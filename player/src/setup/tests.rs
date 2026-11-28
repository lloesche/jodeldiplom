//! Setup on a made-up web: what is got, checked and stamped, and what is
//! not got again.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::work::download::Sha256;
use crate::work::story_models as story;
use crate::work::told::Told;

/// Every needed model's path, when all are there and checked.
fn ready(manifest: &[Need], places: &Places) -> Option<Vec<PathBuf>> {
    manifest
        .iter()
        .map(|need| match look(need, places) {
            Lookup::Checked(p) => Some(p),
            _ => None,
        })
        .collect()
}

/// The paths setup gave, without the models' names.
fn got(paths: Paths) -> Vec<PathBuf> {
    paths.into_iter().map(|(_, p)| p).collect()
}

/// `data`'s [`Sum`].
fn sum(data: &[u8]) -> Sum {
    Sum {
        bytes: data.len() as u64,
        sha256: sha(data),
    }
}

fn sha(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.hex()
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "jodel-setup-{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A made-up web: each URL's bytes; packing concatenates the parts.
struct Fake {
    web: Vec<(String, Vec<u8>)>,
    offline: std::sync::atomic::AtomicBool,
    /// Every download fails on the disk.
    disk_full: std::sync::atomic::AtomicBool,
    /// The files on the web are others now (their bytes reversed).
    changed: std::sync::atomic::AtomicBool,
    /// A download panics.
    panics: std::sync::atomic::AtomicBool,
    /// Downloads asked for (offline too).
    downloads: AtomicUsize,
    free: Option<u64>,
    memory: Option<u64>,
}

impl Work for Fake {
    fn download(&self, part: &Part, to: &Path, progress: &mut dyn FnMut(u64)) -> Result<(), Fault> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.offline.load(Ordering::SeqCst) {
            return Err(Fault::Web("no network".into()));
        }
        assert!(!self.panics.load(Ordering::SeqCst), "a download panicked");
        if self.disk_full.load(Ordering::SeqCst) {
            return Err(Fault::Disk("there is no space left on the device".into()));
        }
        let mut data = self
            .web
            .iter()
            .find(|(u, _)| *u == part.url)
            .unwrap()
            .1
            .clone();
        if self.changed.load(Ordering::SeqCst) {
            data.reverse();
        }
        progress(data.len() as u64);
        if data.len() as u64 != part.sum.bytes || sha(&data) != part.sum.sha256 {
            return Err(Fault::Wrong("not the file".into()));
        }
        // As the real one: beside, then over a file there (made writable).
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        let partial = to.with_extension("partial");
        fs::write(&partial, &data).map_err(|e| Fault::Disk(e.to_string()))?;
        durable::make_replaceable(to);
        fs::rename(&partial, to).map_err(|e| Fault::Disk(e.to_string()))
    }

    fn pack(
        &self,
        parts: &Path,
        to: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String> {
        let mut out = Vec::new();
        for name in ["a", "b"] {
            out.extend(fs::read(parts.join(name)).map_err(|e| e.to_string())?);
        }
        progress(2, 2);
        fs::write(to, out).map_err(|e| e.to_string())
    }

    fn free_space(&self, _dir: &Path) -> Option<u64> {
        self.free
    }

    fn memory(&self) -> Option<u64> {
        self.memory
    }
}

fn fake() -> (Vec<Need>, Fake) {
    let speech = b"the speech model".to_vec();
    let (a, b) = (b"part a, ".to_vec(), b"part b".to_vec());
    let packed: Vec<u8> = a.iter().chain(&b).copied().collect();
    let part = |name: &str, data: &[u8]| {
        let part = Part {
            url: format!("https://example.test/{name}"),
            sum: sum(data),
        };
        (name.to_owned(), part)
    };
    let manifest = vec![
        Need {
            file: "speech.bin".into(),
            title: "the speech model".into(),
            credit: "us".into(),
            why: "timing".into(),
            source: Source::Download(Part {
                url: "https://example.test/speech".into(),
                sum: sum(&speech),
            }),
        },
        Need {
            file: "asr.bin".into(),
            title: "the recogniser".into(),
            credit: "us".into(),
            why: "checking".into(),
            source: Source::Pack {
                packed: sum(&packed),
                parts: vec![part("a", &a), part("b", &b)],
                why: "smaller".into(),
            },
        },
    ];
    let web = vec![
        ("https://example.test/speech".into(), speech),
        ("https://example.test/a".into(), a),
        ("https://example.test/b".into(), b),
    ];
    (
        manifest,
        Fake {
            web,
            offline: false.into(),
            disk_full: false.into(),
            changed: false.into(),
            panics: false.into(),
            downloads: AtomicUsize::new(0),
            free: Some(1 << 40),
            memory: Some(16 << 30),
        },
    )
}

fn places(dir: &Path) -> Places {
    Places {
        look: vec![dir.join("beside"), dir.join("store")],
        store: dir.join("store"),
    }
}

/// Setup started, and what it tells.
fn start(manifest: Vec<Need>, places: Places, work: Arc<dyn Work>) -> (Setup, Arc<Told>) {
    let told = Told::new();
    let setup = Setup::start(move || manifest, places, work, told.notify(), None);
    (setup, told)
}

/// Where setup ends: finished or failed.
fn settle(setup: &Setup, told: &Told) -> Status {
    told.until("setup settles", || {
        let s = setup.status();
        (s.finished.is_some() || s.failure.is_some()).then_some(s)
    })
}

/// Whether `text` is a SHA-256 (or a revision's 40 digits) in lower-case
/// hex.
fn hex(text: &str, len: usize) -> bool {
    text.len() == len && text.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// Every file of every model the player can need, as [`Need::files`]
/// gives it: where it goes, where it comes from, its length and SHA-256.
fn every_file(m: &[Need]) -> Vec<(PathBuf, &Part)> {
    let store = Path::new("store");
    m.iter()
        .flat_map(|need| need.files(&need.at(store)))
        .collect()
}

#[test]
fn the_built_in_manifest_names_the_models_at_fixed_revisions() {
    let without = manifest_for(false);
    assert_eq!(without.len(), 2);
    let m = manifest_for(true);
    assert_eq!(m[..2], without[..], "the speech models everywhere");
    let files: Vec<&str> = m.iter().map(|n| n.file.as_str()).collect();
    assert_eq!(
        files,
        [
            models::MODEL_FILE,
            models::ASR_FILE,
            crate::work::scenes::DEPTH_FILE,
            crate::work::scenes::MATTE_FILE,
            story::LANGUAGE_FILE,
            story::TRANSFORMER_DIR,
            story::PIPELINE_DIR,
            story::ENCODER_DIR,
        ]
    );
    let all = every_file(&m);
    for (path, part) in &all {
        assert!(hex(&part.sum.sha256, 64), "{}: {part:?}", path.display());
        assert!(part.sum.bytes > 0, "{}", path.display());
        let url = &part.url;
        assert!(url.starts_with("https://huggingface.co/"), "{url}");
        // Every file but the word-timing model's (as it always was) at a
        // revision of 40 digits.
        if path.ends_with(models::MODEL_FILE) {
            continue;
        }
        let (_, rest) = url.split_once("/resolve/").expect(url);
        let (revision, _) = rest.split_once('/').expect(url);
        assert!(hex(revision, 40), "{url}");
    }
    // A folder's files and a checkpoint's parts at the paths their
    // repository has them; the story's language model as it names it.
    for need in &m {
        let parts = match &need.source {
            Source::Folder(files) => files,
            Source::Pack { parts, .. } => parts,
            Source::Download(_) => continue,
        };
        for (name, part) in parts {
            assert!(part.url.ends_with(&format!("/{name}")), "{part:?}");
        }
    }
    assert!(story::LANGUAGE_URL.ends_with("/gemma-4-E4B-it-Q4_K_M.gguf"));
    assert!(story::LANGUAGE_FILE.ends_with("/gemma-4-E4B-it-Q4_K_M.gguf"));
    // No two files at one place.
    let mut paths: Vec<&PathBuf> = all.iter().map(|(path, _)| path).collect();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), all.len());
    // The story's models under `story/`, each folder's files at the paths
    // its repository has them.
    for need in &m[4..] {
        assert!(need.file.starts_with("story/"), "{}", need.file);
    }
    let Source::Folder(files) = &m[6].source else {
        panic!("the pipeline's folder")
    };
    assert!(files.iter().any(
        |(name, f)| name == "vae/diffusion_pytorch_model.safetensors"
            && f.url.ends_with("/vae/diffusion_pytorch_model.safetensors")
    ));
}

#[test]
fn the_models_found_are_known_by_their_paths() {
    let store = Path::new("/data/models");
    for gpu in [false, true] {
        let paths: Paths = manifest_for(gpu)
            .iter()
            .map(|n| (n.file.clone(), n.at(store)))
            .collect();
        let found = Found::of(&paths).unwrap();
        assert_eq!(found.speech, store.join(models::MODEL_FILE));
        assert_eq!(found.scenes.is_some(), gpu);
        assert_eq!(
            found.story,
            gpu.then(|| story::StoryModels {
                language: store.join("story").join("gemma-4-E4B-it-Q4_K_M.gguf"),
                transformer: store.join("story").join("flux2-klein-4b-fp8"),
                pipeline: store.join("story").join("flux2-klein-4b"),
                encoder: store.join("story").join("qwen3-4b-fp8"),
            })
        );
    }
}

/// A model's folder on the made-up web: two files, one in a folder of
/// its own; and the web with them.
fn folder_need() -> (Need, Vec<(String, Vec<u8>)>) {
    let (weights, settings) = (b"the weights".to_vec(), b"{}".to_vec());
    let part = |name: &str, data: &[u8]| {
        let part = Part {
            url: format!("https://example.test/folder/{name}"),
            sum: sum(data),
        };
        (name.to_owned(), part)
    };
    let need = Need {
        file: "story/painter".into(),
        title: "the painter".into(),
        credit: "us".into(),
        why: "pictures".into(),
        source: Source::Folder(vec![
            part("w.bin", &weights),
            part("vae/config.json", &settings),
        ]),
    };
    let web = vec![
        ("https://example.test/folder/w.bin".into(), weights),
        (
            "https://example.test/folder/vae/config.json".into(),
            settings,
        ),
    ];
    (need, web)
}

/// A model's folder is got in one step, its files at their paths in it,
/// each stamped: the next start takes it without a download. A file of it
/// left from a run that stopped is kept, the others got.
#[test]
fn a_models_folder_is_got_in_one_step_and_then_taken_as_it_is() {
    let dir = temp("folder");
    let (manifest, mut work) = fake();
    let (folder, web) = folder_need();
    work.web.extend(web);
    let manifest: Vec<Need> = manifest.into_iter().chain([folder]).collect();
    let places = places(&dir);
    // The weights from a run that stopped.
    let at = dir.join("store").join("story").join("painter");
    fs::create_dir_all(&at).unwrap();
    fs::write(at.join("w.bin"), b"the weights").unwrap();
    assert_eq!(look(&manifest[2], &places), Lookup::Missing);
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(s.failure, None);
    let paths = got(s.finished.unwrap());
    assert_eq!(paths[2], at);
    assert_eq!(fs::read(at.join("vae").join("config.json")).unwrap(), b"{}");
    // Speech, the checkpoint, the packing, the cleaning, the folder: its
    // weights kept, so only its settings' bytes said.
    assert_eq!(s.steps.len(), 5);
    assert_eq!(
        s.steps[4].title, "Downloading the painter (us, 1 KB)",
        "{:?}",
        s.steps
    );
    // Speech, a, b, and the folder's settings.
    assert_eq!(work.downloads.load(Ordering::SeqCst), 4);
    // The checkpoint's stamps went with it.
    let stamps = fs::read_to_string(places.store.join(stamp::STAMP)).unwrap();
    assert!(!stamps.contains("asr.bin.download"), "{stamps}");
    // The next start: all checked, nothing got.
    assert_eq!(ready(&manifest, &places), Some(paths));
    let (setup, told) = start(manifest, places, work.clone());
    let s = settle(&setup, &told);
    assert!(s.steps.is_empty(), "{:?}", s.steps);
    assert_eq!(work.downloads.load(Ordering::SeqCst), 4);
}

/// A folder beside the executable is taken whole from there once its
/// files check; one with a wrong file is got afresh into the store, the
/// copy beside left as it is.
#[test]
fn a_folder_beside_the_executable_is_checked_and_a_wrong_one_got_afresh() {
    let dir = temp("folder-beside");
    let (manifest, mut work) = fake();
    let (folder, web) = folder_need();
    work.web.extend(web);
    let manifest: Vec<Need> = manifest.into_iter().chain([folder]).collect();
    let places = places(&dir);
    let beside = dir.join("beside").join("story").join("painter");
    fs::create_dir_all(beside.join("vae")).unwrap();
    fs::write(beside.join("w.bin"), b"the weights").unwrap();
    fs::write(beside.join("vae").join("config.json"), b"{}").unwrap();
    assert_eq!(
        look(&manifest[2], &places),
        Lookup::Unchecked(beside.clone())
    );
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let paths = got(settle(&setup, &told).finished.expect("finished"));
    assert_eq!(paths[2], beside, "the right one is used where it is");
    assert_eq!(look(&manifest[2], &places), Lookup::Checked(beside.clone()));

    // A file of it wrong (its length right): the folder got into the store.
    let dir = temp("folder-wrong");
    let places = self::places(&dir);
    let beside = dir.join("beside").join("story").join("painter");
    fs::create_dir_all(beside.join("vae")).unwrap();
    fs::write(beside.join("w.bin"), b"the WEIGHTS").unwrap();
    fs::write(beside.join("vae").join("config.json"), b"{}").unwrap();
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    let paths = got(s.finished.expect("finished"));
    assert_eq!(paths[2], dir.join("store").join("story").join("painter"));
    assert_eq!(fs::read(beside.join("w.bin")).unwrap(), b"the WEIGHTS");
    assert_eq!(ready(&manifest, &places), Some(paths));
}

/// A folder in the store with one file wrong: only that file is got
/// again, and the step says only its bytes.
#[test]
fn a_folder_with_one_wrong_file_gets_only_that_one_again() {
    let dir = temp("folder-one-wrong");
    let (manifest, mut work) = fake();
    let (folder, web) = folder_need();
    work.web.extend(web);
    let manifest: Vec<Need> = manifest.into_iter().chain([folder]).collect();
    let places = places(&dir);
    let at = dir.join("store").join("story").join("painter");
    fs::create_dir_all(at.join("vae")).unwrap();
    fs::write(at.join("w.bin"), b"the weights").unwrap();
    fs::write(at.join("vae").join("config.json"), b"[]").unwrap();
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(got(s.finished.expect("finished"))[2], at);
    let titles: Vec<&str> = s.steps.iter().map(|st| st.title.as_str()).collect();
    assert!(
        titles.contains(&"Checking the painter (1 KB) against its built-in checksums"),
        "{titles:?}"
    );
    assert!(
        titles.contains(&"Downloading the painter (us, 1 KB)"),
        "{titles:?}"
    );
    // Speech, a, b, and the settings again: the weights kept.
    assert_eq!(work.downloads.load(Ordering::SeqCst), 4);
    assert_eq!(fs::read(at.join("vae").join("config.json")).unwrap(), b"{}");
    assert!(ready(&manifest, &places).is_some());
}

#[test]
fn everything_is_got_checked_and_stamped_then_nothing_more() {
    let dir = temp("all");
    let (manifest, work) = fake();
    let work = Arc::new(work);
    let places = places(&dir);
    assert_eq!(ready(&manifest, &places), None);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(s.failure, None);
    let paths = got(s.finished.clone().unwrap());
    assert_eq!(
        paths,
        vec![dir.join("store/speech.bin"), dir.join("store/asr.bin")]
    );
    assert_eq!(fs::read(&paths[1]).unwrap(), b"part a, part b");
    // Speech, the checkpoint, the packing, the cleaning.
    assert_eq!(s.steps.len(), 4);
    assert!(s.steps.iter().all(|st| st.state == StepState::Done));
    assert_eq!(s.overall(), 1.0);
    assert!(told.count() > 0);
    assert!(
        !dir.join("store/asr.bin.download").exists(),
        "the parts removed"
    );
    // The next start: ready at once, nothing downloaded.
    assert_eq!(ready(&manifest, &places), Some(paths));
    assert_eq!(work.downloads.load(Ordering::SeqCst), 3);
}

#[test]
fn a_file_beside_the_executable_is_checked_once_and_a_wrong_one_replaced() {
    let dir = temp("beside");
    let (manifest, work) = fake();
    let places = places(&dir);
    fs::create_dir_all(dir.join("beside")).unwrap();
    // The speech model beside, right; the recogniser beside, wrong (its
    // length right, its bytes not).
    fs::write(dir.join("beside/speech.bin"), b"the speech model").unwrap();
    fs::write(dir.join("beside/asr.bin"), b"part A, part B").unwrap();
    assert_eq!(
        look(&manifest[0], &places),
        Lookup::Unchecked(dir.join("beside/speech.bin"))
    );
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    let paths = got(s.finished.expect("finished"));
    assert_eq!(
        paths[0],
        dir.join("beside/speech.bin"),
        "the right one is used where it is"
    );
    assert_eq!(
        paths[1],
        dir.join("store/asr.bin"),
        "the wrong one is got afresh"
    );
    assert!(dir.join("beside/asr.bin").exists(), "a file not ours stays");
    // Two checks, then the recogniser's checkpoint, packing, cleaning.
    assert_eq!(s.steps.len(), 5);
    assert_eq!(work.downloads.load(Ordering::SeqCst), 2);
    // The next start passes over the wrong one for the checked one.
    assert_eq!(ready(&manifest, &places), Some(paths));
}

#[test]
fn offline_it_waits_for_try_again_and_goes_on_from_there() {
    let dir = temp("offline");
    let (manifest, work) = fake();
    work.offline.store(true, Ordering::SeqCst);
    let work = Arc::new(work);
    let (setup, told) = start(manifest, places(&dir), work.clone());
    let s = settle(&setup, &told);
    let f = s.failure.expect("a failure");
    assert!(f.what.contains("no network"), "{f:?}");
    assert!(f.advice.contains("online"), "{f:?}");
    assert_eq!(s.finished, None);
    // Where it ends once tried again (a download asked for again).
    let retried = || {
        let tried = work.downloads.load(Ordering::SeqCst);
        setup.retry();
        told.until("setup settles again", || {
            let s = setup.status();
            let again = work.downloads.load(Ordering::SeqCst) > tried;
            (again && (s.finished.is_some() || s.failure.is_some())).then_some(s)
        })
    };
    // Still offline: it fails again.
    assert!(retried().failure.is_some());
    work.offline.store(false, Ordering::SeqCst);
    let s = retried();
    assert_eq!(s.failure, None);
    assert!(s.finished.is_some());
}

#[test]
fn too_little_disk_or_memory_is_said_before_anything_is_fetched() {
    let dir = temp("room");
    let (manifest, mut work) = fake();
    work.free = Some(10);
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places(&dir), work.clone());
    let f = settle(&setup, &told).failure.expect("no room");
    assert!(f.what.contains("not enough free space"), "{f:?}");
    assert_eq!(work.downloads.load(Ordering::SeqCst), 0);

    let dir = temp("memory");
    let (manifest, mut work) = fake();
    work.memory = Some(4 << 30);
    let (setup, told) = start(manifest, places(&dir), Arc::new(work));
    let f = settle(&setup, &told).failure.expect("too little memory");
    assert!(f.what.contains("memory"), "{f:?}");
    assert!(f.advice.contains("Copy the folder"), "{f:?}");
}

/// While the data folder moves, the models not yet moved are checked
/// where they still are (nothing fetched); moved, they are checked once
/// where they went.
#[test]
fn models_are_found_where_they_are_while_the_data_folder_moves() {
    let dir = temp("moved");
    let (manifest, work) = fake();
    let old = dir.join("old/models");
    fs::create_dir_all(&old).unwrap();
    fs::write(old.join("speech.bin"), b"the speech model").unwrap();
    fs::write(old.join("asr.bin"), b"part a, part b").unwrap();
    let places = places(&dir);
    crate::store::mover::begin(
        &dir.join("settings"),
        &[(old.clone(), places.store.clone())],
    )
    .unwrap();
    assert_eq!(
        look(&manifest[0], &places),
        Lookup::Unchecked(old.join("speech.bin"))
    );
    let work = Arc::new(work);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(s.failure, None);
    assert_eq!(
        got(s.finished.unwrap()),
        vec![old.join("speech.bin"), old.join("asr.bin")]
    );
    assert_eq!(work.downloads.load(Ordering::SeqCst), 0, "nothing fetched");
    // The mover brings them over: checked once where they went.
    fs::create_dir_all(&places.store).unwrap();
    for name in ["speech.bin", "asr.bin"] {
        fs::rename(old.join(name), places.store.join(name)).unwrap();
    }
    assert_eq!(ready(&manifest, &places), None);
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(
        got(s.finished.unwrap()),
        vec![
            places.store.join("speech.bin"),
            places.store.join("asr.bin")
        ]
    );
    assert_eq!(work.downloads.load(Ordering::SeqCst), 0, "nothing fetched");
    assert!(ready(&manifest, &places).is_some());
}

#[test]
fn a_stamp_goes_stale_when_the_file_changes() {
    let dir = temp("stale");
    let (manifest, work) = fake();
    let places = places(&dir);
    let (setup, told) = start(manifest.clone(), places.clone(), Arc::new(work));
    settle(&setup, &told).finished.expect("finished");
    // Rewritten, the same length: not taken without a check. Its time
    // set a second on, so that it differs on any file system's clock.
    let path = dir.join("store/speech.bin");
    let was = fs::metadata(&path).unwrap().modified().unwrap();
    fs::write(&path, b"the speech MODEL").unwrap();
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_modified(was + Duration::from_secs(1)).unwrap();
    drop(file);
    assert_eq!(look(&manifest[0], &places), Lookup::Unchecked(path));
    assert_eq!(ready(&manifest, &places), None);
}

/// Setup run once over `manifest` into a fresh `places` under `name`,
/// everything got: its folder, the places, and the made-up web.
fn set_up(name: &str, manifest: &[Need], work: Fake) -> (PathBuf, Places, Arc<Fake>) {
    let dir = temp(name);
    let places = places(&dir);
    let work = Arc::new(work);
    let (setup, told) = start(manifest.to_vec(), places.clone(), work.clone());
    settle(&setup, &told).finished.expect("finished");
    (dir, places, work)
}

/// The lines a repair says, in order.
type Said = Arc<std::sync::Mutex<Vec<Result<String, String>>>>;

/// The repair of `manifest` set up in `places` (each model in the store),
/// what tells it, and what it says.
fn repairing(
    manifest: &[Need],
    places: &Places,
    work: &Arc<Fake>,
) -> (Arc<Repair>, Unreadable, Said) {
    let paths: Paths = (manifest.iter())
        .map(|n| (n.file.clone(), n.at(&places.store)))
        .collect();
    let repair = Repair::new(manifest.to_vec(), places.clone(), work.clone(), &paths);
    let said: Said = Arc::default();
    let heard = Arc::clone(&said);
    repair.tell_to(Box::new(move |line| heard.lock().unwrap().push(line)));
    let unreadable = repair.unreadable();
    (repair, unreadable, said)
}

/// Where the repair of `file` ends (no longer with its worker).
fn settled(repair: &Repair, file: &str) -> repair::State {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match repair.state(file) {
            Some(repair::State::Busy) | None => {}
            Some(state) => return state,
        }
        assert!(Instant::now() < deadline, "{file} not settled");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `unreadable` told that the model of `file` could not be read.
fn fails(unreadable: &Unreadable, file: &str) {
    let _ = unreadable.report(file, Err::<(), _>("not read as its format".to_owned()));
}

/// `path` written `data`, its time set to `at`.
fn rewrite(path: &Path, data: &[u8], at: std::time::SystemTime) {
    fs::write(path, data).unwrap();
    let file = fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_modified(at).unwrap();
}

/// A file cut short (a download stopped, the disk) or grown: found at the
/// start by its length alone, nothing hashed, and only that file got
/// again.
#[test]
fn a_file_of_the_wrong_length_is_got_again_alone_without_hashing() {
    let (manifest, mut work) = fake();
    let (folder, web) = folder_need();
    work.web.extend(web);
    let manifest: Vec<Need> = manifest.into_iter().chain([folder]).collect();
    let (dir, places, work) = set_up("wrong-length", &manifest, work);
    for wrong in [&b"the wei"[..], b"the weights and more"] {
        let at = dir.join("store/story/painter/w.bin");
        fs::write(&at, wrong).unwrap();
        assert_eq!(look(&manifest[2], &places), Lookup::Missing);
        let before = work.downloads.load(Ordering::SeqCst);
        let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
        let s = settle(&setup, &told);
        let titles: Vec<&str> = s.steps.iter().map(|st| st.title.as_str()).collect();
        assert_eq!(titles, ["Downloading the painter (us, 1 KB)"]);
        assert_eq!(work.downloads.load(Ordering::SeqCst), before + 1);
        assert_eq!(fs::read(&at).unwrap(), b"the weights");
        assert!(ready(&manifest, &places).is_some());
    }
}

/// A model gone while the player runs (here the packed recogniser) is
/// got again alone once it is found gone, the user told as it begins and
/// as it ends.
#[test]
fn a_model_gone_while_the_player_runs_is_got_again_alone() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-gone", &manifest, work);
    let asr = dir.join("store/asr.bin");
    fs::remove_file(&asr).unwrap();
    let before = work.downloads.load(Ordering::SeqCst);
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    assert!(!unreadable.present("asr.bin", &asr));
    assert_eq!(settled(&repair, "asr.bin"), repair::State::Whole);
    // Its checkpoint's two parts; the speech model left alone.
    assert_eq!(work.downloads.load(Ordering::SeqCst), before + 2);
    assert_eq!(fs::read(&asr).unwrap(), b"part a, part b");
    assert!(!dir.join("store/asr.bin.download").exists());
    assert!(ready(&manifest, &places).is_some());
    assert_eq!(
        *said.lock().unwrap(),
        [
            Ok("The recogniser is missing or damaged: getting it again.".to_owned()),
            Ok("The recogniser is got again and checked.".to_owned()),
        ]
    );
}

/// A model damaged at its length, its time kept (so that the start's
/// stat takes it as checked), is found by its checksum once it cannot be
/// read, and got again; its loader's failed setup is then tried again at
/// once, not after its while.
#[test]
fn a_model_damaged_at_its_length_is_found_by_its_checksum_once_it_cannot_be_read() {
    struct Net;
    impl jodel_gpu::OnDevice for Net {
        fn lost(&self) -> bool {
            false
        }
    }
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-damaged", &manifest, work);
    let path = dir.join("store/speech.bin");
    let was = fs::metadata(&path).unwrap().modified().unwrap();
    rewrite(&path, b"the speech MODEL", was);
    assert_eq!(look(&manifest[0], &places), Lookup::Checked(path.clone()));
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    let (read, told) = (path.clone(), unreadable.clone());
    let kept = Arc::new(jodel_gpu::Kept::new(
        || None,
        Box::new(move || {
            let right = fs::read(&read).map_err(|e| e.to_string())? == b"the speech model";
            let opened = right.then_some(Net).ok_or_else(|| "not a model".to_owned());
            Ok((told.report("speech.bin", opened)?, "a GPU".to_owned()))
        }),
    ));
    unreadable.retries("speech.bin", &kept);
    assert!(!kept.usable());
    assert_eq!(settled(&repair, "speech.bin"), repair::State::Whole);
    assert_eq!(fs::read(&path).unwrap(), b"the speech model");
    assert_eq!(said.lock().unwrap().len(), 2);
    assert!(ready(&manifest, &places).is_some());
    assert!(kept.usable(), "tried again at once");
}

/// A whole model that will not set up (the GPU's fault, say) is hashed
/// once: nothing got, nothing said, and not looked at again while it stays
/// as it is.
#[test]
fn a_whole_model_that_will_not_set_up_is_hashed_once_and_left() {
    let (manifest, work) = fake();
    let (_dir, places, work) = set_up("repair-whole", &manifest, work);
    let before = work.downloads.load(Ordering::SeqCst);
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    assert_eq!(settled(&repair, "speech.bin"), repair::State::Whole);
    fails(&unreadable, "speech.bin");
    assert_eq!(
        repair.state("speech.bin"),
        Some(repair::State::Whole),
        "not again"
    );
    fails(&unreadable, "not a model");
    assert_eq!(repair.state("not a model"), None);
    assert_eq!(work.downloads.load(Ordering::SeqCst), before);
    assert!(said.lock().unwrap().is_empty());
    assert!(ready(&manifest, &places).is_some(), "stamped again");
}

/// A model that cannot be got again (offline) is said, then rests a
/// while: the failed reads meanwhile ask nothing.
#[test]
fn a_model_that_cannot_be_got_again_is_said_and_rests() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-offline", &manifest, work);
    fs::remove_file(dir.join("store/speech.bin")).unwrap();
    work.offline.store(true, Ordering::SeqCst);
    let before = work.downloads.load(Ordering::SeqCst);
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    let failed = settled(&repair, "speech.bin");
    assert!(matches!(failed, repair::State::Failed(_)), "{failed:?}");
    let Some(Err(line)) = said.lock().unwrap().last().cloned() else {
        panic!("not said");
    };
    assert!(
        line.contains("no network") && line.contains("online"),
        "{line}"
    );
    fails(&unreadable, "speech.bin");
    assert_eq!(repair.state("speech.bin"), Some(failed), "it rests");
    assert_eq!(work.downloads.load(Ordering::SeqCst), before + 1);
}

/// A model whose source sends another file now is given up for the run:
/// said once, never fetched again however often it fails to read.
#[test]
fn a_model_whose_source_sends_another_file_is_given_up_for_the_run() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-wrong", &manifest, work);
    fs::remove_file(dir.join("store/speech.bin")).unwrap();
    work.changed.store(true, Ordering::SeqCst);
    let before = work.downloads.load(Ordering::SeqCst);
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    assert_eq!(settled(&repair, "speech.bin"), repair::State::GivenUp);
    for _ in 0..3 {
        fails(&unreadable, "speech.bin");
    }
    assert_eq!(repair.state("speech.bin"), Some(repair::State::GivenUp));
    assert_eq!(work.downloads.load(Ordering::SeqCst), before + 1);
    let Some(Err(line)) = said.lock().unwrap().last().cloned() else {
        panic!("not said");
    };
    assert!(line.contains("not tried again"), "{line}");
}

/// Two models told at once (the image model's two folders, say) are
/// checked one after the other: both got again, and both stamped (no
/// write to the stamps lost).
#[test]
fn two_models_told_at_once_are_both_got_again_and_stamped() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-two", &manifest, work);
    let speech = dir.join("store/speech.bin");
    let was = fs::metadata(&speech).unwrap().modified().unwrap();
    rewrite(&speech, b"the speech MODEL", was);
    fs::remove_file(dir.join("store/asr.bin")).unwrap();
    let (repair, unreadable, said) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    fails(&unreadable, "asr.bin");
    assert_eq!(settled(&repair, "speech.bin"), repair::State::Whole);
    assert_eq!(settled(&repair, "asr.bin"), repair::State::Whole);
    assert_eq!(said.lock().unwrap().len(), 4);
    assert_eq!(ready(&manifest, &places).map(|p| p.len()), Some(2));
}

/// While the data folder moves, a model gone is got again where the data
/// folder goes, its stamp there: not into the folder left.
#[test]
fn a_model_is_got_again_where_the_data_folder_is_moving_to() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-moving", &manifest, work);
    let moved = dir.join("moved");
    crate::store::mover::begin(
        &dir.join("settings"),
        &[(places.store.clone(), moved.clone())],
    )
    .unwrap();
    fs::remove_file(places.store.join("speech.bin")).unwrap();
    let (repair, unreadable, _) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    assert_eq!(settled(&repair, "speech.bin"), repair::State::Whole);
    assert_eq!(
        fs::read(moved.join("speech.bin")).unwrap(),
        b"the speech model"
    );
    assert!(!places.store.join("speech.bin").exists());
    let stamps = fs::read_to_string(moved.join(stamp::STAMP)).unwrap();
    assert!(stamps.contains("speech.bin"), "{stamps}");
    // Checked where it went (the rest comes over with the move).
    let now = Places {
        look: vec![dir.join("beside"), moved.clone()],
        store: moved.clone(),
    };
    let at = moved.join("speech.bin");
    assert_eq!(look(&manifest[0], &now), Lookup::Checked(at));
}

/// A damaged model made read-only, and a read-only stamp file, are
/// replaced: they are the player's own.
#[test]
fn a_read_only_damaged_model_and_stamp_file_are_replaced() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("read-only", &manifest, work);
    let path = dir.join("store/speech.bin");
    let was = fs::metadata(&path).unwrap().modified().unwrap();
    rewrite(&path, b"the speech MODEL", was + Duration::from_secs(1));
    for file in [&path, &places.store.join(stamp::STAMP)] {
        let mut permissions = fs::metadata(file).unwrap().permissions();
        permissions.set_readonly(true);
        fs::set_permissions(file, permissions).unwrap();
    }
    let (setup, told) = start(manifest.clone(), places.clone(), work.clone());
    let s = settle(&setup, &told);
    assert_eq!(s.failure, None);
    assert_eq!(fs::read(&path).unwrap(), b"the speech model");
    assert!(ready(&manifest, &places).is_some());
}

/// A download the disk let down says to check the data folder, not the
/// network.
#[test]
fn a_disk_fault_is_not_said_as_the_networks() {
    let dir = temp("disk-fault");
    let (manifest, work) = fake();
    work.disk_full.store(true, Ordering::SeqCst);
    let (setup, told) = start(manifest, places(&dir), Arc::new(work));
    let f = settle(&setup, &told).failure.expect("a failure");
    assert!(f.what.contains("no space left"), "{f:?}");
    assert!(f.advice.contains("data folder can be written"), "{f:?}");
}

/// Downloads cut off that no model needed goes on from any more (one no
/// longer needed on this machine, or renamed) are removed after setup;
/// one a model goes on from stays.
#[test]
fn downloads_cut_off_that_no_model_needs_are_removed() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("partials", &manifest, work);
    let store = &places.store;
    let unneeded = [
        store.join("old-model.bin.partial"),
        store.join("story").join("painter").join("w.bin.partial"),
    ];
    let wanted = store.join("asr.bin.download").join("a.partial");
    for file in unneeded.iter().chain([&wanted]) {
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(file, b"cut off").unwrap();
    }
    let (setup, told) = start(manifest, places.clone(), work);
    settle(&setup, &told).finished.expect("finished");
    for file in &unneeded {
        assert!(!file.exists(), "{}", file.display());
    }
    assert!(wanted.exists(), "a model's own is gone on from later");
    assert!(dir.join("store/speech.bin").is_file());
}
/// A repair that panics fails that model alone: the worker goes on, and
/// the next model told is got again.
#[test]
fn a_repair_that_panics_fails_its_model_and_the_next_is_still_repaired() {
    let (manifest, work) = fake();
    let (dir, places, work) = set_up("repair-panics", &manifest, work);
    fs::remove_file(dir.join("store/speech.bin")).unwrap();
    work.panics.store(true, Ordering::SeqCst);
    let (repair, unreadable, _) = repairing(&manifest, &places, &work);
    fails(&unreadable, "speech.bin");
    let failed = settled(&repair, "speech.bin");
    assert!(matches!(failed, repair::State::Failed(_)), "{failed:?}");
    work.panics.store(false, Ordering::SeqCst);
    fs::remove_file(dir.join("store/asr.bin")).unwrap();
    fails(&unreadable, "asr.bin");
    assert_eq!(settled(&repair, "asr.bin"), repair::State::Whole);
    assert!(dir.join("store/asr.bin").is_file());
}
