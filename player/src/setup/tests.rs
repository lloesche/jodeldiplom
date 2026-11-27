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
    /// Downloads asked for (offline too).
    downloads: AtomicUsize,
    free: Option<u64>,
    memory: Option<u64>,
}

impl Work for Fake {
    fn download(
        &self,
        part: &Part,
        to: &Path,
        progress: &mut dyn FnMut(u64),
    ) -> Result<(), String> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self.offline.load(Ordering::SeqCst) {
            return Err("no network".into());
        }
        let data = &self.web.iter().find(|(u, _)| *u == part.url).unwrap().1;
        progress(data.len() as u64);
        if data.len() as u64 != part.sum.bytes || sha(data) != part.sum.sha256 {
            return Err("not the file".into());
        }
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(to, data).map_err(|e| e.to_string())
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
            crate::work::cover::DEPTH_FILE,
            crate::work::cover::MATTE_FILE,
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
        assert_eq!(found.covers.is_some(), gpu);
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
