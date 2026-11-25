//! The player's self-check, and its setup when that fails.
//!
//! Most of what the player needs is inside its executable (the beat and key
//! networks, the stem separator, the singer's isolator, the fonts). Four
//! models are too big for it: the speech model that times karaoke's words
//! ([`crate::lyrics::models::MODEL_FILE`]), the speech recogniser that checks
//! fetched lyrics belong to the song ([`crate::lyrics::models::ASR_FILE`]), and the
//! depth and cut-out models that take covers apart for karaoke's background
//! ([`crate::cover::DEPTH_FILE`], [`crate::cover::MATTE_FILE`]). Their
//! names, lengths and SHA-256s are built in ([`manifest`]).
//!
//! At every start [`ready`] looks for them in `models` beside the
//! executable, in a macOS app's `Resources/models`, and in `models` beside
//! the settings ([`Places`]). A file is taken once it has matched its
//! checksum: the first time it is hashed, and a stamp (its length and when
//! it was written) spares later starts the hashing. When every file is
//! there and stamped, the player starts at once and never goes online for
//! them.
//!
//! Otherwise [`Setup`] gets them, on a thread of its own, while the
//! window shows a screen that cannot be skipped ([`crate::view::setup`]):
//! each step said with what it does and why, a bar for the step and one for
//! the whole. A failure (no network, a download not what it should be, too
//! little disk space or memory) is said with what to do, and the player
//! can only try again or quit: without the files, the player is not what
//! it should be.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use crate::browser::Notify;
use crate::download::Sha256;
use crate::lyrics::models;

/// A file downloaded as it is, checked against its length and SHA-256.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Its name where it is kept.
    pub name: String,
    /// Where it comes from.
    pub url: String,
    /// Its length in bytes.
    pub bytes: u64,
    /// Its SHA-256, in lower-case hex.
    pub sha256: String,
    /// What it is, as the screen says it ("the recogniser's weights").
    pub title: String,
}

/// How a needed file is got.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Downloaded as it is.
    Download {
        /// Where from.
        url: String,
    },
    /// Made here from parts downloaded (a checkpoint packed at 4 bits).
    Pack {
        /// The parts.
        parts: Vec<Part>,
        /// Why it is packed, as the screen says it.
        why: String,
    },
}

/// A file the player needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Need {
    /// Its file name.
    pub file: String,
    /// Its length in bytes.
    pub bytes: u64,
    /// Its SHA-256, in lower-case hex.
    pub sha256: String,
    /// What it is, as the screen names it ("the word-timing model").
    pub title: String,
    /// Whose it is ("wav2vec2 by Meta"), said where it is downloaded.
    pub credit: String,
    /// What it is for, as the screen says it.
    pub why: String,
    /// How it is got.
    pub source: Source,
}

/// The files the player needs beside its executable, with their checksums:
/// the speech models, and the covers' models where this machine has a
/// usable GPU (they run nowhere else).
pub fn manifest() -> Vec<Need> {
    manifest_for(jodel_gpu::gpu_name().is_some())
}

/// The models the player has found, by what they are ([`Found::of`] the
/// paths setup gives in the manifest's order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    /// The word-timing model ([`models::MODEL_FILE`]).
    pub speech: PathBuf,
    /// The speech recogniser ([`models::ASR_FILE`]).
    pub asr: PathBuf,
    /// The covers' depth and cut-out models, where the machine needs them.
    pub covers: Option<crate::cover::CoverModels>,
}

impl Found {
    /// The models among `paths` (setup's, every needed file's path), each
    /// known by its file's name; `None` without the speech models.
    pub fn of(paths: &[PathBuf]) -> Option<Found> {
        let named = |file: &str| {
            paths
                .iter()
                .find(|p| p.file_name().is_some_and(|n| n == file))
                .cloned()
        };
        let covers = match (
            named(crate::cover::DEPTH_FILE),
            named(crate::cover::MATTE_FILE),
        ) {
            (Some(depth), Some(matte)) => Some(crate::cover::CoverModels { depth, matte }),
            _ => None,
        };
        Some(Found {
            speech: named(models::MODEL_FILE)?,
            asr: named(models::ASR_FILE)?,
            covers,
        })
    }
}

/// [`manifest`] for a machine with a usable GPU (`gpu`) or without.
pub fn manifest_for(gpu: bool) -> Vec<Need> {
    let asr_parts = models::ASR_PARTS
        .iter()
        .map(|&(name, bytes, sha)| Part {
            name: name.to_owned(),
            url: format!("{}/{name}", models::ASR_BASE),
            bytes,
            sha256: sha.to_owned(),
            title: match name {
                "model.safetensors" => "the speech recogniser's weights",
                "vocab.json" => "its vocabulary",
                _ => "its tokenizer's settings",
            }
            .to_owned(),
        })
        .collect();
    let mut needs = vec![
        Need {
            file: models::MODEL_FILE.to_owned(),
            bytes: models::MODEL_BYTES,
            sha256: models::MODEL_SHA256.to_owned(),
            title: "the word-timing model".to_owned(),
            credit: "wav2vec2 by Meta".to_owned(),
            why: "It hears where each sung word falls, so karaoke and the Transition tab's lyrics \
                  light up word by word."
                .to_owned(),
            source: Source::Download {
                url: models::MODEL_URL.to_owned(),
            },
        },
        Need {
            file: models::ASR_FILE.to_owned(),
            bytes: models::ASR_BYTES,
            sha256: models::ASR_SHA256.to_owned(),
            title: "the speech recogniser".to_owned(),
            credit: "Qwen3-ASR 0.6B by Alibaba's Qwen team".to_owned(),
            why: "It listens to the singing to check that lyrics found online belong to the \
                  song."
                .to_owned(),
            source: Source::Pack {
                parts: asr_parts,
                why: "Packed at 4 bits, with our calibration from 1,000 songs, it keeps its \
                      accuracy in a third of the memory: 589 MB instead of 1.9 GB."
                    .to_owned(),
            },
        },
    ];
    if !gpu {
        return needs;
    }
    needs.extend([
        Need {
            file: crate::cover::DEPTH_FILE.to_owned(),
            bytes: crate::cover::DEPTH_BYTES,
            sha256: crate::cover::DEPTH_SHA256.to_owned(),
            title: "the depth model".to_owned(),
            credit: "Depth Anything 3 Small by ByteDance Seed".to_owned(),
            why: "It sees how near each part of a cover is, so karaoke's background can move \
                  it in depth."
                .to_owned(),
            source: Source::Download {
                url: crate::cover::DEPTH_URL.to_owned(),
            },
        },
        Need {
            file: crate::cover::MATTE_FILE.to_owned(),
            bytes: crate::cover::MATTE_BYTES,
            sha256: crate::cover::MATTE_SHA256.to_owned(),
            title: "the cover's cut-out model".to_owned(),
            credit: "BEN2 by Prama".to_owned(),
            why: "It cuts the singer (or whatever stands out) off a cover, so karaoke's \
                  background can light and move them on their own."
                .to_owned(),
            source: Source::Download {
                url: crate::cover::MATTE_URL.to_owned(),
            },
        },
    ]);
    needs
}

/// Where the needed files are looked for, and where setup puts them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Places {
    /// The folders looked in, in order (the store last).
    pub look: Vec<PathBuf>,
    /// Where setup puts what it gets, and keeps its stamp: `models` beside
    /// the settings.
    pub store: PathBuf,
}

impl Places {
    /// The player's: `models` beside the executable and in a macOS app's
    /// `Resources` (where models sent along with the player sit), then
    /// `models` in the data folder `data`.
    pub fn player(data: &crate::dirs::Data) -> Places {
        let store = data.models();
        let mut look = Vec::new();
        if let Some(dir) = std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(Path::parent)
        {
            look.push(dir.join(models::MODEL_DIR));
            look.push(dir.join("..").join("Resources").join(models::MODEL_DIR));
        }
        look.push(store.clone());
        Places { look, store }
    }
}

/// The stamp's file in the store: a line a file checked, `SHA-256 length
/// written-at path` (written-at in ns since 1970).
pub const STAMP: &str = "checked.txt";

/// When `meta`'s file was last written, in ns since 1970.
fn written(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos())
}

fn stamp_lines(store: &Path) -> Vec<String> {
    fs::read_to_string(store.join(STAMP))
        .map(|t| t.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn stamp_line(sha256: &str, path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    Some(format!(
        "{sha256} {} {} {}",
        meta.len(),
        written(&meta),
        path.display()
    ))
}

/// Whether `path` was checked against `sha256` as it is now.
fn stamped(store: &Path, sha256: &str, path: &Path) -> bool {
    stamp_line(sha256, path).is_some_and(|line| stamp_lines(store).contains(&line))
}

/// Records `path` as checked against `sha256` (its earlier lines gone).
fn stamp(store: &Path, sha256: &str, path: &Path) -> Result<(), String> {
    let Some(line) = stamp_line(sha256, path) else {
        return Err(format!("{}: gone", path.display()));
    };
    let tail = format!(" {}", path.display());
    let mut lines: Vec<String> = stamp_lines(store)
        .into_iter()
        .filter(|l| !l.ends_with(&tail))
        .collect();
    lines.push(line);
    durable::write_atomic(&store.join(STAMP), (lines.join("\n") + "\n").as_bytes())
        .map_err(|e| format!("{}: {e}", store.join(STAMP).display()))
}

/// Where a needed file was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// There and checked (stamped) as it is.
    Checked(PathBuf),
    /// There at its length, not checked as it is: to be hashed.
    Unchecked(PathBuf),
    /// Nowhere at its length.
    Missing,
}

/// Where `need` is: a copy checked anywhere first, else the first at its
/// length (a copy beside the executable that is wrong is passed over once
/// a good one is in the store).
pub fn look(need: &Need, places: &Places) -> Lookup {
    // Through the mover: while the data folder moves, one not yet moved is
    // where it was.
    let candidates: Vec<PathBuf> = places
        .look
        .iter()
        .map(|d| crate::mover::locate(&d.join(&need.file)))
        .filter(|p| fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() == need.bytes))
        .collect();
    if let Some(p) = candidates
        .iter()
        .find(|p| stamped(&places.store, &need.sha256, p))
    {
        return Lookup::Checked(p.clone());
    }
    candidates
        .into_iter()
        .next()
        .map_or(Lookup::Missing, Lookup::Unchecked)
}

/// Every needed file's path, when all are there and checked: the player
/// can start without setup.
pub fn ready(manifest: &[Need], places: &Places) -> Option<Vec<PathBuf>> {
    manifest
        .iter()
        .map(|need| match look(need, places) {
            Lookup::Checked(p) => Some(p),
            _ => None,
        })
        .collect()
}

/// What setup needs of the system: downloads, packing, the disk and the
/// memory. The player's is [`Real`]; tests make their own.
pub trait Work: Send + Sync {
    /// Downloads `part` to `to` (checked against its length and SHA-256
    /// before it takes the name), telling `progress` the bytes so far.
    fn download(&self, part: &Part, to: &Path, progress: &mut dyn FnMut(u64))
        -> Result<(), String>;
    /// Packs the checkpoint in `parts` into `to`, telling `progress` the
    /// tensors packed so far and in all.
    fn pack(
        &self,
        parts: &Path,
        to: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String>;
    /// The bytes free on the disk `dir` is on, where it can be told.
    fn free_space(&self, dir: &Path) -> Option<u64>;
    /// The computer's memory in bytes, where it can be told.
    fn memory(&self) -> Option<u64>;
}

/// The memory packing takes at its height (the checkpoint read in f32, and
/// what it is packed into), with room for the rest of the system.
pub const PACK_MEMORY: u64 = 7_500_000_000;

/// The player's: from the web, packed by [`jodel_lyrics::asr::packed`].
pub struct Real;

impl Work for Real {
    fn download(
        &self,
        part: &Part,
        to: &Path,
        progress: &mut dyn FnMut(u64),
    ) -> Result<(), String> {
        crate::download::download(&part.url, to, part.bytes, &part.sha256, |done, _| {
            progress(done)
        })
    }

    fn pack(
        &self,
        parts: &Path,
        to: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String> {
        use jodel_lyrics::asr::packed::PackedModel;
        let model = jodel_lyrics::asr::Qwen3Asr::open(parts)?;
        let statistics = jodel_lyrics::asr::quant::Statistics::built_in();
        let packed = PackedModel::from_model_with(
            &model,
            |_| 4,
            Some(&statistics),
            |done, all| progress(done as u64, all as u64),
        );
        drop(model);
        durable::write_atomic(to, &packed.to_bytes()).map_err(|e| format!("{}: {e}", to.display()))
    }

    fn free_space(&self, dir: &Path) -> Option<u64> {
        system::free_space(dir)
    }

    fn memory(&self) -> Option<u64> {
        system::memory()
    }
}

/// What a step's progress counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Bytes (downloaded, or hashed).
    Bytes,
    /// Tensors packed.
    Tensors,
}

/// Where a step stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    /// Still to come.
    ToDo,
    /// Under way.
    Working,
    /// Done.
    Done,
}

/// A step, as the screen shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// What it does ("Downloading the word-timing model (378 MB)").
    pub title: String,
    /// Why ("It hears where each sung word falls…").
    pub why: String,
    /// Where it stands.
    pub state: StepState,
    /// Its share of the whole, as a rough time.
    weight: f64,
}

/// Why setup stopped, and what to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// What went wrong.
    pub what: String,
    /// What to do about it.
    pub advice: String,
}

/// Where setup stands, for the screen.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Status {
    /// Its steps, in order.
    pub steps: Vec<Step>,
    /// The step under way.
    pub current: Option<usize>,
    /// The step's progress: so far, in all (0: not known), counting what.
    pub done: u64,
    /// See [`Status::done`].
    pub total: u64,
    /// See [`Status::done`].
    pub unit: Option<Unit>,
    /// A download's speed, bytes a second, once it can be told.
    pub rate: Option<f64>,
    /// Why it stopped: it waits for [`Setup::retry`].
    pub failure: Option<Failure>,
    /// Every needed file's path, in the manifest's order: the player can
    /// start.
    pub finished: Option<Vec<PathBuf>>,
}

impl Status {
    /// The share of the current step done (0 to 1).
    pub fn step_share(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0)
        }
    }

    /// The share of the whole done (0 to 1).
    pub fn overall(&self) -> f64 {
        let all: f64 = self.steps.iter().map(|s| s.weight).sum();
        if all <= 0.0 {
            return if self.finished.is_some() { 1.0 } else { 0.0 };
        }
        let done: f64 = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| match s.state {
                StepState::Done => s.weight,
                StepState::Working if Some(i) == self.current => s.weight * self.step_share(),
                _ => 0.0,
            })
            .sum();
        (done / all).clamp(0.0, 1.0)
    }
}

/// How long hashing a byte takes against downloading one, for the weights.
const HASH_WEIGHT: f64 = 0.05;
/// Packing's weight, in downloaded bytes' worth (about a minute).
const PACK_WEIGHT: f64 = 1.5e9;

/// One thing setup does.
#[derive(Clone, Debug)]
enum Task {
    /// Hash a file found at its length.
    Check(usize, PathBuf),
    /// Download a file as it is, into the store.
    Fetch(usize),
    /// Download a part of a file to pack.
    FetchPart(usize, usize),
    /// Pack the parts, check the result.
    Pack(usize),
    /// Remove the parts.
    Clean(usize),
}

fn megabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{} MB", bytes.div_ceil(1_000_000))
    } else {
        format!("{} KB", bytes.div_ceil(1_000).max(1))
    }
}

/// The folder a packed file's parts are downloaded into.
fn parts_dir(places: &Places, need: &Need) -> PathBuf {
    places.store.join(format!("{}.download", need.file))
}

fn capital(text: &str) -> String {
    let mut c = text.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

fn step_of(task: &Task, manifest: &[Need]) -> Step {
    let need = |i: usize| &manifest[i];
    let (title, why, weight) = match task {
        Task::Check(i, _) => (
            format!(
                "Checking {} ({}) against its built-in checksum",
                need(*i).title,
                megabytes(need(*i).bytes)
            ),
            need(*i).why.clone(),
            need(*i).bytes as f64 * HASH_WEIGHT,
        ),
        Task::Fetch(i) => (
            format!(
                "Downloading {} ({}, {})",
                need(*i).title,
                need(*i).credit,
                megabytes(need(*i).bytes)
            ),
            need(*i).why.clone(),
            need(*i).bytes as f64,
        ),
        Task::FetchPart(i, p) => {
            let Source::Pack { parts, .. } = &need(*i).source else {
                unreachable!("a part of a file not packed")
            };
            let part = &parts[*p];
            (
                // The first part says whose it is, as a file got whole does.
                if *p == 0 {
                    format!(
                        "Downloading {} ({}, {})",
                        part.title,
                        need(*i).credit,
                        megabytes(part.bytes)
                    )
                } else {
                    format!("Downloading {} ({})", part.title, megabytes(part.bytes))
                },
                if *p == 0 {
                    need(*i).why.clone()
                } else {
                    format!("Part of {}.", need(*i).title)
                },
                part.bytes as f64,
            )
        }
        Task::Pack(i) => {
            let Source::Pack { why, .. } = &need(*i).source else {
                unreachable!("packing a file not packed")
            };
            (
                format!("Packing {} at 4 bits, then checking it", need(*i).title),
                why.clone(),
                PACK_WEIGHT,
            )
        }
        Task::Clean(i) => {
            let Source::Pack { parts, .. } = &need(*i).source else {
                unreachable!("cleaning after a file not packed")
            };
            let all: u64 = parts.iter().map(|p| p.bytes).sum();
            (
                format!("Removing the download ({})", megabytes(all)),
                "Only the packed file is kept.".to_owned(),
                1.0,
            )
        }
    };
    Step {
        title: capital(&title),
        why,
        state: StepState::ToDo,
        weight,
    }
}

/// The tasks that get `need` (number `i`) into the store.
fn obtain(i: usize, need: &Need) -> Vec<Task> {
    match &need.source {
        Source::Download { .. } => vec![Task::Fetch(i)],
        Source::Pack { parts, .. } => (0..parts.len())
            .map(|p| Task::FetchPart(i, p))
            .chain([Task::Pack(i), Task::Clean(i)])
            .collect(),
    }
}

/// The bytes the store needs free for `tasks`: what is downloaded and what
/// is made (a packed file's parts are there with it until removed).
fn space_for(tasks: &[Task], manifest: &[Need]) -> u64 {
    tasks
        .iter()
        .map(|t| match t {
            Task::Fetch(i) | Task::Pack(i) => manifest[*i].bytes,
            Task::FetchPart(i, p) => match &manifest[*i].source {
                Source::Pack { parts, .. } => parts[*p].bytes,
                Source::Download { .. } => 0,
            },
            Task::Check(..) | Task::Clean(_) => 0,
        })
        .sum()
}

/// Hashes `path`, telling `progress` the bytes so far.
fn hash_file(path: &Path, progress: &mut dyn FnMut(u64)) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        done += n as u64;
        progress(done);
    }
    Ok(hash.hex())
}

struct Shared {
    status: Mutex<Status>,
    /// Set by [`Setup::retry`], taken by the worker waiting after a failure.
    again: Mutex<bool>,
    told: Condvar,
    notify: Notify,
}

/// Setup under way, on a thread of its own.
pub struct Setup {
    shared: Arc<Shared>,
}

impl Setup {
    /// Starts getting what `manifest` needs into `places`, through `work`,
    /// telling `notify` as it goes.
    pub fn start(
        manifest: Vec<Need>,
        places: Places,
        work: Arc<dyn Work>,
        notify: Notify,
    ) -> Setup {
        let shared = Arc::new(Shared {
            status: Mutex::new(Status::default()),
            again: Mutex::new(false),
            told: Condvar::new(),
            notify,
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("setup".into())
            .spawn(move || run(&manifest, &places, &*work, &worker))
            .expect("a thread for setup");
        Setup { shared }
    }

    /// Where it stands now.
    pub fn status(&self) -> Status {
        self.shared
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// After a failure: tries again from where it stopped.
    pub fn retry(&self) {
        let mut again = self.shared.again.lock().unwrap_or_else(|e| e.into_inner());
        *again = true;
        self.shared.told.notify_all();
    }
}

/// Tells the screen, at most every 30 ms between steps' ends.
struct Teller<'a> {
    shared: &'a Shared,
    last: Instant,
}

impl Teller<'_> {
    fn set(&mut self, force: bool, f: impl FnOnce(&mut Status)) {
        f(&mut self.shared.status.lock().unwrap_or_else(|e| e.into_inner()));
        if force || self.last.elapsed() >= Duration::from_millis(30) {
            self.last = Instant::now();
            (self.shared.notify)();
        }
    }
}

fn run(manifest: &[Need], places: &Places, work: &dyn Work, shared: &Shared) {
    loop {
        let mut teller = Teller {
            shared,
            last: Instant::now(),
        };
        match attempt(manifest, places, work, &mut teller) {
            Ok(paths) => {
                teller.set(true, |s| {
                    s.current = None;
                    s.finished = Some(paths);
                });
                return;
            }
            Err(failure) => {
                teller.set(true, |s| s.failure = Some(failure));
                let mut again = shared.again.lock().unwrap_or_else(|e| e.into_inner());
                while !*again {
                    again = shared.told.wait(again).unwrap_or_else(|e| e.into_inner());
                }
                *again = false;
                drop(again);
                teller.set(true, |s| s.failure = None);
            }
        }
    }
}

fn failure(what: impl Into<String>, advice: impl Into<String>) -> Failure {
    Failure {
        what: what.into(),
        advice: advice.into(),
    }
}

/// What to do when the network let a step down.
const NETWORK: &str = "Check that this computer is online, then try again. Once the files \
                       are here, the player never needs the network for them again.";

/// One go at getting everything: the paths, or why it stopped.
fn attempt(
    manifest: &[Need],
    places: &Places,
    work: &dyn Work,
    teller: &mut Teller,
) -> Result<Vec<PathBuf>, Failure> {
    let mut paths: Vec<Option<PathBuf>> = vec![None; manifest.len()];
    let mut tasks = Vec::new();
    for (i, need) in manifest.iter().enumerate() {
        match look(need, places) {
            Lookup::Checked(p) => paths[i] = Some(p),
            Lookup::Unchecked(p) => tasks.push(Task::Check(i, p)),
            Lookup::Missing => tasks.extend(obtain(i, need)),
        }
    }
    teller.set(true, |s| {
        *s = Status {
            steps: tasks.iter().map(|t| step_of(t, manifest)).collect(),
            ..Status::default()
        }
    });
    let mut k = 0;
    while k < tasks.len() {
        let task = tasks[k].clone();
        // Before the first download, the room on the disk for all of them.
        if !matches!(task, Task::Check(..))
            && !tasks[..k].iter().any(|t| !matches!(t, Task::Check(..)))
        {
            let need = space_for(&tasks[k..], manifest);
            if let Some(free) = work.free_space(&places.store) {
                if free < need + need / 20 {
                    return Err(failure(
                        format!(
                            "There is not enough free space on the disk {} is on: setup needs {}, there are {} free.",
                            places.store.display(),
                            megabytes(need + need / 20),
                            megabytes(free)
                        ),
                        "Free some space on that disk, then try again.",
                    ));
                }
            }
        }
        teller.set(true, |s| {
            s.current = Some(k);
            s.steps[k].state = StepState::Working;
            s.done = 0;
            s.total = 0;
            s.unit = None;
            s.rate = None;
        });
        match &task {
            Task::Check(i, path) => {
                // Where it is as it is opened: the data folder may have
                // moved it since it was found.
                let path = &crate::mover::locate(path);
                let need = &manifest[*i];
                teller.set(true, |s| {
                    s.total = need.bytes;
                    s.unit = Some(Unit::Bytes);
                });
                let digest = hash_file(path, &mut |done| teller.set(false, |s| s.done = done))
                    .map_err(|e| failure(e, "Check that the disk can be read, then try again."))?;
                if digest == need.sha256 {
                    // Where it is now: the data folder may have moved it
                    // while it was hashed (the same file, renamed).
                    let at = crate::mover::locate(path);
                    stamp(&places.store, &need.sha256, &at)
                        .map_err(|e| failure(e, "Check that the data folder can be written to."))?;
                    paths[*i] = Some(at);
                } else {
                    // Not the file: got afresh into the store (one there is
                    // ours to replace; one elsewhere is passed over).
                    if path.starts_with(&places.store) {
                        let _ = fs::remove_file(path);
                    }
                    let more = obtain(*i, need);
                    teller.set(true, |s| {
                        let steps: Vec<Step> = more.iter().map(|t| step_of(t, manifest)).collect();
                        s.steps.splice(k + 1..k + 1, steps);
                    });
                    tasks.splice(k + 1..k + 1, more);
                }
            }
            Task::Fetch(i) => {
                let need = &manifest[*i];
                let Source::Download { url } = &need.source else {
                    unreachable!("fetching a file that is packed")
                };
                let part = Part {
                    name: need.file.clone(),
                    url: url.clone(),
                    bytes: need.bytes,
                    sha256: need.sha256.clone(),
                    title: need.title.clone(),
                };
                let to = places.store.join(&need.file);
                fetch(work, &part, &to, teller)?;
                stamp(&places.store, &need.sha256, &to).map_err(|e| {
                    failure(e, "Check that the settings' folder can be written to.")
                })?;
                paths[*i] = Some(to);
            }
            Task::FetchPart(i, p) => {
                let need = &manifest[*i];
                let Source::Pack { parts, .. } = &need.source else {
                    unreachable!("a part of a file not packed")
                };
                let part = &parts[*p];
                let to = parts_dir(places, need).join(&part.name);
                // A part from a run that stopped is kept if it is right.
                let kept = fs::metadata(&to).is_ok_and(|m| m.len() == part.bytes) && {
                    teller.set(true, |s| {
                        s.total = part.bytes;
                        s.unit = Some(Unit::Bytes);
                    });
                    hash_file(&to, &mut |done| teller.set(false, |s| s.done = done))
                        .is_ok_and(|d| d == part.sha256)
                };
                if !kept {
                    fetch(work, part, &to, teller)?;
                }
            }
            Task::Pack(i) => {
                let need = &manifest[*i];
                if let Some(memory) = work.memory() {
                    if memory < PACK_MEMORY {
                        return Err(failure(
                            format!(
                                "Packing {} needs about {} of memory; this computer has {}.",
                                need.title,
                                megabytes(PACK_MEMORY),
                                megabytes(memory)
                            ),
                            format!(
                                "Copy the folder {} from a computer that has set the player up into {}, then try again.",
                                models::MODEL_DIR,
                                places.store.display()
                            ),
                        ));
                    }
                }
                let dir = parts_dir(places, need);
                let partial = places.store.join(format!("{}.partial", need.file));
                teller.set(true, |s| s.unit = Some(Unit::Tensors));
                work.pack(&dir, &partial, &mut |done, all| {
                    teller.set(false, |s| {
                        s.done = done;
                        s.total = all;
                    })
                })
                .map_err(|e| {
                    failure(
                        format!("Packing {} failed: {e}", need.title),
                        "Try again; if it fails again, the download may be damaged: it is fetched afresh then.",
                    )
                })?;
                teller.set(true, |s| {
                    s.done = 0;
                    s.total = need.bytes;
                    s.unit = Some(Unit::Bytes);
                });
                let digest = hash_file(&partial, &mut |done| teller.set(false, |s| s.done = done))
                    .map_err(|e| failure(e, "Check that the disk can be read, then try again."))?;
                if digest != need.sha256 {
                    let _ = fs::remove_file(&partial);
                    let _ = fs::remove_dir_all(&dir);
                    return Err(failure(
                        format!(
                            "{} came out different from what it should be (SHA-256 {digest}).",
                            capital(&need.title)
                        ),
                        "Try again: the download is fetched afresh. If it comes out different again, this build of the player packs differently from the one its checksum was made with.",
                    ));
                }
                let to = places.store.join(&need.file);
                durable::rename_durable(&partial, &to)
                    .map_err(|e| failure(format!("{}: {e}", to.display()), "Try again."))?;
                stamp(&places.store, &need.sha256, &to).map_err(|e| {
                    failure(e, "Check that the settings' folder can be written to.")
                })?;
                paths[*i] = Some(to);
            }
            Task::Clean(i) => {
                let _ = fs::remove_dir_all(parts_dir(places, &manifest[*i]));
            }
        }
        teller.set(true, |s| {
            s.steps[k].state = StepState::Done;
            s.done = s.total;
        });
        k += 1;
    }
    paths
        .into_iter()
        .zip(manifest)
        .map(|(p, need)| {
            p.ok_or_else(|| {
                failure(
                    format!("{} is still missing.", capital(&need.title)),
                    "Try again.",
                )
            })
        })
        .collect()
}

/// Downloads `part` to `to`, telling the screen the bytes and the speed.
fn fetch(work: &dyn Work, part: &Part, to: &Path, teller: &mut Teller) -> Result<(), Failure> {
    teller.set(true, |s| {
        s.total = part.bytes;
        s.unit = Some(Unit::Bytes);
    });
    let started = Instant::now();
    work.download(part, to, &mut |done| {
        let seconds = started.elapsed().as_secs_f64();
        teller.set(false, |s| {
            s.done = done;
            s.rate = (seconds > 1.0).then(|| done as f64 / seconds);
        })
    })
    .map_err(|e| failure(format!("Downloading {} failed: {e}", part.title), NETWORK))
}

/// The disk's free space and the memory, from the system.
mod system {
    use std::path::Path;

    /// The bytes free on the disk `dir` is on (or the nearest folder
    /// above it that exists).
    pub fn free_space(dir: &Path) -> Option<u64> {
        let mut at = dir;
        while !at.exists() {
            at = at.parent()?;
        }
        free_at(at)
    }

    #[cfg(windows)]
    fn free_at(dir: &Path) -> Option<u64> {
        use std::os::windows::ffi::OsStrExt;
        use windows::core::PCWSTR;
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let mut free = 0u64;
        // SAFETY: a NUL-terminated path and a u64 to write into.
        unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut free), None, None) }.ok()?;
        Some(free)
    }

    #[cfg(unix)]
    fn free_at(dir: &Path) -> Option<u64> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        // SAFETY: a NUL-terminated path and a statvfs to write into.
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        (unsafe { libc::statvfs(path.as_ptr(), &mut stat) } == 0)
            .then(|| stat.f_bavail as u64 * stat.f_frsize as u64)
    }

    #[cfg(not(any(windows, unix)))]
    fn free_at(_dir: &Path) -> Option<u64> {
        None
    }

    /// The computer's memory in bytes.
    #[cfg(windows)]
    pub fn memory() -> Option<u64> {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: a MEMORYSTATUSEX with its length set.
        unsafe { GlobalMemoryStatusEx(&mut status) }.ok()?;
        Some(status.ullTotalPhys)
    }

    /// The computer's memory in bytes.
    #[cfg(target_os = "macos")]
    pub fn memory() -> Option<u64> {
        let mut bytes = 0u64;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: a NUL-terminated name and a u64 of its length.
        let ok = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut bytes as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } == 0;
        ok.then_some(bytes)
    }

    /// The computer's memory in bytes.
    #[cfg(all(unix, not(target_os = "macos")))]
    pub fn memory() -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }

    /// The computer's memory in bytes.
    #[cfg(not(any(windows, unix)))]
    pub fn memory() -> Option<u64> {
        None
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: the stand-in downloader and packer, and model files where an older player left them"
)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
            if self.offline.load(Ordering::SeqCst) {
                return Err("no network".into());
            }
            self.downloads.fetch_add(1, Ordering::SeqCst);
            let data = &self.web.iter().find(|(u, _)| *u == part.url).unwrap().1;
            progress(data.len() as u64);
            if data.len() as u64 != part.bytes || sha(data) != part.sha256 {
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
        let part = |name: &str, data: &[u8]| Part {
            name: name.into(),
            url: format!("https://example.test/{name}"),
            bytes: data.len() as u64,
            sha256: sha(data),
            title: format!("part {name}"),
        };
        let manifest = vec![
            Need {
                file: "speech.bin".into(),
                bytes: speech.len() as u64,
                sha256: sha(&speech),
                title: "the speech model".into(),
                credit: "us".into(),
                why: "timing".into(),
                source: Source::Download {
                    url: "https://example.test/speech".into(),
                },
            },
            Need {
                file: "asr.bin".into(),
                bytes: packed.len() as u64,
                sha256: sha(&packed),
                title: "the recogniser".into(),
                credit: "us".into(),
                why: "checking".into(),
                source: Source::Pack {
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

    /// Runs setup to its end or a failure.
    fn settle(setup: &Setup) -> Status {
        for _ in 0..500 {
            let s = setup.status();
            if s.finished.is_some() || s.failure.is_some() {
                return s;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("setup did not settle: {:?}", setup.status());
    }

    #[test]
    fn the_built_in_manifest_names_the_four_models() {
        assert_eq!(manifest_for(false).len(), 2);
        let m = manifest_for(true);
        assert_eq!(m.len(), 4);
        assert_eq!(m[0].file, models::MODEL_FILE);
        assert_eq!(m[1].file, models::ASR_FILE);
        assert_eq!(m[2].file, crate::cover::DEPTH_FILE);
        assert_eq!(m[3].file, crate::cover::MATTE_FILE);
        for need in &m {
            assert_eq!(need.sha256.len(), 64);
            assert!(need.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn everything_is_got_checked_and_stamped_then_nothing_more() {
        let dir = temp("all");
        let (manifest, work) = fake();
        let work = Arc::new(work);
        let places = places(&dir);
        assert_eq!(ready(&manifest, &places), None);
        let told = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&told);
        let setup = Setup::start(
            manifest.clone(),
            places.clone(),
            work.clone(),
            Arc::new(move || {
                t.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let s = settle(&setup);
        assert_eq!(s.failure, None);
        let paths = s.finished.clone().unwrap();
        assert_eq!(
            paths,
            vec![dir.join("store/speech.bin"), dir.join("store/asr.bin")]
        );
        assert_eq!(fs::read(&paths[1]).unwrap(), b"part a, part b");
        // Speech, part a, part b, the packing, the cleaning.
        assert_eq!(s.steps.len(), 5);
        assert!(s.steps.iter().all(|st| st.state == StepState::Done));
        assert_eq!(s.overall(), 1.0);
        assert!(told.load(Ordering::SeqCst) > 0);
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
        let setup = Setup::start(
            manifest.clone(),
            places.clone(),
            work.clone(),
            Arc::new(|| {}),
        );
        let s = settle(&setup);
        let paths = s.finished.expect("finished");
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
        // Two checks, then the recogniser's parts, packing, cleaning.
        assert_eq!(s.steps.len(), 6);
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
        let setup = Setup::start(manifest, places(&dir), work.clone(), Arc::new(|| {}));
        let s = settle(&setup);
        let f = s.failure.expect("a failure");
        assert!(f.what.contains("no network"), "{f:?}");
        assert!(f.advice.contains("online"), "{f:?}");
        assert_eq!(s.finished, None);
        // Still offline: it fails again.
        setup.retry();
        std::thread::sleep(Duration::from_millis(50));
        assert!(settle(&setup).failure.is_some());
        work.offline.store(false, Ordering::SeqCst);
        setup.retry();
        std::thread::sleep(Duration::from_millis(50));
        let s = settle(&setup);
        assert_eq!(s.failure, None);
        assert!(s.finished.is_some());
    }

    #[test]
    fn too_little_disk_or_memory_is_said_before_anything_is_fetched() {
        let dir = temp("room");
        let (manifest, mut work) = fake();
        work.free = Some(10);
        let work = Arc::new(work);
        let setup = Setup::start(
            manifest.clone(),
            places(&dir),
            work.clone(),
            Arc::new(|| {}),
        );
        let f = settle(&setup).failure.expect("no room");
        assert!(f.what.contains("not enough free space"), "{f:?}");
        assert_eq!(work.downloads.load(Ordering::SeqCst), 0);

        let dir = temp("memory");
        let (manifest, mut work) = fake();
        work.memory = Some(4 << 30);
        let setup = Setup::start(manifest, places(&dir), Arc::new(work), Arc::new(|| {}));
        let f = settle(&setup).failure.expect("too little memory");
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
        crate::mover::begin(
            &dir.join("settings"),
            &[(old.clone(), places.store.clone())],
        )
        .unwrap();
        assert_eq!(
            look(&manifest[0], &places),
            Lookup::Unchecked(old.join("speech.bin"))
        );
        let work = Arc::new(work);
        let setup = Setup::start(
            manifest.clone(),
            places.clone(),
            work.clone(),
            Arc::new(|| {}),
        );
        let s = settle(&setup);
        assert_eq!(s.failure, None);
        assert_eq!(
            s.finished.unwrap(),
            vec![old.join("speech.bin"), old.join("asr.bin")]
        );
        assert_eq!(work.downloads.load(Ordering::SeqCst), 0, "nothing fetched");
        // The mover brings them over: checked once where they went.
        fs::create_dir_all(&places.store).unwrap();
        for name in ["speech.bin", "asr.bin"] {
            fs::rename(old.join(name), places.store.join(name)).unwrap();
        }
        assert_eq!(ready(&manifest, &places), None);
        let setup = Setup::start(
            manifest.clone(),
            places.clone(),
            work.clone(),
            Arc::new(|| {}),
        );
        let s = settle(&setup);
        assert_eq!(
            s.finished.unwrap(),
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
        let setup = Setup::start(
            manifest.clone(),
            places.clone(),
            Arc::new(work),
            Arc::new(|| {}),
        );
        settle(&setup).finished.expect("finished");
        // Rewritten, the same length: not taken without a check.
        let path = dir.join("store/speech.bin");
        std::thread::sleep(Duration::from_millis(20));
        fs::write(&path, b"the speech MODEL").unwrap();
        assert_eq!(look(&manifest[0], &places), Lookup::Unchecked(path));
        assert_eq!(ready(&manifest, &places), None);
    }
}
