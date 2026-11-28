//! The player's self-check, and its setup when that fails.
//!
//! Most of what the player needs is inside its executable (the beat and key
//! networks, the stem separator, the singer's isolator, the fonts). The
//! models too big for it are got here ([`manifest`]): the speech model that
//! times karaoke's words ([`crate::work::lyrics::models::MODEL_FILE`]), the
//! speech recogniser that checks fetched lyrics belong to the song
//! ([`crate::work::lyrics::models::ASR_FILE`]), and where the machine has a
//! usable GPU karaoke story mode's language and image models
//! ([`crate::work::story_models`]) and the depth and cut-out models that
//! take its scenes apart ([`crate::work::scenes::DEPTH_FILE`],
//! [`crate::work::scenes::MATTE_FILE`]). A model is one file,
//! or a folder of files as its repository has them; every file's name,
//! length and SHA-256 is built in.
//!
//! At every start [`Setup`], on a thread of its own, chooses the graphics
//! card (the manifest depends on it) and looks for them ([`look`]) in
//! `models` beside the executable, in a macOS app's `Resources/models`,
//! and in `models` beside the settings ([`Places`]). A file is taken once
//! it has matched its checksum: the first time it is hashed, and a stamp
//! (its length and when it was written) spares later starts the hashing.
//! When every file is there and stamped, nothing is got and the player
//! never goes online for them. A file missing or of another length is got
//! again alone; while the player runs, one that cannot be read is checked
//! and got again ([`Repair`]).
//!
//! Otherwise setup gets them. Then the player's own start ([`Then`]): its
//! steps follow the files' on the same screen. The window shows it from
//! its first frame ([`crate::view::setup`]), and it cannot be skipped:
//! each step said with what it does and why, a bar for a file's step and
//! one for the whole. A failure (no network, a download not what it should
//! be, too little disk space or memory) is said with what to do, and the
//! player can only try again or quit: without the files, the player is
//! not what it should be.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::work::lyrics::models;
use crate::work::outbox::Notify;
use crate::work::{Job, Teller};

mod manifest;
mod repair;
mod stamp;
mod status;
mod system;
#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: the stand-in downloader and packer, and model files where an older player left them"
)]
mod tests;

pub use crate::work::download::Fault;
pub use manifest::{manifest, manifest_for, under, Found, Need, Part, Source, Sum};
pub use repair::{Repair, Unreadable};
use stamp::{forget, hashes_to, known, present, stamp, verified, STORE};
use status::{failure, lasting};
pub use status::{size, Failure, Paths, Status, Step, StepState, Unit};

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
    pub fn player(data: &crate::store::dirs::Data) -> Places {
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

/// Where a needed model was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// There and checked (stamped) as it is: its path.
    Checked(PathBuf),
    /// There at its lengths, not checked as it is: to be hashed.
    Unchecked(PathBuf),
    /// Nowhere whole at its lengths.
    Missing,
}

/// Where `need` is: a copy checked anywhere first, else the first whole at
/// its lengths (a copy beside the executable that is wrong is passed over
/// once a good one is in the store). A model's folder is taken from one
/// place, all its files there. The path is where it is now
/// ([`crate::store::mover::locate`]).
pub fn look(need: &Need, places: &Places) -> Lookup {
    let whole: Vec<(PathBuf, bool)> = places
        .look
        .iter()
        .filter_map(|d| {
            let at = need.at(d);
            let expected = need.expected(&at);
            expected
                .iter()
                .all(|(path, sum)| present(path, sum))
                .then(|| {
                    let checked = expected
                        .iter()
                        .all(|(path, sum)| known(&places.store, path, sum));
                    (crate::store::mover::locate(&at), checked)
                })
        })
        .collect();
    if let Some((at, _)) = whole.iter().find(|(_, checked)| *checked) {
        return Lookup::Checked(at.clone());
    }
    whole
        .into_iter()
        .next()
        .map_or(Lookup::Missing, |(at, _)| Lookup::Unchecked(at))
}

/// What setup needs of the system: downloads, packing, the disk and the
/// memory. The player's is [`Real`]; tests make their own.
pub trait Work: Send + Sync {
    /// Downloads `part` to `to` (checked against its length and SHA-256
    /// before it takes the name), telling `progress` the bytes so far; why
    /// not, the web's fault or the disk's.
    fn download(&self, part: &Part, to: &Path, progress: &mut dyn FnMut(u64)) -> Result<(), Fault>;
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

/// The bits each weight of a packed checkpoint is held in.
pub const PACK_BITS: u32 = 4;

/// The player's: from the web, packed by [`jodel_lyrics::asr::packed`].
pub struct Real;

impl Work for Real {
    fn download(&self, part: &Part, to: &Path, progress: &mut dyn FnMut(u64)) -> Result<(), Fault> {
        crate::work::download::download(
            &part.url,
            to,
            part.sum.bytes,
            &part.sum.sha256,
            |done, _| progress(done),
        )
    }

    fn pack(
        &self,
        parts: &Path,
        to: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String> {
        use jodel_lyrics::asr::packed::PackedModel;
        let statistics = jodel_lyrics::asr::quant::Statistics::built_in();
        let packed = PackedModel::from_checkpoint(
            parts,
            |_| PACK_BITS,
            Some(&statistics),
            |done, all| progress(done as u64, all as u64),
        )?;
        durable::write_atomic(to, &packed.to_bytes()).map_err(|e| format!("{}: {e}", to.display()))
    }

    fn free_space(&self, dir: &Path) -> Option<u64> {
        system::free_space(dir)
    }

    fn memory(&self) -> Option<u64> {
        system::memory()
    }
}

/// A step of the player's own start weighs as much as hashing this many
/// bytes: next to a download it is nothing, alone the steps share the bar.
const START_WEIGHT: f64 = 1e6;

/// The player's own start, once its files are here ([`Setup::start`]):
/// its steps, in order, and what does them, given every needed model's
/// path and the repair of the models while the player runs ([`Repair`]),
/// and told as each step after the first begins.
pub struct Then {
    /// Each step's title and why.
    pub steps: Vec<(String, String)>,
    /// Does the steps.
    pub run: Starts,
}

/// What does the player's own start's steps ([`Then::run`]).
pub type Starts = Box<dyn FnOnce(Paths, Arc<Repair>, &mut dyn FnMut()) + Send>;

/// How long hashing a byte takes against downloading one, for the weights.
const HASH_WEIGHT: f64 = 0.05;
/// Packing's weight, in downloaded bytes' worth (about a minute).
const PACK_WEIGHT: f64 = 1.5e9;

/// One thing setup does.
#[derive(Clone, Debug)]
enum Task {
    /// Hash a model found whole at its lengths (at its path).
    Check(usize, PathBuf),
    /// Download those of a model's files (by their place in
    /// [`Need::files`]) not in the store yet: its file or its folder's, or
    /// the checkpoint a packed file is made from.
    Fetch(usize, Vec<usize>),
    /// Pack the checkpoint, check the result.
    Pack(usize),
    /// Remove the checkpoint.
    Clean(usize),
}

impl Task {
    /// Whether it writes to the store (it needs room on the disk).
    fn writes(&self) -> bool {
        matches!(self, Task::Fetch(..) | Task::Pack(_))
    }
}

/// The folder a packed file's checkpoint is downloaded into.
fn parts_dir(places: &Places, need: &Need) -> PathBuf {
    under(&places.store, &format!("{}.download", need.file))
}

/// Where `need`'s downloads go: the model's file or folder in the store,
/// or a packed file's checkpoint folder.
fn fetched_at(places: &Places, need: &Need) -> PathBuf {
    if need.packed().is_some() {
        parts_dir(places, need)
    } else {
        need.at(&places.store)
    }
}

/// The bytes of those of `need`'s files that `which` names.
fn files_bytes(need: &Need, which: impl IntoIterator<Item = usize>) -> u64 {
    let files = need.files(Path::new(""));
    which.into_iter().map(|f| files[f].1.sum.bytes).sum()
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
        Task::Check(i, at) => (
            format!(
                "Checking {} ({}) against its built-in {}",
                need(*i).title,
                size(need(*i).bytes()),
                if need(*i).expected(at).len() > 1 {
                    "checksums"
                } else {
                    "checksum"
                }
            ),
            need(*i).why.clone(),
            need(*i).bytes() as f64 * HASH_WEIGHT,
        ),
        Task::Fetch(i, missing) => {
            let bytes = files_bytes(need(*i), missing.iter().copied());
            (
                format!(
                    "Downloading {} ({}, {})",
                    need(*i).title,
                    need(*i).credit,
                    size(bytes)
                ),
                need(*i).why.clone(),
                bytes as f64,
            )
        }
        Task::Pack(i) => (
            format!(
                "Packing {} at {PACK_BITS} bits, then checking it",
                need(*i).title
            ),
            need(*i)
                .packed()
                .map(|(_, why)| why.to_owned())
                .unwrap_or_default(),
            PACK_WEIGHT,
        ),
        Task::Clean(i) => (
            format!(
                "Removing the download ({})",
                size(need(*i).downloaded_bytes())
            ),
            "Only the packed file is kept.".to_owned(),
            1.0,
        ),
    };
    Step {
        weight,
        ..Step::new(capital(&title), why, StepState::ToDo)
    }
}

/// The tasks that get `need` (number `i`) into the store: its files not
/// there and known downloaded (a packed file's checkpoint, then packed and
/// removed).
fn obtain(i: usize, need: &Need, places: &Places) -> Vec<Task> {
    let missing: Vec<usize> = need
        .files(&fetched_at(places, need))
        .iter()
        .enumerate()
        .filter(|(_, (path, part))| !known(&places.store, path, &part.sum))
        .map(|(f, _)| f)
        .collect();
    let mut tasks = Vec::new();
    if !missing.is_empty() {
        tasks.push(Task::Fetch(i, missing));
    }
    if need.packed().is_some() {
        tasks.extend([Task::Pack(i), Task::Clean(i)]);
    }
    tasks
}

/// The bytes the store needs free for `tasks`: what is downloaded and what
/// is made (a packed file's checkpoint is there with it until removed).
fn space_for(tasks: &[Task], manifest: &[Need]) -> u64 {
    tasks
        .iter()
        .map(|t| match t {
            Task::Fetch(i, missing) => files_bytes(&manifest[*i], missing.iter().copied()),
            Task::Pack(i) => manifest[*i].bytes(),
            Task::Check(..) | Task::Clean(_) => 0,
        })
        .sum()
}

/// Setup under way, on a thread of its own.
pub struct Setup {
    job: Arc<Job<Status>>,
}

impl Setup {
    /// Starts getting what `manifest` says is needed (it may take a while:
    /// the player's chooses the graphics card) into `places`, through
    /// `work`, then the player's own start (`then`), telling `notify` as it
    /// goes. All of it on a thread of its own: it returns at once.
    pub fn start(
        manifest: impl FnOnce() -> Vec<Need> + Send + 'static,
        places: Places,
        work: Arc<dyn Work>,
        notify: Notify,
        then: Option<Then>,
    ) -> Setup {
        let (steps, then) = then.map_or((Vec::new(), None), |t| (t.steps, Some(t.run)));
        let status = Status {
            looking: true,
            steps: steps
                .into_iter()
                .map(|(title, why)| Step {
                    start: true,
                    weight: START_WEIGHT,
                    ..Step::new(title, why, StepState::ToDo)
                })
                .collect(),
            ..Status::default()
        };
        let job = Job::new(status, notify);
        let worker = Arc::clone(&job);
        std::thread::Builder::new()
            .name("setup".into())
            .spawn(move || run(manifest, &places, work, &worker, then))
            .expect("a thread for setup");
        Setup { job }
    }

    /// Where it stands now.
    pub fn status(&self) -> Status {
        self.job.status()
    }

    /// After a failure: tries again from where it stopped.
    pub fn retry(&self) {
        self.job.retry();
    }
}

fn run(
    manifest: impl FnOnce() -> Vec<Need>,
    places: &Places,
    work: Arc<dyn Work>,
    job: &Job<Status>,
    then: Option<Starts>,
) {
    // Tells the screen, at most every 30 ms between steps' ends.
    let mut teller = job.teller(Duration::from_millis(30));
    let started = Instant::now();
    let manifest = manifest();
    crate::trace::took("setup: the files known", started, true);
    let paths = loop {
        match attempt(&manifest, places, &*work, &mut teller) {
            Ok(paths) => break paths,
            Err(failure) => {
                teller.set(true, |s| s.failure = Some(failure));
                job.wait_for_retry();
                teller.set(true, |s| s.failure = None);
            }
        }
    };
    forget_partials(&manifest, places);
    teller.set(true, |s| s.current = None);
    if let Some(then) = then {
        let repair = Repair::new(manifest, places.clone(), work, &paths);
        start_player(then, paths.clone(), repair, &mut teller);
    }
    teller.set(true, |s| {
        s.current = None;
        s.finished = Some(paths);
    });
}

/// Removes the downloads cut off in the store ([`durable::Resumable`])
/// that no file of `manifest` goes on from (a model no longer needed on
/// this machine, or renamed by a later player): kept on purpose to be gone
/// on from, they would otherwise stay for good.
fn forget_partials(manifest: &[Need], places: &Places) {
    let wanted: std::collections::HashSet<PathBuf> = (manifest.iter())
        .flat_map(|need| need.files(&fetched_at(places, need)))
        .map(|(path, _)| durable::Resumable::part_of(&path))
        .collect();
    let mut dirs = vec![places.store.clone()];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let (path, kind) = (entry.path(), entry.file_type());
            // Not followed through a link.
            if kind.as_ref().is_ok_and(fs::FileType::is_dir) {
                dirs.push(path);
            } else if path.to_string_lossy().ends_with(durable::RESUMABLE)
                && !wanted.contains(&path)
            {
                crate::trace::note(format_args!(
                    "setup: {} is no model's: removed",
                    path.display()
                ));
                let _ = fs::remove_file(&path);
            }
        }
    }
}

/// The player's own start, its steps (after the files') told as they go.
fn start_player(then: Starts, paths: Paths, repair: Arc<Repair>, teller: &mut Teller<'_, Status>) {
    let mut begun = Instant::now();
    // The step under way done, the next one begun.
    let mut next = || {
        teller.set(true, |s| {
            let k = match s.current {
                Some(k) => {
                    s.steps[k].state = StepState::Done;
                    crate::trace::took(format_args!("setup: {}", s.steps[k].title), begun, true);
                    k + 1
                }
                None => s
                    .steps
                    .iter()
                    .position(|st| st.start)
                    .unwrap_or(s.steps.len()),
            };
            s.current = (k < s.steps.len()).then_some(k);
            if let Some(step) = s.steps.get_mut(k) {
                step.state = StepState::Working;
            }
        });
        begun = Instant::now();
    };
    next();
    then(paths, repair, &mut next);
    next();
}

/// What to do when the network let a step down.
const NETWORK: &str = "Check that this computer is online, then try again. Once the files \
                       are here, the player never needs the network for them again.";

/// What to do when what the server sends is not the file built in.
const WRONG: &str = "The file where it is downloaded from is not the one this player \
                     checks against. Try again later; if it stays so, a newer player may \
                     know the new file.";

/// One go at getting everything: the paths, or why it stopped. It looks
/// (every model's path known from then on: where it was found, else where
/// setup puts it), lays the steps out, checks the disk's room before the
/// first that writes, and does each step in turn.
fn attempt(
    manifest: &[Need],
    places: &Places,
    work: &dyn Work,
    teller: &mut Teller<'_, Status>,
) -> Result<Paths, Failure> {
    let mut paths = Vec::with_capacity(manifest.len());
    let mut tasks = Vec::new();
    for (i, need) in manifest.iter().enumerate() {
        paths.push(match look(need, places) {
            Lookup::Checked(p) => p,
            Lookup::Unchecked(p) => {
                tasks.push(Task::Check(i, p.clone()));
                p
            }
            Lookup::Missing => {
                tasks.extend(obtain(i, need, places));
                need.at(&places.store)
            }
        });
    }
    // The files' steps, before the player's own start's.
    teller.set(true, |s| {
        s.steps.retain(|step| step.start);
        s.steps
            .splice(0..0, tasks.iter().map(|t| step_of(t, manifest)));
        s.looking = false;
        s.begin(None);
    });
    let mut room_checked = false;
    let mut k = 0;
    while k < tasks.len() {
        let task = tasks[k].clone();
        // Before the first that writes, the room on the disk for all.
        if task.writes() && !room_checked {
            room(&tasks[k..], manifest, places, work)?;
            room_checked = true;
        }
        teller.set(true, |s| s.begin(Some(k)));
        match &task {
            Task::Check(i, at) => {
                if !check_need(&manifest[*i], at, places, teller)? {
                    // Not the files: got afresh into the store.
                    paths[*i] = manifest[*i].at(&places.store);
                    let more = obtain(*i, &manifest[*i], places);
                    teller.set(true, |s| {
                        let steps: Vec<Step> = more.iter().map(|t| step_of(t, manifest)).collect();
                        s.steps.splice(k + 1..k + 1, steps);
                    });
                    tasks.splice(k + 1..k + 1, more);
                }
            }
            Task::Fetch(i, missing) => fetch_need(&manifest[*i], missing, places, work, teller)?,
            Task::Pack(i) => pack_need(&manifest[*i], places, work, teller)?,
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
    Ok(manifest
        .iter()
        .map(|need| need.file.clone())
        .zip(paths)
        .collect())
}

/// Whether the disk the store is on has room for `tasks` (and a twentieth
/// more).
fn room(
    tasks: &[Task],
    manifest: &[Need],
    places: &Places,
    work: &dyn Work,
) -> Result<(), Failure> {
    let need = space_for(tasks, manifest);
    let with_margin = need + need / 20;
    match work.free_space(&places.store) {
        Some(free) if free < with_margin => Err(failure(
            format!(
                "There is not enough free space on the disk {} is on: setup needs {}, there are {} free.",
                places.store.display(),
                size(with_margin),
                size(free)
            ),
            "Free some space on that disk, then try again.",
        )),
        _ => Ok(()),
    }
}

/// Removes the wrong file of the store `path` names, where it goes and
/// where it was ([`crate::store::mover::remove`]: a copy left in the old
/// place would come over again), its read-only mark taken off first.
fn remove(path: &Path) {
    durable::make_replaceable(&crate::store::mover::locate(path));
    let _ = crate::store::mover::remove(path);
}

/// Checks `need` found whole at `at`: whether every file of it is right.
/// The wrong ones in the store are removed (one elsewhere is passed over).
fn check_need(
    need: &Need,
    at: &Path,
    places: &Places,
    teller: &mut Teller<'_, Status>,
) -> Result<bool, Failure> {
    teller.set(true, |s| {
        s.total = need.bytes();
        s.unit = Some(Unit::Bytes);
    });
    let mut before = 0;
    let mut right = true;
    for (path, sum) in need.expected(at) {
        if !verified(&places.store, &path, sum, before, teller)? {
            right = false;
            // One of the store's (where it goes, or not moved there yet).
            if crate::store::mover::current(&path).starts_with(&places.store) {
                remove(&path);
            }
        }
        before += sum.bytes;
    }
    Ok(right)
}

/// Downloads those of `need`'s files that `missing` names, each checked
/// and stamped; one already there and right (from a run that stopped) is
/// kept.
fn fetch_need(
    need: &Need,
    missing: &[usize],
    places: &Places,
    work: &dyn Work,
    teller: &mut Teller<'_, Status>,
) -> Result<(), Failure> {
    teller.set(true, |s| {
        s.total = files_bytes(need, missing.iter().copied());
        s.unit = Some(Unit::Bytes);
    });
    let files = need.files(&fetched_at(places, need));
    let mut before = 0;
    for &f in missing {
        let (to, part) = &files[f];
        if !verified(&places.store, to, &part.sum, before, teller)? {
            let started = Instant::now();
            work.download(part, to, &mut |done| {
                let seconds = started.elapsed().as_secs_f64();
                teller.set(false, |s| {
                    s.done = before + done;
                    s.rate = (seconds > 1.0).then(|| done as f64 / seconds);
                })
            })
            .map_err(|fault| {
                let what = format!("Downloading {} failed: {}", need.title, fault.why());
                match fault {
                    Fault::Web(_) => failure(what, NETWORK),
                    Fault::Wrong(_) => lasting(what, WRONG),
                    Fault::Disk(_) => failure(what, STORE),
                }
            })?;
            stamp(&places.store, &part.sum.sha256, to)?;
        }
        before += part.sum.bytes;
    }
    Ok(())
}

/// Packs `need`'s checkpoint, downloaded beside the store's file, into
/// the file, checked against its SHA-256 before it takes the name.
fn pack_need(
    need: &Need,
    places: &Places,
    work: &dyn Work,
    teller: &mut Teller<'_, Status>,
) -> Result<(), Failure> {
    // A model downloaded as it is has nothing to pack.
    let Some((packed, _)) = need.packed() else {
        return Ok(());
    };
    let sha256 = &packed.sha256;
    if let Some(memory) = work.memory() {
        if memory < PACK_MEMORY {
            return Err(failure(
                format!(
                    "Packing {} needs about {} of memory; this computer has {}.",
                    need.title,
                    size(PACK_MEMORY),
                    size(memory)
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
    let to = need.at(&places.store);
    // Under a temporary name until it checks (a crash's swept).
    let partial = durable::temporary(&to);
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
            format!(
                "Try again. If it fails again, the download may be damaged: remove the folder {} and try again, and it is fetched afresh.",
                dir.display()
            ),
        )
    })?;
    teller.set(true, |s| {
        s.done = 0;
        s.total = need.bytes();
        s.unit = Some(Unit::Bytes);
    });
    if let Some(digest) = hashes_to(&partial, sha256, 0, teller)? {
        let _ = fs::remove_file(&partial);
        let _ = fs::remove_dir_all(&dir);
        // Its download checked, so packing again gives the same.
        return Err(lasting(
            format!(
                "{} came out different from what it should be (SHA-256 {digest}).",
                capital(&need.title)
            ),
            "Try again: the download is fetched afresh. If it comes out different again, this build of the player packs differently from the one its checksum was made with.",
        ));
    }
    durable::make_replaceable(&to);
    durable::rename_durable(&partial, &to)
        .map_err(|e| failure(format!("{}: {e}", to.display()), "Try again."))?;
    stamp(&places.store, sha256, &to)
}
