//! Beats and tempo of the tracks being played: the analysis in the
//! background, and a cache of its results on disk.
//!
//! A file is decoded as it is stored, mixed to mono, resampled to the
//! analysis rate and given to [`jodel_rhythm::Analyzer`]; the same mono, at
//! 44.1 kHz (its first [`KEY_SECONDS`]), tells the key
//! ([`jodel_rhythm::key`]). With stems (a
//! [`RhythmRecipe`] asking for them, and a GPU), the same decode also goes
//! as stereo at 44.1 kHz through the separator on the GPU
//! ([`jodel_gpu::GpuSeparator`]), chunk by chunk, and the cues listen to the
//! stems. One [`RhythmWorker`] thread analyses the files it is asked for,
//! in the order asked, and answers with [`Action::Rhythm`]; it first looks
//! for the GPU and says what it found ([`Action::Gpu`]). Results are kept
//! in the data folder's `analyses`, one small text file per track and
//! recipe with its waveform beside it ([`kept`]), named by a hash of the
//! file's path, size and modification time, [`jodel_rhythm::VERSION`] and
//! the recipe ([`cache_key`]), so a changed file, a changed analysis or
//! another recipe is analysed afresh.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jodel_dsp::FULL_SCALE;
use jodel_gpu::GpuSeparator;
use jodel_playback::{Resampler, Source, StereoFeed};
use jodel_rhythm::stems::{PartsBuilder, Separator, Splitter};
use jodel_rhythm::waveform::{Lane, LaneBuilder, Melody, Waveform};
use jodel_rhythm::{Analysis, Analyzer, Evidence, Method, SAMPLE_RATE};

use crate::core::{Action, Facts, Heard, Job, RhythmFacts, RhythmRecipe, Rhythmed};
use crate::work::outbox::Outbox;
use crate::work::stages::{self, Stage};
use crate::work::{Lane as Ask, Lanes, Queue};

mod report;
pub use report::{describe, recent, remember, Timings, RECENT_KEPT};
mod kept;
use kept::load_analysis;
pub(crate) use kept::{beats_text, parse_beats};
pub use kept::{cache_key, from_text, is_stored, key_parts, load, store, to_text};

/// The seconds of a track the key is found from (a long mix is not held
/// whole: ten minutes is about 100 MB).
pub const KEY_SECONDS: usize = 600;

/// What was heard in a file.
pub struct Listened {
    /// The evidence for the analysis.
    pub evidence: Evidence,
    /// Why the stems could not be used, when they were asked for and there
    /// was a GPU (the cues then come from the fast split).
    pub stems_failed: Option<String>,
    /// The mix's waveform, and the stems' when they were separated.
    pub waveform: Waveform,
    /// How long each part took.
    pub timings: Timings,
    /// The key it is in, where it could be told.
    pub key: Option<jodel_rhythm::key::Key>,
}

impl Listened {
    /// The analysis: the beat decided, the key found.
    pub fn analysis(&self) -> Analysis {
        Analysis {
            key: self.key,
            ..self.evidence.conclude()
        }
    }
}

/// Decodes `path` and analyses it by `recipe`.
pub fn analyze_file(path: &Path, recipe: &RhythmRecipe) -> Result<Analysis, String> {
    listen_file(path, recipe).map(|l| l.analysis())
}

/// The separator to use for `recipe`: the GPU's, when stems are asked for
/// and there is one; on a background thread its own, apart from the stems
/// of the track playing.
fn separator(recipe: &RhythmRecipe) -> Option<&'static GpuSeparator> {
    recipe.stems.then(GpuSeparator::for_this_thread).flatten()
}

/// The key the tags name, where one does (`INITIALKEY`, ID3's `TKEY`, or
/// `KEY`) in a spelling [`jodel_rhythm::key::Key::from_tag`] reads.
pub fn key_tag(tags: &jodel_playback::Tags) -> Option<jodel_rhythm::key::Key> {
    ["INITIALKEY", "TKEY", "KEY"]
        .iter()
        .find_map(|name| tags.get(name).and_then(jodel_rhythm::key::Key::from_tag))
}

/// The vocals stem as it is separated, mono at the pitch network's rate
/// (the blocks come in order, each sample once), for its melody.
struct VoiceFeed {
    resampler: Resampler,
    mono: Vec<f64>,
    out: Vec<f64>,
    voice: Vec<f32>,
}

impl VoiceFeed {
    fn new() -> VoiceFeed {
        VoiceFeed {
            resampler: Resampler::new(jodel_playback::stems::RATE, jodel_rhythm::pitch::RATE, 1),
            mono: Vec::new(),
            out: Vec::new(),
            voice: Vec::new(),
        }
    }

    fn take(&mut self, b: &jodel_rhythm::stems::StemBlock) {
        let [l, r] = b.stems[jodel_playback::stems::Stem::Vocals];
        self.mono.clear();
        self.mono.extend(
            l.iter()
                .zip(r)
                .map(|(a, c)| (f64::from(*a) + f64::from(*c)) / 2.0),
        );
        self.out.clear();
        self.resampler.process_f64(&self.mono, &mut self.out);
        self.voice.extend(self.out.iter().map(|&v| v as f32));
    }

    /// The voice's melody ([`jodel_rhythm::pitch`]), once every block is in.
    fn melody(mut self) -> Option<Melody> {
        self.out.clear();
        self.resampler.finish_f64(&mut self.out);
        self.voice.extend(self.out.iter().map(|&v| v as f32));
        (!self.voice.is_empty())
            .then(|| Melody::from_pitch(&jodel_rhythm::pitch::pitch_of(&self.voice)))
    }
}

/// Seconds of stereo the stems may fall behind the decoding: enough for
/// most tracks whole, so the GPU and the network work side by side; a
/// longer track's decoding waits (about 100 MB at most).
const STEMS_BEHIND: usize = 300;

/// Decodes `path` and gathers what `recipe` hears in it, before the
/// analysis decides on the beat.
pub fn listen_file(path: &Path, recipe: &RhythmRecipe) -> Result<Listened, String> {
    listen_keeping(path, recipe, None)
}

/// [`listen_file`], and with stems separated, their audio kept in the
/// stems file `keep` ([`crate::store::stems`]), so they are never separated
/// again.
pub fn listen_keeping(
    path: &Path,
    recipe: &RhythmRecipe,
    keep: Option<PathBuf>,
) -> Result<Listened, String> {
    listen_reporting(path, recipe, keep, None)
}

/// Told of each part of an analysis as it is done ([`Heard`]).
pub type Report = Arc<dyn Fn(Heard) + Send + Sync>;

/// [`listen_keeping`], telling `report` of the key once the file is
/// decoded, of the stems so far as each chunk is separated (where the
/// file says how long it is) and once all are.
///
/// A recipe with work on the GPU is heard as one job of the GPU's stages
/// ([`stages::run`]): the separator's ([`Stage::Stems`]) for its stems,
/// the beat network's ([`Stage::Beats`]) for its network. The whole of it
/// is the one job: the separator is fed as the file decodes and the
/// network runs on what is decoded, so the GPU is at work from the first
/// second decoded to the last.
pub fn listen_reporting(
    path: &Path,
    recipe: &RhythmRecipe,
    keep: Option<PathBuf>,
    report: Option<Report>,
) -> Result<Listened, String> {
    let stems = recipe.stems.then_some(Stage::Stems);
    let network = recipe
        .methods
        .contains(&Method::Network)
        .then_some(Stage::Beats);
    let needs: Vec<Stage> = stems.into_iter().chain(network).collect();
    if needs.is_empty() {
        return listen_now(path, recipe, keep, report);
    }
    let name = crate::work::song_name(path);
    stages::run(&needs, &name, || listen_now(path, recipe, keep, report))
}

/// The background separator and the beat network, registered for their
/// stages ([`Stage::Stems`], [`Stage::Beats`]) once.
pub(crate) fn register_networks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        stages::register(Stage::Stems, GpuSeparator::background_resident());
        stages::register(Stage::Beats, jodel_gpu::GpuNetwork::shared_resident());
    });
}

/// [`listen_reporting`], now.
fn listen_now(
    path: &Path,
    recipe: &RhythmRecipe,
    keep: Option<PathBuf>,
    report: Option<Report>,
) -> Result<Listened, String> {
    let started = Instant::now();
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    let (rate, channels, tagged, frames) = {
        let info = source.info();
        (
            info.rate,
            usize::from(info.channels).max(1),
            key_tag(&info.tags),
            info.frames,
        )
    };
    // The song's length in the waveform's frames, for the stems so far.
    let lane_frames =
        frames.map(|f| (f as f64 * jodel_rhythm::waveform::FPS / f64::from(rate)) as usize);
    let mut resampler = (rate != SAMPLE_RATE).then(|| Resampler::new(rate, SAMPLE_RATE, 1));
    // The same mono at the key's rate, its first KEY_SECONDS.
    let key_rate = jodel_rhythm::key::RATE;
    let mut to_key = (rate != key_rate).then(|| Resampler::new(rate, key_rate, 1));
    // A key in the tags is taken as it is, and none heard.
    let key_most = if tagged.is_some() {
        0
    } else {
        KEY_SECONDS * key_rate as usize
    };
    let mut for_key: Vec<f32> = Vec::new();
    let mut key_out = Vec::new();
    let mut analyzer = Analyzer::with_methods(&recipe.methods);
    // The beat network on the GPU where there is one (a piece it fails on
    // runs on the processor), set up only for a recipe that runs it.
    let network = recipe
        .methods
        .contains(&Method::Network)
        .then(jodel_gpu::GpuNetwork::shared)
        .flatten();
    let network_before = network.map(|n| (n, n.tally()));
    if let Some(network) = network {
        analyzer.run_network_on(network);
    }
    let mut mix = LaneBuilder::new(SAMPLE_RATE);
    // The stems in a thread of their own, fed a second of stereo at a
    // time, so the GPU works while the file decodes and the network runs.
    let gpu = separator(recipe);
    let separator_name = gpu.map(|g| g.name());
    let stems_before = gpu.map(|g| (g, g.tally()));
    let background = jodel_gpu::pace::is_background();
    if background && gpu.is_some_and(GpuSeparator::plays) {
        crate::trace::note(format_args!(
            "beats: the background separator does not fit; {}'s stems on the track playing's, beside the GPU's stages",
            path.display()
        ));
    }
    let stems_report = report.clone();
    let stems = gpu.map(|gpu| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(STEMS_BEHIND);
        // The stems' audio into their file as they come (a file that cannot
        // be written is left out: they are separated live then); what
        // wants them meanwhile waits for it.
        let mut writer = keep
            .as_deref()
            .and_then(|to| crate::store::stems::Keeping::create(to, path, rate, &gpu.name()));
        let worker = thread::spawn(move || {
            if background {
                jodel_gpu::pace::background();
            }
            let began = Instant::now();
            let mut splitter = Splitter::overlapped(gpu);
            let mut parts = PartsBuilder::new();
            let mut voice = VoiceFeed::new();
            let mut take = |b: &jodel_rhythm::stems::StemBlock| {
                parts.take(b);
                voice.take(b);
                if let Some(w) = writer.as_mut() {
                    if w.push(b.stems).is_err() {
                        writer = None;
                    }
                }
                // Each chunk: the stems' waveforms so far, as they are.
                if let (Some(report), Some(frames)) = (&stems_report, lane_frames) {
                    let so_far = crate::core::prep::Separating::of(parts.lanes(), frames);
                    report(Heard::Separating(Arc::new(so_far)));
                }
            };
            for (left, right) in rx {
                splitter.push(&left, &right, &mut take)?;
            }
            splitter.finish(&mut take)?;
            if let Some(w) = writer {
                let _ = w.finish();
            }
            if let Some(report) = &stems_report {
                report(Heard::Stems);
            }
            let parts = parts.finish();
            let melody = voice.melody();
            Ok::<_, String>((parts, melody, began.elapsed()))
        });
        (StereoFeed::new(rate), tx, worker)
    });
    let (mut feed, stems_tx, stems_worker) = match stems {
        Some((feed, tx, worker)) => (Some(feed), Some(tx), Some(worker)),
        None => (None, None, None),
    };
    // A worker that stopped (on an error) says why when joined.
    let mut put = |left, right| {
        if let Some(tx) = &stems_tx {
            let _ = tx.send((left, right));
        }
    };
    let (mut block, mut mono, mut out, mut samples) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let scale = 1.0 / (FULL_SCALE * channels as f64);
    loop {
        block.clear();
        let frames = source
            .read_unquantized(&mut block)
            .map_err(|e| e.to_string())?;
        if frames == 0 {
            break;
        }
        if let Some(feed) = feed.as_mut() {
            feed.push(&block, channels, &mut put);
        }
        mono.clear();
        mono.extend(
            block
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f64>() * scale),
        );
        if for_key.len() < key_most {
            let at_key = match to_key.as_mut() {
                Some(r) => {
                    key_out.clear();
                    r.process_f64(&mono, &mut key_out);
                    &key_out
                }
                None => &mono,
            };
            let room = key_most - for_key.len();
            for_key.extend(at_key.iter().take(room).map(|&s| s as f32));
        }
        let ready = match resampler.as_mut() {
            Some(r) => {
                out.clear();
                r.process_f64(&mono, &mut out);
                &out
            }
            None => &mono,
        };
        samples.clear();
        samples.extend(ready.iter().map(|&s| s as f32));
        mix.push(&samples);
        analyzer.push(&samples);
    }
    if let Some(r) = resampler.as_mut() {
        out.clear();
        r.finish_f64(&mut out);
        samples.clear();
        samples.extend(out.iter().map(|&s| s as f32));
        mix.push(&samples);
        analyzer.push(&samples);
    }
    if let Some(feed) = feed {
        feed.finish(&mut put);
    }
    drop(stems_tx);
    if let Some(r) = to_key.as_mut().filter(|_| for_key.len() < key_most) {
        key_out.clear();
        r.finish_f64(&mut key_out);
        let room = key_most - for_key.len();
        for_key.extend(key_out.iter().take(room).map(|&s| s as f32));
    }
    let key = tagged.or_else(|| jodel_rhythm::key::key_of(&for_key));
    drop(for_key);
    if let Some(report) = &report {
        report(Heard::Key(key));
    }
    let decode = started.elapsed();
    let listening = Instant::now();
    let failed = Mutex::new(None);
    let lanes: Mutex<Option<[Lane; 4]>> = Mutex::new(None);
    let melody: Mutex<Option<Melody>> = Mutex::new(None);
    let stems_times: Mutex<(Option<Duration>, Option<Duration>)> = Mutex::new((None, None));
    let evidence = analyzer.listen_with(|| {
        let worker = stems_worker?;
        let waiting = Instant::now();
        let joined = worker.join();
        let waited = waiting.elapsed();
        match joined {
            Ok(Ok((mut parts, heard, took))) => {
                *stems_times.lock().unwrap_or_else(|e| e.into_inner()) = (Some(took), Some(waited));
                *lanes.lock().unwrap_or_else(|e| e.into_inner()) = parts.lanes.take();
                *melody.lock().unwrap_or_else(|e| e.into_inner()) = heard;
                Some(parts)
            }
            Ok(Err(e)) => {
                *failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                None
            }
            Err(_) => {
                *failed.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some("the stems stopped".into());
                None
            }
        }
    });
    let listen = listening.elapsed();
    let (stems_took, stems_waited) = stems_times.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(Listened {
        evidence,
        stems_failed: failed.into_inner().unwrap_or_else(|e| e.into_inner()),
        waveform: Waveform {
            mix: mix.finish(),
            stems: lanes.into_inner().unwrap_or_else(|e| e.into_inner()),
            melody: melody.into_inner().unwrap_or_else(|e| e.into_inner()),
        },
        timings: Timings {
            decode,
            separator: separator_name,
            stems: stems_took,
            stems_waited,
            stems_gpu: stems_took
                .and(stems_before)
                .map(|(g, before)| g.tally().since(&before)),
            listen,
            network_gpu: network_before.map(|(n, before)| n.tally().since(&before)),
        },
        key,
    })
}

/// What the worker is asked for, in two steps: each file first looked for
/// in the cache ([`load`]), then the files not there analysed. Both in two
/// lanes: wanted now by `recipe`, and to get into the cache ahead by
/// `ahead_recipe` (with a cache only). A file found missing for an earlier
/// ask ([`Lanes::asked`]) is not analysed.
struct Wanted {
    load: Lanes<PathBuf>,
    analyse: Lanes<PathBuf>,
    recipe: RhythmRecipe,
    ahead_recipe: Option<RhythmRecipe>,
}

impl Wanted {
    /// The recipe the files of `lane` are asked by.
    fn recipe(&self, lane: Ask) -> Option<RhythmRecipe> {
        match lane {
            Ask::Now => Some(self.recipe.clone()),
            Ask::Ahead => self.ahead_recipe.clone(),
        }
    }
}

/// Analyses files in the background, one at a time, and answers from the
/// cache at once.
///
/// Two threads: one takes what is wanted from the cache (and says which
/// files to get ready are there already), so a file analysed before never
/// waits behind an analysis under way; the other analyses the rest, the
/// files wanted first.
pub struct RhythmWorker {
    queue: Arc<Queue<Wanted>>,
}

impl RhythmWorker {
    /// Starts the threads; the analysing one first looks for the GPU
    /// (answered with [`Action::Gpu`]). Results go to `outbox` as
    /// [`Action::Rhythm`] and are kept in `cache` (if any); each file whose
    /// analysis is in the cache (or failed) is told with [`Action::Ready`].
    pub fn new(outbox: Outbox, cache: Option<PathBuf>, grids: Option<PathBuf>) -> RhythmWorker {
        register_networks();
        let queue = Arc::new(Queue::new(Wanted {
            load: Lanes::default(),
            analyse: Lanes::default(),
            recipe: RhythmRecipe::default(),
            ahead_recipe: None,
        }));
        {
            let (queue, outbox) = (Arc::clone(&queue), outbox.clone());
            let (cache, grids) = (cache.clone(), grids.clone());
            thread::spawn(move || {
                load_wanted(&queue, &outbox, cache.as_deref(), grids.as_deref());
            });
        }
        let worker = Arc::clone(&queue);
        thread::spawn(move || {
            // Its GPU work gives way to the window's frames.
            jodel_gpu::pace::background();
            // Only the GPU's name for now: the separator is set up when
            // stems are first used.
            if !outbox.send(Action::Gpu(jodel_gpu::gpu_name())) {
                return;
            }
            analyse_wanted(&worker, &outbox, cache.as_deref(), grids.as_deref());
        });
        RhythmWorker { queue }
    }

    /// Answers exactly `paths`, in this order, by `recipe` from now on: at
    /// once those in the cache, then the others as they are analysed (what
    /// was asked for before and not started yet is dropped).
    pub fn want(&self, paths: Vec<PathBuf>, recipe: RhythmRecipe) {
        self.queue.change(|wanted| {
            wanted.load.want(paths);
            wanted.analyse.want([]);
            wanted.recipe = recipe;
        });
    }

    /// Gets `paths` into the cache by `recipe`, in this order, after what
    /// is wanted (what was asked for before and not started is dropped);
    /// nothing without a cache or a recipe. Each is answered with
    /// [`Action::Working`] while it is analysed and [`Action::Ready`] once
    /// it is in the cache (at once for those there already), not with its
    /// analysis.
    pub fn prepare(&self, paths: Vec<PathBuf>, recipe: Option<RhythmRecipe>) {
        self.queue.change(|wanted| {
            wanted.load.prepare(paths);
            wanted.analyse.prepare([]);
            wanted.ahead_recipe = recipe;
        });
    }
}

/// Tells the app that the analysis of `path` is in the cache (or failed):
/// nothing is left to do for it; with what it found (`analysis`), for the
/// queue's icons. False when the app is gone.
fn tell_ready(path: &Path, analysis: Option<&Analysis>, outbox: &Outbox) -> bool {
    outbox.send(Action::Ready {
        path: path.to_owned(),
        job: Job::Rhythm,
        facts: analysis.map(|a| Facts::Rhythm(RhythmFacts::of(a))),
    })
}

/// A track's analysis as the app takes it: with its grid fixed by hand
/// (from `grids`) where there is one.
fn rhythmed(
    path: &Path,
    analysis: Analysis,
    waveform: Waveform,
    grids: Option<&Path>,
) -> Arc<Rhythmed> {
    let found = Rhythmed::new(analysis, Some(waveform));
    let fixed = grids.and_then(|dir| crate::store::grids::load(dir, path));
    Arc::new(match fixed {
        Some(grid) => found.with_grid(&grid),
        None => found,
    })
}

/// The loading thread's loop: each file wanted from the cache, at once
/// (one not there goes to be analysed), and each file to get ready looked
/// for there (one not there goes to be analysed ahead).
fn load_wanted(queue: &Queue<Wanted>, outbox: &Outbox, cache: Option<&Path>, grids: Option<&Path>) {
    let next = |wanted: &mut Wanted| {
        let (path, lane) = wanted.load.pop()?;
        Some((path, lane, wanted.recipe(lane), wanted.load.asked(lane)))
    };
    while let Some((path, lane, recipe, asked)) = queue.next(next) {
        let Some(recipe) = recipe else {
            continue;
        };
        // Not there (and not asked anew since): analysed.
        let to_analyse = |path: PathBuf| {
            queue.change(|wanted| {
                if wanted.load.asked(lane) == asked {
                    wanted.analyse.lane(lane).push_back(path);
                }
            });
        };
        if lane == Ask::Now {
            if let Some((analysis, waveform)) = cache.and_then(|dir| load(dir, &path, &recipe)) {
                // Ready once its stems are kept too (else getting it ready
                // ahead separates them).
                let whole = cache.is_some_and(|dir| is_stored(dir, &path, &recipe));
                let facts = analysis.clone();
                let found = rhythmed(&path, analysis, waveform, grids);
                if !outbox.send(Action::Rhythm(path.clone(), recipe, Ok(found)))
                    || (whole && !tell_ready(&path, Some(&facts), outbox))
                {
                    return;
                }
                continue;
            }
            to_analyse(path);
        } else if let Some(analysis) = cache
            .filter(|dir| is_stored(dir, &path, &recipe))
            .and_then(|dir| load_analysis(dir, &path, &recipe))
        {
            if !tell_ready(&path, Some(&analysis), outbox) {
                return;
            }
        } else if cache.is_some() {
            // Not kept, or kept but not read (damaged): analysed again.
            to_analyse(path);
        }
    }
}

/// Analyses `path` by `recipe` and keeps it in `cache` (if any), saying
/// on `outbox` that it is at work on it meanwhile. `None` when the app
/// is gone.
fn analyse(
    path: &Path,
    recipe: &RhythmRecipe,
    outbox: &Outbox,
    cache: Option<&Path>,
) -> Option<Result<(Analysis, Waveform), String>> {
    let working = |on: bool| {
        outbox.send(Action::Working {
            path: path.to_owned(),
            job: Job::Rhythm,
            on,
        })
    };
    if !working(true) {
        return None;
    }
    let started = Instant::now();
    // With stems: their audio kept, beside the analyses.
    let stems_dir = cache
        .filter(|_| recipe.stems)
        .map(crate::store::stems::beside);
    let keep = stems_dir
        .as_deref()
        .and_then(|dir| crate::store::stems::file_for(dir, path))
        .map(|file| crate::store::mover::current(&file));
    let report: Report = {
        let (outbox, path) = (outbox.clone(), path.to_owned());
        Arc::new(move |part| {
            outbox.send(Action::Heard {
                path: path.clone(),
                part,
            });
        })
    };
    let result = match listen_reporting(path, recipe, keep, Some(report)) {
        Ok(listened) => {
            if let Some(e) = &listened.stems_failed {
                let name = crate::work::song_name(path);
                let note = format!("Stems failed for {name} ({e}): the mix's cues used");
                if !outbox.send(Action::Failed(note)) {
                    return None;
                }
            }
            let deciding = Instant::now();
            let analysis = listened.analysis();
            let decide = deciding.elapsed();
            let at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let entry = describe(
                at,
                path,
                recipe,
                &listened,
                &analysis,
                decide,
                started.elapsed(),
            );
            crate::trace::note(format_args!("beats: {}", entry.replace('\n', "; ")));
            if let Some(dir) = cache {
                let _ = store(dir, path, &analysis, &listened.waveform);
                let _ = remember(dir, &entry);
            }
            let beats = crate::core::prep::BeatsFound::of(&analysis, &listened.waveform.mix);
            if !outbox.send(Action::Heard {
                path: path.to_owned(),
                part: Heard::Beats(Arc::new(beats)),
            }) {
                return None;
            }
            if let Some(dir) = &stems_dir {
                crate::store::stems::prune(dir, crate::store::stems::BUDGET);
            }
            Ok((analysis, listened.waveform))
        }
        Err(e) => Err(e),
    };
    // In the cache now (or it failed): ready, as far as it goes.
    let analysis = result.as_ref().ok().map(|(a, _)| a);
    (working(false) && tell_ready(path, analysis, outbox)).then_some(result)
}

/// The analysing thread's loop: the next file wanted that the cache did
/// not have (it may have since), else one to get ready ahead (with a
/// cache only).
fn analyse_wanted(
    queue: &Queue<Wanted>,
    outbox: &Outbox,
    cache: Option<&Path>,
    grids: Option<&Path>,
) {
    let next = |wanted: &mut Wanted| {
        if let Some(path) = wanted.analyse.lane(Ask::Now).pop_front() {
            return Some((path, wanted.recipe.clone(), Ask::Now));
        }
        let recipe = wanted.ahead_recipe.clone().filter(|_| cache.is_some())?;
        let path = wanted.analyse.lane(Ask::Ahead).pop_front()?;
        Some((path, recipe, Ask::Ahead))
    };
    while let Some((path, recipe, lane)) = queue.next(next) {
        if lane == Ask::Ahead {
            // Kept and read; else (not kept, or damaged) analysed again.
            let kept = cache
                .filter(|dir| is_stored(dir, &path, &recipe))
                .and_then(|dir| load_analysis(dir, &path, &recipe));
            let gone = match kept {
                Some(analysis) => !tell_ready(&path, Some(&analysis), outbox),
                None => analyse(&path, &recipe, outbox, cache).is_none(),
            };
            if gone {
                return;
            }
            continue;
        }
        let result = match cache.and_then(|dir| load(dir, &path, &recipe)) {
            Some(found) => {
                if !tell_ready(&path, Some(&found.0), outbox) {
                    return;
                }
                Ok(found)
            }
            None => match analyse(&path, &recipe, outbox, cache) {
                Some(result) => result,
                None => return,
            },
        };
        let result = result.map(|(analysis, waveform)| rhythmed(&path, analysis, waveform, grids));
        if !outbox.send(Action::Rhythm(path, recipe, result)) {
            return;
        }
    }
}

/// Both threads told to stop: each ends once it waits again (the analysing
/// one after the file it is at). Not joined: an analysis under way would
/// hold up the player's end.
impl Drop for RhythmWorker {
    fn drop(&mut self) {
        self.queue.stop();
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: tracks that change under their analysis"
)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn the_last_three_analyses_are_kept_newest_first() {
        let dir = jodel_testkit::TempDir::new("recent-test");
        assert!(recent(&dir).is_empty());
        for i in 1..=4 {
            remember(&dir, &format!("at {i}\ntrack: {i}.flac (3:00)\n")).unwrap();
        }
        let kept = recent(&dir);
        assert_eq!(kept.len(), RECENT_KEPT);
        assert_eq!(kept[0], "at 4\ntrack: 4.flac (3:00)");
        assert_eq!(kept[2], "at 2\ntrack: 2.flac (3:00)");
        // As --diag prints them: the time as how long ago.
        let lines = crate::diag::recent_lines(&kept, 3_604);
        assert_eq!(lines[0], "1 h ago");
        assert_eq!(lines[1], "track: 4.flac (3:00)");
        assert_eq!(lines[2], "");
        assert_eq!(
            crate::diag::recent_lines(&[], 0),
            vec!["none yet (analyses taken from the cache are not counted)".to_owned()]
        );
    }

    /// A 16-bit mono WAV of `seconds` of clicks at 120 BPM.
    fn clicks_wav(path: &Path, seconds: usize) {
        let rate = 44_100usize;
        let n = rate * seconds;
        let samples: Vec<i16> = (0..n)
            .map(|i| if i % (rate / 2) < 200 { 12_000 } else { 0 })
            .collect();
        fs::write(path, jodel_dsp::wav::pcm16(rate as u32, 1, &samples)).unwrap();
    }

    /// A file in the cache is answered at once, even while another is
    /// being analysed; one to get ready that is in the cache is ready at
    /// once, and one analysed ahead is ready when its analysis is kept.
    #[test]
    fn the_cache_answers_while_an_analysis_is_under_way() {
        let dir = jodel_testkit::TempDir::new("rhythm-loader");
        let cache = dir.join("cache");
        let recipe = RhythmRecipe {
            methods: vec![Method::Flux],
            stems: false,
        };
        // A: analysed before.
        let a = dir.join("a.wav");
        fs::write(&a, b"kept").unwrap();
        let analysis = Analysis {
            beats: vec![0.5, 1.0, 1.5],
            methods: vec![Method::Flux],
            ..Analysis::default()
        };
        let wave = Waveform {
            mix: Lane {
                peak: vec![1, 2],
                rms: vec![0, 1],
            },
            stems: None,
            melody: None,
        };
        store(&cache, &a, &analysis, &wave).unwrap();
        // B: long, analysed ahead.
        let b = dir.join("b.wav");
        clicks_wav(&b, 240);
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = RhythmWorker::new(Outbox::new(tx, Arc::new(|| {})), Some(cache.clone()), None);
        worker.prepare(vec![b.clone()], Some(recipe.clone()));
        let wait = Duration::from_secs(120);
        let next = || loop {
            match rx.recv_timeout(wait).unwrap() {
                Action::Gpu(_) => continue,
                other => return other,
            }
        };
        assert!(
            matches!(next(), Action::Working { ref path, on: true, .. } if *path == b),
            "B's analysis starts"
        );
        worker.want(vec![a.clone()], recipe.clone());
        let mut seen = Vec::new();
        loop {
            match next() {
                Action::Rhythm(path, _, Ok(found)) => {
                    assert_eq!(path, a);
                    assert_eq!(found.analysis.beats, analysis.beats);
                    seen.push("A answered");
                }
                Action::Ready { path, .. } if path == a => seen.push("A ready"),
                // B's parts as they are done.
                Action::Heard { path, .. } if path == b => {}
                Action::Working {
                    path, on: false, ..
                } if path == b => {
                    seen.push("B done");
                    break;
                }
                other => panic!("{other:?}"),
            }
            if seen.len() == 2 {
                break;
            }
        }
        assert_eq!(seen, ["A answered", "A ready"], "A came before B was done");
        // Getting A ready: it is there already.
        worker.prepare(vec![a.clone(), b.clone()], Some(recipe.clone()));
        let mut ready = Vec::new();
        while ready.len() < 2 {
            match next() {
                Action::Ready { path, .. } => ready.push(path),
                Action::Working { .. } | Action::Heard { .. } => {}
                other => panic!("{other:?}"),
            }
        }
        // B was analysed ahead (and kept) meanwhile, or is now: both ready.
        assert!(ready.contains(&a) && ready.contains(&b), "{ready:?}");
        drop(worker);
    }

    /// Dropped while both its threads wait, the worker ends them both:
    /// what they share goes once neither holds it.
    #[test]
    fn dropping_the_worker_ends_both_threads() {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = RhythmWorker::new(Outbox::new(tx, Arc::new(|| {})), None, None);
        // The analysing thread has told the GPU's name: it waits for work.
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(60)),
            Ok(Action::Gpu(_))
        ));
        let shared = Arc::downgrade(&worker.queue);
        drop(worker);
        let started = Instant::now();
        while shared.strong_count() > 0 {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "a thread still waits"
            );
            thread::sleep(Duration::from_millis(5));
        }
        drop(rx);
    }
}
