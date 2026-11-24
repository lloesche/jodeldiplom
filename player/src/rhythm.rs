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
//! in the user's cache folder, one small text file per track and recipe,
//! named by a hash of its path, size, modification time,
//! [`jodel_rhythm::VERSION`] and the recipe, so a changed file, a changed
//! analysis or another recipe is analysed afresh.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jodel_gpu::GpuSeparator;
use jodel_playback::{Resampler, Source, StereoFeed};
use jodel_rhythm::meter::{Grouping, Span};
use jodel_rhythm::stems::{PartsBuilder, Separator, Splitter};
use jodel_rhythm::waveform::{Lane, LaneBuilder, Melody, Waveform};
use jodel_rhythm::{
    Analysis, Analyzer, CandidateReport, Evidence, Method, Region, SAMPLE_RATE, VERSION,
};

use crate::browser::Notify;
use crate::cache;
use crate::core::{Action, Facts, Heard, Job, RhythmFacts, RhythmRecipe, Rhythmed};

mod report;
pub use report::{describe, recent, remember, Timings, RECENT_KEPT};

/// Full scale of the decoder's 24-bit units.
const FULL_SCALE: f64 = 8_388_608.0;

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
    let pick = if jodel_gpu::pace::is_background() {
        GpuSeparator::background
    } else {
        GpuSeparator::shared
    };
    recipe.stems.then(pick).flatten()
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
    vocals: usize,
    resampler: Resampler,
    mono: Vec<f64>,
    out: Vec<f64>,
    voice: Vec<f32>,
}

impl VoiceFeed {
    fn new() -> VoiceFeed {
        use jodel_playback::stems::Stem;
        VoiceFeed {
            vocals: Stem::ALL
                .iter()
                .position(|&s| s == Stem::Vocals)
                .unwrap_or(3),
            resampler: Resampler::new(jodel_playback::stems::RATE, jodel_rhythm::pitch::RATE, 1),
            mono: Vec::new(),
            out: Vec::new(),
            voice: Vec::new(),
        }
    }

    fn take(&mut self, b: &jodel_rhythm::stems::StemBlock) {
        let [l, r] = b.stems[self.vocals];
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
/// stems file `keep` ([`crate::stems_cache`]), so they are never separated
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

/// [`listen_keeping`], telling `report` of the key once the file is decoded
/// and of the stems once separated.
pub fn listen_reporting(
    path: &Path,
    recipe: &RhythmRecipe,
    keep: Option<PathBuf>,
    report: Option<Report>,
) -> Result<Listened, String> {
    let started = Instant::now();
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    let (rate, channels, tagged) = {
        let info = source.info();
        (
            info.rate,
            usize::from(info.channels).max(1),
            key_tag(&info.tags),
        )
    };
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
    // runs on the processor).
    let network = jodel_gpu::GpuNetwork::shared();
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
    let stems_report = report.clone();
    let stems = gpu.map(|gpu| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(STEMS_BEHIND);
        // The stems' audio into their file as they come (a file that cannot
        // be written is left out: they are separated live then); what
        // wants them meanwhile waits for it.
        let mut writer = keep
            .as_deref()
            .and_then(|to| crate::stems_cache::Keeping::create(to, path, rate, &gpu.name()));
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
        report(Heard::Key(key.is_some()));
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

/// Where analyses are kept: `analyses` in the data folder of the usual
/// settings ([`crate::dirs::Data::current`]; the player itself is told its
/// own).
pub fn cache_dir() -> Option<PathBuf> {
    crate::dirs::Data::current().map(|d| d.analyses())
}

/// What identifies a file's analysis: its path, size and modification
/// time, the analysis version, the methods and whether the cues listened
/// to stems.
pub fn cache_key(path: &Path, methods: &[Method], stems: bool) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let ids: Vec<&str> = methods.iter().map(|m| m.id()).collect();
    let stems = if stems { "+stems" } else { "" };
    Some(format!(
        "{} {} {modified} v{VERSION} {}{stems}",
        path.display(),
        meta.len(),
        ids.join("+")
    ))
}

/// FNV-1a, 64 bits.
pub fn hash(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The cache file for `key` in `dir`.
fn cache_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{:016x}.txt", hash(key)))
}

/// Beat times are stored in units of 0.1 ms.
const TICKS: f64 = 10_000.0;

/// Appends `beats` to `s` as ` d1 d2 ...`: each the ticks since the last.
pub(crate) fn beats_text(s: &mut String, beats: &[f64]) {
    let mut last = 0i64;
    for &b in beats {
        let t = (b * TICKS).round() as i64;
        let _ = write!(s, " {}", t - last);
        last = t;
    }
}

/// The beats in [`beats_text`]'s words.
pub(crate) fn parse_beats(words: &str) -> Option<Vec<f64>> {
    let mut t = 0i64;
    words
        .split_whitespace()
        .map(|d| {
            t += d.parse::<i64>().ok()?;
            Some(t as f64 / TICKS)
        })
        .collect()
}

/// An analysis as the cache file's text, under its `key`, in
/// [`cache::ANALYSIS`]'s version.
pub fn to_text(key: &str, a: &Analysis) -> String {
    cache::ANALYSIS.wrap(&analysis_text(key, a))
}

/// The cache file's text after its header.
fn analysis_text(key: &str, a: &Analysis) -> String {
    let mut s = format!("jodel-rhythm {VERSION}\nkey {}\n", key.replace('\n', " "));
    let _ = writeln!(s, "duration {:.4}", a.duration);
    let _ = writeln!(s, "confidence {:.4}", a.confidence);
    let _ = writeln!(s, "chosen {}", a.chosen);
    for g in &a.groupings {
        let _ = writeln!(
            s,
            "grouping {:.4} {} {}",
            g.start, g.grouping.pulses, g.grouping.phase
        );
    }
    let ids: Vec<&str> = a.methods.iter().map(|m| m.id()).collect();
    let _ = writeln!(s, "methods {}", ids.join(" "));
    let _ = writeln!(s, "stems {}", a.stems);
    match a.key {
        Some(k) if k.tagged => {
            let _ = writeln!(s, "tonality {} tag", k.class);
        }
        Some(k) => {
            let _ = writeln!(s, "tonality {} {:.4}", k.class, k.confidence);
        }
        None => {}
    }
    s.push_str("beats");
    beats_text(&mut s, &a.beats);
    s.push_str("\ndownbeats");
    for i in &a.downbeats {
        let _ = write!(s, " {i}");
    }
    s.push('\n');
    for r in &a.regions {
        let _ = writeln!(s, "region {:.4} {:.4} {:.4}", r.start, r.end, r.confidence);
    }
    for c in &a.candidates {
        let _ = writeln!(s, "candidate {:.3} {:.4} {}", c.bpm, c.agreement, c.name);
    }
    s
}

/// The analysis in a cache file's text, if it was stored under `key` in
/// [`cache::ANALYSIS`]'s version.
pub fn from_text(key: &str, text: &str) -> Option<Analysis> {
    let mut lines = cache::ANALYSIS.body(text)?.lines();
    if lines.next()? != format!("jodel-rhythm {VERSION}") {
        return None;
    }
    if lines.next()?.strip_prefix("key ")? != key.replace('\n', " ") {
        return None;
    }
    let mut a = Analysis::default();
    for line in lines {
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        match word {
            "duration" => a.duration = rest.parse().ok()?,
            "confidence" => a.confidence = rest.parse().ok()?,
            "chosen" => a.chosen = rest.to_owned(),
            "grouping" => {
                let mut parts = rest.split_whitespace();
                let start = parts.next()?.parse().ok()?;
                let pulses = parts.next()?.parse().ok()?;
                let phase = parts.next()?.parse().ok()?;
                a.groupings.push(Span {
                    start,
                    grouping: Grouping::new(pulses, phase)?,
                });
            }
            "stems" => a.stems = rest.parse().ok()?,
            "tonality" => {
                let (class, confidence) = rest.split_once(' ')?;
                let class: u8 = class.parse().ok()?;
                if class >= 24 {
                    return None;
                }
                let tagged = confidence == "tag";
                a.key = Some(jodel_rhythm::key::Key {
                    class,
                    confidence: if tagged {
                        1.0
                    } else {
                        confidence.parse().ok()?
                    },
                    tagged,
                });
            }
            "methods" => {
                for id in rest.split_whitespace() {
                    a.methods.push(Method::parse(id)?);
                }
            }
            "beats" => a.beats = parse_beats(rest)?,
            "downbeats" => {
                for i in rest.split_whitespace() {
                    a.downbeats.push(i.parse().ok()?);
                }
            }
            "region" => {
                let v: Vec<f64> = rest
                    .split_whitespace()
                    .map(|x| x.parse().ok())
                    .collect::<Option<_>>()?;
                let [start, end, confidence] = v[..] else {
                    return None;
                };
                a.regions.push(Region {
                    start,
                    end,
                    confidence,
                });
            }
            "candidate" => {
                let mut parts = rest.splitn(3, ' ');
                a.candidates.push(CandidateReport {
                    bpm: parts.next()?.parse().ok()?,
                    agreement: parts.next()?.parse().ok()?,
                    name: parts.next()?.to_owned(),
                });
            }
            _ => return None,
        }
    }
    a.derive();
    Some(a)
}

/// The analysis of `path` by `recipe` from the cache in `dir`, without its
/// waveform.
fn load_analysis(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> Option<Analysis> {
    let key = cache_key(path, &recipe.methods, recipe.stems)?;
    let file = crate::mover::locate(&cache_file(dir, &key));
    from_text(&key, &fs::read_to_string(&file).ok()?)
}

/// The stored analysis of `path` by `recipe` and its waveform, if there
/// are both for the file as it is now.
pub fn load(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> Option<(Analysis, Waveform)> {
    let key = cache_key(path, &recipe.methods, recipe.stems)?;
    // Where it is now: while the data folder moves, not yet moved is where
    // it was.
    let file = crate::mover::locate(&cache_file(dir, &key));
    let analysis = from_text(&key, &fs::read_to_string(&file).ok()?)?;
    let waveform = Waveform::from_bytes(&fs::read(file.with_extension("wave")).ok()?)?;
    Some((analysis, waveform))
}

/// Writes `bytes` to `file` whole, then renames it into place.
fn write_whole(file: &Path, bytes: &[u8]) -> std::io::Result<()> {
    durable::write_atomic(file, bytes)
}

/// Stores the analysis of `path` and its waveform, under the methods it
/// used (the waveform first: an analysis without one is not loaded).
pub fn store(
    dir: &Path,
    path: &Path,
    analysis: &Analysis,
    waveform: &Waveform,
) -> std::io::Result<()> {
    let Some(key) = cache_key(path, &analysis.methods, analysis.stems) else {
        return Ok(());
    };
    // Where it goes now: while the data folder moves, into the new one.
    let dir = &crate::mover::current(dir);
    fs::create_dir_all(dir)?;
    let file = cache_file(dir, &key);
    write_whole(&file.with_extension("wave"), &waveform.to_bytes())?;
    write_whole(&file, to_text(&key, analysis).as_bytes())
}

struct Wanted {
    /// Files wanted, to be taken from the cache at once ([`load`]).
    load: VecDeque<PathBuf>,
    /// Files wanted that are not in the cache, to be analysed.
    queue: VecDeque<PathBuf>,
    recipe: RhythmRecipe,
    /// Counts [`RhythmWorker::want`]s: what was found missing for an
    /// earlier one is not analysed.
    asked: u64,
    /// Files to get into the cache after those, by `ahead_recipe`: to be
    /// looked for in the cache, then those not there analysed.
    check: VecDeque<PathBuf>,
    ahead: VecDeque<PathBuf>,
    ahead_recipe: Option<RhythmRecipe>,
    /// Counts [`RhythmWorker::prepare`]s, the same way.
    prepared: u64,
    stop: bool,
}

/// Whether the analysis of `path` by `recipe` is in the cache in `dir`
/// (without reading it), and with stems, the stems themselves beside it
/// ([`crate::stems_cache::beside`]).
pub fn is_stored(dir: &Path, path: &Path, recipe: &RhythmRecipe) -> bool {
    let analysed = cache_key(path, &recipe.methods, recipe.stems).is_some_and(|key| {
        let file = cache_file(dir, &key);
        file.is_file() && file.with_extension("wave").is_file()
    });
    analysed
        && (!recipe.stems || crate::stems_cache::is_kept(&crate::stems_cache::beside(dir), path))
}

/// Analyses files in the background, one at a time, and answers from the
/// cache at once.
///
/// Two threads: one takes what is wanted from the cache (and says which
/// files to get ready are there already), so a file analysed before never
/// waits behind an analysis under way; the other analyses the rest, the
/// files wanted first.
pub struct RhythmWorker {
    shared: Arc<(Mutex<Wanted>, Condvar)>,
}

impl RhythmWorker {
    /// Starts the threads; the analysing one first looks for the GPU
    /// (answered with [`Action::Gpu`]). Results go to `sender` as
    /// [`Action::Rhythm`] and are kept in `cache` (if any); each file whose
    /// analysis is in the cache (or failed) is told with [`Action::Ready`].
    pub fn new(
        sender: Sender<Action>,
        notify: Notify,
        cache: Option<PathBuf>,
        grids: Option<PathBuf>,
    ) -> RhythmWorker {
        let shared = Arc::new((
            Mutex::new(Wanted {
                load: VecDeque::new(),
                queue: VecDeque::new(),
                recipe: RhythmRecipe::default(),
                asked: 0,
                check: VecDeque::new(),
                ahead: VecDeque::new(),
                ahead_recipe: None,
                prepared: 0,
                stop: false,
            }),
            Condvar::new(),
        ));
        {
            let worker = Arc::clone(&shared);
            let (sender, notify) = (sender.clone(), Arc::clone(&notify));
            let (cache, grids) = (cache.clone(), grids.clone());
            thread::spawn(move || {
                load_wanted(
                    &worker,
                    &sender,
                    &notify,
                    cache.as_deref(),
                    grids.as_deref(),
                );
            });
        }
        let worker = Arc::clone(&shared);
        thread::spawn(move || {
            // Its GPU work gives way to the window's frames.
            jodel_gpu::pace::background();
            // Only the GPU's name for now: the separator is set up when
            // stems are first used.
            if sender.send(Action::Gpu(jodel_gpu::gpu_name())).is_err() {
                return;
            }
            notify();
            work(
                &worker,
                &sender,
                &notify,
                cache.as_deref(),
                grids.as_deref(),
            );
        });
        RhythmWorker { shared }
    }

    /// Answers exactly `paths`, in this order, by `recipe` from now on: at
    /// once those in the cache, then the others as they are analysed (what
    /// was asked for before and not started yet is dropped).
    pub fn want(&self, paths: Vec<PathBuf>, recipe: RhythmRecipe) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        wanted.load = paths.into();
        wanted.queue.clear();
        wanted.recipe = recipe;
        wanted.asked += 1;
        ready.notify_all();
    }

    /// Gets `paths` into the cache by `recipe`, in this order, after what
    /// is wanted (what was asked for before and not started is dropped);
    /// nothing without a cache or a recipe. Each is answered with
    /// [`Action::Working`] while it is analysed and [`Action::Ready`] once
    /// it is in the cache (at once for those there already), not with its
    /// analysis.
    pub fn prepare(&self, paths: Vec<PathBuf>, recipe: Option<RhythmRecipe>) {
        let (lock, ready) = &*self.shared;
        let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
        wanted.check = paths.into();
        wanted.ahead.clear();
        wanted.ahead_recipe = recipe;
        wanted.prepared += 1;
        ready.notify_all();
    }
}

/// Tells the app that the analysis of `path` is in the cache (or failed):
/// nothing is left to do for it; with what it found (`analysis`), for the
/// queue's icons. False when the app is gone.
fn tell_ready(
    path: &Path,
    analysis: Option<&Analysis>,
    sender: &Sender<Action>,
    notify: &Notify,
) -> bool {
    let sent = sender.send(Action::Ready {
        path: path.to_owned(),
        job: Job::Rhythm,
        facts: analysis.map(|a| Facts::Rhythm(RhythmFacts::of(a))),
    });
    notify();
    sent.is_ok()
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
    let fixed = grids.and_then(|dir| crate::grids::load(dir, path));
    Arc::new(match fixed {
        Some(grid) => found.with_grid(&grid),
        None => found,
    })
}

/// The loading thread's loop: each file wanted from the cache, at once
/// (one not there goes to be analysed), and each file to get ready looked
/// for there (one not there goes to be analysed ahead).
fn load_wanted(
    shared: &(Mutex<Wanted>, Condvar),
    sender: &Sender<Action>,
    notify: &Notify,
    cache: Option<&Path>,
    grids: Option<&Path>,
) {
    let (lock, ready) = shared;
    loop {
        let (path, recipe, wanted_now, asked) = {
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if wanted.stop {
                    return;
                }
                if let Some(path) = wanted.load.pop_front() {
                    break (path, Some(wanted.recipe.clone()), true, wanted.asked);
                }
                if let Some(path) = wanted.check.pop_front() {
                    break (path, wanted.ahead_recipe.clone(), false, wanted.prepared);
                }
                wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(recipe) = recipe else {
            continue;
        };
        if wanted_now {
            if let Some((analysis, waveform)) = cache.and_then(|dir| load(dir, &path, &recipe)) {
                // Ready once its stems are kept too (else getting it ready
                // ahead separates them).
                let whole = cache.is_some_and(|dir| is_stored(dir, &path, &recipe));
                let facts = analysis.clone();
                let found = rhythmed(&path, analysis, waveform, grids);
                if sender
                    .send(Action::Rhythm(path.clone(), recipe, Ok(found)))
                    .is_err()
                    || (whole && !tell_ready(&path, Some(&facts), sender, notify))
                {
                    return;
                }
                continue;
            }
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            if wanted.asked == asked {
                wanted.queue.push_back(path);
                ready.notify_all();
            }
        } else if cache.is_some_and(|dir| is_stored(dir, &path, &recipe)) {
            let analysis = cache.and_then(|dir| load_analysis(dir, &path, &recipe));
            if !tell_ready(&path, analysis.as_ref(), sender, notify) {
                return;
            }
        } else if cache.is_some() {
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            if wanted.prepared == asked {
                wanted.ahead.push_back(path);
                ready.notify_all();
            }
        }
    }
}

/// Analyses `path` by `recipe` and keeps it in `cache` (if any), saying
/// on `sender` that it is at work on it meanwhile. `None` when the app
/// is gone.
fn analyse(
    path: &Path,
    recipe: &RhythmRecipe,
    sender: &Sender<Action>,
    notify: &Notify,
    cache: Option<&Path>,
) -> Option<Result<(Analysis, Waveform), String>> {
    let working = |on: bool| {
        let sent = sender.send(Action::Working {
            path: path.to_owned(),
            job: Job::Rhythm,
            on,
        });
        notify();
        sent.is_ok()
    };
    if !working(true) {
        return None;
    }
    let started = Instant::now();
    // With stems: their audio kept, beside the analyses.
    let stems_dir = cache
        .filter(|_| recipe.stems)
        .map(crate::stems_cache::beside);
    let keep = stems_dir
        .as_deref()
        .and_then(|dir| crate::stems_cache::file_for(dir, path))
        .map(|file| crate::mover::current(&file));
    let report: Report = {
        let (sender, notify, path) = (sender.clone(), Arc::clone(notify), path.to_owned());
        Arc::new(move |part| {
            if sender
                .send(Action::Heard {
                    path: path.clone(),
                    part,
                })
                .is_ok()
            {
                notify();
            }
        })
    };
    let result = match listen_reporting(path, recipe, keep, Some(report)) {
        Ok(listened) => {
            if let Some(e) = &listened.stems_failed {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                let note = format!("Stems failed for {name} ({e}): the mix's cues used");
                if sender.send(Action::Failed(note)).is_err() {
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
            if let Some(dir) = &stems_dir {
                crate::stems_cache::prune(dir, crate::stems_cache::BUDGET);
            }
            Ok((analysis, listened.waveform))
        }
        Err(e) => Err(e),
    };
    // In the cache now (or it failed): ready, as far as it goes.
    let analysis = result.as_ref().ok().map(|(a, _)| a);
    (working(false) && tell_ready(path, analysis, sender, notify)).then_some(result)
}

/// The analysing thread's loop: the next file wanted that the cache did
/// not have (it may have since), else one to get ready ahead (with a
/// cache only).
fn work(
    shared: &(Mutex<Wanted>, Condvar),
    sender: &Sender<Action>,
    notify: &Notify,
    cache: Option<&Path>,
    grids: Option<&Path>,
) {
    loop {
        let (path, recipe, ahead) = {
            let (lock, ready) = shared;
            let mut wanted = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if wanted.stop {
                    return;
                }
                if let Some(path) = wanted.queue.pop_front() {
                    break (path, wanted.recipe.clone(), false);
                }
                if let (Some(recipe), Some(_)) = (wanted.ahead_recipe.clone(), cache) {
                    if let Some(path) = wanted.ahead.pop_front() {
                        break (path, recipe, true);
                    }
                }
                wanted = ready.wait(wanted).unwrap_or_else(|e| e.into_inner());
            }
        };
        if ahead {
            let stored = cache.is_some_and(|dir| is_stored(dir, &path, &recipe));
            let gone = if stored {
                let analysis = cache.and_then(|dir| load_analysis(dir, &path, &recipe));
                !tell_ready(&path, analysis.as_ref(), sender, notify)
            } else {
                analyse(&path, &recipe, sender, notify, cache).is_none()
            };
            if gone {
                return;
            }
            continue;
        }
        let result = match cache.and_then(|dir| load(dir, &path, &recipe)) {
            Some(found) => {
                if !tell_ready(&path, Some(&found.0), sender, notify) {
                    return;
                }
                Ok(found)
            }
            None => match analyse(&path, &recipe, sender, notify, cache) {
                Some(result) => result,
                None => return,
            },
        };
        let result = result.map(|(analysis, waveform)| rhythmed(&path, analysis, waveform, grids));
        if sender.send(Action::Rhythm(path, recipe, result)).is_err() {
            return;
        }
        notify();
    }
}
impl Drop for RhythmWorker {
    fn drop(&mut self) {
        let (lock, ready) = &*self.shared;
        lock.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        ready.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_analysis_round_trips_through_its_text_under_its_key_only() {
        let truth = jodel_rhythm::synth::beat_times(0.5, 19.5, |_| 120.0);
        let samples = jodel_rhythm::synth::band(20.0, &truth, 4, 0.5).samples;
        let mut a = jodel_rhythm::analyze_with(&samples, &[Method::Flux]);
        assert!(a.beats.len() > 30);
        a.key = Some(jodel_rhythm::key::Key {
            class: 7,
            confidence: 0.4375,
            tagged: false,
        });
        let text = to_text("song.flac 123 456 v1", &a);
        let back = from_text("song.flac 123 456 v1", &text).unwrap();
        assert_eq!(back.beats.len(), a.beats.len());
        assert!(back
            .beats
            .iter()
            .zip(&a.beats)
            .all(|(x, y)| (x - y).abs() < 1e-4));
        assert_eq!(back.downbeats, a.downbeats);
        assert_eq!(back.regions.len(), a.regions.len());
        assert_eq!(back.candidates.len(), a.candidates.len());
        assert_eq!(back.chosen, a.chosen);
        assert_eq!(back.groupings.len(), a.groupings.len());
        assert!(back
            .groupings
            .iter()
            .zip(&a.groupings)
            .all(|(x, y)| x.grouping == y.grouping && (x.start - y.start).abs() < 1e-4));
        assert_eq!(back.methods, vec![Method::Flux]);
        assert_eq!(back.key, a.key);
        a.key = jodel_rhythm::key::Key::from_tag("Ebm");
        let tagged = from_text("song.flac 123 456 v1", &to_text("song.flac 123 456 v1", &a));
        assert_eq!(tagged.unwrap().key, a.key);
        let (s, t) = (a.summary.unwrap(), back.summary.unwrap());
        assert!((s.bpm - t.bpm).abs() < 0.01);
        assert!(from_text("song.flac 123 999 v1", &text).is_none());
    }

    #[test]
    fn the_last_three_analyses_are_kept_newest_first() {
        let dir = std::env::temp_dir().join(format!("jodel-recent-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
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
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cache_follows_the_file() {
        let dir = std::env::temp_dir().join(format!("jodel-rhythm-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("track.wav");
        fs::write(&file, b"one").unwrap();
        let cache = dir.join("cache");
        let recipe = |methods: &[Method]| RhythmRecipe {
            methods: methods.to_vec(),
            stems: false,
        };
        let a = Analysis {
            beats: vec![0.5, 1.0, 1.5, 2.0, 2.5],
            confidence: 0.9,
            methods: vec![Method::Network],
            ..Analysis::default()
        };
        let wave = Waveform {
            mix: Lane {
                peak: vec![1, 2, 3],
                rms: vec![0, 1, 2],
            },
            stems: None,
            melody: None,
        };
        store(&cache, &file, &a, &wave).unwrap();
        let (back, back_wave) = load(&cache, &file, &recipe(&[Method::Network])).unwrap();
        assert_eq!(back.beats, a.beats);
        assert_eq!(back_wave, wave);
        // Other methods, or stems: analysed afresh.
        assert!(load(&cache, &file, &recipe(&Method::ALL)).is_none());
        let with_stems = RhythmRecipe {
            stems: true,
            ..recipe(&[Method::Network])
        };
        assert!(load(&cache, &file, &with_stems).is_none());
        // Another size: analysed afresh.
        fs::write(&file, b"longer").unwrap();
        assert!(load(&cache, &file, &recipe(&[Method::Network])).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// A 16-bit mono WAV of `seconds` of clicks at 120 BPM.
    fn clicks_wav(path: &Path, seconds: usize) {
        let rate = 44_100usize;
        let n = rate * seconds;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v: i16 = if i % (rate / 2) < 200 { 12_000 } else { 0 };
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&(rate as u32).to_le_bytes());
        wav.extend_from_slice(&(rate as u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        fs::write(path, wav).unwrap();
    }

    /// A file in the cache is answered at once, even while another is
    /// being analysed; one to get ready that is in the cache is ready at
    /// once, and one analysed ahead is ready when its analysis is kept.
    #[test]
    fn the_cache_answers_while_an_analysis_is_under_way() {
        let dir = std::env::temp_dir().join(format!("jodel-rhythm-loader-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
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
        let worker = RhythmWorker::new(tx, Arc::new(|| {}), Some(cache.clone()), None);
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
        let _ = fs::remove_dir_all(&dir);
    }
}
