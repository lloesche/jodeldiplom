//! Beat, downbeat and tempo analysis of a whole track.
//!
//! # Pipeline
//!
//! ```text
//! mono samples at 22.05 kHz → log-mel spectrogram (~100 frames a second)
//! listen  (Analyzer::listen → Evidence)
//!   parts:      drums and tonal sound (a harmonic/percussive split)
//!   cues:       kick, snare, hats, bass and chord change, from the parts
//!   candidates: Method::Flux, spectral flux over all bands and below
//!                 250 Hz, each through the Hmm (tempo × beat position,
//!                 Viterbi) and Plp (local pulse) trackers, and with
//!                 stems the drum stem's flux through the Hmm, voting
//!                 only where the drums play (presence);
//!               Method::Network, the Beat This! network's beat
//!                 probability (from its own spectrogram, 50 frames a
//!                 second) through Hmm, Plp and Peaks
//! decide  (Evidence::decide → Analysis)
//!   vote:    the candidate grid the others agree with most, a confidence
//!   meter:   which grouping of that grid's pulse is the beat (×2 … ×¼),
//!              from the cues and the candidates
//!   align:   each beat onto the sharp onset under it
//!   bars:    downbeats in bars of 3 or 4, from the network's downbeat
//!              probability, the kick and the chord changes
//!   regions: the candidates' agreement per 16 beats; local tempo and a
//!              summary
//! ```
//!
//! The candidates fail in different ways (the low-band flux hears only
//! kick drums and bass, the pulse tracker smooths over gaps the HMM
//! follows, the network knows music but not this track), so their
//! agreement is a confidence that needs no ground truth. A new method
//! joins the vote as one more [`Method`]. The candidates all follow the
//! steadiest pulse, which is not always the beat a listener counts; the
//! meter hears what the listener does (the backbeat, accents, chord
//! changes) to choose it.
//!
//! # Use
//!
//! ```
//! use jodel_rhythm::{Analyzer, SAMPLE_RATE};
//!
//! let mut analyzer = Analyzer::new();
//! analyzer.push(&vec![0.0; SAMPLE_RATE as usize]);
//! let analysis = analyzer.finish();
//! assert!(analysis.beats.is_empty()); // silence has no beat
//! ```

#![warn(missing_docs)]

pub mod activation;
pub mod cues;
mod fft;
pub mod frontend;
pub mod key;
pub mod meter;
pub mod metrics;
pub mod net;
pub mod parts;
pub mod pitch;
pub mod presence;
pub mod stems;
pub mod summary;
pub mod synth;
pub mod track;
pub mod vote;
pub mod waveform;
pub mod yin;

use activation::{Activation, Curve, Flux};
use cues::Cues;
use frontend::{Frontend, FrontendConfig};
use net::{NetFrontend, Network};
use parts::Parts;
pub use summary::Summary;
use track::{Hmm, Peaks, Plp, Tracker};
pub use vote::Region;

/// The sample rate the analysis takes, in hertz.
pub const SAMPLE_RATE: u32 = 22_050;

/// Changes whenever the analysis gives different results, so stored
/// analyses can be recomputed.
pub const VERSION: u32 = 5;

/// A region whose candidates agree at least this much has a clear beat.
pub const CLEAR: f64 = 0.5;

/// A way of finding beats; each gives several candidates to the vote.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Method {
    /// Spectral flux over all bands and below 250 Hz, each through the
    /// HMM and the pulse tracker; with stems, also that of the drum stem
    /// through the HMM where the drums play.
    Flux,
    /// Our inference of the Beat This! network, its beat probability
    /// through the HMM, the pulse tracker and its own peaks.
    Network,
}

impl Method {
    /// Every method.
    pub const ALL: [Method; 2] = [Method::Flux, Method::Network];

    /// The name in settings and stored analyses.
    pub fn id(self) -> &'static str {
        match self {
            Method::Flux => "flux",
            Method::Network => "network",
        }
    }

    /// The name people read.
    pub fn label(self) -> &'static str {
        match self {
            Method::Flux => "Spectral flux",
            Method::Network => "Beat This! network",
        }
    }

    /// The method named `id`.
    pub fn parse(id: &str) -> Option<Method> {
        Method::ALL.into_iter().find(|m| m.id() == id)
    }

    /// The candidates' trackers with their say in the vote and the meter.
    /// The network's say is larger: it knows tempo levels flux does not.
    fn trackers(self) -> Vec<(Box<dyn Tracker>, f64)> {
        match self {
            Method::Flux => vec![
                (Box::new(Hmm::default()), 1.0),
                (Box::new(Plp::default()), 0.75),
            ],
            Method::Network => vec![
                (Box::new(Hmm::default()), 1.5),
                (Box::new(Peaks::default()), 1.5),
                (Box::new(Plp::default()), 1.0),
            ],
        }
    }
}

/// One candidate's result, for reports.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateReport {
    /// `activation/tracker`.
    pub name: String,
    /// Its median tempo, 0 without beats.
    pub bpm: f64,
    /// Its mean agreement with the other candidates.
    pub agreement: f64,
}

/// The analysis of a track.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Analysis {
    /// The track's length, in seconds.
    pub duration: f64,
    /// The beats, in seconds, ascending.
    pub beats: Vec<f64>,
    /// The indices of the beats that start a bar.
    pub downbeats: Vec<usize>,
    /// Each beat's local tempo, in beats per minute.
    pub local_bpm: Vec<f64>,
    /// The tempo in one look, from where the beat is clear; `None` when
    /// it is clear nowhere.
    pub summary: Option<Summary>,
    /// How much the candidates agree with the result, 0 to 1.
    pub confidence: f64,
    /// The result in regions of 16 beats with their own agreement.
    pub regions: Vec<Region>,
    /// The candidate whose grid the beats came from (`activation/tracker`).
    pub chosen: String,
    /// How the beats group that grid's pulse, stretch by stretch (see
    /// [`meter::describe`]).
    pub groupings: Vec<meter::Span>,
    /// The methods that ran.
    pub methods: Vec<Method>,
    /// Whether the cues came from separated stems (else from the fast
    /// split of the mix).
    pub stems: bool,
    /// Every candidate.
    pub candidates: Vec<CandidateReport>,
    /// The key the track is in ([`key::key_of`], from the track at
    /// 44.1 kHz, which the caller hears it at); `None` until found.
    pub key: Option<key::Key>,
}

impl Analysis {
    /// The index of the last beat at or before `time`.
    pub fn beat_at(&self, time: f64) -> Option<usize> {
        self.beats.partition_point(|&b| b <= time).checked_sub(1)
    }

    /// The local tempo at `time`: that of the beat at or before it (the
    /// first beat before the first).
    pub fn bpm_at(&self, time: f64) -> Option<f64> {
        let i = self.beat_at(time).unwrap_or(0);
        self.local_bpm.get(i).copied()
    }

    /// Whether beat `index` starts a bar.
    pub fn is_downbeat(&self, index: usize) -> bool {
        self.downbeats.binary_search(&index).is_ok()
    }

    /// The region around `time`, if one covers it.
    pub fn region_at(&self, time: f64) -> Option<&Region> {
        let i = self
            .regions
            .partition_point(|r| r.start <= time)
            .checked_sub(1)?;
        let region = &self.regions[i];
        // Up to the next region's start, or a beat past the last one's end.
        let end = match self.regions.get(i + 1) {
            Some(next) => next.start,
            None => region.end + 60.0 / self.bpm_at(region.end).unwrap_or(60.0),
        };
        (time < end).then_some(region)
    }

    /// Whether the beat is clear at `time`: the candidates agree there.
    pub fn is_clear_at(&self, time: f64) -> bool {
        self.region_at(time).is_some_and(|r| r.confidence >= CLEAR)
    }

    /// Fills in [`Analysis::local_bpm`] and [`Analysis::summary`] from the
    /// beats and the regions: the summary counts only the beats where the
    /// beat is clear, so an intro without a beat does not blur it.
    pub fn derive(&mut self) {
        self.local_bpm = summary::local_bpm(&self.beats);
        let trusted: Vec<bool> = self.beats.iter().map(|&b| self.is_clear_at(b)).collect();
        self.summary = summary::summarize(&self.beats, &self.local_bpm, &trusted);
    }
}

/// Takes mono samples at [`SAMPLE_RATE`] as they come and analyses them
/// at the end.
pub struct Analyzer {
    methods: Vec<Method>,
    /// The flux spectrogram; the cues and the alignment come from it, so
    /// it always runs.
    frontend: Frontend,
    network: Option<NetFrontend>,
    /// Where the network's layers run, if not here ([`net::NetRunner`]).
    runner: Option<&'static dyn net::NetRunner>,
    samples: u64,
}

impl Default for Analyzer {
    fn default() -> Analyzer {
        Analyzer::new()
    }
}

/// Everything the analysis heard in a track, before it decides on the
/// beat.
#[derive(Clone, Debug, Default)]
pub struct Evidence {
    /// The track's length, in seconds.
    pub duration: f64,
    /// The methods that ran.
    pub methods: Vec<Method>,
    /// Whether the cues came from separated stems.
    pub stems: bool,
    /// Full-band onsets, which the beats are moved onto at the end.
    pub onsets: Curve,
    /// What a listener hears in the beat.
    pub cues: Cues,
    /// Every method's candidates, the network's first (the first wins
    /// ties), the stems' last.
    pub candidates: Vec<vote::Candidate>,
    /// The network's downbeat probability, when it ran.
    pub downbeat: Option<Curve>,
}

/// How far a beat may move onto the onset under it, in seconds.
const ALIGN: f64 = 0.03;
/// How strong that onset must be (the flux is 0 to 1).
const ONSET: f32 = 0.2;

/// Moves each beat onto the strongest full-band onset within [`ALIGN`]
/// of it, where there is a clear one. The network places beats only to
/// within a frame or so (20 ms); the onsets are sharper, at 100 frames a
/// second.
fn align(beats: &mut [f64], onsets: &Curve) {
    let fps = onsets.fps;
    let v = &onsets.values;
    for b in beats.iter_mut() {
        let lo = ((*b - ALIGN) * fps).ceil().max(0.0) as usize;
        let hi = ((*b + ALIGN) * fps).floor().max(0.0) as usize + 1;
        if let Some(frame) = track::peak_in(v, lo, hi) {
            if v[frame.round() as usize] >= ONSET {
                *b = frame / fps;
            }
        }
    }
}

/// The candidates of `method`'s trackers on `curve`, tracked side by side.
fn candidates_of(method: Method, name: &str, curve: &Curve) -> Vec<vote::Candidate> {
    std::thread::scope(|s| {
        let runs: Vec<_> = method
            .trackers()
            .into_iter()
            .map(|(tracker, weight)| {
                s.spawn(move || {
                    vote::Candidate::new(
                        format!("{name}/{}", tracker.name()),
                        tracker.track(curve),
                        weight,
                    )
                })
            })
            .collect();
        runs.into_iter()
            .map(|h| h.join().expect("a tracker"))
            .collect()
    })
}

/// The tracker of the drum stem's flux, and its say in the vote.
///
/// Measured on a list of 59 tracks: the drum stem through the HMM agreed
/// with the other candidates 0.83 on average and raised the vote's mean
/// confidence; through the pulse tracker it agreed 0.75 and the bass
/// stem 0.37 (a bass line's onsets are rarely periodic), and both
/// lowered it. So the drums alone vote, through the HMM.
const DRUMS_WEIGHT: f64 = 1.0;

/// The drum stem's candidate, from parts a stem separator gave and the
/// cues of them (none from the fast split). Its flux is tracked where the
/// drums are present (it is zero elsewhere), and the candidate keeps its
/// beats and its say where they also pulse ([`presence`]).
fn stem_candidates(parts: &Parts, cues: &Cues) -> Vec<vote::Candidate> {
    let Some(present) = &cues.drums_present else {
        return Vec::new();
    };
    let onsets = Flux::FULL.curve(&parts.drums);
    let say = presence::say(present, &presence::pulsing(&onsets));
    let tracker = Hmm::default();
    let mut beats = tracker.track(&presence::scaled(&onsets, present));
    beats.retain(|&b| presence::at(&say, b) >= presence::KEEP);
    vec![vote::Candidate {
        say: Some(say),
        ..vote::Candidate::new(format!("drums/{}", tracker.name()), beats, DRUMS_WEIGHT)
    }]
}

/// The logistic function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A curve of the network's probabilities; all zero when it found no
/// beat anywhere, so the trackers find none either.
fn probabilities(logits: &[f32]) -> Curve {
    let mut values: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
    if values.iter().all(|&v| v < 0.5) {
        values.fill(0.0);
    }
    Curve {
        values,
        fps: net::mel::FPS,
    }
}

/// A candidate's median tempo, 0 without beats.
fn median_bpm(beats: &[f64]) -> f64 {
    summary::summarize(beats, &summary::local_bpm(beats), &vec![true; beats.len()])
        .map_or(0.0, |s| s.bpm)
}

impl Analyzer {
    /// An analyzer for one track with every method.
    pub fn new() -> Analyzer {
        Analyzer::with_methods(&Method::ALL)
    }

    /// An analyzer with `methods` (every method if none).
    pub fn with_methods(methods: &[Method]) -> Analyzer {
        let mut methods = methods.to_vec();
        if methods.is_empty() {
            methods = Method::ALL.to_vec();
        }
        methods.sort_unstable();
        methods.dedup();
        Analyzer {
            network: methods.contains(&Method::Network).then(NetFrontend::new),
            runner: None,
            methods,
            frontend: Frontend::new(FrontendConfig::FLUX),
            samples: 0,
        }
    }

    /// Runs the network's layers on `runner` (a GPU) rather than here; a
    /// piece it fails on still runs here.
    pub fn run_network_on(&mut self, runner: &'static dyn net::NetRunner) {
        self.runner = Some(runner);
    }

    /// Takes in more samples (full scale ±1).
    pub fn push(&mut self, samples: &[f32]) {
        self.samples += samples.len() as u64;
        self.frontend.push(samples);
        if let Some(network) = &mut self.network {
            network.push(samples);
        }
    }

    /// Everything heard in what was pushed: the cues (from the fast split
    /// into parts) and every method's candidates.
    pub fn listen(self) -> Evidence {
        self.listen_with(|| None)
    }

    /// Everything heard in what was pushed, the cues from the parts
    /// `parts` gives when it gives some (the stems, see [`stems`]), else
    /// from the fast split. `parts` is called on a thread of its own while
    /// the network and the flux candidates are worked out, so the stems
    /// may still be finishing elsewhere when this is called.
    pub fn listen_with(self, parts: impl FnOnce() -> Option<Parts> + Send) -> Evidence {
        let duration = self.samples as f64 / f64::from(SAMPLE_RATE);
        let mel = self.frontend.finish();
        let flux = self.methods.contains(&Method::Flux);
        let (logits, (cues, stems, stem_candidates), onsets, flux_candidates) =
            std::thread::scope(|s| {
                let runner = self.runner;
                let network = self.network.map(|frontend| {
                    s.spawn(move || Network::shared().predict_with(&frontend.finish(), runner))
                });
                let cues = s.spawn(|| match parts() {
                    Some(parts) => {
                        let cues = Cues::new(&parts);
                        let found = if flux {
                            stem_candidates(&parts, &cues)
                        } else {
                            Vec::new()
                        };
                        (cues, true, found)
                    }
                    None => (Cues::new(&Parts::split(&mel)), false, Vec::new()),
                });
                let onsets = Flux::FULL.curve(&mel);
                let mut flux_candidates = Vec::new();
                if flux {
                    flux_candidates.extend(candidates_of(Method::Flux, Flux::FULL.name(), &onsets));
                    let low = Flux::LOW.curve(&mel);
                    flux_candidates.extend(candidates_of(Method::Flux, Flux::LOW.name(), &low));
                }
                let logits = network.map(|h| h.join().expect("the network"));
                (
                    logits,
                    cues.join().expect("the cues"),
                    onsets,
                    flux_candidates,
                )
            });
        let mut candidates = Vec::new();
        let mut downbeat = None;
        if let Some(logits) = logits {
            candidates.extend(candidates_of(
                Method::Network,
                "net",
                &probabilities(&logits.beat),
            ));
            downbeat = Some(probabilities(&logits.downbeat));
        }
        candidates.extend(flux_candidates);
        candidates.extend(stem_candidates);
        Evidence {
            duration,
            methods: self.methods,
            stems,
            onsets,
            cues,
            candidates,
            downbeat,
        }
    }

    /// Analyses what was pushed.
    pub fn finish(self) -> Analysis {
        self.listen().conclude()
    }
}

impl Evidence {
    /// The analysis from what was heard, with the meter's weights fitted
    /// for where its cues came from (stems or the mix).
    pub fn conclude(&self) -> Analysis {
        self.decide(&meter::Weights::for_parts(self.stems))
    }

    /// The analysis from what was heard, with the meter's `weights`.
    pub fn decide(&self, weights: &meter::Weights) -> Analysis {
        let outcome = vote::vote(&self.candidates);
        let candidates = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| CandidateReport {
                name: c.name.clone(),
                bpm: median_bpm(&c.beats),
                agreement: outcome.as_ref().map_or(0.0, |o| o.agreements[i]),
            })
            .collect();
        let mut analysis = Analysis {
            duration: self.duration,
            candidates,
            methods: self.methods.clone(),
            stems: self.stems,
            ..Analysis::default()
        };
        let Some(outcome) = outcome else {
            return analysis;
        };
        let grid = &self.candidates[outcome.chosen].beats;
        let (mut beats, groupings) = meter::beats(grid, &self.candidates, &self.cues, weights);
        align(&mut beats, &self.onsets);
        analysis.downbeats = meter::downbeats(&beats, &self.cues, self.downbeat.as_ref());
        analysis.regions = vote::regions(&beats, &self.candidates, outcome.chosen);
        analysis.beats = beats;
        analysis.groupings = groupings;
        analysis.chosen = self.candidates[outcome.chosen].name.clone();
        analysis.confidence = outcome.confidence;
        analysis.derive();
        analysis
    }
}

/// Analyses a whole track of mono samples at [`SAMPLE_RATE`] with every
/// method.
pub fn analyze(samples: &[f32]) -> Analysis {
    analyze_with(samples, &Method::ALL)
}

/// Analyses a whole track of mono samples at [`SAMPLE_RATE`] with
/// `methods`.
pub fn analyze_with(samples: &[f32], methods: &[Method]) -> Analysis {
    let mut analyzer = Analyzer::with_methods(methods);
    analyzer.push(samples);
    analyzer.finish()
}
