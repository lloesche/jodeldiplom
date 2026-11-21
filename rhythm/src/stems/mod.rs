//! Stems: the music separated into bass, drums, other instruments and
//! vocals, by our own inference of the Mini-BS-RoFormer network.
//!
//! The network runs on 6-second chunks of stereo at 44.1 kHz
//! ([`RATE`]): the chunk's STFT goes in, a complex mask per
//! stem comes out ([`Separator::masks`]), and each stem is the mask times
//! the mixture, transformed back. Chunks overlap by 1.2 s and are
//! cross-faded (a linear fade of 0.6 s at each end, the overlap added
//! and divided by the summed fades), as the network's own inference does,
//! with less overlap.
//!
//! The network's work is a [`Separator`], so it can run elsewhere: the
//! built-in [`CpuSeparator`] is the reference, and a GPU crate gives
//! another. [`Splitter`] feeds a separator chunk by chunk as the samples
//! come and hands on the finished stems; [`PartsBuilder`] turns them into
//! the [`Parts`] the cues listen to.

mod model;
pub(crate) mod stft;

use std::sync::OnceLock;

use crate::fft::Complex;
use crate::frontend::{Frontend, FrontendConfig};
use crate::net::layers::par_chunks;
use crate::net::weights::Weights;
use crate::parts::Parts;
use crate::presence::Levels;
use crate::waveform::LaneBuilder;
use model::{Model, FRAME};
pub use model::{BAND_BINS, DIM};
use stft::{Stft, BINS};

/// The sample rate the separator takes, in hertz.
pub const RATE: u32 = 44_100;
/// The stems, in the network's order.
pub const STEMS: usize = 4;
/// Samples per chunk (6 s).
pub const CHUNK: usize = 6 * RATE as usize;
/// Samples between chunks (4.8 s: 1.2 s of overlap).
pub const STEP: usize = CHUNK * 4 / 5;

/// The built-in weights.
static WEIGHTS: &[u8] = include_bytes!("../../models/mini_bs_roformer.bin");

/// A stem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stem {
    /// Bass guitar, synth bass, double bass.
    Bass,
    /// Drums and percussion.
    Drums,
    /// Everything else: guitars, keys, strings.
    Other,
    /// Voices.
    Vocals,
}

impl Stem {
    /// The stems, in the network's order.
    pub const ALL: [Stem; STEMS] = [Stem::Bass, Stem::Drums, Stem::Other, Stem::Vocals];

    /// Its name in files.
    pub fn id(self) -> &'static str {
        match self {
            Stem::Bass => "bass",
            Stem::Drums => "drums",
            Stem::Other => "other",
            Stem::Vocals => "vocals",
        }
    }
}

/// Something that runs the network: from a chunk's STFT to the stems'
/// masks.
pub trait Separator: Send + Sync {
    /// What it runs on, for reports.
    fn name(&self) -> String;

    /// The masks of a chunk of `frames` frames: `stft` holds, per frame,
    /// each bin's two channels' real and imaginary parts (`frames × 4100`
    /// values); the result holds [`STEMS`] blocks of masks in the same
    /// layout.
    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String>;
}

/// The network on the processor: the reference, and slow (several minutes
/// for a track).
pub struct CpuSeparator {
    model: Model,
}

impl CpuSeparator {
    /// The built-in network, loaded on first use.
    pub fn shared() -> &'static CpuSeparator {
        static SEPARATOR: OnceLock<CpuSeparator> = OnceLock::new();
        SEPARATOR.get_or_init(|| CpuSeparator {
            model: Model::load(weights()).expect("the built-in separator weights"),
        })
    }
}

/// The built-in weights, parsed (for other separators to load).
pub fn weights() -> Weights {
    Weights::parse(WEIGHTS).expect("the built-in separator weights")
}

impl Separator for CpuSeparator {
    fn name(&self) -> String {
        "the processor".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        Ok(self.model.masks(frames, stft, crate::net::threads()))
    }
}

/// The fade over a chunk: a linear rise over its first tenth, a fall over
/// its last.
fn fade() -> Vec<f32> {
    let edge = CHUNK / 10;
    (0..CHUNK)
        .map(|i| {
            let rise = i.min(CHUNK - 1 - i);
            if rise < edge {
                rise as f32 / (edge - 1) as f32
            } else {
                1.0
            }
        })
        .collect()
}

/// The two channels' STFTs packed for [`Separator::masks`].
fn pack(channels: &[Vec<Complex>; 2]) -> (usize, Vec<f32>) {
    let frames = channels[0].len() / BINS;
    let mut packed = vec![0.0f32; frames * FRAME];
    for t in 0..frames {
        for f in 0..BINS {
            for (c, spectrum) in channels.iter().enumerate() {
                let x = spectrum[t * BINS + f];
                packed[t * FRAME + (f * 2 + c) * 2] = x.re;
                packed[t * FRAME + (f * 2 + c) * 2 + 1] = x.im;
            }
        }
    }
    (frames, packed)
}

/// The STFT of stereo `left`/`right` (more than half a frame long) as
/// [`Separator::masks`] takes it: the frame count and the packed values.
pub fn packed_stft(left: &[f32], right: &[f32]) -> (usize, Vec<f32>) {
    let mut stft = Stft::new();
    pack(&[stft.forward(left), stft.forward(right)])
}

/// A chunk transformed, waiting for its masks.
struct Forward {
    /// Each channel's STFT.
    channels: [Vec<Complex>; 2],
    /// Its frames.
    frames: usize,
    /// The samples per channel.
    len: usize,
}

/// The STFTs of one chunk of stereo `left`/`right` (the channels side by
/// side on the analysis's threads), and the values packed for
/// [`Separator::masks`].
fn forward(left: &[f32], right: &[f32]) -> (Forward, Vec<f32>) {
    let threads = crate::net::threads();
    let mut spectra: Vec<Vec<Complex>> = vec![Vec::new(); 2];
    par_chunks(&mut spectra, 1, threads, |c, slot| {
        slot[0] = Stft::new().forward(if c == 0 { left } else { right });
    });
    let channels: [Vec<Complex>; 2] = [
        std::mem::take(&mut spectra[0]),
        std::mem::take(&mut spectra[1]),
    ];
    let (frames, packed) = pack(&channels);
    (
        Forward {
            channels,
            frames,
            len: left.len(),
        },
        packed,
    )
}

/// A transformed chunk's stems through its `masks`: `[stem][channel]`
/// samples, each stem and channel turned back side by side on the
/// analysis's threads.
fn inverse(chunk: &Forward, masks: &[f32]) -> Result<Vec<[Vec<f32>; 2]>, String> {
    let (frames, len, channels) = (chunk.frames, chunk.len, &chunk.channels);
    if masks.len() != STEMS * frames * FRAME {
        return Err(format!("{} masks for {frames} frames", masks.len()));
    }
    let threads = crate::net::threads();
    let inverse = |s: usize, c: usize| {
        let mask = &masks[s * frames * FRAME..][..frames * FRAME];
        let spectrum = &channels[c];
        let masked: Vec<Complex> = (0..frames * BINS)
            .map(|i| {
                let (t, f) = (i / BINS, i % BINS);
                let at = t * FRAME + (f * 2 + c) * 2;
                spectrum[i].mul(Complex::new(mask[at], mask[at + 1]))
            })
            .collect();
        Stft::new().inverse(&masked, len)
    };
    let mut back: Vec<Vec<f32>> = vec![Vec::new(); 2 * STEMS];
    par_chunks(&mut back, 1, threads, |i, slot| {
        slot[0] = inverse(i / 2, i % 2)
    });
    let mut back = back.into_iter();
    Ok((0..STEMS)
        .map(|_| {
            [
                back.next().unwrap_or_default(),
                back.next().unwrap_or_default(),
            ]
        })
        .collect())
}

/// Finished stems: from sample `start` (of the track), per stem its left
/// and right channel.
pub struct StemBlock<'a> {
    /// The first sample's place in the track.
    pub start: usize,
    /// `[stem][channel]` samples.
    pub stems: [[&'a [f32]; 2]; STEMS],
}

/// A thread running a separator on chunks sent to it, in order, so the
/// caller transforms other chunks meanwhile.
struct Worker {
    to: Option<std::sync::mpsc::SyncSender<(usize, Vec<f32>)>>,
    from: std::sync::mpsc::Receiver<Result<Vec<f32>, String>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    fn start(separator: &'static dyn Separator) -> Worker {
        let (to, jobs) = std::sync::mpsc::sync_channel::<(usize, Vec<f32>)>(2);
        let (done, from) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("jodel-stems-masks".into())
            .spawn(move || {
                for (frames, packed) in jobs {
                    if done.send(separator.masks(frames, &packed)).is_err() {
                        return;
                    }
                }
            })
            .ok();
        Worker {
            to: Some(to),
            from,
            thread,
        }
    }

    fn send(&self, frames: usize, packed: Vec<f32>) -> Result<(), String> {
        self.to
            .as_ref()
            .and_then(|to| to.send((frames, packed)).ok())
            .ok_or_else(|| "the separator's thread stopped".to_owned())
    }

    fn receive(&self) -> Result<Vec<f32>, String> {
        self.from
            .recv()
            .map_err(|_| "the separator's thread stopped".to_owned())?
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the jobs ends the thread once it has done the one it has.
        self.to.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A chunk on the separator's thread: where it starts, and its transform.
struct InFlight {
    start: usize,
    chunk: Forward,
}

/// Separates stereo at [`RATE`] chunk by chunk as it comes, handing each
/// stretch of finished stems to a sink.
///
/// [`Splitter::new`] runs each chunk through in turn: transform, masks,
/// transform back. [`Splitter::overlapped`] has a thread of its own wait
/// for the masks (on the GPU), so the next chunk is transformed and the
/// one before turned back meanwhile: the same stems, sooner, but each
/// handed on only once the next chunk is in (or at [`Splitter::finish`]).
pub struct Splitter<'a> {
    separator: &'a dyn Separator,
    worker: Option<Worker>,
    /// The chunk whose masks the worker is making.
    in_flight: Option<InFlight>,
    fade: Vec<f32>,
    /// Input from sample `base` on, per channel.
    input: [Vec<f32>; 2],
    /// The faded stems' sum and the fades' sum, from `base` on.
    sum: Vec<[Vec<f32>; 2]>,
    weight: Vec<f32>,
    base: usize,
    /// The next chunk's first sample.
    next: usize,
    /// Samples taken in.
    taken: usize,
}

impl<'a> Splitter<'a> {
    /// A splitter running `separator`, a chunk at a time.
    pub fn new(separator: &'a dyn Separator) -> Splitter<'a> {
        Splitter {
            separator,
            worker: None,
            in_flight: None,
            fade: fade(),
            input: Default::default(),
            sum: (0..STEMS).map(|_| Default::default()).collect(),
            weight: Vec::new(),
            base: 0,
            next: 0,
            taken: 0,
        }
    }

    /// Takes in more stereo samples (full scale ±1), separating every
    /// chunk that is complete and handing on what is finished (overlapped:
    /// up to the chunk on the worker).
    pub fn push(
        &mut self,
        left: &[f32],
        right: &[f32],
        sink: &mut dyn FnMut(&StemBlock),
    ) -> Result<(), String> {
        self.input[0].extend_from_slice(left);
        self.input[1].extend_from_slice(right);
        self.taken += left.len();
        while self.next + CHUNK <= self.taken {
            self.chunk()?;
            self.emit(self.finished(), sink);
        }
        Ok(())
    }

    /// Separates the rest (the last chunk padded with silence) and hands
    /// on everything.
    pub fn finish(mut self, sink: &mut dyn FnMut(&StemBlock)) -> Result<(), String> {
        while self.next < self.taken && (self.next == 0 || self.next + CHUNK - STEP < self.taken) {
            self.chunk()?;
        }
        self.land()?;
        let end = self.taken;
        self.emit(end, sink);
        Ok(())
    }

    /// Where the stems stop being finished: the first sample a chunk not
    /// added in yet still adds to.
    fn finished(&self) -> usize {
        self.in_flight.as_ref().map_or(self.next, |f| f.start)
    }

    /// Separates the chunk at `next`, at once or (overlapped) by sending it
    /// to the worker and adding in the one sent before.
    fn chunk(&mut self) -> Result<(), String> {
        let start = self.next;
        let at = start - self.base;
        let piece = |c: &Vec<f32>| {
            let mut p = c[at.min(c.len())..].to_vec();
            p.resize(CHUNK, 0.0);
            p
        };
        let (left, right) = (piece(&self.input[0]), piece(&self.input[1]));
        self.next += STEP;
        let (chunk, packed) = forward(&left, &right);
        let Some(worker) = &self.worker else {
            let masks = self.separator.masks(chunk.frames, &packed)?;
            let stems = inverse(&chunk, &masks)?;
            self.add(start, &stems);
            return Ok(());
        };
        // This one goes to the worker first, so it starts on it as soon as
        // the one before is done; that one is turned back meanwhile.
        worker.send(chunk.frames, packed)?;
        let before = self.in_flight.replace(InFlight { start, chunk });
        if let Some(before) = before {
            let masks = worker.receive()?;
            let stems = inverse(&before.chunk, &masks)?;
            self.add(before.start, &stems);
        }
        Ok(())
    }

    /// Adds in the chunk on the worker, if any.
    fn land(&mut self) -> Result<(), String> {
        if let (Some(worker), Some(last)) = (&self.worker, self.in_flight.take()) {
            let masks = worker.receive()?;
            let stems = inverse(&last.chunk, &masks)?;
            self.add(last.start, &stems);
        }
        Ok(())
    }

    /// Adds the stems of the chunk from sample `start` in, faded.
    fn add(&mut self, start: usize, stems: &[[Vec<f32>; 2]]) {
        let at = start - self.base;
        let need = at + CHUNK;
        if self.weight.len() < need {
            self.weight.resize(need, 0.0);
            for stem in &mut self.sum {
                for c in stem.iter_mut() {
                    c.resize(need, 0.0);
                }
            }
        }
        for (i, f) in self.fade.iter().enumerate() {
            self.weight[at + i] += f;
        }
        for (sum, stem) in self.sum.iter_mut().zip(stems) {
            for (c, channel) in sum.iter_mut().zip(stem) {
                for (i, (s, f)) in channel.iter().zip(&self.fade).enumerate() {
                    c[at + i] += s * f;
                }
            }
        }
    }

    /// Hands on the finished stems before sample `until` and drops them.
    fn emit(&mut self, until: usize, sink: &mut dyn FnMut(&StemBlock)) {
        let n = (until - self.base).min(self.weight.len());
        if n == 0 {
            return;
        }
        let mut done: Vec<[Vec<f32>; 2]> = Vec::with_capacity(STEMS);
        for stem in &self.sum {
            done.push([0, 1].map(|c| {
                (0..n)
                    .map(|i| stem[c][i] / self.weight[i].max(1e-8))
                    .collect()
            }));
        }
        let stems: [[&[f32]; 2]; STEMS] =
            std::array::from_fn(|s| [done[s][0].as_slice(), done[s][1].as_slice()]);
        sink(&StemBlock {
            start: self.base,
            stems,
        });
        self.weight.drain(..n);
        for stem in &mut self.sum {
            for c in stem.iter_mut() {
                c.drain(..n);
            }
        }
        for c in &mut self.input {
            c.drain(..n.min(c.len()));
        }
        self.base += n;
    }
}

impl Splitter<'static> {
    /// A splitter whose `separator` makes a chunk's masks on a thread of
    /// its own while the next chunk is transformed and the one before
    /// turned back (see [`Splitter`]).
    pub fn overlapped(separator: &'static dyn Separator) -> Splitter<'static> {
        Splitter {
            worker: Some(Worker::start(separator)),
            ..Splitter::new(separator)
        }
    }
}

/// The front end for the parts from stems: the fast split's mel bands,
/// at [`RATE`] (2,048-sample frames every 441 samples, 100 a second).
pub const PARTS_FRONTEND: FrontendConfig = FrontendConfig {
    sample_rate: RATE,
    frame: 2048,
    hop: 441,
    bands: 128,
    low_hz: 30.0,
    high_hz: 10_000.0,
};

/// Builds the [`Parts`] from finished stems: the drums as `drums`, bass
/// and other instruments as `tonal` (the vocals are left out: sung
/// syllables would blur the chord changes), each in mono; and the energy
/// of the drums, the bass and all stems together in each 441-sample frame
/// (100 a second, the frontend's hop); and each stem's waveform.
pub struct PartsBuilder {
    drums: Frontend,
    tonal: Frontend,
    mono: Vec<f32>,
    levels: Levels,
    lanes: [LaneBuilder; STEMS],
    /// The energy summed so far in the frame being filled, and its samples.
    open: [f64; 3],
    filled: usize,
}

impl Default for PartsBuilder {
    fn default() -> PartsBuilder {
        PartsBuilder::new()
    }
}

/// Samples per energy frame.
const LEVEL_HOP: usize = PARTS_FRONTEND.hop;

impl PartsBuilder {
    /// An empty builder.
    pub fn new() -> PartsBuilder {
        PartsBuilder {
            drums: Frontend::new(PARTS_FRONTEND),
            tonal: Frontend::new(PARTS_FRONTEND),
            mono: Vec::new(),
            levels: Levels {
                fps: f64::from(RATE) / LEVEL_HOP as f64,
                ..Levels::default()
            },
            open: [0.0; 3],
            filled: 0,
            lanes: std::array::from_fn(|_| LaneBuilder::new(RATE)),
        }
    }

    /// Takes in a block of finished stems.
    pub fn take(&mut self, block: &StemBlock) {
        let n = block.stems[0][0].len();
        let mut tonal = Vec::with_capacity(n);
        self.mono.clear();
        for i in 0..n {
            let mono: [f32; STEMS] =
                std::array::from_fn(|s| 0.5 * (block.stems[s][0][i] + block.stems[s][1][i]));
            for (lane, &x) in self.lanes.iter_mut().zip(&mono) {
                lane.push_one(x);
            }
            let [bass, drums, other, vocals] = mono;
            self.mono.push(drums);
            tonal.push(bass + other);
            let all = bass + drums + other + vocals;
            self.open[0] += f64::from(drums * drums);
            self.open[1] += f64::from(bass * bass);
            self.open[2] += f64::from(all * all);
            self.filled += 1;
            if self.filled == LEVEL_HOP {
                self.close_level();
            }
        }
        self.drums.push(&self.mono);
        self.tonal.push(&tonal);
    }

    /// Ends the energy frame being filled.
    fn close_level(&mut self) {
        let n = self.filled.max(1) as f64;
        self.levels.drums.push((self.open[0] / n) as f32);
        self.levels.bass.push((self.open[1] / n) as f32);
        self.levels.total.push((self.open[2] / n) as f32);
        self.open = [0.0; 3];
        self.filled = 0;
    }

    /// The parts.
    pub fn finish(mut self) -> Parts {
        if self.filled > 0 {
            self.close_level();
        }
        Parts {
            drums: self.drums.finish(),
            tonal: self.tonal.finish(),
            levels: Some(self.levels),
            lanes: Some(self.lanes.map(LaneBuilder::finish)),
        }
    }
}

/// The parts of a whole stereo track at [`RATE`], through `separator`.
pub fn parts(separator: &dyn Separator, left: &[f32], right: &[f32]) -> Result<Parts, String> {
    let mut builder = PartsBuilder::new();
    let mut splitter = Splitter::new(separator);
    splitter.push(left, right, &mut |b| builder.take(b))?;
    splitter.finish(&mut |b| builder.take(b))?;
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clip of the reference run (`tests/fixtures/stems.txt`): 3 s of
    /// bass, a tone on each side, kicks every half second and hi-hat noise
    /// between them.
    fn clip() -> [Vec<f32>; 2] {
        let n = 3 * RATE as usize;
        let sr = f64::from(RATE);
        let mut seed: u64 = 0x1234_5678;
        let (mut left, mut right) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            seed = (seed * 1_664_525 + 1_013_904_223) % (1 << 32);
            let noise = (seed >> 8) as f64 / (1u64 << 24) as f64 - 0.5;
            let t = i as f64 / sr;
            let kick_t = t % 0.5;
            let kick = 0.6 * (2.0 * PI * 60.0 * kick_t).sin() * (-kick_t / 0.08).exp();
            let hat_t = (t - 0.25).rem_euclid(0.5);
            let hat = if t >= 0.25 {
                0.4 * noise * (-hat_t / 0.02).exp()
            } else {
                0.0
            };
            let bass = 0.3 * (2.0 * PI * 110.0 * t).sin();
            left.push((bass + kick + hat + 0.2 * (2.0 * PI * 440.0 * t).sin()) as f32);
            right.push((bass + kick + hat + 0.2 * (2.0 * PI * 554.37 * t + 0.5).sin()) as f32);
        }
        [left, right]
    }

    use std::f64::consts::PI;

    /// The clip's STFT, packed for the network, and its masks.
    fn masks_of_clip() -> (usize, Vec<Complex>, Vec<Complex>, Vec<f32>) {
        let [left, right] = clip();
        let mut stft = Stft::new();
        let (l, r) = (stft.forward(&left), stft.forward(&right));
        let frames = l.len() / BINS;
        let mut packed = vec![0.0f32; frames * FRAME];
        for t in 0..frames {
            for f in 0..BINS {
                for (c, x) in [l[t * BINS + f], r[t * BINS + f]].into_iter().enumerate() {
                    packed[t * FRAME + (f * 2 + c) * 2] = x.re;
                    packed[t * FRAME + (f * 2 + c) * 2 + 1] = x.im;
                }
            }
        }
        let masks = CpuSeparator::shared().masks(frames, &packed).unwrap();
        (frames, l, r, masks)
    }

    /// Each stem's masked magnitude summed per frame and per band, as in
    /// the fixture.
    fn summary(
        frames: usize,
        l: &[Complex],
        r: &[Complex],
        masks: &[f32],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        (0..STEMS)
            .map(|s| {
                let mask = &masks[s * frames * FRAME..][..frames * FRAME];
                let mut per_frame = vec![0.0f64; frames];
                let mut per_bin = vec![0.0f64; BINS];
                for t in 0..frames {
                    for f in 0..BINS {
                        for (c, x) in [l[t * BINS + f], r[t * BINS + f]].into_iter().enumerate() {
                            let at = t * FRAME + (f * 2 + c) * 2;
                            let m = Complex::new(mask[at], mask[at + 1]);
                            let v = f64::from(x.mul(m).norm());
                            per_frame[t] += v;
                            per_bin[f] += v;
                        }
                    }
                }
                let mut per_band = Vec::new();
                let mut start = 0;
                for &b in &BAND_BINS {
                    per_band.push(per_bin[start..start + b].iter().sum());
                    start += b;
                }
                (per_frame, per_band)
            })
            .collect()
    }

    /// Every stem's masked magnitude per frame and per band against the
    /// authors' PyTorch code on the same clip, within 0.1 % (of each
    /// stem's largest value).
    #[test]
    fn the_stems_match_the_reference() {
        let text = include_str!("../../tests/fixtures/stems.txt");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let frames: usize = lines
            .next()
            .unwrap()
            .strip_prefix("frames ")
            .unwrap()
            .parse()
            .unwrap();
        let numbers = |line: &str, word: &str| -> Vec<f64> {
            line.strip_prefix(word)
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect()
        };
        let (got_frames, l, r, masks) = masks_of_clip();
        assert_eq!(got_frames, frames);
        for (s, (per_frame, per_band)) in summary(frames, &l, &r, &masks).iter().enumerate() {
            let want_frame = numbers(lines.next().unwrap(), "frame ");
            let want_band = numbers(lines.next().unwrap(), "band ");
            for (got, want) in [(per_frame, &want_frame), (per_band, &want_band)] {
                assert_eq!(got.len(), want.len());
                let top = want.iter().fold(0.0f64, |m, v| m.max(*v));
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g - w).abs() <= 0.001 * top,
                        "stem {s} #{i}: {g} vs {w} (top {top})"
                    );
                }
            }
        }
    }

    /// Passes the whole mixture into stem `into`, nothing into the others.
    struct Passthrough {
        into: usize,
    }

    impl Separator for Passthrough {
        fn name(&self) -> String {
            "passthrough".into()
        }

        fn masks(&self, frames: usize, _stft: &[f32]) -> Result<Vec<f32>, String> {
            let mut masks = vec![0.0f32; STEMS * frames * FRAME];
            for v in masks[self.into * frames * FRAME..][..frames * FRAME].chunks_mut(2) {
                v[0] = 1.0;
            }
            Ok(masks)
        }
    }

    #[test]
    fn the_stem_lanes_follow_each_stem() {
        // 8 s of a tone, all of it passed to the vocals: their lane holds
        // the tone, the others nothing.
        let n = 8 * RATE as usize;
        let x: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * 0.03).sin()).collect();
        let separator = Passthrough { into: 3 };
        let parts = parts(&separator, &x, &x).unwrap();
        let lanes = parts.lanes.expect("the stem lanes");
        assert!(lanes.iter().all(|l| l.len() == 800), "{}", lanes[0].len());
        // -6 dB peaks, away from the track's faded ends.
        let vocals = lanes[3].peak_between(2.0, 6.0);
        assert!((vocals - 0.9).abs() < 0.02, "{vocals}");
        for lane in &lanes[..3] {
            assert!(lane.peak_between(2.0, 6.0) < 0.05);
        }
    }

    #[test]
    fn chunks_are_cross_faded_back_into_the_track() {
        // 15 s: three chunks, the last padded; pushed in uneven pieces.
        let n = 15 * RATE as usize;
        let signal = |c: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * (7 + c)) % 1009) as f32 / 1009.0 - 0.5)
                .collect()
        };
        let (left, right) = (signal(0), signal(1));
        let separator = Passthrough { into: 2 };
        let mut splitter = Splitter::new(&separator);
        let mut got: Vec<Vec<f32>> = vec![Vec::new(); 2 * STEMS];
        let mut next = 0;
        let mut sink = |b: &StemBlock| {
            assert_eq!(b.start, next);
            next += b.stems[0][0].len();
            for s in 0..STEMS {
                for c in 0..2 {
                    got[2 * s + c].extend_from_slice(b.stems[s][c]);
                }
            }
        };
        let mut at = 0;
        for piece in [1000, 300_000, 77_777, 200_000] {
            let end = (at + piece).min(n);
            splitter
                .push(&left[at..end], &right[at..end], &mut sink)
                .unwrap();
            at = end;
        }
        splitter.push(&left[at..], &right[at..], &mut sink).unwrap();
        splitter.finish(&mut sink).unwrap();
        assert_eq!(got[4].len(), n);
        // The first sample has no weight; every other one comes back.
        for (c, want) in [&left, &right].into_iter().enumerate() {
            for i in 1..n {
                let g = got[4 + c][i];
                assert!(
                    (g - want[i]).abs() < 1e-4,
                    "channel {c} sample {i}: {g} vs {}",
                    want[i]
                );
            }
        }
        assert!(got[0]
            .iter()
            .chain(&got[2])
            .chain(&got[6])
            .all(|v| v.abs() < 1e-6));
    }

    /// Masks that depend on the chunk (each value its own share), so a
    /// chunk paired with another's masks would show.
    struct ByContent;

    impl Separator for ByContent {
        fn name(&self) -> String {
            "by content".into()
        }

        fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
            let mut masks = vec![0.0f32; STEMS * frames * FRAME];
            for (s, block) in masks.chunks_mut(frames * FRAME).enumerate() {
                for (m, x) in block.iter_mut().zip(stft) {
                    *m = 1.0 / (1.0 + (x.abs() * (s + 1) as f32));
                }
            }
            Ok(masks)
        }
    }

    #[test]
    fn overlapped_the_stems_are_the_same_bit_for_bit() {
        static SEPARATOR: ByContent = ByContent;
        // 20 s pushed in uneven pieces: four chunks, the last padded.
        let n = 20 * RATE as usize;
        let signal = |c: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * (11 + c)) % 997) as f32 / 997.0 - 0.5)
                .collect()
        };
        let (left, right) = (signal(0), signal(1));
        let run = |mut splitter: Splitter<'static>| {
            let mut got: Vec<Vec<f32>> = vec![Vec::new(); 2 * STEMS];
            let mut next = 0;
            let mut sink = |b: &StemBlock| {
                assert_eq!(b.start, next, "every sample once, in order");
                next += b.stems[0][0].len();
                for s in 0..STEMS {
                    for c in 0..2 {
                        got[2 * s + c].extend_from_slice(b.stems[s][c]);
                    }
                }
            };
            let mut at = 0;
            for piece in [44_100, 300_000, 1_000, 250_000, 133_333] {
                let end = (at + piece).min(n);
                splitter
                    .push(&left[at..end], &right[at..end], &mut sink)
                    .unwrap();
                at = end;
            }
            splitter.push(&left[at..], &right[at..], &mut sink).unwrap();
            splitter.finish(&mut sink).unwrap();
            got
        };
        let one_by_one = run(Splitter::new(&SEPARATOR));
        let overlapped = run(Splitter::overlapped(&SEPARATOR));
        assert_eq!(one_by_one[0].len(), n);
        assert!(one_by_one == overlapped);
    }

    #[test]
    fn kicks_go_to_the_drums_and_a_chord_to_the_tonal_part() {
        // 6 s: a kick every half second over a held chord.
        let n = CHUNK;
        let sr = f64::from(RATE);
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let k = t % 0.5;
                let kick = 0.7
                    * (2.0 * PI * (50.0 + 100.0 * (-k / 0.04).exp()) * k).sin()
                    * (-k / 0.12).exp();
                let chord: f64 = [220.0, 277.18, 329.63]
                    .iter()
                    .map(|f| 0.08 * (2.0 * PI * f * t).sin())
                    .sum();
                (kick + chord) as f32
            })
            .collect();
        let parts = parts(CpuSeparator::shared(), &x, &x).unwrap();
        let energy =
            |mel: &crate::frontend::Mel, lo: f64, hi: f64, frames: &dyn Fn(usize) -> bool| {
                let bands = mel.bands_between(lo, hi);
                let (mut sum, mut count) = (0.0, 0.0);
                for t in (0..mel.frames()).filter(|&t| frames(t)) {
                    for b in bands.clone() {
                        sum += f64::from(mel.frame(t)[b]);
                        count += 1.0;
                    }
                }
                sum / count
            };
        // Right on the kicks (100 frames a second, a kick every 50), low down.
        let on_kicks = |t: usize| t > 50 && t % 50 < 3;
        let between = |t: usize| t > 50 && (20..30).contains(&(t % 50));
        let kick_drums = energy(&parts.drums, 40.0, 120.0, &on_kicks);
        let kick_tonal = energy(&parts.tonal, 40.0, 120.0, &on_kicks);
        assert!(
            kick_drums > kick_tonal,
            "kick: drums {kick_drums} tonal {kick_tonal}"
        );
        let chord_tonal = energy(&parts.tonal, 200.0, 350.0, &between);
        let chord_drums = energy(&parts.drums, 200.0, 350.0, &between);
        assert!(
            chord_tonal > chord_drums,
            "chord: tonal {chord_tonal} drums {chord_drums}"
        );
    }
}
