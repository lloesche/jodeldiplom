//! Echo cancellation for the microphone: the music the speakers play,
//! heard again by the microphone, taken out of it, so what is left is the
//! singer.
//!
//! The microphone is only listened to, never played (see
//! [`crate::input`]), so there is no loop: the echo is the music alone,
//! and the music is known exactly. The canceller is given, block by block,
//! what the microphone took in and what went to the speakers at the same
//! moment (the reference). One rule above the others: **the singer is
//! never taken out.** Some of the music's echo may be left in the voice;
//! the voice is never lost.
//!
//! It works in two parts:
//!
//! 1. **Finding the delay**: how the music's level moves in a few bands is
//!    matched against how the microphone's does, over up to [`MAX_DELAY`];
//!    the filter's window is moved to where the echo starts. Most echoes
//!    start well inside the window as it is (the output's and the input's
//!    latency, a few tens of ms); a wireless speaker's do not.
//! 2. **A linear echo canceller**: an adaptive filter learns how the
//!    reference reaches the microphone (the speaker, the room) and
//!    subtracts its prediction of the echo. It is a partitioned-block
//!    frequency-domain adaptive filter ([`TAIL`] long), normalised per
//!    frequency by the reference's power. Two of them: a shadow that learns
//!    while nobody sings, and the filter in use, which takes the shadow over
//!    only once it leaves clearly less. Singing never teaches either (a held
//!    note would draw them towards taking the note out). A room that
//!    changes while someone seems to sing (the volume turned up: the
//!    leftover jumps as a voice's would) is learnt by a trial filter apart,
//!    which takes over only once it has left far less for a second running.
//!    The filters learn along the reference prewhitened (each frequency's
//!    steady part, a held tone, taken out): learnt along a tone, the path is
//!    right for that note only. A step is never more than
//!    [`WHITE_GAIN_MOST`] times what the reference itself would make it,
//!    where whitening takes most of a frequency away (a held bass note):
//!    beyond that the filter grows without end. A filter that went beyond
//!    numbers anyway, or whose estimate is far louder than all the
//!    microphone heard (no echo can be), is started over at once, never
//!    left to spoil the canceller until the player starts again; its
//!    estimate takes nothing out.
//!
//! The voice it gives is the microphone less the echo the filter predicts,
//! nothing taken out beyond that: what the music does not account for (the
//! singer) passes exactly as it came. There is no residual suppressor: on
//! real songs through a simulated laptop (`aec_room`, an example of
//! `jodel-playback`) one took 2 to 3 dB more of the echo out where nobody
//! sang, up to as much of a singer under the echo, and left the voice no
//! further over the rest while they sang.
//!
//! Each block is [`BLOCK`] samples at the canceller's rate (about 44.1 or
//! 48 kHz); the voice of a block comes out with it.

use std::f32::consts::PI;

/// Samples in a block.
pub const BLOCK: usize = 128;
/// The echo the filter learns, in seconds from where its window starts:
/// the way through the air and the room's tail.
pub const TAIL: f32 = 0.2;
/// The longest delay before the echo starts that the window is moved to,
/// in seconds.
pub const MAX_DELAY: f32 = 1.0;

/// The transform's length: two blocks.
const N: usize = 2 * BLOCK;
/// Its bins (0 to Nyquist).
const BINS: usize = N / 2 + 1;
/// The filter's step size while it learns.
const STEP: f32 = 0.6;
/// Blocks after the last one sung over before the meter measures again
/// (about 100 ms).
const HOLD: u32 = 40;
/// A block leaving this many times the echo the filter in use usually
/// leaves is sung over.
const SINGING: f32 = 10.0;
/// While someone sings, the trial filter takes over only after leaving this
/// share of what the filter in use leaves (−6 dB) for this many blocks
/// running (about a second): a room that changed (the volume turned up)
/// does that soon; a voice, which the music does not predict, never does,
/// while a held note may draw the trial near it for moments.
const CLEARLY: f32 = 0.25;
const CLEARLY_FOR: u32 = 375;

/// How many times larger than the whitened reference's the reference's
/// own amplitude may make a step (a step is STEP times this at most, under
/// the 2 beyond which it grows without end). On five real recordings from a
/// laptop the filter that learns had grown beyond numbers within a song
/// without it; at 2 the canceller learnt too slowly (9 of its tests failed),
/// at 4 and 6 it grew again.
const WHITE_GAIN_MOST: f32 = 3.0;
/// An echo estimate over this many times the microphone's power (+20 dB)
/// in a block is no echo (the echo is part of what the microphone hears):
/// the filter that made it has gone astray. Within a block of a working
/// filter an estimate runs up to 9 dB over (a singer's sound against the
/// echo's); started over at +6 dB, a song through a laptop lost a third of
/// what was taken out. One gone astray ran 30 to 45 dB over.
const ASTRAY: f32 = 100.0;
/// Blocks between copies of one filter to the other (about 20 ms).
const SWAP_EVERY: u32 = 8;
/// Partitions whose impulse response is constrained per block.
const CONSTRAINED: usize = 8;
/// The smoothing of the powers the detector and the meter use, per
/// block (about 30 ms).
const SMOOTH: f32 = 0.1;
/// Blocks between looks at where the window should be (about 170 ms).
const PLACE_EVERY: u32 = 64;
/// Blocks the window is given to learn where it is before the delay
/// finder may move it (about 2 s of music).
const LEARN_FIRST: u32 = 750;
/// Partitions the window starts before the echo's first sound.
const MARGIN: usize = 6;
/// The delay finder's bands: bins from each edge to the next (about
/// 190 Hz to 6.7 kHz at 48 kHz).
const EDGES: [usize; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 25, 30, 36];
const BANDS: usize = EDGES.len() - 1;
/// Blocks after the delay that stands out that the room's tail may make
/// stand out too (about 100 ms).
const TAIL_AFTER: usize = 40;
/// Looks running the same delay must stand out in.
const LOOKS: u32 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C32 {
    re: f32,
    im: f32,
}

impl C32 {
    fn mul(self, o: C32) -> C32 {
        C32 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    fn conj_mul(self, o: C32) -> C32 {
        // conj(self) * o
        C32 {
            re: self.re * o.re + self.im * o.im,
            im: self.re * o.im - self.im * o.re,
        }
    }

    fn norm(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// A radix-2 transform of [`N`] points.
struct Fft {
    twiddles: Vec<C32>,
    order: Vec<usize>,
}

impl Fft {
    fn new() -> Fft {
        let bits = N.trailing_zeros();
        Fft {
            twiddles: (0..N / 2)
                .map(|k| {
                    let a = -2.0 * PI * k as f32 / N as f32;
                    C32 {
                        re: a.cos(),
                        im: a.sin(),
                    }
                })
                .collect(),
            order: (0..N)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    /// In place; `inverse` without the 1/N.
    fn run(&self, x: &mut [C32; N], inverse: bool) {
        for (i, &j) in self.order.iter().enumerate() {
            if j > i {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= N {
            let stride = N / len;
            for start in (0..N).step_by(len) {
                for k in 0..len / 2 {
                    let mut w = self.twiddles[k * stride];
                    if inverse {
                        w.im = -w.im;
                    }
                    let a = x[start + k];
                    let b = x[start + k + len / 2].mul(w);
                    x[start + k] = C32 {
                        re: a.re + b.re,
                        im: a.im + b.im,
                    };
                    x[start + k + len / 2] = C32 {
                        re: a.re - b.re,
                        im: a.im - b.im,
                    };
                }
            }
            len *= 2;
        }
    }

    /// The bins of `a` then `b` (each [`BLOCK`] samples, `a` older), as the
    /// transform of their concatenation.
    fn real(&self, a: &[f32], b: &[f32]) -> [C32; BINS] {
        let mut x = [C32::default(); N];
        for (i, v) in a.iter().chain(b).enumerate().take(N) {
            x[i].re = *v;
        }
        self.run(&mut x, false);
        let mut out = [C32::default(); BINS];
        out.copy_from_slice(&x[..BINS]);
        out
    }

    /// The real samples of the bins `bins` (and their mirror), scaled.
    fn inverse(&self, bins: &[C32; BINS]) -> [f32; N] {
        let mut x = [C32::default(); N];
        x[..BINS].copy_from_slice(bins);
        for k in 1..N / 2 {
            x[N - k] = C32 {
                re: bins[k].re,
                im: -bins[k].im,
            };
        }
        self.run(&mut x, true);
        let mut out = [0.0f32; N];
        for (o, v) in out.iter_mut().zip(x.iter()) {
            *o = v.re / N as f32;
        }
        out
    }
}

/// The reference's bins, block by block, as far back as the window may
/// reach: a ring.
struct History {
    spectra: Vec<[C32; BINS]>,
    /// The newest.
    head: usize,
}

impl History {
    fn new(len: usize) -> History {
        History {
            spectra: vec![[C32::default(); BINS]; len],
            head: 0,
        }
    }

    fn push(&mut self, x: [C32; BINS]) {
        self.head = (self.head + 1) % self.spectra.len();
        self.spectra[self.head] = x;
    }

    /// The bins `age` blocks back (0: the newest).
    fn at(&self, age: usize) -> &[C32; BINS] {
        let len = self.spectra.len();
        &self.spectra[(self.head + len - age % len) % len]
    }

    fn clear(&mut self) {
        for s in &mut self.spectra {
            *s = [C32::default(); BINS];
        }
    }
}

/// How fast the whitener follows each frequency's steady part (per block).
const WHITE_FOLLOW: f32 = 0.02;
/// How much of a steady tone it may take out (about 30 dB).
const WHITE_MOST: f32 = 0.97;

/// The reference whitened for learning: in each frequency, what of this
/// frame the frame two blocks before (which shares no sample with it)
/// predicts, taken out. A steady tone turns the same way every block, so
/// it goes; noise and the music's changes stay. The shadow steps along the
/// whitened reference, against the error it left as it is, and is
/// normalised by the whitened reference's power: a tone no longer
/// outweighs the rest of its frequency band. Along the tone itself, the
/// filter fits that one note, and the rest of the band, where a voice heard
/// back may be, takes many seconds to learn; whitened, all of it learns at
/// once. (Whitening the error too would match the reference only if the
/// filter held still between the frames.)
struct Whitener {
    /// Each bin's predictor: the smoothed product of a frame with the one
    /// two before, and the earlier one's power.
    cross: [C32; BINS],
    earlier: [f32; BINS],
}

impl Whitener {
    fn new() -> Whitener {
        Whitener {
            cross: [C32::default(); BINS],
            earlier: [0.0; BINS],
        }
    }

    /// Bin `k`'s predictor (at most [`WHITE_MOST`] of the earlier frame).
    fn predictor(&self, k: usize) -> C32 {
        let p = self.earlier[k];
        if p <= 1e-20 {
            return C32::default();
        }
        let c = C32 {
            re: self.cross[k].re / p,
            im: self.cross[k].im / p,
        };
        let size = c.norm().sqrt();
        if size > WHITE_MOST {
            let f = WHITE_MOST / size;
            C32 {
                re: c.re * f,
                im: c.im * f,
            }
        } else {
            c
        }
    }

    /// The reference's frame `x` whitened, `before` the frame two blocks
    /// earlier; the predictors learn from them first.
    fn reference(&mut self, x: &[C32; BINS], before: &[C32; BINS]) -> [C32; BINS] {
        let mut out = [C32::default(); BINS];
        for k in 0..BINS {
            let g = before[k].conj_mul(x[k]);
            self.cross[k].re += (g.re - self.cross[k].re) * WHITE_FOLLOW;
            self.cross[k].im += (g.im - self.cross[k].im) * WHITE_FOLLOW;
            self.earlier[k] += (before[k].norm() - self.earlier[k]) * WHITE_FOLLOW;
            let c = self.predictor(k).mul(before[k]);
            out[k] = C32 {
                re: x[k].re - c.re,
                im: x[k].im - c.im,
            };
        }
        out
    }
}
/// The echo `weights` predict from the reference's history, the window
/// starting `offset` blocks back: the time samples of the transform, the
/// newer half the block's.
fn predict(fft: &Fft, weights: &[[C32; BINS]], history: &History, offset: usize) -> [f32; N] {
    let mut y = [C32::default(); BINS];
    for (p, w) in weights.iter().enumerate() {
        let x = history.at(offset + p);
        for k in 0..BINS {
            let v = w[k].mul(x[k]);
            y[k].re += v.re;
            y[k].im += v.im;
        }
    }
    fft.inverse(&y)
}

/// Finds how late the echo comes: how the reference's level moves in
/// [`BANDS`] bands, block by block, against how the microphone's does,
/// at every delay up to [`MAX_DELAY`].
struct DelayFinder {
    /// The reference's band levels (each less its recent mean), a ring.
    reference: Vec<[f32; BANDS]>,
    head: usize,
    reference_mean: [f32; BANDS],
    mic_mean: [f32; BANDS],
    /// Per delay in blocks: the smoothed product of the two.
    products: Vec<f32>,
    /// The two's smoothed powers.
    reference_power: f32,
    mic_power: f32,
    /// Blocks seen with the reference playing.
    seen: u32,
    /// The delay that stood out at the last looks, and in how many
    /// running.
    candidate: Option<(usize, u32)>,
}

impl DelayFinder {
    fn new(lags: usize) -> DelayFinder {
        DelayFinder {
            reference: vec![[0.0; BANDS]; lags],
            head: 0,
            reference_mean: [0.0; BANDS],
            mic_mean: [0.0; BANDS],
            products: vec![0.0; lags],
            reference_power: 0.0,
            mic_power: 0.0,
            seen: 0,
            candidate: None,
        }
    }

    fn clear(&mut self) {
        self.reference.fill([0.0; BANDS]);
        self.reference_mean = [0.0; BANDS];
        self.mic_mean = [0.0; BANDS];
        self.products.fill(0.0);
        self.reference_power = 0.0;
        self.mic_power = 0.0;
        self.seen = 0;
        self.candidate = None;
    }

    /// The bands' log levels, less their recent means (about 50 ms): how
    /// the level moves.
    fn moves(bins: &[C32; BINS], mean: &mut [f32; BANDS]) -> [f32; BANDS] {
        let mut out = [0.0f32; BANDS];
        for b in 0..BANDS {
            let p: f32 = bins[EDGES[b]..EDGES[b + 1]].iter().map(|c| c.norm()).sum();
            let level = (p + 1e-10).ln();
            mean[b] += (level - mean[b]) * 0.05;
            out[b] = level - mean[b];
        }
        out
    }

    /// One block of the reference and the microphone (as bins).
    fn push(&mut self, reference: &[C32; BINS], mic: &[C32; BINS]) {
        let r = Self::moves(reference, &mut self.reference_mean);
        let m = Self::moves(mic, &mut self.mic_mean);
        let lags = self.reference.len();
        self.head = (self.head + 1) % lags;
        self.reference[self.head] = r;
        let a = 1.0 / 400.0;
        let dot = |x: &[f32; BANDS], y: &[f32; BANDS]| -> f32 {
            x.iter().zip(y).map(|(a, b)| a * b).sum()
        };
        self.reference_power += (dot(&r, &r) - self.reference_power) * a;
        self.mic_power += (dot(&m, &m) - self.mic_power) * a;
        for (lag, p) in self.products.iter_mut().enumerate() {
            let old = &self.reference[(self.head + lags - lag) % lags];
            *p += (dot(&m, old) - *p) * a;
        }
        self.seen = self.seen.saturating_add(1);
    }

    /// Looks at the delays: one that stands out (well over any other,
    /// the room's tail after it aside) the same in a few looks running is
    /// the echo's.
    fn look(&mut self) {
        let norm = (self.reference_power * self.mic_power).sqrt();
        let best = self
            .products
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(l, &p)| (l, p));
        let candidate = best.filter(|&(best, top)| {
            let other = self
                .products
                .iter()
                .enumerate()
                .filter(|(l, _)| *l + 4 < best || *l > best + TAIL_AFTER)
                .map(|(_, &p)| p)
                .fold(0.0f32, f32::max);
            norm > 0.0 && top / norm > 0.03 && top > 1.5 * other
        });
        match (candidate, self.candidate) {
            (Some((l, _)), Some((was, n))) if l.abs_diff(was) <= 2 => {
                self.candidate = Some((l, n + 1));
            }
            (Some((l, _)), _) => self.candidate = Some((l, 1)),
            (None, _) => self.candidate = None,
        }
    }

    /// The delay in blocks, once it has stood out a few looks running.
    fn lag(&self) -> Option<usize> {
        if self.seen < LEARN_FIRST {
            return None;
        }
        self.candidate.filter(|c| c.1 >= LOOKS).map(|c| c.0)
    }
}

/// How the canceller is doing, for a meter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EchoStatus {
    /// How much quieter the echo is after the canceller than before, in
    /// dB, while only the reference is heard (0 before it has learnt).
    pub removed_db: f32,
    /// How much of it the filter takes out, as it measures that to tell
    /// singing from the echo it leaves, in dB.
    pub filter_db: f32,
    /// It has learnt the echo's way (the filter is converged).
    pub learnt: bool,
    /// Seconds after the reference that the echo is loudest, once learnt.
    pub delay: Option<f32>,
}

/// The canceller ([`crate::aec`]).
pub struct EchoCanceller {
    fft: Fft,
    /// Its rate (samples a second), for the delay it reports.
    rate: u32,
    /// The filter in use: a partition per [`BLOCK`] of the window, newest
    /// first.
    weights: Vec<[C32; BINS]>,
    /// The filter that learns, and what it leaves (smoothed); blocks
    /// since one was copied to the other.
    shadow: Vec<[C32; BINS]>,
    shadow_level: f32,
    since_swap: u32,
    /// The reference's bins, each the transform of two blocks, as far back
    /// as the window may go; where the window starts, in blocks back.
    history: History,
    offset: usize,
    /// The reference whitened, and its bins like `history`'s: what the
    /// shadow learns from.
    whitener: Whitener,
    white_history: History,

    /// Finds where the window should start.
    finder: DelayFinder,
    /// Blocks with the reference playing since the window last moved;
    /// blocks since the last look at it; the partition where the filter
    /// in use has the most.
    placed: u32,
    since_place: u32,
    peak: usize,
    /// The reference's power per bin over the window.
    power: [f32; BINS],
    /// The last reference and microphone blocks (the transforms take two).
    last_reference: [f32; BLOCK],
    last_mic: [f32; BLOCK],
    /// The partition whose gradient is constrained next.
    turn: usize,

    /// Smoothed block powers: microphone, error, echo estimate, reference.
    mic_level: f32,
    error_level: f32,
    echo_level: f32,
    reference_level: f32,
    /// The trial filter (while someone sings), what it leaves (smoothed),
    /// whether it is on, and the blocks running it has left [`CLEARLY`]
    /// less than the filter in use.
    trial: Vec<[C32; BINS]>,
    trial_level: f32,
    trying: bool,
    better_for: u32,
    /// What is left, smoothed.
    out_level: f32,
    /// Blocks since the last one sung over.
    since_singing: u32,
    /// The echo removed, as the meter shows it and as the filter measures
    /// it, in dB.
    removed_db: f32,
    filter_db: f32,
    learnt: bool,
}

impl EchoCanceller {
    /// A canceller at `rate` (samples a second; it sets the window's and
    /// the delay's length in blocks).
    pub fn new(rate: u32) -> EchoCanceller {
        let blocks =
            |seconds: f32| ((seconds * rate as f32) / BLOCK as f32).ceil().max(1.0) as usize;
        let partitions = blocks(TAIL);
        let lags = blocks(MAX_DELAY);

        EchoCanceller {
            fft: Fft::new(),
            rate,
            weights: vec![[C32::default(); BINS]; partitions],
            shadow: vec![[C32::default(); BINS]; partitions],
            shadow_level: 0.0,
            since_swap: 0,
            history: History::new(lags + partitions),
            offset: 0,
            whitener: Whitener::new(),
            white_history: History::new(lags + partitions),

            finder: DelayFinder::new(lags),
            placed: 0,
            since_place: 0,
            peak: 0,
            power: [0.0; BINS],
            last_reference: [0.0; BLOCK],
            last_mic: [0.0; BLOCK],
            turn: 0,

            mic_level: 0.0,
            error_level: 0.0,
            echo_level: 0.0,
            reference_level: 0.0,
            trial: vec![[C32::default(); BINS]; partitions],
            trial_level: 0.0,
            trying: false,
            better_for: 0,
            out_level: 0.0,
            since_singing: u32::MAX,
            removed_db: 0.0,
            filter_db: 0.0,
            learnt: false,
        }
    }

    /// The rate it runs at (only the delay it reports changes).
    pub fn set_rate(&mut self, rate: u32) {
        self.rate = rate;
    }

    /// How it is doing.
    pub fn status(&self) -> EchoStatus {
        EchoStatus {
            removed_db: self.removed_db,
            filter_db: if self.learnt { self.filter_db } else { 0.0 },
            learnt: self.learnt,
            delay: self
                .learnt
                .then(|| ((self.offset + self.peak) * BLOCK) as f32 / self.rate.max(1) as f32),
        }
    }

    /// Forgets what it learnt, and where the echo was (a new output).
    pub fn reset(&mut self) {
        self.forget();
        self.history.clear();
        self.white_history.clear();
        self.finder.clear();
        self.offset = 0;
        self.placed = 0;
    }

    /// Forgets the filters (the echo's way has changed), and what they
    /// left: with nothing learnt they leave the microphone as it is, so
    /// their levels start over from its level (a level kept from before,
    /// a filter gone astray's above all, would keep the shadow from ever
    /// taking over).
    fn forget(&mut self) {
        for w in self
            .weights
            .iter_mut()
            .chain(self.shadow.iter_mut())
            .chain(self.trial.iter_mut())
        {
            *w = [C32::default(); BINS];
        }
        let mic = if self.mic_level.is_finite() {
            self.mic_level
        } else {
            0.0
        };
        self.mic_level = mic;
        self.error_level = mic;
        self.shadow_level = mic;
        self.echo_level = 0.0;
        self.trial_level = 0.0;
        self.trying = false;
        self.better_for = 0;
        self.since_swap = 0;
        self.learnt = false;
        self.removed_db = 0.0;
        self.filter_db = 0.0;
        self.peak = 0;
    }

    /// One block: `mic` what the microphone took in, `reference` what went
    /// to the speakers at the same moment; the voice without the echo goes
    /// to `out`.
    pub fn process(&mut self, mic: &[f32], reference: &[f32], out: &mut [f32]) {
        assert!(mic.len() == BLOCK && reference.len() == BLOCK && out.len() == BLOCK);
        // Nothing but numbers goes in (a device's glitch must not stick).
        let clean = |s: &[f32]| -> [f32; BLOCK] {
            std::array::from_fn(|i| if s[i].is_finite() { s[i] } else { 0.0 })
        };
        let (mic, reference) = (&clean(mic)[..], &clean(reference)[..]);
        // The reference's newest two blocks into the history; the
        // microphone's beside them for the delay finder.
        let x = self.fft.real(&self.last_reference, reference);
        let m = self.fft.real(&self.last_mic, mic);
        self.last_reference.copy_from_slice(reference);
        self.last_mic.copy_from_slice(mic);
        self.history.push(x);
        let xw = self.whitener.reference(&x, self.history.at(2));
        self.white_history.push(xw);
        // The echo estimates: each filter over its window.
        let y_time = predict(&self.fft, &self.weights, &self.history, self.offset);
        let echo = &y_time[BLOCK..];
        let mut error = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            error[i] = mic[i] - echo[i];
        }
        let mean = |s: &[f32]| s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32;
        let quiet = 1e-9;
        let pm = mean(mic);
        // An estimate beyond numbers, or far louder than all the microphone
        // heard (the echo is part of that), is no echo: the filter in use
        // has gone astray (learnt from a voice over music near silence, it
        // meets the music at full level). It takes nothing out, the voice is
        // never lost to it, and it is started over below.
        let astray = |estimate: &[f32]| {
            !estimate.iter().all(|v| v.is_finite()) || mean(estimate) > ASTRAY * pm + quiet
        };
        let in_use_astray = astray(echo);
        if in_use_astray {
            error.copy_from_slice(mic);
        }
        let shadow_time = predict(&self.fft, &self.shadow, &self.history, self.offset);
        let shadow_astray = astray(&shadow_time[BLOCK..]);
        let mut shadow_error = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            shadow_error[i] = mic[i] - shadow_time[BLOCK + i];
        }
        // Levels, for the detector and the meter (an estimate gone astray
        // counts for nothing).
        let py = if in_use_astray { 0.0 } else { mean(echo) };
        let (pe, px) = (mean(&error), mean(reference));
        let smooth = |a: &mut f32, v: f32| *a += (v - *a) * SMOOTH;
        smooth(&mut self.mic_level, pm);
        smooth(&mut self.error_level, pe);
        smooth(&mut self.echo_level, py);
        smooth(&mut self.reference_level, px);
        smooth(&mut self.shadow_level, mean(&shadow_error));
        // A filter gone astray (or beyond numbers) is started over, never
        // left to spoil every comparison with its level from then on: the
        // shadow from the filter in use, the trial dropped, the filter in use
        // (and everything learnt) forgotten. What the shadow left this block
        // goes with it (it is then the filter in use's), or its next step
        // would spoil it again at once.
        if in_use_astray || !(self.error_level.is_finite() && self.echo_level.is_finite()) {
            self.forget();
            shadow_error.copy_from_slice(mic);
        } else if shadow_astray || !self.shadow_level.is_finite() {
            self.shadow.clone_from(&self.weights);
            self.shadow_level = self.error_level;
            shadow_error = error;
        }
        if !self.trial_level.is_finite() {
            self.trying = false;
        }
        let reference_on = self.reference_level > quiet;
        if reference_on {
            self.finder.push(&x, &m);
            self.placed = self.placed.saturating_add(1);
        }
        // Double talk: once learnt, the filter in use leaves about the
        // share of the echo its own cut says (of the echo as it predicts
        // it, which singing does not change); a block leaving several
        // times that is something else in the microphone (someone
        // singing).
        let cut = 10f32.powf(self.filter_db / 10.0);
        let usual = self.echo_level.max(py) / cut;
        let singing = self.learnt && pe > SINGING * usual + 10.0 * quiet;
        // The shadow learns while the music plays and nobody sings (a held
        // note would draw it towards taking the note out: the music has the
        // note's frequency in it too). The filter in use takes it over once
        // it leaves clearly less, and gives it back what it has when it has
        // gone astray.
        self.since_swap = self.since_swap.saturating_add(1);
        if self.since_swap > SWAP_EVERY && reference_on && !singing {
            if self.shadow_level < 0.7 * self.error_level && self.shadow_level < self.mic_level {
                self.weights.clone_from(&self.shadow);
                self.constrain_weights();
                self.error_level = self.shadow_level;
                self.since_swap = 0;
            } else if self.shadow_level > 4.0 * self.error_level + quiet {
                self.shadow.clone_from(&self.weights);
                self.shadow_level = self.error_level;
                self.since_swap = 0;
            }
        }
        // While someone sings, a trial filter learns from where the filter
        // in use is, apart: a room that changed (the volume turned up)
        // looks like singing to the detector for good, and would never be
        // learnt. The trial takes over only once it has left far less than
        // the filter in use for a second running, which a room that changed
        // does soon and a voice never does (the music does not predict
        // it). It is dropped when the singing stops.
        if singing && reference_on {
            if !self.trying {
                self.trying = true;
                self.trial.clone_from(&self.weights);
                self.trial_level = self.error_level;
                self.better_for = 0;
            }
            let trial = predict(&self.fft, &self.trial, &self.history, self.offset);
            let mut trial_error = [0.0f32; BLOCK];
            for i in 0..BLOCK {
                trial_error[i] = mic[i] - trial[BLOCK + i];
            }
            smooth(&mut self.trial_level, mean(&trial_error));
            self.better_for = if self.trial_level < CLEARLY * self.error_level {
                self.better_for.saturating_add(1)
            } else {
                0
            };
            if self.better_for >= CLEARLY_FOR && self.trial_level < self.mic_level {
                self.weights.clone_from(&self.trial);
                self.constrain_weights();
                self.shadow.clone_from(&self.weights);
                self.error_level = self.trial_level;
                self.shadow_level = self.trial_level;
                // Its cut is measured again.
                self.filter_db = 0.0;
                self.trying = false;
            } else {
                self.adapt(&trial_error, true);
            }
        } else {
            self.trying = false;
        }
        // Worse than nothing: once learnt, the echo's way changed
        // completely; start over. Before, the filter in use never fitted it
        // (taken over from the shadow while it fitted a voice for moments):
        // it is dropped, and the shadow learns on (forgetting it as well
        // cost a song through a laptop a third of what was taken out).
        if self.error_level > 2.0 * self.mic_level + quiet && pm > quiet {
            if self.learnt {
                self.forget();
            } else {
                for w in &mut self.weights {
                    *w = [C32::default(); BINS];
                }
                self.error_level = self.mic_level;
                self.echo_level = 0.0;
            }
        }
        // The filter's own cut, while the reference plays and nobody sings
        // over it: once learnt, only blocks whose leftover is about the
        // usual (singing never is) count. Not until a while after the last
        // block sung: the smoothed levels still carry its tail, and a cut
        // that looks smaller makes singing look like the usual leftover.
        let echo_only = !self.learnt || pe < 4.0 * usual;
        let settled = !self.learnt || self.since_singing > HOLD;
        if reference_on && !singing && echo_only && settled && self.mic_level > quiet {
            let db = 10.0 * (self.mic_level / self.error_level.max(1e-12)).log10();
            self.filter_db += (db.clamp(0.0, 60.0) - self.filter_db) * SMOOTH;
            if self.filter_db > 6.0 {
                self.learnt = true;
            }
        }
        if reference_on && !singing {
            self.adapt(&shadow_error, false);
        }
        self.since_place = self.since_place.saturating_add(1);
        if self.since_place >= PLACE_EVERY {
            self.since_place = 0;
            self.place();
        }
        // The voice: the microphone less the echo predicted, and nothing
        // more taken out.
        out.copy_from_slice(&error);
        // What is left (compared with the microphone's smoothed level); not
        // until a while after the last block sung, whose tail the levels
        // still carry.
        smooth(&mut self.out_level, pe);
        self.since_singing = if singing {
            0
        } else {
            self.since_singing.saturating_add(1)
        };
        let settled = self.since_singing > HOLD;
        if reference_on && settled && echo_only && self.mic_level > quiet {
            let db = 10.0 * (self.mic_level / self.out_level.max(1e-12)).log10();
            self.removed_db += (db.clamp(0.0, 80.0) - self.removed_db) * SMOOTH;
        }
    }

    /// Where the window should start. Once learnt, the echo's loudest part
    /// is kept near its start (so the room's tail fits after it); before,
    /// the delay finder's delay, if the window does not reach it.
    fn place(&mut self) {
        let partitions = self.weights.len();
        let furthest = self.history.spectra.len() - partitions;
        let energy = |w: &[C32; BINS]| w.iter().map(|c| c.norm()).sum::<f32>();
        self.peak = self
            .weights
            .iter()
            .map(energy)
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(p, _)| p);
        let to = if self.learnt {
            if self.peak <= partitions / 3 {
                return;
            }
            self.offset + self.peak - MARGIN
        } else {
            self.finder.look();
            if self.placed < LEARN_FIRST {
                return;
            }
            let Some(lag) = self.finder.lag() else {
                return;
            };
            if lag >= self.offset && lag <= self.offset + partitions / 2 {
                return;
            }
            lag.saturating_sub(MARGIN)
        };
        self.move_to(to.min(furthest));
    }

    /// Moves the window to start `to` blocks back, the filters with it
    /// (what they learnt of the delays both windows hold is kept).
    fn move_to(&mut self, to: usize) {
        if to == self.offset {
            return;
        }

        let partitions = self.weights.len();
        for w in [&mut self.weights, &mut self.shadow] {
            if to > self.offset {
                let by = (to - self.offset).min(partitions);
                w.rotate_left(by);
                for p in &mut w[partitions - by..] {
                    *p = [C32::default(); BINS];
                }
            } else {
                let by = (self.offset - to).min(partitions);
                w.rotate_right(by);
                for p in &mut w[..by] {
                    *p = [C32::default(); BINS];
                }
            }
        }
        self.peak = self.peak.saturating_sub(to.saturating_sub(self.offset));
        self.offset = to;
        self.placed = 0;
    }

    /// Every partition of the filter in use kept to its first half: a
    /// linear filter, no circular wrap left. The shadow is constrained a few
    /// partitions a block; what it has not been yet fits the music it
    /// learnt from, not a voice that starts and stops.
    fn constrain_weights(&mut self) {
        for p in 0..self.weights.len() {
            let mut t = self.fft.inverse(&self.weights[p]);
            for v in &mut t[BLOCK..] {
                *v = 0.0;
            }
            self.weights[p] = self.fft.real(&t[..BLOCK], &t[BLOCK..]);
        }
    }

    /// One step of the shadow (or with `trial`, the trial filter) towards
    /// the error `error` it left, along the whitened reference
    /// ([`Whitener`]).
    fn adapt(&mut self, error: &[f32; BLOCK], trial: bool) {
        let mut filter = std::mem::take(if trial {
            &mut self.trial
        } else {
            &mut self.shadow
        });
        let e = self.fft.real(&[0.0; BLOCK], error);
        // Each bin's step is normalised by the whitened reference's power
        // over the window, but never by less than the reference's own over
        // [`WHITE_GAIN_MOST`] squared. The filter acts on the reference as it
        // is and learns along it whitened; where whitening takes most of a
        // bin away (a held bass note, most of a song's power, which a
        // laptop does not even play), a step by the whitened power alone is
        // that many times too large, and grew the filter without end.
        self.power = [0.0; BINS];
        let mut plain = [0.0f32; BINS];
        for p in 0..filter.len() {
            let x = self.white_history.at(self.offset + p);
            for (power, x) in self.power.iter_mut().zip(x) {
                *power += x.norm();
            }
            let x = self.history.at(self.offset + p);
            for (power, x) in plain.iter_mut().zip(x) {
                *power += x.norm();
            }
        }
        for (power, plain) in self.power.iter_mut().zip(plain) {
            *power = power.max(plain / (WHITE_GAIN_MOST * WHITE_GAIN_MOST));
        }
        let floor = self.power.iter().sum::<f32>() / BINS as f32 * 1e-5 + 1e-10;
        for (p, w) in filter.iter_mut().enumerate() {
            let xh = self.white_history.at(self.offset + p);
            for k in 0..BINS {
                let g = xh[k].conj_mul(e[k]);
                let step = STEP / (self.power[k] + floor);
                w[k].re += step * g.re;
                w[k].im += step * g.im;
            }
        }
        // Some partitions constrained per block, in turn: each one's
        // impulse response kept to its first half (a linear, not a
        // circular, convolution), so wrapped errors never build up.
        for _ in 0..CONSTRAINED.min(filter.len()) {
            let p = self.turn % filter.len();
            self.turn = self.turn.wrapping_add(1);
            let mut t = self.fft.inverse(&filter[p]);
            for v in &mut t[BLOCK..] {
                *v = 0.0;
            }
            filter[p] = self.fft.real(&t[..BLOCK], &t[BLOCK..]);
        }
        if trial {
            self.trial = filter;
        } else {
            self.shadow = filter;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// A repeatable noise, -0.5 to 0.5.
    struct Noise(u32);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }
    }

    /// Music-like: noise coloured by a one-pole low-pass, beating
    /// tones on it.
    fn music(n: usize, seed: u32) -> Vec<f32> {
        let mut noise = Noise(seed);
        let mut lp = 0.0f32;
        (0..n)
            .map(|i| {
                lp += (noise.next() - lp) * 0.2;
                let t = i as f32 / RATE as f32;
                0.4 * lp
                    + 0.08 * (2.0 * PI * 220.0 * t).sin()
                    + 0.05 * (2.0 * PI * 554.0 * t).sin() * (2.0 * PI * 0.5 * t).sin()
            })
            .collect()
    }

    /// A room: a delay, then a decaying tail, at a gain.
    fn room(delay_ms: f32, gain: f32) -> Vec<f32> {
        let delay = (delay_ms / 1000.0 * RATE as f32) as usize;
        let mut noise = Noise(99);
        let tail = (0.04 * RATE as f32) as usize;
        let mut h = vec![0.0f32; delay + tail];
        h[delay] = gain;
        for i in 1..tail {
            h[delay + i] = gain * 0.6 * noise.next() * (-(i as f32) / (0.008 * RATE as f32)).exp();
        }
        h
    }

    fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; x.len()];
        for (i, &hv) in h.iter().enumerate() {
            if hv == 0.0 {
                continue;
            }
            for n in i..x.len() {
                y[n] += hv * x[n - i];
            }
        }
        y
    }

    /// A voice: a gliding tone in bursts (syllables).
    fn voice(n: usize, from: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                if i < from {
                    return 0.0;
                }
                let t = (i - from) as f32 / RATE as f32;
                let on = ((t * 3.0).fract() < 0.7) as u8 as f32;
                let f = 300.0 + 150.0 * (2.0 * PI * 0.7 * t).sin();
                0.25 * on * (2.0 * PI * f * t).sin()
            })
            .collect()
    }

    fn power(s: &[f32]) -> f32 {
        s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32
    }

    fn db(p: f32) -> f32 {
        10.0 * p.max(1e-20).log10()
    }

    /// Runs the canceller over whole signals; the output aligned back by
    /// its block.
    fn run(mic: &[f32], reference: &[f32]) -> (Vec<f32>, EchoCanceller) {
        let mut aec = EchoCanceller::new(RATE);
        let mut out = vec![0.0f32; mic.len()];
        let mut block = [0.0f32; BLOCK];
        for b in 0..mic.len() / BLOCK {
            let r = b * BLOCK..(b + 1) * BLOCK;
            aec.process(&mic[r.clone()], &reference[r.clone()], &mut block);
            out[b * BLOCK..(b + 1) * BLOCK].copy_from_slice(&block);
        }
        (out, aec)
    }

    /// The music's echo through a room 45 ms away is taken out by more
    /// than 25 dB once learnt, and a voice singing over it afterwards
    /// comes through, the filter holding still meanwhile.
    #[test]
    fn the_echo_goes_and_a_voice_over_it_stays() {
        let n = RATE as usize * 12;
        let reference = music(n, 7);
        let echo = convolve(&reference, &room(45.0, 0.5));
        let sung = voice(n, RATE as usize * 7);
        let mut noise = Noise(3);
        let mic: Vec<f32> = (0..n)
            .map(|i| echo[i] + sung[i] + 1e-4 * noise.next())
            .collect();
        let (out, aec) = run(&mic, &reference);
        let s = RATE as usize;
        // Music alone, after learning for a few seconds.
        let before = power(&echo[4 * s..7 * s]);
        let after = power(&out[4 * s..7 * s]);
        let removed = db(before) - db(after);
        assert!(removed > 25.0, "echo removed {removed:.1} dB");
        assert!(aec.status().learnt);
        // The meter says about as much, singing or not.
        assert!(aec.status().removed_db > 25.0, "{:?}", aec.status());
        // Singing over it: the voice is what comes out, the echo far under
        // it.
        let w = 8 * s..12 * s;
        let residual: Vec<f32> = w.clone().map(|i| out[i] - sung[i]).collect();
        let kept = db(power(&out[w.clone()])) - db(power(&sung[w.clone()]));
        let clean = db(power(&sung[w.clone()])) - db(power(&residual));
        eprintln!(
            "echo removed {removed:.1} dB (the meter says {:.1} dB); voice kept at {kept:.1} dB, {clean:.1} dB over the rest",
            aec.status().removed_db
        );
        assert!(kept > -3.0, "the voice kept at {kept:.1} dB");
        assert!(clean > 10.0, "the voice {clean:.1} dB over what is not it");
    }

    /// Nothing played: the microphone passes as it is.
    #[test]
    fn without_a_reference_the_voice_passes() {
        let n = RATE as usize * 3;
        let sung = voice(n, 0);
        let (out, _) = run(&sung, &vec![0.0; n]);
        let w = RATE as usize..2 * RATE as usize;
        let residual: Vec<f32> = w.clone().map(|i| out[i] - sung[i]).collect();
        assert!(db(power(&sung[w])) - db(power(&residual)) > 30.0);
    }

    /// The echo's way changing (the speaker moved): it learns the new one.
    #[test]
    fn a_changed_room_is_learnt_again() {
        let n = RATE as usize * 14;
        let reference = music(n, 11);
        let first = convolve(&reference, &room(30.0, 0.6));
        let second = convolve(&reference, &room(80.0, 0.4));
        let half = n / 2;
        let mic: Vec<f32> = (0..n)
            .map(|i| if i < half { first[i] } else { second[i] })
            .collect();
        let (out, _) = run(&mic, &reference);
        let s = RATE as usize;
        let w = n - 3 * s..n;
        let removed = db(power(&second[w.clone()])) - db(power(&out[w]));
        assert!(removed > 25.0, "removed {removed:.1} dB after the change");
    }

    /// The volume turned up 12 dB once learnt: the leftover jumps by as much
    /// as a singer would, yet the new loudness is learnt within seconds
    /// (held still while that looked like singing, it never was).
    #[test]
    fn the_volume_turned_up_is_learnt_again() {
        let n = RATE as usize * 16;
        let s = RATE as usize;
        let reference = music(n, 13);
        let quiet = convolve(&reference, &room(40.0, 0.25));
        let mic: Vec<f32> = (0..n)
            .map(|i| if i < 8 * s { quiet[i] } else { 4.0 * quiet[i] })
            .collect();
        let (out, aec) = run(&mic, &reference);
        let w = 12 * s..16 * s;
        let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
        eprintln!(
            "the volume turned up: {removed:.1} dB removed after, {:?}",
            aec.status()
        );
        assert!(
            removed > 25.0,
            "removed {removed:.1} dB after the volume went up"
        );
        assert!(aec.status().learnt);
    }

    /// A laptop's speakers in a living room: a limiter and soft clipping on
    /// the way out, no bass, a long reverberant tail (`rt60` seconds), the
    /// echo arriving `delay_ms` late.
    fn laptop(reference: &[f32], delay_ms: f32, rt60: f32) -> Vec<f32> {
        // Speaker: high-pass, limiter (fast attack, slow release), tanh.
        let mut hp_x = 0.0f32;
        let mut hp_y = 0.0f32;
        let mut env = 0.0f32;
        let a = (-2.0 * PI * 150.0 / RATE as f32).exp();
        let spoken: Vec<f32> = reference
            .iter()
            .map(|&x| {
                hp_y = a * (hp_y + x - hp_x);
                hp_x = x;
                let level = hp_y.abs();
                let k = if level > env { 0.3 } else { 0.0005 };
                env += (level - env) * k;
                let gain = (0.2 / env.max(1e-6)).min(1.0);
                (1.8 * hp_y * gain).tanh() / 1.8
            })
            .collect();
        // Room: the direct sound, then a diffuse tail.
        let delay = (delay_ms / 1000.0 * RATE as f32) as usize;
        let tail = (rt60 * RATE as f32) as usize;
        let mut noise = Noise(5);
        let mut h = vec![0.0f32; delay + tail];
        h[delay] = 0.6;
        let decay = 6.91 / (rt60 * RATE as f32);
        for i in 1..tail {
            h[delay + i] = 0.25 * noise.next() * (-(i as f32) * decay).exp();
        }
        convolve(&spoken, &h)
    }

    /// The meter's reading: a peak falling over 0.3 s, as the insert's.
    fn meter(s: &[f32]) -> Vec<f32> {
        let decay = (-1.0 / (RATE as f32 * 0.3)).exp();
        let mut p = 0.0f32;
        s.iter()
            .map(|v| {
                p = v.abs().max(p * decay);
                p
            })
            .collect()
    }

    /// What the canceller does in a laptop's room: how far the meter falls
    /// with the music only (8 to 12 s in, once learnt), how much of a
    /// voice singing over it after is kept (dB) and how far over the rest
    /// of what comes out (dB), and the delay it found.
    struct Laptop {
        meter_db: f32,
        kept_db: f32,
        clean_db: f32,
        delay: Option<f32>,
    }

    fn laptop_room(delay_ms: f32, rt60: f32, singer: f32) -> Laptop {
        let n = RATE as usize * 16;
        let s = RATE as usize;
        let reference = music(n, 21);
        let echo = laptop(&reference, delay_ms, rt60);
        let sung: Vec<f32> = voice(n, 12 * s).iter().map(|v| singer * v).collect();
        let mut noise = Noise(8);
        let mic: Vec<f32> = (0..n)
            .map(|i| echo[i] + sung[i] + 3e-4 * noise.next())
            .collect();
        let (out, aec) = run(&mic, &reference);
        let w = 8 * s..12 * s;
        let peak = |m: Vec<f32>| m[w.clone()].iter().cloned().fold(0.0f32, f32::max);
        let meter_db = 20.0 * (peak(meter(&mic)) / peak(meter(&out)).max(1e-9)).log10();
        let v = 13 * s..16 * s;
        let (kept_db, clean_db) = kept(&out[v.clone()], &sung[v]);
        let found = Laptop {
            meter_db,
            kept_db,
            clean_db,
            delay: aec.status().delay,
        };
        eprintln!(
            "{delay_ms} ms late, rt60 {rt60} s, the singer at {singer}: the meter falls {meter_db:.1} dB; the voice kept at {kept_db:.1} dB, {clean_db:.1} dB over the rest; the delay found {:?}",
            found.delay
        );
        found
    }

    /// How much of `voice` is in `out` (dB), and how far over the rest
    /// of `out` it is (dB).
    fn kept(out: &[f32], voice: &[f32]) -> (f32, f32) {
        let a = out.iter().zip(voice).map(|(o, v)| o * v).sum::<f32>()
            / voice.iter().map(|v| v * v).sum::<f32>().max(1e-20);
        let rest: Vec<f32> = out.iter().zip(voice).map(|(o, v)| o - a * v).collect();
        let part: Vec<f32> = voice.iter().map(|v| a * v).collect();
        (
            20.0 * a.abs().max(1e-9).log10(),
            db(power(&part)) - db(power(&rest)),
        )
    }

    /// The music alone moves the meter by less than a 20 dB step; a voice
    /// as loud as the echo comes through whole (within 0.5 dB), 10 dB over
    /// the rest; the delay is found to within 15 ms.
    fn check(delay_ms: f32, rt60: f32) {
        let found = laptop_room(delay_ms, rt60, 1.0);
        assert!(
            found.meter_db > 20.0,
            "the meter falls {:.1} dB",
            found.meter_db
        );
        assert!(
            found.kept_db.abs() < 0.5,
            "the voice kept at {:.1} dB",
            found.kept_db
        );
        assert!(
            found.clean_db > 10.0,
            "the voice {:.1} dB over the rest",
            found.clean_db
        );
        let delay = found.delay.expect("a delay found") * 1000.0;
        assert!(
            (delay - delay_ms).abs() < 15.0,
            "the delay found {delay:.0} ms"
        );
    }

    #[test]
    fn a_laptop_in_a_small_room() {
        check(40.0, 0.15);
    }

    #[test]
    fn a_laptop_in_a_living_room() {
        check(40.0, 0.4);
    }

    /// The window moves after the echo's loudest part, for more of the
    /// room's tail.
    #[test]
    fn speakers_further_away() {
        check(120.0, 0.4);
    }

    /// A wireless speaker: the echo starts past the window as it is, and
    /// the delay finder moves it there.
    #[test]
    fn a_wireless_speaker() {
        check(300.0, 0.3);
    }

    /// A singer as a singer sounds, from `from`: syllables of 0.15 to 0.5 s
    /// with 50 to 150 ms between them, each a voiced tone (its fundamental
    /// between 150 and 350 Hz with a vibrato, harmonics to 4 kHz falling
    /// 6 dB an octave) rising in 20 ms; every fourth soft (−15 dB); every
    /// third starting with a consonant (noise above 3 kHz, 50 ms). Gives
    /// the samples, and each syllable's (start, end, soft).
    fn singer(n: usize, from: usize) -> (Vec<f32>, Vec<(usize, usize, bool)>) {
        let mut out = vec![0.0f32; n];
        let mut syllables = Vec::new();
        let mut noise = Noise(77);
        let mut at = from;
        let mut k = 0usize;
        let rate = RATE as f32;
        while at < n {
            let long = (0.15 + 0.35 * ((k * 7 % 11) as f32 / 10.0)) * rate;
            let end = (at + long as usize).min(n);
            let soft = k % 4 == 3;
            let level = if soft { 0.18 * 0.178 } else { 0.18 };
            let f0 = 150.0 + 200.0 * ((k * 5 % 9) as f32 / 8.0);
            let mut phase = 0.0f32;
            let mut last = 0.0f32;
            for (j, i) in (at..end).enumerate() {
                let t = j as f32 / rate;
                let f = f0 * (1.0 + 0.02 * (2.0 * PI * 5.5 * t).sin());
                phase += 2.0 * PI * f / rate;
                let mut v = 0.0;
                let mut h = 1;
                while h as f32 * f < 4000.0 {
                    v += (h as f32 * phase).sin() / h as f32;
                    h += 1;
                }
                let attack = (t / 0.02).min(1.0);
                let release = (((end - i) as f32 / rate) / 0.03).min(1.0);
                let mut s = level * 0.5 * v * attack * release;
                if k % 3 == 0 && t < 0.05 {
                    let x = noise.next();
                    s += level * 1.2 * (x - last);
                    last = x;
                }
                out[i] = s;
            }
            syllables.push((at, end, soft));
            at = end + ((0.05 + 0.1 * ((k * 3 % 7) as f32 / 6.0)) * rate) as usize;
            k += 1;
        }
        (out, syllables)
    }

    /// How a singer comes through a laptop's room (40 ms, 0.4 s of tail),
    /// once learnt: each syllable's energy out against its own (dB), the
    /// first 60 ms of each (the onsets), the soft ones; and how far the
    /// meter falls with the music alone.
    struct Sung {
        meter_db: f32,
        median: f32,
        worst: f32,
        onsets: f32,
        soft: f32,
    }

    fn sung_through(singer_level: f32) -> Sung {
        let n = RATE as usize * 20;
        let s = RATE as usize;
        let reference = music(n, 21);
        let echo = laptop(&reference, 40.0, 0.4);
        let (voice, syllables) = singer(n, 10 * s);
        let voice: Vec<f32> = voice.iter().map(|v| singer_level * v).collect();
        let mut noise = Noise(8);
        let mic: Vec<f32> = (0..n)
            .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
            .collect();
        let (out, _) = run(&mic, &reference);
        let w = 6 * s..10 * s;
        let peak = |m: Vec<f32>| m[w.clone()].iter().cloned().fold(0.0f32, f32::max);
        let meter_db = 20.0 * (peak(meter(&mic)) / peak(meter(&out)).max(1e-9)).log10();
        let kept_in = |r: std::ops::Range<usize>| db(power(&out[r.clone()])) - db(power(&voice[r]));
        let mut all: Vec<f32> = syllables.iter().map(|&(a, b, _)| kept_in(a..b)).collect();

        let mut onsets: Vec<f32> = syllables
            .iter()
            .map(|&(a, b, _)| kept_in(a..(a + s * 6 / 100).min(b)))
            .collect();
        let mut soft: Vec<f32> = syllables
            .iter()
            .filter(|x| x.2)
            .map(|&(a, b, _)| kept_in(a..b))
            .collect();
        let median = |v: &mut Vec<f32>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let sung = Sung {
            meter_db,
            median: median(&mut all),
            worst: all[all.len() / 10],
            onsets: median(&mut onsets),
            soft: median(&mut soft),
        };
        let sung_power = syllables
            .iter()
            .map(|&(a, b, _)| power(&voice[a..b]))
            .sum::<f32>()
            / syllables.len() as f32;
        let over = db(power(&echo[w.clone()])) - db(sung_power);
        eprintln!(
            "a singer at {singer_level} (the echo {over:.1} dB over them): the meter falls {:.1} dB; syllables kept {:.1} dB (median), {:.1} dB (worst tenth), onsets {:.1} dB, soft ones {:.1} dB",
            sung.meter_db, sung.median, sung.worst, sung.onsets, sung.soft
        );
        sung
    }

    /// A singer through a laptop's room, from twice as loud as the music's
    /// echo at the microphone to 24 dB under it: no syllable loses anything
    /// (the soft ones and the onsets neither: the canceller never takes the
    /// singer out), and the music alone moves the meter by less than a
    /// 20 dB step. Under the echo, what the filter leaves of it adds to the
    /// syllables (a few dB to the soft ones of a singer 24 dB under it).
    #[test]
    fn a_singer_through_a_laptop() {
        for level in [2.0, 1.0, 0.3, 0.1] {
            let sung = sung_through(level);
            assert!(
                sung.meter_db > 20.0,
                "the meter falls {:.1} dB",
                sung.meter_db
            );
            assert!(sung.median > -0.5, "syllables kept {:.1} dB", sung.median);
            assert!(
                sung.worst > -1.0,
                "the worst tenth kept {:.1} dB",
                sung.worst
            );
            assert!(sung.onsets > -1.0, "onsets kept {:.1} dB", sung.onsets);
            assert!(sung.soft > -1.0, "soft syllables kept {:.1} dB", sung.soft);
        }
    }

    /// A singer quieter than the music's echo: heard whole.
    #[test]
    fn a_quiet_singer() {
        let found = laptop_room(40.0, 0.4, 0.5);
        assert!(
            found.meter_db > 20.0,
            "the meter falls {:.1} dB",
            found.meter_db
        );
        assert!(
            found.kept_db.abs() < 0.5,
            "the voice kept at {:.1} dB",
            found.kept_db
        );
    }

    /// A band of held chords over a held bass (as a song's band plays with
    /// its singer turned down), a singer in its key over its echo through
    /// a laptop, the microphone with an offset and a rumble of its own:
    /// the filters stay well within the microphone's level, the echo is
    /// still taken out once the singer stops, and the singer is kept. (On a
    /// real recording the filter that learns grew beyond numbers without
    /// [`WHITE_GAIN_MOST`]; this signal does not drive it that far, so it
    /// holds what must always hold.)
    #[test]
    fn a_steady_band_and_a_singer_never_blow_the_filters_up() {
        let n = RATE as usize * 20;
        let s = RATE as usize;
        let mut noise = Noise(5);
        let reference: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                // A chord changing every two seconds, held, under it its
                // root two octaves down, loud (most of a song's power is
                // its bass), and a little noise. The bass, held, whitened
                // is all but gone, and the laptop does not play it: there
                // its steps were the ones too large.
                let root = [220.0, 196.0, 247.0, 165.0][(t / 2.0) as usize % 4];
                let chord: f32 = [1.0, 1.26, 1.5, 2.0]
                    .iter()
                    .map(|r| (2.0 * PI * root * r * t).sin())
                    .sum();
                let bass = (2.0 * PI * root / 4.0 * t).sin();
                0.08 * chord + 0.3 * bass + 0.002 * noise.next()
            })
            .collect();
        let echo = laptop(&reference, 80.0, 0.4);
        // The singer on the chord's notes (in its key, as a singer is):
        // their harmonics land on the band's held tones.
        let mut syllables = Vec::new();
        let voice: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                if !(4.0..16.0).contains(&t) {
                    return 0.0;
                }
                let root = [220.0, 196.0, 247.0, 165.0][(t / 2.0) as usize % 4];
                let note = root * [1.0, 1.26, 1.5][((t * 2.0) as usize) % 3];
                let f = note * (1.0 + 0.005 * (2.0 * PI * 5.5 * t).sin());
                let on = ((t * 2.0).fract() < 0.85) as u8 as f32;
                let v: f32 = (1..8)
                    .map(|h| (2.0 * PI * f * h as f32 * t).sin() / h as f32)
                    .sum();
                0.12 * on * v
            })
            .collect();
        for k in 8..32 {
            let a = k * s / 2;
            syllables.push((a + s / 20, a + s * 8 / 20, false));
        }
        // A laptop's microphone: an offset of its own and a slow rumble
        // under the room (none of it the music's).
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                echo[i] + voice[i] + 0.01 + 0.004 * (2.0 * PI * 0.7 * t).sin()
            })
            .collect();
        let mut aec = EchoCanceller::new(RATE);
        let mut out = vec![0.0f32; n];
        let mut block = [0.0f32; BLOCK];
        let mut worst = f32::MIN;
        for b in 0..n / BLOCK {
            let r = b * BLOCK..(b + 1) * BLOCK;
            aec.process(&mic[r.clone()], &reference[r.clone()], &mut block);
            out[r].copy_from_slice(&block);
            worst = worst.max(aec.shadow_level / aec.mic_level.max(1e-12));
        }
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            worst < 10.0,
            "the shadow left {:.1} dB over the microphone",
            db(worst)
        );
        let w = 17 * s..20 * s;
        let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
        assert!(removed > 3.0, "the echo removed {removed:.1} dB after");
        let kept: Vec<f32> = syllables
            .iter()
            .filter(|&&(_, b, _)| b < 16 * s)
            .map(|&(a, b, _)| db(power(&out[a..b])) - db(power(&voice[a..b])))
            .collect();
        let median = {
            let mut k = kept.clone();
            k.sort_by(|a, b| a.total_cmp(b));
            k[k.len() / 2]
        };
        eprintln!("a steady band: shadow at most {:.1} dB over the microphone, removed {removed:.1} dB, syllables kept {median:.1} dB", db(worst));
        assert!(median > -0.5, "syllables kept {median:.1} dB");
    }

    /// A filter in use far too loud: one learnt from a singer over music
    /// near silence (a song's quiet start, whose band stem keeps a trace of
    /// the singer) met the music at full level with an estimate 45 dB over
    /// all the microphone heard, on a real song through a made-up laptop.
    /// This signal does not lead it there, so the filters are spoilt by
    /// hand. The estimate takes nothing out, no block comes out louder than
    /// the microphone, and the echo is learnt again.
    #[test]
    fn a_filter_far_too_loud_takes_nothing_out_and_starts_over() {
        let n = RATE as usize * 12;
        let s = RATE as usize;
        let reference = music(n, 13);
        let echo = convolve(&reference, &room(40.0, 0.4));
        let mut noise = Noise(17);
        let mic: Vec<f32> = (0..n).map(|i| echo[i] + 1e-4 * noise.next()).collect();
        let mut aec = EchoCanceller::new(RATE);
        let mut out = vec![0.0f32; n];
        let mut block = [0.0f32; BLOCK];
        let mut worst = f32::MIN;
        for b in 0..n / BLOCK {
            if b * BLOCK == 4 * s {
                assert!(aec.status().learnt);
                for w in aec.weights.iter_mut().chain(aec.shadow.iter_mut()) {
                    for c in w.iter_mut() {
                        c.re *= 200.0;
                        c.im *= 200.0;
                    }
                }
            }
            let r = b * BLOCK..(b + 1) * BLOCK;
            aec.process(&mic[r.clone()], &reference[r.clone()], &mut block);
            out[r.clone()].copy_from_slice(&block);
            if b * BLOCK >= 4 * s {
                worst = worst.max(db(power(&block)) - db(power(&mic[r])));
            }
        }
        assert!(
            worst < 0.5,
            "a block came out {worst:.1} dB over the microphone"
        );
        assert!(aec.status().learnt, "{:?}", aec.status());
        let w = 10 * s..12 * s;
        let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
        assert!(removed > 20.0, "the echo removed {removed:.1} dB after");
    }

    /// Filters gone beyond numbers (whatever did it) are started over, and
    /// the canceller learns again: nothing stays spoilt until the player
    /// starts again.
    #[test]
    fn filters_beyond_numbers_start_over() {
        let n = RATE as usize * 12;
        let s = RATE as usize;
        let reference = music(n, 11);
        let echo = convolve(&reference, &room(30.0, 0.5));
        let mut aec = EchoCanceller::new(RATE);
        let mut out = vec![0.0f32; n];
        let mut block = [0.0f32; BLOCK];
        for b in 0..n / BLOCK {
            if b * BLOCK == 4 * s {
                assert!(aec.status().learnt);
                // The shadow, then the filter in use too, spoilt.
                aec.shadow[0][3].re = f32::NAN;
                aec.shadow_level = f32::NAN;
            }
            if b * BLOCK == 6 * s {
                aec.weights[0][3].re = f32::INFINITY;
            }
            let r = b * BLOCK..(b + 1) * BLOCK;
            aec.process(&echo[r.clone()], &reference[r.clone()], &mut block);
            out[r].copy_from_slice(&block);
        }
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(aec.shadow_level.is_finite() && aec.error_level.is_finite());
        assert!(aec.status().learnt, "{:?}", aec.status());
        let w = 10 * s..12 * s;
        let removed = db(power(&echo[w.clone()])) - db(power(&out[w]));
        assert!(removed > 20.0, "the echo removed {removed:.1} dB after");
    }

    /// A note held for four seconds over the music, once learnt: the
    /// filter learns nothing from it (it would learn to take the note out),
    /// stays learnt, and the note comes through whole.
    #[test]
    fn a_held_note_never_spoils_the_filter() {
        let n = RATE as usize * 10;
        let s = RATE as usize;
        let reference = music(n, 7);
        let echo = convolve(&reference, &room(30.0, 0.5));
        let note: Vec<f32> = (0..n)
            .map(|i| {
                if i < 6 * s {
                    0.0
                } else {
                    0.2 * (2.0 * PI * 330.0 * i as f32 / RATE as f32).sin()
                }
            })
            .collect();
        let mic: Vec<f32> = (0..n).map(|i| echo[i] + note[i]).collect();
        let (out, aec) = run(&mic, &reference);
        let w = 7 * s..10 * s;
        let (kept_db, clean_db) = kept(&out[w.clone()], &note[w]);
        eprintln!(
            "a held note: kept {kept_db:.1} dB, {clean_db:.1} dB over the rest, {:?}",
            aec.status()
        );
        assert!(aec.status().learnt, "the filter was spoilt");
        assert!(kept_db.abs() < 0.5, "the note kept at {kept_db:.1} dB");
        assert!(clean_db > 20.0, "the note {clean_db:.1} dB over the rest");
    }

    #[test]
    fn the_transform_inverts() {
        let fft = Fft::new();
        let a: Vec<f32> = (0..BLOCK).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..BLOCK).map(|i| (i as f32 * 0.11).cos()).collect();
        let x = fft.real(&a, &b);
        let back = fft.inverse(&x);
        for i in 0..BLOCK {
            assert!((back[i] - a[i]).abs() < 1e-4 && (back[BLOCK + i] - b[i]).abs() < 1e-4);
        }
    }
}
