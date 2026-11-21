//! Sample-rate conversion, for the routes where no rate the device takes
//! carries the stream bit-perfect.
//!
//! [`Resampler`] is a synchronous rational polyphase FIR: the rate changes
//! by `up / down`, the exact ratio of the two rates reduced by their
//! greatest common divisor, for any pair of rates, up or down. The filter
//! ([`FilterDesign`]) is a Kaiser-windowed sinc designed in `f64` when the
//! resampler opens:
//!
//! - passband flat to `min(20 kHz, 0.45 · min(from, to))`, ripple far below
//!   0.001 dB;
//! - stopband from half the lower of the two rates, attenuated by at least
//!   [`STOPBAND_DB`] (the design aims at [`DESIGN_DB`]): images when going
//!   up, aliases when going down;
//! - linear phase (symmetric taps), with its group delay compensated: the
//!   first output frame is the instant of the first input frame, and a
//!   stream of `n` input frames gives exactly [`output_frames`]`(n)`
//!   output frames, `round(n · up / down)`.
//!
//! All arithmetic is in `f64`. The integer path
//! ([`Resampler::process`]) takes and gives 24-bit samples right-aligned in
//! `i32` and requantizes once, with TPDF dither of ±1 LSB from a seeded
//! generator (the same input gives the same output), saturating and
//! counting the samples that clip ([`Resampler::clipped`]). The `f64` path
//! ([`Resampler::process_f64`]) gives the unquantized output, for analysis.
//!
//! Each stream starts from silence and ends by flushing the filter with
//! silence ([`Resampler::finish`]), so a stream's output depends only on
//! its own input: a track resampled from a seek point is the same as a
//! fresh resample of the track from there, and gapless tracks add their
//! exact lengths. A stream whose length is known better than its input
//! tells it (a stem mix, whose input at the separator's rate ends only to
//! a frame of that rate) ends at another output frame instead
//! ([`Resampler::finish_at`]): the same frames up to where both have them,
//! then silence filtered on, or none past it.

use std::sync::{Arc, Mutex};

pub use jodel_output::Requantizer;

/// The highest passband edge, in Hz.
pub const PASSBAND_LIMIT_HZ: f64 = 20_000.0;
/// The passband edge as a fraction of the lower rate (when below
/// [`PASSBAND_LIMIT_HZ`]).
pub const PASSBAND_FRACTION: f64 = 0.45;
/// The guaranteed stopband attenuation, in dB.
pub const STOPBAND_DB: f64 = 160.0;
/// The attenuation the filter is designed for, in dB: a margin over
/// [`STOPBAND_DB`] for the length estimate.
pub const DESIGN_DB: f64 = 166.0;

/// Coefficients per phase are padded to a multiple of this, for the
/// multiply-accumulate loop.
const LANES: usize = 8;
/// Designs kept for reuse (a seek reopens the resampler).
const CACHED_DESIGNS: usize = 4;
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The ratio `to / from` in lowest terms, as `(up, down)`.
///
/// # Panics
///
/// If either rate is 0.
pub fn reduce(from: u32, to: u32) -> (u32, u32) {
    assert!(from > 0 && to > 0, "rates must be positive");
    let g = gcd(u64::from(from), u64::from(to)) as u32;
    (to / g, from / g)
}

/// The output frames for `input` frames from `from` Hz to `to` Hz:
/// `round(input · to / from)`, halves rounded up.
pub fn output_frames(from: u32, to: u32, input: u64) -> u64 {
    let (up, down) = reduce(from, to);
    rounded(input, up, down)
}

fn rounded(input: u64, up: u32, down: u32) -> u64 {
    let (up, down) = (u128::from(up), u128::from(down));
    ((2 * u128::from(input) * up + down) / (2 * down)) as u64
}

/// The zeroth-order modified Bessel function of the first kind.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-18 {
        term *= quarter / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

pub(crate) fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// The polyphase filter for one pair of rates.
///
/// The prototype is a lowpass at `up · from` Hz with `2 · half · up + 1`
/// symmetric taps (gain `up`); phase `p` holds the taps `p + up · q` for
/// `q` in `-half..=half`, so each output frame is one dot product of
/// [`FilterDesign::taps_per_phase`] coefficients with as many input frames.
#[derive(Debug)]
pub struct FilterDesign {
    from: u32,
    to: u32,
    up: usize,
    down: usize,
    half: usize,
    taps: usize,
    lead: usize,
    passband: f64,
    stopband: f64,
    beta: f64,
    /// Phase-major; phase `p`, entry `m` multiplies input frame
    /// `i - half - lead + m` for the output frame at input position `i`.
    coefficients: Vec<f64>,
}

impl FilterDesign {
    /// Designs the filter from `from` Hz to `to` Hz.
    ///
    /// # Panics
    ///
    /// If either rate is 0.
    pub fn new(from: u32, to: u32) -> FilterDesign {
        let (up, down) = reduce(from, to);
        let (up, down) = (up as usize, down as usize);
        let low = f64::from(from.min(to));
        let passband = PASSBAND_LIMIT_HZ.min(PASSBAND_FRACTION * low);
        let stopband = 0.5 * low;
        let proto_rate = up as f64 * f64::from(from);
        // Kaiser's estimates for the window and the length.
        let beta = 0.1102 * (DESIGN_DB - 8.7);
        let transition = 2.0 * std::f64::consts::PI * (stopband - passband) / proto_rate;
        let length = (DESIGN_DB - 7.95) / (2.285 * transition) + 1.0;
        let half = ((length / 2.0) / up as f64).ceil().max(1.0) as usize;
        let span = 2 * half + 1;
        let taps = span.div_ceil(LANES) * LANES;
        let lead = taps - span;

        let reach = (half * up) as f64;
        let cutoff = (passband + stopband) / proto_rate; // 2 · fc / fs
        let i0_beta = bessel_i0(beta);
        let tap = |n: i64| -> f64 {
            let n = n as f64;
            if n.abs() > reach {
                return 0.0;
            }
            let r = n / reach;
            let window = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
            up as f64 * cutoff * sinc(cutoff * n) * window
        };
        let mut coefficients = vec![0.0; up * taps];
        for p in 0..up {
            let row = &mut coefficients[p * taps..(p + 1) * taps];
            for (m, c) in row.iter_mut().enumerate().skip(lead) {
                let q = (half + lead) as i64 - m as i64;
                *c = tap(p as i64 + up as i64 * q);
            }
        }
        FilterDesign {
            from,
            to,
            up,
            down,
            half,
            taps,
            lead,
            passband,
            stopband,
            beta,
            coefficients,
        }
    }

    /// The shared design for `from` → `to`, from a small cache.
    pub fn shared(from: u32, to: u32) -> Arc<FilterDesign> {
        static CACHE: Mutex<Vec<Arc<FilterDesign>>> = Mutex::new(Vec::new());
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = cache.iter().position(|d| d.from == from && d.to == to) {
            let design = cache.remove(at);
            cache.push(design.clone());
            return design;
        }
        let design = Arc::new(FilterDesign::new(from, to));
        if cache.len() >= CACHED_DESIGNS {
            cache.remove(0);
        }
        cache.push(design.clone());
        design
    }

    /// The input rate.
    pub fn from_rate(&self) -> u32 {
        self.from
    }

    /// The output rate.
    pub fn to_rate(&self) -> u32 {
        self.to
    }

    /// `(up, down)`: the output rate over the input rate, in lowest terms.
    pub fn ratio(&self) -> (u32, u32) {
        (self.up as u32, self.down as u32)
    }

    /// Coefficients per phase (multiply-accumulates per output sample),
    /// including the zero padding to a multiple of 8.
    pub fn taps_per_phase(&self) -> usize {
        self.taps
    }

    /// Input frames on each side of an output frame's instant that it
    /// depends on: the filter's group delay, in input frames, which the
    /// resampler compensates.
    pub fn half_width(&self) -> usize {
        self.half
    }

    /// The filter's group delay in seconds (compensated: the output is not
    /// delayed, but the resampler holds this much input before it emits).
    pub fn delay_seconds(&self) -> f64 {
        self.half as f64 / f64::from(self.from)
    }

    /// The passband edge, in Hz.
    pub fn passband_hz(&self) -> f64 {
        self.passband
    }

    /// The stopband edge, in Hz (half the lower rate).
    pub fn stopband_hz(&self) -> f64 {
        self.stopband
    }

    /// The Kaiser window's shape parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Bytes held by the coefficient table.
    pub fn table_bytes(&self) -> usize {
        self.coefficients.len() * std::mem::size_of::<f64>()
    }

    /// The prototype lowpass at `up · from` Hz: `2 · half · up + 1`
    /// symmetric taps, centre in the middle, DC gain `up`. Rebuilt from the
    /// phases, so it is exactly what the resampler multiplies with.
    pub fn prototype(&self) -> Vec<f64> {
        let reach = self.half * self.up;
        let mut out = vec![0.0; 2 * reach + 1];
        for p in 0..self.up {
            let row = &self.coefficients[p * self.taps..(p + 1) * self.taps];
            for (m, &c) in row.iter().enumerate().skip(self.lead) {
                let q = (self.half + self.lead) as i64 - m as i64;
                let n = p as i64 + self.up as i64 * q;
                if n.unsigned_abs() as usize <= reach {
                    out[(n + reach as i64) as usize] = c;
                }
            }
        }
        out
    }

    fn phase(&self, p: usize) -> &[f64] {
        &self.coefficients[p * self.taps..(p + 1) * self.taps]
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for k in 0..LANES {
            acc[k] += x[k] * y[k];
        }
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// A streaming resampler for interleaved frames of a fixed channel count.
///
/// Feed it with [`Resampler::process`] (or [`Resampler::process_f64`]) in
/// chunks of any size, end the stream with [`Resampler::finish`] (or
/// [`Resampler::finish_f64`]); the output does not depend on the chunking.
/// [`Resampler::reset`] starts a new stream.
pub struct Resampler {
    design: Arc<FilterDesign>,
    channels: usize,
    /// Per channel: input frames from `base` on.
    history: Vec<Vec<f64>>,
    /// The input frame index of `history[c][0]` (negative: leading silence).
    base: i64,
    /// Input frames given.
    fed: u64,
    /// Silent frames appended at the end.
    padded: u64,
    /// Output frames made.
    produced: u64,
    /// For the next output frame `j`: `j · down = index · up + phase`.
    index: i64,
    phase: usize,
    finished: bool,
    /// The input's scale ([`Resampler::set_gain`]).
    gain: f64,
    requantizer: Requantizer,
    scratch: Vec<f64>,
}

impl Resampler {
    /// A resampler from `from` Hz to `to` Hz for `channels` channels, with
    /// the shared design ([`FilterDesign::shared`]).
    ///
    /// # Panics
    ///
    /// If a rate or `channels` is 0.
    pub fn new(from: u32, to: u32, channels: usize) -> Resampler {
        Resampler::with_design(FilterDesign::shared(from, to), channels)
    }

    /// A resampler with this design.
    ///
    /// # Panics
    ///
    /// If `channels` is 0.
    pub fn with_design(design: Arc<FilterDesign>, channels: usize) -> Resampler {
        assert!(channels > 0, "a resampler needs a channel");
        let mut resampler = Resampler {
            design,
            channels,
            history: vec![Vec::new(); channels],
            base: 0,
            fed: 0,
            padded: 0,
            produced: 0,
            index: 0,
            phase: 0,
            finished: false,
            gain: 1.0,
            requantizer: Requantizer::new(),
            scratch: Vec::new(),
        };
        resampler.reset();
        resampler
    }

    /// Starts a new stream: silence before it, the dither generator back at
    /// its seed, the clip count at zero.
    pub fn reset(&mut self) {
        let lead = self.design.half + self.design.lead;
        for channel in &mut self.history {
            channel.clear();
            channel.resize(lead, 0.0);
        }
        self.base = -(lead as i64);
        self.fed = 0;
        self.padded = 0;
        self.produced = 0;
        self.index = 0;
        self.phase = 0;
        self.finished = false;
        self.requantizer.reset();
    }

    /// Scales every input sample by `gain` (1: none; a headroom below 1)
    /// in the `f64` computation, before the one requantization: the output
    /// is the resample of the input scaled by `gain`, with no extra
    /// rounding. It stays across [`Resampler::reset`].
    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    /// The input's scale (1: none).
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// The filter.
    pub fn design(&self) -> &Arc<FilterDesign> {
        &self.design
    }

    /// Channels per frame.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Input frames given since the stream started.
    pub fn input_frames(&self) -> u64 {
        self.fed
    }

    /// Output frames made since the stream started.
    pub fn output_frames(&self) -> u64 {
        self.produced
    }

    /// Output samples that clipped in requantization since the stream
    /// started (saturated to the 24-bit range).
    pub fn clipped(&self) -> u64 {
        self.requantizer.clipped()
    }

    /// Whether the stream has been finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Resamples interleaved 24-bit samples (right-aligned in `i32`) and
    /// appends the 24-bit output frames that are ready to `out`.
    ///
    /// # Panics
    ///
    /// After [`Resampler::finish`], or if `input` is not whole frames.
    pub fn process(&mut self, input: &[i32], out: &mut Vec<i32>) {
        self.push(input.iter().map(|&s| f64::from(s)), input.len());
        self.run_quantized(out, self.end());
    }

    /// Ends the stream: flushes the filter and appends the last 24-bit
    /// output frames, so that the stream's output has exactly
    /// [`output_frames`] frames.
    pub fn finish(&mut self, out: &mut Vec<i32>) {
        self.finish_at(self.end(), out);
    }

    /// [`Resampler::finish`] with the stream ending at output frame
    /// `frames` (counted from its start) instead of where its input ends:
    /// the input goes on as silence up to there, or the output stops short
    /// of its end. Frames already made stay made.
    pub fn finish_at(&mut self, frames: u64, out: &mut Vec<i32>) {
        self.close(frames);
        self.run_quantized(out, frames);
    }

    /// [`Resampler::process`] for samples that are not 24-bit integers
    /// yet: `f64` samples in 24-bit units (full scale ±2²³) in, 24-bit
    /// samples out, requantized once.
    ///
    /// # Panics
    ///
    /// After [`Resampler::finish`], or if `input` is not whole frames.
    pub fn process_float(&mut self, input: &[f64], out: &mut Vec<i32>) {
        self.push(input.iter().copied(), input.len());
        self.run_quantized(out, self.end());
    }

    /// [`Resampler::process`] without requantization: `f64` samples in
    /// and out (any scale; the filter is linear).
    pub fn process_f64(&mut self, input: &[f64], out: &mut Vec<f64>) {
        self.push(input.iter().copied(), input.len());
        self.run(out, self.end());
    }

    /// [`Resampler::finish`] without requantization.
    pub fn finish_f64(&mut self, out: &mut Vec<f64>) {
        let end = self.end();
        self.close(end);
        self.run(out, end);
    }

    fn push(&mut self, samples: impl Iterator<Item = f64>, len: usize) {
        assert!(!self.finished, "the stream was finished; reset first");
        assert_eq!(len % self.channels, 0, "input must be whole frames");
        let channels = self.channels;
        let frames = len / channels;
        let gain = self.gain;
        // Exact for a gain of 1.
        let samples = samples.map(move |s| s * gain);
        for channel in &mut self.history {
            channel.reserve(frames);
        }
        if channels == 1 {
            self.history[0].extend(samples);
        } else {
            let mut samples = samples;
            for _ in 0..frames {
                for channel in &mut self.history {
                    channel.push(samples.next().unwrap_or(0.0));
                }
            }
        }
        self.fed += frames as u64;
    }

    /// Ends the stream at output frame `end`: appends the silence its last
    /// frames reach past the input (the filter's half, and the input
    /// frames up to `end` the input does not have).
    fn close(&mut self, end: u64) {
        assert!(!self.finished, "the stream was finished; reset first");
        let (up, down) = (self.design.up as u128, self.design.down as u128);
        let reach = (u128::from(end) * down).div_ceil(up) as u64;
        let silence = self.design.half as u64 + reach.saturating_sub(self.fed);
        for channel in &mut self.history {
            channel.resize(channel.len() + silence as usize, 0.0);
        }
        self.padded = silence;
        self.finished = true;
    }

    /// The stream's exact length in output frames, by its input so far.
    fn end(&self) -> u64 {
        rounded(self.fed, self.design.up as u32, self.design.down as u32)
    }

    /// Makes every output frame whose input is there, up to output frame
    /// `limit`, and appends them interleaved to `out`.
    fn run(&mut self, out: &mut Vec<f64>, limit: u64) {
        let design = &*self.design;
        let (up, down) = (design.up, design.down);
        let reach = (design.half + design.lead) as i64;
        let available = (self.fed + self.padded) as i64;
        let half = design.half as i64;
        while self.produced < limit && self.index + half < available {
            let start = (self.index - reach - self.base) as usize;
            let coefficients = design.phase(self.phase);
            for channel in &self.history {
                out.push(dot(coefficients, &channel[start..start + design.taps]));
            }
            self.produced += 1;
            self.phase += down;
            self.index += (self.phase / up) as i64;
            self.phase %= up;
        }
        // Drop the input no later output frame needs.
        let keep_from = (self.index - reach - self.base).max(0) as usize;
        let keep_from = keep_from.min(self.history[0].len());
        if keep_from > 4096 && keep_from * 2 > self.history[0].len() {
            for channel in &mut self.history {
                channel.drain(..keep_from);
            }
            self.base += keep_from as i64;
        }
    }

    fn run_quantized(&mut self, out: &mut Vec<i32>, limit: u64) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        self.run(&mut scratch, limit);
        out.reserve(scratch.len());
        out.extend(scratch.iter().map(|&y| self.requantizer.quantize(y)));
        self.scratch = scratch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_reduce() {
        assert_eq!(reduce(44_100, 48_000), (160, 147));
        assert_eq!(reduce(48_000, 44_100), (147, 160));
        assert_eq!(reduce(44_100, 192_000), (640, 147));
        assert_eq!(reduce(384_000, 48_000), (1, 8));
        assert_eq!(reduce(352_800, 192_000), (80, 147));
        assert_eq!(output_frames(44_100, 48_000, 44_100), 48_000);
        assert_eq!(output_frames(44_100, 48_000, 1), 1);
        assert_eq!(output_frames(384_000, 48_000, 4), 1); // 0.5 rounds up
        assert_eq!(output_frames(384_000, 48_000, 3), 0);
    }

    #[test]
    fn bessel_matches_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-14);
        let i10 = 2_815.716_628_466_254;
        assert!((bessel_i0(10.0) - i10).abs() / i10 < 1e-14);
    }

    #[test]
    fn prototype_is_symmetric_with_gain_up() {
        let d = FilterDesign::new(44_100, 48_000);
        let h = d.prototype();
        let n = h.len();
        for k in 0..n / 2 {
            assert_eq!(h[k], h[n - 1 - k]);
        }
        let dc: f64 = h.iter().sum();
        assert!((dc / 160.0 - 1.0).abs() < 1e-7, "{dc}");
        assert_eq!(d.taps_per_phase() % LANES, 0);
    }

    #[test]
    fn dither_is_reproducible_and_clips_are_counted() {
        let input: Vec<i32> = (0..2000)
            .flat_map(|k| {
                let s = if (k / 20) % 2 == 0 {
                    8_388_607
                } else {
                    -8_388_608
                };
                [s, s / 2]
            })
            .collect();
        let run = || {
            let mut r = Resampler::new(48_000, 44_100, 2);
            let mut out = Vec::new();
            r.process(&input, &mut out);
            r.finish(&mut out);
            (out, r.clipped())
        };
        let (a, clipped) = run();
        let (b, _) = run();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2 * 1838);
        // A full-scale square wave overshoots (Gibbs) in the left channel.
        assert!(clipped > 0);
    }

    /// A stream ended at another frame than its input's end: the same
    /// frames up to where both have them, then silence filtered on, or
    /// none past it.
    #[test]
    fn a_stream_ends_at_the_frame_it_is_told() {
        let input: Vec<i32> = (0..1000).flat_map(|k| [k * 1000, -k * 500]).collect();
        for (from, to) in [(44_100, 192_000), (192_000, 44_100)] {
            let run = |end: Option<u64>| {
                let mut r = Resampler::new(from, to, 2);
                let mut out = Vec::new();
                r.process(&input, &mut out);
                match end {
                    Some(frames) => r.finish_at(frames, &mut out),
                    None => r.finish(&mut out),
                }
                out
            };
            let whole = run(None);
            let frames = output_frames(from, to, 1000);
            assert_eq!(whole.len() as u64, 2 * frames);
            assert_eq!(run(Some(frames)), whole);
            let longer = run(Some(frames + 3));
            assert_eq!(longer.len() as u64, 2 * (frames + 3));
            assert_eq!(longer[..whole.len()], whole[..]);
            let shorter = run(Some(frames - 3));
            assert_eq!(shorter[..], whole[..2 * (frames - 3) as usize]);
        }
    }
}
