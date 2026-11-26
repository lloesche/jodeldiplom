//! Time-stretching that keeps the pitch: one mono stream played faster or
//! slower than it was recorded, every partial at its own frequency.
//!
//! [`Stretcher`] is a phase vocoder. The input is cut into Hann-windowed
//! frames (2048 samples at 44.1 and 48 kHz, longer at higher rates) whose
//! centres advance by the speed times the synthesis hop (a quarter frame);
//! each frame's spectrum is laid down again a fixed synthesis hop after the
//! last one, with its phases advanced so that every partial continues
//! where the previous frame left it. Three things keep the result clean:
//!
//! - **Identity phase locking.** Each spectral peak's phase is advanced by
//!   its measured frequency; the bins around it (up to the lowest bin
//!   between it and the next peak) keep their phase relative to the peak
//!   as measured, so a partial's bins move together and the sound does not
//!   go hollow.
//! - **Phase reset at onsets.** When many bins jump in level from one frame
//!   to the next (a magnitude-weighted share of the spectrum that at least
//!   doubled), those bins take the measured phase unchanged, which puts the
//!   attack back where it was in the frame instead of smearing it. A
//!   caller that knows its onsets (from a drum detector, say) can force a
//!   reset of the whole frame with [`Stretcher::onset`].
//! - **Smooth speed.** [`Stretcher::set_speed`] sets a target the speed
//!   glides to over a few frames; frames overlap by three quarters, so the
//!   change is spread over the output without a step.
//!
//! # Time
//!
//! The output carries no delay: output sample `j` plays the input around
//! [`Stretcher::position_at`]`(j)`, a piecewise-linear map through the
//! frame centres that starts at 0 (at speed 1 the output equals the input
//! sample for sample). [`Stretcher::input_position`] is that map at the
//! next sample [`Stretcher::pull`] will return, the hook for locking the
//! stream to a time map. To produce a sample the stretcher needs input
//! ahead of its position: [`Stretcher::latency`] says how much, and a pull
//! returns fewer samples than asked (or none) until it has been pushed.
//! After [`Stretcher::finish`] the input is taken as silent past its end
//! and the output stops at the sample whose position reaches the end, so a
//! stream of `n` samples at a constant speed `s` comes out `⌈n / s⌉` long.
//!
//! Samples are `f64` at any full scale: nothing in the stretcher depends on
//! the level.

use std::collections::VecDeque;
use std::f64::consts::PI;

use jodel_dsp::fft::{Complex, Fft};
use jodel_dsp::window::hann;

/// The slowest speed accepted (input seconds per output second).
pub const MIN_SPEED: f64 = 0.25;
/// The fastest speed accepted.
pub const MAX_SPEED: f64 = 4.0;
/// The share of the way to the target speed covered at each frame.
const GLIDE: f64 = 0.5;
/// A bin is new at a frame when its magnitude grew by at least this factor.
const RISE: f64 = 2.0;
/// A frame is an onset when new bins carry at least this share of its
/// magnitude.
const ONSET_SHARE: f64 = 0.3;
/// A peak stands at least this far above the bins two away on each side.
const PEAK_RATIO: f64 = 1.5;
/// The per-frame decay of the level an onset is measured against.
const LEVEL_DECAY: f64 = 0.99;
/// Frames quieter than this share of the recent level are no onset.
const LEVEL_FLOOR: f64 = 1e-4;
/// Frame centres kept for the time map.
const HISTORY: usize = 8;
/// No peak owns a bin.
const NO_PEAK: usize = usize::MAX;

/// A pitch-preserving time-stretcher for one mono stream.
#[derive(Clone, Debug)]
pub struct Stretcher {
    /// The frame length (a power of two).
    size: usize,
    /// The synthesis hop, a quarter of the frame.
    hop: usize,
    /// The transform for `size` points.
    fft: Fft<f64>,
    /// The periodic Hann window, used for analysis and synthesis.
    window: Vec<f64>,
    /// The speed the stretcher glides to.
    target: f64,
    /// The speed of the last frame.
    speed: f64,
    /// Whether onsets are found in the spectrum.
    detect: bool,

    /// Input not yet behind every frame to come.
    input: VecDeque<f64>,
    /// The input index of `input[0]`.
    input_start: i64,
    /// Input samples pushed so far.
    pushed: i64,
    /// Whether the input has ended.
    finished: bool,
    /// Input indices where the caller marked an onset, ascending.
    hints: VecDeque<i64>,

    /// The output and input centres of the most recent frames.
    centres: VecDeque<(i64, f64)>,
    /// The input sample the last frame was centred on.
    last_read: i64,
    /// The last frame's measured magnitudes.
    prev_mag: Vec<f64>,
    /// The last frame's measured phases.
    prev_phase: Vec<f64>,
    /// The phases the last frame was laid down with.
    synth_phase: Vec<f64>,
    /// The recent frame level (decaying maximum of the magnitude sum).
    level: f64,

    /// The overlap-add buffer, one frame long.
    acc: Vec<f64>,
    /// The output index of `acc[0]`.
    acc_start: i64,
    /// Finished output not yet pulled.
    ready: VecDeque<f64>,
    /// Output samples pulled so far.
    emitted: i64,
    /// Whether the output has reached the end of the input.
    done: bool,

    /// Scratch: the frame's spectrum.
    spectrum: Vec<Complex<f64>>,
    /// Scratch: the frame's magnitudes.
    mag: Vec<f64>,
    /// Scratch: the frame's phases.
    phase: Vec<f64>,
    /// Scratch: the phases each bin advances to on its own.
    advanced: Vec<f64>,
    /// Scratch: the peaks.
    peaks: Vec<usize>,
    /// Scratch: the peak each bin is locked to.
    owner: Vec<usize>,
    /// Scratch: the bins that take their measured phase.
    reset: Vec<bool>,
}

impl Stretcher {
    /// A stretcher for a stream at `rate` Hz, at speed 1. The frame is the
    /// shortest power of two of at least 2048 samples and 40 ms.
    pub fn new(rate: u32) -> Stretcher {
        let mut size = 2048;
        while (size as f64) < f64::from(rate) * 0.04 {
            size *= 2;
        }
        let hop = size / 4;
        let bins = size / 2 + 1;
        Stretcher {
            size,
            hop,
            fft: Fft::new(size),
            window: hann(size),
            target: 1.0,
            speed: 1.0,
            detect: true,
            input: VecDeque::new(),
            input_start: 0,
            pushed: 0,
            finished: false,
            hints: VecDeque::new(),
            centres: VecDeque::with_capacity(HISTORY + 1),
            last_read: 0,
            prev_mag: vec![0.0; bins],
            prev_phase: vec![0.0; bins],
            synth_phase: vec![0.0; bins],
            level: 0.0,
            acc: vec![0.0; size],
            acc_start: -((size - hop) as i64),
            ready: VecDeque::new(),
            emitted: 0,
            done: false,
            spectrum: vec![Complex::default(); size],
            mag: vec![0.0; bins],
            phase: vec![0.0; bins],
            advanced: vec![0.0; bins],
            peaks: Vec::new(),
            owner: vec![NO_PEAK; bins],
            reset: vec![false; bins],
        }
    }

    /// The frame length in samples.
    pub fn frame_len(&self) -> usize {
        self.size
    }

    /// Sets the speed, in input seconds per output second (above 1 plays
    /// faster), clamped to [`MIN_SPEED`]..=[`MAX_SPEED`]; anything not
    /// finite is ignored. Before the first frame it applies at once; after,
    /// the speed glides there over a few frames.
    pub fn set_speed(&mut self, speed: f64) {
        if !speed.is_finite() {
            return;
        }
        self.target = speed.clamp(MIN_SPEED, MAX_SPEED);
        if self.centres.is_empty() {
            self.speed = self.target;
        }
    }

    /// The speed of the last frame (the target before the first).
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Marks whether a transient starts at the next input sample pushed:
    /// `true` makes the frame centred nearest it take every measured phase
    /// unchanged; `false` does nothing, so a detector's verdict can be
    /// passed straight in.
    pub fn onset(&mut self, here: bool) {
        if here && self.hints.back() != Some(&self.pushed) {
            self.hints.push_back(self.pushed);
        }
    }

    /// Turns finding onsets in the spectrum on or off (on by default);
    /// onsets marked with [`Stretcher::onset`] reset either way.
    pub fn detect_onsets(&mut self, on: bool) {
        self.detect = on;
    }

    /// Appends input. Ignored after [`Stretcher::finish`]. Input is kept
    /// until every frame that reads it is made, which happens in
    /// [`Stretcher::pull`].
    pub fn push(&mut self, samples: &[f64]) {
        if self.finished {
            return;
        }
        self.input.extend(samples.iter().copied());
        self.pushed += samples.len() as i64;
    }

    /// Ends the input: what follows is silence, and the output stops where
    /// its position reaches the last input sample.
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Appends up to `frames` output samples to `out` and returns how many.
    /// Fewer means the stretcher needs more input (see
    /// [`Stretcher::latency`]) or, after [`Stretcher::finish`], that the
    /// output has ended.
    pub fn pull(&mut self, out: &mut Vec<f64>, frames: usize) -> usize {
        while self.ready.len() < frames && self.make_frame() {}
        let n = frames.min(self.ready.len());
        out.extend(self.ready.drain(..n));
        self.emitted += n as i64;
        n
    }

    /// Whether the input has ended and every output sample was pulled.
    pub fn is_done(&self) -> bool {
        self.done && self.ready.is_empty()
    }

    /// Output samples pulled so far.
    pub fn output_position(&self) -> u64 {
        self.emitted as u64
    }

    /// The input position (in samples, fractional) of the next output
    /// sample [`Stretcher::pull`] returns.
    pub fn input_position(&self) -> f64 {
        self.position_at(self.emitted)
    }

    /// The input position of output sample `j`: linear between the centres
    /// of the frames around it, and beyond the frames made so far, straight
    /// on at the current speed.
    pub fn position_at(&self, j: i64) -> f64 {
        let Some(&(last_d, last_c)) = self.centres.back() else {
            return j as f64 * self.speed;
        };
        if j >= last_d {
            return last_c + (j - last_d) as f64 * self.speed;
        }
        let mut later = (last_d, last_c);
        for &(d, c) in self.centres.iter().rev().skip(1) {
            if j >= d {
                return c + (j - d) as f64 * (later.1 - c) / (later.0 - d) as f64;
            }
            later = (d, c);
        }
        let (d, c) = later;
        c + (j - d) as f64 * self.speed
    }

    /// The output sample (fractional) whose input position is `position`:
    /// [`Stretcher::position_at`] the other way (it rises), so a reader of
    /// the output can follow the input's time through it.
    pub fn output_at(&self, position: f64) -> f64 {
        let speed = self.speed.max(MIN_SPEED);
        let Some(&(last_d, last_c)) = self.centres.back() else {
            return position / speed;
        };
        if position >= last_c {
            return last_d as f64 + (position - last_c) / speed;
        }
        let mut later = (last_d, last_c);
        for &(d, c) in self.centres.iter().rev().skip(1) {
            if position >= c {
                let span = later.1 - c;
                let step = if span > 0.0 {
                    (position - c) / span
                } else {
                    0.0
                };
                return d as f64 + step * (later.0 - d) as f64;
            }
            later = (d, c);
        }
        let (d, c) = later;
        d as f64 + (position - c) / speed
    }

    /// The most input, in samples past [`Stretcher::input_position`], that
    /// a pull may need before it returns a sample: half a frame on each
    /// side of the frame centre, the centre up to half a frame of output
    /// ahead at the faster of the current and target speeds.
    pub fn latency(&self) -> usize {
        let half = (self.size / 2) as f64;
        (half * (1.0 + self.speed.max(self.target)) + 1.0).ceil() as usize
    }

    /// The input sample at index `i` (silence outside the input).
    fn sample(&self, i: i64) -> f64 {
        if i < self.input_start || i >= self.pushed {
            0.0
        } else {
            self.input[(i - self.input_start) as usize]
        }
    }

    /// Makes the next frame and moves the output it finishes to `ready`;
    /// false when the input for it is not there yet or the output ended.
    fn make_frame(&mut self) -> bool {
        if self.done {
            return false;
        }
        let first = self.centres.is_empty();
        let speed = if first {
            self.target
        } else {
            let s = self.speed + (self.target - self.speed) * GLIDE;
            if (self.target - s).abs() < 1e-12 {
                self.target
            } else {
                s
            }
        };
        let hop = self.hop as f64;
        let (centre_out, centre_in) = match self.centres.back() {
            None => {
                let d = -((self.size / 2 - self.hop) as i64);
                (d, d as f64 * speed)
            }
            Some(&(d, c)) => (d + self.hop as i64, c + hop * speed),
        };
        let read = centre_in.round() as i64;
        let half = (self.size / 2) as i64;
        let start = read - half;
        if !self.finished && start + self.size as i64 > self.pushed {
            return false;
        }
        self.speed = speed;

        // Analysis.
        let mut spectrum = std::mem::take(&mut self.spectrum);
        for (i, (z, w)) in spectrum.iter_mut().zip(&self.window).enumerate() {
            *z = Complex {
                re: self.sample(start + i as i64) * w,
                im: 0.0,
            };
        }
        self.spectrum = spectrum;
        self.fft.forward(&mut self.spectrum);
        let bins = self.size / 2 + 1;
        for ((z, mag), phase) in self.spectrum.iter().zip(&mut self.mag).zip(&mut self.phase) {
            *mag = z.re.hypot(z.im);
            *phase = z.im.atan2(z.re);
        }

        // Onsets: the bins that jumped, and whether they carry the frame.
        let total: f64 = self.mag.iter().sum();
        self.level = (self.level * LEVEL_DECAY).max(total);
        let mut rising = 0.0;
        for ((reset, &mag), &prev) in self.reset.iter_mut().zip(&self.mag).zip(&self.prev_mag) {
            *reset = mag > RISE * prev;
            if *reset {
                rising += mag;
            }
        }
        let audible = total > 0.0 && total >= self.level * LEVEL_FLOOR;
        let detected = self.detect && audible && rising >= ONSET_SHARE * total;
        let reach = hop * speed / 2.0;
        let mut hinted = false;
        while let Some(&h) = self.hints.front() {
            let offset = (h - read) as f64;
            if offset >= reach {
                break;
            }
            if offset >= -reach {
                hinted = true;
            }
            self.hints.pop_front();
        }
        if first || hinted {
            self.reset.fill(true);
        } else if !detected {
            self.reset.fill(false);
        }

        // Phases.
        if first {
            self.synth_phase.copy_from_slice(&self.phase);
        } else {
            let step = (read - self.last_read) as f64;
            for b in 0..bins {
                let omega = 2.0 * PI * b as f64 / self.size as f64;
                let deviation = wrap(self.phase[b] - self.prev_phase[b] - omega * step);
                let frequency = omega + deviation / step;
                self.advanced[b] = wrap(self.synth_phase[b] + frequency * hop);
            }
            self.find_peaks();
            for b in 0..bins {
                let p = self.owner[b];
                self.synth_phase[b] = if self.reset[b] {
                    self.phase[b]
                } else if p == NO_PEAK {
                    self.advanced[b]
                } else {
                    let peak = if self.reset[p] {
                        self.phase[p]
                    } else {
                        self.advanced[p]
                    };
                    wrap(peak + self.phase[b] - self.phase[p])
                };
            }
            // The edge bins are real: they keep their sign.
            self.synth_phase[0] = self.phase[0];
            self.synth_phase[bins - 1] = self.phase[bins - 1];
        }

        // Synthesis.
        for ((z, &mag), &phase) in self
            .spectrum
            .iter_mut()
            .zip(&self.mag)
            .zip(&self.synth_phase)
        {
            let (sin, cos) = phase.sin_cos();
            *z = Complex {
                re: mag * cos,
                im: mag * sin,
            };
        }
        for b in 1..self.size / 2 {
            self.spectrum[self.size - b] = self.spectrum[b].conj();
        }
        self.fft.inverse(&mut self.spectrum);
        // The squared window sums to 1.5 at a hop of a quarter frame.
        let gain = 1.0 / 1.5;
        for ((a, z), w) in self.acc.iter_mut().zip(&self.spectrum).zip(&self.window) {
            *a += z.re * w * gain;
        }

        std::mem::swap(&mut self.prev_mag, &mut self.mag);
        std::mem::swap(&mut self.prev_phase, &mut self.phase);
        self.last_read = read;
        self.centres.push_back((centre_out, centre_in));
        if self.centres.len() > HISTORY {
            self.centres.pop_front();
        }

        // The first hop of the buffer is final.
        for i in 0..self.hop {
            let j = self.acc_start + i as i64;
            if j < 0 {
                continue;
            }
            if self.finished && self.position_at(j) >= self.pushed as f64 {
                self.done = true;
                break;
            }
            self.ready.push_back(self.acc[i]);
        }
        self.acc.drain(..self.hop);
        self.acc.resize(self.size, 0.0);
        self.acc_start += self.hop as i64;

        // Input before this frame's start is behind every frame to come.
        while self.input_start < start && !self.input.is_empty() {
            self.input.pop_front();
            self.input_start += 1;
        }
        true
    }

    /// Finds the peaks of `mag` and gives every bin to one: the bins up to
    /// the lowest one between two peaks go to the lower peak.
    fn find_peaks(&mut self) {
        let bins = self.mag.len();
        let m = &self.mag;
        self.peaks.clear();
        for b in 2..bins - 2 {
            if m[b] > m[b - 1] && m[b] >= m[b + 1] && m[b] > PEAK_RATIO * m[b - 2].max(m[b + 2]) {
                self.peaks.push(b);
            }
        }
        if self.peaks.is_empty() {
            self.owner.fill(NO_PEAK);
            return;
        }
        let mut from = 0;
        for (i, &p) in self.peaks.iter().enumerate() {
            let to = match self.peaks.get(i + 1) {
                None => bins - 1,
                Some(&next) => {
                    let mut low = p;
                    for b in p + 1..next {
                        if m[b] < m[low] {
                            low = b;
                        }
                    }
                    low
                }
            };
            self.owner[from..=to].fill(p);
            from = to + 1;
        }
    }
}

/// `x` wrapped to -π..=π.
fn wrap(x: f64) -> f64 {
    x - 2.0 * PI * (x / (2.0 * PI)).round()
}

#[cfg(test)]
mod tests {
    use jodel_dsp::window::hann_at;

    use super::*;

    const RATE: u32 = 44_100;

    /// `input` through a stretcher at `speed`, pushed and pulled in uneven
    /// chunks.
    fn stretch(input: &[f64], speed: f64, detect: bool) -> Vec<f64> {
        let mut s = Stretcher::new(RATE);
        s.set_speed(speed);
        s.detect_onsets(detect);
        let mut out = Vec::new();
        for chunk in input.chunks(1000) {
            s.push(chunk);
            while s.pull(&mut out, 700) > 0 {}
        }
        s.finish();
        while s.pull(&mut out, 700) > 0 {}
        assert!(s.is_done());
        out
    }

    fn tone(freq: f64, amplitude: f64, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| amplitude * (2.0 * PI * freq * i as f64 / f64::from(RATE)).sin())
            .collect()
    }

    /// The bin of the largest magnitude in a Hann-windowed transform of
    /// `x` (a power of two long).
    fn peak_bin(x: &[f64]) -> usize {
        let n = x.len();
        let fft = Fft::new(n);
        let mut data: Vec<Complex<f64>> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| Complex::new(v * hann_at(i, n), 0.0))
            .collect();
        fft.forward(&mut data);
        (1..n / 2)
            .max_by(|&a, &b| {
                let ma = data[a].re.hypot(data[a].im);
                let mb = data[b].re.hypot(data[b].im);
                ma.total_cmp(&mb)
            })
            .unwrap()
    }
    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }

    /// A deterministic noise source in -1..1.
    fn noise(len: usize, mut seed: u64) -> Vec<f64> {
        (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    #[test]
    fn a_tone_keeps_its_pitch_and_level_at_the_expected_length() {
        let len = 3 * RATE as usize;
        for (speed, amplitude) in [(0.8, 0.5), (1.25, 0.5), (1.25, 8_388_607.0)] {
            let input = tone(440.0, amplitude, len);
            let out = stretch(&input, speed, true);
            let expected = (len as f64 / speed).ceil();
            assert!(
                (out.len() as f64 - expected).abs() <= 512.0,
                "speed {speed}: {} samples, expected {expected}",
                out.len()
            );
            let n = 1 << 16;
            let middle = &out[4096..4096 + n];
            let bin = peak_bin(middle) as f64;
            let want = 440.0 * n as f64 / f64::from(RATE);
            assert!(
                (bin - want).abs() <= 1.0,
                "speed {speed}: bin {bin}, want {want:.2}"
            );
            let level = rms(middle) / (amplitude / 2f64.sqrt());
            assert!((level - 1.0).abs() < 0.05, "speed {speed}: level {level}");
        }
    }

    /// Where each click of `clicks` comes out of `out` at `speed`: the
    /// distance of the loudest sample from where the time map puts the
    /// click, its level, and the share of the energy within 10 ms of it.
    fn click_report(out: &[f64], clicks: &[usize], speed: f64) -> Vec<(usize, f64, f64)> {
        let rate = f64::from(RATE);
        let tight = (0.010 * rate) as usize;
        let around = (0.2 * rate) as usize;
        clicks
            .iter()
            .map(|&c| {
                let at = (c as f64 / speed).round() as usize;
                let region = &out[at - around..at + around];
                let (peak, value) = region
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .map(|(i, v)| (at - around + i, v.abs()))
                    .unwrap();
                let all: f64 = region.iter().map(|v| v * v).sum();
                let near: f64 = out[peak - tight..=peak + tight].iter().map(|v| v * v).sum();
                (peak.abs_diff(at), value, near / all)
            })
            .collect()
    }

    /// Clicks once every half second, stretched by 1.1: each comes out as a
    /// single sharp click where the time map puts it, which it does not
    /// without the phase reset.
    #[test]
    fn clicks_stay_sharp_and_in_place() {
        let rate = f64::from(RATE);
        let speed = 1.1;
        let clicks: Vec<usize> = (0..10)
            .map(|m| (rate * (0.25 + 0.5 * m as f64)) as usize)
            .collect();
        let mut input = vec![0.0; 5 * RATE as usize];
        for &c in &clicks {
            input[c] = 1.0;
        }
        let within = (0.003 * rate) as usize;
        let out = stretch(&input, speed, true);
        for (&c, (off, value, near)) in clicks.iter().zip(click_report(&out, &clicks, speed)) {
            assert!(off <= within, "click at {c}: {off} samples off");
            assert!(value > 0.3, "click at {c}: peak {value}");
            assert!(
                near > 0.98,
                "click at {c}: {near:.4} of the energy within 10 ms"
            );
        }
        let smeared = stretch(&input, speed, false);
        let worst = click_report(&smeared, &clicks, speed)
            .iter()
            .map(|r| r.2)
            .fold(1.0, f64::min);
        assert!(
            worst < 0.9,
            "without resets the worst click still has {worst:.4} within 10 ms"
        );
    }

    /// At speed 1 the output is the input, with no delay.
    #[test]
    fn speed_one_reproduces_the_input() {
        let len = 2 * RATE as usize;
        let mut input = tone(300.0, 0.4, len);
        for (x, n) in input.iter_mut().zip(noise(len, 3)) {
            *x += 0.2 * n;
        }
        for i in (5000..len).step_by(11_025) {
            input[i] += 0.8;
        }
        let out = stretch(&input, 1.0, true);
        assert_eq!(out.len(), input.len());
        let signal: f64 = input.iter().map(|v| v * v).sum();
        let error: f64 = input.iter().zip(&out).map(|(a, b)| (a - b) * (a - b)).sum();
        let snr = 10.0 * (signal / error).log10();
        assert!(snr > 40.0, "SNR {snr:.1} dB");
    }

    /// Speed changes mid-stream leave no step in the output.
    #[test]
    fn speed_changes_are_smooth() {
        let len = 4 * RATE as usize;
        let input: Vec<f64> = tone(440.0, 0.5, len)
            .iter()
            .zip(tone(1000.0, 0.5, len))
            .map(|(a, b)| a + b)
            .collect();
        let mut s = Stretcher::new(RATE);
        let mut out = Vec::new();
        let mut fed = 0;
        let speeds = [1.0, 1.3, 1.3, 0.8, 0.95, 1.05, 1.0];
        let mut pulls = 0;
        while !s.is_done() {
            s.set_speed(speeds[(pulls / 40) % speeds.len()]);
            let want = (s.input_position() as usize + s.latency()).min(len);
            if want > fed {
                s.push(&input[fed..want]);
                fed = want;
            }
            if fed == len {
                s.finish();
            }
            assert!(
                s.pull(&mut out, 300) > 0 || s.is_done(),
                "starved at pull {pulls}"
            );
            pulls += 1;
        }
        let omega = |f: f64| 2.0 * PI * f / f64::from(RATE);
        let bound = 0.5 * omega(440.0) + 0.5 * omega(1000.0);
        let jump = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f64::max);
        assert!(
            jump < 1.2 * bound,
            "jump {jump:.4}, the input's largest step {bound:.4}"
        );
    }

    /// The input position follows the speed and ends at the input's end.
    #[test]
    fn input_position_follows_the_speed() {
        let len = 2 * RATE as usize;
        let input = noise(len, 11);
        let mut s = Stretcher::new(RATE);
        s.set_speed(1.25);
        s.push(&input);
        s.finish();
        let mut out = Vec::new();
        assert_eq!(s.input_position(), 0.0);
        s.pull(&mut out, 10_000);
        assert!(
            (s.input_position() - 12_500.0).abs() < 0.5,
            "{}",
            s.input_position()
        );
        s.set_speed(0.9);
        let mut last = s.input_position();
        loop {
            let n = s.pull(&mut out, 500);
            if n == 0 {
                break;
            }
            let now = s.input_position();
            let step = (now - last) / n as f64;
            assert!((0.89..=1.26).contains(&step), "step {step}");
            last = now;
        }
        assert!(
            (last - len as f64).abs() < 1.0,
            "ended at {last}, input {len}"
        );
    }
}
