//! Summing the inputs that are not heard as written: in `f64` with each
//! input's gains, then a look-ahead limiter that turns the gain down only
//! where the sum could and does go over full scale (a smooth gain
//! envelope, no waveshaping), then one requantization to the device's bits
//! with TPDF dither. Where the gains add up to at most 1 (a lone input
//! faded) the sum cannot go over, and the limiter leaves it alone.

use crate::dither::Requantizer;
use crate::format::StreamFormat;

/// Full scale of the 24-bit samples mixed.
const FULL: f64 = 8_388_607.0;
/// How far the limiter looks ahead: the time its gain takes to come down
/// to a peak (linearly).
const ATTACK_SECONDS: f64 = 0.002;
/// How long its gain takes to come back from silence to 1 (linearly).
const RELEASE_SECONDS: f64 = 0.1;

/// An input's gains over the frames summed.
#[derive(Clone, Copy)]
pub(crate) enum Gains<'a> {
    /// The same for every frame.
    Flat(f64),
    /// One per frame.
    Each(&'a [f64]),
}

impl Gains<'_> {
    /// The gain of every frame, when it is one.
    pub(crate) fn flat(self) -> Option<f64> {
        match self {
            Gains::Flat(g) => Some(g),
            Gains::Each(_) => None,
        }
    }

    /// Adds `samples` (`channels` per frame) at these gains to `sums`.
    pub(crate) fn accumulate(self, sums: &mut [f64], samples: &[i32], channels: usize) {
        let ch = channels.max(1);
        for (i, (s, &x)) in sums.iter_mut().zip(samples).enumerate() {
            let g = match self {
                Gains::Flat(g) => g,
                Gains::Each(g) => g[i / ch],
            };
            *s += g * f64::from(x);
        }
    }
}

/// How frames went out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Made {
    /// Nothing heard: digital silence.
    Silence,
    /// One input as written, bit for bit.
    Passthrough,
    /// Summed with gains, limited and dithered.
    Mixed,
}

/// The mixing state: the frames summed and not given out yet (the
/// limiter's look-ahead), its gain, and the dither.
pub(crate) struct Mix {
    channels: usize,
    requantizer: Requantizer,
    bits: u32,
    /// The most the limiter lets through: full scale less one device LSB,
    /// so the dither cannot take it over.
    ceiling: f64,
    /// Frames the limiter looks ahead.
    ahead: usize,
    attack_step: f64,
    release_step: f64,
    /// The limiter's gain after the last frame given out.
    gain: f64,
    /// The limiter turned a frame given out down since the last
    /// [`Mix::take_limited`].
    limited: bool,
    /// Summed samples (interleaved) not given out yet: the held-back ones
    /// first, then those being summed.
    sums: Vec<f64>,
    /// Per frame of `sums`: the inputs' gains added up (their sizes).
    weight: Vec<f64>,
    /// Per frame of `sums`: the gain that keeps it under full scale, then
    /// the limiter's envelope.
    env: Vec<f64>,
}

impl Mix {
    /// Mixing for a device in `format`.
    pub(crate) fn new(format: StreamFormat) -> Mix {
        let rate = f64::from(format.rate.max(1));
        let ahead = ((ATTACK_SECONDS * rate).round() as usize).max(1);
        let bits = format.sample.valid_bits().min(24);
        Mix {
            channels: usize::from(format.channels).max(1),
            requantizer: Requantizer::new(),
            bits,
            ceiling: FULL - f64::from(1u32 << (24 - bits)),
            ahead,
            attack_step: 1.0 / ahead as f64,
            release_step: 1.0 / (RELEASE_SECONDS * rate),
            gain: 1.0,
            limited: false,
            sums: Vec::new(),
            weight: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Samples that clipped when requantized, since the start.
    pub(crate) fn clipped(&self) -> u64 {
        self.requantizer.clipped()
    }

    /// Whether frames may go out as an input wrote them: nothing held
    /// back, the limiter's gain at 1.
    pub(crate) fn at_rest(&self) -> bool {
        self.sums.is_empty() && self.gain == 1.0
    }

    /// Whether the limiter turned a frame given out down since the last
    /// look; the next look starts.
    pub(crate) fn take_limited(&mut self) -> bool {
        std::mem::take(&mut self.limited)
    }

    /// Frames held back.
    pub(crate) fn held(&self) -> usize {
        self.sums.len() / self.channels
    }

    /// Room for summing chunks of `frames` frames (and what the limiter
    /// holds back), so that it never allocates.
    pub(crate) fn reserve(&mut self, frames: usize) {
        let frames = frames + self.ahead;
        let samples = frames * self.channels;
        self.sums.reserve(samples.saturating_sub(self.sums.len()));
        self.weight
            .reserve(frames.saturating_sub(self.weight.len()));
        self.env.reserve(frames.saturating_sub(self.env.len()));
    }

    /// Starts summing `samples` more samples (whole frames), after those
    /// held back.
    pub(crate) fn begin(&mut self, samples: usize) {
        let held = self.sums.len();
        self.sums.resize(held + samples, 0.0);
        self.weight.resize(self.sums.len() / self.channels, 0.0);
    }

    /// Adds an input's `samples` (the length [`Mix::begin`] was given) with
    /// its gains.
    pub(crate) fn add(&mut self, samples: &[i32], gain: Gains) {
        let ch = self.channels;
        let start = self.sums.len() - samples.len();
        let frames = samples.len() / ch;
        let weight = self.weight.len() - frames;
        gain.accumulate(&mut self.sums[start..], samples, ch);
        for (f, w) in self.weight[weight..].iter_mut().enumerate() {
            *w += match gain {
                Gains::Flat(g) => g.abs(),
                Gains::Each(g) => g[f].abs(),
            };
        }
    }

    /// Limits and requantizes the frames summed, appending to `out` all but
    /// the limiter's look-ahead; with `settle` (what comes next may go out
    /// as written), all of them.
    pub(crate) fn end(&mut self, settle: bool, out: &mut Vec<i32>) {
        let ch = self.channels;
        let frames = self.sums.len() / ch;
        let give = if settle {
            frames
        } else {
            frames.saturating_sub(self.ahead)
        };
        // The gain each frame needs; then, from the end, a linear attack
        // into every dip (frames after the sums need none).
        let ceiling = self.ceiling;
        self.env.clear();
        self.env
            .extend(self.sums.chunks_exact(ch).zip(&self.weight).map(|(f, &w)| {
                let peak = f.iter().fold(0.0f64, |m, s| m.max(s.abs()));
                if w > 1.0 && peak > ceiling {
                    ceiling / peak
                } else {
                    1.0
                }
            }));
        let mut next = 1.0f64;
        for e in self.env.iter_mut().rev() {
            next = e.min(next + self.attack_step);
            *e = next;
        }
        for (f, frame) in self.sums.chunks_exact(ch).take(give).enumerate() {
            self.gain = self.env[f].min(self.gain + self.release_step);
            self.limited |= self.gain < 1.0;
            for &s in frame {
                out.push(self.requantizer.quantize_to(s * self.gain, self.bits));
            }
        }
        self.sums.drain(..give * ch);
        self.weight.drain(..give);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SampleFormat;

    const RATE: u32 = 48_000;

    fn mix(sample: SampleFormat) -> Mix {
        Mix::new(StreamFormat::new(RATE, 2, sample))
    }

    #[test]
    fn two_inputs_sum_and_are_dithered_once() {
        let n = 4_000;
        let a: Vec<i32> = (0..n)
            .map(|k| ((k as f64 * 0.01).sin() * 3e6) as i32)
            .collect();
        let b: Vec<i32> = (0..n)
            .map(|k| ((k as f64 * 0.003).cos() * 4e6) as i32)
            .collect();
        let gains: Vec<f64> = (0..n / 2).map(|f| f as f64 / (n / 2) as f64).collect();
        let mut m = mix(SampleFormat::S32);
        let mut out = Vec::new();
        m.begin(n);
        m.add(&a, Gains::Flat(0.5));
        m.add(&b, Gains::Each(&gains));
        m.end(true, &mut out);
        assert_eq!(out.len(), n);
        assert!(!m.take_limited());
        let mut q = Requantizer::new();
        for (i, &y) in out.iter().enumerate() {
            let sum = 0.5 * f64::from(a[i]) + gains[i / 2] * f64::from(b[i]);
            assert_eq!(y, q.quantize(sum), "{i}");
        }
        assert!(m.at_rest());
    }

    #[test]
    fn a_lone_fade_is_never_limited() {
        // A master peaking at full scale, faded out: only the gains and
        // the dither, however loud.
        let n = 2 * 4_800;
        let x: Vec<i32> = (0..n)
            .map(|k| if k % 4 < 2 { 8_388_607 } else { -8_388_608 })
            .collect();
        let gains: Vec<f64> = (0..n / 2)
            .map(|f| 1.0 - f as f64 / (n / 2) as f64)
            .collect();
        let mut m = mix(SampleFormat::S32);
        let mut out = Vec::new();
        m.begin(n);
        m.add(&x, Gains::Each(&gains));
        m.end(true, &mut out);
        for (i, &y) in out.iter().enumerate() {
            let want = gains[i / 2] * f64::from(x[i]);
            assert!((f64::from(y) - want).abs() < 1.5, "{i}: {y} for {want}");
        }
        assert!(m.at_rest());
    }

    #[test]
    fn a_sum_over_full_scale_is_turned_down_smoothly_ahead_of_its_peak() {
        let mut m = mix(SampleFormat::S32);
        let ahead = m.ahead;
        let mut out = Vec::new();
        // Quiet, then two inputs at 3/4 of full scale together (1.5×).
        let n = 2 * 2_000;
        let x: Vec<i32> = (0..n)
            .map(|k| if k >= 2 * 1_000 { 6_291_456 } else { 1_000 })
            .collect();
        m.begin(n);
        m.add(&x, Gains::Flat(1.0));
        m.add(&x, Gains::Flat(1.0));
        m.end(false, &mut out);
        // The look-ahead is held back.
        assert_eq!(out.len(), n - 2 * ahead);
        assert_eq!(m.held(), ahead);
        let left: Vec<i32> = out.iter().step_by(2).copied().collect();
        // Never over, and no step: the gain comes down over the attack
        // before the loud part, then holds it at full scale.
        assert!(left.iter().all(|&v| v <= 8_388_607));
        for f in 1_000 - ahead..1_000 {
            let g_was = f64::from(left[f - 1]) / 2_000.0;
            let g = f64::from(left[f]) / 2_000.0;
            assert!(g <= g_was + 0.01 && g_was - g < 2.0 / ahead as f64, "{f}");
        }
        let loud = &left[1_000 + 10..];
        assert!(
            loud.iter().all(|&v| (v - 8_388_607).abs() <= 2),
            "{:?}",
            &loud[..4]
        );
        assert_eq!(m.clipped(), 0);
        assert!(m.take_limited() && !m.take_limited());
        assert!(!m.at_rest());
        // What comes after, quiet: the rest given out, the gain coming back.
        m.begin(2);
        m.add(&[0, 0], Gains::Flat(1.0));
        m.end(true, &mut out);
        assert_eq!(out.len(), n + 2);
        assert!(!m.at_rest(), "still releasing");
    }

    #[test]
    fn a_16_bit_device_gets_16_bit_samples_dithered() {
        let mut m = mix(SampleFormat::S16);
        let mut out = Vec::new();
        let n = 2 * 10_000;
        let x = vec![256_064; n]; // 1000.25 16-bit LSBs
        m.begin(n);
        m.add(&x, Gains::Flat(0.999_999));
        m.end(true, &mut out);
        assert!(out.iter().all(|&v| v & 0xFF == 0));
        let mean = out.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        assert!((mean - 256_064.0 * 0.999_999).abs() < 4.0, "{mean}");
    }
}
