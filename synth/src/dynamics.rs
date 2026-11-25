//! The effects bus's dynamics, so stacked hits never turn harsh or clip: a
//! smooth feed-forward compressor ([`Compressor`]) and a look-ahead peak
//! limiter ([`Limiter`]), both on a stereo block in place, both linked (one
//! gain for the two sides, so the image stays where it is). Their state is
//! fixed in size; nothing allocates, at their making or as they run.
//!
//! The compressor works in decibels: the louder side's level goes through
//! the static curve (unity below the knee, the ratio above it, a quadratic
//! between), and the gain reduction it asks for is smoothed by a peak
//! detector, decoupled: a release stage that follows a rise at once and
//! falls back over the release, and an attack stage after it that
//! follows the release stage's output over the attack. Smoothing the
//! reduction, not the level, keeps the attack and release times what they
//! say whatever the ratio.
//!
//! The limiter holds the audio back by [`LOOKAHEAD`] (and two samples
//! more) so its gain can be down before a peak arrives. For each sample it
//! takes the gain that keeps it under the ceiling, its peak counted as the
//! largest of the sample and the two points halfway to its neighbours (a
//! cubic through four samples: so a peak between two samples counts too).
//! The least of those gains over the look-ahead, let back up towards 1
//! over the release (never above that least), is averaged over the
//! look-ahead: the gain falls smoothly in a straight line ahead of a peak,
//! yet every one of the averaged gains is at most the one the delayed
//! sample needs, so their mean is too. A last clamp at the ceiling only
//! ever catches rounding.
//!
//! Sources: the gain computer with its soft knee, the log-domain detector
//! and the smooth decoupled peak detector after D. Giannoulis, M. Massberg
//! and J. D. Reiss, "Digital Dynamic Range Compressor Design--A Tutorial
//! and Analysis" (J. Audio Eng. Soc. 60(6), 2012); the limiter's sliding
//! minimum and moving average are our own, as is the bound above; the
//! midpoint the four-point Lagrange (cubic) interpolation at one half.

use crate::dsp::Rate;
use crate::BLOCK;

/// The compressor's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompressorSettings {
    /// Where it starts, dBFS (the knee's middle).
    pub threshold: f32,
    /// The ratio above it, 1 (none) and up.
    pub ratio: f32,
    /// The knee's width, dB (0: a hard knee).
    pub knee: f32,
    /// How quickly it pulls down, seconds.
    pub attack: f32,
    /// How quickly it lets go, seconds.
    pub release: f32,
    /// The gain after it, dB.
    pub makeup: f32,
}

impl Default for CompressorSettings {
    fn default() -> CompressorSettings {
        CompressorSettings {
            threshold: -12.0,
            ratio: 3.0,
            knee: 6.0,
            attack: 0.005,
            release: 0.12,
            makeup: 0.0,
        }
    }
}

/// What a smoothing stage keeps a sample to move by 1 - 1/e in `seconds`.
fn keep(seconds: f32, rate: f32) -> f32 {
    if seconds <= 0.0 {
        0.0
    } else {
        (-1.0 / (seconds * rate)).exp()
    }
}

/// A feed-forward compressor, linked across the two sides.
#[derive(Clone, Copy, Debug)]
pub struct Compressor {
    settings: CompressorSettings,
    attack: f32,
    release: f32,
    /// The release stage's and the attack stage's reduction, dB.
    held: f32,
    reduction: f32,
    rate: f32,
}

impl Compressor {
    /// One at `rate`, at its default settings, not reducing.
    pub fn new(rate: Rate) -> Compressor {
        let mut c = Compressor {
            settings: CompressorSettings::default(),
            attack: 0.0,
            release: 0.0,
            held: 0.0,
            reduction: 0.0,
            rate: rate.hz(),
        };
        c.set(CompressorSettings::default());
        c
    }

    /// Its settings now.
    pub fn settings(&self) -> CompressorSettings {
        self.settings
    }

    /// Takes `s` from the next sample on.
    pub fn set(&mut self, s: CompressorSettings) {
        self.settings = s;
        self.attack = keep(s.attack, self.rate);
        self.release = keep(s.release, self.rate);
    }

    /// The static curve: the level out for a steady level `db` in (dBFS),
    /// before the makeup.
    pub fn curve(&self, db: f32) -> f32 {
        let s = self.settings;
        let slope = 1.0 / s.ratio.max(1.0) - 1.0;
        let over = db - s.threshold;
        let w = s.knee.max(0.0);
        if 2.0 * over <= -w {
            db
        } else if 2.0 * over < w {
            let x = over + 0.5 * w;
            db + slope * x * x / (2.0 * w)
        } else {
            db + slope * over
        }
    }

    /// Its gain reduction now, dB (0 or more).
    pub fn reduction(&self) -> f32 {
        self.reduction
    }

    /// The block `left` and `right` through, in place.
    pub fn process(&mut self, left: &mut [f32; BLOCK], right: &mut [f32; BLOCK]) {
        let makeup = self.settings.makeup;
        let (a, r) = (self.attack, self.release);
        for (l, rr) in left.iter_mut().zip(right.iter_mut()) {
            let level = 20.0 * l.abs().max(rr.abs()).max(1e-6).log10();
            let want = level - self.curve(level);
            self.held = want.max(r * self.held + (1.0 - r) * want);
            self.reduction = a * self.reduction + (1.0 - a) * self.held;
            let gain = 10f32.powf((makeup - self.reduction) / 20.0);
            *l *= gain;
            *rr *= gain;
        }
    }
}

/// The limiter's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimiterSettings {
    /// The most any sample comes out at, dBFS.
    pub ceiling: f32,
    /// How quickly the gain comes back up, seconds.
    pub release: f32,
}

impl Default for LimiterSettings {
    fn default() -> LimiterSettings {
        LimiterSettings {
            ceiling: -1.0,
            release: 0.08,
        }
    }
}

/// How far the limiter looks ahead, seconds.
pub const LOOKAHEAD: f32 = 0.0015;

/// The size of its rings: more than the look-ahead and the detection's
/// two samples at the highest rate (288 and 2 at 192 kHz).
const SPAN: usize = 512;

/// A look-ahead peak limiter, linked across the two sides.
#[derive(Clone, Debug)]
pub struct Limiter {
    settings: LimiterSettings,
    ceiling: f32,
    release: f32,
    /// The look-ahead, samples.
    look: usize,
    /// The audio held back, a ring a side, written at `n`.
    audio: [[f32; SPAN]; 2],
    /// The last four samples in, a side, oldest first, and the louder
    /// midpoint between the two oldest of the last step.
    recent: [[f32; 4]; 2],
    mid: f32,
    /// The sliding minimum: its gains and the step each came in on, a
    /// ring from `head`, `len` of them, rising from the front.
    min_gain: [f32; SPAN],
    min_at: [u64; SPAN],
    head: usize,
    len: usize,
    /// The released gain, the ring of the last `look + 1` of them and
    /// their sum.
    gain: f32,
    gains: [f32; SPAN],
    sum: f64,
    /// The step.
    n: u64,
    rate: f32,
}

impl Limiter {
    /// One at `rate`, at its default settings, silent.
    pub fn new(rate: Rate) -> Limiter {
        let look = ((LOOKAHEAD * rate.hz()).round() as usize).max(1);
        let mut l = Limiter {
            settings: LimiterSettings::default(),
            ceiling: 1.0,
            release: 0.0,
            look,
            audio: [[0.0; SPAN]; 2],
            recent: [[0.0; 4]; 2],
            mid: 0.0,
            min_gain: [0.0; SPAN],
            min_at: [0; SPAN],
            head: 0,
            len: 0,
            gain: 1.0,
            gains: [1.0; SPAN],
            sum: (look + 1) as f64,
            n: 0,
            rate: rate.hz(),
        };
        l.set(LimiterSettings::default());
        l
    }

    /// Its settings now.
    pub fn settings(&self) -> LimiterSettings {
        self.settings
    }

    /// Takes `s` from the next sample on.
    pub fn set(&mut self, s: LimiterSettings) {
        self.settings = s;
        self.ceiling = 10f32.powf(s.ceiling.min(0.0) / 20.0);
        self.release = keep(s.release, self.rate);
    }

    /// How late its output is, samples: the look-ahead and two.
    pub fn latency(&self) -> usize {
        self.look + 2
    }

    /// Its gain now (1: not limiting).
    pub fn gain(&self) -> f32 {
        (self.sum / (self.look + 1) as f64).min(1.0) as f32
    }

    /// The block `left` and `right` through, in place, [`Limiter::latency`]
    /// samples late.
    pub fn process(&mut self, left: &mut [f32; BLOCK], right: &mut [f32; BLOCK]) {
        let span = self.look + 1;
        let delay = self.latency();
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let at = self.n as usize % SPAN;
            self.audio[0][at] = *l;
            self.audio[1][at] = *r;
            // The peak of the sample two back: itself and its midpoints.
            let mut sample = 0.0f32;
            let mut mid = 0.0f32;
            for (side, x) in self.recent.iter_mut().zip([*l, *r]) {
                side.copy_within(1.., 0);
                side[3] = x;
                let [a, b, c, d] = *side;
                mid = mid.max(((9.0 * (b + c) - (a + d)) * (1.0 / 16.0)).abs());
                sample = sample.max(b.abs());
            }
            let peak = sample.max(mid).max(self.mid);
            self.mid = mid;
            let need = if peak > self.ceiling {
                self.ceiling / peak
            } else {
                1.0
            };
            // The least need over the look-ahead.
            while self.len > 0 && self.min_gain[(self.head + self.len - 1) % SPAN] >= need {
                self.len -= 1;
            }
            let back = (self.head + self.len) % SPAN;
            self.min_gain[back] = need;
            self.min_at[back] = self.n;
            self.len += 1;
            while self.min_at[self.head] + (span as u64) <= self.n {
                self.head = (self.head + 1) % SPAN;
                self.len -= 1;
            }
            let least = self.min_gain[self.head];
            // Back up over the release, never above the least.
            self.gain = least.min(self.gain + (1.0 - self.gain) * (1.0 - self.release));
            let slot = self.n as usize % span;
            self.sum += f64::from(self.gain) - f64::from(self.gains[slot]);
            self.gains[slot] = self.gain;
            let g = (self.sum / span as f64).min(1.0) as f32;
            let out = (self.n as usize + SPAN - delay) % SPAN;
            let c = self.ceiling;
            *l = (self.audio[0][out] * g).clamp(-c, c);
            *r = (self.audio[1][out] * g).clamp(-c, c);
            self.n += 1;
        }
    }
}
