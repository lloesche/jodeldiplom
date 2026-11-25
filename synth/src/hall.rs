//! A stereo hall: a feedback delay network, a mono send in and a stereo
//! tail out, for the host's reverb send. Its lines are
//! made once, at the rate, for the largest size and pre-delay; nothing
//! allocates as it runs.
//!
//! The send waits out the pre-delay, then four all-passes in series
//! diffuse it (their lengths and gains Dattorro's input diffusers), and it
//! goes into eight delay lines, with a sign each. The lines' outputs are
//! mixed by an 8 x 8 Hadamard matrix (orthogonal, so lossless: every line
//! feeds every other with the same weight) back into their inputs. Each
//! line's length is drawn out of the size, a few prime-ish milliseconds
//! from about 30 to 100 apart, so their echoes never line up; each is read
//! a little nearer and further on a slow sine of its own, so the modes of
//! the network wander and the tail does not ring metallic.
//!
//! The decay is a reverberation time, to -60 dB: each line loses, a pass,
//! what its length's share of that time takes, by a one-pole low-pass
//! whose gain at DC sets the time there and whose gain at the Nyquist sets
//! a shorter one up high (the damping), so every line decays at the same
//! rate whatever its length, and the tail darkens as it fades. The left
//! and right sides read the lines through two orthogonal rows of signs:
//! the two sides are uncorrelated.
//!
//! Sources: the feedback delay network and its per-line absorption for a
//! frequency-dependent reverberation time after J.-M. Jot and A. Chaigne,
//! "Digital Delay Networks for Designing Artificial Reverberators" (AES
//! 90th Convention, Paris, 1991, preprint 3030); the input diffusers and
//! the modulated lines after J. Dattorro, "Effect Design, Part 1:
//! Reverberator and Other Filters" (J. Audio Eng. Soc. 45(9), 1997); the
//! Hadamard feedback matrix as J. O. Smith III, "Physical Audio Signal
//! Processing" (W3K, 2010), presents FDNs. The absorption filter's two
//! gains are our own: a one-pole y = a x + b y' has the gain a / (1 - b) at
//! DC and a / (1 + b) at the Nyquist, which gives a and b for any two.

use crate::dsp::{self, flush, Rate};
use crate::BLOCK;

/// The hall's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HallSettings {
    /// The hall's size, 0 to 1: its lines from 0.3 to 1 times their full
    /// length.
    pub size: f32,
    /// The reverberation time at low frequencies, seconds to -60 dB (0.1
    /// to 30).
    pub decay: f32,
    /// How much shorter it is up high, 0 (not at all) to 1 (a seventh as
    /// long at the Nyquist).
    pub damp: f32,
    /// The wait before the tail starts, seconds (up to [`PREDELAY_MAX`]).
    pub predelay: f32,
    /// How loud the tail is, 0 (silent) to 1.
    pub wet: f32,
    /// The stereo width, 0 (mono) to 1.
    pub width: f32,
}

impl Default for HallSettings {
    fn default() -> HallSettings {
        HallSettings {
            size: 0.7,
            decay: 2.5,
            damp: 0.4,
            predelay: 0.02,
            wet: 0.3,
            width: 1.0,
        }
    }
}

/// The longest pre-delay, seconds.
pub const PREDELAY_MAX: f32 = 0.25;

/// The lines.
const LINES: usize = 8;

/// Their lengths at full size, milliseconds.
const LENGTHS: [f32; LINES] = [31.7, 39.1, 46.3, 53.9, 61.3, 71.9, 83.3, 97.1];

/// How far each is read nearer and further, seconds, and how often, Hz.
const WANDER: f32 = 0.0004;
const WANDER_HZ: [f32; LINES] = [0.13, 0.19, 0.23, 0.29, 0.37, 0.41, 0.47, 0.53];

/// Dattorro's input diffusers: lengths in samples at 29 761 Hz, and gains.
const DIFFUSERS: [(f32, f32); 4] = [(142.0, 0.75), (107.0, 0.75), (379.0, 0.625), (277.0, 0.625)];
const DATTORRO_RATE: f32 = 29_761.0;

/// The signs the send goes into the lines with, and the two rows of the
/// Hadamard matrix the sides read them through.
const INTO: [f32; LINES] = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0];
const LEFT: [f32; LINES] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
const RIGHT: [f32; LINES] = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

/// The send's level into the network and the tail's out of it.
const INPUT: f32 = 0.35;
const OUTPUT: f32 = 0.6;

/// A delay line of a power of two, read anywhere behind the point it is
/// written at.
#[derive(Clone, Debug)]
struct Ring {
    buf: Vec<f32>,
    mask: usize,
}

impl Ring {
    /// One holding at least `len` samples back.
    fn new(len: usize) -> Ring {
        let size = (len + 2).next_power_of_two();
        Ring {
            buf: vec![0.0; size],
            mask: size - 1,
        }
    }

    /// What was written `back` samples before `at` (`back` at least 1).
    #[inline]
    fn at(&self, at: usize, back: usize) -> f32 {
        self.buf[at.wrapping_sub(back) & self.mask]
    }

    /// The same `back` samples ago, fractional, between its neighbours.
    #[inline]
    fn read(&self, at: usize, back: f32) -> f32 {
        let whole = back as usize;
        let frac = back - whole as f32;
        let a = self.at(at, whole);
        a + (self.at(at, whole + 1) - a) * frac
    }

    #[inline]
    fn write(&mut self, at: usize, x: f32) {
        self.buf[at & self.mask] = x;
    }
}

/// One line of the network: its ring, its length now, its wander, and its
/// absorption (a one-pole: `a` of the input, `b` of its last output).
#[derive(Clone, Debug)]
struct Line {
    ring: Ring,
    len: f32,
    lfo: u32,
    lfo_inc: u32,
    a: f32,
    b: f32,
    last: f32,
}

/// A Schroeder all-pass of its own ring.
#[derive(Clone, Debug)]
struct Diffuser {
    ring: Ring,
    len: usize,
    gain: f32,
}

/// The hall: a mono send in, a stereo tail out.
#[derive(Clone, Debug)]
pub struct Hall {
    lines: Vec<Line>,
    diffusers: Vec<Diffuser>,
    pre: Ring,
    /// The pre-delay now, samples.
    wait: usize,
    /// Where the rings are written.
    at: usize,
    /// The wander's depth, samples.
    wander: f32,
    rate: f32,
    settings: HallSettings,
}

impl Hall {
    /// A silent one at `rate`, at its default settings.
    pub fn new(rate: Rate) -> Hall {
        let hz = rate.hz();
        let wander = WANDER * hz;
        let lines = LENGTHS
            .iter()
            .zip(WANDER_HZ)
            .enumerate()
            .map(|(k, (&ms, lfo_hz))| Line {
                ring: Ring::new((ms * 0.001 * hz + 2.0 * wander) as usize + 2),
                len: 0.0,
                lfo: (k as u32).wrapping_mul(0x2000_0000),
                lfo_inc: rate.inc_hz(lfo_hz),
                a: 0.0,
                b: 0.0,
                last: 0.0,
            })
            .collect();
        let diffusers = DIFFUSERS
            .iter()
            .map(|&(n, gain)| {
                let len = ((n * hz / DATTORRO_RATE).round() as usize).max(1);
                Diffuser {
                    ring: Ring::new(len),
                    len,
                    gain,
                }
            })
            .collect();
        let mut h = Hall {
            lines,
            diffusers,
            pre: Ring::new((PREDELAY_MAX * hz) as usize + 1),
            wait: 0,
            at: 0,
            wander,
            rate: hz,
            settings: HallSettings::default(),
        };
        h.set(HallSettings::default());
        h
    }

    /// Its settings now.
    pub fn settings(&self) -> HallSettings {
        self.settings
    }

    /// Takes `s` from the next sample on (a new size moves the lines'
    /// lengths at once: best changed while it is quiet).
    pub fn set(&mut self, s: HallSettings) {
        self.settings = s;
        let scale = 0.3 + 0.7 * s.size.clamp(0.0, 1.0);
        let decay = s.decay.clamp(0.1, 30.0);
        let high = decay * (1.0 - 0.857 * s.damp.clamp(0.0, 1.0));
        for (line, ms) in self.lines.iter_mut().zip(LENGTHS) {
            line.len = ms * 0.001 * self.rate * scale;
            // -60 dB over the time: each pass (its length on average, the
            // wander's middle) loses its share.
            let pass = line.len + self.wander;
            let g0 = 10f32.powf(-3.0 * pass / (decay * self.rate));
            let gpi = 10f32.powf(-3.0 * pass / (high * self.rate));
            line.b = (g0 - gpi) / (g0 + gpi);
            line.a = g0 * (1.0 - line.b);
        }
        self.wait = (s.predelay.clamp(0.0, PREDELAY_MAX) * self.rate) as usize;
    }

    /// The block's send `send` through, its tail added to `left` and
    /// `right`. At wet 0 they are left exactly as they were (the network
    /// still runs, so the tail is there when the wet comes back).
    pub fn process(
        &mut self,
        send: &[f32; BLOCK],
        left: &mut [f32; BLOCK],
        right: &mut [f32; BLOCK],
    ) {
        let s = self.settings;
        let wet = s.wet.max(0.0) * OUTPUT;
        let width = s.width.clamp(0.0, 1.0);
        let wet1 = wet * (0.5 + 0.5 * width);
        let wet2 = wet * (0.5 - 0.5 * width);
        for i in 0..BLOCK {
            let at = self.at;
            self.pre.write(at, send[i] * INPUT);
            let mut x = if self.wait == 0 {
                send[i] * INPUT
            } else {
                self.pre.at(at, self.wait)
            };
            for d in &mut self.diffusers {
                let delayed = d.ring.at(at, d.len);
                let w = x + d.gain * delayed;
                d.ring.write(at, flush(w));
                x = delayed - d.gain * w;
            }
            let mut y = [0.0f32; LINES];
            for (yk, line) in y.iter_mut().zip(&mut self.lines) {
                let back = line.len + self.wander * (1.0 + dsp::sine(line.lfo));
                line.lfo = line.lfo.wrapping_add(line.lfo_inc);
                line.last = flush(line.a * line.ring.read(at, back) + line.b * line.last);
                *yk = line.last;
            }
            let (mut l, mut r) = (0.0, 0.0);
            for k in 0..LINES {
                l += y[k] * LEFT[k];
                r += y[k] * RIGHT[k];
            }
            hadamard(&mut y);
            for ((line, yk), sign) in self.lines.iter_mut().zip(y).zip(INTO) {
                line.ring.write(at, yk + x * sign);
            }
            self.at = at.wrapping_add(1);
            if wet != 0.0 {
                left[i] += l * wet1 + r * wet2;
                right[i] += r * wet1 + l * wet2;
            }
        }
    }
}

/// `y` times the 8 x 8 Hadamard matrix, scaled to be orthogonal (1 /
/// sqrt 8), by three rounds of sums and differences.
#[inline]
fn hadamard(y: &mut [f32; LINES]) {
    let mut h = 1;
    while h < LINES {
        for i in (0..LINES).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (y[j], y[j + h]);
                y[j] = a + b;
                y[j + h] = a - b;
            }
        }
        h *= 2;
    }
    for v in y.iter_mut() {
        *v *= std::f32::consts::FRAC_1_SQRT_2 * 0.5;
    }
}
