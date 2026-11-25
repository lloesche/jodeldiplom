//! A stereo chorus and ensemble: each side's signal into a delay line read
//! by up to [`TAPS`] taps, each tap's delay swept around a centre by a
//! slow LFO (a sine or a triangle), the taps summed over the square root
//! of their count (near-uncorrelated copies, so their power is the input's
//! and a chorused pad keeps its level) and mixed with the dry signal. The taps' LFOs are spread evenly over a cycle (three a
//! third of a cycle apart: an ensemble, whose sweeps cancel each other's
//! pitch wobble), and the right side's are set off from the left's by up
//! to half the spacing (one tap: half a cycle), so the two sides move
//! apart and the sound widens.
//!
//! A tap reads between samples by linear interpolation, which keeps a
//! moving read smooth. A change of the centre delay or the depth glides
//! there over about 10 ms, so it does not jump the read point (a click).
//! The lines are made once, at the rate; nothing allocates as it runs. At
//! a mix of 0 the output is the input, bit for bit (the lines are still
//! fed, so raising the mix starts from a full line).
//!
//! It is a stand-alone processor on a stereo signal: [`Chorus::frame`]
//! takes a frame, [`Chorus::process`] a block in place.
//!
//! Sources: J. Dattorro, "Effect Design, Part 2: Delay-Line Modulation and
//! Chorus" (J. Audio Eng. Soc. 45(10), 1997): the chorus as a delay line
//! whose tap a low-frequency oscillator sweeps around a nominal delay of a
//! few milliseconds, mixed with the dry signal, and the interpolated read
//! of a modulated tap. The ensemble's evenly spread phases and the stereo
//! offset are our own.

use crate::dsp::{self, Rate};
use crate::BLOCK;

/// The taps a side, at the most.
pub const TAPS: usize = 3;

/// The longest centre delay, ms.
pub const DELAY_MAX: f32 = 25.0;

/// The shortest delay a tap reads, ms (the depth is held so the sweep
/// never goes below it).
pub const DELAY_MIN: f32 = 0.5;

/// The LFOs' wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// A sine: the sweep slows at its ends.
    Sine,
    /// A triangle: the sweep's speed (so the pitch's offset) is constant
    /// between its turns.
    Triangle,
}

/// The chorus's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChorusSettings {
    /// The LFOs' rate, Hz.
    pub rate: f32,
    /// How far a tap's delay sweeps either way, ms (held between 0 and
    /// the centre less [`DELAY_MIN`]).
    pub depth: f32,
    /// The centre delay, ms (1 to [`DELAY_MAX`]).
    pub delay: f32,
    /// The wet signal's share of the output, 0 (dry) to 1 (wet alone: a
    /// vibrato with one tap).
    pub mix: f32,
    /// How far the right side's LFOs are set off from the left's, 0 (the
    /// same: a mono chorus) to 1 (half the taps' spacing).
    pub spread: f32,
    /// The LFOs' wave.
    pub wave: Wave,
    /// The taps a side, 1 to [`TAPS`].
    pub taps: usize,
}

impl Default for ChorusSettings {
    /// A lush three-tap ensemble, half wet.
    fn default() -> ChorusSettings {
        ChorusSettings {
            rate: 0.6,
            depth: 2.0,
            delay: 12.0,
            mix: 0.5,
            spread: 1.0,
            wave: Wave::Sine,
            taps: TAPS,
        }
    }
}

/// The stereo chorus.
#[derive(Clone, Debug)]
pub struct Chorus {
    /// The lines, a power of two long, and where the next frame goes.
    left: Vec<f32>,
    right: Vec<f32>,
    at: usize,
    /// The LFOs' shared phase, its step a sample, and each tap's offset
    /// from it, a side.
    phase: u32,
    inc: u32,
    offsets: [[u32; TAPS]; 2],
    /// The centre delay and the depth now and where they glide, samples.
    delay: f32,
    depth: f32,
    target: (f32, f32),
    /// The glide's share a sample.
    glide: f32,
    settings: ChorusSettings,
    rate: Rate,
}

impl Chorus {
    /// A silent one at `rate`, at [`ChorusSettings::default`].
    pub fn new(rate: Rate) -> Chorus {
        let longest = ((DELAY_MAX * 2.0) * 0.001 * rate.hz()) as usize + 2;
        let len = longest.next_power_of_two();
        let mut c = Chorus {
            left: vec![0.0; len],
            right: vec![0.0; len],
            at: 0,
            phase: 0,
            inc: 0,
            offsets: [[0; TAPS]; 2],
            delay: 0.0,
            depth: 0.0,
            target: (0.0, 0.0),
            glide: rate.pole(16.0),
            settings: ChorusSettings::default(),
            rate,
        };
        c.set(ChorusSettings::default());
        (c.delay, c.depth) = c.target;
        c
    }

    /// Its settings now.
    pub fn settings(&self) -> ChorusSettings {
        self.settings
    }

    /// Takes `s` from the next frame on, the delay and the depth gliding
    /// there.
    pub fn set(&mut self, s: ChorusSettings) {
        self.settings = s;
        let taps = s.taps.clamp(1, TAPS);
        let ms = self.rate.hz() * 0.001;
        let delay = s.delay.clamp(1.0, DELAY_MAX);
        let depth = s.depth.clamp(0.0, delay - DELAY_MIN);
        self.target = (delay * ms, depth * ms);
        self.inc = self.rate.inc_hz(s.rate);
        let spacing = 1.0 / taps as f32;
        let apart = s.spread.clamp(0.0, 1.0) * 0.5 * spacing;
        for k in 0..TAPS {
            let at = k as f32 * spacing;
            self.offsets[0][k] = dsp::turns(at);
            self.offsets[1][k] = dsp::turns(at + apart);
        }
    }

    /// One frame `l`, `r` through; the frame out.
    #[inline]
    pub fn frame(&mut self, l: f32, r: f32) -> (f32, f32) {
        let mask = self.left.len() - 1;
        self.left[self.at] = l;
        self.right[self.at] = r;
        self.delay += (self.target.0 - self.delay) * self.glide;
        self.depth += (self.target.1 - self.depth) * self.glide;
        let s = self.settings;
        let out = if s.mix == 0.0 {
            (l, r)
        } else {
            let taps = s.taps.clamp(1, TAPS);
            let mix = s.mix.clamp(0.0, 1.0);
            let wl = self.wet(0, taps);
            let wr = self.wet(1, taps);
            (l + (wl - l) * mix, r + (wr - r) * mix)
        };
        self.phase = self.phase.wrapping_add(self.inc);
        self.at = (self.at + 1) & mask;
        out
    }

    /// A block of `left` and `right` through, in place.
    pub fn process(&mut self, left: &mut [f32; BLOCK], right: &mut [f32; BLOCK]) {
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            (*l, *r) = self.frame(*l, *r);
        }
    }

    /// Side `side`'s first `taps` taps now, summed over the square root of
    /// their count.
    #[inline]
    fn wet(&self, side: usize, taps: usize) -> f32 {
        let line = if side == 0 { &self.left } else { &self.right };
        let mask = line.len() - 1;
        let mut sum = 0.0;
        for &o in &self.offsets[side][..taps] {
            let ph = self.phase.wrapping_add(o);
            let lfo = match self.settings.wave {
                Wave::Sine => dsp::sine(ph),
                Wave::Triangle => dsp::triangle(ph),
            };
            let d = self.delay + self.depth * lfo;
            let back = d as usize;
            let frac = d - back as f32;
            let a = line[self.at.wrapping_sub(back) & mask];
            let b = line[self.at.wrapping_sub(back + 1) & mask];
            sum += a + (b - a) * frac;
        }
        sum / (taps as f32).sqrt()
    }
}
