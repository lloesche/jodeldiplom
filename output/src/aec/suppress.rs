//! What the canceller's filter leaves of the echo, taken down for the
//! pitch the game hears: a residual echo suppressor.
//!
//! The filter predicts the echo it can; what it cannot (a speaker's
//! distortion, a room's movement, the share a laptop's speakers play their
//! own way) stays in the voice, on a MacBook's own speakers as loud as the
//! singer. A pitch tracker needs one tone to stand out of each frame: under
//! a band as loud as the voice it finds none. The leftover has the echo's
//! shape, though, a share of it frequency by frequency: that share is
//! learnt where only the echo is heard (per bin, the leftover's power
//! against the estimate's, followed in the log so a voice the detector
//! missed moves it little), and each bin of the voice is then turned down
//! by what of it that share of the estimate accounts for (a gain of
//! √(1 − leftover/power), never under [`FLOOR`]). Where the singer is the
//! louder, the bin passes nearly whole; where the leftover is, it goes.
//!
//! It works on frames of two blocks under a √Hann window, overlapped by
//! one block (whole again when nothing is taken down): what comes out of a
//! block given is the block before it, which its caller keeps track of.
//! What it learnt goes with the filter it was learnt of: when the
//! canceller's filter is another ([`Echo::generation`]), it learns again.

use super::fft::{Fft, BINS, N};
use super::{Echo, BLOCK};

/// The least a bin is let through (−20 dB).
const FLOOR: f32 = 0.1;
/// How many times the leftover learnt is taken out: a little more than
/// once, as its share moves from moment to moment about what was learnt.
const OVER: f32 = 1.5;
/// How fast the share follows, per block where only the echo is heard
/// (about 1.5 s at 44.1 kHz).
const LEARN: f32 = 1.0 / 500.0;
/// A block's share counts at most this far over the share learnt (the
/// natural log of 10 dB): a voice the detector missed is far over it.
const LEAP: f32 = 2.3;
/// The share learnt stays within these (−30 and +6 dB, natural log).
const SHARE_LEAST: f32 = -6.9;
const SHARE_MOST: f32 = 1.4;
/// Blocks learnt from before anything is taken down (about 0.7 s).
const LEARN_FIRST: u32 = 250;
/// How long the leftover lingers after the estimate falls, in seconds: the
/// leftover is late against the estimate by moments.
const LINGER: f32 = 0.02;

/// The residual echo suppressor (part 3 of [`crate::aec`]).
pub struct Suppressor {
    fft: Fft,
    window: [f32; N],
    /// The last block of the canceller's voice and of its echo estimate.
    last_voice: [f32; BLOCK],
    last_echo: [f32; BLOCK],
    /// The newer half of the last frame, for the overlap-add.
    tail: [f32; BLOCK],
    /// Per bin, the natural log of the leftover's share of the estimate's
    /// power, how many blocks it was learnt from, and of which filter.
    share: [f32; BINS],
    learnt: u32,
    generation: u32,
    /// Per bin, the leftover expected now (it lingers, [`LINGER`]).
    leftover: [f32; BINS],
    linger: f32,
}

impl Suppressor {
    /// A suppressor for a canceller at `rate`.
    pub fn new(rate: u32) -> Suppressor {
        let window = std::array::from_fn(|i| (std::f32::consts::PI * i as f32 / N as f32).sin());
        let mut s = Suppressor {
            fft: Fft::new(),
            window,
            last_voice: [0.0; BLOCK],
            last_echo: [0.0; BLOCK],
            tail: [0.0; BLOCK],
            share: [0.0; BINS],
            learnt: 0,
            generation: 0,
            leftover: [0.0; BINS],
            linger: 0.0,
        };
        s.set_rate(rate);
        s.reset();
        s
    }

    /// The rate it runs at.
    pub fn set_rate(&mut self, rate: u32) {
        self.linger = (-(BLOCK as f32) / (rate.max(1) as f32 * LINGER)).exp();
    }

    /// Forgets what it learnt (a new output: a new echo's way).
    pub fn reset(&mut self) {
        self.restart();
        self.share = [SHARE_LEAST; BINS];
        self.learnt = 0;
    }

    /// The blocks start again after a gap: what comes out of the next one
    /// given is nothing's.
    pub fn restart(&mut self) {
        self.last_voice = [0.0; BLOCK];
        self.last_echo = [0.0; BLOCK];
        self.tail = [0.0; BLOCK];
        self.leftover = [0.0; BINS];
    }

    /// Takes the canceller's block: `voice` what it gave, `echo` what it
    /// subtracted ([`super::EchoCanceller::process`]). The block before,
    /// suppressed, goes to `out`.
    pub fn process(&mut self, voice: &[f32], echo: &Echo, out: &mut [f32]) {
        if echo.generation != self.generation {
            self.generation = echo.generation;
            self.share = [SHARE_LEAST; BINS];
            self.learnt = 0;
        }
        let voice: [f32; BLOCK] = std::array::from_fn(|i| voice[i]);
        let apply = echo.learnt && self.learnt >= LEARN_FIRST;
        self.frame(&voice, &echo.estimate, echo.echo_only, apply, out);
    }

    /// Takes a block of the voice with no canceller before it (it is off):
    /// passed whole, a block late like the others.
    pub fn pass(&mut self, voice: &[f32], out: &mut [f32]) {
        let voice: [f32; BLOCK] = std::array::from_fn(|i| voice[i]);
        self.frame(&voice, &[0.0; BLOCK], false, false, out);
    }

    fn frame(
        &mut self,
        voice: &[f32; BLOCK],
        echo: &[f32; BLOCK],
        learn: bool,
        apply: bool,
        out: &mut [f32],
    ) {
        let windowed = |a: &[f32; BLOCK], b: &[f32; BLOCK]| -> [f32; N] {
            std::array::from_fn(|i| {
                let v = if i < BLOCK { a[i] } else { b[i - BLOCK] };
                v * self.window[i]
            })
        };
        let e = windowed(&self.last_voice, voice);
        let y = windowed(&self.last_echo, echo);
        let mut bins = self.fft.real(&e[..BLOCK], &e[BLOCK..]);
        let estimate = self.fft.real(&y[..BLOCK], &y[BLOCK..]);
        let total: f32 = bins.iter().map(|c| c.norm()).sum::<f32>() / BINS as f32;
        let tiny = 1e-6 * total + 1e-20;
        if learn {
            for k in 0..BINS {
                let p = bins[k].norm() + tiny;
                let q = estimate[k].norm() + tiny;
                let r = (p / q).ln().min(self.share[k] + LEAP);
                self.share[k] =
                    (self.share[k] + (r - self.share[k]) * LEARN).clamp(SHARE_LEAST, SHARE_MOST);
            }
            self.learnt = self.learnt.saturating_add(1);
        }
        for k in 0..BINS {
            let now = self.share[k].exp() * estimate[k].norm();
            self.leftover[k] = now.max(self.leftover[k] * self.linger);
            if apply {
                let p = bins[k].norm() + tiny;
                let g = (1.0 - OVER * self.leftover[k] / p)
                    .max(0.0)
                    .sqrt()
                    .max(FLOOR);
                bins[k] = bins[k].scale(g);
            }
        }
        let back = self.fft.inverse(&bins);
        for i in 0..BLOCK {
            out[i] = self.tail[i] + back[i] * self.window[i];
            self.tail[i] = back[BLOCK + i] * self.window[BLOCK + i];
        }
        self.last_voice = *voice;
        self.last_echo = *echo;
    }
}
