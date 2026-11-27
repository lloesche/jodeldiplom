//! STRING: a plucked string by extended Karplus-Strong, for harps, plucks
//! and kotos.
//!
//! A delay line one period long holds the string's travelling wave. At a
//! pluck one period of noise is added into it, low-passed (a softer, darker
//! pluck) and combed by the pluck position (the harmonics with a node there
//! left out). Each pass around the loop goes through a first-order
//! all-pass for the fractional part of the period (in tune at any pitch,
//! without linear interpolation's loss), a one-pole low-pass (the losses
//! rising with frequency), four first-order all-passes with a negative
//! coefficient (stiffness: the high harmonics going sharp, as in a piano or
//! a thick wire), and a gain that sets the fundamental's ring time exactly,
//! net of the low-pass's loss at the pitch. The delay line is shortened by
//! the filters' exact phase delay at the pitch, so the string stays in tune
//! whatever its brightness and stiffness.
//!
//! What the host's [`VoiceMod`] does here: `phase_inc` sets the pitch, from
//! the rate over [`LINE`] - 8 up to a sixth of the rate; `shape` moves the
//! stiffness ([`SHAPE`]); `bright` lifts the brightness ([`BRIGHT`]). The
//! velocity brightens the pluck; the level is the host's.
//!
//! The delay lines are long (16 KB each), so they are not in the voice: the
//! engine keeps a pool of [`LINES`] of them, made once, and a voice borrows
//! one at its first pluck and gives it back when the host lets it go. A
//! pluck with none free is silent (more than [`LINES`] strings at once).
//!
//! Sources: Karplus and Strong (1983, "Digital synthesis of plucked-string
//! and drum timbres", Computer Music Journal 7(2)); Jaffe and Smith (1983,
//! "Extensions of the Karplus-Strong plucked-string algorithm", same issue)
//! for the all-pass tuning, the decay-time and brightness control, the
//! pluck-position comb and the dynamics low-pass on the pluck; Van Duyne
//! and Smith (1994, "A simplified approach to modeling dispersion caused by
//! stiffness in strings and plates", ICMC) for the stiffness all-passes.

use crate::dsp::{self, Quiet, Rate};
use crate::params::{field, Param};
use crate::{Engine, NoteOn, VoiceMod, BLOCK};
use std::cell::RefCell;
use std::f32::consts::{PI, TAU};

/// The strings that can sound at once (each with a delay line of its own).
pub const LINES: usize = 32;

/// The string's delay line, samples (a power of two): the lowest pitch is
/// 11.7 Hz at 48 kHz, 23.5 Hz at 96 kHz (lower ones play there).
pub const LINE: usize = 4096;
const MASK: usize = LINE - 1;

/// The stiffness all-passes.
const STAGES: usize = 4;

/// The highest loop gain: just below 1, so nothing in the loop can grow.
const LOOP: f32 = 0.99999;

/// The output gain: a pluck peaks at about 0.6 to 1.6 before the voice
/// level.
const OUTPUT: f32 = 0.7;

/// STRING's parameters, each 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// How long the fundamental rings to -60 dB: 0.05 s x 400^decay, so 0
    /// is 50 ms, 0.5 is 1 s, 1 is 20 s.
    pub decay: f32,
    /// The string's and the pluck's brightness: the loop low-pass's corner
    /// 2^(1 + 6 brightness) times the pitch, the pluck's a little higher. A
    /// long decay raises the loop's corner as far as it must for the
    /// fundamental to ring that long.
    pub brightness: f32,
    /// Where it is plucked, a share of its length from 0.02 (at the
    /// bridge: thin and bright) to 0.5 (the middle: round, the even
    /// harmonics gone). Read at each pluck.
    pub position: f32,
    /// Its stiffness: 0 a plain (harmonic) string, 1 a stiff one (fading
    /// out above about 600 Hz, gone by 3 kHz).
    pub stiffness: f32,
}

impl Default for Params {
    fn default() -> Params {
        PLUCK
    }
}

/// STRING's parameters as data, with their ranges.
pub const PARAMS: &[Param<Params>] = &[
    field!("string", "decay", decay, 0.0, 1.0, "", Linear),
    field!("string", "brightness", brightness, 0.0, 1.0, "", Linear),
    field!("string", "position", position, 0.0, 1.0, "", Linear),
    field!("string", "stiffness", stiffness, 0.0, 1.0, "", Linear),
];

/// A bright pluck, a second's ring.
pub const PLUCK: Params = Params {
    decay: 0.6,
    brightness: 0.55,
    position: 0.25,
    stiffness: 0.0,
};

/// A harp: round, long, a little stiff.
pub const HARP: Params = Params {
    decay: 0.75,
    brightness: 0.45,
    position: 0.4,
    stiffness: 0.15,
};

/// The presets, by name.
pub const PRESETS: [(&str, Params); 2] = [("PLUCK", PLUCK), ("HARP", HARP)];

/// What the host's shape moves: the stiffness, by up to half its range.
pub const SHAPE: &str = "stiffness";

/// What the host's brightness opens: the string's brightness, a quarter of
/// its range an octave.
pub const BRIGHT: &str = "brightness";

/// A line of the pool, lent to one voice at a time: it cannot be copied,
/// so no two voices ever hold it, and it goes back to the pool at most
/// once.
#[derive(Debug)]
struct Line(u16);

impl Line {
    fn index(&self) -> usize {
        usize::from(self.0)
    }
}

/// One voice: where its string is (the line it borrowed) and its loop
/// filters. It owns its line, so it is never copied.
#[derive(Debug, Default)]
pub struct Voice {
    /// The line borrowed from the pool (none: not plucked yet, or none was
    /// free).
    line: Option<Line>,
    /// Where the next sample is written.
    write: usize,
    /// The tuning all-pass's last input and output.
    tune_x: f32,
    tune_y: f32,
    /// The loss low-pass's output.
    loss: f32,
    /// The stiffness all-passes' last inputs and outputs.
    stiff_x: [f32; STAGES],
    stiff_y: [f32; STAGES],
    /// The output's DC blocker (outside the loop, so not in the tuning).
    dc_x: f32,
    dc_y: f32,
    quiet: Quiet,
}

/// The delay lines, and those free: a stack of fixed size, as the lines
/// lent are owned (each comes back once, so it never overflows).
struct Pool {
    lines: Box<[[f32; LINE]]>,
    free: [u16; LINES],
    count: usize,
}

impl Pool {
    /// A line to lend, if one is free.
    fn take(&mut self) -> Option<Line> {
        self.count = self.count.checked_sub(1)?;
        Some(Line(self.free[self.count]))
    }

    /// `line` given back.
    fn give(&mut self, line: Line) {
        debug_assert!(self.count < LINES, "a line given back twice");
        if let Some(slot) = self.free.get_mut(self.count) {
            *slot = line.0;
            self.count += 1;
        }
    }
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pool {{ {} of {} lines free }}",
            self.count,
            self.lines.len()
        )
    }
}

/// STRING at one rate, with its pool of lines.
#[derive(Debug)]
pub struct KarplusString {
    rate: Rate,
    /// The output DC blocker's pole: a corner of about 10 Hz.
    dc_pole: f32,
    pool: RefCell<Pool>,
}

/// The phase delay, samples, of the first-order all-pass
/// (c + z^-1) / (1 + c z^-1) at `w` radians a sample.
fn allpass_delay(c: f32, w: f32) -> f32 {
    let (s, co) = w.sin_cos();
    let phase = (-s).atan2(c + co) - (-c * s).atan2(1.0 + c * co);
    -phase / w
}

/// One-pole noise, low-passed by `a` (y += (1 - a) (x - y)), after `wait`
/// samples of silence.
struct Burst {
    rng: u32,
    a: f32,
    y: f32,
    wait: usize,
}

impl Burst {
    fn next(&mut self) -> f32 {
        if self.wait > 0 {
            self.wait -= 1;
            return 0.0;
        }
        self.y += (1.0 - self.a) * (dsp::white(&mut self.rng) - self.y);
        self.y
    }
}

impl Engine for KarplusString {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> KarplusString {
        let mut free = [0; LINES];
        for (k, f) in free.iter_mut().enumerate() {
            *f = (LINES - 1 - k) as u16;
        }
        KarplusString {
            rate,
            dc_pole: 1.0 - TAU * 10.0 / rate.hz(),
            pool: RefCell::new(Pool {
                lines: vec![[0.0; LINE]; LINES].into_boxed_slice(),
                free,
                count: LINES,
            }),
        }
    }

    /// A pluck: into a line borrowed (and cleared) at the first, or added
    /// to what still rings there (a retrigger plucks the sounding string
    /// again).
    fn note_on(&self, p: &Params, v: &mut Voice, n: &NoteOn) {
        let mut pool = self.pool.borrow_mut();
        let k = match &v.line {
            Some(line) => line.index(),
            None => {
                // The first pluck: a line taken, and cleared.
                let Some(line) = pool.take() else { return };
                let k = line.index();
                pool.lines[k] = [0.0; LINE];
                v.line = Some(line);
                k
            }
        };
        let line = &mut pool.lines[k];
        let velocity = n.velocity.clamp(0.0, 1.0);
        let f0 = dsp::hz(n.note) / self.rate.hz();
        let period = (1.0 / f0).clamp(8.0, (LINE - 8) as f32).round() as usize;
        let f0 = 1.0 / period as f32;
        // The pluck's low-pass (Jaffe and Smith's dynamics filter): a
        // harder pluck is brighter.
        let octaves = 2.0 + 6.0 * p.brightness.clamp(0.0, 1.0) * (0.6 + 0.4 * velocity);
        let corner = (f0 * octaves.exp2()).min(0.45);
        let a = (-TAU * corner).exp();
        // The comb gives harmonic k a gain of 2 sin(pi k x): keep its mean
        // power over the harmonics the low-pass lets through at 1, and the
        // low-passed noise's RMS too ((1 - a) / (3 (1 + a)) its variance).
        let x = 0.02 + 0.48 * p.position.clamp(0.0, 1.0);
        let harmonics = ((corner / f0) as usize).clamp(1, 64);
        let power = (1..=harmonics)
            .map(|k| (2.0 * (PI * k as f32 * x).sin()).powi(2))
            .sum::<f32>()
            / harmonics as f32;
        let rms = ((1.0 - a) / (3.0 * (1.0 + a))).sqrt();
        let gain = 1.0 / (power.sqrt().max(0.1) * rms);
        // The pluck and the same pluck from the position's distance later,
        // subtracted: a comb with notches at the harmonics with a node at x.
        let burst = |wait| Burst {
            rng: dsp::seed(n.seed),
            a,
            y: 0.0,
            wait,
        };
        let (mut pluck, mut echo) = (burst(0), burst((x * period as f32).round() as usize));
        // Into the period about to be read, added to what rings there.
        for i in 0..period {
            let at = (v.write + LINE - period + i) & MASK;
            line[at] += gain * (pluck.next() - echo.next());
        }
        v.quiet.restart();
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let n = out.len().min(BLOCK);
        let Some(k) = v.line.as_ref().map(Line::index) else {
            return;
        };
        if v.quiet.done() {
            return;
        }
        let mut pool = self.pool.borrow_mut();
        let line = &mut pool.lines[k];
        let lowest = 1.0 / (LINE - 8) as f32;
        let f0 = (m.phase_inc as f32 * (1.0 / 4_294_967_296.0)).clamp(lowest, 1.0 / 6.0);
        let w = TAU * f0;
        let period = 1.0 / f0;

        // The loss a period for the ring time: -60 dB over ring f0 periods.
        let ring = 0.05 * 400f32.powf(p.decay.clamp(0.0, 1.0));
        let per_period = 10f32.powf(-3.0 * period / (ring * self.rate.hz()));
        // The loss low-pass, y += (1 - a) (x - y). Its gain at DC is 1, so
        // the loop gain (at most LOOP) cannot make up for more loss at the
        // pitch than the ring time allows: a dark string rung long gets the
        // corner it needs, solving (1 - a)^2 = h^2 (1 - 2a cos w + a^2).
        let brightness = (p.brightness + 0.25 * m.bright).clamp(0.0, 1.0);
        let corner = (f0 * (1.0 + 6.0 * brightness).exp2()).min(0.45);
        let (s, c) = w.sin_cos();
        let gain_at = |a: f32| (1.0 - a) / (1.0 - 2.0 * a * c + a * a).sqrt();
        let mut a = (-TAU * corner).exp();
        let h = (per_period / LOOP).min(1.0);
        if gain_at(a) < h {
            let (q, b) = (1.0 - h * h, 1.0 - h * h * c);
            a = if q > 0.0 {
                (b - (b * b - q * q).max(0.0).sqrt()) / q
            } else {
                0.0
            };
        }
        let loss_delay = (a * s).atan2(1.0 - a * c) / w;
        let gain = (per_period / gain_at(a)).min(LOOP);

        // Stiffness, fading out at high pitches where the all-passes' own
        // delay would not fit in a period.
        let fit = ((period - 16.0) / 64.0).clamp(0.0, 1.0);
        let stiff = -0.85 * (p.stiffness + 0.5 * m.shape).clamp(0.0, 1.0) * fit;
        let stiff_delay = STAGES as f32 * allpass_delay(stiff, w);

        // The rest of the period: whole samples, and a fraction of 0.1 to
        // 1.1 for the tuning all-pass (near 0 its pole would sit at -1).
        let rest = period - loss_delay - stiff_delay;
        let whole = (rest - 0.1).floor().max(1.0);
        let fraction = rest - whole;
        let tune = (1.0 - fraction) / (1.0 + fraction);
        let whole = whole as usize;

        let mut y = [0.0f32; BLOCK];
        for yi in &mut y[..n] {
            let read = line[(v.write + LINE - whole) & MASK];
            let tuned = tune * read + v.tune_x - tune * v.tune_y;
            v.tune_x = read;
            v.tune_y = tuned;
            v.loss += (1.0 - a) * (tuned - v.loss);
            let mut s = v.loss;
            for (x1, y1) in v.stiff_x.iter_mut().zip(&mut v.stiff_y) {
                let out = stiff * s + *x1 - stiff * *y1;
                *x1 = s;
                *y1 = out;
                s = out;
            }
            s *= gain;
            line[v.write] = s;
            v.write = (v.write + 1) & MASK;
            v.dc_y = s - v.dc_x + self.dc_pole * v.dc_y;
            v.dc_x = s;
            *yi = v.dc_y * OUTPUT;
        }
        v.tune_y = dsp::flush(v.tune_y);
        v.loss = dsp::flush(v.loss);
        v.stiff_y = v.stiff_y.map(dsp::flush);
        v.dc_y = dsp::flush(v.dc_y);
        v.quiet.update(&y[..n], false);
        for (o, s) in out[..n].iter_mut().zip(&y[..n]) {
            *o += s;
        }
    }

    fn done(&self, v: &Voice) -> bool {
        v.line.is_none() || v.quiet.done()
    }

    fn free(&self, v: &mut Voice) {
        if let Some(k) = v.line.take() {
            self.pool.borrow_mut().give(k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every voice plucked, more than there are lines: each line lent to
    /// one voice only, the rest silent; each given back once however often
    /// its voice is let go, and all of them lent again after.
    #[test]
    fn a_line_is_lent_to_one_voice_and_comes_back_once() {
        let e = KarplusString::new(Rate::new(48_000));
        let n = NoteOn {
            note: 60.0,
            velocity: 1.0,
            seed: 1,
            pan: 0.0,
            shape: 0.0,
            glide: None,
            line: 0,
        };
        let mut voices: Vec<Voice> = (0..LINES + 3).map(|_| Voice::default()).collect();
        for v in &mut voices {
            e.note_on(&PLUCK, v, &n);
        }
        let mut lent: Vec<usize> = voices
            .iter()
            .filter_map(|v| v.line.as_ref().map(Line::index))
            .collect();
        assert_eq!(lent.len(), LINES, "the rest silent");
        lent.sort_unstable();
        lent.dedup();
        assert_eq!(lent.len(), LINES, "no line lent twice");
        assert_eq!(e.pool.borrow().count, 0);
        for v in &mut voices {
            e.free(v);
            e.free(v);
        }
        assert_eq!(e.pool.borrow().count, LINES);
        for v in &mut voices[..LINES] {
            e.note_on(&PLUCK, v, &n);
        }
        assert!(voices[..LINES].iter().all(|v| v.line.is_some()));
        assert_eq!(e.pool.borrow().count, 0);
    }
}
