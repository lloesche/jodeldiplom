//! The building blocks the engines share: a sine table read with
//! interpolation, band-limited saw and pulse oscillators on 32-bit
//! wrapping phases, a soft clip and a soft knee, xorshift noise, a chip's
//! noise register, Simper's trapezoidal state-variable filter (low-pass,
//! band-pass, high-pass and notch from one update), and a crush (a sample
//! hold and fewer bits).
//!
//! Signals are f32 with 1.0 at full scale. Phases are u32 and wrap, a
//! cycle being 2^32. The table is made at compile time.
//!
//! Sources: the polyBLEP residual from V. Välimäki and A. Huovilainen,
//! "Antialiasing Oscillators in Subtractive Synthesis" (IEEE Signal
//! Processing Magazine, 2007) and V. Välimäki, J. Pekonen and J. Nam,
//! "Perceptually informed synthesis of bandlimited classical waveforms
//! using integrated polynomial interpolation" (JASA, 2012); the filter from
//! A. Simper, "Linear Trapezoidal Integrated SVF" (Cytomic technical
//! paper, 2013); xorshift32 from G. Marsaglia, "Xorshift RNGs" (Journal of
//! Statistical Software, 2003); the noise register from the NES APU's
//! noise channel as the nesdev wiki documents it ("APU Noise"); the soft
//! clip a Padé approximant of tanh.

use crate::BLOCK;
use jodel_dsp::xorshift32;
use std::f64::consts::PI;

/// The lowest cutoff a filter takes, Hz.
pub const CUTOFF_MIN: f32 = 20.0;

/// The highest, Hz (and never above 0.45 of the rate).
pub const CUTOFF_MAX: f32 = 20_000.0;

/// One cycle of a sine in 1024 steps, and the first point again at the
/// end, so an interpolation never wraps.
pub static SINE: [f32; 1025] = sine_table();

/// sin(2 pi k / n) for `n` a multiple of 4, from its first quadrant (so
/// it is exactly symmetric), by the Taylor series.
pub(crate) const fn sin_turn(k: usize, n: usize) -> f64 {
    let j = k % n;
    let q = n / 4;
    let (m, sign) = if j <= q {
        (j, 1.0)
    } else if j <= 2 * q {
        (2 * q - j, 1.0)
    } else if j <= 3 * q {
        (j - 2 * q, -1.0)
    } else {
        (n - j, -1.0)
    };
    let x = 2.0 * PI * m as f64 / n as f64;
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut i = 1;
    while i < 14 {
        term = -term * x2 / ((2 * i) * (2 * i + 1)) as f64;
        sum += term;
        i += 1;
    }
    sign * sum
}

const fn sine_table() -> [f32; 1025] {
    let mut t = [0.0; 1025];
    let mut i = 0;
    while i < 1025 {
        t[i] = sin_turn(i, 1024) as f32;
        i += 1;
    }
    t
}

/// The sine at phase `ph` (2^32 a cycle), interpolated between the
/// table's points.
#[inline]
pub fn sine(ph: u32) -> f32 {
    let i = (ph >> 22) as usize;
    let f = (ph & 0x3F_FFFF) as f32 * (1.0 / 4_194_304.0);
    let a = SINE[i];
    a + (SINE[i + 1] - a) * f
}

/// A phase (0 to 1) as a u32 phase.
#[inline]
pub fn turns(x: f32) -> u32 {
    ((x - x.floor()) * 4_294_967_296.0) as u32
}

/// The two-sample polyBLEP residual of a unit downward step (a saw's
/// wrap, halved) for phase `ph` stepping by `inc`: on the sample after
/// the wrap and on the one before it.
#[inline]
pub fn blep(ph: u32, inc: u32) -> f32 {
    if inc == 0 {
        return 0.0;
    }
    if ph < inc {
        let t = ph as f32 / inc as f32;
        return t + t - t * t - 1.0;
    }
    if ph > u32::MAX - inc {
        let t = (u32::MAX - ph) as f32 / inc as f32;
        return (1.0 - t) * (1.0 - t);
    }
    0.0
}

/// A band-limited saw, -1 to 1, at phase `ph` stepping by `inc`.
#[inline]
pub fn saw(ph: u32, inc: u32) -> f32 {
    ph as f32 * (1.0 / 2_147_483_648.0) - 1.0 - blep(ph, inc)
}

/// A band-limited pulse, high for `width` of a cycle (2^31: a square),
/// the difference of two saws: a swing of 2 with no DC (a square from -1
/// to 1).
#[inline]
pub fn pulse(ph: u32, inc: u32, width: u32) -> f32 {
    saw(ph.wrapping_sub(width), inc) - saw(ph, inc)
}

/// A triangle, -1 at phase 0 and 1 half a cycle on (its corners are soft
/// enough to leave as they are).
#[inline]
pub fn triangle(ph: u32) -> f32 {
    let u = ph as f32 * (1.0 / 1_073_741_824.0);
    if u < 2.0 {
        u - 1.0
    } else {
        3.0 - u
    }
}

/// tanh(x), by its Padé approximant x (27 + x^2) / (27 + 9 x^2), held at
/// ±1 past ±3 (where the two meet).
#[inline]
pub fn softclip(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

/// `y` as it is up to `k`, above it rounded off towards 1: only the
/// peaks saturate.
#[inline]
pub fn soft_knee(y: f32, k: f32) -> f32 {
    let a = y.abs();
    if a <= k {
        return y;
    }
    (k + (1.0 - k) * softclip((a - k) / (1.0 - k))).copysign(y)
}

/// A [`xorshift32`] state from `seed`: mixed first (MurmurHash3's finaliser), so
/// small or neighbouring seeds start unalike; never 0, which xorshift
/// cannot leave.
pub fn seed(seed: u32) -> u32 {
    let mut x = seed ^ 0x6A09_E667;
    x = (x ^ (x >> 16)).wrapping_mul(0x85EB_CA6B);
    x = (x ^ (x >> 13)).wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    if x == 0 {
        0x9E37_79B9
    } else {
        x
    }
}

/// 0 to 1 from [`xorshift32`] state `s`.
#[inline]
pub fn unit(s: &mut u32) -> f32 {
    (xorshift32(s) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `x`, or 0 once it has decayed below 1e-15 (-300 dB): a state left to
/// decay on its own would reach denormal numbers, which cost a hundred
/// times as much to compute with.
#[inline]
pub fn flush(x: f32) -> f32 {
    if x.abs() < 1e-15 {
        0.0
    } else {
        x
    }
}

/// Counts the blocks a struck or plucked voice has been near silent, so
/// its engine can say it has rung out.
#[derive(Clone, Copy, Debug, Default)]
pub struct Quiet(u16);

impl Quiet {
    /// Below this peak (-80 dB) a block counts as silent.
    const FLOOR: f32 = 1e-4;
    /// This many silent blocks in a row (about 10 ms at 48 kHz) end it.
    const BLOCKS: u16 = 16;

    /// Struck again: not silent, whatever came before.
    pub fn restart(&mut self) {
        self.0 = 0;
    }

    /// Notes one rendered block (its peak before the amplitude ramp);
    /// `exciting`: something still feeds the voice.
    pub fn update(&mut self, block: &[f32], exciting: bool) {
        let peak = block.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        self.0 = if exciting || peak >= Self::FLOOR {
            0
        } else {
            self.0.saturating_add(1)
        };
    }

    /// Whether the voice has rung out.
    pub fn done(&self) -> bool {
        self.0 >= Self::BLOCKS
    }
}

/// A white noise sample, -1 to 1, from [`xorshift32`] state `s`.
#[inline]
pub fn white(s: &mut u32) -> f32 {
    (xorshift32(s) >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
}

/// The 15-bit noise register a step on: bit 0 fed back with bit 1 (the
/// long loop, 32767 steps) or bit 6 (`short`: 93 steps from the state 1),
/// into bit 14. Its output is bit 0.
#[inline]
pub fn chip_noise(reg: u32, short: bool) -> u32 {
    let fb = (reg ^ (reg >> if short { 6 } else { 1 })) & 1;
    (reg >> 1) | (fb << 14)
}

/// The register `reg` clocked for a sample: its clock's phase `phase`
/// (2^32 a step) moved on by `inc`, which may be many steps a sample (at
/// most 256 are taken).
#[inline]
pub fn clock_noise(reg: &mut u32, phase: &mut u32, inc: u64, short: bool) {
    let at = u64::from(*phase) + inc;
    *phase = at as u32;
    for _ in 0..(at >> 32).min(256) {
        *reg = chip_noise(*reg, short);
    }
}

/// The damping of a filter at resonance `res` (0 to 1): 2 (a Q of 0.5)
/// down to 0.05 (a Q of 20).
pub fn damping(res: f32) -> f32 {
    2.0 - res.clamp(0.0, 1.0) * 1.95
}

/// Simper's trapezoidal state-variable filter, its coefficients for one
/// block: stable at any cutoff and damping.
#[derive(Clone, Copy, Debug)]
pub struct Svf {
    a1: f32,
    a2: f32,
    a3: f32,
    k: f32,
}

/// The state of one [`Svf`]: its two integrators.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SvfState {
    ic1: f32,
    ic2: f32,
}

/// The outputs of one update of an [`Svf`].
#[derive(Clone, Copy, Debug)]
pub struct Taps {
    /// Low-pass.
    pub lp: f32,
    /// Band-pass (peaking at 1 / damping).
    pub bp: f32,
    /// High-pass.
    pub hp: f32,
    /// Notch.
    pub notch: f32,
}

impl Svf {
    /// The filter at `hz` (held between [`CUTOFF_MIN`] and
    /// [`CUTOFF_MAX`], and under 0.45 of the rate) with damping `k`.
    pub fn new(hz: f32, k: f32, rate: Rate) -> Svf {
        let f = hz.clamp(CUTOFF_MIN, CUTOFF_MAX).min(0.45 * rate.hz());
        let g = (std::f32::consts::PI * f / rate.hz()).tan();
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        Svf {
            a1,
            a2,
            a3: g * a2,
            k,
        }
    }

    /// Its damping.
    pub fn k(&self) -> f32 {
        self.k
    }

    /// One sample `x` through, the state in `s`.
    #[inline]
    pub fn run(&self, s: &mut SvfState, x: f32) -> Taps {
        let v3 = x - s.ic2;
        let v1 = self.a1 * s.ic1 + self.a2 * v3;
        let v2 = s.ic2 + self.a2 * s.ic1 + self.a3 * v3;
        s.ic1 = 2.0 * v1 - s.ic1;
        s.ic2 = 2.0 * v2 - s.ic2;
        let hp = x - self.k * v1 - v2;
        Taps {
            lp: v2,
            bp: v1,
            hp,
            notch: v2 + hp,
        }
    }
}

/// A crush: the signal held at a lower rate and rounded to fewer bits,
/// as a cheap converter does.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Crush {
    /// The hold's phase and the sample it holds.
    phase: u32,
    held: f32,
}

/// What a [`Crush`] does for a block: its hold's phase increment (0: no
/// hold) and the levels each side of 0 it rounds to (0: not rounded).
#[derive(Clone, Copy, Debug)]
pub struct CrushSetting {
    inc: u32,
    levels: f32,
}

impl CrushSetting {
    /// A hold at `hz` (0: none) and `bits` (0: none), at `rate`.
    pub fn new(hz: f32, bits: u8, rate: Rate) -> CrushSetting {
        CrushSetting {
            inc: rate.inc_hz(hz),
            levels: if bits > 0 {
                (1u32 << (bits.min(24) - 1)) as f32
            } else {
                0.0
            },
        }
    }
}

impl Crush {
    /// `x` crushed as `s` says.
    #[inline]
    pub fn run(&mut self, x: f32, s: &CrushSetting) -> f32 {
        let mut y = x;
        if s.inc > 0 {
            let old = self.phase;
            self.phase = self.phase.wrapping_add(s.inc);
            if self.phase < old {
                self.held = x;
            }
            y = self.held;
        }
        if s.levels > 0.0 {
            y = (y * s.levels).round() / s.levels;
        }
        y
    }
}
/// A sample rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    hz: f32,
}

impl Rate {
    /// The lowest rate the synth takes.
    pub const MIN: u32 = 32_000;
    /// The highest.
    pub const MAX: u32 = 192_000;

    /// The rate `hz`.
    ///
    /// # Panics
    ///
    /// If it is not from [`Rate::MIN`] to [`Rate::MAX`].
    pub fn new(hz: u32) -> Rate {
        assert!(
            (Rate::MIN..=Rate::MAX).contains(&hz),
            "a rate of {hz} Hz: the synth runs from {} to {} Hz",
            Rate::MIN,
            Rate::MAX
        );
        Rate { hz: hz as f32 }
    }

    /// Its samples a second.
    pub fn hz(self) -> f32 {
        self.hz
    }

    /// The phase increment a sample of `hz`, up to a whole cycle.
    pub fn inc_hz(self, hz: f32) -> u32 {
        (f64::from(hz.max(0.0)) / f64::from(self.hz) * 4_294_967_296.0).min(f64::from(u32::MAX))
            as u32
    }

    /// The same for a clock that may step many times a sample (for
    /// [`clock_noise`]).
    pub fn clock_inc(self, hz: f32) -> u64 {
        (f64::from(hz.max(0.0)) / f64::from(self.hz) * 4_294_967_296.0).min(256.0 * 4_294_967_296.0)
            as u64
    }

    /// The phase increment a sample of MIDI note `note`, held from 0 to
    /// 135 (19.9 kHz).
    pub fn inc(self, note: f32) -> u32 {
        self.inc_hz(hz(note.clamp(0.0, 135.0)))
    }

    /// The share a one-pole low-pass at `hz` moves towards its input a
    /// sample.
    pub fn pole(self, hz: f32) -> f32 {
        1.0 - (-std::f32::consts::TAU * hz.max(0.0) / self.hz).exp()
    }

    /// The same a block.
    pub fn block_pole(self, hz: f32) -> f32 {
        1.0 - (-std::f32::consts::TAU * hz.max(0.0) * BLOCK as f32 / self.hz).exp()
    }
}

/// The frequency of MIDI note `note`, Hz (A4, note 69, at 440 Hz).
pub fn hz(note: f32) -> f32 {
    440.0 * ((note - 69.0) / 12.0).exp2()
}

/// The MIDI note of frequency `hz` (fractional): [`hz`]'s inverse.
pub fn note(hz: f64) -> f32 {
    (69.0 + 12.0 * (hz / 440.0).log2()) as f32
}

/// Middle C, Hz: [`hz`] of note 60.
pub const MIDDLE_C: f32 = 261.625_57;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_note_and_its_frequency_are_one_another_s() {
        assert!((hz(60.0) - MIDDLE_C).abs() < 1e-3);
        for n in [21.0f32, 60.0, 69.0, 100.5] {
            assert!((note(f64::from(hz(n))) - n).abs() < 1e-4, "{n}");
        }
    }

    #[test]
    fn the_sine_is_accurate() {
        for (i, &s) in SINE.iter().enumerate() {
            let want = (2.0 * PI * i as f64 / 1024.0).sin();
            assert!((f64::from(s) - want).abs() < 1e-7, "{i}");
        }
        for k in 0..1000u32 {
            let ph = k.wrapping_mul(0x9E37_79B9);
            let want = (2.0 * PI * f64::from(ph) / 4_294_967_296.0).sin();
            assert!((f64::from(sine(ph)) - want).abs() < 2e-5);
        }
    }

    #[test]
    fn the_soft_clip_is_close_to_tanh() {
        for k in -400..=400 {
            let x = k as f32 / 100.0;
            assert!((softclip(x) - x.tanh()).abs() < 0.025, "{x}");
        }
        assert_eq!(soft_knee(0.3, 0.5), 0.3);
        assert!(soft_knee(10.0, 0.5) <= 1.0);
    }

    #[test]
    fn a_blep_saw_and_pulse_have_no_jump_left() {
        // Through a wrap the band-limited waves move by about a step's
        // worth, where the plain ones jump by 2.
        let inc = 1u32 << 26;
        for wave in [saw as fn(u32, u32) -> f32, |p, i| pulse(p, i, 1 << 30)] {
            let mut ph = 0u32.wrapping_sub(3 * inc);
            let mut last = wave(ph, inc);
            for _ in 0..80 {
                ph = ph.wrapping_add(inc);
                let s = wave(ph, inc);
                assert!((s - last).abs() <= 1.0 + 1e-6, "{last} -> {s}");
                last = s;
            }
        }
        // A pulse has no DC.
        let mean: f32 = (0..64u32)
            .map(|k| pulse(k << 26, inc, 1 << 30))
            .sum::<f32>()
            / 64.0;
        assert!(mean.abs() < 0.02, "{mean}");
    }

    #[test]
    fn the_filter_passes_and_stops_what_it_should() {
        let rate = Rate::new(48_000);
        let svf = Svf::new(1_000.0, damping(0.0), rate);
        let gain = |hz: f32| {
            let mut s = SvfState::default();
            let mut peak = [0.0f32; 4];
            for n in 0..48_000 {
                let x = (std::f32::consts::TAU * hz * n as f32 / 48_000.0).sin();
                let t = svf.run(&mut s, x);
                if n > 24_000 {
                    for (p, v) in peak.iter_mut().zip([t.lp, t.bp, t.hp, t.notch]) {
                        *p = p.max(v.abs());
                    }
                }
            }
            peak
        };
        let (low, mid, high) = (gain(100.0), gain(1_000.0), gain(10_000.0));
        assert!(low[0] > 0.95 && high[0] < 0.02, "{low:?} {high:?}");
        assert!(high[2] > 0.95 && low[2] < 0.02, "{low:?} {high:?}");
        assert!(mid[1] > 0.45 && mid[3] < 0.05, "{mid:?}");
    }

    #[test]
    fn the_chip_register_loops_as_documented() {
        for (short, period) in [(true, 93), (false, 32_767)] {
            let mut r = chip_noise(1, short);
            let mut n = 1;
            while r != 1 {
                r = chip_noise(r, short);
                n += 1;
            }
            assert_eq!(n, period);
        }
    }

    #[test]
    fn a_note_is_its_frequency() {
        let a4 = Rate::new(48_000).inc(69.0);
        assert!((f64::from(a4) - 440.0 / 48_000.0 * 4_294_967_296.0).abs() < 2.0);
        assert!((Rate::new(48_000).pole(1e9) - 1.0).abs() < 1e-6);
    }
}
