//! NOISE: noise through a resonant filter, for whooshes, risers, ticks,
//! drawers, crackle and wind.
//!
//! The [`Source`] is white noise, sparse random impulses ("dust"), or a
//! chip's 15-bit noise register clocked above the key (its short loop a
//! pitched, metallic tone). The colour tilts white noise through pink to
//! brown. The filter (low-, band- or high-pass) follows the key and is
//! made up for what it takes away, so a sweep keeps its level. The drift
//! is a slow random wander of the cutoff and the clock; the crush holds
//! samples and drops bits.
//!
//! The host's brightness moves the cutoff ([`BRIGHT`]), its shape the
//! colour ([`SHAPE`]), and the pitch the key. Each note's noise is seeded
//! from its seed, so a note sounds the same every time.
//!
//! Sources: white noise by xorshift32 (Marsaglia 2003); pink noise by Paul
//! Kellet's "economy" three-pole filter (musicdsp.org, "Filter to make
//! pink noise from white"); brown noise by a leaky integrator; impulses
//! after SuperCollider's Dust2 (random impulses of random size and sign at
//! a mean density); the register after the NES APU's noise channel
//! (nesdev wiki, "APU Noise"); the filter by Simper's trapezoidal SVF
//! (Cytomic, 2013).

use crate::dsp::{self, Rate, Svf, SvfState};
use crate::{Engine, NoteOn, VoiceMod, BLOCK};

/// Where the noise comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// White noise.
    White,
    /// Random impulses at the density.
    Dust,
    /// The chip register clocked the clock's semitones above the key: its
    /// long loop (32767 steps, a hiss) or its short one (93 steps, a
    /// metallic tone, on the key at a clock of [`SHORT_AT_KEY`]).
    Register {
        /// The short loop.
        short: bool,
    },
}

/// The clock, semitones above the key, that puts the short loop's pitch
/// on the key (93 steps a period: 12 log2 93).
pub const SHORT_AT_KEY: f32 = 78.469_22;

/// The filter's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Low-pass.
    LowPass,
    /// Band-pass, its peak at unity.
    BandPass,
    /// High-pass.
    HighPass,
}

/// NOISE's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The source.
    pub source: Source,
    /// The colour of white noise: 0 white, 0.5 pink, 1 brown.
    pub color: f32,
    /// [`Source::Dust`]'s impulses a second.
    pub density: f32,
    /// [`Source::Register`]'s clock, semitones above the key.
    pub clock: f32,
    /// The filter's output.
    pub filter: Filter,
    /// Its cutoff at middle C, Hz.
    pub cutoff: f32,
    /// Its resonance, 0 to 1.
    pub res: f32,
    /// How the cutoff follows the pitch around middle C: 1 a semitone a
    /// semitone.
    pub key_track: f32,
    /// How far the drift wanders, semitones either way (0: none).
    pub drift: f32,
    /// How often it picks a new place, a second.
    pub drift_rate: f32,
    /// The rate the crush holds samples at, Hz (0: none).
    pub crush: f32,
    /// The bits the output is rounded to (0: not rounded).
    pub bits: u8,
}

impl Default for Params {
    fn default() -> Params {
        QUIET
    }
}

/// [`Params::default`]: white noise through an open low-pass.
const QUIET: Params = Params {
    source: Source::White,
    color: 0.0,
    density: 0.0,
    clock: 0.0,
    filter: Filter::LowPass,
    cutoff: 8_000.0,
    res: 0.0,
    key_track: 0.0,
    drift: 0.0,
    drift_rate: 1.0,
    crush: 0.0,
    bits: 0,
};

/// Air through a band: the riser's and the whoosh's.
pub const WHOOSH: Params = Params {
    color: 0.4,
    filter: Filter::BandPass,
    cutoff: 800.0,
    res: 0.55,
    key_track: 1.0,
    drift: 3.0,
    drift_rate: 1.5,
    ..QUIET
};

/// A rough slide: a cash drawer.
pub const DRAWER: Params = Params {
    color: 0.2,
    filter: Filter::BandPass,
    cutoff: 1_800.0,
    res: 0.4,
    key_track: 1.0,
    drift: 1.0,
    drift_rate: 6.0,
    ..QUIET
};

/// A bright click.
pub const TICK: Params = Params {
    filter: Filter::BandPass,
    cutoff: 5_000.0,
    res: 0.5,
    ..QUIET
};

/// Crackle.
pub const CRACKLE: Params = Params {
    source: Source::Dust,
    density: 60.0,
    filter: Filter::BandPass,
    cutoff: 3_000.0,
    res: 0.3,
    ..QUIET
};

/// Wind, wandering.
pub const WIND: Params = Params {
    color: 0.5,
    filter: Filter::BandPass,
    cutoff: 600.0,
    res: 0.8,
    drift: 7.0,
    drift_rate: 0.4,
    ..QUIET
};

/// The short loop on the key: a metallic, pitched clink.
pub const METAL: Params = Params {
    source: Source::Register { short: true },
    clock: SHORT_AT_KEY,
    cutoff: 6_000.0,
    res: 0.1,
    key_track: 1.0,
    ..QUIET
};

/// The long loop clocked low: a crunchy arcade hiss.
pub const ARCADE: Params = Params {
    source: Source::Register { short: false },
    clock: 60.0,
    cutoff: 12_000.0,
    key_track: 0.5,
    ..QUIET
};

/// The presets, by name.
pub const PRESETS: [(&str, Params); 7] = [
    ("WHOOSH", WHOOSH),
    ("DRAWER", DRAWER),
    ("TICK", TICK),
    ("CRACKLE", CRACKLE),
    ("WIND", WIND),
    ("METAL", METAL),
    ("ARCADE", ARCADE),
];

/// What the host's shape moves: the colour, by up to half its range.
pub const SHAPE: &str = "colour";

/// What the host's brightness opens: the cutoff, an octave an octave.
pub const BRIGHT: &str = "cutoff";

/// Kellet's pink poles and their gains (designed at 44.1 kHz), his direct
/// path, and what brings the sum to the level of the white noise in.
const KELLET: [(f32, f32); 3] = [
    (0.997_65, 0.099_046),
    (0.963, 0.296_516_4),
    (0.57, 1.052_691_3),
];
const KELLET_DIRECT: f32 = 0.184_8;
const PINK_GAIN: f32 = 0.34;

/// The corner of the brown noise's integrator, Hz.
const BROWN_HZ: f32 = 20.0;

/// One NOISE voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    filter: SvfState,
    /// The pink poles and the brown one.
    pink: [f32; 3],
    brown: f32,
    rng: u32,
    /// Where the drift is, where it goes, and the blocks until it picks
    /// again.
    drift: f32,
    target: f32,
    countdown: i32,
    /// The register and its clock's phase.
    register: u32,
    clock: u32,
    crush: dsp::Crush,
}

/// NOISE at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    rate: Rate,
    /// Kellet's poles and gains moved to this rate (each pole keeping its
    /// corner, each section its gain at DC).
    pink: [(f32, f32); 3],
    /// The brown integrator's pole and gain (unit variance for unit white).
    brown: (f32, f32),
}

impl Engine for Noise {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Noise {
        let at = 44_100.0 / rate.hz();
        let pink = KELLET.map(|(p, g)| {
            let q = p.powf(at);
            (q, g * (1.0 - q) / (1.0 - p))
        });
        let p = (-std::f32::consts::TAU * BROWN_HZ / rate.hz()).exp();
        Noise {
            rate,
            pink,
            brown: (p, (1.0 - p * p).sqrt()),
        }
    }

    /// Each note its own noise (from its seed); the drift picks a new place
    /// at once. A retrigger keeps the filter, the colour filters, the
    /// register and its clock, the crush and where the drift is.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        v.rng = dsp::seed(n.seed);
        v.countdown = 0;
        if v.register == 0 {
            v.register = 1;
        }
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let mut rng = v.rng;
        if p.drift > 0.0 {
            v.countdown -= 1;
            if v.countdown <= 0 {
                v.target = dsp::white(&mut rng);
                let seconds = (1.0 + 0.5 * dsp::white(&mut rng)) / p.drift_rate.max(0.01);
                v.countdown = (seconds * self.rate.hz() / BLOCK as f32) as i32;
            }
            v.drift += (v.target - v.drift) * self.rate.block_pole(p.drift_rate);
        }
        let wander = v.drift * p.drift;
        let color = (p.color + m.shape * 0.5).clamp(0.0, 1.0);
        let semis = p.key_track * (m.pitch - 60.0) + wander;
        let top = 0.45 * self.rate.hz();
        let hz = (p.cutoff * (m.bright + semis / 12.0).exp2()).clamp(dsp::CUTOFF_MIN, top);
        let svf = Svf::new(hz, dsp::damping(p.res), self.rate);
        let k = svf.k();
        // Make up what the filter takes from white noise (by the share of
        // the band it lets through), less for pink and brown, at most x16,
        // so whatever the band its RMS is about a quarter of full scale.
        let nyquist = 0.5 * self.rate.hz();
        let band = match p.filter {
            Filter::LowPass => hz,
            Filter::BandPass => k * hz * std::f32::consts::FRAC_PI_2,
            Filter::HighPass => nyquist - hz,
        };
        let makeup = 0.45
            * (nyquist / band.max(1.0))
                .sqrt()
                .clamp(1.0, 16.0)
                .powf(1.0 - 0.75 * color);
        // An impulse a sample with the density's chance.
        let dust = self.rate.inc_hz(p.density);
        let clock = self.rate.clock_inc(dsp::hz(m.pitch + p.clock + wander));
        let crush = dsp::CrushSetting::new(p.crush, p.bits, self.rate);
        let mut filter = v.filter;
        let [mut b0, mut b1, mut b2] = v.pink;
        let mut brown = v.brown;
        for o in out.iter_mut() {
            let w = match p.source {
                Source::White => dsp::white(&mut rng),
                Source::Dust => {
                    if dsp::xorshift(&mut rng) < dust {
                        let size = 0.75 + 0.25 * dsp::white(&mut rng);
                        size.copysign(dsp::white(&mut rng))
                    } else {
                        0.0
                    }
                }
                Source::Register { short } => {
                    dsp::clock_noise(&mut v.register, &mut v.clock, clock, short);
                    if v.register & 1 != 0 {
                        0.5
                    } else {
                        -0.5
                    }
                }
            };
            let mut x = w;
            if color > 0.0 {
                let [(p0, g0), (p1, g1), (p2, g2)] = self.pink;
                b0 = b0 * p0 + w * g0;
                b1 = b1 * p1 + w * g1;
                b2 = b2 * p2 + w * g2;
                let pink = (b0 + b1 + b2 + w * KELLET_DIRECT) * PINK_GAIN;
                x = if color <= 0.5 {
                    w + (pink - w) * (2.0 * color)
                } else {
                    brown = brown * self.brown.0 + w * self.brown.1;
                    pink + (brown - pink) * (2.0 * color - 1.0)
                };
            }
            let t = svf.run(&mut filter, x);
            let y = match p.filter {
                Filter::LowPass => t.lp,
                Filter::BandPass => t.bp * k,
                Filter::HighPass => t.hp,
            };
            let y = dsp::soft_knee(y * makeup, 0.8);
            *o += v.crush.run(y, &crush);
        }
        v.filter = filter;
        v.pink = [b0, b1, b2];
        v.brown = brown;
        v.rng = rng;
    }
}
