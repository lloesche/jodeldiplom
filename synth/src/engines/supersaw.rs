//! SUPERSAW: a stack of detuned saws for pads, choirs, big chords and
//! risers.
//!
//! Up to seven band-limited saws: one on the note and up to three pairs
//! around it, each pair detuned either way. The pairs are spaced unevenly,
//! the inner ones close and the outer one at the full spread (pair i of n
//! at (i / n)^1.6 of it), so the beating is slow near the note and fast at
//! the edges, as the analysed supersaw's is. The mix sets the pairs' level
//! against the centre saw (0: the centre alone, a plain saw; 1: all
//! equal), the sum normalised by its RMS so the loudness holds as the mix
//! moves. The saws start at phases drawn from the note's seed: in phase,
//! their wraps would line up into a spike on the first cycle. A sine an
//! octave down can go under them. A resonant state-variable low-pass that
//! follows the note shapes the lot, and a soft knee rounds the rare peaks
//! where many saws line up. The amplitude is the host's (a slow ADSR
//! attack makes the pad's swell).
//!
//! What the host's [`VoiceMod`] does here: `phase_inc` is the centre saw's
//! pitch, and `pitch` the note the cutoff follows; the brightness
//! (octaves, with the filter envelope, [`BRIGHT`]) moves the cutoff; the
//! shape widens the spread ([`SHAPE`]: times 1 + shape, so -1 closes it to
//! one saw's pitch).
//!
//! Sources: A. Szabo, "How to Emulate the Super Saw" (thesis, 2010), an
//! analysis of the Roland JP-8000's supersaw: seven saws, detuned
//! unevenly around the centre, the centre's and the sides' levels set
//! apart by its mix, their phases free rather than reset; the saws by
//! polyBLEP (Välimäki and Huovilainen 2007; Välimäki, Pekonen and Nam
//! 2012), the filter by Simper's trapezoidal SVF (Cytomic, 2013), the
//! seed by xorshift32 (Marsaglia 2003), as in [`crate::dsp`]. The spacing
//! curve and the mix law are our own.

use crate::dsp::{self, Rate, Svf, SvfState};
use crate::params::{field, Param, Scale};
use crate::{Engine, NoteOn, VoiceMod};

/// The saws of a voice, at the most.
pub const SAWS: usize = 7;

/// How the pairs spread out: pair i of n at (i / n)^CURVE of the spread.
const CURVE: f32 = 1.6;

/// The saws' sum at its RMS-normalised level (a saw's RMS is 0.58, so
/// about 0.35 before the filter).
const GAIN: f32 = 0.6;

/// Where the soft knee starts.
const KNEE: f32 = 0.7;

/// SUPERSAW's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The detuned pairs around the centre saw, 0 to 3 (1 to 7 saws).
    pub pairs: u8,
    /// The outer pair's detune either way, cents.
    pub detune: f32,
    /// The pairs' level against the centre saw, 0 to 1.
    pub mix: f32,
    /// The sine an octave down, 0 to 1.
    pub sub: f32,
    /// The low-pass's cutoff at middle C, Hz.
    pub cutoff: f32,
    /// Its resonance, 0 to 1.
    pub res: f32,
    /// How the cutoff follows the note, octaves an octave.
    pub key_track: f32,
}

impl Default for Params {
    fn default() -> Params {
        PLAIN
    }
}

/// SUPERSAW's parameters as data, with their ranges.
pub const PARAMS: &[Param<Params>] = &[
    Param::range(
        ("saws", "pairs"),
        (0.0, 3.0, "", Scale::Whole),
        |p| f32::from(p.pairs),
        |p, v| p.pairs = v as u8,
    ),
    field!("saws", "detune", detune, 0.0, 100.0, "ct", Linear),
    field!("saws", "mix", mix, 0.0, 1.0, "", Linear),
    field!("saws", "sub", sub, 0.0, 1.0, "", Linear),
    field!("filter", "cutoff", cutoff, 20.0, 20_000.0, "Hz", Log),
    field!("filter", "res", res, 0.0, 1.0, "", Linear),
    field!("filter", "key track", key_track, 0.0, 2.0, "", Linear),
];

/// [`Params::default`]: seven saws, an open filter.
const PLAIN: Params = Params {
    pairs: 3,
    detune: 20.0,
    mix: 0.6,
    sub: 0.0,
    cutoff: 12_000.0,
    res: 0.0,
    key_track: 0.0,
};

/// The held glow under a streak: soft, bright and airy, a gentle spread
/// under a low-pass well up in the treble.
pub const SHIMMER_PAD: Params = Params {
    detune: 22.0,
    mix: 0.55,
    cutoff: 5_000.0,
    res: 0.1,
    key_track: 0.5,
    ..PLAIN
};

/// A choir-like pad: a narrow spread, a little sub, a low, slightly
/// resonant cutoff that leaves a vowel-like hump.
pub const CHOIR_PAD: Params = Params {
    detune: 12.0,
    mix: 0.45,
    sub: 0.15,
    cutoff: 1_400.0,
    res: 0.3,
    key_track: 0.6,
    ..PLAIN
};

/// A big, wide chord: the widest spread, the pairs nearly as loud as the
/// centre, a sub under it.
pub const BIG_CHORD: Params = Params {
    detune: 35.0,
    mix: 0.8,
    sub: 0.3,
    cutoff: 3_200.0,
    res: 0.15,
    key_track: 0.4,
    ..PLAIN
};

/// A riser's pad: wide and dark, to be opened by the filter envelope and
/// lifted by the pitch glide.
pub const RISER_PAD: Params = Params {
    detune: 28.0,
    mix: 0.7,
    sub: 0.2,
    cutoff: 450.0,
    res: 0.35,
    key_track: 0.3,
    ..PLAIN
};

/// What the host's shape moves: the spread, times 1 + shape (closed at
/// -1, twice as wide at 1).
pub const SHAPE: &str = "detune spread";

/// What the host's brightness opens: the cutoff, an octave an octave.
pub const BRIGHT: &str = "cutoff";

/// The presets, by name.
pub const PRESETS: [(&str, Params); 4] = [
    ("SHIMMER PAD", SHIMMER_PAD),
    ("CHOIR PAD", CHOIR_PAD),
    ("BIG CHORD", BIG_CHORD),
    ("RISER PAD", RISER_PAD),
];

/// One SUPERSAW voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    /// The saws' phases, the centre first, then the pairs inside out.
    ph: [u32; SAWS],
    sub: u32,
    filter: SvfState,
    /// The phases have been drawn.
    seeded: bool,
}

/// SUPERSAW at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Supersaw {
    rate: Rate,
}

impl Engine for Supersaw {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Supersaw {
        Supersaw { rate }
    }

    /// A voice from silence draws its saws' phases from the note's seed. A
    /// voice still sounding (a mono patch played again) keeps its phases
    /// and its filter's state, so the pad carries on under the new note.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        if !v.seeded {
            let mut s = dsp::seed(n.seed);
            for ph in &mut v.ph {
                *ph = dsp::xorshift(&mut s);
            }
            v.seeded = true;
        }
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let pairs = usize::from(p.pairs.min(3));
        let saws = 1 + 2 * pairs;
        let mix = p.mix.clamp(0.0, 1.0);
        let spread = (p.detune * (1.0 + m.shape)).max(0.0) / 1_200.0;
        let norm = GAIN / (1.0 + 2.0 * pairs as f32 * mix * mix).sqrt();
        let mut inc = [m.phase_inc; SAWS];
        let mut gain = [norm; SAWS];
        for i in 0..pairs {
            let at = ((i + 1) as f32 / pairs as f32).powf(CURVE) * spread;
            for (k, sign) in [(1 + 2 * i, -1.0), (2 + 2 * i, 1.0)] {
                inc[k] = (f64::from(m.phase_inc) * f64::from(sign * at).exp2())
                    .min(f64::from(u32::MAX)) as u32;
                gain[k] = norm * mix;
            }
        }
        let inc_sub = m.phase_inc >> 1;
        let sub = p.sub.clamp(0.0, 1.0) * GAIN;
        let hz = p.cutoff * (m.bright + p.key_track * (m.pitch - 60.0) / 12.0).exp2();
        let svf = Svf::new(hz, dsp::damping(p.res), self.rate);
        let mut ph = v.ph;
        for o in out.iter_mut() {
            let mut s = 0.0;
            for k in 0..saws {
                s += dsp::saw(ph[k], inc[k]) * gain[k];
                ph[k] = ph[k].wrapping_add(inc[k]);
            }
            if sub > 0.0 {
                s += dsp::sine(v.sub) * sub;
                v.sub = v.sub.wrapping_add(inc_sub);
            }
            let y = svf.run(&mut v.filter, s).lp;
            *o += dsp::soft_knee(y, KNEE);
        }
        v.ph = ph;
    }
}
