//! FM: four sine operators modulating one another's phase, for bells, an
//! electric piano, glints, sparkles, chimes, a marimba and brass.
//!
//! Each operator is a sine at a ratio of the note's frequency (or at a
//! fixed frequency), detuned in cents, with an envelope of its own: a
//! linear attack, an exponential decay to its sustain level, held there
//! while the note is. An algorithm routes them: a modulator's output is
//! added to the phase of the operators below it, and the carriers' outputs
//! are summed. A carrier's level is its share of the output; a modulator's
//! is its modulation index, the peak phase deviation in radians, which
//! sets how far the sidebands reach. Operator 4 can modulate itself: its
//! last two samples, averaged (which keeps the loop from ringing at half
//! the rate), are fed back into its phase, turning its sine towards a saw
//! and, past about 2 radians, noise.
//!
//! Each operator's velocity sensitivity scales its level with the note's
//! velocity, so a harder note can be louder and, through its modulators,
//! brighter. The note's own envelope (the host's ADSR, which the voice
//! keeps each block as the host asks it for its amplitude) reaches the
//! operators through their release: each level is multiplied by that
//! envelope to the operator's release power, so as a held note is let go a
//! modulator with a release of 1 fades its index with the note (the sound
//! darkens as it dies), and a carrier with one fades faster than the note.
//!
//! Every block the modulators' indices are kept within the band: by
//! Carson's rule a modulator at f with index I spreads its target's
//! spectrum about (I + 1) f either way, so an index is held where that
//! reaches 0.45 of the rate above the target, and an operator at or above
//! 0.45 of the rate is left out. High notes stay clean instead of folding
//! back as inharmonic noise. The key scale thins the modulation further
//! above middle C.
//!
//! What the host's [`VoiceMod`] does here: `phase_inc` is the note's
//! frequency, the ratio operators' base; `pitch` places the key scale;
//! the brightness (octaves, [`BRIGHT`]) and the shape ([`SHAPE`]) multiply
//! every modulator's index, by 2^(brightness / 2) and 2^shape. The voice
//! ends by itself once every sounding carrier that does not sustain has
//! decayed 80 dB.
//!
//! Sources: J. M. Chowning, "The Synthesis of Complex Audio Spectra by
//! Means of Frequency Modulation" (J. Audio Eng. Soc. 21(7), 1973): the
//! sidebands at the carrier plus and minus multiples of the modulator,
//! weighted by the Bessel functions of the index, inharmonic spectra from
//! irrational ratios (his bells at 1 : 1.4, the index decaying with the
//! amplitude), and the brass whose index follows its envelope; the four
//! operators, the algorithms and the feedback on operator 4 as Yamaha's
//! four-operator synthesizers have them, from their owner's manuals'
//! published descriptions (the DX21, for one, offers eight such
//! algorithms); the band limit after J. R. Carson, "Notes on the Theory of
//! Modulation" (Proc. IRE 10, 1922). The sine is [`dsp::sine`].

use crate::dsp::{self, Rate};
use crate::params::{field, on, pick, when, Param, Scale};
use crate::{Engine, NoteOn, VoiceMod};
use std::f32::consts::TAU;

/// The operators of a voice.
pub const OPERATORS: usize = 4;

/// The highest frequency an operator or a sideband reaches, cycles a
/// sample (21.6 kHz at 48 kHz).
const TOP: f32 = 0.45;

/// How far an exponential decay has fallen at its end: 60 dB.
const FLOOR: f32 = 0.001;

/// Below this (-80 dB) a carrier's envelope counts as silent.
const QUIET: f32 = 1e-4;

/// How an operator's frequency is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Freq {
    /// A multiple of the note's frequency.
    Ratio(f32),
    /// A fixed frequency, Hz, whatever the note.
    Hz(f32),
}

/// One operator: a sine, its frequency, its level and its envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Operator {
    /// Its frequency.
    pub freq: Freq,
    /// Its detune, cents.
    pub detune: f32,
    /// A carrier's level, 0 to 1 (the carriers' levels share the output:
    /// summed past 1, they are scaled down to it); a modulator's index,
    /// radians (0 a pure carrier, about 1 a few sidebands, 5 bright).
    pub level: f32,
    /// The rise from silence (or from where a retrigger finds it) to full,
    /// seconds.
    pub attack: f32,
    /// The fall from full to the sustain level (to within 60 dB of it),
    /// seconds; 0 goes there at once.
    pub decay: f32,
    /// The level held while the note is, 0 to 1. A carrier at 0 rings out
    /// with its decay.
    pub sustain: f32,
    /// How much it follows the note's envelope: its level times that
    /// envelope to this power (0: not at all; 1: fading with the note as
    /// it is let go).
    pub release: f32,
    /// How much the velocity sets its level, 0 (not at all) to 1 (the
    /// level times the velocity).
    pub velocity: f32,
}

/// How the operators are routed, operator 1 always a carrier. "a -> b":
/// operator a modulates b; operators named alone are carriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// 4 -> 3 -> 2 -> 1: one stack, the richest.
    Stack,
    /// 3 and 4 -> 2 -> 1.
    TwoIntoStack,
    /// 3 -> 2 -> 1, and 4 (the fed-back one) -> 1.
    Branch,
    /// 4 -> 3 -> 1, and 2 -> 1.
    BranchStack,
    /// 2, 3 and 4 each -> 1.
    ThreeIntoOne,
    /// 2 -> 1 and 4 -> 3: two pairs side by side.
    TwoPairs,
    /// 4 -> 1, 2 and 3: one modulator under three carriers.
    OneIntoThree,
    /// 4 -> 3, with 1 and 2 plain sines.
    PairAndTwo,
    /// 1, 2, 3 and 4 all plain sines (an additive organ).
    Parallel,
}

/// Every algorithm.
pub const ALGORITHMS: [Algorithm; 9] = [
    Algorithm::Stack,
    Algorithm::TwoIntoStack,
    Algorithm::Branch,
    Algorithm::BranchStack,
    Algorithm::ThreeIntoOne,
    Algorithm::TwoPairs,
    Algorithm::OneIntoThree,
    Algorithm::PairAndTwo,
    Algorithm::Parallel,
];

impl Algorithm {
    /// For each operator (0 is operator 1) the operators modulating it, a
    /// bit each, and the carriers, a bit each. A modulator is always
    /// numbered above what it modulates.
    fn routing(self) -> ([u8; OPERATORS], u8) {
        const fn op(n: u8) -> u8 {
            1 << (n - 1)
        }
        match self {
            Algorithm::Stack => ([op(2), op(3), op(4), 0], op(1)),
            Algorithm::TwoIntoStack => ([op(2), op(3) | op(4), 0, 0], op(1)),
            Algorithm::Branch => ([op(2) | op(4), op(3), 0, 0], op(1)),
            Algorithm::BranchStack => ([op(2) | op(3), 0, op(4), 0], op(1)),
            Algorithm::ThreeIntoOne => ([op(2) | op(3) | op(4), 0, 0, 0], op(1)),
            Algorithm::TwoPairs => ([op(2), 0, op(4), 0], op(1) | op(3)),
            Algorithm::OneIntoThree => ([op(4), op(4), op(4), 0], op(1) | op(2) | op(3)),
            Algorithm::PairAndTwo => ([0, 0, op(4), 0], op(1) | op(2) | op(3)),
            Algorithm::Parallel => ([0; OPERATORS], op(1) | op(2) | op(3) | op(4)),
        }
    }
}

/// FM's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The routing.
    pub algorithm: Algorithm,
    /// The operators, operator 1 first.
    pub ops: [Operator; OPERATORS],
    /// Operator 4's feedback, radians (0 none, about 1 a saw-like edge,
    /// past 2 noise).
    pub feedback: f32,
    /// How the modulation thins out above middle C: the indices times
    /// 2^-key_scale an octave up (0: the same on every note).
    pub key_scale: f32,
}

impl Default for Params {
    fn default() -> Params {
        SINE
    }
}

/// The algorithms' names, in [`ALGORITHMS`]' order.
const ALGORITHM_NAMES: [&str; 9] = [
    "4>3>2>1",
    "3+4>2>1",
    "3>2>1, 4>1",
    "4>3>1, 2>1",
    "2+3+4>1",
    "2>1, 4>3",
    "4>1+2+3",
    "4>3, 1, 2",
    "1, 2, 3, 4",
];

/// Middle C, Hz: where an operator's ratio and its fixed frequency meet,
/// as it is switched from one to the other.
const MIDDLE_C: f32 = 261.625_57;

/// An operator's frequency as a ratio (a fixed one as at middle C).
fn ratio(f: Freq) -> f32 {
    match f {
        Freq::Ratio(r) => r,
        Freq::Hz(hz) => hz / MIDDLE_C,
    }
}

/// An operator's frequency in hertz (a ratio as at middle C).
fn hz(f: Freq) -> f32 {
    match f {
        Freq::Ratio(r) => r * MIDDLE_C,
        Freq::Hz(hz) => hz,
    }
}

/// Operator `$k`'s parameters (0 is operator 1), in group `$group`.
macro_rules! op {
    ($group:expr, $k:expr) => {
        [
            Param::toggle(
                ($group, "fixed"),
                |p| on(matches!(p.ops[$k].freq, Freq::Hz(_))),
                |p, v| {
                    let f = p.ops[$k].freq;
                    p.ops[$k].freq = if v >= 0.5 {
                        Freq::Hz(hz(f))
                    } else {
                        Freq::Ratio(ratio(f))
                    };
                },
            ),
            when(
                Param::range(
                    ($group, "ratio"),
                    (0.0, 32.0, "", Scale::Fine),
                    |p| ratio(p.ops[$k].freq),
                    |p, v| p.ops[$k].freq = Freq::Ratio(v),
                ),
                |p| matches!(p.ops[$k].freq, Freq::Ratio(_)),
            ),
            when(
                Param::range(
                    ($group, "hz"),
                    (1.0, 20_000.0, "Hz", Scale::Log),
                    |p| hz(p.ops[$k].freq),
                    |p, v| p.ops[$k].freq = Freq::Hz(v),
                ),
                |p| matches!(p.ops[$k].freq, Freq::Hz(_)),
            ),
            op_field!($group, "detune", $k, detune, -100.0, 100.0, "ct", Linear),
            op_field!($group, "level", $k, level, 0.0, 8.0, "", Fine),
            op_field!($group, "attack", $k, attack, 0.0, 4.0, "s", Fine),
            op_field!($group, "decay", $k, decay, 0.0, 12.0, "s", Fine),
            op_field!($group, "sustain", $k, sustain, 0.0, 1.0, "", Linear),
            op_field!($group, "release", $k, release, 0.0, 2.0, "", Linear),
            op_field!($group, "velocity", $k, velocity, 0.0, 1.0, "", Linear),
        ]
    };
}

/// A number of operator `$k`'s.
macro_rules! op_field {
    ($group:expr, $name:expr, $k:expr, $f:ident, $min:expr, $max:expr, $unit:expr, $scale:ident) => {
        Param::range(
            ($group, $name),
            ($min, $max, $unit, Scale::$scale),
            |p| p.ops[$k].$f,
            |p, v| p.ops[$k].$f = v,
        )
    };
}

/// The parameters of each operator.
const OP_PARAMS: [[Param<Params>; 10]; OPERATORS] = [
    op!("op 1", 0),
    op!("op 2", 1),
    op!("op 3", 2),
    op!("op 4", 3),
];

/// FM's parameters as data, with their ranges: the routing, the feedback
/// and the key scaling, then each operator's.
pub const PARAMS: &[Param<Params>] = &{
    let mut all = [OP_PARAMS[0][0]; 3 + 10 * OPERATORS];
    all[0] = pick!("fm", "algorithm", algorithm, ALGORITHMS, ALGORITHM_NAMES);
    all[1] = field!("fm", "feedback", feedback, 0.0, 3.0, "rad", Linear);
    all[2] = field!("fm", "key scale", key_scale, 0.0, 2.0, "", Linear);
    let mut k = 0;
    while k < 10 * OPERATORS {
        all[3 + k] = OP_PARAMS[k / 10][k % 10];
        k += 1;
    }
    all
};

/// An operator at the note's frequency, silent, held at full.
const OP: Operator = Operator {
    freq: Freq::Ratio(1.0),
    detune: 0.0,
    level: 0.0,
    attack: 0.0,
    decay: 0.0,
    sustain: 1.0,
    release: 0.0,
    velocity: 0.0,
};

/// An operator struck: at `ratio` and `level`, no attack, decaying to
/// silence over `decay` seconds, `velocity` sensitive.
const fn struck(ratio: f32, level: f32, decay: f32, velocity: f32) -> Operator {
    Operator {
        freq: Freq::Ratio(ratio),
        level,
        decay,
        sustain: 0.0,
        velocity,
        ..OP
    }
}

/// [`Params::default`]: operator 1 alone, a sine.
const SINE: Params = Params {
    algorithm: Algorithm::Stack,
    ops: [Operator { level: 1.0, ..OP }, OP, OP, OP],
    feedback: 0.0,
    key_scale: 0.0,
};

/// A clean, singing glass bell: two sines at the note a cent and a half
/// apart (a slow beat) ring long, an octave pair two octaves up glints
/// briefly, its modulator at 1.4 throwing a few inharmonic sidebands that
/// are gone within a third of a second.
pub const GLASS_BELL: Params = Params {
    algorithm: Algorithm::PairAndTwo,
    ops: [
        struck(1.0, 0.42, 4.0, 0.3),
        Operator {
            detune: 1.5,
            ..struck(1.0, 0.3, 3.0, 0.3)
        },
        struck(4.0, 0.16, 1.2, 0.5),
        struck(1.4, 1.1, 0.35, 0.7),
    ],
    feedback: 0.0,
    key_scale: 0.5,
};

/// An electric piano: a body pair at 1 : 1 whose index falls from 1.6 to a
/// quarter of it (softer notes rounder), and a tine pair whose modulator
/// at 14 gives the bell-like strike of the first tenth of a second.
pub const E_PIANO: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        struck(1.0, 0.6, 3.0, 0.4),
        Operator {
            sustain: 0.25,
            release: 1.0,
            ..struck(1.0, 1.6, 1.2, 0.8)
        },
        Operator {
            detune: 3.0,
            ..struck(1.0, 0.35, 1.0, 0.5)
        },
        struck(14.0, 0.9, 0.12, 1.0),
    ],
    feedback: 0.0,
    key_scale: 0.4,
};

/// A coin's high metallic "ting": two inharmonic pairs (2.41 into the
/// note, 5.19 into 2.76 with a touch of feedback), struck hard and short.
pub const COIN_GLINT: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        struck(1.0, 0.5, 0.6, 0.3),
        struck(2.41, 1.4, 0.15, 0.6),
        struck(2.76, 0.35, 0.35, 0.3),
        struck(5.19, 0.8, 0.06, 0.6),
    ],
    feedback: 0.5,
    key_scale: 0.3,
};

/// A short, bright glitter for arpeggiated sparkles: the note and its
/// octave, each brightened by a harmonic modulator for its first few tens
/// of milliseconds.
pub const SPARKLE: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        struck(1.0, 0.55, 0.4, 0.3),
        struck(3.0, 1.2, 0.07, 0.8),
        Operator {
            detune: 6.0,
            ..struck(2.0, 0.3, 0.22, 0.3)
        },
        struck(5.0, 0.7, 0.04, 0.8),
    ],
    feedback: 0.0,
    key_scale: 0.5,
};

/// A chime: three carriers at a bar's first modes (1, 2.76, 5.40), the
/// higher ringing shorter, under one modulator's brief clang.
pub const CHIME: Params = Params {
    algorithm: Algorithm::OneIntoThree,
    ops: [
        struck(1.0, 0.5, 3.5, 0.3),
        struck(2.76, 0.25, 2.0, 0.3),
        struck(5.4, 0.12, 1.0, 0.3),
        struck(1.0, 0.7, 0.2, 0.6),
    ],
    feedback: 0.0,
    key_scale: 0.4,
};

/// A marimba: the bar's note under a knock of odd harmonics (a modulator
/// at 4) gone in 50 ms, and its second mode, two octaves up, ringing
/// short.
pub const MARIMBA: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        struck(1.0, 0.75, 0.9, 0.3),
        struck(4.0, 0.9, 0.05, 0.8),
        struck(4.0, 0.18, 0.25, 0.3),
        struck(1.0, 0.4, 0.03, 0.6),
    ],
    feedback: 0.0,
    key_scale: 0.3,
};

/// Brass: two pairs at 1 : 1, a few cents apart, their indices swelling
/// with the attack and settling (Chowning's brass), fading with the note
/// as it is let go; a touch of feedback for the edge.
pub const BRASS: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        Operator {
            attack: 0.05,
            decay: 0.5,
            sustain: 0.85,
            velocity: 0.2,
            level: 0.55,
            ..OP
        },
        Operator {
            attack: 0.08,
            decay: 0.6,
            sustain: 0.6,
            release: 1.0,
            velocity: 0.6,
            level: 3.0,
            ..OP
        },
        Operator {
            detune: 8.0,
            attack: 0.05,
            decay: 0.5,
            sustain: 0.85,
            velocity: 0.2,
            level: 0.4,
            ..OP
        },
        Operator {
            attack: 0.1,
            decay: 0.6,
            sustain: 0.5,
            release: 1.0,
            velocity: 0.6,
            level: 2.4,
            ..OP
        },
    ],
    feedback: 0.4,
    key_scale: 0.3,
};

/// A deep bell: Chowning's 1 : 1.4, its index falling from 2.4 over a few
/// seconds so the clang settles into the prime, over a hum an octave down
/// whose own modulator gives the quint and the partials above it.
pub const BELL: Params = Params {
    algorithm: Algorithm::TwoPairs,
    ops: [
        struck(1.0, 0.5, 8.0, 0.2),
        struck(1.4, 2.4, 3.5, 0.6),
        struck(0.5, 0.35, 10.0, 0.2),
        struck(1.0, 0.9, 4.0, 0.6),
    ],
    feedback: 0.0,
    key_scale: 0.0,
};

/// What the host's shape moves: every modulator's index, times 2^shape
/// (from half to twice it).
pub const SHAPE: &str = "modulation index";

/// What the host's brightness opens: every modulator's index, times
/// 2^(brightness / 2), the carriers as they are.
pub const BRIGHT: &str = "modulation index";

/// The presets, by name.
pub const PRESETS: [(&str, Params); 8] = [
    ("GLASS BELL", GLASS_BELL),
    ("E-PIANO", E_PIANO),
    ("COIN GLINT", COIN_GLINT),
    ("SPARKLE", SPARKLE),
    ("CHIME", CHIME),
    ("MARIMBA FM", MARIMBA),
    ("BRASS FM", BRASS),
    ("BELL", BELL),
];

/// One FM voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    /// The operators' phases.
    ph: [u32; OPERATORS],
    /// Operator 4's last two samples (its sine, before its level).
    fb: [f32; 2],
    /// Seconds since the note started.
    t: f32,
    /// Each envelope's level as the note started, and now.
    from: [f32; OPERATORS],
    env: [f32; OPERATORS],
    /// Each operator's gain at the end of the last block (a carrier's
    /// amplitude, a modulator's index in turns), where the next ramps from.
    gain: [f32; OPERATORS],
    velocity: f32,
    /// The note's envelope (the host's ADSR) this block.
    note_env: f32,
    /// Every carrier has rung out.
    ended: bool,
}

/// FM at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Fm {
    rate: Rate,
}

impl Fm {
    /// The phase increment a sample of operator `op` on a note stepping by
    /// `note_inc`.
    fn inc(&self, op: &Operator, note_inc: u32) -> u32 {
        let detune = (op.detune / 1_200.0).exp2();
        match op.freq {
            Freq::Ratio(r) => (f64::from(note_inc) * f64::from(r.max(0.0) * detune))
                .min(f64::from(u32::MAX)) as u32,
            Freq::Hz(hz) => self.rate.inc_hz(hz * detune),
        }
    }
}

/// An operator's envelope `t` seconds into the note, its attack rising
/// from `from`.
fn envelope(op: &Operator, from: f32, t: f32) -> f32 {
    if t < op.attack {
        return from + (1.0 - from) * (t / op.attack);
    }
    let sustain = op.sustain.clamp(0.0, 1.0);
    if op.decay <= 0.0 {
        return sustain;
    }
    sustain + (1.0 - sustain) * FLOOR.powf((t - op.attack) / op.decay)
}

/// A phase offset of `x` turns (either way, any size) as a u32 phase.
#[inline]
fn offset(x: f32) -> u32 {
    (x * 4_294_967_296.0) as i64 as u32
}

impl Engine for Fm {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Fm {
        Fm { rate }
    }

    /// Starts the operators' envelopes. A voice still sounding (a mono
    /// patch played again) keeps its phases, its feedback and its
    /// operators' levels: each envelope's attack rises from where it is,
    /// and the levels glide there from where they were, so a retrigger
    /// never clicks.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        v.from = v.env;
        v.t = 0.0;
        v.velocity = n.velocity.clamp(0.0, 1.0);
        v.note_env = 1.0;
        v.ended = false;
    }

    /// The ADSR's own amplitude; its level kept for the operators'
    /// releases.
    fn amp(&self, _p: &Params, v: &mut Voice, adsr: f32) -> f32 {
        v.note_env = adsr.clamp(0.0, 1.0);
        adsr
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }
        let (mods, carriers) = p.algorithm.routing();
        let bit = |k: usize| 1u8 << k;
        v.t += out.len() as f32 / self.rate.hz();
        let inc: [u32; OPERATORS] = std::array::from_fn(|k| self.inc(&p.ops[k], m.phase_inc));
        let f: [f32; OPERATORS] = inc.map(|i| i as f32 * (1.0 / 4_294_967_296.0));
        let deepen = (0.5 * m.bright + m.shape).exp2()
            * (-p.key_scale * (m.pitch - 60.0).max(0.0) / 12.0).exp2();
        let total = (0..OPERATORS)
            .filter(|&k| carriers & bit(k) != 0)
            .map(|k| p.ops[k].level.max(0.0))
            .sum::<f32>()
            .max(1.0);
        let mut target = [0.0f32; OPERATORS];
        let mut sounding = false;
        for (k, op) in p.ops.iter().enumerate() {
            let e = envelope(op, v.from[k], v.t);
            v.env[k] = e;
            let touch = 1.0 - op.velocity.clamp(0.0, 1.0) * (1.0 - v.velocity);
            let fade = v.note_env.powf(op.release.max(0.0));
            let level = if f[k] >= TOP {
                0.0
            } else {
                op.level.max(0.0) * e * touch * fade
            };
            target[k] = if carriers & bit(k) != 0 {
                if op.level > 0.0 && (op.sustain > 0.0 || v.t <= op.attack || e >= QUIET) {
                    sounding = true;
                }
                level / total
            } else {
                // Within Carson's band of each operator it modulates.
                let room = (0..OPERATORS)
                    .filter(|&j| mods[j] & bit(k) != 0)
                    .map(|j| (TOP - f[j]) / f[k].max(1e-9) - 1.0)
                    .fold(f32::MAX, f32::min);
                (level * deepen).min(room.max(0.0)) / TAU
            };
        }
        v.ended = !sounding;

        // The weights: w[k][j] 1 where operator j modulates k; c[k] 1 for
        // a carrier.
        let w: [[f32; OPERATORS]; OPERATORS] =
            std::array::from_fn(|k| std::array::from_fn(|j| f32::from(mods[k] & bit(j) != 0)));
        let c: [f32; OPERATORS] = std::array::from_fn(|k| f32::from(carriers & bit(k) != 0));
        let steps = out.len() as f32;
        let dg: [f32; OPERATORS] = std::array::from_fn(|k| (target[k] - v.gain[k]) / steps);
        let feedback = p.feedback.max(0.0) / TAU * 0.5;
        let mut ph = v.ph;
        let mut g = v.gain;
        let mut fb = v.fb;
        for o in out.iter_mut() {
            let s4 = dsp::sine(ph[3].wrapping_add(offset((fb[0] + fb[1]) * feedback)));
            fb = [s4, fb[0]];
            let mut y = [0.0f32; OPERATORS];
            y[3] = s4 * g[3];
            for k in (0..3).rev() {
                let pm = w[k][1] * y[1] + w[k][2] * y[2] + w[k][3] * y[3];
                y[k] = dsp::sine(ph[k].wrapping_add(offset(pm))) * g[k];
            }
            let s = c[0] * y[0] + c[1] * y[1] + c[2] * y[2] + c[3] * y[3];
            for k in 0..OPERATORS {
                ph[k] = ph[k].wrapping_add(inc[k]);
                g[k] += dg[k];
            }
            *o += s;
        }
        v.ph = ph;
        v.gain = target;
        v.fb = fb;
    }

    fn done(&self, v: &Voice) -> bool {
        v.ended
    }
}
