//! MODAL: a mallet strikes a body whose modes ring as a bank of
//! resonators, for bells, slot bells, coins, glass, kalimba and marimba.
//!
//! Each of the [`MODES`] modes is a damped sinusoid: a coupled-form
//! resonator (a rotation by the mode's frequency, scaled by its decay a
//! sample), which stays accurate at low frequencies where a direct-form
//! two-pole would need more precision than f32 has. Its impulse response is
//! r^n sin(n theta): unit amplitude whatever the frequency, so a mode's
//! level is just its gain. The decay is a ring time to -60 dB for the
//! lowest mode, shorter for the higher ones as the brightness falls (losses
//! rising with frequency, as air and internal friction do). The modes can
//! come in split pairs a fraction of a hertz apart, as an imperfectly round
//! glass or bell has them: they beat.
//!
//! The mallet is a force pulse, half a sine over the contact time, as a
//! Hertzian contact gives. Its spectrum falls off above about 1.5 over the
//! contact time, so a hard mallet (a short contact) excites the high modes
//! and a soft one only the low ones. A noise burst under the same window
//! adds a scrape or a jingle. Striking a body that still rings damps it
//! first, over the contact, to [`RESTRIKE`] of its amplitude (the mallet
//! rests on it a moment), so however fast it is struck its level grows
//! towards a ceiling of twice a strike's, never past it.
//!
//! What the host's [`VoiceMod`] does here: `phase_inc` sets the lowest
//! mode's frequency; `shape` moves the body ([`SHAPE`]); `bright` hardens
//! the mallet and lifts the brightness ([`BRIGHT`]). The velocity shortens
//! the contact (a harder hit is brighter); the level is the host's.
//!
//! Sources: modal synthesis after Adrien (1991, "The missing link: modal
//! synthesis", in De Poli, Piccialli and Roads, "Representations of Musical
//! Signals") and Cook (2002, "Real Sound Synthesis for Interactive
//! Applications"); the coupled-form resonator after Gold and Rader (1969,
//! "Digital Processing of Signals"); the half-sine mallet after Fletcher
//! and Rossing ("The Physics of Musical Instruments", 2nd ed. 1998), whose
//! beam and ring theory, bell partials and Leissa's (1969) plate modes
//! give the body tables. The wine glass's ratios, 1 : 2.3 : 4.2 : 6.6, are
//! our choice by ear, between the lower ratios measured glasses show and
//! thin-ring theory's 1 : 2.83 : 5.42 : 8.77; that an imperfectly round
//! glass splits each mode into a beating pair is as French (1983, "In vino
//! veritas: A study of wineglass acoustics", Am. J. Phys. 51) describes.

use crate::dsp::{self, Quiet, Rate};
use crate::{Engine, NoteOn, VoiceMod, BLOCK};
use std::f32::consts::{PI, TAU};

/// The modes a voice.
pub const MODES: usize = 24;

/// Modes rendered together, their state in registers.
const BATCH: usize = 4;

/// Modes at or above this frequency (cycles a sample: 21.6 kHz at 48 kHz)
/// are left out.
const TOP: f32 = 0.45;

/// The output gain: a hard hit peaks between about 0.2 and 1.
const OUTPUT: f32 = 0.36;

/// The share of a ringing body's amplitude a re-strike leaves.
pub const RESTRIKE: f32 = 0.5;

/// The mode frequency ratios of the bodies the body parameter sweeps, the
/// lowest mode first (always 1: the played pitch).
const BODIES: [[f32; MODES]; 6] = [
    // Harmonic: a string or a tube.
    [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ],
    // A marimba bar: the first three modes tuned 1 : 4 : 10 by the undercut,
    // the rest the free-free beam's (beta_n / beta_1)^2 scaled on from 10.
    [
        1.0, 4.0, 10.0, 16.53, 24.69, 34.49, 45.92, 58.98, 73.67, 90.0, 108.0, 127.6, 148.8, 171.6,
        196.1, 222.2, 250.0, 279.4, 310.4, 343.1, 377.3, 413.3, 450.8, 490.0,
    ],
    // A kalimba tine: a clamped-free beam, (beta_n / beta_1)^2 with
    // beta = 1.8751, 4.6941, 7.8548, 10.9955, 14.1372, then (n - 1/2) pi.
    [
        1.0, 6.267, 17.55, 34.39, 56.84, 84.91, 118.6, 157.9, 202.8, 253.3, 309.5, 371.2, 438.6,
        511.6, 590.2, 674.4, 764.2, 859.7, 960.7, 1067.0, 1180.0, 1298.0, 1421.0, 1550.0,
    ],
    // A wine glass: its (2,0) to (5,0) modes 1 : 2.3 : 4.2 : 6.6 (chosen by
    // ear between measured glasses and thin-ring theory), the rest a thin
    // ring's n (n^2 - 1) / sqrt(n^2 + 1) scaled on from there.
    [
        1.0, 2.3, 4.2, 6.6, 9.685, 13.33, 17.53, 22.3, 27.62, 33.52, 39.97, 46.98, 54.55, 62.68,
        71.37, 80.59, 90.45, 100.8, 111.7, 123.3, 135.3, 147.9, 161.1, 174.9,
    ],
    // A tuned church bell, relative to the prime (the hum an octave below
    // left out): prime, tierce (a minor third), quint, nominal (an octave)
    // and the higher partials, of the order the bell literature (Fletcher
    // and Rossing) gives for a tuned bell, rounded by us; above the twelfth
    // mode the spacing grows by 0.05 a mode.
    [
        1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166, 5.433, 6.796, 8.215, 9.687, 11.21,
        12.78, 14.4, 16.07, 17.8, 19.57, 21.39, 23.26, 25.18, 27.16, 29.18, 31.25,
    ],
    // A metal plate: the first eleven modes about those of a clamped
    // circular plate (Leissa 1969: lambda^2 = 10.22, 21.26, 34.88, 39.77,
    // ...), the rest evenly spaced, as a plate's grow about evenly.
    [
        1.0, 2.081, 3.414, 3.893, 4.996, 5.954, 6.819, 8.279, 8.722, 8.882, 10.87, 11.67, 12.47,
        13.27, 14.07, 14.87, 15.67, 16.47, 17.27, 18.07, 18.87, 19.67, 20.47, 21.27,
    ],
];

/// The body parameter of each table.
pub const HARMONIC: f32 = 0.0;
/// A marimba bar.
pub const BAR: f32 = 0.2;
/// A kalimba tine.
pub const TINE: f32 = 0.4;
/// A wine glass.
pub const GLASS_BODY: f32 = 0.6;
/// A church bell.
pub const BELL_BODY: f32 = 0.8;
/// A metal plate.
pub const PLATE: f32 = 1.0;

/// MODAL's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The body, 0 to 1 through the tables ([`HARMONIC`], [`BAR`],
    /// [`TINE`], [`GLASS_BODY`], [`BELL_BODY`], [`PLATE`]), the ratios
    /// between neighbours interpolated.
    pub body: f32,
    /// How long the lowest mode rings to -60 dB, 0 to 1: 0.02 s x
    /// 10^(4 decay), so 0 is 20 ms, 0.5 is 2 s, 1 is 200 s.
    pub decay: f32,
    /// How long the high modes ring against the lowest, 0 to 1: a mode k
    /// times its frequency rings k^(-2 (1 - brightness)) as long.
    pub brightness: f32,
    /// Where the body is struck, 0 to 1: mode k sounds as sin(pi k x), x
    /// from 0.02 (the edge: all modes) to 0.5 (the middle: the odd ones).
    pub position: f32,
    /// The mallet, 0 to 1: the contact lasts 0.2 ms x 2^(5 (1 - hardness)),
    /// from 6.4 ms (soft felt) to 0.2 ms (metal), at most 0.6 of a period.
    pub hardness: f32,
    /// A noise burst under the contact, 0 (none) to 1 (as much energy as
    /// the pulse).
    pub noise: f32,
    /// The modes in pairs this many hertz apart, beating (0: single modes,
    /// [`MODES`] of them; split, half as many pairs).
    pub split: f32,
}

impl Default for Params {
    fn default() -> Params {
        STRUCK
    }
}

/// [`Params::default`]: a harmonic body, struck plainly.
const STRUCK: Params = Params {
    body: HARMONIC,
    decay: 0.5,
    brightness: 0.5,
    position: 0.3,
    hardness: 0.6,
    noise: 0.0,
    split: 0.0,
};

/// A church bell.
pub const BELL: Params = Params {
    body: BELL_BODY,
    decay: 0.72,
    brightness: 0.5,
    position: 0.25,
    hardness: 0.7,
    ..STRUCK
};

/// A small, bright casino bell: struck hard and fast it rings on, its
/// pairs shimmering.
pub const SLOT_BELL: Params = Params {
    body: 0.85,
    decay: 0.45,
    brightness: 0.75,
    position: 0.2,
    hardness: 1.0,
    split: 3.0,
    ..STRUCK
};

/// A coin: a small plate, a hard hit and a little scrape.
pub const COIN: Params = Params {
    body: PLATE,
    decay: 0.38,
    brightness: 0.8,
    position: 0.15,
    hardness: 1.0,
    noise: 0.3,
    ..STRUCK
};

/// A wine glass: a few inharmonic modes ringing long, split a few tenths
/// of a hertz into slow beats, tapped hard enough for a tiny bright tick.
pub const GLASS: Params = Params {
    body: GLASS_BODY,
    decay: 0.7,
    brightness: 0.65,
    position: 0.3,
    hardness: 0.8,
    noise: 0.02,
    split: 0.3,
};

/// A kalimba tine.
pub const KALIMBA: Params = Params {
    body: TINE,
    decay: 0.5,
    brightness: 0.35,
    position: 0.45,
    hardness: 0.35,
    ..STRUCK
};

/// A marimba bar.
pub const MARIMBA: Params = Params {
    body: BAR,
    decay: 0.45,
    brightness: 0.4,
    position: 0.25,
    hardness: 0.35,
    ..STRUCK
};

/// What the host's shape moves: the body, by up to half its range.
pub const SHAPE: &str = "body";

/// What the host's brightness opens: the mallet, its contact shorter by an
/// octave an octave (at the strike), and the high modes' ring, a quarter of
/// the brightness's range an octave.
pub const BRIGHT: &str = "mallet / brightness";

/// The presets, by name.
pub const PRESETS: [(&str, Params); 6] = [
    ("BELL", BELL),
    ("SLOT BELL", SLOT_BELL),
    ("COIN", COIN),
    ("GLASS", GLASS),
    ("KALIMBA", KALIMBA),
    ("MARIMBA", MARIMBA),
];

/// One voice: the modes, their coefficients, the strike.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    /// The resonators' state: the rotating pair a mode.
    x: [f32; MODES],
    y: [f32; MODES],
    /// Each mode's amplitude from the strike position.
    gain: [f32; MODES],
    /// The coefficients r cos theta and r sin theta a mode, its output
    /// gain, and how many modes sound, for the inputs in `key` (none yet:
    /// `active` 0).
    rc: [f32; MODES],
    rs: [f32; MODES],
    out: [f32; MODES],
    active: usize,
    key: [u32; 6],
    /// The contact pulse: due at the next block, its length, the samples
    /// of it done, its level and its noise's, and the damping a sample
    /// while it lasts (1: none).
    strike: bool,
    contact: u32,
    elapsed: u32,
    level: f32,
    noise: f32,
    damp: f32,
    /// Struck while still ringing.
    ringing: bool,
    rng: u32,
    velocity: f32,
    quiet: Quiet,
}

/// MODAL at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Modal {
    rate: Rate,
}

/// The ratios of the body at `body`, between the two nearest tables.
fn ratios(body: f32) -> [f32; MODES] {
    let x = body.clamp(0.0, 1.0) * (BODIES.len() - 1) as f32;
    let i = (x as usize).min(BODIES.len() - 2);
    let t = x - i as f32;
    let (a, b) = (&BODIES[i], &BODIES[i + 1]);
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
}

impl Modal {
    /// The modes' coefficients for this block's pitch `f0` (cycles a
    /// sample), body and damping.
    fn tune(&self, p: &Params, v: &mut Voice, f0: f32, body: f32, bright: f32) {
        let key = [
            f0.to_bits(),
            body.to_bits(),
            bright.to_bits(),
            p.decay.to_bits(),
            p.brightness.to_bits(),
            p.split.to_bits(),
        ];
        if key == v.key && v.active > 0 {
            return;
        }
        v.key = key;
        let ratio = ratios(body);
        let ring = 0.02 * 10f32.powf(4.0 * p.decay.clamp(0.0, 1.0));
        let brightness = (p.brightness + 0.25 * bright).clamp(0.0, 1.0);
        let tilt = -2.0 * (1.0 - brightness);
        let split = p.split.max(0.0) / self.rate.hz();
        let paired = split > 0.0;
        // Each pair shares its mode's level (struck in phase, together they
        // start as the one mode would).
        let share = if paired { 0.5 } else { 1.0 };
        v.active = 0;
        for k in 0..MODES {
            let j = if paired { k / 2 } else { k };
            let f = f0 * ratio[j] + if paired && k % 2 == 1 { split } else { 0.0 };
            if f >= TOP {
                break;
            }
            let seconds = (ring * ratio[j].powf(tilt)).max(1e-3);
            // -60 dB over the ring time: ln(1000) = 6.9078.
            let r = (-6.907_755 / (seconds * self.rate.hz())).exp();
            let (s, c) = (TAU * f).sin_cos();
            v.rc[k] = r * c;
            v.rs[k] = r * s;
            // An even spread over many high modes would sum to a click far
            // louder than the ring: tilt them as a struck body radiates.
            v.out[k] = v.gain[j] / ratio[j].sqrt() * share;
            v.active = k + 1;
        }
    }

    /// Sets up the contact pulse at the strike.
    fn strike(&self, p: &Params, v: &mut Voice, f0: f32, bright: f32) {
        let hardness = p.hardness.clamp(0.0, 1.0);
        let seconds = 0.0002 * (5.0 * (1.0 - hardness) - bright).exp2() * (1.0 - 0.3 * v.velocity);
        // Never longer than 0.6 of a period: the lowest mode is always struck.
        let samples = (seconds * self.rate.hz()).min(0.6 / f0);
        v.contact = (samples.round() as u32).clamp(1, 4096);
        v.elapsed = 0;
        let length = v.contact as f32;
        // The half sine's samples sum to 1: the low modes start at their
        // gains, whatever the contact.
        v.level = PI / (2.0 * length);
        // The burst's energy: noise times the pulse's, pi^2 / (8 length)
        // against a windowed uniform noise's length / 6 a unit amplitude^2.
        v.noise = p.noise.clamp(0.0, 1.0).sqrt() * PI * 0.75f32.sqrt() / length;
        v.damp = if v.ringing {
            RESTRIKE.powf(1.0 / length)
        } else {
            1.0
        };
        v.strike = false;
    }
}

/// The modes `b..b + BATCH` rung through the block's `force`, damped by
/// `damp` a sample (`DAMPED`), into `y`.
#[inline]
fn ring<const DAMPED: bool>(v: &mut Voice, b: usize, force: &[f32], damp: &[f32], y: &mut [f32]) {
    let lanes = BATCH.min(v.active - b);
    // A lane past the last sounding mode has zero coefficients and gain.
    let mut rc = [0.0f32; BATCH];
    let mut rs = [0.0f32; BATCH];
    let mut a = [0.0f32; BATCH];
    rc[..lanes].copy_from_slice(&v.rc[b..b + lanes]);
    rs[..lanes].copy_from_slice(&v.rs[b..b + lanes]);
    a[..lanes].copy_from_slice(&v.out[b..b + lanes]);
    let mut xs: [f32; BATCH] = std::array::from_fn(|j| v.x[b + j]);
    let mut ys: [f32; BATCH] = std::array::from_fn(|j| v.y[b + j]);
    for (i, yi) in y.iter_mut().enumerate() {
        let (fi, g) = (force[i], if DAMPED { damp[i] } else { 1.0 });
        let mut sum = 0.0;
        for j in 0..BATCH {
            let x = (rc[j] * xs[j] - rs[j] * ys[j]) * g + fi;
            let yy = (rs[j] * xs[j] + rc[j] * ys[j]) * g;
            xs[j] = x;
            ys[j] = yy;
            sum += a[j] * yy;
        }
        *yi += sum * OUTPUT;
    }
    for j in 0..BATCH {
        v.x[b + j] = dsp::flush(xs[j]);
        v.y[b + j] = dsp::flush(ys[j]);
    }
}

impl Engine for Modal {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Modal {
        Modal { rate }
    }

    /// A strike: from silence, or on the ringing body (a retrigger keeps
    /// the modes ringing, damped by the strike as [`RESTRIKE`] says).
    fn note_on(&self, p: &Params, v: &mut Voice, n: &NoteOn) {
        let x = 0.02 + 0.48 * p.position.clamp(0.0, 1.0);
        for (k, g) in v.gain.iter_mut().enumerate() {
            *g = (PI * (k + 1) as f32 * x).sin();
        }
        // The same energy at every position.
        let rms = (v.gain.iter().map(|g| g * g).sum::<f32>() / MODES as f32).sqrt();
        let norm = 1.0 / rms.max(1e-3);
        for g in &mut v.gain {
            *g *= norm;
        }
        v.ringing = v.active > 0 && !v.quiet.done();
        v.velocity = n.velocity.clamp(0.0, 1.0);
        v.strike = true;
        v.rng = dsp::seed(n.seed);
        v.active = 0;
        v.quiet.restart();
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let n = out.len().min(BLOCK);
        if v.quiet.done() {
            return;
        }
        let f0 = m.phase_inc as f32 * (1.0 / 4_294_967_296.0);
        let body = (p.body + 0.5 * m.shape).clamp(0.0, 1.0);
        if v.strike {
            self.strike(p, v, f0, m.bright);
        }
        self.tune(p, v, f0, body, m.bright);

        // The force on the body this block, the half-sine contact with
        // noise under the same window, and the damping while it lasts.
        let mut force = [0.0f32; BLOCK];
        let mut damp = [1.0f32; BLOCK];
        let damped = v.damp < 1.0 && v.elapsed < v.contact;
        let length = v.contact as f32;
        for (f, d) in force[..n].iter_mut().zip(&mut damp[..n]) {
            if v.elapsed >= v.contact {
                break;
            }
            let window = (PI * (v.elapsed as f32 + 0.5) / length).sin();
            *f = window * (v.level + v.noise * dsp::white(&mut v.rng));
            *d = v.damp;
            v.elapsed += 1;
        }
        let mut y = [0.0f32; BLOCK];
        for b in (0..v.active).step_by(BATCH) {
            if damped {
                ring::<true>(v, b, &force[..n], &damp[..n], &mut y[..n]);
            } else {
                ring::<false>(v, b, &force[..n], &damp[..n], &mut y[..n]);
            }
        }
        v.quiet.update(&y[..n], v.elapsed < v.contact);
        for (o, s) in out[..n].iter_mut().zip(&y[..n]) {
            *o += s;
        }
    }

    fn done(&self, v: &Voice) -> bool {
        v.quiet.done()
    }
}
