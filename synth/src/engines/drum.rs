//! DRUM: synthesized drums, one voice a hit: kicks, snares, a clap, closed
//! and open hats, toms, a rim, a tick and a crash.
//!
//! Every [`Kind`] is made of the same few parts, in its own measure:
//!
//! - A body: one or two sine modes, each decaying exponentially, their
//!   pitch starting `bend` semitones up and falling back with the time
//!   constant `bend_time` (a kick's sweep, a tom's drop). The TR-808's bass
//!   drum is a band-pass network rung by a short trigger pulse, its pitch
//!   jumping up during the attack and sighing back; the kick here is that
//!   shape, not that circuit. A snare's two modes sit at an ideal
//!   membrane's first two, 1 : 1.593.
//! - Metal: the six square oscillators the TR-808's hats and cymbal share,
//!   at their nominal 205.3, 304.4, 369.6, 522.7, 540 and 800 Hz (scaled
//!   together by the tune) and their duty cycle of 47.98 %, summed through
//!   a band-pass (about 7.1 kHz for the hats, 3.4 kHz for the cymbal, as
//!   the circuit's two band-passes are centred) and a high-pass after the
//!   envelope.
//! - Noise: white noise through a high-pass (a kick's click, a rim's
//!   crack, the sizzle of the metal) or a band-pass (a snare's wires, a
//!   clap, a tom's skin), decaying over its own `snap_decay`. A clap is noise alone: four bursts
//!   8 to 10 ms apart, each falling over `snap_decay`, then a tail over
//!   `decay` from the last.
//!
//! Then a drive (a saturation blended in, so 0 leaves the sound as it is),
//! a soft knee and the level.
//!
//! The pitch is the voice's: `tune` is its frequency at middle C (note 60),
//! and the note played moves it a semitone a semitone, so a patch is played
//! on note 60 for its own tune (the host's pitch modulation moves it the
//! same way). The body's modes, the squares and the filters all follow it.
//!
//! What the host's [`VoiceMod`] does here: the brightness (`bright`,
//! octaves) moves the noise filter and the metal's band-pass and high-pass
//! by as many octaves (on a clap its band, on a kick its click), the shape
//! adds up to half of its range to `snappy`, and the pitch moves the tune
//! as above. The amplitude is rendered at unity; the host's level and
//! velocity scale it.
//!
//! A drum is a one-shot: its voice rings out by itself and says so
//! ([`Engine::done`]) once every part is below -80 dB, so a patch plays it
//! with a gate longer than its longest decay.
//!
//! Sources: the TR-808 bass drum's shape (a band-pass network rung by a
//! trigger pulse, a jump in pitch at the attack and a falling "sigh") after
//! K. J. Werner, J. S. Abel and J. O. Smith, "A Physically-Informed,
//! Circuit-Bendable, Digital Model of the Roland TR-808 Bass Drum Circuit"
//! (DAFx-14, Erlangen, 2014); the six oscillators' frequencies and duty
//! cycle, and the band-passes' centres, from the same authors' "The TR-808
//! Cymbal: a Physically-Informed, Circuit-Bendable, Digital Model"
//! (ICMC/SMC 2014, Athens); the drums' anatomy (a tuned body, a noise
//! component with its own envelope, metallic clusters of square waves)
//! after G. Reid's "Synth Secrets" in Sound On Sound ("Synthesizing Drums:
//! The Bass Drum", "Practical Bass Drum Synthesis", "Synthesizing Drums:
//! The Snare Drum", "Practical Snare Drum Synthesis", "Synthesizing
//! Realistic Cymbals" and "Practical Cymbal Synthesis", 2001-2002); the
//! membrane's mode ratio from N. H. Fletcher and T. D. Rossing, "The
//! Physics of Musical Instruments" (2nd ed., 1998); the squares by the
//! polyBLEP pulse and the filters by Simper's SVF, as [`dsp`] cites them.

use crate::dsp::{self, Rate, Svf, SvfState};
use crate::params::{field, options, pick, Param};
use crate::{Engine, NoteOn, VoiceMod};

/// What a voice plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A kick: a swept sine, a click and drive.
    Kick,
    /// A snare: two membrane modes and high-passed noise.
    Snare,
    /// A clap: band-passed noise in bursts, then a tail.
    Clap,
    /// A closed hat: the six squares, bright and short.
    ClosedHat,
    /// An open hat: the same, ringing on.
    OpenHat,
    /// A tom: a sine falling in pitch and a little noise.
    Tom,
    /// A rim: two short inharmonic modes and a click, driven.
    Rim,
    /// A crash or ride: the squares lower, with noise, long.
    Cymbal,
}

/// DRUM's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The voice.
    pub kind: Kind,
    /// Its frequency at note 60, Hz: a body's lowest mode, the metal's
    /// lowest square, a clap's band.
    pub tune: f32,
    /// How long the body (or the metal, or a clap's tail) rings to -60 dB,
    /// seconds.
    pub decay: f32,
    /// How far above the tune the body's pitch starts, semitones.
    pub bend: f32,
    /// How quickly it falls back, seconds (its time constant).
    pub bend_time: f32,
    /// The brightness of the noise and the metal, 0 to 1 (two octaves).
    pub tone: f32,
    /// The noise's share (a click, the wires, the sizzle; on a clap the
    /// bursts against the tail), 0 to 1.
    pub snappy: f32,
    /// How long the noise (a clap's each burst) rings to -60 dB, seconds.
    pub snap_decay: f32,
    /// The saturation, 0 (none) to 1.
    pub drive: f32,
    /// The level, 0 to 1.
    pub level: f32,
}

impl Default for Params {
    fn default() -> Params {
        KICK_808
    }
}

options! {
    /// The kinds, in their order, and their names.
    KINDS, KIND_NAMES: Kind = [
        Kind::Kick => "kick",
        Kind::Snare => "snare",
        Kind::Clap => "clap",
        Kind::ClosedHat => "closed hat",
        Kind::OpenHat => "open hat",
        Kind::Tom => "tom",
        Kind::Rim => "rim",
        Kind::Cymbal => "cymbal",
    ]
}

/// DRUM's parameters as data, with their ranges.
pub const PARAMS: &[Param<Params>] = &[
    pick!("drum", "kind", kind, KINDS, KIND_NAMES),
    field!("drum", "tune", tune, 20.0, 5_000.0, "Hz", Log),
    field!("drum", "decay", decay, 0.0, 6.0, "s", Fine),
    field!("drum", "level", level, 0.0, 1.0, "", Linear),
    field!("body", "bend", bend, 0.0, 48.0, "st", Linear),
    field!("body", "bend time", bend_time, 0.0, 0.2, "s", Fine),
    field!("noise", "tone", tone, 0.0, 1.0, "", Linear),
    field!("noise", "snappy", snappy, 0.0, 1.0, "", Linear),
    field!("noise", "snap decay", snap_decay, 0.0, 2.0, "s", Fine),
    field!("noise", "drive", drive, 0.0, 1.0, "", Linear),
];

/// The parts a kind is made of, against its tune.
struct Recipe {
    /// The body's modes: ratio to the tune, level, and ring time against
    /// the decay (a level of 0: none).
    modes: [(f32, f32, f32); 2],
    /// The squares' level (0: none), their band-pass's centre against the
    /// tune, and the high-pass after it against that centre.
    metal: f32,
    band: f32,
    high: f32,
    /// The noise's level at full `snappy`, its filter (band-pass, else
    /// high-pass), centre against the tune and damping.
    noise: f32,
    band_pass: bool,
    at: f32,
    k: f32,
}

/// The TR-808's six oscillators against the lowest: 205.3, 304.4, 369.6,
/// 522.7, 540 and 800 Hz.
const SQUARES: [f32; 6] = [
    1.0,
    304.4 / 205.3,
    369.6 / 205.3,
    522.7 / 205.3,
    540.0 / 205.3,
    800.0 / 205.3,
];

/// Their duty cycle, 47.98 %, as a pulse width (2^32 a cycle).
const DUTY: u32 = (0.4798 * 4_294_967_296.0) as u32;

/// The level that brings the squares' band to about the body's.
const METAL_GAIN: f32 = 5.0;

/// The clap's bursts, seconds from the hit.
const BURSTS: [f32; 4] = [0.0, 0.0085, 0.0175, 0.027];

/// Below this (-80 dB) a part is silent.
const FLOOR: f32 = 1e-4;

const fn recipe(kind: Kind) -> Recipe {
    const NONE: Recipe = Recipe {
        modes: [(1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
        metal: 0.0,
        band: 1.0,
        high: 1.0,
        noise: 0.0,
        band_pass: false,
        at: 1.0,
        k: 1.4,
    };
    match kind {
        Kind::Kick => Recipe {
            modes: [(1.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
            noise: 0.6,
            at: 60.0,
            ..NONE
        },
        Kind::Snare => Recipe {
            modes: [(1.0, 0.65, 1.0), (1.593, 0.35, 0.7)],
            noise: 0.8,
            band_pass: true,
            at: 25.0,
            k: 1.8,
            ..NONE
        },
        Kind::Clap => Recipe {
            noise: 3.0,
            band_pass: true,
            k: 0.7,
            ..NONE
        },
        Kind::ClosedHat | Kind::OpenHat => Recipe {
            metal: 1.0,
            band: 7_100.0 / 205.3,
            high: 0.9,
            noise: 0.4,
            at: 40.0,
            ..NONE
        },
        Kind::Tom => Recipe {
            modes: [(1.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
            noise: 0.5,
            band_pass: true,
            at: 6.0,
            k: 1.0,
            ..NONE
        },
        Kind::Rim => Recipe {
            modes: [(1.0, 0.6, 1.0), (2.75, 0.4, 0.8)],
            noise: 0.8,
            at: 6.0,
            ..NONE
        },
        Kind::Cymbal => Recipe {
            metal: 1.0,
            band: 3_440.0 / 205.3,
            high: 0.7,
            noise: 0.6,
            at: 40.0,
            ..NONE
        },
    }
}

/// The 808-style kick: deep and long, a gentle sweep, a soft click.
pub const KICK_808: Params = Params {
    kind: Kind::Kick,
    tune: 49.0,
    decay: 1.2,
    bend: 7.0,
    bend_time: 0.03,
    tone: 0.4,
    snappy: 0.15,
    snap_decay: 0.004,
    drive: 0.1,
    level: 1.0,
};

/// A punchy kick: shorter, a fast deep sweep, a harder click, driven.
pub const KICK: Params = Params {
    tune: 55.0,
    decay: 0.35,
    bend: 30.0,
    bend_time: 0.012,
    tone: 0.6,
    snappy: 0.4,
    snap_decay: 0.006,
    drive: 0.45,
    ..KICK_808
};

/// The game's downbeat thump under the slam and the rung hits: low, a
/// sweep from two octaves up, a click and drive.
pub const IMPACT: Params = Params {
    tune: 41.0,
    decay: 0.9,
    bend: 26.0,
    bend_time: 0.02,
    tone: 0.35,
    snappy: 0.35,
    snap_decay: 0.008,
    drive: 0.6,
    ..KICK_808
};

/// A snare.
pub const SNARE: Params = Params {
    kind: Kind::Snare,
    tune: 185.0,
    decay: 0.18,
    bend: 4.0,
    bend_time: 0.012,
    tone: 0.55,
    snappy: 0.75,
    snap_decay: 0.22,
    drive: 0.1,
    level: 1.0,
};

/// A hand clap.
pub const CLAP: Params = Params {
    kind: Kind::Clap,
    tune: 1_100.0,
    decay: 0.3,
    bend: 0.0,
    bend_time: 0.01,
    tone: 0.5,
    snappy: 0.6,
    snap_decay: 0.009,
    drive: 0.0,
    level: 1.0,
};

/// A closed hat.
pub const HAT: Params = Params {
    kind: Kind::ClosedHat,
    tune: 205.3,
    decay: 0.06,
    bend: 0.0,
    bend_time: 0.01,
    tone: 0.5,
    snappy: 0.1,
    snap_decay: 0.04,
    drive: 0.0,
    level: 1.0,
};

/// An open hat.
pub const OPEN_HAT: Params = Params {
    kind: Kind::OpenHat,
    decay: 0.45,
    snap_decay: 0.3,
    ..HAT
};

/// A high tom.
pub const TOM_HIGH: Params = Params {
    kind: Kind::Tom,
    tune: 196.0,
    decay: 0.35,
    bend: 5.0,
    bend_time: 0.06,
    tone: 0.5,
    snappy: 0.15,
    snap_decay: 0.05,
    drive: 0.1,
    level: 1.0,
};

/// A low tom.
pub const TOM_LOW: Params = Params {
    tune: 98.0,
    decay: 0.55,
    bend_time: 0.08,
    ..TOM_HIGH
};

/// A rim shot.
pub const RIM: Params = Params {
    kind: Kind::Rim,
    tune: 480.0,
    decay: 0.03,
    bend: 0.0,
    bend_time: 0.01,
    tone: 0.6,
    snappy: 0.5,
    snap_decay: 0.012,
    drive: 0.5,
    level: 1.0,
};

/// A short bright tick, for a countdown or a cursor.
pub const TICK: Params = Params {
    tune: 1_500.0,
    decay: 0.012,
    tone: 0.9,
    snappy: 0.6,
    snap_decay: 0.006,
    drive: 0.2,
    ..RIM
};

/// A crash.
pub const CRASH: Params = Params {
    kind: Kind::Cymbal,
    tune: 205.3,
    decay: 2.2,
    bend: 0.0,
    bend_time: 0.01,
    tone: 0.6,
    snappy: 0.3,
    snap_decay: 1.2,
    drive: 0.0,
    level: 1.0,
};

/// What the host's shape moves: `snappy`, by up to half its range.
pub const SHAPE: &str = "snappy";

/// What the host's brightness opens: the noise filter and the metal's
/// band, an octave an octave.
pub const BRIGHT: &str = "noise filter / metal band";

/// The presets, by name.
///
/// On every one the shape adds to `snappy` (more click on a kick, more
/// wires on a snare, sharper bursts on a clap, more sizzle on the hats and
/// the crash, more skin on a tom, more crack on a rim) and the brightness
/// moves the noise's filter (and on the hats and the crash the metal's
/// band) by its octaves.
pub const PRESETS: [(&str, Params); 12] = [
    ("IMPACT", IMPACT),
    ("KICK 808", KICK_808),
    ("KICK", KICK),
    ("SNARE", SNARE),
    ("CLAP", CLAP),
    ("HAT", HAT),
    ("OPEN HAT", OPEN_HAT),
    ("TOM HIGH", TOM_HIGH),
    ("TOM LOW", TOM_LOW),
    ("RIM", RIM),
    ("TICK", TICK),
    ("CRASH", CRASH),
];

/// The slam: a kick, a clap and a crash cued on one frame, one voice each
/// (three patches, the same note 60): together one hit.
pub const SLAM: [(&str, Params); 3] = [
    (
        "SLAM KICK",
        Params {
            snappy: 0.5,
            drive: 0.7,
            ..IMPACT
        },
    ),
    (
        "SLAM CLAP",
        Params {
            tone: 0.65,
            decay: 0.4,
            level: 0.8,
            ..CLAP
        },
    ),
    (
        "SLAM CRASH",
        Params {
            decay: 1.8,
            tone: 0.7,
            level: 0.7,
            ..CRASH
        },
    ),
];

/// One DRUM voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    /// Samples since the hit.
    t: u32,
    /// The body's modes: phases and envelopes.
    phase: [u32; 2],
    body: [f32; 2],
    /// The squares' phases (free-running, kept on a retrigger) and their
    /// envelope.
    squares: [u32; 6],
    metal: f32,
    /// The noise's envelope (a clap's bursts), a clap's tail, and the
    /// bursts begun.
    noise: f32,
    tail: f32,
    bursts: usize,
    /// The noise's filter and the metal's two.
    filter: SvfState,
    band: SvfState,
    high: SvfState,
    rng: u32,
    /// Rung out.
    ended: bool,
}

/// DRUM at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Drum {
    rate: Rate,
}

impl Drum {
    /// What an envelope keeps a sample to fall 60 dB in `seconds`.
    fn fall(&self, seconds: f32) -> f32 {
        (-6.907_755 / (seconds.max(1e-3) * self.rate.hz())).exp()
    }

    /// The body's pitch factor `t` samples into the hit.
    fn bend(&self, p: &Params, t: u32) -> f32 {
        let tau = p.bend_time.max(1e-4) * self.rate.hz();
        (p.bend * (-(t as f32) / tau).exp() / 12.0).exp2()
    }

    /// One block of `v` at unity, added to `out`.
    fn hit(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let r = recipe(p.kind);
        let n = out.len();
        let key = p.tune.max(1.0) * ((m.pitch - 60.0) / 12.0).exp2();
        let tone = 2.0 * (p.tone.clamp(0.0, 1.0) - 0.5) + m.bright;
        let snappy = (p.snappy + 0.5 * m.shape).clamp(0.0, 1.0);
        let top = 0.45 * self.rate.hz();

        // The body's increments at the block's ends, the sweep linear
        // between them.
        let (b0, b1) = (self.bend(p, v.t), self.bend(p, v.t + n as u32));
        let base = self.rate.inc_hz(key) as f32;
        let mut inc = [0.0f32; 2];
        let mut step = [0.0f32; 2];
        let mut keep = [0.0f32; 2];
        for j in 0..2 {
            let (ratio, _, ring) = r.modes[j];
            inc[j] = (base * ratio * b0).min(0.45 * 4_294_967_296.0);
            step[j] = ((base * ratio * b1).min(0.45 * 4_294_967_296.0) - inc[j]) / n as f32;
            keep[j] = self.fall(p.decay * ring);
        }
        let gains = [r.modes[0].1, r.modes[1].1];

        // The metal: its squares, band and high-pass.
        let metal = r.metal > 0.0;
        let squares: [u32; 6] = std::array::from_fn(|k| self.rate.inc_hz(key * SQUARES[k]));
        let centre = (key * r.band * tone.exp2()).min(top);
        let band = Svf::new(centre, 1.0, self.rate);
        let high = Svf::new(centre * r.high, 1.2, self.rate);
        let metal_keep = self.fall(p.decay);

        // The noise: its filter and envelopes.
        let noise_level = r.noise * snappy;
        let filter = Svf::new((key * r.at * tone.exp2()).min(top), r.k, self.rate);
        let noise_keep = self.fall(p.snap_decay);
        let tail_keep = self.fall(p.decay);
        let tail_level = 1.0 - 0.6 * snappy;
        let clap = p.kind == Kind::Clap;
        let next_burst = |b: usize| (BURSTS[b] * self.rate.hz()) as u32;

        let drive = p.drive.clamp(0.0, 1.0);
        let push = 1.0 + 7.0 * drive;
        let pushed = dsp::softclip(push);
        let level = p.level.clamp(0.0, 1.0);

        for o in out.iter_mut() {
            let mut y = 0.0;
            for j in 0..2 {
                if gains[j] > 0.0 {
                    y += dsp::sine(v.phase[j]) * gains[j] * v.body[j];
                    v.phase[j] = v.phase[j].wrapping_add(inc[j] as u32);
                    inc[j] += step[j];
                    v.body[j] *= keep[j];
                }
            }
            if metal {
                let mut sum = 0.0;
                for (ph, &i) in v.squares.iter_mut().zip(&squares) {
                    sum += dsp::pulse(*ph, i, DUTY);
                    *ph = ph.wrapping_add(i);
                }
                let b = band.run(&mut v.band, sum * (1.0 / 6.0)).bp;
                let h = high.run(&mut v.high, b * v.metal).hp;
                y += h * METAL_GAIN * r.metal;
                v.metal *= metal_keep;
            }
            if noise_level > 0.0 {
                if clap && v.bursts < BURSTS.len() && v.t >= next_burst(v.bursts) {
                    v.noise = 1.0;
                    v.bursts += 1;
                    if v.bursts == BURSTS.len() {
                        v.tail = tail_level;
                    }
                }
                let taps = filter.run(&mut v.filter, dsp::white(&mut v.rng));
                let x = if r.band_pass { taps.bp * r.k } else { taps.hp };
                y += x * (v.noise + v.tail) * noise_level;
                v.noise *= noise_keep;
                v.tail *= tail_keep;
            }
            if drive > 0.0 {
                y += drive * (dsp::softclip(push * y) / pushed - y);
            }
            *o += dsp::soft_knee(y, 0.8) * level;
            v.t = v.t.saturating_add(1);
        }
        for e in v
            .body
            .iter_mut()
            .chain([&mut v.metal, &mut v.noise, &mut v.tail])
        {
            *e = dsp::flush(*e);
        }
        let loud = (v.body[0] * gains[0])
            .max(v.body[1] * gains[1])
            .max(v.metal * r.metal)
            .max((v.noise + v.tail) * noise_level);
        let pending = clap && v.bursts < BURSTS.len();
        v.ended = loud < FLOOR && !pending;
    }
}

impl Engine for Drum {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Drum {
        Drum { rate }
    }

    /// Strikes the drum: every envelope at full, the body's modes from
    /// phase 0, a clap's bursts from the first. A retrigger (a mono patch
    /// played again) strikes it the same way, but the filters ring on and
    /// the squares keep their phases (they run free, as the 808's do); on a
    /// voice from silence the squares start from phases drawn from the
    /// seed.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        v.rng = dsp::seed(n.seed);
        if v.t == 0 {
            for ph in &mut v.squares {
                *ph = jodel_dsp::xorshift32(&mut v.rng);
            }
        }
        v.t = 0;
        v.phase = [0; 2];
        v.body = [1.0; 2];
        v.metal = 1.0;
        v.noise = 1.0;
        v.tail = 0.0;
        v.bursts = 0;
        v.ended = false;
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        if v.ended {
            return;
        }
        self.hit(p, v, m, out);
    }

    fn done(&self, v: &Voice) -> bool {
        v.ended
    }
}
