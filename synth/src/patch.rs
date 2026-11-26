//! A patch: what a note on it sounds like. An engine and its parameters,
//! an ADSR (and how far it brightens the sound), glides of the pitch and the
//! shape, an LFO, its level, how long a note holds, its sends to the
//! shared reverb, delay and chorus, the random spreads each note draws from
//! its seed, and whether it plays one voice at a time (sliding from note to
//! note).
//!
//! A patch is plain data, built with struct update syntax from
//! [`Patch::new`]:
//!
//! ```
//! use jodel_synth::engines::lofi;
//! use jodel_synth::patch::{Adsr, Patch};
//!
//! let coin = Patch {
//!     adsr: Adsr { decay: 0.4, sustain: 0.0, ..Adsr::default() },
//!     reverb: 0.2,
//!     ..Patch::new(lofi::Params::default().into())
//! };
//! # let _ = coin;
//! ```

use crate::engines::AnyParams;
use crate::params::{self, field, on, switch, when, Knobs, Param, Scale, Spec};

/// An ADSR: a linear attack, an exponential decay to the sustain level,
/// an exponential release; times in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adsr {
    /// The rise from silence (or from where a retrigger finds it) to full.
    pub attack: f32,
    /// The fall from full to the sustain level (to within 60 dB of it).
    pub decay: f32,
    /// The level held while the note is, 0 to 1. At 0 the note ends once
    /// its decay is over.
    pub sustain: f32,
    /// The fall to silence (60 dB down) after the note is let go, when the
    /// note ends.
    pub release: f32,
}

impl Default for Adsr {
    fn default() -> Adsr {
        Adsr::GATE
    }
}

impl Adsr {
    /// A plain gate: full at once, held, released over 0.1 s.
    pub const GATE: Adsr = Adsr {
        attack: 0.0,
        decay: 0.0,
        sustain: 1.0,
        release: 0.1,
    };

    /// Struck: no attack, a decay of `decay` seconds to silence (a short
    /// release, should it be let go first).
    pub const fn hit(decay: f32) -> Adsr {
        Adsr {
            attack: 0.0,
            decay,
            sustain: 0.0,
            release: 0.05,
        }
    }

    /// Held: a rise of `attack` seconds to full, held there until let go,
    /// a release of `release`.
    pub const fn held(attack: f32, release: f32) -> Adsr {
        Adsr {
            attack,
            decay: 0.0,
            sustain: 1.0,
            release,
        }
    }
}

/// A glide into place: a note starts `from` away from where it sits (its
/// pitch, in semitones, or its shape) and arrives there after `time`
/// seconds, as (1 - t / time)^`curve` (1 linear, above 1 quickly then
/// slowly, below 1 slowly then quickly). For the pitch: up from below for
/// a riser or an up-chirp, down from above for a "pew"; for the shape, a
/// vowel sliding into another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glide {
    /// Where the note starts, from where it sits.
    pub from: f32,
    /// How long it takes to arrive, seconds.
    pub time: f32,
    /// The curve of the way there.
    pub curve: f32,
}

impl Default for Glide {
    fn default() -> Glide {
        Glide::NONE
    }
}

impl Glide {
    /// None: the note starts where it sits.
    pub const NONE: Glide = Glide {
        from: 0.0,
        time: 0.0,
        curve: 1.0,
    };

    /// The offset `t` seconds into the note.
    pub fn at(&self, t: f32) -> f32 {
        if self.from == 0.0 || self.time <= 0.0 {
            return 0.0;
        }
        self.from * (1.0 - (t / self.time).min(1.0)).powf(self.curve)
    }
}

/// How fast an LFO runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LfoRate {
    /// Cycles a second.
    Hz(f32),
    /// Beats a cycle, at the tempo the synth is told.
    Beats(f32),
}

/// A sine LFO, a voice's own (it starts with the note, after a delay and
/// fading in, as a singer's vibrato does), and how deep it moves the
/// pitch, the brightness, the amplitude and the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lfo {
    /// Its rate.
    pub rate: LfoRate,
    /// Semitones up and down (a vibrato).
    pub pitch: f32,
    /// Octaves of brightness up and down.
    pub bright: f32,
    /// How far it dips the amplitude, 0 to 1 (a tremolo).
    pub amp: f32,
    /// How far it moves the shape either way (the shape runs from -1 to
    /// 1).
    pub shape: f32,
    /// Seconds after the note starts before it moves anything.
    pub delay: f32,
    /// Seconds it then takes to come in fully.
    pub fade: f32,
}

impl Default for Lfo {
    fn default() -> Lfo {
        Lfo::NONE
    }
}

impl Lfo {
    /// None: at 5 Hz, moving nothing.
    pub const NONE: Lfo = Lfo {
        rate: LfoRate::Hz(5.0),
        pitch: 0.0,
        bright: 0.0,
        amp: 0.0,
        shape: 0.0,
        delay: 0.0,
        fade: 0.0,
    };
}

/// How much each note of a patch differs from the next, drawn from its
/// seed (so the same seed, the same note): each up to the amount either
/// way, the start only later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    /// Detune, cents either way.
    pub detune: f32,
    /// Level, dB either way.
    pub level: f32,
    /// A later start, up to this many seconds.
    pub start: f32,
    /// Where in its cycle the LFO starts, up to this share of a cycle (1:
    /// anywhere; voices of a chord then wobble apart).
    pub lfo_phase: f32,
}

impl Default for Spread {
    fn default() -> Spread {
        Spread::NONE
    }
}

impl Spread {
    /// None: every note the same.
    pub const NONE: Spread = Spread {
        detune: 0.0,
        level: 0.0,
        start: 0.0,
        lfo_phase: 0.0,
    };
}

/// What a note on it sounds like.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch {
    /// The engine and its parameters.
    pub engine: AnyParams,
    /// The amplitude's envelope.
    pub adsr: Adsr,
    /// How far the envelope at full brightens the sound, octaves (see
    /// [`crate::VoiceMod::bright`]).
    pub filter_env: f32,
    /// The pitch's glide, semitones.
    pub pitch: Glide,
    /// The shape's glide (-1 to 1, as the shape's modulation).
    pub shape: Glide,
    /// The LFO.
    pub lfo: Lfo,
    /// The level, dB.
    pub level: f32,
    /// How long a note is held before it is let go, seconds (unless the
    /// note is cued with a length of its own).
    pub gate: f32,
    /// The send to the shared reverb (the hall), 0 to 1.
    pub reverb: f32,
    /// The send to the shared delay, 0 to 1.
    pub delay: f32,
    /// The send to the shared chorus, 0 to 1 (its return comes on top of
    /// the dry sound).
    pub chorus: f32,
    /// The random spreads.
    pub spread: Spread,
    /// One voice at a time: a new note retriggers the sounding one on its
    /// frame, keeping what its engine keeps (see each engine's
    /// `note_on`).
    pub mono: bool,
    /// A mono patch's portamento: seconds a retrigger takes to slide its
    /// pitch and its note's shape from where they were to the new note's,
    /// in a straight line (0: at once). So a voice sings legato: one
    /// syllable's pitch and vowel gliding into the next's.
    pub slide: f32,
}

impl Patch {
    /// A patch of `engine`: no envelopes beyond a plain gate (held 0.25 s,
    /// released over 0.1 s), no LFO, no sends, no spreads, at 0 dB.
    pub const fn new(engine: AnyParams) -> Patch {
        Patch {
            engine,
            adsr: Adsr::GATE,
            filter_env: 0.0,
            pitch: Glide::NONE,
            shape: Glide::NONE,
            lfo: Lfo::NONE,
            level: 0.0,
            gate: 0.25,
            reverb: 0.0,
            delay: 0.0,
            chorus: 0.0,
            spread: Spread::NONE,
            mono: false,
            slide: 0.0,
        }
    }
}

/// An LFO's rate in cycles a second (one in beats as at the synth's
/// starting tempo, 120 a minute).
fn lfo_hz(r: LfoRate) -> f32 {
    match r {
        LfoRate::Hz(hz) => hz,
        LfoRate::Beats(beats) => 2.0 / beats.max(1e-3),
    }
}

/// An LFO's rate in beats a cycle (one in hertz as at 120 a minute).
fn lfo_beats(r: LfoRate) -> f32 {
    match r {
        LfoRate::Hz(hz) => 2.0 / hz.max(1e-3),
        LfoRate::Beats(beats) => beats,
    }
}

/// A patch's own parameters as data, with their ranges (its engine's are
/// the engine's: [`AnyParams`] as [`Knobs`]).
pub const PARAMS: &[Param<Patch>] = &[
    field!("envelope", "attack", adsr.attack, 0.0, 4.0, "s", Fine),
    field!("envelope", "decay", adsr.decay, 0.0, 8.0, "s", Fine),
    field!("envelope", "sustain", adsr.sustain, 0.0, 1.0, "", Linear),
    field!("envelope", "release", adsr.release, 0.0, 8.0, "s", Fine),
    field!("voice", "level", level, -48.0, 12.0, "dB", Linear),
    field!("voice", "gate", gate, 0.0, 8.0, "s", Fine),
    field!("voice", "brighten", filter_env, -6.0, 8.0, "oct", Linear),
    switch!("voice", "mono", mono),
    when(field!("voice", "slide", slide, 0.0, 1.0, "s", Fine), |p| {
        p.mono
    }),
    field!("pitch", "from", pitch.from, -48.0, 48.0, "st", Linear),
    field!("pitch", "time", pitch.time, 0.0, 4.0, "s", Fine),
    field!("pitch", "curve", pitch.curve, 0.1, 8.0, "", Log),
    field!("shape", "from", shape.from, -2.0, 2.0, "", Linear),
    field!("shape", "time", shape.time, 0.0, 4.0, "s", Fine),
    field!("shape", "curve", shape.curve, 0.1, 8.0, "", Log),
    Param::toggle(
        ("lfo", "synced"),
        |p| on(matches!(p.lfo.rate, LfoRate::Beats(_))),
        |p, v| {
            let r = p.lfo.rate;
            p.lfo.rate = if v >= 0.5 {
                LfoRate::Beats(lfo_beats(r))
            } else {
                LfoRate::Hz(lfo_hz(r))
            };
        },
    ),
    when(
        Param::range(
            ("lfo", "rate"),
            (0.05, 40.0, "Hz", Scale::Log),
            |p| lfo_hz(p.lfo.rate),
            |p, v| p.lfo.rate = LfoRate::Hz(v),
        ),
        |p| matches!(p.lfo.rate, LfoRate::Hz(_)),
    ),
    when(
        Param::range(
            ("lfo", "beats"),
            (0.0625, 32.0, "beats", Scale::Log),
            |p| lfo_beats(p.lfo.rate),
            |p, v| p.lfo.rate = LfoRate::Beats(v),
        ),
        |p| matches!(p.lfo.rate, LfoRate::Beats(_)),
    ),
    field!("lfo", "pitch", lfo.pitch, 0.0, 12.0, "st", Fine),
    field!("lfo", "bright", lfo.bright, 0.0, 4.0, "oct", Linear),
    field!("lfo", "amp", lfo.amp, 0.0, 1.0, "", Linear),
    field!("lfo", "shape", lfo.shape, 0.0, 2.0, "", Linear),
    field!("lfo", "delay", lfo.delay, 0.0, 4.0, "s", Fine),
    field!("lfo", "fade", lfo.fade, 0.0, 4.0, "s", Fine),
    field!("sends", "reverb", reverb, 0.0, 1.0, "", Linear),
    field!("sends", "delay", delay, 0.0, 1.0, "", Linear),
    field!("sends", "chorus", chorus, 0.0, 1.0, "", Linear),
    field!("spread", "detune", spread.detune, 0.0, 100.0, "ct", Fine),
    field!("spread", "level", spread.level, 0.0, 12.0, "dB", Linear),
    field!("spread", "start", spread.start, 0.0, 0.2, "s", Fine),
    field!(
        "spread",
        "lfo phase",
        spread.lfo_phase,
        0.0,
        1.0,
        "",
        Linear
    ),
];

/// A patch's parameters: its own ([`PARAMS`]) first, then its engine's.
impl Knobs for Patch {
    fn count(&self) -> usize {
        PARAMS.len() + self.engine.count()
    }

    fn spec(&self, k: usize) -> Option<Spec> {
        match k.checked_sub(PARAMS.len()) {
            None => params::spec_of(PARAMS, k),
            Some(e) => self.engine.spec(e),
        }
    }

    fn applies(&self, k: usize) -> bool {
        match k.checked_sub(PARAMS.len()) {
            None => params::applies_to(PARAMS, self, k),
            Some(e) => self.engine.applies(e),
        }
    }

    fn get(&self, k: usize) -> Option<f32> {
        match k.checked_sub(PARAMS.len()) {
            None => params::get_in(PARAMS, self, k),
            Some(e) => self.engine.get(e),
        }
    }

    fn set(&mut self, k: usize, v: f32) -> bool {
        match k.checked_sub(PARAMS.len()) {
            None => params::set_in(PARAMS, self, k, v),
            Some(e) => self.engine.set(e, v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every parameter of every engine's default, set to its value, is as
    /// it was; read back after a set, it is what was set (clamped); each
    /// key is a patch's once.
    #[test]
    fn every_parameter_reads_back_what_it_is_set_to() {
        for name in AnyParams::ENGINES {
            let mut p = Patch::new(AnyParams::of_engine(name).unwrap());
            let mut keys: Vec<String> = (0..p.count()).map(|k| p.spec(k).unwrap().key()).collect();
            let n = keys.len();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), n, "{name}: a key twice");
            for k in 0..p.count() {
                if !p.applies(k) {
                    continue;
                }
                let before = p;
                let v = p.get(k).unwrap();
                assert!(p.set(k, v));
                assert_eq!(p, before, "{name} {:?}", p.spec(k));
                let to = p.spec(k).unwrap().kind.at(0.3);
                p.set(k, to);
                if p.applies(k) {
                    let got = p.get(k).unwrap();
                    assert!((got - to).abs() <= 1e-3 * to.abs().max(1.0), "{name} {k}");
                }
                p = before;
            }
            assert!(!p.set(p.count(), 0.0));
        }
    }

    #[test]
    fn a_glide_arrives() {
        let p = Glide {
            from: 24.0,
            time: 0.1,
            curve: 2.0,
        };
        assert_eq!(p.at(0.0), 24.0);
        assert!((p.at(0.05) - 6.0).abs() < 1e-4);
        assert_eq!(p.at(0.2), 0.0);
    }
}
