//! A patch: what a note on it sounds like. An engine and its parameters,
//! an ADSR (and how far it brightens the sound), glides of the pitch and the
//! shape, an LFO, its level, how long a note holds, its sends to the
//! shared reverb, delay and chorus, and the random spreads each note draws from
//! its seed.
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
//!     ..Patch::new(lofi::Params::default())
//! };
//! # let _ = coin;
//! ```

use crate::engines::AnyParams;

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
        Adsr {
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.1,
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
        Glide {
            from: 0.0,
            time: 0.0,
            curve: 1.0,
        }
    }
}

impl Glide {
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
        Lfo {
            rate: LfoRate::Hz(5.0),
            pitch: 0.0,
            bright: 0.0,
            amp: 0.0,
            shape: 0.0,
            delay: 0.0,
            fade: 0.0,
        }
    }
}

/// How much each note of a patch differs from the next, drawn from its
/// seed (so the same seed, the same note): each up to the amount either
/// way, the start only later.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
}

impl Patch {
    /// A patch of `engine`: no envelopes beyond a plain gate (held 0.25 s,
    /// released over 0.1 s), no LFO, no sends, no spreads, at 0 dB.
    pub fn new(engine: impl Into<AnyParams>) -> Patch {
        Patch {
            engine: engine.into(),
            adsr: Adsr::default(),
            filter_env: 0.0,
            pitch: Glide::default(),
            shape: Glide::default(),
            lfo: Lfo::default(),
            level: 0.0,
            gate: 0.25,
            reverb: 0.0,
            delay: 0.0,
            chorus: 0.0,
            spread: Spread::default(),
            mono: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
