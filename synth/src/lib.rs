//! A small real-time synthesizer for the karaoke game's sound effects.
//!
//! The sounds come from our own engines, each written from public
//! descriptions of its technique (named in its module): [`engines::noise`]
//! (noise from hiss to a chip's register, through a resonant filter),
//! [`engines::analog`] (two band-limited oscillators, drive and a
//! state-variable filter), [`engines::lofi`] (a chip voice),
//! [`engines::phase`] (phase distortion), [`engines::modal`] (a struck
//! body's modes), [`engines::string`] (a plucked string),
//! [`engines::formant`] (a sung or whispered vowel), [`engines::fm`] (four
//! operators modulating one another's phase), [`engines::supersaw`] (a
//! stack of detuned saws) and [`engines::drum`] (synthesized drums). They
//! share the building blocks in [`dsp`].
//!
//! The host ([`Synth`]) plays them on a pool of voices whose size it is
//! given once, each voice on a [`Patch`] (an engine and its parameters, an
//! ADSR, glides of the pitch and the shape, an LFO, sends to the shared
//! [`hall::Hall`], [`delay::Delay`] and [`chorus::Chorus`], and random
//! spreads drawn from each note's seed). Its bus ends in a compressor and
//! a limiter ([`dynamics`]), so a pile of sounds never clips. Notes are cued on the frame they are to start on
//! ([`Synth::note`]), so a conductor can land them exactly on beat frames.
//!
//! Everything runs in blocks of [`BLOCK`] samples: the engines take their
//! modulation once a block ([`VoiceMod`]) and render at unity level; the
//! host ramps each voice's amplitude linearly across the block, so nothing
//! clicks and the cost a sample stays low. After construction nothing
//! allocates: the voices, the cues and the effects' lines are all fixed in
//! size, so the synth can render on an audio thread. Rendering offline
//! ([`Synth::render_to`]) is the same code.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod chorus;
pub mod delay;
pub mod dsp;
pub mod dynamics;
pub mod engines;
mod env;
pub mod hall;
mod host;
pub mod patch;

pub use host::{Bank, PatchId, Synth, VoiceView, CUES, GENTLE};
pub use patch::Patch;

use dsp::Rate;

/// The samples of a block: the engines take their modulation once a block.
pub const BLOCK: usize = 32;

/// What the host tells an engine's voice for one block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceMod {
    /// The phase increment a sample of the voice's pitch (2^32 a cycle).
    pub phase_inc: u32,
    /// The same pitch as a MIDI note (fractional; the pitch glide, the LFO
    /// and the note's detune in it).
    pub pitch: f32,
    /// The brightness, in octaves either way (the patch's filter envelope
    /// and LFO): each engine opens what makes it brighter by it, as its
    /// module's `BRIGHT` says (a filter's cutoff, a tone control, the
    /// phase distortion's depth, a mallet's hardness).
    pub bright: f32,
    /// The shape's modulation, -1 to 1 (0: the parameters' own shape),
    /// moving what its module's `SHAPE` says.
    pub shape: f32,
}

/// A note starting on a voice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoteOn {
    /// Its pitch as a MIDI note (60 is middle C).
    pub note: f32,
    /// Its velocity, 0 to 1.
    pub velocity: f32,
    /// Its own random seed: the same seed, the same sound.
    pub seed: u32,
    /// Where it sits, -1 left to 1 right.
    pub pan: f32,
}

/// A sound engine: its parameters, the state of one of its voices, and
/// how that voice starts and sounds.
///
/// An engine is made for one sample rate and keeps what follows from it
/// (and, for an engine that needs one, a fixed pool its voices borrow
/// from); what a voice needs is in its own state.
pub trait Engine: Sized {
    /// Its parameters (one patch's sound).
    type Params: Copy;
    /// The state of one voice. `Default` is a voice from silence.
    type Voice: Default + Clone;

    /// The engine at `rate`.
    fn new(rate: Rate) -> Self;

    /// Starts `v` on note `n`: a voice from silence, or one still sounding
    /// (a mono patch played again), whose state the engine's module says
    /// it keeps.
    fn note_on(&self, p: &Self::Params, v: &mut Self::Voice, n: &NoteOn);

    /// Adds `out.len()` samples (at most [`BLOCK`]) of `v` to `out`, mono,
    /// at unity level (a full-scale signal adds 1): adds, never
    /// overwrites. The host applies the amplitude.
    fn render(&self, p: &Self::Params, v: &mut Self::Voice, m: &VoiceMod, out: &mut [f32]);

    /// The voice's amplitude this block from the ADSR's `adsr`: the ADSR's
    /// own, unless the engine has an envelope of its own.
    fn amp(&self, _p: &Self::Params, _v: &mut Self::Voice, adsr: f32) -> f32 {
        adsr
    }

    /// Whether the voice has ended by itself (checked once a block).
    fn done(&self, _v: &Self::Voice) -> bool {
        false
    }

    /// The host lets `v` go (it ended, or was taken for another note): it
    /// gives back what its `note_on` borrowed from the engine's pool.
    fn free(&self, _v: &mut Self::Voice) {}
}
