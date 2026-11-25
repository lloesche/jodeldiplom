//! The engines, and the host's one view of all of them: [`Engines`] holds
//! one of each, made for a rate, and plays [`AnyParams`] on an
//! [`AnyVoice`] of the same engine (a patch keeps its engine with its
//! parameters and its voices are made from them, so they always match).
//!
//! An engine joins by a line in the `engines!` list below: its variant
//! name, its module, and its type there, with its `Params` and `Voice`
//! beside it in that module.
//!
//! Changing a patch to another engine (a future Synth tab's `set_patch`)
//! must let that patch's sounding voices go first (fading them out): their
//! voices are the old engine's.

pub mod analog;
pub mod drum;
pub mod fm;
pub mod formant;
pub mod lofi;
pub mod modal;
pub mod noise;
pub mod phase;
pub mod string;
pub mod supersaw;

use crate::dsp::Rate;
use crate::{Engine, NoteOn, VoiceMod};

/// A voice played with another engine's parameters: never, as the host
/// makes voices from their patch. Checked in debug builds; nothing plays
/// otherwise.
#[cold]
fn mismatch() {
    debug_assert!(false, "a voice played with another engine's parameters");
}

macro_rules! engines {
    ($($(#[$doc:meta])* $name:ident($module:ident::$engine:ident),)*) => {
        /// One of each engine, made for a rate.
        #[derive(Debug)]
        pub struct Engines {
            $($(#[$doc])* pub $module: $module::$engine,)*
        }

        /// An engine's parameters.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub enum AnyParams {
            $($(#[$doc])* $name($module::Params),)*
        }

        /// The state of a voice of an engine, as large as the largest (a
        /// modal voice's modes, about 0.6 KB): the pool holds one a voice,
        /// made once, where a voice boxed apart would allocate as it starts.
        #[derive(Debug)]
        #[allow(clippy::large_enum_variant)]
        pub enum AnyVoice {
            $($(#[$doc])* $name($module::Voice),)*
        }

        $(
            impl From<$module::Params> for AnyParams {
                fn from(p: $module::Params) -> AnyParams {
                    AnyParams::$name(p)
                }
            }
        )*

        impl Engines {
            /// Every engine at `rate`.
            pub fn new(rate: Rate) -> Engines {
                Engines {
                    $($module: $module::$engine::new(rate),)*
                }
            }

            /// A voice from silence for `p`'s engine.
            pub fn voice(&self, p: &AnyParams) -> AnyVoice {
                match p {
                    $(AnyParams::$name(_) => AnyVoice::$name(Default::default()),)*
                }
            }

            /// [`Engine::note_on`].
            pub fn note_on(&self, p: &AnyParams, v: &mut AnyVoice, n: &NoteOn) {
                match (p, v) {
                    $((AnyParams::$name(p), AnyVoice::$name(v)) => self.$module.note_on(p, v, n),)*
                    _ => mismatch(),
                }
            }

            /// [`Engine::render`].
            pub fn render(&self, p: &AnyParams, v: &mut AnyVoice, m: &VoiceMod, out: &mut [f32]) {
                match (p, v) {
                    $((AnyParams::$name(p), AnyVoice::$name(v)) => self.$module.render(p, v, m, out),)*
                    _ => mismatch(),
                }
            }

            /// [`Engine::amp`].
            pub fn amp(&self, p: &AnyParams, v: &mut AnyVoice, adsr: f32) -> f32 {
                match (p, v) {
                    $((AnyParams::$name(p), AnyVoice::$name(v)) => self.$module.amp(p, v, adsr),)*
                    _ => {
                        mismatch();
                        0.0
                    }
                }
            }

            /// [`Engine::done`].
            pub fn done(&self, v: &AnyVoice) -> bool {
                match v {
                    $(AnyVoice::$name(v) => self.$module.done(v),)*
                }
            }

            /// [`Engine::free`].
            pub fn free(&self, v: &mut AnyVoice) {
                match v {
                    $(AnyVoice::$name(v) => self.$module.free(v),)*
                }
            }
        }
    };
}

engines! {
    /// NOISE ([`noise`]).
    Noise(noise::Noise),
    /// ANALOG ([`analog`]).
    Analog(analog::Analog),
    /// LOFI ([`lofi`]).
    Lofi(lofi::Lofi),
    /// PHASE ([`phase`]).
    Phase(phase::Phase),
    /// MODAL ([`modal`]).
    Modal(modal::Modal),
    /// STRING ([`string`]).
    String(string::KarplusString),
    /// FORMANT ([`formant`]).
    Formant(formant::Formant),
    /// FM ([`fm`]).
    Fm(fm::Fm),
    /// SUPERSAW ([`supersaw`]).
    Supersaw(supersaw::Supersaw),
    /// DRUM ([`drum`]).
    Drum(drum::Drum),
}
