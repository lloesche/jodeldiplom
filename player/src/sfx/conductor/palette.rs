//! The conductor's instruments, as data: the atoms its sounds are made of
//! ([`Atom`], every one in [`Atom::ALL`]), each one patch of our own synth
//! ([`jodel_synth`], [`patch`]), and the bank they are made into
//! ([`Palette`]): the bubble's pop, its body and glint, three coins for
//! the clinks, the zap, ticks and pips, a whirr and a muted horn for the
//! wind-down, a pluck, a bar chime, brass, a choir's "aah", a shimmer pad,
//! glitter, a riser and a whoosh, the register's clack, thud, bell and
//! drawer, a low tonic, the slam's drums, and the diploma's sheet's own
//! ([`super::sheet`]): its bells, glass, plucks and hum, its thumps and
//! knocks, the paper's rustle and the air under it, snaps and cracks of
//! paper, the air a stamp pushes down, a roll under a fanfare, and the
//! pen's nib.
//!
//! Levels set the sounds against each other: the pops soft, the clinks
//! quieter still, the big ones bigger; the synth's bus lifts them all
//! together over the song ([`super::LEVEL`], after its compressor, into
//! its limiter at -1 dBFS, which keeps a pile of them from clipping). The
//! held ones (the pad, the riser) are given their lengths note by note.

use std::ops::Index;

use jodel_synth::engines::{analog, drum, fm, formant, modal, noise, phase, supersaw, AnyParams};
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::{Bank, Patch, PatchId};

use crate::core::singing::{STAMP_DOWN, UNROLL};

/// The atoms, and all of them in their order.
macro_rules! atoms {
    ($($name:ident,)*) => {
        /// What a sound is made of: one patch each ([`patch`]).
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(super) enum Atom {
            $($name,)*
        }

        impl Atom {
            /// Every atom, in its order.
            pub(super) const ALL: [Atom; [$(Atom::$name),*].len()] = [$(Atom::$name),*];
        }
    };
}

atoms! {
    Pop,
    PopBody,
    Glint,
    CoinA,
    CoinB,
    CoinC,
    Zap,
    Tick,
    Pip,
    Whirr,
    Wah,
    Pluck,
    Chime,
    BrassSaw,
    BrassPd,
    Choir,
    PadTonic,
    PadFifth,
    Glitter,
    Riser,
    Whoosh,
    Clack,
    Thud,
    Ching,
    Drawer,
    Low,
    SlamKick,
    SlamClap,
    SlamCrash,
    Ping,
    Bell,
    Glass,
    Plucked,
    Hum,
    Thump,
    Knock,
    Rustle,
    Air,
    Snap,
    Crack,
    Rush,
    Roll,
    Nib,
}

/// The coins, taken in turn; the brass's two layers; the shimmer's two
/// voices (the tonic and the fifth); the slam's three drums.
pub(super) const COINS: [Atom; 3] = [Atom::CoinA, Atom::CoinB, Atom::CoinC];
pub(super) const BRASS: [Atom; 2] = [Atom::BrassSaw, Atom::BrassPd];
pub(super) const PADS: [Atom; 2] = [Atom::PadTonic, Atom::PadFifth];
pub(super) const SLAM: [Atom; 3] = [Atom::SlamKick, Atom::SlamClap, Atom::SlamCrash];

/// `engine` with `adsr` at `level` dB.
const fn voice(engine: AnyParams, adsr: Adsr, level: f32) -> Patch {
    Patch {
        adsr,
        level,
        ..Patch::new(engine)
    }
}

/// A coin: a small plate struck hard, here, so hard, ringing so long; each
/// note a little apart in pitch, level and time.
const fn coin(position: f32, hardness: f32, decay: f32) -> Patch {
    Patch {
        reverb: 0.12,
        spread: Spread {
            detune: 15.0,
            level: 1.5,
            start: 0.008,
            lfo_phase: 0.0,
        },
        ..voice(
            AnyParams::Modal(modal::Params {
                position,
                hardness,
                decay,
                ..modal::COIN
            }),
            Adsr::hit(0.16),
            -19.0,
        )
    }
}

/// Brass: two saws and phase distortion together, opened by the envelope.
const fn brass(engine: AnyParams, filter_env: f32, level: f32) -> Patch {
    Patch {
        filter_env,
        reverb: 0.25,
        spread: Spread {
            detune: 6.0,
            level: 1.0,
            start: 0.006,
            lfo_phase: 0.0,
        },
        ..voice(
            engine,
            Adsr {
                attack: 0.03,
                decay: 0.4,
                sustain: 0.6,
                release: 0.25,
            },
            level,
        )
    }
}

/// The held shimmer: a bright, airy pad high up, breathing in the song's
/// tempo; one voice a note, held as long as its notes say.
const PAD: Patch = Patch {
    lfo: Lfo {
        rate: LfoRate::Beats(2.0),
        bright: 0.5,
        amp: 0.35,
        ..Lfo::NONE
    },
    reverb: 0.4,
    chorus: 0.6,
    mono: true,
    ..voice(
        AnyParams::Supersaw(supersaw::SHIMMER_PAD),
        Adsr::held(0.8, 0.15),
        -31.0,
    )
};

/// A slam's drum: held far longer than it rings, so the drum ends itself.
const fn slam(d: drum::Params) -> Patch {
    Patch {
        gate: 4.0,
        reverb: 0.2,
        ..voice(AnyParams::Drum(d), Adsr::held(0.0, 0.3), -11.0)
    }
}

/// Noise through a band on the note, `res` sharp, `color` from white (0)
/// to brown (1): its cutoff on middle C, following the key a semitone a
/// semitone, so a note's pitch is the band's.
const fn band(res: f32, color: f32) -> noise::Params {
    noise::Params {
        color,
        filter: noise::Filter::BandPass,
        cutoff: MIDDLE_C,
        res,
        key_track: 1.0,
        ..noise::TICK
    }
}

/// Middle C, Hz.
const MIDDLE_C: f32 = 261.625_57;

/// Struck: a decay of `decay` seconds to silence (60 dB down), let go
/// over `release` when the note is shorter.
const fn struck(decay: f32, release: f32) -> Adsr {
    Adsr {
        attack: 0.001,
        decay,
        sustain: 0.0,
        release,
    }
}

/// An atom's patch.
pub(super) const fn patch(a: Atom) -> Patch {
    match a {
        // The bubble's pop: a sine flicking up into its note, gone in a
        // tenth of a second.
        Atom::Pop => Patch {
            pitch: Glide {
                from: -7.0,
                time: 0.025,
                curve: 2.0,
            },
            reverb: 0.08,
            ..voice(AnyParams::Analog(analog::SINE), Adsr::hit(0.09), -5.0)
        },
        // Its body: a kalimba tine on the same note, so the pop is a note.
        Atom::PopBody => Patch {
            reverb: 0.12,
            spread: Spread {
                detune: 4.0,
                level: 1.0,
                ..Spread::NONE
            },
            ..voice(AnyParams::Modal(modal::KALIMBA), Adsr::hit(0.3), -4.0)
        },
        // A glint an octave over the pop, louder as the streak is higher.
        Atom::Glint => Patch {
            reverb: 0.15,
            ..voice(AnyParams::Fm(fm::COIN_GLINT), Adsr::hit(0.2), -20.0)
        },
        Atom::CoinA => coin(0.15, 1.0, 0.36),
        Atom::CoinB => coin(0.24, 0.9, 0.4),
        Atom::CoinC => coin(0.32, 0.95, 0.33),
        // The rescue shot: a synced saw diving two and a half octaves, an
        // echo after it.
        Atom::Zap => Patch {
            pitch: Glide {
                from: 30.0,
                time: 0.18,
                curve: 2.5,
            },
            delay: 0.3,
            ..voice(AnyParams::Analog(analog::ZAP), Adsr::hit(0.25), -9.0)
        },
        // A short bright tick: a reel's, a count-up's.
        Atom::Tick => voice(AnyParams::Drum(drum::TICK), Adsr::hit(0.03), -14.0),
        // A pip: a short dark pluck under a tick.
        Atom::Pip => Patch {
            filter_env: 3.0,
            reverb: 0.1,
            ..voice(AnyParams::Analog(analog::PLUCK), Adsr::hit(0.07), -5.0)
        },
        // The reels' whirr running down: air sinking two octaves.
        Atom::Whirr => Patch {
            pitch: Glide {
                from: 24.0,
                time: 0.6,
                curve: 0.8,
            },
            reverb: 0.15,
            ..voice(
                AnyParams::Noise(noise::WHOOSH),
                Adsr {
                    attack: 0.01,
                    decay: 0.4,
                    sustain: 0.4,
                    release: 0.2,
                },
                -18.0,
            )
        },
        // A muted horn's "wah", sliding down a semitone as it opens and
        // closes; kind.
        Atom::Wah => Patch {
            filter_env: 2.2,
            pitch: Glide {
                from: 1.0,
                time: 0.15,
                curve: 1.0,
            },
            lfo: Lfo {
                rate: LfoRate::Hz(5.0),
                pitch: 0.1,
                fade: 0.1,
                ..Lfo::NONE
            },
            reverb: 0.2,
            ..voice(
                AnyParams::Analog(analog::BRASS),
                Adsr {
                    attack: 0.03,
                    decay: 0.18,
                    sustain: 0.4,
                    release: 0.25,
                },
                -13.0,
            )
        },
        // The arpeggios' pluck.
        Atom::Pluck => Patch {
            filter_env: 4.0,
            reverb: 0.15,
            delay: 0.18,
            ..voice(AnyParams::Analog(analog::PLUCK), Adsr::hit(0.3), -4.0)
        },
        // A bar chime: the stingers' bell, the tier chimes', the tremolo's.
        Atom::Chime => Patch {
            reverb: 0.3,
            spread: Spread {
                level: 1.0,
                ..Spread::NONE
            },
            ..voice(AnyParams::Fm(fm::CHIME), Adsr::hit(1.2), -7.0)
        },
        Atom::BrassSaw => brass(AnyParams::Analog(analog::BRASS), 3.0, -17.0),
        Atom::BrassPd => brass(AnyParams::Phase(phase::BRASS), 2.6, -18.0),
        // A choir's soft "aah", its vibrato coming in.
        Atom::Choir => Patch {
            lfo: Lfo {
                rate: LfoRate::Hz(5.2),
                pitch: 0.12,
                delay: 0.15,
                fade: 0.3,
                ..Lfo::NONE
            },
            reverb: 0.45,
            chorus: 0.4,
            spread: Spread {
                detune: 8.0,
                level: 1.0,
                start: 0.02,
                lfo_phase: 1.0,
            },
            ..voice(
                AnyParams::Formant(formant::Params {
                    vowel: formant::A,
                    ..formant::AWW
                }),
                Adsr {
                    attack: 0.08,
                    decay: 0.3,
                    sustain: 0.8,
                    release: 0.6,
                },
                -13.0,
            )
        },
        Atom::PadTonic | Atom::PadFifth => PAD,
        // Glitter: a short bright sparkle.
        Atom::Glitter => Patch {
            reverb: 0.3,
            delay: 0.15,
            spread: Spread {
                detune: 8.0,
                level: 2.0,
                ..Spread::NONE
            },
            ..voice(AnyParams::Fm(fm::SPARKLE), Adsr::hit(0.2), -17.0)
        },
        // The riser: air in a band, rising two octaves and swelling into
        // its note, held as long as its note says.
        Atom::Riser => Patch {
            pitch: Glide {
                from: -24.0,
                time: 0.8,
                curve: 0.8,
            },
            reverb: 0.2,
            mono: true,
            ..voice(
                AnyParams::Noise(noise::WHOOSH),
                Adsr::held(0.8, 0.05),
                -14.0,
            )
        },
        // A whoosh up into a payout.
        Atom::Whoosh => Patch {
            pitch: Glide {
                from: -24.0,
                time: 0.5,
                curve: 0.6,
            },
            reverb: 0.25,
            ..voice(
                AnyParams::Noise(noise::WHOOSH),
                Adsr::held(0.4, 0.08),
                -14.0,
            )
        },
        // The register's lever clacking down.
        Atom::Clack => voice(
            AnyParams::Noise(noise::Params {
                cutoff: 1_400.0,
                res: 0.6,
                ..noise::TICK
            }),
            Adsr::hit(0.04),
            -9.0,
        ),
        // The lever's thud: a sine falling an octave.
        Atom::Thud => Patch {
            pitch: Glide {
                from: 12.0,
                time: 0.04,
                curve: 1.5,
            },
            ..voice(AnyParams::Analog(analog::SINE), Adsr::hit(0.12), -12.0)
        },
        // The register's "ching": phase distortion's bell, a little less
        // bent.
        Atom::Ching => Patch {
            filter_env: 0.4,
            reverb: 0.35,
            ..voice(
                AnyParams::Phase(phase::Params {
                    dcw: 0.1,
                    ..phase::BELL
                }),
                Adsr::hit(1.1),
                -12.0,
            )
        },
        // The drawer sliding out.
        Atom::Drawer => Patch {
            pitch: Glide {
                from: -12.0,
                time: 0.25,
                curve: 1.0,
            },
            ..voice(
                AnyParams::Noise(noise::DRAWER),
                Adsr::held(0.02, 0.06),
                -10.0,
            )
        },
        // A low tonic under the big chords: a round square bass.
        Atom::Low => Patch {
            reverb: 0.2,
            ..voice(
                AnyParams::Analog(analog::BASS),
                Adsr {
                    attack: 0.005,
                    decay: 0.9,
                    sustain: 0.0,
                    release: 0.2,
                },
                -14.0,
            )
        },
        Atom::SlamKick => slam(drum::SLAM[0].1),
        Atom::SlamClap => slam(drum::SLAM[1].1),
        Atom::SlamCrash => slam(drum::SLAM[2].1),
        // The sheet's bells: a chime's partials, a ping dying soon, a bell
        // ringing on; each note as long as its sound holds it.
        Atom::Ping => Patch {
            reverb: 0.15,
            ..voice(AnyParams::Fm(fm::CHIME), struck(0.6, 0.05), -6.0)
        },
        Atom::Bell => Patch {
            reverb: 0.25,
            ..voice(AnyParams::Fm(fm::CHIME), struck(2.5, 0.08), -6.0)
        },
        // A glassy grain: a sparkle without the glitter's echo.
        Atom::Glass => Patch {
            reverb: 0.15,
            ..voice(AnyParams::Fm(fm::SPARKLE), struck(0.5, 0.04), -8.0)
        },
        // A pluck ringing as long as its note.
        Atom::Plucked => Patch {
            filter_env: 4.0,
            reverb: 0.2,
            ..voice(AnyParams::Analog(analog::PLUCK), struck(1.5, 0.08), -6.0)
        },
        // A soft chord's sines, swelling in, a slow vibrato coming in.
        Atom::Hum => Patch {
            lfo: Lfo {
                rate: LfoRate::Hz(4.5),
                pitch: 0.06,
                fade: 0.3,
                ..Lfo::NONE
            },
            reverb: 0.35,
            ..voice(
                AnyParams::Analog(analog::SINE),
                Adsr {
                    attack: 0.1,
                    decay: 1.2,
                    sustain: 0.4,
                    release: 0.35,
                },
                -6.0,
            )
        },
        // A sine sliding down an octave and more: a stamp's thunk, a
        // seal's thump.
        Atom::Thump => Patch {
            pitch: Glide {
                from: 14.0,
                time: 0.08,
                curve: 2.0,
            },
            reverb: 0.08,
            ..voice(AnyParams::Analog(analog::SINE), Adsr::hit(0.35), 0.0)
        },
        // A small sine knock, sliding a little: the desk under a pen's
        // tap, the paper under a value landing, the sheet lying flat.
        Atom::Knock => Patch {
            pitch: Glide {
                from: 5.0,
                time: 0.04,
                curve: 2.0,
            },
            reverb: 0.05,
            ..voice(AnyParams::Analog(analog::SINE), Adsr::hit(0.1), -6.0)
        },
        // The sheet unrolling: the paper's rustle, its band rising as the
        // sheet goes and wandering (the crinkles in it), as loud as its
        // front is fast; and the air under it.
        Atom::Rustle => Patch {
            pitch: Glide {
                from: -19.0,
                time: UNROLL as f32,
                curve: 1.0,
            },
            reverb: 0.1,
            ..voice(
                AnyParams::Noise(noise::Params {
                    drift: 2.0,
                    drift_rate: 30.0,
                    ..band(0.3, 0.2)
                }),
                struck(1.1, 0.05),
                -6.0,
            )
        },
        Atom::Air => Patch {
            reverb: 0.1,
            ..voice(AnyParams::Noise(band(0.17, 0.3)), struck(1.1, 0.05), -6.0)
        },
        // A tiny burst of paper: a crinkle, a nib's click, a flick.
        Atom::Snap => Patch {
            pitch: Glide {
                from: 6.0,
                time: 0.015,
                curve: 1.0,
            },
            ..voice(AnyParams::Noise(band(0.75, 0.0)), Adsr::hit(0.03), -6.0)
        },
        // A short burst falling: a slap, a crack, the paper flapping.
        Atom::Crack => Patch {
            pitch: Glide {
                from: 8.0,
                time: 0.05,
                curve: 1.5,
            },
            reverb: 0.05,
            ..voice(AnyParams::Noise(band(0.5, 0.0)), Adsr::hit(0.25), -6.0)
        },
        // The air a stamp pushes as it comes down, rising and swelling.
        Atom::Rush => Patch {
            pitch: Glide {
                from: -24.0,
                time: STAMP_DOWN as f32,
                curve: 1.0,
            },
            ..voice(
                AnyParams::Noise(band(0.5, 0.3)),
                Adsr {
                    attack: 0.09,
                    decay: 0.08,
                    sustain: 0.0,
                    release: 0.02,
                },
                -6.0,
            )
        },
        // A roll of noise swelling under a fanfare.
        Atom::Roll => Patch {
            pitch: Glide {
                from: -24.0,
                time: 0.5,
                curve: 1.0,
            },
            reverb: 0.3,
            ..voice(
                AnyParams::Noise(band(0.45, 0.0)),
                Adsr {
                    attack: 0.35,
                    decay: 0.5,
                    sustain: 0.0,
                    release: 0.1,
                },
                -6.0,
            )
        },
        // The pen's nib on paper: a grain of its scratch, rising over one
        // grain, held as long as its note, let go over two.
        Atom::Nib => voice(
            AnyParams::Noise(band(0.6, 0.0)),
            Adsr {
                attack: super::pen::GRAIN as f32,
                decay: 0.0,
                sustain: 1.0,
                release: 2.0 * super::pen::GRAIN as f32,
            },
            -16.0,
        ),
    }
}

/// The atoms' patches in a synth's bank.
#[derive(Clone, Copy, Debug)]
pub(super) struct Palette([PatchId; Atom::ALL.len()]);

impl Palette {
    /// The bank of every atom's patch, and where each is in it.
    pub(super) fn bank() -> (Bank, Palette) {
        let mut bank = Bank::new();
        let ids = Atom::ALL.map(|a| bank.add(patch(a)));
        (bank, Palette(ids))
    }
}

impl Index<Atom> for Palette {
    type Output = PatchId;

    fn index(&self, a: Atom) -> &PatchId {
        &self.0[a as usize]
    }
}
