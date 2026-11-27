//! Spooky: the game's Halloween. The tiers in slime green, pumpkin,
//! witch's purple, a ghostly mint and moonlight; the count-in in pumpkin.
//! The fire burns green (the screen's edges, the lane's flames, the line
//! sung), the marquee's bulbs are jack-o'-lanterns grinning, the coins
//! candy, the angels ghosts. A cold grade over the backdrop, fog creeping
//! in at the edges with eyes blinking in it ([`Fx::Fog`]). The pops
//! bubble like a cauldron over a skeleton's xylophone, the coins clink
//! as candy in a jar, the chimes toll as bells, the choir sings a ghost's
//! "ooh", the streak lost wails down like a theremin, and the
//! announcer is a ghost, moaning its calls out of the fog.

use jodel_synth::engines::{analog, fm, formant, modal, AnyParams};
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::Patch;

use super::{BulbShape, Flight, Grade, Palette, Ramp, Theme};
use crate::gui::fx::{Fx, Use};
use crate::gui::stage::palette::Rgb;
use crate::sfx::conductor::Atom;

/// The Halloween theme.
pub static SPOOKY: Theme = Theme {
    name: "spooky",
    title: "Spooky",
    about: "Halloween: green fire, pumpkins, candy, ghosts and fog",
    colors: Some([
        [0.55, 1.0, 0.3],
        [1.0, 0.55, 0.1],
        [0.74, 0.46, 1.0],
        [0.3, 1.0, 0.78],
        [0.9, 1.0, 0.94],
    ]),
    accent: Some([1.0, 140.0 / 255.0, 26.0 / 255.0]),
    // The classic fire's heats in greens.
    fire: Ramp([
        [0.0155, 0.45, 0.0185],
        [0.066, 1.0, 0.1],
        [0.2015, 1.0, 0.2875],
        [0.5, 1.0, 0.571],
        [0.925, 1.0, 0.866],
    ]),
    // Pumpkin orange in purple-black sockets.
    bulbs: Palette {
        on: [1.0, 0.58, 0.12],
        glow: [1.0, 0.36, 0.02],
        off: [0.32, 0.13, 0.03],
        socket: [0.07, 0.03, 0.09],
        rim: [0.5, 0.22, 0.6],
    },
    bulb: BulbShape::Jack,
    coins: Some(&CANDY),
    angels: Flight::Ghosts,
    sounds: &[
        (Atom::Pop, BLOOP),
        (Atom::PopBody, BONES),
        (Atom::Glint, candy(0.4, 0.3, -20.0)),
        (Atom::CoinA, candy(0.2, 0.35, -21.0)),
        (Atom::CoinB, candy(0.3, 0.4, -21.0)),
        (Atom::CoinC, candy(0.38, 0.32, -21.0)),
        (Atom::Wah, THEREMIN),
        (Atom::Chime, TOLL),
        (Atom::Ching, CHURCH),
        (Atom::Choir, GHOSTS),
        (Atom::Singer, GHOST),
    ],
    always: &[Use::of(Fx::Fog, 0.8)],
    grade: Grade {
        tint: [0.5, 0.66, 0.95],
        amount: 0.8,
    },
    look: None,
};

/// The sweets' colours the coins come in (sRGB): strawberry, mint, lime,
/// grape, orange, blue raspberry.
const CANDY: [Rgb; 6] = [
    [1.0, 0.42, 0.72],
    [0.4, 1.0, 0.7],
    [0.7, 1.0, 0.25],
    [0.72, 0.45, 1.0],
    [1.0, 0.55, 0.15],
    [0.35, 0.85, 1.0],
];

/// A word's pop: a bubble of a cauldron, a sine welling up an octave into
/// its note.
const BLOOP: Patch = Patch {
    pitch: Glide {
        from: -12.0,
        time: 0.06,
        curve: 1.6,
    },
    adsr: Adsr::hit(0.12),
    level: -5.0,
    reverb: 0.12,
    ..Patch::new(AnyParams::Analog(analog::SINE))
};

/// Its body: a skeleton's xylophone, a hard bar of bone.
const BONES: Patch = Patch {
    adsr: Adsr::hit(0.25),
    level: -4.0,
    reverb: 0.18,
    ..Patch::new(AnyParams::Modal(modal::Params {
        hardness: 0.95,
        brightness: 0.6,
        decay: 0.3,
        ..modal::MARIMBA
    }))
};

/// A boiled sweet clinking on glass, struck at `position`, ringing
/// `decay`, at `level` dB; each a little apart.
const fn candy(position: f32, decay: f32, level: f32) -> Patch {
    Patch {
        adsr: Adsr::hit(0.18),
        level,
        reverb: 0.15,
        spread: Spread {
            detune: 20.0,
            level: 1.5,
            start: 0.008,
            lfo_phase: 0.0,
        },
        ..Patch::new(AnyParams::Modal(modal::Params {
            position,
            decay,
            ..modal::GLASS
        }))
    }
}

/// The streak lost: a theremin wailing down, its vibrato wide.
const THEREMIN: Patch = Patch {
    pitch: Glide {
        from: 3.0,
        time: 0.45,
        curve: 0.8,
    },
    lfo: Lfo {
        rate: LfoRate::Hz(6.0),
        pitch: 0.45,
        fade: 0.08,
        ..Lfo::NONE
    },
    adsr: Adsr {
        attack: 0.06,
        decay: 0.35,
        sustain: 0.6,
        release: 0.45,
    },
    level: -12.0,
    reverb: 0.35,
    ..Patch::new(AnyParams::Analog(analog::SINE))
};

/// The stingers' and chimes' bell: a deep bell tolling.
const TOLL: Patch = Patch {
    adsr: Adsr::hit(1.8),
    level: -8.0,
    reverb: 0.45,
    ..Patch::new(AnyParams::Fm(fm::BELL))
};

/// The register's bell: a church bell.
const CHURCH: Patch = Patch {
    adsr: Adsr::hit(1.6),
    level: -12.0,
    reverb: 0.45,
    ..Patch::new(AnyParams::Modal(modal::BELL))
};

/// The choir: ghosts' "ooh", swooping up into the note, wobbling wide.
const GHOSTS: Patch = Patch {
    pitch: Glide {
        from: -3.0,
        time: 0.3,
        curve: 1.2,
    },
    lfo: Lfo {
        rate: LfoRate::Hz(4.2),
        pitch: 0.3,
        delay: 0.1,
        fade: 0.4,
        ..Lfo::NONE
    },
    adsr: Adsr {
        attack: 0.15,
        decay: 0.4,
        sustain: 0.85,
        release: 0.9,
    },
    level: -13.0,
    reverb: 0.6,
    chorus: 0.6,
    spread: Spread {
        detune: 10.0,
        level: 1.0,
        start: 0.03,
        lfo_phase: 1.0,
    },
    ..Patch::new(AnyParams::Formant(formant::Params {
        breath: 0.45,
        size: 1.0,
        ..formant::OOH
    }))
};

/// The announcer as a ghost: hollow and breathy, sinking into each
/// syllable from above and gliding slowly from one to the next, a wide,
/// slow wobble on the long ones, echoing round a crypt. It calls on the
/// same beats and notes as ever (the conductor's): only its voice is the
/// theme's.
const GHOST: Patch = Patch {
    pitch: Glide {
        from: 2.0,
        time: 0.12,
        curve: 1.5,
    },
    lfo: Lfo {
        rate: LfoRate::Hz(4.0),
        pitch: 0.55,
        bright: 0.2,
        delay: 0.1,
        fade: 0.25,
        ..Lfo::NONE
    },
    reverb: 0.5,
    delay: 0.3,
    chorus: 0.5,
    slide: 0.09,
    level: -1.0,
    engine: AnyParams::Formant(formant::Params {
        vowel: formant::A,
        breath: 0.5,
        size: 0.35,
        soft: 0.75,
        width: 3.0,
    }),
    ..crate::sfx::conductor::patch(Atom::Singer)
};
