//! Spooky: the game's Halloween. The tiers in slime green, pumpkin,
//! witch's purple, a ghostly mint and moonlight; the count-in in pumpkin.
//! The fire burns green (the screen's edges, the lane's flames, the line
//! sung), the marquee's bulbs are jack-o'-lanterns grinning, the coins
//! candy, the angels ghosts. A cold grade over the backdrop, fog creeping
//! in at the edges with eyes blinking in it ([`Fx::Fog`]). The pops
//! bubble like a cauldron over a skeleton's xylophone, the coins clink
//! as candy in a jar, the chimes toll as bells, the choir sings a ghost's
//! "ooh", and the streak lost wails down like a theremin.

use jodel_synth::engines::{analog, fm, formant, modal, AnyParams};
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::Patch;

use super::Theme;
use crate::gui::fx::{Fx, Use};
use crate::sfx::conductor::Atom;

/// The Halloween theme.
pub static SPOOKY: Theme = Theme {
    name: "spooky",
    colors: &[
        ("Good", [0.55, 1.0, 0.3]),
        ("Great", [1.0, 0.55, 0.1]),
        ("Awesome", [0.74, 0.46, 1.0]),
        ("Amazing", [0.3, 1.0, 0.78]),
        ("Diva", [0.9, 1.0, 0.94]),
    ],
    accent: Some([255, 140, 26]),
    looks: &[
        // Green fire, its pulse and breath as made.
        (Fx::EdgeFire, [1.0, 1.0, 1.0, 1.0]),
        (Fx::LyricFire, [1.0, 1.0, 0.0, 0.0]),
        (Fx::Flames, [1.0, 0.0, 1.0, 0.0]),
        // Jack-o'-lanterns, chasing and blinking as made.
        (Fx::Bulbs, [1.0, 1.0, 1.0, 1.0]),
        // Candy and ghosts.
        (Fx::Fountain, [1.0, 0.0, 0.0, 0.0]),
        (Fx::Rain, [1.0, 0.0, 0.0, 0.0]),
        (Fx::Flick, [1.0, 0.0, 0.0, 0.0]),
        (Fx::Jackpot, [1.0, 0.0, 0.0, 0.0]),
        (Fx::Angels, [1.0, 0.0, 0.0, 0.0]),
    ],
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
    ],
    always: &[Use::of(Fx::Fog, 0.8)],
    grade: [0.5, 0.66, 0.95, 0.8],
};

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
