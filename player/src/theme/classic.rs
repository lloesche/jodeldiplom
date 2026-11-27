//! Classic: the game as it is made. The tiers in their rows' colours, the
//! count-in in the accent, every sound the palette's; the fire deep red to
//! white hot, the marquee warm white bulbs on gold, the coins gold, the
//! angels angels; nothing held throughout and no grade.

use super::{BulbShape, Flight, Grade, Palette, Ramp, Theme};

/// The game as it is made.
pub static CLASSIC: Theme = Theme {
    name: "classic",
    title: "Classic",
    about: "the game as it is made",
    colors: None,
    accent: None,
    fire: Ramp([
        [0.45, 0.03, 0.01],
        [1.0, 0.2, 0.02],
        [1.0, 0.55, 0.08],
        [1.0, 0.88, 0.35],
        [1.0, 0.98, 0.85],
    ]),
    bulbs: Palette {
        on: [1.0, 0.86, 0.5],
        glow: [1.0, 0.68, 0.22],
        off: [0.34, 0.25, 0.16],
        socket: [0.1, 0.06, 0.03],
        rim: [0.86, 0.64, 0.28],
    },
    bulb: BulbShape::Bulb,
    coins: None,
    angels: Flight::Angels,
    sounds: &[],
    always: &[],
    grade: Grade::NONE,
    look: None,
};
