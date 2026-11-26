//! How each kind of burst moves ([`MOTIONS`]), its line by its code:
//! how long it lasts, its quads on the processor, its call and maths in
//! the shader, and bursts of it to hold the shader to; and how each look
//! of a quad is drawn ([`LOOKS`]).

use super::confetti;
use super::shards;
use super::{burst, scored, Kind, Look, Spray, Sprite, DRAW_SPARKLE, SCORED_LIFE};
use crate::gui::fx::{angels, coins, jackpot, slam, step_up};
use crate::view::bubbles::BURST;

/// How a kind of burst moves: how long it lasts, quad `q` of a burst
/// `age` seconds on (none where it does not show), the same in the shader
/// (called with `at`, `rgb`, `size`, `seed`, `q`, `age` and `drift`: how
/// far it has gone across) and its maths there, and bursts of it to hold
/// the shader to.
#[derive(Clone, Copy, Debug)]
pub struct Motion {
    /// The kind.
    pub kind: Kind,
    /// How long it lasts, in seconds.
    pub life: f32,
    /// Its quads.
    pub quad: fn(&Spray, u32, f32) -> Option<Sprite>,
    /// Its call in the shader.
    pub call: &'static str,
    /// Its WGSL (none: the particles' own, [`super::shader`]).
    pub wgsl: Option<fn() -> String>,
    /// Bursts of it for the shader's check.
    pub probes: &'static [Spray],
}

/// How a quad of a look is drawn: how far its quad reaches each way from
/// its middle (WGSL, from `half`, half its size: room for its glow), the
/// function that draws it (called with the pixel, its middle, `half`, its
/// turn and its colour and how much it shows) and that function's WGSL.
#[derive(Clone, Copy, Debug)]
pub struct Drawing {
    /// The look.
    pub look: Look,
    /// How far its quad reaches.
    pub reach: &'static str,
    /// Its function's name.
    pub draw: &'static str,
    /// Its function.
    pub wgsl: &'static str,
}

/// Every look's drawing, by its code: a new look is a line here and its
/// function, both selected in the particles' shader by its code
/// ([`super::shader`]: `reach_of`, `draw_of`).
pub const LOOKS: &[Drawing] = &[
    Drawing {
        look: Look::Sparkle,
        reach: "half + vec2<f32>(1.0)",
        draw: "draw_sparkle",
        wgsl: DRAW_SPARKLE,
    },
    Drawing {
        look: Look::Sheen,
        reach: "half + vec2<f32>(half.y + 2.0)",
        draw: "draw_sheen",
        wgsl: shards::DRAW_SHEEN,
    },
    Drawing {
        look: Look::Shard,
        reach: "vec2<f32>(half.x + half.y * 3.0 + 2.0)",
        draw: "draw_shard",
        wgsl: shards::DRAW_SHARD,
    },
    Drawing {
        look: Look::Drop,
        reach: "vec2<f32>(half.x + 2.0)",
        draw: "draw_drop",
        wgsl: shards::DRAW_DROP,
    },
    Drawing {
        look: Look::Note,
        reach: "vec2<f32>(length(half) * 1.5 + 2.0)",
        draw: "draw_note",
        wgsl: step_up::DRAW_NOTE,
    },
    Drawing {
        look: Look::Bit,
        reach: "vec2<f32>(length(half) * 1.5 + 2.0)",
        draw: "draw_bit",
        wgsl: step_up::DRAW_BIT,
    },
    Drawing {
        look: Look::Angel,
        reach: "half * 1.9 + vec2<f32>(4.0)",
        draw: "draw_angel",
        wgsl: angels::DRAW_ANGEL,
    },
    Drawing {
        look: Look::Coin,
        reach: "half * 1.5 + vec2<f32>(2.0)",
        draw: "draw_coin",
        wgsl: jackpot::DRAW_COIN,
    },
    Drawing {
        look: Look::Gold,
        // Its glint reaches past its radius.
        reach: "half * 1.9 + vec2<f32>(3.0)",
        draw: "draw_gold",
        wgsl: coins::DRAW_GOLD,
    },
    Drawing {
        look: Look::Puff,
        reach: "half * 1.5 + vec2<f32>(1.0)",
        draw: "draw_puff",
        wgsl: slam::DRAW_PUFF,
    },
    Drawing {
        look: Look::Ghost,
        reach: "half * 1.4 + vec2<f32>(3.0)",
        draw: "draw_ghost",
        wgsl: angels::DRAW_GHOST,
    },
];

/// A probe: a burst of `kind` from `at` (`half` its size), `size` large,
/// scattered by `seed`, going across at `drift`, from 10 s.
const fn probe(kind: Kind, at: (f32, f32), half: (f32, f32), size: f32, seed: u32) -> Spray {
    Spray {
        at,
        half,
        start: 10.0,
        drift: 0.0,
        size,
        seed,
        rgb: [1.0, 0.75, 0.24],
        kind,
    }
}

/// `s` going across at `drift`.
const fn drifting(s: Spray, drift: f32) -> Spray {
    Spray { drift, ..s }
}

/// Every kind's motion, by its code.
pub const MOTIONS: &[Motion] = &[
    Motion {
        kind: Kind::Burst,
        life: BURST as f32,
        quad: burst,
        call: "burst(at, rgb, size, seed, q, age, drift)",
        wgsl: Some(shards::wgsl),
        probes: &[
            probe(Kind::Burst, (400.0, 300.0), (30.0, 4.0), 41.0, 7),
            drifting(
                probe(Kind::Burst, (180.0, 160.0), (84.0, 23.0), 94.0, 104_729),
                -150.0,
            ),
        ],
    },
    Motion {
        kind: Kind::Scored,
        life: SCORED_LIFE,
        quad: scored,
        call: "scored(at, rgb, size, seed, q, age, drift)",
        wgsl: None,
        probes: &[drifting(
            probe(Kind::Scored, (200.0, 330.0), (0.0, 0.0), 20.0, 4242),
            -150.0,
        )],
    },
    Motion {
        kind: Kind::StepUp,
        life: step_up::STEP_LIFE,
        quad: step_up::step_up,
        call: "step_up(at, size, seed, q, age, drift)",
        wgsl: Some(step_up::wgsl),
        probes: &[
            drifting(
                probe(Kind::StepUp, (600.0, 330.0), (0.0, 0.0), 20.0, 99),
                -150.0,
            ),
            probe(Kind::StepUp, (220.0, 200.0), (0.0, 0.0), 47.0, 31_337),
        ],
    },
    Motion {
        kind: Kind::Angels,
        life: angels::ANGELS_LIFE,
        quad: angels::angel,
        call: "angel(at, rgb, size, seed, q, age)",
        wgsl: Some(angels::wgsl),
        probes: &[probe(
            Kind::Angels,
            (800.0, -90.0),
            (720.0, 675.0),
            90.0,
            4242,
        )],
    },
    Motion {
        kind: Kind::Jackpot,
        life: jackpot::JACKPOT_LIFE,
        quad: jackpot::coin,
        call: "coin(at, rgb, size, seed, q, age)",
        wgsl: Some(jackpot::wgsl),
        probes: &[probe(Kind::Jackpot, (900.0, 70.0), (0.0, 0.0), 30.0, 777)],
    },
    Motion {
        kind: Kind::Confetti,
        life: confetti::CONFETTI_LIFE,
        quad: confetti::piece,
        call: "confetti(at, rgb, size, seed, q, age)",
        wgsl: Some(confetti::wgsl),
        probes: &[probe(
            Kind::Confetti,
            (800.0, 0.0),
            (800.0, 900.0),
            20.0,
            5150,
        )],
    },
    Motion {
        kind: Kind::CoinFountain,
        life: coins::FOUNTAIN_LIFE,
        quad: coins::fountain,
        call: "coin_fountain(at, rgb, size, seed, q, age)",
        wgsl: Some(coins::fountain_wgsl),
        probes: &[probe(
            Kind::CoinFountain,
            (900.0, 600.0),
            (0.0, 0.0),
            34.0,
            4242,
        )],
    },
    Motion {
        kind: Kind::CoinRain,
        life: coins::RAIN_LIFE,
        quad: coins::rain,
        call: "coin_rain(at, rgb, size, seed, q, age)",
        wgsl: Some(coins::rain_wgsl),
        probes: &[
            probe(Kind::CoinRain, (1512.0, -60.0), (1512.0, 1964.0), 30.0, 99),
            probe(
                Kind::CoinRain,
                (1512.0, -60.0),
                (1512.0, 1964.0),
                30.0,
                99 + 0x9E37,
            ),
        ],
    },
    Motion {
        kind: Kind::CoinFlick,
        life: coins::FLICK_LIFE,
        quad: coins::flick,
        call: "coin_flick(at, rgb, size, seed, q, age)",
        wgsl: Some(coins::flick_wgsl),
        probes: &[
            probe(Kind::CoinFlick, (500.0, 500.0), (0.0, 0.0), 14.0, 7),
            probe(Kind::CoinFlick, (500.0, 500.0), (0.0, 0.0), 14.0, 8),
        ],
    },
    Motion {
        kind: Kind::CoinBurst,
        life: coins::BURST_LIFE,
        quad: coins::burst,
        call: "coin_burst(at, rgb, size, seed, q, age)",
        // In the rain's: they fall alike.
        wgsl: None,
        probes: &[probe(
            Kind::CoinBurst,
            (1512.0, -60.0),
            (1512.0, 1964.0),
            30.0,
            41,
        )],
    },
    Motion {
        kind: Kind::SlamDust,
        life: slam::DUST_LIFE,
        quad: slam::dust,
        call: "slam_dust(at, rgb, size, seed, q, age)",
        wgsl: Some(slam::dust_wgsl),
        probes: &[
            probe(Kind::SlamDust, (1512.0, 800.0), (420.0, 0.0), 32.0, 5),
            probe(Kind::SlamDust, (700.0, 400.0), (150.0, 0.0), 12.0, 77),
        ],
    },
    Motion {
        kind: Kind::Ghosts,
        life: angels::ANGELS_LIFE,
        quad: angels::ghost,
        call: "ghost(at, rgb, size, seed, q, age)",
        wgsl: Some(angels::ghost_wgsl),
        probes: &[probe(
            Kind::Ghosts,
            (800.0, -90.0),
            (720.0, 675.0),
            90.0,
            4243,
        )],
    },
];
