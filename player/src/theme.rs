//! The karaoke game's themes: a [`Theme`] is one data file
//! ([`classic`], [`spooky`]) saying how the tiers ([`crate::tiers`]),
//! their effects ([`crate::gui::fx`]) and their sounds
//! ([`crate::sfx::conductor`]) look and sound, and how the picture is
//! graded. What it may override it names by its own key: a tier's colour
//! ([`Theme::color`], by the tier), the count-in's accent
//! ([`Theme::accent`]), an atom's patch ([`Theme::patch`]); whatever it
//! leaves out is as the tier's row or the palette has it. How it dresses
//! the effects it always says: the fire's colours ([`Ramp`]), the marquee's
//! bulbs ([`Palette`], [`BulbShape`]), the coins ([`Theme::coins`]), the
//! angels ([`Flight`]); the shaders read those from one uniform
//! ([`Theme::uniform`], [`wgsl`]). It holds effects throughout (the fog at
//! the edges: [`Theme::always`]) and grades the backdrop ([`Grade`]).
//!
//! [`CLASSIC`] is the game as it is made. [`SPOOKY`] is its Halloween:
//! green fire, pumpkins for bulbs, candy for coins, ghosts for angels, a
//! cold backdrop and fog at the edges with eyes in it, and a ghostly
//! choir, tolling bells and a skeleton's xylophone. The theme is a setting
//! ([`crate::config::Config::karaoke_theme`]); everything reads it as the
//! theme in force now ([`crate::core::State::theme`]), so a change is the
//! next frame's, its sounds' patches told the conductor's synth as it
//! plays.

mod classic;
mod spooky;

use jodel_synth::Patch;

use crate::gui::fx::{Fx, Use};
use crate::gui::stage::palette::Rgb;
use crate::sfx::conductor::{self, Atom};
use crate::tiers::Tier;

pub use classic::CLASSIC;
pub use spooky::SPOOKY;

/// How many colours fire runs through ([`Ramp`]).
pub const FIRE_STOPS: usize = 5;
/// How many colours a marquee's bulbs have ([`Palette`]).
pub const BULB_COLORS: usize = 5;

/// Fire's colours, the one fire every burning effect draws
/// ([`crate::gui::fx`]: the screen's edges, the lane's flames, the line
/// sung), coldest first: its colour at heat 0, and those it is wholly by
/// heat 0.3, 0.55, 0.8 and 1, eased between (as made: deep red, red,
/// orange, yellow, white hot).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp(pub [Rgb; FIRE_STOPS]);

impl Ramp {
    /// As the shaders read it ([`Uniform::fire`]): a vec4 a colour.
    pub fn uniform(&self) -> [[f32; 4]; FIRE_STOPS] {
        self.0.map(vec4)
    }

    /// The ramp in the vec4s `v` as the shaders read it (the first
    /// [`FIRE_STOPS`]).
    pub fn of_uniform(v: &[[f32; 4]]) -> Ramp {
        Ramp(std::array::from_fn(|i| [v[i][0], v[i][1], v[i][2]]))
    }
}

/// `c` as a vec4 of the uniform (its last unused).
fn vec4(c: Rgb) -> [f32; 4] {
    [c[0], c[1], c[2], 0.0]
}

/// The marquee's bulbs' colours: a bulb lit, its glow, a bulb unlit, its
/// socket, the socket's rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// A bulb lit.
    pub on: Rgb,
    /// Its glow.
    pub glow: Rgb,
    /// A bulb unlit.
    pub off: Rgb,
    /// Its socket.
    pub socket: Rgb,
    /// The socket's rim.
    pub rim: Rgb,
}

impl Palette {
    /// As the shaders read it ([`Uniform::bulbs`]): its colours in their
    /// order here, a vec4 each.
    pub fn uniform(&self) -> [[f32; 4]; BULB_COLORS] {
        [self.on, self.glow, self.off, self.socket, self.rim].map(vec4)
    }

    /// The palette in the vec4s `v` as the shaders read it (the first
    /// [`BULB_COLORS`]).
    pub fn of_uniform(v: &[[f32; 4]]) -> Palette {
        let c = |i: usize| [v[i][0], v[i][1], v[i][2]];
        Palette {
            on: c(0),
            glow: c(1),
            off: c(2),
            socket: c(3),
            rim: c(4),
        }
    }
}

/// What a marquee's bulb is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulbShape {
    /// A round light bulb.
    Bulb,
    /// A jack-o'-lantern grinning, its stalk on top
    /// ([`crate::gui::fx::bulbs::jack`]).
    Jack,
}

impl BulbShape {
    /// How much of a jack-o'-lantern it is: 0 a bulb, 1 a jack-o'-lantern
    /// (what the bulbs' maths mixes by).
    pub fn pumpkin(self) -> f32 {
        match self {
            BulbShape::Bulb => 0.0,
            BulbShape::Jack => 1.0,
        }
    }

    /// As the shaders read it ([`Uniform::bulb`]): `x` how much of a
    /// jack-o'-lantern it is ([`BulbShape::pumpkin`]).
    pub fn uniform(self) -> [f32; 4] {
        [self.pumpkin(), 0.0, 0.0, 0.0]
    }

    /// The shape in the vec4 `v` as the shaders read it: a jack-o'-lantern
    /// from half of one up.
    pub fn of_uniform(v: [f32; 4]) -> BulbShape {
        if v[0] >= 0.5 {
            BulbShape::Jack
        } else {
            BulbShape::Bulb
        }
    }
}

/// What comes down while the singing is at the top
/// ([`crate::gui::fx::angels`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flight {
    /// A flight of angels.
    Angels,
    /// A flight of little sheet ghosts.
    Ghosts,
}

/// The backdrop's grade: what its colours are tinted to once desaturated
/// (sRGB), and how much (0: none, to 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    /// The tint.
    pub tint: Rgb,
    /// How much.
    pub amount: f32,
}

impl Grade {
    /// No grade at all.
    pub const NONE: Grade = Grade {
        tint: [1.0; 3],
        amount: 0.0,
    };

    /// As the backdrop's shaders read it: the tint, then the amount.
    pub fn uniform(self) -> [f32; 4] {
        let [r, g, b] = self.tint;
        [r, g, b, self.amount]
    }
}

/// A theme: its name (as the settings file has it), its title and what it
/// is in a few words (as the settings screen and the Synth tab show it),
/// what it overrides (the tiers' colours, the count-in's accent, atoms'
/// patches), how it dresses the effects, the effects it holds throughout,
/// the backdrop's grade and its look.
#[derive(Debug)]
pub struct Theme {
    /// Its name.
    pub name: &'static str,
    /// Its title.
    pub title: &'static str,
    /// What it is, in a few words.
    pub about: &'static str,
    /// The tiers' colours (sRGB), Good first, if it has its own.
    pub colors: Option<[Rgb; Tier::COUNT]>,
    /// The count-in's accent (sRGB), if it has its own.
    pub accent: Option<Rgb>,
    /// The fire's colours.
    pub fire: Ramp,
    /// The marquee's bulbs' colours.
    pub bulbs: Palette,
    /// What a bulb is.
    pub bulb: BulbShape,
    /// The colours the coins come in, each spray of them one by its seed
    /// (none: gold).
    pub coins: Option<&'static [Rgb]>,
    /// What comes down at the top.
    pub angels: Flight,
    /// The atoms' patches.
    pub sounds: &'static [(Atom, Patch)],
    /// The effects held throughout, at their strengths, whatever the
    /// singing (moved by their mods).
    pub always: &'static [Use],
    /// The backdrop's grade.
    pub grade: Grade,
    /// The look over the backdrop, a preset by its id
    /// ([`jodel_looks::presets`]; none: none, the backdrop as it is). A
    /// song's own look, where it has one, is drawn instead
    /// ([`crate::core::State::look`]).
    pub look: Option<&'static str>,
}

/// The themes, the first the default.
pub static THEMES: [&Theme; 2] = [&CLASSIC, &SPOOKY];

impl Theme {
    /// The theme named `name`, if there is one.
    pub fn named(name: &str) -> Option<&'static Theme> {
        THEMES.into_iter().find(|t| t.name == name)
    }

    /// `tier`'s colour (sRGB).
    pub fn color(&self, tier: Tier) -> Rgb {
        self.colors.map_or(tier.spec().color, |c| c[tier.index()])
    }

    /// The count-in's accent (sRGB).
    pub fn accent(&self) -> Rgb {
        self.accent
            .unwrap_or(crate::view::palette::ACCENT.map(|b| f32::from(b) / 255.0))
    }

    /// `atom`'s patch.
    pub fn patch(&self, atom: Atom) -> Patch {
        (self.sounds.iter())
            .find(|(a, _)| *a == atom)
            .map_or_else(|| conductor::patch(atom), |&(_, p)| p)
    }

    /// Whether it holds `fx` throughout.
    pub fn holds(&self, fx: Fx) -> bool {
        self.always.iter().any(|u| u.fx == fx)
    }

    /// How it dresses the effects, as the shaders read it ([`wgsl`]).
    pub fn uniform(&self) -> Uniform {
        Uniform {
            fire: self.fire.uniform(),
            bulbs: self.bulbs.uniform(),
            bulb: self.bulb.uniform(),
        }
    }
}

/// How a theme dresses the effects, as the shaders read it (`theme`): the
/// fire's colours, the bulbs' and what a bulb is (`x`: 1 a
/// jack-o'-lantern, 0 a bulb), each colour a vec4 (its last unused).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
    /// The fire's colours.
    pub fire: [[f32; 4]; FIRE_STOPS],
    /// The bulbs' colours.
    pub bulbs: [[f32; 4]; BULB_COLORS],
    /// What a bulb is.
    pub bulb: [f32; 4],
}

/// How many bytes [`Uniform`] is.
pub const UNIFORM_BYTES: u64 = std::mem::size_of::<Uniform>() as u64;

/// [`Uniform`] as the shaders declare it (`struct Theme`; the stage's
/// shader binds one as `theme`), with how many colours its arrays have
/// (`FIRE_STOPS`, `BULB_COLORS`).
pub fn wgsl() -> String {
    format!(
        "const FIRE_STOPS: u32 = {FIRE_STOPS}u;
const BULB_COLORS: u32 = {BULB_COLORS}u;
struct Theme {{
    fire: array<vec4<f32>, FIRE_STOPS>,
    bulbs: array<vec4<f32>, BULB_COLORS>,
    bulb: vec4<f32>,
}};
"
    )
}

impl PartialEq for Theme {
    /// Themes are one by name.
    fn eq(&self, other: &Theme) -> bool {
        self.name == other.name
    }
}

impl Eq for Theme {}

impl Default for &'static Theme {
    fn default() -> &'static Theme {
        &CLASSIC
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every theme named apart, every patch it overrides an atom's once,
    /// every effect it holds throughout once and a screen pass; its grade's
    /// amount 0 to 1, its coins' colours some, its look a preset.
    #[test]
    fn every_theme_overrides_what_there_is_once() {
        let mut names: Vec<&str> = THEMES.iter().map(|t| t.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), THEMES.len());
        for t in THEMES {
            assert_eq!(Theme::named(t.name), Some(t));
            for (i, (atom, _)) in t.sounds.iter().enumerate() {
                assert!(t.sounds[i + 1..].iter().all(|(a, _)| a != atom), "{atom:?}");
            }
            for (i, u) in t.always.iter().enumerate() {
                assert!(t.always[i + 1..].iter().all(|v| v.fx != u.fx), "{:?}", u.fx);
                assert!(
                    u.fx.spec().is_screen(),
                    "{:?}: held throughout, it is a screen pass",
                    u.fx
                );
            }
            assert!((0.0..=1.0).contains(&t.grade.amount), "{}", t.name);
            assert!(t.coins.is_none_or(|c| !c.is_empty()), "{}", t.name);
            if let Some(id) = t.look {
                assert!(
                    jodel_looks::presets::named(id).is_some(),
                    "{}: {id}",
                    t.name
                );
            }
        }
        assert_eq!(Theme::named("fabulous"), None);
    }

    /// Classic is the game as made: every tier's colour its row's, every
    /// patch the palette's, the bulbs bulbs, the coins gold, the angels
    /// angels; nothing held and no grade.
    #[test]
    fn classic_overrides_nothing() {
        let c: &Theme = Default::default();
        assert_eq!(c, &CLASSIC);
        for t in Tier::ALL {
            assert_eq!(c.color(t), t.spec().color);
        }
        for a in Atom::ALL {
            assert_eq!(c.patch(a), conductor::patch(a));
        }
        let accent = crate::view::palette::ACCENT.map(|b| f32::from(b) / 255.0);
        assert_eq!(c.accent(), accent);
        assert_eq!(
            (c.bulb, c.coins, c.angels),
            (BulbShape::Bulb, None, Flight::Angels)
        );
        assert!(c.always.is_empty() && c.grade == Grade::NONE && c.look.is_none());
    }

    /// Every atom's patch in every theme is within its parameters' ranges
    /// (the Synth tab's sliders reach every value built in, and a patch
    /// kept as built reads back as built).
    #[test]
    fn every_patch_is_within_its_parameters_ranges() {
        for t in THEMES {
            for a in Atom::ALL {
                let p = t.patch(a);
                for n in p.knobs().filter(|n| n.applies) {
                    let (spec, v) = (n.spec, n.value);
                    let to = spec.kind.clamp(v);
                    assert!(
                        (to - v).abs() <= 1e-6 * v.abs().max(1.0),
                        "{} {a:?}: {} is {v}, out of {:?}",
                        t.name,
                        spec.key(),
                        spec.kind
                    );
                }
            }
        }
    }

    /// Spooky looks and sounds other than classic: every tier its own
    /// colour, its own accent, the fire green, jack-o'-lanterns, candy,
    /// ghosts, a cold grade, the fog held.
    #[test]
    fn spooky_is_its_own() {
        for t in Tier::ALL {
            assert_ne!(SPOOKY.color(t), t.spec().color, "{}", t.name());
        }
        assert_ne!(SPOOKY.accent(), CLASSIC.accent());
        let green = |r: &Ramp| r.0.iter().all(|c| c[1] >= c[0] && c[1] >= c[2]);
        assert!(green(&SPOOKY.fire) && !green(&CLASSIC.fire));
        assert_ne!(SPOOKY.bulbs, CLASSIC.bulbs);
        let dressed = (SPOOKY.bulb, SPOOKY.coins.is_some(), SPOOKY.angels);
        assert_eq!(dressed, (BulbShape::Jack, true, Flight::Ghosts));
        let g = SPOOKY.grade;
        assert!(g.amount > 0.0 && g.tint[2] > g.tint[0]);
        assert!(SPOOKY.holds(Fx::Fog));
        assert!(!SPOOKY.sounds.is_empty());
    }

    /// The uniform lies where the shaders read it: every field a whole
    /// number of vec4s, in the order `struct Theme` declares them.
    #[test]
    fn the_uniform_lies_where_the_shaders_read_it() {
        use std::mem::offset_of;
        let fields = [
            ("fire", offset_of!(Uniform, fire), "FIRE_STOPS", FIRE_STOPS),
            (
                "bulbs",
                offset_of!(Uniform, bulbs),
                "BULB_COLORS",
                BULB_COLORS,
            ),
            ("bulb", offset_of!(Uniform, bulb), "", 1),
        ];
        let declared = wgsl();
        let mut at = 0;
        for (name, offset, count, vec4s) in fields {
            assert_eq!(offset, at, "{name}");
            let ty = match count {
                "" => "vec4<f32>".to_owned(),
                n => {
                    assert!(declared.contains(&format!("const {n}: u32 = {vec4s}u;")));
                    format!("array<vec4<f32>, {n}>")
                }
            };
            assert!(declared.contains(&format!("    {name}: {ty},")), "{name}");
            at += 16 * vec4s;
        }
        assert_eq!(UNIFORM_BYTES, at as u64);
        let u = SPOOKY.uniform();
        assert_eq!(u.fire[0][..3], SPOOKY.fire.0[0]);
        assert_eq!(u.bulbs[4][..3], SPOOKY.bulbs.rim);
        assert_eq!(u.bulb[0], 1.0);
        assert_eq!(CLASSIC.uniform().bulb[0], 0.0);
    }
}
