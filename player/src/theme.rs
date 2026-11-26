//! The karaoke game's themes: a [`Theme`] overrides, by key, how the
//! tiers ([`crate::tiers`]), their effects ([`crate::gui::fx`]) and their
//! sounds ([`crate::sfx::conductor`]) look and sound, and may grade the
//! picture. Everything that has a colour, a look or a patch the theme may
//! change reads it through the theme in force: a tier's colour
//! ([`Theme::color`]), the count-in's accent ([`Theme::accent`]), an
//! effect's look ([`Theme::fx_params`], the four numbers its module
//! names), an atom's patch ([`Theme::patch`]); what a theme holds
//! throughout (the fog at the edges: [`Theme::always`]) and the cold
//! grade over the backdrop ([`Theme::grade`]). Whatever a theme leaves
//! out is as the tier's row, the effect's spec or the palette has it.
//!
//! [`CLASSIC`] overrides nothing: the game as it is made. [`SPOOKY`]
//! ([`spooky`]) is its Halloween: green fire, pumpkins for bulbs, candy
//! for coins, ghosts for angels, a cold backdrop and fog at the edges with
//! eyes in it, and a ghostly choir, tolling bells and a skeleton's
//! xylophone. The theme is a setting ([`crate::config::Config::karaoke_theme`]).

mod spooky;

use jodel_synth::Patch;

use crate::gui::fx::{Fx, Looks, Use};
use crate::sfx::conductor::{self, Atom};
use crate::tiers::Tier;

pub use spooky::SPOOKY;

/// A theme: its name (as the settings file has it), and what it overrides:
/// tiers' colours (by the tier's name), the count-in's accent, effects'
/// looks, atoms' patches; the effects it holds throughout, and the
/// backdrop's grade.
#[derive(Debug)]
pub struct Theme {
    /// Its name.
    pub name: &'static str,
    /// The tiers' colours (sRGB), by their names.
    pub colors: &'static [(&'static str, [f32; 3])],
    /// The count-in's accent (sRGB bytes), if it has its own.
    pub accent: Option<[u8; 3]>,
    /// The effects' looks.
    pub looks: &'static [(Fx, [f32; 4])],
    /// The atoms' patches.
    pub sounds: &'static [(Atom, Patch)],
    /// The effects held throughout, at their strengths, whatever the
    /// singing (moved by their mods).
    pub always: &'static [Use],
    /// The backdrop's grade: what its colours are tinted to once
    /// desaturated (sRGB), and how much (0: none).
    pub grade: [f32; 4],
}

/// The game as it is made: nothing overridden.
pub static CLASSIC: Theme = Theme {
    name: "classic",
    colors: &[],
    accent: None,
    looks: &[],
    sounds: &[],
    always: &[],
    grade: [1.0, 1.0, 1.0, 0.0],
};

/// The themes, the first the default.
pub static THEMES: [&Theme; 2] = [&CLASSIC, &SPOOKY];

impl Theme {
    /// The theme named `name`, if there is one.
    pub fn named(name: &str) -> Option<&'static Theme> {
        THEMES.into_iter().find(|t| t.name == name)
    }

    /// `tier`'s colour (sRGB).
    pub fn color(&self, tier: Tier) -> [f32; 3] {
        (self.colors.iter())
            .find(|(name, _)| *name == tier.name())
            .map_or(tier.spec().color, |&(_, c)| c)
    }

    /// The count-in's accent (sRGB bytes).
    pub fn accent(&self) -> [u8; 3] {
        self.accent.unwrap_or(crate::view::palette::ACCENT)
    }

    /// `fx`'s look ([`crate::gui::fx::Spec::params`]).
    pub fn fx_params(&self, fx: Fx) -> [f32; 4] {
        (self.looks.iter())
            .find(|(f, _)| *f == fx)
            .map_or(fx.spec().params, |&(_, p)| p)
    }

    /// Every effect's look.
    pub fn looks(&self) -> Looks {
        let mut looks = Looks::SPECS;
        for &(fx, p) in self.looks {
            looks.0[fx.index()] = p;
        }
        looks
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

    /// Every theme named apart, every key one there is and each once: a
    /// tier's colour by a tier's name, an effect's look, an atom's patch,
    /// an effect held throughout; a grade's amount 0 to 1.
    #[test]
    fn every_theme_overrides_what_there_is_once() {
        let mut names: Vec<&str> = THEMES.iter().map(|t| t.name).collect();
        names.dedup();
        assert_eq!(names.len(), THEMES.len());
        for t in THEMES {
            assert_eq!(Theme::named(t.name), Some(t));
            for (i, (name, _)) in t.colors.iter().enumerate() {
                assert_eq!(Tier::named(name).name(), *name);
                assert!(t.colors[i + 1..].iter().all(|(n, _)| n != name), "{name}");
            }
            for (i, (fx, _)) in t.looks.iter().enumerate() {
                assert!(t.looks[i + 1..].iter().all(|(f, _)| f != fx), "{fx:?}");
            }
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
            assert!((0.0..=1.0).contains(&t.grade[3]), "{}", t.name);
        }
        assert_eq!(Theme::named("fabulous"), None);
    }

    /// Classic is the game as made: every tier's colour its row's, every
    /// look its effect's spec's, every patch the palette's, nothing held
    /// and no grade.
    #[test]
    fn classic_overrides_nothing() {
        let c: &Theme = Default::default();
        assert_eq!(c, &CLASSIC);
        for t in Tier::ALL {
            assert_eq!(c.color(t), t.spec().color);
        }
        assert_eq!(c.looks(), Looks::SPECS);
        for a in Atom::ALL {
            assert_eq!(c.patch(a), conductor::patch(a));
        }
        assert_eq!(c.accent(), crate::view::palette::ACCENT);
        assert!(c.always.is_empty() && c.grade[3] == 0.0);
    }

    /// Spooky looks and sounds other than classic: every tier its own
    /// colour, its own accent, the fire green, a cold grade, the fog held.
    #[test]
    fn spooky_is_its_own() {
        for t in Tier::ALL {
            assert_ne!(SPOOKY.color(t), t.spec().color, "{}", t.name());
        }
        assert_ne!(SPOOKY.accent(), CLASSIC.accent());
        assert_eq!(SPOOKY.fx_params(Fx::EdgeFire)[0], 1.0);
        assert!(SPOOKY.grade[3] > 0.0 && SPOOKY.grade[2] > SPOOKY.grade[0]);
        assert!(SPOOKY.holds(Fx::Fog));
        assert!(!SPOOKY.sounds.is_empty());
    }
}
