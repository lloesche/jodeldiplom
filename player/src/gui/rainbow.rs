//! A streak's shine as a clock the looks share: while the singing shines
//! (a [`Glow`] level above 0: the glow of the tier the last phrase
//! reached, 0 to 1, [`crate::tiers::TierSpec::glow`]) it eases in over
//! [`RISE`] from how strong it showed; lost, it collapses over
//! [`COLLAPSE`] ([`shine_at`]). The rainbow's bands, swirl, rim and text
//! ([`super::fx::rainbow`]), the ring of the time left
//! ([`super::glory::ring_at`]) and the tiers' held looks
//! ([`crate::tiers::held`]) all go by it; the rainbow's colours are
//! [`hue`].
//!
//! The processor gives only the level (when it changed, and from what);
//! everything moves by the time in the shaders. These functions are their
//! reference ([`stage_wgsl`]), held to them on the GPU
//! (`super::shader_check`).

use crate::core::shine::Tier;
use crate::tiers::TIERS;

/// How long a level takes to come in, in seconds.
pub const RISE: f32 = 0.25;
/// How long the shine takes to collapse into the playhead when lost.
pub const COLLAPSE: f32 = 0.5;
/// The least strength that shows.
pub const SHOWS: f32 = 0.00025;
/// How strong the shine is once the swirl, the rim, the text's tint and
/// the time left's ring are whole in the rainbow (they come in as it
/// grows to it): the first tier's glow.
pub const WHOLE: f32 = TIERS[0].glow;

/// A streak's shine as the shaders take it: its level (0: none, up to the
/// top's 1), how strong it showed when it changed (its level before, or
/// part way there) and when that was (seconds, the singing's clock).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Glow {
    /// Its level.
    pub level: f32,
    /// How strong it showed as it changed.
    pub from: f32,
    /// When it changed.
    pub since: f32,
}

impl Glow {
    /// How strong it is at `time`, and how far it has collapsed
    /// ([`shine_at`]).
    pub fn at(&self, time: f32) -> (f32, f32) {
        shine_at(self, time)
    }

    /// Whether anything of it shows at `time`.
    pub fn shows(&self, time: f32) -> bool {
        self.at(time).0 > SHOWS
    }

    /// As the shaders take it: level, from, since.
    pub fn triple(&self) -> [f32; 3] {
        [self.level, self.from, self.since]
    }
}

pub(crate) fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// `x` held to 0 to 1.
pub(crate) fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// WGSL's `smoothstep`.
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    smooth((x - e0) / (e1 - e0))
}

pub(crate) fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// How far `p` is outside a box `half` its size round the origin, its
/// corners `r` round (negative inside): the shaders' `round_box`.
pub fn round_box(p: (f32, f32), half: (f32, f32), r: f32) -> f32 {
    let (qx, qy) = (p.0.abs() - half.0 + r, p.1.abs() - half.1 + r);
    (qx.max(0.0).hypot(qy.max(0.0))) + qx.max(qy).min(0.0) - r
}

/// A colour of the rainbow, `h` round it (a whole turn every 1).
pub fn hue(h: f32) -> [f32; 3] {
    let h = fract(h) * 6.0;
    [
        ((h - 3.0).abs() - 1.0).clamp(0.0, 1.0),
        (2.0 - (h - 2.0).abs()).clamp(0.0, 1.0),
        (2.0 - (h - 4.0).abs()).clamp(0.0, 1.0),
    ]
}

/// How strong `s` is at `time`, and how far it has collapsed (0 to 1):
/// easing over [`RISE`] from how strong it showed to its level, up or down
/// a level alike; lost (its level 0), collapsing over [`COLLAPSE`].
pub fn shine_at(s: &Glow, time: f32) -> (f32, f32) {
    let t = (time - s.since).max(0.0);
    if s.level <= 0.0 && s.from > 0.0 {
        let c = (t / COLLAPSE).clamp(0.0, 1.0);
        (s.from * (1.0 - c), c)
    } else {
        (s.from + (s.level - s.from) * smooth(t / RISE), 0.0)
    }
}

/// [`hue`] and [`shine_at`] in WGSL (and `TAU`, which the stage's shader
/// has not otherwise), with [`crate::tiers::held`] over the rows' glows
/// (and `TIERS`, how many there are, for the rest of the shader): for the
/// stage's shader.
pub fn stage_wgsl() -> String {
    format!(
        "
const TAU: f32 = {:?};
const RISE: f32 = {RISE:?};
const COLLAPSE: f32 = {COLLAPSE:?};
const SHOWS: f32 = {SHOWS:?};
const WHOLE: f32 = {WHOLE:?};
const TIERS: u32 = {}u;
const GLOWS: array<f32, TIERS> = {};
{HUE}{SHINE}",
        super::sparks::TAU,
        Tier::COUNT,
        crate::tiers::wgsl_array(TIERS.iter().map(|t| t.glow)),
    )
}

/// [`hue`] in WGSL.
pub const HUE: &str = r"
fn hue(h0: f32) -> vec3<f32> {
    let h = fract(h0) * 6.0;
    return vec3<f32>(
        clamp(abs(h - 3.0) - 1.0, 0.0, 1.0),
        clamp(2.0 - abs(h - 2.0), 0.0, 1.0),
        clamp(2.0 - abs(h - 4.0), 0.0, 1.0),
    );
}
";

const SHINE: &str = r"
// How strong a shine (level, from, since) is at `time`, and how far it
// has collapsed.
fn shine_at(s: vec3<f32>, time: f32) -> vec2<f32> {
    let t = max(time - s.z, 0.0);
    if s.x <= 0.0 && s.y > 0.0 {
        let c = clamp(t / COLLAPSE, 0.0, 1.0);
        return vec2<f32>(s.y * (1.0 - c), c);
    }
    return vec2<f32>(s.y + (s.x - s.y) * smoothstep(0.0, 1.0, t / RISE), 0.0);
}

// A held look at glow `k`: the straight line through none at 0 and each
// row's value at its glow, the top's from there on.
fn held(k: f32, values: array<f32, TIERS>) -> f32 {
    var a = vec2<f32>(0.0, 0.0);
    for (var i = 0u; i < TIERS; i = i + 1u) {
        let b = vec2<f32>(GLOWS[i], values[i]);
        if k < b.x {
            return a.y + (b.y - a.y) * ((k - a.x) / (b.x - a.x));
        }
        a = b;
    }
    return a.y;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    fn shine(level: f32, from: f32) -> Glow {
        Glow {
            level,
            from,
            since: 10.0,
        }
    }

    /// A new level comes in over [`RISE`]; lost, the shine collapses over
    /// [`COLLAPSE`] and is gone.
    #[test]
    fn a_new_level_rises_and_a_lost_one_collapses() {
        let rise = shine(0.5, 0.25);
        assert_eq!(shine_at(&rise, 10.0).0, 0.25);
        assert!((shine_at(&rise, 10.0 + RISE).0 - 0.5).abs() < 1e-6);
        assert_eq!(shine_at(&rise, 99.0), (0.5, 0.0));
        let lost = shine(0.0, 0.75);
        let (k, c) = shine_at(&lost, 10.25);
        assert!((c - 0.5).abs() < 1e-6 && (k - 0.375).abs() < 1e-6);
        assert_eq!(shine_at(&lost, 10.0 + COLLAPSE), (0.0, 1.0));
        assert!(!lost.shows(10.0 + COLLAPSE) && lost.shows(10.1));
    }

    /// Down a tier (Amazing's shine to Great's) it eases down over [`RISE`]
    /// and shines on at Great's: no collapse.
    #[test]
    fn a_drop_of_a_level_eases_down_and_shines_on() {
        let drop = shine(0.5, 0.75);
        assert_eq!(drop.at(10.0), (0.75, 0.0));
        let half = drop.at(10.0 + RISE / 2.0);
        assert!(half.0 < 0.75 && half.0 > 0.5 && half.1 == 0.0, "{half:?}");
        assert_eq!(drop.at(99.0), (0.5, 0.0));
        assert!(drop.shows(99.0));
    }

    #[test]
    fn the_rainbow_goes_round() {
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4);
        assert!(near(hue(0.0), [1.0, 0.0, 0.0]));
        assert!(near(hue(1.0 / 3.0), [0.0, 1.0, 0.0]));
        assert!(near(hue(2.0 / 3.0), [0.0, 0.0, 1.0]));
        assert!(near(hue(-1.0 / 3.0), hue(2.0 / 3.0)));
    }
}
