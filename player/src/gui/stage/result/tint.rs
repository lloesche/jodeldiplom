//! The diploma's tint: from the cover's palette where its layers
//! have one (its most vivid colour lights the screen and its border, a
//! deep shade of it its ink), else the stage's own colours.

use super::super::{Rgb, ACTIVE, CORAL, GOLD};
use crate::cover::Swatch;

/// The screen's tint: its light's colour (vivid), the ink's (deep), the
/// border's and the ribbons' two, and a light one for its small words off
/// the paper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tint {
    /// The light's.
    pub glow: Rgb,
    /// The ink's.
    pub ink: Rgb,
    /// The border's and the ribbons'.
    pub ribbon: (Rgb, Rgb),
    /// The small words'.
    pub soft: Rgb,
}

fn hsv(h: f32, s: f32, v: f32) -> Rgb {
    let c = crate::gui::shine::hue(h);
    c.map(|k| v * (1.0 - s + s * k))
}

/// `c`'s hue, saturation and value.
fn to_hsv(c: Rgb) -> (f32, f32, f32) {
    let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = max - min;
    if d < 1e-4 {
        return (0.0, 0.0, max);
    }
    let h = if max == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    (h / 6.0, d / max, max)
}

impl Tint {
    /// The stage's own: gold light, a deep violet ink, coral and blue.
    pub const STAGE: Tint = Tint {
        glow: GOLD,
        ink: [0.07, 0.05, 0.14],
        ribbon: (CORAL, [0.32, 0.52, 1.0]),
        soft: ACTIVE,
    };

    /// From a cover's palette (largest share first): its most vivid colour
    /// (that is not too dark) for the light, the ribbons and the words, a
    /// deep shade of it for the ink; the stage's own without one.
    pub fn of(palette: &[Swatch]) -> Tint {
        let rgb = |s: &Swatch| s.rgb.map(|v| f32::from(v) / 255.0);
        let vivid = |s: &&Swatch| {
            let (_, sat, val) = to_hsv(rgb(s));
            sat * val.min(0.7) * (0.25 + s.share).sqrt()
        };
        let Some(best) = palette.iter().max_by(|a, b| vivid(a).total_cmp(&vivid(b))) else {
            return Tint::STAGE;
        };
        let (h, s, _) = to_hsv(rgb(best));
        if s < 0.15 {
            // A grey cover: its own deep shade, the stage's light.
            let (_, _, v) = to_hsv(rgb(&palette[0]));
            let g = 0.05 + 0.1 * v;
            return Tint {
                ink: [g, g, g * 1.1],
                ..Tint::STAGE
            };
        }
        // The ribbon's second colour: another of the cover's, far enough
        // round the wheel, else the complement.
        let other = palette
            .iter()
            .map(|p| to_hsv(rgb(p)))
            .find(|&(oh, os, _)| os > 0.25 && (0.5 - ((oh - h).rem_euclid(1.0) - 0.5).abs()) > 0.12)
            .map_or(h + 0.45, |(oh, _, _)| oh);
        Tint {
            glow: hsv(h, s.max(0.6), 1.0),
            ink: hsv(h, s.clamp(0.4, 0.7), 0.17),
            ribbon: (hsv(h, s.max(0.65), 0.95), hsv(other, 0.65, 0.95)),
            soft: hsv(h, 0.3, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cover's most vivid colour lights the screen; a grey one keeps
    /// the stage's light; none, the stage's own.
    #[test]
    fn the_tint_is_the_cover_s() {
        assert_eq!(Tint::of(&[]), Tint::STAGE);
        let swatch = |rgb, share| Swatch { rgb, share };
        let blue = Tint::of(&[swatch([40, 40, 40], 0.6), swatch([30, 60, 200], 0.2)]);
        assert!(blue.glow[2] > blue.glow[0] && blue.glow[2] > 0.9);
        assert!(blue.ink.iter().all(|&c| c < 0.25));
        let grey = Tint::of(&[swatch([120, 120, 120], 1.0)]);
        assert_eq!(grey.glow, Tint::STAGE.glow);
    }
}
