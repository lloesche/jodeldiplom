//! What the window's iced side takes beside the native look
//! ([`super::look`]): the ground under the old canvas, the face iced
//! draws in unless told, and mixing two colours.

use iced::font::{Family, Weight};
use iced::{Color, Font};

use crate::view::palette;

/// The window's ground under the old canvas.
pub const BG: Color = Color::from_rgb8(palette::BG[0], palette::BG[1], palette::BG[2]);

/// The labels and lines: a proportional face, the built-in Atkinson
/// Hyperlegible Next (iced's own without it).
pub fn ui() -> Font {
    match crate::gui::fonts::lyrics_font() {
        Some(_) => Font {
            family: Family::Name("Atkinson Hyperlegible Next"),
            weight: Weight::Semibold,
            ..Font::DEFAULT
        },
        None => Font::DEFAULT,
    }
}

/// `t` of the way from `a` to `b`, each channel straight.
pub fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}
