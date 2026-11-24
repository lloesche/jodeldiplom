//! The window's look in iced: the player's colours as iced's, the fonts,
//! and the styles of the console, the strips, the scribble strips, the
//! buttons and the lamps: dark brushed panels, backlit labels, buttons that
//! glow when on.

use iced::font::{Family, Weight};
use iced::widget::{button, container};
use iced::{border, gradient, Background, Border, Color, Font, Radians, Shadow, Vector};

use crate::view::colors;

/// A cell colour as iced's.
pub const fn rgb(c: ratatui::style::Color) -> Color {
    match c {
        ratatui::style::Color::Rgb(r, g, b) => Color {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        },
        _ => Color::WHITE,
    }
}

pub const BG: Color = rgb(colors::BG);
pub const ACCENT: Color = rgb(colors::ACCENT);
pub const DIM: Color = rgb(colors::DIM);
pub const TEXT: Color = rgb(colors::TEXT);
pub const BORDER: Color = rgb(colors::BORDER);
pub const BUTTON: Color = rgb(colors::BUTTON);
pub const GREEN: Color = rgb(colors::GREEN);
pub const WARN: Color = rgb(colors::WARN);
pub const FAIL: Color = rgb(colors::FAIL);
pub const SOURCE: Color = rgb(colors::SOURCE);

/// Text on a lit scribble strip or button.
pub const ON_LIT: Color = Color::from_rgb(0.08, 0.08, 0.09);

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

/// Numbers that change (levels, clocks): the player's fixed-width face
/// (Sarasa Fixed J, else Cascadia Mono), so they do not jitter.
pub fn digits() -> Font {
    if crate::gui::fonts::built_in_fonts().0 {
        Font::new("Sarasa Fixed J")
    } else {
        Font::new("Cascadia Mono")
    }
}

pub fn with_alpha(c: Color, a: f32) -> Color {
    Color { a, ..c }
}

pub fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

fn vertical(top: Color, bottom: Color) -> Background {
    Background::Gradient(
        gradient::Linear::new(Radians(std::f32::consts::PI))
            .add_stop(0.0, top)
            .add_stop(1.0, bottom)
            .into(),
    )
}

/// The desk's console: brushed dark metal.
pub fn console() -> container::Style {
    container::Style {
        background: Some(vertical(
            Color::from_rgb8(44, 44, 52),
            Color::from_rgb8(26, 26, 31),
        )),
        border: Border {
            color: Color::from_rgb8(70, 70, 82),
            width: 1.0,
            radius: 10.0.into(),
        },
        shadow: Shadow {
            color: with_alpha(Color::BLACK, 0.6),
            offset: Vector::new(0.0, 6.0),
            blur_radius: 18.0,
        },
        ..container::Style::default()
    }
}

/// A strip's panel; lit at its edge while selected.
pub fn strip(selected: bool, master: bool) -> container::Style {
    let (top, bottom) = if master {
        (Color::from_rgb8(40, 40, 50), Color::from_rgb8(28, 28, 35))
    } else {
        (Color::from_rgb8(36, 36, 43), Color::from_rgb8(27, 27, 32))
    };
    container::Style {
        background: Some(vertical(top, bottom)),
        border: Border {
            color: if selected {
                with_alpha(ACCENT, 0.85)
            } else {
                Color::from_rgb8(16, 16, 19)
            },
            width: if selected { 1.5 } else { 1.0 },
            radius: 6.0.into(),
        },
        shadow: if selected {
            Shadow {
                color: with_alpha(ACCENT, 0.25),
                offset: Vector::ZERO,
                blur_radius: 10.0,
            }
        } else {
            Shadow::default()
        },
        ..container::Style::default()
    }
}

/// The scribble strip: a backlit label, lit in the accent while selected.
pub fn scribble(selected: bool, status: button::Status) -> button::Style {
    let hover = matches!(status, button::Status::Hovered);
    let bg = if selected {
        ACCENT
    } else if hover {
        Color::from_rgb8(52, 52, 64)
    } else {
        Color::from_rgb8(20, 20, 25)
    };
    button::Style {
        background: Some(Background::Color(bg)),
        text_color: if selected { ON_LIT } else { TEXT },
        border: Border {
            color: Color::from_rgb8(8, 8, 10),
            width: 1.0,
            radius: 3.0.into(),
        },
        shadow: if selected {
            Shadow {
                color: with_alpha(ACCENT, 0.45),
                offset: Vector::ZERO,
                blur_radius: 8.0,
            }
        } else {
            Shadow::default()
        },
        ..button::Style::default()
    }
}

/// A desk's square push button (mute, solo, echo): lit in `color` with a
/// glow when on.
pub fn push(on: bool, color: Color, status: button::Status) -> button::Style {
    let pressed = matches!(status, button::Status::Pressed);
    let hover = matches!(status, button::Status::Hovered);
    let (top, bottom) = if on {
        (mix(color, Color::WHITE, 0.25), color)
    } else if hover {
        (Color::from_rgb8(70, 70, 82), Color::from_rgb8(48, 48, 58))
    } else {
        (Color::from_rgb8(60, 60, 70), Color::from_rgb8(38, 38, 46))
    };
    button::Style {
        background: Some(if pressed {
            vertical(bottom, top)
        } else {
            vertical(top, bottom)
        }),
        text_color: if on { ON_LIT } else { DIM },
        border: Border {
            color: Color::from_rgb8(10, 10, 12),
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: if on {
            Shadow {
                color: with_alpha(color, 0.55),
                offset: Vector::ZERO,
                blur_radius: 10.0,
            }
        } else {
            Shadow {
                color: with_alpha(Color::BLACK, 0.5),
                offset: Vector::new(0.0, 2.0),
                blur_radius: 2.0,
            }
        },
        ..button::Style::default()
    }
}

/// A lamp: a small LED, glowing in `color` when lit.
pub fn lamp(lit: bool, color: Color) -> container::Style {
    container::Style {
        background: Some(Background::Color(if lit {
            mix(color, Color::WHITE, 0.15)
        } else {
            Color::from_rgb8(46, 46, 54)
        })),
        border: Border {
            color: if lit {
                color
            } else {
                Color::from_rgb8(16, 16, 19)
            },
            width: 1.0,
            radius: 4.0.into(),
        },
        shadow: if lit {
            Shadow {
                color: with_alpha(color, 0.8),
                offset: Vector::ZERO,
                blur_radius: 6.0,
            }
        } else {
            Shadow::default()
        },
        ..container::Style::default()
    }
}

/// The level's readout: a small dark window.
pub fn readout() -> container::Style {
    container::Style {
        background: Some(Background::Color(Color::from_rgb8(12, 12, 15))),
        border: Border {
            color: Color::from_rgb8(56, 56, 66),
            width: 1.0,
            radius: 3.0.into(),
        },
        ..container::Style::default()
    }
}

/// A bar's track and its lit part (the ring, the device's volume).
pub fn bar(color: Color) -> container::Style {
    container::Style {
        background: Some(Background::Color(color)),
        border: border::rounded(2),
        ..container::Style::default()
    }
}
