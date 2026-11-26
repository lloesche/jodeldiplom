//! The native look's tokens, in one place: the colours, the spacing, the
//! corners, the type scale, the rows' heights and the motion. Every screen
//! drawn in iced's widgets takes its look from here and from
//! [`super::widgets`], never its own numbers.
//!
//! Calm and dark, one accent: layered dark surfaces ([`BG`], [`SURFACE`],
//! [`RAISED`]) with the music and the covers as the colour, the accent
//! ([`ACCENT`]) marking what is active. The colours start from the
//! player's palette ([`crate::view::palette`]), so it stays recognisable.

use std::time::Duration;

use iced::Color;

use crate::view::palette;

/// A colour of the palette as iced's.
const fn rgb([r, g, b]: [u8; 3]) -> Color {
    Color::from_rgb8(r, g, b)
}

/// The window's ground.
pub const BG: Color = rgb(palette::BG);
/// Panels, the list pane, the bottom bar.
pub const SURFACE: Color = Color::from_rgb8(0x17, 0x17, 0x1d);
/// Cards, menus, the palette overlay, hovered rows.
pub const RAISED: Color = Color::from_rgb8(0x1f, 0x1f, 0x27);
/// The selected row (with an accent bar at its left).
pub const SELECTED: Color = Color::from_rgb8(0x2a, 0x2a, 0x36);
/// The 1 px edges that remain, and the empty part of a track.
pub const EDGE: Color = Color::from_rgb8(0x2c, 0x2c, 0x36);
/// Primary text.
pub const TEXT: Color = rgb(palette::TEXT);
/// Secondary text and labels.
pub const TEXT_DIM: Color = Color::from_rgb8(0x9a, 0x9a, 0xa8);
/// Hints, and what cannot be used now.
pub const TEXT_FAINT: Color = Color::from_rgb8(0x6e, 0x6e, 0x7c);
/// The play head, the active tab, the focused control, the selection bar.
pub const ACCENT: Color = rgb(palette::ACCENT);
/// Text on the accent.
pub const ACCENT_INK: Color = Color::from_rgb8(0x16, 0x16, 0x0f);
/// Done, chosen, good news.
pub const GOOD: Color = rgb(palette::GREEN);
/// Warnings.
pub const WARN: Color = rgb(palette::WARN);
/// Errors.
pub const FAIL: Color = rgb(palette::FAIL);
/// Folders in the browser.
pub const FOLDER: Color = rgb(palette::FOLDER);
/// The file's rate, the spectrum's curve.
pub const SOURCE: Color = rgb(palette::SOURCE);

/// `color` at opacity `alpha`.
pub const fn alpha(color: Color, alpha: f32) -> Color {
    Color { a: alpha, ..color }
}

/// `share` of `tone` over `ground`, as one solid colour: mixed as the
/// eye sees it, so a faint tint stays faint (blended by its alpha on the
/// GPU, in linear light, a bright tone over the dark comes out strong).
pub fn tint(tone: Color, ground: Color, share: f32) -> Color {
    crate::gui::style::mix(ground, tone, share)
}

/// `color` as the bytes the pixel painters take (the lanes' tiles).
pub fn bytes(color: Color) -> [u8; 3] {
    let [r, g, b, _] = color.into_rgba8();
    [r, g, b]
}

/// The spacing scale, in points: between and around everything.
pub mod space {
    /// Within a group: an icon and its label, a title and its line.
    pub const XS: f32 = 4.0;
    /// Between controls side by side.
    pub const S: f32 = 8.0;
    /// Inside a row or a card.
    pub const M: f32 = 12.0;
    /// A panel's padding.
    pub const L: f32 = 16.0;
    /// Between the groups of a panel.
    pub const XL: f32 = 24.0;
    /// Round a panel that is a whole screen.
    pub const XXL: f32 = 32.0;
}

/// The corners' radii, in points.
pub mod radius {
    /// Controls: buttons, fields, keys.
    pub const CONTROL: f32 = 6.0;
    /// Panels and cards.
    pub const PANEL: f32 = 10.0;
    /// Pills: chips, toggles, tracks.
    pub const PILL: f32 = 999.0;
}

/// The type scale, in points.
pub mod size {
    /// Captions, details, key hints.
    pub const CAPTION: f32 = 12.0;
    /// Labels (semi-bold, dim).
    pub const LABEL: f32 = 13.0;
    /// Body text, rows, buttons.
    pub const BODY: f32 = 14.0;
    /// A screen's or a panel's title (bold).
    pub const TITLE: f32 = 18.0;
    /// The one big line of a screen (bold): Now Playing's title.
    pub const DISPLAY: f32 = 28.0;
}

/// Heights of the repeated pieces, in points.
pub mod height {
    /// A list row.
    pub const ROW: f32 = 34.0;
    /// A dense table's row.
    pub const DENSE_ROW: f32 = 28.0;
    /// A button.
    pub const CONTROL: f32 = 34.0;
    /// A chip.
    pub const CHIP: f32 = 26.0;
    /// A small button, in a dense panel (a mixer strip's).
    pub const SMALL_CONTROL: f32 = 24.0;
    /// A progress bar.
    pub const BAR: f32 = 6.0;
    /// A slider's or the seek bar's track.
    pub const TRACK: f32 = 4.0;
    /// An icon beside body text, and in a button.
    pub const ICON: f32 = 16.0;
}

/// Lines' widths, in points.
pub mod line {
    /// An edge, where a surface needs one.
    pub const EDGE: f32 = 1.0;
    /// The ring round the focused control.
    pub const FOCUS: f32 = 2.0;
}

/// The motion: what eases, and how long it takes.
pub mod motion {
    use super::Duration;

    /// Hover and selection ease in.
    pub const HOVER: Duration = Duration::from_millis(120);
    /// Panels and overlays fade and slide in.
    pub const PANEL: Duration = Duration::from_millis(160);
}

/// The faces of the type scale: Nunito (built in) for the interface,
/// iced's own sans-serif without it (the tests). Nunito's figures are all
/// one width (tabular), so numbers that change do not jitter in it.
pub mod font {
    use iced::font::{Family, Weight};
    use iced::Font;

    use crate::gui::fonts;

    /// The interface's face at `weight`.
    fn nunito(weight: Weight) -> Font {
        Font {
            family: match fonts::words_font() {
                Some(_) => Family::Name("Nunito"),
                None => Family::SansSerif,
            },
            weight,
            ..Font::DEFAULT
        }
    }

    /// Body text and captions.
    pub fn body() -> Font {
        nunito(Weight::Medium)
    }

    /// Labels and buttons.
    pub fn label() -> Font {
        nunito(Weight::Semibold)
    }

    /// Titles and the display line.
    pub fn title() -> Font {
        nunito(Weight::Bold)
    }

    /// Figures that change (times, rates, counts, shares).
    pub fn numbers() -> Font {
        nunito(Weight::Semibold)
    }
}
