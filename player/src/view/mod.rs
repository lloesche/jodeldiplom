//! The ratatui widgets that draw the [`State`] into cells, which the window
//! rasterises, in the player's colours ([`palette`]).
//!
//! Pictures (the cover, the lanes' waveforms, the bar numbers, Karaoke
//! mode's picture) go through [`Pictures`]: the window leaves their cells
//! empty and paints pixels there after the frame is drawn.
//!
//! Each part of the screen has its widgets: the box of a tab of its own
//! (Karaoke, the Mixer's place). They are laid out on a grid of cells
//! ([`layout`]), with text measured in cells ([`cells`]), and take the
//! mouse there ([`mouse`]). The window draws the rest from their models,
//! over the cells left to it: the list pane with its tabs ([`list`]), the
//! Transition tab ([`transition`]), the right column's text under the
//! cover or the summary under the list ([`info`]), the command palette
//! over everything ([`command_palette`]) and the bottom bar ([`bottom`],
//! the spectrum's [`plot`]); and whole screens: Now playing
//! ([`now::NowPlaying`]), the settings ([`settings`]) and the keys
//! ([`help`]).

pub mod bottom;
pub mod bubbles;
pub mod cells;
pub mod command_palette;
pub mod help;
pub mod info;
pub mod karaoke;
pub mod lanes;
pub mod layout;
pub mod list;
pub mod mouse;
pub mod now;
pub mod palette;
pub mod plot;
pub mod settings;
pub mod setup;
pub mod sync;
pub mod transition;

use ratatui::buffer::Buffer;
use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Widget};
use ratatui::Frame;

use crate::core::State;
use crate::view::cells::width;
use crate::view::layout::{Grid, Layout};

/// The palette's colours as the cells' colours, for the widgets.
mod colors {
    use ratatui::style::Color;

    use super::palette;

    /// A colour of the palette as a cell's.
    const fn cell([r, g, b]: [u8; 3]) -> Color {
        Color::Rgb(r, g, b)
    }

    pub const BG: Color = cell(palette::BG);
    pub const ACCENT: Color = cell(palette::ACCENT);
    pub const DIM: Color = cell(palette::DIM);
    pub const TEXT: Color = cell(palette::TEXT);
    pub const BORDER: Color = cell(palette::BORDER);
    pub const BUTTON: Color = cell(palette::BUTTON);
    pub const WARN: Color = cell(palette::WARN);
    pub const FAIL: Color = cell(palette::FAIL);
    pub const TRACK: Color = cell(palette::TRACK);
}

use colors::*;

/// Draws the pictures: the window paints them in pixels where the widgets
/// left their cells empty; [`CellPictures`] leaves them empty.
pub trait Pictures {
    /// The cover (or a placeholder) into `area`. With `below_line` the
    /// area's first row has a border line through its middle (the list
    /// pane's top): the cover starts under that line, half a cell down,
    /// at the left; otherwise it is centred.
    fn cover(&mut self, frame: &mut Frame, area: Rect, below_line: bool);
    /// A lane's waveform ([`lanes::WavePlot`]): the Karaoke tab's voice
    /// lane.
    fn wave(&mut self, frame: &mut Frame, plot: &lanes::WavePlot);
    /// Karaoke mode's picture ([`karaoke::KaraokePicture`]) over the
    /// cells of `area`.
    fn karaoke(&mut self, frame: &mut Frame, area: Rect, picture: &karaoke::KaraokePicture);
}

/// Everything a frame shows.
pub struct Scene<'a> {
    /// The UI state.
    pub state: &'a State,
    /// The grid it is laid out on.
    pub grid: Grid,
    /// Seconds into the playing track as heard now ([`crate::app::App::heard`]),
    /// for what follows it (the Transition tab, Karaoke mode's picture).
    pub heard: f64,
}

/// Draws the whole screen.
pub fn draw(frame: &mut Frame, scene: &Scene, pictures: &mut dyn Pictures) {
    let area = frame.area();
    let bg = Style::new().bg(BG).fg(TEXT);
    Block::new().style(bg).render(area, frame.buffer_mut());
    // The layout is on the scene's grid; a frame of another size (the
    // window resized since the grid was fitted) stays blank until the grid
    // catches up, as nothing laid out for it would fit.
    if (area.width, area.height) != (scene.grid.cols, scene.grid.rows) {
        return;
    }
    let layout = layout::compute(scene.state, scene.grid);
    match &layout {
        Layout::TooSmall(area) => {
            let text = format!(
                "Make the window larger ({} × {} at least)",
                layout::MIN_SIZE.0,
                layout::MIN_SIZE.1
            );
            let y = area.y + area.height / 2;
            frame.render_widget(
                Paragraph::new(text)
                    .style(Style::new().fg(DIM))
                    .alignment(HorizontalAlignment::Center),
                Rect::new(area.x, y, area.width, 1),
            );
        }
        Layout::Main(m) => {
            draw_tab(frame, scene, m, pictures);
            // The right column's cover; the window draws the text under it.
            if let Some(cover) = m.cover {
                pictures.cover(frame, cover, true);
            }
        }
        // Drawn by the window in iced's widgets.
        Layout::Native => {}
        Layout::Karaoke(k) => karaoke::draw_mode(frame, scene, k, pictures),
    }
}

/// The Mixer tab's bottom border: its keys.
const DESK_FOOTER: &str = "← → strip · ↑ ↓ fader · home 0 dB · m mute · s solo · e echo";

/// A tab of its own (Karaoke, the Mixer) in its box (the window draws the
/// tabs over its top border, and the Mixer's desk inside it), its keys in
/// its bottom border; nothing for the browser, the queue and the
/// Transition tab, which the window draws whole.
fn draw_tab(frame: &mut Frame, scene: &Scene, m: &layout::MainLayout, pictures: &mut dyn Pictures) {
    use crate::core::Side;
    let state = scene.state;
    let footer: String = match state.side {
        Side::Karaoke if state.karaoke.sync.is_some() => sync::footer(state).into(),
        Side::Karaoke => "enter Karaoke mode · ↑ ↓ row · ← → its value · s sync".into(),
        Side::Mixer => DESK_FOOTER.into(),
        // The window draws the browser, the queue and the Transition tab
        // whole.
        Side::Browser | Side::Queue | Side::Transition => return,
    };
    let block = bordered(Line::default()).title_bottom(Line::from(Span::styled(
        format!(" {footer} "),
        Style::new().fg(DIM),
    )));
    frame.render_widget(block, m.list);
    match state.side {
        Side::Karaoke => {
            if let Some(layout) = &m.karaoke {
                karaoke::draw_tab(frame, scene, layout, pictures);
            }
        }
        // The window draws the desk in iced's widgets over its rows
        // ([`crate::gui::mixer`]).
        Side::Mixer | Side::Browser | Side::Queue | Side::Transition => {}
    }
}

fn bordered(title: Line<'static>) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .title(title)
}

/// Shortens `text` to `max` cells, keeping its start (a space before the
/// `…` stays).
fn keep_start(text: &str, max: usize) -> String {
    if width(text) <= max {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 1;
    for c in text.chars() {
        let w = width(c.encode_utf8(&mut [0; 4]));
        if used + w > max {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push('…');
    out
}

/// A level as `n` blocks, `share` of them lit (▮) in `on`, the rest
/// unlit (▯): the microphone's meter, a ring's fill, the device's volume.
pub(crate) fn blocks(share: f64, n: usize, on: ratatui::style::Color) -> [Span<'static>; 2] {
    let lit = ((share.clamp(0.0, 1.0) * n as f64).round() as usize).min(n);
    [
        Span::styled("▮".repeat(lit), Style::new().fg(on)),
        Span::styled("▯".repeat(n - lit), Style::new().fg(TRACK)),
    ]
}

/// The pictures' cells as the window leaves them, without painting them:
/// for the tests and the text dumps of the widgets. The cover is the
/// placeholder (as the window shows it without art); the other pictures'
/// cells are blank.
pub struct CellPictures;

/// `area` blank, in the background colour, as the window leaves it under a
/// picture it paints.
pub fn blank(frame: &mut Frame, area: Rect) {
    Block::new()
        .style(Style::new().bg(BG))
        .render(area, frame.buffer_mut());
}

impl Pictures for CellPictures {
    fn cover(&mut self, frame: &mut Frame, area: Rect, _below_line: bool) {
        placeholder(frame, area);
    }

    fn wave(&mut self, frame: &mut Frame, plot: &lanes::WavePlot) {
        blank(frame, plot.area);
    }

    fn karaoke(&mut self, frame: &mut Frame, area: Rect, _picture: &karaoke::KaraokePicture) {
        blank(frame, area);
    }
}

/// The box drawn where there is no cover.
pub fn placeholder(frame: &mut Frame, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height > 0 {
        frame.render_widget(
            Paragraph::new(Span::styled("♪", Style::new().fg(BORDER)))
                .alignment(HorizontalAlignment::Center),
            Rect::new(inner.x, inner.y + inner.height / 2, inner.width, 1),
        );
    }
}

/// The text of a buffer, one line per row (for text dumps and tests).
pub fn buffer_text(buffer: &Buffer) -> String {
    let mut out = String::new();
    for y in 0..buffer.area.height {
        let mut row = String::new();
        let mut skip = 0;
        for x in 0..buffer.area.width {
            if skip > 0 {
                skip -= 1;
                continue;
            }
            let symbol = buffer[(x, y)].symbol();
            row.push_str(symbol);
            skip = width(symbol).saturating_sub(1);
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}
