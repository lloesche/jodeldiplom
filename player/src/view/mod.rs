//! The ratatui widgets that draw the [`State`] into cells, which the window
//! rasterises, in the player's colours ([`palette`]).
//!
//! Pictures (the cover, the spectrum, the lanes' waveforms, the bar
//! numbers, Karaoke mode's picture) go through [`Pictures`]: the window
//! leaves their cells empty and paints pixels there after the frame is
//! drawn. The spectrum's axis labels are cells ([`plot`]).
//!
//! Each part of the screen has its widgets: the list pane ([`list`]), the
//! command palette over it ([`command_palette`]), what is playing ([`now`]:
//! the right column and Now playing), the settings ([`settings`]), the
//! keys ([`help`]) and the bottom bar ([`bottom`]). They are laid out on a
//! grid of cells ([`layout`]), with text measured in cells ([`cells`]),
//! and take the mouse there ([`mouse`]).

mod bottom;
pub mod bubbles;
pub mod cells;
mod command_palette;
mod help;
pub mod karaoke;
pub mod lanes;
pub mod layout;
mod list;
pub mod mouse;
mod now;
pub mod palette;
pub mod plot;
mod settings;
pub mod setup;
pub mod sync;
pub mod transition;

pub use plot::Plot;

use ratatui::buffer::Buffer;
use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Widget};
use ratatui::Frame;

use crate::core::State;
use crate::spectrum::SpectrumFrame;
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
    pub const SELECTED: Color = cell(palette::SELECTED);
    pub const BUTTON: Color = cell(palette::BUTTON);
    pub const GREEN: Color = cell(palette::GREEN);
    pub const WARN: Color = cell(palette::WARN);
    pub const FAIL: Color = cell(palette::FAIL);
    pub const SOURCE: Color = cell(palette::SOURCE);
    pub const FOLDER: Color = cell(palette::FOLDER);
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
    /// The spectrum's curve and grid into `plot.area` (its labels are
    /// drawn as cells by the caller).
    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot);
    /// Whether text can go over the cover just drawn (over a cover, not
    /// over the placeholder).
    fn text_over_cover(&self) -> bool {
        false
    }
    /// The cells of `caption` are text over the cover in `cover`: the
    /// window darkens the cover behind them so they read on any cover.
    fn caption(&mut self, _cover: Rect, _caption: Rect) {}
    /// A lane's waveform ([`lanes::WavePlot`]), on the Transition tab or the
    /// Karaoke tab's voice lane.
    fn wave(&mut self, frame: &mut Frame, plot: &lanes::WavePlot);
    /// A Transition block's bar numbers ([`transition::RulerPlot`]), where
    /// the bars are, to the pixel.
    fn ruler(&mut self, frame: &mut Frame, plot: &transition::RulerPlot);
    /// Karaoke mode's picture ([`karaoke::KaraokePicture`]).
    fn karaoke(&mut self, frame: &mut Frame, picture: &karaoke::KaraokePicture);
}

/// Everything a frame shows.
pub struct Scene<'a> {
    /// The UI state.
    pub state: &'a State,
    /// The grid it is laid out on.
    pub grid: Grid,
    /// The spectrum heard now.
    pub spectrum: &'a SpectrumFrame,
    /// The pointer's cell, if it is in the window: the title, artist and
    /// album links are underlined while it is over them.
    pub pointer: Option<(u16, u16)>,
    /// Seconds into the playing track as heard now ([`crate::app::App::heard`]),
    /// for the beat pulse.
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
            list::draw_pane(frame, scene, m, pictures);
            now::draw_column(frame, scene, m, pictures);
        }
        Layout::NowPlaying(n) => now::draw_now(frame, scene, n, pictures),
        Layout::Settings(s) => settings::draw(frame, scene.state, s),
        Layout::Help(h) => help::draw(frame, scene.state, h),
        Layout::Karaoke(k) => karaoke::draw_mode(frame, scene, k, pictures),
    }
    if let Some(bottom) = layout.bottom() {
        bottom::draw(frame, scene, bottom, pictures);
    }
}

fn bordered(title: Line<'static>) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(BORDER))
        .title(title)
}

/// Shortens `text` to `max` cells, keeping its start (unlike
/// [`cells::shorten`], a space before the `…` stays).
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

/// `left` shortened to fit `w` cells with `right` right-aligned after it.
fn row_line(left: Vec<Span<'static>>, right: Vec<Span<'static>>, w: usize) -> Line<'static> {
    let right_w: usize = right.iter().map(|s| width(&s.content)).sum();
    let room = w.saturating_sub(right_w + 1);
    let mut used = 0;
    let mut spans = Vec::new();
    for span in left {
        let sw = width(&span.content);
        if used + sw > room {
            let cut = keep_start(&span.content, room.saturating_sub(used));
            used += width(&cut);
            spans.push(Span::styled(cut, span.style));
            break;
        }
        used += sw;
        spans.push(span);
    }
    if right_w > 0 && w > right_w {
        spans.push(Span::raw(" ".repeat(w - right_w - used)));
        spans.extend(right);
    }
    Line::from(spans)
}

/// A search line in `rect`: `/`, what is typed (with the caret while it
/// is edited), and how many it leaves `shown` at the right where there is
/// room (the browser's filter, the help's search).
fn search_line(frame: &mut Frame, rect: Rect, typed: &str, editing: bool, shown: usize) {
    let mut spans = vec![
        Span::styled("/ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(typed.to_owned(), Style::new().fg(TEXT)),
    ];
    if editing {
        spans.push(Span::styled("▏", Style::new().fg(ACCENT)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), rect);
    let count = format!("{shown} shown");
    let w = width(&count) as u16;
    if rect.width > w + 10 {
        frame.render_widget(
            Paragraph::new(Span::styled(count, Style::new().fg(DIM))),
            Rect::new(rect.right() - w, rect.y, w, 1),
        );
    }
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

    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot) {
        blank(frame, plot.area);
    }

    fn wave(&mut self, frame: &mut Frame, plot: &lanes::WavePlot) {
        blank(frame, plot.area);
    }

    fn ruler(&mut self, frame: &mut Frame, plot: &transition::RulerPlot) {
        blank(frame, plot.area);
    }

    fn karaoke(&mut self, frame: &mut Frame, picture: &karaoke::KaraokePicture) {
        blank(frame, picture.area);
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
