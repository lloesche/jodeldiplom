//! What the screens say, as models the window draws in iced's widgets:
//! the list pane with its tabs ([`list`]), the Transition tab
//! ([`transition`]), the Karaoke tab and Karaoke mode ([`karaoke`], the
//! sync's panel [`sync`]), the right column's text under the cover or the
//! summary under the list ([`info`]), the command palette over everything
//! ([`command_palette`]) and the bottom bar ([`bottom`], the spectrum's
//! [`plot`]); and whole screens: Now playing ([`now::NowPlaying`]), the
//! settings ([`settings`]) and the keys ([`help`]), in the player's
//! colours ([`palette`]).
//!
//! What is left of the old cell renderer: [`draw`] fills the old canvas's
//! cells with the background (no screen is drawn into cells any more), on
//! a grid of cells ([`layout`]), with text measured in cells ([`cells`]),
//! the pointer over it in cells ([`mouse`]).

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
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Widget};
use ratatui::Frame;

use crate::core::State;
use crate::view::cells::width;
use crate::view::layout::Grid;

/// Everything a frame shows.
pub struct Scene<'a> {
    /// The UI state.
    pub state: &'a State,
    /// The grid it is laid out on.
    pub grid: Grid,
    /// Seconds into the playing track as heard now ([`crate::app::App::heard`]).
    pub heard: f64,
}

/// Draws the whole screen: the background, as every screen is the
/// window's own widgets now.
pub fn draw(frame: &mut Frame, _scene: &Scene) {
    let [r, g, b] = palette::BG;
    let [tr, tg, tb] = palette::TEXT;
    let style = Style::new()
        .bg(Color::Rgb(r, g, b))
        .fg(Color::Rgb(tr, tg, tb));
    let area = frame.area();
    Block::new().style(style).render(area, frame.buffer_mut());
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
