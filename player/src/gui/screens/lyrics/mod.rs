//! The lyrics editor ([`crate::view::lyrics::Editor`]) in iced's widgets,
//! the whole window over the bottom bar: its header (the song, whether its
//! lyrics are edited by hand, undo, redo, Revert and Done, [`panel`]), the
//! time across the view, the voice's waveform as a lane the window paints
//! ([`spots`], through [`crate::gui::lanes`]) with the words over it as
//! boxes ([`words`]: a press holds one, its middle or an edge), the
//! voice's level over each word under it, the word selected with its line
//! or the text being written ([`panel`]), and the keys at the foot.
//!
//! The layout is in points ([`plan`]), every edge on the window's pixels,
//! so the lane sits where the view leaves room for it. Over the lane the
//! view takes the pointer: a press on a word holds it
//! ([`Grab::LyricWord`]), one off the words puts the playhead there, the
//! secondary button drags the view ([`Grab::LyricView`]), the wheel zooms
//! it round the time under the pointer.

mod panel;
mod words;

pub use panel::FIELD;
pub use words::{part_id, LANE};

use iced::widget::{container, pin, stack, Space, Stack};
use iced::{Background, Border, Element, Fill, Point, Rectangle, Size, Widget as _};

use super::{page, Parts};
use crate::core::lanes::WordLines;
use crate::core::{Screen, State};
use crate::gui::lanes::{LaneSpot, Part, Rows};
use crate::gui::look::{self, radius, space};
use crate::gui::metrics;
use crate::gui::pixels::PixelRect;
use crate::gui::window::Message;
use crate::view::lanes::{lane_color, WavePlot};
use crate::view::lyrics::{wave, Editor};

/// The header's row, in points.
const HEADER: f32 = 44.0;
/// The time's row, over the lane.
const RULER: f32 = 22.0;
/// The lane's lowest and highest.
const LEAST_LANE: f32 = 64.0;
const MOST_LANE: f32 = 560.0;
/// The voice's levels' row, under the lane.
const LEVELS: f32 = 30.0;
/// The panel of the word selected.
const PANEL: f32 = 140.0;
/// The keys' row at the foot.
const FOOTER: f32 = 28.0;
/// Between the lane and the rows over and under it.
const NEAR: f32 = 2.0;

/// Where the screen goes: the window over the bottom bar, in points, and
/// the window's pixels a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// The page.
    pub page: Rectangle,
    /// Pixels a point.
    pub scale: f32,
}

impl Room {
    /// The screen's room while it shows, in a window of `size` (points),
    /// `scale` pixels a point: the window over the bottom bar.
    pub fn of(state: &State, size: Size, scale: f32) -> Option<Room> {
        if state.screen != Screen::Lyrics || state.karaoke.editor.is_none() {
            return None;
        }
        let bar = metrics::EDGE + metrics::BAND;
        Some(Room {
            page: Rectangle::new(
                Point::ORIGIN,
                Size::new(size.width, (size.height - bar).max(0.0)),
            ),
            scale: scale.max(0.1),
        })
    }

    /// `v` points on the window's pixels.
    fn snap(&self, v: f32) -> f32 {
        (v * self.scale).round() / self.scale
    }

    /// `x`, `y`, `w`, `h` with its edges on the window's pixels.
    fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> Rectangle {
        let (x0, y0) = (self.snap(x), self.snap(y));
        Rectangle::new(
            Point::new(x0, y0),
            Size::new(
                (self.snap(x + w) - x0).max(0.0),
                (self.snap(y + h) - y0).max(0.0),
            ),
        )
    }

    /// `rect` in the window's pixels.
    fn pixels(&self, rect: Rectangle) -> PixelRect {
        let px = |v: f32| (v * self.scale).round().max(0.0) as usize;
        (
            px(rect.x),
            px(rect.y),
            px(rect.x + rect.width) - px(rect.x),
            px(rect.y + rect.height) - px(rect.y),
        )
    }
}

/// The screen laid out in its room, in the window's points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// The header.
    pub header: Rectangle,
    /// The time across the view.
    pub ruler: Rectangle,
    /// The lane: the voice's waveform, the words over it.
    pub lane: Rectangle,
    /// The voice's level over each word.
    pub levels: Rectangle,
    /// The word selected, its line, the text being written.
    pub panel: Rectangle,
    /// The keys.
    pub footer: Rectangle,
}

/// The width of the screen's rows (the lane's among them) in a window
/// `width` points wide.
fn across(width: f32) -> f32 {
    (width - 2.0 * space::XL).max(0.0)
}

/// The width of the lane in a window `width` points wide, in letters
/// `letter` points wide: what `z` fits its words to
/// ([`crate::core::Fit::lyric_letters`]).
pub fn lane_letters(width: f32, letter: f32) -> u16 {
    if letter <= 0.0 {
        return 0;
    }
    (across(width) / letter)
        .floor()
        .clamp(0.0, f32::from(u16::MAX)) as u16
}

/// The screen in `room`: the header at the top, the keys at the foot, the
/// lane as high as there is room for (up to [`MOST_LANE`]) between the
/// time over it and the levels and the panel under it.
pub fn plan(room: &Room) -> Plan {
    let p = room.page;
    let (x, w) = (p.x + space::XL, across(p.width));
    let top = p.y + space::L;
    let header = room.rect(x, top, w, HEADER);
    let ruler_y = top + HEADER + space::M;
    let ruler = room.rect(x, ruler_y, w, RULER);
    let lane_y = ruler_y + RULER + NEAR;
    let footer_y = p.y + p.height - space::M - FOOTER;
    let under = NEAR + LEVELS + space::L + PANEL + space::M;
    let lane_h = (footer_y - under - lane_y).clamp(LEAST_LANE, MOST_LANE);
    // On the pixels, so the waveform's rows are whole.
    let lane_h = (lane_h * room.scale).floor() / room.scale;
    let lane = room.rect(x, lane_y, w, lane_h);
    let levels_y = lane_y + lane_h + NEAR;
    let levels = room.rect(x, levels_y, w, LEVELS);
    let panel = room.rect(x, levels_y + LEVELS + space::L, w, PANEL);
    let footer = room.rect(x, footer_y, w, FOOTER);
    Plan {
        header,
        ruler,
        lane,
        levels,
        panel,
        footer,
    }
}

/// What the lane of the screen in `room` draws, and where, in the window's
/// pixels: the voice's waveform over the seconds in view, the bars' lines,
/// the playhead. None before the song's waveform is in.
pub fn spots(state: &State, room: &Room) -> Vec<LaneSpot> {
    let (Some(wave), Some((start, end))) =
        (wave(state), crate::core::lyric_edit::screen::window(state))
    else {
        return Vec::new();
    };
    let lane = plan(room).lane;
    let rect = room.pixels(lane);
    let plot = WavePlot {
        waveform: wave.waveform,
        lane: wave.lane,
        start,
        end,
        color: lane_color(wave.lane),
        downbeats: wave.downbeats,
        unclear: Vec::new(),
        selection: None,
        shown: None,
        playhead: crate::core::lyric_edit::screen::playhead(state),
        cursor: None,
        cue: None,
        transition: None,
        gain: Vec::new(),
        edit: None,
        lyrics: None,
        text_scale: 1,
        lines: WordLines::ONE,
        follow: None,
        deep: false,
        key: None,
    };
    vec![LaneSpot {
        rect,
        part: Part::Wave(
            Box::new(plot),
            Rows {
                above: 0,
                full: rect.3,
            },
        ),
    }]
}

/// The screen for `state` over the bottom bar, the window's lanes
/// (`parts`) drawn over its page and the words over them.
pub(super) fn view<'a>(state: &'a State, parts: &Parts<'a>) -> Element<'a, Message> {
    let (Some(room), Some(editor)) = (Room::of(state, parts.size, parts.scale), Editor::of(state))
    else {
        return page(state, Space::new().boxed(), parts);
    };
    let plan = plan(&room);
    let at = |element: Element<'a, Message>, r: Rectangle| -> Element<'a, Message> {
        pin(container(element).width(r.width).height(r.height))
            .x(r.x)
            .y(r.y)
            .boxed()
    };
    let layers: Vec<Element<'a, Message>> = vec![
        ground(),
        at(panel::header(&editor), plan.header),
        at(words::ruler(&editor, plan.ruler), plan.ruler),
        at(lane_card(), plan.lane),
        at(words::levels(&editor, plan.levels), plan.levels),
        at(panel::panel(&editor), plan.panel),
        at(panel::footer(&editor), plan.footer),
    ];
    let screen = Stack::with_children(layers)
        .width(room.page.width)
        .height(room.page.height);
    stack![
        page(state, screen.boxed(), parts),
        parts.lanes.view(Point::ORIGIN),
        words::overlay(&editor, plan.lane),
    ]
    .boxed()
}
/// The screen's ground.
fn ground<'a>() -> Element<'a, Message> {
    container(Space::new())
        .width(Fill)
        .height(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::BG)),
            ..container::Style::default()
        })
        .boxed()
}

/// Where the lane goes: a card, as the lane's own (the lane drawn over it;
/// alone while the song has no waveform yet).
fn lane_card<'a>() -> Element<'a, Message> {
    container(Space::new())
        .width(Fill)
        .height(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::RAISED)),
            border: Border {
                radius: radius::PANEL.into(),
                ..Border::default()
            },
            ..container::Style::default()
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A room of `w` × `h` points at scale `scale`.
    fn room(w: f32, h: f32, scale: f32) -> Room {
        Room {
            page: Rectangle::new(Point::ORIGIN, Size::new(w, h)),
            scale,
        }
    }

    /// The rows go down the page in order, inside it, the lane as high as
    /// allowed where there is room; every edge on the pixels at scale 2.
    #[test]
    fn the_rows_go_down_the_page_and_stay_on_its_pixels() {
        for (w, h, scale) in [
            (1280.0, 740.0, 1.0),
            (1512.0, 920.0, 2.0),
            (900.0, 500.0, 1.0),
        ] {
            let p = plan(&room(w, h, scale));
            let rows = [p.header, p.ruler, p.lane, p.levels, p.panel, p.footer];
            for pair in rows.windows(2) {
                assert!(pair[0].y + pair[0].height <= pair[1].y + 0.01, "{p:?}");
            }
            assert!(p.footer.y + p.footer.height <= h);
            assert!(p.lane.height >= LEAST_LANE && p.lane.height <= MOST_LANE);
            assert_eq!((p.lane.x, p.lane.width), (p.ruler.x, p.ruler.width));
            for r in rows {
                for v in [r.x, r.y, r.width, r.height] {
                    assert_eq!((v * scale).fract(), 0.0, "{r:?}");
                }
            }
        }
        assert_eq!(plan(&room(1280.0, 1400.0, 1.0)).lane.height, MOST_LANE);
    }
}
