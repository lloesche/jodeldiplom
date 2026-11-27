//! The Transition tab in iced's widgets, in the main screen's side pane
//! ([`Room`]): a panel, on it a
//! block for the playing track and one for the next where there is room
//! (and the one left, sliding up out of view after a hand-over), each a
//! header of facts ([`header`]), the whole track in one row, the bar
//! numbers and the lanes with their labels ([`block`]); the keys in its
//! footer with the button that zooms to fit the lyrics.
//!
//! The layout is in points ([`plan`]), every edge on the window's pixels,
//! so the lanes the window paints ([`lanes`], through
//! [`crate::gui::lanes`]) sit where this view leaves room for them. The
//! view takes the pointer over them: a press on the lanes puts the cursor
//! and a drag selects bars ([`Grab::Lanes`]), the secondary button drags
//! the view ([`Grab::Pan`]), the wheel zooms the block around the pointer
//! (over the whole track it scrolls the view, and a press there seeks).

mod block;
mod header;

pub use block::id;

use iced::widget::{container, opaque, pin, stack, Space};
use iced::{Background, Border, Element, Point, Rectangle, Size, Widget as _};

use crate::core::lanes::{Block, Lane};
use crate::core::transition::viewport;
use crate::core::{Screen, Side, State};
use crate::gui::lanes::{LaneSpot, Part, Rows};
use crate::gui::look::{self, radius, space};
use crate::gui::metrics;
use crate::gui::pixels::Snap;
use crate::gui::window::Message;
use crate::view::transition::{block_lanes, lane_width, plots, LABELS, PAD};

/// A block's header row, in points.
const HEADER: f32 = 28.0;
/// The whole track's row.
const OVERVIEW: f32 = 18.0;
/// The bar numbers' row.
const RULER: f32 = 16.0;
/// Under the header, under the whole track, under the bar numbers.
const UNDER_HEADER: f32 = 2.0;
const UNDER_OVERVIEW: f32 = 4.0;
const UNDER_RULER: f32 = 2.0;
/// Between two lanes, and between two blocks.
const LANE_GAP: f32 = 3.0;
const BLOCK_GAP: f32 = 16.0;
/// The lowest and the highest a lane is.
const LEAST_LANE: f32 = 18.0;
const MOST_LANE: f32 = 64.0;
/// The footer's row.
const FOOTER: f32 = 30.0;
/// The least a lane shows of itself for its label to show beside it.
const LABEL: f32 = 20.0;

/// Where the tab goes: the side pane in the window, in points, and the
/// window's pixels a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// The pane.
    pub pane: Rectangle,
    /// Pixels a point.
    pub scale: f32,
}

impl Room {
    /// The tab's room while it shows on the main screen in a window of
    /// `size`, `scale` pixels a point ([`metrics::main`]).
    pub fn of(state: &State, size: Size, scale: f32) -> Option<Room> {
        if state.screen != Screen::Main || state.side != Side::Transition {
            return None;
        }
        Some(Room {
            pane: metrics::main(size, Side::Transition).pane,
            scale: scale.max(0.1),
        })
    }

    /// The window's pixels its points land on.
    fn px(&self) -> Snap {
        Snap(self.scale)
    }
}

/// A lane as laid out: the part of it in view, how much of it is above
/// that and how high it is (it is cut while its block slides), and its
/// label's place where it shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedLane {
    /// The lane.
    pub lane: Lane,
    /// Its part in view.
    pub rect: Rectangle,
    /// Points of it above the view.
    pub above: f32,
    /// Its whole height.
    pub full: f32,
    /// Its label's place, left of it.
    pub label: Option<Rectangle>,
}

/// A block as laid out: each row where it shows (none where it is out of
/// view), its lanes in view.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// The block.
    pub block: Block,
    /// Its header.
    pub header: Option<Rectangle>,
    /// The whole track.
    pub overview: Option<Rectangle>,
    /// The bar numbers.
    pub ruler: Option<Rectangle>,
    /// Its lanes in view, top first.
    pub lanes: Vec<PlacedLane>,
}

impl Placed {
    /// Where the bar numbers and the lanes take the pointer: from the
    /// first in view to the last.
    pub fn reach(&self) -> Option<Rectangle> {
        let top = self.ruler.or(self.lanes.first().map(|l| l.rect))?;
        let bottom = self.lanes.last().map_or(top, |l| l.rect);
        Some(Rectangle {
            height: bottom.y + bottom.height - top.y,
            ..top
        })
    }

    /// Whether its first lane is cut at the top (the block slides out of
    /// view).
    pub fn cut(&self) -> bool {
        self.lanes.first().is_some_and(|l| l.above > 0.0)
    }
}

/// The tab laid out in its room.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// The panel under it all.
    pub panel: Rectangle,
    /// The blocks that show, top first.
    pub blocks: Vec<Placed>,
    /// The footer's row.
    pub footer: Rectangle,
}

/// The tab for `state` in `room`: the playing track's block, then the next
/// one's where both fit with lanes [`LEAST_LANE`] high or more (as high as
/// both fit, up to [`MOST_LANE`]); while one track gives way to the next,
/// the one left above them, sliding up out of view. All in the window's
/// points, on its pixels.
pub fn plan(state: &State, room: &Room) -> Plan {
    let p = room.pane;
    let panel = room.px().rect(p);
    let left = p.x + PAD;
    let width = (p.width - 2.0 * PAD).max(0.0);
    let lanes_x = left + LABELS;
    let lanes_w = lane_width(p.width);
    let footer = room.px().rect(Rectangle::new(
        Point::new(left, panel.y + panel.height - space::S - FOOTER),
        Size::new(width, FOOTER),
    ));
    let (top, bottom) = (p.y + PAD, footer.y - space::S);
    let fixed = HEADER + UNDER_HEADER + OVERVIEW + UNDER_OVERVIEW + RULER + UNDER_RULER;
    let need = |n: usize, h: f32| fixed + n as f32 * h + n.saturating_sub(1) as f32 * LANE_GAP;
    let counts = [
        block_lanes(state, Block::Now).len(),
        block_lanes(state, Block::Next).len(),
    ];
    // The tallest lanes both blocks fit with, else the playing track's
    // alone; on the pixels.
    let floor = |h: f32| (h * room.scale).floor() / room.scale;
    let room_for = |blocks: &[usize]| {
        let n: usize = blocks.iter().sum();
        let gaps: usize = blocks.iter().map(|k| k.saturating_sub(1)).sum();
        let over = blocks.len() as f32 * fixed
            + blocks.len().saturating_sub(1) as f32 * BLOCK_GAP
            + gaps as f32 * LANE_GAP;
        floor(((bottom - top - over) / n.max(1) as f32).min(MOST_LANE))
    };
    let both = room_for(&counts);
    let mut order: Vec<Block> = if both >= LEAST_LANE {
        vec![Block::Now, Block::Next]
    } else if room_for(&counts[..1]) >= LEAST_LANE {
        vec![Block::Now]
    } else {
        Vec::new()
    };
    let h = if order.len() == 2 {
        both
    } else {
        room_for(&counts[..1])
    };
    let mut y = top;
    // After a hand-over, the track left slides up out of view and the
    // playing one up into its place, the next one in under it.
    if let (Some(done), 2) = (viewport::slide(state), order.len()) {
        let pitch = need(block_lanes(state, Block::Prev).len(), h) + BLOCK_GAP;
        y -= room.px().snap(done as f32 * pitch);
        order.insert(0, Block::Prev);
    }
    let inside = |r: Rectangle| r.y >= top - 0.01 && r.y + r.height <= bottom + 0.01;
    let row = |y: f32, x: f32, w: f32, h: f32| {
        let r = room
            .px()
            .rect(Rectangle::new(Point::new(x, y), Size::new(w, h)));
        inside(r).then_some(r)
    };
    let mut blocks = Vec::new();
    for block in order {
        let first = y;
        let header = row(y, left, width, HEADER);
        y += HEADER + UNDER_HEADER;
        let overview = row(y, lanes_x, lanes_w, OVERVIEW);
        y += OVERVIEW + UNDER_OVERVIEW;
        let ruler = row(y, lanes_x, lanes_w, RULER);
        y += RULER + UNDER_RULER;
        let mut lanes = Vec::new();
        for &lane in block_lanes(state, block) {
            let (from, to) = (y.max(top), (y + h).min(bottom));
            if to > from {
                let rect = room.px().rect(Rectangle::new(
                    Point::new(lanes_x, from),
                    Size::new(lanes_w, to - from),
                ));
                let label = (rect.height >= LABEL).then(|| Rectangle {
                    x: room.px().snap(left),
                    width: LABELS - space::S,
                    ..rect
                });
                lanes.push(PlacedLane {
                    lane,
                    rect,
                    above: room.px().snap(from - y),
                    full: h,
                    label,
                });
            }
            y += h + LANE_GAP;
        }
        y += BLOCK_GAP - LANE_GAP;
        if first >= bottom || y <= top {
            continue;
        }
        blocks.push(Placed {
            block,
            header,
            overview,
            ruler,
            lanes,
        });
    }
    Plan {
        panel,
        blocks,
        footer,
    }
}

/// What the lanes of the tab in `room` draw, `heard` seconds into the
/// playing track, and where, in the window's pixels.
pub fn spots(state: &State, room: &Room, heard: f64) -> Vec<LaneSpot> {
    let plan = plan(state, room);
    let mut out = Vec::new();
    for placed in &plan.blocks {
        let Some(plots) = plots(state, placed.block, heard, !placed.cut()) else {
            continue;
        };
        let rows = |above: f32, full: f32| Rows {
            above: (above * room.scale).round() as usize,
            full: (full * room.scale).round() as usize,
        };
        if let Some(r) = placed.overview {
            out.push(LaneSpot {
                rect: room.px().pixels(r),
                part: Part::Wave(Box::new(plots.overview), rows(0.0, r.height)),
            });
        }
        if let Some(r) = placed.ruler {
            out.push(LaneSpot {
                rect: room.px().pixels(r),
                part: Part::Ruler(Box::new(plots.ruler)),
            });
        }
        for l in &placed.lanes {
            let Some(plot) = plots.lanes.iter().find(|p| p.lane == l.lane) else {
                continue;
            };
            let rect = room.px().pixels(l.rect);
            let rows = rows(l.above, l.full);
            out.push(LaneSpot {
                rect,
                part: Part::Wave(
                    Box::new(plot.clone()),
                    Rows {
                        full: rows.full.max(rect.3 + rows.above),
                        ..rows
                    },
                ),
            });
        }
    }
    out
}

/// The tab for `state` in `room`, filling the pane: its panel, the
/// blocks' headers, labels and the places that take the pointer, and the
/// footer; the lanes go over it ([`spots`]).
pub fn view<'a>(state: &State, room: &Room) -> Element<'a, Message> {
    let plan = plan(state, room);
    let heard = state.heard();
    let origin = room.pane.position();
    let at = |element: Element<'a, Message>, r: Rectangle| -> Element<'a, Message> {
        pin(container(element).width(r.width).height(r.height))
            .x(r.x - origin.x)
            .y(r.y - origin.y)
            .boxed()
    };
    let mut layers: Vec<Element<'a, Message>> = vec![at(ground(), plan.panel)];
    for placed in &plan.blocks {
        block::view(state, placed, heard, &mut |e, r| layers.push(at(e, r)));
    }
    layers.push(at(header::footer(state, plan.footer.width), plan.footer));
    opaque(
        stack(layers)
            .width(room.pane.width)
            .height(room.pane.height),
    )
    .capture_scroll(true)
    .boxed()
}

/// The panel the tab is on.
fn ground<'a>() -> Element<'a, Message> {
    container(Space::new())
        .width(iced::Fill)
        .height(iced::Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::SURFACE)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: radius::PANEL.into(),
            },
            ..container::Style::default()
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// A room of `w` × `h` points at scale `scale`.
    fn room(w: f32, h: f32, scale: f32) -> Room {
        Room {
            pane: Rectangle::new(Point::ORIGIN, Size::new(w, h)),
            scale,
        }
    }

    fn state() -> State {
        let mut s = State::new("/m".into(), Config::default());
        s.side = Side::Transition;
        s.rhythm_recipe.stems = true;
        s
    }

    /// Tall: both blocks, every lane as high as allowed or as both fit,
    /// in order down the pane and inside it; the lanes, the whole track
    /// and the bar numbers in one column, as wide as the fit says.
    #[test]
    fn both_blocks_fit_with_their_lanes_in_one_column() {
        let s = state();
        let r = room(1280.0, 740.0, 1.0);
        let p = plan(&s, &r);
        assert_eq!(
            p.blocks.iter().map(|b| b.block).collect::<Vec<_>>(),
            [Block::Now, Block::Next]
        );
        let mut last = 0.0;
        for b in &p.blocks {
            let ruler = b.ruler.unwrap();
            assert_eq!(b.overview.unwrap().x, ruler.x);
            assert_eq!(b.lanes.len(), 5);
            for l in &b.lanes {
                assert_eq!((l.rect.x, l.rect.width), (ruler.x, ruler.width));
                assert!(l.rect.y >= last, "{l:?}");
                assert!(l.rect.height >= LEAST_LANE && l.rect.height <= MOST_LANE);
                assert!(l.label.is_some());
                last = l.rect.y + l.rect.height;
            }
        }
        assert!(last <= p.footer.y);
        assert_eq!(
            p.blocks[0].lanes[0].rect.width,
            lane_width(1280.0),
            "the fit's width"
        );
    }

    /// Short: the playing track's block alone; too short for it: none.
    /// Every edge on the pixels, at scale 2 too.
    #[test]
    fn short_the_next_block_goes_and_the_edges_stay_on_pixels() {
        let s = state();
        let p = plan(&s, &room(900.0, 300.0, 1.0));
        assert_eq!(p.blocks.len(), 1);
        assert_eq!(p.blocks[0].block, Block::Now);
        assert!(plan(&s, &room(900.0, 120.0, 1.0)).blocks.is_empty());
        let r = room(901.0, 611.0, 2.0);
        let p = plan(&s, &r);
        for l in p.blocks.iter().flat_map(|b| &b.lanes) {
            for v in [l.rect.x, l.rect.y, l.rect.width, l.rect.height] {
                assert_eq!((v * 2.0).fract(), 0.0, "{l:?}");
            }
        }
    }
}
