//! A block of the Transition tab, as laid out ([`Placed`]): its header,
//! its lanes' labels (small tags in their stems' colours: grey where the
//! stem is off, lit where its fade is edited), and the places that take
//! the pointer over the lanes the window paints there: the bar numbers and
//! the lanes (a press puts the cursor, a drag selects, the secondary
//! button drags the view, the wheel zooms around the pointer) and the
//! whole track (a press seeks there or moves the next track's view, the
//! secondary button drags the span shown, the wheel scrolls it). The block
//! sliding away after a hand-over takes no pointer.

use iced::mouse::ScrollDelta;
use iced::widget::container;
use iced::{Alignment, Element, Point, Rectangle, Widget as _};

use super::header::header;
use super::Placed;
use crate::core::lanes::{self, Block};
use crate::core::timeview::PanOf;
use crate::core::transition::viewport;
use crate::core::{Action, State};
use crate::gui::look;
use crate::gui::pointer::{Grab, WheelOn};
use crate::gui::widgets;
use crate::gui::window::Message;
use crate::view::transition::{lane_tags, Header};

/// The share of `bounds`'s width left of `at` (0 to 1).
fn share(bounds: Rectangle, at: Point) -> f64 {
    if bounds.width <= 0.0 {
        return 0.0;
    }
    f64::from(((at.x - bounds.x) / bounds.width).clamp(0.0, 1.0))
}

/// The wheel over `on`: a step up and a step down, each `wheel(steps)`.
fn steps(on: WheelOn, delta: ScrollDelta, wheel: impl Fn(i32) -> Action) -> Message {
    Message::Steps(on, delta, Box::new((wheel(-1), wheel(1))))
}

/// The name the tests find a block's lanes by (`lanes-now`), or its whole
/// track (`overview-next`).
pub fn id(what: &str, block: Block) -> String {
    let block = match block {
        Block::Now => "now",
        Block::Next => "next",
        Block::Prev => "last",
    };
    format!("{what}-{block}")
}

/// Block `placed` of `state`, `heard` seconds into the playing track: each
/// piece handed to `put` with its place in the window.
pub(super) fn view<'a>(
    state: &State,
    placed: &Placed,
    heard: f64,
    put: &mut dyn FnMut(Element<'a, Message>, Rectangle),
) {
    let block = placed.block;
    if let Some(r) = placed.header {
        put(header(&Header::of(state, block, heard), r.width), r);
    }
    let tags = lane_tags(state, block);
    for l in &placed.lanes {
        let (Some(r), Some(tag)) = (l.label, tags.iter().find(|t| t.lane == l.lane)) else {
            continue;
        };
        let color = iced::Color::from_rgb8(tag.color[0], tag.color[1], tag.color[2]);
        let label = tag.lane.label();
        let tag = match (tag.edited, tag.off) {
            (true, _) => widgets::lit_tag(label, color),
            (false, true) => widgets::tag(label, look::TEXT_FAINT),
            (false, false) => widgets::tag(label, color),
        };
        put(container(tag).center_y(r.height).boxed(), r);
    }
    if block == Block::Prev {
        return;
    }
    if let Some(r) = placed.reach() {
        let (start, end) = viewport::window(state, block, heard);
        let lanes = widgets::area()
            .id(id("lanes", block))
            .on_press(move |bounds, at| Message::Grab(Grab::Lanes { block, bounds }, at))
            .on_right_press(move |bounds, at| {
                let pan = Grab::Pan {
                    of: PanOf::Block {
                        block,
                        overview: false,
                    },
                    bounds,
                };
                Message::Grab(pan, at)
            })
            .on_scroll(move |bounds, at, delta| {
                // The time under the pointer stays where it is.
                let time = start + share(bounds, at) * (end - start);
                steps(WheelOn::Lanes(block, false), delta, |steps| {
                    Action::TransitionWheel {
                        block,
                        steps,
                        scroll: false,
                        at: Some(time),
                    }
                })
            });
        put(lanes.boxed(), r);
    }
    if let Some(r) = placed.overview {
        let duration = lanes::rhythmed(state, block).map_or(0.0, |r| r.length());
        let whole = widgets::area()
            .id(id("overview", block))
            .on_press(move |bounds, at| {
                Message::Act(Action::TransitionJump {
                    block,
                    time: share(bounds, at) * duration,
                })
            })
            .on_right_press(move |bounds, at| {
                let pan = Grab::Pan {
                    of: PanOf::Block {
                        block,
                        overview: true,
                    },
                    bounds,
                };
                Message::Grab(pan, at)
            })
            .on_scroll(move |_, _, delta| {
                steps(WheelOn::Lanes(block, true), delta, |steps| {
                    Action::TransitionWheel {
                        block,
                        steps,
                        scroll: true,
                        at: None,
                    }
                })
            });
        put(container(whole).align_y(Alignment::Center).boxed(), r);
    }
}
