//! A setting's state at a glance: a switch ([`switch`]: a pill with its
//! knob at the right and lit in the accent while on) and a check
//! ([`check`]: a box, or a disc for one of a few, filled in the accent
//! while on). They take no click themselves: the row they are on does.

use iced::widget::{container, space};
use iced::{Alignment, Background, Border, Color, Element, Length, Widget as _};

use crate::gui::look::{self, radius};
use crate::gui::widgets::{icon, Icon};

/// A switch's width and height, in points.
const SWITCH: (f32, f32) = (36.0, 20.0);
/// Its knob's, across.
const KNOB: f32 = 14.0;
/// A check's, across.
const CHECK: f32 = 18.0;

/// A switch, `on` or off; dimmed where it cannot be turned on
/// (`usable` false).
pub fn switch<'a, Message: 'a>(on: bool, usable: bool) -> Element<'a, Message> {
    let (track, knob) = match (on, usable) {
        (true, _) => (look::ACCENT, look::ACCENT_INK),
        (false, true) => (look::EDGE, look::TEXT_DIM),
        (false, false) => (look::RAISED, look::TEXT_FAINT),
    };
    let knob = container(space())
        .width(KNOB)
        .height(KNOB)
        .style(move |_| round(knob, Color::TRANSPARENT, radius::PILL));
    container(knob)
        .width(SWITCH.0)
        .height(SWITCH.1)
        .padding((SWITCH.1 - KNOB) / 2.0)
        .align_x(if on { Alignment::End } else { Alignment::Start })
        .align_y(Alignment::Center)
        .style(move |_| round(track, Color::TRANSPARENT, radius::PILL))
        .boxed()
}

/// A check, `on` or off: a disc where it is one of a few (`round`), a box
/// otherwise.
pub fn check<'a, Message: 'a>(on: bool, round_: bool) -> Element<'a, Message> {
    let corner = if round_ { radius::PILL } else { 5.0 };
    let mark: Element<'a, Message> = match (on, round_) {
        (false, _) => space().boxed(),
        (true, true) => container(space())
            .width(8.0)
            .height(8.0)
            .style(|_| round(look::ACCENT_INK, Color::TRANSPARENT, radius::PILL))
            .boxed(),
        (true, false) => icon(Icon::Check).size(14.0).color(look::ACCENT_INK).boxed(),
    };
    let (fill, edge) = if on {
        (look::ACCENT, Color::TRANSPARENT)
    } else {
        (Color::TRANSPARENT, look::TEXT_FAINT)
    };
    container(mark)
        .center_x(Length::Fixed(CHECK))
        .center_y(Length::Fixed(CHECK))
        .style(move |_| round(fill, edge, corner))
        .boxed()
}

/// A container's look: `fill`, an `edge` (transparent: none), corners of
/// `corner`.
fn round(fill: Color, edge: Color, corner: f32) -> container::Style {
    container::Style {
        background: Some(Background::Color(fill)),
        border: Border {
            color: edge,
            width: if edge.a > 0.0 { 1.5 } else { 0.0 },
            radius: corner.into(),
        },
        ..container::Style::default()
    }
}
