//! The bottom bar ([`crate::view::bottom`]) in iced's widgets, along the
//! window's foot ([`crate::gui::metrics`]): the spectrum heard at its top
//! where it has one, then a band of its own: the time heard, the seek bar
//! and the length on a line; under it the transport (play or pause in
//! the accent), the stems' switches, the message in the middle, and the
//! settings and the help at the right.
//!
//! The buttons send the core's actions ([`Message::Act`]); a press on the
//! seek bar grabs the pointer ([`Grab::Seek`]): the window follows it
//! anywhere and seeks where it is let go of. The wheel over the bar
//! scrolls what the screen scrolls ([`Message::Wheel`]).

use iced::widget::{column, container, mouse_area, row, text, Row, Space};
use iced::{Alignment, Background, Element, Fill, Padding, Widget as _};

use crate::core::keymap::Op;
use crate::core::{Action, State};
use crate::gui::look::{self, size, space};
use crate::gui::pointer::Grab;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::spectrum::SpectrumFrame;
use crate::view::bottom::{Bar, Press, Tone};

/// The width the times take beside the seek bar, in points: room for
/// `1:23:45` in tabular figures.
const TIME: f32 = 52.0;

/// The bottom bar for `state`, `width` points wide, its top `spectrum`
/// points the spectrum `heard` (none at 0).
pub fn view<'a>(
    state: &State,
    heard: &SpectrumFrame,
    width: f32,
    spectrum: f32,
) -> Element<'a, Message> {
    let bar = Bar::of(state, width);
    let top: Element<'a, Message> = if spectrum > 0.0 {
        container(widgets::spectrum(heard))
            .height(spectrum)
            .width(Fill)
            .boxed()
    } else {
        Space::new().height(0).boxed()
    };
    // The seek bar's line and the buttons fill the band's three rows;
    // what is left of the window under them is shared above and below.
    let band = container(column![seek(&bar), buttons(&bar)])
        .padding(Padding::from([0.0, space::L]))
        .width(Fill)
        .height(Fill)
        .align_y(Alignment::Center)
        .style(|_| container::Style {
            background: Some(Background::Color(look::SURFACE)),
            ..container::Style::default()
        });
    let edge =
        container(Space::new().width(Fill).height(look::line::EDGE)).style(|_| container::Style {
            background: Some(Background::Color(look::EDGE)),
            ..container::Style::default()
        });
    mouse_area(column![top, edge, band])
        .on_scroll(Message::Wheel)
        .boxed()
}

/// The time heard, the seek bar and the length.
fn seek<'a>(bar: &Bar) -> Element<'a, Message> {
    let seek = &bar.seek;
    let tone = if seek.track {
        look::TEXT_DIM
    } else {
        look::TEXT_FAINT
    };
    let time = |s: &str| {
        text(s.to_owned())
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(tone)
    };
    let mut track = widgets::track(seek.share).held(seek.held).id("seek");
    if seek.track {
        track = track.on_press(|bounds, at| Message::Grab(Grab::Seek(bounds), at));
    }
    row![
        time(&seek.elapsed).width(TIME).align_x(Alignment::End),
        Element::from(track),
        time(&seek.total).width(TIME),
    ]
    .spacing(space::M)
    .align_y(Alignment::Center)
    .boxed()
}

/// The transport, the stems' switches, the message and the ends.
fn buttons<'a>(bar: &Bar) -> Element<'a, Message> {
    let mut left = Row::new().spacing(space::XS).align_y(Alignment::Center);
    for press in &bar.transport {
        left = left.push(transport(press));
    }
    if !bar.stems.is_empty() {
        left = left.push(Space::new().width(space::S).boxed());
    }
    for press in &bar.stems {
        left = left.push(Element::from(
            widgets::chip(press.label.clone())
                .struck(press.off)
                .on_press(act(press.op)),
        ));
    }
    let message = text(bar.message.clone())
        .size(size::LABEL)
        .font(look::font::body())
        .color(match bar.tone {
            Tone::Text => look::TEXT,
            Tone::Dim => look::TEXT_FAINT,
            Tone::Fail => look::FAIL,
        })
        .wrapping(text::Wrapping::None)
        .ellipsis(text::Ellipsis::End)
        .width(Fill)
        .align_x(Alignment::Center);
    let mut right = Row::new().spacing(space::XS).align_y(Alignment::Center);
    for press in &bar.ends {
        let icon = match press.op {
            Op::Settings => Icon::Settings,
            _ => Icon::Help,
        };
        let button = if bar.named_ends {
            widgets::ghost(press.label.clone()).leading(icon)
        } else {
            widgets::button::icon(icon)
        };
        right = right.push(Element::from(
            button
                .id(press.label.to_lowercase())
                .on_press(act(press.op)),
        ));
    }
    row![left, message, right]
        .spacing(space::L)
        .align_y(Alignment::Center)
        .boxed()
}

/// A transport button: play or pause in the accent, the rest quiet.
fn transport<'a>(press: &Press) -> Element<'a, Message> {
    let (icon, round) = match (press.op, press.label.as_str()) {
        (Op::PlayPause, "pause") => (Icon::Pause, true),
        (Op::PlayPause, _) => (Icon::Play, true),
        (Op::Previous, _) => (Icon::Previous, false),
        (Op::Next, _) => (Icon::Next, false),
        _ => (Icon::Stop, false),
    };
    let button = if round {
        widgets::round(icon)
    } else {
        widgets::button::icon(icon)
    };
    button
        .id(press.label.clone())
        .on_press(act(press.op))
        .into()
}

/// The core's `op`.
fn act(op: Op) -> Message {
    Message::Act(Action::Op(op))
}
