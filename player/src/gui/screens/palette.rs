//! The command palette ([`crate::view::command_palette`]) in iced's
//! widgets, over everything while it is open: the window dimmed under it
//! (the shade fading in) and a card near the top in the middle sliding
//! in, its shadow growing (over [`look::motion::PANEL`]): what is typed,
//! the commands it matches with
//! their keys (the one selected marked, as a list's row), and the keys
//! that work it.
//!
//! A click on a command runs it ([`Action::PaletteRun`]); a press beside
//! the card closes the palette, as `esc` does. Nothing under it takes the
//! pointer meanwhile.

use iced::animation::Easing;
use iced::widget::{
    column, container, mouse_area, opaque, responsive, row, stack, text, Column, Row, Space,
};
use iced::{
    Alignment, Animation, Background, Border, Color, Element, Fill, Padding, Shadow, Size, Vector,
    Widget as _,
};

use crate::core::keymap::Op;
use crate::core::{Action, State};
use crate::gui::look::{self, height, radius, size, space};
use crate::gui::widgets::{self, list_row, Icon};
use crate::gui::window::Message;
use crate::view::command_palette::Palette;

/// The card's widest, in points.
const WIDTH: f32 = 600.0;
/// The most commands in view.
const ROWS: usize = 8;
/// The line where the command is typed.
const INPUT: f32 = 52.0;
/// The line of keys at the card's foot.
const KEYS: f32 = 36.0;
/// How far the card slides in from, in points.
const SLIDE: f32 = 10.0;

/// The palette over the window, where it is open.
pub fn view<'a>(state: &'a State) -> Option<Element<'a, Message>> {
    state.palette.as_ref()?;
    Some(responsive(move |window| over(state, window)).boxed())
}

/// The palette in a window of `window`'s size: the commands that fit.
fn over<'a>(state: &State, window: Size) -> Element<'a, Message> {
    let top = (window.height * 0.12).clamp(space::L, 96.0);
    let room = window.height - top - INPUT - KEYS - 2.0 * space::XL;
    let rows = ((room / height::ROW).floor().max(1.0) as usize).min(ROWS);
    let Some(palette) = Palette::of(state, rows) else {
        return Space::new().boxed();
    };
    let width = WIDTH.min(window.width - 2.0 * space::XL).max(0.0);
    iced::widget::transition(
        1.0,
        || {
            Animation::new(0.0)
                .duration(look::motion::PANEL)
                .easing(Easing::EaseOut)
        },
        move |shown: &Animation<f32>, at| scene(&palette, width, top, shown.interpolate(at)),
    )
    .boxed()
}

/// The dimmed window and the card on it, `shown` of the way in (0 to 1).
fn scene<'a>(palette: &Palette, width: f32, top: f32, shown: f32) -> Element<'a, Message> {
    let dim = mouse_area(
        container(Space::new().width(Fill).height(Fill))
            .width(Fill)
            .height(Fill)
            .style(move |_| container::Style {
                background: Some(Background::Color(look::alpha(Color::BLACK, 0.5 * shown))),
                ..container::Style::default()
            }),
    )
    .on_press(Message::Act(Action::Op(Op::Back)));
    let card = container(opaque(card(palette, width, shown)))
        .padding(Padding {
            top: top - SLIDE * (1.0 - shown),
            ..Padding::ZERO
        })
        .center_x(Fill);
    stack![opaque(dim), card].boxed()
}

/// The card: what is typed, the commands, the keys; its shadow `shown`
/// of the way in.
fn card<'a>(palette: &Palette, width: f32, shown: f32) -> Element<'a, Message> {
    let caret = container(Space::new().width(2).height(20)).style(move |_| container::Style {
        background: Some(Background::Color(look::ACCENT)),
        ..container::Style::default()
    });
    let typed: Element<'a, Message> = if palette.typed.is_empty() {
        row![
            caret,
            text("Type a command")
                .size(size::BODY + 2.0)
                .font(look::font::body())
                .color(look::TEXT_FAINT)
        ]
        .spacing(space::XS)
        .align_y(Alignment::Center)
        .boxed()
    } else {
        row![
            text(palette.typed.clone())
                .size(size::BODY + 2.0)
                .font(look::font::label())
                .color(look::TEXT)
                .wrapping(text::Wrapping::None),
            caret
        ]
        .spacing(1)
        .align_y(Alignment::Center)
        .boxed()
    };
    let input = container(
        row![
            widgets::icon(Icon::Command).size(18.0).color(look::ACCENT),
            typed
        ]
        .spacing(space::M)
        .align_y(Alignment::Center),
    )
    .padding(Padding::from([0.0, space::L]))
    .center_y(INPUT)
    .width(Fill);
    let mut list =
        Column::<Element<'a, Message>>::new().padding(Padding::from([space::XS + 2.0, space::S]));
    for m in &palette.matches {
        let mut keys = Row::<Element<'a, Message>>::new()
            .spacing(space::XS)
            .align_y(Alignment::Center);
        for key in &m.keys {
            keys = keys.push(widgets::key(key.clone()));
        }
        let content = row![
            text(m.name)
                .size(size::BODY)
                .font(if m.selected {
                    look::font::label()
                } else {
                    look::font::body()
                })
                .color(look::TEXT)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End)
                .width(Fill),
            keys
        ]
        .spacing(space::M)
        .align_y(Alignment::Center);
        list = list.push(Element::from(
            list_row(content.boxed())
                .ground(look::RAISED)
                .selected(m.selected)
                .on_press(Message::Act(Action::PaletteRun(m.index))),
        ));
    }
    if palette.total == 0 {
        list = list.push(
            container(
                text("No command matches")
                    .size(size::BODY)
                    .font(look::font::body())
                    .color(look::TEXT_FAINT),
            )
            .padding(Padding::from([space::M, space::M]))
            .boxed(),
        );
    }
    let hint = |s: String| {
        text(s)
            .size(size::CAPTION)
            .font(look::font::body())
            .color(look::TEXT_FAINT)
    };
    let keys = container(
        row![
            widgets::key("↑ ↓"),
            hint("move".into()),
            Space::new().width(space::S),
            widgets::key("enter"),
            hint("run".into()),
            Space::new().width(space::S),
            widgets::key("esc"),
            hint("close".into()),
            Space::new().width(Fill),
            hint(match palette.total {
                1 => "1 command".into(),
                n => format!("{n} commands"),
            }),
        ]
        .spacing(space::S)
        .align_y(Alignment::Center),
    )
    .padding(Padding::from([0.0, space::L]))
    .center_y(KEYS)
    .width(Fill);
    let rule = || {
        container(Space::new().width(Fill).height(look::line::EDGE)).style(move |_| {
            container::Style {
                background: Some(Background::Color(look::EDGE)),
                ..container::Style::default()
            }
        })
    };
    container(column![input, rule(), list, rule(), keys])
        .width(width)
        .style(move |_| container::Style {
            background: Some(Background::Color(look::RAISED)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: radius::PANEL.into(),
            },
            shadow: Shadow {
                color: look::alpha(Color::BLACK, 0.55 * shown),
                offset: Vector::new(0.0, 16.0),
                blur_radius: 48.0,
            },
            ..container::Style::default()
        })
        .boxed()
}
