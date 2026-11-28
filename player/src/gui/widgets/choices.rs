//! A value that opens a list, and the list under it
//! ([`crate::view::choices`]): the value on a raised chip with `▾`
//! ([`opener`]), the entries as a card's rows, each its mark, name and
//! detail, taking a click ([`list`]), set in under the value or floating
//! over what is under it ([`floating`]). The settings' device and data
//! folder, the microphone on the Karaoke tab and the Mixer tab.

use iced::widget::{container, row, text, Column, Container, Space};
use iced::{Alignment, Background, Border, Color, Element, Padding, Shadow, Vector, Widget as _};

use super::{badge, icon, item, Icon};
use crate::gui::look::{self, radius, size, space};
use crate::view::choices::{Entry, List, Mark};

/// `value` on a raised chip with `▾`, shortened at `ellipsis` where it
/// does not fit (a path at its start, a name at its end).
pub fn opener<'a, Message: 'a>(
    value: impl Into<String>,
    ellipsis: text::Ellipsis,
) -> Element<'a, Message> {
    container(
        row![
            text(value.into())
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT)
                .wrapping(text::Wrapping::None)
                .ellipsis(ellipsis),
            text("▾").size(size::CAPTION).color(look::TEXT_DIM),
        ]
        .spacing(space::S)
        .align_y(Alignment::Center),
    )
    .center_y(look::height::CHIP + 4.0)
    .padding(Padding::from([0.0, space::M]))
    .style(|_| container::Style {
        background: Some(Background::Color(look::RAISED)),
        border: Border {
            color: look::EDGE,
            width: look::line::EDGE,
            radius: radius::CONTROL.into(),
        },
        ..container::Style::default()
    })
    .boxed()
}

/// `content` (a list) floating over what is under it, as the command
/// palette does: raised, edged, its shadow under it.
pub fn floating<'a, Message: 'a>(
    content: Element<'a, Message>,
) -> Container<'a, Element<'a, Message>> {
    container(content)
        .padding(space::S)
        .style(|_| container::Style {
            background: Some(Background::Color(look::RAISED)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: radius::PANEL.into(),
            },
            shadow: Shadow {
                color: look::alpha(Color::BLACK, 0.55),
                offset: Vector::new(0.0, 8.0),
                blur_radius: 24.0,
            },
            ..container::Style::default()
        })
}

/// `list`'s entries set in `indent` points (and a little more), each
/// sending what `on` makes of it when clicked, its widget's id `id` and
/// its index; what shows while it is listed or empty.
pub fn list<'a, Message: Clone + 'a>(
    list: &List,
    indent: f32,
    id: &str,
    on: impl Fn(&Entry) -> Message,
) -> Element<'a, Message> {
    let mut all = Column::new().spacing(2.0).padding(Padding {
        left: indent + space::L,
        top: 2.0,
        ..Padding::ZERO
    });
    for e in &list.entries {
        all = all.push(entry(e, format!("{id}-{}", e.index), on(e)));
    }
    if let Some(note) = &list.note {
        all = all.push(
            container(
                text(note.clone())
                    .size(size::LABEL)
                    .font(look::font::body())
                    .color(look::TEXT_DIM),
            )
            .padding(Padding::from([space::S, space::M]))
            .boxed(),
        );
    }
    all.boxed()
}

/// An entry of a list: its mark, name and detail, sending `press`.
fn entry<'a, Message: Clone + 'a>(e: &Entry, id: String, press: Message) -> Element<'a, Message> {
    let mark: Element<'a, Message> = match e.mark {
        Mark::Chosen => badge(Space::new(), 8.0, look::GOOD, look::GOOD).boxed(),
        Mark::Device => badge(Space::new(), 8.0, Color::TRANSPARENT, look::TEXT_FAINT).boxed(),
        Mark::Here => icon(Icon::Check).size(14.0).color(look::GOOD).boxed(),
        Mark::Up => text("↑").size(size::BODY).color(look::TEXT_DIM).boxed(),
        Mark::Folder => text("▸").size(size::BODY).color(look::FOLDER).boxed(),
    };
    let name_color = match e.mark {
        Mark::Folder => look::FOLDER,
        Mark::Up => look::TEXT_DIM,
        _ => look::TEXT,
    };
    let mut line = row![
        container(mark).center_x(16.0),
        text(e.name.clone())
            .size(size::BODY)
            .font(if e.selected {
                look::font::label()
            } else {
                look::font::body()
            })
            .color(name_color)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::End),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center);
    if !e.detail.is_empty() {
        line = line.push(
            text(e.detail.clone())
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End)
                .boxed(),
        );
    }
    item(line.boxed())
        .padding(Padding::from([6.0, space::M]))
        .selected(e.selected)
        .id(id)
        .on_press(press)
        .boxed()
}
