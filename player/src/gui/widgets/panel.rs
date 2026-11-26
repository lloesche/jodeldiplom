//! Surfaces: a panel ([`panel`]: the list pane, a screen's body) and a
//! card on it ([`card`]); a tinted note for what needs attention
//! ([`note`]); a disc for a status mark ([`badge`]); a tinted label on a
//! pill ([`tag`]), named ([`named_tag`]) or lit ([`lit_tag`]), and a
//! key's cap ([`key`]). Grouping comes from these and from space, not
//! from lines.

use iced::widget::{container, row, text, Container};
use iced::{Background, Border, Color, Element, Padding, Widget as _};

use crate::gui::look::{self, radius, size, space};

/// `content` on a panel: the `surface`, padded, its corners rounded.
pub fn panel<'a, W>(content: W) -> Container<'a, W> {
    surface(content, look::SURFACE)
}

/// `content` on a card: `raised` above a panel.
pub fn card<'a, W>(content: W) -> Container<'a, W> {
    surface(content, look::RAISED)
}

/// `content` on a note tinted with `tone` (a status colour) on a panel:
/// what needs attention, set apart.
pub fn note<'a, W>(content: W, tone: Color) -> Container<'a, W> {
    container(content)
        .padding(space::L)
        .style(move |_| container::Style {
            background: Some(Background::Color(look::tint(tone, look::SURFACE, 0.08))),
            border: Border {
                color: look::tint(tone, look::SURFACE, 0.3),
                width: look::line::EDGE,
                radius: radius::PANEL.into(),
            },
            ..container::Style::default()
        })
}

/// `content` centred on a disc `size` points across, of `fill` with an
/// `edge` (transparent: none): a status mark.
pub fn badge<'a, W>(content: W, size: f32, fill: Color, edge: Color) -> Container<'a, W> {
    container(content)
        .center_x(size)
        .center_y(size)
        .style(move |_| container::Style {
            background: Some(Background::Color(fill)),
            border: Border {
                color: edge,
                width: if edge.a > 0.0 { look::line::EDGE } else { 0.0 },
                radius: radius::PILL.into(),
            },
            ..container::Style::default()
        })
}

/// `label` on a pill tinted with `tone`: a file's format and rate, a
/// tier. It says, it does not switch (a switch is [`super::chip`]).
pub fn tag<'a, Message: 'a>(label: impl Into<String>, tone: Color) -> Element<'a, Message> {
    // See-through, so it shows on a selected row as on the panel.
    pill(caption(label, tone).boxed(), look::alpha(tone, 0.13))
}

/// `value` in `tone` after its name `label`, faint, on a pill tinted with
/// `tone`: a fact named (the transition's preset, the next track's
/// tempo).
pub fn named_tag<'a, Message: 'a>(
    label: impl Into<String>,
    value: impl Into<String>,
    tone: Color,
) -> Element<'a, Message> {
    let name = text(label.into())
        .size(size::CAPTION)
        .font(look::font::body())
        .color(look::TEXT_FAINT)
        .wrapping(text::Wrapping::None);
    pill(
        row![name, caption(value, tone)].spacing(space::XS).boxed(),
        look::alpha(tone, 0.13),
    )
}

/// `label` in the accent's ink on a pill of `tone`: a tag lit (the
/// Transition tab's lane whose fade is being edited).
pub fn lit_tag<'a, Message: 'a>(label: impl Into<String>, tone: Color) -> Element<'a, Message> {
    pill(caption(label, look::ACCENT_INK).boxed(), tone)
}

/// A tag's words, in `tone`.
fn caption<'a>(label: impl Into<String>, tone: Color) -> iced::widget::Text<'a> {
    text(label.into())
        .size(size::CAPTION)
        .font(look::font::label())
        .color(tone)
        .wrapping(text::Wrapping::None)
}

/// `content` on a pill of `fill`: a tag.
fn pill<'a, Message: 'a>(content: Element<'a, Message>, fill: Color) -> Element<'a, Message> {
    container(content)
        .padding(Padding::from([2.0, space::S]))
        .style(move |_| container::Style {
            background: Some(Background::Color(fill)),
            border: Border {
                radius: radius::PILL.into(),
                ..Border::default()
            },
            ..container::Style::default()
        })
        .boxed()
}

/// A key's cap, `key` on it: what to press for a command.
pub fn key<'a, Message: 'a>(key: impl Into<String>) -> Element<'a, Message> {
    keycap(key, look::SELECTED, look::TEXT)
}

/// A key's cap of `fill`, `key` on it in `ink` (a button's key after its
/// label takes the button's tones).
pub fn keycap<'a, Message: 'a>(
    key: impl Into<String>,
    fill: Color,
    ink: Color,
) -> Element<'a, Message> {
    container(
        text(key.into())
            .size(size::CAPTION)
            .font(look::font::label())
            .color(ink)
            .wrapping(text::Wrapping::None),
    )
    .padding(Padding::from([1.0, 6.0]))
    .style(move |_| container::Style {
        background: Some(Background::Color(fill)),
        border: Border {
            radius: radius::CONTROL.into(),
            ..Border::default()
        },
        ..container::Style::default()
    })
    .boxed()
}

fn surface<'a, W>(content: W, fill: Color) -> Container<'a, W> {
    container(content)
        .padding(space::L)
        .style(move |_| container::Style {
            background: Some(Background::Color(fill)),
            border: Border {
                radius: radius::PANEL.into(),
                ..Border::default()
            },
            ..container::Style::default()
        })
}
