//! Surfaces: a panel ([`panel`]: the list pane, a screen's body) and a
//! card on it ([`card`]); a tinted note for what needs attention
//! ([`note`]); a disc for a status mark ([`badge`]). Grouping comes from
//! these and from space, not from lines.

use iced::widget::{container, Container};
use iced::{Background, Border, Color};

use crate::gui::look::{self, radius, space};

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
