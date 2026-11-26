//! The help ([`crate::view::help`]) in iced's widgets: every key by
//! context, each section a card, what the keys do with the keys as key
//! caps at the right; the title, the search while there is one and the
//! keys that move about at the top. Every line is one [`measure::LINE`] high, so the
//! lines the window reports to the core are the lines shown, and its
//! scrolling and paging are the page's. The wheel scrolls it, three lines
//! a step, as the keys do.

use iced::widget::{container, mouse_area, responsive, row, text, Column, Row, Space};
use iced::{Alignment, Background, Border, Element, Fill, Length, Padding, Widget as _};

use crate::core::State;
use crate::gui::look::{self, radius, size, space};
use crate::gui::widgets::{hints, key};
use crate::gui::window::Message;
use crate::view::help::{self, measure, Line, Page, Search, TITLE};

/// The widest the cards go, in points.
const WIDEST: f32 = 880.0;

/// The help for `state`.
pub fn view<'a>(state: &State) -> Element<'a, Message> {
    let page = Page::of(state);
    let content = responsive(move |size| {
        let width = (size.width - 2.0 * space::XXL).clamp(0.0, WIDEST);
        container(lines(&page).width(width)).center_x(Fill).boxed()
    });
    mouse_area(
        container(content)
            .width(Fill)
            .height(Fill)
            .padding(Padding {
                top: measure::TOP,
                bottom: measure::BOTTOM,
                ..Padding::ZERO
            })
            .style(|_| container::Style {
                background: Some(Background::Color(look::BG)),
                ..container::Style::default()
            }),
    )
    .on_scroll(Message::Wheel)
    .boxed()
}

/// The page's column: the header, then the lines.
fn lines<'a>(page: &Page) -> Column<Element<'a, Message>> {
    let mut header = Row::new()
        .spacing(space::L)
        .align_y(Alignment::Center)
        .height(measure::HEADER)
        .push(
            text(TITLE)
                .size(size::TITLE)
                .font(look::font::title())
                .color(look::TEXT)
                .boxed(),
        );
    if let Some(search) = &page.search {
        header = header.push(searched(search));
    }
    header = header
        .push(Space::new().width(Fill).boxed())
        .push(hints(&help::hints()).boxed());
    let mut all = Column::new()
        .push(header.boxed())
        .push(Space::new().height(measure::UNDER_HEADER).boxed());
    if let Some(empty) = &page.empty {
        all = all.push(line_box(
            text(empty.clone())
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT_DIM)
                .boxed(),
        ));
    }
    blocks(&page.lines)
        .into_iter()
        .fold(all, |all, block| all.push(block))
}

/// The search as a field: `/`, what is typed with the caret while it is
/// typed, and how many keys it leaves.
fn searched<'a>(search: &Search) -> Element<'a, Message> {
    let mut typed = Row::new()
        .spacing(space::S)
        .align_y(Alignment::Center)
        .push(key("/"))
        .push(
            text(search.typed.clone())
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT)
                .boxed(),
        );
    if search.editing {
        typed = typed.push(
            container(Space::new())
                .width(2.0)
                .height(size::BODY + 4.0)
                .style(|_| container::Style {
                    background: Some(Background::Color(look::ACCENT)),
                    ..container::Style::default()
                })
                .boxed(),
        );
    }
    let field = row![
        typed,
        Space::new().width(Fill),
        text(format!("{} shown", search.shown))
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(look::TEXT_FAINT),
    ]
    .spacing(space::M)
    .align_y(Alignment::Center);
    let edge = if search.editing {
        look::alpha(look::ACCENT, 0.6)
    } else {
        look::EDGE
    };
    container(field)
        .width(Length::Fixed(320.0))
        .center_y(measure::HEADER)
        .padding(Padding::from([0.0, space::S]))
        .style(move |_| container::Style {
            background: Some(Background::Color(look::SURFACE)),
            border: Border {
                color: edge,
                width: look::line::EDGE,
                radius: radius::CONTROL.into(),
            },
            ..container::Style::default()
        })
        .boxed()
}

/// The lines as cards: each run of a section's lines on one, the gaps
/// between them as space.
fn blocks<'a>(lines: &[Line]) -> Vec<Element<'a, Message>> {
    let mut out = Vec::new();
    let mut card: Vec<Element<'a, Message>> = Vec::new();
    let close = |card: &mut Vec<Element<'a, Message>>, out: &mut Vec<Element<'a, Message>>| {
        if !card.is_empty() {
            let lines = Column::with_children(std::mem::take(card)).width(Fill);
            out.push(
                container(lines)
                    .padding(Padding::from([0.0, space::L]))
                    .width(Fill)
                    .style(|_| container::Style {
                        background: Some(Background::Color(look::SURFACE)),
                        border: Border {
                            radius: radius::PANEL.into(),
                            ..Border::default()
                        },
                        ..container::Style::default()
                    })
                    .boxed(),
            );
        }
    };
    for line in lines {
        match line {
            Line::Title(title) => card.push(line_box(
                text(*title)
                    .size(size::BODY)
                    .font(look::font::title())
                    .color(look::TEXT)
                    .boxed(),
            )),
            Line::Keys { keys, what } => card.push(keyed(keys, what)),
            Line::Gap => {
                close(&mut card, &mut out);
                out.push(Space::new().height(measure::LINE).boxed());
            }
        }
    }
    close(&mut card, &mut out);
    out
}

/// `content` on a line of its own, in the middle of its height.
fn line_box<'a>(content: Element<'a, Message>) -> Element<'a, Message> {
    container(content).center_y(measure::LINE).boxed()
}

/// A key's line: what the keys do, the keys as caps at the right.
fn keyed<'a>(keys: &[String], what: &'static str) -> Element<'a, Message> {
    let caps = keys
        .iter()
        .fold(Row::new().spacing(space::XS), |caps, name| {
            caps.push(key(name.clone()))
        })
        .align_y(Alignment::Center);
    row![
        text(what)
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT_DIM)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::End)
            .width(Fill),
        caps,
    ]
    .spacing(space::L)
    .align_y(Alignment::Center)
    .height(measure::LINE)
    .boxed()
}
