//! Now playing ([`crate::view::now::NowPlaying`]) in iced's widgets: the
//! cover as large as the window holds it, in the middle, the caption over
//! its bottom on a shade (the title in the display size, the artist and
//! the album, the format's chips, the tempo with its beat pulse); with no
//! cover, a placeholder of its size and the caption under it.
//!
//! The title, the artist and the album are links (underlined under the
//! pointer), each going where its key goes; a click anywhere else
//! returns, as the key does.

use iced::widget::{
    column, container, mouse_area, responsive, rich_text, span, stack, text, Column, Row, Space,
};
use iced::{Alignment, Background, Border, Element, Fill, Padding, Size, Widget as _};

use crate::core::keymap::Op;
use crate::core::text::{Seg, Tone};
use crate::core::{Action, Screen, State};
use crate::gui::look::{self, radius, size, space};
use crate::gui::widgets::cover::{shade, Align, Covers, Spot};
use crate::gui::window::Message;
use crate::view::now::{NowPlaying, Pulse};

/// The room round the cover, in points.
const MARGIN: f32 = space::XL;

/// How far above the caption the cover starts to darken, in points.
const LEAD: f32 = 96.0;

/// The room the caption takes under a placeholder, in points (with no
/// cover it goes under it).
const CAPTION: f32 = 150.0;

/// Where Now playing's cover goes in a window of `size` (points) at
/// `scale` pixels a point, while it shows: the window but for a margin,
/// the cover in the middle of it.
pub fn spot(state: &State, size: Size, scale: f32) -> Option<Spot> {
    if state.screen != Screen::NowPlaying {
        return None;
    }
    let px = |points: f32| (points * scale).max(0.0).round() as usize;
    Some(Spot {
        place: (
            px(MARGIN),
            px(MARGIN),
            px(size.width - 2.0 * MARGIN),
            px(size.height - 2.0 * MARGIN),
        ),
        align: Align::Center,
    })
}

/// Now playing for `state`, the cover from `covers`.
pub fn view<'a>(state: &'a State, covers: &'a Covers) -> Element<'a, Message> {
    let caption = NowPlaying::of(state, state.heard());
    let body: Element<'a, Message> = match covers.art() {
        Some(art) => {
            let over = column![
                Space::new().height(Fill),
                shade(words(&caption)).width(Fill).padding(Padding {
                    top: LEAD,
                    right: space::XXL,
                    bottom: space::XXL,
                    left: space::XXL,
                }),
            ];
            container(stack![art, over])
                .center(Fill)
                .padding(MARGIN)
                .boxed()
        }
        None => responsive(move |window| placeholder(&caption, window)).boxed(),
    };
    mouse_area(
        container(body)
            .width(Fill)
            .height(Fill)
            .style(|_| container::Style {
                background: Some(Background::Color(look::BG)),
                ..container::Style::default()
            }),
    )
    .on_press(Message::Act(Action::Op(Op::NowPlaying)))
    .boxed()
}

/// With no cover: a placeholder as large as the cover would be, the
/// caption's room left under it, and the caption there.
fn placeholder<'a>(caption: &NowPlaying, window: Size) -> Element<'a, Message> {
    let side = (window.width - 2.0 * MARGIN)
        .min(window.height - 2.0 * MARGIN - CAPTION)
        .max(0.0);
    let note = text("♪")
        .size(side / 4.0)
        .font(look::font::title())
        .color(look::EDGE);
    let square = container(note)
        .center_x(side)
        .center_y(side)
        .style(|_| container::Style {
            background: Some(Background::Color(look::SURFACE)),
            border: Border {
                color: look::EDGE,
                width: look::line::EDGE,
                radius: radius::PANEL.into(),
            },
            ..container::Style::default()
        });
    let under = container(words(caption)).width(side).padding(Padding {
        top: space::XL,
        ..Padding::ZERO
    });
    container(column![square, under])
        .center(Fill)
        .padding(MARGIN)
        .boxed()
}

/// The caption: the title, the artist and the album, the chips and the
/// tempo.
fn words<'a>(caption: &NowPlaying) -> Element<'a, Message> {
    let title = linked(
        std::slice::from_ref(&caption.title),
        size::DISPLAY,
        look::font::title(),
    );
    let mut lines = Column::new().spacing(space::S).push(title);
    if !caption.by.is_empty() {
        lines = lines.push(linked(&caption.by, size::TITLE, look::font::body()));
    }
    let mut facts = Row::new().spacing(space::S).align_y(Alignment::Center);
    for chip in &caption.chips {
        facts = facts.push(chip_of(chip));
    }
    if let Some(tempo) = &caption.tempo {
        facts = facts.push(Space::new().width(space::XS).boxed());
        facts = facts.push(beat(tempo, caption.pulse));
    }
    if !caption.chips.is_empty() || caption.tempo.is_some() {
        lines = lines.push(Space::new().height(space::XS).boxed());
        lines = lines.push(facts.boxed());
    }
    lines.boxed()
}

/// `segs` as one paragraph at `size` in `font`, the links among them
/// clickable (underlined under the pointer).
fn linked<'a>(segs: &[Seg], size: f32, font: iced::Font) -> Element<'a, Message> {
    let spans: Vec<_> = segs
        .iter()
        .map(|s| {
            let color = match s.tone {
                Tone::Title | Tone::TitleLink | Tone::ArtistLink => look::TEXT,
                Tone::AlbumLink => look::TEXT_DIM,
                _ => look::TEXT_FAINT,
            };
            span(s.text.clone())
                .color(color)
                .link_maybe(NowPlaying::link(s.tone))
        })
        .collect();
    rich_text(spans)
        .size(size)
        .font(font)
        .on_link_click(|op| Message::Act(Action::Op(op)))
        .boxed()
}

/// A chip of the caption: the format and rate in the source's tone, the
/// status light's word quiet.
fn chip_of<'a>(chip: &Seg) -> Element<'a, Message> {
    let tone = match chip.tone {
        Tone::Source => look::SOURCE,
        Tone::Warn => look::WARN,
        _ => look::TEXT_DIM,
    };
    container(
        text(chip.text.clone())
            .size(size::LABEL)
            .font(look::font::numbers())
            .color(tone),
    )
    .center_y(look::height::CHIP)
    .padding(Padding::from([0.0, space::M]))
    .style(move |_| container::Style {
        background: Some(Background::Color(look::alpha(look::BG, 0.55))),
        border: Border {
            color: look::alpha(tone, 0.35),
            width: look::line::EDGE,
            radius: radius::PILL.into(),
        },
        ..container::Style::default()
    })
    .boxed()
}

/// The tempo line, its `♩` lit with the beat (in the accent on a bar's
/// first beat).
fn beat<'a>(tempo: &[Seg], pulse: Pulse) -> Element<'a, Message> {
    let spans: Vec<_> = tempo
        .iter()
        .map(|s| {
            let (color, font) = match (s.tone, pulse) {
                (Tone::Pulse, Pulse::Bar) => (look::ACCENT, look::font::title()),
                (Tone::Pulse, Pulse::Beat) => (look::TEXT, look::font::title()),
                (Tone::Pulse, Pulse::Rest) => (look::TEXT_FAINT, look::font::body()),
                _ => (look::TEXT_DIM, look::font::numbers()),
            };
            span::<Op>(s.text.clone()).color(color).font(font)
        })
        .collect();
    rich_text(spans).size(size::BODY).boxed()
}
