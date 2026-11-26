//! The right column's preparation view ([`crate::view::prep::PrepView`])
//! in iced's widgets: in the cover's place a card with the latest result
//! big (the beat's waveform with its ticks and the play head's pulse, the
//! key on a keyboard, the stems' lanes filling, the words lit in time, the
//! cover's subject lifting off it), the headline and the song under it,
//! then the steps one under the other (each its icon, what it does or
//! found, and a spinner while it runs or a tick once found). A story of
//! the song, once one is told, shows in the card: its words typed out, its
//! scenes fading in. The wheel over it scrolls the list.

use iced::widget::text::Wrapping;
use iced::widget::{column, container, rich_text, row, span, text, Column, Row, Space};
use iced::{Alignment, Background, Border, Element, Fill, Length, Rectangle, Size, Widget as _};

use crate::core::State;
use crate::gui::look::{self, radius, size, space};
use crate::gui::metrics;
use crate::gui::style::mix;
use crate::gui::widgets::cover::Covers;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::prep::{Hero, Kind, PrepView, Stage, Stand, StoryView};

/// A step's icons.
const ICON: f32 = 14.0;
/// The pulse's metronome beside the tempo.
const PULSE: f32 = 22.0;
/// A stem's lane.
const LANE: f32 = 22.0;
/// The column of a stem's name before its lane.
const LANE_NAME: f32 = 52.0;
/// A story scene's picture.
const SCENE: f32 = 64.0;
/// The room the card leaves under it: the headline, the song and every
/// step.
const BELOW: f32 = 200.0;
/// The least the card is shown at.
const LEAST_CARD: f32 = 96.0;

/// The card's place in the main screen `m` lays out: at the cover's, as
/// wide, as high as leaves room for the steps under it; none where the
/// column has no room for it.
pub fn card_place(m: &metrics::Main) -> Option<Rectangle> {
    let (cover, right) = (m.cover?, m.right?);
    let height = cover
        .height
        .min(right.height - BELOW - metrics::UNDER_COVER);
    (height >= LEAST_CARD).then(|| Rectangle::new(cover.position(), Size::new(cover.width, height)))
}

/// The column for `view`, the card in its `place` (where the column has
/// room for one), `width` points wide.
pub fn view<'a>(
    view: &PrepView,
    state: &State,
    place: Option<Rectangle>,
    covers: &'a Covers,
    width: f32,
) -> Element<'a, Message> {
    let mut parts = Column::<Element<'a, Message>>::new().width(width);
    if let Some(place) = place {
        parts = parts
            .push(
                container(card(hero(view, covers, place.width.min(place.height))))
                    .width(place.width)
                    .height(place.height)
                    .boxed(),
            )
            .push(Space::new().height(metrics::UNDER_COVER).boxed());
    }
    let found = view
        .stages
        .iter()
        .any(|s| s.stand == Stand::Found && s.reveal < 1.0);
    let headline = text(view.headline.clone())
        .size(size::TITLE)
        .font(look::font::title())
        .color(if found { look::ACCENT } else { look::TEXT });
    let about = text(if view.done {
        format!("Ready: {}", view.title)
    } else {
        format!("Getting ready: {}", view.title)
    })
    .size(size::CAPTION)
    .font(look::font::body())
    .color(look::TEXT_DIM)
    .wrapping(Wrapping::None)
    .ellipsis(iced::widget::text::Ellipsis::End);
    parts = parts.push(column![headline, about].spacing(space::XS / 2.0).boxed());
    let steps = view
        .stages
        .iter()
        .fold(Column::new().spacing(space::XS), |c, s| {
            c.push(stage(state, s))
        });
    parts = parts.push(
        container(steps)
            .padding(iced::Padding::from([space::S, 0.0]))
            .boxed(),
    );
    parts.spacing(space::S).boxed()
}

/// `content` on the card in the cover's place: raised, its corners
/// rounded, centred.
fn card<'a>(content: Element<'a, Message>) -> Element<'a, Message> {
    container(content)
        .center(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::RAISED)),
            border: Border {
                radius: radius::PANEL.into(),
                ..Border::default()
            },
            ..container::Style::default()
        })
        .clip(true)
        .boxed()
}

/// What the card shows: the hero, `side` points across.
fn hero<'a>(view: &PrepView, covers: &'a Covers, side: f32) -> Element<'a, Message> {
    let inner = (side - 2.0 * space::L).max(0.0);
    let padded =
        |content: Element<'a, Message>| container(content).padding(space::L).center(Fill).boxed();
    match &view.hero {
        Hero::Waiting => padded(
            column![
                widgets::icon(Icon::Busy).size(28.0).color(look::TEXT_FAINT),
                text("Its turn comes")
                    .size(size::LABEL)
                    .font(look::font::label())
                    .color(look::TEXT_FAINT),
            ]
            .spacing(space::S)
            .align_x(Alignment::Center)
            .boxed(),
        ),
        Hero::Beat(b) => {
            let (lit, bar) = b.pulse;
            let pulse = mix(
                look::TEXT_FAINT,
                if bar { look::ACCENT } else { look::TEXT },
                lit,
            );
            let bpm = view
                .stages
                .iter()
                .find(|s| s.kind == Kind::Beat)
                .and_then(|s| s.line.strip_prefix("Found it: "))
                .unwrap_or("")
                .to_owned();
            padded(
                column![
                    row![
                        widgets::icon(Icon::Tempo).size(PULSE).color(pulse),
                        text(bpm)
                            .size(size::DISPLAY)
                            .font(look::font::numbers())
                            .color(look::TEXT),
                    ]
                    .spacing(space::S)
                    .align_y(Alignment::Center)
                    .boxed(),
                    widgets::wave(b.peaks.clone(), look::alpha(look::TEXT_DIM, 0.8))
                        .ticks(b.ticks.clone())
                        .head(b.head, b.pulse)
                        .view(inner * 0.4),
                ]
                .spacing(space::M)
                .align_x(Alignment::Center)
                .boxed(),
            )
        }
        Hero::Key(k) => {
            const NAMES: [&str; 12] = [
                "C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B",
            ];
            let labels = (k.lit.iter())
                .map(|&n| (n, NAMES[usize::from(n % 12)].to_owned()))
                .collect();
            padded(
                column![
                    text(k.name.clone())
                        .size(size::DISPLAY)
                        .font(look::font::title())
                        .color(look::TEXT),
                    container(widgets::piano(2, k.lit.clone(), labels))
                        .width(Fill)
                        .height(inner * 0.4),
                ]
                .spacing(space::M)
                .align_x(Alignment::Center)
                .boxed(),
            )
        }
        Hero::Stems(s) => {
            let lanes = s
                .lanes
                .iter()
                .fold(Column::new().spacing(space::S), |c, (name, peaks)| {
                    c.push(
                        row![
                            text(*name)
                                .size(size::CAPTION)
                                .font(look::font::label())
                                .color(look::TEXT_DIM)
                                .width(LANE_NAME),
                            widgets::wave(peaks.clone(), look::ACCENT)
                                .filled(s.share)
                                .view(LANE),
                        ]
                        .spacing(space::S)
                        .align_y(Alignment::Center),
                    )
                });
            padded(
                column![lanes, widgets::progress(f64::from(s.share), look::ACCENT),]
                    .spacing(space::M)
                    .boxed(),
            )
        }
        Hero::Words(w) => {
            let lines = w.lines.iter().fold(
                Column::new().spacing(space::S).align_x(Alignment::Center),
                |c, line| {
                    let spans: Vec<_> = (line.iter().enumerate())
                        .map(|(k, (word, lit))| {
                            let gap = if k == 0 { "" } else { " " };
                            span::<()>(format!("{gap}{word}")).color(mix(
                                look::TEXT_FAINT,
                                look::ACCENT,
                                *lit,
                            ))
                        })
                        .collect();
                    c.push(
                        rich_text(spans)
                            .size(size::TITLE)
                            .font(look::font::title())
                            .align_x(Alignment::Center)
                            .width(Fill),
                    )
                },
            );
            padded(lines.boxed())
        }
        Hero::Story(story) => padded(story_slot(story, covers)),
        Hero::Cover(c) => match covers.cut(c.sway, c.lift) {
            Some(cut) => container(cut).center(Fill).boxed(),
            None => padded(Space::new().boxed()),
        },
    }
}

/// A step's line: its icon, what it says, and its mark (a spinner while
/// it runs, a tick once found).
fn stage<'a>(state: &State, s: &Stage) -> Element<'a, Message> {
    let icon = match s.kind {
        Kind::Beat => Icon::Tempo,
        Kind::Key => Icon::Key,
        Kind::Stems => Icon::Stems,
        Kind::Words => Icon::Lyrics,
        Kind::Cover => Icon::Cover,
    };
    let (tone, words) = match s.stand {
        Stand::Found => (mix(look::TEXT_FAINT, look::ACCENT, s.reveal), look::TEXT),
        Stand::Running => (look::ACCENT, look::TEXT),
        Stand::Waiting | Stand::Missing => (look::TEXT_FAINT, look::TEXT_FAINT),
    };
    let mark: Element<'a, Message> = match s.stand {
        Stand::Running => widgets::icon(Icon::Busy)
            .size(ICON)
            .color(look::ACCENT)
            .turn(super::list::turn(state))
            .into(),
        Stand::Found => widgets::icon(Icon::Check)
            .size(ICON)
            .color(look::alpha(look::GOOD, s.reveal))
            .into(),
        Stand::Waiting | Stand::Missing => Space::new().width(ICON).boxed(),
    };
    row![
        widgets::icon(icon).size(ICON).color(tone),
        text(s.line.clone())
            .size(size::BODY)
            .font(look::font::body())
            .color(words)
            .wrapping(Wrapping::None)
            .ellipsis(iced::widget::text::Ellipsis::End)
            .width(Fill),
        mark,
    ]
    .spacing(space::S)
    .align_y(Alignment::Center)
    .boxed()
}

/// The story's slot: its words as written so far (a caret after them
/// while they are), its scenes fading in as they are drawn under them.
fn story_slot<'a>(story: &StoryView, covers: &'a Covers) -> Element<'a, Message> {
    let caret = if story.caret { "▏" } else { "" };
    let words = text(format!("{}{caret}", story.text))
        .size(size::BODY)
        .font(look::font::body())
        .color(look::TEXT_DIM)
        .width(Fill);
    let scenes = (0..story.scenes.len())
        .filter_map(|k| covers.scene(k, story.scenes[k].1, SCENE))
        .fold(
            Row::<Element<'a, Message>>::new().spacing(space::S),
            |r, scene| r.push(scene),
        );
    column![words, scenes]
        .spacing(space::S)
        .width(Length::Fill)
        .boxed()
}
