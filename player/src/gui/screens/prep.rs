//! The right column's preparation view ([`crate::view::prep::PrepView`])
//! in iced's widgets: in the cover's place a card with the latest result
//! big (the beat's waveform with its ticks and the play head's pulse, the
//! key on a keyboard, the stems' lanes filling, the words lit in time), or
//! the song's cover; the headline and the song under it, then the steps
//! one under the other (each its icon, what it does or found, and a
//! spinner while it runs or a tick once found). While the song's story is
//! made, it is built on the cover in the card: its words typed out over
//! the cover darkened, its scenes taking the cover's place tile by tile.
//! The wheel over it scrolls the list.

use iced::widget::text::Wrapping;
use iced::widget::{column, container, rich_text, row, span, stack, text, Column, Space};
use iced::{Alignment, Background, Border, Element, Fill, Rectangle, Size, Widget as _};

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
/// The most of a story's words shown over its cover, in characters: its
/// last ones.
const STORY_TAIL: usize = 360;
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
        Hero::Cover => covers.card().map_or_else(
            || super::karaoke::tab::placeholder(side, side),
            |cover| container(cover).center(Fill).boxed(),
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
        Hero::Story(story) => story_slot(story, covers, side),
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
        Kind::Story => Icon::Story,
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

/// The story built on the cover, `side` points across at most: the cover
/// darkened, its scenes over their tiles as they are drawn
/// ([`Covers::building`]; the card's raised ground without a cover), and
/// over it the last of its words as written so far (a caret after them
/// while they are), until its first scene is drawn.
fn story_slot<'a>(story: &StoryView, covers: &'a Covers, side: f32) -> Element<'a, Message> {
    let built = covers
        .building(story)
        .unwrap_or_else(|| Space::new().width(side).height(side).boxed());
    if !story.scenes.is_empty() {
        return container(built).center(Fill).boxed();
    }
    let caret = if story.caret { "▏" } else { "" };
    let words = text(format!("{}{caret}", tail(&story.text, STORY_TAIL)))
        .size(size::BODY)
        .font(look::font::body())
        .color(look::TEXT)
        .width(Fill);
    let over = container(words)
        .padding(space::L)
        .width(Fill)
        .height(Fill)
        .align_y(Alignment::End)
        .clip(true);
    container(stack![built, over]).center(Fill).boxed()
}

/// The last `most` characters of `text`, from a word's start where there
/// is one in them.
fn tail(text: &str, most: usize) -> &str {
    let from = most
        .checked_sub(1)
        .and_then(|k| text.char_indices().rev().nth(k));
    let Some((at, _)) = from else {
        return if most == 0 { "" } else { text };
    };
    let rest = &text[at..];
    if at == 0 || text[..at].ends_with(char::is_whitespace) {
        return rest;
    }
    rest.find(char::is_whitespace)
        .map_or(rest, |space| rest[space..].trim_start())
}

#[cfg(test)]
mod tests {
    use super::tail;

    /// The story's last words shown over the cover: a text shorter than
    /// the most whole; a longer one from a word's start within its last
    /// characters, never inside a character of several bytes.
    #[test]
    fn the_story_s_last_words_start_on_a_word() {
        assert_eq!(tail("A harbour at dusk", 40), "A harbour at dusk");
        assert_eq!(tail("A harbour at dusk", 9), "at dusk");
        assert_eq!(tail("Ünter dem Möwenflug über dem Kai", 12), "über dem Kai");
        assert_eq!(tail("Möwenflug", 3), "lug", "no space: its last ones");
        assert_eq!(tail("", 5), "");
    }
}
