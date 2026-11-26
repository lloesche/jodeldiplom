//! The tab's rows of words: a block's header (which track, its title, its
//! facts as chips at the right), and the footer (the zoom, or the fade
//! being edited, the keys as caps with what they do, and the button that
//! zooms to fit the lyrics). Where a row has not room for all, the facts
//! (or keys) that matter least go first.

use iced::widget::{container, row, text, Row, Space};
use iced::{Alignment, Background, Border, Element, Fill, Widget as _};

use crate::core::keymap::Op;
use crate::core::{Action, State};
use crate::gui::look::{self, radius, size, space};
use crate::gui::widgets;
use crate::gui::window::Message;
use crate::view::transition::{fit_lyrics, Fact, Footer, Header, Tone};

/// An average letter of a caption's, in points: to tell what fits (iced
/// lays the row out after it is made).
const LETTER: f32 = 6.6;
/// A tag's padding, both sides together.
const TAG_PAD: f32 = 2.0 * space::S;
/// The least the title keeps of a header's width.
const TITLE: f32 = 140.0;
/// The block's name's column.
const NAME: f32 = 40.0;
/// The focus mark.
const MARK: f32 = 3.0;

/// About how wide `text` is in a caption.
fn wide(text: &str) -> f32 {
    text.chars().count() as f32 * LETTER
}

/// About how wide a fact's chip is.
fn fact_width(fact: &Fact) -> f32 {
    let label = fact.label.map_or(0.0, |l| wide(l) + space::XS);
    let pad = if fact.tone == Tone::Hint {
        0.0
    } else {
        TAG_PAD
    };
    label + wide(&fact.text) + pad
}

/// Of `items` (in order, each with how long it stays and how wide it is),
/// those that fit `room` points with `gap` between them: the ones that
/// stay the shortest go first.
fn fitting<T>(
    items: &[T],
    room: f32,
    gap: f32,
    keep: impl Fn(&T) -> u8,
    width: impl Fn(&T) -> f32,
) -> Vec<&T> {
    let mut shown: Vec<bool> = vec![true; items.len()];
    let total = |shown: &[bool]| -> f32 {
        items
            .iter()
            .zip(shown)
            .filter(|(_, s)| **s)
            .map(|(i, _)| width(i) + gap)
            .sum()
    };
    while total(&shown) > room {
        let Some(least) = (0..items.len())
            .filter(|&i| shown[i])
            .min_by_key(|&i| (keep(&items[i]), std::cmp::Reverse(i)))
        else {
            break;
        };
        shown[least] = false;
    }
    items
        .iter()
        .zip(shown)
        .filter(|(_, s)| *s)
        .map(|(i, _)| i)
        .collect()
}

/// A fact as a chip (a hint as words alone).
fn fact<'a>(fact: &Fact) -> Element<'a, Message> {
    let tone = match fact.tone {
        Tone::Chip => look::TEXT_DIM,
        Tone::Lit => return widgets::lit_tag(fact.text.clone(), look::ACCENT),
        Tone::Warn => look::WARN,
        Tone::Hint => {
            return text(fact.text.clone())
                .size(size::CAPTION)
                .font(look::font::body())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None)
                .boxed()
        }
    };
    match fact.label {
        Some(label) => widgets::named_tag(label, fact.text.clone(), tone),
        None => widgets::tag(fact.text.clone(), tone),
    }
}

/// A block's header, `width` points wide: the focus mark, the block's
/// name, the title (shortened at its end), and the facts that fit.
pub(super) fn header<'a>(h: &Header, width: f32) -> Element<'a, Message> {
    let mark: Element<'a, Message> = if h.focused {
        container(Space::new().width(MARK).height(14.0))
            .style(|_| container::Style {
                background: Some(Background::Color(look::ACCENT)),
                border: Border {
                    radius: radius::PILL.into(),
                    ..Border::default()
                },
                ..container::Style::default()
            })
            .boxed()
    } else {
        Space::new().width(MARK).boxed()
    };
    let name = text(h.name)
        .size(size::LABEL)
        .font(look::font::label())
        .color(if h.focused {
            look::ACCENT
        } else {
            look::TEXT_DIM
        })
        .wrapping(text::Wrapping::None)
        .width(NAME);
    let title = match &h.title {
        Ok(title) => text(title.clone())
            .size(size::BODY)
            .font(look::font::label())
            .color(look::TEXT),
        Err(what) => text(*what)
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT_FAINT),
    }
    .wrapping(text::Wrapping::None)
    .ellipsis(text::Ellipsis::End)
    .width(Fill);
    let room = width - MARK - NAME - TITLE - 3.0 * space::S;
    let mut facts = Row::new().spacing(space::S).align_y(Alignment::Center);
    for f in fitting(&h.facts, room, space::S, |f| f.keep, fact_width) {
        facts = facts.push(fact(f));
    }
    row![mark, name, title, facts]
        .spacing(space::S)
        .align_y(Alignment::Center)
        .height(Fill)
        .boxed()
}

/// About how wide a key's hint is ([`widgets::hints`]: its cap, a gap,
/// its words).
fn hint_width((key, what): &(String, &str)) -> f32 {
    wide(key) + 12.0 + space::S + wide(what)
}

/// The tab's footer for `state`, `width` points wide: the focused track's
/// zoom (or the fade edited), the keys that fit, and the button that zooms
/// to fit the lyrics.
pub(super) fn footer<'a>(state: &State, width: f32) -> Element<'a, Message> {
    let f = Footer::of(state);
    let lead: Element<'a, Message> = if f.editing {
        row![
            widgets::lit_tag("editing", look::ACCENT),
            text(f.zoom.clone())
                .size(size::LABEL)
                .font(look::font::label())
                .color(look::TEXT)
                .wrapping(text::Wrapping::None)
        ]
        .spacing(space::S)
        .align_y(Alignment::Center)
        .boxed()
    } else {
        widgets::tag(f.zoom.clone(), look::TEXT_DIM)
    };
    let (key, what) = fit_lyrics();
    let fit = widgets::secondary(what)
        .key(key)
        .small()
        .on_press(Message::Act(Action::Op(Op::TransitionFitLyrics)));
    let lead_w = wide(&f.zoom) + if f.editing { 70.0 } else { TAG_PAD };
    let room = width - lead_w - 130.0 - space::L;
    // The keys in order, the last going first.
    let shown: Vec<(String, &str)> = fitting(&f.keys, room, space::L, |_| 0, hint_width)
        .into_iter()
        .cloned()
        .collect();
    let keys = widgets::hints(&shown);
    row![lead, keys, Space::new().width(Fill), Element::from(fit)]
        .spacing(space::L)
        .align_y(Alignment::Center)
        .height(Fill)
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What does not fit goes, the least kept first, the later of equals
    /// before the earlier; the rest keep their order.
    #[test]
    fn what_matters_least_goes_first() {
        let items = [(3u8, 50.0f32), (1, 50.0), (2, 50.0), (1, 50.0)];
        let keep = |i: &(u8, f32)| i.0;
        let width = |i: &(u8, f32)| i.1;
        let all = fitting(&items, 1000.0, 10.0, keep, width);
        assert_eq!(all.len(), 4);
        let three = fitting(&items, 180.0, 10.0, keep, width);
        assert_eq!(three, [&items[0], &items[1], &items[2]]);
        let one = fitting(&items, 60.0, 10.0, keep, width);
        assert_eq!(one, [&items[0]]);
        assert!(fitting(&items, 10.0, 10.0, keep, width).is_empty());
    }
}
