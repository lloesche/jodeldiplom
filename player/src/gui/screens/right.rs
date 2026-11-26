//! The main screen's right column's text ([`crate::view::info`]) in
//! iced's widgets, under the cover at the column's left: the title, the
//! artist and the album, each on a line (ending in `…` where it is longer)
//! and a link to where it is in the browser (underlined under the
//! pointer); then what the file is as a tag beside the tempo and key, a
//! metronome before them lit on each beat (the beat pulse, in the accent
//! on a bar's first); then the way to the device as tags joined by arrows,
//! the device's brighter and lit under the pointer: its click opens the
//! device's setting. Where the window is too narrow for the column, the
//! summary under the list: the play state, the title and the artist, then
//! what the file is and its way, on two lines; a click on it opens Now
//! playing.
//!
//! The wheel over either scrolls the list ([`Message::Wheel`]), as over
//! the bottom bar.

use iced::widget::text::{Ellipsis, Wrapping};
use iced::widget::{column, container, hover, mouse_area, rich_text, row, span, text, Row};
use iced::{mouse, Alignment, Color, Element, Fill, Font, Padding, Widget as _};
use jodel_playback::PlayState;

use crate::core::keymap::Op;
use crate::core::{Action, SettingsRow, State};
use crate::gui::look::{self, size, space};
use crate::gui::style::mix;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::info::{self, Info, Tag, TagKind, NOTHING};

/// The metronome before the tempo.
const PULSE: f32 = 14.0;
/// The play state's icon before the summary's title.
const SUMMARY_ICON: f32 = 12.0;

/// The column's text for `state`, `width` points wide at the left of its
/// place; the wheel scrolls the list over all of the place.
pub fn view<'a>(state: &State, width: f32) -> Element<'a, Message> {
    mouse_area(container(text_of(state)).width(width))
        .on_scroll(Message::Wheel)
        .boxed()
}

/// The column's text.
fn text_of<'a>(state: &State) -> Element<'a, Message> {
    let info = Info::of(state);
    let Some(track) = info.track else {
        return column![
            text(NOTHING)
                .size(size::TITLE)
                .font(look::font::title())
                .color(look::TEXT_DIM),
            text(info::nothing_hint())
                .size(size::CAPTION)
                .font(look::font::body())
                .color(look::TEXT_FAINT),
        ]
        .spacing(space::XS)
        .boxed();
    };
    let title = look::font::title();
    let mut parts = column![link(
        track.title,
        Op::GoTrack,
        size::TITLE,
        title,
        look::TEXT
    )]
    .spacing(space::XS / 2.0);
    if let Some(artist) = track.artist {
        let font = look::font::label();
        parts = parts.push(link(artist, Op::GoArtist, size::BODY, font, look::TEXT));
    }
    if let Some(album) = track.album {
        let font = look::font::body();
        parts = parts.push(link(album, Op::GoAlbum, size::BODY, font, look::TEXT_DIM));
    }
    let (status, way): (Vec<&Tag>, Vec<&Tag>) =
        track.tags.iter().partition(|t| t.kind == TagKind::Status);
    let mut facts = Row::with_children(
        status
            .into_iter()
            .map(|t| widgets::tag(t.text.clone(), tone(t.kind))),
    )
    .spacing(space::S)
    .align_y(Alignment::Center);
    if let Some(tempo) = track.tempo {
        facts = facts.push(
            row![
                widgets::icon(Icon::Tempo)
                    .size(PULSE)
                    .color(pulse(state, track.play)),
                text(tempo)
                    .size(size::LABEL)
                    .font(look::font::numbers())
                    .color(look::TEXT_DIM)
                    .wrapping(Wrapping::None)
                    .ellipsis(Ellipsis::End),
            ]
            .spacing(space::XS)
            .align_y(Alignment::Center)
            .boxed(),
        );
    }
    let way = way
        .into_iter()
        .fold(
            Row::new().spacing(space::XS).align_y(Alignment::Center),
            |r, tag| r.push(step(tag)),
        )
        .wrap()
        .vertical_spacing(space::XS);
    column![parts, column![facts, way].spacing(space::S)]
        .spacing(space::M)
        .width(Fill)
        .boxed()
}

/// `words` as a link to where `op` goes, in `font` at `size` in `color`.
fn link<'a>(words: String, op: Op, size: f32, font: Font, color: Color) -> Element<'a, Message> {
    widgets::link(words, size, font, color, Message::Act(Action::Op(op)))
}

/// A tag's colour.
fn tone(kind: TagKind) -> Color {
    match kind {
        TagKind::Source => look::SOURCE,
        TagKind::Warn => look::WARN,
        TagKind::Status | TagKind::Detail | TagKind::Device => look::TEXT_DIM,
    }
}

/// A tag of the way: after an arrow where it is a step; the device's
/// opens its setting.
fn step<'a>(tag: &Tag) -> Element<'a, Message> {
    let shown: Element<'a, Message> = if tag.kind == TagKind::Device {
        // Brighter than the others, and lit under the pointer: its click
        // opens the device's setting.
        let lit = widgets::tag(tag.text.clone(), look::TEXT);
        mouse_area(hover(widgets::tag(tag.text.clone(), look::TEXT), lit))
            .on_press(Message::Act(Action::OpenSettings(SettingsRow::Device)))
            .interaction(mouse::Interaction::Pointer)
            .boxed()
    } else {
        widgets::tag(tag.text.clone(), tone(tag.kind))
    };
    if !tag.step {
        return shown;
    }
    row![arrow(), shown]
        .spacing(space::XS)
        .align_y(Alignment::Center)
        .boxed()
}

/// The arrow before a step of the way.
fn arrow<'a>() -> Element<'a, Message> {
    text("→")
        .size(size::CAPTION)
        .font(look::font::body())
        .color(look::TEXT_FAINT)
        .boxed()
}

/// The metronome's colour now: lit on each beat (in the accent on a
/// bar's first), fading back to dim.
fn pulse(state: &State, play: PlayState) -> Color {
    let dim = look::TEXT_FAINT;
    if play != PlayState::Playing {
        return dim;
    }
    let heard = state.heard();
    match info::pulse(state, heard) {
        Some(beat) => {
            let lit = if beat.downbeat {
                look::ACCENT
            } else {
                look::TEXT
            };
            mix(dim, lit, beat.lit)
        }
        None => dim,
    }
}

/// The summary under the list for `state`: a click on it opens Now
/// playing.
pub fn summary<'a>(state: &State) -> Element<'a, Message> {
    let info = Info::of(state);
    let content: Element<'a, Message> = match info.track {
        None => text(NOTHING)
            .size(size::BODY)
            .font(look::font::body())
            .color(look::TEXT_DIM)
            .boxed(),
        Some(track) => {
            let icon = match track.play {
                PlayState::Playing => Icon::Play,
                PlayState::Paused => Icon::Pause,
                PlayState::Stopped => Icon::Stop,
            };
            let mut first = track.title;
            if let Some(artist) = track.artist {
                first.push_str(" — ");
                first.push_str(&artist);
            }
            let first = text(first)
                .size(size::BODY)
                .font(look::font::label())
                .color(look::TEXT)
                .wrapping(Wrapping::None)
                .ellipsis(Ellipsis::End);
            // The status light apart, the way as the signal path says it;
            // what does not fit is cut at the edge.
            let mut second = Vec::new();
            for (i, tag) in track.tags.iter().enumerate() {
                let after = i.checked_sub(1).map(|i| track.tags[i].kind);
                let gap = match after {
                    _ if tag.step => " → ",
                    None => "",
                    Some(TagKind::Status) => "  ",
                    Some(_) => ", ",
                };
                second.push(span::<()>(gap).color(look::TEXT_FAINT));
                let color = match tag.kind {
                    TagKind::Device => look::TEXT,
                    kind => tone(kind),
                };
                second.push(span(tag.text.clone()).color(color));
            }
            let second = rich_text(second)
                .size(size::CAPTION)
                .font(look::font::body())
                .wrapping(Wrapping::None);
            // The second line under the title, past the play state's icon.
            column![
                row![
                    widgets::icon(icon).size(SUMMARY_ICON).color(look::ACCENT),
                    first
                ]
                .spacing(space::S)
                .align_y(Alignment::Center),
                container(second).padding(Padding {
                    left: SUMMARY_ICON + space::S,
                    ..Padding::ZERO
                }),
            ]
            .boxed()
        }
    };
    mouse_area(container(content).center_y(Fill).width(Fill))
        .on_press(Message::Act(Action::Op(Op::NowPlaying)))
        .on_scroll(Message::Wheel)
        .interaction(mouse::Interaction::Pointer)
        .boxed()
}
