//! The lyrics editor's header ([`header`]: the song, whether its lyrics
//! are edited by hand, undo, redo, Revert, asked first, and Done), the
//! panel of the word selected ([`panel`]: its text and times, set by hand,
//! not sung, the voice's level over it, its line's words to pick, or the
//! text being written in a field) and the keys at the foot ([`footer`]).
//! Every button sends the core what its key does.

use iced::widget::{column, container, row, text, Row, Space};
use iced::{Alignment, Element, Fill, Padding, Widget as _};

use crate::core::keymap::{self, Context, Op};
use crate::core::lyric_edit::{Edit, LyricsEdit, Selection};
use crate::core::Action;
use crate::gui::look::{self, size, space};
use crate::gui::widgets::{self, button, Icon};
use crate::gui::window::Message;
use crate::view::lyrics::{Editor, ASK, DONE, REVERT};

/// The name the tests find the text field by.
pub const FIELD: &str = "lyrics-text";

/// What a press asks of the editor.
fn edit(what: LyricsEdit) -> Message {
    Message::Act(Action::LyricsEdit(what))
}

/// What a press does: `op`, as its key does.
fn op(op: Op) -> Message {
    Message::Act(Action::Op(op))
}

/// `op`'s key in the editor.
fn key(op: Op) -> String {
    keymap::hint(Context::LyricsEditor, op)
}

/// One line of text, cut where it is too long.
fn line<'a>(said: impl Into<String>, size: f32, tone: iced::Color) -> Element<'a, Message> {
    text(said.into())
        .size(size)
        .font(look::font::body())
        .color(tone)
        .wrapping(text::Wrapping::None)
        .ellipsis(text::Ellipsis::End)
        .boxed()
}

/// The header: the song (its title, its artist), whether its lyrics are
/// edited by hand, undo and redo, Revert (asked first) and Done.
pub(super) fn header<'a>(editor: &Editor) -> Element<'a, Message> {
    let title = column![
        text(editor.title.clone())
            .size(size::TITLE)
            .font(look::font::title())
            .color(look::TEXT)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::End),
        line(
            if editor.artist.is_empty() {
                "Lyrics".to_owned()
            } else {
                format!("Lyrics · {}", editor.artist)
            },
            size::LABEL,
            look::TEXT_DIM
        ),
    ]
    .spacing(2.0)
    .width(Fill);
    let was = if editor.edited {
        widgets::tag("edited by hand", look::ACCENT)
    } else {
        widgets::tag("as found", look::TEXT_DIM)
    };
    let undo = button::icon(Icon::Undo).id("lyrics-undo");
    let redo = button::icon(Icon::Redo).id("lyrics-redo");
    let undo = if editor.undo {
        undo.on_press(op(Op::LyricsUndo))
    } else {
        undo
    };
    let redo = if editor.redo {
        redo.on_press(op(Op::LyricsRedo))
    } else {
        redo
    };
    let mut bar = Row::<Element<'a, Message>>::new()
        .spacing(space::S)
        .align_y(Alignment::Center)
        .push(
            widgets::icon(Icon::Lyrics)
                .size(22.0)
                .color(look::TEXT_DIM)
                .into(),
        )
        .push(Space::new().width(space::XS).boxed())
        .push(title.boxed())
        .push(was)
        .push(Space::new().width(space::M).boxed())
        .push(undo.into())
        .push(redo.into())
        .push(Space::new().width(space::S).boxed());
    if editor.asking {
        bar = bar
            .push(line(ASK, size::LABEL, look::WARN))
            .push(
                widgets::primary(REVERT)
                    .on_press(edit(LyricsEdit::Revert))
                    .into(),
            )
            .push(
                widgets::ghost("Keep them")
                    .key(key(Op::Back))
                    .on_press(edit(LyricsEdit::AskRevert(false)))
                    .into(),
            );
    } else {
        bar = bar.push(
            widgets::secondary(REVERT)
                .leading(Icon::Retry)
                .on_press(edit(LyricsEdit::AskRevert(true)))
                .into(),
        );
    }
    bar.push(
        widgets::primary(DONE)
            .key(key(Op::Back))
            .on_press(edit(LyricsEdit::Close))
            .into(),
    )
    .height(Fill)
    .boxed()
}

/// A name in the panel's column of names.
fn name<'a>(said: &'static str) -> Element<'a, Message> {
    text(said)
        .size(size::LABEL)
        .font(look::font::label())
        .color(look::TEXT_DIM)
        .wrapping(text::Wrapping::None)
        .width(48.0)
        .boxed()
}

/// The panel of the word selected, on a card: its text, whether it was set
/// by hand and its times; not sung and the voice's level over it, with
/// their buttons; then its line's words (a press selects one) and the
/// buttons that write the word or the line, or the text being written.
pub(super) fn panel<'a>(editor: &Editor) -> Element<'a, Message> {
    let Some(s) = &editor.selected else {
        return widgets::card(line(
            "No words: the song's text has none to time.",
            size::BODY,
            look::TEXT_DIM,
        ))
        .padding(space::L)
        .width(Fill)
        .height(Fill)
        .boxed();
    };
    let mut first = Row::<Element<'a, Message>>::new()
        .spacing(space::M)
        .align_y(Alignment::Center)
        .push(
            text(s.text.clone())
                .size(size::TITLE)
                .font(look::font::title())
                .color(if s.not_sung {
                    look::TEXT_DIM
                } else {
                    look::TEXT
                })
                .wrapping(text::Wrapping::None)
                .boxed(),
        );
    if s.pinned {
        first = first.push(
            row![
                widgets::icon(Icon::Pin).size(14.0).color(look::SOURCE),
                text("set by hand")
                    .size(size::CAPTION)
                    .font(look::font::label())
                    .color(look::SOURCE)
            ]
            .spacing(space::XS)
            .align_y(Alignment::Center)
            .boxed(),
        );
    }
    first = first.push(Space::new().width(Fill).boxed()).push(
        text(s.times.clone())
            .size(size::BODY)
            .font(look::font::numbers())
            .color(look::TEXT_DIM)
            .wrapping(text::Wrapping::None)
            .boxed(),
    );
    let not_sung = widgets::chip("Not sung")
        .lit(s.not_sung.then_some(look::WARN))
        .struck(!s.not_sung)
        .key(key(Op::LyricsNotSung))
        .on_press(op(Op::LyricsNotSung));
    let small =
        |label: &'static str, o: Op| widgets::ghost(label).small().key(key(o)).on_press(op(o));
    let reset = widgets::ghost("As the slider")
        .small()
        .key(key(Op::LyricsLevelReset));
    let reset = if s.own_level {
        reset.on_press(op(Op::LyricsLevelReset))
    } else {
        reset
    };
    let second = row![
        Element::from(not_sung),
        Space::new().width(space::M),
        name("Voice"),
        Element::from(small("Quieter", Op::LyricsLevel(-1))),
        text(s.level.clone())
            .size(size::BODY)
            .font(look::font::numbers())
            .color(if s.own_level {
                look::ACCENT
            } else {
                look::TEXT
            })
            .wrapping(text::Wrapping::None),
        Element::from(small("Louder", Op::LyricsLevel(1))),
        Element::from(reset),
        Space::new().width(Fill),
        Element::from(
            widgets::ghost("Take out")
                .small()
                .key(key(Op::LyricsDelete))
                .on_press(op(Op::LyricsDelete))
        ),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center);
    let third: Element<'a, Message> = match &editor.draft {
        Some(draft) => {
            let what = if draft.line { "the line" } else { "the word" };
            row![
                name(if draft.line { "Line" } else { "Word" }),
                widgets::field(format!("{what}: none takes it out"), draft.text.clone())
                    .id(FIELD)
                    .on_input(|t| edit(LyricsEdit::Draft(t)))
                    .width(Fill),
                Element::from(
                    widgets::primary("Keep")
                        .key(keymap::hint(Context::LyricsText, Op::Accept))
                        .on_press(op(Op::Accept))
                ),
                Element::from(
                    widgets::ghost("Leave")
                        .key(keymap::hint(Context::LyricsText, Op::Back))
                        .on_press(edit(LyricsEdit::Discard))
                ),
            ]
            .spacing(space::S)
            .align_y(Alignment::Center)
            .boxed()
        }
        None => {
            let words = editor.line.iter().fold(
                Row::<Element<'a, Message>>::new().spacing(space::XS),
                |all, w| {
                    let at = w.at;
                    all.push(
                        widgets::chip(w.text.clone())
                            .lit(w.selected.then_some(look::ACCENT))
                            .struck(w.not_sung)
                            .on_press(edit(LyricsEdit::Edit(Edit::Select(Selection::Word(at)))))
                            .into(),
                    )
                },
            );
            row![
                name("Line"),
                container(words).width(Fill).clip(true),
                Element::from(
                    widgets::secondary("Write word")
                        .small()
                        .key(key(Op::LyricsWrite))
                        .on_press(op(Op::LyricsWrite))
                ),
                Element::from(
                    widgets::secondary("Write line")
                        .small()
                        .key(key(Op::LyricsWriteLine))
                        .on_press(op(Op::LyricsWriteLine))
                ),
            ]
            .spacing(space::S)
            .align_y(Alignment::Center)
            .boxed()
        }
    };
    widgets::card(column![first, second, third].spacing(space::M))
        .padding(Padding::from([space::M, space::L]))
        .width(Fill)
        .height(Fill)
        .boxed()
}

/// The keys at the foot, and what stands in for the voice's waveform at
/// the right while it is not the voice's.
pub(super) fn footer<'a>(editor: &Editor) -> Element<'a, Message> {
    let pairs: Vec<(String, &str)> = editor
        .keys
        .iter()
        .map(|h| (h.keys.clone(), h.what))
        .collect();
    let mut out = Row::<Element<'a, Message>>::new()
        .align_y(Alignment::Center)
        .push(widgets::hints(&pairs).boxed())
        .push(Space::new().width(Fill).boxed());
    if let Some(note) = editor.note {
        out = out.push(line(note, size::CAPTION, look::TEXT_FAINT));
    }
    out.height(Fill).boxed()
}
