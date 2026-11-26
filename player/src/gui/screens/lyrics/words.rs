//! The lyrics editor's words on their lane: the time across the view over
//! it ([`ruler`]), each word a box over the voice's waveform where it is
//! sung ([`overlay`]: its text at its top, a pin where it was set by hand,
//! dimmed where it is not to be sung, ringed in the accent while selected,
//! its edges grips), and the voice's level over each under it
//! ([`levels`]).
//!
//! A press on a box holds its start, its middle or its end
//! ([`Grab::LyricWord`]); one on the waveform off the words puts the
//! playhead there; the secondary button drags the view
//! ([`Grab::LyricView`]); the wheel zooms round the time under the pointer.

use iced::mouse::{Interaction, ScrollDelta};
use iced::widget::{column, container, pin, row, text, Space, Stack};
use iced::{Background, Border, Color, Element, Fill, Point, Rectangle, Widget as _};

use crate::core::lyric_edit::screen::{Input, Part, LEVEL_LEAST, LEVEL_MOST};
use crate::core::lyric_edit::{LyricsEdit, WordAt};
use crate::core::Action;
use crate::gui::look::{self, size, space};
use crate::gui::pointer::{Grab, WheelOn};
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::lyrics::{Editor, WordBox};

/// A box's edges that take a press, in points (a third of a narrow box).
const GRIP: f32 = 7.0;
/// The grips drawn on the word selected.
const GRIP_MARK: f32 = 3.0;
/// The corners of a word's box.
const CORNER: f32 = 4.0;
/// A box's text, in from its edges.
const INSET: f32 = 6.0;

/// The name the tests find the lane by.
pub const LANE: &str = "lyrics-lane";

/// The name the tests find a word's part by: `lyric-0-3-end`.
pub fn part_id(at: WordAt, part: Part) -> String {
    let part = match part {
        Part::Start => "start",
        Part::Middle => "middle",
        Part::End => "end",
    };
    format!("lyric-{}-{}-{part}", at.line, at.word)
}

/// Where `t` seconds falls across `width` points of the view.
fn x_of(editor: &Editor, t: f64, width: f32) -> f32 {
    let (a, b) = editor.window;
    ((t - a) / (b - a).max(1e-9)) as f32 * width
}

/// `content` pinned at (`x`, `y`) in its stack, `w` × `h`.
fn at<'a>(content: Element<'a, Message>, x: f32, y: f32, w: f32, h: f32) -> Element<'a, Message> {
    pin(container(content).width(w).height(h)).x(x).y(y).boxed()
}

/// A bar of `color`.
fn bar<'a>(color: Color) -> Element<'a, Message> {
    container(Space::new())
        .width(Fill)
        .height(Fill)
        .style(move |_| container::Style {
            background: Some(Background::Color(color)),
            ..container::Style::default()
        })
        .boxed()
}

/// The time across the view over the lane, in `rect`: a mark at each
/// round step and the time beside it, the playhead's place in the accent.
pub(super) fn ruler<'a>(editor: &Editor, rect: Rectangle) -> Element<'a, Message> {
    let (w, h) = (rect.width, rect.height);
    let mut layers: Vec<Element<'a, Message>> = Vec::new();
    for (t, said) in &editor.ticks {
        let x = x_of(editor, *t, w).round();
        layers.push(at(bar(look::TEXT_FAINT), x, h - 6.0, 1.0, 6.0));
        layers.push(
            pin(text(said.clone())
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(look::TEXT_DIM)
                .wrapping(text::Wrapping::None))
            .x(x + space::XS)
            .y(1.0)
            .boxed(),
        );
    }
    if let Some(p) = editor.playhead.filter(|p| *p >= editor.window.0) {
        let x = x_of(editor, p, w);
        if x < w {
            layers.push(at(bar(look::ACCENT), x.floor(), 0.0, 2.0, h));
        }
    }
    Stack::with_children(layers)
        .width(w)
        .height(h)
        .clip(true)
        .boxed()
}

/// The voice's level over each word in view under the lane, in `rect`: a
/// bar a word, as high as the level (from [`LEVEL_LEAST`] to
/// [`LEVEL_MOST`]), lit where it is the word's own, in the accent while
/// selected; a hairline at the bottom.
pub(super) fn levels<'a>(editor: &Editor, rect: Rectangle) -> Element<'a, Message> {
    let (w, h) = (rect.width, rect.height);
    let mut layers: Vec<Element<'a, Message>> = vec![at(bar(look::EDGE), 0.0, h - 1.0, w, 1.0)];
    let span = f64::from(LEVEL_MOST - LEVEL_LEAST);
    for word in &editor.words {
        let x0 = x_of(editor, word.start, w).max(0.0);
        let x1 = x_of(editor, word.end, w).min(w);
        if x1 - x0 < 1.0 {
            continue;
        }
        let share = ((word.level - f64::from(LEVEL_LEAST)) / span).clamp(0.0, 1.0) as f32;
        let tall = (share * (h - 3.0)).max(2.0);
        let tone = match (word.selected, word.own) {
            (true, true) => look::ACCENT,
            (true, false) => look::TEXT_DIM,
            (false, true) => look::alpha(look::ACCENT, 0.5),
            (false, false) => look::alpha(look::TEXT_FAINT, 0.45),
        };
        layers.push(at(
            container(Space::new())
                .width(Fill)
                .height(Fill)
                .style(move |_| container::Style {
                    background: Some(Background::Color(tone)),
                    border: Border {
                        radius: iced::border::Radius::new(0.0).top(2.0),
                        ..Border::default()
                    },
                    ..container::Style::default()
                })
                .boxed(),
            x0 + 1.0,
            h - 1.0 - tall,
            (x1 - x0 - 2.0).max(1.0),
            tall,
        ));
    }
    Stack::with_children(layers)
        .width(w)
        .height(h)
        .clip(true)
        .boxed()
}

/// Over the lane (`lane`, in the window's points): what takes the pointer
/// on it, and each word in view as a box.
pub(super) fn overlay<'a>(editor: &Editor, lane: Rectangle) -> Element<'a, Message> {
    let window = editor.window;
    let ground = widgets::area()
        .id(LANE)
        .interaction(Interaction::Crosshair)
        .on_press(move |_, at| {
            let seek = Input::Seek(time_at(window, lane, at));
            Message::Act(Action::LyricsEdit(LyricsEdit::Screen(seek)))
        })
        .on_right_press(move |_, at| Message::Grab(Grab::LyricView(lane), at))
        .on_scroll(move |_, at, delta| zoom(window, lane, at, delta));
    let mut layers: Vec<Element<'a, Message>> =
        vec![at(ground.boxed(), lane.x, lane.y, lane.width, lane.height)];
    for word in &editor.words {
        let x0 = x_of(editor, word.start, lane.width);
        let x1 = x_of(editor, word.end, lane.width);
        // The part in view, a point in from its neighbours.
        let (from, to) = (x0.max(0.0) + 1.0, x1.min(lane.width) - 1.0);
        if to - from < 2.0 {
            continue;
        }
        let cut = (x0 < 0.0, x1 > lane.width);
        layers.push(at(
            word_box(word, to - from, (window, lane), cut),
            lane.x + from,
            lane.y + 1.0,
            to - from,
            lane.height - 2.0,
        ));
    }
    Stack::with_children(layers)
        .width(Fill)
        .height(Fill)
        .boxed()
}

/// The seconds into the song under `at` across `lane`, which shows
/// `window`.
fn time_at((start, end): (f64, f64), lane: Rectangle, at: Point) -> f64 {
    let share = f64::from(((at.x - lane.x) / lane.width.max(1.0)).clamp(0.0, 1.0));
    start + share * (end - start)
}

/// The wheel's `delta` at `at` over `lane` (showing `window`): the view
/// zoomed round the time there, in while it turns up.
fn zoom(window: (f64, f64), lane: Rectangle, at: Point, delta: ScrollDelta) -> Message {
    let time = time_at(window, lane, at);
    let step = |steps| Action::LyricsEdit(LyricsEdit::Screen(Input::Zoom { steps, at: time }));
    Message::Steps(WheelOn::Lyrics, delta, Box::new((step(1), step(-1))))
}

/// `w`'s box, `width` points wide over the lane (`lane`, showing
/// `window`), its start or end out of view (`cut`) taking no press; the
/// secondary button and the wheel over it as over the lane.
fn word_box<'a>(
    w: &WordBox,
    width: f32,
    (window, lane): ((f64, f64), Rectangle),
    (cut_start, cut_end): (bool, bool),
) -> Element<'a, Message> {
    let (fill, edge, ink, ring) = match (w.selected, w.held.is_some(), w.not_sung) {
        (_, true, _) => (
            look::alpha(look::ACCENT, 0.16),
            look::ACCENT,
            look::ACCENT,
            2.0,
        ),
        (true, false, true) => (
            look::alpha(look::ACCENT, 0.08),
            look::ACCENT,
            look::TEXT_DIM,
            2.0,
        ),
        (true, false, false) => (
            look::alpha(look::ACCENT, 0.08),
            look::ACCENT,
            look::TEXT,
            2.0,
        ),
        (false, _, true) => (
            look::alpha(look::TEXT_FAINT, 0.05),
            look::alpha(look::TEXT_FAINT, 0.45),
            look::TEXT_FAINT,
            1.0,
        ),
        (false, _, false) if w.pinned => (
            look::alpha(look::SOURCE, 0.12),
            look::alpha(look::SOURCE, 0.8),
            look::TEXT,
            1.0,
        ),
        (false, _, false) => (
            look::alpha(look::TEXT, 0.06),
            look::alpha(look::TEXT, 0.26),
            look::TEXT,
            1.0,
        ),
    };
    let ink = if w.heard && !w.not_sung {
        look::ACCENT
    } else {
        ink
    };
    let mut label = row![text(w.text.clone())
        .size(size::LABEL)
        .font(look::font::label())
        .color(ink)
        .wrapping(text::Wrapping::None)]
    .spacing(space::XS)
    .align_y(iced::Alignment::Center);
    if w.pinned {
        label = label.push(Element::from(widgets::icon(Icon::Pin).size(12.0).color(
            if w.selected {
                look::ACCENT
            } else {
                look::SOURCE
            },
        )));
    }
    let mut lines = column![label].spacing(2.0);
    if w.not_sung {
        lines = lines.push(
            text("not sung")
                .size(size::CAPTION)
                .font(look::font::body())
                .color(look::TEXT_FAINT)
                .wrapping(text::Wrapping::None)
                .boxed(),
        );
    }
    let corner = Border {
        color: edge,
        width: ring,
        radius: iced::border::Radius::new(0.0)
            .left(if cut_start { 0.0 } else { CORNER })
            .right(if cut_end { 0.0 } else { CORNER }),
    };
    let visual = container(lines)
        .padding([space::XS, INSET])
        .width(Fill)
        .height(Fill)
        .clip(true)
        .style(move |_| container::Style {
            background: Some(Background::Color(fill)),
            border: corner,
            ..container::Style::default()
        });
    let grip = (width / 3.0).min(GRIP);
    let handle = |part: Part| {
        let word = w.at;
        let interaction = match part {
            Part::Middle if w.held.is_some() => Interaction::Grabbing,
            Part::Middle => Interaction::Grab,
            Part::Start | Part::End => Interaction::ResizingHorizontally,
        };
        widgets::area()
            .id(part_id(word, part))
            .interaction(interaction)
            .on_press(move |_, at| Message::Grab(Grab::LyricWord { word, part, lane }, at))
            .on_right_press(move |_, at| Message::Grab(Grab::LyricView(lane), at))
            .on_scroll(move |_, at, delta| zoom(window, lane, at, delta))
            .boxed()
    };
    let side = |part: Part, cut: bool| -> Element<'a, Message> {
        if cut {
            Space::new().width(grip).height(Fill).boxed()
        } else {
            container(handle(part)).width(grip).height(Fill).boxed()
        }
    };
    let handles = row![
        side(Part::Start, cut_start),
        container(handle(Part::Middle)).width(Fill).height(Fill),
        side(Part::End, cut_end),
    ];
    let mut layers: Vec<Element<'a, Message>> = vec![visual.boxed()];
    // The selected word's edges drawn as grips.
    if w.selected {
        let mark = |show: bool| -> Element<'a, Message> {
            if show {
                container(bar(look::ACCENT))
                    .width(GRIP_MARK)
                    .height(Fill)
                    .padding([space::M, 0.0])
                    .boxed()
            } else {
                Space::new().width(GRIP_MARK).boxed()
            }
        };
        layers.push(
            row![mark(!cut_start), Space::new().width(Fill), mark(!cut_end)]
                .padding([0.0, 2.0])
                .width(Fill)
                .height(Fill)
                .boxed(),
        );
    }
    layers.push(handles.width(Fill).height(Fill).boxed());
    Stack::with_children(layers)
        .width(Fill)
        .height(Fill)
        .boxed()
}
