//! The Karaoke tab ([`crate::view::karaoke::Tab`]) in iced's widgets, over
//! the pane the cell layout leaves for it (the tabs on its top edge are
//! the list pane's, [`super::list`]), on the list pane's panel: beside the
//! cover (the window's own widget over the cell layout's place,
//! [`crate::gui::widgets::cover`]; a card in its place without art), the
//! song with the button into the mode, then the controls (the sliders,
//! the sync's row under the delay; the game's switches) and what is known
//! (the lyrics of the song and the next two, the microphone and its
//! meter, the echo canceller, the word timing; a few notes on the mode),
//! side by side where there is room, the keys at the pane's foot. Under
//! the pane, where the voice's lane goes, what it says while it has
//! nothing to show ([`voice`]). While a sync shows, its panel over the
//! window ([`sync`]): only its buttons take a press.
//!
//! The buttons and the switches send the core's actions
//! ([`Message::Act`]; the button into the mode its second meaning with
//! shift held, [`Message::OpShifted`]); a press on a slider's track holds
//! the pointer ([`Grab::Slider`]): the slider follows it anywhere until it
//! is let go of. The wheel over a slider's row steps it
//! ([`Message::Steps`]).

use iced::widget::{
    column, container, mouse_area, opaque, pin, rich_text, row, span, stack, text, Column, Row,
    Space,
};
use iced::{
    Alignment, Background, Border, Color, Element, Fill, Length, Padding, Rectangle, Shadow,
    Vector, Widget as _,
};

use crate::core::karaoke::HEARD_WARNING;
use crate::core::keymap::Op;
use crate::core::Action;
use crate::gui::look::{self, radius, size, space};
use crate::gui::pointer::{Grab, WheelOn};
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;
use crate::view::karaoke::{
    track_id, Control, Hint, LyricsRow, Mic, SliderRow, Switch, Tab, ENTER, MEASURE,
};
use crate::view::sync::{Dot, Panel, Said, HEARS, SAY, TITLE};

/// The narrowest a column of the tab gets with another beside it.
const COLUMN: f32 = 280.0;
/// A control's row: a slider, the sync, a switch.
const ROW: f32 = 32.0;
/// The names' column of the controls and of what is known.
const NAME: f32 = 56.0;
/// A slider's value, right of its track: `−200 ms` in tabular figures.
const VALUE: f32 = 64.0;
/// A switch's width: the widest name with its key.
const SWITCH: f32 = 104.0;
/// The microphone's meter.
const METER: f32 = 96.0;
/// The line of keys at the pane's foot.
const KEYS: f32 = 28.0;
/// The sync's panel's widest.
const PANEL: f32 = 480.0;
/// A click's dot on the sync's panel.
const DOT: f32 = 14.0;

/// Where the tab is: the pane's size, a cell's, and where the cover is in
/// the pane (its tabs take the top row of cells, its panel's edges run
/// through the middle of the border cells, as the list's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// The pane's width.
    pub width: f32,
    /// The pane's height.
    pub height: f32,
    /// A cell's width and height.
    pub cell: (f32, f32),
    /// The cover's square, from the pane's top left.
    pub cover: Rectangle,
}

/// The tab at `room`, its spinners at `turn` of a turn.
pub fn view<'a>(tab: &Tab, turn: f32, room: Room) -> Element<'a, Message> {
    let (cw, ch) = room.cell;
    let ground = super::list::ground(super::list::Room {
        width: room.width,
        height: room.height,
        cell: room.cell,
    });
    let cover = room.cover;
    let (pad, top) = (cw / 2.0 + space::L, cover.y.max(ch + space::S));
    // Beside a large cover, the text right of it, level with its top; under
    // a small one (a narrow pane), the song beside it and the rest under
    // them, the pane's whole width.
    let beside = cover.height >= room.height / 2.0;
    let left = if beside && cover.width > 0.0 {
        cover.x + cover.width + space::XL
    } else {
        pad
    };
    let wide = room.width - left - pad >= 2.0 * COLUMN + space::XL;
    let controls = column![sliders(&tab.controls), switches(&tab.switches)].spacing(space::M);
    let known = known(tab, turn);
    let body: Element<'a, Message> = if wide {
        row![
            container(controls).width(Fill),
            container(known).width(Fill)
        ]
        .spacing(space::XL)
        .boxed()
    } else {
        column![controls, known].spacing(space::M).boxed()
    };
    let head: Element<'a, Message> = if beside || cover.width <= 0.0 {
        head(tab)
    } else {
        row![
            Space::new()
                .width(cover.x + cover.width + space::XL - pad)
                .height(cover.height),
            head(tab)
        ]
        .boxed()
    };
    let content = container(column![head, body].spacing(space::L))
        .padding(Padding {
            top,
            right: pad,
            bottom: ch / 2.0 + KEYS + space::S,
            left,
        })
        .width(Fill)
        .height(Fill)
        .clip(true);
    let foot = container(hints(&tab.keys))
        .padding(Padding {
            left: cw / 2.0 + space::L,
            bottom: ch / 2.0 + space::XS,
            ..Padding::ZERO
        })
        .width(Fill)
        .height(Fill)
        .align_y(Alignment::End);
    let mut layers = vec![ground];
    if cover.width > 0.0 {
        layers.push(
            pin(placeholder(cover.width, cover.height))
                .x(cover.x)
                .y(cover.y)
                .boxed(),
        );
    }
    layers.push(content.boxed());
    layers.push(foot.boxed());
    iced::widget::Stack::with_children(layers)
        .width(room.width)
        .height(room.height)
        .boxed()
}

/// What stands in the cover's place without art (the cover drawn over it
/// otherwise): a card with a note on it.
fn placeholder<'a>(width: f32, height: f32) -> Element<'a, Message> {
    container(
        widgets::icon(Icon::Note)
            .size((width.min(height) / 5.0).clamp(16.0, 64.0))
            .color(look::TEXT_FAINT),
    )
    .center_x(width)
    .center_y(height)
    .style(|_| container::Style {
        background: Some(Background::Color(look::RAISED)),
        border: Border {
            radius: radius::PANEL.into(),
            ..Border::default()
        },
        ..container::Style::default()
    })
    .boxed()
}

/// The song: its title, its artist (and key), and the button into the
/// mode, greyed with why while the song cannot be sung (a press still
/// says why on the status line).
fn head<'a>(tab: &Tab) -> Element<'a, Message> {
    let press = Message::OpShifted(tab.enter.op, tab.enter.shifted);
    let enter = match tab.enter.why {
        None => widgets::primary(ENTER),
        Some(_) => widgets::secondary(ENTER),
    }
    .leading(Icon::Play)
    .key("enter")
    .on_press(press);
    let mut button = Row::<Element<'a, Message>>::new()
        .spacing(space::M)
        .align_y(Alignment::Center)
        .push(enter.into());
    if let Some(why) = tab.enter.why {
        button = button.push(dim(why, look::TEXT_DIM));
    }
    let mut lines = Column::<Element<'a, Message>>::new()
        .spacing(space::XS)
        .push(
            text(tab.title.clone())
                .size(size::TITLE)
                .font(look::font::title())
                .color(look::TEXT)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End)
                .boxed(),
        );
    if !tab.artist.is_empty() {
        lines = lines.push(
            text(tab.artist.clone())
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT_DIM)
                .wrapping(text::Wrapping::None)
                .ellipsis(text::Ellipsis::End)
                .boxed(),
        );
    }
    column![lines, button].spacing(space::M).boxed()
}

/// A name in the names' column, `tone`d.
fn name<'a>(label: &'static str, tone: Color) -> Element<'a, Message> {
    text(label)
        .size(size::LABEL)
        .font(look::font::label())
        .color(tone)
        .wrapping(text::Wrapping::None)
        .width(NAME)
        .boxed()
}

/// A dim line of text that wraps.
fn dim<'a>(said: impl Into<String>, tone: Color) -> Element<'a, Message> {
    text(said.into())
        .size(size::LABEL)
        .font(look::font::body())
        .color(tone)
        .width(Fill)
        .boxed()
}

/// The sliders on a card, the sync's row under the delay.
fn sliders<'a>(controls: &[Control]) -> Element<'a, Message> {
    let rows = controls.iter().map(|c| match c {
        Control::Slider(s) => slider(s),
        Control::Sync(from) => sync_row(from),
    });
    widgets::card(Column::with_children(rows))
        .padding(Padding::from([space::S, space::L]))
        .width(Fill)
        .boxed()
}

/// A slider's row: its name (in the accent while the arrow keys move it),
/// its track, its value; the wheel over the row steps it.
fn slider<'a>(s: &SliderRow) -> Element<'a, Message> {
    let tone = match (s.focused, s.dim) {
        (true, _) => look::ACCENT,
        (false, true) => look::TEXT_FAINT,
        (false, false) => look::TEXT_DIM,
    };
    let slider = s.slider;
    let track = widgets::track(s.share)
        .held(s.held)
        .id(track_id(slider))
        .on_press(move |bounds, at| Message::Grab(Grab::Slider(slider, bounds), at));
    let value = text(s.value.clone())
        .size(size::BODY)
        .font(look::font::numbers())
        .color(if s.dim { look::TEXT_FAINT } else { look::TEXT })
        .wrapping(text::Wrapping::None)
        .width(VALUE)
        .align_x(Alignment::End);
    let line = row![
        name(slider.label(), tone),
        container(track).width(Fill),
        value
    ]
    .spacing(space::M)
    .height(ROW)
    .align_y(Alignment::Center);
    mouse_area(line)
        .on_scroll(move |delta| {
            let step = |by| Action::KaraokeStep { slider, by };
            Message::Steps(
                WheelOn::Slider(slider),
                delta,
                Box::new((step(1), step(-1))),
            )
        })
        .boxed()
}

/// The sync's row: its button, and where the delay in force comes from.
fn sync_row<'a>(from: &str) -> Element<'a, Message> {
    row![
        name("Sync", look::TEXT_DIM),
        Element::from(
            widgets::secondary(MEASURE)
                .small()
                .key("s")
                .on_press(Message::Act(Action::Op(Op::KaraokeSync)))
        ),
        text(from.to_owned())
            .size(size::LABEL)
            .font(look::font::body())
            .color(look::TEXT_DIM)
            .wrapping(text::Wrapping::None)
            .ellipsis(text::Ellipsis::End)
            .width(Fill),
    ]
    .spacing(space::M)
    .height(ROW)
    .align_y(Alignment::Center)
    .boxed()
}

/// The game's switches on a card: each lit while on, struck through while
/// off, what it means beside it.
fn switches<'a>(switches: &[Switch]) -> Element<'a, Message> {
    let rows = switches.iter().map(|s| {
        let mut chip = widgets::chip(s.label)
            .lit(s.on.then_some(look::ACCENT))
            .struck(!s.on)
            .width(SWITCH)
            .on_press(Message::Act(Action::Op(s.op)));
        if let Some(key) = s.key {
            chip = chip.key(key);
        }
        row![
            Element::from(chip),
            text(s.what)
                .size(size::LABEL)
                .font(look::font::body())
                .color(if s.on {
                    look::TEXT_DIM
                } else {
                    look::TEXT_FAINT
                })
                .width(Fill),
        ]
        .spacing(space::M)
        .padding(Padding::from([space::XS, 0.0]))
        .align_y(Alignment::Center)
        .boxed()
    });
    widgets::card(Column::with_children(rows))
        .padding(Padding::from([space::S, space::L]))
        .width(Fill)
        .boxed()
}

/// A busy arc at `turn` of a turn.
fn busy<'a>(turn: f32) -> Element<'a, Message> {
    widgets::icon(Icon::Busy)
        .size(14.0)
        .color(look::ACCENT)
        .turn(turn)
        .into()
}

/// What is known, on a card: the lyrics of the song and the next two, the
/// microphone, the echo canceller, the word timing; the notes under it.
fn known<'a>(tab: &Tab, turn: f32) -> Element<'a, Message> {
    let mut rows = Column::<Element<'a, Message>>::new().spacing(space::XS + 2.0);
    for l in &tab.lyrics {
        rows = rows.push(lyrics_row(l, turn));
        if l.heard && l.playing {
            rows = rows.push(
                row![
                    Space::new().width(NAME),
                    widgets::icon(Icon::Alert).size(14.0).color(look::WARN),
                    dim(HEARD_WARNING, look::WARN)
                ]
                .spacing(space::S)
                .boxed(),
            );
        }
    }
    if !tab.lyrics.is_empty() {
        rows = rows.push(Space::new().height(space::XS).boxed());
    }
    rows = rows.push(mic(&tab.mic, turn));
    if let Some(echo) = &tab.echo {
        rows = rows.push(
            row![
                name("Echo", look::TEXT_DIM),
                dim(echo.clone(), look::TEXT_DIM)
            ]
            .spacing(space::S)
            .boxed(),
        );
    }
    rows = rows.push(
        row![
            name("Words", look::TEXT_DIM),
            dim(tab.words, look::TEXT_DIM)
        ]
        .spacing(space::S)
        .boxed(),
    );
    let card = widgets::card(rows)
        .padding(Padding::from([space::M, space::L]))
        .width(Fill);
    let notes = Column::with_children(tab.notes.iter().map(|n| {
        text(*n)
            .size(size::CAPTION)
            .font(look::font::body())
            .color(look::TEXT_FAINT)
            .width(Fill)
            .boxed()
    }))
    .spacing(2.0)
    .padding(Padding::from([0.0, space::XS]));
    column![card, notes].spacing(space::M).boxed()
}

/// A song's lyrics: the song playing first ("Lyrics"), then the next two.
fn lyrics_row<'a>(l: &LyricsRow, turn: f32) -> Element<'a, Message> {
    let lead: Element<'a, Message> = if l.looking {
        busy(turn)
    } else {
        widgets::icon(Icon::Lyrics)
            .size(14.0)
            .color(if l.what.is_empty() {
                look::TEXT_FAINT
            } else if l.heard {
                look::WARN
            } else {
                look::TEXT_DIM
            })
            .into()
    };
    let what_tone = if l.heard { look::WARN } else { look::TEXT_DIM };
    let mut spans = vec![span::<()>(l.name.clone())
        .color(look::TEXT)
        .font(if l.playing {
            look::font::label()
        } else {
            look::font::body()
        })];
    if !l.what.is_empty() {
        spans.push(span(format!("  {}", l.what)).color(what_tone));
    }
    row![
        name(if l.playing { "Lyrics" } else { "Next" }, look::TEXT_DIM),
        lead,
        rich_text(spans)
            .size(size::LABEL)
            .font(look::font::body())
            .width(Fill)
    ]
    .spacing(space::S)
    .align_y(Alignment::Center)
    .boxed()
}

/// `keys` as key caps and what each does, in a line.
fn hints<'a>(keys: &[Hint]) -> Element<'a, Message> {
    let pairs: Vec<(&str, &str)> = keys.iter().map(|h| (h.keys, h.what)).collect();
    widgets::hints(&pairs).height(KEYS).boxed()
}

/// The microphone: its meter and name, opening, silent or not open.
fn mic<'a>(m: &Mic, turn: f32) -> Element<'a, Message> {
    let mut line = Row::<Element<'a, Message>>::new()
        .spacing(space::S)
        .align_y(Alignment::Center)
        .push(name("Mic", look::TEXT_DIM));
    let mut under = None;
    match m {
        Mic::Opening => {
            line = line.push(busy(turn)).push(dim("opening…", look::TEXT_DIM));
        }
        Mic::Open { name, level } => {
            line = line
                .push(
                    widgets::icon(Icon::Mic)
                        .size(14.0)
                        .color(look::TEXT_DIM)
                        .into(),
                )
                .push(
                    container(widgets::progress(*level, look::ACCENT))
                        .width(METER)
                        .center_y(14.0)
                        .boxed(),
                )
                .push(
                    text(name.clone())
                        .size(size::LABEL)
                        .font(look::font::body())
                        .color(look::TEXT)
                        .wrapping(text::Wrapping::None)
                        .ellipsis(text::Ellipsis::End)
                        .boxed(),
                );
        }
        Mic::Silent { name, hint } => {
            line = line
                .push(
                    widgets::icon(Icon::Alert)
                        .size(14.0)
                        .color(look::FAIL)
                        .into(),
                )
                .push(dim(format!("{name}: only silence comes in"), look::FAIL));
            under = Some(*hint);
        }
        Mic::Failed(why) => {
            line = line
                .push(
                    widgets::icon(Icon::Alert)
                        .size(14.0)
                        .color(look::FAIL)
                        .into(),
                )
                .push(
                    text(format!("not open: {why}"))
                        .size(size::LABEL)
                        .font(look::font::body())
                        .color(look::FAIL)
                        .wrapping(text::Wrapping::None)
                        .ellipsis(text::Ellipsis::End)
                        .boxed(),
                );
        }
    }
    if let Some(key) = m.key() {
        line = line
            .push(Space::new().width(Fill).boxed())
            .push(hints(&[key]));
    }
    let mut out = Column::<Element<'a, Message>>::new()
        .spacing(space::XS)
        .push(line.boxed());
    if let Some(hint) = under {
        out = out.push(
            row![
                Space::new().width(NAME + space::S),
                dim(hint, look::TEXT_DIM)
            ]
            .boxed(),
        );
    }
    out.boxed()
}

/// What the voice's lane says while it has nothing to show, over its
/// place (`width` × `height` points): on a card, as the lane's.
pub fn voice<'a>(note: &'static str, width: f32, height: f32) -> Element<'a, Message> {
    container(
        row![
            widgets::icon(Icon::Mic).size(16.0).color(look::TEXT_FAINT),
            text(note)
                .size(size::BODY)
                .font(look::font::body())
                .color(look::TEXT_DIM)
        ]
        .spacing(space::S)
        .align_y(Alignment::Center),
    )
    .center_x(width)
    .center_y(height)
    .style(|_| container::Style {
        background: Some(Background::Color(look::SURFACE)),
        border: Border {
            radius: radius::PANEL.into(),
            ..Border::default()
        },
        ..container::Style::default()
    })
    .boxed()
}

/// The sync's panel over the window: the window dimmed under it, nothing
/// there taking a press; on a card in the middle what to do, the clicks
/// as dots (played, answered), how many were heard, how it ended, and its
/// buttons.
pub fn sync<'a>(panel: &Panel) -> Element<'a, Message> {
    let shade = container(Space::new().width(Fill).height(Fill))
        .width(Fill)
        .height(Fill)
        .style(|_| container::Style {
            background: Some(Background::Color(look::alpha(Color::BLACK, 0.5))),
            ..container::Style::default()
        });
    let mut dots = Row::<Element<'a, Message>>::new()
        .spacing(space::S)
        .align_y(Alignment::Center);
    for dot in &panel.dots {
        let (fill, edge) = match dot {
            Dot::Coming => (Color::TRANSPARENT, look::TEXT_FAINT),
            Dot::Played => (look::TEXT_DIM, Color::TRANSPARENT),
            Dot::Answered => (look::ACCENT, Color::TRANSPARENT),
        };
        dots = dots.push(widgets::badge(Space::new(), DOT, fill, edge).boxed());
    }
    dots = dots.push(Space::new().width(space::S).boxed()).push(
        text(panel.heard.clone())
            .size(size::LABEL)
            .font(look::font::numbers())
            .color(look::TEXT_DIM)
            .boxed(),
    );
    let said: Element<'a, Message> = match &panel.said {
        None => Space::new().height(size::TITLE + space::XS).boxed(),
        Some(Said::Measured(ms)) => rich_text![
            span::<()>("Your delay: ").color(look::TEXT),
            span(format!("{ms} ms"))
                .color(look::ACCENT)
                .font(look::font::title()),
        ]
        .size(size::TITLE)
        .font(look::font::label())
        .boxed(),
        Some(Said::TooFew(t)) => dim(t.clone(), look::WARN),
        Some(Said::Failed(t)) => dim(t.clone(), look::FAIL),
    };
    let buttons = Row::<Element<'a, Message>>::with_children(panel.buttons.iter().map(|b| {
        let button = if b.primary {
            widgets::primary(b.label)
        } else {
            widgets::secondary(b.label)
        };
        button
            .key(b.key)
            .id(b.id())
            .on_press(Message::Act(Action::Op(b.op)))
            .into()
    }))
    .spacing(space::S);
    let card = container(
        column![
            text(TITLE)
                .size(size::TITLE)
                .font(look::font::title())
                .color(look::TEXT),
            column![
                text(SAY)
                    .size(size::BODY)
                    .font(look::font::label())
                    .color(look::TEXT),
                text(HEARS)
                    .size(size::LABEL)
                    .font(look::font::body())
                    .color(look::TEXT_DIM),
            ]
            .spacing(space::XS),
            dots,
            said,
            buttons,
        ]
        .spacing(space::L),
    )
    .padding(space::XL)
    .width(PANEL)
    .style(|_| container::Style {
        background: Some(Background::Color(look::RAISED)),
        border: Border {
            color: look::EDGE,
            width: look::line::EDGE,
            radius: radius::PANEL.into(),
        },
        shadow: Shadow {
            color: look::alpha(Color::BLACK, 0.55),
            offset: Vector::new(0.0, 16.0),
            blur_radius: 48.0,
        },
        ..container::Style::default()
    });
    stack![
        opaque(shade),
        container(opaque(card)).center(Fill).padding(space::XL)
    ]
    .width(Length::Fill)
    .height(Length::Fill)
    .boxed()
}
