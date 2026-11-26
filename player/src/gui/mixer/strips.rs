//! Each strip of the Mixer tab as widgets, in the look of
//! [`crate::gui::look`]: a card ([`look::SURFACE`], raised and edged in the
//! accent while selected) with its name on a pill (lit in the accent while
//! selected), the lines under it (what it carries, [`super::said`]), its
//! tall section ([`super::fader`]), its level and its buttons (the shared
//! [`crate::gui::widgets::button`]s, small, lit while on); the bracket over
//! the stems.

use iced::widget::text::Ellipsis;
use iced::widget::{column, container, row, space, text, Column, Row};
use iced::{Alignment, Background, Border, Color, Element, Fill, Length, Widget as _};

use super::fader::{self, Fader, MeterLook, Scale};
use super::said::{self, Line, Piece, Push, Said, Tone};
use crate::core::desk::{Channel, DeskAction, Pair, StripKind};
use crate::core::{Action, State};
use crate::gui::look::{self, height, radius, size};
use crate::gui::widgets::button;
use crate::gui::window::Message;

/// A line's height under a strip's name.
const LINE: f32 = 16.0;
/// The lines every strip's tall section sits under where there is room
/// (the microphone has the most), so the sections line up as on a desk.
const LINES: usize = 6;
/// What a strip's other parts take of its height at least (the
/// microphone's, which has the most, with a short tall section).
const OTHER_PARTS: f32 = 220.0;
/// The mute and solo buttons' width together, at most.
const PUSHES: f32 = 64.0;
/// The room inside a strip's card, across and down.
const PAD_X: f32 = 3.0;
const PAD_Y: f32 = 8.0;
/// The room between a strip's parts.
const PARTS_GAP: f32 = 6.0;
/// The name's pill and the level's readout: their height.
const PILL: f32 = 22.0;
/// A lamp's size.
const LAMP: f32 = 6.0;

/// The look's colour for a tone.
fn color(tone: Tone) -> Color {
    match tone {
        Tone::Text => look::TEXT,
        Tone::Dim => look::TEXT_DIM,
        Tone::Faint => look::TEXT_FAINT,
        Tone::Accent => look::ACCENT,
        Tone::Good => look::GOOD,
        Tone::Warn => look::WARN,
        Tone::Fail => look::FAIL,
        Tone::Source => look::SOURCE,
    }
}

/// A rounded fill: a lamp, a bar, a pill.
fn fill(color: Color, corner: f32) -> container::Style {
    container::Style {
        background: Some(Background::Color(color)),
        border: Border {
            radius: corner.into(),
            ..Border::default()
        },
        ..container::Style::default()
    }
}

/// The bracket over the stems: `── ● STEMS heard ──`.
pub(super) fn bracket(state: &State, width: f32) -> Element<'static, Message> {
    let (lit, said) = said::stems_said(state);
    let rule = || {
        container(space().height(1))
            .width(Fill)
            .style(|_| fill(look::EDGE, 0.0))
    };
    container(
        row![
            rule(),
            lamp(lit, Tone::Good),
            text("STEMS")
                .size(size::CAPTION)
                .font(look::font::label())
                .color(look::TEXT_DIM),
            text(said)
                .size(size::CAPTION)
                .font(look::font::body())
                .color(if lit { look::GOOD } else { look::TEXT_FAINT }),
            rule(),
        ]
        .spacing(look::space::XS + 2.0)
        .align_y(Alignment::Center),
    )
    .width(width)
    .height(super::layout::BRACKET)
    .center_y(super::layout::BRACKET)
    .boxed()
}

fn lamp(lit: bool, tone: Tone) -> Element<'static, Message> {
    let c = if lit { color(tone) } else { look::EDGE };
    container(space().width(LAMP).height(LAMP))
        .style(move |_| fill(c, radius::PILL))
        .boxed()
}

fn line_view(line: Line) -> Element<'static, Message> {
    let mut r = Row::new()
        .spacing(look::space::XS)
        .align_y(Alignment::Center)
        .height(LINE);
    for piece in line {
        let e: Element<'static, Message> = match piece {
            Piece::Text(s, c) => text(s)
                .size(size::CAPTION)
                .font(look::font::body())
                .color(color(c))
                .wrapping(text::Wrapping::None)
                .boxed(),
            Piece::Digits(s, c) => text(s)
                .size(size::CAPTION)
                .font(look::font::numbers())
                .color(color(c))
                .wrapping(text::Wrapping::None)
                .boxed(),
            Piece::Lamp(lit, c) => lamp(lit, c),
            Piece::Bar(share, w, c) => {
                let lit = (w * share.clamp(0.0, 1.0)).round();
                let tone = color(c);
                container(
                    row![
                        container(space().width(lit).height(height::TRACK))
                            .style(move |_| fill(tone, radius::PILL)),
                        space().width(w - lit),
                    ]
                    .height(height::TRACK),
                )
                .style(|_| fill(look::EDGE, radius::PILL))
                .boxed()
            }
        };
        r = r.push(e);
    }
    r.boxed()
}

/// The room for the lines under a name on a strip `h` points high: as
/// many whole lines as it has room for, up to [`LINES`].
pub(super) fn info_height(h: f32) -> f32 {
    ((h - OTHER_PARTS) / LINE).floor().clamp(0.0, LINES as f32) * LINE
}

/// The lines under a name that fit `h` points, `h` high whatever their
/// number.
fn info(lines: Vec<Line>, h: f32) -> Element<'static, Message> {
    let count = (h / LINE).floor().max(0.0) as usize;
    let mut c = column![].spacing(0);
    for l in lines.into_iter().take(count) {
        c = c.push(line_view(l));
    }
    container(c.clip(true)).height(h).width(Fill).boxed()
}

/// The name on its pill: lit in the accent while the strip is selected
/// (a press anywhere on the strip selects it); a stem's a size smaller, on
/// its narrow strip.
fn name(said: &Said) -> Element<'static, Message> {
    let lit = said.selected;
    container(
        text(said.name.clone())
            .size(if said.channel.is_part() {
                size::CAPTION
            } else {
                size::LABEL
            })
            .font(look::font::label())
            .color(if lit { look::ACCENT_INK } else { look::TEXT })
            .wrapping(text::Wrapping::None)
            .ellipsis(Ellipsis::End),
    )
    .center_x(Fill)
    .center_y(PILL)
    .style(move |_| {
        fill(
            if lit { look::ACCENT } else { look::RAISED },
            radius::CONTROL,
        )
    })
    .boxed()
}

/// A push button: small, lit in its tone while on.
fn push(p: Push) -> button::Button<'static, Message> {
    button::secondary(p.label)
        .small()
        .lit(p.on.then_some(color(p.tone)))
        .on_press(Message::Act(Action::Desk(DeskAction::Press(p.button))))
}

/// A letter's width in the numbers' face at the readout's size, about
/// (its figures are tabular).
const FIGURE: f32 = size::CAPTION * 0.55;

/// A readout on one line, its height the same whatever it says: what it
/// says goes on a strip `w` wide whole, without its unit (`" dB"`) where
/// it would not fit.
fn readout((said, tone): (String, Tone), w: f32) -> Element<'static, Message> {
    let room = w - 2.0 * PAD_X - 2.0 * look::space::XS;
    let fits = |s: &str| s.chars().count() as f32 * FIGURE <= room;
    let said = match said.strip_suffix(" dB") {
        Some(bare) if !fits(&said) => bare.to_owned(),
        _ => said,
    };
    container(
        text(said)
            .size(size::CAPTION)
            .font(look::font::numbers())
            .color(color(tone))
            .wrapping(text::Wrapping::None)
            .ellipsis(Ellipsis::End),
    )
    .center_x(Fill)
    .center_y(PILL)
    .style(|_| fill(look::BG, radius::CONTROL))
    .boxed()
}

fn looks(pair: &Pair, now_ms: u64) -> [MeterLook; 2] {
    pair.each_ref().map(|m| MeterLook {
        db: m.db,
        peak_db: m.peak_db,
        clipped: m.clipped(now_ms),
    })
}

/// A strip's card: raised, edged in the accent, while selected.
fn card(selected: bool) -> container::Style {
    container::Style {
        background: Some(Background::Color(if selected {
            look::RAISED
        } else {
            look::SURFACE
        })),
        border: Border {
            color: if selected {
                look::tint(look::ACCENT, look::RAISED, 0.55)
            } else {
                Color::TRANSPARENT
            },
            width: if selected { look::line::EDGE } else { 0.0 },
            radius: radius::PANEL.into(),
        },
        ..container::Style::default()
    }
}

/// channel's strip, w wide, `lines` points of lines under its name.
pub(super) fn strip<'a>(
    state: &'a State,
    caps: &'a super::Caps,
    channel: Channel,
    w: f32,
    lines: f32,
) -> Element<'a, Message> {
    let said = said::said(state, channel);
    let selected = said.selected;
    let body: Column<Element<'a, Message>> = match channel {
        Channel::Strip(kind) => fader_strip(state, caps, kind, said, (w, lines)),
        Channel::Mic => mic_strip(state, said, (w, lines)),
        Channel::Main => master(state, said, lines),
    };
    container(body.spacing(PARTS_GAP))
        .padding([PAD_Y, PAD_X])
        .width(w)
        .height(Fill)
        .style(move |_| card(selected))
        .boxed()
}

fn fader_strip<'a>(
    state: &'a State,
    caps: &'a super::Caps,
    kind: StripKind,
    said: Said,
    (w, lines): (f32, f32),
) -> Column<Element<'a, Message>> {
    // A stem's strip is narrow: no scale of its own (the player's, beside
    // it, reads it).
    let section = fader::view(Fader {
        channel: Channel::Strip(kind),
        cap: Some(caps.of(kind)),
        selected: said.selected,
        meters: looks(state.desk.meters(kind), state.now_ms),
        scale: Scale::Taper,
        labels: !kind.is_part(),
        wide: false,
    });
    let mut pushes: Row<Element<'a, Message>> = Row::new().spacing(look::space::XS);
    for p in said.pushes.iter().copied() {
        pushes = pushes.push(push(p).width(Fill).into());
    }
    column![
        name(&said),
        info(said.lines, lines),
        section,
        said.level
            .map_or_else(|| space().boxed(), |l| readout(l, w)),
        container(pushes.width((w - 2.0 * PAD_X).clamp(0.0, PUSHES)))
            .center_x(Fill)
            .boxed(),
    ]
}

fn mic_strip(state: &State, said: Said, (w, lines): (f32, f32)) -> Column<Element<'_, Message>> {
    let r = &state.desk.readings;
    let meters = [&r.mic_in, &r.mic_voice].map(|m| MeterLook {
        db: m.db,
        peak_db: m.peak_db,
        clipped: m.clipped(state.now_ms),
    });
    let mut top = column![name(&said), info(said.lines, lines)].spacing(PARTS_GAP);
    for p in said.pushes.iter().copied() {
        top = top.push(container(Element::from(push(p))).center_x(Fill).boxed());
    }
    let caption = |s: &'static str| {
        text(s)
            .size(size::CAPTION)
            .font(look::font::label())
            .color(look::TEXT_FAINT)
    };
    let mut c = column![
        top,
        row![
            space().width(Fill),
            caption("IN"),
            caption("VOICE"),
            space().width(Fill)
        ]
        .spacing(look::space::M),
        fader::view(Fader {
            channel: Channel::Mic,
            cap: None,
            selected: said.selected,
            meters,
            scale: Scale::Dbfs60,
            labels: true,
            wide: false,
        }),
        said.level
            .map_or_else(|| space().boxed(), |l| readout(l, w)),
    ];
    for l in said.foot {
        c = c.push(container(line_view(l)).center_x(Fill).boxed());
    }
    c
}

fn master(state: &State, said: Said, lines: f32) -> Column<Element<'_, Message>> {
    let mut c = column![
        name(&said),
        info(said.lines, lines),
        fader::view(Fader {
            channel: Channel::Main,
            cap: None,
            selected: said.selected,
            meters: looks(&state.desk.readings.master, state.now_ms),
            scale: Scale::Dbfs60,
            labels: true,
            wide: true,
        }),
    ];
    for l in said.foot {
        c = c.push(container(line_view(l)).center_x(Fill).boxed());
    }
    c.width(Length::Fill)
}
