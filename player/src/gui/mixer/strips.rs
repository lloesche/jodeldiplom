//! Each strip of the Mixer tab as widgets: its scribble strip, the lines
//! under it (what it carries, [`super::said`]), its tall section
//! ([`super::fader`]), its level and its buttons; the bracket over the
//! stems.

use iced::widget::{button, column, container, row, space, text, Row};
use iced::{Alignment, Color, Element, Fill, Length, Widget as _};

use super::fader::{self, Fader, MeterLook, Scale};
use super::said::{self, Line, Piece};
use crate::core::desk::{self, taper, Channel, DeskAction, DeskButton, Pair, StripKind};
use crate::core::State;
use crate::gui::style::{self, ACCENT, DIM, FAIL, GREEN, TEXT, WARN};
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
const PUSHES: f32 = 68.0;

/// The bracket over the stems: `── ● STEMS heard ──`.
pub(super) fn bracket(state: &State, width: f32) -> Element<'static, Message> {
    let (lit, said) = said::stems_said(state);
    let rule = || {
        container(space().height(1))
            .width(Fill)
            .style(|_| style::bar(style::BORDER))
    };
    container(
        row![
            rule(),
            lamp(lit, GREEN),
            text("STEMS").size(12).color(TEXT),
            text(said).size(12).color(if lit { GREEN } else { DIM }),
            rule(),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
    )
    .width(width)
    .height(22)
    .center_y(22)
    .boxed()
}

fn lamp(lit: bool, color: Color) -> Element<'static, Message> {
    container(space().width(8).height(8))
        .style(move |_| style::lamp(lit, color))
        .boxed()
}

fn line_view(line: Line) -> Element<'static, Message> {
    let mut r = Row::new()
        .spacing(4)
        .align_y(Alignment::Center)
        .height(LINE);
    for piece in line {
        let e: Element<'static, Message> = match piece {
            Piece::Text(s, c) => text(s)
                .size(12)
                .color(c)
                .wrapping(text::Wrapping::None)
                .boxed(),
            Piece::Digits(s, c) => text(s)
                .size(12)
                .font(style::digits())
                .color(c)
                .wrapping(text::Wrapping::None)
                .boxed(),
            Piece::Lamp(lit, c) => lamp(lit, c),
            Piece::Bar(share, w, c) => {
                let lit = (w * share.clamp(0.0, 1.0)).round();
                container(
                    row![
                        container(space().width(lit).height(5)).style(move |_| style::bar(c)),
                        space().width(w - lit),
                    ]
                    .height(5),
                )
                .style(|_| style::bar(style::mix(style::BUTTON, Color::BLACK, 0.3)))
                .boxed()
            }
        };
        r = r.push(e);
    }
    r.boxed()
}

/// How many lines a strip h pixels high has room for under its name.
pub(super) fn lines_for(h: f32) -> usize {
    (((h - OTHER_PARTS) / LINE).floor().max(0.0) as usize).min(LINES)
}

/// The first count of lines under a name, count high whatever their
/// number.
fn info(lines: Vec<Line>, count: usize) -> Element<'static, Message> {
    let mut c = column![].spacing(0);
    for l in lines.into_iter().take(count) {
        c = c.push(line_view(l));
    }
    container(c.clip(true))
        .height(LINE * count as f32)
        .width(Fill)
        .boxed()
}

fn scribble(label: &str, channel: Channel, selected: bool) -> Element<'static, Message> {
    button(
        text(label.to_owned())
            .size(13)
            .width(Fill)
            .align_x(Alignment::Center)
            .wrapping(text::Wrapping::None),
    )
    .padding([3, 4])
    .width(Fill)
    .style(move |_, status| style::scribble(selected, status))
    .on_press(Message::Desk(DeskAction::Select(channel)))
    .boxed()
}

fn push(label: &'static str, on: bool, color: Color, b: DeskButton) -> Element<'static, Message> {
    button(text(label).size(12).align_x(Alignment::Center).width(Fill))
        .width(Fill)
        .padding([3, 0])
        .style(move |_, status| style::push(on, color, status))
        .on_press(Message::Desk(DeskAction::Press(b)))
        .boxed()
}

fn readout(said: String, color: Color) -> Element<'static, Message> {
    container(text(said).size(12).font(style::digits()).color(color))
        .padding([2, 6])
        .center_x(Fill)
        .style(|_| style::readout())
        .boxed()
}

fn looks(pair: &Pair, now_ms: u64) -> [MeterLook; 2] {
    pair.each_ref().map(|m| MeterLook {
        db: m.db,
        peak_db: m.peak_db,
        clipped: m.clipped(now_ms),
    })
}

/// channel's strip, w wide, lines lines under its name.
pub(super) fn strip<'a>(
    state: &'a State,
    caps: &'a super::Caps,
    channel: Channel,
    w: f32,
    lines: usize,
) -> Element<'a, Message> {
    let selected = state.desk.selected == channel;
    let body: Element<'a, Message> = match channel {
        Channel::Strip(kind) => fader_strip(state, caps, kind, (selected, w, lines)),
        Channel::Mic => mic_strip(state, selected, lines),
        Channel::Main => master(state, selected, lines),
    };
    container(body)
        .padding([8, 6])
        .width(w)
        .height(Fill)
        .style(move |_| style::strip(selected, channel == Channel::Main))
        .boxed()
}

fn fader_strip<'a>(
    state: &'a State,
    caps: &'a super::Caps,
    kind: StripKind,
    (selected, w, lines): (bool, f32, usize),
) -> Element<'a, Message> {
    let setting = state.desk.setting(kind);
    // A stem's strip is narrow: its short name, no scale of its own (the
    // player's, beside it, reads it).
    let (label, scale) = match kind {
        StripKind::Drums | StripKind::Bass | StripKind::Vocals | StripKind::Other => {
            (kind.short_label(), false)
        }
        StripKind::Player | StripKind::Preview | StripKind::Effects => (kind.label(), true),
    };
    let section = fader::view(Fader {
        channel: Channel::Strip(kind),
        cap: Some(caps.of(kind)),
        selected,
        meters: looks(state.desk.meters(kind), state.now_ms),
        scale: Scale::Taper,
        labels: scale,
        wide: false,
    });
    let value = readout(
        taper::label(setting.db),
        if setting.db == 0.0 { TEXT } else { WARN },
    );
    column![
        scribble(label, Channel::Strip(kind), selected),
        info(said::source_lines(state, kind), lines),
        section,
        value,
        // Two square-ish buttons, narrower on a stem's narrow strip.
        container(
            row![
                push("M", setting.mute, FAIL, DeskButton::Mute(kind)),
                push("S", setting.solo, ACCENT, DeskButton::Solo(kind)),
            ]
            .spacing(4)
            .width((w - 12.0).clamp(0.0, PUSHES))
        )
        .center_x(Fill),
    ]
    .spacing(6)
    .boxed()
}

fn mic_strip(state: &State, selected: bool, lines: usize) -> Element<'_, Message> {
    let open = desk::listens(state) && state.karaoke.mic.name.is_some();
    let r = &state.desk.readings;
    let meters = [&r.mic_in, &r.mic_voice].map(|m| MeterLook {
        db: m.db,
        peak_db: m.peak_db,
        clipped: m.clipped(state.now_ms),
    });
    let mut top = column![
        scribble("MIC", Channel::Mic, selected),
        info(said::mic_lines(state), lines)
    ]
    .spacing(6);
    if open {
        top = top.push(
            row![
                space().width(Fill),
                button(text("ECHO").size(12))
                    .padding([3, 8])
                    .style({
                        let on = state.config.karaoke_echo;
                        move |_, status| style::push(on, GREEN, status)
                    })
                    .on_press(Message::Desk(DeskAction::Press(DeskButton::Echo))),
                space().width(Fill),
            ]
            .boxed(),
        );
    }
    column![
        top,
        row![
            space().width(Fill),
            text("IN").size(11).color(DIM),
            text("VOICE").size(11).color(DIM),
            space().width(Fill)
        ]
        .spacing(12),
        fader::view(Fader {
            channel: Channel::Mic,
            cap: None,
            selected,
            meters,
            scale: Scale::Dbfs60,
            labels: true,
            wide: false,
        }),
        readout("LISTEN ONLY".into(), DIM),
        container(line_view(said::taps_line(state))).center_x(Fill),
    ]
    .spacing(6)
    .boxed()
}

fn master(state: &State, selected: bool, lines: usize) -> Element<'_, Message> {
    let name = match state.desk.master.as_ref() {
        Some(m) if !m.device.is_empty() => format!("MAIN · {}", m.device),
        _ => "MAIN".to_owned(),
    };
    column![
        scribble(&name, Channel::Main, selected),
        info(said::master_lines(state), lines),
        fader::view(Fader {
            channel: Channel::Main,
            cap: None,
            selected,
            meters: looks(&state.desk.readings.master, state.now_ms),
            scale: Scale::Dbfs60,
            labels: true,
            wide: true,
        }),
        container(line_view(said::volume_line(state))).center_x(Fill),
        container(text("adds no gain").size(11).color(DIM)).center_x(Fill),
    ]
    .spacing(6)
    .width(Length::Fill)
    .boxed()
}
