//! The Mixer tab in iced's own widgets, laid out in points ([`layout`]):
//! the strips side by side on the window's ground, each a card, the
//! player's stems beside it under their bracket, the preview, the
//! microphone and the master; the strips shown are the core's rule, so
//! the keys go round the same ones. The faders, scales and meters are a
//! canvas each ([`fader`]); what each strip says is [`said`], its widgets
//! [`strips`]; the look is [`crate::gui::look`]'s. Everything sends the
//! core's [`DeskAction`]s; the wheel over a strip goes to the window,
//! which turns it into steps.
//!
//! [`DeskAction`]: crate::core::desk::DeskAction

mod fader;
pub mod layout;
pub mod said;
mod strips;

use std::fmt::Write as _;
use std::time::Instant;

use iced::widget::{container, mouse_area, pin, space, stack};
use iced::{Element, Size, Widget as _};

use crate::core::desk::{taper, DeskAction, StripKind};
use crate::core::{Action, State};
use crate::gui::look;
use crate::gui::window::Message;

/// The room each side of a strip's card inside its place on the desk, in
/// points: the stems' cards, side by side, stand this twice apart.
const INSET: f32 = 1.0;

/// The Mixer tab, `size` points, its strips where [`layout`] has them.
/// The layers are the same whatever shows (the ground, the stems' bracket
/// or nothing, the strips), so iced keeps each widget's state in its
/// place.
pub fn view<'a>(state: &'a State, caps: &'a Caps, size: Size) -> Element<'a, Message> {
    let desk = layout::compute(size, state.desk.has_stems());
    let ground = container(space())
        .width(size.width)
        .height(size.height)
        .style(|_| container::Style {
            background: Some(look::BG.into()),
            ..container::Style::default()
        });
    let bracket = match desk.stems {
        Some(r) => pin(strips::bracket(state, r.width)).x(r.x).y(r.y).boxed(),
        None => space().boxed(),
    };
    let mut layers: Vec<Element<'a, Message>> = vec![ground.boxed(), bracket];
    // The strips' lines as many as the tallest has room for, a stem's as
    // much less as it stands lower, so their tall sections line up.
    let tallest = desk
        .strips
        .iter()
        .map(|s| s.frame.height)
        .fold(0.0, f32::max);
    let top = desk
        .strips
        .iter()
        .map(|s| s.frame.y)
        .fold(size.height, f32::min);
    let info = strips::info_height(tallest);
    for s in &desk.strips {
        let r = s.frame;
        let channel = s.channel;
        let lines = (info - (r.y - top)).max(0.0);
        // A click anywhere on a strip selects it; the wheel over it moves
        // its fader (the master's: the player's volume).
        let strip = mouse_area(strips::strip(
            state,
            caps,
            channel,
            r.width - 2.0 * INSET,
            lines,
        ))
        .on_press(Message::Act(Action::Desk(DeskAction::Select(channel))))
        .on_scroll(move |delta| Message::DeskWheel(channel, delta));
        layers.push(
            pin(container(strip).height(r.height).clip(true))
                .x(r.x + INSET)
                .y(r.y)
                .boxed(),
        );
    }
    container(stack(layers))
        .width(size.width)
        .height(size.height)
        .boxed()
}

/// The Mixer tab in `size` points as an outline: the strips shown with
/// where they are and how wide, the stems' bracket, then each strip as it
/// reads ([`said::Said`]), its tones in brackets.
pub fn outline(state: &State, size: Size) -> String {
    let desk = layout::compute(size, state.desk.has_stems());
    let mut out = String::new();
    let _ = writeln!(out, "desk {} × {}", size.width, size.height);
    if let Some(r) = desk.stems {
        let _ = writeln!(
            out,
            "[bracket] {} at {}, {} wide",
            said::stems(state),
            r.x,
            r.width
        );
    }
    for s in &desk.strips {
        let said = said::said(state, s.channel);
        let _ = writeln!(out, "[strip] at {}, {} wide", s.frame.x, s.frame.width);
        let _ = write!(out, "{said}");
    }
    out
}

/// Each fader's cap as drawn: it glides to where the core has it (a held
/// one is where the pointer puts it).
#[derive(Debug, Default)]
pub struct Caps {
    at: [f32; StripKind::COUNT],
    last: Option<Instant>,
    moving: bool,
}

/// How long a cap takes to most of the way (to within 1/e).
const GLIDE_S: f32 = 0.03;

impl Caps {
    /// The caps a frame on, for `state`'s settings.
    pub fn step(&mut self, state: &State) {
        let now = Instant::now();
        let first = self.last.is_none();
        let dt = self
            .last
            .map_or(0.0, |at| now.duration_since(at).as_secs_f32())
            .min(0.1);
        self.last = Some(now);
        let held = state.desk.dragging.map(|d| d.strip);
        self.moving = false;
        for kind in StripKind::ALL {
            let to = taper::position_of(state.desk.setting(kind).db) as f32;
            let cap = &mut self.at[kind.index()];
            if first || held == Some(kind) || (*cap - to).abs() < 0.0005 {
                *cap = to;
            } else {
                *cap += (to - *cap) * (1.0 - (-dt / GLIDE_S).exp());
                self.moving = true;
            }
        }
    }

    /// Whether a cap is still on its way.
    pub fn moving(&self) -> bool {
        self.moving
    }

    /// `kind`'s cap, as a position on its travel (0: the bottom).
    pub fn of(&self, kind: StripKind) -> f32 {
        self.at[kind.index()]
    }
}
