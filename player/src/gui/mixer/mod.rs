//! The Mixer tab in iced's own widgets, over its place on the old canvas:
//! a console with the strips side by side where the core lays them out
//! ([`crate::view::layout::desk`], so the keys go round the strips shown),
//! the player's stems beside it under their bracket, the preview, the
//! microphone and the master. The faders, scales and meters are a canvas
//! each ([`fader`]); what each strip says is [`said`], its widgets
//! [`strips`]; the look is [`crate::gui::style`]. Everything sends the
//! core's [`DeskAction`]s; the wheel over a strip goes to the window, which
//! turns it into steps.
//!
//! [`DeskAction`]: crate::core::desk::DeskAction

mod fader;
pub mod said;
mod strips;

use std::time::Instant;

use iced::widget::{container, mouse_area, pin, space, stack};
use iced::{Element, Fill, Widget as _};
use ratatui::layout::Rect;

use crate::core::desk::{taper, DeskAction, StripKind};
use crate::core::{Action, State};
use crate::gui::style;
use crate::gui::window::Message;
use crate::view::layout::desk::DeskLayout;

/// The room between two strips, and around the console's edge, in pixels.
const GAP: f32 = 3.0;

/// The Mixer tab in `area` (its cells on the old canvas, `cell` a cell's
/// size in pixels), its strips where `desk` has them. The layers are the
/// same whatever shows (the console, the stems' bracket or nothing, the
/// strips), so iced keeps each widget's state in its place.
pub fn view<'a>(
    state: &'a State,
    caps: &'a Caps,
    desk: &DeskLayout,
    area: Rect,
    cell: (f32, f32),
) -> Element<'a, Message> {
    let (cw, ch) = cell;
    let (w, h) = (f32::from(area.width) * cw, f32::from(area.height) * ch);
    // A cell rect of the desk in pixels inside the console.
    let place = |r: Rect| {
        (
            f32::from(r.x.saturating_sub(area.x)) * cw,
            f32::from(r.y.saturating_sub(area.y)) * ch,
            f32::from(r.width) * cw,
            f32::from(r.height) * ch,
        )
    };
    let console = container(space())
        .width(Fill)
        .height(Fill)
        .style(|_| style::console());
    let bracket = match desk.stems {
        Some(row) => {
            let (x, y, w, h) = place(row);
            pin(container(strips::bracket(state, w)).height(h))
                .x(x)
                .y(y)
                .boxed()
        }
        None => space().boxed(),
    };
    let mut layers: Vec<Element<'a, Message>> = vec![console.boxed(), bracket];
    // The strips' lines as many as the shortest has room for, so their
    // tall sections line up.
    let shortest = desk
        .channels
        .iter()
        .map(|c| place(c.frame).3)
        .fold(h, f32::min);
    let lines = strips::lines_for(shortest - 2.0 * GAP);
    for c in &desk.channels {
        let (x, y, w, h) = place(c.frame);
        let channel = c.channel;
        // A click anywhere on a strip selects it; the wheel over it moves
        // its fader (the master's: the device's volume).
        let strip = mouse_area(strips::strip(state, caps, channel, w - 2.0 * GAP, lines))
            .on_press(Message::Act(Action::Desk(DeskAction::Select(channel))))
            .on_scroll(move |delta| Message::DeskWheel(channel, delta));
        layers.push(
            pin(container(strip).height(h - 2.0 * GAP).clip(true))
                .x(x + GAP)
                .y(y + GAP)
                .boxed(),
        );
    }
    container(stack(layers)).width(w).height(h).boxed()
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
