//! The Synth tab in iced's widgets ([`crate::core::synth_tab`]), drawn
//! from its model ([`crate::view::synth::Synth`]) in the side pane under
//! the tabs: a panel, on it the header (the key and the tempo the sounds
//! play in, each a step either way or the song's again, and the theme
//! compared), the game's sounds by moment on the left ([`sounds`]: a click
//! plays one), the selected sound's patch in the middle ([`patch`]: its
//! atoms, its engine, its envelope's curve and sliders, its parameters,
//! keep and revert), and the keyboard across the foot (two octaves from
//! C4, the keys held lit, the computer's keys written on theirs). A
//! slider's press holds the pointer ([`Grab::SynthKnob`]), and so does the
//! keyboard's ([`Grab::Piano`]): the key under it sounds while it is held.

mod patch;
mod sounds;

pub use patch::knob_id;
pub use sounds::id as sound_id;

use iced::widget::{column, container, opaque, row, stack, text, Space};
use iced::{Alignment, Element, Fill, Size, Widget as _};

use super::list;
use crate::core::synth_tab::{SynthAction, KEYS};
use crate::core::{Action, State};
use crate::gui::look::{self, height, size, space};
use crate::gui::pointer::Grab;
use crate::gui::widgets::{self, Reveal};
use crate::gui::window::Message;
use crate::theme::THEMES;
use crate::view::synth::{Keyboard, Synth};

/// The sounds' column, in points.
const SOUNDS_WIDTH: f32 = 280.0;
/// The keyboard's row.
const KEYBOARD: f32 = 104.0;
/// The id of the keyboard's area (the tests press it).
pub const PIANO: &str = "synth-piano";

/// What keeps the sound the keys are on in view.
pub fn reveal(state: &State) -> Reveal {
    Reveal::new(sounds::SCROLLED, sounds::id(state.synth.sound))
}

/// The tab in a pane of `size`: its panel, and on it the header, the
/// sounds, the patch and the keyboard.
pub fn view<'a>(state: &State, size: Size) -> Element<'a, Message> {
    let m = Synth::of(state);
    let body = column![
        header(&m),
        row![
            container(sounds::view(&m)).width(SOUNDS_WIDTH).height(Fill),
            container(patch::view(&m)).width(Fill).height(Fill),
        ]
        .spacing(space::L)
        .height(Fill),
        keyboard(&m.keyboard),
    ]
    .spacing(space::M);
    let body = container(body)
        .padding(space::L)
        .width(size.width)
        .height(size.height);
    opaque(
        stack![list::ground(), body]
            .width(size.width)
            .height(size.height),
    )
    .boxed()
}

/// A step either way round a value, the value between: `‹ A minor ›`.
fn stepper<'a>(
    name: &'static str,
    value: String,
    song: bool,
    step: impl Fn(i32) -> SynthAction,
) -> Element<'a, Message> {
    let button = |label: &'static str, by: i32| {
        widgets::ghost(label)
            .small()
            .on_press(Message::Act(Action::Synth(step(by))))
    };
    row![
        text(name)
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_FAINT),
        Element::from(button("‹", -1)),
        Element::from(
            widgets::ghost(value)
                .small()
                .lit((!song).then_some(look::ACCENT))
                .on_press(Message::Act(Action::Synth(step(0)))),
        ),
        Element::from(button("›", 1)),
    ]
    .spacing(space::XS)
    .align_y(Alignment::Center)
    .boxed()
}

/// The header: the key and the tempo the sounds play in, and the theme
/// compared.
fn header<'a>(m: &Synth) -> Element<'a, Message> {
    let key_said = if m.key_chosen {
        m.key.clone()
    } else {
        format!("{} (song)", m.key)
    };
    let tempo_said = if m.tempo_chosen {
        format!("{:.0} bpm", m.tempo)
    } else {
        format!("{:.0} bpm (song)", m.tempo)
    };
    let themes = widgets::segmented(THEMES.map(|t| {
        widgets::segment(t.title)
            .selected(t == m.theme)
            .on_press(Message::Act(Action::Synth(SynthAction::Theme(t))))
    }));
    row![
        stepper("Key", key_said, !m.key_chosen, SynthAction::KeyStep),
        stepper("Tempo", tempo_said, !m.tempo_chosen, SynthAction::TempoStep),
        Space::new().width(Fill),
        text("Theme")
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_FAINT),
        themes,
    ]
    .spacing(space::L)
    .height(height::CONTROL)
    .align_y(Alignment::Center)
    .boxed()
}

/// The keyboard across the foot: two octaves from C4, the keys held lit,
/// the computer's keys written on theirs; what it plays beside it.
fn keyboard<'a>(k: &Keyboard) -> Element<'a, Message> {
    let keys = stack![
        widgets::piano(KEYS / 12, k.lit.clone(), k.labels.clone()),
        widgets::area()
            .on_press(|bounds, at| Message::Grab(Grab::Piano(bounds), at))
            .id(PIANO)
            .boxed(),
    ]
    .width(Fill)
    .height(Fill);
    let said = column![
        text("Keyboard")
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM),
        text(format!("plays {}", k.atom.name()))
            .size(size::BODY)
            .color(look::TEXT),
        text(format!("keys from C{} · z x: octave", 4 + k.octave))
            .size(size::CAPTION)
            .color(look::TEXT_FAINT),
    ]
    .spacing(space::XS)
    .width(SOUNDS_WIDTH - space::L);
    row![said, keys]
        .spacing(space::L)
        .height(KEYBOARD)
        .align_y(Alignment::Center)
        .boxed()
}
