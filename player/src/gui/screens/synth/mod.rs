//! The Synth tab in iced's widgets ([`crate::core::synth_tab`]), in the
//! side pane under the tabs: a panel, on it the header (the key and the
//! tempo the sounds play in, each a step either way or the song's again,
//! and the theme compared), the game's sounds by moment on the left
//! ([`sounds`]: a click plays one), the selected sound's patch in the
//! middle ([`patch`]: its atoms, its engine, its envelope's curve and
//! sliders, its parameters, keep and revert), and the keyboard across the
//! foot (two octaves from C4, the keys held lit, the computer's keys
//! written on theirs). A slider's press holds the pointer
//! ([`Grab::SynthKnob`]), and so does the keyboard's ([`Grab::Piano`]):
//! the key under it sounds while it is held.

mod patch;
mod sounds;

pub use patch::knob_id;
pub use sounds::id as sound_id;

use std::fmt::Write as _;

use iced::widget::{column, container, opaque, row, stack, text, Space};
use iced::{Alignment, Element, Fill, Size, Widget as _};
use jodel_synth::params::Knobs;

use super::list;
use crate::core::synth_tab::{self, SynthAction, KEYS, LOW};
use crate::core::{Action, State};
use crate::gui::look::{self, height, size, space};
use crate::gui::pointer::Grab;
use crate::gui::widgets::{self, Reveal};
use crate::gui::window::Message;
use crate::sfx::bench::SOUNDS;
use crate::theme::THEMES;

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
    let body = column![
        header(state),
        row![
            container(sounds::view(state))
                .width(SOUNDS_WIDTH)
                .height(Fill),
            container(patch::view(state)).width(Fill).height(Fill),
        ]
        .spacing(space::L)
        .height(Fill),
        keyboard(state),
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
fn header<'a>(state: &State) -> Element<'a, Message> {
    let key = synth_tab::key(state).map_or_else(|| "no key".to_owned(), |k| k.name());
    let key_said = match state.synth.key {
        Some(_) => key,
        None => format!("{key} (song)"),
    };
    let tempo = synth_tab::tempo(state);
    let tempo_said = match state.synth.tempo {
        Some(_) => format!("{tempo:.0} bpm"),
        None => format!("{tempo:.0} bpm (song)"),
    };
    let theme = synth_tab::theme(state);
    let themes = widgets::segmented(THEMES.map(|t| {
        widgets::segment(t.title)
            .selected(t == theme)
            .on_press(Message::Act(Action::Synth(SynthAction::Theme(t))))
    }));
    row![
        stepper(
            "Key",
            key_said,
            state.synth.key.is_none(),
            SynthAction::KeyStep
        ),
        stepper(
            "Tempo",
            tempo_said,
            state.synth.tempo.is_none(),
            SynthAction::TempoStep
        ),
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
fn keyboard<'a>(state: &State) -> Element<'a, Message> {
    let tab = &state.synth;
    let on_screen = |note: f32| {
        let k = (note - LOW).round();
        (0.0..f32::from(KEYS)).contains(&k).then_some(k as u8)
    };
    let lit: Vec<u8> = (tab.held.iter().map(|h| h.1))
        .chain(tab.mouse)
        .filter_map(on_screen)
        .collect();
    let mut labels: Vec<(u8, String)> = computer_keys()
        .into_iter()
        .filter_map(|(k, c)| {
            let note = LOW + 12.0 * tab.octave as f32 + f32::from(k);
            Some((on_screen(note)?, c.to_string()))
        })
        .collect();
    for c in [0u8, 12] {
        if !labels.iter().any(|(k, _)| *k == c) {
            labels.push((c, format!("C{}", 4 + c / 12)));
        }
    }
    let atom = synth_tab::atom(state);
    let keys = stack![
        widgets::piano(KEYS / 12, lit, labels),
        widgets::area()
            .on_press(|bounds, at| Message::Grab(Grab::Piano(bounds), at))
            .id(PIANO)
            .boxed(),
    ]
    .width(Fill)
    .height(Fill);
    let octave = 4 + tab.octave;
    let said = column![
        text("Keyboard")
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT_DIM),
        text(format!("plays {}", atom.name()))
            .size(size::BODY)
            .color(look::TEXT),
        text(format!("keys from C{octave} · z x: octave"))
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

/// The computer's keys of the keyboard, each its place over the lowest C:
/// the Synth tab's rows of the keymap.
fn computer_keys() -> Vec<(u8, char)> {
    use crate::core::keymap::{Context, KeyCode, Op, KEYMAP};
    KEYMAP
        .iter()
        .filter(|b| b.context == Context::Synth)
        .filter_map(|b| match (b.op, b.keys.first().map(|k| k.code)) {
            (Op::SynthNote(k), Some(KeyCode::Char(c))) => Some((u8::try_from(k).ok()?, c)),
            _ => None,
        })
        .collect()
}

/// The tab as an outline: the key, the tempo and the theme; the sounds by
/// moment, the selected one marked; the patch shown, each parameter that
/// applies with its value; the keys held.
pub fn outline(state: &State) -> String {
    let mut out = String::new();
    let key = synth_tab::key(state).map_or_else(|| "no key".to_owned(), |k| k.name());
    let theme = synth_tab::theme(state);
    let _ = writeln!(
        out,
        "synth: {key}, {:.0} bpm, {}",
        synth_tab::tempo(state),
        theme.name
    );
    let mut moment = None;
    for (k, s) in SOUNDS.iter().enumerate() {
        if moment != Some(s.moment) {
            moment = Some(s.moment);
            let _ = writeln!(out, "[{}]", s.moment.name());
        }
        let mark = if k == state.synth.sound { ">" } else { " " };
        let _ = writeln!(out, "{mark} {}", s.name);
    }
    let atom = synth_tab::atom(state);
    let p = synth_tab::patch(state, theme, atom);
    let edited = synth_tab::edited(state, theme, atom);
    let status = patch::status(edited);
    let _ = writeln!(
        out,
        "patch {} ({}) on {}",
        atom.name(),
        status,
        p.engine.engine()
    );
    for k in (0..p.count()).filter(|&k| p.applies(k)) {
        if let (Some(spec), Some(v)) = (p.spec(k), p.get(k)) {
            let _ = writeln!(out, "  {} = {}", spec.key(), spec.kind.text(v));
        }
    }
    let held: Vec<String> = (state.synth.held.iter().map(|h| h.1))
        .chain(state.synth.mouse)
        .map(|n| format!("{n:.0}"))
        .collect();
    let _ = writeln!(out, "keys held: {}", held.join(" "));
    out
}
