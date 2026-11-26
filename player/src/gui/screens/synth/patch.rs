//! The Synth tab's middle: the selected sound (its name, its moment, the
//! layers its tier switches on, and Play), its atoms (one shown), and the
//! shown atom's patch: its engine (a segmented choice), its envelope (the
//! curve beside its four sliders), its parameters in their groups (each a
//! slider with its range, a choice, a switch: from the patch's tables),
//! a value off the built-in one in the accent; and where the patch stands
//! (as built, edited, kept) with Keep and Revert.

use iced::widget::{column, container, row, scrollable, text, Column, Row, Space};
use iced::{Alignment, Color, Element, Fill, Length, Padding, Widget as _};
use jodel_synth::engines::AnyParams;
use jodel_synth::params::{Kind, Knobs, Spec};
use jodel_synth::Patch;

use crate::core::synth_tab::{self, Edited, SynthAction};
use crate::core::{Action, State};
use crate::gui::look::{self, size, space};
use crate::gui::pointer::Grab;
use crate::gui::widgets::{self, Icon};
use crate::gui::window::Message;

/// A parameter's name, and its value, in points.
const NAME: f32 = 92.0;
const VALUE: f32 = 76.0;
/// A slider's row.
const ROW: f32 = 26.0;
/// The envelope's curve.
const CURVE: (f32, f32) = (220.0, 104.0);
/// The choices shown side by side (more: a step either way).
const SIDE_BY_SIDE: usize = 4;

/// Where an atom's patch stands.
pub fn status(edited: Option<&Edited>) -> &'static str {
    match edited {
        None => "as built",
        Some(e) if e.kept == Some(e.patch) => "kept",
        Some(e) if e.kept.is_some() => "kept, edited since",
        Some(_) => "edited",
    }
}

/// The id of a parameter's track: its key's words joined by `-`.
pub fn knob_id(spec: &Spec) -> String {
    format!("synth-knob-{}", spec.key().replace(' ', "-"))
}

/// The middle of the tab.
pub fn view<'a>(state: &State) -> Element<'a, Message> {
    let theme = synth_tab::theme(state);
    let atom = synth_tab::atom(state);
    let p = synth_tab::patch(state, theme, atom);
    let built = theme.patch(atom);
    let edited = synth_tab::edited(state, theme, atom);
    let held = state.synth.dragging;
    let groups = scrollable(params(&p, &built, held))
        .id("synth-params")
        .height(Fill);
    column![
        title(state),
        atoms(state),
        engines(&p),
        envelope(&p, &built, held),
        groups,
        footer(edited),
    ]
    .spacing(space::M)
    .width(Fill)
    .height(Fill)
    .boxed()
}

/// The sound's name and moment, the layers its tier switches on, Play.
fn title<'a>(state: &State) -> Element<'a, Message> {
    let sound = synth_tab::sound(state);
    let theme = synth_tab::theme(state);
    let name = column![
        text(sound.moment.name())
            .size(size::CAPTION)
            .font(look::font::label())
            .color(look::TEXT_FAINT),
        text(sound.name.clone())
            .size(size::TITLE)
            .font(look::font::title())
            .color(look::TEXT)
            .wrapping(text::Wrapping::None),
    ]
    .spacing(2.0);
    let tone = sound.tier.map_or(look::TEXT_DIM, |t| {
        let [r, g, b] = theme.color(t);
        Color::from_rgb(r, g, b)
    });
    let layers = (sound.layers.iter())
        .fold(
            Row::<Element<'a, Message>>::new().spacing(space::XS),
            |r, l| r.push(widgets::tag(*l, tone)),
        )
        .wrap();
    let play = widgets::primary("Play")
        .leading(Icon::Play)
        .key("enter")
        .id("synth-play")
        .on_press(Message::Act(Action::Synth(SynthAction::Pick(
            state.synth.sound,
        ))));
    row![
        name,
        container(layers)
            .width(Fill)
            .padding(Padding::from([0.0, space::M])),
        Element::from(play)
    ]
    .spacing(space::M)
    .align_y(Alignment::Center)
    .boxed()
}

/// The sound's atoms, the one shown lit; an edited one marked.
fn atoms<'a>(state: &State) -> Element<'a, Message> {
    let sound = synth_tab::sound(state);
    let theme = synth_tab::theme(state);
    let shown = synth_tab::atom(state);
    let chips = sound.atoms.iter().enumerate().fold(
        Row::<Element<'a, Message>>::new().spacing(space::XS),
        |r, (i, &a)| {
            let mark = if synth_tab::edited(state, theme, a).is_some() {
                " •"
            } else {
                ""
            };
            r.push(Element::from(
                widgets::chip(format!("{}{mark}", a.name()))
                    .lit((a == shown).then_some(look::ACCENT))
                    .on_press(Message::Act(Action::Synth(SynthAction::Atom(i)))),
            ))
        },
    );
    row![label("Plays"), chips.wrap()]
        .spacing(space::M)
        .align_y(Alignment::Center)
        .boxed()
}

/// The engine, a segmented choice.
fn engines<'a>(p: &Patch) -> Element<'a, Message> {
    let now = p.engine.engine();
    let segments = AnyParams::ENGINES.map(|name| {
        widgets::segment(name)
            .selected(name == now)
            .on_press(Message::Act(Action::Synth(SynthAction::Engine(name))))
    });
    row![label("Engine"), widgets::segmented(segments)]
        .spacing(space::M)
        .align_y(Alignment::Center)
        .boxed()
}

/// A dim label at a row's left.
fn label<'a>(said: &'static str) -> Element<'a, Message> {
    text(said)
        .size(size::LABEL)
        .font(look::font::label())
        .color(look::TEXT_DIM)
        .width(NAME - space::M)
        .boxed()
}

/// The envelope: its curve beside its four sliders.
fn envelope<'a>(p: &Patch, built: &Patch, held: Option<usize>) -> Element<'a, Message> {
    let sliders = (0..p.count())
        .filter(|&k| p.spec(k).is_some_and(|s| s.group == "envelope"))
        .fold(
            Column::<Element<'a, Message>>::new().spacing(2.0),
            |c, k| c.push(knob(p, built, k, held)),
        );
    let a = p.adsr;
    let curve = container(widgets::envelope(a.attack, a.decay, a.sustain, a.release))
        .width(CURVE.0)
        .height(CURVE.1);
    widgets::card(
        row![curve, container(sliders).width(Fill)]
            .spacing(space::L)
            .align_y(Alignment::Center),
    )
    .padding(Padding::from([space::S, space::M]))
    .width(Fill)
    .boxed()
}

/// The parameters that apply but the envelope's, in their groups (the
/// engine's first), in two columns.
fn params<'a>(p: &Patch, built: &Patch, held: Option<usize>) -> Element<'a, Message> {
    let own = jodel_synth::patch::PARAMS.len();
    let order = (own..p.count()).chain(0..own);
    let mut groups: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for k in order.filter(|&k| p.applies(k)) {
        let Some(spec) = p.spec(k).filter(|s| s.group != "envelope") else {
            continue;
        };
        match groups.iter_mut().find(|(g, _)| *g == spec.group) {
            Some((_, ks)) => ks.push(k),
            None => groups.push((spec.group, vec![k])),
        }
    }
    // Into two columns, each group to the shorter.
    let mut columns: [(usize, Column<Element<'a, Message>>); 2] = [
        (0, Column::new().spacing(space::M)),
        (0, Column::new().spacing(space::M)),
    ];
    for (group, ks) in groups {
        let rows = ks.len();
        let card = ks.into_iter().fold(
            Column::<Element<'a, Message>>::new().spacing(2.0),
            |c, k| c.push(knob(p, built, k, held)),
        );
        let block = column![
            text(group.to_uppercase())
                .size(size::CAPTION)
                .font(look::font::label())
                .color(look::TEXT_FAINT),
            widgets::card(card)
                .padding(Padding::from([space::XS, space::M]))
                .width(Fill),
        ]
        .spacing(space::XS);
        let shorter = usize::from(columns[1].0 < columns[0].0);
        let (n, c) = std::mem::replace(&mut columns[shorter], (0, Column::new()));
        columns[shorter] = (n + rows + 1, c.push(block.boxed()));
    }
    let [(_, left), (_, right)] = columns;
    row![left.width(Fill), right.width(Fill)]
        .spacing(space::L)
        .padding(Padding {
            right: space::M,
            ..Padding::ZERO
        })
        .boxed()
}

/// Parameter `k`'s row: its name; a slider with its value, a choice, or
/// a switch; in the accent where it is off the built-in patch's.
fn knob<'a>(p: &Patch, built: &Patch, k: usize, held: Option<usize>) -> Element<'a, Message> {
    let (Some(spec), Some(v)) = (p.spec(k), p.get(k)) else {
        return Space::new().boxed();
    };
    let same_engine = built.engine.engine() == p.engine.engine();
    let off = !(same_engine || k < jodel_synth::patch::PARAMS.len()) || built.get(k) != Some(v);
    let tone = if off { look::ACCENT } else { look::TEXT };
    let set = |value: f32| Message::Act(Action::Synth(SynthAction::Set { knob: k, value }));
    let name = text(spec.name)
        .size(size::LABEL)
        .color(look::TEXT_DIM)
        .wrapping(text::Wrapping::None)
        .width(NAME);
    let control: Element<'a, Message> = match spec.kind {
        Kind::Range { .. } => {
            let track = widgets::track(f64::from(spec.kind.share(v)))
                .held(held == Some(k))
                .id(knob_id(&spec))
                .on_press(move |bounds, at| Message::Grab(Grab::SynthKnob(k, bounds), at));
            row![
                container(track).width(Fill),
                text(spec.kind.text(v))
                    .size(size::LABEL)
                    .font(look::font::numbers())
                    .color(tone)
                    .wrapping(text::Wrapping::None)
                    .width(VALUE)
                    .align_x(Alignment::End),
            ]
            .spacing(space::M)
            .align_y(Alignment::Center)
            .boxed()
        }
        Kind::Choice(options) if options.len() <= SIDE_BY_SIDE => {
            let segments = options.iter().enumerate().map(|(i, o)| {
                widgets::segment(*o)
                    .selected(i as f32 == v)
                    .on_press(set(i as f32))
            });
            widgets::segmented(segments)
        }
        Kind::Choice(options) => {
            let n = options.len() as f32;
            row![
                Element::from(
                    widgets::ghost("‹")
                        .small()
                        .on_press(set((v - 1.0).rem_euclid(n)))
                ),
                text(spec.kind.text(v))
                    .size(size::LABEL)
                    .color(tone)
                    .wrapping(text::Wrapping::None)
                    .width(Length::Fill)
                    .align_x(Alignment::Center),
                Element::from(widgets::ghost("›").small().on_press(set((v + 1.0) % n))),
            ]
            .align_y(Alignment::Center)
            .boxed()
        }
        Kind::Toggle => Element::from(
            widgets::chip(spec.kind.text(v))
                .lit((v >= 0.5).then_some(tone))
                .struck(v < 0.5)
                .on_press(set(1.0 - v)),
        ),
    };
    row![name, container(control).width(Fill)]
        .spacing(space::S)
        .height(ROW)
        .align_y(Alignment::Center)
        .boxed()
}

/// Where the patch stands, and Keep and Revert where they do something.
fn footer<'a>(edited: Option<&Edited>) -> Element<'a, Message> {
    let act = |a: SynthAction| Message::Act(Action::Synth(a));
    let mut keep = widgets::secondary("Keep").key("K").id("synth-keep");
    if edited.is_some_and(|e| e.kept != Some(e.patch)) {
        keep = keep.on_press(act(SynthAction::Keep));
    }
    let mut revert = widgets::ghost("As built").key("del").id("synth-revert");
    if edited.is_some() {
        revert = revert.on_press(act(SynthAction::Revert));
    }
    let tone = match edited {
        None => look::TEXT_FAINT,
        Some(e) if e.kept == Some(e.patch) => look::GOOD,
        Some(_) => look::ACCENT,
    };
    row![
        text(format!("The patch: {}", status(edited)))
            .size(size::LABEL)
            .font(look::font::label())
            .color(tone),
        Space::new().width(Fill),
        Element::from(revert),
        Element::from(keep),
    ]
    .spacing(space::S)
    .align_y(Alignment::Center)
    .boxed()
}
