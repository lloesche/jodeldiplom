//! The Shader tab in iced's widgets ([`crate::core::shader_tab`]), in the
//! side pane under the tabs: the presets by family on the left (the
//! search over them), the stack in the middle (its passes, a click
//! selecting one; the selected pass's knobs, each a slider with its range
//! and meaning, its driver a step either way and how far it moves it; the
//! passes to add, by family), and on the right the song's pictures drawn
//! through the look as Karaoke mode draws them (its stage: the backdrop,
//! the look over it, the lyrics), what the guards measure under it
//! ([`readout`]), Copy and Keep. A slider's press holds the pointer
//! ([`Grab::ShaderKnob`]).

pub mod readout;

use std::fmt::Write as _;

use iced::widget::{column, container, opaque, row, scrollable, stack, text, Column, Space};
use iced::{Alignment, Element, Fill, Padding, Size, Widget as _};
use jodel_looks::passes::PASSES;
use jodel_looks::{guard, Family};

use super::list;
use crate::core::shader_tab::{self, KnobRef, Part, Search, ShaderAction};
use crate::core::{Action, State};
use crate::gui::look::{self, size, space};
use crate::gui::pointer::Grab;
use crate::gui::stage_view::StageView;
use crate::gui::widgets;
use crate::gui::window::Message;

/// The presets' column and the stack's, in points.
const PRESETS_WIDTH: f32 = 260.0;
const STACK_WIDTH: f32 = 400.0;
/// A knob's name and value.
const NAME: f32 = 96.0;
const VALUE: f32 = 64.0;

/// The id of a slider's track: its part and slot (`shader-knob-0`,
/// `shader-amount-8`, ...).
pub fn knob_id(at: KnobRef) -> String {
    let part = match at.part {
        Part::Value => "knob",
        Part::Amount => "amount",
    };
    format!("shader-{part}-{}", at.slot)
}

/// What keeps the preset the keys are on in view.
pub fn reveal(state: &State) -> widgets::Reveal {
    widgets::Reveal::new(PRESETS, preset_id(state.shader.preset))
}

/// The id of the scrollable the presets are in.
const PRESETS: &str = "shader-presets";

/// The id of preset `k`'s row.
pub fn preset_id(k: usize) -> String {
    format!("shader-preset-{k}")
}

fn act(a: ShaderAction) -> Message {
    Message::Act(Action::Shader(a))
}

/// A caption over a group.
fn caption<'a>(s: String) -> Element<'a, Message> {
    text(s)
        .size(size::CAPTION)
        .font(look::font::label())
        .color(look::TEXT_FAINT)
        .boxed()
}

/// The tab in a pane of `size`: the presets, the stack, the picture.
pub fn view<'a>(state: &State, size: Size, stage: &StageView) -> Element<'a, Message> {
    let body = row![
        container(presets(state)).width(PRESETS_WIDTH).height(Fill),
        container(stack_view(state)).width(STACK_WIDTH).height(Fill),
        container(picture(state, stage)).width(Fill).height(Fill),
    ]
    .spacing(space::L);
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

/// The search, and the presets by family under it.
fn presets<'a>(state: &State) -> Element<'a, Message> {
    let listed = shader_tab::listed(state);
    let search = match &state.shader.search {
        Some(Search::Typing(s)) => format!("search: {s}▏"),
        Some(Search::Held(s)) => format!("search: {s}"),
        None => "/ to search".to_owned(),
    };
    let title = format!("Looks: {} by family, and every pass", listed.len());
    let mut list = Column::<Element<'a, Message>>::new().spacing(2.0);
    let mut family = None;
    for (k, (f, l)) in listed.iter().enumerate() {
        if family != Some(*f) {
            family = Some(*f);
            list = list.push(
                container(caption(f.to_uppercase()))
                    .padding(Padding {
                        top: if k == 0 { space::XS } else { space::M },
                        bottom: space::XS,
                        left: space::M,
                        right: 0.0,
                    })
                    .boxed(),
            );
        }
        let selected = k == state.shader.preset;
        let line = text(l.title().to_owned())
            .size(size::BODY)
            .color(if selected { look::TEXT } else { look::TEXT_DIM })
            .wrapping(text::Wrapping::None)
            .width(Fill);
        let r = widgets::list_row(
            container(line)
                .padding(Padding::from([0.0, space::M]))
                .boxed(),
        )
        .selected(selected)
        .on_press(act(ShaderAction::Pick(k)));
        list = list.push(container(r).id(preset_id(k)).boxed());
    }
    column![
        text(title)
            .size(size::LABEL)
            .font(look::font::label())
            .color(look::TEXT),
        text(search).size(size::LABEL).color(look::TEXT_DIM),
        widgets::card(
            scrollable(list.padding(Padding::from([space::XS, 0.0])))
                .id(PRESETS)
                .height(Fill),
        )
        .padding(space::XS)
        .width(Fill)
        .height(Fill),
    ]
    .spacing(space::S)
    .boxed()
}

/// The stack: its passes, the selected one's knobs, the passes to add.
fn stack_view<'a>(state: &State) -> Element<'a, Message> {
    let look = shader_tab::look(state);
    let mut passes = Column::<Element<'a, Message>>::new().spacing(2.0);
    let selected = shader_tab::layer(state).map(|(i, _, _)| i);
    for (i, l) in look.layers.iter().enumerate() {
        let known = jodel_looks::passes::named(&l.pass).is_some();
        let name = text(l.pass.clone())
            .size(size::BODY)
            .color(if known { look::TEXT } else { look::FAIL })
            .width(Fill);
        let button = |label: &'static str, a: ShaderAction| {
            Element::from(widgets::ghost(label).small().on_press(act(a)))
        };
        let line = row![
            name,
            button("↑", ShaderAction::Move(i, -1)),
            button("↓", ShaderAction::Move(i, 1)),
            button("✕", ShaderAction::Remove(i)),
        ]
        .spacing(space::XS)
        .align_y(Alignment::Center)
        .padding(Padding::from([0.0, space::M]));
        let r = widgets::list_row(line.boxed())
            .selected(selected == Some(i))
            .on_press(act(ShaderAction::Layer(i)));
        passes = passes.push(container(r).boxed());
    }
    column![
        text(look.title().to_owned())
            .size(size::TITLE)
            .font(look::font::title())
            .color(look::TEXT)
            .wrapping(text::Wrapping::None),
        widgets::card(passes).padding(space::XS).width(Fill),
        scrollable(knobs(state)).id("shader-knobs").height(Fill),
        caption("ADD A PASS".to_owned()),
        widgets::card(scrollable(adds()).id("shader-add").height(160.0))
            .padding(space::XS)
            .width(Fill),
    ]
    .spacing(space::M)
    .boxed()
}

/// The passes to add, by family, a click adding one after the selected.
fn adds<'a>() -> Element<'a, Message> {
    let mut list = Column::<Element<'a, Message>>::new().spacing(space::XS);
    for family in Family::ALL {
        let mut names = iced::widget::Row::<Element<'a, Message>>::new().spacing(space::XS);
        for p in PASSES.iter().filter(|p| p.family == family) {
            names = names.push(Element::from(
                widgets::ghost(p.name)
                    .small()
                    .on_press(act(ShaderAction::Add(p.name))),
            ));
        }
        list = list.push(caption(family.name().to_uppercase()));
        list = list.push(container(names.wrap()).boxed());
    }
    list.padding(space::S).boxed()
}

/// The selected pass's knobs and its mix, each a slider; under each, its
/// driver and how far it moves it.
fn knobs<'a>(state: &State) -> Element<'a, Message> {
    let Some((_, layer, pass)) = shader_tab::layer(state) else {
        return text("No pass: add one below")
            .size(size::BODY)
            .color(look::TEXT_FAINT)
            .boxed();
    };
    let held = state.shader.dragging;
    let mut rows = Column::<Element<'a, Message>>::new().spacing(space::S);
    rows = rows.push(
        text(pass.about.to_owned())
            .size(size::CAPTION)
            .color(look::TEXT_DIM)
            .boxed(),
    );
    for (slot, k) in pass.slots() {
        let (lo, hi) = (k.lo, k.hi);
        let v = shader_tab::value(layer, pass, slot);
        let name = k.name;
        let value = KnobRef {
            slot,
            part: Part::Value,
        };
        let share = f64::from((v - lo) / (hi - lo).max(1e-6));
        let track = widgets::track(share)
            .held(held == Some(value))
            .id(knob_id(value))
            .on_press(move |bounds, at| Message::Grab(Grab::ShaderKnob(value, bounds), at));
        let off = (v - k.default).abs() > 1e-4;
        rows = rows.push(
            row![
                text(name)
                    .size(size::LABEL)
                    .color(look::TEXT_DIM)
                    .wrapping(text::Wrapping::None)
                    .width(NAME),
                container(track).width(Fill),
                text(format!("{v:.2}"))
                    .size(size::LABEL)
                    .font(look::font::numbers())
                    .color(if off { look::ACCENT } else { look::TEXT })
                    .width(VALUE)
                    .align_x(Alignment::End),
            ]
            .spacing(space::M)
            .align_y(Alignment::Center)
            .boxed(),
        );
        let drive = shader_tab::drive(layer, pass, slot);
        let driver = drive.map_or("still".to_owned(), |(d, _)| d.name().to_owned());
        let step = |label: &'static str, by: i32| {
            Element::from(
                widgets::ghost(label)
                    .small()
                    .on_press(act(ShaderAction::Driver { slot, by })),
            )
        };
        let amount: Element<'a, Message> = match drive {
            Some((_, a)) => {
                let amount = KnobRef {
                    slot,
                    part: Part::Amount,
                };
                let t = widgets::track(f64::from((a + 1.0) / 2.0))
                    .held(held == Some(amount))
                    .id(knob_id(amount))
                    .on_press(move |bounds, at| {
                        Message::Grab(Grab::ShaderKnob(amount, bounds), at)
                    });
                row![
                    container(t).width(Fill),
                    text(format!("{a:+.2}"))
                        .size(size::CAPTION)
                        .font(look::font::numbers())
                        .color(look::TEXT_DIM)
                        .width(VALUE)
                        .align_x(Alignment::End),
                ]
                .spacing(space::M)
                .align_y(Alignment::Center)
                .boxed()
            }
            None => Space::new().width(Fill).boxed(),
        };
        rows = rows.push(
            row![
                text(format!("{lo}–{hi} {}", k.unit.name()))
                    .size(size::CAPTION)
                    .color(look::TEXT_FAINT)
                    .wrapping(text::Wrapping::None)
                    .width(NAME),
                step("‹", -1),
                text(driver)
                    .size(size::CAPTION)
                    .color(if drive.is_some() {
                        look::ACCENT
                    } else {
                        look::TEXT_FAINT
                    })
                    .wrapping(text::Wrapping::None)
                    .width(84.0)
                    .align_x(Alignment::Center),
                step("›", 1),
                amount,
            ]
            .spacing(space::XS)
            .align_y(Alignment::Center)
            .boxed(),
        );
        rows = rows.push(
            text(k.about.to_owned())
                .size(size::CAPTION)
                .color(look::TEXT_FAINT)
                .boxed(),
        );
    }
    rows.padding(Padding {
        right: space::M,
        ..Padding::ZERO
    })
    .boxed()
}

/// The song's pictures through the look, what the guards measure, Copy and
/// Keep.
fn picture<'a>(state: &State, stage: &StageView) -> Element<'a, Message> {
    let r = &state.shader.readout;
    let checked = shader_tab::look(state).check();
    let cost = checked.cost();
    let ok = |good: bool| if good { look::GOOD } else { look::WARN };
    let readout = row![
        text(format!("flashes {} a second at most", r.flashes))
            .size(size::LABEL)
            .color(ok(r.flashes <= 3)),
        text(format!("lyrics {:.1} : 1", r.contrast))
            .size(size::LABEL)
            .color(ok(r.contrast >= 3.0)),
        text(format!("cost {cost:.1} of {}", guard::BUDGET))
            .size(size::LABEL)
            .color(ok(cost <= guard::BUDGET && checked.dropped.is_empty())),
        text(format!(
            "clipped {:.0} %, light {:.2}, strength {:.0} %",
            r.clipped * 100.0,
            r.mean,
            r.strength * 100.0
        ))
        .size(size::LABEL)
        .color(ok(r.strength >= 1.0)),
        Space::new().width(Fill),
        Element::from(
            widgets::secondary("Copy JSON")
                .key("C")
                .on_press(act(ShaderAction::Copy))
        ),
        Element::from(
            widgets::primary("Keep")
                .key("K")
                .on_press(act(ShaderAction::Keep))
        ),
    ]
    .spacing(space::L)
    .align_y(Alignment::Center);
    let dropped: Element<'a, Message> = if checked.dropped.is_empty() {
        Space::new().boxed()
    } else {
        text(format!(
            "dropped for the budget: {}",
            checked.dropped.join(", ")
        ))
        .size(size::CAPTION)
        .color(look::WARN)
        .boxed()
    };
    let shown: Element<'a, Message> = if stage.gpu() {
        stage.view()
    } else {
        container(text(super::karaoke::mode::NO_STAGE).color(look::TEXT_DIM))
            .center(Fill)
            .boxed()
    };
    column![
        container(shown).width(Fill).height(Fill).clip(true),
        readout,
        dropped
    ]
    .spacing(space::S)
    .boxed()
}

/// The tab as an outline: the presets listed (the selected marked), the
/// stack, the selected pass's knobs with their drivers, the readout.
pub fn outline(state: &State) -> String {
    let mut out = String::new();
    for (k, (f, l)) in shader_tab::listed(state).iter().enumerate() {
        let mark = if k == state.shader.preset { ">" } else { " " };
        let _ = writeln!(out, "{mark} [{f}] {}", l.id);
    }
    let look = shader_tab::look(state);
    let _ = writeln!(out, "stack {}:", look.id);
    for l in &look.layers {
        let _ = writeln!(out, "  {}", l.pass);
    }
    if let Some((_, layer, pass)) = shader_tab::layer(state) {
        for (slot, k) in pass.slots() {
            let v = shader_tab::value(layer, pass, slot);
            let d = shader_tab::drive(layer, pass, slot)
                .map_or(String::new(), |(d, a)| format!(" by {} {a:+.2}", d.name()));
            let _ = writeln!(out, "  {} = {v:.3}{d}", k.name);
        }
    }
    out
}
