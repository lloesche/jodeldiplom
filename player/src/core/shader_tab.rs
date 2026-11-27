//! The Shader tab ([`Side::Shader`]): a workbench for Karaoke mode's looks
//! ([`jodel_looks`]). The song's pictures (its cover today) are drawn as
//! the mode draws them, the backdrop and the stage over it, with the
//! tab's look in force while it shows ([`State::look`]), moved by the song
//! playing exactly as in the game (no song: a made-up beat). The presets
//! are listed by family (the ones kept here first), searched by `/`; one
//! picked is the stack edited: its passes added, taken out and moved, each
//! pass's knobs on sliders (their ranges and meanings from the pass), each
//! knob's driver chosen and how far it moves it. What the guards measure
//! on what is drawn shows beside it ([`Readout`]): the most flashes in a
//! second, the lyrics' contrast, the frame's exposure and the strength the
//! look draws at; the stack's cost against the budget beside them. The
//! stack can be copied as JSON, and kept in the data folder as a preset of
//! the user's own ([`crate::store::looks`]).

use jodel_looks::presets;
use jodel_looks::{Driver, Layer, Look, Pass};

use super::keymap::Op;
use super::{Effect, Side, State};

/// The family the presets kept on the tab are listed in.
pub const YOURS: &str = "yours";

/// What the guards measured on what the tab drew lately: the most
/// flashes in any second of the last few, the lyric band's contrast
/// (white words against its brightest tenth, 1 to 21), the frame's
/// exposure (the share of it clipped, its mean light) and the strength the
/// look draws at to keep it (1: as made). (What the stack costs and what
/// the budget drops are the look's own: [`Look::check`].)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readout {
    /// The most flashes in a second.
    pub flashes: usize,
    /// The lyrics' contrast.
    pub contrast: f32,
    /// The share clipped.
    pub clipped: f32,
    /// The mean light.
    pub mean: f32,
    /// The strength it draws at.
    pub strength: f32,
}

/// The Shader tab.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderTab {
    /// The stack edited (none: the first preset's).
    pub look: Option<Look>,
    /// The preset selected in the list ([`listed`]).
    pub preset: usize,
    /// The search (none: none).
    pub search: Option<Search>,
    /// The pass of the stack selected.
    pub layer: usize,
    /// The slider the pointer holds.
    pub dragging: Option<KnobRef>,
    /// The presets kept on the tab, from the data folder.
    pub kept: Vec<Look>,
    /// What the guards measured.
    pub readout: Readout,
}

/// The search over the presets: being typed, or held (typed, then
/// accepted).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Search {
    /// Being typed.
    Typing(String),
    /// Held.
    Held(String),
}

impl Search {
    /// What it searches for.
    pub fn text(&self) -> &str {
        match self {
            Search::Typing(t) | Search::Held(t) => t,
        }
    }
}

/// A slider of the selected pass: a slot's ([`Pass::slots`]: a knob's, or
/// the mix's) value, or how far its driver moves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnobRef {
    /// The slot.
    pub slot: usize,
    /// Which of its sliders.
    pub part: Part,
}

/// Which of a slot's sliders: its value, or its driver's amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Its value.
    Value,
    /// Its driver's amount.
    Amount,
}

/// The pointer and the window on the tab.
#[derive(Clone, Debug, PartialEq)]
pub enum ShaderAction {
    /// The preset at its place in [`listed`] picked: the stack is it.
    Pick(usize),
    /// A pass of the stack selected.
    Layer(usize),
    /// A pass added after the one selected.
    Add(&'static str),
    /// A pass of the stack taken out.
    Remove(usize),
    /// A pass moved so many places on (back, when negative).
    Move(usize, i32),
    /// A slider of the selected pass set to `share` of its way (a value
    /// from the least to the most; a driver's amount from -1 to 1).
    Set {
        /// The slider.
        at: KnobRef,
        /// Where, 0 to 1.
        share: f32,
    },
    /// The slider let go of.
    SlideEnd,
    /// A slot's driver a step round (none, then every driver).
    Driver {
        /// The slot.
        slot: usize,
        /// Which way.
        by: i32,
    },
    /// The search's text.
    Search(String),
    /// The stack copied, as JSON.
    Copy,
    /// The stack kept as a preset of the user's own.
    Keep,
    /// What the guards measured on the frames drawn.
    Readout(Readout),
}

/// Whether the tab shows.
pub fn shows(state: &State) -> bool {
    state.showing(Side::Shader)
}

/// The looks the list shows: the user's own, then every preset family's,
/// then every pass as a look of its own under its pass family's group,
/// each with its group's name, as the search finds them (none: all).
pub fn listed(state: &State) -> Vec<(&str, &Look)> {
    let words: Vec<String> = (state.shader.search.as_ref())
        .map(|s| s.text().to_lowercase())
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let found = |family: &str, l: &Look| {
        let text = format!("{family} {} {} {}", l.id, l.name, l.tags.join(" ")).to_lowercase();
        words.iter().all(|w| text.contains(w.as_str()))
    };
    let kept = state.shader.kept.iter().map(|l| (YOURS, l));
    let built = (presets::families().iter())
        .chain(presets::pass_families())
        .flat_map(|f| f.looks.iter().map(move |l| (f.family.as_str(), l)));
    kept.chain(built).filter(|(f, l)| found(f, l)).collect()
}

/// The stack edited: the tab's, else the first preset's (the built-in
/// presets are never none: a test holds them).
pub fn look(state: &State) -> &Look {
    match &state.shader.look {
        Some(l) => l,
        None => &presets::all()[0],
    }
}

/// The pass of the stack selected, and its place.
pub fn layer(state: &State) -> Option<(usize, &Layer, &'static Pass)> {
    let look = look(state);
    let i = state.shader.layer.min(look.layers.len().checked_sub(1)?);
    let l = &look.layers[i];
    Some((i, l, jodel_looks::passes::named(&l.pass)?))
}

/// The value of `layer`'s slot `slot` on its `pass` ([`Layer::values`]).
pub fn value(layer: &Layer, pass: &Pass, slot: usize) -> f32 {
    layer.values(pass).get(slot).copied().unwrap_or(0.0)
}

/// The music moving `layer`'s slot `slot` on its `pass` (none: none):
/// what moves it and how far ([`Layer::drives`]).
pub fn drive(layer: &Layer, pass: &Pass, slot: usize) -> Option<(Driver, f32)> {
    let name = pass.slot(slot)?.name;
    (layer.drives(pass).iter())
        .find(|d| d.knob == name)
        .map(|d| (d.by, d.amount))
}

/// The stack changed by `change`, the selected pass kept in it; the tab's
/// own from now on.
fn edit(state: &mut State, change: impl FnOnce(&mut Look, &mut usize)) {
    let mut look = look(state).clone();
    let mut at = state.shader.layer;
    change(&mut look, &mut at);
    state.shader.layer = at.min(look.layers.len().saturating_sub(1));
    state.shader.look = Some(look);
}

/// The selected pass changed by `change`.
fn edit_layer(state: &mut State, change: impl FnOnce(&mut Layer, &'static Pass)) {
    let Some((i, _, pass)) = layer(state) else {
        return;
    };
    edit(state, |look, _| change(&mut look.layers[i], pass));
}

/// `layer`'s drives on its `pass` as its own (as they move it:
/// [`Layer::drives`]).
fn own_drives<'a>(layer: &'a mut Layer, pass: &Pass) -> &'a mut Vec<jodel_looks::Drive> {
    layer.drive.get_or_insert_with(|| pass.natural.to_vec())
}

/// The preset at `k` in the list selected, and the stack made it.
fn pick(state: &mut State, k: usize) {
    let listed = listed(state);
    let Some(&(_, look)) = listed.get(k.min(listed.len().saturating_sub(1))) else {
        return;
    };
    let look = look.clone();
    state.shader.preset = k.min(listed.len() - 1);
    state.shader.layer = 0;
    state.shader.look = Some(look);
}

/// The first preset of the family `by` families on from the selected
/// one's.
fn family_step(state: &mut State, by: i32) {
    let listed = listed(state);
    let mut families: Vec<&str> = listed.iter().map(|(f, _)| *f).collect();
    families.dedup();
    let now = listed.get(state.shader.preset).map(|(f, _)| *f);
    let at = families.iter().position(|f| Some(*f) == now).unwrap_or(0) as i32;
    let to = families.get((at + by).clamp(0, families.len() as i32 - 1).max(0) as usize);
    if let Some(k) = to.and_then(|to| listed.iter().position(|(f, _)| f == to)) {
        pick(state, k);
    }
}

/// The stack kept as a preset of the user's own: under its own id (a
/// preset's again gets `my-` before it), replacing one of the same id.
fn keep(state: &mut State, fx: &mut Vec<Effect>) {
    let mut look = look(state).clone();
    if !state.shader.kept.iter().any(|l| l.id == look.id) {
        let base = format!("my-{}", look.id.trim_start_matches("my-"));
        let mut id = base.clone();
        let mut n = 2;
        while state.shader.kept.iter().any(|l| l.id == id) || presets::named(&id).is_some() {
            id = format!("{base}-{n}");
            n += 1;
        }
        look.id = id;
        if !look.name.starts_with("My ") {
            look.name = format!("My {}", look.name);
        }
    }
    let kept = &mut state.shader.kept;
    match kept.iter_mut().find(|l| l.id == look.id) {
        Some(l) => *l = look.clone(),
        None => kept.insert(0, look.clone()),
    }
    state.info(format!("{} kept", look.id));
    state.shader.look = Some(look.clone());
    fx.push(Effect::KeepLook(look));
}

/// What the pointer and the window do on the tab.
pub fn act(state: &mut State, what: ShaderAction, fx: &mut Vec<Effect>) {
    match what {
        ShaderAction::Pick(k) => pick(state, k),
        ShaderAction::Layer(i) => state.shader.layer = i,
        ShaderAction::Add(name) => edit(state, |look, at| {
            let to = if look.layers.is_empty() { 0 } else { *at + 1 };
            look.layers
                .insert(to.min(look.layers.len()), Layer::of(name));
            *at = to;
        }),
        ShaderAction::Remove(i) => edit(state, |look, at| {
            if i < look.layers.len() {
                look.layers.remove(i);
                *at = (*at).min(look.layers.len().saturating_sub(1));
            }
        }),
        ShaderAction::Move(i, by) => edit(state, |look, at| {
            let n = look.layers.len() as i32;
            let to = (i as i32 + by).clamp(0, (n - 1).max(0)) as usize;
            if i < look.layers.len() && to != i {
                let l = look.layers.remove(i);
                look.layers.insert(to, l);
                *at = to;
            }
        }),
        ShaderAction::Set { at, share } => {
            state.shader.dragging = Some(at);
            let share = share.clamp(0.0, 1.0);
            edit_layer(state, |l, pass| {
                let Some(k) = pass.slot(at.slot) else {
                    return;
                };
                match at.part {
                    Part::Value => l.set(pass, at.slot, k.lo + (k.hi - k.lo) * share),
                    Part::Amount => {
                        if let Some(d) = own_drives(l, pass).iter_mut().find(|d| d.knob == k.name) {
                            d.amount = share * 2.0 - 1.0;
                        }
                    }
                }
            });
        }
        ShaderAction::SlideEnd => state.shader.dragging = None,
        ShaderAction::Driver { slot, by } => edit_layer(state, |l, pass| {
            let Some(name) = pass.slot(slot).map(|k| k.name) else {
                return;
            };
            let now = drive(l, pass, slot).map(|(d, _)| d);
            let at = now.map_or(0, |d| 1 + d as i32);
            let n = Driver::COUNT as i32 + 1;
            let to = (at + by).rem_euclid(n);
            let drives = own_drives(l, pass);
            let amount = (drives.iter().find(|d| d.knob == name)).map_or(0.5, |d| d.amount);
            drives.retain(|d| d.knob != name);
            if to > 0 {
                drives.push(jodel_looks::Drive {
                    knob: name.into(),
                    by: Driver::ALL[(to - 1) as usize],
                    amount,
                });
            }
        }),
        ShaderAction::Search(text) => {
            state.shader.search = Some(Search::Typing(text));
            state.shader.preset = 0;
        }
        ShaderAction::Copy => {
            fx.push(Effect::CopyText(look(state).to_json()));
            state.info("The stack copied, as JSON");
        }
        ShaderAction::Keep => keep(state, fx),
        ShaderAction::Readout(r) => state.shader.readout = r,
    }
}

/// A character typed into the search, while it is typed; whether it was.
pub fn typed(state: &mut State, c: char) -> bool {
    if !shows(state) {
        return false;
    }
    match &mut state.shader.search {
        Some(Search::Typing(text)) => {
            text.push(c);
            state.shader.preset = 0;
            true
        }
        _ => false,
    }
}

/// Whether the search is being typed (its keys are the search's).
pub fn searching(state: &State) -> bool {
    shows(state) && matches!(state.shader.search, Some(Search::Typing(_)))
}

/// What a key does on the tab; whether it was the tab's.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if searching(state) {
        match op {
            Op::Erase => {
                if let Some(Search::Typing(text)) = &mut state.shader.search {
                    text.pop();
                }
            }
            Op::Accept => {
                if let Some(Search::Typing(text)) = state.shader.search.take() {
                    state.shader.search = Some(Search::Held(text));
                }
                pick(state, 0);
            }
            Op::Back => state.shader.search = None,
            _ => return false,
        }
        return true;
    }
    let n = listed(state).len();
    let at = state.shader.preset;
    match op {
        Op::Up => pick(state, at.saturating_sub(1)),
        Op::Down => pick(state, (at + 1).min(n.saturating_sub(1))),
        Op::PageUp => family_step(state, -1),
        Op::PageDown => family_step(state, 1),
        Op::Top => pick(state, 0),
        Op::Bottom => pick(state, n.saturating_sub(1)),
        Op::Open => pick(state, at),
        Op::Filter => state.shader.search = Some(Search::Typing(String::new())),
        Op::Back if state.shader.search.is_some() => state.shader.search = None,
        Op::ShaderLayer(by) => {
            let n = look(state).layers.len().max(1) as i32;
            state.shader.layer = (state.shader.layer as i32 + i32::from(by)).rem_euclid(n) as usize;
        }
        Op::ShaderMove(by) => act(
            state,
            ShaderAction::Move(state.shader.layer, i32::from(by)),
            fx,
        ),
        Op::ShaderRemove => act(state, ShaderAction::Remove(state.shader.layer), fx),
        Op::ShaderCopy => act(state, ShaderAction::Copy, fx),
        Op::ShaderKeep => act(state, ShaderAction::Keep, fx),
        _ => return false,
    }
    true
}

/// After any action: a slider held is let go of once the tab is left.
pub(crate) fn settle(state: &mut State) {
    if !shows(state) {
        state.shader.dragging = None;
    }
}
