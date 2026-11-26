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
//! second, the lyrics' contrast, the stack's cost against the budget. The
//! stack can be copied as JSON, and kept in the data folder as a preset of
//! the user's own ([`crate::store::looks`]).

use jodel_looks::presets;
use jodel_looks::{Drive, Driver, Layer, Look, Pass, MIX};

use super::keymap::Op;
use super::{Effect, Side, State};

/// The family the presets kept on the tab are listed in.
pub const YOURS: &str = "yours";

/// What the guards measured on what the tab drew lately: the most
/// flashes in any second of the last few, the lyric band's contrast
/// (white words against its brightest tenth, 1 to 21), the stack's cost,
/// and the passes the budget dropped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readout {
    /// The most flashes in a second.
    pub flashes: usize,
    /// The lyrics' contrast.
    pub contrast: f32,
    /// What the stack costs.
    pub cost: f32,
    /// The passes dropped.
    pub dropped: Vec<&'static str>,
}

/// The Shader tab.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderTab {
    /// The stack edited (none: the first preset's).
    pub look: Option<Look>,
    /// The preset selected in the list ([`listed`]).
    pub preset: usize,
    /// The search (none: none), and whether it is being typed.
    pub search: Option<(String, bool)>,
    /// The pass of the stack selected.
    pub layer: usize,
    /// The knob whose slider the pointer holds (a knob's place, or
    /// [`MIX`]), and whether it is its driver's amount.
    pub dragging: Option<(usize, bool)>,
    /// The presets kept on the tab, from the data folder.
    pub kept: Vec<Look>,
    /// What the guards measured.
    pub readout: Readout,
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
    /// A knob of the selected pass (its place, or [`MIX`]) set to `share`
    /// of its way.
    Slide {
        /// The knob.
        knob: usize,
        /// Where, 0 to 1.
        share: f32,
    },
    /// A knob's driver's amount set to `share` of its way (-1 at 0, 1 at
    /// 1).
    Amount {
        /// The knob.
        knob: usize,
        /// Where, 0 to 1.
        share: f32,
    },
    /// The slider let go of.
    SlideEnd,
    /// A knob's driver a step round (none, then every driver).
    Driver {
        /// The knob.
        knob: usize,
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

/// The presets the list shows: the user's own, then every family's, each
/// with its family, as the search finds them (none: all).
pub fn listed(state: &State) -> Vec<(&str, &Look)> {
    let words: Vec<String> = (state.shader.search.as_ref())
        .map(|(s, _)| s.to_lowercase())
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let found = |family: &str, l: &Look| {
        let text = format!("{family} {} {} {}", l.id, l.name, l.tags.join(" ")).to_lowercase();
        words.iter().all(|w| text.contains(w.as_str()))
    };
    let kept = state.shader.kept.iter().map(|l| (YOURS, l));
    let built = presets::families()
        .iter()
        .flat_map(|f| f.looks.iter().map(move |l| (f.family.as_str(), l)));
    kept.chain(built).filter(|(f, l)| found(f, l)).collect()
}

/// The stack edited: the tab's, else the first preset's.
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

/// A knob's value in `layer` of `pass` (its place, or [`MIX`]).
pub fn value(layer: &Layer, pass: &Pass, knob: usize) -> f32 {
    if knob == MIX {
        return layer.mix;
    }
    pass.knobs.get(knob).map_or(0.0, |k| {
        layer.knobs.get(k.name).map_or(k.default, |&v| k.clamp(v))
    })
}

/// A knob's range (its place, or [`MIX`]: 0 to 1).
pub fn range(pass: &Pass, knob: usize) -> (f32, f32) {
    pass.knobs.get(knob).map_or((0.0, 1.0), |k| (k.lo, k.hi))
}

/// The knob's name (its place, or [`MIX`]: `mix`).
pub fn knob_name(pass: &Pass, knob: usize) -> &'static str {
    pass.knobs.get(knob).map_or("mix", |k| k.name)
}

/// The music moving `layer`'s knob (none: none): its own drives, or the
/// pass's natural ones where it has none of its own.
pub fn drive(layer: &Layer, pass: &Pass, knob: usize) -> Option<(Driver, f32)> {
    let name = knob_name(pass, knob);
    match &layer.drive {
        Some(d) => d.iter().find(|d| d.knob == name).map(|d| (d.by, d.amount)),
        None => (pass.natural.iter())
            .find(|n| n.knob == name)
            .map(|n| (n.by, n.amount)),
    }
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

/// `layer`'s drives as its own (the pass's natural ones, where it had
/// none of its own).
fn own_drives<'a>(layer: &'a mut Layer, pass: &Pass) -> &'a mut Vec<Drive> {
    layer.drive.get_or_insert_with(|| {
        (pass.natural.iter())
            .map(|n| Drive {
                knob: n.knob.to_owned(),
                by: n.by,
                amount: n.amount,
            })
            .collect()
    })
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
        if look.name.is_empty() || !look.name.starts_with("My ") {
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
        ShaderAction::Slide { knob, share } => {
            state.shader.dragging = Some((knob, false));
            edit_layer(state, |l, pass| {
                let (lo, hi) = range(pass, knob);
                let v = lo + (hi - lo) * share.clamp(0.0, 1.0);
                match pass.knobs.get(knob) {
                    Some(k) => {
                        l.knobs.insert(k.name.to_owned(), k.clamp(v));
                    }
                    None => l.mix = v,
                }
            });
        }
        ShaderAction::Amount { knob, share } => {
            state.shader.dragging = Some((knob, true));
            edit_layer(state, |l, pass| {
                let name = knob_name(pass, knob);
                if let Some(d) = own_drives(l, pass).iter_mut().find(|d| d.knob == name) {
                    d.amount = share.clamp(0.0, 1.0) * 2.0 - 1.0;
                }
            });
        }
        ShaderAction::SlideEnd => state.shader.dragging = None,
        ShaderAction::Driver { knob, by } => edit_layer(state, |l, pass| {
            let name = knob_name(pass, knob);
            let now = drive(l, pass, knob).map(|(d, _)| d);
            let at = now.map_or(0, |d| 1 + d as i32);
            let n = Driver::COUNT as i32 + 1;
            let to = (at + by).rem_euclid(n);
            let drives = own_drives(l, pass);
            let amount = (drives.iter().find(|d| d.knob == name)).map_or(0.5, |d| d.amount);
            drives.retain(|d| d.knob != name);
            if to > 0 {
                drives.push(Drive {
                    knob: name.to_owned(),
                    by: Driver::ALL[(to - 1) as usize],
                    amount,
                });
            }
        }),
        ShaderAction::Search(text) => {
            state.shader.search = Some((text, true));
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
        Some((text, true)) => {
            text.push(c);
            state.shader.preset = 0;
            true
        }
        _ => false,
    }
}

/// Whether the search is being typed (its keys are the search's).
pub fn searching(state: &State) -> bool {
    shows(state) && state.shader.search.as_ref().is_some_and(|(_, t)| *t)
}

/// What a key does on the tab; whether it was the tab's.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if searching(state) {
        match op {
            Op::Erase => {
                if let Some((text, _)) = &mut state.shader.search {
                    text.pop();
                }
            }
            Op::Accept => {
                if let Some((_, typing)) = &mut state.shader.search {
                    *typing = false;
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
        Op::Filter => state.shader.search = Some((String::new(), true)),
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
        Op::ShaderKeep => keep(state, fx),
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
