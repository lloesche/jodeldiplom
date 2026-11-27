//! Karaoke mode's looks: a song's own look over whatever its backdrop
//! draws (its cover today, scenes made for it later), as data.
//!
//! - A [`Look`] is an ordered stack of passes ([`Layer`]: a pass by name,
//!   its knobs, how the music moves them, how much of it shows), written
//!   as JSON by a person, a preset ([`presets`]) or a language model,
//!   which never writes shader code: it picks from the catalogue
//!   ([`catalogue`]).
//! - A pass ([`pass::Pass`], the library in [`passes`]) declares its
//!   knobs (name, meaning, range, default, unit, how far the music may
//!   move it, how much light it moves), what it costs, what moves it by
//!   itself (its natural [`Drive`]s) and its maths twice: WGSL for the GPU and
//!   Rust for the processor ([`math`] has WGSL's builtins and the shared
//!   helpers in both), held equal on the GPU by this crate's
//!   `tests/gpu.rs` (and checked as WGSL anywhere).
//! - The music ([`drive`]) is read at the song's time as heard: curves
//!   and moments worked out once a song before it plays ([`drive::Curves`]),
//!   and what the singing knows live.
//! - The guards ([`Look::check`], [`guard`]) hold whatever a look asks:
//!   every number to its range, the music's swing to its limits and the
//!   light it moves under the flash limits, the lyric band readable, the
//!   stack within its GPU budget (its dearest optional pass dropped,
//!   always the same one).
//! - The same look, song and time give the same frame: nothing is random
//!   but by a seed made of the look and the song ([`seed`]).

pub mod catalogue;
pub mod demo;
pub mod drive;
mod faults;
pub mod guard;
pub mod math;
pub mod pass;
pub mod passes;
pub mod presets;
pub mod render;
pub mod wgsl;

use std::borrow::Cow;
use std::collections::BTreeMap;

pub use drive::{Curves, Driver, Feed, Signals};
pub use pass::{Ctx, Family, Knob, Pass, Values, BAND, KNOBS, MIX, MIX_KNOB, PALETTE};

/// A look: its id (unique: a preset's name, or one a model made for a
/// song), its name and what it is for (for people), its tags (genres,
/// themes, eras), and its stack of passes, drawn in order over the
/// backdrop.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Look {
    /// Its id.
    pub id: String,
    /// Its name.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// What it is for.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub about: String,
    /// Its tags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Its passes, in order.
    pub layers: Vec<Layer>,
}

/// A pass in a look: which (by name), its knobs set (by name; those left
/// out at their defaults), how the music moves them (none given: the
/// pass's own way; an empty list: still), how much of it shows over what
/// is under it (0 to 1; the music may move that too, as the knob `mix`),
/// and whether the budget must keep it (else it is optional: the first
/// to go is the dearest).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Layer {
    /// The pass.
    pub pass: String,
    /// Its knobs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub knobs: BTreeMap<String, f32>,
    /// How the music moves them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drive: Option<Vec<Drive>>,
    /// How much of it shows.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub mix: f32,
    /// Whether the budget keeps it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub keep: bool,
}

fn one() -> f32 {
    1.0
}

fn is_one(v: &f32) -> bool {
    *v == 1.0
}

/// A knob moved by the music: `knob` (or `mix`) by `by`, `amount` of the
/// knob's swing (-1 to 1: the swing is the most the music may move it).
/// A look's own, or a pass's natural one ([`pass::natural`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Drive {
    /// The knob.
    pub knob: Cow<'static, str>,
    /// What moves it.
    pub by: Driver,
    /// How far.
    pub amount: f32,
}

impl Layer {
    /// `pass` as it is made: its knobs at their defaults, moved as it
    /// moves itself.
    pub fn of(pass: &str) -> Layer {
        Layer {
            pass: pass.to_owned(),
            knobs: BTreeMap::new(),
            drive: None,
            mix: 1.0,
            keep: false,
        }
    }

    /// Its values on `pass` (its pass), a slot each ([`Pass::slots`]):
    /// each knob as it is set, held to its range, else at its default; its
    /// mix held to 0 to 1; the slots `pass` has not none.
    pub fn values(&self, pass: &Pass) -> Values {
        let mut v = [0.0; KNOBS + 1];
        for (i, k) in pass.slots() {
            let set = if i == MIX {
                Some(self.mix)
            } else {
                self.knobs.get(k.name).copied()
            };
            v[i] = set.map_or(k.default, |x| k.clamp(x));
        }
        v
    }

    /// Its slot `slot` on `pass` (its pass) set to `v`, held to the slot's
    /// range ([`Knob::clamp`]); a slot `pass` has not: nothing.
    pub fn set(&mut self, pass: &Pass, slot: usize, v: f32) {
        let Some(k) = pass.slot(slot) else {
            return;
        };
        if slot == MIX {
            self.mix = k.clamp(v);
        } else {
            self.knobs.insert(k.name.to_owned(), k.clamp(v));
        }
    }

    /// Its drives as they move it on `pass` (its pass): those it gives,
    /// else its pass's own.
    pub fn drives<'a>(&'a self, pass: &'a Pass) -> &'a [Drive] {
        self.drive.as_deref().unwrap_or(pass.natural)
    }
}

impl Look {
    /// It within the budget: the passes [`Look::check`] would drop for it
    /// let go (the dearest not kept first, the later of equals, one at a
    /// time until it is within; a pass there is none of costs nothing and
    /// stays, for [`Look::faults`] to say).
    pub fn within_budget(&mut self) {
        loop {
            let known: Vec<(usize, (f32, bool))> = (self.layers.iter().enumerate())
                .filter_map(|(i, l)| passes::named(&l.pass).map(|p| (i, (p.cost, l.keep))))
                .collect();
            let stack: Vec<(f32, bool)> = known.iter().map(|&(_, s)| s).collect();
            let Some(k) = over_budget(&stack) else {
                return;
            };
            self.layers.remove(known[k].0);
        }
    }

    /// Its name for people (none: its id).
    pub fn title(&self) -> &str {
        if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        }
    }

    /// The look written as `json`.
    pub fn from_json(json: &str) -> Result<Look, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// It as JSON, laid out for people.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// It made safe to draw ([`Checked`]): passes there are none of
    /// dropped, every number held to its range (knobs, mixes, the
    /// music's amounts), the music moving each knob within its swing and
    /// the light it moves within the flash limits ([`guard::FAST_LIGHT`],
    /// [`guard::SLOW_LIGHT`]), at most [`guard::MOST_LAYERS`] passes and
    /// within [`guard::BUDGET`] (the dearest optional pass dropped, the
    /// later of equals, until it is). What was changed is said
    /// ([`Checked::notes`]).
    pub fn check(&self) -> Checked {
        let mut notes = Vec::new();
        let mut layers: Vec<CheckedLayer> = Vec::new();
        for layer in &self.layers {
            let Some(pass) = passes::named(&layer.pass) else {
                notes.push(format!("no pass {:?}: left out", layer.pass));
                continue;
            };
            let values = layer.values(pass);
            for (name, &v) in &layer.knobs {
                match pass.knob(name) {
                    Some((i, k)) => {
                        if values[i] != v && !k.wraps() {
                            notes.push(format!("{}.{name}: {v} held to {}", pass.name, values[i]));
                        }
                    }
                    None => notes.push(format!("{} has no knob {name:?}: left out", pass.name)),
                }
            }
            let mut drives = Vec::new();
            for d in layer.drives(pass) {
                let amount = if d.amount.is_finite() {
                    d.amount.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                match pass.drivable(&d.knob) {
                    Some((slot, swing, light)) if swing > 0.0 && amount != 0.0 => {
                        drives.push(CheckedDrive {
                            slot,
                            by: d.by,
                            amount: amount * swing,
                            light,
                        });
                    }
                    Some(_) => {}
                    None => notes.push(format!("{} has no knob {:?} to move", pass.name, d.knob)),
                }
            }
            layers.push(CheckedLayer {
                pass,
                values,
                drives,
                keep: layer.keep,
            });
        }
        let mut dropped = Vec::new();
        while let Some(i) = over_budget(&stack_of(&layers)) {
            let gone = layers.remove(i);
            notes.push(format!("{} dropped: over the budget", gone.pass.name));
            dropped.push(gone.pass.name);
        }
        guard::cap_light(&mut layers);
        Checked {
            id: self.id.clone(),
            layers,
            dropped,
            notes,
        }
    }
}

/// What a stack of checked layers costs.
fn cost(layers: &[CheckedLayer]) -> f32 {
    layers.iter().map(|l| l.pass.cost).sum()
}

/// A stack of checked layers as the budget sees it: each pass's cost, and
/// whether the budget must keep it.
fn stack_of(layers: &[CheckedLayer]) -> Vec<(f32, bool)> {
    layers.iter().map(|l| (l.pass.cost, l.keep)).collect()
}

/// The pass of `stack` (each its cost and whether the budget must keep
/// it) that goes first while the stack is over the budget
/// ([`guard::over`]): the dearest not kept (the later of equals), else the
/// last; none within the budget.
fn over_budget(stack: &[(f32, bool)]) -> Option<usize> {
    let cost: f32 = stack.iter().map(|&(cost, _)| cost).sum();
    if !guard::over(stack.len(), cost) {
        return None;
    }
    (0..stack.len())
        .filter(|&i| !stack[i].1)
        .max_by(|&a, &b| stack[a].0.total_cmp(&stack[b].0).then(a.cmp(&b)))
        .or(stack.len().checked_sub(1))
}

/// A look made safe to draw ([`Look::check`]): its id, its passes as
/// they draw, the passes the budget dropped and what was changed.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    /// Its id.
    pub id: String,
    /// Its passes.
    pub layers: Vec<CheckedLayer>,
    /// The passes dropped.
    pub dropped: Vec<&'static str>,
    /// What was changed, for people.
    pub notes: Vec<String>,
}

/// A pass as it draws in a checked look: the pass, its values (its knobs
/// in its order and its mix, [`Layer::values`]), how the music moves them
/// and whether the budget keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedLayer {
    /// The pass.
    pub pass: &'static Pass,
    /// Its values.
    pub values: Values,
    /// How the music moves it.
    pub drives: Vec<CheckedDrive>,
    /// Whether the budget keeps it.
    pub keep: bool,
}

/// A knob moved by the music, checked: its slot ([`Pass::slots`]), what
/// moves it, how far (in the knob's units, the flash
/// guard's share taken off) and the light a unit of it moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckedDrive {
    /// Its slot.
    pub slot: usize,
    /// What moves it.
    pub by: Driver,
    /// How far.
    pub amount: f32,
    /// The light a unit moves.
    pub light: f32,
}

/// What every pass of a frame shares: the target's size in pixels, the
/// song's time as heard, the look's seed, the frame's step (seconds),
/// whether the frame before is to be forgotten, the lyric band (top and
/// bottom, shares of the height; none: nothing kept readable), the
/// cover's three colours and how strongly the look draws (its passes'
/// mixes times it, 0 to 1: 1 as it is made, less where its frames over the
/// song's picture would be washed out or blacked out: [`guard::strength`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moment {
    /// Width and height.
    pub size: (u32, u32),
    /// The time.
    pub t: f64,
    /// The seed.
    pub seed: u32,
    /// The step.
    pub dt: f32,
    /// Whether the frame before is forgotten.
    pub forget: bool,
    /// The lyric band.
    pub band: Option<(f32, f32)>,
    /// The cover's colours.
    pub palette: [[f32; 3]; 3],
    /// The strength.
    pub strength: f32,
}

impl Moment {
    /// What a moment a look is measured at ([`guard::strength`],
    /// [`guard::over_song`]) starts from, each caller giving its own size,
    /// time, step and seed: the song's start, the frame before forgotten,
    /// no lyric band, [`PALETTE`], at full strength.
    pub const MEASURING: Moment = Moment {
        size: (1, 1),
        t: 0.0,
        seed: 0,
        dt: 0.0,
        forget: true,
        band: None,
        palette: PALETTE,
        strength: 1.0,
    };
}

impl Checked {
    /// What it costs the GPU ([`Pass::cost`] added up).
    pub fn cost(&self) -> f32 {
        cost(&self.layers)
    }

    /// Whether any of its passes reads the frame before.
    pub fn history(&self) -> bool {
        self.layers.iter().any(|l| l.pass.history)
    }

    /// Each pass's numbers at `moment` with the music `signals` (its
    /// data from `curves`): its knobs moved and held to their ranges.
    pub fn frame(
        &self,
        moment: &Moment,
        signals: &Signals,
        curves: Option<&Curves>,
    ) -> Vec<(&'static Pass, Ctx)> {
        let (w, h) = (moment.size.0.max(1) as f32, moment.size.1.max(1) as f32);
        let feed = Feed {
            t: moment.t,
            signals,
            curves,
        };
        let pal = moment.palette.map(|c| [c[0], c[1], c[2], 0.0]);
        let (top, bottom) = moment.band.unwrap_or((2.0, 2.0));
        self.layers
            .iter()
            .map(|l| {
                let mut v = l.values;
                for d in &l.drives {
                    v[d.slot] += d.amount * signals.get(d.by);
                }
                for (i, knob) in l.pass.slots() {
                    v[i] = knob.clamp(v[i]);
                }
                let mut k = [0.0; KNOBS];
                k.copy_from_slice(&v[..KNOBS]);
                let data = l.pass.data.map_or([[0.0; 4]; pass::DATA], |f| f(&feed));
                let ctx = Ctx {
                    size: [w, h, 1.0 / w, 1.0 / h],
                    time: [
                        moment.t as f32,
                        (moment.seed >> 8) as f32,
                        moment.dt.clamp(0.0, 0.25),
                        if moment.forget { 1.0 } else { 0.0 },
                    ],
                    k,
                    band: [top, bottom, l.pass.relief, guard::CEILING],
                    pal,
                    mix: [v[MIX] * moment.strength.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
                    data,
                };
                (l.pass, ctx)
            })
            .collect()
    }
}

/// The seed of a look for a song: the same look and song, the same
/// frames; another song, another grain.
pub fn seed(look: &str, song: &str) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for b in look.bytes().chain([0]).chain(song.bytes()) {
        h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
    math::mix32(h)
}

#[cfg(test)]
mod tests;
