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
//!   itself ([`pass::Natural`]) and its maths twice: WGSL for the GPU and
//!   Rust for the processor ([`math`] has WGSL's builtins and the shared
//!   helpers in both), held equal on the GPU by the player's
//!   `shader_check`.
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
pub mod guard;
pub mod math;
pub mod pass;
pub mod passes;
pub mod presets;
pub mod render;
pub mod wgsl;

use std::collections::BTreeMap;

pub use drive::{Curves, Driver, Feed, Signals};
pub use pass::{Ctx, Family, Knob, Pass};

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
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Drive {
    /// The knob.
    pub knob: String,
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
}

impl Look {
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
            let mut knobs = pass.defaults();
            for (name, &v) in &layer.knobs {
                match pass.knob(name) {
                    Some((i, k)) => {
                        knobs[i] = k.clamp(v);
                        if knobs[i] != v && !k.wraps() {
                            notes.push(format!("{}.{name}: {v} held to {}", pass.name, knobs[i]));
                        }
                    }
                    None => notes.push(format!("{} has no knob {name:?}: left out", pass.name)),
                }
            }
            let asked: Vec<(String, Driver, f32)> = match &layer.drive {
                Some(d) => d.iter().map(|d| (d.knob.clone(), d.by, d.amount)).collect(),
                None => (pass.natural.iter())
                    .map(|n| (n.knob.to_owned(), n.by, n.amount))
                    .collect(),
            };
            let mut drives = Vec::new();
            for (name, by, amount) in asked {
                let amount = if amount.is_finite() {
                    amount.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let slot = if name == "mix" {
                    Some((MIX, guard::MIX_SWING, guard::MIX_LIGHT))
                } else {
                    pass.knob(&name)
                        .map(|(i, k)| (i, k.swing, k.light / (k.hi - k.lo).max(1e-6)))
                };
                match slot {
                    Some((slot, swing, light)) if swing > 0.0 && amount != 0.0 => {
                        drives.push(CheckedDrive {
                            slot,
                            by,
                            amount: amount * swing,
                            light,
                        });
                    }
                    Some(_) => {}
                    None => notes.push(format!("{} has no knob {name:?} to move", pass.name)),
                }
            }
            layers.push(CheckedLayer {
                pass,
                knobs,
                mix: if layer.mix.is_finite() {
                    layer.mix.clamp(0.0, 1.0)
                } else {
                    1.0
                },
                drives,
                keep: layer.keep,
            });
        }
        let mut dropped = Vec::new();
        while layers.len() > guard::MOST_LAYERS || cost(&layers) > guard::BUDGET {
            let dearest = (0..layers.len())
                .filter(|&i| !layers[i].keep)
                .max_by(|&a, &b| {
                    layers[a]
                        .pass
                        .cost
                        .total_cmp(&layers[b].pass.cost)
                        .then(a.cmp(&b))
                });
            let Some(i) = dearest.or(layers.len().checked_sub(1)) else {
                break;
            };
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
            strength: 1.0,
        }
    }
}

/// The slot of a layer's mix among its knobs' (as a drive moves it).
pub const MIX: usize = 8;

/// What a stack of checked layers costs.
fn cost(layers: &[CheckedLayer]) -> f32 {
    layers.iter().map(|l| l.pass.cost).sum()
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
    /// How strongly it draws (its passes' mixes times it, 0 to 1): 1 as
    /// it is made, less where its frames over a picture would be washed
    /// out or blacked out ([`guard::strength`]).
    pub strength: f32,
}

/// A pass as it draws in a checked look: the pass, its knobs (in its
/// order), its mix, how the music moves them and whether the budget keeps
/// it.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedLayer {
    /// The pass.
    pub pass: &'static Pass,
    /// Its knobs.
    pub knobs: [f32; 8],
    /// Its mix.
    pub mix: f32,
    /// How the music moves it.
    pub drives: Vec<CheckedDrive>,
    /// Whether the budget keeps it.
    pub keep: bool,
}

/// A knob moved by the music, checked: its slot (a knob's place, or
/// [`MIX`]), what moves it, how far (in the knob's units, the flash
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
/// bottom, shares of the height; none: nothing kept readable) and the
/// cover's three colours.
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
}

/// A palette to measure a look with where no cover gives one: warm,
/// cool and near black.
pub const MEASURING_PALETTE: [[f32; 3]; 3] = [[0.9, 0.5, 0.2], [0.2, 0.4, 0.8], [0.1, 0.1, 0.12]];

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
                let mut k = l.knobs;
                let mut mix = l.mix;
                for d in &l.drives {
                    let by = d.amount * signals.get(d.by);
                    if d.slot == MIX {
                        mix += by;
                    } else {
                        k[d.slot] += by;
                    }
                }
                for (i, knob) in l.pass.knobs.iter().enumerate() {
                    k[i] = knob.clamp(k[i]);
                }
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
                    mix: [
                        mix.clamp(0.0, 1.0) * self.strength.clamp(0.0, 1.0),
                        0.0,
                        0.0,
                        0.0,
                    ],
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
