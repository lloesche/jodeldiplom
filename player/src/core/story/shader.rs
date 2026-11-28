//! The shaders over a song's pictures as the model writes them
//! ([`Shader`], read forgivingly: [`Shader::of`]), and the look they make
//! ([`Shader::to_look`]): the preset, the passes added after its passes in
//! the order given, the knobs moved, the music moving what was asked (on
//! top of what moves each pass already, a knob asked for anew moved only
//! as asked), the mood among its tags. Nothing of it is refused: a name no
//! one knows (the preset, a pass to add) is asked of the model once more
//! with the rest of the plan's questions (`super::follow`), and what is
//! still unknown then, or anything else off the catalogue, is mended and
//! said: a pass, a knob, a driver taken as the nearest there is
//! ([`super::rules::nearest_by`]; a pass or a knob only where it is near
//! enough, else left out), a pass moved but not in the look added to
//! it, a number held to its range, a knob moved twice moved once; over
//! the budget, the dearest passes not kept go ([`Look::within_budget`]),
//! as the catalogue tells the model.

use std::collections::BTreeMap;

use jodel_looks::{passes, presets, Drive, Driver, Layer, Look};
use serde_json::Value;

use super::repair::{list, number, text};
use super::rules::{near, nearest_by, MOOD};

/// What a look made for a song has after its preset's id.
pub const SONG_ID: &str = "+song";

/// How far the music moves a knob where the model said not.
const AMOUNT: f32 = 0.5;

/// The shaders as the model writes them: a preset by its id, passes added
/// after its passes (in order), knobs moved pass by pass, what in the
/// music moves which knob (or a pass's `mix`), and the mood in a few
/// words; every name as written, known or not.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shader {
    /// The preset.
    pub preset: String,
    /// Passes added after the preset's, in order (one it has stays).
    pub add: Vec<String>,
    /// Knobs moved, pass by pass.
    pub knobs: BTreeMap<String, BTreeMap<String, f32>>,
    /// What the music moves.
    pub drive: Vec<Moved>,
    /// The mood.
    pub mood: Vec<String>,
}

/// A knob of a pass (or its `mix`) moved by the music: `by` a driver,
/// `amount` of the knob's swing, -1 to 1; the names as written.
#[derive(Clone, Debug, PartialEq)]
pub struct Moved {
    /// The pass.
    pub pass: String,
    /// Its knob, or `mix` (none: one the music moves).
    pub knob: Option<String>,
    /// What moves it.
    pub by: String,
    /// How far.
    pub amount: f32,
}

impl Shader {
    /// The shaders in `value` (the answer's `"shader"`), read forgivingly:
    /// a field missing is empty, a knob set to a number alone (not by its
    /// name) sets the pass's `amount` (else its first knob), a value that is no number left
    /// out, a drive without its pass left out (its amount missing:
    /// [`AMOUNT`]); the mood words past the most left out, each mended and
    /// said in `notes`.
    pub fn of(value: Option<&Value>, notes: &mut Vec<String>) -> Shader {
        let field = |key: &str| value.and_then(|v| v.get(key));
        let names = |key: &str| -> Vec<String> {
            (list(field(key)).iter())
                .filter_map(|v| text(Some(v)))
                .filter(|s| !s.is_empty())
                .collect()
        };
        let mut knobs: BTreeMap<String, BTreeMap<String, f32>> = BTreeMap::new();
        if let Some(Value::Object(by_pass)) = field("knobs") {
            for (pass, set) in by_pass {
                let set = match set {
                    Value::Object(set) => set.clone(),
                    alone => {
                        let first = passes::named(pass)
                            .and_then(|p| p.knob("amount").map(|(_, k)| k).or(p.knobs.first()));
                        let Some(first) = first else {
                            notes.push(format!(
                                "the knob of {pass} set to {alone} left out: it has none"
                            ));
                            continue;
                        };
                        notes.push(format!(
                            "{pass} set to {alone} alone: taken as its {}",
                            first.name
                        ));
                        [(first.name.to_owned(), alone.clone())]
                            .into_iter()
                            .collect()
                    }
                };
                for (knob, v) in set {
                    match number(Some(&v)) {
                        Some(n) => {
                            knobs
                                .entry(pass.clone())
                                .or_default()
                                .insert(knob, n as f32);
                        }
                        None => {
                            notes.push(format!("{pass}.{knob} set to {v}, no number: left out"))
                        }
                    }
                }
            }
        }
        let mut drive = Vec::new();
        for d in list(field("drive")) {
            let Some(pass) = text(d.get("pass")).filter(|p| !p.is_empty()) else {
                notes.push(format!("a drive without its pass left out ({d})"));
                continue;
            };
            drive.push(Moved {
                pass,
                knob: text(d.get("knob")).filter(|k| !k.is_empty()),
                by: text(d.get("by")).unwrap_or_default(),
                amount: number(d.get("amount")).map_or(AMOUNT, |a| a as f32),
            });
        }
        let mut mood = names("mood");
        notes.extend(MOOD.hold(&mut mood));
        Shader {
            preset: text(field("preset")).unwrap_or_default(),
            add: names("add"),
            knobs,
            drive,
            mood,
        }
    }

    /// The look it makes, about `about`, everything off the catalogue
    /// mended and said in `notes` (see the module's doc): a preset that is
    /// none taken as the nearest there is to what was written (where
    /// nothing was, to `genre`), a pass added that is none as the nearest
    /// near enough, else left out.
    pub fn to_look(&self, about: &str, genre: &str, notes: &mut Vec<String>) -> Look {
        let preset = presets::named(&self.preset).unwrap_or_else(|| {
            let to = if self.preset.is_empty() {
                genre
            } else {
                &self.preset
            };
            let all = presets::all();
            let preset = nearest_by(to, all, |p| &p.id).expect("the catalogue has presets");
            notes.push(format!(
                "the preset {:?} taken as {:?}",
                self.preset, preset.id
            ));
            preset
        });
        let mut look = preset.clone();
        look.id = format!("{}{SONG_ID}", preset.id);
        look.name = format!("{}, the song's own", preset.name);
        look.about = about.to_owned();
        for mood in &self.mood {
            if !look.tags.contains(mood) {
                look.tags.push(mood.clone());
            }
        }
        for pass in &self.add {
            let Some(name) = known_pass(pass, notes) else {
                continue;
            };
            if !look.layers.iter().any(|l| l.pass == name) {
                look.layers.push(Layer::of(name));
            }
        }
        for (pass, knobs) in &self.knobs {
            if let Some(layer) = layer_of(&mut look, pass, notes) {
                set_knobs(layer, knobs, notes);
            }
        }
        let mut moved: Vec<(String, Drive)> = Vec::new();
        for d in &self.drive {
            let Some(layer) = layer_of(&mut look, &d.pass, notes) else {
                continue;
            };
            let pass = layer.pass.clone();
            let Some(drive) = drive_of(&pass, d, notes) else {
                continue;
            };
            if (moved.iter()).any(|(p, o)| *p == pass && o.knob == drive.knob) {
                notes.push(format!("{pass}.{} moved twice: the first kept", drive.knob));
                continue;
            }
            moved.push((pass, drive));
        }
        for layer in &mut look.layers {
            let asked: Vec<&Drive> = (moved.iter())
                .filter(|(p, _)| *p == layer.pass)
                .map(|(_, d)| d)
                .collect();
            let Some(pass) = passes::named(&layer.pass).filter(|_| !asked.is_empty()) else {
                continue;
            };
            let mut drive = layer.drives(pass).to_vec();
            drive.retain(|d| !asked.iter().any(|a| a.knob == d.knob));
            drive.extend(asked.into_iter().cloned());
            layer.drive = Some(drive);
        }
        let passes_before = look.layers.len();
        look.within_budget();
        if look.layers.len() < passes_before {
            notes.push(format!(
                "over the budget: {} of {passes_before} passes kept",
                look.layers.len()
            ));
        }
        notes.extend(look.faults().into_iter().map(|f| format!("the look: {f}")));
        look
    }
}

/// The pass named `written`, else the nearest there is where it is near
/// enough ([`near`]); else none (each said in `notes`).
fn known_pass(written: &str, notes: &mut Vec<String>) -> Option<&'static str> {
    if let Some(pass) = passes::named(written) {
        return Some(pass.name);
    }
    let pass = nearest_by(written, passes::PASSES, |p| p.name).filter(|p| near(written, p.name));
    match pass {
        Some(pass) => notes.push(format!("the pass {written:?} taken as {:?}", pass.name)),
        None => notes.push(format!("the pass {written:?} is none: left out")),
    }
    pass.map(|p| p.name)
}

/// The layer of `look` a knob or a drive of the pass `written` moves: the
/// pass's own; a pass the look has not added to it; one there is none of
/// taken as the nearest of the look's where it is near enough, else as
/// the nearest there is where that is (added); else none (each said in
/// `notes`).
fn layer_of<'a>(
    look: &'a mut Look,
    written: &str,
    notes: &mut Vec<String>,
) -> Option<&'a mut Layer> {
    let has = |look: &Look, name: &str| look.layers.iter().position(|l| l.pass == name);
    if let Some(k) = has(look, written) {
        return Some(&mut look.layers[k]);
    }
    let names: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
    let own = (nearest_by(written, &names, |n| n))
        .filter(|n| passes::named(written).is_none() && near(written, n))
        .map(|n| (*n).to_owned());
    if let Some(k) = own.and_then(|name| has(look, &name)) {
        notes.push(format!(
            "the pass {written:?} moved taken as {:?}",
            look.layers[k].pass
        ));
        return Some(&mut look.layers[k]);
    }
    let name = known_pass(written, notes)?;
    match has(look, name) {
        Some(k) => Some(&mut look.layers[k]),
        None => {
            notes.push(format!("{name} is moved, so it is added to the look"));
            look.layers.push(Layer::of(name));
            look.layers.last_mut()
        }
    }
}

/// `knobs` set on `layer`: each named as its pass names it (else the
/// nearest it has, where it is near enough, else left out; `mix`, the
/// layer's mix), held to its range (each mending said in `notes`).
fn set_knobs(layer: &mut Layer, knobs: &BTreeMap<String, f32>, notes: &mut Vec<String>) {
    let Some(pass) = passes::named(&layer.pass) else {
        return;
    };
    for (written, &v) in knobs {
        if written == "mix" {
            layer.mix = held(&format!("{}.mix", pass.name), v, (0.0, 1.0), notes);
            continue;
        }
        let knob = (pass.knob(written).map(|(_, k)| k)).or_else(|| {
            nearest_by(written, pass.knobs, |k| k.name).filter(|k| near(written, k.name))
        });
        let Some(knob) = knob else {
            notes.push(format!("{} has no knob {written:?}: left out", pass.name));
            continue;
        };
        let name = knob.name;
        if name != written {
            notes.push(format!(
                "{}.{written} taken as {}.{name}",
                pass.name, pass.name
            ));
        }
        let set = knob.clamp(v);
        if set != v {
            notes.push(format!("{}.{name} set to {v}: taken as {set}", pass.name));
        }
        layer.knobs.insert(name.to_owned(), set);
    }
}

/// `v`, the value of `what`, held within `range` (said in `notes`).
fn held(what: &str, v: f32, (lo, hi): (f32, f32), notes: &mut Vec<String>) -> f32 {
    let set = if v.is_finite() { v.clamp(lo, hi) } else { hi };
    if set != v {
        notes.push(format!("{what} set to {v}: taken as {set}"));
    }
    set
}

/// The drive `d` asks of the pass `pass` (its name the look's): its knob
/// one the music can move (none written: the first; else the nearest
/// such where it is near enough; none where there is no such), its driver
/// one there is (else the nearest where it is near enough; none where
/// none is), its amount within -1 to 1, each mending said in `notes`.
fn drive_of(pass: &str, d: &Moved, notes: &mut Vec<String>) -> Option<Drive> {
    let p = passes::named(pass)?;
    let movable: Vec<&'static str> = (p.slots())
        .filter(|(_, k)| k.swing > 0.0)
        .map(|(_, k)| k.name)
        .collect();
    let written = d.knob.as_deref().unwrap_or_default();
    let knob = match written {
        "" => movable.first().copied(),
        _ => nearest_by(written, &movable, |k| k)
            .filter(|k| near(written, k))
            .copied(),
    };
    let Some(knob) = knob else {
        notes.push(format!(
            "{pass}.{written} cannot be moved by the music: its drive left out"
        ));
        return None;
    };
    if knob != written {
        notes.push(format!("{pass}.{written} moved: taken as {pass}.{knob}"));
    }
    let by = nearest_by(&d.by, &Driver::ALL, |x| x.name()).filter(|x| near(&d.by, x.name()));
    let Some(&by) = by else {
        notes.push(format!(
            "{pass}.{knob} moved by {:?}, no driver: its drive left out",
            d.by
        ));
        return None;
    };
    if by.name() != d.by {
        notes.push(format!(
            "{pass}.{knob} moved by {:?}: taken as {:?}",
            d.by,
            by.name()
        ));
    }
    Some(Drive {
        knob: knob.to_owned().into(),
        by,
        amount: held(
            &format!("{pass}.{knob}'s amount"),
            d.amount,
            (-1.0, 1.0),
            notes,
        ),
    })
}
