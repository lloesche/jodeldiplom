//! The shaders over a song's pictures as the model writes them
//! ([`Shader`]), and the look they make ([`Shader::to_look`]): the
//! preset, the passes added after its passes in the order given, the
//! knobs moved, the music moving what was asked (on top of what moves
//! each pass already, a knob asked for anew moved only as asked), the
//! mood among its tags; held to the catalogue strictly
//! ([`jodel_looks::Look::faults`]) but for one thing: over the budget,
//! the dearest passes not kept go ([`Look::within_budget`]), as the
//! catalogue tells the model, since a plan that is otherwise right should
//! not be asked for again for the shaders' cost alone. Every other way
//! off the catalogue, a number out of its range too, is refused, and the
//! model asked again with the error.

use std::collections::BTreeMap;

use jodel_looks::{passes, presets, Drive, Driver, Layer, Look};

use super::rules::MOOD;

/// What a look made for a song has after its preset's id.
pub const SONG_ID: &str = "+song";

/// The shaders as the model writes them: a preset by its id, passes added
/// after its passes (in order), knobs moved pass by pass (each knob to a
/// value in its range), what in the music moves which knob (or a pass's
/// `mix`), and the mood in a few words.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shader {
    /// The preset.
    pub preset: String,
    /// Passes added after the preset's, in order (one it has stays).
    #[serde(default)]
    pub add: Vec<String>,
    /// Knobs moved, pass by pass.
    #[serde(default)]
    pub knobs: BTreeMap<String, BTreeMap<String, f32>>,
    /// What the music moves.
    #[serde(default)]
    pub drive: Vec<Moved>,
    /// The mood.
    pub mood: Vec<String>,
}

/// A knob of a pass (or its `mix`) moved by the music: `by` a driver,
/// `amount` of the knob's swing, -1 to 1.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moved {
    /// The pass.
    pub pass: String,
    /// Its knob, or `mix`.
    pub knob: String,
    /// What moves it.
    pub by: Driver,
    /// How far.
    pub amount: f32,
}

impl Shader {
    /// The look it makes, about `about`, or every way it leaves the
    /// catalogue: a preset there is none of, a pass added there is none of,
    /// knobs or drives of a pass the look has not, mood words not as many
    /// as asked, and the look's own faults ([`Look::faults`]). A pass to
    /// add that the look has already stays where it is (the model names
    /// in `add` the passes it moves as often as those it adds).
    pub fn to_look(&self, about: &str) -> Result<Look, Vec<String>> {
        let mut faults: Vec<String> = MOOD.check(self.mood.len()).err().into_iter().collect();
        if self.mood.iter().any(|m| m.trim().is_empty()) {
            faults.push("a mood word is empty".into());
        }
        let Some(preset) = presets::named(&self.preset) else {
            faults.push(format!(
                "the shader's preset {:?} is not in the catalogue",
                self.preset
            ));
            return Err(faults);
        };
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
            if !look.layers.iter().any(|l| l.pass == *pass) {
                look.layers.push(Layer::of(pass));
            }
        }
        let has = |pass: &str| look.layers.iter().any(|l| l.pass == pass);
        let stack = || {
            let passes: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
            passes.join(", ")
        };
        for pass in self.knobs.keys().chain(self.drive.iter().map(|d| &d.pass)) {
            if !has(pass) {
                faults.push(format!(
                    "{pass} is moved, but the look has no {pass} (its passes: {}; add it to \"add\")",
                    stack()
                ));
            }
        }
        for layer in &mut look.layers {
            if let Some(knobs) = self.knobs.get(&layer.pass) {
                layer
                    .knobs
                    .extend(knobs.iter().map(|(k, &v)| (k.clone(), v)));
            }
            let asked: Vec<&Moved> = (self.drive.iter())
                .filter(|d| d.pass == layer.pass)
                .collect();
            if asked.is_empty() {
                continue;
            }
            let Some(pass) = passes::named(&layer.pass) else {
                continue;
            };
            let mut drive = layer.drives(pass).to_vec();
            drive.retain(|d| !asked.iter().any(|a| a.knob == d.knob));
            drive.extend(asked.iter().map(|a| Drive {
                knob: a.knob.clone().into(),
                by: a.by,
                amount: a.amount,
            }));
            layer.drive = Some(drive);
        }
        // Over the budget, the dearest passes not kept go, as the catalogue
        // tells the model they do: its faults are the fitted look's.
        look.within_budget();
        faults.extend(look.faults());
        if faults.is_empty() {
            Ok(look)
        } else {
            Err(faults)
        }
    }
}
