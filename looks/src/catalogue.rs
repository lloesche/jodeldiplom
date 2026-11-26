//! The menu a model is given to choose a song's look from, made from the
//! passes' own definitions (so it is never out of step with them): every
//! pass (its family, what it does, its knobs with their ranges, units,
//! defaults and how far the music may move them, its cost, how the music
//! moves it by itself, what it goes well with), every driver (what it is,
//! how fast it moves) and every preset (its id, name, tags and stack), as
//! JSON ([`json`]) or Markdown ([`markdown`], what `docs/looks-catalogue.md`
//! is).

use serde::Serialize;

use crate::drive::{Driver, Speed};
use crate::pass::{Family, Knob, Natural};
use crate::passes::PASSES;
use crate::{guard, presets};

#[derive(Serialize)]
struct PassEntry {
    name: &'static str,
    family: Family,
    about: &'static str,
    cost: f32,
    knobs: &'static [Knob],
    moved_by: &'static [Natural],
    pairs: &'static [&'static str],
}

#[derive(Serialize)]
struct DriverEntry {
    name: &'static str,
    about: &'static str,
    speed: Speed,
}

#[derive(Serialize)]
struct PresetEntry<'a> {
    id: &'a str,
    name: &'a str,
    family: &'a str,
    tags: &'a [String],
    passes: Vec<&'a str>,
}

#[derive(Serialize)]
struct Rules {
    budget: f32,
    most_passes: usize,
    fast_light: f32,
    slow_light: f32,
    lyric_ceiling: f32,
}

#[derive(Serialize)]
struct Catalogue<'a> {
    rules: Rules,
    passes: Vec<PassEntry>,
    drivers: Vec<DriverEntry>,
    presets: Vec<PresetEntry<'a>>,
}

fn catalogue() -> Catalogue<'static> {
    Catalogue {
        rules: Rules {
            budget: guard::BUDGET,
            most_passes: guard::MOST_LAYERS,
            fast_light: guard::FAST_LIGHT,
            slow_light: guard::SLOW_LIGHT,
            lyric_ceiling: guard::CEILING,
        },
        passes: PASSES
            .iter()
            .map(|p| PassEntry {
                name: p.name,
                family: p.family,
                about: p.about,
                cost: p.cost,
                knobs: p.knobs,
                moved_by: p.natural,
                pairs: p.pairs,
            })
            .collect(),
        drivers: Driver::ALL
            .iter()
            .map(|d| DriverEntry {
                name: d.name(),
                about: d.about(),
                speed: d.speed(),
            })
            .collect(),
        presets: presets::families()
            .iter()
            .flat_map(|f| {
                f.looks.iter().map(|l| PresetEntry {
                    id: &l.id,
                    name: &l.name,
                    family: &f.family,
                    tags: &l.tags,
                    passes: l.layers.iter().map(|x| x.pass.as_str()).collect(),
                })
            })
            .collect(),
    }
}

/// The catalogue as JSON.
pub fn json() -> String {
    serde_json::to_string_pretty(&catalogue()).unwrap_or_default()
}

/// A number as the catalogue writes it: as short as it is.
fn n(v: f32) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

/// The catalogue as Markdown: how to write a look, the passes family by
/// family with their knobs, the drivers, the presets.
pub fn markdown() -> String {
    let c = catalogue();
    let mut out = String::new();
    out.push_str("# The looks' catalogue\n\n");
    out.push_str("Made from the passes' own definitions by `cargo run -p jodel-looks --example catalogue`; a test holds it to them. How to use it: `docs/looks.md`.\n\n");
    out.push_str("A look is JSON: an `id`, and `layers`, each a `pass` by name with the `knobs` it sets (by name; left out: the default), how the music moves them (`drive`: a list of `{\"knob\", \"by\", \"amount\"}`, the amount a share of the knob's swing from -1 to 1; left out: the pass's own drivers; `[]`: still), its `mix` over what is under it (0 to 1; `mix` can be driven too) and `keep` (the budget never drops it).\n\n");
    out.push_str(&format!(
        "Rules held whatever is asked: every number to its range; at most {} passes costing {} in all (over it, the dearest pass not kept goes); the fast drivers together may move {} of the screen's light, the slow ones {}; the lyric band's light under {} (encoded).\n\n",
        c.rules.most_passes,
        n(c.rules.budget),
        n(c.rules.fast_light),
        n(c.rules.slow_light),
        n(c.rules.lyric_ceiling)
    ));
    out.push_str("## Passes\n\n");
    for family in Family::ALL {
        let of: Vec<&PassEntry> = c.passes.iter().filter(|p| p.family == family).collect();
        if of.is_empty() {
            continue;
        }
        out.push_str(&format!("### {}\n\n", family.name()));
        for p in of {
            out.push_str(&format!(
                "#### `{}` (cost {})\n\n{}.\n\n",
                p.name,
                n(p.cost),
                p.about
            ));
            if !p.knobs.is_empty() {
                out.push_str("| knob | range | default | unit | music's swing | what it does |\n|---|---|---|---|---|---|\n");
                for k in p.knobs {
                    out.push_str(&format!(
                        "| `{}` | {} to {} | {} | {} | {} | {} |\n",
                        k.name,
                        n(k.lo),
                        n(k.hi),
                        n(k.default),
                        k.unit,
                        n(k.swing),
                        k.about
                    ));
                }
                out.push('\n');
            }
            if !p.moved_by.is_empty() {
                let by: Vec<String> = p
                    .moved_by
                    .iter()
                    .map(|m| format!("`{}` by `{}` ({})", m.knob, m.by.name(), n(m.amount)))
                    .collect();
                out.push_str(&format!("Moved by itself: {}.\n\n", by.join(", ")));
            }
            if !p.pairs.is_empty() {
                let pairs: Vec<String> = p.pairs.iter().map(|x| format!("`{x}`")).collect();
                out.push_str(&format!("Goes well with: {}.\n\n", pairs.join(", ")));
            }
        }
    }
    out.push_str(
        "## Drivers\n\n| driver | speed | what it is, what it is good for |\n|---|---|---|\n",
    );
    for d in &c.drivers {
        let speed = match d.speed {
            Speed::Fast => "fast",
            Speed::Slow => "slow",
        };
        out.push_str(&format!("| `{}` | {speed} | {} |\n", d.name, d.about));
    }
    out.push_str("\n## Presets\n\n");
    for f in presets::families() {
        out.push_str(&format!("### {}\n\n{}\n\n", f.family, f.about));
        for l in &f.looks {
            let passes: Vec<String> = l.layers.iter().map(|x| format!("`{}`", x.pass)).collect();
            out.push_str(&format!(
                "- `{}` {}: {} ({})\n",
                l.id,
                l.name,
                passes.join(" → "),
                l.tags.join(", ")
            ));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    /// The catalogue in the docs is the one the passes make (written by
    /// `cargo run -p jodel-looks --example catalogue`), and its JSON reads.
    #[test]
    fn the_catalogue_is_as_the_passes_are() {
        let docs =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/looks-catalogue.md");
        let kept = std::fs::read_to_string(docs).unwrap_or_default();
        assert!(
            kept == super::markdown(),
            "docs/looks-catalogue.md is not the passes' (cargo run -p jodel-looks --example catalogue)"
        );
        let json: serde_json::Value = serde_json::from_str(&super::json()).expect("JSON");
        assert_eq!(
            json["passes"].as_array().map(Vec::len),
            Some(crate::passes::PASSES.len())
        );
    }
}
