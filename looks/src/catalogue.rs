//! The menu a model is given to choose a song's look from, made from the
//! passes' own definitions (so it is never out of step with them): every
//! pass (its family, what it does, its knobs with their ranges, units,
//! defaults and how far the music may move them, its cost, how the music
//! moves it by itself, what it goes well with), every driver (what it is,
//! how fast it moves) and every preset (its id, name, tags and stack), as
//! JSON ([`json`]) or Markdown ([`markdown`], what `docs/looks-catalogue.md`
//! is), or in brief where the whole would not fit a model's prompt
//! ([`brief`]: the karaoke story's plan).

use serde::Serialize;

use crate::drive::{Driver, Speed};
use crate::pass::{Family, Knob, Natural};
use crate::passes::PASSES;
use crate::presets::Kind;
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

/// The catalogue in brief, for a model's prompt where the whole of it
/// would not fit: its rules; the presets family by family, each stack
/// made from no other once, its id quoted and its passes in order
/// (`"rock" curve>halation>grain>chromatic`), the presets made from it
/// after it, each its id and the passes it adds (`"rock-70s"
/// +split_tone`); every pass's cost and knobs (with their ranges, where
/// not 0 to 1); and the drivers, fast and slow. Made once.
pub fn brief() -> &'static str {
    static BRIEF: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    BRIEF.get_or_init(|| {
        let c = catalogue();
        let mut out = format!(
            "The catalogue. A look is a preset, and passes added after its passes: at most {} passes in all, costing at most {}.\nPresets, family by family, each id in quotes: \"genre\" a>b, \"genre-era\" +c, \"genre-theme\" would be the presets genre (its passes a then b), genre-era (a, b, then c) and genre-theme (a and b).\n",
            c.rules.most_passes,
            n(c.rules.budget)
        );
        for family in presets::families() {
            out.push_str(&format!("- {}: {}\n", family.family, stacks(&family.looks)));
        }
        out.push_str(
            "The passes, each its cost and its knobs, each knob 0 to 1 unless its range is given:\n",
        );
        for p in &c.passes {
            let knobs: Vec<String> = (p.knobs.iter())
                .map(|k| {
                    if (k.lo, k.hi) == (0.0, 1.0) {
                        k.name.to_owned()
                    } else {
                        format!("{} ({} to {})", k.name, n(k.lo), n(k.hi))
                    }
                })
                .collect();
            out.push_str(&format!(
                "- {} (cost {}): {}\n",
                p.name,
                n(p.cost),
                knobs.join(", ")
            ));
        }
        for (speed, name) in [(Speed::Fast, "fast"), (Speed::Slow, "slow")] {
            let of: Vec<&str> = (c.drivers.iter())
                .filter(|d| d.speed == speed)
                .map(|d| d.name)
                .collect();
            out.push_str(&format!("Drivers, {name}: {}\n", of.join(", ")));
        }
        out
    })
}

/// A family's presets as [`brief`] has them: each stack made from no
/// other (a root) with its passes, then the presets made from it (whose
/// id is the root's and more after a dash, and whose passes start with
/// the root's), each with the passes it adds; in the family's order.
fn stacks(looks: &[crate::Look]) -> String {
    fn passes(l: &crate::Look) -> Vec<&str> {
        l.layers.iter().map(|x| x.pass.as_str()).collect()
    }
    let made_from = |l: &crate::Look, root: &crate::Look| {
        l.id.strip_prefix(root.id.as_str())
            .is_some_and(|rest| rest.starts_with('-'))
            && passes(l).starts_with(&passes(root))
    };
    // Each look's root: the shortest it is made from, else itself.
    let roots: Vec<usize> = (looks.iter().enumerate())
        .map(|(k, l)| {
            (looks.iter().enumerate())
                .filter(|(_, r)| made_from(l, r))
                .min_by_key(|(_, r)| r.id.len())
                .map_or(k, |(j, _)| j)
        })
        .collect();
    let mut out: Vec<String> = Vec::new();
    for (k, root) in looks.iter().enumerate() {
        if roots[k] != k {
            continue;
        }
        let base = passes(root).len();
        let mut s = format!("\"{}\" {}", root.id, passes(root).join(">"));
        for (j, l) in looks.iter().enumerate() {
            if j != k && roots[j] == k {
                s.push_str(&format!(", \"{}\"", l.id));
                for p in &passes(l)[base..] {
                    s.push_str(&format!(" +{p}"));
                }
            }
        }
        out.push(s);
    }
    out.join(", ")
}

/// A number as the catalogue writes it: as short as it is.
pub(crate) fn n(v: f32) -> String {
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
    out.push_str(&matrix());
    out
}

/// The keys of `family` (a genre, an era, a theme): the tag of each of
/// its own looks, tagged with its key alone.
fn keys(family: &presets::Family) -> impl Iterator<Item = &str> {
    family.looks.iter().filter_map(|l| match l.tags.as_slice() {
        [key] => Some(key.as_str()),
        _ => None,
    })
}

/// The keys of every family of `kind`.
fn keys_of(kind: Kind) -> Vec<&'static str> {
    (presets::families().iter())
        .filter(|f| f.kind == kind)
        .flat_map(keys)
        .collect()
}

/// The genre × theme × era matrix, worked examples for the model: for
/// each genre the presets made for it (tagged with its key first), by
/// what was added to its own look (the tags after its key: an era's key,
/// a theme's, or both): the era its music comes from, the themes its
/// songs often sing of, and the two together.
fn matrix() -> String {
    let mut out = String::from(
        "## The matrix: genre × theme × era\n\nEach genre's own look, and the same look in its era and for the themes its songs often sing of (the presets' ids; each a starting point to move the knobs from). Any genre takes any era's or theme's layers the same way: `eras` and `themes` above have them alone.\n\n| genre | its look | its era | its themes | era and theme |\n|---|---|---|---|---|\n",
    );
    let (eras, themes) = (keys_of(Kind::Era), keys_of(Kind::Theme));
    for f in (presets::families().iter()).filter(|f| f.kind == Kind::Genre) {
        for key in keys(f) {
            let of: Vec<&crate::Look> = f
                .looks
                .iter()
                .filter(|l| l.tags.first().is_some_and(|t| t == key))
                .collect();
            let ids = |pick: &dyn Fn(&crate::Look) -> bool| -> String {
                let ids: Vec<String> = of
                    .iter()
                    .filter(|l| pick(l))
                    .map(|l| format!("`{}`", l.id))
                    .collect();
                ids.join(", ")
            };
            // Whether what was added to its own look has one of `of`.
            let adds = |l: &crate::Look, of: &[&str]| {
                (l.tags[1..].iter()).any(|t| of.contains(&t.as_str()))
            };
            out.push_str(&format!(
                "| {key} | `{key}` | {} | {} | {} |\n",
                ids(&|l| adds(l, &eras) && !adds(l, &themes)),
                ids(&|l| adds(l, &themes) && !adds(l, &eras)),
                ids(&|l| adds(l, &eras) && adds(l, &themes)),
            ));
        }
    }
    out.push('\n');
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

    /// The catalogue in brief names every preset with its passes (read
    /// back from it, each preset's as it is), every pass with its cost
    /// and knobs, every driver, and the rules.
    #[test]
    fn the_brief_has_the_whole_catalogue() {
        let brief = super::brief();
        let mut read: Vec<(String, Vec<String>)> = Vec::new();
        for family in crate::presets::families() {
            let line = (brief.lines())
                .find_map(|l| l.strip_prefix(&format!("- {}: ", family.family)))
                .expect(&family.family);
            // A root's passes ("\"id\" a>b"), then those made from it
            // ("\"id\" +c +d").
            let mut root: Vec<String> = Vec::new();
            for item in line.split(", ") {
                let (id, rest) = item[1..].split_once('"').expect(item);
                let rest = rest.trim();
                if rest.is_empty() || rest.starts_with('+') {
                    let added = rest.split_whitespace().map(|p| p[1..].to_owned());
                    read.push((id.to_owned(), root.iter().cloned().chain(added).collect()));
                } else {
                    root = rest.split('>').map(str::to_owned).collect();
                    read.push((id.to_owned(), root.clone()));
                }
            }
        }
        let mut all: Vec<(String, Vec<String>)> = (crate::presets::all().iter())
            .map(|l| {
                (
                    l.id.clone(),
                    l.layers.iter().map(|x| x.pass.clone()).collect(),
                )
            })
            .collect();
        read.sort();
        all.sort();
        assert_eq!(read, all);
        assert!(brief.contains(
            "- rock: \"rock\" curve>halation>grain>chromatic, \"rock-70s\" +split_tone, \"rock-rebellion\", \"rock-freedom\" +lens_flare +light_leak, "
        ));
        assert!(brief.contains("- grain (cost 1): amount, size (0.6 to 3), chroma\n"));
        assert!(brief.contains(&format!(
            "at most {} passes in all, costing at most {}",
            crate::guard::MOST_LAYERS,
            super::n(crate::guard::BUDGET)
        )));
        let drivers: String = (brief.lines())
            .filter_map(|l| l.strip_prefix("Drivers, "))
            .filter_map(|l| l.split_once(": "))
            .map(|(_, names)| format!("{names}, "))
            .collect();
        for d in crate::Driver::ALL {
            assert!(drivers.contains(&format!("{}, ", d.name())), "{}", d.name());
        }
    }
}
