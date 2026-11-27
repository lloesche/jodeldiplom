//! The presets: looks made by hand from the passes, as data, one JSON file
//! a family (`looks/presets/<family>.json`: its name, what it is, its
//! kind, its looks), built in; and every pass as a look of its own, a
//! family a pass family ([`pass_families`]). A look for a song can be a preset as it is, a preset
//! with its knobs moved, or a stack of its own: all are checked the same
//! way ([`crate::Look::check`]).

use std::sync::OnceLock;

use crate::Look;

/// A family of presets as its file has it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Family {
    /// Its name.
    pub family: String,
    /// What its looks are made for.
    pub kind: Kind,
    /// What it is.
    pub about: String,
    /// Its looks.
    pub looks: Vec<Look>,
}

/// What a family's looks are made for: a genre's songs (each genre's own
/// look, then in an era, for a theme, for both), an era, a theme of the
/// lyrics, a mood of light, a medium, the music drawn, or a pass alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A genre's.
    Genre,
    /// An era's.
    Era,
    /// A theme's.
    Theme,
    /// A mood's.
    Mood,
    /// A medium's.
    Media,
    /// The music drawn.
    Visualizer,
    /// A pass alone ([`pass_families`]; never a file's).
    #[serde(skip_deserializing)]
    Pass,
}

/// Every family's file, built in.
const FILES: &[&str] = &[
    include_str!("../presets/rock.json"),
    include_str!("../presets/pop.json"),
    include_str!("../presets/electronic.json"),
    include_str!("../presets/urban.json"),
    include_str!("../presets/roots.json"),
    include_str!("../presets/classical.json"),
    include_str!("../presets/world.json"),
    include_str!("../presets/occasions.json"),
    include_str!("../presets/eras.json"),
    include_str!("../presets/themes.json"),
    include_str!("../presets/moods.json"),
    include_str!("../presets/media.json"),
    include_str!("../presets/visualizer.json"),
];

/// Every family, as its file has it (a file that does not read is a
/// fault the tests find: none of its looks).
pub fn families() -> &'static [Family] {
    static ALL: OnceLock<Vec<Family>> = OnceLock::new();
    ALL.get_or_init(|| {
        FILES
            .iter()
            .filter_map(|f| serde_json::from_str::<Family>(f).ok())
            .collect()
    })
}

/// Every preset, family by family.
pub fn all() -> &'static [Look] {
    static ALL: OnceLock<Vec<Look>> = OnceLock::new();
    ALL.get_or_init(|| {
        families()
            .iter()
            .flat_map(|f| f.looks.iter().cloned())
            .collect()
    })
}

/// Every pass as a look of its own (its id the pass's name), at its
/// knobs' defaults, moved as it moves itself, a family for each pass
/// family (`grade passes`, ...: [`Kind::Pass`]): what a browser of the
/// looks lists after the presets.
pub fn pass_families() -> &'static [Family] {
    static ALL: OnceLock<Vec<Family>> = OnceLock::new();
    ALL.get_or_init(|| {
        (crate::pass::Family::ALL.iter())
            .map(|&family| Family {
                family: format!("{} passes", family.name()),
                kind: Kind::Pass,
                about: format!("Every {} pass as a look of its own.", family.name()),
                looks: (crate::passes::PASSES.iter())
                    .filter(|p| p.family == family)
                    .map(|p| Look {
                        id: p.name.to_owned(),
                        name: p.name.replace('_', " "),
                        about: p.about.to_owned(),
                        tags: vec![family.name().to_owned()],
                        layers: vec![crate::Layer::of(p.name)],
                    })
                    .collect(),
            })
            .filter(|f| !f.looks.is_empty())
            .collect()
    })
}

/// The look `id`: a preset, or a pass as a look of its own.
pub fn named(id: &str) -> Option<&'static Look> {
    family_of(id)?.looks.iter().find(|l| l.id == id)
}

/// The family of the look `id` (a preset's, or a pass's as a look of its
/// own).
pub fn family_of(id: &str) -> Option<&'static Family> {
    (families().iter())
        .chain(pass_families())
        .find(|f| f.looks.iter().any(|l| l.id == id))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every family's file reads, every preset's id is unique, and every
    /// preset is drawn as it asks: no pass it names missing, no knob,
    /// none of its numbers held, nothing dropped for the budget.
    #[test]
    fn every_preset_reads_and_asks_only_what_there_is() {
        assert_eq!(families().len(), FILES.len(), "every file reads");
        let mut ids = std::collections::HashSet::new();
        for p in all() {
            assert!(ids.insert(p.id.as_str()), "{} twice", p.id);
            assert!(!p.name.is_empty() && !p.layers.is_empty(), "{}", p.id);
            let c = p.check();
            assert!(c.notes.is_empty(), "{}: {:?}", p.id, c.notes);
        }
        assert!(all().len() >= 300, "{}", all().len());
        for p in pass_families().iter().flat_map(|f| &f.looks) {
            assert!(ids.insert(p.id.as_str()), "{}: a preset and a pass", p.id);
            assert!(p.check().notes.is_empty(), "{}", p.id);
            assert_eq!(named(&p.id), Some(p));
        }
        let passes: usize = pass_families().iter().map(|f| f.looks.len()).sum();
        assert_eq!(passes, crate::passes::PASSES.len());
    }
}
