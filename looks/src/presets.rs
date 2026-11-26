//! The presets: looks made by hand from the passes, as data, one JSON file
//! a family (`looks/presets/<family>.json`: its name, what it is, its
//! looks), built in. A look for a song can be a preset as it is, a preset
//! with its knobs moved, or a stack of its own: all are checked the same
//! way ([`crate::Look::check`]).

use std::sync::OnceLock;

use crate::Look;

/// A family of presets as its file has it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Family {
    /// Its name.
    pub family: String,
    /// What it is.
    pub about: String,
    /// Its looks.
    pub looks: Vec<Look>,
}

/// Every family's file, built in.
const FILES: &[&str] = &[include_str!("../presets/examples.json")];

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

/// The preset `id`.
pub fn named(id: &str) -> Option<&'static Look> {
    all().iter().find(|l| l.id == id)
}

/// The family of the preset `id`.
pub fn family_of(id: &str) -> Option<&'static Family> {
    families()
        .iter()
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
        assert!(all().len() >= 12, "{}", all().len());
    }
}
