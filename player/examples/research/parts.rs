//! What the recogniser heard, as the research tools keep it: rows of
//! `START<TAB>END<TAB>TEXT`.

use std::path::Path;

use jodel_lyrics::asr::Part;

/// The recogniser's parts or glimpses in `path`, if it is there.
pub fn read_parts(path: &Path) -> Option<Vec<Part>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .filter_map(|row| {
                let mut cells = row.splitn(3, '\t');
                Some(Part {
                    start: cells.next()?.trim().parse().ok()?,
                    end: cells.next()?.trim().parse().ok()?,
                    text: cells.next().unwrap_or("").to_owned(),
                })
            })
            .collect(),
    )
}
