//! The best scores sung, a song and a level each: `scores.txt` in the data
//! folder ([`crate::cache::SCORES`]). The singer's own record, not a cache:
//! every version of the file is read.
//!
//! After the header, a line a best score: the level, the points, the words
//! hit, the words scored, and the song (its path), tab apart.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::cache::SCORES;
use crate::core::singing::Level;

/// A best score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Best {
    /// The points.
    pub points: u32,
    /// The words hit.
    pub hit: u32,
    /// The words scored.
    pub words: u32,
}

/// The best scores, by song (its path) and level.
pub type Bests = HashMap<(String, Level), Best>;

/// `bests` as the file holds them, in order (by song, then level).
pub fn to_text(bests: &Bests) -> String {
    let mut keys: Vec<&(String, Level)> = bests.keys().collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0).then((a.1 as u8).cmp(&(b.1 as u8))));
    let mut body = String::new();
    for key in keys {
        let b = &bests[key];
        let _ = writeln!(
            body,
            "{}\t{}\t{}\t{}\t{}",
            key.1.name(),
            b.points,
            b.hit,
            b.words,
            key.0
        );
    }
    SCORES.wrap(&body)
}

/// The best scores in a file's `text` (every version; lines not read
/// skipped).
pub fn from_text(text: &str) -> Bests {
    let body = SCORES.open(text).map_or("", |(_, rest)| rest);
    let mut out = Bests::new();
    for line in body.lines() {
        let mut parts = line.splitn(5, '\t');
        let (Some(level), Some(points), Some(hit), Some(words), Some(song)) = (
            parts.next().and_then(Level::parse),
            parts.next().and_then(|p| p.parse().ok()),
            parts.next().and_then(|p| p.parse().ok()),
            parts.next().and_then(|p| p.parse().ok()),
            parts.next(),
        ) else {
            continue;
        };
        out.insert((song.to_owned(), level), Best { points, hit, words });
    }
    out
}

/// The best scores in the file at `path` (none when there is none).
pub fn load(path: &Path) -> Bests {
    std::fs::read_to_string(path)
        .map(|t| from_text(&t))
        .unwrap_or_default()
}

/// Writes `bests` to `path`, durably.
pub fn save(path: &Path, bests: &Bests) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    durable::write_atomic(path, to_text(bests).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_go_to_text_and_back() {
        let mut bests = Bests::new();
        bests.insert(
            ("Z:/a b/song.flac".into(), Level::Hard),
            Best {
                points: 3400,
                hit: 34,
                words: 40,
            },
        );
        bests.insert(
            ("Z:/a b/song.flac".into(), Level::Easy),
            Best {
                points: 3900,
                hit: 39,
                words: 40,
            },
        );
        let text = to_text(&bests);
        assert!(text.starts_with("jodeldiplom-cache scores 1"));
        assert_eq!(from_text(&text), bests);
        // A line not read is skipped, the rest kept.
        let broken = text.replace("hard\t3400", "harder\t3400");
        assert_eq!(from_text(&broken).len(), 1);
    }
}
