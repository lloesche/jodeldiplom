//! The best scores sung, a song each: `scores.txt` in the data folder
//! ([`crate::cache::SCORES`]). The singer's own record, not a cache: every
//! version of the file is read.
//!
//! After the header, a line a best score: the points, the words hit, the
//! words scored, and the song (its path), tab apart. Version 1 had a level
//! in front (there were three): read without it, a song's best of them
//! kept.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::cache::SCORES;

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

/// The best scores, by song (its path).
pub type Bests = HashMap<String, Best>;

/// `bests` as the file holds them, in order (by song).
pub fn to_text(bests: &Bests) -> String {
    let mut songs: Vec<&String> = bests.keys().collect();
    songs.sort();
    let mut body = String::new();
    for song in songs {
        let b = &bests[song];
        let _ = writeln!(body, "{}\t{}\t{}\t{song}", b.points, b.hit, b.words);
    }
    SCORES.wrap(&body)
}

/// The best scores in a file's `text` (every version; lines not read
/// skipped).
pub fn from_text(text: &str) -> Bests {
    let (version, body) = SCORES.open(text).unwrap_or((SCORES.version, ""));
    let mut out = Bests::new();
    for line in body.lines() {
        // Version 1's level in front, left out.
        let line = match version {
            1 => line.split_once('\t').map_or("", |(_, rest)| rest),
            _ => line,
        };
        let mut parts = line.splitn(4, '\t');
        let (Some(points), Some(hit), Some(words), Some(song)) = (
            parts.next().and_then(|p| p.parse().ok()),
            parts.next().and_then(|p| p.parse().ok()),
            parts.next().and_then(|p| p.parse().ok()),
            parts.next(),
        ) else {
            continue;
        };
        let best = Best { points, hit, words };
        match out.get(song) {
            Some(was) if was.points >= points => {}
            _ => {
                out.insert(song.to_owned(), best);
            }
        }
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
    durable::write_atomic(path, to_text(bests).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_go_to_text_and_back() {
        let mut bests = Bests::new();
        bests.insert(
            "Z:/a b/song.flac".into(),
            Best {
                points: 3400,
                hit: 34,
                words: 40,
            },
        );
        bests.insert(
            "Z:/other.flac".into(),
            Best {
                points: 900,
                hit: 9,
                words: 12,
            },
        );
        let text = to_text(&bests);
        assert!(text.starts_with("jodeldiplom-cache scores 2"));
        assert_eq!(from_text(&text), bests);
        // A line not read is skipped, the rest kept.
        let broken = text.replace("3400\t34", "x\t34");
        assert_eq!(from_text(&broken).len(), 1);
    }

    /// Version 1's lines had a level: read without it, a song's best of
    /// its levels kept.
    #[test]
    fn version_one_is_read_without_its_levels() {
        let v1 = "jodeldiplom-cache scores 1 by jodeldiplom x\n\
                  easy\t3900\t39\t40\tZ:/a.flac\n\
                  hard\t3400\t34\t40\tZ:/a.flac\n\
                  medium\t100\t1\t9\tZ:/b.flac\n";
        let bests = from_text(v1);
        assert_eq!(bests.len(), 2);
        assert_eq!(bests["Z:/a.flac"].points, 3900);
        assert_eq!(bests["Z:/b.flac"].words, 9);
    }
}
