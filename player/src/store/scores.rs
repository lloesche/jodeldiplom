//! The best scores sung, a song each: `scores.txt` in the data folder
//! ([`crate::store::cache::SCORES`]). The singer's own record, not a cache:
//! every version of the file is read.
//!
//! After the header, a line a best score: the points, the words hit, the
//! words scored, and the song (its path), tab apart. Version 1 had a level
//! in front (there were three): read without it, a song's best of them
//! kept.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::store::cache::SCORES;

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

/// The best scores in the file at `path`, wherever it is while the data
/// folder moves ([`crate::store::mover::read`]); none when there is none.
pub fn load(path: &Path) -> Bests {
    read(path).unwrap_or_default()
}

/// The file at `path` read ([`load`]): none when there is none, an
/// error when one there cannot be read.
fn read(path: &Path) -> std::io::Result<Bests> {
    match crate::store::mover::read(path) {
        Ok(bytes) => Ok(from_text(&String::from_utf8_lossy(&bytes))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Bests::new()),
        Err(e) => Err(e),
    }
}

/// Merges `other` into `bests`: a song's higher score wins.
fn merge(bests: &mut Bests, other: Bests) {
    for (song, best) in other {
        match bests.get(&song) {
            Some(was) if was.points >= best.points => {}
            _ => {
                bests.insert(song, best);
            }
        }
    }
}

/// Writes `bests` to `path` (where it goes while the data folder moves:
/// [`crate::store::mover::current`]), durably, with every best the file held
/// already: one the player did not read (the file was still in the old
/// place, or could not be read at its start) is never written over. A file
/// there that cannot be read is left as it is (the error).
pub fn save(path: &Path, bests: &Bests) -> std::io::Result<()> {
    let mut all = read(path)?;
    merge(&mut all, bests.clone());
    durable::write_atomic(
        &crate::store::mover::current(path),
        to_text(&all).as_bytes(),
    )
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: a data folder's scores as an earlier run left them"
)]
mod tests {
    use super::*;

    fn best(points: u32) -> Best {
        Best {
            points,
            hit: points / 100,
            words: 40,
        }
    }

    /// A data folder moving from `old` to `new` (its journal in
    /// `settings`), the bests of A and B in the old one.
    fn moving(
        tag: &str,
    ) -> (
        jodel_testkit::TempDir,
        std::path::PathBuf,
        crate::store::dirs::Data,
        crate::store::dirs::Data,
    ) {
        let dir = jodel_testkit::TempDir::new(&format!("scores-{tag}"));
        let (old, new) = (
            crate::store::dirs::Data(dir.join("old")),
            crate::store::dirs::Data(dir.join("new")),
        );
        std::fs::create_dir_all(&old.0).unwrap();
        let earlier: Bests = [("A".to_owned(), best(3000)), ("B".to_owned(), best(1200))].into();
        std::fs::write(old.scores(), to_text(&earlier)).unwrap();
        let settings = dir.join("settings");
        crate::store::mover::begin(&settings, &[(old.0.clone(), new.0.clone())]).unwrap();
        (dir, settings, old, new)
    }

    /// Started again while the data folder moves (the player knows only the
    /// new place), the bests still in the old one load.
    #[test]
    fn bests_load_while_the_data_folder_moves() {
        let (_dir, _, old, new) = moving("load");
        let bests = load(&new.scores());
        assert_eq!(bests.len(), 2);
        assert_eq!(bests["A"].points, 3000);
        assert_eq!(load(&old.scores()), bests);
    }

    /// A best sung while the data folder moves is kept with every earlier
    /// one, through the move's end: even written from bests that never
    /// read the old file, nothing of it is written over.
    #[test]
    fn a_best_sung_during_a_move_survives_it_with_the_older_ones() {
        let (_dir, settings, old, new) = moving("sung");
        // C sung, and B beaten (A not), as the player holds them.
        let sung: Bests = [("C".to_owned(), best(900)), ("B".to_owned(), best(2500))].into();
        save(&new.scores(), &sung).unwrap();
        assert!(new.scores().is_file(), "written where it goes");
        crate::store::mover::finish(&settings);
        assert!(!old.0.exists(), "the old folder is gone");
        let after = load(&new.scores());
        let points = |s: &str| after[s].points;
        assert_eq!(after.len(), 3, "{after:?}");
        assert_eq!((points("A"), points("B"), points("C")), (3000, 2500, 900));
        // A lower score saved later beats nothing.
        save(&new.scores(), &[("A".to_owned(), best(10))].into()).unwrap();
        assert_eq!(load(&new.scores())["A"].points, 3000);
    }

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
