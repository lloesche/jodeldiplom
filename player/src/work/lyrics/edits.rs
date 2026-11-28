//! A song's lyrics as edited by hand, kept as the user's own work: in
//! `lyric-edits` in the data folder, beside the lyrics' cache, never
//! forgotten with it (`--clean-cache`).
//!
//! The edits are the whole text as edited (a word or a line replaced, put
//! in or taken out: the edited text takes the place of the one found), and
//! for each word what was set by hand ([`jodel_lyrics::WordEdit`]): its
//! start and end moved, marked as not to be sung, the voice's level over
//! it. They are kept as a [`Lyrics`] with only those times
//! ([`Lyrics::edits`]); here are their file and where it is kept.
//!
//! A file a song, named by a hash of its path and size (as the grids fixed
//! by hand are: a file whose tags are saved again keeps its edits),
//! [`cache::LYRIC_EDITS`]:
//!
//! ```text
//! jodeldiplom-cache lyric-edits 1 by jodeldiplom 0.1.0
//! key /music/a song.flac 31415926
//! source lrclib
//! line
//! word - - - Hold
//! word 12.340000 12.900000 - me
//! word - - not-sung (close)
//! break
//! line
//! word 20.000000 20.500000 not-sung,voice:-3 Never
//! ```
//!
//! `key` is the song's path and size; `source` where the text was found
//! ([`Source::name`]); then the lines in order, `line` for a sung one and
//! `break` for a break in the singing, each `word` after its line as
//! `word START END MARKS TEXT`: the times set by hand in seconds (`- -`:
//! none), `MARKS` `-` or `not-sung` and `voice:DB` (dB against the voice as
//! recorded) joined by a comma, and the word as written (no white space).
//!
//! The song is timed again with its edits whenever they change, the words
//! set by hand held where they were set and the rest fitted around them;
//! that timing is a cache (`{key}.edited.lrc` in the lyrics' cache,
//! [`cache::LYRICS_EDITED`]), named by the edits it was made with
//! ([`Lyrics::edits_stamp`]), and made again by a new aligner as any
//! timing is.

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

use jodel_lyrics::{Line, Lyrics, Source, Timing, Word};

use crate::store::cache;

/// What finds a song's edits: its path and size.
fn key(path: &Path) -> Option<String> {
    let size = std::fs::metadata(path).ok()?.len();
    Some(format!("{} {size}", path.display()))
}

/// `edits` as the file for the song `key` names holds them, in
/// [`cache::LYRIC_EDITS`]'s version.
pub fn to_text(key: &str, edits: &Lyrics) -> String {
    let mut s = format!(
        "key {}\nsource {}\n",
        key.replace('\n', " "),
        edits.source.name()
    );
    let time = |t: Option<f64>| t.map_or_else(|| "-".to_owned(), |t| format!("{t:.6}"));
    for line in &edits.lines {
        s.push_str(if line.is_break { "break\n" } else { "line\n" });
        for w in &line.words {
            let (start, end) = w.edit.pin.map_or((None, None), |(a, b)| (Some(a), Some(b)));
            let mut marks = Vec::new();
            if w.edit.not_sung {
                marks.push("not-sung".to_owned());
            }
            if let Some(db) = w.edit.voice_db {
                marks.push(format!("voice:{db}"));
            }
            let marks = if marks.is_empty() {
                "-".to_owned()
            } else {
                marks.join(",")
            };
            let text: String = w.text.split_whitespace().collect::<Vec<_>>().join("_");
            let _ = writeln!(s, "word {} {} {marks} {text}", time(start), time(end));
        }
    }
    cache::LYRIC_EDITS.wrap(&s)
}

/// The edits in a file's `text`, if it was written for the song `key` (in
/// any version of [`cache::LYRIC_EDITS`] up to this player's: the user's
/// work is kept).
pub fn from_text(key: &str, text: &str) -> Option<Lyrics> {
    let (version, body) = cache::LYRIC_EDITS.open(text)?;
    if version > cache::LYRIC_EDITS.version {
        return None;
    }
    let mut lines = body.lines();
    if lines.next()?.strip_prefix("key ")? != key.replace('\n', " ") {
        return None;
    }
    let source = Source::parse(lines.next()?.strip_prefix("source ")?)?;
    let mut out = Lyrics {
        lines: Vec::new(),
        source,
        timing: Timing::default(),
    };
    for line in lines {
        match line {
            "line" | "break" => out.lines.push(Line {
                is_break: line == "break",
                ..Line::default()
            }),
            _ => {
                let mut fields = line.strip_prefix("word ")?.splitn(4, ' ');
                let (start, end, marks, text) = (
                    fields.next()?,
                    fields.next()?,
                    fields.next()?,
                    fields.next()?,
                );
                if text.is_empty() {
                    return None;
                }
                let mut w = Word::new(text);
                if start != "-" || end != "-" {
                    let (a, b): (f64, f64) = (start.parse().ok()?, end.parse().ok()?);
                    if !(a.is_finite() && b.is_finite() && a <= b) {
                        return None;
                    }
                    w.pin_to(a, b);
                }
                for mark in marks.split(',').filter(|m| *m != "-") {
                    match mark.split_once(':') {
                        None if mark == "not-sung" => w.edit.not_sung = true,
                        Some(("voice", db)) => w.edit.voice_db = Some(db.parse().ok()?),
                        _ => return None,
                    }
                }
                out.lines.last_mut()?.words.push(w);
            }
        }
    }
    for line in &mut out.lines {
        line.mark_backing();
    }
    Some(out)
}

/// The songs' edits, kept in a folder.
#[derive(Clone, Debug)]
pub struct EditStore {
    /// The folder (`lyric-edits` in the data folder:
    /// [`crate::store::dirs::Data::lyric_edits`]).
    pub dir: PathBuf,
}

impl EditStore {
    /// The edits kept in `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> EditStore {
        EditStore { dir: dir.into() }
    }

    /// The file of the edits kept under `key`.
    fn file(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{}.txt", cache::name(key)))
    }

    /// The edits of the song at `path`, if it has any (where they are now:
    /// while the data folder moves, not yet moved is where it was).
    pub fn load(&self, path: &Path) -> Option<Lyrics> {
        let key = key(path)?;
        let bytes = crate::store::mover::read(&self.file(&key)).ok()?;
        from_text(&key, &String::from_utf8(bytes).ok()?)
    }

    /// Keeps `edits` as the song at `path`'s, whole or not at all; `None`
    /// forgets its edits (its text as found, timed by the aligner again).
    pub fn save(&self, path: &Path, edits: Option<&Lyrics>) -> io::Result<()> {
        let key = key(path).ok_or_else(|| io::Error::other("the song cannot be read"))?;
        let file = self.file(&key);
        match edits {
            Some(edits) => durable::write_atomic(
                &crate::store::mover::current(&file),
                to_text(&key, edits).as_bytes(),
            ),
            None => crate::store::mover::remove(&file),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edits as the editor leaves them: a word moved, one not to be sung,
    /// one with the voice set over it, a break.
    fn sample() -> Lyrics {
        let mut l = Lyrics::parse_text("Hold me (close)\n♪\nNever let go\n", Source::Lrclib);
        l.lines[0].words[1].pin_to(12.34, 12.9);
        l.lines[0].words[2].edit.not_sung = true;
        let w = &mut l.lines[2].words[0];
        w.pin_to(20.0, 20.5);
        (w.edit.not_sung, w.edit.voice_db) = (true, Some(-3.0));
        l.edits()
    }

    #[test]
    fn edits_read_back_as_they_were_written() {
        let edits = sample();
        assert!(edits.lines[1].is_break);
        assert!(edits.lines[0].words[2].backing);
        let text = to_text("/music/a.flac 1", &edits);
        assert_eq!(from_text("/music/a.flac 1", &text), Some(edits.clone()));
        // Another song's file is not this one's.
        assert_eq!(from_text("/music/b.flac 1", &text), None);
    }

    /// Edits kept are found again, also while the data folder moves (where
    /// they still are until they come over), new ones go where the folder
    /// goes, and edits forgotten are gone.
    #[test]
    fn edits_are_kept_and_found_while_the_data_folder_moves() {
        let dir = jodel_testkit::TempDir::new("edits");
        let song = dir.join("music/a song.flac");
        durable::write_atomic(&song, b"not music").unwrap();
        let (old, new) = (dir.join("old"), dir.join("new"));
        let store = EditStore::new(old.join("lyric-edits"));
        assert_eq!(store.load(&song), None);
        let edits = sample();
        store.save(&song, Some(&edits)).unwrap();
        assert_eq!(store.load(&song), Some(edits.clone()));
        // Another song: none.
        let other = dir.join("music/b.flac");
        durable::write_atomic(&other, b"other").unwrap();
        assert_eq!(store.load(&other), None);
        // The data folder moves: found where they are, then where they went.
        let settings = dir.join("settings");
        crate::store::mover::begin(&settings, &[(old.clone(), new.clone())]).unwrap();
        assert_eq!(store.load(&song), Some(edits.clone()));
        crate::store::mover::finish(&settings);
        let moved = EditStore::new(new.join("lyric-edits"));
        assert_eq!(moved.load(&song), Some(edits.clone()));
        assert_eq!(store.load(&song), Some(edits.clone()));
        // New edits go where the folder is now; forgotten, gone.
        let mut more = edits.clone();
        more.lines[0].words[0].edit.not_sung = true;
        store.save(&song, Some(&more)).unwrap();
        assert_eq!(moved.load(&song), Some(more));
        store.save(&song, None).unwrap();
        assert_eq!(store.load(&song), None);
        assert_eq!(moved.load(&song), None);
    }
}
