//! What the player keeps on disk between runs, and the schema of each kind
//! of file.
//!
//! | Schema | Files |
//! |---|---|
//! | [`ANALYSIS`] | a track's beats, bars and tempo (`rhythm/*.txt`) |
//! | [`WAVEFORM`] | its waveforms (`rhythm/*.wave`, binary: `JDLW` and the version byte) |
//! | [`LYRICS_TEXT`] | its lyrics as found (`lyrics/*.lrc`) |
//! | [`LYRICS_NONE`] | that none were found, and when (`lyrics/*.none`) |
//! | [`LYRICS_TIMED`] | its lyrics timed word by word (`lyrics/*.timed.lrc`) |
//! | [`LYRIC_EDITS`] | its lyrics as edited by hand (`lyric-edits/*.txt`) |
//! | [`LYRICS_EDITED`] | its lyrics as edited, timed word by word (`lyrics/*.edited.lrc`) |
//! | [`GRID`] | a beat grid fixed by hand (`grids/*.txt`) |
//! | [`SCORES`] | the best scores sung, a song each (`scores.txt`) |
//! | [`SHINE`] | the original singer's phrases measured, the shine's yardstick (`shine/*.txt`) |
//! | [`LOUDNESS`] | a song's loudness as the karaoke plays it (`loudness/*.txt`) |
//! | [`DELAYS`] | the singer's delays, an output and a microphone each (`delays.txt`, beside the settings) |
//! | [`COVER_LAYERS`] | a cover taken apart (`covers/*.layers`, binary: `JDLC` and the version byte) |
//! | [`COVER_SONG`] | which cover taken apart is a song's (`covers/songs/*.txt`) |
//! | [`COVER_ALBUM`] | an album's covers taken apart, with their perceptual hashes (`covers/albums/*.txt`) |
//! | [`PATCH`] | a game sound's patch kept on the Synth tab (`patches/*.txt`) |
//! | [`STORY_PLAN`] | a song's story planned for the karaoke (`stories/*.txt`) |
//! | [`STORY_SCENES`] | a song's story scenes: what made them, the look over them, each one's span and picture (`scenes/<song>/scenes.txt`, beside the pictures) |
//! | [`SCENE_PARTS`] | what is derived from a story scene's picture for the backdrop (`scene-parts/<picture>.parts`, binary: `JDLS` and the version byte) |
//!
//! A text file starts with a header naming its schema and the version of
//! it the file was written in, and the player that wrote it (for the
//! record only: [`Schema::header`]). A cache file without a header is not
//! read: it is as if there were none, and is made again. One of the user's
//! own ([`Schema::user_data`]) was written before there were headers, in
//! version 1.
//!
//! The contract:
//!
//! - **Read only in the version written.** A cache file is read only when
//!   its schema version is the one this player writes; any other is left
//!   alone and made again. The player's own version plays no part.
//! - **A change, a new version.** A change to what a schema's files hold,
//!   or how, raises its version here. The tests in `tests/schemas.rs` pin
//!   the files each version writes, byte for byte, and fail on a change
//!   until the version is raised and a sample of the new version added.
//! - **The same version, the same files.** While a schema's version stays,
//!   every player reads the files of it that any player wrote.
//! - **The user's work is kept.** Grids fixed by hand, the best scores
//!   sung, the delays measured, the lyrics edited and the patches kept are
//!   not a cache ([`Schema::user_data`]): a new version of [`GRID`],
//!   [`SCORES`], [`DELAYS`], [`LYRIC_EDITS`] or [`PATCH`] still reads every
//!   older one, a file without a header (from before there were headers)
//!   as version 1.
//!
//! What a file says may also depend on how it was made: an analysis on
//! [`jodel_rhythm::VERSION`] (in the name of its file, so another one is
//! made afresh), timed lyrics on their aligner (named in the file, and a
//! timing by another than the one the player times with now is timed
//! again: `crate::work::lyrics::WordTimer::maker`), a cover taken apart on its
//! models and the way it is taken apart (`crate::work::cover::VERSION`, in the
//! file, compared when it is read), a story's plan on its language model and
//! what it was asked (`crate::work::story::VERSION` and the prompt's hash, in
//! the file, compared when it is read), a song's scenes' parts on theirs
//! (`crate::work::scenes::VERSION`, in the file, compared when it is read, and
//! each picture's hash). That is apart from the schema: a
//! better aligner raises its own version (`jodel_lyrics::VERSION`,
//! `jodel_lyrics::speech::maker`), not the schema's.

/// A kind of file kept on disk, and the version of its layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schema {
    /// Its name.
    pub name: &'static str,
    /// The version this player writes (and reads).
    pub version: u32,
    /// The user's own work, not a cache: a file of it without a header
    /// was written before there were headers, and is read as version 1
    /// (a cache's is not read: it is made again).
    pub user_data: bool,
}

/// A track's analysis: beats, bars, groupings, regions and candidates,
/// and its key. Version 1 had no key.
pub const ANALYSIS: Schema = Schema {
    name: "analysis",
    version: 2,
    user_data: false,
};

/// A track's waveforms (binary, versioned by its own byte).
pub const WAVEFORM: Schema = Schema {
    name: "waveform",
    version: jodel_rhythm::waveform::FORMAT as u32,
    user_data: false,
};

/// Lyrics as found: their words, a line each, the source in a
/// `[source:…]` tag, and when the source synced a line, a
/// `[#:synced START END]` line after it (the scaffold the aligner holds
/// the line near: `jodel_lyrics::scaffold`). Version 1 kept the times they
/// came with as the lines' own; version 2 kept no times at all.
pub const LYRICS_TEXT: Schema = Schema {
    name: "lyrics",
    version: 3,
    user_data: false,
};

/// That no lyrics were found (or the track has no words), and when.
pub const LYRICS_NONE: Schema = Schema {
    name: "lyrics-none",
    version: 1,
    user_data: false,
};

/// Lyrics timed word by word by us: Enhanced LRC. Version 1 was held to
/// the line times the text came with; version 2 kept no letters' times
/// (`[#:letters ...]`); version 3's text was not checked against what is sung
/// (`jodel_lyrics::prepare::fitting`), and version 4 kept no fit (`[fit:...]`).
pub const LYRICS_TIMED: Schema = Schema {
    name: "lyrics-timed",
    version: 5,
    user_data: false,
};

/// A song's lyrics as edited by hand ([`crate::work::lyrics::edits`]): its text
/// and, word by word, the times set by hand and the marks.
pub const LYRIC_EDITS: Schema = Schema {
    name: "lyric-edits",
    version: 1,
    user_data: true,
};

/// A song's lyrics as edited, timed word by word by us with the words set
/// by hand held (`crate::work::lyrics::edits`): the edits' stamp
/// (`crate::work::lyrics::edits::stamp`), then Enhanced LRC as
/// [`LYRICS_TIMED`] has it.
pub const LYRICS_EDITED: Schema = Schema {
    name: "lyrics-edited",
    version: 1,
    user_data: false,
};

/// A beat grid fixed by hand.
pub const GRID: Schema = Schema {
    name: "grid",
    version: 1,
    user_data: true,
};

/// The best scores sung ([`crate::store::scores`]). Version 1 kept one a level
/// (there were three).
pub const SCORES: Schema = Schema {
    name: "scores",
    version: 2,
    user_data: true,
};

/// The original singer's phrases as the shine holds a singer to them
/// ([`crate::work::shine`]); what they hold is versioned apart, by
/// [`crate::core::shine::reference::VERSION`] in the file.
pub const SHINE: Schema = Schema {
    name: "shine",
    version: 1,
    user_data: false,
};

/// A song's loudness as the karaoke plays it ([`crate::work::loudness`]): its
/// integrated loudness, and what it was measured from (the stems and the
/// voice's level, or the file).
pub const LOUDNESS: Schema = Schema {
    name: "loudness",
    version: 1,
    user_data: false,
};

/// The singer's delays measured and nudged ([`crate::store::delays`]).
pub const DELAYS: Schema = Schema {
    name: "delays",
    version: 2,
    user_data: true,
};

/// A cover taken apart: what made it and its palette, then its plate,
/// subject, depth, plate's depth and rim as PNGs (binary, versioned by its
/// own byte).
pub const COVER_LAYERS: Schema = Schema {
    name: "cover-layers",
    version: 1,
    user_data: false,
};

/// A song's cover: which cover taken apart is its (or none).
pub const COVER_SONG: Schema = Schema {
    name: "cover-song",
    version: 1,
    user_data: false,
};

/// An album's covers taken apart: each one's pixels' hash and perceptual
/// hash.
pub const COVER_ALBUM: Schema = Schema {
    name: "cover-album",
    version: 1,
    user_data: false,
};

/// A game sound's patch as kept on the Synth tab ([`crate::store::patches`]):
/// its theme and atom, its engine, then each parameter that applies by
/// its name.
pub const PATCH: Schema = Schema {
    name: "patch",
    version: 1,
    user_data: true,
};

/// A song's story planned for the karaoke ([`crate::work::story`]): what
/// made it and what it was asked (compared when it is read), then the
/// plan as JSON, or why the song has none.
pub const STORY_PLAN: Schema = Schema {
    name: "story-plan",
    version: 1,
    user_data: false,
};

/// A song's story scenes as the story's drawing step leaves them: what
/// made them (a hash of what they hold), the song's look over them, then a
/// line a scene, its span and its picture. Version 1 had neither.
pub const STORY_SCENES: Schema = Schema {
    name: "story-scenes",
    version: 2,
    user_data: false,
};

/// What is derived from a story scene's picture: what made it, the
/// picture's hash, its split, heads and palette, then its nearness and (in
/// layers) its layers' shares and plate as PNGs (binary, versioned by its
/// own byte).
pub const SCENE_PARTS: Schema = Schema {
    name: "scene-parts",
    version: 1,
    user_data: false,
};

/// Every schema.
pub const ALL: [Schema; 19] = [
    ANALYSIS,
    WAVEFORM,
    LYRICS_TEXT,
    LYRICS_NONE,
    LYRICS_TIMED,
    LYRIC_EDITS,
    LYRICS_EDITED,
    GRID,
    SCORES,
    SHINE,
    LOUDNESS,
    DELAYS,
    COVER_LAYERS,
    COVER_SONG,
    COVER_ALBUM,
    PATCH,
    STORY_PLAN,
    STORY_SCENES,
    SCENE_PARTS,
];

/// The first word of a header.
const MARK: &str = "jodeldiplom-cache";

/// The player that writes the files.
pub const PLAYER: &str = concat!("jodeldiplom ", env!("CARGO_PKG_VERSION"));

impl Schema {
    /// The header of a file of this schema: `jodeldiplom-cache analysis 1
    /// by jodeldiplom 0.1.0`.
    pub fn header(&self) -> String {
        format!("{MARK} {} {} by {PLAYER}", self.name, self.version)
    }

    /// `body` with the header in front.
    pub fn wrap(&self, body: &str) -> String {
        format!("{}\n{body}", self.header())
    }

    /// The version of this schema a file's `text` was written in, and the
    /// rest of it; `None` for a file of another schema. A file without a
    /// header is version 1 of the user's own work, and not read as a cache
    /// ([`Schema::user_data`]).
    pub fn open<'a>(&self, text: &'a str) -> Option<(u32, &'a str)> {
        let Some(after) = text.strip_prefix(MARK) else {
            return self.user_data.then_some((1, text));
        };
        let (first, rest) = after.split_once('\n').unwrap_or((after, ""));
        let mut words = first.split_whitespace();
        if words.next()? != self.name {
            return None;
        }
        let version = words.next()?.parse().ok()?;
        Some((version, rest))
    }

    /// The rest of a file's `text` after its header, when it was written in
    /// this player's version of the schema.
    pub fn body<'a>(&self, text: &'a str) -> Option<&'a str> {
        let (version, rest) = self.open(text)?;
        (version == self.version).then_some(rest)
    }
}

/// A binary cache file of `schema` (the covers' layers, the story scenes'
/// parts): `magic`, the schema's version byte, then each of `parts` after
/// its length (four bytes, little-endian).
pub fn frame(magic: &[u8; 4], schema: &Schema, parts: &[&[u8]]) -> Vec<u8> {
    let mut out = magic.to_vec();
    out.push(schema.version as u8);
    for part in parts {
        out.extend((part.len() as u32).to_le_bytes());
        out.extend(*part);
    }
    out
}

/// The length of a [`frame`]d file's first part, from its first 9 bytes,
/// when it is of `magic` and `schema`'s version.
pub fn first_len(head: &[u8], magic: &[u8; 4], schema: &Schema) -> Option<usize> {
    (head.len() >= 9 && &head[..4] == magic && u32::from(head[4]) == schema.version)
        .then(|| u32::from_le_bytes([head[5], head[6], head[7], head[8]]) as usize)
}

/// The parts of a [`frame`]d file of `magic` and `schema`'s version, when
/// it is whole.
pub fn unframe<'a>(bytes: &'a [u8], magic: &[u8; 4], schema: &Schema) -> Option<Vec<&'a [u8]>> {
    first_len(bytes, magic, schema)?;
    let mut parts = Vec::new();
    let mut at = 5;
    while at < bytes.len() {
        let len = u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?) as usize;
        parts.push(bytes.get(at + 4..at + 4 + len)?);
        at += 4 + len;
    }
    Some(parts)
}

/// FNV-1a, 64 bits, of `text`: what the caches name their files by
/// ([`name`]).
pub fn hash(text: &str) -> u64 {
    jodel_dsp::fnv1a(text.as_bytes())
}

/// What `text` is named by in a cache: its [`hash`] in 16 hex figures.
pub fn name(text: &str) -> String {
    format!("{:016x}", hash(text))
}

/// A file as the caches know it, `PATH SIZE NANOS`: its path, size and
/// modification time (nanoseconds since 1970); the start of its
/// analysis's key (`crate::work::rhythm::cache_key`) and what its lyrics and
/// stems are kept by (`crate::work::lyrics::store::key_of`,
/// `crate::store::stems::file_of`). None when it cannot be read.
pub fn identity(path: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Some(format!("{} {} {modified}", path.display(), meta.len()))
}

/// The file in `dir` that keeps `what` of the song at `path` (whether or
/// not it is there): named by the hash of the song's [`identity`] and
/// `what`, so a file changed is looked up afresh; `None` when the song
/// cannot be read.
pub fn file_for(
    dir: &std::path::Path,
    path: &std::path::Path,
    what: &str,
) -> Option<std::path::PathBuf> {
    let key = format!("{} {what}", identity(path)?);
    Some(dir.join(format!("{}.txt", name(&key))))
}

/// An [`identity`]'s parts: the path, the size and the modification time.
pub fn identity_parts(identity: &str) -> Option<(&std::path::Path, u64, u128)> {
    let mut parts = identity.rsplitn(3, ' ');
    let modified = parts.next()?.parse().ok()?;
    let size = parts.next()?.parse().ok()?;
    Some((std::path::Path::new(parts.next()?), size, modified))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_names_the_schema_its_version_and_the_player() {
        // A schema's first version, as the analysis's was.
        let first = Schema {
            name: "analysis",
            version: 1,
            user_data: false,
        };
        let text = first.wrap("body\n");
        assert!(text.starts_with("jodeldiplom-cache analysis 1 by jodeldiplom "));
        assert_eq!(first.body(&text), Some("body\n"));
        // Another schema's, another version's: not read.
        assert_eq!(GRID.body(&text), None);
        let later = Schema {
            name: "analysis",
            version: 2,
            user_data: false,
        };
        assert_eq!(later.body(&text), None);
        assert_eq!(later.open(&text), Some((1, "body\n")));
        // No header: a cache's is not read; the user's own is version 1,
        // as written before there were headers.
        assert_eq!(first.open("jodel-rhythm 9\n"), None);
        assert_eq!(first.body("jodel-rhythm 9\n"), None);
        assert_eq!(GRID.open("jodel-grid 1\n"), Some((1, "jodel-grid 1\n")));
        let users: Vec<&str> = ALL.iter().filter(|s| s.user_data).map(|s| s.name).collect();
        assert_eq!(users, ["lyric-edits", "grid", "scores", "delays", "patch"]);
        // Another player wrote it: read all the same.
        let other = "jodeldiplom-cache analysis 1 by jodeldiplom 9.9.9\nbody\n";
        assert_eq!(first.body(other), Some("body\n"));
    }

    #[test]
    fn the_schemas_have_names_of_their_own() {
        let mut names: Vec<&str> = ALL.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
    }
}
