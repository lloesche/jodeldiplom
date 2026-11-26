//! The lanes a track is drawn in, as the Transition and Karaoke tabs show
//! them: which track a block of lanes shows, the lanes in it, its analysis
//! and its lyrics, and how the lyrics' words are laid out along a lane.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_playback::Stem;

use super::{Rhythm, Rhythmed, State};

/// A block of lanes: a track, by where it is in the queue from the one
/// playing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Block {
    /// The track playing.
    #[default]
    Now,
    /// The next one in the queue.
    Next,
    /// The one before the playing track: shown only while it slides away
    /// after a transition into the playing one
    /// ([`crate::core::transition::viewport::slide`]).
    Prev,
}

impl Block {
    /// The two the keys act on, top first.
    pub const BOTH: [Block; 2] = [Block::Now, Block::Next];
}

/// A lane of a block: the whole song or one stem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// The whole song.
    Song,
    /// One stem.
    Stem(Stem),
}

impl Lane {
    /// The lanes with stems, top first.
    pub const ALL: [Lane; 5] = [
        Lane::Song,
        Lane::Stem(Stem::Drums),
        Lane::Stem(Stem::Bass),
        Lane::Stem(Stem::Other),
        Lane::Stem(Stem::Vocals),
    ];

    /// Its label.
    pub fn label(self) -> &'static str {
        match self {
            Lane::Song => "song",
            Lane::Stem(stem) => stem.id(),
        }
    }

    /// Its place among a waveform's stems ([`Stem::ALL`]'s order: bass,
    /// drums, other, vocals); `None` for the whole song.
    pub fn stem_index(self) -> Option<usize> {
        match self {
            Lane::Song => None,
            Lane::Stem(stem) => Stem::ALL.iter().position(|&s| s == stem),
        }
    }
}

/// The track a block shows: the one playing, the one after it, or the one
/// before it.
pub fn path(state: &State, block: Block) -> Option<&PathBuf> {
    state.queue.paths.get(index(state, block)?)
}

/// The queue index of the track a block shows (it may be past the queue).
pub fn index(state: &State, block: Block) -> Option<usize> {
    let index = state.playing.track.as_ref()?.index;
    match block {
        Block::Now => Some(index),
        Block::Next => Some(index + 1),
        Block::Prev => index.checked_sub(1),
    }
}

/// The analysis and waveform of a block's track, once in.
pub fn rhythmed(state: &State, block: Block) -> Option<&Arc<Rhythmed>> {
    match state.rhythm.get(path(state, block)?)? {
        Rhythm::Ready(r) => Some(r),
        _ => None,
    }
}

/// The lyrics of a block's track, once found.
pub fn lyrics(state: &State, block: Block) -> Option<Arc<jodel_lyrics::Lyrics>> {
    use super::karaoke::{lyrics_of, SongLyrics};
    match index(state, block).and_then(|i| lyrics_of(state, i)) {
        Some(SongLyrics::Found { shown, .. }) => Some(Arc::clone(shown)),
        _ => None,
    }
}

/// The timed words of `lyrics`, in order.
pub fn timed_words(lyrics: &jodel_lyrics::Lyrics) -> Vec<&jodel_lyrics::Word> {
    lyrics
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .filter(|w| w.is_timed())
        .collect()
}

/// The words of `lyrics` a zoom is fitted to: those timed where they were
/// heard (or given their times by the source), not those the aligner did
/// not hear and timed between their neighbours, often squeezed into a gap
/// (an echo of backing words in a fifth of a second would set the zoom of
/// the whole song). They are still drawn where there is room. All the
/// timed words when none were heard.
pub fn fitted_words(lyrics: &jodel_lyrics::Lyrics) -> Vec<&jodel_lyrics::Word> {
    let timed = timed_words(lyrics);
    let heard: Vec<&jodel_lyrics::Word> = timed
        .iter()
        .copied()
        .filter(|w| w.confidence >= jodel_lyrics::CONFIDENT)
        .collect();
    if heard.is_empty() {
        timed
    } else {
        heard
    }
}

/// The most cells a second a lane's words are laid out at to fit them:
/// faster, they would scroll by too fast to read.
pub const LYRICS_MOST_PER: f64 = 40.0;
/// The least (a lane over a song's quiet stretch shows no more than this).
pub const LYRICS_LEAST_PER: f64 = 3.0;

/// The lines a lane's words go on, and whether a word with no room is
/// left out or goes on later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordLines {
    /// Lines of words: the first, then each under the one before.
    pub rows: usize,
    /// A word with no room anywhere near where it is sung still goes in,
    /// after the one before (late), rather than being left out.
    pub keep_all: bool,
}

impl WordLines {
    /// One line; words with no room left out (the Transition tab's lanes).
    pub const ONE: WordLines = WordLines {
        rows: 1,
        keep_all: false,
    };
    /// One line; every word goes in, each where it is sung (at the zoom
    /// that fits them all), but for words sung at the same moment, which
    /// go on after the one before (the Karaoke tab's voice lane).
    pub const KARAOKE: WordLines = WordLines {
        rows: 1,
        keep_all: true,
    };
}

/// Lays `words` (timed, in order: the whole song's) out along a lane: each
/// at `x_of(its start)` (the lane's units, cells or pixels, from a fixed
/// origin), a character `cell` units wide at `per` units a second, as wide
/// as it is sung (up to the next word's start, never cut below its
/// characters), on the first of `lines.rows` lines where its place is free
/// (a word crowding the one before on the first line goes under it, where
/// it is sung). Where it is taken on every line, it goes after the word on
/// the line that frees first (a character between them), or is left out
/// once that is past where it ends (unless `lines.keep_all`).
/// `place(k, at, wide, row)` is given each word laid out (its index, where
/// it starts, its width in characters and its line) and says whether to go
/// on. Returns how many were left out (or, with `keep_all`, went in late).
/// The words are laid out from the song's start, so where one stands
/// never depends on which others are in view.
pub fn lay_out_words(
    words: &[&jodel_lyrics::Word],
    x_of: &dyn Fn(f64) -> f64,
    per: f64,
    cell: f64,
    lines: WordLines,
    place: &mut dyn FnMut(usize, f64, usize, usize) -> bool,
) -> usize {
    let rows = lines.rows.clamp(1, 4);
    let chars_per_second = per / cell.max(1e-9);
    let mut free = [f64::NEG_INFINITY; 4];
    let mut left_out = 0;
    for (k, w) in words.iter().enumerate() {
        let (Some(start), Some(end)) = (w.start, w.end) else {
            continue;
        };
        let letters = w.text.chars().count();
        if letters == 0 {
            continue;
        }
        let x = x_of(start);
        // The first line free where it is sung, else the one that frees
        // first.
        let (row, at) = match (0..rows).find(|&r| free[r] <= x) {
            Some(r) => (r, x),
            None => {
                let r = (0..rows)
                    .min_by(|&a, &b| free[a].total_cmp(&free[b]))
                    .unwrap_or(0);
                (r, free[r])
            }
        };
        if at > x_of(end) + cell {
            left_out += 1;
            if !lines.keep_all {
                continue;
            }
        }
        let next = words
            .get(k + 1)
            .and_then(|n| n.start)
            .map_or(f64::INFINITY, x_of);
        let room = ((next - at) / cell - 1.0).floor().clamp(0.0, 1e6) as usize;
        let sung = ((end - start) * chars_per_second).floor().max(0.0) as usize;
        let wide = sung.min(room).max(letters);
        free[row] = at + (wide + 1) as f64 * cell;
        if !place(k, at, wide, row) {
            break;
        }
    }
    left_out
}

/// How many of `words` a lane of `rows` lines leaves out at `per` cells a
/// second.
pub fn words_left_out(words: &[&jodel_lyrics::Word], per: f64, rows: usize) -> usize {
    let lines = WordLines {
        rows,
        keep_all: false,
    };
    lay_out_words(
        words,
        &|t| (t * per).floor(),
        per,
        1.0,
        lines,
        &mut |_, _, _, _| true,
    )
}

/// The fewest cells a second at which a lane of `rows` lines leaves out no
/// more of `words` than it must (those it leaves out even at
/// [`LYRICS_MOST_PER`]: words sung at the same moment), a little more for
/// room; `None` without words.
pub fn lyrics_per_second(words: &[&jodel_lyrics::Word], rows: usize) -> Option<f64> {
    if words.is_empty() {
        return None;
    }
    let least = words_left_out(words, LYRICS_MOST_PER, rows);
    let fits = |per: f64| words_left_out(words, per, rows) <= least;
    if fits(LYRICS_LEAST_PER) {
        return Some(LYRICS_LEAST_PER);
    }
    let (mut lo, mut hi) = (LYRICS_LEAST_PER, LYRICS_MOST_PER);
    while hi / lo > 1.01 {
        let mid = (lo * hi).sqrt();
        if fits(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let roomy = (hi * 1.1).min(LYRICS_MOST_PER);
    Some(if fits(roomy) { roomy } else { hi })
}
