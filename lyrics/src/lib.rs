//! `jodel-lyrics`: the lyrics of a song for karaoke, and when each word is
//! sung.
//!
//! # What is here
//!
//! - The model: [`Lyrics`] is [`Line`]s of [`Word`]s, each word with an
//!   optional start and end and a confidence, with a record of where the
//!   text came from ([`Source`]) and who timed it ([`Timing`]).
//! - [`lrc`]: plain text, LRC (line times) and Enhanced LRC (word times)
//!   in, Enhanced LRC out. Section tags (`[Chorus]`, `Verse 2:`) are
//!   dropped; breaks (`♪`, `[Instrumental]`, an empty timed line) are
//!   kept as one mark, written `♪`; words in parentheses are kept as
//!   backing words, which the singer may leave out.
//! - [`tokens`]: the words as a speech model's letters (upper case, no
//!   diacritics, words joined by `|`), in the vocabulary of
//!   `facebook/wav2vec2-base-960h`.
//! - [`ctc`]: forced alignment of those letters to a model's per-frame
//!   log-probabilities (a Viterbi path with blanks), with skippable
//!   backing words and what else is known of the voice as costs.
//! - [`scaffold`]: a synced text's line times (LRCLIB's) laid on the text
//!   timed, checked against the song aligned without them, and, where they
//!   fit, the lines' words held near them.
//! - [`aligner`]: it put together, on a speech model's emissions
//!   ([`aligner::Acoustic`]): the [`Aligner`] the player uses.
//! - [`spread`]: timing without a model: line times split across words by
//!   syllables, plain text spread over where the voice is heard, and the
//!   filling-in of words whose times were not trusted.
//! - [`sources`]: the text from a file's tags or a `.lrc` file beside it;
//!   [`lrclib`]: the LRCLIB web service's addresses and answers (the
//!   request itself is the caller's).
//! - [`json`]: a small JSON reader for those answers.
//! - [`words`]: words as they are compared, the lyrics' and a
//!   recogniser's, and the words two texts have in common, in order.
//! - [`fit`]: whether a text fits a recording, its words against what the
//!   recogniser ([`asr`]) heard.
//! - [`speech`]: the speech model on the CPU, `wav2vec2-base-960h` from
//!   its `model.safetensors`, giving the CTC emissions [`ctc`] aligns.
//!   It matches PyTorch's logits to about 1e-4 of their range; the GPU's
//!   version (`jodel_gpu::GpuSpeech`) matches it as closely, 25 times as fast.
//! - [`asr`]: a speech recogniser on the CPU, `Qwen3-ASR-0.6B` from its
//!   `model.safetensors`: what is sung, as text. It gets about a third of
//!   a song's words wrong, so it does not replace fetched lyrics, but it
//!   tells whether they fit the recording and where their lines are. It
//!   matches the official model's logits to about 1e-5 and writes the same
//!   tokens. [`asr::quant`] is its 4-bit weights' format, which the GPU's
//!   version (`jodel_gpu::GpuAsr`) runs on, 20 times as fast.
//!
//! The two interfaces, [`LyricsSource`] (text for a track) and
//! [`Aligner`] (word times for a text against the voice), leave room for a
//! transcribing source and other aligners later. Every result says which
//! source and which aligner, at which version, made it, so a better one
//! can time a song again.
//!
//! No dependencies.

#![warn(missing_docs)]

use std::path::PathBuf;
use std::sync::Arc;

pub mod aligner;
pub mod anchors;
pub mod asr;
pub mod ctc;
pub mod fit;
pub mod glimpses;
pub mod json;
pub mod lines;
pub mod lrc;
pub mod lrclib;
mod nn;
pub mod repeats;
pub mod scaffold;
pub mod sources;
pub mod speech;
pub mod spread;
pub mod tokens;
pub mod words;

/// The version of the timing in this crate. A result made by an older one
/// can be timed again. 2: the letters' times are kept ([`Word::letters`]).
/// 3: a word's letters are kept from bridging a pause in the voice
/// ([`ctc::CtcSettings::inside_word`]). 4: the words pinned where a
/// recogniser heard them, the opening pulled together
/// ([`ctc::CtcSettings::window_penalty`], [`ctc::CtcSettings::opening_pull`]).
/// 5: the repeats a recogniser heard and the text leaves out put in
/// ([`repeats`]). 6: the recogniser's glimpses: a line's start where they
/// heard it, no line spread over a stretch where they heard no words
/// ([`glimpses`]); a line's first words heard weakly at a sung pace
/// before the first heard surely ([`spread::interpolate`]). 7: the lines held
/// near their source's synced times where those fit the recording
/// ([`scaffold`]). 8: a file's synced lyrics tag kept as the scaffold of
/// its plain one ([`sources::found_from`]); a text fits on one scale
/// ([`fit::fits`]).
pub const VERSION: u32 = 8;

/// The rate the aligners take the voice at, in Hz.
pub const VOCALS_RATE: u32 = 16_000;

/// Words with at least this confidence keep their own times; others are
/// timed between their neighbours, and shown at the line's pace.
pub const CONFIDENT: f32 = 0.4;

/// A word's confidence when its times were given by the source (an
/// Enhanced LRC file): full.
pub const GIVEN: f32 = 1.0;

/// One word of a line, as written (punctuation included).
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// The word as written, with its punctuation and parentheses.
    pub text: String,
    /// When it starts, in seconds from the start of the track.
    pub start: Option<f64>,
    /// When it ends.
    pub end: Option<f64>,
    /// How sure its times are, 0 to 1: [`GIVEN`] for times from the
    /// source, the mean frame posterior for aligned words, 0 for times
    /// spread or filled in between neighbours.
    pub confidence: f32,
    /// Whether it is inside parentheses: a backing part, which the voice
    /// may not carry.
    pub backing: bool,
    /// When each character of `text` is sung, in seconds after `start`, as
    /// the aligner heard its letters (one a character; one not sung, such
    /// as punctuation, at the letter's before it); empty when not known
    /// (the word's times given or spread rather than heard).
    pub letters: Vec<f32>,
}

impl Word {
    /// An untimed word.
    pub fn new(text: impl Into<String>) -> Word {
        Word {
            text: text.into(),
            start: None,
            end: None,
            confidence: 0.0,
            backing: false,
            letters: Vec::new(),
        }
    }

    /// When character `i` of the word is sung (seconds into the track):
    /// its letter's time, else spread evenly over the word; `None` for an
    /// untimed word.
    pub fn letter_at(&self, i: usize) -> Option<f64> {
        let (start, end) = (self.start?, self.end?);
        match self.letters.get(i) {
            Some(&offset) => Some(start + f64::from(offset)),
            None => {
                let n = self.text.chars().count().max(1);
                Some(start + (end - start) * i as f64 / n as f64)
            }
        }
    }

    /// Whether it has both times.
    pub fn is_timed(&self) -> bool {
        self.start.is_some() && self.end.is_some()
    }
}

/// One line of the lyrics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Line {
    /// Its words; none for a break.
    pub words: Vec<Word>,
    /// When the line starts, as an LRC file gives it.
    pub time: Option<f64>,
    /// When the line is sung as the text's source synced it (LRCLIB's
    /// synced lyrics, an LRC file): kept with a found text as a scaffold
    /// for the aligner, which holds the line's words near it only where it
    /// fits the recording ([`scaffold`]). Never shown, never a line's time.
    pub synced: Option<Synced>,
    /// Whether the whole line is in parentheses.
    pub backing: bool,
    /// Whether it marks a break in the singing rather than being sung: an
    /// empty timed line, a note (`♪`), `[Instrumental]`, `(Guitar solo)`
    /// and the like, all read as one, written as [`BREAK`].
    pub is_break: bool,
}

/// How a break in the singing is written, and shown.
pub const BREAK: &str = "♪";

/// When a line is sung as its text's source synced it ([`Line::synced`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Synced {
    /// Its time, in seconds from the start of the track.
    pub start: f64,
    /// The source's next time after it (the next line's, or a break's):
    /// by then the line is over. `None` for the last.
    pub end: Option<f64>,
}

impl Line {
    /// A line of `words` (as written, split at white space), marking the
    /// words in parentheses as backing.
    pub fn from_text(text: &str) -> Line {
        let mut line = Line {
            words: text.split_whitespace().map(Word::new).collect(),
            ..Line::default()
        };
        line.mark_backing();
        line
    }

    /// Marks the words inside parentheses, and the line when they are all
    /// of it. A parenthesis opens only at the start of a word, so `word(s)`
    /// stays a sung word.
    pub fn mark_backing(&mut self) {
        let mut depth = 0usize;
        for word in &mut self.words {
            let opens = word
                .text
                .trim_start_matches(['"', '\'', '“', '‘'])
                .starts_with('(');
            word.backing = depth > 0 || opens;
            for c in word.text.chars() {
                match c {
                    '(' if opens || depth > 0 => depth += 1,
                    ')' => depth = depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        self.backing = !self.words.is_empty() && self.words.iter().all(|w| w.backing);
    }

    /// The words joined by spaces.
    pub fn text(&self) -> String {
        let words: Vec<&str> = self.words.iter().map(|w| w.text.as_str()).collect();
        words.join(" ")
    }

    /// When it starts: its time, else its first timed word's start.
    pub fn start(&self) -> Option<f64> {
        self.time
            .or_else(|| self.words.iter().find_map(|w| w.start))
    }
}

/// Where a text came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A lyrics tag in the file.
    Tag,
    /// A `.lrc` file beside it.
    Sidecar,
    /// The LRCLIB web service.
    Lrclib,
    /// Heard from the voice by a speech model.
    Transcribed(Maker),
    /// Not known (a text given by hand).
    Unknown,
}

impl Source {
    /// Its name in a file's header: `tag`, `sidecar`, `lrclib`,
    /// `transcribed <name> <version>` or `unknown`.
    pub fn name(&self) -> String {
        match self {
            Source::Tag => "tag".into(),
            Source::Sidecar => "sidecar".into(),
            Source::Lrclib => "lrclib".into(),
            Source::Transcribed(m) => format!("transcribed {}", m.to_text()),
            Source::Unknown => "unknown".into(),
        }
    }

    /// The source a [`Source::name`] names.
    pub fn parse(text: &str) -> Option<Source> {
        let text = text.trim();
        Some(match text {
            "tag" => Source::Tag,
            "sidecar" => Source::Sidecar,
            "lrclib" => Source::Lrclib,
            "unknown" => Source::Unknown,
            _ => Source::Transcribed(Maker::parse(text.strip_prefix("transcribed ")?)?),
        })
    }
}

/// What made a result, and at which version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Maker {
    /// Its name, without spaces (`ctc:wav2vec2-base-960h`, `syllables`).
    pub name: String,
    /// Its version: a higher one may do better.
    pub version: u32,
}

impl Maker {
    /// A maker.
    pub fn new(name: impl Into<String>, version: u32) -> Maker {
        Maker {
            name: name.into(),
            version,
        }
    }

    /// `name version`.
    pub fn to_text(&self) -> String {
        format!("{} {}", self.name, self.version)
    }

    /// The maker in a [`Maker::to_text`].
    pub fn parse(text: &str) -> Option<Maker> {
        let (name, version) = text.trim().rsplit_once(' ')?;
        Some(Maker::new(name.trim(), version.parse().ok()?))
    }
}

/// How much of a text is timed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Detail {
    /// Not at all.
    #[default]
    None,
    /// Lines only.
    Lines,
    /// Words.
    Words,
}

/// How a text is timed, and by whom.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    /// How much of it.
    pub detail: Detail,
    /// The aligner that made the times; `None`: they came with the text.
    pub by: Option<Maker>,
    /// How well the text fit what the speech recogniser heard in the voice,
    /// when that was checked ([`fit`]).
    pub fit: Option<Fit>,
}

/// How well a text fit what was heard in the voice ([`fit::lyrics_fit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fit {
    /// The share of its words heard in order, in whole percent
    /// ([`fit::percent_of`]).
    pub percent: u8,
    /// Whether it took the place of the text first found, which fit less.
    pub swapped: bool,
}

impl Fit {
    /// A fit of `share` (0 to 1).
    pub fn new(share: f32, swapped: bool) -> Fit {
        Fit {
            percent: fit::percent_of(share),
            swapped,
        }
    }

    /// Whether the text fits ([`fit::FIT_LEAST_PERCENT`]).
    pub fn fits(&self) -> bool {
        self.percent >= fit::FIT_LEAST_PERCENT
    }

    /// `78`, or `24 swapped`.
    pub fn to_text(&self) -> String {
        if self.swapped {
            format!("{} swapped", self.percent)
        } else {
            self.percent.to_string()
        }
    }

    /// The fit in a [`Fit::to_text`].
    pub fn parse(text: &str) -> Option<Fit> {
        let mut words = text.split_whitespace();
        let percent = words.next()?.parse::<u8>().ok().filter(|p| *p <= 100)?;
        Some(Fit {
            percent,
            swapped: words.next() == Some("swapped"),
        })
    }
}

/// The lyrics of a track.
#[derive(Clone, Debug, PartialEq)]
pub struct Lyrics {
    /// The lines, in the order sung.
    pub lines: Vec<Line>,
    /// Where the text came from.
    pub source: Source,
    /// How it is timed.
    pub timing: Timing,
}

impl Lyrics {
    /// The lyrics in `text` (plain, LRC or Enhanced LRC), from `source`.
    pub fn parse(text: &str, source: Source) -> Lyrics {
        lrc::parse(text, source)
    }

    /// The words of `text` (plain, LRC or Enhanced LRC) from `source`, and
    /// its line times as [`Line::synced`] (each line from its time to the
    /// next one's): every word time it came with is dropped, the lines'
    /// times are kept only as the scaffold, and the lines without words (an
    /// LRC file's pauses) go. What is found
    /// elsewhere (a tag, a file beside the track, LRCLIB) is read this way;
    /// only our own aligner times the words.
    pub fn parse_text(text: &str, source: Source) -> Lyrics {
        let mut lyrics = lrc::parse(text, source);
        for k in 0..lyrics.lines.len() {
            let Some(start) = lyrics.lines[k].time else {
                continue;
            };
            let end = lyrics.lines[k + 1..]
                .iter()
                .find_map(|l| l.time.filter(|&t| t > start));
            lyrics.lines[k].synced = Some(Synced { start, end });
        }
        lyrics.drop_times();
        lyrics
    }

    /// Drops every time but the scaffold ([`Line::synced`]): the lines',
    /// the words' (and their confidence), and the lines without words (a
    /// break stays, marked).
    pub fn drop_times(&mut self) {
        self.lines.retain(|l| !l.words.is_empty() || l.is_break);
        for line in &mut self.lines {
            line.time = None;
            for word in &mut line.words {
                word.start = None;
                word.end = None;
                word.confidence = 0.0;
                word.letters.clear();
            }
        }
        self.timing = Timing::default();
    }

    /// As Enhanced LRC (plain lines where nothing is timed).
    pub fn to_lrc(&self) -> String {
        lrc::write(self)
    }

    /// Whether there are no words.
    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|l| l.words.is_empty())
    }

    /// How much is timed, from the lines and words themselves.
    pub fn detail_of_lines(lines: &[Line]) -> Detail {
        if lines.iter().flat_map(|l| &l.words).any(Word::is_timed) {
            Detail::Words
        } else if lines.iter().any(|l| l.time.is_some()) {
            Detail::Lines
        } else {
            Detail::None
        }
    }
}

/// A track to find lyrics for.
#[derive(Clone, Debug, Default)]
pub struct Track {
    /// The file.
    pub path: PathBuf,
    /// Its title.
    pub title: Option<String>,
    /// Its artist.
    pub artist: Option<String>,
    /// Its album.
    pub album: Option<String>,
    /// Its length in seconds.
    pub duration: Option<f64>,
    /// Its tags as (key, value), as the decoder gives them.
    pub tags: Vec<(String, String)>,
    /// The voice at [`VOCALS_RATE`], mono, when it has been separated
    /// (for a source that transcribes).
    pub vocals: Option<Arc<[f32]>>,
}

/// What a source found for a track.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// Lyrics.
    Lyrics(Lyrics),
    /// The track has no words.
    Instrumental,
    /// Nothing.
    NotFound,
}

/// Something that gives the text of a track, with line or word times when
/// it has them.
pub trait LyricsSource {
    /// Its name, for messages.
    fn name(&self) -> &str;

    /// What it has for `track`; `Err` when it could not tell (it may on
    /// another try).
    fn find(&self, track: &Track) -> Result<Found, String>;
}

/// Something that times the words of a text against the voice.
pub trait Aligner {
    /// Its name and version, recorded in its results.
    fn maker(&self) -> Maker;

    /// `lyrics` with word times, from the voice at [`VOCALS_RATE`] (mono).
    /// The result's [`Timing::by`] names this aligner.
    fn align(&self, vocals_16k: &[f32], lyrics: &Lyrics) -> Lyrics;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backing_words_are_the_parenthesised_ones() {
        let line = Line::from_text("I need you (need you) now word(s)");
        let backing: Vec<bool> = line.words.iter().map(|w| w.backing).collect();
        assert_eq!(backing, [false, false, false, true, true, false, false]);
        assert!(!line.backing);
        let line = Line::from_text("(Oh, oh oh)");
        assert!(line.backing);
        assert!(line.words.iter().all(|w| w.backing));
        assert!(!Line::from_text("").backing);
    }

    #[test]
    fn a_fit_names_itself() {
        let f = Fit::new(0.784, false);
        assert_eq!(f.percent, 78);
        assert!(f.fits());
        assert_eq!(Fit::parse(&f.to_text()), Some(f));
        let swapped = Fit::new(0.36, true);
        assert_eq!(swapped.to_text(), "36 swapped");
        assert_eq!(Fit::parse("36 swapped"), Some(swapped));
        assert!(!Fit::new(0.31, false).fits());
        assert_eq!(Fit::parse("120"), None);
    }

    #[test]
    fn a_fit_is_judged_on_one_scale() {
        // Just under the least: 35 % rounded, but not a fit, kept or not.
        let f = Fit::new(0.347, false);
        assert!(!fit::fits(0.347) && !f.fits());
        assert_eq!(f.percent, 34);
        assert!(!Fit::parse(&f.to_text()).unwrap().fits());
        // Every share from just under the least to just over: the same
        // verdict both ways, turning once.
        let least = f32::from(fit::FIT_LEAST_PERCENT) / 100.0;
        let near = 1 << 16;
        let mut turns = 0;
        let mut before = false;
        for bits in least.to_bits() - near..=least.to_bits() + near {
            let share = f32::from_bits(bits);
            let fits = fit::fits(share);
            assert_eq!(Fit::new(share, false).fits(), fits, "{share}");
            turns += usize::from(fits != before);
            before = fits;
        }
        assert_eq!(turns, 1);
        assert!(before, "a share just over the least fits");
    }

    #[test]
    fn sources_and_makers_name_themselves() {
        for source in [
            Source::Tag,
            Source::Sidecar,
            Source::Lrclib,
            Source::Unknown,
            Source::Transcribed(Maker::new("qwen3-asr", 2)),
        ] {
            assert_eq!(Source::parse(&source.name()), Some(source));
        }
        assert_eq!(Source::parse("telepathy"), None);
        assert_eq!(
            Maker::parse("ctc:wav2vec2-base-960h 3"),
            Some(Maker::new("ctc:wav2vec2-base-960h", 3))
        );
    }
}
