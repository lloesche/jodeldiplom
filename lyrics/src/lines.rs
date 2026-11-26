//! Where a stream of timed words breaks into lines: for lyrics that came
//! without lines of their own (heard from the voice by a speech
//! recogniser). [`transcript_lyrics`] gives such lyrics a first cut, to
//! be timed; [`relined`] makes their lines again from the timing.
//!
//! The most likely breaks over the whole song, by dynamic programming:
//! each break scored by how its pause and the word before it (how much
//! longer it lasts than the song's words do) speak for a line ending, each
//! line by how likely its length is, in words and in seconds. The scores
//! are log-likelihood ratios learnt from 33 songs with real lines (LRCLIB's
//! texts timed by our aligner; `research/lines`, `line_bench`). The
//! recogniser's punctuation counts for a break too, and much: a sentence's
//! end (`.`, `!`, `?`) and, less, a comma. Those weights, and how readily a
//! break is made, were tuned on 31 real transcripts (our recogniser and
//! aligner over songs whose real lines are known; `research/asrbench`,
//! the `line_asr` example), half the songs tuning, the other half scoring,
//! both ways: F1 0.64 exactly and 0.75 within a word against the real
//! lines (the weights before: 0.60 and 0.73; cutting at stops, else every
//! 9 words: 0.60 and 0.71). Rhymes, and lines kept to the bars, were tried:
//! the bars help rap with lines a bar long, but nothing overall.

use crate::asr::Transcript;
use crate::{Line, Lyrics, Maker, Source, Timing};

/// The upper edges of the pause bins, in seconds (the last is open).
const GAP_EDGES: [f64; 11] = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5];
/// How much more likely a break is than not, by pause bin (log).
const GAP_SCORE: [f64; 12] = [
    -0.451, -1.448, -1.607, -0.987, -0.59, 0.067, 1.201, 1.524, 2.228, 2.662, 2.894, 2.806,
];
/// The upper edges of the onset bins: a word's onset after the one before,
/// against the song's median (the last is open).
const ONSET_EDGES: [f64; 9] = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.5];
/// How much more likely a break is than not, by onset bin (log).
const ONSET_SCORE: [f64; 10] = [
    -0.769, -1.961, -2.141, -1.229, -0.671, -0.313, 0.577, 1.094, 1.829, 2.591,
];
/// A break's prior at any boundary (log), and how much more readily one is
/// made than that (tuned on real transcripts: none).
const PRIOR: f64 = -1.907;
/// A line's log length in seconds: mean and spread, and its weight.
const SECONDS: (f64, f64, f64) = (0.943, 0.588, 0.5);
/// A line's log length in words: mean and spread, and its weight.
const WORDS: (f64, f64, f64) = (1.795, 0.474, 1.0);
/// What a sentence's end before a boundary adds to a break there (log;
/// tuned on real transcripts, flat from 5 to 8).
const SENTENCE_END: f64 = 6.0;
/// What a comma before a boundary adds to a break there (log; tuned, flat
/// from 2.5 to 3.5).
const COMMA: f64 = 3.0;
/// The most words a line gets (more are hard to read as it is sung; a cap
/// of 12 costs the benchmark nothing, real lines rarely having over 10).
const MOST_WORDS: usize = 12;
/// The most words a line of a transcript's first cut gets (a longer
/// sentence is cut into even pieces).
const HEARD_LINE_MOST: usize = 9;

fn bin(edges: &[f64], x: f64) -> usize {
    edges.iter().position(|&e| x <= e).unwrap_or(edges.len())
}

/// Where lines start in a stream of words timed `(start, end)` (seconds),
/// after the first: word indices, ascending. `stops` says what each word
/// ends with as written.
pub fn breaks(times: &[(f64, f64)], stops: &[Stop]) -> Vec<usize> {
    let n = times.len();
    if n < 2 {
        return Vec::new();
    }
    let onsets: Vec<f64> = (1..n).map(|k| times[k].0 - times[k - 1].0).collect();
    let median = {
        let mut sorted = onsets.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2].max(1e-3)
    };
    // What speaks for a break before word k (k ≥ 1).
    let boundary: Vec<f64> = (0..n)
        .map(|k| {
            if k == 0 {
                return 0.0;
            }
            let gap = times[k].0 - times[k - 1].1;
            let onset = onsets[k - 1] / median;
            let sentence = match stops.get(k - 1) {
                Some(Stop::Sentence) => SENTENCE_END,
                Some(Stop::Comma) => COMMA,
                _ => 0.0,
            };
            GAP_SCORE[bin(&GAP_EDGES, gap)]
                + ONSET_SCORE[bin(&ONSET_EDGES, onset)]
                + PRIOR
                + sentence
        })
        .collect();
    let line = |a: usize, b: usize| -> f64 {
        let seconds = (times[b - 1].1 - times[a].0).max(0.2);
        let zs = (seconds.ln() - SECONDS.0) / SECONDS.1;
        let zw = (((b - a) as f64).ln() - WORDS.0) / WORDS.1;
        -0.5 * SECONDS.2 * zs * zs - 0.5 * WORDS.2 * zw * zw
    };
    // best[b]: the best score of words [0, b) with a line ending at b;
    // from[b]: where that line starts.
    let mut best = vec![f64::NEG_INFINITY; n + 1];
    let mut from = vec![0usize; n + 1];
    best[0] = 0.0;
    for b in 1..=n {
        for a in b.saturating_sub(MOST_WORDS)..b {
            if best[a] == f64::NEG_INFINITY {
                continue;
            }
            let v = best[a] + if a > 0 { boundary[a] } else { 0.0 } + line(a, b);
            if v > best[b] {
                best[b] = v;
                from[b] = a;
            }
        }
    }
    let mut out = Vec::new();
    let mut b = n;
    while b > 0 {
        let a = from[b];
        if a > 0 {
            out.push(a);
        }
        b = a;
    }
    out.reverse();
    out
}

/// What a word ends with as written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// Nothing.
    None,
    /// A comma (or a semicolon, a colon, a dash).
    Comma,
    /// A sentence's end.
    Sentence,
}

impl Stop {
    /// What `word` ends with.
    pub fn of(word: &str) -> Stop {
        let w = word.trim_end_matches(['"', '\'', '’', '”', ')']);
        if w.ends_with(['.', '!', '?', '…', '。', '！', '？']) {
            Stop::Sentence
        } else if w.ends_with([',', ';', ':', '–', '—', '，', '、']) {
            Stop::Comma
        } else {
            Stop::None
        }
    }
}

/// Lyrics from what a speech recogniser heard, by `maker`, as first cut
/// before their words are timed (then [`relined`] by the timing): each
/// part's words cut into lines at the ends of sentences as the recogniser
/// writes them (`.`, `!`, `?`, `;`, `…`, `。`, `！`, `？` ending a word, no
/// quotes looked through), at commas where a sentence is long, and into
/// even pieces of at most [`HEARD_LINE_MOST`] words; untimed. Their source
/// is [`Source::Transcribed`].
pub fn transcript_lyrics(transcript: &Transcript, maker: Maker) -> Lyrics {
    let mut lines = Vec::new();
    for part in &transcript.parts {
        let mut sentence: Vec<&str> = Vec::new();
        let words: Vec<&str> = part.text.split_whitespace().collect();
        for (k, word) in words.iter().enumerate() {
            sentence.push(word);
            let ends = word.ends_with(['.', '!', '?', ';', '…', '。', '！', '？'])
                || (word.ends_with([',', '，']) && sentence.len() >= 5)
                || k + 1 == words.len();
            if ends {
                let pieces = sentence.len().div_ceil(HEARD_LINE_MOST);
                let each = sentence.len().div_ceil(pieces);
                for piece in sentence.chunks(each) {
                    lines.push(Line::from_text(&piece.join(" ")));
                }
                sentence.clear();
            }
        }
    }
    Lyrics {
        lines,
        source: Source::Transcribed(maker),
        timing: Timing::default(),
    }
}

/// `lyrics` (every word timed) with their lines made again where the words'
/// timing says they break ([`breaks`]); each line's time its first word's.
/// Breaks in the singing are left out. Unchanged when a word is untimed.
pub fn relined(lyrics: &Lyrics) -> Lyrics {
    let words: Vec<&crate::Word> = lyrics.lines.iter().flat_map(|l| &l.words).collect();
    let times: Option<Vec<(f64, f64)>> = words.iter().map(|w| w.start.zip(w.end)).collect();
    let Some(times) = times else {
        return lyrics.clone();
    };
    let ends: Vec<Stop> = words.iter().map(|w| Stop::of(&w.text)).collect();
    let mut starts = breaks(&times, &ends);
    starts.insert(0, 0);
    starts.push(words.len());
    let lines = starts
        .windows(2)
        .filter(|p| p[1] > p[0])
        .map(|p| {
            let mut line = Line {
                words: words[p[0]..p[1]].iter().map(|w| (*w).clone()).collect(),
                time: words[p[0]].start,
                ..Line::default()
            };
            line.mark_backing();
            line
        })
        .collect();
    Lyrics {
        lines,
        source: lyrics.source.clone(),
        timing: lyrics.timing.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words a third of a second apart, two lines of five with a pause of
    /// `pause` between, at ends of phrases held a little.
    fn song(pause: f64, lines: usize) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut t = 0.0;
        for _ in 0..lines {
            for k in 0..5 {
                let held = if k == 4 { 0.6 } else { 0.25 };
                out.push((t, t + held));
                t += held + 0.08;
            }
            t += pause;
        }
        out
    }

    #[test]
    fn lines_break_at_the_pauses() {
        let times = song(0.6, 4);
        let ends = vec![Stop::None; times.len()];
        assert_eq!(breaks(&times, &ends), [5, 10, 15]);
    }

    #[test]
    fn no_pauses_still_gives_lines_no_longer_than_the_most() {
        let times: Vec<(f64, f64)> = (0..40)
            .map(|k| (k as f64 * 0.4, k as f64 * 0.4 + 0.3))
            .collect();
        let ends = vec![Stop::None; times.len()];
        let b = breaks(&times, &ends);
        let mut starts = vec![0];
        starts.extend(&b);
        starts.push(40);
        let lengths: Vec<usize> = starts.windows(2).map(|p| p[1] - p[0]).collect();
        assert!(
            lengths.len() > 1 && lengths.iter().all(|&l| (3..=MOST_WORDS).contains(&l)),
            "{lengths:?}"
        );
    }

    #[test]
    fn a_transcript_is_cut_into_lines_at_its_sentences() {
        let part = |start: f64, text: &str| crate::asr::Part {
            start,
            end: start + 20.0,
            text: text.into(),
        };
        let heard = Transcript {
            parts: vec![
                part(
                    0.0,
                    "Short one! Then a sentence that goes on, and on and on and on and on and on for ever",
                ),
                part(20.0, "no stops at all here"),
            ],
            ..Default::default()
        };
        let lyrics = transcript_lyrics(&heard, Maker::new("heard", 1));
        let texts: Vec<String> = lyrics.lines.iter().map(Line::text).collect();
        assert_eq!(
            texts,
            [
                "Short one!",
                "Then a sentence that goes on,",
                "and on and on and on",
                "and on and on for ever",
                "no stops at all here"
            ]
        );
        assert!(lyrics
            .lines
            .iter()
            .all(|l| l.words.len() <= HEARD_LINE_MOST));
        assert_eq!(lyrics.source, Source::Transcribed(Maker::new("heard", 1)));
    }

    #[test]
    fn relined_lyrics_keep_every_word_and_its_time() {
        let times = song(0.6, 3);
        let mut line = Line::from_text("a b c d e f g h i j k l m n o.");
        for (w, &(s, e)) in line.words.iter_mut().zip(&times) {
            w.start = Some(s);
            w.end = Some(e);
        }
        let lyrics = Lyrics {
            lines: vec![line],
            source: crate::Source::Unknown,
            timing: crate::Timing::default(),
        };
        let out = relined(&lyrics);
        let texts: Vec<String> = out.lines.iter().map(Line::text).collect();
        assert_eq!(texts, ["a b c d e", "f g h i j", "k l m n o."]);
        assert_eq!(out.lines[1].time, Some(times[5].0));
        // Untimed: as it was.
        let mut untimed = lyrics.clone();
        untimed.lines[0].words[3].start = None;
        assert_eq!(relined(&untimed), untimed);
    }
}
