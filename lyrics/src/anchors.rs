//! Where a recogniser heard the lyrics' words: its transcript comes in
//! parts of 20 to 30 s ([`crate::asr::Part`]), so a word of the lyrics
//! matched to a word it heard was sung within that part, give or take
//! [`SLACK`]. The aligner takes these windows as soft pins
//! ([`crate::ctc::Hints::windows`]).
//!
//! The words are matched in order (the longest common subsequence, as
//! [`crate::fit`] counts them). Three things keep a window from being
//! wrong:
//!
//! - Only runs of [`RUN`] words matched in a row, heard together in one
//!   part, count: a single common word ("I", "the") matched anywhere does
//!   not.
//! - A line whose pinned words (its first word matched, and the next
//!   heard with it) were also heard in another part (a chorus heard again,
//!   perhaps a different number of times than the text has it) may be
//!   matched to the wrong one, so it is not pinned.
//! - A part where the recogniser fell into a loop ([`looped`]: one 26 s
//!   part of a song had "cry me a river" sixty times) is not used.
//!
//! On 138 songs, 4 % of lines pinned this way had LRCLIB's time outside
//! their window before the last two rules; with them, the pins (and the
//! opening pull) broke 5 lines that were right and fixed 42.

use std::collections::HashSet;

use crate::asr::Part;
use crate::fit::words;
use crate::Lyrics;

/// How many words matched in a row make a run that counts.
pub const RUN: usize = 3;

/// The seconds a window reaches past its part on either side (a word cut
/// between parts, a part's edge).
pub const SLACK: f64 = 1.0;

/// For each word of `lyrics` (line after line), the seconds within which a
/// recogniser heard it start, by its transcript's `parts`: for each line,
/// its first word matched in a run that counts, unless what it was heard
/// with was heard in another part too; `None` for the others.
pub fn windows(lyrics: &Lyrics, parts: &[Part]) -> Vec<Option<(f64, f64)>> {
    // The heard words, each with its part.
    let mut heard = Vec::new();
    let mut part_of = Vec::new();
    for (k, p) in parts.iter().enumerate() {
        if looped(p) {
            continue;
        }
        for w in words(&p.text) {
            heard.push(w);
            part_of.push(k);
        }
    }
    // The text's words, each with the lyrics' word it is of (a word may
    // compare as two: "five-thirty"), and that word's line.
    let mut text = Vec::new();
    let mut word_of = Vec::new();
    let mut line_of = Vec::new();
    for (l, line) in lyrics.lines.iter().enumerate() {
        for w in &line.words {
            for c in words(&w.text) {
                text.push(c);
                word_of.push(line_of.len());
            }
            line_of.push(l);
        }
    }
    let pairs = common(&text, &heard);
    // The text words in a run (matched in a row, heard in a row in one
    // part): the heard word each is matched to, and where its run ends.
    let mut good: Vec<Option<(usize, usize)>> = vec![None; text.len()];
    let mut k = 0;
    while k < pairs.len() {
        let mut e = k;
        while e + 1 < pairs.len()
            && pairs[e + 1].0 == pairs[e].0 + 1
            && pairs[e + 1].1 == pairs[e].1 + 1
            && part_of[pairs[e + 1].1] == part_of[pairs[k].1]
        {
            e += 1;
        }
        if e + 1 - k >= RUN {
            for &(i, j) in &pairs[k..=e] {
                good[i] = Some((j, pairs[e].1 + 1));
            }
        }
        k = e + 1;
    }
    // The heard words from `j` (at most RUN, within its run) heard in
    // another part too.
    let elsewhere = |j: usize, end: usize| {
        let phrase = &heard[j..end.min(j + RUN)];
        (0..=heard.len() - phrase.len()).any(|i| {
            part_of[i] != part_of[j]
                && part_of[i + phrase.len() - 1] == part_of[i]
                && heard[i..i + phrase.len()] == *phrase
        })
    };
    let mut out = vec![None; line_of.len()];
    let mut done_line = None;
    for (i, g) in good.iter().enumerate() {
        let Some((j, end)) = *g else {
            continue;
        };
        let w = word_of[i];
        // A word's start is that of its first compared part.
        if done_line == Some(line_of[w]) || (i > 0 && word_of[i - 1] == w) {
            continue;
        }
        done_line = Some(line_of[w]);
        if !elsewhere(j, end) {
            let p = &parts[part_of[j]];
            out[w] = Some((p.start - SLACK, p.end + SLACK));
        }
    }
    out
}

/// Whether the recogniser fell into a loop in `part`: more words than are
/// sung in its time (over 4 a second), most of them one phrase over again
/// (under a third of its runs of three words different).
pub fn looped(part: &Part) -> bool {
    let w = words(&part.text);
    let seconds = (part.end - part.start).max(1.0);
    if (w.len() as f64) < 4.0 * seconds || w.len() < 3 {
        return false;
    }
    let threes: HashSet<&[String]> = w.windows(3).collect();
    (threes.len() as f64) < (w.len() - 2) as f64 / 3.0
}

/// The pairs `(i, j)` of a longest common subsequence of `a` and `b`.
pub(crate) fn common<T: PartialEq>(a: &[T], b: &[T]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut t = vec![0u16; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[at(i, j)] = if a[i] == b[j] {
                t[at(i + 1, j + 1)].saturating_add(1)
            } else {
                t[at(i + 1, j)].max(t[at(i, j + 1)])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if t[at(i + 1, j)] >= t[at(i, j + 1)] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Line, Source, Timing};

    fn lyrics(lines: &[&str]) -> Lyrics {
        Lyrics {
            lines: lines.iter().map(|t| Line::from_text(t)).collect(),
            source: Source::Unknown,
            timing: Timing::default(),
        }
    }

    fn part(start: f64, end: f64, text: &str) -> Part {
        Part {
            start,
            end,
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_line_heard_in_a_part_is_pinned_to_it() {
        let text = lyrics(&[
            "Here we go down the river",
            "I",
            "Under the bridge, the water's cold",
        ]);
        let parts = [
            part(0.0, 25.0, "Here we go down the river."),
            part(25.0, 50.0, "Hi. Under the bridge the water's cold."),
        ];
        let w = windows(&text, &parts);
        // A window on each line's first word in a run; none on "I" (one
        // word), none on the rest of a line.
        let want = |a: f64, b: f64| Some((a - SLACK, b + SLACK));
        assert_eq!(w.len(), 6 + 1 + 6);
        assert_eq!(w[0], want(0.0, 25.0));
        assert!(w[1..7].iter().all(Option::is_none));
        assert_eq!(w[7], want(25.0, 50.0));
        assert!(w[8..].iter().all(Option::is_none));
    }

    #[test]
    fn a_chorus_heard_in_two_parts_is_not_pinned() {
        let text = lyrics(&[
            "Keep it off my wave",
            "Keep it off my wave",
            "Gone away now",
        ]);
        let parts = [
            part(0.0, 25.0, "Keep it off my wave."),
            part(25.0, 50.0, "Keep it off my wave. Gone away now."),
        ];
        let w = windows(&text, &parts);
        assert!(w[..10].iter().all(Option::is_none), "{w:?}");
        assert_eq!(w[10], Some((25.0 - SLACK, 50.0 + SLACK)));
    }

    #[test]
    fn a_loop_is_not_heard() {
        let looping = part(0.0, 10.0, &"cry me a river ".repeat(30));
        assert!(looped(&looping));
        // A chorus sung four times in a part is no loop; nor are fast
        // words that differ.
        assert!(!looped(&part(0.0, 26.0, &"cry me a river ".repeat(4))));
        let fast: String = (0..60).map(|i| format!("w{i} ")).collect();
        assert!(!looped(&part(0.0, 10.0, &fast)));
        let text = lyrics(&["Cry me a river"]);
        assert_eq!(windows(&text, &[looping]), vec![None; 4]);
    }
}
