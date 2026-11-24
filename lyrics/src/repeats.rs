//! Repeats a text leaves out: a chorus or hook written once and sung
//! twice. The aligner has nothing for the second time, so it stretches
//! the words around it over it, and the lines after it drift (DMX's
//! "Ruff Ryders' Anthem" opens with its hook twice, its text has it
//! once).
//!
//! What a recogniser heard says where: a run of at least [`GAP`] heard
//! words the text has nothing for (none of them in the longest common
//! subsequence of the text's words and the heard ones) that is mostly
//! lines the text has elsewhere, sung up to [`TIMES`] times in a row (at
//! least [`LEAST`] words in common, [`OF_BLOCK`] of those lines' words
//! and [`OF_GAP`] of the run), gets those lines again there: at the first
//! boundary between lines after the last word matched.
//!
//! On 152 songs with LRCLIB's texts (`research/asrbench/repeats.py`) it put
//! lines in for 22: each a chorus, hook or outro sung more often than
//! written (DMX's hook four times over), the share of the heard words the
//! text accounts for up by 1 to 30 points.

use crate::asr::Part;
use crate::fit::words;
use crate::{Line, Lyrics};

/// Heard words in a row the text has nothing for, at least.
pub const GAP: usize = 6;
/// Words of the lines put in heard in the run, at least.
pub const LEAST: usize = 5;
/// The share of the lines' words heard in the run, at least.
pub const OF_BLOCK: f32 = 0.6;
/// The share of the run the lines account for, at least.
pub const OF_GAP: f32 = 0.5;
/// Lines put in as one repeat, at most.
pub const MOST_LINES: usize = 8;
/// Times in a row the lines are put in, at most (a hook sung twice, an
/// outro's line sung over and over).
pub const TIMES: usize = 4;

/// `lyrics` with the repeats the recogniser heard (in its transcript's
/// `parts`) and the text leaves out put in, each line again as it is
/// written but without a time of its own (nor a synced one). Unchanged
/// when none.
pub fn with_repeats(lyrics: &Lyrics, parts: &[Part]) -> Lyrics {
    let heard: Vec<String> = parts
        .iter()
        .filter(|p| !crate::anchors::looped(p))
        .flat_map(|p| words(&p.text))
        .collect();
    let lines: Vec<Vec<String>> = lyrics
        .lines
        .iter()
        .map(|l| l.words.iter().flat_map(|w| words(&w.text)).collect())
        .collect();
    let mut text = Vec::new();
    let mut line_of = Vec::new();
    let mut first_of = Vec::new();
    for (k, l) in lines.iter().enumerate() {
        first_of.push(text.len());
        for w in l {
            text.push(w.clone());
            line_of.push(k);
        }
    }
    let pairs = crate::anchors::common(&text, &heard);
    // The heard runs between matched words, each with the line before
    // which it was sung.
    let mut inserts = Vec::new();
    let mut bounds = Vec::with_capacity(pairs.len() + 2);
    bounds.push((None, None));
    bounds.extend(pairs.iter().map(|&(i, j)| (Some(i), Some(j))));
    bounds.push((None, None));
    for k in 0..bounds.len() - 1 {
        let (ti, hi) = bounds[k];
        let hj = if k + 1 == bounds.len() - 1 {
            heard.len()
        } else {
            bounds[k + 1].1.unwrap()
        };
        let a = hi.map_or(0, |h| h + 1);
        if hj < a + GAP {
            continue;
        }
        // At the first boundary between lines after the last word matched
        // (the rest of its line perhaps not heard), before the next one.
        let next = bounds[k + 1].0.unwrap_or(text.len());
        let mut t = ti.map_or(0, |t| t + 1);
        if t < text.len() && first_of[line_of[t]] != t {
            t = first_of.get(line_of[t] + 1).copied().unwrap_or(text.len());
            if t > next {
                continue;
            }
        }
        let at = if t < text.len() {
            line_of[t]
        } else {
            lines.len()
        };
        if let Some(block) = best_block(&lines, &lyrics.lines, &heard[a..hj]) {
            inserts.push((at, block));
        }
    }
    if inserts.is_empty() {
        return lyrics.clone();
    }
    let mut out = lyrics.clone();
    for &(at, (l1, l2, times)) in inserts.iter().rev() {
        let again: Vec<Line> = (0..times)
            .flat_map(|_| &lyrics.lines[l1..=l2])
            .map(|l| Line {
                time: None,
                synced: None,
                ..l.clone()
            })
            .collect();
        out.lines.splice(at..at, again);
    }
    out
}

/// The run of whole lines (starting and ending on sung ones, at most
/// [`MOST_LINES`]), sung up to [`TIMES`] times in a row, that best
/// accounts for the heard words `gap`, when one does well enough: its
/// first and last line and how many times.
fn best_block(
    lines: &[Vec<String>],
    written: &[Line],
    gap: &[String],
) -> Option<(usize, usize, usize)> {
    let mut best: Option<(usize, (usize, usize, usize))> = None;
    let n = gap.len() as f32;
    for l1 in 0..lines.len() {
        if written[l1].is_break || lines[l1].is_empty() {
            continue;
        }
        let mut block: Vec<String> = Vec::new();
        for (l2, line) in lines.iter().enumerate().skip(l1).take(MOST_LINES) {
            block.extend(line.iter().cloned());
            if block.len() as f32 > 1.5 * n {
                break;
            }
            if line.is_empty() {
                continue;
            }
            for times in 1..=TIMES {
                let len = (block.len() * times) as f32;
                if len > 1.5 * n {
                    break;
                }
                if len < 0.5 * n {
                    continue;
                }
                let words: Vec<String> = (0..times).flat_map(|_| block.iter().cloned()).collect();
                let c = crate::anchors::common(&words, gap).len();
                let good = c >= LEAST && c as f32 >= OF_BLOCK * len && c as f32 >= OF_GAP * n;
                if good && best.is_none_or(|b| c > b.0) {
                    best = Some((c, (l1, l2, times)));
                }
            }
        }
    }
    best.map(|b| b.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Source, Timing};

    fn lyrics(lines: &[&str]) -> Lyrics {
        Lyrics {
            lines: lines.iter().map(|t| Line::from_text(t)).collect(),
            source: Source::Unknown,
            timing: Timing::default(),
        }
    }

    fn part(text: &str) -> Part {
        Part {
            start: 0.0,
            end: 25.0,
            text: text.to_owned(),
        }
    }

    fn texts(l: &Lyrics) -> Vec<String> {
        l.lines.iter().map(Line::text).collect()
    }

    const HOOK: [&str; 3] = [
        "Stop, drop, shut 'em down open up shop",
        "Oh, no",
        "That's how Ruff Ryders roll",
    ];

    #[test]
    fn a_hook_sung_twice_and_written_once_goes_in_again() {
        let mut text = HOOK.to_vec();
        text.extend([
            "Niggaz wanna try, niggaz wanna lie",
            "Then niggaz wonder why",
        ]);
        let parts = [
            part("Uh. Something new. Stop. Drop. Shut them down. Open up shop. Oh no, that's how rough riders roll."),
            part("Stop. Drop, shut 'em down over them shop. Oh no, that's how rough riders roll. Niggas wanna try, niggas wanna lie, then niggas wonder why."),
        ];
        let out = with_repeats(&lyrics(&text), &parts);
        let mut want: Vec<&str> = HOOK.to_vec();
        want.extend(HOOK);
        want.extend([
            "Niggaz wanna try, niggaz wanna lie",
            "Then niggaz wonder why",
        ]);
        assert_eq!(texts(&out), want);
    }

    #[test]
    fn a_hook_sung_twice_over_after_a_line_half_heard_goes_in_twice_after_it() {
        let mut text = HOOK.to_vec();
        text.extend([
            "Think you holdin weight? Then you haven't met the Apes",
            "Is ya'll niggaz crazy?",
        ]);
        let parts = [
            part("Stop drop shut em down open up shop oh no that's how Ruff Ryders roll."),
            part(
                "Think you holding weight then you haven't met the apex. Stop. Drop, shut 'em down, \
                 open up shop. Oh no, that's how rough riders roll. Stop, drop, shut 'em down, \
                 open up shop. Oh no, that's how rough riders roll. Is y'all niggas crazy?",
            ),
        ];
        let out = with_repeats(&lyrics(&text), &parts);
        let mut want: Vec<&str> = HOOK.to_vec();
        want.push("Think you holdin weight? Then you haven't met the Apes");
        want.extend(HOOK);
        want.extend(HOOK);
        want.push("Is ya'll niggaz crazy?");
        assert_eq!(texts(&out), want);
    }

    #[test]
    fn a_text_sung_as_written_is_left_alone() {
        let text = lyrics(&HOOK);
        let parts = [part(
            "Stop drop shut em down open up shop oh no that's how Ruff Ryders roll",
        )];
        assert_eq!(with_repeats(&text, &parts), text);
    }

    #[test]
    fn words_the_text_does_not_have_are_not_made_a_repeat() {
        // An intro spoken over the beat, nothing like the lines.
        let text = lyrics(&HOOK);
        let parts = [part(
            "Yo yo this is the intro we are live tonight from the city ladies and gentlemen \
             Stop drop shut em down open up shop oh no that's how Ruff Ryders roll",
        )];
        assert_eq!(with_repeats(&text, &parts), text);
    }

    #[test]
    fn a_repeated_line_has_no_time_of_its_own() {
        let mut text = lyrics(&HOOK);
        text.lines[0].time = Some(12.0);
        let parts = [part(
            "stop drop shut em down open up shop oh no that's how ruff ryders roll \
             stop drop shut em down open up shop oh no that's how ruff ryders roll",
        )];
        let out = with_repeats(&text, &parts);
        assert_eq!(out.lines.len(), 6);
        let times: Vec<Option<f64>> = out.lines.iter().map(|l| l.time).collect();
        assert_eq!(times.iter().filter(|t| t.is_some()).count(), 1);
    }
}
