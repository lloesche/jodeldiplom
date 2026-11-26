//! What a recogniser hears in short glimpses of the voice (a sliding
//! window): [`GLIMPSE`]
//! seconds every [`HOP`] seconds, so every moment is heard twice, each
//! glimpse on its own. A transcript's parts ([`crate::asr::Part`], 20 to
//! 30 s) say in which half minute a line was sung; the glimpses say it to a
//! second or two, and where nothing with words was sung at all. The aligner
//! takes both ([`crate::ctc::Hints::glimpses`]):
//!
//! - **Where no words were heard** ([`unheard`]): a stretch every glimpse
//!   over it heard nothing in (a choir's pad, a synth left in the voice,
//!   a held breath) is no place for a line to go on through, so a line's
//!   words are not spread over it.
//! - **Where a line starts** ([`line_starts`]): near where the alignment
//!   first put a line, the glimpses that heard its first words (with each
//!   other) say from when to when it may start, and those near that heard
//!   none of them (an echo, a "yeah, yeah" a line's first word was taken
//!   from) say when it does not.
//!
//! wav2vec2 hears letters in sounds with no words (a chant's "yo" as
//! "you", a choir's "oh"); the recogniser, which hears whole words in their
//! context, does not.

use crate::asr::Part;
use crate::ctc::FrameClock;
use crate::words::{words, Compared};
use crate::Lyrics;

/// A glimpse's length, in seconds.
pub const GLIMPSE: f64 = 3.0;
/// The time from one glimpse's start to the next's.
pub const HOP: f64 = 1.5;
/// How far inside a glimpse a moment must be to count as heard in it (its
/// edges cut words, which the recogniser may or may not hear).
pub const INSIDE: f64 = 0.25;
/// How far inside a glimpse a line's start must be for the glimpse not
/// hearing it to tell (a word at its edge is often lost).
pub const MARGIN: f64 = 0.75;
/// How far before a glimpse that heard a line's start the line may start
/// (a word cut by its edge, heard whole).
pub const SLACK: f64 = 0.5;
/// How far from where the alignment first put a line the glimpses are
/// looked at for it, in seconds: they move a line a little, the
/// transcript's parts move it far.
pub const NEAR: f64 = 4.0;
/// How many of a line's first words a glimpse that heard none of them
/// tells it does not start in.
pub const RUN: usize = 3;
/// How much nearer to where the alignment first put a line one time its
/// first words were heard must be than any other for the glimpses to
/// tell, in seconds.
pub const APART: f64 = 1.0;

/// The glimpses of a voice `seconds` long: (start, end), from 0 on.
pub fn glimpses_of(seconds: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut at = 0.0;
    while at < seconds {
        out.push((at, (at + GLIMPSE).min(seconds)));
        if at + GLIMPSE >= seconds {
            break;
        }
        at += HOP;
    }
    out
}

/// The glimpses of `voice` (16 kHz mono), each heard by `hear` (the words
/// it heard); one where the voice is not heard at all
/// ([`crate::spread::vocal_stretches`]) heard as nothing without asking.
pub fn hear_glimpses(
    voice: &[f32],
    mut hear: impl FnMut(&[f32]) -> Result<String, String>,
) -> Result<Vec<Part>, String> {
    hear_glimpses_at_once(voice, |pieces| pieces.iter().map(|p| hear(p)).collect())
}

/// [`hear_glimpses`] with the glimpses where the voice is heard all given
/// to `hear` at once (in order; it gives each one's words), for a
/// recogniser that hears many together.
pub fn hear_glimpses_at_once(
    voice: &[f32],
    hear: impl FnOnce(&[&[f32]]) -> Result<Vec<String>, String>,
) -> Result<Vec<Part>, String> {
    let rate = f64::from(crate::VOCALS_RATE);
    let sung = crate::spread::vocal_stretches(voice);
    let mut out = Vec::new();
    let mut pieces = Vec::new();
    for (start, end) in glimpses_of(voice.len() as f64 / rate) {
        if sung.iter().any(|&(a, b)| a < end && start < b) {
            let from = ((start * rate) as usize).min(voice.len());
            let to = ((end * rate) as usize).min(voice.len());
            pieces.push((out.len(), &voice[from..to]));
        }
        out.push(Part {
            start,
            end,
            text: String::new(),
        });
    }
    let heard = hear(&pieces.iter().map(|p| p.1).collect::<Vec<_>>())?;
    if heard.len() != pieces.len() {
        return Err(format!(
            "{} glimpses heard of {}",
            heard.len(),
            pieces.len()
        ));
    }
    for ((at, _), text) in pieces.into_iter().zip(heard) {
        out[at].text = text;
    }
    Ok(out)
}

/// For each frame of `clock`, whether no words were heard there: some
/// glimpse has it well inside ([`INSIDE`], [`FrameClock::inward`]) and
/// none of those heard a word. (Only between words heard, or only short
/// stretches, did worse on 138 songs: an intro or a long break a line was
/// taken into is as common as a verse the recogniser hears nothing of.)
pub fn unheard(glimpses: &[Part], clock: FrameClock) -> Vec<bool> {
    let mut covered = vec![false; clock.frames];
    let mut heard = vec![false; clock.frames];
    for g in glimpses {
        let any = !words(&g.text).is_empty();
        for f in clock.inward(g.start, g.end, INSIDE) {
            covered[f] = true;
            heard[f] |= any;
        }
    }
    covered.iter().zip(&heard).map(|(&c, &h)| c && !h).collect()
}

/// The sounds sung without words, each letter once ("oooh" is "oh").
const VOCABLES: [&str; 22] = [
    "o", "oh", "ah", "a", "aha", "uh", "u", "m", "hm", "mh", "la", "na", "da", "du", "ya", "ye",
    "yeah", "yo", "hey", "ha", "whoa", "woah",
];

/// Whether `word` is a sound sung without words ("Oooh", "la", "yeah"):
/// a line of nothing else is sung where the glimpses hear no words.
pub fn vocable(word: &str) -> bool {
    word.split(|c: char| !c.is_alphabetic()).all(|piece| {
        let mut letters = String::new();
        for l in piece.chars().flat_map(char::to_lowercase) {
            if !letters.ends_with(l) {
                letters.push(l);
            }
        }
        letters.is_empty() || VOCABLES.contains(&letters.as_str())
    })
}

/// Whether `heard` has the start of a line whose first words are `text`
/// (at least two): `surely`, its first [`RUN`] in a row; else its first two
/// in a row, or a word and then its second and third (its first heard as
/// another, "so" for "say").
fn starts(heard: &[String], text: &[String], surely: bool) -> bool {
    let run = &text[..text.len().min(RUN)];
    if surely {
        return heard.windows(run.len()).any(|w| w == run);
    }
    heard.windows(2).any(|w| w == &text[..2])
        || (text.len() >= 3 && heard.windows(3).any(|w| w[1..] == text[1..3]))
}

/// Where the glimpses heard a line start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    /// The earliest it may start, in seconds.
    pub from: f64,
    /// The latest.
    pub to: f64,
}

/// For each line of `lyrics`, the seconds within which its first word
/// starts by the glimpses near `first` (where the alignment first put it):
/// of the glimpses near that heard its start ([`RUN`] of its first words in
/// a row; when none did, its first two, or a word and its second
/// and third), the run of overlapping ones nearest it (a time it was
/// sung), from their latest start (a little before, [`SLACK`]) to their
/// earliest end; less the middles ([`MARGIN`]) of those near that heard
/// none of its first [`RUN`] words where they meet that span's ends.
/// `None` for a line of one word, one put nowhere, one no glimpse near
/// heard, one heard about as near twice ([`APART`]), or one the glimpses
/// disagree on.
pub fn line_starts(
    lyrics: &Lyrics,
    glimpses: &[Part],
    first: &[Option<f64>],
) -> Vec<Option<Start>> {
    let heard: Vec<Vec<String>> = glimpses.iter().map(|g| words(&g.text)).collect();
    let compared = Compared::of(lyrics);
    (0..compared.lines())
        .zip(first)
        .map(|(line, at)| {
            let at = (*at)?;
            let text = compared.of_line(line);
            if text.len() < 2 {
                return None;
            }
            let run = &text[..text.len().min(RUN)];
            let near = |g: &Part| g.end > at - NEAR && g.start < at + NEAR;
            // The glimpses near that heard it (its first words all, else
            // fewer), in runs of overlapping ones (each run a time it was
            // sung): the run nearest, when no other is about as near.
            let mut runs: Vec<Vec<&Part>> = Vec::new();
            for surely in [true, false] {
                for (g, h) in glimpses.iter().zip(&heard) {
                    if !(near(g) && starts(h, text, surely)) {
                        continue;
                    }
                    match runs.last_mut() {
                        Some(r) if r.last().is_some_and(|p| g.start < p.end) => r.push(g),
                        _ => runs.push(vec![g]),
                    }
                }
                if !runs.is_empty() {
                    break;
                }
            }
            let off = |r: &Vec<&Part>| {
                let (a, b) = (r[0].start, r[r.len() - 1].end);
                (a - at).max(at - b).max(0.0)
            };
            runs.sort_by(|a, b| off(a).total_cmp(&off(b)));
            let best = runs.first()?;
            if runs.get(1).is_some_and(|r| off(r) - off(best) < APART) {
                return None;
            }
            let from = best
                .iter()
                .map(|g| g.start - SLACK)
                .fold(f64::NEG_INFINITY, f64::max);
            let to = best.iter().map(|g| g.end).fold(f64::INFINITY, f64::min);
            if from >= to {
                return None;
            }
            let (mut from, mut to) = (from, to);
            // The middles of the glimpses that heard none of its first
            // words, where they meet the span's ends.
            let mut none: Vec<(f64, f64)> = glimpses
                .iter()
                .zip(&heard)
                .filter(|(g, h)| near(g) && !run.iter().any(|w| h.contains(w)))
                .map(|(g, _)| (g.start + MARGIN, g.end - MARGIN))
                .filter(|(a, b)| a < b)
                .collect();
            none.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(a, b) in &none {
                if a <= from && from < b {
                    from = b;
                }
            }
            for &(a, b) in none.iter().rev() {
                if a < to && to <= b {
                    to = a;
                }
            }
            (from < to).then_some(Start { from, to })
        })
        .collect()
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

    fn spans(starts: Vec<Option<Start>>) -> Vec<Option<(f64, f64)>> {
        starts.iter().map(|s| s.map(|s| (s.from, s.to))).collect()
    }

    fn glimpse(start: f64, text: &str) -> Part {
        Part {
            start,
            end: start + GLIMPSE,
            text: text.to_owned(),
        }
    }

    #[test]
    fn sounds_without_words_are_told() {
        for w in ["Oooh", "ooh", "Ahh,", "la", "Na-na", "Yeah!", "mmm", "Whoa"] {
            assert!(vocable(w), "{w}");
        }
        for w in ["oh-no", "home", "love", "you", "Hallo", "nah"] {
            assert!(!vocable(w), "{w}");
        }
    }

    #[test]
    fn every_moment_is_heard_twice() {
        let g = glimpses_of(7.0);
        assert_eq!(
            g,
            vec![(0.0, 3.0), (1.5, 4.5), (3.0, 6.0), (4.5, 7.0)],
            "the last one cut at the end"
        );
        assert_eq!(glimpses_of(2.0), vec![(0.0, 2.0)]);
        assert!(glimpses_of(0.0).is_empty());
    }

    #[test]
    fn only_glimpses_where_the_voice_is_heard_are_heard() {
        let rate = crate::VOCALS_RATE as usize;
        let voice: Vec<f32> = (0..6 * rate)
            .map(|i| {
                let t = i as f64 / rate as f64;
                if (1.0..2.5).contains(&t) {
                    (t * 440.0 * std::f64::consts::TAU).sin() as f32 * 0.3
                } else {
                    0.0
                }
            })
            .collect();
        let mut asked = Vec::new();
        let g = hear_glimpses(&voice, |piece| {
            asked.push(piece.len());
            Ok("La.".to_owned())
        })
        .unwrap();
        assert_eq!(asked, vec![3 * rate, 3 * rate]);
        let texts: Vec<&str> = g.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, ["La.", "La.", ""]);
    }

    #[test]
    fn a_stretch_every_glimpse_heard_nothing_in_is_unheard() {
        let g = [
            glimpse(0.0, "It doesn't hurt me."),
            glimpse(1.5, ""),
            glimpse(3.0, ""),
            glimpse(4.5, "Oh."),
        ];
        let clock = FrameClock {
            frame_seconds: 0.02,
            frames: 375,
        };
        let u = unheard(&g, clock);
        let at = |t: f64| u[(t / 0.02) as usize];
        // Heard by the first, or near its end; unheard where the two
        // empty ones alone have it; heard again in the last.
        assert!(!at(1.0) && !at(2.7));
        assert!(at(3.0) && at(4.6));
        assert!(!at(5.0) && !at(7.0));
    }

    #[test]
    fn a_line_starts_after_the_chant_heard_before_it() {
        // "Do you" first put on a "yeah, yeah, yo" at 28.1; the line is
        // sung at 29.4.
        let text = lyrics(&["It doesn't hurt me", "Do you wanna feel how it feels?"]);
        let g = [
            glimpse(24.0, "It doesn't hurt me."),
            glimpse(25.5, "It doesn't hurt me."),
            glimpse(27.0, "Yeah, yeah, yeah."),
            glimpse(28.5, "Do you wanna feel how it feels?"),
            glimpse(30.0, "Feel how it feels."),
        ];
        let w = spans(line_starts(&text, &g, &[Some(26.1), Some(28.1)]));
        // From the end of the chant's glimpse's middle to the start of the
        // middle of the next that heard none of it.
        assert_eq!(w[1], Some((27.0 + GLIMPSE - MARGIN, 30.0 + MARGIN)));
        let (a, b) = w[0].unwrap();
        assert!(a <= 25.9 && 25.9 <= b, "{a} {b}");
        // A line put nowhere, or of one word, gets nothing.
        assert_eq!(
            spans(line_starts(&text, &g, &[None, None])),
            vec![None, None]
        );
        let one = lyrics(&["Oh"]);
        assert_eq!(spans(line_starts(&one, &g, &[Some(28.0)])), vec![None]);
    }

    #[test]
    fn a_first_word_heard_as_another_still_tells() {
        // "Say" heard as "so", after a glimpse that heard nothing.
        let text = lyrics(&["Say if I only could"]);
        let g = [
            glimpse(234.0, "With no problems."),
            glimpse(235.5, ""),
            glimpse(237.0, "So if I can."),
        ];
        assert_eq!(
            spans(line_starts(&text, &g, &[Some(235.7)])),
            vec![Some((235.5 + GLIMPSE - MARGIN, 237.0 + GLIMPSE))]
        );
    }

    #[test]
    fn a_line_heard_twice_about_as_near_is_left_alone() {
        let text = lyrics(&["Keep it off my wave"]);
        let g = [
            glimpse(9.0, "Keep it off my wave."),
            glimpse(12.5, "Keep it off."),
        ];
        assert_eq!(spans(line_starts(&text, &g, &[Some(12.0)])), vec![None]);
        // One much nearer than the other tells.
        let g = [
            glimpse(10.5, "Keep it off my wave."),
            glimpse(15.0, "Keep it off."),
        ];
        assert_eq!(
            spans(line_starts(&text, &g, &[Some(12.0)])),
            vec![Some((10.5 - SLACK, 10.5 + GLIMPSE))]
        );
    }
}
