//! Timing without a model, and the filling-in of words whose times are not
//! trusted.
//!
//! - [`interpolate`]: words under a confidence threshold (or with no times)
//!   are given times between their trusted neighbours in the line, in
//!   proportion to their syllables. A line's LRC time stands in for a
//!   neighbour at its start, the next line's start for one at its end.
//!   With room to spare, the words go at no slower than
//!   [`SLOWEST`] a syllable from their line's start; with only one
//!   neighbour, at [`TYPICAL`].
//! - [`split_lines`]: line times only, split across the words.
//! - [`spread_over`]: no times at all: the words in syllable proportion
//!   over the stretches where the voice is heard ([`vocal_stretches`]).
//! - [`fallback`], the timing when there is no model, picks between the
//!   two: a synced text's line times ([`crate::Line::synced`]) split, else
//!   the words spread over the voice.

use crate::ctc::FrameClock;
use crate::{Detail, Line, Lyrics, Maker, Timing, Word, CONFIDENT, VERSION, VOCALS_RATE};

/// The slowest pace words are filled in at, in seconds a syllable.
pub const SLOWEST: f64 = 1.0;
/// The pace words are filled in at with a neighbour on one side only.
pub const TYPICAL: f64 = 0.3;

/// What the timing without a model records in its results.
pub fn maker() -> Maker {
    Maker::new("syllables", VERSION)
}

/// The syllables in `word`, estimated: the groups of vowels (`y` counted
/// as one), less a silent final `e`; at least one for a word with a
/// letter. A word in another script counts a syllable a character (right
/// for CJK); punctuation alone counts none.
pub fn syllables(word: &str) -> usize {
    let letters: Vec<char> = word
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    if letters.is_empty() {
        return 0;
    }
    if !letters.iter().all(|c| c.is_ascii()) {
        let latin = crate::tokens::normalize(word).map(|w| w.to_lowercase());
        return match latin {
            Some(w) => syllables(&w),
            None => letters.len(),
        };
    }
    let vowel = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
    let mut groups = 0;
    let mut before = false;
    for &c in &letters {
        let v = vowel(c);
        if v && !before {
            groups += 1;
        }
        before = v;
    }
    let n = letters.len();
    let silent_e = n > 2 && letters[n - 1] == 'e' && {
        let prev = letters[n - 2];
        // `-le` after a consonant is a syllable of its own (ta-ble).
        let le_syllable = prev == 'l' && !vowel(letters[n - 3]);
        !(vowel(prev) || le_syllable)
    };
    if silent_e && groups > 1 {
        groups -= 1;
    }
    groups.max(1)
}

/// Gives the words of `lines` under `confident` (or without both times)
/// times between their neighbours, by syllables; a word set by hand
/// ([`crate::WordEdit::pin`]) keeps its own. Words that cannot be bounded
/// on either side are left as they are.
pub fn interpolate(lines: &mut [Line], confident: f32) {
    let known = |w: &Word| w.is_timed() && (w.edit.pin.is_some() || w.confidence >= confident);
    for l in 0..lines.len() {
        let count = lines[l].words.len();
        let mut i = 0;
        while i < count {
            if known(&lines[l].words[i]) {
                i += 1;
                continue;
            }
            let a = i;
            while i < count && !known(&lines[l].words[i]) {
                i += 1;
            }
            let b = i;
            // The neighbours: (time, whether it is in this line).
            let left = if a > 0 {
                lines[l].words[a - 1].end.map(|t| (t, true))
            } else if let Some(t) = lines[l].time {
                Some((t, true))
            } else {
                lines[..l]
                    .iter()
                    .rev()
                    .flat_map(|line| line.words.iter().rev())
                    .find(|w| known(w))
                    .and_then(|w| w.end)
                    .map(|t| (t, false))
            };
            let right = if b < count {
                lines[l].words[b].start.map(|t| (t, true))
            } else {
                lines[l + 1..]
                    .iter()
                    .find_map(|line| {
                        line.time
                            .or_else(|| line.words.iter().find(|w| known(w)).and_then(|w| w.start))
                    })
                    .map(|t| (t, false))
            };
            let weights: Vec<f64> = lines[l].words[a..b]
                .iter()
                .map(|w| syllables(&w.text) as f64)
                .collect();
            let total = weights.iter().sum::<f64>().max(1.0);
            let (start, length) = match (left, right) {
                (Some((l0, in_line)), Some((r0, _))) if r0 >= l0 => {
                    let room = r0 - l0;
                    // Between two times of the line, its words fill them;
                    // beside one (its first words before its first sure
                    // one, its last after its last) they go at a sung pace,
                    // a line being sung in one go; a whole line between
                    // two others at the slowest.
                    let length = if in_line && right.is_some_and(|r| r.1) {
                        room
                    } else if in_line || right.is_some_and(|r| r.1) {
                        room.min(total * TYPICAL)
                    } else {
                        room.min(total * SLOWEST)
                    };
                    // From the start of the line, else up to its end.
                    if in_line {
                        (l0, length)
                    } else {
                        (r0 - length, length)
                    }
                }
                (Some((l0, _)), _) => (l0, total * TYPICAL),
                (None, Some((r0, _))) => {
                    let length = r0.min(total * TYPICAL);
                    (r0 - length, length)
                }
                (None, None) => continue,
            };
            across(&mut lines[l].words[a..b], start, length, &weights);
        }
    }
}

/// `words` timed one after another from `start` for `length` seconds, each
/// its share by `weights` (one a word); their letters' times dropped.
pub fn across(words: &mut [Word], start: f64, length: f64, weights: &[f64]) {
    let total = weights.iter().sum::<f64>().max(1.0);
    let mut at = 0.0;
    for (w, weight) in words.iter_mut().zip(weights) {
        w.start = Some(start + length * at / total);
        at += weight;
        w.end = Some(start + length * at / total);
        w.letters.clear();
    }
}

/// `lyrics` with each line's words timed across it by syllables, from the
/// line times (a word's own trusted times are kept).
pub fn split_lines(lyrics: &Lyrics) -> Lyrics {
    let mut out = lyrics.clone();
    interpolate(&mut out.lines, CONFIDENT);
    out.timing = Timing {
        detail: Lyrics::detail_of_lines(&out.lines),
        by: Some(maker()),
        fit: None,
    };
    out
}

/// `lyrics` spread over `stretches` (start and end seconds, in order) in
/// proportion to the syllables: the words go along the stretches as if
/// they were one; a word that would straddle a gap starts after it. Each
/// line's time is its first word's start.
pub fn spread_over(lyrics: &Lyrics, stretches: &[(f64, f64)]) -> Lyrics {
    let mut out = lyrics.clone();
    let active: f64 = stretches.iter().map(|(s, e)| (e - s).max(0.0)).sum();
    let total: f64 = out
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .map(|w| syllables(&w.text) as f64)
        .sum();
    if active <= 0.0 || total <= 0.0 {
        return out;
    }
    // The time `x` seconds of voice in, and which stretch that is.
    let place = |x: f64| -> (f64, usize) {
        let mut before = 0.0;
        for (k, &(s, e)) in stretches.iter().enumerate() {
            let length = (e - s).max(0.0);
            if x <= before + length || k + 1 == stretches.len() {
                return (s + (x - before).min(length), k);
            }
            before += length;
        }
        (0.0, 0)
    };
    let mut at = 0.0;
    for line in &mut out.lines {
        for w in &mut line.words {
            let from = at;
            at += syllables(&w.text) as f64 * active / total;
            let (mut start, k) = place(from);
            let (end, k_end) = place(at);
            if k_end != k {
                start = stretches[k_end].0.min(end);
            }
            w.clear_times();
            (w.start, w.end) = (Some(start), Some(end));
        }
        line.time = line.words.first().and_then(|w| w.start);
    }
    out.timing = Timing {
        detail: Detail::Words,
        by: Some(maker()),
        fit: None,
    };
    out
}

/// Where the voice is heard in `vocals_16k` (mono at [`VOCALS_RATE`]), as
/// (start, end) in seconds: 20 ms frames within 30 dB of the loudest
/// (its 95th percentile), and above -60 dB, with gaps under 0.4 s closed
/// and stretches under 0.25 s dropped.
pub fn vocal_stretches(vocals_16k: &[f32]) -> Vec<(f64, f64)> {
    let size = VOCALS_RATE as usize / 50;
    let level: Vec<f64> = vocals_16k
        .chunks(size)
        .map(|c| {
            let power =
                c.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / c.len() as f64;
            10.0 * power.max(1e-12).log10()
        })
        .collect();
    if level.is_empty() {
        return Vec::new();
    }
    let mut sorted = level.clone();
    sorted.sort_by(f64::total_cmp);
    let loud = sorted[((sorted.len() - 1) as f64 * 0.95) as usize];
    let floor = (loud - 30.0).max(-60.0);
    let clock = FrameClock::of_rate(VOCALS_RATE, size, level.len());
    let mut stretches: Vec<(f64, f64)> = Vec::new();
    for (f, &db) in level.iter().enumerate() {
        if db <= floor {
            continue;
        }
        let (s, e) = (clock.time(f), clock.time(f + 1));
        match stretches.last_mut() {
            Some(last) if s - last.1 < 0.4 => last.1 = e,
            _ => stretches.push((s, e)),
        }
    }
    stretches.retain(|(s, e)| e - s >= 0.25);
    stretches
}

/// For each frame of `clock`, whether it lies outside every one of
/// `stretches` (each rounded outward, [`FrameClock::outward`]): a pause in
/// the voice.
pub fn pauses(stretches: &[(f64, f64)], clock: FrameClock) -> Vec<bool> {
    let mut out = vec![true; clock.frames];
    for &(s, e) in stretches {
        out[clock.outward(s, e)].fill(false);
    }
    out
}

/// Times for `lyrics` without a model ([`maker`]): when its source synced
/// its lines ([`Line::synced`]), each line from its synced start, its words
/// split by syllables ([`split_lines`]); else the words spread over where
/// the voice in `vocals_16k` is heard ([`spread_over`]).
pub fn fallback(vocals_16k: &[f32], lyrics: &Lyrics) -> Lyrics {
    if lyrics.lines.iter().all(|l| l.synced.is_none()) {
        return spread_over(lyrics, &vocal_stretches(vocals_16k));
    }
    let mut synced = lyrics.clone();
    for line in &mut synced.lines {
        line.time = line.synced.map(|s| s.start);
    }
    split_lines(&synced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Source;

    #[test]
    fn syllables_are_counted() {
        for (word, n) in [
            ("hello", 2),
            ("there", 1),
            ("the", 1),
            ("beautiful", 3),
            ("little", 2),
            ("time", 1),
            ("Ooh,", 1),
            ("sunshine", 2),
            ("rhythm", 1),
            ("Mädchen", 2),
            ("愛してる", 4),
            ("—", 0),
        ] {
            assert_eq!(syllables(word), n, "{word}");
        }
    }

    fn lines(texts: &[(&str, Option<f64>)]) -> Lyrics {
        Lyrics {
            lines: texts
                .iter()
                .map(|&(t, time)| Line {
                    time,
                    ..Line::from_text(t)
                })
                .collect(),
            source: Source::Unknown,
            timing: Timing {
                detail: Detail::Lines,
                by: None,
                fit: None,
            },
        }
    }

    fn word_times(l: &Line) -> Vec<(f64, f64)> {
        l.words
            .iter()
            .map(|w| (w.start.unwrap(), w.end.unwrap()))
            .collect()
    }

    fn near(a: &[(f64, f64)], b: &[(f64, f64)]) {
        assert_eq!(a.len(), b.len(), "{a:?}");
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x.0 - y.0).abs() < 1e-9 && (x.1 - y.1).abs() < 1e-9,
                "{a:?} != {b:?}"
            );
        }
    }

    #[test]
    fn line_times_are_split_by_syllables() {
        // 2 + 1 + 1 + 2 = 6 syllables from the line's time at the sung
        // pace (well inside the 3 s to the next line).
        let out = split_lines(&lines(&[
            ("hello there my darling", Some(10.0)),
            ("go", Some(13.0)),
        ]));
        let p = TYPICAL;
        near(
            &word_times(&out.lines[0]),
            &[
                (10.0, 10.0 + 2.0 * p),
                (10.0 + 2.0 * p, 10.0 + 3.0 * p),
                (10.0 + 3.0 * p, 10.0 + 4.0 * p),
                (10.0 + 4.0 * p, 10.0 + 6.0 * p),
            ],
        );
        // A line with less time than that fills it.
        let tight = split_lines(&lines(&[("hello there", Some(1.0)), ("go", Some(1.6))]));
        near(&word_times(&tight.lines[0]), &[(1.0, 1.4), (1.4, 1.6)]);
        // The last line has no next one: the typical pace.
        near(&word_times(&out.lines[1]), &[(13.0, 13.0 + TYPICAL)]);
        assert_eq!(out.timing.detail, Detail::Words);
        assert_eq!(out.timing.by, Some(Maker::new("syllables", VERSION)));
        assert!(out.lines[0].words.iter().all(|w| w.confidence == 0.0));
    }

    #[test]
    fn a_long_gap_before_the_next_line_is_not_filled() {
        // Two syllables, 20 s to the next line: at the sung pace.
        let out = split_lines(&lines(&[("oh no", Some(5.0)), ("back", Some(25.0))]));
        near(
            &word_times(&out.lines[0]),
            &[(5.0, 5.0 + TYPICAL), (5.0 + TYPICAL, 5.0 + 2.0 * TYPICAL)],
        );
    }

    #[test]
    fn without_a_model_a_synced_text_is_timed_from_its_synced_lines() {
        // Its source's line times are only its scaffold, yet without a
        // model they time its words as an LRC file's line times would.
        let lrc = "[00:10.00]hello there my darling\n[00:13.00]go\n";
        let text = Lyrics::parse_text(lrc, Source::Lrclib);
        assert!(text.lines.iter().all(|l| l.time.is_none()));
        let out = fallback(&[], &text);
        let lines = split_lines(&Lyrics::parse(lrc, Source::Lrclib));
        assert_eq!(out.lines.len(), 2);
        for (a, b) in out.lines.iter().zip(&lines.lines) {
            near(&word_times(a), &word_times(b));
        }
        assert_eq!(out.lines[0].words[0].start, Some(10.0));
        assert_eq!(out.timing.detail, Detail::Words);
        assert_eq!(out.timing.by, Some(maker()));
    }

    #[test]
    fn untrusted_words_are_filled_in_between_trusted_ones() {
        let mut l = lines(&[("one two three four", Some(0.0))]);
        let words = &mut l.lines[0].words;
        let set = |w: &mut Word, s: f64, e: f64, c: f32| {
            w.start = Some(s);
            w.end = Some(e);
            w.confidence = c;
        };
        set(&mut words[0], 1.0, 1.5, 0.9);
        set(&mut words[1], 7.0, 7.2, 0.1);
        words[2].start = None;
        set(&mut words[3], 3.5, 4.0, 0.8);
        interpolate(&mut l.lines, CONFIDENT);
        // "two" and "three" (one syllable each) share 1.5 to 3.5.
        near(
            &word_times(&l.lines[0]),
            &[(1.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.0)],
        );
        assert_eq!(l.lines[0].words[1].confidence, 0.1);
    }

    #[test]
    fn a_line_s_unsure_first_words_go_at_a_sung_pace_before_its_first_sure_one() {
        // The line before ends at 2 s; "if I" heard weakly, "only" surely
        // at 10 s: they are sung just before it, not stretched back.
        let mut l = lines(&[("gone", None), ("if I only", None)]);
        let set = |w: &mut Word, s: f64, e: f64, c: f32| {
            w.start = Some(s);
            w.end = Some(e);
            w.confidence = c;
        };
        set(&mut l.lines[0].words[0], 1.5, 2.0, 0.9);
        set(&mut l.lines[1].words[0], 3.0, 3.1, 0.1);
        set(&mut l.lines[1].words[1], 3.1, 3.2, 0.1);
        set(&mut l.lines[1].words[2], 10.0, 10.4, 0.9);
        interpolate(&mut l.lines, CONFIDENT);
        near(
            &word_times(&l.lines[1]),
            &[
                (10.0 - 2.0 * TYPICAL, 10.0 - TYPICAL),
                (10.0 - TYPICAL, 10.0),
                (10.0, 10.4),
            ],
        );
    }

    #[test]
    fn plain_text_is_spread_over_the_voice() {
        let plain = Lyrics {
            lines: vec![Line::from_text("la la"), Line::from_text("la la")],
            source: Source::Unknown,
            timing: Timing::default(),
        };
        // 2 s of voice, then a gap, then 2 s more: four syllables, one a
        // second of voice.
        let out = spread_over(&plain, &[(1.0, 3.0), (10.0, 12.0)]);
        near(&word_times(&out.lines[0]), &[(1.0, 2.0), (2.0, 3.0)]);
        near(&word_times(&out.lines[1]), &[(10.0, 11.0), (11.0, 12.0)]);
        assert_eq!(out.lines[1].time, Some(10.0));
        // A word straddling the gap starts after it.
        let odd = Lyrics {
            lines: vec![Line::from_text("la la la")],
            ..plain.clone()
        };
        let out = spread_over(&odd, &[(0.0, 1.0), (5.0, 6.0)]);
        let w = word_times(&out.lines[0]);
        assert!((w[1].0 - 5.0).abs() < 1e-9 && (w[1].1 - (5.0 + 1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn the_voice_is_found_where_it_is_loud() {
        let rate = VOCALS_RATE as usize;
        let mut v = vec![0.0f32; rate * 6];
        // A tone from 1 s to 2.5 s, and from 4 s to 5 s with a 0.2 s drop.
        for (i, s) in v.iter_mut().enumerate() {
            let t = i as f64 / rate as f64;
            let on =
                (1.0..2.5).contains(&t) || ((4.0..5.0).contains(&t) && !(4.4..4.6).contains(&t));
            if on {
                *s = (0.3 * (t * 440.0 * std::f64::consts::TAU).sin()) as f32;
            }
        }
        let found = vocal_stretches(&v);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!((found[0].0 - 1.0).abs() < 0.03 && (found[0].1 - 2.5).abs() < 0.03);
        assert!((found[1].0 - 4.0).abs() < 0.03 && (found[1].1 - 5.0).abs() < 0.03);
        let plain = Lyrics {
            lines: vec![Line::from_text("la la")],
            source: Source::Unknown,
            timing: Timing::default(),
        };
        let out = fallback(&v, &plain);
        assert_eq!(out.lines[0].words[0].start, Some(found[0].0));
        assert_eq!(out.timing.by, Some(maker()));
        // The pauses are the frames outside them; the short drop is none.
        let clock = FrameClock {
            frame_seconds: 0.02,
            frames: 300,
        };
        let p = pauses(&found, clock);
        for (f, pause) in [
            (40, true),
            (60, false),
            (140, true),
            (225, false),
            (290, true),
        ] {
            assert_eq!(p[f], pause, "frame {f}");
        }
    }
}
