//! The source's line times as a scaffold for the aligner.
//!
//! LRCLIB's synced lyrics say when each line is sung, to a second or so,
//! but often for another version of the song (a live one, a longer one, a
//! radio edit with another intro), and a text's lines say nothing of its
//! words. Our own alignment places the words, and is right on most lines;
//! where it goes wrong, it goes far (a line put in an intro's chant, a
//! chorus taken for another, a word held on over a pause). So the source's
//! line times are a scaffold, not the timing:
//!
//! 1. **Laid on the text** ([`attach`]): a record's synced text and its
//!    plain one are matched word for word; a line of the text whose first
//!    word starts a synced line gets that line's time, and the time after
//!    it ([`crate::Synced`]). A synced text that is not the same text gets
//!    none.
//! 2. **Checked against the recording** ([`check`]): the song is aligned
//!    without them, and the lines' starts compared with theirs: the median
//!    of the differences is the offset between the versions, and the
//!    share of lines within [`AGREE`] of it says whether they fit
//!    ([`Verdict::trusted`]). One that does not fit is passed over, and the
//!    song timed as without.
//! 3. **Holding the words** ([`windows`], [`align`]): where they fit, the
//!    song is aligned again with each line's words costing
//!    ([`crate::ctc::CtcSettings::scaffold`]) to start, and for every
//!    frame, outside their line's window, from [`BEFORE`] before its time
//!    to [`AFTER`] after the next (both moved by the offset). Within the
//!    window the words go where they are heard, so the lines still switch
//!    on our words, not on the source's times.
//! 4. **Not where the voice says otherwise** ([`held`], [`HEARD_LESS`]):
//!    a fitting source still has lines wrong (an intro's lines all synced
//!    at its first second, a hook's four times synced in one), so a line
//!    gets no window where its source gives it less time than its words
//!    take ([`FASTEST`]), where the voice is heard less at its source's
//!    time than where we put it, or where the recogniser's glimpses heard
//!    it where we put it and not there; and a line the window moved to
//!    where the model no longer hears its words surely is let go again.
//!
//! Measured on 153 songs (`research/lrcscaffold`): of the lines in songs
//! whose times fit, about one in twenty was over 2.5 s from its source's
//! time; where the recogniser's glimpses could tell, the source was right
//! about as often as we were, so it is no authority line by line.

use crate::ctc::{align_held, CtcSettings, Emissions, FrameClock, Hints};
use crate::tokens::Vocab;
use crate::words::{common, words, Compared};
use crate::{Line, Lyrics, Synced};

/// The least share of a text's words, and of the synced text's, the two
/// must have in common, in order, for the synced times to be laid on it.
pub const SAME_WORDS: f32 = 0.8;

/// How far, in seconds, a line's start may be from the source's time (with
/// the offset) to agree.
pub const AGREE: f64 = 1.0;

/// The least share of lines that agree for the source's times to fit.
pub const AGREE_LEAST: f32 = 0.6;

/// The fewest lines with times, aligned, for the check to tell.
pub const LEAST_LINES: usize = 8;

/// How far before its source's time (with the offset) a line's window
/// opens, in seconds: the source's times are a second or two off.
pub const BEFORE: f64 = 2.5;

/// How far after the source's next time (with the offset) a line's window
/// closes, in seconds.
pub const AFTER: f64 = 2.0;

/// Lays the line times of `synced` (a synced text of the same song, read
/// with [`Lyrics::parse_text`]) on the lines of `text` as their
/// [`crate::Line::synced`]: the two texts' words matched in order (a
/// longest common subsequence), a line of `text` whose first word is
/// matched to the first of a synced line gets that line's start, and the
/// end of the synced line its last word matched is in. Only when at least
/// [`SAME_WORDS`] of either text's words are matched: otherwise (another
/// text, or one with its repeats written out) none. The number of lines
/// given a time.
pub fn attach(text: &mut Lyrics, synced: &Lyrics) -> usize {
    for line in &mut text.lines {
        line.synced = None;
    }
    let ours = Compared::of(text);
    let theirs = Compared::of(synced);
    let pairs = common(&ours.words, &theirs.words);
    let enough = |n: usize| n > 0 && pairs.len() as f32 >= SAME_WORDS * n as f32;
    if !enough(ours.words.len()) || !enough(theirs.words.len()) {
        return 0;
    }
    let mut pair_of = vec![None; ours.words.len()];
    for &(i, j) in &pairs {
        pair_of[i] = Some(j);
    }
    let mut given = 0;
    let mut i = 0;
    while i < ours.words.len() {
        let line = ours.line[i];
        let next = ours.first_of(line + 1);
        let start = pair_of[i]
            .filter(|&j| theirs.first_of(theirs.line[j]) == j)
            .and_then(|j| synced.lines[theirs.line[j]].synced);
        let last = (i..next).rev().find_map(|k| pair_of[k]);
        if let (Some(start), Some(last)) = (start, last) {
            let end = synced.lines[theirs.line[last]]
                .synced
                .map_or(start.end, |s| s.end);
            text.lines[line].synced = Some(Synced {
                start: start.start,
                end,
            });
            given += 1;
        }
        i = next;
    }
    given
}

/// How a source's line times sit against a song aligned without them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict {
    /// The median of our lines' starts less the source's times, in
    /// seconds: how much later the recording has its lines.
    pub offset: f64,
    /// The share of lines whose start is within [`AGREE`] of the source's
    /// time with the offset.
    pub agree: f32,
    /// How many lines were compared.
    pub lines: usize,
}

impl Verdict {
    /// Whether the source's times fit the recording: [`LEAST_LINES`] lines
    /// compared at least, [`AGREE_LEAST`] of them agreeing.
    pub fn trusted(&self) -> bool {
        self.lines >= LEAST_LINES && self.agree >= AGREE_LEAST
    }
}

/// How the source's line times ([`crate::Line::synced`]) of `free` (a
/// song aligned without them) sit against its lines' starts (each line's
/// first word's); `None` when no line has both.
pub fn check(free: &Lyrics) -> Option<Verdict> {
    let mut diffs: Vec<f64> = free
        .lines
        .iter()
        .filter_map(|l| Some(l.words.first()?.start? - l.synced?.start))
        .collect();
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(f64::total_cmp);
    let n = diffs.len();
    let offset = if n % 2 == 1 {
        diffs[n / 2]
    } else {
        (diffs[n / 2 - 1] + diffs[n / 2]) / 2.0
    };
    let agree = diffs
        .iter()
        .filter(|d| (*d - offset).abs() <= AGREE)
        .count();
    Some(Verdict {
        offset,
        agree: agree as f32 / n as f32,
        lines: n,
    })
}

/// When the source has line `k` of `lines` over: its own end, else the
/// next line's time; `None` when neither is known.
pub fn line_end(lines: &[Line], k: usize) -> Option<f64> {
    lines[k].synced?.end.or_else(|| {
        lines[k + 1..]
            .iter()
            .find_map(|l| l.synced)
            .map(|s| s.start)
    })
}

/// For each line of `lyrics` (a recording `seconds` long), the seconds its
/// words are held within by the source's times with `verdict`'s offset:
/// from [`BEFORE`] before its time to [`AFTER`] after its end
/// ([`line_end`]; else no end). `None` for a line the source has no time
/// for, one it gives less time than its words could be sung in
/// ([`FASTEST`]: a run of lines synced together, a second apart, is not
/// when they are sung), and one whose window lies outside the recording
/// (it would squeeze the line into its edge).
pub fn windows(lyrics: &Lyrics, verdict: &Verdict, seconds: f64) -> Vec<Option<(f64, f64)>> {
    let lines = &lyrics.lines;
    let compared = Compared::of(lyrics);
    (0..lines.len())
        .map(|k| {
            let synced = lines[k].synced?;
            let end = line_end(lines, k);
            let words = compared.of_line(k).len();
            if end.is_some_and(|e| e - synced.start < words as f64 / FASTEST) {
                return None;
            }
            let window = (
                synced.start + verdict.offset - BEFORE,
                end.map_or(f64::INFINITY, |e| e + verdict.offset + AFTER),
            );
            (window.0 < seconds && window.1 > 0.0).then_some(window)
        })
        .collect()
}

/// The most words a second a line is sung at (fast rap is about six).
pub const FASTEST: f64 = 8.0;

/// [`windows`] for the lines of `free` (the song aligned without them),
/// less those of the lines the voice says the source has wrong. A line
/// `free` puts outside its window is left where it is when the voice is
/// heard less where the source has it, over the line's length, than where
/// it was put (by [`VOICE_MORE`] of it, by `hints`' pauses of `clock`'s
/// frames), or when the recogniser's glimpses heard its first words
/// ([`HEARD_WORDS`] of them) near where it was put and not near the
/// source's time. The recording is `clock`'s frames.
pub fn held(
    free: &Lyrics,
    verdict: &Verdict,
    hints: &Hints,
    clock: FrameClock,
) -> Vec<Option<(f64, f64)>> {
    let mut out = windows(free, verdict, clock.seconds());
    let compared = Compared::of(free);
    for (k, (line, window)) in free.lines.iter().zip(&mut out).enumerate() {
        let (Some((lo, hi)), Some(synced)) = (*window, line.synced) else {
            continue;
        };
        let (Some(ours), Some(end)) = (
            line.words.first().and_then(|w| w.start),
            line.words.iter().rev().find_map(|w| w.end),
        ) else {
            continue;
        };
        if (lo..=hi).contains(&ours) {
            continue;
        }
        let theirs = synced.start + verdict.offset;
        let length = (end - ours).max(0.5);
        let voiced = |at: f64| voiced(&hints.pauses, clock, at, at + length);
        let first = compared.of_line(k);
        let first = &first[..first.len().min(5)];
        let heard = |at: f64| heard_near(first, &hints.glimpses, at) >= HEARD_WORDS;
        let quieter = voiced(theirs) + VOICE_MORE < voiced(ours);
        let unheard = first.len() >= HEARD_WORDS && heard(ours) && !heard(theirs);
        if quieter || unheard {
            *window = None;
        }
    }
    out
}

/// How much more of a line's length the voice must be heard in where the
/// song aligned without the source's times puts it than where the source
/// has it, for the source's time of that line to be passed over.
pub const VOICE_MORE: f64 = 0.3;

/// How many of a line's first words (in order) a glimpse must hear for the
/// line to be heard there.
pub const HEARD_WORDS: usize = 2;

/// The share of the frames of `clock` from `a` to `b` seconds (rounded
/// outward, [`FrameClock::outward`], within those `pauses` covers) where
/// the voice is heard (not `pauses`); all of them when the pauses are not
/// known.
fn voiced(pauses: &[bool], clock: FrameClock, a: f64, b: f64) -> f64 {
    if pauses.is_empty() {
        return 1.0;
    }
    let frames = FrameClock {
        frames: pauses.len(),
        ..clock
    }
    .outward(a, b);
    if frames.is_empty() {
        return 0.0;
    }
    pauses[frames.clone()].iter().filter(|p| !**p).count() as f64 / frames.len() as f64
}

/// The most of `first` (a line's first words) a glimpse starting near `at`
/// (from a second after it to two before) heard, in order.
fn heard_near(first: &[String], glimpses: &[crate::asr::Part], at: f64) -> usize {
    glimpses
        .iter()
        .filter(|g| g.start - 1.0 <= at && at <= g.start + 2.0)
        .map(|g| common(first, &words(&g.text)).len())
        .max()
        .unwrap_or(0)
}

/// How much more of a line's words a line moved into its window may have
/// not heard surely ([`crate::CONFIDENT`]) than where it was, before it is
/// let go again: held where the model hears nothing of it, its words would
/// only be spread over whatever is there (silence, too).
pub const HEARD_LESS: f32 = 0.25;

/// The lines of `held` (aligned in their windows) moved from where `free`
/// had them and with [`HEARD_LESS`] more of their words not heard surely.
fn heard_less(free: &Lyrics, held: &Lyrics) -> Vec<usize> {
    let unsure = |l: &Line| {
        let n = l
            .words
            .iter()
            .filter(|w| w.confidence < crate::CONFIDENT)
            .count();
        n as f32 / l.words.len().max(1) as f32
    };
    let start = |l: &Line| l.words.first().and_then(|w| w.start);
    (0..free.lines.len())
        .filter(|&k| {
            let (a, b) = (&free.lines[k], &held.lines[k]);
            let moved = match (start(a), start(b)) {
                (Some(x), Some(y)) => (x - y).abs() > AGREE,
                _ => false,
            };
            moved && unsure(b) > unsure(a) + HEARD_LESS
        })
        .collect()
}

/// `lyrics` aligned to `emissions` ([`align_held`]), held to its
/// source's line times where they fit the recording: aligned once without
/// them, checked ([`check`]), and aligned again in their windows
/// ([`held`]) when they fit and `settings` cost anything outside them; the
/// lines then heard less surely than before ([`HEARD_LESS`]) let go, and
/// aligned once more (as without them when that finds no path). `None`
/// when no path fits.
///
/// When no line is held, the song is not aligned again: held nowhere,
/// nothing costs more than without the source's times, so the path, and
/// the lyrics, would be the same.
pub fn align(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
) -> Option<Lyrics> {
    let free = align_held(emissions, lyrics, vocab, settings, hints, &[])?;
    let Some(v) = check(&free).filter(|v| v.trusted() && settings.scaffold > 0.0) else {
        return Some(free);
    };
    let mut lines = held(&free, &v, hints, emissions.clock());
    if lines.iter().all(Option::is_none) {
        return Some(free);
    }
    let Some(first) = align_held(emissions, lyrics, vocab, settings, hints, &lines) else {
        return Some(free);
    };
    let lost = heard_less(&free, &first);
    if lost.is_empty() {
        return Some(first);
    }
    for k in lost {
        lines[k] = None;
    }
    let again = align_held(emissions, lyrics, vocab, settings, hints, &lines);
    Some(again.unwrap_or(free))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctc::tests::{plant_at, planted, FS};
    use crate::ctc::{align_emissions, align_held, align_path, heard, CtcSettings, Hints};
    use crate::tokens::WAV2VEC2_960H;
    use crate::{Line, Source, Timing};

    fn text(lrc: &str) -> Lyrics {
        Lyrics::parse_text(lrc, Source::Lrclib)
    }

    fn synced_of(l: &Lyrics) -> Vec<Option<(f64, Option<f64>)>> {
        l.lines
            .iter()
            .map(|l| l.synced.map(|s| (s.start, s.end)))
            .collect()
    }

    #[test]
    fn a_synced_text_lays_its_times_on_the_same_words() {
        let synced = text(
            "[00:10.00]Hold me close\n[00:14.00]Never let me go\n[00:20.00]\n\
             [00:30.00]Hold me close\n[00:34.50]And now I know\n",
        );
        // Its own lines: each to the next time, a break's too.
        assert_eq!(
            synced_of(&synced),
            [
                Some((10.0, Some(14.0))),
                Some((14.0, Some(20.0))),
                Some((20.0, Some(30.0))),
                Some((30.0, Some(34.5))),
                Some((34.5, None)),
            ]
        );
        // The same words written without times, a line split in two and
        // two joined: a line starting mid-way has none.
        let mut plain = text("Hold me close\nNever let me\ngo\nHold me close, and now I know\n");
        assert_eq!(attach(&mut plain, &synced), 3);
        assert_eq!(
            synced_of(&plain),
            [
                Some((10.0, Some(14.0))),
                Some((14.0, Some(20.0))),
                None,
                Some((30.0, None)),
            ]
        );
    }

    #[test]
    fn a_synced_text_of_other_words_lays_none() {
        let synced = text("[00:10.00]Hold me close\n[00:14.00]Never let me go\n");
        let mut other = text("Show me the way\nTo go home\n");
        assert_eq!(attach(&mut other, &synced), 0);
        assert!(other.lines.iter().all(|l| l.synced.is_none()));
        // Nor one with its chorus written out twice, where a line could be
        // given either time.
        let twice = text(
            "[00:10.00]Hold me close\n[00:14.00]Never let me go\n\
             [00:40.00]Hold me close\n[00:44.00]Never let me go\n",
        );
        let mut once = text("Hold me close\nNever let me go\n");
        assert_eq!(attach(&mut once, &twice), 0);
    }

    /// Lines of one word each: our start and the source's time.
    fn aligned(starts: &[(f64, f64)]) -> Lyrics {
        Lyrics {
            lines: starts
                .iter()
                .map(|&(ours, theirs)| {
                    let mut line = Line::from_text("la");
                    line.words[0].start = Some(ours);
                    line.words[0].end = Some(ours + 0.5);
                    line.synced = Some(Synced {
                        start: theirs,
                        end: None,
                    });
                    line
                })
                .collect(),
            source: Source::Lrclib,
            timing: Timing::default(),
        }
    }

    #[test]
    fn times_that_fit_the_recording_are_trusted_with_their_offset() {
        // A version 2 s later, a line put elsewhere by our timing.
        let mut lines: Vec<(f64, f64)> = (0..10)
            .map(|k| {
                let t = 10.0 + 4.0 * f64::from(k);
                (t + 2.0 + 0.1 * f64::from(k % 3), t)
            })
            .collect();
        lines[6].0 += 15.0;
        let v = check(&aligned(&lines)).unwrap();
        assert!((v.offset - 2.1).abs() < 1e-9, "{v:?}");
        assert_eq!((v.agree, v.lines), (0.9, 10));
        assert!(v.trusted());
    }

    #[test]
    fn times_of_another_version_are_not_trusted() {
        // Its second verse 30 s later than ours, its last chorus 50 s.
        let lines: Vec<(f64, f64)> = (0..12)
            .map(|k| {
                let t = 10.0 + 4.0 * f64::from(k);
                let shift = match k {
                    0..4 => 0.0,
                    4..8 => 30.0,
                    _ => 50.0,
                };
                (t, t + shift)
            })
            .collect();
        let v = check(&aligned(&lines)).unwrap();
        assert!(!v.trusted(), "{v:?}");
        // Too few lines to tell, however well they agree.
        let few: Vec<(f64, f64)> = (0..5).map(|k| (f64::from(k), f64::from(k))).collect();
        let v = check(&aligned(&few)).unwrap();
        assert_eq!(v.agree, 1.0);
        assert!(!v.trusted());
        // Nothing to compare.
        assert_eq!(check(&text("la\nla\n")), None);
    }

    #[test]
    fn a_window_runs_from_before_a_line_to_after_its_end() {
        let mut lyrics = text("[00:10.00]one\n[00:14.00]two\n[00:20.00]three\n");
        lyrics.lines[1].synced = Some(Synced {
            start: 14.0,
            end: None,
        });
        lyrics.lines.push(Line::from_text("four"));
        let v = Verdict {
            offset: 2.0,
            agree: 1.0,
            lines: 10,
        };
        let w = windows(&lyrics, &v, 100.0);
        assert_eq!(w[0], Some((12.0 - BEFORE, 16.0 + AFTER)));
        // No end of its own: the next line's time.
        assert_eq!(w[1], Some((16.0 - BEFORE, 22.0 + AFTER)));
        assert_eq!(w[2], Some((22.0 - BEFORE, f64::INFINITY)));
        assert_eq!(w[3], None);
        // Five words synced to 0.3 s: not when they are sung.
        let squashed = text("[00:10.00]one two three four five\n[00:10.30]six\n[00:12.00]seven\n");
        let w = windows(&squashed, &v, 100.0);
        assert_eq!(w[0], None);
        let (a, b) = w[1].unwrap();
        assert!((a - (12.3 - BEFORE)).abs() < 1e-9 && (b - (14.0 + AFTER)).abs() < 1e-9);
        // Past the recording's end, or all before its start: none.
        let w = windows(&lyrics, &v, 18.0);
        assert!(w[0].is_some() && w[1].is_some());
        assert_eq!(w[2], None, "opens at 19.5 s");
        let early = Verdict { offset: -30.0, ..v };
        assert!(windows(&lyrics, &early, 100.0)[..2]
            .iter()
            .all(Option::is_none));
    }

    #[test]
    fn a_line_the_voice_says_the_source_has_wrong_is_left_free() {
        // Ten lines that agree, but the source has the last at 30 s,
        // where the voice is silent, and the one before it at 26 s, where
        // it is sung but the glimpses heard none of it (they heard it where
        // we put it, at 17 s).
        let mut lines: Vec<(f64, f64)> = (0..10)
            .map(|k| {
                let t = 1.0 + 2.0 * f64::from(k);
                (t, t)
            })
            .collect();
        lines[9].1 = 30.0;
        lines[8].1 = 26.0;
        let mut free = aligned(&lines);
        free.lines[8].words = Line::from_text("come home").words;
        free.lines[8].words[0].start = Some(17.0);
        free.lines[8].words[1].end = Some(17.8);
        let v = check(&free).unwrap();
        assert!(v.trusted(), "{v:?}");
        let mut pauses = vec![false; 2000];
        pauses[1425..1750].fill(true);
        let glimpse = |start: f64, text: &str| crate::asr::Part {
            start,
            end: start + 3.0,
            text: text.into(),
        };
        let hints = Hints {
            pauses,
            glimpses: vec![glimpse(16.5, "Come home."), glimpse(25.5, "Oh.")],
            ..Hints::default()
        };
        let clock = FrameClock {
            frame_seconds: FS,
            frames: 2000,
        };
        let all = windows(&free, &v, clock.seconds());
        let kept = held(&free, &v, &hints, clock);
        assert_eq!(kept[..8], all[..8]);
        assert!(all[8].is_some() && all[9].is_some());
        assert_eq!((kept[8], kept[9]), (None, None));
        // Nothing heard or measured against them: all kept.
        assert_eq!(held(&free, &v, &Hints::default(), clock), all);
    }

    /// One-word lines sung surely every 2 s from 1 s; the last, "ten",
    /// also heard (more surely than where it is sung, with its letters and
    /// the blanks under them at `weak`, at 19 s) in the outro's echo at
    /// 23 s.
    fn ten_lines(weak: (f32, f32)) -> (crate::ctc::Emissions, Lyrics) {
        const WORDS: [&str; 10] = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ];
        let mut e = planted(1250, &[], 0.9);
        let mut lrc = String::new();
        for (k, word) in WORDS.iter().enumerate() {
            let at = 1.0 + 2.0 * k as f64;
            let frame = (at / FS).round() as usize;
            if k < 9 {
                plant_at(&mut e, word, frame, 0.95, 0.04);
            } else {
                plant_at(&mut e, word, frame, weak.0, weak.1);
            }
            lrc.push_str(&format!("[{}]{word}\n", crate::lrc::format_time(at)));
        }
        plant_at(&mut e, "ten", 1150, 0.7, 0.25);
        lrc.push_str("[00:20.50]\n");
        (e, text(&lrc))
    }

    #[test]
    fn a_word_dragged_into_the_outro_is_held_in_its_line() {
        let (e, lyrics) = ten_lines((0.5, 0.45));
        let settings = CtcSettings::default();
        let free =
            align_emissions(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        assert_eq!(free.lines[9].words[0].start, Some(1150.0 * FS));
        let kept = align(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        let verdict = check(&free).unwrap();
        assert!(
            verdict.trusted() && verdict.offset.abs() < 0.1,
            "{verdict:?}"
        );
        assert_eq!(kept.lines[9].words[0].start, Some(950.0 * FS));
        // The lines that were right stay where they were.
        for k in 0..9 {
            assert_eq!(kept.lines[k].words[0].start, free.lines[k].words[0].start);
        }
        // Without the scaffold's cost, as before.
        let off = CtcSettings {
            scaffold: 0.0,
            ..settings
        };
        let out = align(&e, &lyrics, &WAV2VEC2_960H, &off, &Hints::default()).unwrap();
        assert_eq!(out, free);
    }

    #[test]
    fn a_line_held_where_it_is_hardly_heard_is_let_go() {
        // "ten" barely heard at its source's time, and the scaffold's cost
        // high enough to hold it there all the same: it would not be heard
        // surely there, as it is in the echo.
        let (e, lyrics) = ten_lines((0.1, 0.2));
        let settings = CtcSettings {
            scaffold: 40.0,
            ..CtcSettings::default()
        };
        let free =
            align_emissions(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        let v = check(&free).unwrap();
        let lines = windows(&free, &v, 25.0);
        let forced = align_held(
            &e,
            &lyrics,
            &WAV2VEC2_960H,
            &settings,
            &Hints::default(),
            &lines,
        )
        .unwrap();
        // Not heard surely, it is timed between its neighbours, in its
        // window.
        let word = &forced.lines[9].words[0];
        let (lo, hi) = lines[9].unwrap();
        assert!(word.confidence < crate::CONFIDENT, "{word:?}");
        assert!((lo..hi).contains(&word.start.unwrap()), "{word:?}");
        assert_eq!(heard_less(&free, &forced), [9]);
        let out = align(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        assert_eq!(out.lines[9].words[0].start, Some(1150.0 * FS));
    }

    #[test]
    fn times_that_do_not_fit_leave_the_timing_as_it_was() {
        let (e, mut lyrics) = ten_lines((0.5, 0.45));
        // Another version's times: its lines everywhere but where they
        // are sung here.
        let shifts = [0.0, 7.0, 31.0, 3.0, 12.0, 45.0, 22.0, 5.0, 60.0, 0.0, 0.0];
        for (line, shift) in lyrics.lines.iter_mut().zip(shifts) {
            if let Some(s) = &mut line.synced {
                s.start += shift;
                s.end = None;
            }
        }
        let settings = CtcSettings::default();
        let free =
            align_emissions(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        let out = align(&e, &lyrics, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        assert!(!check(&free).unwrap().trusted());
        assert_eq!(out, free);
    }

    #[test]
    fn a_held_note_stays_held_within_its_window() {
        // "go" held from 0.2 s (a sure G) to 4 s (its O), the voice
        // singing on throughout; its line's time 0.5 s, over at 5 s.
        let (g, o) = (
            WAV2VEC2_960H.id('G').unwrap(),
            WAV2VEC2_960H.id('O').unwrap(),
        );
        let e = planted(300, &[(10, g), (200, o)], 0.9);
        let lyrics = text("[00:00.50]go\n[00:05.00]\n");
        let hints = Hints {
            pauses: vec![false; 300],
            ..Hints::default()
        };
        let settings = CtcSettings::default();
        let free = align_emissions(&e, &lyrics, &WAV2VEC2_960H, &settings, &hints).unwrap();
        let verdict = Verdict {
            offset: 0.0,
            agree: 1.0,
            lines: 10,
        };
        let lines = windows(&lyrics, &verdict, 6.0);
        let kept = align_held(&e, &lyrics, &WAV2VEC2_960H, &settings, &hints, &lines).unwrap();
        let word = &kept.lines[0].words[0];
        assert_eq!((word.start, word.end), (Some(10.0 * FS), Some(201.0 * FS)));
        assert_eq!(kept, free);
    }

    #[test]
    fn a_song_held_on_no_line_is_aligned_as_without_its_times() {
        // Every cost there is: pauses, pins, glimpses.
        let (e, lyrics) = ten_lines((0.5, 0.45));
        let settings = CtcSettings::default();
        let hints = Hints {
            pauses: (0..e.frames()).map(|f| f % 97 < 5).collect(),
            windows: vec![Some((0.0, 25.0)); 10],
            glimpses: vec![crate::asr::Part {
                start: 14.5,
                end: 17.5,
                text: "Seven, eight.".into(),
            }],
        };
        // Held nowhere, the alignment align skips is the one without the
        // source's times, to the bit.
        let free = align_path(&e, &lyrics, &WAV2VEC2_960H, &settings, &hints, &[]).unwrap();
        let nowhere = vec![None; lyrics.lines.len()];
        let again = align_path(&e, &lyrics, &WAV2VEC2_960H, &settings, &hints, &nowhere).unwrap();
        assert_eq!(again, free);
        let heard = |path| heard(&e, &lyrics, &WAV2VEC2_960H, path).to_bits();
        assert_eq!(heard(&again.1), heard(&free.1));
        // A source whose every line is synced too fast to be sung there
        // fits, holds no line, and leaves the song as without it.
        let mut squeezed = lyrics.clone();
        for line in &mut squeezed.lines {
            if line.is_break {
                line.synced = None;
            } else if let Some(s) = &mut line.synced {
                s.end = Some(s.start + 0.01);
            }
        }
        let free = align_held(&e, &squeezed, &WAV2VEC2_960H, &settings, &hints, &[]).unwrap();
        let v = check(&free).unwrap();
        assert!(v.trusted(), "{v:?}");
        assert!(held(&free, &v, &hints, e.clock())
            .iter()
            .all(Option::is_none));
        let out = align(&e, &squeezed, &WAV2VEC2_960H, &settings, &hints).unwrap();
        assert_eq!(out, free);
    }
}
