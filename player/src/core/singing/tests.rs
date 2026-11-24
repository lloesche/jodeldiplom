//! The singing's tests: notes under their letters, the scoring and the
//! key shift.

use std::path::Path;

use super::*;

/// A melody sure at `cents` over the frames `runs` give.
fn melody(frames: usize, runs: &[(usize, usize, u16)]) -> Melody {
    let mut m = Melody {
        cents: vec![0; frames],
        sure: vec![0; frames],
    };
    for &(a, b, c) in runs {
        for i in a..b {
            m.cents[i] = c;
            m.sure[i] = 255;
        }
    }
    m
}

fn word(text: &str, s: f64, e: f64) -> jodel_lyrics::Word {
    jodel_lyrics::Word {
        start: Some(s),
        end: Some(e),
        confidence: 0.9,
        ..jodel_lyrics::Word::new(text)
    }
}

fn line(words: Vec<jodel_lyrics::Word>) -> jodel_lyrics::Line {
    jodel_lyrics::Line {
        words,
        ..jodel_lyrics::Line::default()
    }
}

#[test]
fn a_line_s_notes_go_under_their_letters_as_high_as_they_are_sung() {
    // "go" on C4; "up" (its "p" from 0.4 to 0.75 s) on C4, then a
    // fifth up, then an octave and a third up (heard an octave wrong:
    // a third up, between).
    let m = melody(
        200,
        &[
            (0, 20, 6000),
            (30, 45, 6000),
            (45, 60, 6700),
            (60, 75, 7600),
        ],
    );
    let mut up = word("up", 0.3, 0.75);
    up.letters = vec![0.0, 0.1];
    let mut go = word("go", 0.0, 0.2);
    go.letters = vec![0.0, 0.1];
    let l = line(vec![go, up]);
    let notes = notes_of(&l, &m);
    let r = |x: f32| (x * 100.0).round() / 100.0;
    let got: Vec<(f32, f32, f32)> = notes
        .notes
        .iter()
        .map(|n| (r(n.from), r(n.to), r(n.height)))
        .collect();
    // The middle is C4 (half the singing); spread over 4 semitones
    // each way at least (here 7: the fifth at the top).
    assert_eq!(
        got,
        [
            (0.0, 2.0, 0.0),
            (3.0, 4.14, 0.0),
            (4.14, 4.57, 1.0),
            (4.57, 5.0, 0.57)
        ]
    );
    // Over the words' whole time, the octave folded back.
    assert_eq!((notes.notes[1].start, notes.notes[3].end), (0.3, 0.75));
    assert_eq!(notes.notes[3].cents, 6400.0);
    // A backing word gets none: here that leaves one note, too few.
    let mut backing = l.clone();
    backing.words[1].backing = true;
    assert!(notes_of(&backing, &m).notes.is_empty());
}

#[test]
fn a_word_without_a_pitch_gets_one_between_its_neighbours() {
    // "low" on F3, "mid" with no pitch at all, "high" a fifth up on C4.
    let m = melody(300, &[(0, 50, 5300), (200, 250, 6000)]);
    let l = line(vec![
        word("low", 0.0, 0.5),
        word("mid", 1.0, 1.5),
        word("high", 2.0, 2.5),
    ]);
    let got: Vec<(f32, f32, f32, f64, f64)> = notes_of(&l, &m)
        .notes
        .iter()
        .map(|n| {
            let h = (n.height * 100.0).round() / 100.0;
            (n.from, n.to, h, n.start, n.end)
        })
        .collect();
    // The middle the lower (half the singing), spread over 7
    // semitones: "mid" halfway, under the whole word and its time.
    assert_eq!(
        got,
        [
            (0.0, 3.0, 0.0, 0.0, 0.5),
            (4.0, 7.0, 0.5, 1.0, 1.5),
            (8.0, 12.0, 1.0, 2.0, 2.5)
        ]
    );
}

/// A song of one line, "la la la la", each word a second on its note:
/// C4, E4, G4, C4.
fn song() -> (Arc<jodel_lyrics::Lyrics>, Arc<Melody>) {
    let m = melody(
        500,
        &[
            (0, 100, 6000),
            (100, 200, 6400),
            (200, 300, 6700),
            (300, 400, 6000),
        ],
    );
    let l = line(
        (0..4)
            .map(|k| word("la", f64::from(k), f64::from(k) + 1.0))
            .collect(),
    );
    let lyrics = jodel_lyrics::Lyrics {
        lines: vec![l],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), Arc::new(m))
}

/// `singing` hears `pitch(t)` every 10 ms from 0 to 4.5 s.
fn sing(singing: &mut Singing, pitch: impl Fn(f64) -> Option<f32>) {
    for i in 0..450 {
        let t = f64::from(i) / 100.0;
        singing.hear(t, pitch(t));
    }
}

fn melody_at(t: f64) -> f32 {
    [6000.0, 6400.0, 6700.0, 6000.0][(t as usize).min(3)]
}

#[test]
fn the_song_sung_scores_every_word() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    sing(&mut s, |t| Some(melody_at(t)));
    assert_eq!(
        (s.tally.hit_words, s.tally.scored_words, s.tally.score),
        (4, 4, 400)
    );
}

#[test]
fn any_key_counts_the_shape_within_three_semitones() {
    let (lyrics, m) = song();
    let score = |off: &dyn Fn(f64) -> f32| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
        sing(&mut s, |t| Some(melody_at(t) + off(t)));
        // The words hit as sung (not rescued).
        s.tally.hit_words - s.tally.rescued
    };
    // A fourth down (the key followed from the first frames on), an
    // octave down: every word.
    assert_eq!(score(&|_| -500.0), 4);
    assert_eq!(score(&|_| -1200.0), 4);
    // The second and third two semitones flat (the key held): still the
    // melody's shape.
    let k = |t: f64| t as usize;
    assert_eq!(
        score(&|t| if k(t) == 1 || k(t) == 2 { -200.0 } else { 0.0 }),
        4
    );
    // The second a fifth off: missed.
    assert_eq!(score(&|t| if k(t) == 1 { -700.0 } else { 0.0 }), 3);
}

#[test]
fn a_hit_is_drawn_on_its_note_a_miss_as_sung_smoothed() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    // The second word a fifth flat (a fourth sharp, octaves aside): off its
    // note, drawn as sung.
    sing(&mut s, |t| {
        Some(melody_at(t) + if t as usize == 1 { -710.0 } else { 0.0 })
    });
    let at = |t: f64| {
        s.heard
            .iter()
            .find(|h| (h.t - t).abs() < 0.001)
            .map(|h| (h.cents, h.hit))
    };
    assert_eq!(at(0.5), Some((6000.0, true)));
    let (cents, hit) = at(1.5).unwrap();
    assert!((cents - 6890.0).abs() < 0.01 && !hit, "{cents}");
    // Off the notes a frame now and then a semitone and a half off: drawn
    // as the rest (the median).
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    sing(&mut s, |t| {
        let i = (t * 100.0).round() as u32;
        Some(melody_at(t) + if i.is_multiple_of(4) { 850.0 } else { 700.0 })
    });
    let h = s.heard.iter().find(|h| (h.t - 0.4).abs() < 0.001).unwrap();
    assert!(h.hit, "the key followed: on its note");
}

/// A word asks a tenth of its note time: 15 % hit is a word, 5 % is not.
#[test]
fn a_word_asks_a_tenth_of_its_note_time() {
    let (lyrics, m) = song();
    let hit_words = |share: f64| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
        // In tune for the first `share` of each word, then nothing.
        sing(&mut s, |t| (t.fract() < share - 1e-9).then(|| melody_at(t)));
        s.tally.hit_words
    };
    assert_eq!(hit_words(0.15), 4);
    assert_eq!(hit_words(0.05), 0);
}

#[test]
fn medals_go_by_the_words() {
    assert_eq!(Medal::of(4, 4), Some(Medal::Gold));
    assert_eq!(Medal::of(3, 4), Some(Medal::Silver));
    assert_eq!(Medal::of(2, 4), Some(Medal::Bronze));
    assert_eq!(Medal::of(1, 4), None);
    assert_eq!(Medal::of(0, 0), None);
}

#[test]
fn a_streak_multiplies_the_points_and_two_misses_end_it() {
    assert_eq!(
        [0, 4, 5, 9, 10, 19, 20, 99].map(multiplier),
        [1, 1, 2, 2, 3, 3, 4, 4]
    );
    // A song of 14 words, a second each on C4: the 12th and 13th missed,
    // the others sung.
    let s = fourteen(&[11, 12]);
    // Words 1-4 at ×1, 5-9 at ×2, 10-11 at ×3; the 12th waits for a
    // rescue, the 13th missed too ends the streak; the 14th at ×1 again.
    assert_eq!(s.tally.score, 4 * 100 + 5 * 200 + 2 * 300 + 100);
    assert_eq!((s.tally.streak, s.tally.best_streak), (1, 11));
    assert_eq!((s.tally.hit_words, s.tally.rescued), (12, 0));
    let last = s.tally.last_scored.unwrap();
    assert_eq!((last.hit, last.points, last.streak), (true, 100, 1));
    let lost = s.moments().iter().filter(|m| m.cue == Cue::StreakLost);
    assert_eq!(lost.count(), 1);
    assert!(!s
        .moments()
        .iter()
        .any(|m| matches!(m.cue, Cue::Shot { .. })));
}

/// A single word missed, the next burst: it fires a shot at the word
/// missed, which is rescued (scored as if sung, the streak going on) and
/// bursts as the shot lands.
#[test]
fn a_single_word_missed_is_rescued_by_the_next_to_burst() {
    let s = fourteen(&[12]);
    // Words 1-4 at ×1, 5-9 at ×2, 10-14 at ×3 (the 13th rescued).
    assert_eq!(s.tally.score, 4 * 100 + 5 * 200 + 5 * 300);
    assert_eq!((s.tally.streak, s.tally.best_streak), (14, 14));
    assert_eq!((s.tally.hit_words, s.tally.rescued), (14, 1));
    assert!(!s.moments().iter().any(|m| m.cue == Cue::StreakLost));
    // The shot: from the 14th's bubble as it bursts, at the 13th.
    let shots: Vec<_> = (s.moments().iter())
        .filter_map(|m| match m.cue {
            Cue::Shot { from, to, .. } => Some((m.at, from, to)),
            _ => None,
        })
        .collect();
    let [(fired, from, to)] = shots[..] else {
        panic!("one shot: {shots:?}");
    };
    let popped = |w: usize| {
        s.moments().iter().find_map(|m| match m.cue {
            Cue::Pop(b) if b.word == w => Some(m.at),
            _ => None,
        })
    };
    assert_eq!((from, to), ((0, 13), (0, 12)));
    // As it bursts, or as the word missed is scored, whichever is later.
    let burst = popped(13).expect("the 14th bursts");
    assert!(
        fired >= burst - 1e-9 && fired < burst + 0.05,
        "{fired} {burst}"
    );
    let rescue = s.rescue_of(0, 12).expect("rescued");
    assert_eq!((rescue.from, rescue.fired), ((0, 13), fired));
    let lands = fired + crate::core::bubbles::FLIGHT;
    assert!((rescue.lands() - lands).abs() < 1e-9);
    let rescued = popped(12).expect("the word rescued bursts");
    assert!((rescued - lands - crate::core::bubbles::SWEEP).abs() < 1e-9);
}

/// A word missed, then the song sought forward past the next word, then
/// a word hit: the word passed unheard ends the wait, nothing is
/// rescued over them (no shot), and the streak goes on.
#[test]
fn a_seek_forward_past_a_word_missed_rescues_nothing() {
    let (lyrics, m) = fourteen_song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    // Words 1 to 11 sung, the 12th missed; on from 12 s to 13.5 s.
    for i in (0..1200).chain(1350..1500) {
        let t = f64::from(i) / 100.0;
        let out = (11.0..12.0).contains(&t);
        s.hear(t, (!out).then_some(6000.0));
    }
    s.finish();
    assert!(!s
        .moments()
        .iter()
        .any(|m| matches!(m.cue, Cue::Shot { .. })));
    assert_eq!(s.tally.rescued, 0);
    assert_eq!(s.rescues().count(), 0);
    // The 12th missed and the 14th hit scored, the 13th not: the streak
    // not lost.
    assert_eq!((s.tally.scored_words, s.tally.streak), (13, 12));
}

/// A song of 14 words, a second each on C4, sung on all but those
/// `missed` (by their place): heard to its end.
fn fourteen(missed: &[usize]) -> Singing {
    let (lyrics, m) = fourteen_song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    for i in 0..1500 {
        let t = f64::from(i) / 100.0;
        let out = missed.contains(&(t as usize));
        s.hear(t, (!out).then_some(6000.0));
    }
    s
}

/// A song of 14 words, a second each on C4, and its melody.
fn fourteen_song() -> (Arc<jodel_lyrics::Lyrics>, Melody) {
    let m = melody(1500, &[(0, 1400, 6000)]);
    let l = line(
        (0..14)
            .map(|k| word("la", f64::from(k), f64::from(k) + 1.0))
            .collect(),
    );
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![l],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    (lyrics, m)
}

#[test]
fn silence_scores_nothing_and_singing_again_from_the_start_starts_over() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    sing(&mut s, |_| None);
    assert_eq!(
        (s.tally.hit_words, s.tally.scored_words, s.tally.score),
        (0, 4, 0)
    );
    // From the start again: over.
    sing(&mut s, |t| Some(melody_at(t)));
    assert_eq!((s.tally.hit_words, s.tally.scored_words), (4, 4));
}

/// Played two semitones up, the notes are, the singing held to them; the
/// heights stay, and a change mid-song moves the notes from then.
#[test]
fn in_another_key_the_notes_move_and_the_singing_is_held_to_them() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.set_key(2);
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    assert_eq!(s.word_at(0.5).map(|(_, c)| c), Some(6200.0));
    sing(&mut s, |t| Some(melody_at(t) + 200.0));
    assert_eq!(s.tally.hit_words, 4);
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &crate::core::singing::melodic(&m));
    let heights: Vec<f32> = s.notes.lines[0].notes.iter().map(|n| n.height).collect();
    s.set_key(2);
    sing(&mut s, |t| Some(melody_at(t) + 200.0));
    let moved: Vec<f32> = s.notes.lines[0].notes.iter().map(|n| n.height).collect();
    assert_eq!(heights, moved);
    assert_eq!(
        s.notes.lines[0].height(6600.0),
        s.notes.lines[0].notes[1].height
    );
    // Down to −1 mid-song: what was heard moves with the notes.
    let before = s.heard[0].cents;
    s.set_key(-1);
    assert_eq!(s.heard[0].cents, before - 300.0);
    assert_eq!(s.word_at(1.5).map(|(_, c)| c), Some(6300.0));
    assert_eq!(s.shift(), -1);
}
