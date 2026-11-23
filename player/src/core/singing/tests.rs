//! The singing's tests: notes under their letters, the levels, the
//! scoring and the key shift.

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
fn the_song_sung_scores_every_word_at_every_level() {
    let (lyrics, m) = song();
    for level in Level::ALL {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        sing(&mut s, |t| Some(melody_at(t)));
        assert_eq!(
            (s.tally.hit_words, s.tally.scored_words, s.tally.score),
            (4, 4, 400),
            "{level:?}"
        );
    }
}

#[test]
fn a_fourth_down_counts_on_easy_and_medium_but_not_on_hard() {
    let (lyrics, m) = song();
    let score = |level| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        sing(&mut s, |t| Some(melody_at(t) - 500.0));
        s.tally.hit_words
    };
    // The key followed from the first frames on: every word.
    assert_eq!(score(Level::Easy), 4);
    assert_eq!(score(Level::Medium), 4);
    assert_eq!(score(Level::Hard), 0);
    // An octave down counts even on hard.
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Hard);
    sing(&mut s, |t| Some(melody_at(t) - 1200.0));
    assert_eq!(s.tally.hit_words, 4);
}

#[test]
fn the_shape_is_enough_on_easy_the_steps_on_medium() {
    let (lyrics, m) = song();
    // Every note 2 semitones off its own way (the key held), the second
    // and third flat by two semitones: within easy's 3, not medium's 1.5.
    let off = |t: f64| {
        let k = t as usize;
        Some(melody_at(t) + if k == 1 || k == 2 { -200.0 } else { 0.0 })
    };
    let score = |level| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        sing(&mut s, off);
        s.tally.hit_words
    };
    assert_eq!(score(Level::Easy), 4);
    assert_eq!(score(Level::Medium), 2);
}

#[test]
fn close_on_medium_scores_half_and_draws_on_the_nearest_semitone() {
    let (lyrics, m) = song();
    // The second word 2 semitones flat: beyond medium's 1.5, within its
    // half hits' 3.
    let off = |t: f64| Some(melody_at(t) + if t as usize == 1 { -210.0 } else { 0.0 });
    let sung = |level| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        sing(&mut s, off);
        s
    };
    let s = sung(Level::Medium);
    assert_eq!(
        (s.tally.hit_words, s.tally.half_words, s.tally.score),
        (3, 1, 350)
    );
    // The half hit kept the streak: the first, third and last hit.
    assert_eq!((s.tally.streak, s.tally.best_streak), (3, 3));
    // Drawn on the note where it hit, a semitone step off where it half
    // hit.
    let at = |s: &Singing, t: f64| {
        s.heard
            .iter()
            .find(|h| (h.t - t).abs() < 0.001)
            .map(|h| (h.cents, h.hit))
    };
    assert_eq!(at(&s, 0.5), Some((6000.0, Some(Hit::Full))));
    assert_eq!(at(&s, 1.5), Some((6200.0, Some(Hit::Half))));
    // On easy it all hits, on its notes.
    let s = sung(Level::Easy);
    assert_eq!(
        (s.tally.hit_words, s.tally.half_words, s.tally.score),
        (4, 0, 400)
    );
    assert_eq!(at(&s, 1.5), Some((6400.0, Some(Hit::Full))));
    // On hard: as sung, no half hits.
    let s = sung(Level::Hard);
    assert_eq!((s.tally.hit_words, s.tally.half_words), (3, 0));
    let (cents, hit) = at(&s, 1.5).unwrap();
    assert!((cents - 6190.0).abs() < 0.01 && hit.is_none(), "{cents}");
}

#[test]
fn the_pitch_drawn_on_hard_is_smoothed() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Hard);
    // A frame now and then a semitone and a half sharp: drawn as the
    // rest.
    sing(&mut s, |t| {
        let i = (t * 100.0).round() as u32;
        Some(melody_at(t) + if i.is_multiple_of(4) { 150.0 } else { 20.0 })
    });
    let h = s.heard.iter().find(|h| (h.t - 0.4).abs() < 0.001).unwrap();
    assert!((h.cents - 6020.0).abs() < 0.01, "{}", h.cents);
}

/// Each level asks its own share of a word's note time: 15 % hit is a
/// word on easy (a tenth), not on medium (a quarter); 30 % is one on
/// medium, not on hard (two fifths).
#[test]
fn each_level_asks_its_own_share_of_a_word() {
    let (lyrics, m) = song();
    let hit_words = |level, share: f64| {
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        // In tune for the first `share` of each word, then nothing.
        sing(&mut s, |t| (t.fract() < share - 1e-9).then(|| melody_at(t)));
        s.tally.hit_words
    };
    assert_eq!(hit_words(Level::Easy, 0.15), 4);
    assert_eq!(hit_words(Level::Medium, 0.15), 0);
    assert_eq!(hit_words(Level::Medium, 0.30), 4);
    assert_eq!(hit_words(Level::Hard, 0.30), 0);
    assert_eq!(hit_words(Level::Hard, 0.45), 4);
}

#[test]
fn medals_go_by_the_words() {
    assert_eq!(Medal::of(4, 0, 4), Some(Medal::Gold));
    // Half the words half hit, the rest hit: 75 %.
    assert_eq!(Medal::of(2, 2, 4), Some(Medal::Silver));
    assert_eq!(Medal::of(2, 0, 4), Some(Medal::Bronze));
    assert_eq!(Medal::of(1, 1, 4), None);
    assert_eq!(Medal::of(0, 0, 0), None);
}

#[test]
fn a_streak_multiplies_the_points_and_a_miss_ends_it() {
    assert_eq!(
        [0, 4, 5, 9, 10, 19, 20, 99].map(multiplier),
        [1, 1, 2, 2, 3, 3, 4, 4]
    );
    // A song of 14 words, a second each on C4: the first 12 sung, the
    // 13th missed, the last sung.
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
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &Arc::new(m), 1, Level::Medium);
    for i in 0..1500 {
        let t = f64::from(i) / 100.0;
        let missed = (12.0..13.0).contains(&t);
        s.hear(t, (!missed).then_some(6000.0));
    }
    // Words 1-4 at ×1, 5-9 at ×2, 10-12 at ×3, the 13th nothing, the
    // 14th at ×1 again.
    assert_eq!(s.tally.score, 4 * 100 + 5 * 200 + 3 * 300 + 100);
    assert_eq!((s.tally.streak, s.tally.best_streak), (1, 12));
    let last = s.tally.last_scored.unwrap();
    assert_eq!(
        (last.hit, last.points, last.streak),
        (Some(Hit::Full), 100, 1)
    );
}

#[test]
fn silence_scores_nothing_and_singing_again_from_the_start_starts_over() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Medium);
    sing(&mut s, |_| None);
    assert_eq!(
        (s.tally.hit_words, s.tally.scored_words, s.tally.score),
        (0, 4, 0)
    );
    // From the start again: over.
    sing(&mut s, |t| Some(melody_at(t)));
    assert_eq!((s.tally.hit_words, s.tally.scored_words), (4, 4));
}

/// Played two semitones up, the notes are: the song sung in its key no
/// longer scores on hard, sung in the new key it does, on every level; the
/// heights stay, and a change mid-song moves the notes from then.
#[test]
fn in_another_key_the_notes_move_and_the_singing_is_held_to_them() {
    let (lyrics, m) = song();
    for level in Level::ALL {
        let mut s = Singing::default();
        s.set_key(2);
        s.prepare(Path::new("a"), &lyrics, &m, 1, level);
        assert_eq!(s.word_at(0.5).map(|(_, c)| c), Some(6200.0));
        sing(&mut s, |t| Some(melody_at(t) + 200.0));
        assert_eq!(s.tally.hit_words, 4, "{level:?}");
    }
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Hard);
    let heights: Vec<f32> = s.lines[0].notes.iter().map(|n| n.height).collect();
    s.set_key(2);
    sing(&mut s, |t| Some(melody_at(t)));
    assert_eq!(
        s.tally.hit_words, 0,
        "the song's own key is two semitones off"
    );
    let moved: Vec<f32> = s.lines[0].notes.iter().map(|n| n.height).collect();
    assert_eq!(heights, moved);
    assert_eq!(s.lines[0].height(6600.0), s.lines[0].notes[1].height);
    // Down to −1 mid-song: what was heard moves with the notes.
    let before = s.heard[0].cents;
    s.set_key(-1);
    assert_eq!(s.heard[0].cents, before - 300.0);
    assert_eq!(s.word_at(1.5).map(|(_, c)| c), Some(6300.0));
    assert_eq!(s.shift(), -1);
}
