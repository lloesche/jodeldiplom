use super::*;
use jodel_rhythm::waveform::Melody;
use std::path::Path;
use std::sync::Arc;

/// One line of words a second each, word `k` on `pitches[k]` (a pair: its
/// first half on the one, its second on the other).
fn song_of(level: Level, pitches: &[(u16, u16)]) -> Singing {
    let n = pitches.len();
    let mut m = Melody {
        cents: vec![0; n * 100 + 100],
        sure: vec![0; n * 100 + 100],
    };
    for (k, &(a, b)) in pitches.iter().enumerate() {
        for i in 0..100 {
            m.cents[k * 100 + i] = if i < 50 { a } else { b };
            m.sure[k * 100 + i] = 255;
        }
    }
    let words = (0..n)
        .map(|k| jodel_lyrics::Word {
            start: Some(k as f64),
            end: Some(k as f64 + 1.0),
            confidence: 0.9,
            ..jodel_lyrics::Word::new("lala")
        })
        .collect();
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines: vec![jodel_lyrics::Line {
            words,
            ..jodel_lyrics::Line::default()
        }],
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &m, 1, level);
    s
}

/// "lala" four times: the first on C4 then G4 (two notes), the others on
/// E4, G4, C4.
fn song(level: Level) -> Singing {
    song_of(
        level,
        &[(6000, 6700), (6400, 6400), (6700, 6700), (6000, 6000)],
    )
}

fn melody_at(s: &Singing, t: f64) -> Option<f32> {
    s.word_at(t).map(|(_, c)| c as f32)
}

/// `s` hears its melody on the first `share(k)` of each word `k` (nothing
/// on the rest), a frame every 10 ms from `from` to `to` seconds.
fn sing(s: &mut Singing, share: impl Fn(usize) -> f64, (from, to): (f64, f64)) {
    let mut i = (from * 100.0).round() as u32;
    while f64::from(i) / 100.0 <= to {
        let t = f64::from(i) / 100.0;
        let on = t.fract() < share(t as usize) - 1e-9;
        let c = melody_at(s, t).filter(|_| on);
        s.hear(t, c);
        i += 1;
    }
}

#[test]
fn a_word_counted_on_easy_fills_all_its_bubbles_at_once() {
    let mut s = song(Level::Easy);
    assert_eq!(s.lines[0].notes.len(), 5, "the first word on two notes");
    // 15 % of each word sung: each counts at its 11th frame hit (a tenth
    // of the 100 it will have had, and one more), its second note not sung
    // at all.
    sing(&mut s, |_| 0.15, (0.0, 4.5));
    let b = line(&s, 0, 4.5);
    for (k, x) in b.notes.iter().enumerate() {
        let w = s.lines[0].notes[k].word as f64;
        assert!(
            x.full_at.is_some_and(|f| (f - (w + 0.10)).abs() < 1e-6) && x.by_word,
            "{k}: {x:?}"
        );
    }
    assert_eq!(b.words[0].notes, 0..2);
    assert_eq!(b.words[0].time, (0.0, 1.0));
    assert!(b.notes[1].spans.is_empty(), "never sung, full all the same");
    assert_eq!(b.words_full(), 4);
    assert_eq!(words_full(&s, 5.0), (4, 4));
}

#[test]
fn medium_fills_where_a_pitch_hit_and_is_full_only_filled() {
    let mut s = song(Level::Medium);
    sing(&mut s, |_| 0.15, (0.0, 4.5));
    let b = line(&s, 0, 4.5);
    assert!(b.notes.iter().all(|x| x.full_at.is_none()));
    assert_eq!(b.notes[0].spans.len(), 1);
    let (a, z, full) = b.notes[0].spans[0];
    assert!(a == 0.0 && (z - 0.15).abs() < 1e-9 && full);
    assert_eq!(words_full(&s, 5.0), (0, 4));
    // Sung all through: each bubble full once 85 % is filled.
    let mut s = song(Level::Medium);
    sing(&mut s, |_| 1.0, (0.0, 4.5));
    let b = line(&s, 0, 4.5);
    assert!(b.notes[0].full_at.is_some_and(|f| (f - 0.42).abs() < 1e-6));
    assert!(b.notes.iter().all(|x| x.full_at.is_some() && !x.by_word));
}

#[test]
fn hard_fills_as_before_and_is_full_only_truly_filled() {
    let mut s = song(Level::Hard);
    sing(&mut s, |_| 0.5, (0.0, 4.5));
    let b = line(&s, 0, 4.5);
    let spans: Vec<(f64, f64, bool)> = b.notes.iter().flat_map(|x| x.spans.clone()).collect();
    assert_eq!(spans.len(), 4, "{spans:?}");
    assert!(spans
        .iter()
        .all(|s| s.2 && ((s.1 - s.0) - 0.5).abs() < 1e-6));
    // The first word's first note (0 to 0.5 s) sung whole: it alone full.
    assert!(b.notes[0].full_at.is_some());
    assert!(b.notes[1..].iter().all(|x| x.full_at.is_none()));
    assert_eq!(b.words_full(), 0);
}

/// On easy a word's bubbles are full exactly when the scoring counts the
/// word hit, at shares about the tenth it asks.
#[test]
fn on_easy_a_word_is_full_exactly_when_the_scoring_counts_it() {
    let shares = [0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.2, 0.0, 0.095, 0.105];
    let pitches: Vec<(u16, u16)> = (0..shares.len())
        .map(|k| [(6000, 6000), (6400, 6700), (6200, 6200)][k % 3])
        .collect();
    let mut s = song_of(Level::Easy, &pitches);
    let mut scored: Vec<(f64, Option<Hit>)> = Vec::new();
    let end = shares.len() as f64 + 0.5;
    let mut i = 0;
    while f64::from(i) / 100.0 <= end {
        let t = f64::from(i) / 100.0;
        let k = t as usize;
        let on = k < shares.len() && t.fract() < shares[k] - 1e-9;
        let c = melody_at(&s, t).filter(|_| on);
        s.hear(t, c);
        if let Some(w) = s
            .tally
            .last_scored
            .filter(|w| scored.last().is_none_or(|l| l.0 != w.end))
        {
            scored.push((w.end, w.hit));
        }
        i += 1;
    }
    let b = line(&s, 0, end);
    assert_eq!(scored.len(), shares.len());
    for (w, &(at, hit)) in b.words.iter().zip(&scored) {
        assert_eq!(w.time.1, at);
        assert_eq!(
            w.full_at.is_some(),
            hit == Some(Hit::Full),
            "word {} ({}): {w:?}",
            w.word,
            shares[w.word]
        );
    }
    assert_eq!(b.words_full() as u32, s.tally.hit_words);
}

/// A bubble is full once, its time fixed by the frames up to it: fed a
/// frame at a time, a time once given never moves.
#[test]
fn each_bubble_is_full_once_its_time_fixed_by_the_frames_up_to_it() {
    let mut s = song(Level::Easy);
    let mut fixed: Vec<Option<f64>> = vec![None; 5];
    for i in 0..=450 {
        let t = f64::from(i) / 100.0;
        sing(&mut s, |_| 0.3, (t, t));
        for (k, x) in line(&s, 0, t).notes.iter().enumerate() {
            match fixed[k] {
                Some(a) => assert_eq!(Some(a), x.full_at, "{k} moved at {t}"),
                None => fixed[k] = x.full_at,
            }
        }
    }
    assert!(fixed.iter().all(Option::is_some));
}

/// Sought back under a second (the delay nudged mid-song), what the frames
/// from there on did is taken back and they are heard again: sung
/// differently the second time, the scoring and the bubbles are those of
/// the singing as it ended up, not of both, and the score before stays.
/// Sought back further, the song starts over.
#[test]
fn a_seek_back_takes_back_what_was_heard_from_there() {
    for level in Level::ALL {
        // First pass: words 0 to 2 sung all through, to 2.8 s (word 0
        // scored, word 1 too); back to 1.9 s; then only 5 % of each word.
        let first = |k: usize| if k < 3 { 1.0 } else { 0.0 };
        let second = |_| 0.05;
        let mut s = song(level);
        sing(&mut s, first, (0.0, 2.8));
        assert_eq!(s.tally.scored_words, 2, "{level:?}");
        sing(&mut s, second, (1.9, 4.5));
        // As if sung so straight through.
        let mut straight = song(level);
        sing(
            &mut straight,
            |k| if k < 1 { 1.0 } else { 0.0 },
            (0.0, 0.99),
        );
        sing(
            &mut straight,
            |k| if k < 3 { 1.0 } else { 0.0 },
            (1.0, 1.89),
        );
        sing(&mut straight, second, (1.9, 4.5));
        let tally = |s: &Singing| {
            (
                s.tally.score,
                s.tally.hit_words,
                s.tally.half_words,
                s.tally.scored_words,
                s.tally.streak,
            )
        };
        assert_eq!(tally(&s), tally(&straight), "{level:?}");
        assert!(s.heard.windows(2).all(|w| w[0].t < w[1].t), "in time");
        assert_eq!(s.heard.len(), straight.heard.len());
        assert_eq!(line(&s, 0, 4.5), line(&straight, 0, 4.5), "{level:?}");
        for w in 0..4 {
            assert_eq!(s.noted(0, w), straight.noted(0, w), "{level:?} {w}");
        }
    }
    // From the start again after 3 s: over, nothing full yet.
    let mut s = song(Level::Medium);
    sing(&mut s, |_| 1.0, (0.0, 3.0));
    sing(&mut s, |_| 0.0, (0.0, 0.3));
    assert_eq!(s.tally.scored_words, 0);
    assert!(line(&s, 0, 0.3).notes.iter().all(|x| x.full_at.is_none()));
}
