//! The shine's tests: the tiers' bars, a phrase measured and held against
//! the original's, weak singing never Diva, a reference as text, the live
//! scoring.

use std::path::Path;
use std::sync::Arc;

use super::measure::yardstick;
use super::reference::{keyed, measure_all, Original, VERSION};
use super::*;
use crate::core::singing::{melodic, Notes, Singing};
use jodel_rhythm::waveform::Melody;

/// A song of two phrases of four words, half a second each, on a melody
/// that moves (C4, E4, G4, D4 over and over), a second apart; its lyrics
/// and melody.
fn song() -> (Arc<jodel_lyrics::Lyrics>, Melody) {
    const PITCH: [u16; 4] = [6000, 6400, 6700, 6200];
    let mut m = Melody {
        cents: vec![0; 700],
        sure: vec![0; 700],
    };
    let mut lines = Vec::new();
    for l in 0..2 {
        let from = 1.0 + 3.0 * l as f64;
        let words = (0..4)
            .map(|k| {
                let s = from + 0.5 * k as f64;
                for i in (s * 100.0) as usize..((s + 0.5) * 100.0) as usize {
                    m.cents[i] = PITCH[k];
                    m.sure[i] = 255;
                }
                jodel_lyrics::Word {
                    start: Some(s),
                    end: Some(s + 0.5),
                    confidence: 0.9,
                    ..jodel_lyrics::Word::new("la")
                }
            })
            .collect();
        lines.push(jodel_lyrics::Line {
            words,
            ..jodel_lyrics::Line::default()
        });
    }
    let lyrics = Arc::new(jodel_lyrics::Lyrics {
        lines,
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    });
    (lyrics, m)
}

fn notes() -> Notes {
    let (lyrics, m) = song();
    Notes::new(&lyrics, &m)
}

/// The frames of a singer `off` cents from the notes (a share `voiced` of
/// the frames with a pitch, as a voice's consonants and breaths have none;
/// only where `sings` says), a frame each 10 ms over 7 s.
fn sung(
    n: &Notes,
    off: impl Fn(usize) -> f64,
    voiced: f64,
    sings: impl Fn(f64) -> bool,
) -> Vec<(f64, Option<f32>)> {
    (0..700)
        .map(|i| {
            let t = f64::from(i) / 100.0;
            let on = (f64::from(i) * 0.618_034).fract() < voiced && sings(t);
            let c = n
                .word_at(t)
                .filter(|_| on)
                .map(|(_, c)| (c + off(i as usize)) as f32);
            (t, c)
        })
        .collect()
}

fn always(_: f64) -> bool {
    true
}

/// The original measured on itself: each phrase's yardstick.
fn yardsticks(n: &Notes, frames: &[(f64, Option<f32>)]) -> Vec<Option<Yardstick>> {
    let frames = keyed(n, frames);
    (0..n.lines.len())
        .map(|l| yardstick(n, l, &frames))
        .collect()
}

/// `frames` against the yardsticks `y`: each phrase's shine.
fn shines_of(n: &Notes, frames: &[(f64, Option<f32>)], y: &[Option<Yardstick>]) -> Vec<Shine> {
    measure_all(n, &keyed(n, frames), y)
        .iter()
        .zip(y)
        .map(|(m, y)| shine(m.as_ref().unwrap(), y.as_ref().unwrap()))
        .collect()
}

#[test]
fn the_original_shines_as_itself_and_silence_not_at_all() {
    let n = notes();
    let original = sung(&n, |_| 0.0, 0.6, always);
    let y = yardsticks(&n, &original);
    assert!(shines(y[0].as_ref().unwrap()));
    let same = shines_of(&n, &original, &y)[0];
    assert!((same.value - 1.0).abs() < 1e-9, "{same:?}");
    assert_eq!(Tier::of(&same, Par::default()), Some(Tier::TOP));
    // Silence: nothing hit, out of time, no tier.
    let quiet = shines_of(&n, &sung(&n, |_| 0.0, 0.0, always), &y)[0];
    assert!(quiet.value < 0.1, "{quiet:?}");
    assert_eq!(quiet.timing, Some(0.15 / 1.15), "out of time: no fit");
    assert_eq!(Tier::of(&quiet, Par::default()), None);
}

#[test]
fn a_singer_off_the_notes_shines_less_in_another_key_alike() {
    let n = notes();
    let y = yardsticks(&n, &sung(&n, |_| 0.0, 0.6, always));
    // A fourth down, as in tune: the singer's own key.
    let fourth = shines_of(&n, &sung(&n, |_| -500.0, 0.6, always), &y)[1];
    assert!(fourth.value > 0.95, "{fourth:?}");
    // Most frames near a semitone off: the pitch's share lower, the
    // coverage whole.
    let wobbly = shines_of(
        &n,
        &sung(&n, |i| [80.0, 0.0, -80.0][i % 3], 0.6, always),
        &y,
    )[1];
    assert!(wobbly.coverage > 0.99, "{wobbly:?}");
    assert!(wobbly.pitch.unwrap() < 0.3, "{wobbly:?}");
    assert!(wobbly.value < fourth.value);
}

#[test]
fn the_bars_are_a_share_of_what_the_original_reaches() {
    let at = |value: f64, pitch: Option<f64>| Shine {
        value,
        coverage: 1.0,
        pitch,
        ..Shine::default()
    };
    let par = Par::default();
    assert_eq!(Tier::of(&at(0.59, None), par), None);
    assert_eq!(Tier::of(&at(0.60, None), par), Some(Tier::named("Good")));
    assert_eq!(Tier::of(&at(0.80, None), par), Some(Tier::named("Great")));
    assert_eq!(Tier::of(&at(0.88, None), par), Some(Tier::named("Amazing")));
    assert_eq!(Tier::of(&at(0.95, Some(0.95)), par), Some(Tier::TOP));
    // Diva asks the pitch too.
    assert_eq!(
        Tier::of(&at(0.95, Some(0.85)), par),
        Some(Tier::named("Amazing"))
    );
    // An original that reaches only 0.8 (its everyday hearings): 90 % of
    // that is Diva, the others as much lower.
    let rough = Par {
        shine: 0.8,
        pitch: 0.8,
    };
    assert_eq!(Tier::of(&at(0.73, Some(0.73)), rough), Some(Tier::TOP));
    assert_eq!(
        Tier::of(&at(0.73, Some(0.70)), rough),
        Some(Tier::named("Amazing"))
    );
    assert_eq!(Tier::of(&at(0.48, None), rough), Some(Tier::named("Good")));
    let shone = Shone::new(0, 1.0, &at(0.72, None), rough);
    assert!((shone.of_top() - 1.0).abs() < 1e-6);
}

/// On a phrase whose original reaches as little of itself as a phrase may
/// and still shine ([`reference::PAR_LEAST`]), near silence and half of it
/// sung (every other word, the rest perfect) are never Diva; a phrase
/// whose original reaches less has no shine at all.
#[test]
fn weak_singing_is_never_diva_however_low_the_par() {
    let n = notes();
    let y = yardsticks(&n, &sung(&n, |_| 0.0, 0.6, always));
    let low = Par {
        shine: reference::PAR_LEAST,
        pitch: reference::PAR_LEAST,
    };
    let half_words = |t: f64| ((t - 1.0) / 0.5).floor() as i64 % 2 == 0;
    for (name, frames) in [
        ("near silence", sung(&n, |_| 0.0, 0.05, always)),
        ("half the words", sung(&n, |_| 0.0, 0.6, half_words)),
    ] {
        for s in shines_of(&n, &frames, &y) {
            assert_ne!(Tier::of(&s, low), Some(Tier::TOP), "{name}: {s:?}");
        }
    }
    let reference = Reference {
        phrases: vec![
            Some(Phrase {
                yardstick: y[0].clone().unwrap(),
                par: Par {
                    shine: reference::PAR_LEAST - 0.01,
                    pitch: 1.0,
                },
            }),
            Some(Phrase {
                yardstick: y[1].clone().unwrap(),
                par: low,
            }),
        ],
    };
    assert!(reference.phrase(0).is_none() && reference.phrase(1).is_some());
}

/// The reference from a voice: the original's own phrases, each one's par
/// what its everyday hearings reach; the same as text and back.
#[test]
fn a_reference_from_a_voice_and_as_text() {
    let n = notes();
    // A voice singing the notes: a sine at each note's pitch, at 16 kHz.
    let rate = 16_000;
    let voice: Vec<f32> = (0..rate as usize * 7)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            match n.word_at(t) {
                Some((_, c)) => {
                    let hz = 440.0 * 2f64.powf((c - 6900.0) / 1200.0);
                    (0.3 * (std::f64::consts::TAU * hz * t).sin()) as f32
                }
                None => 0.0,
            }
        })
        .collect();
    let r = Original::new(&n, &voice, rate).reference();
    assert_eq!(r.phrases.len(), 2);
    assert_eq!(r.shining(), 2);
    for p in r.phrases.iter().flatten() {
        assert!(p.par.shine > 0.8 && p.par.pitch > 0.8, "{:?}", p.par);
    }
    let text = r.to_text();
    assert_eq!(Reference::from_text(&text), Some(r));
    assert_eq!(
        Reference::from_text(&text.replace(
            &format!("version {VERSION}"),
            &format!("version {}", VERSION - 1)
        )),
        None
    );
}

/// A reference of `n`'s phrases from `frames`, each at the par given.
fn reference_of(n: &Notes, frames: &[(f64, Option<f32>)]) -> Arc<Reference> {
    Arc::new(Reference {
        phrases: yardsticks(n, frames)
            .into_iter()
            .map(|y| {
                y.map(|yardstick| Phrase {
                    yardstick,
                    par: Par::default(),
                })
            })
            .collect(),
    })
}

/// The live singing: each phrase scored as its frames are in, against the
/// reference set, as a moment; none without one; the last scored by
/// [`Singing::finish`].
#[test]
fn the_singing_scores_each_phrase_as_it_ends() {
    let (lyrics, m) = song();
    let mut s = Singing::default();
    s.prepare(Path::new("a"), &lyrics, &melodic(&m));
    let frames = sung(&s.notes, |_| 0.0, 0.6, always);
    for &(t, c) in &frames {
        s.hear(t, c);
    }
    assert_eq!(s.shone().count(), 0, "no reference, no shine");
    s.restart();
    s.set_reference(Some(reference_of(&s.notes, &frames)));
    for &(t, c) in &frames[..400] {
        s.hear(t, c);
    }
    // The first phrase (1 to 3 s) scored once its frames are in.
    assert_eq!(s.shone().count(), 1);
    let last = *s.last_phrase().unwrap();
    assert_eq!((last.line, last.tier), (0, Some(Tier::TOP)));
    // The second (4 to 6 s), its frames stopping as it ends: scored as
    // the song is finished.
    for &(t, c) in &frames[400..605] {
        s.hear(t, c);
    }
    assert_eq!(s.shone().count(), 1);
    s.finish();
    let tiers: Vec<_> = s.shone().map(|p| p.tier).collect();
    assert_eq!(tiers, [Some(Tier::TOP), Some(Tier::TOP)]);
}
