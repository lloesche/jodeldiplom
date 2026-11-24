use super::*;
use crate::tokens::{normalize, WAV2VEC2_960H};
use crate::{Line, Source};

pub(crate) const FS: f64 = 0.02;

/// Emissions of `frames` frames where the planted token of each frame
/// (the blank where none is) has probability `p` and the others share
/// the rest.
pub(crate) fn planted(frames: usize, plants: &[(usize, u32)], p: f32) -> Emissions {
    let mut on = vec![WAV2VEC2_960H.blank; frames];
    for &(f, t) in plants {
        on[f] = t;
    }
    emissions_of(&on, p)
}

pub(crate) fn emissions_of(on: &[u32], p: f32) -> Emissions {
    let v = WAV2VEC2_960H.len();
    let other = ((1.0 - p) / (v - 1) as f32).ln();
    let mut logp = vec![other; on.len() * v];
    for (f, &t) in on.iter().enumerate() {
        logp[f * v + t as usize] = p.ln();
    }
    Emissions {
        logp,
        tokens: v,
        frame_seconds: FS,
    }
}

/// Plants `word`'s letters from `frame`, `each` frames a letter with a
/// blank between, then a word break after; gives the frame after.
pub(crate) fn plant(on: &mut [u32], word: &str, frame: usize, each: usize) -> usize {
    let mut f = frame;
    let letters = normalize(word).unwrap();
    for (i, c) in letters.chars().enumerate() {
        if i > 0 {
            f += 1; // a blank
        }
        for _ in 0..each {
            on[f] = WAV2VEC2_960H.id(c).unwrap();
            f += 1;
        }
    }
    f
}

pub(crate) fn lyrics(lines: &[&str]) -> Lyrics {
    Lyrics {
        lines: lines.iter().map(|text| Line::from_text(text)).collect(),
        source: Source::Unknown,
        timing: Timing::default(),
    }
}

fn times(l: &Lyrics) -> Vec<(f64, f64)> {
    l.lines
        .iter()
        .flat_map(|l| &l.words)
        .map(|w| (w.start.unwrap(), w.end.unwrap()))
        .collect()
}

fn close(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.0 - y.0).abs() < 1e-9 && (x.1 - y.1).abs() < 1e-9)
}

#[test]
fn the_path_finds_planted_word_times_exactly() {
    let mut on = vec![WAV2VEC2_960H.blank; 400];
    let sep = WAV2VEC2_960H.separator;
    // "Hello world, all ok": words at planted frames, a break between.
    let mut planted_at = Vec::new();
    let mut f = 20;
    for (word, gap) in [("Hello", 15), ("world,", 40), ("all", 7), ("ok", 0)] {
        let end = plant(&mut on, word, f, 2);
        planted_at.push((f as f64 * FS, end as f64 * FS));
        if gap > 0 {
            on[end + gap / 2] = sep;
        }
        f = end + gap;
    }
    let e = emissions_of(&on, 0.9);
    let text = lyrics(&["Hello world,", "all ok"]);
    let out = align_emissions(
        &e,
        &text,
        &WAV2VEC2_960H,
        &CtcSettings::default(),
        &Hints::default(),
    )
    .unwrap();
    assert!(close(&times(&out), &planted_at), "{:?}", times(&out));
    assert!(out
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .all(|w| w.confidence > 0.85));
    assert_eq!(out.timing.detail, Detail::Words);
    // Each letter where it was planted (three frames apart); the comma,
    // not sung, at the letter's before it.
    let words: Vec<&crate::Word> = out.lines.iter().flat_map(|l| &l.words).collect();
    let hundredths = |w: &crate::Word| -> Vec<u32> {
        w.letters
            .iter()
            .map(|t| (t * 100.0).round() as u32)
            .collect()
    };
    assert_eq!(hundredths(words[0]), [0, 6, 12, 18, 24]);
    assert_eq!(hundredths(words[1]), [0, 6, 12, 18, 24, 24]);
    let third = words[0].letter_at(2).unwrap();
    assert!((third - (20.0 + 6.0) * FS).abs() < 1e-6, "{third}");
}

#[test]
fn repeated_letters_need_a_blank_between() {
    // "LL" can only be two tokens with a blank between.
    let l = WAV2VEC2_960H.id('L').unwrap();
    let e = planted(3, &[(0, l), (2, l)], 0.9);
    let path = force_align(&e, &[l, l], 0, &[], &Costs::default()).unwrap();
    assert_eq!(path.states, [1, 2, 3]);
    let e = planted(1, &[(0, l)], 0.9);
    assert!(force_align(&e, &[l, l], 0, &[], &Costs::default()).is_none());
}

#[test]
fn a_word_is_held_over_singing_but_not_over_a_pause() {
    // "go": a sure G at frame 10 and an O at 200, and a weaker G at
    // 196. From the first G the word is surer; it is a held note
    // where the voice sings on, and letters heard in a pause where it
    // does not.
    let (g, o) = (
        WAV2VEC2_960H.id('G').unwrap(),
        WAV2VEC2_960H.id('O').unwrap(),
    );
    let mut e = planted(300, &[(10, g), (200, o)], 0.9);
    let v = WAV2VEC2_960H.len();
    e.logp[196 * v + g as usize] = 0.5f32.ln();
    e.logp[196 * v + WAV2VEC2_960H.blank as usize] = 0.4f32.ln();
    let text = lyrics(&["go"]);
    let settings = CtcSettings::default();
    let start = |pauses: &[bool]| {
        let hints = Hints {
            pauses: pauses.to_vec(),
            ..Hints::default()
        };
        align_emissions(&e, &text, &WAV2VEC2_960H, &settings, &hints)
            .unwrap()
            .lines[0]
            .words[0]
            .start
            .unwrap()
    };
    assert_eq!(start(&[]), 10.0 * FS);
    assert_eq!(start(&[false; 300]), 10.0 * FS);
    let mut pauses = vec![false; 300];
    pauses[11..196].fill(true);
    assert_eq!(start(&pauses), 196.0 * FS);
    // Between words, a pause is free.
    let mut on = vec![WAV2VEC2_960H.blank; 300];
    let a = plant(&mut on, "go", 10, 2);
    on[a + 1] = WAV2VEC2_960H.separator;
    let b = plant(&mut on, "on", 200, 2);
    let e = emissions_of(&on, 0.9);
    let out = align_emissions(
        &e,
        &lyrics(&["go on"]),
        &WAV2VEC2_960H,
        &settings,
        &Hints {
            pauses: vec![true; 300],
            ..Hints::default()
        },
    )
    .unwrap();
    assert!(close(
        &times(&out),
        &[(10.0 * FS, a as f64 * FS), (200.0 * FS, b as f64 * FS)]
    ));
}

/// Frame `f` of `e` with `token` at probability `p`, the blank at
/// `blank`, the other tokens sharing the rest.
pub(crate) fn set(e: &mut Emissions, f: usize, token: u32, p: f32, blank: f32) {
    let v = WAV2VEC2_960H.len();
    let other = ((1.0 - p - blank) / (v - 2) as f32).ln();
    for t in 0..v {
        e.logp[f * v + t] = other;
    }
    e.logp[f * v + token as usize] = p.ln();
    e.logp[f * v + WAV2VEC2_960H.blank as usize] = blank.ln();
}

/// Plants `word` in `e` from `frame`, two frames a letter and a blank
/// between, each letter at `p` and the blank under it at `blank`; gives
/// the frame after.
pub(crate) fn plant_at(e: &mut Emissions, word: &str, frame: usize, p: f32, blank: f32) -> usize {
    let mut f = frame;
    for (i, c) in normalize(word).unwrap().chars().enumerate() {
        if i > 0 {
            f += 1;
        }
        for _ in 0..2 {
            set(e, f, WAV2VEC2_960H.id(c).unwrap(), p, blank);
            f += 1;
        }
    }
    f
}

#[test]
fn a_word_heard_elsewhere_costs_its_pin() {
    // "sun" heard surely at 0.4 s (a sound like it) and less surely
    // at 6 s, where a recogniser heard it.
    let mut e = planted(400, &[], 0.9);
    plant_at(&mut e, "sun", 20, 0.9, 0.09);
    plant_at(&mut e, "sun", 300, 0.4, 0.5);
    let text = lyrics(&["sun"]);
    let start = |settings: &CtcSettings, windows: Vec<Option<(f64, f64)>>| {
        let hints = Hints {
            windows,
            ..Hints::default()
        };
        align_emissions(&e, &text, &WAV2VEC2_960H, settings, &hints)
            .unwrap()
            .lines[0]
            .words[0]
            .start
            .unwrap()
    };
    let settings = CtcSettings::default();
    assert_eq!(start(&settings, vec![]), 20.0 * FS);
    assert_eq!(start(&settings, vec![Some((5.0, 7.0))]), 300.0 * FS);
    // Unpinned, or a window that holds the sure one, changes nothing.
    let free = CtcSettings {
        window_penalty: 0.0,
        ..settings.clone()
    };
    assert_eq!(start(&free, vec![Some((5.0, 7.0))]), 20.0 * FS);
    assert_eq!(start(&settings, vec![Some((0.0, 2.0))]), 20.0 * FS);
}

#[test]
fn the_opening_stays_near_the_lines_after_it() {
    // "go", then "one two three four" sung surely from 8 s. "go" is
    // heard a little more surely at 0.4 s (something sung earlier)
    // than at 7.4 s, just before the rest.
    let mut e = planted(600, &[], 0.9);
    plant_at(&mut e, "go", 20, 0.7, 0.25);
    plant_at(&mut e, "go", 370, 0.6, 0.35);
    let mut f = 400;
    for word in ["one", "two", "three", "four"] {
        f = plant_at(&mut e, word, f, 0.95, 0.04);
        set(&mut e, f + 1, WAV2VEC2_960H.separator, 0.95, 0.04);
        f += 4;
    }
    let text = lyrics(&["go", "one two three four"]);
    let start = |settings: &CtcSettings| {
        align_emissions(&e, &text, &WAV2VEC2_960H, settings, &Hints::default())
            .unwrap()
            .lines[0]
            .words[0]
            .start
            .unwrap()
    };
    let loose = CtcSettings {
        opening_pull: 0.0,
        ..CtcSettings::default()
    };
    assert_eq!(start(&loose), 20.0 * FS);
    assert_eq!(start(&CtcSettings::default()), 370.0 * FS);
}

fn glimpse(start: f64, text: &str) -> Part {
    Part {
        start,
        end: start + crate::glimpses::GLIMPSE,
        text: text.to_owned(),
    }
}

#[test]
fn a_line_starts_where_the_glimpses_heard_it_not_on_the_chant_before() {
    // "it hurts" at 1 s; a chant's "yo" at 3 s heard surely as "you";
    // the line "do you wanna" sung at 4.6 s, its first two words heard
    // less surely.
    let mut e = planted(500, &[], 0.9);
    let mut f = plant_at(&mut e, "it", 50, 0.95, 0.04);
    plant_at(&mut e, "hurts", f + 2, 0.95, 0.04);
    plant_at(&mut e, "you", 150, 0.95, 0.04);
    f = plant_at(&mut e, "do", 230, 0.5, 0.45);
    f = plant_at(&mut e, "you", f + 2, 0.5, 0.45);
    plant_at(&mut e, "wanna", f + 2, 0.95, 0.04);
    let text = lyrics(&["it hurts", "do you wanna"]);
    let start = |glimpses: Vec<Part>| {
        let hints = Hints {
            glimpses,
            ..Hints::default()
        };
        align_emissions(&e, &text, &WAV2VEC2_960H, &CtcSettings::default(), &hints)
            .unwrap()
            .lines[1]
            .words[0]
            .start
            .unwrap()
    };
    assert!(start(Vec::new()) < 3.0, "{}", start(Vec::new()));
    let heard = vec![
        glimpse(0.0, "It hurts."),
        glimpse(1.5, "It hurts. Yeah, yeah."),
        glimpse(3.0, "Yeah. Do you wanna?"),
        glimpse(4.5, "Do you wanna?"),
        glimpse(6.0, ""),
    ];
    assert_eq!(start(heard), 230.0 * FS);
}

#[test]
fn a_line_does_not_go_on_over_a_stretch_with_no_words_heard() {
    // "come home" sung at 1 s; an "oh" at 6 s, where nothing with words
    // was sung, sounds more like "home" than the real one.
    let mut e = planted(500, &[], 0.9);
    let f = plant_at(&mut e, "come", 50, 0.95, 0.04);
    plant_at(&mut e, "home", f + 2, 0.5, 0.45);
    plant_at(&mut e, "home", 300, 0.6, 0.35);
    let text = lyrics(&["come home"]);
    let end = |glimpses: Vec<Part>| {
        let hints = Hints {
            glimpses,
            ..Hints::default()
        };
        align_emissions(&e, &text, &WAV2VEC2_960H, &CtcSettings::default(), &hints)
            .unwrap()
            .lines[0]
            .words[1]
            .end
            .unwrap()
    };
    assert!(end(Vec::new()) > 6.0);
    let heard = vec![
        glimpse(0.0, "Come home."),
        glimpse(1.5, ""),
        glimpse(3.0, ""),
        glimpse(4.5, ""),
        glimpse(6.0, ""),
        glimpse(7.5, "Go."),
    ];
    assert!(end(heard) < 2.0);
}

#[test]
fn a_backing_word_not_sung_is_left_out_and_filled_in() {
    let mut on = vec![WAV2VEC2_960H.blank; 300];
    let sep = WAV2VEC2_960H.separator;
    let a = plant(&mut on, "come", 10, 2);
    on[a + 5] = sep;
    let b = plant(&mut on, "home", a + 40, 2);
    let e = emissions_of(&on, 0.9);
    let text = lyrics(&["come (yeah yeah) home"]);
    let out = align_emissions(
        &e,
        &text,
        &WAV2VEC2_960H,
        &CtcSettings::default(),
        &Hints::default(),
    )
    .unwrap();
    let w = &out.lines[0].words;
    assert_eq!(
        (w[0].start, w[0].end),
        (Some(10.0 * FS), Some(a as f64 * FS))
    );
    assert_eq!(
        (w[3].start, w[3].end),
        (Some((a + 40) as f64 * FS), Some(b as f64 * FS))
    );
    // Left out: timed between, with no confidence.
    for backing in &w[1..3] {
        assert_eq!(backing.confidence, 0.0);
        let (s, e) = (backing.start.unwrap(), backing.end.unwrap());
        assert!(s >= w[0].end.unwrap() && e <= w[3].start.unwrap() && s < e);
    }
    // Sung, the same words are found.
    let mut on2 = vec![WAV2VEC2_960H.blank; 300];
    let a = plant(&mut on2, "come", 10, 2);
    on2[a + 2] = sep;
    let y1 = plant(&mut on2, "yeah", a + 5, 2);
    on2[y1 + 2] = sep;
    let y2 = plant(&mut on2, "yeah", y1 + 5, 2);
    on2[y2 + 2] = sep;
    plant(&mut on2, "home", y2 + 5, 2);
    let e = emissions_of(&on2, 0.9);
    let out = align_emissions(
        &e,
        &text,
        &WAV2VEC2_960H,
        &CtcSettings::default(),
        &Hints::default(),
    )
    .unwrap();
    let w = &out.lines[0].words;
    assert_eq!(w[1].start, Some((a + 5) as f64 * FS));
    assert_eq!(w[2].start, Some((y1 + 5) as f64 * FS));
    assert!(w[1].confidence > 0.8);
}

#[test]
fn too_few_frames_give_no_path() {
    let mut on = vec![WAV2VEC2_960H.blank; 200];
    plant(&mut on, "hi", 20, 2);
    let short = emissions_of(&on[..2], 0.9);
    let text = lyrics(&["hi", "ho"]);
    let settings = CtcSettings::default();
    assert!(align_emissions(&short, &text, &WAV2VEC2_960H, &settings, &Hints::default()).is_none());
}
