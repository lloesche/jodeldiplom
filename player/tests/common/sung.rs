//! The made-up song the stage's digest and its GPU timings sing
//! (`tests/fx_digest.rs`, `examples/frame_cost.rs`, `examples/fx_frames.rs`,
//! each taking this file as a module of its own): its words on their
//! notes, a frame's picture of it, the music under it, a cover put back
//! together from its layers, and effects put into a frame as the stage
//! puts them.

#![allow(dead_code)]

use std::sync::Arc;

use jodeldiplom::core::shine::{Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::cover::CoverLayers;
use jodeldiplom::gui::backdrop::music::Levels;
use jodeldiplom::gui::fx::{self, At, Board, By, Effects, Music, Out, Text};
use jodeldiplom::gui::particles::Spray;
use jodeldiplom::gui::rainbow::Glow;
use jodeldiplom::gui::stage::{over_at, screen_at, Lane, Stage};
use jodeldiplom::gui::stage_gpu::{kind, Instance, StageFrame};
use jodeldiplom::gui::streak::Step;
use jodeldiplom::tiers::TIERS;
use jodeldiplom::view::karaoke::{timed_lines, KaraokePicture, Score};
use ratatui::layout::Rect;

/// Frames a second.
pub const FPS: f64 = 60.0;
/// The song's words: text, from, to (seconds), pitch (hundredths of a MIDI
/// note), line by line, again every [`LOOP`] seconds.
pub const WORDS: [&[(&str, f64, f64, u16)]; 2] = [
    &[
        ("Shine", 0.4, 1.2, 6200),
        ("on", 1.3, 1.8, 6400),
        ("tonight", 1.9, 3.4, 6700),
    ],
    &[
        ("we", 4.2, 4.7, 6400),
        ("sing", 4.8, 5.6, 6900),
        ("along", 5.7, 7.4, 6600),
    ],
];
pub const LOOP: f64 = 8.0;
/// How long the song is, in seconds.
pub const LENGTH: f64 = 40.0;

/// The song: its lyrics, timed word by word, and its melody (each word on
/// its note, sure).
pub fn song() -> (Arc<jodel_lyrics::Lyrics>, jodel_rhythm::waveform::Melody) {
    let n = (LENGTH * 100.0) as usize;
    let mut m = jodel_rhythm::waveform::Melody {
        cents: vec![0; n],
        sure: vec![0; n],
    };
    let mut lines = Vec::new();
    let mut at = 0.0;
    while at + LOOP <= LENGTH {
        for words in WORDS {
            let words = words
                .iter()
                .map(|&(text, s, e, c)| {
                    for i in ((at + s) * 100.0) as usize..((at + e) * 100.0) as usize {
                        m.cents[i] = c;
                        m.sure[i] = 255;
                    }
                    jodel_lyrics::Word {
                        start: Some(at + s),
                        end: Some(at + e),
                        confidence: 0.9,
                        ..jodel_lyrics::Word::new(text)
                    }
                })
                .collect();
            lines.push(jodel_lyrics::Line {
                words,
                ..jodel_lyrics::Line::default()
            });
        }
        at += LOOP;
    }
    let lyrics = jodel_lyrics::Lyrics {
        lines,
        source: jodel_lyrics::Source::Unknown,
        timing: jodel_lyrics::Timing::default(),
    };
    (Arc::new(lyrics), m)
}

/// Karaoke mode's picture at `now` of the lines `sung`, scored as
/// `singing` has it.
pub fn picture_at(sung: &[&jodel_lyrics::Line], now: f64, singing: &Singing) -> KaraokePicture {
    KaraokePicture {
        area: Rect::new(0, 0, 100, 24),
        bar: jodeldiplom::core::karaoke::controls::bar(Rect::new(0, 0, 100, 24)),
        lines: timed_lines(sung, now).unwrap_or_default(),
        count: None,
        cue: None,
        controls: None,
        start: None,
        before: Vec::new(),
        moved: 1.0,
        score: Some(Score {
            points: singing.tally.score,
            streak: singing.tally.streak,
        }),
        result: None,
    }
}

/// The music under it: every stem middling.
pub fn levels() -> Levels {
    Levels {
        bass: 0.5,
        vocals: 0.6,
        other: 0.5,
    }
}

/// The cover as it was: the subject over the plate.
pub fn cover_of(l: &CoverLayers) -> image::RgbImage {
    let mut cover = l.plate.clone();
    if let Some(s) = &l.subject {
        for (p, q) in cover.pixels_mut().zip(s.pixels()) {
            let a = f32::from(q.0[3]) / 255.0;
            for c in 0..3 {
                p.0[c] = (f32::from(p.0[c]) * (1.0 - a) + f32::from(q.0[c]) * a).round() as u8;
            }
        }
    }
    cover
}

/// When and how effects are put into a frame ([`with_effects`]): fired
/// at `at` (the singing's clock: a word hit then too), to `music` (as it
/// is at a moment of the singing's clock), by a phrase reaching a row or
/// that row's rung reached (`by`).
#[derive(Clone, Copy)]
pub struct Fired<'a> {
    pub at: f64,
    pub music: &'a dyn Fn(f64) -> Music,
    pub by: By,
}

/// No music, at any moment.
pub fn still(_: f64) -> Music {
    Music::NEUTRAL
}

impl Fired<'_> {
    /// Fired at `at` by a phrase reaching the first row, with no music.
    pub fn phrase(at: f64) -> Fired<'static> {
        Fired {
            at,
            music: &still,
            by: By::Phrase(0),
        }
    }
}

/// What `board` draws at `at` in `frame`: its shapes into `shapes`, its
/// bursts into `sprays` (no words, no shake: those only over everything,
/// as [`with_effects`] lets them be).
fn draw(
    board: &mut Board,
    (at, frame): (&At, &fx::Frame),
    shapes: &mut Vec<Instance>,
    sprays: &mut Vec<Spray>,
) {
    let mut out = Out {
        shapes,
        sprays,
        letters: &mut Vec::new(),
        shake: &mut [0.0; 2],
    };
    board.draw(at, frame, &mut out);
}

/// None at all.
static NONE: Effects = Effects::NONE;

/// `base` (a frame of `stage` over `area`, at its clock) with `effects`
/// in it, as the board draws them from the row `fired` names listing
/// them, the singing at its glow, fired as it says: their screen passes,
/// the line sung in the strongest of their treatments, their bursts from
/// the playhead, what they draw over everything as it is at the frame's
/// clock (a word slammed down, lettered by `stage`, the screen shaken),
/// their looks and the music.
pub fn with_effects(
    stage: &mut Stage,
    base: &StageFrame,
    area: (f32, f32, f32, f32),
    effects: &Effects,
    fired: Fired,
) -> StageFrame {
    let clock = f64::from(base.clock);
    let Fired {
        at: fired,
        music,
        by,
    } = fired;
    let (By::Phrase(row) | By::Rung(row)) = by;
    let tier = Tier::ALL[row];
    // The rows up to the one listing the effects, the singing at its glow.
    let rows: Vec<(f32, &Effects)> = (0..=row)
        .map(|i| (TIERS[i].glow, if i == row { effects } else { &NONE }))
        .collect();
    let rows = || rows.iter().copied();
    let glow = TIERS[row].glow;
    let held = Step {
        tier: Some(tier),
        glow: Glow {
            level: glow,
            from: glow,
            since: -10.0,
        },
        at_rung: matches!(by, By::Rung(_)).then_some(tier),
        ..Step::default()
    };
    let step = match by {
        By::Phrase(_) => Step {
            phrase: Some(Shone {
                line: 0,
                end: fired,
                value: 1.0,
                top: 0.9,
                tier: Some(tier),
            }),
            word: Some(fired),
            ..held
        },
        By::Rung(_) => Step {
            rung: Some((row.checked_sub(1).map(|r| Tier::ALL[r]), Some(tier))),
            word: Some(fired),
            ..held
        },
    };
    let mut board = Board::default();
    let mut f = base.clone();
    let drawn = board.frame_of(rows(), &step, &music(fired), fired);
    let lane = Lane::new(area, clock);
    let glass = lane.glass(area);
    let screen = screen_at(area, glass);
    draw(
        &mut board,
        (&screen, &drawn),
        &mut f.screen,
        &mut Vec::new(),
    );
    if let (Some(t), At::Screen { band, .. }) = (fx::treatment(Text::Sung, &drawn), screen) {
        // The line sung: the text in the lyric band with a split along it.
        let sung = f
            .instances
            .iter_mut()
            .filter(|i| i.params[0] == kind::TEXT && i.params[1].is_finite() && i.quad[3] > band.0);
        sung.for_each(|i| t.apply(i));
    }
    let y = glass.middle.1;
    let tip = (lane.head + lane.pill * 0.42, y);
    let word = (lane.head + lane.pill * 3.0, y - lane.pill * 1.6);
    let playhead = At::Playhead {
        tip,
        pill: lane.pill,
        pps: 0.0,
        word: Some(word),
        area,
    };
    draw(
        &mut board,
        (&playhead, &drawn),
        &mut Vec::new(),
        &mut f.sprays,
    );
    // What plays out over the frames after (the rain's bursts on the beat,
    // a word slammed down), frame by frame up to the frame's clock, to the
    // music then, drawn over everything as it is at the frame's clock.
    let over = over_at(area, glass);
    let after = At::Playhead {
        tip,
        pill: lane.pill,
        pps: 0.0,
        word: None,
        area,
    };
    let (mut t, mut now) = (fired, drawn);
    while t < clock {
        draw(&mut board, (&over, &now), &mut Vec::new(), &mut f.sprays);
        t = (t + 1.0 / FPS).min(clock);
        now = board.frame_of(rows(), &held, &music(t), t);
        draw(&mut board, (&after, &now), &mut Vec::new(), &mut f.sprays);
    }
    // The last: its words lettered as the stage letters them, its shake the
    // frame's.
    let mut letters = Vec::new();
    let mut out = Out {
        shapes: &mut f.top,
        sprays: &mut f.sprays,
        letters: &mut letters,
        shake: &mut f.shake,
    };
    board.draw(&over, &now, &mut out);
    stage.letter_into(&letters, &mut f);
    let looks = drawn.looks();
    for u in effects.lists().into_iter().flatten() {
        let i = u.fx.index();
        f.looks.0[i] = looks.0[i];
    }
    f.music = music(clock);
    f
}
