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
use jodeldiplom::gui::fx::{self, At, Board, Effects, Music, Text};
use jodeldiplom::gui::rainbow::Glow;
use jodeldiplom::gui::stage::{screen_at, Lane};
use jodeldiplom::gui::stage_gpu::{kind, StageFrame};
use jodeldiplom::gui::streak::Step;
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
            tier: singing.last_phrase().and_then(|p| p.tier),
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

/// `base` (a frame of the stage over `area`, at its clock) with `effects`
/// in it, as the board draws them from a row of them at full glow, its
/// bursts fired at `fired` (the singing's clock: a phrase and a word hit
/// then): their screen passes, the line sung in the strongest of their
/// treatments, their bursts from the playhead, and their looks.
pub fn with_effects(
    base: &StageFrame,
    area: (f32, f32, f32, f32),
    effects: &Effects,
    fired: f64,
) -> StageFrame {
    let clock = f64::from(base.clock);
    let shone = Shone {
        line: 0,
        end: fired,
        value: 1.0,
        top: 0.9,
        tier: Some(Tier::ALL[0]),
    };
    let step = Step {
        tier: Some(Tier::ALL[0]),
        glow: Glow {
            level: 1.0,
            from: 1.0,
            since: -10.0,
        },
        phrase: Some(shone),
        word: Some(fired),
        ..Step::default()
    };
    let mut board = Board::default();
    let mut f = base.clone();
    let drawn = board.frame_of([(1.0, effects)].into_iter(), &step, &Music::NEUTRAL, fired);
    let lane = Lane::new(area, clock);
    let glass = lane.glass(area);
    let screen = screen_at(area, glass);
    board.shapes(&screen, &drawn, &mut f.screen);
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
    board.sprays(&playhead, &drawn, &mut f.sprays);
    let looks = drawn.looks();
    for u in effects.lists().into_iter().flatten() {
        let i = u.fx.index();
        f.looks.0[i] = looks.0[i];
    }
    f
}
