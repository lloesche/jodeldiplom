//! The stage's frames through a scripted singing, held to a digest (with
//! its lane, and without it: the lyrics set in the middle): every
//! tier in turn, Good to Diva, the streak lost, Diva again, a seek back
//! (the scoring started over) and Diva once more, at 60 frames a second
//! for half a minute, the streak climbing through every rung of the
//! multiplier and starting over, again and again. Every
//! frame's shapes (over the stage, over everything, the light, the screen
//! passes), its tubes'
//! lines and its bursts go into the digest; a shape's kind by its name and
//! text by the rainbow it shows in ([`shine_of`], only while it shows), so
//! moving an effect's code or renumbering its kinds changes nothing here
//! unless what is drawn changes. The song is the one the GPU's timings
//! sing (`common/sung.rs`).
//!
//! On Windows x86_64 only: the digest takes every float drawn to its last
//! bit, and the system's maths library rounds a few of those differently
//! on Linux and macOS.

#![cfg(all(windows, target_arch = "x86_64"))]

use std::path::Path;
use std::sync::Arc;

use jodel_dsp::Fnv1a;
use jodeldiplom::core::shine::{Reference, Shone, Tier};
use jodeldiplom::core::singing::Singing;
use jodeldiplom::gui::fx::rainbow::shine_of;
use jodeldiplom::gui::particles::Spray;
use jodeldiplom::gui::stage::gpu::{Instance, Shape, StageFrame};
use jodeldiplom::gui::stage::{Song, Stage, Tint};

#[path = "common/sung.rs"]
mod sung;
use sung::{picture_at, song, FPS, LENGTH};

/// The frames: how many in all, and the one the song is sought back at
/// (by [`SEEK`] seconds).
const FRAMES: u32 = 1800;
const SOUGHT: u32 = 1320;
const SEEK: f64 = 10.0;

/// The phrases decided, by frame: every tier in turn from frame 120 to
/// 840, none, the top twice, then the second tier.
fn decisions() -> Vec<(u32, Option<Tier>)> {
    let last = (Tier::COUNT - 1).max(1) as u32;
    let mut all: Vec<(u32, Option<Tier>)> = (Tier::ALL.into_iter().zip(0..))
        .map(|(t, i)| (120 + 720 * i / last, Some(t)))
        .collect();
    all.extend([
        (1080, None),
        (1200, Some(Tier::TOP)),
        (1500, Some(Tier::TOP)),
        (1620, Some(Tier::ALL[1])),
    ]);
    all
}

/// A 64-bit FNV-1a digest.
struct Digest(Fnv1a);

impl Digest {
    fn word(&mut self, w: u32) {
        self.0.write(&w.to_le_bytes());
    }

    fn floats(&mut self, v: &[f32]) {
        for x in v {
            self.word(x.to_bits());
        }
    }

    fn text(&mut self, s: &str) {
        for b in s.bytes() {
            self.word(u32::from(b));
        }
        self.word(0xFFFF_FFFF);
    }

    /// A shape: its kind by name, text by the rainbow it shows in.
    fn instance(&mut self, i: &Instance, clock: f32) {
        let mut i = *i;
        self.text(Shape::name_of(i.params[0]));
        if i.is(Shape::Text) {
            let shown = shine_of(&i).filter(|g| g.shows(clock));
            match shown {
                Some(g) => self.floats(&[1.0, g.level, g.from, g.since]),
                None => self.floats(&[0.0]),
            }
            i.params[3] = 0.0;
            i.strain = [0.0; 4];
        }
        i.params[0] = 0.0;
        for v in [i.quad, i.shape, i.color, i.color2, i.params, i.uv, i.strain] {
            self.floats(&v);
        }
    }

    fn spray(&mut self, s: &Spray) {
        self.text(&format!("{:?}", s.kind));
        self.floats(&[s.at.0, s.at.1, s.half.0, s.half.1, s.start, s.drift]);
        self.floats(&[s.size, s.rgb[0], s.rgb[1], s.rgb[2]]);
        self.word(s.seed);
    }

    fn frame(&mut self, f: &StageFrame) {
        self.floats(&[f.clock]);
        let lists = [
            (&f.instances, 1),
            (&f.top, 2),
            (&f.lights, 3),
            (&f.screen, 7),
        ];
        for (list, mark) in lists {
            self.word(mark);
            for i in list {
                self.instance(i, f.clock);
            }
        }
        self.word(4);
        for p in &f.paths {
            self.floats(p);
        }
        for (list, mark) in [(&f.sprays, 5), (&f.light_sprays, 6)] {
            self.word(mark);
            for s in list {
                self.spray(s);
            }
        }
    }
}

/// The digest of the scripted singing's frames as the window's stage draws
/// them (see the module's docs; with its `lane`, else the lyrics set in
/// the middle), with each kind of shape's count over all of them.
fn digest(lane: bool) -> (u64, std::collections::BTreeMap<String, usize>) {
    let (lyrics, melody) = song();
    let sung: Vec<&jodel_lyrics::Line> = lyrics.lines.iter().collect();
    let mut singing = Singing::default();
    singing.prepare(
        Path::new("song"),
        &lyrics,
        &jodeldiplom::core::singing::melodic(&melody),
    );
    let phrases = vec![None; singing.notes.lines.len()];
    singing.set_reference(Some(Arc::new(Reference { phrases })));
    let mut stage = Stage::default();
    let mut d = Digest(Fnv1a::new());
    let mut counts = std::collections::BTreeMap::new();
    let mut heard = 0usize;
    let decided = decisions();
    // The streak climbing past the top's rung, then over again.
    let climb = Tier::TOP.spec().streak + 15;
    for f in 0..FRAMES {
        let mut now = f64::from(f) / FPS;
        if f >= SOUGHT {
            now -= SEEK;
            if f == SOUGHT {
                heard = (now * 100.0) as usize;
            }
        }
        while (heard as f64) / 100.0 <= now {
            let t = heard as f64 / 100.0;
            let c = singing.word_at(t).map(|(_, c)| c as f32);
            singing.hear(t, c);
            heard += 1;
        }
        if let Some(&(_, tier)) = decided.iter().find(|(at, _)| *at == f) {
            let shone = Shone {
                line: (f / 240) as usize,
                end: now,
                value: 1.0,
                top: 0.9,
                tier,
            };
            singing.decided(now, shone);
        }
        singing.tally.streak = (f / 12) % climb;
        let picture = picture_at(&sung, now, &singing);
        let song = Song::new(&singing, sung.clone(), Some(LENGTH));
        let area = (0.0, 0.0, 1600.0, 900.0);
        let step = (now, 1.0 / 60.0);
        let frame = stage.scene((&song, lane), &picture, area, step, &Tint::STAGE);
        let frame = frame.expect("a stage");
        d.frame(&frame);
        for i in frame
            .instances
            .iter()
            .chain(&frame.top)
            .chain(&frame.lights)
            .chain(&frame.screen)
        {
            *counts
                .entry(Shape::name_of(i.params[0]).to_owned())
                .or_insert(0) += 1;
        }
        for s in frame.sprays.iter().chain(&frame.light_sprays) {
            *counts.entry(format!("{:?}", s.kind)).or_insert(0) += 1;
        }
    }
    (d.0.finish(), counts)
}

/// The scripted singing draws the frames it drew when the digest was
/// taken: the tiers' looks, the rainbow, the bursts and the light all as
/// they were, and every tier's look in it (the lane's glow, the flames,
/// the bands, the swirl, the edges on fire, the bulbs, the coins, the
/// word slammed).
#[test]
fn the_stage_draws_the_same_frames() {
    let (digest, counts) = digest(true);
    for name in [
        "bands",
        "swirl",
        "glow",
        "blaze",
        "meter",
        "badge",
        "StepUp",
        "Angels",
        "Jackpot",
        "edge fire",
        "bulbs",
        "CoinFlick",
        "slam",
        "SlamDust",
    ] {
        assert!(
            counts.get(name).is_some_and(|&n| n > 0),
            "{name}: {counts:?}"
        );
    }
    assert_eq!(digest, 1_429_602_034_178_520_279, "{counts:?}");
}

/// The same singing without the lane (the notes off): the lyrics set in
/// the middle and the score draw the frames they drew when the digest was
/// taken.
#[test]
fn the_lyrics_set_in_the_middle_draw_the_same_frames() {
    let (digest, counts) = digest(false);
    assert!(counts.get("text").is_some_and(|&n| n > 0), "{counts:?}");
    assert_eq!(digest, 5_741_503_954_211_909_980, "{counts:?}");
}
