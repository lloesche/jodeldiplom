//! The tiers' effects at the stage's points ([`super::super::fx`]): where
//! they draw their light round the lane, as screen passes and over
//! everything ([`Stage::lane_at`], [`Stage::screen_at`],
//! [`Stage::over_at`]) and what they keep clear there ([`Stage::clear`]:
//! from the placements of what they keep clear, so they never drift
//! apart), each slot drawn with the screen's shake they make added
//! up over the frame ([`Stage::effects_at`]), and their big words lettered
//! from the stage's atlas ([`Stage::over`]: before any other text of the
//! frame, and only where the atlas has room, so it never starts again
//! under text already lettered), with their light and their bursts.

use super::super::fx::{self, At, Clear, Letters};
use super::super::particles::Spray;
use super::super::stage_gpu::{Instance, StageFrame};
use super::{lines, Lane, Stage};

impl Stage {
    /// What the effects keep clear on a stage in the window's `area` round
    /// `lane` ([`Clear`]): the score's corner, the multiplier's badge, the
    /// meter and its verdict, each where it is placed.
    pub fn clear(&mut self, area: (f32, f32, f32, f32), lane: &Lane) -> Clear {
        Clear {
            score: super::score::corner(area),
            badge: self.badge_box(lane, area),
            meter: self.meter_spot(area, lane).rect,
        }
    }

    /// Where the effects' screen passes draw over a stage in the window's
    /// `area` round `lane`: the area, the lyric band they keep mostly
    /// clear, what else they keep clear ([`Stage::clear`]), the lane's
    /// glass.
    pub fn screen_at(&mut self, area: (f32, f32, f32, f32), lane: &Lane) -> At {
        At::Screen {
            area,
            band: lines::band(area),
            clear: self.clear(area, lane),
            lane: lane.glass(area),
        }
    }

    /// Where the effects draw their light round the lane on a stage in the
    /// window's `area` round `lane` with `effects`: the lane's glass, a
    /// pill's height, the area, the lyric band, and how strongly the
    /// marquee's bulbs show round it (the flames set them on fire).
    pub fn lane_at(area: (f32, f32, f32, f32), lane: &Lane, effects: &fx::Frame) -> At {
        At::Lane {
            lane: lane.glass(area),
            pill: lane.pill,
            area,
            band: lines::band(area),
            bulbs: effects.live(fx::Fx::Bulbs).strength,
        }
    }

    /// Where the effects draw over everything on a stage in the window's
    /// `area` round `lane`: the area, the lyric band, and the room big
    /// words land in: the lane's glass behind the playhead (what has been
    /// sung), short of the playhead by a pill and of the glass's edges by
    /// a little air; under the HUD's band (the badge, the meter), over
    /// the lyric band, never over what is coming to be sung.
    pub fn over_at(&self, area: (f32, f32, f32, f32), lane: &Lane) -> At {
        let [left, top, _, bottom] = lane.glass(area).rect();
        let air = area.3 * super::AIR;
        At::Over {
            area,
            band: lines::band(area),
            room: [
                left + air,
                top + air,
                (lane.head - lane.pill).max(left + air),
                bottom - air,
            ],
        }
    }

    /// The effects at `at` in `effects`: their shapes into `shapes`, their
    /// bursts into `sprays`, the shake they make added to the frame's.
    /// Only over everything do they letter words ([`Stage::over`]).
    pub(super) fn effects_at(
        &mut self,
        at: &At,
        effects: &fx::Frame,
        (shapes, sprays): (&mut Vec<Instance>, &mut Vec<Spray>),
    ) {
        let mut letters = Vec::new();
        let mut out = fx::Out {
            shapes,
            sprays,
            letters: &mut letters,
            shake: &mut self.shake,
        };
        self.board.draw(at, effects, &mut out);
        debug_assert!(letters.is_empty(), "{:?} letters nothing", at.layer());
    }

    /// What the effects draw over everything at `at` in `effects`: their
    /// shapes and their words lettered, and the words' light; their bursts
    /// go with the stage's, the shake they make onto the frame's.
    pub(super) fn over(&mut self, at: &At, effects: &fx::Frame) -> (Vec<Instance>, Vec<Instance>) {
        let (mut shapes, mut sprays, mut letters) = (Vec::new(), Vec::new(), Vec::new());
        let mut out = fx::Out {
            shapes: &mut shapes,
            sprays: &mut sprays,
            letters: &mut letters,
            shake: &mut self.shake,
        };
        self.board.draw(at, effects, &mut out);
        let mut light = Vec::new();
        sprays.extend(self.letter(&letters, (&mut shapes, &mut light)));
        self.sprays.extend(sprays);
        (shapes, light)
    }

    /// `letters` lettered into `frame` as the stage letters them (over
    /// everything, their light, their bursts, the text new to the atlas):
    /// for a frame put together apart from [`Stage::frame`].
    pub fn letter_into(&mut self, letters: &[Letters], frame: &mut StageFrame) {
        let sprays = self.letter(letters, (&mut frame.top, &mut frame.lights));
        frame.sprays.extend(sprays);
        frame.uploads.append(&mut self.atlas.uploads);
    }

    /// `letters` lettered into `out`, their light into `light`
    /// ([`Letters`]): each put into the atlas where it has room (smaller
    /// where it would be too wide), its instance given its box about its
    /// foot, its quad round that and its place in the atlas. Gives back
    /// the bursts thrown from their feet, each as wide as its word.
    fn letter(
        &mut self,
        letters: &[Letters],
        (out, light): (&mut Vec<Instance>, &mut Vec<Instance>),
    ) -> Vec<Spray> {
        let mut bursts = Vec::new();
        for l in letters {
            let wide = self.measure(l.face, &l.text, l.px);
            let px = if wide > l.widest {
                (l.px * l.widest / wide).floor()
            } else {
                l.px
            };
            let big = self.faces.get(l.face);
            let Some(placed) = self.atlas.place_if_room(big, &l.text, px) else {
                continue;
            };
            let (u, v, w, h) = placed.rect;
            let (sx, sy) = l.scale;
            let (bw, bh) = (w as f32 / placed.made * sx, h as f32 / placed.made * sy);
            let (fx, fy) = l.foot;
            let shape = [fx - bw / 2.0, fy - bh, fx + bw / 2.0, fy];
            let m = l.margin * sx.max(sy) * px / l.px;
            let word = Instance {
                quad: [shape[0] - m, shape[1] - m, shape[2] + m, shape[3] + m],
                shape,
                uv: [u as f32, v as f32, (u + w) as f32, (v + h) as f32],
                ..l.shape
            };
            out.push(word);
            if l.glow > 0.0 {
                let mut lit = word;
                lit.params[3] = l.glow;
                light.push(lit);
            }
            bursts.extend(l.burst.map(|b| Spray {
                half: (bw / 2.0, b.half.1),
                ..b
            }));
        }
        bursts
    }
}

#[cfg(test)]
mod tests {
    use super::super::Lane;
    use super::*;
    use crate::gui::big_text::Face;
    use crate::gui::fx::{Effects, Fx, Music, Use};
    use crate::gui::particles::Kind;
    use crate::gui::stage_gpu::ATLAS;

    /// A word to letter, `px` large, scaled by `scale`.
    fn word(text: &str, px: f32, scale: (f32, f32)) -> Letters {
        Letters {
            text: text.to_owned(),
            face: Face::SHOUT,
            px,
            widest: 10_000.0,
            foot: (500.0, 300.0),
            scale,
            margin: 10.0,
            shape: Instance {
                params: [Fx::Slam.kind(0), 1.0, 0.0, 0.0],
                ..Instance::default()
            },
            glow: 0.0,
            burst: None,
        }
    }

    /// Lettered: its box standing on its foot, scaled about it; its quad
    /// its box and the margin (scaled as the box is) round it; its place in
    /// the atlas; too wide, made smaller; glowing, a copy in the light; its
    /// burst as wide as the word.
    #[test]
    fn a_word_is_lettered_on_its_foot_with_its_margin() {
        let mut stage = Stage::default();
        let (mut out, mut light) = (Vec::new(), Vec::new());
        let bursts = stage.letter(&[word("DIVA!", 80.0, (1.0, 1.0))], (&mut out, &mut light));
        assert!(bursts.is_empty() && light.is_empty());
        let a = out[0];
        assert_eq!(
            (a.shape[3], (a.shape[0] + a.shape[2]) / 2.0),
            (300.0, 500.0)
        );
        assert_eq!(
            a.quad,
            [
                a.shape[0] - 10.0,
                a.shape[1] - 10.0,
                a.shape[2] + 10.0,
                310.0
            ]
        );
        let (uw, uh) = (a.uv[2] - a.uv[0], a.uv[3] - a.uv[1]);
        assert_eq!((uw, uh), (a.shape[2] - a.shape[0], a.shape[3] - a.shape[1]));
        assert_eq!(a.params[0], Fx::Slam.kind(0));
        // Squashed: wider, lower, its margin as the larger way.
        let mut squashed = word("DIVA!", 80.0, (1.3, 0.7));
        squashed.glow = 0.5;
        squashed.burst = Some(crate::gui::fx::slam::slam_dust(
            (500.0, 300.0),
            0.0,
            9.0,
            1.0,
            3,
            [1.0; 3],
        ));
        let bursts = stage.letter(&[squashed], (&mut out, &mut light));
        let b = out[1];
        assert!((b.shape[2] - b.shape[0] - 1.3 * uw).abs() < 1e-3);
        assert!((b.shape[3] - b.shape[1] - 0.7 * uh).abs() < 1e-3);
        assert!((b.quad[2] - b.shape[2] - 13.0).abs() < 1e-3);
        assert_eq!(light.len(), 1);
        assert_eq!((light[0].params[3], light[0].quad), (0.5, b.quad));
        assert_eq!(bursts.len(), 1);
        assert_eq!(bursts[0].kind, Kind::SlamDust);
        assert!((bursts[0].half.0 * 2.0 - (b.shape[2] - b.shape[0])).abs() < 1e-3);
        // Too wide: smaller, no wider than it may be.
        let mut wide = word("AMAZING!", 80.0, (1.0, 1.0));
        wide.widest = 150.0;
        stage.letter(&[wide], (&mut out, &mut light));
        let c = out[2];
        assert!(c.shape[2] - c.shape[0] <= 150.0 + 8.0, "{c:?}");
    }

    /// The atlas too full for a word: not lettered, and the atlas not
    /// started again (the text already in it keeps its place).
    #[test]
    fn a_word_with_no_room_in_the_atlas_waits() {
        let mut stage = Stage::default();
        let (mut out, mut light) = (Vec::new(), Vec::new());
        stage.letter(&[word("GOOD!", 60.0, (1.0, 1.0))], (&mut out, &mut light));
        let kept = out[0].uv;
        stage.atlas.pen = (0, ATLAS - 20, 10);
        stage.letter(&[word("DIVA!", 120.0, (1.0, 1.0))], (&mut out, &mut light));
        assert_eq!(out.len(), 1, "no room: not lettered");
        stage.letter(&[word("GOOD!", 60.0, (1.0, 1.0))], (&mut out, &mut light));
        assert_eq!(out[1].uv, kept, "the text before keeps its place");
    }

    /// The screen passes get the lyric band (from the line sung's top to
    /// the next's bottom, under the lane), what they keep clear (the
    /// score's corner top right, the badge, the meter) and the lane's
    /// glass, and an effect drawing there carries them.
    #[test]
    fn the_screen_passes_get_the_lyric_band() {
        let area = (0.0, 0.0, 1600.0, 900.0);
        let mut stage = Stage::default();
        let at = stage.screen_at(area, &Lane::new(area, 0.0));
        let At::Screen {
            band, lane, clear, ..
        } = at
        else {
            panic!("{at:?}");
        };
        let score = clear.score;
        assert_eq!(band.0, 900.0 * 0.68);
        assert!(band.1 > 900.0 * 0.78 && band.1 < 900.0, "{band:?}");
        assert!(lane.rect()[3] < band.0, "the lane over the band: {lane:?}");
        assert_eq!((score[1], score[2]), (0.0, 1600.0), "top right: {score:?}");
        assert!(score[0] > 1200.0 && score[3] > 0.0, "{score:?}");
        static FIRE: [Use; 1] = [Use::of(Fx::EdgeFire, 0.8)];
        let row = Effects {
            held: &FIRE,
            ..Effects::NONE
        };
        let step = crate::gui::streak::Step {
            tier: Some(crate::tiers::Tier::ALL[0]),
            glow: crate::gui::rainbow::Glow {
                level: 1.0,
                from: 1.0,
                since: 0.0,
            },
            ..Default::default()
        };
        let frame = (stage.board).frame_of([(1.0, &row)].into_iter(), &step, &Music::NEUTRAL, 3.0);
        let mut screen = Vec::new();
        stage.effects_at(&at, &frame, (&mut screen, &mut Vec::new()));
        assert!(!screen.is_empty());
        for i in &screen {
            assert_eq!((i.shape[0], i.shape[1]), band);
            assert_eq!((i.color2, i.color, i.uv), (score, clear.badge, clear.meter));
            assert_eq!(i.params[1], 0.8);
        }
    }

    /// The window's sizes the HUD is held clear at: the Mac's screen (in
    /// pixels and in points), 4K, wide, square, upright and narrow
    /// windows (the narrowest the window may be).
    const SIZES: [(f32, f32); 9] = [
        (3024.0, 1964.0),
        (1512.0, 982.0),
        (3840.0, 2160.0),
        (1600.0, 1000.0),
        (2560.0, 1080.0),
        (1200.0, 1200.0),
        (900.0, 1400.0),
        (500.0, 900.0),
        (480.0, 1000.0),
    ];

    /// Whether boxes `a` and `b` (left, top, right, bottom) are apart.
    fn apart(a: [f32; 4], b: [f32; 4]) -> bool {
        a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1]
    }

    /// The HUD over the lane at every size: the badge round the playhead,
    /// the meter right of it, the score's corner right of that, none over
    /// another, all on the stage in the HUD's band, over the highest the
    /// bulbs light (lit or not); the badge and the meter as large as
    /// before where the band holds them (a wide window), smaller where
    /// not, never under a readable size; the room the word slammed down
    /// lands in under it all.
    #[test]
    fn the_hud_sits_over_the_bulbs_and_apart() {
        let mut stage = Stage::default();
        for (w, h) in SIZES {
            let area = (0.0, 0.0, w, h);
            let lane = Lane::new(area, 0.0);
            let (top, bottom) = lane.hud(area);
            let lit = crate::gui::fx::bulbs::lit_top(lane.glass(area).rect(), (w, h));
            assert!(top >= 0.0 && bottom < lit && top < bottom, "{w}x{h}");
            let Clear {
                score,
                badge,
                meter,
            } = stage.clear(area, &lane);
            for (name, b) in [("badge", badge), ("meter", meter)] {
                assert!(b[0] >= 0.0 && b[2] <= w, "{w}x{h}: the {name} {b:?}");
                assert!(
                    b[1] >= top - 0.5 && b[3] <= bottom + 0.5,
                    "{w}x{h}: the {name} {b:?}"
                );
                assert!(b[3] - b[1] >= 14.0, "{w}x{h}: the {name} readable {b:?}");
            }
            assert!(
                badge[0] < lane.head && badge[2] > lane.head,
                "{w}x{h}: {badge:?}"
            );
            let all = (badge, meter, score);
            assert!(
                apart(badge, meter) && apart(meter, score) && apart(badge, score),
                "{w}x{h}: {all:?}"
            );
            // The word slammed down lands under it all.
            let At::Over { room, .. } = stage.over_at(area, &lane) else {
                unreachable!()
            };
            assert!(
                [badge, meter, score].iter().all(|b| b[3] <= room[1]),
                "{w}x{h}: the slam's room {room:?}, {all:?}"
            );
        }
    }

    /// Every screen pass at full strength, flaring on a bar's first beat
    /// with the bass loud, lights nothing at all in what the singer reads
    /// (the badge, the meter and its verdict, the score's corner) at every
    /// size: the bulbs by where they are, the fire by keeping them clear
    /// (each worked out as its shader works it out: a screen pass this
    /// does not know fails here until it is added).
    #[test]
    fn the_screen_passes_keep_the_hud_clear() {
        use crate::gui::fx::{bulbs, edge_fire, fog};
        let music = Music {
            stems: [1.0; 4],
            ..Music::steady(8.0, 120.0, 4, 1.0, 0.0)
        };
        assert!(music.downbeat > 0.9 && music.sure > 0.9, "{music:?}");
        let mut stage = Stage::default();
        for (w, h) in SIZES {
            let area = (0.0, 0.0, w, h);
            let lane = Lane::new(area, 0.0);
            let at = stage.screen_at(area, &lane);
            let At::Screen { clear, .. } = at else {
                panic!("{at:?}");
            };
            for fx in Fx::ALL.into_iter().filter(|f| f.spec().is_screen()) {
                let uses: &'static [Use] = Box::leak(Box::new([Use::of(fx, 1.0)]));
                let row = Effects {
                    held: uses,
                    ..Effects::NONE
                };
                let step = crate::gui::streak::Step {
                    tier: Some(crate::tiers::Tier::TOP),
                    glow: crate::gui::rainbow::Glow {
                        level: 1.0,
                        from: 1.0,
                        since: -10.0,
                    },
                    ..Default::default()
                };
                stage.board = fx::Board::default();
                let frame = (stage.board).frame_of([(1.0, &row)].into_iter(), &step, &music, 10.0);
                let mut shapes = Vec::new();
                stage.effects_at(&at, &frame, (&mut shapes, &mut Vec::new()));
                assert!(!shapes.is_empty(), "{fx:?} draws");
                let look = fx.spec().params;
                for i in &shapes {
                    let light = |px: (f32, f32), time: f32| -> [f32; 4] {
                        let s = i.params[1];
                        match i.params[0] {
                            k if k == bulbs::BULBS => {
                                bulbs::bulbs(px, (w, h), time, s, i.shape, look, &music)
                            }
                            k if k == edge_fire::EDGE_FIRE => edge_fire::edge_fire(
                                px,
                                (w, h),
                                time,
                                (s, (i.shape[0], i.shape[1])),
                                &clear,
                                look,
                                &music,
                            ),
                            k if k == fog::FOG => fog::fog(
                                px,
                                (w, h),
                                time,
                                (s, (i.shape[0], i.shape[1])),
                                &clear,
                                look,
                                &music,
                            ),
                            k => panic!("{fx:?}: a screen pass ({k}) this does not know"),
                        }
                    };
                    for [l, t, r, b] in clear.boxes() {
                        for y in 0..=16 {
                            let y = t + (b - t) * y as f32 / 16.0;
                            for x in 0..=48 {
                                let x = l + (r - l) * x as f32 / 48.0;
                                for time in [0.4, 2.9, 7.7] {
                                    let c = light((x, y), time);
                                    assert_eq!(c, [0.0; 4], "{fx:?} {w}x{h} at {x} {y}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The light round the lane at full strength (its glow, its flames
    /// setting the marquee's bulbs on fire, the bulbs `bulbs` strong), to
    /// `music`, on a stage the window's `size`: its shapes, the HUD's
    /// band's bottom and the lyric band.
    fn lane_light(bulbs: f32, music: &Music, size: (f32, f32)) -> (Vec<Instance>, f32, (f32, f32)) {
        let uses: &'static [Use] = Box::leak(Box::new([
            Use::of(Fx::Glow, 1.0),
            Use::of(Fx::Flames, 1.0),
            Use::of(Fx::Bulbs, bulbs),
        ]));
        let row = Effects {
            held: uses,
            ..Effects::NONE
        };
        let step = crate::gui::streak::Step {
            tier: Some(crate::tiers::Tier::TOP),
            glow: crate::gui::rainbow::Glow {
                level: 1.0,
                from: 1.0,
                since: -10.0,
            },
            ..Default::default()
        };
        let mut stage = Stage::default();
        let area = (0.0, 0.0, size.0, size.1);
        let lane = Lane::new(area, 0.0);
        let frame = (stage.board).frame_of([(1.0, &row)].into_iter(), &step, music, 10.0);
        let at = Stage::lane_at(area, &lane, &frame);
        let mut shapes = Vec::new();
        stage.effects_at(&at, &frame, (&mut shapes, &mut Vec::new()));
        (shapes, lane.hud(area).1, lines::band(area))
    }

    /// The music surging: a bar's first beat, the grid sure, the bass loud.
    fn surging() -> Music {
        Music {
            stems: [1.0; 4],
            ..Music::steady(8.0, 120.0, 4, 1.0, 0.0)
        }
    }

    /// The light of a shape round the lane (its glow or its flames) at
    /// `p`, `time` seconds on, as its shader works it out.
    fn lit(i: &Instance, p: (f32, f32), time: f32) -> f32 {
        use crate::gui::fx::{flames, glow};
        match i.params[0] {
            k if k == flames::BLAZE => {
                let c = flames::flames_at(p, i.shape, i.params, i.color, i.color2, (time, 0.0));
                c[0] + c[1] + c[2]
            }
            k if k == glow::GLOW => {
                let d = crate::gui::rainbow::round_box(
                    (p.0 - i.shape[0], p.1 - i.shape[1]),
                    (i.shape[2], i.shape[3]),
                    i.shape[3],
                );
                let [sure, kick, _, _] = i.color2;
                glow::glow_at(d, i.params[1], time, i.params[2], (sure, kick))
            }
            k => panic!("a shape round the lane ({k}) this does not know"),
        }
    }

    /// The light round the lane at full strength, surging on a bar's
    /// first beat with the bass loud, lights nothing over the HUD's band's
    /// bottom nor in the lyric band, at every size: every shape it draws is
    /// within them.
    #[test]
    fn the_lane_s_light_keeps_off_the_hud_and_the_lyrics() {
        for size in SIZES {
            let (shapes, hud, band) = lane_light(1.0, &surging(), size);
            assert_eq!(shapes.len(), 3, "{size:?}: the glow and two edges' flames");
            for i in &shapes {
                assert!(
                    i.quad[1] >= hud && i.quad[3] <= band.0,
                    "{size:?}: {:?} between {hud} and {}",
                    i.quad,
                    band.0
                );
            }
        }
    }

    /// Nothing of the light round the lane is cut off: at every size,
    /// with the bulbs off, coming in and full, still and surging, every
    /// shape's light is gone (none at all) at each edge of its quad, so no
    /// straight edge of it ever shows.
    #[test]
    fn the_lane_s_light_is_gone_at_its_quads_edges() {
        let musics = [
            Music::NEUTRAL,
            surging(),
            Music::steady(8.5, 120.0, 4, 1.0, 0.0),
        ];
        for size in SIZES {
            for bulbs in [0.0, 0.2, 0.6, 1.0] {
                for music in &musics {
                    let (shapes, _, _) = lane_light(bulbs, music, size);
                    for i in &shapes {
                        let [l, t, r, b] = i.quad;
                        for n in 0..=96 {
                            let (x, y) =
                                (l + (r - l) * n as f32 / 96.0, t + (b - t) * n as f32 / 96.0);
                            for p in [(x, t), (x, b), (l, y), (r, y)] {
                                for time in [0.3, 1.7, 4.1] {
                                    let v = lit(i, p, time);
                                    assert_eq!(v, 0.0, "{size:?} {bulbs}: {:?} at {p:?}", i.quad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
