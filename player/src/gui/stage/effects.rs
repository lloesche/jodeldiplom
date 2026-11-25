//! The tiers' effects at the stage's points ([`super::super::fx`]): where
//! they draw as screen passes and over everything ([`screen_at`],
//! [`over_at`]), each slot drawn with the screen's shake they make added
//! up over the frame ([`Stage::effects_at`]), and their big words lettered
//! from the stage's atlas ([`Stage::over`]: before any other text of the
//! frame, and only where the atlas has room, so it never starts again
//! under text already lettered), with their light and their bursts.

use super::super::fx::{self, At, Letters};
use super::super::particles::Spray;
use super::super::stage_gpu::{Instance, StageFrame};
use super::{lines, LaneBox, Stage};

/// Where the effects' screen passes draw over a stage in the window's
/// `area` round its `lane`: the area, the lyric band and the score's
/// corner they keep mostly clear, the lane.
pub fn screen_at(area: (f32, f32, f32, f32), lane: LaneBox) -> At {
    At::Screen {
        area,
        band: lines::band(area),
        score: super::super::karaoke::score_corner(area),
        lane,
    }
}

/// Where the effects draw over everything on a stage in the window's
/// `area` round its `lane`: the area, the lane, the lyric band they keep
/// clear.
pub fn over_at(area: (f32, f32, f32, f32), lane: LaneBox) -> At {
    At::Over {
        area,
        lane,
        band: lines::band(area),
    }
}

impl Stage {
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
    /// the next's bottom, under the lane), the score's corner (top right)
    /// and the lane's glass, and an effect drawing there carries them.
    #[test]
    fn the_screen_passes_get_the_lyric_band() {
        let area = (0.0, 0.0, 1600.0, 900.0);
        let glass = Lane::new(area, 0.0).glass(area);
        let at = screen_at(area, glass);
        let At::Screen {
            band, lane, score, ..
        } = at
        else {
            panic!("{at:?}");
        };
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
        let mut stage = Stage::default();
        let frame = (stage.board).frame_of([(1.0, &row)].into_iter(), &step, &Music::NEUTRAL, 3.0);
        let mut screen = Vec::new();
        stage.effects_at(&at, &frame, (&mut screen, &mut Vec::new()));
        assert!(!screen.is_empty());
        for i in &screen {
            assert_eq!((i.shape[0], i.shape[1]), band);
            assert_eq!(i.color2, score);
            assert_eq!(i.params[1], 0.8);
        }
    }
}
