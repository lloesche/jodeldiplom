//! The stage's scene as the window draws it ([`Stage::scene`]): the result
//! screen while one shows; else the song's lane where the stage draws it;
//! and START's pill over either while Karaoke mode waits for it
//! ([`super::start`]); nothing where none of those shows. What takes
//! clicks in it is kept as drawn ([`Stage::targets`]).

use super::super::stage_gpu::StageFrame;
use super::start::{Line, Look};
use super::{Song, Stage, Tint};
use crate::core::keymap::Op;
use crate::view::karaoke::KaraokePicture;

impl Stage {
    /// The stage for `picture` in the window's `area` (pixels): the result
    /// screen (in `tint`) while one shows; else `song`'s lane where the
    /// stage draws it (`lane`: [`super::wanted`]) at `heard` seconds into
    /// it, `dt` after the frame before; and START's pill over either (on
    /// the singing's clock alone without the lane) while the mode waits
    /// for it. None where there is nothing to draw.
    pub fn scene(
        &mut self,
        (song, lane): (&Song, bool),
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
        (heard, dt): (f64, f32),
        tint: &Tint,
    ) -> Option<StageFrame> {
        self.targets.clear();
        if let Some(frame) = self.result(picture, tint, area) {
            return Some(frame);
        }
        if !lane && picture.start.is_none() {
            return None;
        }
        let mut frame = if lane {
            self.frame(song, picture, area, heard, dt)
        } else {
            StageFrame::bare(crate::core::lag::sung_clock(heard, song.late) as f32)
        };
        if let Some(start) = picture.start {
            let line = Line::of(picture, area);
            self.start_pill(&mut frame.top, &line, None, Look::still(start.ready));
            frame.uploads.append(&mut self.atlas.uploads);
        }
        Some(frame)
    }

    /// What takes clicks on the stage as drawn last, where (left, top,
    /// right, bottom in the window's pixels), and what a click does there:
    /// START's pill while it shows.
    pub fn targets(&self) -> &[([f32; 4], Op)] {
        &self.targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::Singing;
    use crate::view::karaoke::Start;

    fn waiting(rows: u16, ready: bool) -> KaraokePicture {
        KaraokePicture {
            bar: 5.0 / f32::from(rows),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: Some(Start { ready }),
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        }
    }

    /// Waiting with no result and no lane: START alone, in the middle,
    /// above where the controls' bar comes (shown or not), its target where
    /// it is drawn; greyed it is still START. Nothing waits: no frame, no
    /// target.
    #[test]
    fn start_shows_alone_above_the_bar_and_is_its_target() {
        let mut stage = Stage::default();
        let singing = Singing::default();
        let song = Song::new(&singing, Vec::new(), None);
        let area = (0.0, 0.0, 1600.0, 900.0);
        for ready in [true, false] {
            let picture = waiting(50, ready);
            let frame = stage.scene((&song, false), &picture, area, (0.0, 0.0), &Tint::STAGE);
            let frame = frame.expect("START");
            let &[(at, op)] = stage.targets() else {
                panic!("{:?}", stage.targets())
            };
            assert_eq!(op, Op::KaraokeStart);
            let [x0, y0, x1, y1] = at;
            assert!(((x0 + x1) / 2.0 - 800.0).abs() < 1.0, "{x0} {x1}");
            // The bar's 5 rows of 50 begin at 810 px.
            assert!(y1 < 810.0 && y0 > 700.0, "{y0} {y1}");
            let gold = frame
                .top
                .iter()
                .find(|i| i.quad[0] <= x0 && i.quad[2] >= x1);
            assert!(gold.is_some(), "the pill drawn round its target");
        }
        let mut idle = waiting(50, true);
        idle.start = None;
        let frame = stage.scene((&song, false), &idle, area, (0.0, 0.0), &Tint::STAGE);
        assert!(frame.is_none() && stage.targets().is_empty());
    }
}
