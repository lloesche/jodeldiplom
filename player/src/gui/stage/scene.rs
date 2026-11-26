//! The stage's scene as the window draws it ([`Stage::scene`]): the result
//! screen while one shows; else the song's lane where the stage draws it
//! or the lyrics set in the middle where it does not, the count-in and the
//! score under either (where the window's pixmap had them); and START's
//! pill over it all while Karaoke mode waits for it ([`super::start`]);
//! nothing where none of those shows. What takes clicks in it is kept as
//! drawn ([`Stage::targets`]).

use super::super::stage_gpu::StageFrame;
use super::lyrics::Which;
use super::start::{Line, Look};
use super::{clip, Song, Stage, Tint};
use crate::core::keymap::Op;
use crate::view::karaoke::KaraokePicture;

impl Stage {
    /// The stage for `picture` in the window's `area` (pixels): the result
    /// screen (in `tint`) while one shows; else `song`'s lane where the
    /// stage draws it (`lane`: [`super::wanted`]) at `heard` seconds into
    /// it, `dt` after the frame before, or the lyrics set in the middle
    /// where it does not; the count-in and the score under it, kept above
    /// the controls' bar while it shows; and START's pill over it all while
    /// the mode waits for it. None where there is nothing to draw.
    pub fn scene(
        &mut self,
        (song, lane): (&Song, bool),
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
        (heard, dt): (f64, f32),
        tint: &Tint,
    ) -> Option<StageFrame> {
        self.targets.clear();
        // What lies under the lane (as the pixmap did): the lines set in
        // the middle, the count-in, the score; over it, the bubbles of
        // the lines set in the middle.
        let (mut under, mut over, mut sprays) = (Vec::new(), Vec::new(), Vec::new());
        let which = if lane { Which::Info } else { Which::All };
        let into = (&mut under, &mut over, &mut sprays);
        let rim = self.lyrics(into, picture, (area, which));
        self.count(&mut under, picture, area);
        self.score(&mut under, picture, area, dt);
        if picture.controls.is_some() {
            let (x, y, w, h) = area;
            clip(&mut under, 0, [x, y, x + w, y + h * (1.0 - picture.bar)]);
        }
        if let Some(frame) = self.result(picture, tint, area) {
            return Some(frame);
        }
        let mut frame = if lane {
            self.frame(song, picture, area, heard, dt)
        } else {
            self.rim_shine((&mut over, &mut sprays), (song, heard), rim);
            StageFrame::bare(crate::core::lag::sung_clock(heard, song.late) as f32)
        };
        under.append(&mut frame.instances);
        under.append(&mut over);
        frame.instances = under;
        frame.sprays.append(&mut sprays);
        if let Some(start) = picture.start {
            let line = Line::of(picture, area);
            self.start_pill(&mut frame.top, &line, None, Look::still(start.ready));
        }
        frame.uploads.append(&mut self.atlas.uploads);
        let empty = frame.instances.is_empty() && frame.top.is_empty() && frame.sprays.is_empty();
        (lane || !empty).then_some(frame)
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
