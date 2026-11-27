//! The stage's scene as the window draws it ([`Stage::scene`]): the result
//! screen while one shows; else the song's lane where the stage draws it
//! or the lyrics set in the middle where it does not, the count-in and the
//! score under either (where the window's pixmap had them); and Next's
//! button over it all while Karaoke mode waits for it ([`super::next`]);
//! nothing where none of those shows. What takes clicks in it is kept as
//! drawn ([`Stage::targets`]).

use super::gpu::StageFrame;
use super::lyrics::Which;
use super::next::{Look, Place};
use super::{clip, Song, Stage, Tint};
use crate::core::keymap::Op;
use crate::view::karaoke::KaraokePicture;

impl Stage {
    /// The stage for `picture` in the window's `area` (pixels): the result
    /// screen (in `tint`) while one shows; else `song`'s lane where the
    /// stage draws it (`lane`: [`super::wanted`]) at `heard` seconds into
    /// it, `dt` after the frame before, or the lyrics set in the middle
    /// where it does not; the count-in and the score under it, kept above
    /// the controls' bar while it shows; and Next's button over it all
    /// while the mode waits for it. None where there is nothing to draw.
    pub fn scene(
        &mut self,
        (song, lane): (&Song, bool),
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
        (heard, dt): (f64, f32),
        tint: &Tint,
    ) -> Option<StageFrame> {
        self.targets.clear();
        self.theme = song.theme;
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
            let place = Place::of(area);
            self.next_button(&mut frame.top, &place, Look::still(start.ready));
        }
        frame.uploads.append(&mut self.atlas.uploads);
        let empty = frame.instances.is_empty() && frame.top.is_empty() && frame.sprays.is_empty();
        (lane || !empty).then_some(frame)
    }

    /// What takes clicks on the stage as drawn last, where (left, top,
    /// right, bottom in the window's pixels), and what a click does there:
    /// Next's button while it shows.
    pub fn targets(&self) -> &[([f32; 4], Op)] {
        &self.targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::karaoke::controls;
    use crate::core::singing::Singing;
    use crate::core::State;
    use crate::gui::screens::karaoke::mode;
    use crate::gui::stage::gpu::Instance;
    use crate::view::karaoke::Start;

    /// The mode waiting for Next in a window `height` pixels high, its
    /// controls' bar as the window lays it out.
    fn waiting(height: f32, ready: bool) -> KaraokePicture {
        KaraokePicture {
            bar: mode::bar_share(height),
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

    /// Waiting with no result and no lane: Next alone, at the right a
    /// little above the middle, the pointer over it never bringing the
    /// controls up (in windows wide, narrow and upright), its target where
    /// it is drawn, its word and arrow on it; greyed it is still Next.
    /// Nothing waits: no frame, no target.
    #[test]
    fn next_shows_alone_at_the_right_out_of_the_controls_reach() {
        let mut stage = Stage::default();
        let singing = Singing::default();
        let song = Song::new(&singing, Vec::new(), None);
        let mut state = State::new("/m".into(), Config::default());
        let sizes = [
            (1600.0, 900.0),
            (1280.0, 800.0),
            (1024.0, 768.0),
            (800.0, 480.0),
            (640.0, 800.0),
            (500.0, 900.0),
            (600.0, 1000.0),
        ];
        for ((w, h), ready) in sizes.into_iter().flat_map(|s| [(s, true), (s, false)]) {
            let area = (0.0, 0.0, w, h);
            let picture = waiting(h, ready);
            let frame = stage.scene((&song, false), &picture, area, (0.0, 0.0), &Tint::STAGE);
            let frame = frame.expect("Next");
            let &[(at, op)] = stage.targets() else {
                panic!("{:?}", stage.targets())
            };
            assert_eq!(op, Op::KaraokeStart);
            let [x0, y0, x1, y1] = at;
            let size = format!("{w} x {h}: {at:?}");
            assert!(x1 < w && x1 > w * 0.95 && x0 > w * 0.6, "{size}");
            assert!(y0 > h * 0.3 && y1 < h * 0.5, "{size}");
            state.fit.controls_reach = mode::reach(h);
            for y in [y0, (y0 + y1) / 2.0, y1] {
                assert!(!controls::in_reach(&state, (h - y) / h), "{size}");
            }
            let inside = |i: &Instance| {
                let [a, b, c, d] = i.quad;
                let (cx, cy) = ((a + c) / 2.0, (b + d) / 2.0);
                (x0..=x1).contains(&cx) && (y0..=y1).contains(&cy)
            };
            assert!(frame.top.len() > 3, "the pill, its word, its arrow");
            assert!(frame.top.iter().all(inside), "{size}");
            let gold = frame
                .top
                .iter()
                .find(|i| i.quad[0] <= x0 && i.quad[2] >= x1);
            assert!(gold.is_some(), "the pill drawn round its target");
        }
        let mut idle = waiting(900.0, true);
        idle.start = None;
        let area = (0.0, 0.0, 1600.0, 900.0);
        let frame = stage.scene((&song, false), &idle, area, (0.0, 0.0), &Tint::STAGE);
        assert!(frame.is_none() && stage.targets().is_empty());
    }
}
