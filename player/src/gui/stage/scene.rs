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
        // What lies under the lane (as the pixmap did): the lines set in
        // the middle, the count-in, the score; over it, the bubbles of
        // the lines set in the middle.
        let (mut under, mut over, mut sprays) = (Vec::new(), Vec::new(), Vec::new());
        let which = if lane { Which::Info } else { Which::All };
        let into = (&mut under, &mut over, &mut sprays);
        let rim = self.lyrics(into, picture, (area, which));
        self.count(&mut under, picture, area, song.theme);
        self.score(&mut under, picture, area, dt);
        if picture.controls.is_some() {
            let (x, y, w, h) = area;
            clip(&mut under, 0, [x, y, x + w, y + h * (1.0 - picture.bar)]);
        }
        let place = Place::of(self, area, picture.reach);
        let dress = (tint, song.theme);
        let (mut frame, button, always) = match self.result(picture, dress, area, &place) {
            Some((frame, button)) => (frame, button, true),
            None => {
                let mut frame = if lane {
                    self.frame(song, picture, area, heard, dt)
                } else {
                    self.rim_shine((&mut over, &mut sprays), (song, heard), rim);
                    let clock = crate::core::lag::sung_clock(heard, song.late) as f32;
                    StageFrame::bare(clock, song.theme)
                };
                under.append(&mut frame.instances);
                under.append(&mut over);
                frame.instances = under;
                frame.sprays.append(&mut sprays);
                let button = picture.next.map(|n| Look::still(n.ready));
                (frame, button, lane)
            }
        };
        if let Some(look) = button {
            self.next_button(&mut frame.top, &place, look);
        }
        frame.uploads.append(&mut self.atlas.uploads);
        let empty = frame.instances.is_empty() && frame.top.is_empty() && frame.sprays.is_empty();
        (always || !empty).then_some(frame)
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
    use crate::view::karaoke::Next;

    /// The mode waiting for Next in a window `height` points high, its
    /// controls' bar and their reach as the window lays them out.
    fn waiting(height: f32, ready: bool) -> KaraokePicture {
        let mut state = State::new("/m".into(), Config::default());
        state.fit.controls_reach = mode::reach(height);
        KaraokePicture {
            bar: mode::bar_share(height),
            reach: controls::reach(&state),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            next: Some(Next { ready }),
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        }
    }

    /// Waiting with no result and no lane: Next alone at the right, its
    /// bottom above the controls' reach (where the pointer brings them
    /// up: points from the window's bottom), in windows wide, narrow,
    /// upright and short; a little above the middle where the window is
    /// tall enough; its target where it is drawn, its word and arrow on
    /// it; greyed it is still Next. Nothing waits: no frame, no target.
    #[test]
    fn next_shows_alone_at_the_right_out_of_the_controls_reach() {
        let mut stage = Stage::default();
        let singing = Singing::default();
        let song = Song::new(&singing, Vec::new(), None);
        let sizes = [
            (1600.0, 900.0),
            (1280.0, 800.0),
            (1024.0, 768.0),
            (800.0, 480.0),
            (640.0, 800.0),
            (500.0, 900.0),
            (600.0, 1000.0),
            (1280.0, 200.0),
        ];
        for ((w, h), ready) in sizes.into_iter().flat_map(|s| [(s, true), (s, false)]) {
            let area = (0.0, 0.0, w, h);
            let picture = waiting(h, ready);
            let frame = stage.scene((&song, false), &picture, area, (0.0, 0.0), &Tint::STAGE);
            let frame = frame.expect("Next");
            let &[(at, op)] = stage.targets() else {
                panic!("{:?}", stage.targets())
            };
            assert_eq!(op, Op::KaraokeNext);
            let [x0, y0, x1, y1] = at;
            let size = format!("{w} x {h}: {at:?}");
            assert!(x1 < w && x1 > w * 0.95 && x0 > w * 0.6, "{size}");
            assert!(y1 <= h * (1.0 - picture.reach), "{size}: {}", picture.reach);
            if h >= 480.0 {
                assert!(y0 > h * 0.3 && y1 < h * 0.5, "{size}");
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
        idle.next = None;
        let area = (0.0, 0.0, 1600.0, 900.0);
        let frame = stage.scene((&song, false), &idle, area, (0.0, 0.0), &Tint::STAGE);
        assert!(frame.is_none() && stage.targets().is_empty());
    }
}
