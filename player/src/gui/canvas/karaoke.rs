//! Karaoke mode in the canvas. With a GPU the stage draws all of it but the
//! controls' bar ([`Canvas::stage_frame`]: the lyrics, the count-in, the
//! score, START, the result) over its backdrop, and the pixmap is only the
//! bar's strip ([`Canvas::strip`]): the controls' text over the backdrop,
//! made darker, painted again only where its cells were redrawn or the bar
//! changed. Without one the whole picture is painted into the pixmap (the
//! cover's ground, the lyrics, the count-in and the score, the bar over
//! them) only where it changed, or moves by itself (the score rolling up).

use std::time::Instant;

use ratatui::layout::Rect;

use super::Canvas;
use crate::art::Art;
use crate::gui::karaoke::{self, Backdrop};
use crate::gui::pixels::{dim, PixelRect, Target, SEE_THROUGH};
use crate::gui::{stage, stage_gpu};
use crate::view::karaoke::KaraokePicture;
use crate::view::layout::controls_bar;

impl Canvas {
    /// Karaoke mode's stage for the frame just drawn, at `heard` seconds
    /// into the song (as `song` has it heard, once analysed), `dt` seconds
    /// after the last; the result screen in `tint` while one shows; none
    /// where the stage does not show.
    pub fn stage_frame(
        &mut self,
        (state, song): (
            &crate::core::State,
            Option<&crate::gui::backdrop::music::Song>,
        ),
        heard: f64,
        dt: f32,
        tint: &stage::Tint,
    ) -> Option<stage_gpu::StageFrame> {
        let (picture, rect) = self.staged.as_ref()?;
        let (x, y, w, h) = *rect;
        let area = (x as f32, y as f32, w as f32, h as f32);
        let song = (&stage::Song::of(state, song), stage::wanted(state));
        self.stage.scene(song, picture, area, (heard, dt), tint)
    }

    /// What the button the stage drew in the frame shown at (`x`, `y`)
    /// (window pixels) does, if there is one: START's pill.
    pub fn target_at(&self, x: f32, y: f32) -> Option<crate::core::keymap::Op> {
        self.staged.as_ref()?;
        let inside = |[x0, y0, x1, y1]: [f32; 4]| (x0..=x1).contains(&x) && (y0..=y1).contains(&y);
        let mut targets = self.stage.targets().iter();
        targets.find(|(at, _)| inside(*at)).map(|&(_, op)| op)
    }

    /// The rows of window pixels (from, to) where the pixmap shows over
    /// Karaoke mode's stage's ground: the controls' bar while it is laid
    /// out; none where there is none (the stage draws all the rest).
    pub fn strip(&self) -> Option<(u32, u32)> {
        self.strip
    }

    /// Whether Karaoke mode's picture, shown, moves by itself: its score
    /// rolling up (the stage's, with a GPU).
    pub(super) fn karaoke_moving(&self) -> bool {
        let moving = if self.gpu {
            self.stage.moving()
        } else {
            self.karaoke_layer.odometer.moving()
        };
        self.shown.is_some() && moving
    }

    /// Karaoke mode's picture `full` over the cells of `area` in the frame
    /// at `now` (`last`: the one in the frame before and its cells, if
    /// there was one). With a GPU it is the stage's to draw, and only the
    /// controls' bar is painted. Painted again only where it changed, the
    /// cells under it were redrawn, or it moves by itself; the mode just
    /// come, its score shows at once. Returns the pixels it changed.
    pub(super) fn karaoke_picture(
        &mut self,
        (area, full): (Rect, KaraokePicture),
        last: Option<&(Rect, Option<f64>)>,
        art: Option<&Art>,
        now: Instant,
    ) -> Vec<PixelRect> {
        if last.is_none() {
            self.karaoke_layer.odometer.step(None, 0.0);
            self.stage.entered();
        }
        let damaged = self.terminal.backend().damage.touches(area);
        let controls = full.controls;
        let afresh = damaged || last.is_none_or(|l| *l != (area, controls));
        if self.gpu {
            self.strip = controls.map(|_| {
                let (_, y, _, h) = self.pixel_rect(controls_bar(area));
                (y as u32, (y + h) as u32)
            });
            self.staged = Some((full, self.pixel_rect(area)));
            self.karaoke = None;
            return if afresh {
                self.paint_bar(area, controls)
            } else {
                Vec::new()
            };
        }
        // No stage, no pill: START said in words.
        let picture = crate::view::karaoke::with_start_cue(&full);
        let kept = self.karaoke.as_ref().is_some_and(|k| k == &picture)
            && !afresh
            && !self.karaoke_layer.odometer.moving();
        let changed = if kept {
            Vec::new()
        } else {
            self.karaoke_layer.at(now);
            self.paint_karaoke((area, &picture), art, afresh)
        };
        self.karaoke = Some(picture);
        changed
    }

    /// The ground the GPU draws Karaoke mode on (all see-through) over the
    /// cells of `area`, the controls' bar darker over it where it is laid
    /// out (`controls`: how far it shows) with its cells' text. Returns the
    /// pixels it changed.
    fn paint_bar(&mut self, cells: Rect, controls: Option<f64>) -> Vec<PixelRect> {
        let area = self.pixel_rect(cells);
        let (ax, ay, aw, ah) = area;
        if self
            .backdrop
            .as_ref()
            .is_none_or(|b| (b.art, b.size) != (None, (aw, ah)))
        {
            self.backdrop = Some(Backdrop::through(aw, ah));
        }
        let bar = controls.map(|_| self.pixel_rect(controls_bar(cells)));
        let in_bar = |y: usize| bar.is_some_and(|(_, by, _, bh)| (by..by + bh).contains(&y));
        let backend = self.terminal.backend_mut();
        let width = backend.width;
        let (x0, x1) = (ax.min(width), (ax + aw).min(width));
        for y in ay..(ay + ah).min(backend.height) {
            let row = &mut backend.pixmap[y * width + x0..y * width + x1];
            let ground = if in_bar(y) {
                dim(SEE_THROUGH, 0.4)
            } else {
                SEE_THROUGH
            };
            row.fill(ground);
        }
        if let Some(fade) = controls {
            self.overlay_controls(cells, fade);
        }
        vec![area]
    }

    /// Karaoke mode's picture, without a GPU: the backdrop (made again for
    /// another cover or size), the lyrics, the count-in and the score, then
    /// the controls' text over it. `full`: all of it; else only the text,
    /// where it was and is (the backdrop put back under the last frame's).
    /// Returns the pixels it changed.
    fn paint_karaoke(
        &mut self,
        (cells, picture): (Rect, &KaraokePicture),
        art: Option<&Art>,
        full: bool,
    ) -> Vec<PixelRect> {
        let area = self.pixel_rect(cells);
        let bar = picture
            .controls
            .map(|_| self.pixel_rect(controls_bar(cells)));
        let art_id = Some(art.map_or(0, |a| a.id));
        let fits = self
            .backdrop
            .as_ref()
            .is_some_and(|b| (b.art, b.size) == (art_id, (area.2, area.3)));
        if !fits {
            self.backdrop = Some(Backdrop::new(art, area.2, area.3));
        }
        let full = full || !fits;
        let Some(backdrop) = &self.backdrop else {
            return Vec::new();
        };
        let backend = self.terminal.backend_mut();
        let (width, height) = (backend.width, backend.height);
        let mut target = Target {
            pixels: &mut backend.pixmap,
            width,
            height,
        };
        let changed = karaoke::paint(
            &mut target,
            &mut self.faces,
            &mut self.karaoke_layer,
            backdrop,
            picture,
            area,
            bar,
            full,
        );
        if let (true, Some(fade)) = (full, picture.controls) {
            self.overlay_controls(cells, fade);
        }
        changed
    }

    /// Karaoke mode's controls over its picture, where they are laid out:
    /// their cells' text and their bar, faded in to `fade` as far as the
    /// bar shows ([`crate::core::karaoke::controls::handover`]).
    fn overlay_controls(&mut self, cells: Rect, fade: f64) {
        let Some(backdrop) = &self.backdrop else {
            return;
        };
        let bar_cells = controls_bar(cells);
        let (area, bar) = (self.pixel_rect(cells), self.pixel_rect(bar_cells));
        let backend = self.terminal.backend_mut();
        backend.overlay(bar_cells);
        let (width, height) = (backend.width, backend.height);
        let mut target = Target {
            pixels: &mut backend.pixmap,
            width,
            height,
        };
        let shown = crate::core::karaoke::controls::handover(fade).1;
        backdrop.fade_bar(&mut target, area, bar, shown as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::karaoke::Score;
    use std::time::Duration;

    /// The screen's cells.
    const AREA: Rect = Rect::new(0, 0, 80, 24);

    /// The picture with the score `points` and nothing else.
    fn scored(points: u32) -> KaraokePicture {
        KaraokePicture {
            bar: 5.0 / 24.0,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: Some(Score { points, streak: 0 }),
            result: None,
        }
    }

    /// A canvas of 640 × 384 pixels, on the GPU or not, and frames 1/60 s
    /// apart from now.
    fn canvas(gpu: bool) -> (Canvas, impl Fn(usize) -> Instant) {
        let mut canvas = Canvas::new(80, 24, crate::gui::font_px(1.0));
        canvas.set_gpu(gpu);
        canvas.fit(640, 384);
        let t0 = Instant::now();
        (canvas, move |f| {
            t0 + Duration::from_secs_f32(f as f32 / 60.0)
        })
    }

    /// One frame of `points` at frame `f`, as [`Canvas::draw`] makes it
    /// (`last`: the frame before's area and controls): whether it changed
    /// a pixel; and the stage's frame where there is a stage.
    fn frame(
        canvas: &mut Canvas,
        last: &mut Option<(Rect, Option<f64>)>,
        points: u32,
        at: Instant,
    ) -> (bool, Option<stage_gpu::StageFrame>) {
        // As each frame starts: no cells drawn under it yet.
        canvas.terminal.backend_mut().reset_damage();
        canvas.staged = None;
        let changed = canvas.karaoke_picture((AREA, scored(points)), last.as_ref(), None, at);
        *last = Some((AREA, None));
        canvas.shown = *last;
        let state = crate::core::State::new("/m".into(), crate::config::Config::default());
        let tint = stage::Tint::STAGE;
        let stage = canvas.stage_frame((&state, None), 0.0, 1.0 / 60.0, &tint);
        (!changed.is_empty(), stage)
    }

    /// Without a GPU: the same picture frame after frame is not painted
    /// again; points added, it is painted every frame while they roll up
    /// (moving), and no more once they are at rest.
    #[test]
    fn the_picture_is_painted_again_while_its_score_rolls_and_then_not() {
        let (mut canvas, at) = canvas(false);
        let mut last = None;
        assert!(frame(&mut canvas, &mut last, 12_400, at(0)).0);
        assert!(!frame(&mut canvas, &mut last, 12_400, at(1)).0);
        assert!(!canvas.moving());
        assert!(frame(&mut canvas, &mut last, 13_200, at(2)).0);
        let mut painted = 1;
        let mut f = 3;
        while canvas.moving() {
            assert!(frame(&mut canvas, &mut last, 13_200, at(f)).0, "frame {f}");
            painted += 1;
            f += 1;
            assert!(f < 120, "still moving");
        }
        // Rolled in and its glow gone: about 0.4 s and 0.3 s of frames.
        assert!(painted > 30, "{painted}");
        assert!(!frame(&mut canvas, &mut last, 13_200, at(f)).0);
    }

    /// With a GPU the stage asks for frames while its score rolls up, and
    /// no more once it is at rest; the pixmap is painted once, and never
    /// for the score.
    #[test]
    fn the_stage_asks_for_frames_while_its_score_rolls() {
        let (mut canvas, at) = canvas(true);
        let mut last = None;
        let (painted, stage) = frame(&mut canvas, &mut last, 12_400, at(0));
        assert!(painted && stage.is_some_and(|s| !s.instances.is_empty()));
        assert!(!frame(&mut canvas, &mut last, 12_400, at(1)).0);
        assert!(!canvas.moving());
        let (painted, _) = frame(&mut canvas, &mut last, 13_200, at(2));
        assert!(!painted);
        let mut f = 3;
        while canvas.moving() {
            assert!(!frame(&mut canvas, &mut last, 13_200, at(f)).0, "frame {f}");
            f += 1;
            assert!(f < 120, "still moving");
        }
        assert!(f > 30, "{f}");
        assert_eq!(canvas.strip(), None, "no controls: no strip");
    }

    /// Points that came while the mode was left (20 000, then 30 000 on
    /// coming back) show at once: nothing rolls up, on the GPU or not.
    #[test]
    fn a_score_come_while_the_mode_was_left_shows_at_once() {
        for gpu in [false, true] {
            let (mut canvas, at) = canvas(gpu);
            frame(&mut canvas, &mut None, 20_000, at(0));
            // Left: a frame without the picture, what it keeps kept.
            canvas.shown = None;
            let mut last = None;
            frame(&mut canvas, &mut last, 30_000, at(600));
            assert!(!canvas.moving(), "gpu {gpu}");
            let (changed, _) = frame(&mut canvas, &mut last, 30_000, at(601));
            assert!(!changed, "gpu {gpu}");
        }
    }
}
