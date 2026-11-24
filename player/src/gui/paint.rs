//! The frames: one decision a tick ([`Paint::plan`]) of how often the
//! ticks come and what the old canvas ([`super::embed`]) draws now: all of
//! it, only the lane that follows the playhead (the GPU scrolls it), or
//! nothing; Karaoke mode's stage with the picture's shine over it; and the
//! frames' line in the trace.

use std::time::{Duration, Instant};

use super::embed::{Drawn, Embed};
use super::stage::Tint;
use super::stage_gpu::{self, StageFrame};
use super::streak;
use super::window::SetupRun;
use crate::app::App;
use crate::core::singing::Motion;
use crate::core::Action;
use crate::view::Scene;

/// A frame of what moves smoothly where the display's refresh is not
/// what paces it: 60 a second.
pub const FRAME: Duration = Duration::from_millis(16);

/// How often the old canvas is drawn while only the spectrum and the
/// clock move: 30 a second, half the work.
pub const CALM: Duration = Duration::from_millis(33);

/// The rest's tick: what the engine and the workers sent is taken in, and
/// what is due is done, at least this often (it comes whatever the pace).
pub const REST: Duration = Duration::from_millis(250);

/// How early a tick may come for a calm frame (the timers' jitter).
const SLACK: Duration = Duration::from_millis(3);

/// How often the ticks come.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    /// Every refresh of the display.
    Refresh,
    /// At this interval.
    Every(Duration),
}

/// What the old canvas draws on a tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draw {
    /// Nothing: the window shows what it has.
    Nothing,
    /// Only the lane that follows the playhead moves, on the GPU.
    LayerOnly,
    /// The whole frame.
    Whole,
}

/// A tick's decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub pace: Pace,
    pub draw: Draw,
}

/// How the window stands, for [`Paint::plan`].
#[derive(Clone, Copy, Debug)]
pub struct Seen {
    /// It can be seen (not minimised, its frames coming).
    pub visible: bool,
    /// The GPU scrolls a lane by itself now.
    pub layer: bool,
    /// A fader's cap on the Mixer tab is gliding.
    pub gliding: bool,
    /// Karaoke mode's stage is drawn (iced draws on the GPU).
    pub stage: bool,
}

/// Where a frame is drawn: the window's size in pixels, its scale factor,
/// and the pointer's cell while it is over the canvas (the links under it
/// are underlined).
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub size: (u32, u32),
    pub scale: f32,
    pub pointer: Option<(u16, u16)>,
}

/// The frames' state.
pub struct Paint {
    /// Input changed what shows: the next tick draws it whole.
    pub dirty: bool,
    /// The engine or the workers sent news: drawn with the next calm frame
    /// while something moves, at once otherwise.
    pub news: bool,
    /// When the last whole frame's tick came.
    last_whole: Option<Instant>,
    /// When the last frame with Karaoke mode's stage was drawn.
    last_stage: Option<Instant>,
    /// When the last frame of what moves began (for the trace).
    last_paced: Option<Instant>,
    /// The singing's streak as the picture shines with it.
    shine: streak::PictureShine,
    frames: crate::trace::Frames,
}

impl Paint {
    pub fn new() -> Paint {
        Paint {
            dirty: true,
            news: false,
            last_whole: None,
            last_stage: None,
            last_paced: None,
            shine: streak::PictureShine::default(),
            frames: crate::trace::Frames::default(),
        }
    }

    /// The one decision of a tick at `at`: how often the ticks come from
    /// now, and what is drawn now.
    pub fn plan(&self, app: &App, seen: Seen, at: Instant) -> Plan {
        let state = &app.state;
        // A song's result moves only where the stage draws it.
        let result = crate::core::karaoke::result_motion(state).filter(|_| seen.stage);
        let smooth = app.smooth() || result == Some(Motion::Revealing);
        let moving = app.animating() || result == Some(Motion::Twinkling);
        let layer = app.layer_moves();
        // The Mixer tab's meters and caps are iced's: every refresh.
        let mixer = crate::core::desk::on_tab(state) || seen.gliding;
        let pace = if self.dirty || smooth || layer || mixer {
            Pace::Refresh
        } else if moving {
            Pace::Every(CALM)
        } else if state
            .due()
            .is_some_and(|due| due.saturating_sub(app.now_ms()) < REST.as_millis() as u64)
        {
            // What is due soon (changed settings, a count-in, a preview)
            // on time.
            Pace::Every(FRAME)
        } else {
            Pace::Every(REST)
        };
        let calm_due = self.calm_due(at);
        let draw = if !seen.visible {
            Draw::Nothing
        } else if self.dirty
            || smooth
            || (self.news && !moving)
            || ((moving || self.news) && calm_due)
        {
            // Input, what moves smoothly, news while nothing else moves;
            // otherwise what moves at the calm rate (news with it).
            Draw::Whole
        } else if layer && seen.layer {
            Draw::LayerOnly
        } else if layer {
            // Without the GPU's layer the lane is in the pixmap.
            Draw::Whole
        } else {
            Draw::Nothing
        };
        Plan { pace, draw }
    }

    /// Whether a calm frame is due at the tick `at`: [`CALM`] after the
    /// last whole frame's tick (not after its drawing ended, or every other
    /// tick would be skipped), a little early allowed.
    fn calm_due(&self, at: Instant) -> bool {
        self.last_whole
            .is_none_or(|w| at.saturating_duration_since(w) + SLACK >= CALM)
    }

    /// The setup screen's frame.
    pub fn setup(&mut self, embed: &Embed, run: &mut SetupRun, place: Place) {
        let status = run.setup.status();
        embed.draw(place.scale, |canvas, _| {
            canvas.fit(place.size.0, place.size.1);
            run.buttons = canvas.draw_widgets(|frame| crate::view::setup::draw(frame, &status));
            Drawn::default()
        });
        self.dirty = false;
    }

    /// Draws what `draw` says, for the tick at `at`, `heard` seconds into
    /// the track. A frame of what moves (no input in it) lets background
    /// GPU work know how the window keeps up.
    pub fn draw(
        &mut self,
        draw: Draw,
        embed: &Embed,
        app: &mut App,
        place: Place,
        (heard, at): (f64, Instant),
    ) {
        if draw == Draw::Nothing {
            return;
        }
        let paced = !self.dirty;
        let started = Instant::now();
        // Frames of what moves, and how far apart they come (a long pause
        // is the window resting), for the trace.
        let gap = paced
            .then(|| self.last_paced.map(|last| at - last))
            .flatten()
            .filter(|&gap| gap < REST);
        if paced {
            self.last_paced = Some(at);
        }
        let what = if draw == Draw::LayerOnly {
            embed.draw(place.scale, |canvas, _| Drawn {
                layer: canvas.layer_frame(heard),
                stage: None,
            });
            "the layer alone".to_owned()
        } else {
            self.last_whole = Some(at);
            self.whole(embed, app, place, heard);
            if crate::core::karaoke::in_mode(&app.state) {
                "Karaoke mode".to_owned()
            } else {
                format!("{:?}, {:?}", app.state.screen, app.state.side)
            }
        };
        let (late, handed) = embed.late();
        if paced {
            jodel_gpu::pace::frame(late);
        }
        self.frames
            .frame(gap, late, started.elapsed(), handed, &what);
    }

    /// The whole frame: the widgets, the pictures, the layer and the stage.
    fn whole(&mut self, embed: &Embed, app: &mut App, place: Place, heard: f64) {
        let drawing = Instant::now();
        let shine = &mut self.shine;
        let dt = self
            .last_stage
            .map_or(0.0, |at| drawing.duration_since(at).as_secs_f32());
        let mut staged = false;
        embed.draw(place.scale, |canvas, gpu| {
            let grid = canvas.fit(place.size.0, place.size.1);
            if grid != app.state.size {
                app.dispatch(Action::Resize(grid.0, grid.1));
            }
            let spectrum = app.spectrum();
            let scene = Scene {
                state: &app.state,
                spectrum: &spectrum,
                pointer: place.pointer,
                heard,
            };
            canvas.draw(&scene, app.art());
            let layer = canvas.layer_frame(heard);
            let stage = gpu.then(|| stage(canvas, app, shine, heard, dt)).flatten();
            staged = stage.is_some();
            Drawn { layer, stage }
        });
        crate::trace::took("frame: drawn", drawing, false);
        self.last_stage = staged.then_some(drawing);
        self.dirty = false;
        self.news = false;
    }
}

/// Karaoke mode's stage, stepped by `dt` seconds, with the picture's
/// bubbles' liquid and bursts and its streak's shine over it; or the
/// result screen alone (on its reveal's clock), tinted by the cover's
/// palette where its layers are ready.
fn stage(
    canvas: &mut super::Canvas,
    app: &App,
    shine: &mut streak::PictureShine,
    heard: f64,
    dt: f32,
) -> Option<StageFrame> {
    let layers = app.cover_layers();
    let tint = Tint::of(layers.as_ref().map_or(&[][..], |l| &l.palette[..]));
    let stage = canvas.stage_frame(&app.state, heard, dt, &tint);
    if crate::core::karaoke::result_shown(&app.state).is_some() {
        return stage;
    }
    let clock = app.state.karaoke.lag.sung_clock(heard);
    let over = canvas.picture_over();
    let song = streak::Sung {
        singing: &app.state.karaoke.singing,
        game: app.state.config.karaoke_game,
    };
    let (edge, ups) = shine.frame(over, song, clock);
    let ours = over.into_iter();
    let bubbles = ours.clone().flat_map(|o| o.bubbles.iter().copied());
    let bubbles: Vec<_> = bubbles.chain(edge).collect();
    let sprays = ours.flat_map(|o| o.sprays.iter()).chain(ups);
    let sprays: Vec<_> = sprays.copied().collect();
    stage_gpu::with_picture(stage, &bubbles, &sprays, clock as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticks every [`CALM`] (or a little early) draw a whole frame each,
    /// at 60 ticks a second every other one: the gate is on the ticks'
    /// instants.
    #[test]
    fn calm_frames_come_at_every_calm_tick() {
        let mut paint = Paint::new();
        let t0 = Instant::now();
        assert!(paint.calm_due(t0));
        paint.last_whole = Some(t0);
        assert!(paint.calm_due(t0 + CALM));
        assert!(paint.calm_due(t0 + CALM - Duration::from_millis(2)));
        assert!(!paint.calm_due(t0 + Duration::from_micros(16_667)));
        assert!(paint.calm_due(t0 + Duration::from_micros(33_333)));
    }
}
