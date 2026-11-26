//! The frames: one decision a tick ([`Paint::plan`]) of how often the
//! ticks come and what the old canvas ([`super::embed`]) draws now: all of
//! it, only the lane that follows the playhead (the GPU scrolls it), or
//! nothing; Karaoke mode's stage with the picture's shine over it; the
//! spectrum heard, for the bottom bar; and the frames' line in the
//! trace.

use std::time::{Duration, Instant};

use super::backdrop::music::Song;
use super::backdrop::Visualiser;
use super::embed::{Drawn, Embed};
use super::stage::Tint;
use super::stage_gpu::{self, StageFrame};
use super::streak;
use crate::app::App;
use crate::core::singing::Motion;
use crate::core::{Action, Fit};
use crate::spectrum::SpectrumFrame;
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
    /// The canvas's pictures move by themselves
    /// ([`super::Canvas::moving`]), whatever else does.
    pub pictures_moving: bool,
}

/// What moves at a tick, for [`Paint::plan`]'s decision.
#[derive(Clone, Copy, Debug, Default)]
struct Moves {
    /// Something moves smoothly: a frame every refresh.
    smooth: bool,
    /// Something moves: a frame at the calm rate at least.
    moving: bool,
    /// A lane follows the playhead.
    layer: bool,
    /// The Mixer tab's meters and caps (iced's): every refresh.
    mixer: bool,
    /// Something is due within the rest's tick.
    due_soon: bool,
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
    /// Karaoke mode's backdrop on the GPU, and when its last frame was;
    /// the song heard, which it and the stage's effects share.
    backdrop: Visualiser,
    heard: Option<std::sync::Arc<Song>>,
    last_backdrop: Option<Instant>,
    frames: crate::trace::Frames,
    /// The spectrum heard, as the last whole frame took it (the bottom
    /// bar draws it).
    pub spectrum: SpectrumFrame,
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
            backdrop: Visualiser::default(),
            heard: None,
            last_backdrop: None,
            frames: crate::trace::Frames::default(),
            spectrum: SpectrumFrame::default(),
        }
    }

    /// The one decision of a tick at `at`: how often the ticks come from
    /// now, and what is drawn now.
    pub fn plan(&self, app: &App, seen: Seen, at: Instant) -> Plan {
        let state = &app.state;
        // A song's result moves only where the stage draws it.
        let result = crate::core::karaoke::result_motion(state).filter(|_| seen.stage);
        // The backdrop settles only in frames drawn: not while minimised.
        let settling = seen.visible && self.backdrop.moving();
        let moves = Moves {
            smooth: app.smooth() || result == Some(Motion::Revealing),
            moving: app.animating() || result == Some(Motion::Twinkling) || settling,
            layer: app.layer_moves(),
            mixer: crate::core::desk::on_tab(state) || seen.gliding,
            due_soon: state
                .due()
                .is_some_and(|due| due.saturating_sub(app.now_ms()) < REST.as_millis() as u64),
        };
        self.decide(moves, seen, at)
    }

    /// [`Paint::plan`]'s decision for what `moves` at the tick `at`.
    fn decide(&self, moves: Moves, seen: Seen, at: Instant) -> Plan {
        let Moves {
            smooth,
            layer,
            mixer,
            ..
        } = moves;
        let moving = moves.moving || seen.pictures_moving;
        let pace = if self.dirty || smooth || layer || mixer {
            Pace::Refresh
        } else if moving {
            Pace::Every(CALM)
        } else if moves.due_soon {
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

    /// Draws what `plan` says, for the tick at `at`, `heard` seconds into
    /// the track. A frame of what moves (no input in it) at the display's
    /// refresh lets background GPU work know how the window keeps up, and
    /// makes it give way ([`jodel_gpu::pace::frame`]); calm frames leave
    /// the GPU time enough and say nothing.
    pub fn draw(
        &mut self,
        plan: Plan,
        embed: &Embed,
        app: &mut App,
        place: Place,
        (heard, at): (f64, Instant),
    ) {
        let draw = plan.draw;
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
                ..Drawn::default()
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
        if paced && plan.pace == Pace::Refresh {
            jodel_gpu::pace::frame(late);
        }
        self.frames
            .frame(gap, late, started.elapsed(), handed, &what);
    }

    /// The whole frame: the widgets, the pictures, the layer, the stage and
    /// the backdrop under it; and the spectrum heard now.
    fn whole(&mut self, embed: &Embed, app: &mut App, place: Place, heard: f64) {
        let drawing = Instant::now();
        let shine = &mut self.shine;
        let since = |last: Option<Instant>| last.map_or(0.0, |at| (drawing - at).as_secs_f32());
        let dt = since(self.last_stage);
        let backdrop_dt = since(self.last_backdrop);
        let visualiser = &mut self.backdrop;
        Song::follow(&mut self.heard, crate::core::karaoke::rhythm_of(&app.state));
        let song = self.heard.as_deref();
        let (mut staged, mut drawn) = (false, false);
        let spectrum = &mut self.spectrum;
        embed.draw(place.scale, |canvas, gpu| {
            canvas.fit(place.size.0, place.size.1);
            canvas.hold(&app.state);
            let grid = canvas.grid();
            // The list's page and the desk as wide as the window draws
            // them, at its cells' size.
            let (cw, ch) = canvas.cell();
            let cell = (cw as f32 / place.scale, ch as f32 / place.scale);
            let fit = Fit {
                desk_width: crate::view::layout::desk_width(grid, cell.0),
                ..super::screens::list::fit(&app.state, grid, cell)
            };
            if fit != app.state.fit {
                app.dispatch(Action::Fit(fit));
            }
            *spectrum = app.spectrum();
            let scene = Scene {
                state: &app.state,
                grid,
                pointer: place.pointer,
                heard,
            };
            canvas.draw(&scene, app.art());
            let layer = canvas.layer_frame(heard);
            let stage = gpu
                .then(|| stage(canvas, (app, song), shine, heard, dt))
                .flatten();
            staged = stage.is_some();
            let backdrop = visualiser.frame((app, song), gpu, heard, place.size, backdrop_dt);
            drawn = backdrop.is_some();
            Drawn {
                layer,
                stage,
                backdrop,
            }
        });
        crate::trace::took("frame: drawn", drawing, false);
        self.last_stage = staged.then_some(drawing);
        self.last_backdrop = drawn.then_some(drawing);
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
    (app, song): (&App, Option<&Song>),
    shine: &mut streak::PictureShine,
    heard: f64,
    dt: f32,
) -> Option<StageFrame> {
    let layers = app.cover_layers();
    let tint = Tint::of(layers.as_ref().map_or(&[][..], |l| &l.layers.palette[..]));
    let stage = canvas.stage_frame((&app.state, song), heard, dt, &tint);
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

    /// The canvas's pictures moving by themselves with nothing else moving
    /// (a score rolling up in a paused song) bring calm frames, each drawn
    /// whole; once they stop, the window rests.
    #[test]
    fn pictures_moving_by_themselves_bring_calm_frames() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let t0 = Instant::now();
        paint.last_whole = Some(t0);
        let seen = |pictures_moving| Seen {
            visible: true,
            layer: false,
            gliding: false,
            stage: true,
            pictures_moving,
        };
        let at = t0 + CALM;
        let plan = paint.decide(Moves::default(), seen(true), at);
        assert_eq!(
            plan,
            Plan {
                pace: Pace::Every(CALM),
                draw: Draw::Whole
            }
        );
        let plan = paint.decide(Moves::default(), seen(false), at);
        assert_eq!(
            plan,
            Plan {
                pace: Pace::Every(REST),
                draw: Draw::Nothing
            }
        );
    }

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
