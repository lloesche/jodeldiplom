//! The frames (the frame scheduler): one decision a tick
//! ([`Paint::plan`]) of how often the ticks come and what is drawn now:
//! the whole frame (what the window's size does to the keys,
//! [`super::metrics::fit`]; the lanes, [`super::lanes`]), only the lanes
//! (where one follows the playhead: no new tiles in most frames), or
//! nothing; in Karaoke mode its stage over its backdrop
//! ([`super::stage_view`]); the spectrum heard, for the bottom bar; and
//! the frames' line in the trace.

use std::time::{Duration, Instant};

use super::backdrop::music::Song;
use super::backdrop::{BackdropFrame, Visualiser};
use super::lanes::{Lanes, Metrics};
use super::stage::Tint;
use super::stage_view::StageView;
use crate::app::App;
use crate::core::karaoke::result::Motion;
use crate::core::Action;
use crate::spectrum::SpectrumFrame;

/// A frame of what moves smoothly where the display's refresh is not
/// what paces it: 60 a second.
pub const FRAME: Duration = Duration::from_millis(16);

/// How often a whole frame is drawn while only the spectrum and the clock
/// move: 30 a second, half the work.
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

/// What is drawn on a tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draw {
    /// Nothing: the window shows what it has.
    Nothing,
    /// Only the lanes move (one follows the playhead): nothing else is
    /// framed, the lanes stand where the playhead is now.
    LanesOnly,
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
    /// A lane that follows the playhead shows.
    pub following: bool,
    /// A fader's cap on the Mixer tab is gliding.
    pub gliding: bool,
    /// Karaoke mode's stage is drawn (iced draws on the GPU).
    pub stage: bool,
    /// What the stage draws moves by itself
    /// ([`super::stage_view::StageView::moving`]), whatever else does.
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
    /// A fader's cap on the Mixer tab glides: every refresh. Its meters
    /// lit are [`Moves::moving`] ([`crate::core::desk::moving`]): they
    /// fall a pixel or two a calm frame.
    gliding: bool,
    /// Something is due within the rest's tick.
    due_soon: bool,
}

/// Where a frame is drawn: the window's size in pixels, its scale
/// factor, and its size in points.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub size: (u32, u32),
    pub scale: f32,
    pub points: iced::Size,
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
    /// Karaoke mode's backdrop on the GPU, and when its last frame was;
    /// the song heard, which it and the stage's effects share.
    backdrop: Visualiser,
    heard: Option<std::sync::Arc<Song>>,
    last_backdrop: Option<Instant>,
    frames: crate::trace::Frames,
    /// The spectrum heard, as the last whole frame took it (the bottom
    /// bar draws it).
    pub spectrum: SpectrumFrame,
    /// What the guards measure on the Shader tab's picture.
    readout: super::screens::shader::readout::Readout,
}

impl Paint {
    pub fn new() -> Paint {
        Paint {
            dirty: true,
            news: false,
            last_whole: None,
            last_stage: None,
            last_paced: None,
            backdrop: Visualiser::default(),
            heard: None,
            last_backdrop: None,
            frames: crate::trace::Frames::default(),
            spectrum: SpectrumFrame::default(),
            readout: Default::default(),
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
            gliding: seen.gliding,
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
            gliding,
            ..
        } = moves;
        let moving = moves.moving || seen.pictures_moving;
        let pace = if self.dirty || smooth || layer || gliding {
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
        } else if layer && seen.following {
            Draw::LanesOnly
        } else if layer {
            // The lanes as drawn last do not follow the playhead yet.
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
        (stage, lanes): (&StageView, &mut Lanes),
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
        let mode = crate::core::karaoke::in_mode(&app.state);
        let what = if draw == Draw::LanesOnly {
            lanes.frame(heard);
            "the lanes alone".to_owned()
        } else {
            self.last_whole = Some(at);
            self.whole((stage, lanes), app, place, heard);
            if mode {
                "Karaoke mode".to_owned()
            } else {
                format!("{:?}, {:?}", app.state.screen, app.state.side)
            }
        };
        // Only Karaoke mode's stage hears how far behind the GPU is.
        let (late, handed) = if mode {
            stage.late()
        } else {
            (false, Duration::ZERO)
        };
        if paced && plan.pace == Pace::Refresh {
            jodel_gpu::pace::frame(late);
        }
        self.frames
            .frame(gap, late, started.elapsed(), handed, &what);
    }

    /// The whole frame: what the core keeps of the window's size, the
    /// lanes; in Karaoke mode its stage stepped over its backdrop; and the
    /// spectrum heard now.
    fn whole(
        &mut self,
        (stage, lanes): (&StageView, &mut Lanes),
        app: &mut App,
        place: Place,
        heard: f64,
    ) {
        let drawing = Instant::now();
        let since = |last: Option<Instant>| last.map_or(0.0, |at| (drawing - at).as_secs_f32());
        let dt = since(self.last_stage);
        let backdrop_dt = since(self.last_backdrop);
        Song::follow(&mut self.heard, crate::core::karaoke::rhythm_of(&app.state));
        let song = self.heard.as_deref();
        // The list's page, the help's, the desk and the Transition tab's
        // lanes as the window lays them out (the lanes' letters a cell of
        // theirs wide); how far up Karaoke mode's controls reach.
        let metrics = Metrics::at(place.scale);
        let letter = metrics.cell.0 as f32 / place.scale;
        let height = place.points.height;
        let fit = super::metrics::fit(place.points, &app.state, letter);
        if fit != app.state.fit {
            app.dispatch(Action::Fit(fit));
        }
        self.spectrum = app.spectrum();
        let gpu = stage.gpu();
        // The Shader tab's picture: the mode's stage at the widget's size
        // (half the window's before it is first drawn).
        let tab = crate::core::shader_tab::shows(&app.state);
        let size = if tab {
            (stage.drawn_size()).unwrap_or((place.size.0 / 2, place.size.1 / 2))
        } else {
            place.size
        };
        let backdrop = self
            .backdrop
            .frame((app, song), gpu, heard, size, backdrop_dt);
        if tab {
            let small = self.backdrop.picture(app.art());
            if let Some(r) = self.readout.step((backdrop.as_ref(), &small), drawing) {
                app.dispatch(Action::Shader(
                    crate::core::shader_tab::ShaderAction::Readout(r),
                ));
            }
        }
        self.last_backdrop = backdrop.is_some().then_some(drawing);
        let spots =
            super::screens::lane_spots(&app.state, place.points, place.scale, metrics, heard);
        let lanes_gpu = lanes.gpu();
        lanes.show(spots, place.scale, lanes_gpu);
        lanes.frame(heard);
        if crate::core::karaoke::stage_shown(&app.state) {
            let bar = if tab {
                0.0
            } else {
                super::screens::karaoke::mode::bar_share(height)
            };
            let picture = crate::view::karaoke::picture(&app.state, bar, heard);
            // The result screen tinted by the palette of the story's scene
            // showing.
            let tint = Tint::of(backdrop.as_ref().map_or(&[][..], BackdropFrame::palette));
            stage.frame(
                (&app.state, song),
                &picture,
                (size, place.scale),
                (heard, dt),
                &tint,
                backdrop,
            );
            self.last_stage = gpu.then_some(drawing);
        } else {
            stage.left();
            self.last_stage = None;
        }
        crate::trace::took("frame: drawn", drawing, false);
        self.dirty = false;
        self.news = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window in view where nothing moves by itself.
    fn still() -> Seen {
        Seen {
            visible: true,
            following: false,
            gliding: false,
            stage: true,
            pictures_moving: false,
        }
    }

    /// The queue shown, every track got ready, nothing playing: no frames
    /// but the rest's ticks, and nothing drawn; a worker's news (an icon
    /// lit) is drawn once, and the window rests again.
    #[test]
    fn an_idle_queue_rests() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let t0 = Instant::now();
        paint.last_whole = Some(t0);
        let rest = Plan {
            pace: Pace::Every(REST),
            draw: Draw::Nothing,
        };
        for k in 1..20 {
            let at = t0 + REST * k;
            assert_eq!(paint.decide(Moves::default(), still(), at), rest);
        }
        paint.news = true;
        let at = t0 + REST * 20;
        let plan = paint.decide(Moves::default(), still(), at);
        assert_eq!(
            plan,
            Plan {
                pace: Pace::Every(REST),
                draw: Draw::Whole
            }
        );
    }

    /// The Mixer tab with nothing on it moving (nothing playing, its
    /// meters dark, no cap gliding, the microphone not listened to:
    /// [`crate::core::desk::moving`] false) rests like any other: the
    /// rest's ticks, nothing drawn.
    #[test]
    fn an_idle_mixer_tab_rests() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let t0 = Instant::now();
        paint.last_whole = Some(t0);
        for k in 1..20 {
            let plan = paint.decide(Moves::default(), still(), t0 + REST * k);
            assert_eq!(
                plan,
                Plan {
                    pace: Pace::Every(REST),
                    draw: Draw::Nothing
                }
            );
        }
    }

    /// The Mixer tab keeps its pace while something on it moves: a cap
    /// gliding or a fader dragged at the refresh; its meters lit (a
    /// track playing, a peak falling, the microphone listened to) at the
    /// calm rate, each calm frame whole.
    #[test]
    fn a_mixer_tab_with_something_moving_keeps_its_pace() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let t0 = Instant::now();
        paint.last_whole = Some(t0);
        let at = t0 + CALM;
        let gliding = Moves {
            gliding: true,
            ..Moves::default()
        };
        assert_eq!(paint.decide(gliding, still(), at).pace, Pace::Refresh);
        // A fader dragged moves smoothly ([`crate::app::App::smooth`]).
        let dragged = Moves {
            smooth: true,
            ..Moves::default()
        };
        assert_eq!(paint.decide(dragged, still(), at).pace, Pace::Refresh);
        let lit = Moves {
            moving: true,
            ..Moves::default()
        };
        assert_eq!(
            paint.decide(lit, still(), at),
            Plan {
                pace: Pace::Every(CALM),
                draw: Draw::Whole
            }
        );
        // Between two calm frames, nothing.
        paint.last_whole = Some(at);
        let between = at + Duration::from_millis(16);
        assert_eq!(paint.decide(lit, still(), between).draw, Draw::Nothing);
    }

    /// A window nobody can see (its frames stopped coming) draws nothing,
    /// whatever moves, so its frames make no background GPU work give way
    /// ([`jodel_gpu::pace::frame`] is told only of frames drawn); its ticks
    /// stay at the refresh, so the first that comes again shows it.
    #[test]
    fn a_window_nobody_can_see_draws_nothing() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let everything = Moves {
            smooth: true,
            moving: true,
            layer: true,
            gliding: true,
            due_soon: true,
        };
        let at = Instant::now();
        let covered = Seen {
            visible: false,
            pictures_moving: true,
            ..still()
        };
        let plan = paint.decide(everything, covered, at);
        assert_eq!(
            plan,
            Plan {
                pace: Pace::Refresh,
                draw: Draw::Nothing
            }
        );
    }

    /// The stage's pictures moving by themselves with nothing else moving
    /// (a score rolling up in a paused song) bring calm frames, each drawn
    /// whole; once they stop, the window rests.
    #[test]
    fn pictures_moving_by_themselves_bring_calm_frames() {
        let mut paint = Paint::new();
        paint.dirty = false;
        let t0 = Instant::now();
        paint.last_whole = Some(t0);
        let seen = |pictures_moving| Seen {
            pictures_moving,
            ..still()
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
