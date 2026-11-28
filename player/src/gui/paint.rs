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

/// The shortest gap between ticks taken for a refresh: shorter ones come
/// from the pacer's redraws while the window goes in or out of fullscreen
/// (every 4 ms), or catch up after a late one.
const SHORTEST: Duration = Duration::from_millis(6);

/// When a tick was due ([`Lateness::tick`]): at the display's refresh,
/// at a timer's interval, or not at a time the window keeps (the rest's
/// tick, which comes whatever the pace).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Due {
    Refresh,
    Every(Duration),
    Rest,
}

/// The ticks as they come, to tell when the window is starved of the
/// GPU. iced gets each frame (`get_current_texture`, on Metal
/// `nextDrawable`, which waits while the GPU is busy with other work) and
/// presents it on the thread that ticks, inside its own loop, so that wait
/// cannot be timed here; it shows as the next tick coming late, a refresh
/// tick and a timer's alike. A tick a quarter of its interval or more past
/// it is late (the pace starts giving way only on several close together:
/// [`jodel_gpu::pace::tick`]): the refresh's interval is the shortest gap
/// between refresh ticks lately, [`SHORTEST`] at least (it creeps up a
/// hundredth a tick, so another display's slower refresh, or a 120 Hz one
/// drawn at 60, is taken in within a second or two), a timer's is its own.
/// Only gaps between two ticks due the same way count, and none after a
/// tick whose own frame left less than a quarter of the interval (the
/// window's own work, not the GPU's). A display past about 165 Hz is not
/// timed at the refresh (its gaps are under [`SHORTEST`]); and a starved
/// window whose frames come steadily at half the refresh is learned as a
/// slower display after a second or so (as a 120 Hz one drawn at 60 is),
/// late long enough to start giving way first: the trace's starved ticks
/// show whether that comes and goes.
#[derive(Debug)]
struct Lateness {
    last: Option<(Instant, Due)>,
    refresh: Option<Duration>,
    /// How long the last tick's frame took to draw.
    drew: Duration,
}

impl Lateness {
    fn new() -> Lateness {
        Lateness {
            last: None,
            refresh: None,
            drew: Duration::ZERO,
        }
    }

    /// A tick at `at`, due as `due`: whether it came late.
    fn tick(&mut self, at: Instant, due: Due) -> bool {
        let drew = std::mem::take(&mut self.drew);
        let before = self.last.replace((at, due));
        if due == Due::Rest {
            self.last = None;
            return false;
        }
        let Some(gap) = before
            .filter(|&(_, d)| d == due)
            .map(|(l, _)| at.saturating_duration_since(l))
            .filter(|&g| g >= SHORTEST && g < REST)
        else {
            return false;
        };
        let interval = match due {
            Due::Every(every) => every,
            _ => {
                let r = self.refresh.map_or(gap, |r| r.mul_f64(1.01).min(gap));
                self.refresh = Some(r);
                r
            }
        };
        // Late, unless the window's own frame left it less than a quarter of
        // the interval.
        drew + interval / 4 < interval && gap >= interval + interval / 4
    }
}

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
    /// The ticks as they come, to tell when the window is starved of the
    /// GPU; whether this tick came late (told with its frame); and how
    /// the last plan paced the ticks (when this one was due).
    lateness: Lateness,
    late: bool,
    paced_by: Option<Pace>,
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
            lateness: Lateness::new(),
            late: false,
            paced_by: None,
        }
    }

    /// A tick at `at`, the rest's (`rest`) or one the plan paced: whether
    /// it came late ([`Lateness`]), told to background GPU work with its
    /// frame ([`Paint::draw`]).
    pub fn ticked(&mut self, at: Instant, rest: bool) {
        let due = match (rest, self.paced_by) {
            (false, Some(Pace::Refresh)) => Due::Refresh,
            (false, Some(Pace::Every(every))) => Due::Every(every),
            _ => Due::Rest,
        };
        self.late = self.lateness.tick(at, due);
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
    /// the track. Background GPU work is told of each tick
    /// ([`jodel_gpu::pace::tick`]): whether it came late ([`Lateness`], or
    /// in Karaoke mode the stage's last frame not through the GPU yet),
    /// and whether what moves (no input in it) is drawn at the display's
    /// refresh; it gives way while that moves, or the ticks come late.
    pub fn draw(
        &mut self,
        plan: Plan,
        (stage, lanes): (&StageView, &mut Lanes),
        app: &mut App,
        place: Place,
        (heard, at): (f64, Instant),
    ) {
        self.paced_by = Some(plan.pace);
        let draw = plan.draw;
        if draw == Draw::Nothing {
            jodel_gpu::pace::tick(std::mem::take(&mut self.late), false);
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
        let smooth = paced && plan.pace == Pace::Refresh;
        let tick_late = std::mem::take(&mut self.late);
        jodel_gpu::pace::tick(tick_late || (smooth && late), smooth);
        let drew = started.elapsed();
        self.lateness.drew = drew;
        self.frames.frame(gap, late, drew, handed, &what);
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

    /// Ticks `gap` apart, due as `due`, from `at` on: how many came late.
    fn late_ticks(
        l: &mut Lateness,
        at: &mut Instant,
        (n, gap): (usize, Duration),
        due: Due,
    ) -> usize {
        (0..n)
            .filter(|_| {
                *at += gap;
                l.tick(*at, due)
            })
            .count()
    }

    /// A tick a quarter of its interval or more late is told, at 60 Hz and
    /// at 120 Hz; one on time, the ticks starting again after a pause or
    /// after the rest's tick, are not.
    #[test]
    fn a_tick_a_quarter_of_a_frame_late_is_told() {
        for hz in [60.0, 120.0] {
            let frame = Duration::from_secs_f64(1.0 / hz);
            let mut l = Lateness::new();
            let mut at = Instant::now();
            assert_eq!(late_ticks(&mut l, &mut at, (30, frame), Due::Refresh), 0);
            at += frame / 5;
            assert!(!l.tick(at, Due::Refresh), "{hz} Hz, a fifth late");
            at += frame + frame / 3;
            assert!(l.tick(at, Due::Refresh), "{hz} Hz, a third late");
            at += 2 * frame;
            assert!(l.tick(at, Due::Refresh), "{hz} Hz, a frame missed");
            at += REST;
            assert!(!l.tick(at, Due::Refresh), "the ticks stopped");
            assert!(!l.tick(at + 5 * frame, Due::Rest));
            at += 9 * frame;
            assert!(!l.tick(at, Due::Refresh), "after the rest's tick");
        }
    }

    /// Calm ticks after ticks at the refresh are on time at their own
    /// interval, and late past it.
    #[test]
    fn calm_ticks_are_late_only_past_their_own_interval() {
        let frame = Duration::from_secs_f64(1.0 / 60.0);
        let mut l = Lateness::new();
        let mut at = Instant::now();
        assert_eq!(late_ticks(&mut l, &mut at, (60, frame), Due::Refresh), 0);
        let calm = Due::Every(CALM);
        assert_eq!(late_ticks(&mut l, &mut at, (60, CALM), calm), 0);
        assert_eq!(late_ticks(&mut l, &mut at, (5, CALM * 3 / 2), calm), 5);
    }

    /// A burst of ticks 4 ms apart (going fullscreen) does not make the
    /// refresh 4 ms; a 120 Hz display drawn at 60 is taken in within a
    /// second or two; a tick after a slow frame of the window's own is not
    /// told.
    #[test]
    fn the_refresh_is_not_fooled_by_bursts_or_slow_frames() {
        let frame = Duration::from_secs_f64(1.0 / 60.0);
        let mut l = Lateness::new();
        let mut at = Instant::now();
        late_ticks(&mut l, &mut at, (60, frame), Due::Refresh);
        late_ticks(
            &mut l,
            &mut at,
            (100, Duration::from_millis(4)),
            Due::Refresh,
        );
        assert_eq!(late_ticks(&mut l, &mut at, (60, frame), Due::Refresh), 0);
        // 120 Hz, then drawn at 60.
        let mut l = Lateness::new();
        late_ticks(&mut l, &mut at, (60, frame / 2), Due::Refresh);
        let early = late_ticks(&mut l, &mut at, (120, frame), Due::Refresh);
        assert!(early < 60, "{early} of the first 120 late");
        assert_eq!(late_ticks(&mut l, &mut at, (60, frame), Due::Refresh), 0);
        // Its own frame slow, leaving less than a quarter of the frame.
        l.drew = frame * 4 / 5;
        at += 2 * frame;
        assert!(!l.tick(at, Due::Refresh), "after a slow frame of its own");
        at += 2 * frame;
        assert!(l.tick(at, Due::Refresh), "late again");
        // At 120 Hz, a frame of 3 ms leaves time enough: a late tick after it
        // is told.
        let fast = frame / 2;
        let mut l = Lateness::new();
        late_ticks(&mut l, &mut at, (60, fast), Due::Refresh);
        l.drew = Duration::from_millis(3);
        at += 2 * fast;
        assert!(
            l.tick(at, Due::Refresh),
            "late after a frame of 3 ms at 120 Hz"
        );
    }

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
    /// whatever moves, so nothing it draws makes background GPU work give
    /// way ([`jodel_gpu::pace::tick`] says nothing moves); its ticks
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
