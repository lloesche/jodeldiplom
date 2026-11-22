//! The engine's side of a transition ([`crate::transition`]).
//!
//! Armed, it waits until the playing track (A) is [`LEAD`] before the
//! window. Then A moves onto a stems route at the open format (the
//! seamless switch; with every gain at one its stems add up to the mix),
//! with its fades as its stems' gains, and the next track (B) opens on a
//! stems route at the same format from a little before where the window
//! meets it, its stems faded by its own lanes and read along the time map
//! ([`Warping`]), so its beats fall on A's. B decodes ahead on the turns a
//! switch gets. From the window's first frame, each new piece of A's
//! output has B's frames for the same output frames added to it; at the
//! window's last frame A is dropped and B plays on, gliding back to its
//! own tempo, and once it is there its position runs straight again and
//! it moves back onto its own route.
//!
//! Without a stem separator both tracks go through [`whole_mix`], which
//! puts the whole music in one stem: their fades are then those of that
//! stem's lanes.
//!
//! The stems on in both tracks are the settings' times a mixer's
//! ([`crate::Command::StemMask`], [`super::stem_mask`]), so a stem muted on a
//! mixer's strip stays out of the mix the mixer hears as main here.

use std::sync::Arc;
use std::time::Duration;

use super::api::{Event, TrackStart};
use super::stem_mask::StemsMask;
use super::track::{rescale, Cue, CueEvent, Heard, HeardWarp, Progress, SegStart, Track};
use super::warping::Warping;
use super::{Engine, SWITCH_TURN};
use crate::error::Error;
use crate::rate::{stems_at, FileFacts, Stage};
use crate::stems::{whole_mix, Gains};
use crate::transition::{Side, TransitionPlan, TransitionState};
use crate::warp::{Glide, TimeMap};

/// Output frames the playing track must hold unwritten for an armed
/// transition to read it again on a switch (what a jump decoded ahead).
const AHEAD_FRAMES: usize = 4096;

/// Seconds before the window that the tracks start to be prepared.
pub(super) const LEAD: f64 = 4.0;

/// Seconds of B's output made before the window (dropped): its resamplers
/// settle there.
const B_LEAD: f64 = 1.0;

/// Seconds of B's input before its first read position (the interpolation
/// reaches back into them).
const B_MARGIN: f64 = 0.05;

/// How long one piece of the mix may wait for B to make its frames (the
/// output holds about half a second); past it, that piece goes without B.
const MIX_PATIENCE: Duration = Duration::from_millis(150);

/// Seconds after the glide back to its tempo that B's position is made
/// straight again.
const SETTLED: f64 = 0.3;

/// Full scale of the 24-bit samples the tracks are mixed in.
const FULL: f64 = 8_388_607.0;

/// Where the mix's peaks start to bend toward full scale (−1 dBFS): two
/// loud masters added go over full scale, and a hard clip there is heard.
const KNEE: f64 = 0.891 * FULL;

/// A sample of the two tracks added: as it is up to [`KNEE`], above it
/// bent smoothly (a tanh with slope 1 at the knee) toward full scale,
/// which it never reaches. Whether it still had to be clipped.
fn soft_limit(sum: i64) -> (i32, bool) {
    let x = sum as f64;
    if x.abs() <= KNEE {
        return (sum as i32, false);
    }
    let room = FULL - KNEE;
    let bent = (KNEE + room * ((x.abs() - KNEE) / room).tanh()).copysign(x);
    let kept = bent.round().clamp(-FULL - 1.0, FULL);
    (kept as i32, kept != bent.round())
}

/// Where a transition is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Waiting for the playing track to come near the window.
    Armed,
    /// B decodes ahead; the window has not started.
    Preparing,
    /// The window is being written: B's output is added to A's.
    Mixing,
    /// B plays alone, gliding back to its own tempo.
    Settling,
}

/// A transition under way.
pub(super) struct Transition {
    plan: Arc<TransitionPlan>,
    /// B's time along the output's (A's before the window, where they are
    /// the same).
    map: Arc<TimeMap>,
    /// A's time along the output's through the window.
    glide: Glide,
    /// The same as a map A is read along, when it glides.
    map_a: Option<Arc<TimeMap>>,
    phase: Phase,
    /// B, decoding ahead (pending) until it takes over.
    b: Option<Track>,
    /// B's output frames count from 0 at A's time `a0`.
    a0: f64,
    /// The output frame of B's output frame 0, once A's timeline is known.
    origin: Option<u64>,
    /// The window's first and last output frames, once known.
    window: Option<(u64, u64)>,
    /// B's output frame of `b.out_buf[b.out_at]`.
    b_next: u64,
    /// Output frames up to which A's output has had B's added.
    mixed_to: u64,
}

impl Transition {
    /// The playing track and the next one, by their indices in the queue.
    pub(super) fn tracks(&self) -> (usize, usize) {
        (self.plan.from, self.plan.into)
    }

    /// A's gains: its lanes at its own time, times the stems on then.
    fn gains_a(plan: &Arc<TransitionPlan>, mask: StemsMask) -> Gains {
        let plan = Arc::clone(plan);
        Arc::new(move |t| {
            let g = plan.gains(Side::From, plan.share(t));
            let on = mask.for_a(t);
            std::array::from_fn(|s| g[s] * on[s])
        })
    }

    /// B's gains: its lanes at the playing track's time where B is at `t`
    /// (the output's time there, along the glide), times the stems on
    /// then.
    fn gains_b(
        plan: &Arc<TransitionPlan>,
        map: &Arc<TimeMap>,
        glide: Glide,
        mask: StemsMask,
    ) -> Gains {
        let (plan, map) = (Arc::clone(plan), Arc::clone(map));
        Arc::new(move |t| {
            let a = glide.at(map.inverse(t));
            let g = plan.gains(Side::Into, plan.share(a));
            let on = mask.for_b(t, || a);
            std::array::from_fn(|s| g[s] * on[s])
        })
    }
}

/// The output frame at which A's time `seconds` is heard, on its timeline.
pub(super) fn output_frame_of(track: &Track, seconds: f64) -> u64 {
    let frame = (seconds * f64::from(track.info.rate)).round().max(0.0) as u64;
    let since = frame.saturating_sub(track.seg_frame);
    track.seg_out + rescale(since, track.route.input_rate, track.route.output_rate)
}

/// A's time at the output frame being written.
pub(super) fn seconds_written(track: &Track, written: u64) -> f64 {
    let since = written.saturating_sub(track.seg_out);
    (track.seg_frame as f64
        + since as f64 * f64::from(track.route.input_rate) / f64::from(track.route.output_rate))
        / f64::from(track.info.rate.max(1))
}

impl Engine {
    /// [`crate::Command::Transition`]: takes the plan if it fits what is
    /// playing.
    pub(super) fn arm_transition(&mut self, plan: TransitionPlan) {
        // One heard already is not replaced (it would be cut off).
        if self.transition_heard() {
            return;
        }
        // Another plan for one armed or preparing: the tracks stay where
        // they are (on their stems), and it prepares again from there.
        if self.transition.is_some() {
            self.forget_transition(None);
        }
        let playing = self.playing.as_ref().map(|t| t.index);
        let why = if playing != Some(plan.from) {
            Some("that track is not the one playing")
        } else if plan.into != plan.from + 1 || plan.into >= self.queue.len() {
            Some("there is no next track to go into")
        } else if plan.length.is_nan() || plan.length <= 0.0 {
            Some("the window is empty")
        } else {
            None
        };
        if let Some(why) = why {
            return self.emit(Event::Transition(TransitionState::Cancelled {
                reason: why.into(),
            }));
        }
        // B on A's beats as they are heard: along the output's time, which
        // runs as A's up to the window and A along the glide in it.
        let beats = plan.time_map();
        let glide = plan.glide_of(&beats);
        let map = Arc::new(beats.through_glide(&glide));
        let map_a = (!glide.is_identity()).then(|| Arc::new(glide.map()));
        let a0 = plan.start - B_LEAD;
        self.transition = Some(Transition {
            plan: Arc::new(plan),
            map,
            glide,
            map_a,
            phase: Phase::Armed,
            b: None,
            a0,
            origin: None,
            window: None,
            b_next: 0,
            mixed_to: 0,
        });
        // A decoded ahead for jumps ([`super::jump`]): what it holds is
        // read again from where it is written, so the transition mixes into
        // what is made from now on.
        let ahead = self
            .playing
            .as_ref()
            .filter(|a| a.unwritten() > AHEAD_FRAMES)
            .map(|a| a.route.clone());
        if let (Some(route), None) = (ahead, &self.switch) {
            let _ = self.begin_switch(route);
        }
        self.emit(Event::Transition(TransitionState::Armed));
    }

    /// Drops the transition, saying why when there is a reason (and it
    /// had not finished), and puts what it changed back: the playing track
    /// loses its fades and goes back to the route the settings choose; a
    /// next track that has taken over runs on a straight timeline from
    /// here. B, if it has not taken over, is dropped in the background.
    pub(super) fn drop_transition(&mut self, reason: Option<&str>) {
        let Some(t) = self.forget_transition(reason) else {
            return;
        };
        if t.phase == Phase::Settling {
            let written = self.out.as_ref().map_or(0, |o| o.output.written());
            self.straighten(&t, written);
        } else {
            let from = t.plan.from;
            if let Some(a) = self.playing.as_mut().filter(|a| a.index == from) {
                a.gains = None;
            }
            if let Some(s) = self.switch.as_mut().filter(|s| s.track.index == from) {
                s.track.gains = None;
            }
        }
        self.reroute();
    }

    /// Drops the transition and B (in the background), saying why when
    /// there is a reason and it had not finished, without putting anything
    /// back (playback is moving elsewhere anyway). Returns what it was.
    pub(super) fn forget_transition(&mut self, reason: Option<&str>) -> Option<Transition> {
        let mut t = self.transition.take()?;
        if let Some(b) = t.b.take() {
            super::drop_later(b);
        }
        // What it queued to be said as it is heard (its window started,
        // say) is not said after it is gone.
        self.cues
            .retain(|c| !matches!(c.event, CueEvent::Transition(_)));
        if t.phase == Phase::Settling {
            // Handed over already: it is done, as far as anyone can hear.
            self.emit(Event::Transition(TransitionState::Finished));
        } else if let Some(reason) = reason {
            self.emit(Event::Transition(TransitionState::Cancelled {
                reason: reason.into(),
            }));
        }
        Some(t)
    }

    /// [`super::Command::Next`] while a transition is heard: the next track
    /// plays on alone at once. Mixing, from where it is heard now (from
    /// its first frame in the window, when the window is not heard yet),
    /// read again there, the playing track cut off; settling (it plays
    /// alone already), on a straight timeline from here. Whether it did.
    pub(super) fn cut_to_next(&mut self) -> bool {
        let Some(t) = self.transition.as_ref() else {
            return false;
        };
        match t.phase {
            Phase::Settling => {
                self.drop_transition(None);
                true
            }
            Phase::Mixing => {
                let (Some(origin), Some(out)) = (t.origin, self.out.as_ref()) else {
                    return false;
                };
                let rate = f64::from(out.info.format.rate.max(1));
                let heard = out.clock.position();
                let frame = t.window.map_or(heard, |(start, _)| heard.max(start));
                let seconds = t
                    .map
                    .at(t.a0 + frame.saturating_sub(origin) as f64 / rate)
                    .max(0.0);
                let into = t.plan.into;
                self.forget_transition(Some("cut to the next track"));
                self.paused = false;
                self.jump(into, seconds);
                true
            }
            Phase::Armed | Phase::Preparing => false,
        }
    }

    /// Whether a transition is heard (mixing, or the next track settling
    /// after it).
    pub(super) fn transition_heard(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| matches!(t.phase, Phase::Mixing | Phase::Settling))
    }

    /// Whether a transition is under way past being armed (its tracks
    /// prepared, mixed or settling), so the stems switched on and off go
    /// into its fades rather than calling it off.
    pub(super) fn transition_running(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| t.phase != Phase::Armed)
    }

    /// The stems heard changed during a transition: its fades take them,
    /// on both tracks, at the time the change is heard at (or at once).
    pub(super) fn retune_transition(&mut self) {
        let mask = self.transition_mask();
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        let (plan, map, glide) = (Arc::clone(&t.plan), Arc::clone(&t.map), t.glide);
        let gains_a = Transition::gains_a(&plan, mask.clone());
        let gains_b = Transition::gains_b(&plan, &map, glide, mask);
        if let Some(b) = t.b.as_mut() {
            b.gains = Some(Arc::clone(&gains_b));
        }
        for track in [
            self.playing.as_mut(),
            self.switch.as_mut().map(|s| &mut s.track),
        ]
        .into_iter()
        .flatten()
        {
            if track.gains.is_none() {
                continue;
            }
            if track.index == plan.from {
                track.gains = Some(Arc::clone(&gains_a));
            } else if track.index == plan.into {
                track.gains = Some(Arc::clone(&gains_b));
            }
        }
    }

    /// Queues a cue at `at`, or after the last one queued.
    fn queue_cue(&mut self, at: u64, event: CueEvent) {
        let at = at.max(self.cues.back().map_or(0, |c| c.at));
        self.cues.push_back(Cue { at, event });
    }

    /// The next track, which has taken over, runs on a straight timeline
    /// from output frame `from` (it is at its own tempo there, or close).
    fn straighten(&mut self, t: &Transition, from: u64) {
        let Some(b) = self.playing.as_mut() else {
            return;
        };
        if b.index != t.plan.into {
            return;
        }
        let rate = self.out.as_ref().map_or(1, |o| o.info.format.rate);
        let origin = t.origin.unwrap_or(from);
        let seconds = t
            .map
            .at(t.a0 + from.saturating_sub(origin) as f64 / f64::from(rate));
        let frame = (seconds * f64::from(b.info.rate)).round().max(0.0) as u64;
        // Output frames counted from the new start, so cues land as before.
        b.emitted = b.emitted.saturating_sub(from.saturating_sub(b.seg_out));
        b.begin_segment(from, frame, SegStart::Continued);
        let heard = Heard {
            index: b.index,
            info: b.info.clone(),
            route: b.route.clone(),
            seg_out: from,
            seg_frame: frame,
            warp: None,
        };
        self.queue_cue(from, CueEvent::Heard(heard));
    }

    /// The transition's turn, when it has one: getting ready near the
    /// window, B decoding ahead, the hand-over at the window's end, and
    /// B's return to a straight timeline. Returns whether it took the turn.
    pub(super) fn transition_turn(&mut self) -> Result<bool, Error> {
        let Some(phase) = self.transition.as_ref().map(|t| t.phase) else {
            return Ok(false);
        };
        let Some(written) = self.out.as_ref().map(|o| o.output.written()) else {
            return Ok(false);
        };
        match phase {
            Phase::Armed => {
                let Some(a) = &self.playing else {
                    return Ok(false);
                };
                let plan = Arc::clone(&self.transition.as_ref().expect("checked").plan);
                let now = seconds_written(a, written);
                if now > plan.start + 0.05 {
                    self.drop_transition(Some("the window has already passed"));
                } else if now >= plan.start - LEAD {
                    // A route switch under way (new stems, new settings) is
                    // the transition's to make now: it moves the track
                    // itself (one already meeting its old route ends first).
                    if self.switch.as_ref().is_some_and(|s| s.spliced) {
                        return Ok(false);
                    }
                    self.cancel_switch();
                    if let Err(why) = self.prepare_transition() {
                        self.drop_transition(Some(&why));
                    }
                }
                Ok(false)
            }
            Phase::Preparing | Phase::Mixing => {
                if phase == Phase::Mixing {
                    let t = self.transition.as_ref().expect("checked");
                    let end = t.window.map(|w| w.1);
                    let drained = self.playing.as_ref().is_some_and(|a| a.unwritten() == 0);
                    if end.is_some_and(|end| written >= end) && drained {
                        let a = self.playing.take();
                        self.hand_over_transition(a);
                        return Ok(true);
                    }
                }
                self.b_turn()
            }
            Phase::Settling => {
                let t = self.transition.as_ref().expect("checked");
                let Some(origin) = t.origin else {
                    return Ok(false);
                };
                let rate = self.out.as_ref().map_or(1, |o| o.info.format.rate);
                let clock = t.a0 + written.saturating_sub(origin) as f64 / f64::from(rate);
                if clock >= t.map.settled_at() + SETTLED {
                    self.settle_transition(written);
                    return Ok(true);
                }
                Ok(false)
            }
        }
    }

    /// Moves A onto a stems route with its fades (unless it is on one),
    /// and opens B ahead of the window.
    fn prepare_transition(&mut self) -> Result<(), String> {
        let (plan, map, a0, glide, map_a) = {
            let t = self.transition.as_ref().expect("checked");
            (
                Arc::clone(&t.plan),
                Arc::clone(&t.map),
                t.a0,
                t.glide,
                t.map_a.clone(),
            )
        };
        let open = self
            .out
            .as_ref()
            .map(|o| o.info.format)
            .ok_or("nothing is playing")?;
        let caps = self.capabilities().map_err(|e| e.to_string())?;
        let settings = self.route_settings();
        let separator = self.separator.is_none().then_some(whole_mix as fn() -> _);
        // The stems the user has on stay as they are (and a change of them
        // to come, at its time).
        let on = self.transition_mask();
        // A on its stems, faded by its lanes.
        let a = self.playing.as_ref().ok_or("nothing is playing")?;
        let a_facts = FileFacts::from(&*a.info);
        // On its stems already, and not read along an earlier plan's glide:
        // only its fades.
        if a.route.stage == Stage::Stems && map_a.is_none() && a.warp.is_none() {
            let a = self.playing.as_mut().expect("checked");
            a.gains = Some(Transition::gains_a(&plan, on.clone()));
        } else {
            // Onto its stems, or (to be read along the glide) onto a fresh
            // track on them.
            let route = if a.route.stage == Stage::Stems {
                a.route.clone()
            } else {
                stems_at(&a_facts, &caps, &settings, &open)
                    .ok_or("the output cannot take the stems of this track")?
            };
            self.begin_switch(route).map_err(|e| e.to_string())?;
            let switch = self.switch.as_mut().expect("just begun");
            switch.track.gains = Some(Transition::gains_a(&plan, on.clone()));
            switch.track.separator = separator;
            if let Some(map_a) = map_a {
                // From its first frame, where the glide is still none (the
                // switch meets the file a second on, on its grid).
                let s = switch.track.seg_frame as f64 / f64::from(switch.track.info.rate.max(1));
                switch.track.warp = Some(Box::new(Warping::new(map_a, s, s, plan.tempo)));
            }
        }
        // B from a little before the window, read along the map.
        let path = self.queue[plan.into].clone();
        let source = crate::source::Source::open(&path).map_err(|e| e.to_string())?;
        let info = source.info().clone();
        let route = stems_at(&FileFacts::from(&*info), &caps, &settings, &open)
            .ok_or("the output cannot take the next track's stems")?;
        let b_origin = (map.at(a0) - B_MARGIN).max(0.0);
        let frame = (b_origin * f64::from(info.rate)).floor() as u64;
        let mut b = Track::new(
            plan.into,
            info.clone(),
            route,
            source,
            self.segments.clone(),
        )
        .map_err(|e| e.to_string())?;
        b.start_at(frame).map_err(|e| e.to_string())?;
        b.pending = true;
        b.begin_segment(0, b.seg_frame, SegStart::Cold);
        b.gains = Some(Transition::gains_b(&plan, &map, glide, on));
        b.separator = separator;
        b.warp = Some(Box::new(Warping::new(
            Arc::clone(&map),
            a0,
            frame as f64 / f64::from(info.rate),
            plan.tempo,
        )));
        let t = self.transition.as_mut().expect("checked");
        t.b = Some(b);
        t.phase = Phase::Preparing;
        Ok(())
    }

    /// Offline ([`super::PlayerOptions::offline`]), the playing track holds
    /// within a second of the window while its move onto its stems is under
    /// way, and the move decodes meanwhile (on a device it could not wait:
    /// the window then goes on as it can). Returns whether it held.
    pub(super) fn hold_for_transition(&mut self) -> Result<bool, Error> {
        let moving = self.switch.as_ref().is_some_and(|s| !s.spliced && !s.done);
        if !self.offline || !moving {
            return Ok(false);
        }
        let (Some(t), Some(a), Some(out)) = (&self.transition, &self.playing, &self.out) else {
            return Ok(false);
        };
        let near = out.output.written() + u64::from(out.info.format.rate);
        if t.phase != Phase::Preparing || near < output_frame_of(a, t.plan.start) {
            return Ok(false);
        }
        let mut switch = self.switch.take().expect("checked");
        match self.produce(&mut switch.track, Duration::from_millis(10)) {
            Ok(Progress::Complete) => switch.done = true,
            Ok(Progress::Busy) => {}
            Err(error) if error.is_skippable() => {
                self.broke(&mut switch.track, error);
                switch.done = true;
            }
            Err(error) => return Err(error),
        }
        self.switch = Some(switch);
        self.try_splice()?;
        Ok(true)
    }

    /// B's turn to decode ahead: when the output holds half its buffer, is
    /// paused, or took nothing of the last write, as a switch's.
    fn b_turn(&mut self) -> Result<bool, Error> {
        let Some(out) = &self.out else {
            return Ok(false);
        };
        let buffered = out.output.written().saturating_sub(out.clock.position());
        let ring = out.info.ring_frames;
        let rate = out.info.format.rate;
        let turn = self.paused
            || self.offline
            || (ring > 0 && buffered.saturating_mul(2) >= ring)
            || self.stalled.is_some_and(|at| at.elapsed() < SWITCH_TURN);
        let t = self.transition.as_mut().expect("checked");
        let Some(b) = t.b.as_ref() else {
            return Ok(false);
        };
        // Far enough ahead: a quarter of a second past the frame being
        // written, once it is laid onto A's timeline (its stems are
        // separated well ahead, so it makes that much at once). Its output
        // is mixed with the stems on as it is made, so a stem switched on
        // a bar (a second ahead or more) is heard there on it too.
        let ahead = ((b.out_buf.len() - b.out_at) / 2) as u64;
        let written = out.output.written();
        let enough = match t.origin {
            Some(origin) => origin + t.b_next + ahead >= written + u64::from(rate) / 4,
            None => ahead >= 2 * u64::from(rate),
        };
        if !turn || enough {
            return Ok(false);
        }
        let mut b = t.b.take().expect("checked");
        let before = b.emitted;
        let result = self.produce(&mut b, Duration::ZERO);
        let waiting = b.emitted == before;
        if let Some(why) = b.stems_error.take() {
            super::drop_later(b);
            self.drop_transition(Some(&format!("the next track's stems failed: {why}")));
            return Ok(true);
        }
        match result {
            Ok(_) => {}
            Err(error) if error.is_skippable() => {
                super::drop_later(b);
                self.drop_transition(Some(&format!("the next track failed: {error}")));
                return Ok(true);
            }
            Err(error) => return Err(error),
        }
        if let Some(t) = self.transition.as_mut() {
            t.b = Some(b);
        }
        Ok(!waiting)
    }

    /// Adds B's output to the piece of A's that was just made (it starts
    /// at the frame being written), cutting A's at the window's end.
    pub(super) fn mix_transition(&mut self, a: &mut Track) -> Result<(), Error> {
        let Some(out) = &self.out else {
            return Ok(());
        };
        let (written, channels, rate) = (
            out.output.written(),
            usize::from(out.info.format.channels),
            out.info.format.rate,
        );
        let Some(t) = self.transition.as_mut() else {
            return Ok(());
        };
        if !matches!(t.phase, Phase::Preparing | Phase::Mixing) || a.out_at != 0 {
            return Ok(());
        }
        // The window and B's timeline on A's (straight while it plays on).
        let (w0, w1) = match t.window {
            Some(w) => w,
            None => {
                let w0 = output_frame_of(a, t.plan.start);
                // A covers its window in the glide's length of output.
                let w1 = if t.glide.is_identity() {
                    output_frame_of(a, t.plan.start + t.plan.length)
                } else {
                    w0 + (t.glide.length() * f64::from(rate)).round() as u64
                };
                let lead = ((t.plan.start - t.a0) * f64::from(rate)).round() as u64;
                t.origin = Some(w0.saturating_sub(lead));
                (w0, w1)
            }
        };
        let frames = (a.out_buf.len() / channels) as u64;
        let (from, to) = (written.max(t.mixed_to), written + frames);
        if to <= w0 {
            return Ok(());
        }
        if t.phase == Phase::Preparing {
            t.phase = Phase::Mixing;
            t.window = Some((w0, w1));
            let unglided = t.map_a.is_some() && a.warp.is_none();
            self.queue_cue(
                w0,
                CueEvent::Transition(TransitionState::Started { output_frame: w0 }),
            );
            if unglided {
                // The next track is laid onto the glide, which the playing
                // one cannot follow: it goes on alone, as it is (its move
                // onto the gliding track dropped), rather than off its
                // beats.
                a.gains = None;
                self.cancel_switch();
                self.drop_transition(Some(
                    "the playing track was not on its stems in time to change its tempo",
                ));
                return Ok(());
            }
        }
        let t = self.transition.as_mut().expect("checked");
        let origin = t.origin.expect("set with the window");
        let first = from.max(w0);
        let last = to.min(w1);
        if a.gains.is_none() {
            // A is not on its stems yet (the move to them has not met the
            // window in time): it fades as a whole, by its lanes' mean.
            let plan = Arc::clone(&t.plan);
            for f in first..last {
                let gains = plan.gains(Side::From, plan.share(seconds_written(a, f)));
                let g = gains.iter().sum::<f64>() / gains.len() as f64;
                let at = ((f - written) as usize) * channels;
                for s in &mut a.out_buf[at..at + channels] {
                    *s = (f64::from(*s) * g).round() as i32;
                }
            }
        }
        let t = self.transition.as_mut().expect("checked");
        let mut b = t.b.take().expect("B is there while mixing");
        let mut f = first;
        let mut spent = Duration::ZERO;
        while f < last {
            let j = f - origin;
            // B's frames before this one are not heard.
            let t = self.transition.as_mut().expect("checked");
            if t.b_next < j {
                let have = ((b.out_buf.len() - b.out_at) / channels) as u64;
                let skip = (j - t.b_next).min(have);
                b.out_at += skip as usize * channels;
                t.b_next += skip;
            }
            let have = ((b.out_buf.len() - b.out_at) / channels) as u64;
            if t.b_next == j && have > 0 {
                let n = have.min(last - f);
                let a_at = ((f - written) as usize) * channels;
                for k in 0..n as usize * channels {
                    let sum = i64::from(a.out_buf[a_at + k]) + i64::from(b.out_buf[b.out_at + k]);
                    let (kept, clipped) = soft_limit(sum);
                    a.mix_clips += u64::from(clipped);
                    a.out_buf[a_at + k] = kept;
                }
                b.out_at += n as usize * channels;
                t.b_next += n;
                f += n;
                continue;
            }
            // B must make more; it may wait for its stems, a little.
            if b.out_at > 1 << 16 {
                b.out_buf.drain(..b.out_at);
                b.out_at = 0;
            }
            if spent >= MIX_PATIENCE && !self.offline {
                // Behind: the rest of this piece goes without it (its
                // frames are skipped when they come, so it stays in time).
                break;
            }
            let before = b.emitted;
            match self.produce(&mut b, Duration::from_millis(10)) {
                Ok(Progress::Complete) => {
                    if b.emitted == before {
                        break;
                    }
                }
                Ok(Progress::Busy) => {}
                Err(error) if error.is_skippable() => break,
                Err(error) => return Err(error),
            }
            if b.emitted == before {
                spent += Duration::from_millis(10);
            }
        }
        let t = self.transition.as_mut().expect("checked");
        t.mixed_to = to.min(w1);
        t.b = Some(b);
        if to > w1 {
            // A ends at the window's end; B goes on from there.
            a.out_buf.truncate(((w1 - written) as usize) * channels);
            a.emitted = a.emitted.saturating_sub(to - w1);
        }
        Ok(())
    }

    /// The window's end is written (or A ended in it): B plays on alone,
    /// and `a` goes.
    pub(super) fn hand_over_transition(&mut self, a: Option<Track>) {
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        let Some(mut b) = t.b.take() else {
            return;
        };
        let written = self.out.as_ref().map_or(0, |o| o.output.written());
        let rate = self.out.as_ref().map_or(1, |o| o.info.format.rate);
        let origin = t.origin.unwrap_or(written);
        let end = written;
        // B's frames before the end are not heard.
        let channels = usize::from(b.route.channels);
        let j = end.saturating_sub(origin);
        if t.b_next < j {
            let have = ((b.out_buf.len() - b.out_at) / channels) as u64;
            let skip = (j - t.b_next).min(have);
            b.out_at += skip as usize * channels;
            t.b_next += skip;
        }
        b.pending = false;
        b.begin_segment(origin, b.seg_frame, SegStart::Continued);
        let warp = HeardWarp {
            map: Arc::clone(&t.map),
            origin,
            a0: t.a0,
            out_rate: rate,
        };
        let seconds = t.map.at(t.a0 + j as f64 / f64::from(rate));
        let from_frame = (seconds * f64::from(b.info.rate)).round().max(0.0) as u64;
        // A move of A onto its stems that has not met it yet goes with it.
        if let Some(switch) = self.switch.take() {
            super::drop_later(switch.track);
        }
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        t.phase = Phase::Settling;
        t.origin = Some(origin);
        if let Some(a) = &a {
            // A's cues past the end are not heard.
            self.cues
                .retain(|c| c.at <= end || !matches!(c.event, CueEvent::Clipped { .. }));
            self.queue_cue(end, CueEvent::Ended { index: a.index });
        }
        // B's held cues from the end on.
        b.held.retain(|c| c.at + origin >= end);
        let out = self.out.as_ref().expect("an output is open");
        let start = TrackStart {
            index: b.index,
            info: b.info.clone(),
            route: b.route.clone(),
            output: out.info.clone(),
            device: self.target.name(),
            bit_perfect: false,
            output_frame: end,
            from_frame,
        };
        let heard = Heard {
            index: b.index,
            info: b.info.clone(),
            route: b.route.clone(),
            seg_out: origin,
            seg_frame: b.seg_frame,
            warp: Some(Arc::new(warp)),
        };
        self.queue_cue(end, CueEvent::Started(Box::new(start), heard));
        self.release(&mut b);
        self.playing = Some(b);
        if let Some(a) = a {
            super::drop_later(a);
        }
    }

    /// B is back at its own tempo: its timeline runs straight from the
    /// frame being written, and it moves back onto its own route.
    fn settle_transition(&mut self, written: u64) {
        let Some(t) = self.transition.take() else {
            return;
        };
        self.straighten(&t, written);
        self.queue_cue(written, CueEvent::Transition(TransitionState::Finished));
        // Back onto its own route, without a break where the format allows.
        self.reroute();
    }

    /// A ended (`a`): inside the window, B takes over at once (and `None`
    /// comes back); before it, the transition is off and `a` comes back
    /// to end as usual.
    pub(super) fn transition_takes_over(&mut self, a: Track) -> Option<Track> {
        match self.transition.as_ref().map(|t| t.phase) {
            Some(Phase::Mixing) => {
                if let Some(t) = self.transition.as_mut() {
                    if let Some((w0, _)) = t.window {
                        let written = self.out.as_ref().map_or(w0, |o| o.output.written());
                        t.window = Some((w0, written.max(w0)));
                    }
                }
                self.hand_over_transition(Some(a));
                None
            }
            Some(Phase::Settling) | None => Some(a),
            Some(_) => {
                self.forget_transition(Some("the track ended before the window"));
                Some(a)
            }
        }
    }
}

impl Engine {
    /// The queue was reordered ([`super::queue`]): a transition follows
    /// its tracks while the next one is still the one after the playing
    /// one, and is called off otherwise.
    pub(super) fn reindex_transition(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        match (map(t.plan.from), map(t.plan.into)) {
            (Some(from), Some(into)) if into == from + 1 => {
                if (from, into) != (t.plan.from, t.plan.into) {
                    t.plan = Arc::new(TransitionPlan {
                        from,
                        into,
                        ..(*t.plan).clone()
                    });
                    if let Some(b) = &mut t.b {
                        b.index = into;
                    }
                }
            }
            _ => self.drop_transition(Some("the queue changed")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mix_bends_its_peaks_under_full_scale() {
        // Up to the knee, as added.
        assert_eq!(soft_limit(1_000_000), (1_000_000, false));
        assert_eq!(soft_limit(-7_000_000), (-7_000_000, false));
        // Over it, under full scale and in order, with no clip.
        let mut last = KNEE as i32;
        for x in [7_600_000_i64, 8_388_607, 9_000_000, 12_000_000, 16_777_214] {
            let (y, clipped) = soft_limit(x);
            assert!(!clipped && y > last && f64::from(y) <= FULL, "{x} {y}");
            assert_eq!(soft_limit(-x).0, -y);
            last = y;
        }
        // Smooth at the knee.
        let k = KNEE as i64;
        assert!((soft_limit(k + 1000).0 as i64 - (k + 1000)).abs() <= 1);
    }
}
