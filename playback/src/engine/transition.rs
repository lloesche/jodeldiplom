//! The engine's side of a transition ([`crate::transition`]): its plan
//! compiled onto the desk, which mixes the playing track's head (A) and a
//! second head ([`super::heads`]) for the next track (B).
//!
//! Armed, it waits until A is [`LEAD`] before the window. Then:
//!
//! - B opens on the second head at the open output's format (its route is
//!   chosen for that format alone, so the device never opens again), read
//!   along the plan's map from its first frame ([`super::along`]), which
//!   the desk link puts on A's stream frame `origin`,
//!   [`crate::transition::NEXT_LEAD`] before the window; it writes ahead on
//!   turns of its own;
//! - with a glide, A reads along the glide;
//! - the plan's lanes become automation of each head's input, on its own
//!   stream frames ([`compile`]): each stem's envelope its part's curve,
//!   their mean main's (for a head heard without parts), A whole and B
//!   silent before the window.
//!
//! A is written up to the window's end and no further; there B is made the
//! desk's Lead, A's output plays out what it holds and closes, and the
//! heads swap: B plays on, settles off its map back onto its own route at
//! the same format, and the transition is finished. The cue clock is B's
//! from the hand-over (its frames count from its first). The output format
//! stays the open one until playback stops or a track is picked.
//!
//! Pausing pauses A, and the link holds B. `Next` in the window fades A out
//! from where the desk mixes it and puts B at its end values, and hands
//! over once A is written to its fade's end. Called off in the window (the
//! queue changed, B failed), B fades out and A comes back whole. Moving
//! playback elsewhere drops B; new settings wait for the end (and the
//! format is no longer held then).

use std::sync::Arc;
use std::time::Duration;

use jodel_output::mixer::{Automation, Fader, GainCurve, Role};

use super::api::{Event, TrackStart};
use super::track::{drop_later, frames_of, rescale, Cue, CueEvent, HeardWarp, Track};
use super::Engine;
use crate::error::Error;
use crate::rate::Settings;
use crate::source::Source;
use crate::stems::{Stem, LANES};
use crate::transition::{Envelope, Side, TransitionPlan, TransitionState, NEXT_LEAD};
use crate::warp::{Glide, TimeMap};

/// Seconds before the window that the tracks start to be prepared.
const LEAD: f64 = 4.0;

/// How long a track takes to fade out when a mix is cut short (`Next`, or
/// called off in its window).
const CUT: Duration = Duration::from_millis(30);

/// Seconds of output between the points of a curve that the plan's
/// envelopes do not give exactly (through a glide, or a mean of lanes).
const STEP: f64 = 0.01;

/// A share of the window just before a point of an envelope.
const EPSILON: f64 = 1e-12;

/// Where a transition is.
enum Phase {
    /// Waiting for the playing track to come near the window.
    Armed,
    /// B writes ahead, and the window is written once it is mixing.
    Prepared(Prepared),
    /// B plays alone, on its way back to its own route. `cut`: `Next` cut
    /// the mix short (it was called off then, and its settling says
    /// nothing).
    Settling { cut: bool },
}

/// A transition's tracks as prepared for its window.
struct Prepared {
    /// The window's first and last stream frames of A.
    window: (u64, u64),
    /// A's stream frame that B's frame 0 is linked to.
    origin: u64,
    /// The heads' automation (A's, B's).
    curves: [Curves; 2],
    /// The window is being written.
    mixing: bool,
    /// `Next` cut the mix short, A ending at this stream frame, its fade
    /// out's end: it hands over there.
    cut: Option<u64>,
}

/// A head's automation: main's curve and each part's, in [`LANES`]' order.
struct Curves {
    main: GainCurve,
    parts: Vec<GainCurve>,
}

impl Curves {
    /// The automation of these curves.
    fn automation(&self) -> Automation {
        self.with(Automation::new(), |c| c.clone())
    }

    /// Each curve made by `curve`, as an automation on top of `base`.
    fn with(&self, base: Automation, curve: impl Fn(&GainCurve) -> GainCurve) -> Automation {
        let main = base.main(curve(&self.main));
        self.parts
            .iter()
            .enumerate()
            .fold(main, |a, (k, c)| a.part(k, curve(c)))
    }
}

/// A gain held from stream frame `frame` on.
fn held(frame: u64, gain: f64) -> GainCurve {
    GainCurve::new(&[(frame, gain)]).expect("one finite point")
}

/// A transition under way.
pub(super) struct Transition {
    plan: Arc<TransitionPlan>,
    /// B's time along A's output's (A's own time before the window, A's
    /// time through the glide's inverse in it).
    map: Arc<TimeMap>,
    /// A's time along its output's through the window.
    glide: Glide,
    /// The settings when it was armed (others since: the format is not held
    /// after it).
    settings: Settings,
    phase: Phase,
}

impl Transition {
    /// Its tracks as prepared, from then until the hand-over.
    fn prepared(&self) -> Option<&Prepared> {
        match &self.phase {
            Phase::Prepared(p) => Some(p),
            _ => None,
        }
    }

    fn prepared_mut(&mut self) -> Option<&mut Prepared> {
        match &mut self.phase {
            Phase::Prepared(p) => Some(p),
            _ => None,
        }
    }

    /// Whether its window is being written (and not handed over yet).
    fn mixing(&self) -> bool {
        self.prepared().is_some_and(|p| p.mixing)
    }
}

/// The output frame at which A's time `seconds` is heard, on its timeline.
fn output_frame_of(track: &Track, seconds: f64) -> u64 {
    let frame = (seconds * f64::from(track.info.rate)).round().max(0.0) as u64;
    let since = frame.saturating_sub(track.seg_frame);
    track.seg_out + rescale(since, track.route.input_rate, track.route.output_rate)
}

/// A's time at the output frame being written.
fn seconds_written(track: &Track, written: u64) -> f64 {
    let since = written.saturating_sub(track.seg_out);
    (track.seg_frame as f64
        + since as f64 * f64::from(track.route.input_rate) / f64::from(track.route.output_rate))
        / f64::from(track.info.rate.max(1))
}

/// The plan's lanes as each head's automation (A's, then B's), at `rate`:
/// with the window from A's stream frame `w0`, and B's frame 0 on A's
/// `origin`. Each part's curve is its stem's envelope at A's time heard on
/// each frame (through the glide's inverse, A's time along the output's),
/// main's their mean; before the window A is whole and B silent.
fn compile(plan: &TransitionPlan, glide: &Glide, w0: u64, origin: u64, rate: u32) -> [Curves; 2] {
    let rate = f64::from(rate);
    // A's stream frame (fractional) where it is at its time `a`.
    let frame_of = |a: f64| w0 as f64 + (glide.inverse(a) - plan.start) * rate;
    let head = |side: Side, before: f64, shift: u64| {
        let lanes = LANES.map(|s: Stem| plan.lane(side, s));
        let lane_curve = |envelopes: &[&Envelope]| {
            let differ = envelopes.windows(2).any(|w| w[0] != w[1]);
            curve(
                plan,
                envelopes,
                before,
                !glide.is_identity() || differ,
                |a| (frame_of(a) - shift as f64).round().max(0.0) as u64,
            )
        };
        Curves {
            main: lane_curve(&lanes),
            parts: lanes.iter().map(|&e| lane_curve(&[e])).collect(),
        }
    };
    [head(Side::From, 1.0, 0), head(Side::Into, 0.0, origin)]
}

/// The curve of the mean of `envelopes` over the window and after it (at
/// `before` before it), on the frames `frame_of` puts A's times on: a point
/// on each of theirs (two on a cut), and every [`STEP`] between when
/// `dense`.
fn curve(
    plan: &TransitionPlan,
    envelopes: &[&Envelope],
    before: f64,
    dense: bool,
    frame_of: impl Fn(f64) -> u64,
) -> GainCurve {
    let mut shares: Vec<f64> = envelopes
        .iter()
        .flat_map(|e| e.points.iter().map(|p| p.at))
        .filter(|&s| s > 0.0)
        .collect();
    let last = shares.iter().copied().fold(0.0, f64::max);
    if dense && plan.length > 0.0 {
        let step = STEP / plan.length;
        shares.extend((1..).map(|k| k as f64 * step).take_while(|&s| s < last));
    }
    shares.sort_by(f64::total_cmp);
    shares.dedup();
    // The mean just before `share` and just after it.
    let mean = |share: f64| {
        let n = envelopes.len().max(1) as f64;
        let (left, right) = envelopes
            .iter()
            .map(|e| around(e, share))
            .fold((0.0, 0.0), |(l, r), (a, b)| (l + a, r + b));
        (left / n, right / n)
    };
    let at = |share: f64| frame_of(plan.start + share * plan.length);
    // Two points on a frame are a cut (smoothed): only where the gain cuts.
    let near = |x: f64, y: f64| (x - y).abs() <= 1e-9;
    let mut points = vec![(at(0.0), before)];
    let (_, first) = mean(0.0);
    if !near(first, before) {
        points.push((at(0.0), first));
    }
    for share in shares {
        let (left, right) = mean(share);
        if !near(left, right) {
            points.push((at(share), left));
        }
        points.push((at(share), right));
    }
    GainCurve::new(&points).expect("frames in order, gains finite")
}

/// `envelope`'s gain just before `share` and just after it: two on a cut
/// there, else the same.
fn around(envelope: &Envelope, share: f64) -> (f64, f64) {
    let at = envelope.gain(share);
    let points = &envelope.points;
    if !points.iter().any(|p| p.at == share && p.cut) {
        return (at, at);
    }
    let after = points
        .iter()
        .rev()
        .find(|p| p.at == share)
        .map_or(at, |p| p.gain);
    (envelope.gain(share - EPSILON), after)
}

impl Engine {
    /// [`crate::Command::Transition`]: takes the plan if it fits what is
    /// playing.
    pub(super) fn arm_transition(&mut self, plan: TransitionPlan) {
        // One heard already is not replaced (it would be cut off).
        if self.transition_heard() {
            return;
        }
        // Another plan for one armed or preparing: its next track goes, and
        // it prepares again from where the tracks are.
        if self.transition.is_some() {
            self.drop_transition(None);
        }
        let playing = self.head.playing.as_ref().map(|t| t.index);
        let why = if self.desk.is_none() {
            Some("needs a desk")
        } else if plan.into != plan.from + 1 || plan.into >= self.queue.len() {
            Some("there is no next track to go into")
        } else if plan.length.is_nan() || plan.length <= 0.0 {
            Some("the window is empty")
        } else if playing != Some(plan.from) {
            Some("that track is not the one playing")
        } else if self.along_under_way() {
            Some("a track is read along a time map")
        } else {
            None
        };
        if let Some(why) = why {
            return self.emit(Event::Transition(TransitionState::Cancelled {
                reason: why.into(),
            }));
        }
        // B on A's beats as they are heard: along A's output's time, which
        // runs as A's up to the window and A along the glide in it.
        let beats = plan.time_map();
        let glide = plan.glide_of(&beats);
        self.transition = Some(Transition {
            map: Arc::new(beats.through_glide(&glide)),
            plan: Arc::new(plan),
            glide,
            settings: self.settings,
            phase: Phase::Armed,
        });
        self.emit(Event::Transition(TransitionState::Armed));
    }

    /// Drops the transition, saying why when there is a reason (and it
    /// had not finished), and puts what it changed back: in its window B
    /// fades out over [`CUT`] (and closes then) and A comes back whole;
    /// before it B goes and A loses its automation; A leaves its glide, and
    /// is routed as the settings choose. One `Next` cut short hands over
    /// now instead (A is faded out already): B plays on.
    pub(super) fn drop_transition(&mut self, reason: Option<&str>) {
        let t = self.transition.as_ref();
        if let Some(cut) = t.and_then(Transition::prepared).and_then(|p| p.cut) {
            if let Some(a) = self.head.playing.take() {
                let written = self.head.out.as_ref().map_or(cut, |o| o.output.written());
                self.hand_over_transition(a, cut.min(written));
            }
        } else if t.is_some_and(Transition::mixing) {
            self.fade_next_out();
        }
        let Some(t) = self.forget_transition(reason) else {
            return;
        };
        let along = self.head.along_at.is_some()
            || self
                .head
                .playing
                .as_ref()
                .is_some_and(|a| a.along.is_some());
        if !matches!(t.phase, Phase::Settling { .. }) && along {
            self.leave_map();
        }
        self.reroute();
    }

    /// Playback moves elsewhere (a seek, a new timeline): the transition is
    /// forgotten, and the output a hand-over left playing out what it holds
    /// closes at once.
    pub(super) fn abandon_transition(&mut self, reason: &str) {
        self.forget_transition(Some(reason));
        self.cut_parked();
    }

    /// Drops the transition, saying why when there is a reason and it had
    /// not finished, without routing anything again (playback is moving
    /// elsewhere anyway): before the hand-over B (unless it is fading out
    /// already) goes and A loses its automation; after it, B loses its
    /// automation. Returns what it was.
    pub(super) fn forget_transition(&mut self, reason: Option<&str>) -> Option<Transition> {
        let t = self.transition.take()?;
        // What it queued to be said as it is heard (its window started,
        // say) is not said after it is gone.
        self.drop_transition_cues();
        let on_air = self.faders().map(|(f, _)| f.clone());
        let called_off = match &t.phase {
            Phase::Settling { cut } => {
                on_air.inspect(Fader::clear_automation);
                // Handed over already: it is done, as far as anyone can
                // hear (one cut short by `Next` was called off then).
                if !cut {
                    self.emit(Event::Transition(TransitionState::Finished));
                }
                return Some(t);
            }
            Phase::Prepared(p) => p.cut.is_some(),
            Phase::Armed => false,
        };
        if self.parked_track().is_some() {
            self.drop_parked();
            on_air.inspect(Fader::clear_automation);
        }
        if let Some(reason) = reason.filter(|_| !called_off) {
            self.emit(Event::Transition(TransitionState::Cancelled {
                reason: reason.into(),
            }));
        }
        Some(t)
    }

    /// The cues a transition queued to be said as they are heard go.
    fn drop_transition_cues(&mut self) {
        self.cues
            .retain(|c| !matches!(c.event, CueEvent::Transition(_)));
    }

    /// Where each head's input is mixed next, on its stream frames (where
    /// a change of its automation is heard from), and the output's rate.
    fn mix_fronts(&self) -> Option<(u64, u64, u32)> {
        let (fa, fb) = self.faders()?;
        let out = self.head.out.as_ref()?;
        let a = fa.mixed_to().unwrap_or_else(|| out.clock.position());
        Some((a, fb.mixed_to().unwrap_or(0), out.info.format.rate))
    }

    /// In the window, the transition is dropped: from where each is mixed
    /// next, B fades out over [`CUT`] and A comes back whole (each curve
    /// from its gain there to one), and B's head plays out what it has and
    /// closes.
    fn fade_next_out(&mut self) {
        let (Some(t), Some((a, b, rate))) = (self.transition.as_ref(), self.mix_fronts()) else {
            return;
        };
        let Some(Prepared { curves, .. }) = t.prepared() else {
            return;
        };
        let fade = frames_of(CUT, rate);
        let back = curves[0].with(Automation::new(), |c| {
            GainCurve::new(&[(a, c.gain(a, rate)), (a + fade, 1.0)]).expect("in order")
        });
        let out = GainCurve::new(&[(b, 1.0), (b + fade, 0.0)]).expect("in order");
        let gone = curves[1].with(Automation::new().input(out), |c| held(b, c.gain(b, rate)));
        if let Some((fa, fb)) = self.faders() {
            let _ = fa.automate(back);
            let _ = fb.automate(gone);
        }
        self.retire_parked();
    }

    /// [`super::Command::Next`] while a transition is heard. Mixing: from
    /// where the desk mixes A next it fades out over [`CUT`] and B goes to
    /// its end values (each automation replaced, which the desk smooths),
    /// A is handed over from once it is written to its fade's end, the mix
    /// called off now (cut to the next track). Settling (B plays alone
    /// already): the track after B. Whether it did either.
    pub(super) fn cut_to_next(&mut self) -> bool {
        let Some(t) = self.transition.as_ref() else {
            return false;
        };
        let p = match &t.phase {
            Phase::Settling { .. } => {
                let next = t.plan.into + 1;
                self.pick();
                self.jump(next, 0.0);
                return true;
            }
            // Cut already: it ends as soon as it can.
            Phase::Prepared(p) if p.cut.is_some() => return true,
            Phase::Prepared(p) if p.mixing => p,
            Phase::Prepared(_) | Phase::Armed => return false,
        };
        let Some((front, _, rate)) = self.mix_fronts() else {
            return false;
        };
        let end = front.max(p.origin);
        let fade = frames_of(CUT, rate);
        let out_a = GainCurve::new(&[(end, 1.0), (end + fade, 0.0)]).expect("in order");
        let a = p.curves[0].with(Automation::new().input(out_a), |c| {
            held(end, c.gain(end, rate))
        });
        let b = p.curves[1].with(Automation::new(), |c| {
            held(0, c.points().last().map_or(1.0, |p| p.1))
        });
        if let Some((fa, fb)) = self.faders() {
            let _ = fa.automate(a);
            let _ = fb.automate(b);
        }
        // A is written on to where its fade ends (no further than the
        // window), and handed over from there; nothing more of the window
        // is said.
        let written = self.head.out.as_ref().map_or(0, |o| o.output.written());
        if let Some(p) = self.transition.as_mut().and_then(Transition::prepared_mut) {
            p.cut = Some(end + fade);
            p.window.1 = p.window.1.min((end + fade).max(written));
        }
        self.drop_transition_cues();
        self.emit(Event::Transition(TransitionState::Cancelled {
            reason: "cut to the next track".into(),
        }));
        // Paused, the next track plays (as next does).
        self.resume();
        true
    }

    /// Whether a transition is heard (mixing, or the next track settling
    /// after it).
    pub(super) fn transition_heard(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| t.mixing() || matches!(t.phase, Phase::Settling { .. }))
    }

    /// Whether a transition is under way past being armed (its tracks
    /// prepared, mixed or settling), so new settings wait for its end
    /// rather than call it off.
    pub(super) fn transition_running(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| !matches!(t.phase, Phase::Armed))
    }

    /// Whether a transition has its tracks where it needs them (prepared or
    /// mixed): the playing track is not moved to another route.
    pub(super) fn transition_holds_route(&self) -> bool {
        self.transition
            .as_ref()
            .is_some_and(|t| t.prepared().is_some())
    }

    /// Queues a cue at `at`, or after the last one queued.
    fn queue_cue(&mut self, at: u64, event: CueEvent) {
        let at = at.max(self.cues.back().map_or(0, |c| c.at));
        self.cues.push_back(Cue { at, event });
    }

    /// The sample index the playing `track` (A) writes no further than
    /// while its transition is prepared or mixed: its window's end.
    pub(super) fn transition_limit(&self, track: &Track) -> Option<usize> {
        let t = self.transition.as_ref()?;
        let end = t.prepared()?.window.1;
        if track.index != t.plan.from || track.pending {
            return None;
        }
        let written = self.head.out.as_ref()?.output.written();
        let ch = usize::from(track.route.channels);
        Some(track.out_at + end.saturating_sub(written) as usize * ch)
    }

    /// The transition's turn, when it has one: getting ready near the
    /// window, the window's start, B's turns, the hand-over at the
    /// window's end, and B back on its own route. Returns whether it took
    /// the turn.
    pub(super) fn transition_turn(&mut self) -> Result<bool, Error> {
        let Some(t) = self.transition.as_ref() else {
            return Ok(false);
        };
        let Some(written) = self.head.out.as_ref().map(|o| o.output.written()) else {
            return Ok(false);
        };
        let start = t.plan.start;
        match &t.phase {
            Phase::Armed => {
                let Some(a) = &self.head.playing else {
                    return Ok(false);
                };
                let now = seconds_written(a, written);
                if now > start + 0.05 {
                    self.drop_transition(Some("the window has already passed"));
                } else if now >= start - LEAD {
                    // A route switch under way (new settings) is dropped:
                    // the transition has the track now (one already meeting
                    // its old route ends first).
                    if self.head.switch.as_ref().is_some_and(|s| s.spliced) {
                        return Ok(false);
                    }
                    self.cancel_switch();
                    if let Err(why) = self.prepare_transition() {
                        self.drop_transition(Some(&why));
                    }
                }
                Ok(false)
            }
            Phase::Prepared(p) => {
                let ((w0, w1), origin, cut, mixing) = (p.window, p.origin, p.cut, p.mixing);
                if !mixing && written >= w0 {
                    let started = TransitionState::Started {
                        head: self.head.id,
                        output_frame: w0,
                        origin,
                    };
                    self.queue_cue(w0, CueEvent::Transition(started));
                    if let Some(p) = self.transition.as_mut().and_then(Transition::prepared_mut) {
                        p.mixing = true;
                    }
                }
                if let Some(a) = self.head.playing.take_if(|_| written >= w1) {
                    self.hand_over_transition(a, cut.map_or(w1, |at| at.min(w1)));
                    return Ok(true);
                }
                match self.parked_turn() {
                    Ok(did) => Ok(did),
                    Err(why) => {
                        self.drop_transition(Some(&format!("the next track failed: {why}")));
                        Ok(true)
                    }
                }
            }
            Phase::Settling { .. } => {
                let back = self
                    .head
                    .playing
                    .as_ref()
                    .is_some_and(|b| b.along.is_none());
                if back && self.head.switch.is_none() {
                    self.settle_transition(written);
                }
                Ok(false)
            }
        }
    }

    /// Prepares the window: B opened on the second head, read along the
    /// map from its first frame and linked to A's frame `origin`; A along
    /// the glide; both heads automated. Why it cannot be, if it cannot.
    fn prepare_transition(&mut self) -> Result<(), String> {
        let t = self.transition.as_ref().expect("armed");
        let (plan, map, glide) = (Arc::clone(&t.plan), Arc::clone(&t.map), t.glide);
        let out = self.head.out.as_ref().ok_or("nothing is playing")?;
        let (open, written, rate) = (out.info.format, out.output.written(), out.info.format.rate);
        // What refuses at once, before anything of B is opened.
        if !self.parked_takes_lanes() {
            return Err("the next track's head takes no stems".into());
        }
        let a = self.head.playing.as_ref().ok_or("nothing is playing")?;
        if let Some(why) = self.along_refusal(a).filter(|_| !glide.is_identity()) {
            return Err(format!("the playing track cannot change its tempo: {why}"));
        }
        let w0 = output_frame_of(a, plan.start);
        let w1 = if glide.is_identity() {
            output_frame_of(a, plan.start + plan.length)
        } else {
            w0 + (glide.length() * f64::from(rate)).round() as u64
        };
        let origin = w0.saturating_sub((NEXT_LEAD * f64::from(rate)).round() as u64);
        // B, routed for the open format alone.
        let source = Source::open(&self.queue[plan.into]).map_err(|e| e.to_string())?;
        let info = source.info().clone();
        if info.channels > 2 {
            return Err("the next track has more than two channels".into());
        }
        let route = self
            .route_at(&info, open)
            .map_err(|none| format!("the next track does not play at {open} ({none})"))?;
        let b = Track::new(plan.into, info, route, source, self.segments.clone())
            .map_err(|e| e.to_string())?;
        self.open_parked(b)
            .map_err(|e| format!("the next track's output did not open: {e}"))?;
        // Its frame 0 at A's time there (at `origin`), along the map, heard
        // from the window on.
        let clock = self.head.out.as_ref().expect("checked").clock.clone();
        let heard = HeardWarp {
            map,
            origin: 0,
            a0: plan.start - (w0 - origin) as f64 / f64::from(rate),
            out_rate: rate,
        };
        self.with_parked(|e| {
            let mut b = e.head.playing.take().expect("just opened");
            e.start_along(&mut b, heard, plan.tempo, (clock, w0));
            e.head.playing = Some(b);
        });
        let (fa, fb) = self.faders().expect("on a desk");
        let (fa, fb) = (fa.clone(), fb.clone());
        fb.follow(&fa, origin).map_err(|e| e.to_string())?;
        if !glide.is_identity() {
            // A along the glide from the frame written next, the glide's
            // time at its stream frame 0 (none before the window).
            let heard = HeardWarp {
                map: Arc::new(glide.map()),
                origin: 0,
                a0: plan.start - w0 as f64 / f64::from(rate),
                out_rate: rate,
            };
            self.read_along_at(heard, written, plan.tempo)
                .map_err(|why| format!("the playing track cannot change its tempo: {why}"))?;
        }
        let curves = compile(&plan, &glide, w0, origin, rate);
        fa.automate(curves[0].automation())
            .and_then(|()| fb.automate(curves[1].automation()))
            .map_err(|e| e.to_string())?;
        let t = self.transition.as_mut().expect("armed");
        t.phase = Phase::Prepared(Prepared {
            window: (w0, w1),
            origin,
            curves,
            mixing: false,
            cut: None,
        });
        Ok(())
    }

    /// A (`a`) is written to `end` (its stream frame; it ended there, or
    /// the window did, or `Next` cut it there): B is made the desk's Lead
    /// before A is a guest again, A's output plays out what it holds, and
    /// the heads swap. A's end and B's start are cued on one frame, on B's
    /// clock from here (A's frames less `origin`).
    pub(super) fn hand_over_transition(&mut self, a: Track, end: u64) {
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        let Some(&Prepared { origin, cut, .. }) = t.prepared() else {
            return;
        };
        t.phase = Phase::Settling { cut: cut.is_some() };
        let from = t.plan.from;
        // The format stays where it is, unless the settings changed.
        let pin = t.settings == self.settings;
        if let Some((fa, fb)) = self.faders() {
            let _ = fb.set_role(Role::Lead);
            let _ = fa.set_role(Role::Guest);
        }
        let open = self.head.out.as_ref().map(|o| o.info.format);
        if let Some(out) = self.head.out.as_mut() {
            let _ = out.output.drain(Some(Duration::ZERO));
        }
        // A's cues past the end are not heard; it ends there.
        self.cues
            .retain(|c| c.at <= end || !matches!(c.event, CueEvent::Clipped { .. }));
        self.queue_cue(end, CueEvent::Ended { index: from });
        self.cancel_switch();
        self.head.along_at = None;
        self.head.pending_jump = None;
        self.drop_jump_fill();
        drop_later(a);
        self.swap_heads();
        // B's clock from here: the cues, and what is heard until B is.
        for cue in &mut self.cues {
            cue.at = cue.at.saturating_sub(origin);
        }
        self.heard = self.heard.as_ref().map(|h| h.rebased(origin));
        self.last_report = None;
        self.pinned = open.filter(|_| pin);
        let Some(mut b) = self.head.playing.take() else {
            return;
        };
        let end = end.saturating_sub(origin);
        b.pending = false;
        b.held.retain(|c| c.at >= end);
        // On air: its clock is its own now, A's no longer.
        if let Some(along) = b.along.as_mut() {
            along.heard_from = None;
        }
        let heard = b.heard();
        let start = TrackStart {
            index: b.index,
            head: self.head.id,
            info: b.info.clone(),
            route: b.route.clone(),
            output: self.head.out.as_ref().expect("B's output").info.clone(),
            device: self.head.target.name(),
            bit_perfect: false,
            output_frame: end,
            from_frame: heard.frame_at(end),
        };
        self.queue_cue(end, CueEvent::Started(Box::new(start), heard));
        self.release(&mut b);
        self.head.playing = Some(b);
    }

    /// B is back on its own route (or ended): its automation goes (it is at
    /// its end values, one in a plan that ends with every stem in), the
    /// transition is finished, and settings that waited apply (the format
    /// no longer held when they changed).
    fn settle_transition(&mut self, written: u64) {
        let Some(t) = self.transition.take() else {
            return;
        };
        if let Some((fader, _)) = self.faders() {
            fader.clear_automation();
        }
        if matches!(t.phase, Phase::Settling { cut: false }) {
            self.queue_cue(written, CueEvent::Transition(TransitionState::Finished));
        }
        if t.settings != self.settings {
            self.pinned = None;
        }
        self.reroute();
    }

    /// A ended (`a`): inside the window, B takes over at once (and `None`
    /// comes back); before it, the transition is off and `a` comes back
    /// to end as usual. B ending as it settles finishes it.
    pub(super) fn transition_takes_over(&mut self, a: Track) -> Option<Track> {
        let written = self.head.out.as_ref().map_or(0, |o| o.output.written());
        let Some(t) = self.transition.as_ref() else {
            return Some(a);
        };
        if t.mixing() {
            self.hand_over_transition(a, written);
            return None;
        }
        if matches!(t.phase, Phase::Settling { .. }) {
            self.settle_transition(written);
        } else {
            self.forget_transition(Some("the track ended before the window"));
        }
        Some(a)
    }

    /// The queue was reordered ([`super::queue`]): a transition follows
    /// its tracks while the next one is still the one after the playing
    /// one, and is called off otherwise (one handed over follows its next
    /// track).
    pub(super) fn reindex_transition(&mut self, map: &dyn Fn(usize) -> Option<usize>) {
        let Some(t) = self.transition.as_mut() else {
            return;
        };
        let settling = matches!(t.phase, Phase::Settling { .. });
        let (from, into) = (map(t.plan.from), map(t.plan.into));
        let (from, into) = match (from, into) {
            (Some(from), Some(into)) if into == from + 1 => (from, into),
            (from, Some(into)) if settling => (from.unwrap_or(t.plan.from), into),
            _ => return self.drop_transition(Some("the queue changed")),
        };
        if (from, into) != (t.plan.from, t.plan.into) {
            t.plan = Arc::new(TransitionPlan {
                from,
                into,
                ..(*t.plan).clone()
            });
            if let Some(b) = self.parked_track() {
                b.index = into;
                for cue in &mut b.held {
                    cue.event.reindex(map);
                }
            }
        }
    }
}
