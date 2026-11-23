//! The player: a command API, one decode thread from the source into the
//! output, and an event stream timed to what is heard.
//!
//! [`Player::new`] starts the thread and returns the handle and the event
//! receiver. Commands are queued and run in order. The thread:
//!
//! - opens each track ([`crate::source::Source`]), chooses its route
//!   ([`crate::rate::choose_route`]) against the target's capabilities, and
//!   opens an output at the route's format;
//! - reads the source and resamples what it reads when the route has a
//!   [`crate::rate::Resample`] step
//!   ([`crate::Resampler`], one stream per track and per seek, so each
//!   track keeps its exact length; samples that are not 24-bit integers
//!   yet, from lossy and floating-point files, go into it unquantized and
//!   are requantized once), and hands the output to
//!   [`jodel_output::Output::write`];
//! - at the end of a track opens the next one at once and, when its route
//!   plays at the same output format, keeps writing into the same output
//!   (gapless); otherwise it lets the output play out, closes it and opens
//!   one at the new format;
//! - seeks by moving the source to the exact frame and flushing the
//!   output;
//! - on new settings ([`Command::SetSettings`]) moves the track being
//!   played to its new route without a break: the new route reads ahead,
//!   from a little before the frame being written, while the old one goes
//!   on playing; once it has caught up, the two meet at an exact frame
//!   after what is written. At the same output format they crossfade over
//!   [`PlayerOptions::switch_fade`] (nothing flushed, the output stays
//!   open); at another format the old route fades out at that frame, the
//!   output plays out and opens at the new format, and the new route
//!   starts there, faded in, from the frame after it. Settings never stop
//!   playback: when the track has no route with them, or its new route
//!   cannot be opened, it keeps playing as it was and a [`Event::Notice`]
//!   says so; the new settings apply to the tracks that start after it;
//! - opens outputs again for up to [`OPEN_BUSY_WAIT`] while the device
//!   reports it is busy (as it can right after an output on it closed).
//!
//! Events about what is heard (track started and ended, clipped samples,
//! errors of skipped tracks, the end of the queue) are queued against the
//! output frame they belong to and sent when the output clock
//! ([`jodel_output::Output::position`]) reaches it. Position events are sent
//! every [`PlayerOptions::position_interval`] while the clock moves.
//!
//! Here: the engine's state, its command loop and its turns of work, new
//! settings, the cues and what is reported. Its other parts: `api` (the
//! commands, events and the handle), `track` (a track as it plays, its
//! cues and fades), `produce` (making a track's output and writing it),
//! `switch` (moving a track to a new route), `play` (opening, starting,
//! seeking, stopping and running into the next track), `jump`, `queue`,
//! `transition`, `warping`, `along` (a head read along a time map), and
//! `lanes` (the stems as
//! parts of an output of lanes).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::Capabilities;

use crate::error::Error;
use crate::rate::{choose_route, FileFacts, Route, Settings};
use crate::source::TrackInfo;
use crate::stem_file::StemStore;
use crate::stems::SeparatorSource;
use crate::target::OutputTarget;

mod along;
mod api;
mod jump;
mod lanes;
mod play;
mod produce;
mod queue;
mod stem_mask;
mod switch;
mod track;
mod transition;
mod warping;

pub use api::{Command, Event, PlayState, Player, PlayerOptions, Position, TrackStart};
pub use jump::JumpAt;
pub use lanes::MakeStems;

use track::{
    drop_later, preroll, Cue, CueEvent, Heard, OpenOutput, Progress, StartAt, Switch, Track,
};

/// How long one output write may block, so commands stay responsive.
const WRITE_WAIT: Duration = Duration::from_millis(10);

/// How long the thread waits for a command while it has nothing to write.
const IDLE_WAIT: Duration = Duration::from_millis(10);

/// How long opening an output is tried again while the device reports it
/// is busy ([`jodel_output::Error::DeviceBusy`]: WASAPI's device in use,
/// Core Audio's hog mode held, ALSA's `EBUSY`), before the open fails.
pub const OPEN_BUSY_WAIT: Duration = Duration::from_millis(1_000);

/// The pause between those tries.
const OPEN_BUSY_STEP: Duration = Duration::from_millis(100);

/// Frames of a stem mix taken at a time.
const STEM_FRAMES: usize = 4096;

/// How far before the frame heard a switch's new route starts decoding, so
/// that it has settled by the time it is heard: a separation (a
/// transition's track onto its stems) has context, a resampler's filter is
/// full.
const PREROLL_STEMS: f64 = 1.0;

const PREROLL_RESAMPLE: f64 = 0.1;

/// How much of a switch's new route is decoded before the old one fades
/// out for an output at another format, so the new output starts at once.
const SWITCH_PREFILL: Duration = Duration::from_millis(200);

/// Seconds of output a switch back onto a track's own route after a
/// transition waits for its stem parts at most ([`Switch::parts_wait`]).
const PARTS_PATIENCE: f64 = 2.0;

/// How long the new route has the thread after a write the output did not
/// take (it is full, paused or held back).
const SWITCH_TURN: Duration = Duration::from_millis(20);

struct Engine {
    target: Box<dyn OutputTarget>,
    caps: Option<Capabilities>,
    settings: Settings,
    position_interval: Duration,
    events: Sender<Event>,
    queue: Vec<PathBuf>,
    out: Option<OpenOutput>,
    paused: bool,
    /// The track being decoded.
    playing: Option<Track>,
    /// The next track, opened, waiting for the output to play out before
    /// the output is reopened at its format.
    next: Option<Track>,
    /// Everything is written; waiting for the output to play out (then
    /// `next` starts, or the queue has ended).
    finishing: bool,
    /// The last track index played when the queue ran out.
    tail: Option<usize>,
    cues: VecDeque<Cue>,
    heard: Option<Heard>,
    last_report: Option<(Instant, u64)>,
    /// Where a transition's stems get their separator (the parts have
    /// theirs in `lanes`).
    separator: Option<SeparatorSource>,
    /// Where tracks' stems are kept.
    stem_store: Option<Arc<dyn StemStore>>,
    /// The stems of a transition's track failed: the transitions after it
    /// take their tracks whole ([`crate::stems::whole_mix`]) until stems
    /// may work again: the mix changes outside a transition
    /// ([`Command::StemMask`]), playback stops (a new queue too), or the
    /// parts of the output are separated again.
    stems_failed: bool,
    /// The playing track's move to another route, under way.
    switch: Option<Switch>,
    /// [`PlayerOptions::switch_fade`].
    switch_fade: Duration,
    /// [`PlayerOptions::offline`].
    offline: bool,
    /// The settings changed while a switch was being handed over: route
    /// the track again once it has been.
    reroute_later: bool,
    /// When the output last took nothing of a write.
    stalled: Option<Instant>,
    /// A transition from the playing track into the next, under way.
    transition: Option<transition::Transition>,
    /// [`Command::StemMask`]: each stem's gain as a mixer plays them, and
    /// the changes of them to come at times of a track.
    stem_gains: stem_mask::StemGains,
    /// [`Command::JumpWindow`]: seconds kept behind, decoded ahead.
    jump_window: (f64, f64),
    /// A [`Command::Along`] waiting for its frame.
    along_at: Option<along::PendingAlong>,
    /// A [`Command::Jump`] waiting for its time.
    pending_jump: Option<jump::PendingJump>,
    /// A reading of the playing track at a jump's target it does not hold.
    jump_fill: Option<Box<jump::JumpFill>>,
    /// Seconds of silence the next track installed starts after
    /// ([`Command::PlayIn`]).
    lead_in: f64,
    /// Output frames of that silence still to write, before the playing
    /// track's.
    lead: u64,
    /// [`Command::HoldAtEnd`].
    hold: bool,
    /// The playing track's stems as parts of the output.
    lanes: lanes::StemLanes,
    /// Where the tracks' segments get their ids.
    segments: track::Segments,
}

impl Engine {
    fn new(target: Box<dyn OutputTarget>, options: PlayerOptions, events: Sender<Event>) -> Engine {
        Engine {
            target,
            caps: options.capabilities,
            settings: options.settings,
            position_interval: options.position_interval,
            events,
            queue: Vec::new(),
            out: None,
            paused: false,
            playing: None,
            next: None,
            finishing: false,
            tail: None,
            cues: VecDeque::new(),
            heard: None,
            last_report: None,
            separator: options.separator,
            stem_store: options.stem_store.clone(),
            stems_failed: false,
            switch: None,
            switch_fade: options.switch_fade,
            offline: options.offline,
            reroute_later: false,
            stalled: None,
            transition: None,
            stem_gains: stem_mask::StemGains::default(),
            jump_window: (0.0, 0.0),
            along_at: None,
            pending_jump: None,
            jump_fill: None,
            lead_in: 0.0,
            lead: 0,
            hold: false,
            lanes: lanes::StemLanes::new(options.separator, options.stem_store, options.offline),
            segments: track::Segments::default(),
        }
    }

    /// Whether there is work to do: a track to write or an output playing
    /// out, or (paused too) a switch's new route to decode.
    fn busy(&self) -> bool {
        let switching = self.playing.is_some() && self.switch.as_ref().is_some_and(|s| !s.spliced);
        let reading = self.playing.is_some() && self.jump_fill.is_some();
        switching || reading || (!self.paused && (self.playing.is_some() || self.finishing))
    }

    fn emit(&self, event: Event) {
        // A receiver that has gone away just stops listening.
        let _ = self.events.send(event);
    }

    fn run(mut self, commands: Receiver<Command>) {
        loop {
            let wait = if self.busy() {
                Duration::ZERO
            } else {
                IDLE_WAIT
            };
            match commands.recv_timeout(wait) {
                Ok(command) => {
                    self.handle(command);
                    while let Ok(command) = commands.try_recv() {
                        self.handle(command);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            self.fire_cues();
            self.report_position(false);
            // The commands may have paused, stopped or started playback.
            if self.busy() {
                if let Err(error) = self.work() {
                    self.fail(None, error);
                }
            }
        }
        self.stop(false);
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Load(paths) => {
                self.stop(true);
                self.queue = paths;
                self.emit_queue();
            }
            Command::Play(index) => {
                self.paused = false;
                self.jump(index, 0.0);
            }
            Command::PlayIn { index, lead } => {
                self.paused = false;
                self.lead_in = if lead.is_finite() { lead.max(0.0) } else { 0.0 };
                self.jump(index, 0.0);
                self.lead_in = 0.0;
            }
            Command::PlayAt { index, share } => {
                self.paused = false;
                let share = if share.is_finite() {
                    share.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.jump_on(index, StartAt::Share(share), None);
            }
            Command::HoldAtEnd(hold) => self.hold = hold,
            Command::Enqueue(paths) => {
                self.queue.extend(paths);
                self.emit_queue();
                if self.finishing && self.next.is_none() {
                    if let Some(last) = self.tail.take() {
                        self.cues
                            .retain(|c| !matches!(c.event, CueEvent::QueueEnded));
                        self.finishing = false;
                        let end = self.out.as_ref().map_or(0, |o| o.output.written());
                        if let Err(error) = self.advance(last + 1, end) {
                            self.fail(None, error);
                        }
                    }
                }
            }
            Command::Remove(index) => self.remove_track(index),
            Command::Move { from, to } => self.move_track(from, to),
            Command::Pause => {
                if self.paused || self.out.is_none() {
                    return;
                }
                if let Some(out) = &mut self.out {
                    if let Err(e) = out.output.pause() {
                        return self.fail(None, e.into());
                    }
                }
                self.paused = true;
                self.report_position(true);
                self.emit(Event::State(PlayState::Paused));
            }
            Command::Resume => {
                if !self.paused {
                    return;
                }
                if let Some(out) = &mut self.out {
                    if let Err(e) = out.output.resume() {
                        return self.fail(None, e.into());
                    }
                }
                self.paused = false;
                self.emit(Event::State(PlayState::Playing));
            }
            Command::Seek(seconds) => self.seek(seconds),
            // While a transition is heard, into its next track at once.
            Command::Next if self.cut_to_next() => {}
            Command::Next => {
                if let Some(index) = self.current_index() {
                    self.paused = false;
                    self.jump(index + 1, 0.0);
                }
            }
            Command::Previous => {
                if let Some(index) = self.current_index() {
                    self.paused = false;
                    self.jump(index.saturating_sub(1), 0.0);
                }
            }
            Command::Stop => self.stop(true),
            Command::SetSettings(settings) => self.set_settings(settings),
            Command::Transition(plan) => self.arm_transition(*plan),
            // A mix under way is not cut off (moving playback does that).
            Command::CancelTransition if !self.transition_heard() => {
                self.drop_transition(Some("called off"));
            }
            Command::CancelTransition => {}
            Command::Jump { index, to, at } => self.jump_within(index, to, at),
            Command::JumpWindow { back, ahead } => self.jump_window = (back, ahead),
            Command::MakeStems(make) => self.lanes.set_make(make),
            Command::StemMask { gains, at } => self.set_stem_mask(gains, at),
            Command::Along { map, from, tempo } => self.read_along(map, from, tempo),
        }
    }

    fn notice(&self, index: Option<usize>, message: String) {
        self.emit(Event::Notice { index, message });
    }

    /// [`Command::SetSettings`]: the track being played moves to its new
    /// route without a break ([`Engine::reroute`]), or keeps the one it has
    /// when there is none or it does not open. It is never skipped and
    /// playback never stops for it.
    fn set_settings(&mut self, settings: Settings) {
        // New settings never call a transition off. Under way, they wait
        // for its end (its tracks' routes are chosen again then); armed, it
        // takes them when it prepares.
        if settings != self.settings && self.transition_running() {
            self.settings = settings;
            let index = self.playing.as_ref().map(|t| t.index);
            self.notice(
                index,
                "The new settings apply once the transition is over".into(),
            );
            return;
        }
        self.settings = settings;
        self.reroute();
    }

    /// The route the settings choose for `info` now, or why there is none
    /// (a sentence).
    fn route_for(&mut self, info: &TrackInfo) -> Result<Route, String> {
        let caps = self
            .capabilities()
            .map_err(|e| format!("Could not ask {} what it takes: {e}", self.target.name()))?;
        choose_route(&FileFacts::from(info), &caps, &self.settings).map_err(|none| {
            format!(
                "No route for this track with these settings on {} ({none})",
                self.target.name()
            )
        })
    }

    /// How `track` is heard when it is read along a map (a head along its
    /// own, the playing track along a transition's glide, the next one
    /// along its beats): then its position is its map's, not a straight
    /// line from its segment.
    fn heard_along(&self, track: &Track) -> Option<Heard> {
        if track.along.is_some() {
            return Some(track.heard());
        }
        self.heard
            .as_ref()
            .filter(|h| h.index == track.index && h.warp.is_some() && track.warp.is_some())
            .cloned()
    }

    fn emit_play_state(&self) {
        self.emit(Event::State(if self.paused {
            PlayState::Paused
        } else {
            PlayState::Playing
        }));
    }

    fn emit_queue(&self) {
        self.emit(Event::QueueChanged {
            queue: self.queue.clone(),
            current: self.heard.as_ref().map(|h| h.index),
        });
    }

    fn current_index(&self) -> Option<usize> {
        self.heard
            .as_ref()
            .map(|h| h.index)
            .or(self.playing.as_ref().map(|t| t.index))
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        if self.caps.is_none() {
            self.caps = Some(self.target.capabilities()?);
        }
        Ok(self.caps.clone().unwrap_or_default())
    }

    /// Queues a cue of `track` at output frame `at` (or after the last one
    /// queued), or holds it while the track is pending. Returns where it
    /// went.
    fn push_cue(&mut self, track: &mut Track, at: u64, event: CueEvent) -> u64 {
        let queue = if track.pending {
            &mut track.held
        } else {
            &mut self.cues
        };
        let at = at.max(queue.back().map_or(0, |c| c.at));
        queue.push_back(Cue { at, event });
        at
    }

    /// Drops everything decoded and queued, without closing the output.
    fn clear_timeline(&mut self) {
        self.forget_transition(Some("playback moved elsewhere"));
        self.cues.clear();
        self.playing = None;
        self.next = None;
        self.finishing = false;
        self.tail = None;
        self.cancel_switch();
        self.reroute_later = false;
        self.along_at = None;
        self.pending_jump = None;
        self.drop_jump_fill();
        self.lead = 0;
    }

    /// Sends the cues the output clock has reached.
    fn fire_cues(&mut self) {
        let position = match &self.out {
            Some(out) => out.clock.position(),
            None => return,
        };
        while self.cues.front().is_some_and(|c| c.at <= position) {
            let cue = self.cues.pop_front().expect("checked");
            match cue.event {
                CueEvent::Started(start, heard) => {
                    self.heard = Some(heard);
                    self.last_report = None;
                    self.emit(Event::TrackStarted(start));
                }
                CueEvent::Clipped { index, samples } => self.emit(Event::Clipped {
                    index,
                    samples,
                    output_frame: cue.at,
                }),
                CueEvent::Ended { index } => {
                    self.report_at(cue.at);
                    self.emit(Event::TrackEnded {
                        index,
                        finished: true,
                    });
                }
                CueEvent::Skipped { index, error } => self.emit(Event::Error {
                    index: Some(index),
                    error,
                    skipped: true,
                }),
                CueEvent::QueueEnded => {
                    self.heard = None;
                    self.emit(Event::QueueEnded);
                }
                CueEvent::Held { index } => {
                    self.heard = None;
                    self.emit(Event::Held { index });
                }
                CueEvent::Transition(state) => self.emit(Event::Transition(state)),
                CueEvent::Heard(heard) => {
                    self.heard = Some(heard);
                    self.last_report = None;
                }
                CueEvent::Jumped { index, seconds } => {
                    self.emit(Event::Jumped { index, seconds });
                    self.report_at(cue.at);
                }
            }
        }
    }

    fn report_at(&mut self, output_frame: u64) {
        let Some(heard) = &self.heard else {
            return;
        };
        let event = Position {
            index: heard.index,
            seconds: heard.seconds_at(output_frame),
            frame: heard.frame_at(output_frame),
            output_frame,
        };
        self.last_report = Some((Instant::now(), output_frame));
        self.emit(Event::Position(event));
    }

    fn report_position(&mut self, force: bool) {
        let (Some(_), Some(out)) = (&self.heard, &self.out) else {
            return;
        };
        let position = out.clock.position();
        if !force {
            if let Some((at, last)) = self.last_report {
                if last == position || at.elapsed() < self.position_interval {
                    return;
                }
            }
        }
        self.report_at(position);
    }

    /// One unit of work: write pending output, decode a write, or finish
    /// the track; or, while finishing, let the output play out. A switch
    /// under way takes its turns first ([`Engine::work_switch`]).
    fn work(&mut self) -> Result<(), Error> {
        if self.lead > 0 && self.playing.is_some() {
            return self.write_lead();
        }
        self.along_turn()?;
        if self.playing.is_some() && self.work_switch()? {
            return Ok(());
        }
        if self.playing.is_some() && self.transition_turn()? {
            return Ok(());
        }
        if self.playing.is_some() && self.hold_for_transition()? {
            return Ok(());
        }
        if self.playing.is_some() && self.jump_fill_turn()? {
            return Ok(());
        }
        if self.jump_turn()? {
            return Ok(());
        }
        if let Some(mut track) = self.playing.take() {
            let progress = match self.step(&mut track) {
                Ok(progress) => progress,
                Err(error) if error.is_skippable() => {
                    self.broke(&mut track, error);
                    Progress::Busy
                }
                Err(error) => return Err(error),
            };
            if let Some(why) = track.stems_error.take() {
                self.stems_stopped(track, &why);
                return Ok(());
            }
            match progress {
                Progress::Busy => {
                    // A transition adds the next track to what was just made.
                    self.mix_transition(&mut track)?;
                    self.playing = Some(track);
                }
                Progress::Complete => self.finish(track)?,
            }
            return Ok(());
        }
        if self.finishing {
            let drained = match &mut self.out {
                Some(out) => out.output.drain(Some(WRITE_WAIT))?,
                None => true,
            };
            if !drained {
                return Ok(());
            }
            self.fire_cues();
            self.finishing = false;
            match self.next.take() {
                Some(track) => {
                    self.close_output();
                    self.install_next(track)?;
                    self.fire_cues();
                }
                None => {
                    self.tail = None;
                    self.close_output();
                    self.heard = None;
                    self.emit(Event::State(PlayState::Stopped));
                }
            }
        }
        Ok(())
    }
}
