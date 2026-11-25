//! The player: a command API, one decode thread from the source into the
//! output, and an event stream timed to what is heard.
//!
//! [`Player::new`] starts the thread and returns the handle and the event
//! receiver. Commands are queued and run in order. The thread:
//!
//! - opens each track ([`crate::source::Source`]), chooses its route
//!   ([`crate::rate::choose_route`]) against the target's capabilities, and
//!   opens an output at the route's format (on a desk, its one format:
//!   every route goes to it, so what follows of formats changing is a
//!   plain device's);
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
//!   says so; the new settings apply to the tracks that start after it.
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
//! `transition` (a transition compiled onto a desk), `heads` (a second
//! output on the desk, for a transition's next track), `warping`, `along`
//! (a head read along a time map), `key` and `shift` (the head on air in
//! another key, on such a map), and `lanes` (the stems as parts of an
//! output of lanes).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use jodel_output::{Capabilities, StreamFormat};

use crate::error::Error;
use crate::rate::{choose_route, FileFacts, NoRoute, Route, Settings};
use crate::source::TrackInfo;
use crate::target::OutputTarget;

mod along;
mod api;
mod heads;
mod jump;
mod key;
mod lanes;
mod play;
mod produce;
mod queue;
mod shift;
mod switch;
mod track;
mod transition;
mod warping;

pub use api::{Command, Event, PlayState, Player, PlayerOptions, Position, TrackStart};
pub use heads::{Head, HeadFaders};
pub use jump::JumpAt;
pub use lanes::MakeStems;
pub use shift::{ratio_of, KEY_RANGE};

use track::{drop_later, preroll, Cue, CueEvent, Heard, Progress, StartAt, Track};

/// How long one output write may block, so commands stay responsive.
const WRITE_WAIT: Duration = Duration::from_millis(10);

/// How long the thread waits for a command while it has nothing to write.
const IDLE_WAIT: Duration = Duration::from_millis(10);

/// Frames of a count-in's silence written at a time.
const SILENCE_FRAMES: usize = 4096;

/// How far before the frame heard a switch's new route starts decoding, so
/// that it has settled by the time it is heard: a resampler's filter is
/// full.
const PREROLL_RESAMPLE: f64 = 0.1;

/// How much of a switch's new route is decoded before the old one fades
/// out for an output at another format, so the new output starts at once.
const SWITCH_PREFILL: Duration = Duration::from_millis(200);

/// How long the new route has the thread after a write the output did not
/// take (it is full, paused or held back).
const SWITCH_TURN: Duration = Duration::from_millis(20);

struct Engine {
    /// The head played on (on a desk, the one on air).
    head: heads::HeadState,
    /// On a desk: the other head, parked ([`heads`]).
    desk: Option<Box<heads::HeadState>>,
    settings: Settings,
    position_interval: Duration,
    events: Sender<Event>,
    queue: Vec<PathBuf>,
    paused: bool,
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
    /// [`PlayerOptions::switch_fade`].
    switch_fade: Duration,
    /// [`PlayerOptions::offline`].
    offline: bool,
    /// A transition from the playing track into the next, under way.
    transition: Option<transition::Transition>,
    /// [`Command::JumpWindow`]: seconds kept behind, decoded ahead.
    jump_window: (f64, f64),
    /// Seconds of silence the next track installed starts after
    /// ([`Command::PlayIn`]).
    lead_in: f64,
    /// [`Command::HoldAtEnd`].
    hold: bool,
    /// Where the tracks' segments get their ids.
    segments: track::Segments,
    /// [`Command::Key`]: the key asked for, and the file it is on (another
    /// picked resets it; none: it is for the one picked next).
    key: i8,
    key_song: Option<PathBuf>,
}

impl Engine {
    fn new(target: Box<dyn OutputTarget>, options: PlayerOptions, events: Sender<Event>) -> Engine {
        let lanes = lanes::StemLanes::new(options.separator, options.stem_store, options.offline);
        Engine {
            head: heads::HeadState::new(0, target, lanes),
            desk: None,
            settings: options.settings,
            position_interval: options.position_interval,
            events,
            queue: Vec::new(),
            paused: false,
            next: None,
            finishing: false,
            tail: None,
            cues: VecDeque::new(),
            heard: None,
            last_report: None,
            switch_fade: options.switch_fade,
            offline: options.offline,
            transition: None,
            jump_window: (0.0, 0.0),
            lead_in: 0.0,
            hold: false,
            segments: track::Segments::default(),
            key: 0,
            key_song: None,
        }
    }

    /// Whether there is work to do: a track to write or an output playing
    /// out, or (paused too) a switch's new route to decode. (A transition's
    /// next track writes on the turns of the playing one.)
    fn busy(&self) -> bool {
        let switching =
            self.head.playing.is_some() && self.head.switch.as_ref().is_some_and(|s| !s.spliced);
        let reading = self.head.playing.is_some() && self.head.jump_fill.is_some();
        switching || reading || (!self.paused && (self.head.playing.is_some() || self.finishing))
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
                self.pick(index);
                self.jump(index, 0.0);
            }
            Command::PlayIn { index, lead } => {
                self.pick(index);
                self.lead_in = if lead.is_finite() { lead.max(0.0) } else { 0.0 };
                self.jump(index, 0.0);
                self.lead_in = 0.0;
            }
            Command::PlayAt { index, share } => {
                self.pick(index);
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
                        let end = self.head.out.as_ref().map_or(0, |o| o.output.written());
                        if let Err(error) = self.advance(last + 1, end) {
                            self.fail(None, error);
                        }
                    }
                }
            }
            Command::Remove(index) => self.remove_track(index),
            Command::Move { from, to } => self.move_track(from, to),
            Command::Pause => {
                if self.paused || self.head.out.is_none() {
                    return;
                }
                if let Some(out) = &mut self.head.out {
                    if let Err(e) = out.output.pause() {
                        return self.fail(None, e.into());
                    }
                }
                if let Err(e) = self.pause_parked(true) {
                    return self.fail(None, e);
                }
                self.paused = true;
                self.report_position(true);
                self.emit(Event::State(PlayState::Paused));
            }
            Command::Resume => self.resume(),
            Command::Seek(seconds) => self.seek(seconds),
            Command::Next => {
                // While a transition is heard, into its next track at once.
                if self.cut_to_next() {
                    return;
                }
                if let Some(index) = self.current_index() {
                    self.pick(index + 1);
                    self.jump(index + 1, 0.0);
                }
            }
            Command::Previous => {
                if let Some(index) = self.current_index() {
                    self.pick(index.saturating_sub(1));
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
            Command::MakeStems(make) => self.every_lanes().for_each(|l| l.set_make(make)),
            Command::Key { semitones } => self.set_key(semitones),
            Command::Along { map, from, tempo } => self.read_along(map, from, tempo),
        }
    }

    /// Track `index` is picked to play (not run into): playback plays, and
    /// the key goes back to 0 unless it is for this track
    /// ([`Engine::key_picked`]).
    fn pick(&mut self, index: usize) {
        self.paused = false;
        self.key_picked(index);
    }

    /// [`Command::Resume`]: the outputs paused play on.
    fn resume(&mut self) {
        if !self.paused {
            return;
        }
        if let Some(out) = &mut self.head.out {
            if let Err(e) = out.output.resume() {
                return self.fail(None, e.into());
            }
        }
        if let Err(e) = self.pause_parked(false) {
            return self.fail(None, e);
        }
        self.paused = false;
        self.emit(Event::State(PlayState::Playing));
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
            let index = self.head.playing.as_ref().map(|t| t.index);
            self.notice(
                index,
                "The new settings apply once the transition is over".into(),
            );
            return;
        }
        if settings != self.settings {
            // Shifted, the head keeps its map until it is back in its key.
            if self.head_shifted() {
                let index = self.head.playing.as_ref().map(|t| t.index);
                self.notice(index, key::SETTINGS_WAIT.into());
            }
        }
        self.settings = settings;
        self.reroute();
    }

    /// The route the settings choose for `info` from what the target takes
    /// (on a desk, its one format), or why there is none (a sentence).
    fn route_for(&mut self, info: &TrackInfo) -> Result<Route, String> {
        let name = self.head.target.name();
        let caps = self
            .target_caps()
            .map_err(|e| format!("Could not ask {name} what it takes: {e}"))?;
        choose_route(&FileFacts::from(info), &caps, &self.settings).map_err(|none| {
            format!("No route for this track with these settings on {name} ({none})")
        })
    }

    /// The route the settings choose for `info` at `format` alone.
    fn route_at(&self, info: &TrackInfo, format: StreamFormat) -> Result<Route, NoRoute> {
        let mut caps = Capabilities::new();
        caps.insert(format);
        choose_route(&FileFacts::from(info), &caps, &self.settings)
    }

    /// What the target takes, asked anew for each route: on a desk its one
    /// format, which is the device's, and changes when the device comes back
    /// at another rate (one round trip to the mixer's thread).
    fn target_caps(&mut self) -> Result<Capabilities, Error> {
        Ok(self.head.target.capabilities()?)
    }

    /// How `track` is heard when it is read along a map (a head along its
    /// own, the playing track along a transition's glide, the next one
    /// along its beats): then its position is its map's, not a straight
    /// line from its segment.
    fn heard_along(&self, track: &Track) -> Option<Heard> {
        track.along.is_some().then(|| track.heard())
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
            .or(self.head.playing.as_ref().map(|t| t.index))
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
        self.abandon_transition("playback moved elsewhere");
        self.cues.clear();
        self.head.playing = None;
        self.next = None;
        self.finishing = false;
        self.tail = None;
        self.cancel_switch();
        self.head.reroute_later = false;
        self.head.along_at = None;
        self.head.pending_jump = None;
        self.drop_jump_fill();
        self.head.lead = 0;
    }

    /// Sends the cues the output clock has reached.
    fn fire_cues(&mut self) {
        let position = match &self.head.out {
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
            head: self.head.id,
            seconds: heard.seconds_at(output_frame),
            frame: heard.frame_at(output_frame),
            output_frame,
            key: heard.key,
        };
        self.last_report = Some((Instant::now(), output_frame));
        self.emit(Event::Position(event));
    }

    fn report_position(&mut self, force: bool) {
        let (Some(_), Some(out)) = (&self.heard, &self.head.out) else {
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
        if self.head.lead > 0 && self.head.playing.is_some() {
            return self.write_lead();
        }
        self.along_turn()?;
        self.tend_parked();
        if self.head.playing.is_some() && self.work_switch()? {
            return Ok(());
        }
        if self.head.playing.is_some() && self.transition_turn()? {
            return Ok(());
        }
        if self.head.playing.is_some() && self.jump_fill_turn()? {
            return Ok(());
        }
        if self.jump_turn()? {
            return Ok(());
        }
        if let Some(mut track) = self.head.playing.take() {
            let progress = match self.step(&mut track) {
                Ok(progress) => progress,
                Err(error) if error.is_skippable() => {
                    self.broke(&mut track, error);
                    Progress::Busy
                }
                Err(error) => return Err(error),
            };
            match progress {
                Progress::Busy => self.head.playing = Some(track),
                Progress::Complete => self.finish(track)?,
            }
            return Ok(());
        }
        if self.finishing {
            let drained = match &mut self.head.out {
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
