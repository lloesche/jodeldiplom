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
//! `transition`, `warping`, and `lanes` (the stems as
//! parts of an output of lanes).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::Capabilities;

use crate::error::Error;
use crate::rate::{choose_route_near, FileFacts, Route, Settings};
use crate::source::TrackInfo;
use crate::stem_file::StemStore;
use crate::stems::{SeparatorSource, StemMix, StemRun};
use crate::target::OutputTarget;

mod api;
mod jump;
mod lanes;
mod play;
mod produce;
mod queue;
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
/// that it has settled by the time it is heard: a separation has context,
/// a resampler's filter is full.
const PREROLL_STEMS: f64 = 1.0;

const PREROLL_RESAMPLE: f64 = 0.1;

/// How much of a switch's new route is decoded before the old one fades
/// out for an output at another format, so the new output starts at once.
const SWITCH_PREFILL: Duration = Duration::from_millis(200);

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
    /// Where stem playback gets its separator.
    separator: Option<SeparatorSource>,
    /// Where tracks' stems are kept.
    stem_store: Option<Arc<dyn StemStore>>,
    /// The stems failed: tracks play their usual routes until the mix
    /// changes.
    stems_failed: bool,
    /// The next track's separation, started once the playing track's file
    /// has been read, so a gapless change finds its stems ready.
    prestart: Option<StemRun>,
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
    /// A change of the stems heard at a time of the playing track
    /// ([`Command::SetStemsAt`]), not yet reached.
    stems_change: Option<StemsChange>,
    /// [`Command::JumpWindow`]: seconds kept behind, decoded ahead.
    jump_window: (f64, f64),
    /// [`Command::KeepStems`].
    keep_stems: bool,
    /// The playing track's stems while it plays its file
    /// ([`Command::KeepStems`]), following what it writes; a stems route
    /// for it takes them over.
    shadow: Option<StemRun>,
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

/// A change of the stems heard at a time of a track: before it the track
/// plays `before`, from it [`Settings::stems`]. It is held to the output
/// frame that time was heard at when it was asked for, so a jump in
/// between does not move it (a bar looped takes it at the loop's end).
#[derive(Clone, Copy, Debug)]
struct StemsChange {
    /// The track.
    index: usize,
    /// The output frame it is heard at.
    out: u64,
    /// Seconds into the track (what a transition's fades go by: its
    /// tracks are read along maps).
    at: f64,
    /// The stems heard before.
    before: StemMix,
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
            prestart: None,
            switch: None,
            switch_fade: options.switch_fade,
            offline: options.offline,
            reroute_later: false,
            stalled: None,
            transition: None,
            stems_change: None,
            jump_window: (0.0, 0.0),
            keep_stems: false,
            shadow: None,
            pending_jump: None,
            jump_fill: None,
            lead_in: 0.0,
            lead: 0,
            hold: false,
            lanes: lanes::StemLanes::new(
                options.separator,
                options.stem_store.clone(),
                options.offline,
            ),
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

    /// The settings routes are chosen with: the stems left out when there
    /// is no separator or they failed.
    fn route_settings(&self) -> Settings {
        let mut settings = self.settings;
        // All on again at a later time: the stems route plays on to then.
        if let Some(change) = self.stems_change {
            if settings.stems.is_full()
                && self
                    .playing
                    .as_ref()
                    .is_some_and(|t| t.index == change.index)
            {
                settings.stems = change.before;
            }
        }
        if self.separator.is_none() || self.stems_failed {
            settings.stems = StemMix::ALL;
        }
        settings
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
            Command::SetSettings(settings) => {
                // Other stems at once: no change waits for its time.
                if settings.stems != self.settings.stems {
                    self.stems_change = None;
                }
                self.set_settings(settings);
            }
            Command::Transition(plan) => self.arm_transition(*plan),
            // A mix under way is not cut off (moving playback does that).
            Command::CancelTransition if !self.transition_heard() => {
                self.drop_transition(Some("called off"));
            }
            Command::CancelTransition => {}
            Command::SetStemsAt { stems, at } => self.set_stems_at(stems, at),
            Command::Jump { index, to, at } => self.jump_within(index, to, at),
            Command::JumpWindow { back, ahead } => self.jump_window = (back, ahead),
            Command::KeepStems(keep) => {
                self.keep_stems = keep;
                if !keep {
                    self.shadow = None;
                }
            }
            Command::MakeStems(make) => self.lanes.set_make(make),
        }
    }

    fn notice(&self, index: Option<usize>, message: String) {
        self.emit(Event::Notice { index, message });
    }

    /// [`Command::SetStemsAt`]: the playing track's stems change at `at`;
    /// a route the new stems need (the first stem off) is moved to now,
    /// and one they no longer need (all on again) once `at` has been made.
    fn set_stems_at(&mut self, stems: StemMix, at: f64) {
        let Some(index) = self.playing.as_ref().map(|t| t.index) else {
            return self.set_settings(Settings {
                stems,
                ..self.settings
            });
        };
        // A change still to come keeps what was heard before it.
        let before = match self.stems_change {
            Some(change) if change.index == index => change.before,
            _ => self.settings.stems,
        };
        // Where the track is heard at `at` as it plays now.
        let out = self
            .playing
            .as_ref()
            .map_or(0, |t| transition::output_frame_of(t, at));
        self.stems_change = (stems != before).then_some(StemsChange {
            index,
            out,
            at,
            before,
        });
        self.set_settings(Settings {
            stems,
            ..self.settings
        });
    }

    /// A stems change whose time has been made is over: with every stem
    /// on again, the track goes back to its own route.
    fn stems_change_turn(&mut self) {
        let Some(change) = self.stems_change else {
            return;
        };
        let (Some(track), Some(out)) = (&self.playing, &self.out) else {
            return;
        };
        if track.index != change.index {
            self.stems_change = None;
            return;
        }
        if out.output.written() < change.out {
            return;
        }
        self.stems_change = None;
        if self.settings.stems.is_full() {
            self.reroute();
        }
    }

    /// [`Command::SetSettings`]: the track being played moves to its new
    /// route without a break ([`Engine::reroute`]), or keeps the one it has
    /// when there is none or it does not open. It is never skipped and
    /// playback never stops for it.
    fn set_settings(&mut self, settings: Settings) {
        // New settings never call a transition off. Under way, other stems
        // go into its fades and the rest waits for its end (its tracks'
        // routes are chosen again then); armed, it takes them when it
        // prepares.
        if settings != self.settings && self.transition_running() {
            let stems = settings.stems != self.settings.stems;
            let others = Settings {
                stems: self.settings.stems,
                ..settings
            } != self.settings;
            if stems {
                self.stems_failed = false;
            }
            self.settings = settings;
            if stems {
                self.retune_transition();
            }
            if others {
                let index = self.playing.as_ref().map(|t| t.index);
                self.notice(
                    index,
                    "The new settings apply once the transition is over".into(),
                );
            }
            return;
        }
        if settings.stems != self.settings.stems {
            // Another mix tries the stems again.
            self.stems_failed = false;
        }
        if settings.stems.is_full() {
            self.prestart = None;
        }
        // A new mix of the same route needs nothing more: the stems that
        // are on are summed as the output is filled.
        self.settings = settings;
        self.reroute();
    }

    /// The route the settings choose for `info` now, or why there is none
    /// (a sentence).
    fn route_for(&mut self, info: &TrackInfo) -> Result<Route, String> {
        let caps = self
            .capabilities()
            .map_err(|e| format!("Could not ask {} what it takes: {e}", self.target.name()))?;
        let open = self.out.as_ref().map(|o| o.info.format);
        choose_route_near(
            &FileFacts::from(info),
            &caps,
            &self.route_settings(),
            open.as_ref(),
        )
        .map_err(|none| {
            format!(
                "No route for this track with these settings on {} ({none})",
                self.target.name()
            )
        })
    }

    /// How `track` is heard when it is read along a map (the playing track
    /// along a transition's glide, the next one along its beats): then its
    /// position is its map's, not a straight line from its segment.
    fn heard_along(&self, track: &Track) -> Option<&Heard> {
        self.heard
            .as_ref()
            .filter(|h| h.index == track.index && h.warp.is_some() && track.warp.is_some())
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
        self.stems_change = None;
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
        self.stems_change_turn();
        self.shadow_turn();
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
