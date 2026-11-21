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

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use jodel_output::{Capabilities, Clock, OpenRequest, Output, StreamFormat, StreamInfo};

use crate::error::Error;
use crate::rate::{choose_route_near, FileFacts, Route, Settings, Stage};
use crate::resample::{Requantizer, Resampler};
use crate::source::{Source, TrackInfo};
use crate::stem_file::StemStore;
use crate::stems::{Gains, Pull, SeparatorSource, StemMix, StemRun};
use crate::transition::{TransitionPlan, TransitionState};
use crate::warp::TimeMap;

mod jump;
mod queue;
mod transition;

pub use jump::JumpAt;

use crate::target::OutputTarget;
use transition::Warping;

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

/// What a [`Player`] can be told to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Stops and replaces the queue (nothing plays until [`Command::Play`]).
    Load(Vec<PathBuf>),
    /// Plays the queue from `index`.
    Play(usize),
    /// Plays track `index` from its start after `lead` seconds of silence
    /// (a count-in). Its [`Event::TrackStarted`] comes as the silence
    /// begins to be heard, and its positions run negative through it
    /// (seconds before the track's first frame), so a count-in drawn from
    /// them falls on what is heard.
    PlayIn {
        /// The track.
        index: usize,
        /// Seconds of silence before it.
        lead: f64,
    },
    /// Plays track `index` from `share` of its length (0 to 1; from its
    /// start when the length is not known), as a preview from a third in.
    PlayAt {
        /// The track.
        index: usize,
        /// Where it starts, as a share of its length.
        share: f64,
    },
    /// Whether playback holds at the end of each track (karaoke): the next
    /// one is not run into; once the track has been heard to its end,
    /// [`Event::Held`] names the track that would have come, and playback
    /// stops ([`Command::PlayIn`] or [`Command::Play`] start it). A track
    /// already running into the next one (its last frames written) still
    /// does.
    HoldAtEnd(bool),
    /// Appends to the queue. If the queue had played out to its end and is
    /// still letting the last track play, playback continues gaplessly.
    Enqueue(Vec<PathBuf>),
    /// Takes track `index` out of the queue. Playback goes on as it was;
    /// the track being heard gives way to the one after it (as
    /// [`Command::Next`]), or playback stops after the last.
    Remove(usize),
    /// Moves track `from` to `to` in the queue (the others close up
    /// behind it). Playback goes on as it was; a transition whose next
    /// track is no longer the one after the playing track is called off.
    Move {
        /// The track's index now.
        from: usize,
        /// Its index after.
        to: usize,
    },
    /// Pauses the output.
    Pause,
    /// Resumes after [`Command::Pause`].
    Resume,
    /// Moves to this many seconds into the track being heard (clamped to
    /// the track). A paused player stays paused.
    Seek(f64),
    /// Plays the next track; while a transition is heard, its next track
    /// plays on alone at once, from where it is heard.
    Next,
    /// Plays the previous track (the first one restarts).
    Previous,
    /// Stops and closes the output.
    Stop,
    /// Replaces the settings. If the track being played would now take
    /// another route, it moves to that route without a break, about as
    /// late as the output is long (at another output format, with the
    /// pause the output needs to open again). If it would have none, or the
    /// new route's output does not open, it keeps its route and position
    /// (an [`Event::Notice`] says so); it is never skipped.
    SetSettings(Settings),
    /// Plays a transition from the playing track into the next
    /// ([`TransitionPlan`]), replacing one armed or preparing (its tracks
    /// stay where they are; one heard is not replaced);
    /// [`Event::Transition`] says how it goes. Only moving playback
    /// elsewhere (play, seek, next, previous, stop, load, the next track
    /// taken out of the queue) calls it off. New stems go into its fades,
    /// at their time; other settings wait for its end.
    Transition(Box<TransitionPlan>),
    /// Calls the transition off, if it is not heard yet (a mix under way
    /// plays to its end).
    CancelTransition,
    /// The stems heard become `stems` from `at` seconds into the playing
    /// track (a bar's first beat, say), to the frame, rather than as soon
    /// as the output has played what it holds ([`Command::SetSettings`]).
    /// Until then the track plays the stems it had. The time is held to
    /// the output frame it is heard at as the track plays now: a jump in
    /// between (a bar looped) does not move it. A time already made (or a
    /// route that cannot be met by then) takes the change as soon as it
    /// can.
    SetStemsAt {
        /// The stems heard from then on.
        stems: StemMix,
        /// Seconds into the playing track.
        at: f64,
    },
    /// The playing track (`index`) goes to `to` seconds, at once or when it
    /// reaches a time ([`JumpAt`]), without a gap: within the stretch it
    /// keeps ([`Command::JumpWindow`]) the output gives back what it has
    /// not played and the target follows, crossfaded over
    /// [`PlayerOptions::switch_fade`]; elsewhere the track is read again
    /// there and takes over once it is ready (later, never with a gap).
    /// [`Event::Jumped`] says when it is heard.
    Jump {
        /// The track (another than the one playing: nothing happens).
        index: usize,
        /// Seconds into it to go to.
        to: f64,
        /// When.
        at: JumpAt,
    },
    /// How much of the playing track is kept behind what is heard and
    /// decoded ahead of it, in seconds, so that a [`Command::Jump`] within
    /// is heard at once (zero: none, as it plays otherwise). A stems route
    /// keeps its separated stems for as long, and mixes the target from
    /// them with the stems that are on. None while a transition is set up.
    JumpWindow {
        /// Seconds kept behind.
        back: f64,
        /// Seconds decoded ahead.
        ahead: f64,
    },
    /// Whether the playing track is separated into its stems while it
    /// plays its file too (all stems on), for its jump window: a stem
    /// switched off then has the stems from before it, so a jump back is
    /// heard at once. Costs a stem separation while it is on; nothing
    /// without a separator or a jump window.
    KeepStems(bool),
    /// The microphone listened to (`None`: none): never played, but lined
    /// up with what the output plays ([`jodel_output::Output::set_tap`], a
    /// tap that only reads the device's buffers), so its echo of the music
    /// can be taken out; set again on every output opened. On an output
    /// that runs no taps the music's echo stays in its voice (an
    /// [`Event::Notice`] says so). What plays never changes.
    SetMic(Option<Mic>),
}

/// A microphone for [`Command::SetMic`]: an open input. Two are equal when
/// they capture the same device the same way.
#[derive(Clone, Debug)]
pub struct Mic(pub jodel_output::input::Input);

impl PartialEq for Mic {
    fn eq(&self, other: &Mic) -> bool {
        self.0.info() == other.0.info()
    }
}

/// Whether the player is playing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayState {
    /// Nothing is playing; no output is open.
    Stopped,
    /// Playing.
    Playing,
    /// Paused; the output stays open.
    Paused,
}

/// What a track plays through, sent when it starts to be heard.
#[derive(Clone, Debug)]
pub struct TrackStart {
    /// The track's index in the queue.
    pub index: usize,
    /// The file: format, tags, cover.
    pub info: Arc<TrackInfo>,
    /// The route the rate policy chose.
    pub route: Route,
    /// What the output was opened at.
    pub output: StreamInfo,
    /// The output target's name (the device).
    pub device: String,
    /// The samples reach the device unchanged: the device is held
    /// exclusively, the route is lossless and the format holds its bits.
    /// Never for lossy files.
    pub bit_perfect: bool,
    /// The output frame it starts at ([`Position::output_frame`]).
    pub output_frame: u64,
    /// The track frame it starts from (0, or where a seek or a switch of
    /// route took it up).
    pub from_frame: u64,
}

/// Where playback is, as heard (from the output clock).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    /// The track's index in the queue.
    pub index: usize,
    /// Seconds into the track.
    pub seconds: f64,
    /// The file frame being heard.
    pub frame: u64,
    /// The output frame being heard (as [`jodel_output::Output::position`]
    /// counts them).
    pub output_frame: u64,
}

/// What the player reports. Track, clip and queue-end events arrive when
/// the output clock reaches the frame they belong to, in order.
#[derive(Clone, Debug)]
pub enum Event {
    /// The queue changed. `current` is the track being heard.
    QueueChanged {
        /// The queue.
        queue: Vec<PathBuf>,
        /// The track being heard, if any.
        current: Option<usize>,
    },
    /// The play state changed.
    State(PlayState),
    /// A track started to be heard.
    TrackStarted(Box<TrackStart>),
    /// The position, as heard.
    Position(Position),
    /// Samples clipped when requantized to 24 bits (saturated to full
    /// scale), as heard: by the resampler (inter-sample overs), or by the
    /// source for lossy and floating-point files (overs in the decoded
    /// signal). Sent at most four times a second of output while it clips.
    Clipped {
        /// The track's index in the queue.
        index: usize,
        /// Samples clipped since the track (or the last seek) started.
        samples: u64,
        /// The output frame from which the count applies.
        output_frame: u64,
    },
    /// A seek within the track being heard took effect.
    Seeked {
        /// The track's index in the queue.
        index: usize,
        /// Where it is now, in seconds.
        seconds: f64,
    },
    /// A track stopped being heard: `finished` when it played to its end,
    /// false when it was left (next, previous, play, stop, load).
    TrackEnded {
        /// The track's index in the queue.
        index: usize,
        /// Whether it played to its end.
        finished: bool,
    },
    /// Something failed. `skipped` errors (an unreadable file, no route,
    /// a decoder rejection) cost only that track, and playback goes on with
    /// the next; others stop playback.
    Error {
        /// The track it concerns, if any.
        index: Option<usize>,
        /// What failed.
        error: Error,
        /// Whether playback went on with the next track.
        skipped: bool,
    },
    /// Something the user should know that changes nothing else: new
    /// settings that the track being heard keeps playing through (no route
    /// with them, or their output did not open).
    Notice {
        /// The track it concerns, if any.
        index: Option<usize>,
        /// A sentence for the user.
        message: String,
    },
    /// The stems of a track stopped (the separator failed): it plays on
    /// its usual route from where they stopped, and every track does until
    /// the mix changes.
    StemsStopped {
        /// The track.
        index: usize,
        /// Why.
        reason: String,
    },
    /// How a transition goes.
    Transition(TransitionState),
    /// A [`Command::Jump`] is heard from now: the track plays on from
    /// `seconds`.
    Jumped {
        /// The track.
        index: usize,
        /// Where it is now.
        seconds: f64,
    },
    /// The last track of the queue played out.
    QueueEnded,
    /// A track played to its end while holding at ends
    /// ([`Command::HoldAtEnd`]): `index` is the one that would have come
    /// next, waiting to be started. Playback stops after it.
    Held {
        /// The next track.
        index: usize,
    },
}

/// How a [`Player`] is set up.
#[derive(Clone, Debug)]
pub struct PlayerOptions {
    /// The rate policy settings.
    pub settings: Settings,
    /// How often position events are sent while playing.
    pub position_interval: Duration,
    /// The target's capabilities, when the caller has probed them already
    /// (otherwise the engine asks the target once, when it first needs
    /// them).
    pub capabilities: Option<Capabilities>,
    /// Where stem playback gets its separator ([`crate::stems`]); `None`:
    /// no stem playback, and [`Settings::stems`] is ignored.
    pub separator: Option<SeparatorSource>,
    /// Where tracks' stems are kept ([`crate::stem_file`]): a track found
    /// there plays its kept stems, and only one that is not is separated.
    pub stem_store: Option<Arc<dyn StemStore>>,
    /// How long a switch of route crossfades at the same output format
    /// (and fades out and in at another); zero: they meet at one frame.
    pub switch_fade: Duration,
    /// The output is not a device playing in real time (a file being
    /// rendered, a test's output): what a transition needs is waited for
    /// rather than gone without. The playing track holds at the window
    /// until it is on the stems the window needs, and each piece of the
    /// mix waits for the next track's frames however long they take. A
    /// device could not wait (it would run dry); a file loses nothing by it.
    pub offline: bool,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        PlayerOptions {
            settings: Settings::default(),
            position_interval: Duration::from_millis(100),
            capabilities: None,
            separator: None,
            stem_store: None,
            switch_fade: Duration::from_millis(20),
            offline: false,
        }
    }
}

/// The handle to the playback engine. Dropping it stops playback and joins
/// the thread.
pub struct Player {
    commands: Option<Sender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl Player {
    /// Starts the engine on `target`. Events arrive on the receiver.
    pub fn new(target: Box<dyn OutputTarget>, options: PlayerOptions) -> (Player, Receiver<Event>) {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let engine = Engine::new(target, options, event_tx);
        let thread = std::thread::Builder::new()
            .name("jodel-playback".into())
            .spawn(move || engine.run(command_rx))
            .expect("spawn the playback thread");
        (
            Player {
                commands: Some(command_tx),
                thread: Some(thread),
            },
            event_rx,
        )
    }

    /// Queues a command. Fails only when the engine thread has stopped.
    pub fn send(&self, command: Command) -> Result<(), Error> {
        self.commands
            .as_ref()
            .ok_or(Error::Stopped)?
            .send(command)
            .map_err(|_| Error::Stopped)
    }

    /// [`Command::Load`].
    pub fn load(&self, paths: Vec<PathBuf>) -> Result<(), Error> {
        self.send(Command::Load(paths))
    }

    /// [`Command::Play`].
    pub fn play(&self, index: usize) -> Result<(), Error> {
        self.send(Command::Play(index))
    }

    /// [`Command::PlayIn`].
    pub fn play_in(&self, index: usize, lead: f64) -> Result<(), Error> {
        self.send(Command::PlayIn { index, lead })
    }

    /// [`Command::HoldAtEnd`].
    pub fn hold_at_end(&self, hold: bool) -> Result<(), Error> {
        self.send(Command::HoldAtEnd(hold))
    }

    /// [`Command::Remove`].
    pub fn remove(&self, index: usize) -> Result<(), Error> {
        self.send(Command::Remove(index))
    }

    /// [`Command::Move`].
    pub fn move_track(&self, from: usize, to: usize) -> Result<(), Error> {
        self.send(Command::Move { from, to })
    }

    /// [`Command::Enqueue`].
    pub fn enqueue(&self, paths: Vec<PathBuf>) -> Result<(), Error> {
        self.send(Command::Enqueue(paths))
    }

    /// [`Command::Pause`].
    pub fn pause(&self) -> Result<(), Error> {
        self.send(Command::Pause)
    }

    /// [`Command::Resume`].
    pub fn resume(&self) -> Result<(), Error> {
        self.send(Command::Resume)
    }

    /// [`Command::Seek`].
    pub fn seek(&self, seconds: f64) -> Result<(), Error> {
        self.send(Command::Seek(seconds))
    }

    /// [`Command::Next`].
    pub fn next(&self) -> Result<(), Error> {
        self.send(Command::Next)
    }

    /// [`Command::Previous`].
    pub fn previous(&self) -> Result<(), Error> {
        self.send(Command::Previous)
    }

    /// [`Command::Stop`].
    pub fn stop(&self) -> Result<(), Error> {
        self.send(Command::Stop)
    }

    /// [`Command::SetSettings`].
    pub fn set_settings(&self, settings: Settings) -> Result<(), Error> {
        self.send(Command::SetSettings(settings))
    }

    /// [`Command::Transition`].
    pub fn transition(&self, plan: TransitionPlan) -> Result<(), Error> {
        self.send(Command::Transition(Box::new(plan)))
    }

    /// [`Command::CancelTransition`].
    pub fn cancel_transition(&self) -> Result<(), Error> {
        self.send(Command::CancelTransition)
    }

    /// [`Command::SetStemsAt`].
    pub fn set_stems_at(&self, stems: StemMix, at: f64) -> Result<(), Error> {
        self.send(Command::SetStemsAt { stems, at })
    }

    /// [`Command::Jump`].
    pub fn jump(&self, index: usize, to: f64, at: JumpAt) -> Result<(), Error> {
        self.send(Command::Jump { index, to, at })
    }

    /// [`Command::JumpWindow`].
    pub fn jump_window(&self, back: f64, ahead: f64) -> Result<(), Error> {
        self.send(Command::JumpWindow { back, ahead })
    }

    /// [`Command::KeepStems`].
    pub fn keep_stems(&self, keep: bool) -> Result<(), Error> {
        self.send(Command::KeepStems(keep))
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        // Closing the channel ends the thread, which stops the output.
        self.commands.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Where in a track playback starts.
#[derive(Clone, Copy, Debug, PartialEq)]
enum StartAt {
    /// Seconds in.
    Seconds(f64),
    /// A share of its length (its start when the length is not known).
    Share(f64),
}

/// The track being heard: what maps the output clock to the track.
#[derive(Clone)]
struct Heard {
    index: usize,
    info: Arc<TrackInfo>,
    route: Route,
    /// The output frame of the segment's first frame.
    seg_out: u64,
    /// The track frame at the segment's start.
    seg_frame: u64,
    /// The track is read along a time map (a transition's next track):
    /// its position comes from that, not from a straight line.
    warp: Option<Arc<HeardWarp>>,
}

/// How a track read along a time map is heard: its output frame `origin`
/// is the map's time `a0`, and output frames go at `out_rate`.
#[derive(Clone)]
struct HeardWarp {
    map: Arc<TimeMap>,
    origin: u64,
    a0: f64,
    out_rate: u32,
}

impl HeardWarp {
    /// The track's seconds at `output_frame`.
    fn seconds_at(&self, output_frame: u64) -> f64 {
        let since = output_frame.saturating_sub(self.origin) as f64;
        self.map
            .at(self.a0 + since / f64::from(self.out_rate.max(1)))
            .max(0.0)
    }
}

/// `frames` at `from` Hz in frames at `to` Hz, rounded down.
fn rescale(frames: u64, from: u32, to: u32) -> u64 {
    (u128::from(frames) * u128::from(to) / u128::from(from.max(1))) as u64
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// A route's smallest exact step: `a` track frames are `b` output frames
/// (a track frame that many from the segment's start is heard at an exact
/// output frame).
fn grid(route: &Route) -> (u64, u64) {
    let (input, output) = (u64::from(route.input_rate), u64::from(route.output_rate));
    let g = gcd(input, output);
    (input / g, output / g)
}

/// `duration` in frames at `rate`.
fn frames_of(duration: Duration, rate: u32) -> u64 {
    (duration.as_secs_f64() * f64::from(rate)).round() as u64
}

/// How many track frames before the switch `route` starts decoding.
fn preroll(route: &Route, rate: u32) -> u64 {
    let mut seconds: f64 = 0.0;
    if route.stage == Stage::Stems {
        seconds = seconds.max(PREROLL_STEMS);
    }
    if route.resample.is_some() {
        seconds = seconds.max(PREROLL_RESAMPLE);
    }
    (seconds * f64::from(rate)).round() as u64
}

/// Drops what `track` just made (from sample `before` on) while it has
/// frames to discard (a stems route's preroll after a jump).
fn discard_made(track: &mut Track, before: usize) {
    if track.discard == 0 {
        return;
    }
    let channels = usize::from(track.route.channels);
    let made = ((track.out_buf.len() - before) / channels) as u64;
    let drop = track.discard.min(made);
    track
        .out_buf
        .drain(before..before + drop as usize * channels);
    track.discard -= drop;
}

/// Drops `value` on a thread of its own: a track's file closes there (on a
/// network share that can take a while) and its buffers are freed, while
/// the playback thread goes on writing.
fn drop_later<T: Send + 'static>(value: T) {
    // If no thread starts, it is dropped here after all.
    let _ = std::thread::Builder::new()
        .name("jodel-drop".into())
        .spawn(move || drop(value));
}

/// The weight of the new side `i` frames into a fade of `frames`: a raised
/// cosine from 0 to 1 (the two sides sum to one).
fn fade_weight(i: usize, frames: usize) -> f64 {
    0.5 - 0.5 * (std::f64::consts::PI * (i as f64 + 0.5) / frames as f64).cos()
}

/// Crossfades from `old` into `new` (interleaved, the same length).
fn crossfade(old: &[i32], new: &mut [i32], channels: usize) {
    let frames = old.len() / channels;
    for (i, (from, to)) in old
        .chunks_exact(channels)
        .zip(new.chunks_exact_mut(channels))
        .enumerate()
    {
        let w = fade_weight(i, frames);
        for (a, b) in from.iter().zip(to) {
            *b = (f64::from(*a) * (1.0 - w) + f64::from(*b) * w).round() as i32;
        }
    }
}

/// Fades `samples` (interleaved) in, or out.
fn fade(samples: &mut [i32], channels: usize, rising: bool) {
    let frames = samples.len() / channels;
    for (i, frame) in samples.chunks_exact_mut(channels).enumerate() {
        let w = fade_weight(i, frames);
        let w = if rising { w } else { 1.0 - w };
        for s in frame {
            *s = (f64::from(*s) * w).round() as i32;
        }
    }
}

impl Heard {
    fn frame_at(&self, output_frame: u64) -> u64 {
        if let Some(w) = &self.warp {
            let frame = (w.seconds_at(output_frame) * f64::from(self.info.rate)) as u64;
            return self.info.frames.map_or(frame, |total| frame.min(total));
        }
        let since = output_frame.saturating_sub(self.seg_out);
        let frame = self.seg_frame + rescale(since, self.route.output_rate, self.route.input_rate);
        self.info.frames.map_or(frame, |total| frame.min(total))
    }

    fn seconds_at(&self, output_frame: u64) -> f64 {
        if let Some(w) = &self.warp {
            let seconds = w.seconds_at(output_frame);
            return match self.info.duration() {
                Some(d) => seconds.min(d.as_secs_f64()),
                None => seconds,
            };
        }
        // Before the segment only in a count-in's silence: negative there.
        let since = output_frame as f64 - self.seg_out as f64;
        let seconds = self.seg_frame as f64 / f64::from(self.info.rate)
            + since / f64::from(self.route.output_rate);
        match self.info.duration() {
            Some(d) => seconds.min(d.as_secs_f64()),
            None => seconds,
        }
    }
}

/// Something to send when the output clock reaches a frame.
enum CueEvent {
    Started(Box<TrackStart>, Heard),
    Clipped {
        index: usize,
        samples: u64,
    },
    Ended {
        index: usize,
    },
    Skipped {
        index: usize,
        error: Error,
    },
    QueueEnded,
    /// Holding at the end: this track is the next one.
    Held {
        index: usize,
    },
    /// A transition's progress, as heard.
    Transition(TransitionState),
    /// The track heard is now mapped by this (its timeline was made
    /// straight again after a transition, or it jumped).
    Heard(Heard),
    /// A jump within the track is heard.
    Jumped {
        index: usize,
        seconds: f64,
    },
}

struct Cue {
    at: u64,
    event: CueEvent,
}

/// A track being decoded and written.
struct Track {
    index: usize,
    info: Arc<TrackInfo>,
    route: Route,
    source: Source,
    /// The rate conversion after the source, when the route has one.
    resampler: Option<Resampler>,
    source_done: bool,
    /// The source's samples on their way into the resampler.
    read_buf: Vec<i32>,
    /// The source's unquantized samples on their way into the resampler.
    wide_buf: Vec<f64>,
    /// Output samples not yet written, from `out_at`.
    out_buf: Vec<i32>,
    out_at: usize,
    /// The output frame of this segment's first frame.
    seg_out: u64,
    /// The track frame where the segment starts.
    seg_frame: u64,
    /// Output frames still to drop before the segment starts (on a stems
    /// route, output frames after a jump's preroll).
    discard: u64,
    /// Output frames made in this segment (after the discard).
    emitted: u64,
    /// The clip count last queued, and the output frame it was queued at.
    clips: Option<(u64, u64)>,
    /// The error that ended the track early, reported once what was
    /// decoded before it is written.
    failure: Option<Error>,
    /// On a [`Stage::Stems`] route: the separation, started when the track
    /// is first stepped (or taken over from [`Engine::prestart`]).
    stems: Option<StemRun>,
    /// Every stem has been taken.
    stems_done: bool,
    /// Why the separation stopped: the track goes back to its usual route.
    stems_error: Option<String>,
    /// The requantization of a stem mix the device takes at its own rate.
    requantizer: Requantizer,
    /// Decoded ahead for a switch, not written yet: its output frames
    /// count from 0 at `seg_frame`, and its cues are held.
    pending: bool,
    /// Cues not queued yet: a pending track's, and once it has met the old
    /// route, those to queue when it is installed (counted from
    /// `seg_out`).
    held: VecDeque<Cue>,
    /// Clips made before the segment heard (in a switch's run-up).
    clip_base: u64,
    /// The route it switched from, to go back to if its output does not
    /// open.
    fallback: Option<Route>,
    /// A transition's fades of its stems, in place of the settings' mix.
    gains: Option<Gains>,
    /// A transition's next track: its stems read along the time map.
    warp: Option<Box<Warping>>,
    /// The separator its stems use, in place of the player's (a
    /// transition without one).
    separator: Option<SeparatorSource>,
    /// Samples clipped when a transition added the next track to this
    /// one's output (past the soft limit's bend, so none in practice).
    mix_clips: u64,
    /// The samples a jump's crossfade replaced, from this frame of
    /// `out_buf`, put back once written ([`jump`]).
    restore: Option<(usize, Vec<i32>)>,
    /// Decoding ahead found the end of the track.
    ahead_done: bool,
}

/// A switch of the track being played to another route, under way.
struct Switch {
    /// The track on the new route, decoding ahead from `seg_frame`.
    track: Track,
    /// The track frames from the old segment's start that the two routes
    /// can meet at are multiples of this.
    grid: u64,
    /// The new route plays at the open output's format.
    fits: bool,
    /// Decoded from the start once already.
    retried: bool,
    /// The new route has played up to the end; it can meet the old one
    /// with what it has.
    done: bool,
    /// The routes have met: the old track's output ends where the new
    /// one's begins.
    spliced: bool,
}

/// What a switch is waiting for.
enum Splice {
    Wait,
    Done,
    Cancel,
}

impl Track {
    fn new(
        index: usize,
        info: Arc<TrackInfo>,
        route: Route,
        mut source: Source,
    ) -> Result<Track, Error> {
        let resampler = route.resample.map(|r| {
            let mut resampler = Resampler::new(r.from, r.to, usize::from(route.channels));
            // The headroom scales the samples before the one
            // requantization.
            resampler.set_gain(r.headroom.gain());
            resampler
        });
        if resampler.is_none() {
            // The one requantization is the source's (a lossy or
            // floating-point file): the headroom goes there.
            source.set_gain(route.headroom.gain());
        }
        Ok(Track {
            index,
            info,
            route,
            source,
            resampler,
            source_done: false,
            read_buf: Vec::new(),
            wide_buf: Vec::new(),
            out_buf: Vec::new(),
            out_at: 0,
            seg_out: 0,
            seg_frame: 0,
            discard: 0,
            emitted: 0,
            clips: None,
            failure: None,
            stems: None,
            stems_done: false,
            stems_error: None,
            requantizer: Requantizer::new(),
            pending: false,
            held: VecDeque::new(),
            gains: None,
            warp: None,
            separator: None,
            mix_clips: 0,
            clip_base: 0,
            fallback: None,
            restore: None,
            ahead_done: false,
        })
    }

    /// The track's output frames waiting to be written.
    fn unwritten(&self) -> usize {
        (self.out_buf.len() - self.out_at) / usize::from(self.route.channels)
    }

    /// Samples clipped by the source's requantization, the resampler and
    /// the stem mix's requantization, since the track opened.
    fn clipped(&self) -> u64 {
        self.source.clipped()
            + self.resampler.as_ref().map_or(0, |r| r.clipped())
            + self.requantizer.clipped()
            + self.mix_clips
    }

    /// Restarts the track at `frame`: the source moves there and the
    /// segment starts again.
    fn restart_at(&mut self, frame: u64) -> Result<(), Error> {
        let frame = self.info.frames.map_or(frame, |total| frame.min(total));
        self.restore = None;
        self.ahead_done = false;
        if self.route.stage == Stage::Stems {
            // A new separation starts there when the track is next stepped.
            self.stems = None;
            self.stems_done = false;
            self.stems_error = None;
            self.requantizer.reset();
        } else {
            if self.source.position() != frame {
                self.source.seek(frame)?;
            }
            self.source_done = false;
        }
        if let Some(resampler) = &mut self.resampler {
            resampler.reset();
        }
        self.out_buf.clear();
        self.out_at = 0;
        self.seg_frame = frame;
        self.discard = 0;
        self.emitted = 0;
        self.clips = None;
        self.failure = None;
        Ok(())
    }

    /// Moves a track just opened to `frame` (nothing to do at 0).
    fn start_at(&mut self, frame: u64) -> Result<(), Error> {
        if frame > 0 {
            self.restart_at(frame)?;
        }
        Ok(())
    }
}

struct OpenOutput {
    output: Box<dyn Output>,
    info: StreamInfo,
    clock: Clock,
}

/// What one unit of work on the current track did.
enum Progress {
    Busy,
    Complete,
}

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
    /// [`Command::SetMic`].
    mic: Option<jodel_output::input::Input>,
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
            mic: None,
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
            Command::SetMic(mic) => {
                self.mic = mic.map(|m| m.0);
                self.apply_mic();
            }
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

    /// Moves the track being played to the route the settings choose for
    /// it: a [`Switch`], decoded ahead and met at an exact frame. The track
    /// waiting for the output to play out takes its new route at once (it
    /// is not heard yet).
    fn reroute(&mut self) {
        // A transition under way has its tracks where it needs them; its
        // end routes them again.
        if self.transition_running() {
            return;
        }
        if self.switch.as_ref().is_some_and(|s| s.spliced) {
            // The old route ends at the frame the new one begins: once
            // the new one is playing, it is routed again.
            self.reroute_later = true;
            return;
        }
        let Some(track) = &self.playing else {
            return self.reroute_next();
        };
        let (index, info, current) = (track.index, track.info.clone(), track.route.clone());
        // A track a transition made (read along its time map, its stems
        // faded by it, or on its stand-in separator) is not on its route as
        // the settings make it once the transition is over, though the
        // route is the same: it moves to a fresh one.
        let leftover = self.transition.is_none()
            && (track.warp.is_some() || track.gains.is_some() || track.separator.is_some());
        let keeping = |why: String| format!("{why}; keeping {} for this track", current.describe());
        let route = match self.route_for(&info) {
            Ok(route) => route,
            Err(why) => {
                self.cancel_switch();
                return self.notice(Some(index), keeping(why));
            }
        };
        if route == current && !leftover {
            self.cancel_switch();
            return;
        }
        if self.switch.as_ref().is_some_and(|s| s.track.route == route) {
            return;
        }
        self.cancel_switch();
        if let Err(error) = self.begin_switch(route) {
            let message = keeping(format!("Could not reopen this track: {error}"));
            self.notice(Some(index), message);
        }
    }

    /// [`Engine::reroute`] for the track waiting for the output to play
    /// out: it takes its new route now, and when that fits the open output,
    /// it follows what was written without a break.
    fn reroute_next(&mut self) {
        let Some(next) = &self.next else {
            return;
        };
        let (index, frame, info, current, fallback) = (
            next.index,
            next.seg_frame,
            next.info.clone(),
            next.route.clone(),
            next.fallback.clone(),
        );
        let keeping = |why: String| format!("{why}; keeping {} for this track", current.describe());
        let route = match self.route_for(&info) {
            Ok(route) if route == current => return,
            Ok(route) => route,
            Err(why) => return self.notice(Some(index), keeping(why)),
        };
        let mut track = match self.prepare_at(index, frame, Some(route)) {
            Ok(track) => track,
            Err(error) => {
                let message = keeping(format!("Could not reopen this track: {error}"));
                return self.notice(Some(index), message);
            }
        };
        // A switch's new route waits for the old one, faded out, to play
        // out: it opens the output again rather than start unfaded after
        // it, and can go back to the old route.
        let switched = fallback.is_some();
        track.fallback = fallback;
        let open = self
            .out
            .as_ref()
            .map(|o| (o.info.format, o.output.written()));
        match open {
            Some((format, end)) if !switched && track.route.fits(&format) => {
                // Nothing was written after the end the track waits at.
                track.seg_out = end;
                let cue = self.started(&track);
                self.cues.push_back(cue);
                self.next = None;
                self.finishing = false;
                self.playing = Some(track);
            }
            _ => self.next = Some(track),
        }
    }

    /// Starts the playing track's move to `route`: the track opens on it a
    /// little before the frame being written ([`preroll`]), on the grid
    /// both routes meet exactly on, and decodes ahead from there.
    fn begin_switch(&mut self, route: Route) -> Result<(), Error> {
        let out = self.out.as_ref().ok_or(Error::Stopped)?;
        let written = out.output.written();
        let fits = route.fits(&out.info.format);
        let old = self.playing.as_ref().ok_or(Error::Stopped)?;
        let (a_old, _) = grid(&old.route);
        let (a_new, _) = grid(&route);
        let step = a_old / gcd(a_old, a_new) * a_new;
        // A track read along a map (the playing one along a transition's
        // glide) is where its map, as heard, puts it.
        let along = self.heard_along(old).map(|h| h.frame_at(written));
        let at = along.unwrap_or_else(|| {
            old.seg_frame
                + rescale(
                    written.saturating_sub(old.seg_out),
                    old.route.output_rate,
                    old.route.input_rate,
                )
        });
        let want = at.saturating_sub(preroll(&route, old.info.rate));
        // On the grid from the old segment's start.
        let first = old.seg_frame % step;
        let start = if want >= first {
            first + (want - first) / step * step
        } else {
            first
        };
        let index = old.index;
        let mut track = self.prepare_at(index, start, Some(route))?;
        track.pending = true;
        self.switch = Some(Switch {
            track,
            grid: step,
            fits,
            retried: false,
            done: false,
            spliced: false,
        });
        Ok(())
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

    /// Opens track `index` at `seconds` and chooses its route.
    fn prepare(&mut self, index: usize, seconds: f64) -> Result<Track, Error> {
        self.prepare_with(index, StartAt::Seconds(seconds), None)
    }

    /// Opens track `index` at `start` on `route`, or on the route the
    /// settings choose for it.
    fn prepare_with(
        &mut self,
        index: usize,
        start: StartAt,
        route: Option<Route>,
    ) -> Result<Track, Error> {
        let mut track = self.open_track(index, route)?;
        let frame = match start {
            StartAt::Seconds(seconds) => {
                (seconds.max(0.0) * f64::from(track.info.rate)).round() as u64
            }
            StartAt::Share(share) => track
                .info
                .frames
                .map_or(0, |n| (n as f64 * share).round() as u64),
        };
        track.start_at(frame)?;
        Ok(track)
    }

    /// [`Engine::prepare_with`] at track frame `frame`.
    fn prepare_at(
        &mut self,
        index: usize,
        frame: u64,
        route: Option<Route>,
    ) -> Result<Track, Error> {
        let mut track = self.open_track(index, route)?;
        track.start_at(frame)?;
        Ok(track)
    }

    /// Opens track `index` at its start on `route`, or on the route the
    /// settings choose for it.
    fn open_track(&mut self, index: usize, route: Option<Route>) -> Result<Track, Error> {
        let path = self.queue[index].clone();
        let source = Source::open(&path)?;
        let info = source.info().clone();
        let route = match route {
            Some(route) => route,
            None => {
                let caps = self.capabilities()?;
                let open = self.out.as_ref().map(|o| o.info.format);
                choose_route_near(
                    &FileFacts::from(&*info),
                    &caps,
                    &self.route_settings(),
                    open.as_ref(),
                )?
            }
        };
        Track::new(index, info, route, source)
    }

    /// Opens an output, trying again while the device is busy, for up to
    /// [`OPEN_BUSY_WAIT`].
    fn open_target(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let deadline = Instant::now() + OPEN_BUSY_WAIT;
        loop {
            match self.target.open(request) {
                Err(jodel_output::Error::DeviceBusy) if Instant::now() < deadline => {
                    std::thread::sleep(OPEN_BUSY_STEP);
                }
                result => return Ok(result?),
            }
        }
    }

    fn open_output(&mut self, route: &Route) -> Result<(), Error> {
        self.close_output();
        let mut output = self.open_target(&route.open_request())?;
        if self.paused {
            output.pause()?;
        }
        let info = output.info().clone();
        let clock = output.clock();
        self.out = Some(OpenOutput {
            output,
            info,
            clock,
        });
        self.apply_mic();
        Ok(())
    }

    /// Lines the microphone (or none) up with what the open output plays:
    /// a tap that only reads its buffers.
    fn apply_mic(&mut self) {
        let Some(out) = &mut self.out else {
            return;
        };
        let tap = self
            .mic
            .as_ref()
            .map(|m| Box::new(m.tap()) as Box<dyn jodel_output::Tap>);
        let wanted = tap.is_some();
        if let Err(e) = out.output.set_tap(tap) {
            if wanted {
                let index = self.playing.as_ref().map(|t| t.index);
                self.notice(
                    index,
                    format!("The microphone cannot hear this output (its echo stays in): {e}"),
                );
            }
        }
    }

    fn close_output(&mut self) {
        if let Some(mut out) = self.out.take() {
            let _ = out.output.close();
        }
    }

    fn started(&self, track: &Track) -> Cue {
        let out = self.out.as_ref().expect("an output is open");
        let format: StreamFormat = out.info.format;
        let bit_perfect = out.info.exclusive && track.route.lossless && track.route.fits(&format);
        let start = TrackStart {
            index: track.index,
            info: track.info.clone(),
            route: track.route.clone(),
            output: out.info.clone(),
            device: self.target.name(),
            bit_perfect,
            output_frame: track.seg_out,
            from_frame: track.seg_frame,
        };
        Cue {
            at: track.seg_out,
            event: CueEvent::Started(
                Box::new(start),
                Heard {
                    index: track.index,
                    info: track.info.clone(),
                    route: track.route.clone(),
                    seg_out: track.seg_out,
                    seg_frame: track.seg_frame,
                    // A track read along a map (the playing one along a
                    // transition's glide) is heard along it, from here,
                    // where the map's time is the track's.
                    warp: track.warp.as_ref().map(|w| {
                        Arc::new(HeardWarp {
                            map: Arc::clone(w.map()),
                            origin: track.seg_out,
                            a0: track.seg_frame as f64 / f64::from(track.info.rate.max(1)),
                            out_rate: track.route.output_rate,
                        })
                    }),
                },
            ),
        }
    }

    /// Makes `track` the one being played now: reuses the output when it
    /// fits (flushing what it holds), otherwise opens one.
    fn install(&mut self, mut track: Track) -> Result<(), Error> {
        let fits = self
            .out
            .as_ref()
            .is_some_and(|o| track.route.fits(&o.info.format));
        if fits {
            let out = self.out.as_mut().expect("checked");
            out.output.flush()?;
        } else {
            self.open_output(&track.route)?;
        }
        let out = self.out.as_ref().expect("an output is open");
        let written = out.output.written();
        // A count-in's silence first: the track is heard after it, and
        // starts to be as it begins.
        let rate = f64::from(out.info.format.rate);
        self.lead = (std::mem::take(&mut self.lead_in) * rate).round() as u64;
        track.seg_out = written + self.lead;
        let mut cue = self.started(&track);
        cue.at = written;
        self.cues.push_back(cue);
        self.release(&mut track);
        self.playing = Some(track);
        Ok(())
    }

    /// Writes some of a count-in's silence ([`Engine::lead`]); while the
    /// output is full, the track decodes ahead behind it, so it is ready
    /// (its stems separated) when the silence ends.
    fn write_lead(&mut self) -> Result<(), Error> {
        let out = self.out.as_mut().ok_or(Error::Stopped)?;
        let channels = usize::from(out.info.format.channels);
        let frames = self.lead.min(STEM_FRAMES as u64) as usize;
        let silence = vec![0i32; frames * channels];
        let took = out.output.write(&silence, Some(WRITE_WAIT))?;
        self.lead -= took as u64;
        if took == 0 {
            self.stalled = Some(Instant::now());
            if let Some(mut track) = self.playing.take() {
                // Ahead by a few seconds at most: the rest as it plays.
                let ahead = u64::from(track.route.output_rate) * 3;
                let result = if (track.unwritten() as u64) < ahead {
                    self.produce(&mut track, Duration::ZERO).map(|_| ())
                } else {
                    Ok(())
                };
                self.playing = Some(track);
                result?;
            }
        }
        Ok(())
    }

    /// Queues `track`'s held cues, from its `seg_out`.
    fn release(&mut self, track: &mut Track) {
        for mut cue in track.held.drain(..) {
            cue.at = (cue.at + track.seg_out).max(self.cues.back().map_or(0, |c| c.at));
            self.cues.push_back(cue);
        }
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

    /// Drops the switch under way, if any (in the background: closing its
    /// file and freeing its buffers can take a while).
    fn cancel_switch(&mut self) {
        if let Some(mut switch) = self.switch.take() {
            // The stems it took over are kept for the next one.
            let file = self.playing.as_ref().is_some_and(|t| {
                t.route.stage != Stage::Stems && t.info.path == switch.track.info.path
            });
            if file && self.shadow.is_none() {
                self.shadow = switch.track.stems.take();
            }
            drop_later(switch.track);
        }
    }

    /// Plays track `index` at `seconds` now, skipping tracks that fail to
    /// open or route.
    fn jump(&mut self, index: usize, seconds: f64) {
        self.jump_on(index, StartAt::Seconds(seconds), None);
    }

    /// [`Engine::jump`] to `start` of the track, on `route` when one is
    /// given.
    fn jump_on(&mut self, index: usize, start: StartAt, route: Option<Route>) {
        // Stems separated ahead for another next track are not wanted.
        let next = self.queue.get(index + 1);
        if self
            .prestart
            .as_ref()
            .is_some_and(|p| Some(&p.path) != next)
        {
            self.prestart = None;
        }
        let previous = self.heard.as_ref().map(|h| h.index);
        self.clear_timeline();
        if let Some(previous) = previous {
            if previous != index {
                self.emit(Event::TrackEnded {
                    index: previous,
                    finished: false,
                });
                self.heard = None;
            }
        }
        let mut index = index;
        let mut start = start;
        let mut route = route;
        loop {
            if index >= self.queue.len() {
                self.close_output();
                if self.heard.take().is_some() || previous.is_some() {
                    self.emit(Event::QueueEnded);
                }
                self.paused = false;
                self.emit(Event::State(PlayState::Stopped));
                return;
            }
            match self.prepare_with(index, start, route.take()) {
                Ok(track) => {
                    if let Err(error) = self.install(track) {
                        return self.fail(Some(index), error);
                    }
                    self.fire_cues();
                    self.emit_play_state();
                    return;
                }
                Err(error) if error.is_skippable() => {
                    self.emit(Event::Error {
                        index: Some(index),
                        error,
                        skipped: true,
                    });
                    index += 1;
                    start = StartAt::Seconds(0.0);
                }
                Err(error) => return self.fail(Some(index), error),
            }
        }
    }

    fn seek(&mut self, seconds: f64) {
        // A count-in under way ends: the track plays from where it went.
        self.lead = 0;
        self.pending_jump = None;
        self.drop_jump_fill();
        let Some(heard) = self.heard.clone() else {
            return;
        };
        // A switch under way, or one waiting for the output to open again,
        // has the route the track is going to: the seek's flush hides the
        // change.
        let switching = self
            .switch
            .take()
            .filter(|s| s.track.index == heard.index)
            .map(|s| s.track.route)
            .or_else(|| {
                self.next
                    .as_ref()
                    .filter(|t| t.index == heard.index)
                    .map(|t| t.route.clone())
            });
        // A transition is off; a track read along one starts afresh on its
        // own route.
        self.forget_transition(Some("a seek"));
        let transitioned = self
            .playing
            .as_ref()
            .is_some_and(|t| t.warp.is_some() || t.gains.is_some());
        if transitioned {
            self.jump_on(heard.index, StartAt::Seconds(seconds), None);
            if let Some(h) = &self.heard {
                let at = h.seconds_at(h.seg_out);
                self.emit(Event::Seeked {
                    index: h.index,
                    seconds: at,
                });
            }
            return;
        }
        let in_place = switching.is_none()
            && self
                .playing
                .as_ref()
                .is_some_and(|t| t.index == heard.index);
        if !in_place {
            // The track keeps the route it is heard on, or goes on to the
            // one it is switching to.
            let route = switching.unwrap_or_else(|| heard.route.clone());
            self.jump_on(heard.index, StartAt::Seconds(seconds), Some(route));
            if let Some(h) = &self.heard {
                let at = h.seconds_at(h.seg_out);
                self.emit(Event::Seeked {
                    index: h.index,
                    seconds: at,
                });
            }
            return;
        }
        let frame = (seconds.max(0.0) * f64::from(heard.info.rate)).round() as u64;
        let result = (|| {
            let out = self.out.as_mut().ok_or(Error::Stopped)?;
            out.output.flush()?;
            let written = out.output.written();
            let track = self.playing.as_mut().expect("checked");
            track.restart_at(frame)?;
            track.seg_out = written;
            Ok::<_, Error>((written, track.seg_frame))
        })();
        match result {
            Ok((seg_out, seg_frame)) => {
                self.cues.clear();
                self.next = None;
                self.finishing = false;
                self.tail = None;
                let heard = self.heard.as_mut().expect("checked");
                heard.seg_out = seg_out;
                heard.seg_frame = seg_frame;
                let (index, at) = (heard.index, heard.seconds_at(seg_out));
                self.emit(Event::Seeked { index, seconds: at });
                self.report_position(true);
            }
            Err(error) if error.is_skippable() => {
                self.emit(Event::Error {
                    index: Some(heard.index),
                    error,
                    skipped: true,
                });
                self.jump(heard.index + 1, 0.0);
            }
            Err(error) => self.fail(Some(heard.index), error),
        }
    }

    /// Stops playback and closes the output.
    fn stop(&mut self, announce: bool) {
        let was_active = self.out.is_some() || self.playing.is_some();
        self.prestart = None;
        self.shadow = None;
        self.clear_timeline();
        self.close_output();
        self.paused = false;
        if let Some(heard) = self.heard.take() {
            if announce {
                self.emit(Event::TrackEnded {
                    index: heard.index,
                    finished: false,
                });
            }
        }
        if announce && was_active {
            self.emit(Event::State(PlayState::Stopped));
        }
    }

    fn fail(&mut self, index: Option<usize>, error: Error) {
        self.emit(Event::Error {
            index,
            error,
            skipped: false,
        });
        self.stop(true);
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

    /// Installs the track that waited for the output to play out. One that
    /// switched to another format goes back to its old route, at the same
    /// frame, when its output does not open (a notice says so).
    fn install_next(&mut self, mut track: Track) -> Result<(), Error> {
        let fallback = track.fallback.take();
        let (index, frame, route) = (track.index, track.seg_frame, track.route.clone());
        let Err(error) = self.install(track) else {
            return Ok(());
        };
        let Some(back) = fallback else {
            return Err(error);
        };
        let again = self
            .prepare_at(index, frame, Some(back.clone()))
            .and_then(|track| self.install(track));
        if again.is_err() {
            return Err(error);
        }
        self.fire_cues();
        let message = format!(
            "{} did not open for {}: {error}; keeping {} for this track",
            self.target.name(),
            route.describe(),
            back.describe()
        );
        self.notice(Some(index), message);
        Ok(())
    }

    /// `track` failed mid-way (`error`, one that costs only the track):
    /// what it decoded plays (with the resampler's tail), then the error is
    /// reported and the next track goes on.
    fn broke(&mut self, track: &mut Track, error: Error) {
        track.source_done = true;
        if track.resampler.as_ref().is_some_and(|r| !r.is_finished()) {
            let before = track.out_buf.len();
            if let Some(resampler) = &mut track.resampler {
                resampler.finish(&mut track.out_buf);
            }
            self.after_output(track, before);
        }
        track.failure = Some(error);
    }

    fn step(&mut self, track: &mut Track) -> Result<Progress, Error> {
        let (back, ahead) = self.window_frames(track);
        if track.out_at < track.out_buf.len() {
            // A jump waiting for its time: written no further than it.
            let limit = self
                .jump_limit(track)
                .map_or(track.out_buf.len(), |l| l.min(track.out_buf.len()));
            let mut frames = 0;
            if track.out_at < limit {
                let out = self.out.as_mut().ok_or(Error::Stopped)?;
                let channels = usize::from(out.info.format.channels);
                frames = out
                    .output
                    .write(&track.out_buf[track.out_at..limit], Some(WRITE_WAIT))?;
                self.stalled = (frames == 0).then(Instant::now);
                track.out_at += frames * channels;
                Self::restore_faded(track, false);
            }
            // The output is full: the window's stretch ahead is decoded (a
            // stems route takes what its stems have separated).
            if frames == 0 && ahead > 0 && !track.ahead_done && track.unwritten() < ahead {
                if let Progress::Complete = self.produce(track, Duration::ZERO)? {
                    track.ahead_done = true;
                }
            }
            if frames == 0 {
                if let Some(run) = track.stems.as_mut() {
                    run.fetch(0, Duration::ZERO);
                }
            }
            return Ok(Progress::Busy);
        }
        // Written: what the window keeps stays, the rest goes.
        Self::trim_written(track, back);
        if let Some(error) = track.failure.take() {
            let end = self.out.as_ref().map_or(0, |o| o.output.written());
            self.cues.push_back(Cue {
                at: end,
                event: CueEvent::Skipped {
                    index: track.index,
                    error,
                },
            });
            return Ok(Progress::Complete);
        }
        self.produce(track, WRITE_WAIT)
    }

    /// Decodes one piece of `track` and appends its output to `out_buf`
    /// (a stem mix waits up to `wait` for its stems).
    fn produce(&mut self, track: &mut Track, wait: Duration) -> Result<Progress, Error> {
        if track.route.stage == Stage::Stems {
            return self.step_stems(track, wait);
        }
        self.step_plain(track)
    }

    /// One read of plain PCM: resampled into the output buffer when the
    /// route has a resampler (unquantized when the file's samples are not
    /// 24-bit integers: lossy, floating-point, 32-bit, so that they are
    /// requantized once), appended to it as read otherwise. Once the source has ended the
    /// resampler is flushed, then the track is complete.
    fn step_plain(&mut self, track: &mut Track) -> Result<Progress, Error> {
        let before = track.out_buf.len();
        if !track.source_done {
            let frames = match &mut track.resampler {
                Some(resampler) if !track.info.fits_24_bits() => {
                    track.wide_buf.clear();
                    let frames = track.source.read_unquantized(&mut track.wide_buf)?;
                    resampler.process_float(&track.wide_buf, &mut track.out_buf);
                    frames
                }
                Some(resampler) => {
                    track.read_buf.clear();
                    let frames = track.source.read(&mut track.read_buf)?;
                    resampler.process(&track.read_buf, &mut track.out_buf);
                    frames
                }
                None => track.source.read(&mut track.out_buf)?,
            };
            if frames > 0 {
                self.after_output(track, before);
                return Ok(Progress::Busy);
            }
            track.source_done = true;
        }
        match &mut track.resampler {
            Some(resampler) if !resampler.is_finished() => {
                resampler.finish(&mut track.out_buf);
                self.after_output(track, before);
                Ok(Progress::Busy)
            }
            _ => Ok(Progress::Complete),
        }
    }

    /// One piece of a [`Stage::Stems`] route: the stems that are on, summed,
    /// resampled or requantized into the output buffer (waiting up to
    /// `wait` for the stems).
    fn step_stems(&mut self, track: &mut Track, wait: Duration) -> Result<Progress, Error> {
        let Some(separator) = track.separator.or(self.separator) else {
            track.stems_error = Some("no separator".into());
            return Ok(Progress::Busy);
        };
        let before = track.out_buf.len();
        if track.stems_done {
            if let Some(resampler) = track.resampler.as_mut().filter(|r| !r.is_finished()) {
                resampler.finish(&mut track.out_buf);
                discard_made(track, before);
                self.after_output(track, before);
                return Ok(Progress::Busy);
            }
            return Ok(Progress::Complete);
        }
        let keep = self.stem_keep(track);
        let run = match track.stems.take() {
            Some(run) => run,
            None => match self.prestart.take() {
                Some(run) if run.path == track.info.path && run.start == track.seg_frame => run,
                other => {
                    self.prestart = other;
                    match self.take_shadow(track) {
                        Some(run) => run,
                        None => StemRun::start(
                            track.info.path.clone(),
                            track.seg_frame,
                            separator,
                            self.stem_store.clone(),
                        ),
                    }
                }
            },
        };
        let run = track.stems.insert(run);
        run.set_keep(keep.0, keep.1);
        track.wide_buf.clear();
        let origin = run.start as f64 / f64::from(track.info.rate.max(1));
        let gains = track.gains.as_ref().map(|g| (g, origin));
        // A change at a time of this track: the stems before it until its
        // frame of the run, the track's time at the change's output frame
        // as it is heard now (a track decoded ahead for a switch is heard
        // on the playing one's timeline).
        let (mix, change) = match self.stems_change {
            Some(c) if c.index == track.index => {
                let timeline =
                    |t: &Track| (t.seg_frame, t.info.rate, t.seg_out, t.route.output_rate);
                let (frame, rate, seg_out, out_rate) = match self.playing.as_ref() {
                    Some(playing) if track.pending => timeline(playing),
                    _ => (
                        track.seg_frame,
                        track.info.rate,
                        track.seg_out,
                        track.route.output_rate,
                    ),
                };
                let seconds = frame as f64 / f64::from(rate.max(1))
                    + (c.out as f64 - seg_out as f64) / f64::from(out_rate.max(1));
                let at = ((seconds - origin) * f64::from(crate::stems::RATE))
                    .round()
                    .max(0.0) as u64;
                (c.before, Some((at, self.settings.stems)))
            }
            _ => (self.settings.stems, None),
        };
        let pull = run.mix(STEM_FRAMES, mix, change, gains, &mut track.wide_buf, wait);
        let read_all = run.read_all();
        if let Some(warp) = track.warp.as_mut() {
            match pull {
                Pull::Mixed => warp.read(&mut track.wide_buf),
                // What the reader holds comes out before the end.
                Pull::End => {
                    warp.end();
                    warp.read(&mut track.wide_buf);
                }
                _ => {}
            }
            if matches!(pull, Pull::End) && !track.wide_buf.is_empty() {
                match track.resampler.as_mut() {
                    Some(resampler) => resampler.process_float(&track.wide_buf, &mut track.out_buf),
                    None => {
                        let quantize = &mut track.requantizer;
                        track
                            .out_buf
                            .extend(track.wide_buf.iter().map(|&y| quantize.quantize(y)));
                    }
                }
                self.after_output(track, before);
                track.stems_done = true;
                return Ok(Progress::Busy);
            }
        }
        match pull {
            Pull::Mixed => {
                match track.resampler.as_mut() {
                    Some(resampler) => resampler.process_float(&track.wide_buf, &mut track.out_buf),
                    None => {
                        let quantize = &mut track.requantizer;
                        track
                            .out_buf
                            .extend(track.wide_buf.iter().map(|&y| quantize.quantize(y)));
                    }
                }
                discard_made(track, before);
                self.after_output(track, before);
            }
            Pull::Waiting => {}
            Pull::End => track.stems_done = true,
            Pull::Failed(why) => track.stems_error = Some(why),
        }
        // The next track's stems start once this file is read.
        let next = self.queue.get(track.index + 1);
        let stale = self.prestart.as_ref().is_none_or(|p| Some(&p.path) != next);
        if read_all && stale {
            let store = self.stem_store.clone();
            self.prestart = next.map(|path| StemRun::start(path.clone(), 0, separator, store));
        }
        Ok(Progress::Busy)
    }

    /// The switch's turn, when it has one: the new route decodes ahead
    /// while the output holds at least half its buffer, is paused, or took
    /// nothing of the last write ([`SWITCH_TURN`]), and meets the old one
    /// once it has caught up; once the old one is written up to where they
    /// meet, the new one takes over. Returns whether it took the turn
    /// (otherwise the old route writes or decodes).
    fn work_switch(&mut self) -> Result<bool, Error> {
        let Some(switch) = &self.switch else {
            return Ok(false);
        };
        if switch.spliced {
            if self.playing.as_ref().map_or(0, Track::unwritten) > 0 {
                return Ok(false);
            }
            self.hand_over();
            return Ok(true);
        }
        let Some(out) = &self.out else {
            return Ok(false);
        };
        let buffered = out.output.written().saturating_sub(out.clock.position());
        let ring = out.info.ring_frames;
        let turn = self.paused
            || self.offline
            || (ring > 0 && buffered.saturating_mul(2) >= ring)
            || self.stalled.is_some_and(|at| at.elapsed() < SWITCH_TURN);
        if !turn {
            return Ok(false);
        }
        let mut switch = self.switch.take().expect("checked");
        let before = switch.track.emitted;
        if !switch.done {
            match self.produce(&mut switch.track, Duration::ZERO) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => switch.done = true,
                Err(error) if error.is_skippable() => {
                    // What it decoded before the error plays, then the
                    // error, as on its own.
                    self.broke(&mut switch.track, error);
                    switch.done = true;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(reason) = switch.track.stems_error.take() {
            // The stems did not start: the track plays on as it is, and
            // every track does until the mix changes.
            self.stems_failed = true;
            self.prestart = None;
            let index = switch.track.index;
            self.emit(Event::StemsStopped { index, reason });
            return Ok(true);
        }
        let waiting = switch.track.route.stage == Stage::Stems
            && switch.track.emitted == before
            && !switch.done;
        self.switch = Some(switch);
        self.try_splice()?;
        Ok(!waiting)
    }

    /// The switch's new route failed before it was heard (`why`): it
    /// decodes once more from the start of the track (a decoder may refuse
    /// to start mid-stream), or else the track keeps its route.
    fn switch_failed(&mut self, switch: Switch, why: &str) {
        let index = switch.track.index;
        let route = switch.track.route.clone();
        let start = switch.track.seg_frame;
        let first = start % switch.grid;
        drop_later(switch.track);
        if !switch.retried && start > first {
            if let Ok(mut track) = self.prepare_at(index, first, Some(route.clone())) {
                track.pending = true;
                self.switch = Some(Switch {
                    track,
                    retried: true,
                    done: false,
                    spliced: false,
                    ..switch
                });
                return;
            }
        }
        let current = self
            .playing
            .as_ref()
            .map_or_else(String::new, |t| t.route.describe());
        let message = format!(
            "{} failed on this track ({why}); keeping {current} for this track",
            route.describe()
        );
        self.notice(Some(index), message);
    }

    /// Meets the old route with the switch's new one if it has caught up
    /// ([`Engine::splice`]); drops the switch when the track ends first.
    fn try_splice(&mut self) -> Result<(), Error> {
        let Some(mut switch) = self.switch.take() else {
            return Ok(());
        };
        let Some(mut old) = self.playing.take() else {
            return Ok(());
        };
        let result = self.splice(&mut old, &mut switch);
        self.playing = Some(old);
        match result? {
            Splice::Wait | Splice::Done => self.switch = Some(switch),
            Splice::Cancel => {
                // A new route that failed before where they would meet.
                match switch.track.failure.take() {
                    Some(error) => self.switch_failed(switch, &error.to_string()),
                    None => drop_later(switch.track),
                }
            }
        }
        Ok(())
    }

    /// Where the old route and the new one meet: at the first track frame
    /// on the grid ([`Switch::grid`]) heard at or after the frame being
    /// written (and, at another format, a fade's length after it), so that
    /// output frame `at` on the old route and frame `j` of the new one are
    /// the same track frame exactly. Once the new route has its output
    /// from there (a crossfade's worth, or at another format enough to
    /// start the new output), the old one is cut at `at`; at the same
    /// format the two crossfade from there, at another the old one fades
    /// out before it and the new one fades in.
    fn splice(&mut self, old: &mut Track, switch: &mut Switch) -> Result<Splice, Error> {
        /// Output frames a pending track keeps before what it may still
        /// meet the old route at, before it drops them.
        const KEEP: u64 = 65_536;
        let Some(out) = &self.out else {
            return Ok(Splice::Wait);
        };
        let written = out.output.written();
        let new_rate = switch.track.route.output_rate;
        let fade_old = frames_of(self.switch_fade, old.route.output_rate);
        let fade_new = frames_of(self.switch_fade, new_rate);
        let (a_old, b_old) = grid(&old.route);
        let (a_new, b_new) = grid(&switch.track.route);
        let step_out = switch.grid / a_old * b_old;
        let start = switch.track.seg_frame;
        // An old track read along a map (the playing one along a
        // transition's glide, called off) has no grid with the output: it
        // meets the new one at output frame `from` itself, where its map
        // puts it (on the new route's grid; the crossfade takes the rest).
        let along = self.heard_along(old).cloned();
        // The grid point at or after output frame `from`, as (track frame,
        // output frame), and the new route's output frame for it.
        let meet = |from: u64| {
            if let Some(h) = &along {
                let frame = start + h.frame_at(from).saturating_sub(start) / a_new * a_new;
                let j = (frame - start) / a_new * b_new;
                return (frame, from.max(old.seg_out), j);
            }
            let n = from.saturating_sub(old.seg_out).div_ceil(step_out);
            let frame = old.seg_frame + n * switch.grid;
            let j = frame.saturating_sub(start) / a_new * b_new;
            (frame, old.seg_out + n * step_out, j)
        };
        let guard = if switch.fits { 0 } else { fade_old };
        let (frame, at, j) = meet(written + guard);
        if old.info.frames.is_some_and(|total| frame >= total) {
            return Ok(Splice::Cancel);
        }
        let new = &mut switch.track;
        let channels = usize::from(new.route.channels);
        let base = new.emitted - (new.out_buf.len() / channels) as u64;
        let lead = if switch.fits {
            fade_new
        } else {
            frames_of(SWITCH_PREFILL, new_rate).max(fade_new)
        };
        if (new.emitted < j + lead || new.emitted <= j) && !switch.done {
            // Not caught up: drop what is before the earliest frame it can
            // still meet the old route at.
            // (It may not have made that much yet.)
            let (_, _, earliest) = meet(written);
            let drop = earliest.min(new.emitted);
            if drop > base + KEEP {
                new.out_buf.drain(..(drop - base) as usize * channels);
            }
            return Ok(Splice::Wait);
        }
        if new.emitted <= j {
            // The new route ends before it.
            return Ok(Splice::Cancel);
        }
        let ahead = new.emitted - j;
        let cross = if switch.fits { fade_old.min(ahead) } else { 0 };
        // The old route's output up to there (and through the crossfade).
        while old.seg_out + old.emitted < at + cross {
            if old.failure.is_some() || old.stems_error.is_some() {
                return Ok(Splice::Cancel);
            }
            match self.produce(old, WRITE_WAIT) {
                Ok(Progress::Busy) => {}
                Ok(Progress::Complete) => return Ok(Splice::Cancel),
                Err(error) if error.is_skippable() => {
                    self.broke(old, error);
                    return Ok(Splice::Cancel);
                }
                Err(error) => return Err(error),
            }
        }
        // The old route stops at `at`.
        let old_channels = usize::from(old.route.channels);
        let keep = old.out_at + (at - written) as usize * old_channels;
        let tail = old.out_buf[keep..keep + cross as usize * old_channels].to_vec();
        old.out_buf.truncate(keep);
        if !switch.fits {
            let from = keep - fade_old as usize * old_channels;
            fade(&mut old.out_buf[from..], old_channels, false);
        }
        let index = old.index;
        self.cues.retain(|c| {
            c.at < at
                || !matches!(
                    &c.event,
                    CueEvent::Clipped { index: i, .. } if *i == index
                ) && !matches!(&c.event, CueEvent::Started(s, _) if s.index == index)
        });
        // The new one starts there.
        let new = &mut switch.track;
        new.out_buf.drain(..(j - base) as usize * channels);
        new.out_at = 0;
        if switch.fits {
            crossfade(&tail, &mut new.out_buf[..tail.len()], channels);
        } else {
            let rise = fade_new.min(ahead) as usize * channels;
            fade(&mut new.out_buf[..rise], channels, true);
        }
        new.emitted -= j;
        new.seg_frame = frame;
        new.pending = false;
        new.clip_base = new.clipped();
        new.clips = None;
        // Its cues from there.
        let mut held = VecDeque::new();
        for cue in std::mem::take(&mut new.held) {
            match cue.event {
                CueEvent::Clipped { .. } => {}
                event => held.push_back(Cue {
                    at: cue.at.saturating_sub(j),
                    event,
                }),
            }
        }
        new.held = held;
        if switch.fits {
            new.seg_out = at;
            let cue = self.started(new);
            self.cues.push_back(cue);
            self.release(new);
        } else {
            new.seg_out = 0;
            new.fallback = Some(old.route.clone());
        }
        switch.spliced = true;
        Ok(Splice::Done)
    }

    /// The old route is written up to where the new one begins: the new
    /// one plays on (or, at another format, waits for the output to play
    /// out and open again).
    fn hand_over(&mut self) {
        let Some(switch) = self.switch.take() else {
            return;
        };
        // The old track, written to where the new one begins (its stems
        // kept for the track on its file again).
        if let Some(mut old) = self.playing.take() {
            if old.index == switch.track.index && switch.track.route.stage != Stage::Stems {
                self.shadow = old.stems.take();
            }
            drop_later(old);
        }
        if switch.fits {
            self.playing = Some(switch.track);
        } else {
            self.next = Some(switch.track);
            self.finishing = true;
        }
        if std::mem::take(&mut self.reroute_later) {
            self.reroute();
        }
    }

    /// The stems of `track` stopped (`why`): it goes on, on its usual
    /// route, from the frame after what was written, with nothing flushed
    /// (in the same output when it fits, else after this one plays out);
    /// the stems rest until the mix changes.
    fn stems_stopped(&mut self, track: Track, why: &str) {
        self.cancel_switch();
        let index = track.index;
        let channels = usize::from(track.route.channels);
        let unwritten = (track.out_buf.len() - track.out_at) / channels;
        let written = track.emitted.saturating_sub(unwritten as u64);
        let frame =
            track.seg_frame + rescale(written, track.route.output_rate, track.route.input_rate);
        let rate = f64::from(track.info.rate.max(1));
        drop(track);
        self.stems_failed = true;
        self.prestart = None;
        self.shadow = None;
        self.emit(Event::StemsStopped {
            index,
            reason: why.to_owned(),
        });
        let end = self.out.as_ref().map_or(0, |o| o.output.written());
        match self.prepare(index, frame as f64 / rate) {
            Ok(mut track) => {
                let fits = self
                    .out
                    .as_ref()
                    .is_some_and(|o| track.route.fits(&o.info.format));
                if fits {
                    track.seg_out = end;
                    let cue = self.started(&track);
                    self.cues.push_back(cue);
                    self.playing = Some(track);
                } else {
                    self.next = Some(track);
                    self.finishing = true;
                }
            }
            Err(error) if error.is_skippable() => {
                self.cues.push_back(Cue {
                    at: end,
                    event: CueEvent::Skipped { index, error },
                });
                if let Err(error) = self.advance(index + 1, end) {
                    self.fail(None, error);
                }
            }
            Err(error) => self.fail(Some(index), error),
        }
    }

    /// Counts the output frames just made and queues a clip count when the
    /// source's requantization or the resampler clipped (at most four a
    /// second of output).
    fn after_output(&mut self, track: &mut Track, before: usize) {
        let start = track.seg_out + track.emitted;
        let made = (track.out_buf.len() - before) / usize::from(track.route.channels);
        track.emitted += made as u64;
        let clipped = track.clipped().saturating_sub(track.clip_base);
        let spacing = u64::from(track.route.output_rate / 4);
        let due = match track.clips {
            None => clipped > 0,
            Some((count, at)) => clipped > count && start >= at + spacing,
        };
        if due {
            let event = CueEvent::Clipped {
                index: track.index,
                samples: clipped,
            };
            let at = self.push_cue(track, start, event);
            track.clips = Some((clipped, at));
        }
    }

    /// `track` is fully written: queue its end and go on with the next.
    fn finish(&mut self, track: Track) -> Result<(), Error> {
        // Inside a transition's window the next track takes over.
        let Some(track) = self.transition_takes_over(track) else {
            return Ok(());
        };
        // A switch of this track has nothing left to switch.
        self.cancel_switch();
        let end = self.out.as_ref().map_or(0, |o| o.output.written());
        // A cue mapped past the last output frame is never heard after the
        // end; it applies at the end at the latest.
        for cue in self.cues.iter_mut() {
            cue.at = cue.at.min(end);
        }
        self.cues.push_back(Cue {
            at: end,
            event: CueEvent::Ended { index: track.index },
        });
        // Holding: the next track waits to be started; playback stops
        // once this one is heard to its end.
        if self.hold && track.index + 1 < self.queue.len() {
            self.cues.push_back(Cue {
                at: end,
                event: CueEvent::Held {
                    index: track.index + 1,
                },
            });
            self.tail = None;
            self.finishing = true;
            return Ok(());
        }
        self.advance(track.index + 1, end)
    }

    /// Goes on with the first track from `index` that opens, at output
    /// frame `end`: gaplessly when it fits the open output.
    fn advance(&mut self, mut index: usize, end: u64) -> Result<(), Error> {
        loop {
            if index >= self.queue.len() {
                self.cues.push_back(Cue {
                    at: end,
                    event: CueEvent::QueueEnded,
                });
                self.tail = index.checked_sub(1);
                self.finishing = true;
                return Ok(());
            }
            match self.prepare(index, 0.0) {
                Ok(mut track) => {
                    let fits = self
                        .out
                        .as_ref()
                        .is_some_and(|o| track.route.fits(&o.info.format));
                    if fits {
                        track.seg_out = end;
                        let cue = self.started(&track);
                        self.cues.push_back(cue);
                        self.playing = Some(track);
                    } else {
                        self.next = Some(track);
                        self.finishing = true;
                    }
                    return Ok(());
                }
                Err(error) if error.is_skippable() => {
                    self.cues.push_back(Cue {
                        at: end,
                        event: CueEvent::Skipped { index, error },
                    });
                    index += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_steps_are_the_smallest_exact_ones() {
        assert_eq!(gcd(44_100, 48_000), 300);
        for (input, output, want) in [
            (44_100u64, 88_200u64, (1, 2)),
            (44_100, 48_000, (147, 160)),
            (96_000, 44_100, (320, 147)),
            (48_000, 48_000, (1, 1)),
        ] {
            let g = gcd(input, output);
            let (a, b) = (input / g, output / g);
            assert_eq!((a, b), want);
            // `a` track frames are exactly `b` output frames, and no fewer.
            assert_eq!(a * output, b * input);
            for n in 1..a {
                assert_ne!(n * output % input, 0);
            }
        }
    }

    #[test]
    fn fades_sum_to_one_and_end_where_they_should() {
        let frames = 64;
        for i in 0..frames {
            let sum = fade_weight(i, frames) + fade_weight(frames - 1 - i, frames);
            assert!((sum - 1.0).abs() < 1e-12);
        }
        assert!(fade_weight(0, frames) < 0.001);
        assert!(fade_weight(frames - 1, frames) > 0.999);
        let old = vec![1_000; 2 * frames];
        let mut new = vec![-1_000; 2 * frames];
        crossfade(&old, &mut new, 2);
        assert!(new[0] > 990 && new[2 * frames - 1] < -990);
        assert!(new.windows(2).all(|w| w[1] <= w[0]));
    }
}
