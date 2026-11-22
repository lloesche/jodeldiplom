//! The engine's interface: the [`Command`]s it takes, the [`Event`]s it
//! sends, and the [`Player`] handle, which starts the decode thread and
//! queues commands for it.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use jodel_output::{Capabilities, StreamInfo};

use crate::error::Error;
use crate::rate::{Route, Settings};
use crate::source::TrackInfo;
use crate::stem_file::StemStore;
use crate::stems::{SeparatorSource, StemMix};
use crate::target::OutputTarget;
use crate::transition::{TransitionPlan, TransitionState};

use super::jump::JumpAt;
use super::lanes::MakeStems;
use super::Engine;

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
    /// When the stems of the tracks played are made as parts of the output
    /// (on an output of lanes, [`jodel_output::lanes`]: main, then drums,
    /// bass, vocals and other, and whether they are there): never (the
    /// default), from the stems kept only, or separated where they are not
    /// kept. The output is the same either way; a mixer may play the parts
    /// in place of main.
    MakeStems(MakeStems),
    /// The stems as a mixer plays them, for what a transition mixes inside
    /// the engine (its two tracks on their stems, faded by its lanes, are
    /// heard as main, so a mixer's stem levels do not reach them there):
    /// each stem's gain in [`crate::stems::LANES`]'s order (0 for one
    /// muted), from `at` seconds into the playing track (a bar's first
    /// beat), else at once. Outside a transition it changes nothing heard;
    /// with gains other than one, the next track's return to its own route
    /// after a transition waits until its stems are parts of the output
    /// again, so a mixer can play them on from there.
    StemMask {
        /// Each stem's gain, drums, bass, vocals, other.
        gains: [f64; crate::stems::STEMS],
        /// Seconds into the playing track; `None`: at once.
        at: Option<f64>,
    },
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
    /// The stems of a track as parts of the output ([`Command::MakeStems`])
    /// failed: the track has none (its output is main alone) until they
    /// are asked for again.
    PartsStopped {
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

    /// [`Command::MakeStems`].
    pub fn make_stems(&self, make: MakeStems) -> Result<(), Error> {
        self.send(Command::MakeStems(make))
    }

    /// [`Command::StemMask`].
    pub fn stem_mask(
        &self,
        gains: [f64; crate::stems::STEMS],
        at: Option<f64>,
    ) -> Result<(), Error> {
        self.send(Command::StemMask { gains, at })
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
