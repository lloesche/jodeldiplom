//! The engine's interface: the [`Command`]s it takes, the [`Event`]s it
//! sends, and the [`Player`] handle, which starts the decode thread and
//! queues commands for it.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use jodel_output::StreamInfo;

use crate::error::Error;
use crate::rate::{Route, Settings};
use crate::source::TrackInfo;
use crate::stem_file::StemStore;
use crate::stems::SeparatorSource;
use crate::target::OutputTarget;
use crate::transition::{TransitionPlan, TransitionState};
#[cfg(feature = "test-support")]
use crate::{transition::TempoMode, warp::TimeMap};

use super::heads::Head;
use super::jump::JumpAt;
use super::lanes::MakeStems;
use super::Engine;

/// What a [`Player`] can be told to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Stops and replaces the queue (nothing plays until [`Command::Play`]).
    Load(Vec<PathBuf>),
    /// Plays the queue from `index`, also when paused (as do the other
    /// commands that start a track: only [`Command::Seek`] stays paused).
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
    /// Plays the next track; while a transition is mixed, its next track
    /// plays on alone at once, from where it is heard (the playing one
    /// fading out over a few milliseconds).
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
    /// [`Event::Transition`] says how it goes. The desk mixes it: the next
    /// track plays on the player's second head ([`Player::on_desk`]; a
    /// player without one calls it off: it needs a desk), linked to the
    /// playing one, and the plan's lanes are automation of both heads'
    /// inputs, so the desk's stem levels apply to both, on their frames.
    /// Only moving playback elsewhere (play, seek, next, previous, stop,
    /// load, the next track taken out of the queue) calls it off; new
    /// settings wait for its end. Once it is handed over, the tracks after
    /// play at the output format open, where they have a route at it, until
    /// playback stops, a track is picked (play, previous, next outside a
    /// mix, load), the queue ends or the settings change.
    Transition(Box<TransitionPlan>),
    /// Calls the transition off, if it is not heard yet (a mix under way
    /// plays to its end).
    CancelTransition,
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
    /// is heard at once (zero: none, as it plays otherwise). None while a
    /// transition is set up, nor for a track read along a map.
    JumpWindow {
        /// Seconds kept behind.
        back: f64,
        /// Seconds decoded ahead.
        ahead: f64,
    },
    /// When the stems of the tracks played are made as parts of the output
    /// (on an output of lanes, [`jodel_output::lanes`]: main, then drums,
    /// bass, vocals and other, and whether they are there): never (the
    /// default), from the stems kept only, or separated where they are not
    /// kept. The output is the same either way; a mixer may play the parts
    /// in place of main.
    MakeStems(MakeStems),
    /// Plays the head on air `semitones` higher (lower when negative),
    /// its tempo kept: −6 to +6 ([`crate::KEY_RANGE`], clamped), 0 its own
    /// key. Every stem but the drums is shifted (on an output of lanes the
    /// head plays its stems, its main their sum, as along a time map);
    /// without stems the whole mix is. Heard from the frame the output gives
    /// back to, about as soon as a jump is, crossfaded; at 0 the head goes
    /// back onto its own route and plays the file bit for bit. The key
    /// stays for the tracks the queue runs into, and goes back to 0 when
    /// another track is picked ([`Command::Play`], [`Command::PlayIn`],
    /// [`Command::PlayAt`], [`Command::Next`], [`Command::Previous`]: the
    /// track it is on picked again keeps it, and one asked with nothing
    /// playing is for the track picked next). [`Event::Key`] says the key
    /// asked for, [`Position::key`] the one heard. Refused (a notice says
    /// why) while a transition is set up, for a track of more than two
    /// channels, and off an output of lanes. While shifted, new settings
    /// wait for the key's way back or the next track, and jumps are
    /// refused (a notice says so).
    Key {
        /// Semitones up (down when negative).
        semitones: i8,
    },
    /// For tests (feature `test-support`): the playing track read along a
    /// time map from an output frame on (on an output of lanes, its stems
    /// read along it as its parts, its main their sum), back on its own
    /// route once the map runs straight, as a transition reads its tracks.
    /// A track of more than two channels is not (an [`Event::Notice`] says
    /// why).
    #[cfg(feature = "test-support")]
    Along {
        /// The track's seconds at each second of the output (its frames
        /// over its rate).
        map: Arc<TimeMap>,
        /// The output frame it is read along the map from (the frame
        /// written next, if that is later).
        from: u64,
        /// How it follows the map.
        tempo: TempoMode,
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
    /// The head it plays on: 0 the target [`Player::new`] or
    /// [`Player::on_desk`] is given first, 1 the second (a transition's next
    /// track starts on the other head; [`Position::output_frame`] counts
    /// that head's frames from then).
    pub head: usize,
    /// The file: format, tags, cover.
    pub info: Arc<TrackInfo>,
    /// The route the rate policy chose.
    pub route: Route,
    /// What the output was opened at.
    pub output: StreamInfo,
    /// The output target's name (the device).
    pub device: String,
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
    /// The head whose output frames `output_frame` counts
    /// ([`TrackStart::head`]).
    pub head: usize,
    /// Seconds into the track.
    pub seconds: f64,
    /// The file frame being heard.
    pub frame: u64,
    /// The output frame being heard (as [`jodel_output::Output::position`]
    /// counts them), of the head played on. At a transition's hand-over
    /// it steps back: from there it counts the next track's head's frames,
    /// from its first (heard [`crate::NEXT_LEAD`] before the window); the
    /// transition's [`TransitionState::Started`] is on the playing head's
    /// frames.
    pub output_frame: u64,
    /// The key heard there, in semitones ([`Command::Key`]).
    pub key: i8,
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
    /// The key the head on air plays in from now, in semitones, as asked
    /// ([`Command::Key`]), reset by a track picked, or kept where a change
    /// was refused; heard within about a tenth of a second
    /// ([`Position::key`] says from which frame).
    Key {
        /// Semitones up (down when negative).
        semitones: i8,
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
    /// Where the stems get their separator ([`crate::stems`]): the parts
    /// of an output of lanes ([`Command::MakeStems`]), a head read along a
    /// map's too. `None`: only stems kept are parts (a head along a map has
    /// the whole mix as one, and a transition fades it whole).
    pub separator: Option<SeparatorSource>,
    /// Where tracks' stems are kept ([`crate::stem_file`]): a track found
    /// there plays its kept stems, and only one that is not is separated.
    pub stem_store: Option<Arc<dyn StemStore>>,
    /// How long a switch of route crossfades; zero: they meet at one
    /// frame.
    pub switch_fade: Duration,
    /// The output is not a device playing in real time (a file being
    /// rendered, a test's output): stems are waited for rather than gone
    /// without, a transition's next track's too (on a desk that plays what
    /// it is given at once, its link holds the playing track at the next
    /// track's first frame until it has begun). A device could not wait
    /// (it would run dry); a file loses nothing by it.
    pub offline: bool,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        PlayerOptions {
            settings: Settings::default(),
            position_interval: Duration::from_millis(100),
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
        Player::start(target, None, options)
    }

    /// Starts the engine on a desk ([`crate::StemDesk::two_heads`]): it
    /// plays on `lead`'s input, and a transition's next track on `guest`'s
    /// (the two heads swap at each hand-over). Events arrive on the
    /// receiver.
    pub fn on_desk(lead: Head, guest: Head, options: PlayerOptions) -> (Player, Receiver<Event>) {
        Player::start(lead.target, Some((lead.faders.input, guest)), options)
    }

    fn start(
        target: Box<dyn OutputTarget>,
        desk: Option<(jodel_output::mixer::Fader, Head)>,
        options: PlayerOptions,
    ) -> (Player, Receiver<Event>) {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let mut engine = Engine::new(target, options, event_tx);
        if let Some((fader, second)) = desk {
            engine = engine.with_heads(fader, second);
        }
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

    /// [`Command::PlayAt`].
    pub fn play_at(&self, index: usize, share: f64) -> Result<(), Error> {
        self.send(Command::PlayAt { index, share })
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

    /// [`Command::Jump`].
    pub fn jump(&self, index: usize, to: f64, at: JumpAt) -> Result<(), Error> {
        self.send(Command::Jump { index, to, at })
    }

    /// [`Command::JumpWindow`].
    pub fn jump_window(&self, back: f64, ahead: f64) -> Result<(), Error> {
        self.send(Command::JumpWindow { back, ahead })
    }

    /// [`Command::MakeStems`].
    pub fn make_stems(&self, make: MakeStems) -> Result<(), Error> {
        self.send(Command::MakeStems(make))
    }

    /// [`Command::Key`].
    pub fn key(&self, semitones: i8) -> Result<(), Error> {
        self.send(Command::Key { semitones })
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
