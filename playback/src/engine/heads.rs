//! Heads: what the engine keeps of one output ([`HeadState`]), and a
//! player's second output on the desk its first is on, where a
//! transition's next track plays ([`super::transition`]).
//!
//! The engine plays on the head on air (its `head`). A player on a desk
//! has another, parked (its `desk`). The parked head takes its turns swapped
//! in ([`Engine::with_parked`]): while a transition is prepared and mixed,
//! the next track is written on it, linked on the desk to the playing one,
//! its cues held (a turn of it queues none: what is engine-wide, the cues
//! and what is heard, stays the head on air's). At the hand-over the two
//! swap for good ([`Engine::swap_heads`]): the next track's head is on air,
//! and the old one's output plays out what it holds and closes
//! ([`Engine::tend_parked`]; as the link's leader it pauses with the head
//! on air), or closes at once when playback moves elsewhere
//! ([`Engine::cut_parked`]). A next track dropped in the window retires so
//! too, once it has faded out.

use std::time::{Duration, Instant};

use jodel_output::mixer::Fader;
use jodel_output::Capabilities;

use super::along::PendingAlong;
use super::jump::{JumpFill, PendingJump};
use super::lanes::StemLanes;
use super::track::{drop_later, OpenOutput, Progress, Switch, Track};
use super::Engine;
use crate::error::Error;
use crate::target::OutputTarget;

/// A player's head on a desk ([`super::Player::on_desk`]): where the engine
/// opens its outputs (a lane input of the desk, [`crate::StemDesk`]) and
/// that input's faders.
pub struct Head {
    /// The input, as a target.
    pub target: Box<dyn OutputTarget>,
    /// Its faders.
    pub faders: HeadFaders,
}

/// The faders of a head's input: they stay that input's whatever plays on
/// it (a transition's hand-over swaps what the heads play, not them).
#[derive(Clone)]
pub struct HeadFaders {
    /// The input's own fader, over main and every stem: by it a transition
    /// links the input to the other head's, automates it (its parts'
    /// curves too) and makes it the desk's Lead or a guest; a controller's
    /// duck goes on top (the preview's).
    pub input: Fader,
    /// Each stem's lane's own fader, in [`crate::stems::LANES`]' order (the
    /// engine leaves them to its controllers: a Karaoke voice on the
    /// vocals, say).
    pub lanes: Vec<Fader>,
}

/// What the engine keeps of one output: its target and what it offers (on
/// a desk, its input's fader), the output open on it, its stems as parts,
/// the track it plays and what moves it (a route switch, a move onto a map
/// waiting for its frame, a jump), a count-in, its stalls.
pub(super) struct HeadState {
    /// Which head it is: 0 the first a player is given, 1 its second.
    pub(super) id: usize,
    pub(super) target: Box<dyn OutputTarget>,
    /// Its input's own fader, on a desk (`None` on a plain target).
    pub(super) fader: Option<Fader>,
    pub(super) caps: Option<Capabilities>,
    pub(super) out: Option<OpenOutput>,
    /// The playing track's stems as parts of the output.
    pub(super) lanes: StemLanes,
    /// The track being decoded.
    pub(super) playing: Option<Track>,
    /// The playing track's move to another route, under way.
    pub(super) switch: Option<Switch>,
    /// The settings changed while a switch was being handed over: route
    /// the track again once it has been.
    pub(super) reroute_later: bool,
    /// A [`super::Command::Along`] waiting for its frame.
    pub(super) along_at: Option<PendingAlong>,
    /// A [`super::Command::Jump`] waiting for its time.
    pub(super) pending_jump: Option<PendingJump>,
    /// A reading of the playing track at a jump's target it does not hold.
    pub(super) jump_fill: Option<Box<JumpFill>>,
    /// Output frames of a count-in's silence still to write, before the
    /// playing track's.
    pub(super) lead: u64,
    /// When the output last took nothing of a write.
    pub(super) stalled: Option<Instant>,
}

impl HeadState {
    pub(super) fn new(
        id: usize,
        target: Box<dyn OutputTarget>,
        caps: Option<Capabilities>,
        lanes: StemLanes,
    ) -> HeadState {
        HeadState {
            id,
            target,
            fader: None,
            caps,
            out: None,
            lanes,
            playing: None,
            switch: None,
            reroute_later: false,
            along_at: None,
            pending_jump: None,
            jump_fill: None,
            lead: 0,
            stalled: None,
        }
    }

    /// Whether its output plays out what it holds, with no track on it (the
    /// head a transition handed over from, or a next track dropped).
    fn retiring(&self) -> bool {
        self.playing.is_none() && self.out.is_some()
    }

    /// Its output closes, and its parts stop.
    fn close(&mut self) {
        if let Some(mut out) = self.out.take() {
            let _ = out.output.close();
        }
        self.lanes.stop();
    }
}

impl Engine {
    /// The engine on a desk: the target given to [`Engine::new`] is the
    /// input whose fader is `first`, and `second` is the other head.
    pub(super) fn with_heads(mut self, first: Fader, second: Head) -> Engine {
        self.head.fader = Some(first);
        let mut parked = HeadState::new(1, second.target, None, self.head.lanes.sibling());
        parked.fader = Some(second.faders.input);
        self.desk = Some(Box::new(parked));
        self
    }

    /// The fader of the head worked on (the one on air, or the parked one
    /// in its turn, [`Engine::with_parked`]), and the other's, on a desk.
    pub(super) fn faders(&self) -> Option<(&Fader, &Fader)> {
        let other = self.desk.as_ref()?.fader.as_ref()?;
        Some((self.head.fader.as_ref()?, other))
    }

    /// The parked head on air and the one on air parked, for good.
    pub(super) fn swap_heads(&mut self) {
        if let Some(parked) = self.desk.as_deref_mut() {
            std::mem::swap(&mut self.head, parked);
        }
    }

    /// Runs `f` with the parked head swapped in (none: `None`). It queues
    /// no cue: the parked head's track holds its own.
    pub(super) fn with_parked<R>(&mut self, f: impl FnOnce(&mut Engine) -> R) -> Option<R> {
        self.desk.as_ref()?;
        self.swap_heads();
        let cues = self.cues.len();
        let result = f(self);
        debug_assert_eq!(self.cues.len(), cues, "a parked head queued a cue");
        self.swap_heads();
        Some(result)
    }

    /// The parked head, on a desk.
    fn parked(&mut self) -> Option<&mut HeadState> {
        self.desk.as_deref_mut()
    }

    /// Opens an output on the parked head for `track`, at its route's
    /// format (the open one's: its route is chosen for that format alone),
    /// with lanes where the target takes them, never paused (its link holds
    /// it), and puts the track there, its cues held.
    pub(super) fn open_parked(&mut self, mut track: Track) -> Result<(), Error> {
        track.pending = true;
        self.with_parked(|e| {
            e.open_output(&track.route, false)?;
            e.head.playing = Some(track);
            Ok(())
        })
        .unwrap_or(Err(Error::Stopped))
    }

    /// Whether the parked head's target takes stems beside main (a next
    /// track is read along its map as them).
    pub(super) fn parked_takes_lanes(&mut self) -> bool {
        self.parked().is_some_and(|p| p.target.lanes() > 1)
    }

    /// Whether the parked head has a track whose output has room: its turn
    /// would write or make some.
    pub(super) fn parked_has_room(&self) -> bool {
        let Some(p) = self.desk.as_deref() else {
            return false;
        };
        let (Some(out), Some(_)) = (&p.out, &p.playing) else {
            return false;
        };
        let ring = out.info.ring_frames;
        ring == 0 || out.output.written().saturating_sub(out.clock.position()) < ring
    }

    /// The parked head's turn while a transition is prepared or mixed, when
    /// its output has room: its track takes a step as the head on air's does
    /// ([`Engine::step`]: what it made written without waiting for room, or
    /// more made). Whether it wrote or made anything, or why it failed.
    pub(super) fn parked_turn(&mut self) -> Result<bool, String> {
        if !self.parked_has_room() {
            return Ok(false);
        }
        let turn = self.with_parked(|e| {
            let mut track = e.head.playing.take().expect("checked");
            let written = e.head.out.as_ref().map_or(0, |o| o.output.written());
            let made = track.emitted;
            let progress = e.step(&mut track);
            let now = e.head.out.as_ref().map_or(0, |o| o.output.written());
            let did = now != written || track.emitted != made;
            e.head.playing = Some(track);
            progress.map(|p| did && matches!(p, Progress::Busy))
        });
        turn.unwrap_or(Ok(false)).map_err(|e| e.to_string())
    }

    /// The parked head's track goes (in the background), and its output
    /// closes now: a transition's next track not heard.
    pub(super) fn drop_parked(&mut self) {
        let Some(p) = self.parked() else {
            return;
        };
        if let Some(track) = p.playing.take() {
            drop_later(track);
        }
        p.along_at = None;
        p.close();
    }

    /// The parked head's track goes, its output playing out what it has
    /// (a next track fading out as its transition is dropped), and closing
    /// then ([`Engine::tend_parked`]).
    pub(super) fn retire_parked(&mut self) {
        if let Some(p) = self.parked() {
            if let Some(track) = p.playing.take() {
                drop_later(track);
            }
            p.along_at = None;
            if let Some(out) = p.out.as_mut() {
                let _ = out.output.drain(Some(Duration::ZERO));
            }
        }
    }

    /// A parked output with no track closes once it has played out what it
    /// holds.
    pub(super) fn tend_parked(&mut self) {
        let Some(p) = self.parked().filter(|p| p.retiring()) else {
            return;
        };
        let out = p.out.as_mut().expect("retiring");
        if !matches!(out.output.drain(Some(Duration::ZERO)), Ok(false)) {
            p.close();
        }
    }

    /// A parked output with no track closes now, played out or not
    /// (playback moves elsewhere).
    pub(super) fn cut_parked(&mut self) {
        if let Some(p) = self.parked().filter(|p| p.retiring()) {
            p.close();
        }
    }

    /// A parked output with no track pauses or resumes with the one on air,
    /// which follows it on the desk, or which it follows.
    pub(super) fn pause_parked(&mut self, pause: bool) -> Result<(), Error> {
        let Some(p) = self.parked().filter(|p| p.retiring()) else {
            return Ok(());
        };
        let out = p.out.as_mut().expect("retiring");
        if pause {
            out.output.pause()?;
        } else {
            out.output.resume()?;
        }
        Ok(())
    }

    /// The parked head's track, for what follows it through the queue.
    pub(super) fn parked_track(&mut self) -> Option<&mut Track> {
        self.parked()?.playing.as_mut()
    }

    /// Every head's stem parts, for what applies to each.
    pub(super) fn every_lanes(&mut self) -> impl Iterator<Item = &mut StemLanes> {
        let parked = self.desk.as_deref_mut().map(|p| &mut p.lanes);
        std::iter::once(&mut self.head.lanes).chain(parked)
    }
}
