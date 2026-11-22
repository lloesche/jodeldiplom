//! The desk the app plays on, and the engine on it: a mixer that owns the
//! output device ([`jodel_output::mixer`]), the player's strip as a group
//! with a strip for each of its stems inside it (drums, bass, vocals,
//! other), the preview's strip, and the engine on a lane input of the
//! player's: its main on the group, each stem's lane on its strip. With
//! every strip as it starts, main alone is heard, at gain 1, and reaches
//! the device bit for bit.
//!
//! Where it plays ([`Output`]): a device, a null output or a memory one.
//! The Mix tab's settings are made on the strips here ([`Engine::apply`]),
//! again on every new mixer, and the desk is looked at for it once a drawn
//! frame ([`Engine::look`]).

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use jodel_output::mixer::{Curve, Fader, MicView, Mixer, MixerOptions, Ramp, Role, Start, Strip};
use jodel_output::{Capabilities, SimClock};
use jodel_playback::{
    any_capabilities, Captures, DeviceTarget, Event, MemoryTarget, NullTarget, OutputTarget,
    Player, PlayerOptions, SpectrumTap,
};

use crate::core::desk::{DeskChange, Look, StripKind};
use crate::core::State;

/// Where the engine plays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    /// A device: `None` the system default, else an id or a name (as
    /// [`jodel_output::find_device`] takes).
    Device(Option<String>),
    /// Nowhere: a null output that accepts every format, on this clock.
    Null(SimClock),
    /// A memory device that keeps what it plays (tests).
    Memory(MemoryOutput),
}

/// A memory device that takes `caps` on `clock` and keeps each stream it
/// plays ([`MemoryTarget`]); every engine started on it (a restart) gets a
/// new one, and [`MemoryOutput::captures`] lists them all.
#[derive(Clone)]
pub struct MemoryOutput {
    caps: Capabilities,
    clock: SimClock,
    opened: Arc<std::sync::Mutex<Vec<Captures>>>,
}

impl MemoryOutput {
    /// A memory device taking `caps`, playing on `clock`.
    pub fn new(caps: Capabilities, clock: SimClock) -> MemoryOutput {
        MemoryOutput {
            caps,
            clock,
            opened: Arc::default(),
        }
    }

    /// What each engine's device played, oldest first.
    pub fn captures(&self) -> Vec<Captures> {
        self.opened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn memory(&self) -> MemoryTarget {
        let target = MemoryTarget::new(self.caps.clone(), self.clock);
        self.opened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(target.captures());
        target
    }
}

impl std::fmt::Debug for MemoryOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MemoryOutput")
    }
}

impl PartialEq for MemoryOutput {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.opened, &other.opened)
    }
}

impl Eq for MemoryOutput {}

impl Output {
    /// The target to open the mixer on, and the id of the device it plays
    /// on (none for the null and the memory ones); why there is none.
    pub(crate) fn target(&self) -> Result<(Box<dyn OutputTarget>, Option<String>), String> {
        Ok(match self {
            Output::Null(clock) => (Box::new(NullTarget::new(any_capabilities(), *clock)), None),
            Output::Memory(memory) => (Box::new(memory.memory()), None),
            Output::Device(query) => {
                let target = DeviceTarget::find(query.as_deref().unwrap_or("default"))
                    .map_err(|e| e.to_string())?;
                let id = target.device().id.clone();
                (Box::new(target), Some(id))
            }
        })
    }
}

/// How long a stem's lane takes to a new gain (the Karaoke tab's voice): as
/// a strip's change, so it never clicks.
const LANE_RAMP: std::time::Duration = std::time::Duration::from_millis(20);

/// The engine and what it plays on. The fields drop in order: the
/// players before the mixer they play on.
pub(crate) struct Engine {
    pub(crate) previews: Option<crate::preview::Previews>,
    pub(crate) player: Player,
    /// The engine's fader on the mixer: over its main and every stem (the
    /// preview's duck).
    pub(crate) lead: Fader,
    /// Each stem's lane's fader ([`StripKind::PARTS`]'s order): the
    /// controllers' gain on top of its strip (the Karaoke tab's voice).
    lanes: Vec<Fader>,
    /// The desk's strips ([`StripKind::index`]), the Mix tab's faders.
    strips: Vec<Strip>,
    /// The microphone's port as last told.
    pub(crate) mic: MicView,
    /// The mixer's monitor: what the device plays.
    pub(crate) tap: SpectrumTap,
    pub(crate) mixer: Mixer,
}

impl Engine {
    /// The mixer on `target` (which takes `capabilities`), its strips, and
    /// the engine on the player's lane input with `options`; the engine's
    /// events.
    pub(crate) fn start(
        target: Box<dyn OutputTarget>,
        capabilities: Option<Capabilities>,
        options: PlayerOptions,
    ) -> (Engine, Receiver<Event>) {
        let mixer = Mixer::open(
            target,
            MixerOptions {
                capabilities,
                ..MixerOptions::default()
            },
        );
        // The fine bass block at 384 kHz, plus the device buffer that runs
        // ahead of what is heard.
        let tap = SpectrumTap::new(1 << 19);
        mixer.set_monitor(Some(Box::new(tap.clone())));
        let strips: Vec<Strip> = {
            let player = mixer.strip(StripKind::Player.name());
            let mut strips = vec![player.clone()];
            for kind in StripKind::ALL.into_iter().skip(1) {
                strips.push(match kind.parent() {
                    Some(_) => mixer.strip_in(&player, kind.name()),
                    None => mixer.strip(kind.name()),
                });
            }
            strips
        };
        let parts: Vec<&Strip> = StripKind::PARTS
            .iter()
            .map(|k| &strips[k.index()])
            .collect();
        let (input, lead, lanes) =
            mixer.input_lanes(&strips[StripKind::Player.index()], &parts, Role::Lead, 1.0);
        let (player, events) = Player::new(Box::new(input), options);
        let engine = Engine {
            previews: None,
            player,
            lead,
            lanes,
            strips,
            mic: MicView::default(),
            tap,
            mixer,
        };
        (engine, events)
    }

    /// The strip of `kind`.
    pub(crate) fn strip(&self, kind: StripKind) -> &Strip {
        &self.strips[kind.index()]
    }

    /// Sets a strip of the mixer as the Mix tab has it, or a stem's lane.
    pub(crate) fn apply(&self, change: DeskChange) {
        match change {
            DeskChange::Level(kind, gain) => self.strip(kind).set_level(gain),
            DeskChange::Mute(kind, on) => self.strip(kind).set_mute(on),
            DeskChange::MuteAt(kind, on, frame) => {
                // On that frame of the player's output, as it plays now (at
                // once before any of it is mixed).
                let start = self.lead.at_stream(frame);
                self.strip(kind).set_mute_at(on, start);
            }
            DeskChange::Solo(kind, on) => self.strip(kind).set_solo(on),
            DeskChange::Lane(kind, gain) => {
                if let Some(lane) = kind.lane().and_then(|k| self.lanes.get(k)) {
                    lane.set(Ramp {
                        start: Start::Now,
                        to: gain,
                        over: LANE_RAMP,
                        curve: Curve::Linear,
                    });
                }
            }
        }
    }

    /// Every setting of the desk on a new mixer (its strips start at 0
    /// dB).
    pub(crate) fn set_up(&self, state: &State) {
        for change in crate::core::desk::changes(state) {
            self.apply(change);
        }
    }

    /// The Mix tab's look for a frame: the desk and the meters since the
    /// last look (the one place they are taken), each fader strip found by
    /// its id.
    pub(crate) fn look(&self) -> Look {
        let ids = StripKind::ALL.map(|kind| self.strip(kind).id());
        Look::of(self.mixer.desk(), self.mixer.meters(), ids)
    }
}
