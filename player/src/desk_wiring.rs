//! The desk the app plays on (its mixer's graph), and the engine on it: a
//! mixer that owns the
//! output device ([`jodel_output::mixer`]), the player's strip as a group
//! with a strip for each of its stems inside it (drums, bass, vocals,
//! other), the preview's strip, and the engine on two heads, each a lane
//! input of the player's ([`StemDesk::two_heads`]): its main on the group,
//! each stem's lane on its strip; and the karaoke game's effects input
//! on a strip of its own ([`crate::sfx`]). The engine plays on the head on air, the
//! desk's Lead; a transition's next track on the other, and the two swap at
//! its hand-over. The user's strips act on both. With every strip as it
//! starts, main alone is heard, at gain 1, and reaches the device bit for
//! bit.
//!
//! Where it plays ([`Output`]): a device, a null output or a memory one.
//! The Mixer tab's settings are made on the strips here ([`Engine::apply`]),
//! again on every new mixer, and the desk is looked at for it once a drawn
//! frame ([`Engine::look`]).

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use jodel_output::mixer::{
    Automation, Curve, Effects, Fader, GainCurve, MicPort, MicView, Mixer, MixerOptions, Ramp,
    Start, Strip,
};
use jodel_output::{Capabilities, SimClock};
use jodel_playback::{
    any_capabilities, Captures, DeviceTarget, Event, HeadFaders, MemoryTarget, NullTarget,
    OutputTarget, Player, PlayerOptions, SpectrumTap, StemDesk,
};

use crate::core::desk::{DeskChange, Look, StripKind};
use crate::core::voice_curve::VoiceCurve;
use crate::core::{DeviceChoice, HeadFrame, State};

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
            Output::Null(clock) => (
                Box::new(NullTarget::new(any_capabilities(48_000), *clock)),
                None,
            ),
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
    /// The engine's heads' faders, by head id
    /// ([`jodel_playback::TrackStart::head`]): the first is the one it
    /// starts on. Its input's fader is the preview's duck, each stem's
    /// lane's ([`StripKind::PARTS`]' order) the Karaoke tab's voice.
    heads: [HeadFaders; 2],
    /// The desk's strips ([`StripKind::index`]), the Mixer tab's faders.
    strips: Vec<Strip>,
    /// The group every strip is in: its level is the player's own volume
    /// ([`crate::core::desk::DeskState::volume`]).
    main: Strip,
    /// The microphone's port as last told.
    pub(crate) mic: MicView,
    /// The karaoke game's sounds, and the effects input they play on (on
    /// the Effects strip).
    sfx: crate::sfx::Sfx,
    effects: Effects,
    /// The mixer's monitor: what the device plays.
    pub(crate) tap: SpectrumTap,
    pub(crate) mixer: Mixer,
}

impl Engine {
    /// The mixer on `target` (which takes `capabilities`), its device
    /// opened at once at the system's mix rate, its strips, and the engine
    /// on the player's two heads with `options`, routing every track to the
    /// desk's one format; the engine's events.
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
        // The player's group and its stems' strips, and the preview's: by
        // kind ([`StripKind::index`]), all in the main group.
        let main = mixer.strip("Main");
        let (desk, lead, guest) =
            StemDesk::two_heads(&mixer, Some(&main), StripKind::Player.name(), 1.0);
        let strips = StripKind::ALL
            .map(|kind| match (kind, kind.stem()) {
                (StripKind::Player, _) => desk.group.clone(),
                (_, Some(stem)) => desk.stem(stem).clone(),
                _ => mixer.strip_in(&main, kind.name()),
            })
            .to_vec();
        let heads = [lead.faders.clone(), guest.faders.clone()];
        let effects = mixer.effects(&strips[StripKind::Effects.index()]);
        let (player, events) = Player::on_desk(lead, guest, options);
        let engine = Engine {
            previews: None,
            player,
            heads,
            strips,
            main,
            mic: MicView::default(),
            sfx: crate::sfx::Sfx::default(),
            effects,
            tap,
            mixer,
        };
        (engine, events)
    }

    /// The strip of `kind`.
    pub(crate) fn strip(&self, kind: StripKind) -> &Strip {
        &self.strips[kind.index()]
    }

    /// Both heads' input faders (the song, for the preview to duck): the
    /// one on air and a transition's next track's alike.
    pub(crate) fn song_faders(&self) -> Vec<Fader> {
        self.heads.iter().map(|h| h.input.clone()).collect()
    }

    /// Sets a strip of the mixer as the Mixer tab has it, or a stem's lane.
    pub(crate) fn apply(&self, change: DeskChange) {
        match change {
            DeskChange::Level(kind, gain) => self.strip(kind).set_level(gain),
            DeskChange::Volume(gain) => self.main.set_level(gain),
            DeskChange::Mute(kind, on) => self.strip(kind).set_mute(on),
            DeskChange::MuteAt(kind, on, at) => {
                // On that frame of that head's output, as it plays now (at
                // once before any of it is mixed).
                self.strip(kind).set_mute_at(on, self.at_stream(at));
            }
            DeskChange::Solo(kind, on) => self.strip(kind).set_solo(on),
            DeskChange::Lane(kind, gain) => {
                // On both heads: the voice holds through a transition.
                let lanes = self.heads.iter().filter_map(|h| h.lanes.get(kind.lane()?));
                for lane in lanes {
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

    /// The voice's curve over the song's words on its head's input: the
    /// vocals part's automation ([`crate::core::voice_curve`]), or its end.
    /// The head's input carries no other automation then (no transition is
    /// armed or mixing).
    pub(crate) fn voice_curve(&self, curve: VoiceCurve) {
        match curve {
            VoiceCurve::Set { head, points } => {
                let vocals = StripKind::Vocals.lane();
                let (Some(h), Some(k), Ok(curve)) =
                    (self.heads.get(head), vocals, GainCurve::new(&points))
                else {
                    return;
                };
                let _ = h.input.automate(Automation::new().part(k, curve));
            }
            VoiceCurve::Clear { head } => {
                if let Some(h) = self.heads.get(head) {
                    h.input.clear_automation();
                }
            }
        }
    }

    /// When `at` is mixed, as its head's output plays now. A head the
    /// engine has not is a bug; it is taken as now.
    fn at_stream(&self, at: HeadFrame) -> Start {
        let Some(head) = self.heads.get(at.head) else {
            debug_assert!(false, "the engine has no head {}", at.head);
            return Start::Now;
        };
        head.input.at_stream(at.frame)
    }

    /// Every setting of the desk on a new mixer (its strips start at 0
    /// dB).
    pub(crate) fn set_up(&self, state: &State) {
        for change in crate::core::desk::changes(state) {
            self.apply(change);
        }
    }

    /// A frame drawn: the karaoke game's new moments cued on the effects
    /// input ([`crate::sfx::Sfx::frame`]).
    pub(crate) fn sound_effects(&mut self, state: &State) {
        let desk = crate::sfx::Desk {
            effects: &self.effects,
            heads: [&self.heads[0].input, &self.heads[1].input],
            timebase: self.mixer.timebase(),
        };
        self.sfx.frame(state, desk);
    }

    /// What the effects input took and dropped.
    pub(crate) fn effects(&self) -> jodel_output::mixer::EffectsView {
        self.effects.view()
    }

    /// What the Synth tab asks played on the effects input
    /// ([`crate::sfx::Sfx::bench`]).
    pub(crate) fn bench(&mut self, state: &State, ask: crate::core::synth_tab::Bench) {
        let desk = crate::sfx::Desk {
            effects: &self.effects,
            heads: [&self.heads[0].input, &self.heads[1].input],
            timebase: self.mixer.timebase(),
        };
        self.sfx.bench(state, desk, ask);
    }

    /// The Mixer tab's look for a frame: the desk and the meters since the
    /// last look (the one place they are taken), each fader strip found by
    /// its id.
    pub(crate) fn look(&self) -> Look {
        let ids = StripKind::ALL.map(|kind| self.strip(kind).id());
        Look::of(self.mixer.desk(), self.mixer.meters(), ids)
    }

    /// Why the microphone cannot hear this output (its echo stays in),
    /// once each time the device refuses the microphone's tap anew.
    pub(crate) fn mic_refused(&mut self) -> Option<String> {
        let mic = self.mixer.mic();
        if mic == self.mic {
            return None;
        }
        self.mic = mic.clone();
        match mic.port {
            MicPort::CannotHear(why) => Some(why),
            _ => None,
        }
    }
}

/// The real devices, each with its mix format in a few words.
pub fn list_devices() -> Result<Vec<DeviceChoice>, String> {
    let devices = jodel_output::devices().map_err(|e| e.to_string())?;
    Ok(devices
        .into_iter()
        .map(|device| {
            let summary = match jodel_output::mix_format(&device.id) {
                Ok(format) => format_summary(format),
                Err(error) => format!("cannot be asked: {error}"),
            };
            let name = if device.is_default {
                format!("{} (default)", device.name)
            } else {
                device.name.clone()
            };
            DeviceChoice {
                id: Some(device.id),
                name,
                summary,
            }
        })
        .collect())
}

/// `48 kHz · F32 · 2 ch`: a device's mix format, the rate the system runs
/// it at first.
pub fn format_summary(format: jodel_output::StreamFormat) -> String {
    format!(
        "{} kHz · {} · {} ch",
        crate::core::text::khz_short(format.rate),
        format.sample,
        format.channels
    )
}
