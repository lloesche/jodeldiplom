//! Looks at the desk, on the caller's thread: [`Mixer::desk`] (how it
//! stands), [`Mixer::meters`] (the peaks since the last look), and the
//! cheap ones the app asks for often ([`Mixer::passthrough`],
//! [`Mixer::mic`]).

use std::sync::atomic::Ordering;

use super::{Mixer, Role, Timebase};
use crate::format::{SampleFormat, StreamFormat};
use crate::output::StreamInfo;

/// The desk as it stands: the master (the device), the strips with their
/// inputs, and the microphone's port. Looking changes nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct DeskView {
    /// What goes to the device.
    pub master: MasterView,
    /// The strips, in the order they were made (a group before its
    /// children).
    pub strips: Vec<StripView>,
    /// The microphone's port.
    pub mic: MicView,
}

/// What goes to the device.
#[derive(Clone, Debug, PartialEq)]
pub struct MasterView {
    /// The device's name.
    pub device: String,
    /// What it runs at (`None`: closed).
    pub stream: Option<StreamInfo>,
    /// The clock, as of the last frames mixed.
    pub timebase: Timebase,
    /// The last frames went to the device as an input wrote them (or were
    /// silence): nothing summed, limited or dithered.
    pub passthrough: bool,
    /// Samples that clipped when a mix was requantized, since the mixer
    /// started (over every opening of the device).
    pub clipped: u64,
}

/// A strip.
#[derive(Clone, Debug, PartialEq)]
pub struct StripView {
    /// Which strip it is ([`super::Strip::id`]; [`Meters::strips`] tells
    /// its peaks by it).
    pub id: u64,
    /// Its name.
    pub name: String,
    /// The group it sits in, by id (`None`: at the top of the desk).
    pub parent: Option<u64>,
    /// The level in force (linear).
    pub level: f64,
    /// Muted.
    pub mute: bool,
    /// Soloed.
    pub solo: bool,
    /// The inputs plugged into it, in the order they were made.
    pub inputs: Vec<InputView>,
}

/// An input of a strip.
#[derive(Clone, Debug, PartialEq)]
pub struct InputView {
    /// What it is to the device's format.
    pub role: Role,
    /// Its controllers' gain (its [`super::Fader`]'s), at the last frame
    /// mixed.
    pub gain: f64,
    /// It was heard in the last frames mixed: it supplied frames at a gain
    /// other than 0.
    pub heard: bool,
    /// The format of the output open on it (`None`: none).
    pub format: Option<StreamFormat>,
    /// Frames queued in that output's ring.
    pub queued: u64,
    /// The ring's capacity in frames.
    pub capacity: u64,
}

/// The microphone's port: a microphone is only listened to. Its tap on the
/// device hears what plays (the echo canceller's sidechain); nothing of it
/// ever reaches the device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MicPort {
    /// No microphone.
    #[default]
    None,
    /// A microphone, and no Lead output open to hear it with.
    Idle,
    /// The microphone hears what the device plays.
    Hearing,
    /// The device runs no taps: the music's echo stays in the voice.
    CannotHear(String),
}

/// How the microphone's port stands.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MicView {
    /// Its state.
    pub port: MicPort,
    /// Times the device refused its tap, since the mixer started: each one
    /// a new [`MicPort::CannotHear`], even with the same reason.
    pub refused: u64,
}

/// The peaks since the last look ([`Mixer::meters`]), each channel's
/// linear (1.0: full scale), up to [`super::METER_CHANNELS`]; none while
/// the device is closed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meters {
    /// What went to the device.
    pub master: Vec<f32>,
    /// Each strip's subtree (its inputs and its children's) summed after
    /// their gains, by [`StripView::id`], in the order the strips were
    /// made.
    pub strips: Vec<(u64, Vec<f32>)>,
    /// The limiter turned the gain down.
    pub limited: bool,
}

/// A stream format as one word (0: none), for an atomic.
pub(crate) fn pack_format(format: Option<StreamFormat>) -> u64 {
    let Some(f) = format else {
        return 0;
    };
    let sample: u64 = match f.sample {
        SampleFormat::S16 => 1,
        SampleFormat::S24Packed => 2,
        SampleFormat::S24In32 => 3,
        SampleFormat::S24In32Low => 4,
        SampleFormat::S32 => 5,
    };
    u64::from(f.rate) | (u64::from(f.channels) << 32) | (sample << 48)
}

/// [`pack_format`] undone.
pub(crate) fn unpack_format(word: u64) -> Option<StreamFormat> {
    let sample = match word >> 48 {
        1 => SampleFormat::S16,
        2 => SampleFormat::S24Packed,
        3 => SampleFormat::S24In32,
        4 => SampleFormat::S24In32Low,
        5 => SampleFormat::S32,
        _ => return None,
    };
    Some(StreamFormat::new(word as u32, (word >> 32) as u16, sample))
}

impl Mixer {
    /// The desk as it stands, gathered from what the mixer thread
    /// publishes. Looking changes nothing (the meters are
    /// [`Mixer::meters`]).
    pub fn desk(&self) -> DeskView {
        let s = &self.shared;
        let strips = s
            .strips()
            .iter()
            .map(|(&id, entry)| {
                let st = &entry.state;
                StripView {
                    id,
                    name: st.name.clone(),
                    parent: st.parent,
                    level: st.level(),
                    mute: st.mute(),
                    solo: st.solo(),
                    inputs: entry
                        .inputs
                        .values()
                        .map(|i| InputView {
                            role: i.role,
                            gain: f64::from_bits(i.gain.load(Ordering::Acquire)),
                            heard: i.heard.load(Ordering::Relaxed),
                            format: unpack_format(i.format.load(Ordering::Relaxed)),
                            queued: i.queued.load(Ordering::Relaxed),
                            capacity: i.capacity.load(Ordering::Relaxed),
                        })
                        .collect(),
                }
            })
            .collect();
        DeskView {
            master: MasterView {
                device: self.name.clone(),
                stream: s.device().clone(),
                timebase: self.timebase(),
                passthrough: self.passthrough(),
                clipped: s.clipped.load(Ordering::Relaxed),
            },
            strips,
            mic: self.mic(),
        }
    }

    /// The peaks and the limiter since the last look, starting the next.
    /// The one who shows the meters looks; others look at
    /// [`Mixer::desk`].
    pub fn meters(&self) -> Meters {
        let s = &self.shared;
        let channels = s
            .device()
            .as_ref()
            .map_or(0, |d| usize::from(d.format.channels));
        Meters {
            master: s.peaks.take(channels),
            strips: s
                .strips()
                .iter()
                .map(|(&id, e)| (id, e.state.peaks.take(channels)))
                .collect(),
            limited: s.limited.swap(false, Ordering::Relaxed),
        }
    }

    /// Whether the last frames went to the device as an input wrote them
    /// (or were silence): [`MasterView::passthrough`], cheaply.
    pub fn passthrough(&self) -> bool {
        self.shared.passthrough.load(Ordering::Relaxed)
    }

    /// How the microphone's port stands: [`DeskView::mic`], cheaply.
    pub fn mic(&self) -> MicView {
        self.shared.mic().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_pack_into_a_word() {
        assert_eq!(unpack_format(pack_format(None)), None);
        for sample in SampleFormat::ALL {
            let f = StreamFormat::new(352_800, 8, sample);
            assert_eq!(unpack_format(pack_format(Some(f))), Some(f));
        }
    }
}
