//! The microphone's port, on the mixer thread: the microphone's tap on the
//! device, set for the Lead output open, so the echo canceller hears the
//! whole mix with the stream frames told in the Lead's frames.
//!
//! The mixer thread only decides ([`Port::fit`]). The helper thread
//! (`aside.rs`) makes each tap, sets it on the device, waiting there for
//! the device thread to take it ([`Fit::run`]), and says how the port
//! stands; so the mixer thread never waits, allocates or frees for it.

use std::sync::Arc;

use super::aside::{Aside, Gone};
use super::desk::{MicPort, MicView};
use super::input::{FrameMap, MappedTap};
use super::MixShared;
use crate::input::Input;
use crate::shared::Shared;

/// What the port is fitted for: the microphone (by how many were set), the
/// device's opening (0: closed), and the Lead output.
type Fitting = (u64, u64, Option<u64>);

/// A fitting of the device's tap, done on the helper thread.
pub(super) enum Fit {
    /// The tap on `device` made for `input`, the Lead's frames told from
    /// entry `from` of `map` on.
    On {
        device: Arc<Shared>,
        input: Input,
        map: Arc<FrameMap>,
        from: u64,
    },
    /// No tap: taken off the device that was tapped (if one was), the
    /// port `port`.
    Off {
        tapped: Option<Arc<Shared>>,
        port: MicPort,
    },
}

impl Fit {
    /// Does it, and keeps how the port stands in `view` and `shared`.
    pub(super) fn run(self, view: &mut MicView, shared: &MixShared) {
        let port = match self {
            Fit::On {
                device,
                input,
                map,
                from,
            } => match device.usable() {
                Ok(()) => {
                    let tap = MappedTap::new(Box::new(input.tap()), map, from);
                    device.set_tap(Some(Box::new(tap)));
                    MicPort::Hearing
                }
                Err(e) => {
                    view.refused += 1;
                    MicPort::CannotHear(e.to_string())
                }
            },
            Fit::Off { tapped, port } => {
                if let Some(device) = tapped {
                    device.set_tap(None);
                }
                port
            }
        };
        let seen = MicView {
            port,
            refused: view.refused,
        };
        if seen != *view {
            *view = seen;
            *shared.mic() = view.clone();
        }
    }
}

/// The microphone and its tap on the device.
pub(super) struct Port {
    input: Option<Input>,
    /// Microphones set so far.
    set: u64,
    fitted: Option<Fitting>,
    /// The Lead output the tap is set for, and the device it is on.
    tap: Option<(u64, Arc<Shared>)>,
    /// Where the Lead's frames went on the device, for every tap: each one
    /// reads it from the entry that was next when it was set.
    map: Arc<FrameMap>,
}

impl Port {
    /// A port with no microphone (on the caller's thread: this allocates).
    pub(super) fn new() -> Port {
        Port {
            input: None,
            set: 0,
            fitted: None,
            tap: None,
            map: Arc::new(FrameMap::new()),
        }
    }

    /// The microphone listened to (`None`: none), tapped with the next
    /// [`Port::fit`]. The one before goes to the helper, after the
    /// fittings handed over before (its tap is off the device by the time
    /// the next one is set).
    pub(super) fn set(&mut self, aside: &mut Aside, input: Option<Input>) {
        self.set += 1;
        if let Some(old) = std::mem::replace(&mut self.input, input) {
            aside.drop_later(Gone::Input(old));
        }
    }

    /// The frame map of the tap, when it is set for stream `stream`.
    pub(super) fn map_for(&self, stream: u64) -> Option<&FrameMap> {
        let tapped = self.tap.as_ref().is_some_and(|(s, _)| *s == stream);
        tapped.then_some(&*self.map)
    }

    /// Fits the port to `device` (the device's stream; `None`: closed) of
    /// opening `epoch` and the Lead output `lead`: the tap set for it, or
    /// taken off with no microphone, device or Lead output. Nothing changes
    /// when nothing it is fitted for has.
    pub(super) fn fit(
        &mut self,
        aside: &mut Aside,
        device: Option<&Arc<Shared>>,
        epoch: u64,
        lead: Option<u64>,
    ) {
        let fitting = (self.set, if device.is_some() { epoch } else { 0 }, lead);
        if self.fitted == Some(fitting) {
            return;
        }
        self.fitted = Some(fitting);
        let tapped = self.tap.take().map(|(_, d)| d);
        match (&self.input, device, lead) {
            (Some(input), Some(device), Some(lead)) => {
                // The device tapped before (it may be another opening) is
                // let go of there too.
                if let Some(old) = tapped {
                    aside.drop_later(Gone::Device(old));
                }
                self.tap = Some((lead, device.clone()));
                aside.tap(Fit::On {
                    device: device.clone(),
                    input: input.clone(),
                    map: self.map.clone(),
                    from: self.map.head(),
                });
            }
            (input, _, _) => {
                let port = if input.is_some() {
                    MicPort::Idle
                } else {
                    MicPort::None
                };
                aside.tap(Fit::Off { tapped, port });
            }
        }
    }
}
