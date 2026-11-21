//! The microphone's port, on the mixer thread: the microphone's tap on the
//! device, set for the Lead output open, so the echo canceller hears the
//! whole mix with the stream frames told in the Lead's frames.

use std::sync::Arc;

use super::desk::{MicPort, MicView};
use super::input::{FrameMap, MappedTap};
use super::MixShared;
use crate::input::Input;
use crate::output::Output;

/// What the port is fitted for: the microphone (by how many were set), the
/// device's opening (0: closed), and the Lead output.
type Fitting = (u64, u64, Option<u64>);

/// The microphone, its tap on the device, and how the port stands.
#[derive(Default)]
pub(super) struct Port {
    input: Option<Input>,
    /// The microphone replaced, let go of once its tap is off the device.
    retired: Option<Input>,
    /// Microphones set so far.
    set: u64,
    fitted: Option<Fitting>,
    /// The Lead output the tap is set for, and its frame map.
    tap: Option<(u64, Arc<FrameMap>)>,
    view: MicView,
}

/// Lets go of a microphone off the mixer thread: when it is the last
/// handle, dropping it stops capture and waits for its cleaner's thread.
/// (With no thread to be had, it is dropped here, as a last resort.)
fn let_go(input: Input) {
    let _ = std::thread::Builder::new()
        .name("jodel-mic-drop".into())
        .spawn(move || drop(input));
}

impl Port {
    /// The microphone listened to (`None`: none), tapped with the next
    /// [`Port::fit`].
    pub(super) fn set(&mut self, input: Option<Input>) {
        self.set += 1;
        if let Some(old) = std::mem::replace(&mut self.input, input) {
            self.retired = Some(old);
        }
    }

    /// The frame map of the tap, when it is set for stream `stream`.
    pub(super) fn map_for(&self, stream: u64) -> Option<&FrameMap> {
        self.tap
            .as_ref()
            .filter(|(s, _)| *s == stream)
            .map(|(_, m)| &**m)
    }

    /// Fits the port to the device `output` (`None`: closed) of opening
    /// `epoch` and the Lead output `lead`: the tap set for it with a fresh
    /// frame map, or taken off with no microphone, device or Lead output.
    /// Nothing changes when nothing it is fitted for has.
    pub(super) fn fit(
        &mut self,
        output: Option<&mut Box<dyn Output>>,
        epoch: u64,
        lead: Option<u64>,
        shared: &MixShared,
    ) {
        let fitting = (self.set, if output.is_some() { epoch } else { 0 }, lead);
        if self.fitted != Some(fitting) {
            self.fitted = Some(fitting);
            self.refit(output, lead, shared);
        }
        // The old microphone's tap is off the device: its last handles go
        // elsewhere.
        if let Some(old) = self.retired.take() {
            let_go(old);
        }
    }

    fn refit(
        &mut self,
        output: Option<&mut Box<dyn Output>>,
        lead: Option<u64>,
        shared: &MixShared,
    ) {
        let mut view = self.view.clone();
        view.port = match (&self.input, output, lead) {
            (Some(input), Some(output), Some(stream)) => {
                let map = Arc::new(FrameMap::new());
                let tap = MappedTap::new(Box::new(input.tap()), map.clone());
                match output.set_tap(Some(Box::new(tap))) {
                    Ok(()) => {
                        self.tap = Some((stream, map));
                        MicPort::Hearing
                    }
                    Err(e) => {
                        self.tap = None;
                        view.refused += 1;
                        MicPort::CannotHear(e.to_string())
                    }
                }
            }
            (input, output, _) => {
                if let (Some(output), Some(_)) = (output, self.tap.take()) {
                    let _ = output.set_tap(None);
                }
                self.tap = None;
                if input.is_some() {
                    MicPort::Idle
                } else {
                    MicPort::None
                }
            }
        };
        if view != self.view {
            self.view = view.clone();
            *shared.mic() = view;
        }
    }
}

impl Drop for Port {
    fn drop(&mut self) {
        for input in [self.input.take(), self.retired.take()]
            .into_iter()
            .flatten()
        {
            let_go(input);
        }
    }
}
