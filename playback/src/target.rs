//! Where the engine plays to: a device, memory, nowhere, or a mixer's input
//! ([`jodel_output::mixer::InputTarget`], several players on one device).
//!
//! The engine asks an [`OutputTarget`] for its [`Capabilities`] (once, for
//! the rate policy) and opens one [`Output`] per stream format it needs.

use std::sync::{Arc, Mutex};

pub use jodel_output::OutputTarget;
use jodel_output::{
    Capabilities, Error, FileOutput, MemorySink, NullOutput, OpenRequest, Output, SampleFormat,
    SimClock, StreamFormat,
};

/// A real output device, opened shared through `jodel-output`.
pub struct DeviceTarget {
    device: jodel_output::Device,
}

impl DeviceTarget {
    /// The device `query` names: `"default"`, an id, a name or a unique
    /// part of a name (see [`jodel_output::find_device`]).
    pub fn find(query: &str) -> Result<DeviceTarget, Error> {
        Ok(DeviceTarget {
            device: jodel_output::find_device(query)?,
        })
    }

    /// The device.
    pub fn device(&self) -> &jodel_output::Device {
        &self.device
    }
}

impl OutputTarget for DeviceTarget {
    fn name(&self) -> String {
        self.device.name.clone()
    }

    /// Its one format, asked anew each time (it is the system's).
    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([jodel_output::mix_format(&self.device.id)?]
            .into_iter()
            .collect())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        Ok(Box::new(jodel_output::open(&self.device.id, request)?))
    }
}

/// Every format at 1 to 8 channels, at `rate`: what a stand-in for a
/// device the system runs at `rate` reports (a device offers one rate).
pub fn any_capabilities(rate: u32) -> Capabilities {
    let mut caps = Capabilities::new();
    for channels in 1..=8 {
        for sample in SampleFormat::ALL {
            caps.insert(StreamFormat::new(rate, channels, sample));
        }
    }
    caps
}

/// A target that plays into nothing, on a simulated clock.
pub struct NullTarget {
    caps: Capabilities,
    clock: SimClock,
}

impl NullTarget {
    /// A null target that accepts `caps`; [`SimClock::Realtime`] plays in
    /// real time, [`SimClock::Instant`] as fast as the engine writes.
    pub fn new(caps: Capabilities, clock: SimClock) -> NullTarget {
        NullTarget { caps, clock }
    }
}

impl OutputTarget for NullTarget {
    fn name(&self) -> String {
        "null output".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok(self.caps.clone())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let format = self
            .caps
            .choose(request.rate, request.channels, &request.formats)
            .ok_or(Error::FormatNotAccepted {
                rate: request.rate,
                channels: request.channels,
            })?;
        Ok(Box::new(NullOutput::new(format, self.clock)))
    }
}

/// One output stream a [`MemoryTarget`] opened: its format and the exact
/// bytes the simulated device played.
#[derive(Clone)]
pub struct Capture {
    /// The stream format.
    pub format: StreamFormat,
    /// The bytes, in the format's layout.
    pub sink: MemorySink,
}

/// A target that records every output stream in memory ([`FileOutput`]),
/// for tests.
pub struct MemoryTarget {
    caps: Capabilities,
    clock: SimClock,
    captures: Arc<Mutex<Vec<Capture>>>,
    lanes: u16,
}

impl MemoryTarget {
    /// A memory target that accepts `caps`.
    pub fn new(caps: Capabilities, clock: SimClock) -> MemoryTarget {
        MemoryTarget {
            caps,
            clock,
            captures: Arc::default(),
            lanes: 1,
        }
    }

    /// A memory target that also takes lane streams of up to `lanes` lanes
    /// ([`jodel_output::lanes`]) and keeps them as written
    /// ([`FileOutput::open_lanes`]): a capture's bytes are then the ring's,
    /// [`jodel_output::lanes::LANE_SAMPLE`] at the lane width a frame.
    pub fn with_lanes(caps: Capabilities, clock: SimClock, lanes: u16) -> MemoryTarget {
        MemoryTarget {
            lanes: lanes.max(1),
            ..MemoryTarget::new(caps, clock)
        }
    }

    /// A handle to the streams opened so far, readable while the engine
    /// owns the target.
    pub fn captures(&self) -> Captures {
        Captures(self.captures.clone())
    }
}

/// The streams a [`MemoryTarget`] has opened, in order.
#[derive(Clone)]
pub struct Captures(Arc<Mutex<Vec<Capture>>>);

impl Captures {
    /// A copy of the list.
    pub fn list(&self) -> Vec<Capture> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl OutputTarget for MemoryTarget {
    fn name(&self) -> String {
        "memory".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok(self.caps.clone())
    }

    fn lanes(&mut self) -> u16 {
        self.lanes
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let sink = MemorySink::default();
        request.check_lanes(self.lanes)?;
        let output = FileOutput::open_lanes(sink.clone(), &self.caps, request, self.clock)?;
        self.captures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Capture {
                format: output.format(),
                sink,
            });
        Ok(Box::new(output))
    }
}
