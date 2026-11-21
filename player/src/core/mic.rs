//! The microphone as the app reads it once a drawn frame
//! ([`MicReading`], in [`crate::core::State::mic`]): the Karaoke tab's
//! meter and echo line and the Mix tab's strip all draw from it.

/// The microphone as read for a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MicReading {
    /// The capture rate.
    pub rate: u32,
    /// The recent peak as captured (linear, 1.0: full scale).
    pub peak: f32,
    /// The singer isolated (linear), while the cleaner gives it: an output
    /// is tapped, so the music's echo can be taken out.
    pub voice: Option<f32>,
    /// The echo canceller is on.
    pub echo_on: bool,
    /// How much quieter the echo is after the canceller, in dB.
    pub removed_db: f32,
    /// It has learnt the echo's way.
    pub learnt: bool,
    /// How late the echo comes, in seconds, once learnt.
    pub delay: Option<f32>,
    /// The clock bridge's drift: how much faster the microphone runs than
    /// the device, in parts per million.
    pub drift_ppm: f64,
    /// The clock bridge's fill, in milliseconds.
    pub fill_ms: f64,
    /// Times the bridge ran dry or over.
    pub xruns: u64,
}

impl MicReading {
    /// The singer's recent peak: isolated where the cleaner gives it,
    /// else as captured (nothing plays to take out).
    pub fn singer(&self) -> f32 {
        self.voice.unwrap_or(self.peak)
    }

    /// The echo canceller, while it runs: the echo it removes in dB,
    /// whether it has learnt the echo's way, and how late the echo comes.
    pub fn echo(&self) -> Option<(f32, bool, Option<f32>)> {
        (self.echo_on && self.voice.is_some()).then_some((self.removed_db, self.learnt, self.delay))
    }
}
