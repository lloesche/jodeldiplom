//! Lanes: one stream carrying a player's outputs side by side, so they share
//! one ring, one write head and one timeline (a seek, a flush, a jump taken
//! back act on all of them at once).
//!
//! A stream of `lanes` lanes (`lanes > 1`) at `channels` channels has
//! [`lane_width`] samples a frame, in this order:
//!
//! ```text
//! [main ch 0 .. ch n-1][part 1 ch 0 .. n-1] .. [part lanes-1 ch 0 .. n-1][control]
//! ```
//!
//! - **main**: the player's output as it would be on its own (the file, bit
//!   for bit). A mixer may pass it through untouched.
//! - **parts** 1 to `lanes - 1`: signals that together make main (stems,
//!   say, in the player's order); this crate only knows them by number. They
//!   are never passed through: they are always mixed.
//! - **control**: one word a frame, a [`Parts`]: whether the parts are valid
//!   on that frame. A writer ends them with [`Parts::Ending`] for
//!   [`ending_frames`] before [`Parts::None`], with one exception: on the
//!   first frame of a new track run into gaplessly (main itself cuts to
//!   another signal there) the word may go from [`Parts::Present`] straight
//!   to [`Parts::None`], and a mixer snaps to main on that cut.
//!
//! The ring holds lane streams in [`LANE_SAMPLE`] whatever the device's
//! format: every format's values fit in it unchanged, so main reaches the
//! device exactly as it would from a stream of its own, and the parts are
//! not requantized once more on the way. A stream of one lane is a plain
//! stream: `channels` samples a frame, in the device's format.

use std::time::Duration;

use crate::format::{SampleFormat, StreamFormat};

/// The sample format of a lane stream's ring.
pub const LANE_SAMPLE: SampleFormat = SampleFormat::S24In32;

/// How long a mixer takes to switch from main to the parts or back: a
/// linear crossfade, as main and the parts' sum are the same signal.
pub const SWITCH_FADE: Duration = Duration::from_millis(10);

/// The frames of [`SWITCH_FADE`] at `rate`, rounded as the mixer rounds its
/// ramps: how long after a frame that says [`Parts::Ending`] the writer
/// keeps the parts valid.
pub fn ending_frames(rate: u32) -> u64 {
    (SWITCH_FADE.as_secs_f64() * f64::from(rate)).round() as u64
}

/// The samples in a frame of a stream of `lanes` lanes at `channels`
/// channels: `channels` for one lane, else `channels × lanes` and the
/// control word.
pub fn lane_width(channels: u16, lanes: u16) -> usize {
    let (channels, lanes) = (usize::from(channels), usize::from(lanes.max(1)));
    if lanes > 1 {
        channels * lanes + 1
    } else {
        channels
    }
}

/// The ring's format for a stream of `lanes` lanes on a device at `format`:
/// `format` itself for one lane, else [`LANE_SAMPLE`] at [`lane_width`]
/// channels.
pub fn lane_format(format: StreamFormat, lanes: u16) -> StreamFormat {
    if lanes > 1 {
        let width = lane_width(format.channels, lanes);
        let width = u16::try_from(width).expect("a lane stream's width fits a u16");
        StreamFormat::new(format.rate, width, LANE_SAMPLE)
    } else {
        format
    }
}

/// The control word of a frame of a lane stream: whether its parts are
/// valid there.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Parts {
    /// No parts: only main is valid (the parts are not made, or not yet).
    #[default]
    None = 0,
    /// The parts are valid, and together make main.
    Present = 1,
    /// The parts are valid for a while yet, and end: whoever mixes them
    /// fades back to main from this frame (the writer keeps them valid for
    /// [`ending_frames`] after).
    Ending = 2,
}

impl Parts {
    /// The word written for it.
    pub fn word(self) -> i32 {
        self as i32
    }

    /// The control word `word` read back (an unknown one is `None`).
    pub fn from_word(word: i32) -> Parts {
        match word {
            1 => Parts::Present,
            2 => Parts::Ending,
            _ => Parts::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_are_side_by_side_with_a_control_word() {
        assert_eq!(lane_width(2, 1), 2);
        assert_eq!(lane_width(2, 5), 11);
        assert_eq!(lane_width(1, 5), 6);
        let plain = StreamFormat::new(48_000, 2, SampleFormat::S16);
        assert_eq!(lane_format(plain, 1), plain);
        assert_eq!(
            lane_format(plain, 5),
            StreamFormat::new(48_000, 11, SampleFormat::S24In32)
        );
        for p in [Parts::None, Parts::Present, Parts::Ending] {
            assert_eq!(Parts::from_word(p.word()), p);
        }
        assert_eq!(Parts::from_word(7), Parts::None);
        assert_eq!((ending_frames(48_000), ending_frames(44_100)), (480, 441));
    }
}
