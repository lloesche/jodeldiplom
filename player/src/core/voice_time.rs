//! Where in the song a block of the singer's voice was sung
//! ([`VoiceClock`]): its stream frame (of the Lead output it was heard
//! with, as the microphone's tap tells it) against the last position the
//! engine told of the playing track.
//!
//! A transition's hand-over moves the tap onto the next track's head as
//! the desk makes it Lead, which is before (or, for blocks still on their
//! way, after) the positions told switch to that head: for a while the
//! blocks are of one head and the position of the other. The two heads'
//! frames are the mix's `origin` apart (the next track's frame 0 is on the
//! playing head's `origin`), so a block is placed through whichever head
//! puts it near what is heard; one near neither is dropped, never placed
//! seconds off in a recording or a score.

use jodel_output::VoiceBlock;

use super::transition::Status;
use super::{Anchor, State};

/// How far from what is heard a block may be placed, in seconds: the
/// voice comes a fraction of a second late, a head's frames are a mix's
/// lead and window apart.
pub const NEAR: f64 = 3.0;

/// The playing track's clock for the voice's blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceClock {
    anchor: Anchor,
    /// The output's rate.
    rate: f64,
    /// While a transition is mixed or settles: what turns the other head's
    /// frames into the anchor's head's.
    shift: Option<i64>,
    /// Seconds into the track heard now.
    heard: f64,
}

impl VoiceClock {
    /// The clock of track `index`, the one playing: none before a position
    /// of it is told.
    pub fn of(state: &State, index: usize) -> Option<VoiceClock> {
        let track = state.playing.track.as_ref().filter(|t| t.index == index)?;
        let anchor = state.playing.anchor.filter(|a| a.index == index)?;
        // The next track's head's frames are `origin` fewer than the one's
        // it was mixed in on.
        let shift = match &state.transition.draft.status {
            Status::Mixing(mix) => {
                let origin = i64::try_from(mix.origin).unwrap_or(i64::MAX);
                Some(if anchor.at.head == mix.start.head {
                    origin
                } else {
                    -origin
                })
            }
            _ => None,
        };
        Some(VoiceClock {
            anchor,
            rate: f64::from(track.output.format.rate.max(1)),
            shift,
            heard: state.heard(),
        })
    }

    /// The seconds into the track of stream frame `frame` of the Lead
    /// output (fractional): through the anchor's head, else the other's,
    /// whichever is near what is heard; `None` when neither is.
    pub fn seconds(&self, frame: f64) -> Option<f64> {
        let at = |f: f64| self.anchor.seconds + (f - self.anchor.at.frame as f64) / self.rate;
        std::iter::once(at(frame))
            .chain(self.shift.map(|s| at(frame + s as f64)))
            .find(|t| (t - self.heard).abs() <= NEAR)
    }

    /// Each of `blocks` that is placed ([`VoiceClock::seconds`]), with its
    /// time in the track and whether it follows on from the one before (a
    /// block dropped between breaks that).
    pub fn place<'a>(&self, blocks: &'a [VoiceBlock]) -> Vec<(&'a VoiceBlock, f64, bool)> {
        let mut gap = false;
        blocks
            .iter()
            .filter_map(|block| {
                let Some(at) = self.seconds(block.stream_frame as f64) else {
                    gap = true;
                    return None;
                };
                Some((block, at, block.continues && !std::mem::take(&mut gap)))
            })
            .collect()
    }
}
