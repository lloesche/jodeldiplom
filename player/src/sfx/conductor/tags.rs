//! The tags the conductor's notes go with ([`jodel_synth::Tag`]), every
//! one: what is called off together, and what a release lets go of (the
//! voices of its tag, whatever their patch: each held sound has a tag of
//! its own). A sound that plays out goes with none ([`Tag::NONE`]).

use jodel_synth::Tag;

/// The top rung's payout, called off with the streak.
pub(crate) const PAYOUT: Tag = Tag(1);

/// The glitter over the top, called off with the streak too.
pub(crate) const STREAK: Tag = Tag(2);

/// The riser into a high rung, held until the rung comes or the streak
/// goes.
pub(crate) const RISER: Tag = Tag(3);

/// The shimmer pad held at the high rungs, let go with the streak.
pub(crate) const SHIMMER: Tag = Tag(4);

/// The Synth tab's keyboard held by the pointer.
pub(crate) const POINTER: Tag = Tag(16);

/// The Synth tab's keyboard held by the computer key at place `k` (0
/// up), each its own (after [`POINTER`]).
pub(crate) fn computer_key(k: u8) -> Tag {
    Tag(POINTER.0.saturating_add(1).saturating_add(k))
}
