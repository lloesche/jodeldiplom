//! What a planner tells a synth: the patches it plays ([`Bank`], each by
//! its [`PatchId`]), the timed instructions it sends ([`Command`]: notes
//! and releases on frames, tagged so those waiting of one part can be
//! called off together, [`Tag`]; the tempo, the bus's level), and where
//! they go ([`Commands`]: a synth, a queue to one, or a list written
//! down).

use crate::patch::Patch;
use crate::NoteOn;

/// A patch in a [`Bank`]: the `n`th added is [`PatchId::nth`]`(n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PatchId(u16);

impl PatchId {
    /// The `n`th patch a bank has (from 0).
    ///
    /// # Panics
    ///
    /// Past 65536 patches.
    pub fn nth(n: usize) -> PatchId {
        PatchId(u16::try_from(n).expect("at most 65536 patches"))
    }

    /// Its place in its bank (from 0).
    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// The patches a synth plays, registered before it is made, in order.
#[derive(Clone, Debug, Default)]
pub struct Bank {
    pub(crate) patches: Vec<Patch>,
}

impl Bank {
    /// An empty bank.
    pub fn new() -> Bank {
        Bank::default()
    }

    /// Adds `patch`; its id in the synth made of this bank (the next in
    /// order: [`PatchId::nth`] of the patches before it).
    ///
    /// # Panics
    ///
    /// Past 65536 patches.
    pub fn add(&mut self, patch: Patch) -> PatchId {
        let id = PatchId::nth(self.patches.len());
        self.patches.push(patch);
        id
    }

    /// Its patches.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

/// What a note or a release is part of: those waiting of one are called
/// off together ([`Command::Cancel`]), and a release lets go of the
/// voices of its tag, whatever their patch ([`Cue::Release`]): a held
/// sound has a tag of its own. [`Tag::NONE`] is no part: only a cancel of
/// everything calls it off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tag(pub u8);

impl Tag {
    /// No part.
    pub const NONE: Tag = Tag(0);
}

/// What a timed command does on its frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cue {
    /// A note of `patch` starts ([`crate::Synth::note`]).
    Note {
        /// Its patch.
        patch: PatchId,
        /// The note.
        note: NoteOn,
        /// How long it is held, frames (none: the patch's gate).
        length: Option<u64>,
    },
    /// The voices still held of the command's tag let go, whatever their
    /// patch, from the block the frame falls in (one already rendered: as
    /// soon as it can).
    Release,
}

/// A timed instruction to a synth: what a planner sends it. Its frames are
/// the synth's own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    /// A note or a release on `frame`, tagged.
    At {
        /// The frame.
        frame: u64,
        /// Its tag.
        tag: Tag,
        /// What it does there.
        cue: Cue,
    },
    /// The notes and releases waiting of `tag` called off (none: all of
    /// them, [`Tag::NONE`]'s too); those sounding sound on.
    Cancel {
        /// The tag called off.
        tag: Option<Tag>,
    },
    /// [`crate::Synth::set_tempo`], from now on.
    Tempo {
        /// Beats a minute.
        bpm: f32,
    },
    /// The bus's level before its limiter, from now on: gliding there
    /// over [`crate::dynamics::LEVEL_GLIDE`] while anything sounds or
    /// waits, at once while nothing does. Not taken unless finite.
    Level {
        /// The level, dB.
        db: f32,
    },
}

impl Command {
    /// Its frame, if it is on one.
    pub fn frame(&self) -> Option<u64> {
        match *self {
            Command::At { frame, .. } => Some(frame),
            Command::Cancel { .. } | Command::Tempo { .. } | Command::Level { .. } => None,
        }
    }

    /// The same on frame `f` (one on no frame, as it is).
    pub fn on(mut self, f: u64) -> Command {
        if let Command::At { frame, .. } = &mut self {
            *frame = f;
        }
        self
    }

    /// Whether it is a note (not a release, nor one on no frame).
    pub fn is_note(&self) -> bool {
        matches!(
            self,
            Command::At {
                cue: Cue::Note { .. },
                ..
            }
        )
    }

    /// Whether a cancel of `tag` (none: of all) calls it off, waiting:
    /// a note or a release of that tag; [`Tag::NONE`]'s only by a cancel of
    /// all.
    pub fn called_off_by(&self, tag: Option<Tag>) -> bool {
        let Command::At { tag: of, .. } = *self else {
            return false;
        };
        match tag {
            None => true,
            Some(t) => t != Tag::NONE && t == of,
        }
    }
}

/// Where a planner's commands go: a synth, a queue to one, or a list.
pub trait Commands {
    /// Takes `c`; whether it was taken (not when full).
    fn send(&mut self, c: Command) -> bool;
}

/// Commands written down, in order (all of them taken).
impl Commands for Vec<Command> {
    fn send(&mut self, c: Command) -> bool {
        self.push(c);
        true
    }
}
