//! A network packed from its CPU model's weights onto a device of its own
//! ([`PackedNet`]: the separator, the beat network, the speech model), and
//! one the process keeps ([`KeptNet`]: the separators and the beat
//! network, set up when first needed and again once their device is lost).

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use super::{Kept, Kernels, Model, OnDevice};
use crate::pace::{Counts, Pieces, Tally};
use crate::packer::Named;
use crate::{Footprint, Resident};

/// A network on its device ([`Model`]), and where its weights are in its
/// weights' buffer, by name.
pub(crate) struct PackedNet<K, B> {
    pub(crate) model: Model<K, B>,
    pub(crate) weights: Named,
}

impl<K: Kernels, B> OnDevice for PackedNet<K, B> {
    fn lost(&self) -> bool {
        self.model.lost()
    }
}

/// A packed network the process keeps ([`Kept`]), and what its runs cost
/// (one count across its setups, read without waiting for a run).
pub(crate) struct KeptNet<K, B> {
    kept: Arc<Kept<PackedNet<K, B>>>,
    counts: Counts,
}

impl<K, B> KeptNet<K, B>
where
    K: Kernels + Send,
    B: Copy + Eq + Hash + Debug + Send + 'static,
{
    /// The network `open` sets up on a device (its runs paced as `pieces`
    /// say; with the GPU's name), set up when first used, taking what
    /// `footprint` works out on the GPU.
    pub(crate) fn new(
        pieces: Pieces,
        footprint: fn() -> Option<Footprint>,
        open: impl Fn(Pieces) -> Result<(PackedNet<K, B>, String), String> + Send + Sync + 'static,
    ) -> KeptNet<K, B> {
        let counts = pieces.counts();
        let set_up = Box::new(move || open(pieces.clone()));
        KeptNet {
            counts,
            kept: Arc::new(Kept::new(footprint, set_up)),
        }
    }

    /// [`KeptNet::new`], set up now; why not.
    pub(crate) fn set_up_now(
        pieces: Pieces,
        footprint: fn() -> Option<Footprint>,
        open: impl Fn(Pieces) -> Result<(PackedNet<K, B>, String), String> + Send + Sync + 'static,
    ) -> Result<KeptNet<K, B>, String> {
        let counts = pieces.counts();
        let set_up = Box::new(move || open(pieces.clone()));
        Ok(KeptNet {
            counts,
            kept: Arc::new(Kept::set_up_now(footprint, set_up)?),
        })
    }

    /// Whether it is set up (set up now where it is not and may be).
    pub(crate) fn usable(&self) -> bool {
        self.kept.usable()
    }

    /// `work` on the network on its device ([`Kept::run`]).
    pub(crate) fn run<R>(
        &self,
        work: impl FnMut(&mut PackedNet<K, B>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.kept.run(work)
    }

    /// What its GPU work cost so far ([`Tally::since`] an earlier tally
    /// says what was done between), without waiting for its work.
    pub(crate) fn tally(&self) -> Tally {
        self.counts.tally()
    }

    /// The GPU's name, once set up.
    pub(crate) fn name(&self) -> String {
        self.kept.name()
    }

    /// The network as a scheduler keeps it: its footprint, and its
    /// release.
    pub(crate) fn resident(&self) -> Arc<dyn Resident> {
        Arc::clone(&self.kept) as Arc<dyn Resident>
    }
}
