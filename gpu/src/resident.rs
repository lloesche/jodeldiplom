//! What a model holds on the GPU, said before it is loaded, so the
//! player's scheduler can plan which models stay and which go.

/// A model's memory on the GPU: what stays between its jobs, and what a
/// job needs on top while it runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Kept while the model is loaded (its weights).
    pub resident: u64,
    /// Needed on top while a job runs (its working buffers).
    pub working: u64,
}

impl Footprint {
    /// Both together: what a job needs at its peak.
    pub fn total(&self) -> u64 {
        self.resident + self.working
    }
}

/// A model a scheduler may let go of: its footprint, known before it is
/// loaded, and its release.
pub trait Resident: Send + Sync {
    /// What it needs, loaded or not.
    fn footprint(&self) -> Footprint;
    /// Lets go of what it holds on the GPU; its next use loads it again.
    fn release(&self);
}
