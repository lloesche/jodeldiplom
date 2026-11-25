//! A network on a device of its own, planned once per input size
//! ([`Model`]).

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::time::Duration;

use super::{bytes, Kernels, Op, Program, Recorded};
use crate::pace::{self, Pieces};

/// The most plans a model keeps: one more, and the older all go first
/// (inputs come in a few sizes: the last chunk or piece of a track is
/// shorter).
const PLANS: usize = 8;

/// A network on a device of its own, in buffers made once (`B` names
/// them), planned once for each input size it sees: a plan's dispatches
/// recorded when first run, and each run one plan and one read back.
pub(crate) struct Model<K, B> {
    pub(crate) program: Program<K>,
    pub(crate) buffers: HashMap<B, wgpu::Buffer>,
    pub(crate) staging: wgpu::Buffer,
    plans: HashMap<usize, Recorded<K>>,
    pub(crate) pieces: Pieces,
}

impl<K: Kernels, B: Copy + Eq + Hash + Debug> Model<K, B> {
    /// `program`'s network with a buffer for each of `sizes` and room for
    /// `read` f32s read back, its runs paced as `pieces` say.
    pub(crate) fn new(
        program: Program<K>,
        sizes: &[(B, usize)],
        read: usize,
        pieces: Pieces,
    ) -> Result<Model<K, B>, String> {
        let buffers = program.buffers(sizes)?;
        let staging = program.staging(read);
        Ok(Model {
            program,
            buffers,
            staging,
            plans: HashMap::new(),
            pieces,
        })
    }

    /// `values` written to the start of `buf`.
    pub(crate) fn write(&self, buf: B, values: &[f32]) {
        self.program
            .queue
            .write_buffer(&self.buffers[&buf], 0, &bytes(values));
    }

    /// The plan `key`, recorded from `ops` if it is not yet.
    fn plan(
        &mut self,
        key: usize,
        ops: impl FnOnce() -> Vec<Op<K, [B; 3], 12>>,
    ) -> Result<(), String> {
        if self.plans.contains_key(&key) {
            return Ok(());
        }
        if self.plans.len() >= PLANS {
            self.plans.clear();
        }
        let buffers = &self.buffers;
        let recorded = self
            .program
            .record(&ops(), |bind: [B; 3]| bind.map(|b| &buffers[&b]).to_vec())?;
        self.plans.insert(key, recorded);
        Ok(())
    }

    /// Runs the plan `key` (recorded from `ops` on first use), `prelude`
    /// before its dispatches: the first `count` f32s of `from`.
    pub(crate) fn run(
        &mut self,
        key: usize,
        ops: impl FnOnce() -> Vec<Op<K, [B; 3], 12>>,
        prelude: impl FnOnce(&mut wgpu::CommandEncoder, &HashMap<B, wgpu::Buffer>),
        (from, count): (B, usize),
    ) -> Result<Vec<f32>, String> {
        self.plan(key, ops)?;
        let plan = &self.plans[&key];
        let (buffers, staging) = (&self.buffers, &self.staging);
        let size = (count * 4) as u64;
        pace::dispatch(
            &self.program,
            &mut self.pieces,
            &plan.groups,
            &plan.dispatches,
            |encoder| prelude(encoder, buffers),
            |encoder| encoder.copy_buffer_to_buffer(&buffers[&from], 0, staging, 0, size),
        )?;
        crate::device::read_back(&self.program.device, staging, count)
    }

    /// The dispatches of the plan `key` that `keep` picks, timed
    /// ([`Program::time`]); `Err` where it is not planned.
    pub(crate) fn time(
        &self,
        key: usize,
        keep: &dyn Fn(usize) -> bool,
    ) -> Result<Duration, String> {
        let plan = self
            .plans
            .get(&key)
            .ok_or_else(|| format!("no plan for {key}"))?;
        self.program.time(plan, keep)
    }
}
