//! A network on a device of its own, planned once per input size
//! ([`Model`]).

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use super::{bytes, guarded, Input, Kernels, ModelOp, OnDevice, Output, Program, Recorded, Write};
use crate::pace::Pieces;
use crate::Footprint;

/// What a network takes on the GPU, from its buffers' shapes: each of
/// `sizes` (its f32s) and the `read` f32s it reads back through, kept while
/// it is set up; and `upload` bytes on top while its weights go up.
pub(crate) fn buffers_footprint<B>(sizes: &[(B, usize)], read: usize, upload: u64) -> Footprint {
    let floats: usize = sizes.iter().map(|&(_, n)| n).sum::<usize>() + read;
    Footprint {
        resident: floats as u64 * 4,
        working: upload,
    }
}

/// The most plans a model keeps: one more, and the older all go first
/// (inputs come in a few sizes: the last chunk or piece of a track is
/// shorter).
const PLANS: usize = 8;

/// A network on a device of its own, in buffers made once (`B` names
/// them), planned once for each input size it sees: a plan's dispatches
/// ([`ModelOp`]s) recorded when first run, and each run one plan and one
/// read back.
pub(crate) struct Model<K, B> {
    pub(crate) program: Program<K>,
    pub(crate) buffers: HashMap<B, wgpu::Buffer>,
    pub(crate) staging: wgpu::Buffer,
    plans: HashMap<usize, Recorded<K, [B; 3]>>,
    pub(crate) pieces: Pieces,
}

impl<K: Kernels, B> OnDevice for Model<K, B> {
    fn lost(&self) -> bool {
        self.program.lost()
    }
}

impl<K: Kernels, B: Copy + Eq + Hash + Debug> Model<K, B> {
    /// `program`'s network with a buffer for each of `sizes`, `writes` put
    /// in theirs (its weights, its tables) and the others zeroed
    /// ([`Program::room`]), and room for `read` f32s read back, its runs
    /// paced as `pieces` say.
    pub(crate) fn new(
        program: Program<K>,
        sizes: &[(B, usize)],
        writes: &[(B, &[f32])],
        read: usize,
        pieces: Pieces,
    ) -> Result<Model<K, B>, String> {
        let written: Vec<(B, Vec<u8>)> = writes.iter().map(|&(b, v)| (b, bytes(v))).collect();
        let writes: Vec<(B, &[u8])> = written.iter().map(|(b, v)| (*b, &v[..])).collect();
        let buffers = program.room(sizes, &writes)?;
        let staging = program.staging(read);
        Ok(Model {
            program,
            buffers,
            staging,
            plans: HashMap::new(),
            pieces,
        })
    }

    /// The plan `key`, recorded from `ops` if it is not yet.
    fn plan(&mut self, key: usize, ops: impl FnOnce() -> Vec<ModelOp<K, B>>) -> Result<(), String> {
        if self.plans.contains_key(&key) {
            return Ok(());
        }
        if self.plans.len() >= PLANS {
            self.plans.clear();
        }
        let buffers = &self.buffers;
        let recorded = guarded(&self.program.device, || {
            self.program
                .record(&ops(), |bind: [B; 3]| bind.map(|b| &buffers[&b]).to_vec())
        })?;
        self.plans.insert(key, recorded);
        Ok(())
    }

    /// Runs the plan `key` (recorded from `ops` on first use): `inputs`
    /// written first, `prelude` before its dispatches; then `output` read
    /// back. A failure on the device is an `Err`.
    pub(crate) fn run(
        &mut self,
        key: usize,
        ops: impl FnOnce() -> Vec<ModelOp<K, B>>,
        inputs: &[Input<B>],
        prelude: impl FnOnce(&mut wgpu::CommandEncoder, &HashMap<B, wgpu::Buffer>),
        output: Output<B>,
    ) -> Result<Vec<f32>, String> {
        self.plan(key, ops)?;
        let buffers = &self.buffers;
        let buffer = |b: B| &buffers[&b];
        let writes: Vec<Write> = inputs.iter().map(|i| i.write(buffer)).collect();
        let read = output.read(buffer, &self.staging);
        self.program.run(
            &mut self.pieces,
            &self.plans[&key],
            &writes,
            |encoder| prelude(encoder, buffers),
            Some(read),
        )
    }
}
