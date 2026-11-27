//! The picture models' GPU ([`Gpu`]: one device for all three parts, set
//! up when first needed and again once lost, with tensor cores where it
//! has them), a part's weights on it ([`Loaded`]), and a plan made on it
//! ([`Made`]: its buffers, its dispatches recorded, run in short
//! submissions).

use std::sync::{Arc, Mutex};

use super::plan::{Buf, Plan};
use super::{source, Kernel};
use crate::engine::{bytes, guarded, Program, Recorded};
use crate::Turned;

/// The workgroup memory the f32 kernels take at most (the attention's,
/// 24 KB), and the tensor-core ones (the matrix multiply's, 44 KB).
const WORKGROUP_MEMORY: u32 = 32 * 1024;
const COOP_MEMORY: u32 = 44 * 1024;

/// The features the tensor-core kernels need.
const COOP: wgpu::Features =
    wgpu::Features::SHADER_F16.union(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);

/// The picture models' GPU.
pub(crate) struct Gpu {
    pub(crate) program: Program<Kernel>,
    /// The GPU's name.
    pub(crate) name: String,
    /// Whether the fast kernels run on tensor cores.
    pub(crate) coop: bool,
}

/// The GPU while set up (never torn down once set up: the parts hold their
/// own memory), until its device is lost.
static GPU: Mutex<Option<Arc<Turned<Gpu>>>> = Mutex::new(None);

/// Whether `adapter` has what the tensor-core kernels need: f16,
/// cooperative matrices of 16 × 16 × 16 in f16 with f32 sums, subgroups
/// of 32 (eight to a workgroup of 256), and 44 KB of workgroup memory.
fn has_tensor_cores(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    adapter.features().contains(COOP)
        && adapter.limits().max_compute_workgroup_storage_size >= COOP_MEMORY
        && info.subgroup_min_size == 32
        && info.subgroup_max_size == 32
        && adapter.cooperative_matrix_properties().iter().any(|p| {
            (p.m_size, p.n_size, p.k_size) == (16, 16, 16)
                && p.ab_type == wgpu::CooperativeScalarType::F16
                && p.cr_type == wgpu::CooperativeScalarType::F32
        })
}

impl Gpu {
    /// The picture models' GPU, set up when first asked for (and again once
    /// its device is lost: the parts on the lost one say so); why not where
    /// it cannot be (asked again, it tries again).
    pub(crate) fn shared() -> Result<Arc<Turned<Gpu>>, String> {
        let mut slot = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(gpu) = slot.as_ref().filter(|gpu| !gpu.program.lost()) {
            return Ok(Arc::clone(gpu));
        }
        let gpu = std::panic::catch_unwind(|| {
            crate::device::open_with(
                "jodel-gpu image",
                |adapter| {
                    if has_tensor_cores(adapter) {
                        COOP
                    } else {
                        wgpu::Features::empty()
                    }
                },
                Gpu::on,
            )
        })
        .map_err(|_| "setting the GPU up panicked".to_owned())??;
        let (mut gpu, name) = gpu;
        gpu.name = name;
        let gpu = Arc::new(Turned::new(gpu));
        *slot = Some(Arc::clone(&gpu));
        Ok(gpu)
    }

    fn on(device: wgpu::Device, queue: wgpu::Queue, limits: &wgpu::Limits) -> Result<Gpu, String> {
        if limits.max_compute_workgroup_storage_size < WORKGROUP_MEMORY {
            return Err(format!(
                "{} bytes of workgroup memory, 32 KB needed",
                limits.max_compute_workgroup_storage_size
            ));
        }
        let coop = device.features().contains(COOP);
        Ok(Gpu {
            program: Program::new(device, queue, limits, "jodel-gpu image", &source(coop)),
            name: String::new(),
            coop,
        })
    }

    /// A storage buffer of `bytes` (a multiple of 4).
    pub(crate) fn buffer(&self, label: &str, bytes: u64) -> Result<wgpu::Buffer, String> {
        self.program.storage(label, (bytes / 4) as usize)
    }

    /// The bytes the device has taken from the GPU's memory now (its
    /// allocator's blocks), where the backend tells.
    pub(crate) fn reserved(&self) -> Option<u64> {
        self.program
            .device
            .generate_allocator_report()
            .map(|r| r.total_reserved_bytes)
    }
}

/// A part's weights on the GPU: its matrices (a few buffers of words) and
/// its f32 tensors.
pub(crate) struct Loaded {
    pub(crate) gpu: Arc<Turned<Gpu>>,
    pub(crate) matrices: Vec<wgpu::Buffer>,
    pub(crate) small: wgpu::Buffer,
}

impl Loaded {
    /// The bytes it holds.
    pub(crate) fn bytes(&self) -> u64 {
        self.matrices.iter().map(wgpu::Buffer::size).sum::<u64>() + self.small.size()
    }

    /// Whether its device was lost: it runs nothing again.
    pub(crate) fn lost(&self) -> bool {
        self.gpu.program.lost()
    }
}

/// A plan made on the device: its buffers, its dispatches recorded, and
/// the buffer its result is read back through.
pub(crate) struct Made {
    slots: Vec<wgpu::Buffer>,
    recorded: Recorded<Kernel>,
    staging: wgpu::Buffer,
    pieces: crate::pace::Pieces,
}

impl Made {
    /// `plan` on `loaded`'s device, `read` f32s to be read back from a run.
    pub(crate) fn new(loaded: &Loaded, plan: &Plan, read: usize) -> Result<Made, String> {
        let program = &loaded.gpu.program;
        guarded(&program.device, || {
            let slots = plan
                .slots
                .iter()
                .enumerate()
                .map(|(i, &len)| program.storage(&format!("image slot {i}"), len))
                .collect::<Result<Vec<_>, String>>()?;
            let staging = program.staging(read);
            let recorded = program.record(&plan.ops, |bufs: [Buf; 5]| {
                bufs.iter()
                    .map(|b| match *b {
                        Buf::W(i) => &loaded.matrices[usize::from(i)],
                        Buf::Small => &loaded.small,
                        Buf::Slot(i) => &slots[usize::from(i)],
                    })
                    .collect()
            })?;
            Ok(Made {
                slots,
                recorded,
                staging,
                pieces: crate::pace::Pieces::resting(),
            })
        })
    }

    /// The bytes its buffers take.
    pub(crate) fn bytes(&self) -> u64 {
        self.slots.iter().map(wgpu::Buffer::size).sum()
    }

    /// Runs it: each of `inputs` (a slot, an offset, values) written first,
    /// then `count` f32s of slot `from` from `offset` read back.
    pub(crate) fn run(
        &mut self,
        loaded: &Loaded,
        inputs: &[(u8, usize, &[f32])],
        (from, offset, count): (u8, usize, usize),
    ) -> Result<Vec<f32>, String> {
        let program = &loaded.gpu.program;
        let (slots, recorded, staging) = (&self.slots, &self.recorded, &self.staging);
        let pieces = &mut self.pieces;
        guarded(&program.device, || {
            for &(slot, at, values) in inputs {
                program.queue.write_buffer(
                    &slots[usize::from(slot)],
                    (at * 4) as u64,
                    &bytes(values),
                );
            }
            program.run(
                pieces,
                (&recorded.groups, &recorded.dispatches),
                |_| {},
                Some((&slots[usize::from(from)], offset, count)),
                staging,
            )
        })
    }
}
