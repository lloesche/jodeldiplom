//! The picture models' GPU ([`Gpu`]: one device for all three parts, set
//! up when first needed and again once lost, with tensor cores where it
//! has them), a part's weights on it ([`Loaded`]), and a plan made on it
//! ([`Made`]: its buffers, its dispatches recorded, run in short
//! submissions).

use std::sync::Arc;

use super::plan::{Buf, Plan};
use super::{source, Kernel};
use crate::engine::{guarded, Input, OnDevice, Output, Program, Recorded, SharedDevice, Write};
use crate::{Precision, Turned};

/// The workgroup memory the f32 kernels take at most (the attention's,
/// 24 KB).
const WORKGROUP_MEMORY: u32 = 32 * 1024;

/// The features the tensor-core kernels need.
pub(crate) const COOP: wgpu::Features =
    wgpu::Features::SHADER_F16.union(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);

/// The picture models' GPU.
pub(crate) struct Gpu {
    pub(crate) program: Program<Kernel>,
    /// The GPU's name.
    pub(crate) name: String,
    /// The tiles the fast kernels run on tensor cores with, where they do.
    pub(crate) coop: Option<Tiles>,
}

/// The cooperative matrices the tensor-core kernels are made of
/// (`coop.wgsl`): their tiles of 16 × 16 as one of 16 × 16 × 16 each, or
/// as four of 8 × 8 × 8 (all Metal offers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tiles {
    Sixteen,
    Eight,
}

impl Tiles {
    /// The side of its cooperative matrices.
    fn side(self) -> u32 {
        match self {
            Tiles::Sixteen => 16,
            Tiles::Eight => 8,
        }
    }

    /// The workgroup memory its kernels take at most (on 16 × 16 the
    /// attention's 46.5 KB, on 8 × 8 the multiply's 29 KB: `coop.wgsl`;
    /// the 8 × 8 attention takes 23 KB; `the_kernels_validate` sums them).
    pub(crate) fn memory(self) -> u32 {
        match self {
            Tiles::Sixteen => 47 * 1024,
            Tiles::Eight => 29 * 1024,
        }
    }
}

/// The GPU while set up (never torn down once set up: the parts hold their
/// own memory), until its device is lost.
static GPU: SharedDevice<Gpu> = SharedDevice::new(Gpu::open);

impl OnDevice for Gpu {
    fn lost(&self) -> bool {
        self.program.lost()
    }
}

/// The tiles the tensor-core kernels take on `adapter`, where it has what
/// they need: f16, cooperative matrices of 16 × 16 × 16 (else of 8 × 8 ×
/// 8) in f16 with f32 sums, subgroups of 32 (eight to a workgroup of 256)
/// and the workgroup memory the kernels take on those tiles
/// ([`Tiles::memory`]). Through Metal
/// the name is all there is to go by: wgpu says subgroups of 4 to 64 for
/// every Metal adapter and gives no vendor or device (0), and it offers the
/// same 8 × 8 matrices on Apple's GPUs (family Apple7) and on the Intel and
/// AMD GPUs of older Macs (Mac2), whose SIMD-groups need not be 32 wide.
/// Apple's are 32, and a simdgroup matrix is one of them; their names start
/// with "Apple", which keeps the others out.
fn tiles(adapter: &wgpu::Adapter) -> Option<Tiles> {
    let info = adapter.get_info();
    let of_32 = (info.subgroup_min_size, info.subgroup_max_size) == (32, 32)
        || (info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple"));
    if !adapter.features().contains(COOP) || !of_32 {
        return None;
    }
    let memory = adapter.limits().max_compute_workgroup_storage_size;
    let properties = adapter.cooperative_matrix_properties();
    [Tiles::Sixteen, Tiles::Eight].into_iter().find(|t| {
        let n = t.side();
        memory >= t.memory()
            && properties.iter().any(|p| {
                (p.m_size, p.n_size, p.k_size) == (n, n, n)
                    && p.ab_type == wgpu::CooperativeScalarType::F16
                    && p.cr_type == wgpu::CooperativeScalarType::F32
            })
    })
}

impl Gpu {
    /// The picture models' GPU, set up when first asked for (and again once
    /// its device is lost: the parts on the lost one say so); why not where
    /// it cannot be (a failed setup stands for a while).
    pub(crate) fn shared() -> Result<Arc<Turned<Gpu>>, String> {
        GPU.get()
    }

    /// The device, with tensor cores where the adapter has them.
    fn open() -> Result<Gpu, String> {
        let (mut gpu, name) = crate::device::open_with(
            "jodel-gpu image",
            |adapter| {
                let coop = tiles(adapter);
                if coop.is_none() {
                    return (
                        wgpu::Features::empty(),
                        wgpu::ExperimentalFeatures::disabled(),
                        None,
                    );
                }
                // SAFETY: wgpu's cooperative matrices are experimental: its
                // checks of what a kernel does with them may be incomplete,
                // so a wrong kernel could misbehave on the device instead of
                // failing validation. That risk is accepted for the two
                // tensor-core kernels, and only on adapters that pass
                // `tiles` (the matrices' shape and types, the
                // subgroup size and workgroup memory they are written for);
                // the tests hold them to the f32 kernels on such a GPU.
                (COOP, unsafe { wgpu::ExperimentalFeatures::enabled() }, coop)
            },
            Gpu::on,
        )?;
        gpu.name = name;
        Ok(gpu)
    }

    /// The picture models' program on `device`, its tensor-core kernels on
    /// `coop`'s tiles (the device has their features: it was asked for them).
    fn on(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        coop: Option<Tiles>,
    ) -> Result<Gpu, String> {
        if limits.max_compute_workgroup_storage_size < WORKGROUP_MEMORY {
            return Err(format!(
                "{} bytes of workgroup memory, 32 KB needed",
                limits.max_compute_workgroup_storage_size
            ));
        }
        Ok(Gpu {
            program: Program::of(
                device,
                queue,
                limits,
                "jodel-gpu image",
                &source(coop),
                &Kernel::on(coop),
            ),
            name: String::new(),
            coop,
        })
    }

    /// A storage buffer of `bytes` (a multiple of 4), in whole vectors
    /// ([`whole_vectors`]).
    pub(crate) fn buffer(&self, label: &str, bytes: u64) -> Result<wgpu::Buffer, String> {
        self.program
            .storage(label, whole_vectors((bytes / 4) as usize))
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

/// `len` f32s rounded up to whole vectors of four: the kernels read their
/// buffers as `vec4`s, and a binding holds only whole ones, so a buffer of
/// 777 values is 194 vectors and value 776 lies past its end. Metal reads
/// such an index as the last vector's (`wgpu` clamps it there), Vulkan
/// happened to read the value; with the last vector whole, both read it.
pub(crate) fn whole_vectors(len: usize) -> usize {
    len.next_multiple_of(4)
}

/// A part's weights on the GPU: its matrices (a few buffers of words) and
/// its f32 tensors.
#[cfg_attr(test, derive(Clone))]
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

    /// The tiles its plans run the fast kernels on with `precision` (on
    /// tensor cores): where it is fast and the GPU has them.
    pub(crate) fn fast(&self, precision: Precision) -> Option<Tiles> {
        self.gpu.coop.filter(|_| precision == Precision::Fast)
    }
}

/// A plan made on the device: its buffers, its dispatches recorded, and
/// the buffer its result is read back through.
pub(crate) struct Made {
    slots: Vec<wgpu::Buffer>,
    recorded: Recorded<Kernel, [Buf; 5]>,
    staging: wgpu::Buffer,
    pieces: crate::pace::Pieces,
}

impl Made {
    /// `plan` on `loaded`'s device, `read` f32s to be read back from a run.
    pub(crate) fn new(loaded: &Loaded, plan: &Plan, read: usize) -> Result<Made, String> {
        debug_assert_eq!(plan.check(), None, "a slot read after it was handed over");
        let program = &loaded.gpu.program;
        guarded(&program.device, || {
            // The slots made and zeroed, paced ([`Program::room`]).
            let sizes: Vec<(usize, usize)> = (plan.slots.iter().enumerate())
                .map(|(i, &len)| (i, whole_vectors(len)))
                .collect();
            let mut made = program.room(&sizes, &[])?;
            let slots: Vec<wgpu::Buffer> = (0..sizes.len())
                .map(|i| made.remove(&i).expect("every slot made"))
                .collect();
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

    /// Runs it: each of `inputs` written first, then `output` read back
    /// (the plan's slots by number).
    pub(crate) fn run(
        &mut self,
        loaded: &Loaded,
        inputs: &[Input<u8>],
        output: Output<u8>,
    ) -> Result<Vec<f32>, String> {
        let slots = &self.slots;
        let slot = |i: u8| &slots[usize::from(i)];
        let writes: Vec<Write> = inputs.iter().map(|i| i.write(slot)).collect();
        let read = output.read(slot, &self.staging);
        let program = &loaded.gpu.program;
        program.run(
            &mut self.pieces,
            &self.recorded,
            &writes,
            |_| {},
            Some(read),
        )
    }
}
