//! The vision networks on the device: one GPU for both ([`Gpu`]: the
//! device and the program of their kernels, an `engine::SharedDevice` set
//! up when first needed and again once its device is lost), and each
//! network's own part ([`Net`]: its weights uploaded once, and the buffers,
//! bind groups and dispatches of its plan for one input size, made when it
//! runs and let go when asked).
//!
//! A network's weight buffer ends in a tail kept for its size-dependent
//! constants (positions and their like), as large as the largest size
//! needs: planning another size rewrites the tail and makes the plan's
//! buffers again, nothing more (no new device, no weights uploaded again).

use std::sync::Arc;

use super::{source, Kernel, Plan};
use crate::engine::{bytes, guarded, Input, OnDevice, Output, Program, Recorded, SharedDevice};
use crate::Turned;

/// The vision networks' GPU.
pub(crate) struct Gpu {
    program: Program<Kernel>,
    /// The GPU's name.
    pub(crate) name: String,
}

/// The GPU while set up (never torn down once set up: it holds no
/// network's memory), until its device is lost.
static GPU: SharedDevice<Gpu> = SharedDevice::new(Gpu::open);

impl OnDevice for Gpu {
    fn lost(&self) -> bool {
        self.program.lost()
    }
}

impl Gpu {
    /// The vision networks' GPU, set up when first asked for (and again
    /// once its device is lost: the networks on the lost one say so,
    /// [`Net::lost`]); why not where it cannot be (a failed setup stands
    /// for a while).
    pub(crate) fn shared() -> Result<Arc<Turned<Gpu>>, String> {
        GPU.get()
    }

    /// The device and the program of the kernels.
    fn open() -> Result<Gpu, String> {
        let (mut gpu, name) = crate::device::open("jodel-gpu vision", Gpu::on)?;
        gpu.name = name;
        Ok(gpu)
    }

    fn on(device: wgpu::Device, queue: wgpu::Queue, limits: &wgpu::Limits) -> Result<Gpu, String> {
        if limits.max_compute_workgroup_storage_size < 16 * 1024 {
            return Err(format!(
                "{} bytes of workgroup memory, 16 KB needed",
                limits.max_compute_workgroup_storage_size
            ));
        }
        Ok(Gpu {
            program: Program::new(device, queue, limits, "jodel-gpu vision", &source()),
            name: String::new(),
        })
    }
}

/// A plan on the device: its buffers, its dispatches recorded, and the
/// buffer its result is read back through.
struct Planned {
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    recorded: Recorded<Kernel, [usize; 3]>,
    bytes: u64,
}

/// One network's part on the GPU.
pub(crate) struct Net {
    gpu: Arc<Turned<Gpu>>,
    weights: wgpu::Buffer,
    /// The weights' f32s; the tail for constants follows them.
    base: usize,
    tail: usize,
    planned: Option<Planned>,
    /// The bytes the last plan's buffers took.
    plan_bytes: u64,
    pieces: crate::pace::Pieces,
}

impl Net {
    /// The network's weights (`base`) on the GPU, with room for `tail`
    /// f32s of constants after them.
    pub(crate) fn new(label: &'static str, base: &[f32], tail: usize) -> Result<Net, String> {
        let gpu = Gpu::shared()?;
        let program = &gpu.program;
        let weights = guarded(&program.device, || {
            let w = program.storage(label, base.len() + tail)?;
            let mut upload = program.upload();
            upload.write(&w, 0, &bytes(base))?;
            upload.finish()?;
            Ok(w)
        })?;
        Ok(Net {
            gpu,
            weights,
            base: base.len(),
            tail,
            planned: None,
            plan_bytes: 0,
            pieces: crate::pace::Pieces::resting(),
        })
    }

    /// The GPU's name.
    pub(crate) fn gpu_name(&self) -> &str {
        &self.gpu.name
    }

    /// Whether its device was lost: it runs nothing again.
    pub(crate) fn lost(&self) -> bool {
        self.gpu.program.lost()
    }

    /// Where the tail starts (constants are packed from there).
    pub(crate) fn tail_origin(&self) -> usize {
        self.base
    }

    /// Whether a plan is made on the device.
    pub(crate) fn is_planned(&self) -> bool {
        self.planned.is_some()
    }

    /// The bytes the network holds on the GPU: its weights and its plan's
    /// buffers.
    pub(crate) fn bytes(&self) -> u64 {
        ((self.base + self.tail) * 4) as u64 + self.planned.as_ref().map_or(0, |p| p.bytes)
    }

    /// The weights' bytes and the last plan's (held while it runs).
    pub(crate) fn sizes(&self) -> (u64, u64) {
        (((self.base + self.tail) * 4) as u64, self.plan_bytes)
    }

    /// Lets the plan's buffers go (the weights stay).
    pub(crate) fn release(&mut self) {
        self.planned = None;
    }

    /// Makes `plan` on the device: `tail` written after the weights, the
    /// plan's buffers, bind groups and parameters; `read` f32s may be read
    /// back from a run. The last plan's buffers go first.
    pub(crate) fn plan(&mut self, plan: &Plan, tail: &[f32], read: usize) -> Result<(), String> {
        if tail.len() > self.tail {
            return Err(format!("{} constants, room for {}", tail.len(), self.tail));
        }
        self.planned = None;
        let gpu = Arc::clone(&self.gpu);
        let program = &gpu.program;
        let (weights, base) = (&self.weights, self.base);
        let planned = guarded(&program.device, || {
            let mut upload = program.upload();
            upload.write(weights, (base * 4) as u64, &bytes(tail))?;
            upload.finish()?;
            // The plan's buffers made and zeroed, paced ([`Program::room`]).
            let sizes: Vec<(usize, usize)> = plan.sizes.iter().copied().enumerate().collect();
            let mut made = program.room(&sizes, &[])?;
            let buffers: Vec<wgpu::Buffer> = (0..sizes.len())
                .map(|i| made.remove(&i).expect("every buffer made"))
                .collect();
            let staging = program.staging(read);
            // A kernel's buffers: src, the weights, dst, then src2.
            let recorded = program.record(&plan.ops, |[src, src2, dst]: [usize; 3]| {
                vec![&buffers[src], weights, &buffers[dst], &buffers[src2]]
            })?;
            Ok(Planned {
                bytes: buffers.iter().map(wgpu::Buffer::size).sum(),
                buffers,
                staging,
                recorded,
            })
        })?;
        self.plan_bytes = planned.bytes;
        self.planned = Some(planned);
        Ok(())
    }

    /// Runs the plan: `input` written first, then `output` read back (the
    /// plan's buffers by index).
    pub(crate) fn run(
        &mut self,
        input: Input<usize>,
        output: Output<usize>,
    ) -> Result<Vec<f32>, String> {
        let p = self.planned.as_ref().ok_or("not planned")?;
        let buffer = |i: usize| &p.buffers[i];
        let (write, read) = (input.write(buffer), output.read(buffer, &p.staging));
        let program = &self.gpu.program;
        program.run(&mut self.pieces, &p.recorded, &[write], |_| {}, Some(read))
    }
}
