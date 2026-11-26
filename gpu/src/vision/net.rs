//! The vision networks on the device: one GPU for both ([`Gpu`]: the
//! device and the program of their kernels, set up when first needed and
//! kept until its device is lost), and each network's own part ([`Net`]: its
//! weights uploaded once, and the buffers, bind groups and dispatches of
//! its plan for one input size, made when it runs and let go when asked).
//!
//! A network's weight buffer ends in a tail kept for its size-dependent
//! constants (positions and their like), as large as the largest size
//! needs: planning another size rewrites the tail and makes the plan's
//! buffers again, nothing more (no new device, no weights uploaded again).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{source, Kernel, Plan};
use crate::engine::{bytes, guarded, Kernels, Program, Recorded};
use crate::Turned;

/// The vision networks' GPU.
pub(crate) struct Gpu {
    program: Program<Kernel>,
    /// The GPU's name.
    pub(crate) name: String,
}

/// The GPU while set up (never torn down once set up: it holds no
/// network's memory), until its device is lost.
static GPU: Mutex<Option<Arc<Turned<Gpu>>>> = Mutex::new(None);

impl Gpu {
    /// The vision networks' GPU, set up when first asked for (and again
    /// once its device is lost: the networks on the lost one say so,
    /// [`Net::lost`]); why not where it cannot be (asked again, it tries
    /// again).
    pub(crate) fn shared() -> Result<Arc<Turned<Gpu>>, String> {
        let mut slot = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(gpu) = slot.as_ref().filter(|gpu| !gpu.program.lost()) {
            return Ok(Arc::clone(gpu));
        }
        let gpu = std::panic::catch_unwind(|| {
            crate::device::open("jodel-gpu vision", |device, queue, limits| {
                Gpu::on(device, queue, limits)
            })
        })
        .map_err(|_| "setting the GPU up panicked".to_owned())??;
        let (mut gpu, name) = gpu;
        gpu.name = name;
        let gpu = Arc::new(Turned::new(gpu));
        *slot = Some(Arc::clone(&gpu));
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
    recorded: Recorded<Kernel>,
    bytes: u64,
}

/// One network's part on the GPU.
pub(crate) struct Net {
    gpu: Arc<Turned<Gpu>>,
    /// The network's name: its buffers' label (and an error's, where one
    /// does not fit).
    label: &'static str,
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
            program.queue.write_buffer(&w, 0, &bytes(base));
            Ok(w)
        })?;
        Ok(Net {
            gpu,
            label,
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
        let (label, weights, base) = (self.label, &self.weights, self.base);
        let planned = guarded(&program.device, || {
            program
                .queue
                .write_buffer(weights, (base * 4) as u64, &bytes(tail));
            let buffers = plan
                .sizes
                .iter()
                .map(|&len| program.storage(label, len))
                .collect::<Result<Vec<_>, String>>()?;
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

    /// Runs the plan: `input` written to buffer `into` first, then
    /// `count` f32s of buffer `from` from `offset` read back.
    pub(crate) fn run(
        &mut self,
        into: usize,
        input: &[f32],
        (from, offset, count): (usize, usize, usize),
    ) -> Result<Vec<f32>, String> {
        let gpu = Arc::clone(&self.gpu);
        let program = &gpu.program;
        let p = self.planned.as_ref().ok_or("not planned")?;
        let pieces = &mut self.pieces;
        guarded(&program.device, || {
            program
                .queue
                .write_buffer(&p.buffers[into], 0, &bytes(input));
            program.run(
                pieces,
                (&p.recorded.groups, &p.recorded.dispatches),
                |_| {},
                Some((&p.buffers[from], offset, count)),
                &p.staging,
            )
        })
    }

    /// Each kernel's share of a run (its dispatches each submitted and
    /// waited for on its own, the best of three, timed from encoding to
    /// done), for a diagnostic: the kernel's name and work, its dispatches,
    /// its milliseconds; the slowest first.
    pub(crate) fn profile(
        &mut self,
        into: usize,
        input: &[f32],
    ) -> Result<Vec<(String, usize, f64)>, String> {
        let program = &self.gpu.program;
        let p = self.planned.as_ref().ok_or("not planned")?;
        program
            .queue
            .write_buffer(&p.buffers[into], 0, &bytes(input));
        let mut took: HashMap<String, (usize, f64)> = HashMap::new();
        for (i, d) in p.recorded.dispatches.iter().enumerate() {
            let best = program.time(&p.recorded, &|j| j == i)?;
            let [x, y, z] = d.groups;
            let entry = took
                .entry(format!("{} {x}x{y}x{z}", d.kernel.entry()))
                .or_default();
            entry.0 += 1;
            entry.1 += best.as_secs_f64() * 1000.0;
        }
        let mut out: Vec<(String, usize, f64)> =
            took.into_iter().map(|(k, (n, ms))| (k, n, ms)).collect();
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        Ok(out)
    }
}
