//! The vision networks on the device: one GPU for both ([`Gpu`]: the
//! device, its kernels and their bind group layout, set up once per process
//! when first needed and kept), and each network's own part ([`Net`]: its
//! weights uploaded once, and the buffers, bind groups and dispatches of
//! its plan for one input size, made when it runs and let go when asked).
//!
//! A network's weight buffer ends in a tail kept for its size-dependent
//! constants (positions and their like), as large as the largest size
//! needs: planning another size rewrites the tail and makes the plan's
//! buffers again, nothing more (no new device, no weights uploaded again).

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use super::{source, Kernel, Plan};
use crate::{bytes, guarded, Turned, PARAM_STRIDE};

/// Bytes of one dispatch's parameters.
const PARAMS: u64 = 96;

/// The vision networks' GPU.
pub(crate) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<Kernel, wgpu::ComputePipeline>,
    /// The most a buffer may be bound with.
    most: u64,
    /// The GPU's name.
    pub(crate) name: String,
}

/// The GPU while set up (never torn down once set up: it holds no
/// network's memory).
static GPU: Mutex<Option<Arc<Turned<Gpu>>>> = Mutex::new(None);

impl Gpu {
    /// The vision networks' GPU, set up when first asked for; why not where
    /// it cannot be (asked again, it tries again).
    pub(crate) fn shared() -> Result<Arc<Turned<Gpu>>, String> {
        let mut slot = GPU.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(gpu) = &*slot {
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
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodel-gpu vision"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(PARAMS),
                    },
                    count: None,
                },
                storage(4, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodel-gpu vision"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodel-gpu vision"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipelines = Kernel::ALL
            .iter()
            .map(|&k| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(k.entry()),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(k.entry()),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (k, pipeline)
            })
            .collect();
        Ok(Gpu {
            device,
            queue,
            layout,
            pipelines,
            most: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            name: String::new(),
        })
    }

    fn buffer(&self, label: &'static str, len: usize) -> Result<wgpu::Buffer, String> {
        let size = (len.max(1) * 4) as u64;
        if size > self.most {
            return Err(format!(
                "the GPU's buffers are too small ({size} bytes needed)"
            ));
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }
}

/// A buffer bound whole at `binding`.
fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

/// A plan on the device: its buffers, bind groups and dispatches.
struct Planned {
    buffers: Vec<wgpu::Buffer>,
    staging: wgpu::Buffer,
    groups: Vec<wgpu::BindGroup>,
    ops: Vec<crate::pace::Op<Kernel>>,
    _params: wgpu::Buffer,
    bytes: u64,
}

/// One network's part on the GPU.
pub(crate) struct Net {
    gpu: Arc<Turned<Gpu>>,
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
        let weights = guarded(&gpu.device, || {
            let w = gpu.buffer(label, base.len() + tail)?;
            gpu.queue.write_buffer(&w, 0, &bytes(base));
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
        let label = self.label;
        let weights = &self.weights;
        let base = self.base;
        let planned = guarded(&gpu.device, || {
            gpu.queue
                .write_buffer(weights, (base * 4) as u64, &bytes(tail));
            let mut total = 0;
            let buffers = plan
                .sizes
                .iter()
                .map(|&len| {
                    total += (len.max(1) * 4) as u64;
                    gpu.buffer(label, len)
                })
                .collect::<Result<Vec<_>, String>>()?;
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (read.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut data = vec![0u8; plan.ops.len().max(1) * PARAM_STRIDE as usize];
            for (i, op) in plan.ops.iter().enumerate() {
                for (j, v) in op.params.iter().enumerate() {
                    let at = i * PARAM_STRIDE as usize + 4 * j;
                    data[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&params, 0, &data);
            // A bind group for each (src, src2, dst) the plan uses.
            let mut groups = Vec::new();
            let mut index: HashMap<[usize; 3], usize> = HashMap::new();
            let mut ops = Vec::with_capacity(plan.ops.len());
            for (i, op) in plan.ops.iter().enumerate() {
                let key = [op.src, op.src2, op.dst];
                let group = *index.entry(key).or_insert_with(|| {
                    groups.push(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: &gpu.layout,
                        entries: &[
                            entry(0, &buffers[key[0]]),
                            entry(1, weights),
                            entry(2, &buffers[key[2]]),
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &params,
                                    offset: 0,
                                    size: NonZeroU64::new(PARAMS),
                                }),
                            },
                            entry(4, &buffers[key[1]]),
                        ],
                    }));
                    groups.len() - 1
                });
                ops.push((op.kernel, group, i as u64 * PARAM_STRIDE, op.groups));
            }
            Ok(Planned {
                buffers,
                staging,
                groups,
                ops,
                _params: params,
                bytes: total,
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
        let p = self.planned.as_ref().ok_or("not planned")?;
        let pieces = &mut self.pieces;
        guarded(&gpu.device, || {
            gpu.queue.write_buffer(&p.buffers[into], 0, &bytes(input));
            crate::pace::dispatch(
                &gpu.device,
                &gpu.queue,
                "jodel-gpu vision",
                pieces,
                &gpu.pipelines,
                &p.groups,
                &p.ops,
                |_| {},
                |encoder| {
                    encoder.copy_buffer_to_buffer(
                        &p.buffers[from],
                        (offset * 4) as u64,
                        &p.staging,
                        0,
                        (count * 4) as u64,
                    );
                },
            )?;
            let slice = p.staging.slice(..(count * 4) as u64);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| e.to_string())?;
            rx.recv()
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let out: Vec<f32> = slice
                .get_mapped_range()
                .map_err(|e| e.to_string())?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            p.staging.unmap();
            Ok(out)
        })
    }

    /// Each kernel's share of a run (its dispatches each submitted and
    /// waited for on its own, the best of three), for a diagnostic: the
    /// kernel's name and work, its dispatches, its milliseconds; the
    /// slowest first.
    pub(crate) fn profile(
        &mut self,
        into: usize,
        input: &[f32],
    ) -> Result<Vec<(String, usize, f64)>, String> {
        let gpu = Arc::clone(&self.gpu);
        let p = self.planned.as_ref().ok_or("not planned")?;
        gpu.queue.write_buffer(&p.buffers[into], 0, &bytes(input));
        let mut took: HashMap<String, (usize, f64)> = HashMap::new();
        for &(kernel, group, offset, [x, y, z]) in &p.ops {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&gpu.pipelines[&kernel]);
                    pass.set_bind_group(0, &p.groups[group], &[offset as u32]);
                    pass.dispatch_workgroups(x, y, z);
                }
                let started = std::time::Instant::now();
                gpu.queue.submit([encoder.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| e.to_string())?;
                best = best.min(started.elapsed().as_secs_f64() * 1000.0);
            }
            let entry = took
                .entry(format!("{} {x}x{y}x{z}", kernel.entry()))
                .or_default();
            entry.0 += 1;
            entry.1 += best;
        }
        let mut out: Vec<(String, usize, f64)> =
            took.into_iter().map(|(k, (n, ms))| (k, n, ms)).collect();
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        Ok(out)
    }
}
