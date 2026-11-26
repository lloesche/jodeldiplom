//! A device and a program of kernels on it ([`Program`]), and dispatches
//! recorded for it ([`Recorded`]).

use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{Access, Dispatch, Kernels, Op, PARAM_STRIDE};
use crate::pace::{self, Pieces};

/// A device and a program of kernels on it: the module, the bind group
/// layout and each kernel's pipeline, with what makes buffers, parameters
/// and bind groups for it; and whether the device was lost.
pub(crate) struct Program<K> {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    label: &'static str,
    /// Set when the device is lost (its callback).
    lost: Arc<AtomicBool>,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<K, wgpu::ComputePipeline>,
    /// The most a buffer may be bound with.
    most: u64,
}

/// Dispatches on the device: their parameters, a bind group for each set
/// of buffers they bind, and what to dispatch.
pub(crate) struct Recorded<K> {
    pub(crate) dispatches: Vec<Dispatch<K>>,
    pub(crate) groups: Vec<wgpu::BindGroup>,
    /// Kept while the bind groups use it.
    _params: wgpu::Buffer,
}

impl<K: Kernels> Program<K> {
    /// `source`'s kernels (every one of `K::ALL`) on the device, labelled
    /// `label`, on the layout `K` says; the device watched for its loss.
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        label: &'static str,
        source: &str,
    ) -> Program<K> {
        let lost = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&lost);
        device.set_device_lost_callback(move |_, _| flag.store(true, Ordering::Release));
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
        let mut entries = vec![
            storage(0, true),
            storage(1, true),
            storage(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(K::PARAM_BYTES),
                },
                count: None,
            },
        ];
        entries.extend(
            (4..)
                .zip(K::EXTRA)
                .map(|(b, &access)| storage(b, access == Access::Read)),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipelines = K::ALL
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
        Program {
            device,
            queue,
            label,
            lost,
            layout,
            pipelines,
            most: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
        }
    }

    /// Its label.
    pub(crate) fn label(&self) -> &'static str {
        self.label
    }

    /// Whether its device was lost (a driver reset, the GPU gone): nothing
    /// runs on it again. The device is polled first, so a loss it found is
    /// told.
    pub(crate) fn lost(&self) -> bool {
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.lost.load(Ordering::Acquire)
    }

    /// The pipeline of `kernel`.
    pub(crate) fn pipeline(&self, kernel: K) -> &wgpu::ComputePipeline {
        &self.pipelines[&kernel]
    }

    /// The most bytes a buffer may be bound with.
    pub(crate) fn most(&self) -> u64 {
        self.most
    }

    /// A storage buffer of `len` f32s labelled `label` (the error names it
    /// where it does not fit), copied to and from.
    pub(crate) fn storage(&self, label: &str, len: usize) -> Result<wgpu::Buffer, String> {
        let size = (len.max(1) * 4) as u64;
        if size > self.most {
            return Err(format!(
                "the GPU's buffers are too small for {label} ({size} bytes)"
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

    /// A storage buffer for each of `sizes` (each one's f32s), labelled by
    /// the program and its name.
    pub(crate) fn buffers<B: Copy + Eq + Hash + std::fmt::Debug>(
        &self,
        sizes: &[(B, usize)],
    ) -> Result<HashMap<B, wgpu::Buffer>, String> {
        sizes
            .iter()
            .map(|&(buf, len)| Ok((buf, self.storage(&format!("{} {buf:?}", self.label), len)?)))
            .collect()
    }

    /// A buffer `len` f32s are read back through.
    pub(crate) fn staging(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// An empty uniform buffer for the parameters of `ops` dispatches.
    pub(crate) fn params_buffer(&self, ops: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: ops.max(1) as u64 * PARAM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The parameters of `ops` written to `buffer`, the dispatch `i`'s at
    /// offset `i * PARAM_STRIDE`; `Err` where a number does not fit 32
    /// bits.
    pub(crate) fn write_params<'a, B: 'a, const P: usize>(
        &self,
        buffer: &wgpu::Buffer,
        ops: impl IntoIterator<Item = &'a Op<K, B, P>>,
    ) -> Result<(), String> {
        debug_assert_eq!((P * 4) as u64, K::PARAM_BYTES, "a dispatch's parameters");
        let mut data = vec![0u8; PARAM_STRIDE as usize];
        for (i, op) in ops.into_iter().enumerate() {
            data.resize((i + 1) * PARAM_STRIDE as usize, 0);
            for (j, &v) in op.params.iter().enumerate() {
                let v = u32::try_from(v).map_err(|_| {
                    format!("{}'s number {j} is {v}, past 32 bits", op.kernel.entry())
                })?;
                let at = i * PARAM_STRIDE as usize + 4 * j;
                data[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        self.queue.write_buffer(buffer, 0, &data);
        Ok(())
    }

    /// The bind group of `buffers` (at bindings 0, 1 and 2, then from 4
    /// on) with the parameters in `params`.
    pub(crate) fn bind_group(
        &self,
        buffers: &[&wgpu::Buffer],
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let mut entries: Vec<wgpu::BindGroupEntry> = (0..)
            .filter(|&b| b != 3)
            .zip(buffers)
            .map(|(binding, buf)| wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: params,
                offset: 0,
                size: NonZeroU64::new(K::PARAM_BYTES),
            }),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.label),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// `ops` on the device: their parameters in a buffer of their own, a
    /// bind group for each set of buffers they bind (`bound` gives a key's
    /// buffers, in binding order: 0, 1, 2, then 4 on), and their dispatches.
    pub(crate) fn record<'a, B: Copy + Eq + Hash, const P: usize>(
        &self,
        ops: &[Op<K, B, P>],
        bound: impl Fn(B) -> Vec<&'a wgpu::Buffer>,
    ) -> Result<Recorded<K>, String> {
        let params = self.params_buffer(ops.len());
        self.write_params(&params, ops)?;
        let mut groups = Vec::new();
        let mut index: HashMap<B, usize> = HashMap::new();
        let dispatches = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = *index.entry(op.buffers).or_insert_with(|| {
                    groups.push(self.bind_group(&bound(op.buffers), &params));
                    groups.len() - 1
                });
                Dispatch {
                    kernel: op.kernel,
                    group,
                    offset: i as u64 * PARAM_STRIDE,
                    groups: op.groups,
                }
            })
            .collect();
        Ok(Recorded {
            dispatches,
            groups,
            _params: params,
        })
    }

    /// `dispatches` (on the bind groups `groups`) in short submissions
    /// ([`pace::dispatch`]), `prelude` before them; then `read`'s f32s (a
    /// buffer, the first one's index, how many) copied to `staging` and
    /// read back (none without `read`).
    pub(crate) fn run(
        &self,
        pieces: &mut Pieces,
        (groups, dispatches): (&[wgpu::BindGroup], &[Dispatch<K>]),
        prelude: impl FnOnce(&mut wgpu::CommandEncoder),
        read: Option<(&wgpu::Buffer, usize, usize)>,
        staging: &wgpu::Buffer,
    ) -> Result<Vec<f32>, String> {
        pace::dispatch(self, pieces, groups, dispatches, prelude, |encoder| {
            if let Some((from, first, count)) = read {
                let size = (count * 4) as u64;
                encoder.copy_buffer_to_buffer(from, (first * 4) as u64, staging, 0, size);
            }
        })?;
        match read {
            Some((_, _, count)) => crate::device::read_back(&self.device, staging, count),
            None => Ok(Vec::new()),
        }
    }

    /// The dispatches of `recorded` that `keep` picks (by index) as one
    /// submission, waited for, three times: the best time, from encoding to
    /// done. For profiles.
    pub(crate) fn time(
        &self,
        recorded: &Recorded<K>,
        keep: &dyn Fn(usize) -> bool,
    ) -> Result<Duration, String> {
        let mut best = Duration::MAX;
        for _ in 0..3 {
            let started = Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(self.label),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(self.label),
                    timestamp_writes: None,
                });
                for (i, d) in recorded.dispatches.iter().enumerate() {
                    if keep(i) {
                        let [x, y, z] = d.groups;
                        pass.set_pipeline(self.pipeline(d.kernel));
                        pass.set_bind_group(0, &recorded.groups[d.group], &[d.offset as u32]);
                        pass.dispatch_workgroups(x, y, z);
                    }
                }
            }
            self.queue.submit([encoder.finish()]);
            self.wait()?;
            best = best.min(started.elapsed());
        }
        Ok(best)
    }

    /// Waits for everything sent to the GPU so far.
    pub(crate) fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}
