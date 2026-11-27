//! A device and a program of kernels on it ([`Program`]), dispatches
//! recorded for it ([`Recorded`]), and their runs: waited for
//! ([`Program::run`]), or left running ([`Program::run_left`]: an
//! [`InFlight`] the caller finishes later).

use std::collections::HashMap;
use std::hash::Hash;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{bytes, guarded, Access, Dispatch, Kernels, Op, PARAM_STRIDE};
use crate::pace::{self, Left, Pieces};
use crate::profile::Stamps;

/// A device and a program of kernels on it: the module, the bind group
/// layout and each kernel's pipeline, with what makes buffers, parameters
/// and bind groups for it; and whether the device was lost.
pub(crate) struct Program<K> {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    label: &'static str,
    /// Set when the device is lost (its callback).
    lost: Arc<AtomicBool>,
    /// Whether the last run on the device was left running
    /// ([`Program::run_left`]): the next run's first wait may wait for it
    /// too ([`pace`]).
    pub(crate) left: AtomicBool,
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<K, wgpu::ComputePipeline>,
    /// The most a buffer may be bound with.
    most: u64,
}

/// Dispatches on the device: what to dispatch, their parameters (in a
/// buffer with room for so many), and a bind group for each set of
/// buffers they bind (`B`, a key its network turns into buffers), kept
/// from one recording to the next ([`Recorded::again`]).
pub(crate) struct Recorded<K, B> {
    dispatches: Vec<Dispatch<K>>,
    groups: Vec<wgpu::BindGroup>,
    /// Each set of buffers' bind group (its index in `groups`).
    index: HashMap<B, usize>,
    /// The dispatches' parameters, kept while the bind groups use it.
    params: wgpu::Buffer,
    /// The most dispatches its parameters have room for.
    room: usize,
}

/// What a run reads back: `count` f32s of `from` from its `first`,
/// copied to `staging` (room for them) and read from there.
#[derive(Clone, Copy)]
pub(crate) struct Read<'a> {
    pub(crate) from: &'a wgpu::Buffer,
    pub(crate) first: usize,
    pub(crate) count: usize,
    pub(crate) staging: &'a wgpu::Buffer,
}

/// What is written before a run: `values` to `to` from its f32 `at`.
#[derive(Clone, Copy)]
pub(crate) struct Write<'a> {
    pub(crate) to: &'a wgpu::Buffer,
    pub(crate) at: usize,
    pub(crate) values: &'a [f32],
}

/// What a network writes before a run of its plan: `values` to its buffer
/// `buf` (`B`, its name for it) from its f32 `at`.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a, B> {
    pub(crate) buf: B,
    pub(crate) at: usize,
    pub(crate) values: &'a [f32],
}

/// What a network reads back of a run of its plan: `count` f32s of its
/// buffer `buf` (`B`, its name for it) from its `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Output<B> {
    pub(crate) buf: B,
    pub(crate) first: usize,
    pub(crate) count: usize,
}

impl<'a, B: Copy> Input<'a, B> {
    /// The write it is, its buffer as `buffer` finds it.
    pub(crate) fn write(&self, buffer: impl Fn(B) -> &'a wgpu::Buffer) -> Write<'a> {
        Write {
            to: buffer(self.buf),
            at: self.at,
            values: self.values,
        }
    }
}

impl<B: Copy> Output<B> {
    /// The read it is, its buffer as `buffer` finds it, through `staging`.
    pub(crate) fn read<'a>(
        self,
        buffer: impl Fn(B) -> &'a wgpu::Buffer,
        staging: &'a wgpu::Buffer,
    ) -> Read<'a> {
        Read {
            from: buffer(self.buf),
            first: self.first,
            count: self.count,
            staging,
        }
    }
}

/// `work` on what `held` holds, locked (`Err(failed)` where a use
/// panicked before), a failure on the device of its program (`program`
/// finds it) an `Err` ([`guarded`]): a use of a model the caller holds.
pub(crate) fn using<I, K, R>(
    held: &Mutex<I>,
    failed: &str,
    program: fn(&I) -> &Program<K>,
    work: impl FnOnce(&mut I) -> Result<R, String>,
) -> Result<R, String> {
    let mut inner = held.lock().map_err(|_| failed.to_owned())?;
    let device = program(&inner).device.clone();
    guarded(&device, || work(&mut inner))
}

/// A run made and maybe still running ([`Program::run_left`]): its last
/// submission where it was left running, and what it reads back
/// ([`Program::finish`] waits for it and reads that).
pub(crate) struct InFlight {
    left: Option<Left>,
    staging: Option<wgpu::Buffer>,
    count: usize,
}

impl<K: Kernels> Program<K> {
    /// `source`'s kernels (all of `K::ALL`) on the device, labelled
    /// `label`, on the layout `K` says; the device watched for its loss.
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        label: &'static str,
        source: &str,
    ) -> Program<K> {
        Program::of(device, queue, limits, label, source, K::ALL)
    }

    /// [`Program::new`] with only `kernels`, those `source` holds (a
    /// program whose kernels depend on the device).
    pub(crate) fn of(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        label: &'static str,
        source: &str,
        kernels: &[K],
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
        let pipelines = kernels
            .iter()
            .map(|&k| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(k.entry()),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(k.entry()),
                    // Workgroup memory is not zeroed at each dispatch's
                    // start (on Metal that costs ~1.9 ms a dispatch): every
                    // kernel writes what it reads first (gpu/README.md,
                    // "Rules for kernels", rule 6).
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
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
            left: AtomicBool::new(false),
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

    /// Room for `room` dispatches recorded at a time ([`Recorded::again`]),
    /// none recorded yet.
    pub(crate) fn recording<B>(&self, room: usize) -> Recorded<K, B> {
        Recorded {
            dispatches: Vec::new(),
            groups: Vec::new(),
            index: HashMap::new(),
            params: self.params_buffer(room),
            room,
        }
    }

    /// `ops` on the device: their parameters in a buffer of their own, a
    /// bind group for each set of buffers they bind (`bound` gives a key's
    /// buffers, in binding order: 0, 1, 2, then 4 on), and their dispatches.
    pub(crate) fn record<'a, B: Copy + Eq + Hash, const P: usize>(
        &self,
        ops: &[Op<K, B, P>],
        bound: impl Fn(B) -> Vec<&'a wgpu::Buffer>,
    ) -> Result<Recorded<K, B>, String> {
        let mut recorded = self.recording(ops.len());
        recorded.again(self, ops, bound)?;
        Ok(recorded)
    }

    /// `recorded`'s dispatches in short submissions ([`pace::dispatch`]),
    /// `writes` written first and `prelude` before the dispatches; then
    /// what `read` says read back (nothing without it). A failure on the
    /// device is an `Err` ([`guarded`]).
    pub(crate) fn run<B>(
        &self,
        pieces: &mut Pieces,
        recorded: &Recorded<K, B>,
        writes: &[Write],
        prelude: impl FnOnce(&mut wgpu::CommandEncoder),
        read: Option<Read>,
    ) -> Result<Vec<f32>, String> {
        guarded(&self.device, || {
            for w in writes {
                let at = (w.at * 4) as u64;
                self.queue.write_buffer(w.to, at, &bytes(w.values));
            }
            pace::dispatch(self, pieces, recorded, (prelude, copy(read)))?;
            match read {
                Some(r) => crate::device::read_back(&self.device, r.staging, r.count),
                None => Ok(Vec::new()),
            }
        })
    }

    /// [`Program::run`]'s dispatches of `recorded` and the copy of what
    /// `read` says, made; the last submission may still run when it
    /// returns ([`pace::dispatch_left`]): the caller does more meanwhile,
    /// then [`Program::finish`]es it.
    pub(crate) fn run_left<B>(
        &self,
        pieces: &mut Pieces,
        recorded: &Recorded<K, B>,
        read: Option<Read>,
    ) -> Result<InFlight, String> {
        guarded(&self.device, || {
            let left = pace::dispatch_left(self, pieces, recorded, copy(read))?;
            Ok(InFlight {
                left,
                staging: read.map(|r| r.staging.clone()),
                count: read.map_or(0, |r| r.count),
            })
        })
    }

    /// Waits for `in_flight` (what it took learnt in `pieces`, those it
    /// was run with) and reads back what it copied.
    pub(crate) fn finish(
        &self,
        in_flight: InFlight,
        pieces: &mut Pieces,
    ) -> Result<Vec<f32>, String> {
        guarded(&self.device, || {
            if let Some(left) = in_flight.left {
                pace::finish(pieces, &self.device, left)?;
            }
            match &in_flight.staging {
                Some(staging) => crate::device::read_back(&self.device, staging, in_flight.count),
                None => Ok(Vec::new()),
            }
        })
    }

    /// `dispatches` (on the bind groups `groups`) into `encoder`: in one
    /// compute pass, or each in a pass of its own that `stamps` times
    /// ([`crate::profile`]).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        groups: &[wgpu::BindGroup],
        dispatches: &[Dispatch<K>],
        stamps: Option<&Stamps>,
    ) {
        let put = |pass: &mut wgpu::ComputePass<'_>, d: &Dispatch<K>| {
            let [x, y, z] = d.groups;
            pass.set_pipeline(self.pipeline(d.kernel));
            pass.set_bind_group(0, &groups[d.group], &[d.offset as u32]);
            pass.dispatch_workgroups(x, y, z);
        };
        match stamps.filter(|s| !s.whole()) {
            None => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(self.label),
                    timestamp_writes: stamps.map(|s| s.writes(0)),
                });
                for d in dispatches {
                    put(&mut pass, d);
                }
                drop(pass);
                if let Some(stamps) = stamps {
                    stamps.resolve(encoder);
                }
            }
            Some(stamps) => {
                for (i, d) in dispatches.iter().enumerate() {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(self.label),
                        timestamp_writes: Some(stamps.writes(i)),
                    });
                    put(&mut pass, d);
                }
                stamps.resolve(encoder);
            }
        }
    }

    /// Writes of a model's weights to its buffers ([`Upload::write`]).
    pub(crate) fn upload(&self) -> Upload<'_, K> {
        Upload {
            program: self,
            staged: 0,
        }
    }

    /// Waits for everything sent to the GPU so far.
    pub(crate) fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// What copies `read`'s values to its staging buffer, after a run's
/// dispatches.
fn copy(read: Option<Read<'_>>) -> impl FnOnce(&mut wgpu::CommandEncoder) + '_ {
    move |encoder| {
        if let Some(r) = read {
            let size = (r.count * 4) as u64;
            encoder.copy_buffer_to_buffer(r.from, (r.first * 4) as u64, r.staging, 0, size);
        }
    }
}

impl<K: Kernels, B: Copy + Eq + Hash> Recorded<K, B> {
    /// `ops` recorded in place of what it held: their parameters written
    /// over the last ones (in the queue's order, so after the work sent
    /// before: a run left running reads its own), a bind group made for
    /// each set of buffers not bound before (`bound` gives a key's buffers,
    /// in binding order: 0, 1, 2, then 4 on); `Err` past its room.
    pub(crate) fn again<'a, const P: usize>(
        &mut self,
        program: &Program<K>,
        ops: &[Op<K, B, P>],
        bound: impl Fn(B) -> Vec<&'a wgpu::Buffer>,
    ) -> Result<(), String> {
        if ops.len() > self.room {
            return Err(format!("{} dispatches, room for {}", ops.len(), self.room));
        }
        program.write_params(&self.params, ops)?;
        let (groups, index, params) = (&mut self.groups, &mut self.index, &self.params);
        self.dispatches = ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let group = *index.entry(op.buffers).or_insert_with(|| {
                    groups.push(program.bind_group(&bound(op.buffers), params));
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
        Ok(())
    }

    /// Its dispatches `times` over, one after the other (dispatches that
    /// read where they are from the GPU: the language model's steps).
    pub(crate) fn repeat(&mut self, times: usize) {
        let once = std::mem::take(&mut self.dispatches);
        self.dispatches = (0..times).flat_map(|_| once.iter().copied()).collect();
    }

    /// The bytes of its parameters' buffer.
    pub(crate) fn params_bytes(&self) -> u64 {
        self.params.size()
    }
}

impl<K, B> Recorded<K, B> {
    /// What it dispatches, in order.
    pub(crate) fn dispatches(&self) -> &[Dispatch<K>] {
        &self.dispatches
    }

    /// The bind groups its dispatches bind (by index).
    pub(crate) fn groups(&self) -> &[wgpu::BindGroup] {
        &self.groups
    }
}

/// The bytes written to the GPU between waits as weights go up: wgpu
/// stages each write until it is sent, so without a wait the staging of a
/// whole model would be held at once.
const UPLOAD_BATCH: u64 = 256 << 20;

/// A model's weights going up to its buffers, the copies waited for every
/// [`UPLOAD_BATCH`] bytes and at the end ([`Upload::finish`]).
pub(crate) struct Upload<'a, K> {
    program: &'a Program<K>,
    staged: u64,
}

impl<K: Kernels> Upload<'_, K> {
    /// `data` into `buffer` from byte `at` (a multiple of 4; a last part
    /// word written with zeros after it).
    pub(crate) fn write(
        &mut self,
        buffer: &wgpu::Buffer,
        at: u64,
        data: &[u8],
    ) -> Result<(), String> {
        debug_assert!(at.is_multiple_of(4), "an upload to byte {at}");
        let queue = &self.program.queue;
        let whole = data.len() & !3;
        queue.write_buffer(buffer, at, &data[..whole]);
        if whole < data.len() {
            let mut tail = [0u8; 4];
            tail[..data.len() - whole].copy_from_slice(&data[whole..]);
            queue.write_buffer(buffer, at + whole as u64, &tail);
        }
        self.staged += data.len() as u64;
        if self.staged >= UPLOAD_BATCH {
            self.staged = 0;
            queue.submit([]);
            self.program.wait()?;
        }
        Ok(())
    }

    /// Every write sent and waited for.
    pub(crate) fn finish(self) -> Result<(), String> {
        self.program.queue.submit([]);
        self.program.wait()
    }
}
