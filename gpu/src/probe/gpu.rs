//! The probe's GPU ([`Gpu`]): a device of its own on the chosen adapter
//! with the features the probes may use, kernels made from WGSL text (a
//! failure is an `Err`, never a panic), buffers of words, runs timed by
//! the wall clock, and their words read back.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

/// The features the probes use where the adapter has them (the
/// cooperative matrices only on a device of their own: [`Gpu::open`]).
pub(super) const WANTED: wgpu::Features = wgpu::Features::SHADER_F16
    .union(wgpu::Features::SUBGROUP)
    .union(wgpu::Features::SUBGROUP_BARRIER);

/// What the cooperative matrices' device asks for.
pub(super) const COOP: wgpu::Features =
    wgpu::Features::SHADER_F16.union(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);

/// A device of the probe's own, on the process's chosen adapter.
pub(super) struct Gpu {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) limits: wgpu::Limits,
}

/// A buffer as a run binds it: the whole of it, or its first bytes.
#[derive(Clone, Copy)]
pub(super) enum Bound<'a> {
    All(&'a wgpu::Buffer),
    First(&'a wgpu::Buffer, u64),
}

/// `work`, a panic in it an `Err` (a probe never takes the process down).
pub(super) fn caught<T>(work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "a panic".into());
        Err(format!("panicked: {why}"))
    })
}

impl Gpu {
    /// A device on the chosen adapter with those of `features` it has
    /// (and wgpu's experimental features where `features` holds the
    /// cooperative matrices and the adapter has them), and every limit
    /// it has.
    pub(super) fn open(label: &'static str, features: wgpu::Features) -> Result<Gpu, String> {
        let (gpu, _) = crate::device::open_with(
            label,
            |adapter| {
                let wanted = adapter.features() & features;
                let experimental =
                    if wanted.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
                        // SAFETY: wgpu's cooperative matrices are experimental:
                        // its checks of a kernel's use of them may be
                        // incomplete, so a wrong kernel could misbehave on the
                        // device instead of failing validation. The probe's
                        // are a single multiply-add of one tile shape the
                        // adapter lists, on a device of their own.
                        unsafe { wgpu::ExperimentalFeatures::enabled() }
                    } else {
                        wgpu::ExperimentalFeatures::disabled()
                    };
                (wanted, experimental, ())
            },
            |device, queue, limits, ()| {
                // An error outside the probes' scopes is printed, not a
                // panic (wgpu's default).
                device.on_uncaptured_error(Arc::new(|e| eprintln!("probe: GPU error: {e}")));
                Ok(Gpu {
                    device,
                    queue,
                    limits: limits.clone(),
                })
            },
        )?;
        Ok(gpu)
    }

    /// Whether the device has all of `features`.
    pub(super) fn has(&self, features: wgpu::Features) -> bool {
        self.device.features().contains(features)
    }

    /// `work` inside the device's error scopes and with panics caught.
    pub(super) fn guarded<T>(&self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        caught(|| crate::engine::guarded(&self.device, work))
    }

    /// The kernel `entry` of `source`.
    pub(super) fn kernel(
        &self,
        source: &str,
        entry: &str,
    ) -> Result<wgpu::ComputePipeline, String> {
        self.kernel_zeroing(source, entry, true)
    }

    /// The kernel `entry` of `source`, its workgroup memory zeroed by wgpu
    /// at the start of each workgroup (`zeroed`, wgpu's default) or not.
    pub(super) fn kernel_zeroing(
        &self,
        source: &str,
        entry: &str,
        zeroed: bool,
    ) -> Result<wgpu::ComputePipeline, String> {
        self.guarded(|| {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jodel-gpu probe"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            Ok(self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: zeroed,
                        ..Default::default()
                    },
                    cache: None,
                }))
        })
    }

    /// A storage buffer holding `words`.
    pub(super) fn words(&self, words: &[u32]) -> Result<wgpu::Buffer, String> {
        self.fits(words.len() as u64 * 4)?;
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.guarded(|| {
            Ok(self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jodel-gpu probe"),
                    contents: &bytes,
                    usage: usage(),
                }))
        })
    }

    /// A storage buffer of `bytes` zeros.
    pub(super) fn zeros(&self, bytes: u64) -> Result<wgpu::Buffer, String> {
        self.fits(bytes)?;
        self.guarded(|| {
            Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodel-gpu probe"),
                size: bytes,
                usage: usage(),
                mapped_at_creation: false,
            }))
        })
    }

    /// `Err` where a binding of `bytes` is past the device's limits.
    fn fits(&self, bytes: u64) -> Result<(), String> {
        let most = self.largest_binding();
        if bytes > most {
            return Err(format!(
                "{bytes} bytes is more than a binding takes ({most})"
            ));
        }
        Ok(())
    }

    /// The most bytes a buffer may be bound with.
    pub(super) fn largest_binding(&self) -> u64 {
        self.limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size)
    }

    /// `buffer` set to zeros.
    pub(super) fn clear(&self, buffer: &wgpu::Buffer) -> Result<(), String> {
        self.guarded(|| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(buffer, 0, None);
            self.queue.submit([encoder.finish()]);
            self.wait()
        })
    }

    /// `kernel` dispatched `times` times in one pass over `groups`
    /// workgroups, the buffers bound at 0, 1, … in order, waited for; the
    /// wall time from the submission to the end.
    pub(super) fn run(
        &self,
        kernel: &wgpu::ComputePipeline,
        buffers: &[Bound],
        groups: [u32; 3],
        times: u32,
    ) -> Result<Duration, String> {
        self.guarded(|| {
            let entries: Vec<_> = (0..)
                .zip(buffers)
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding,
                    resource: match *b {
                        Bound::All(buffer) => buffer.as_entire_binding(),
                        Bound::First(buffer, bytes) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(bytes),
                            })
                        }
                    },
                })
                .collect();
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodel-gpu probe"),
                layout: &kernel.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(kernel);
                pass.set_bind_group(0, &group, &[]);
                for _ in 0..times {
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
            }
            let command = encoder.finish();
            let started = Instant::now();
            self.queue.submit([command]);
            self.wait()?;
            Ok(started.elapsed())
        })
    }

    /// Waits for the queue's work.
    fn wait(&self) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// The words of `buffer`.
    pub(super) fn read(&self, buffer: &wgpu::Buffer) -> Result<Vec<u32>, String> {
        self.guarded(|| {
            let size = buffer.size();
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodel-gpu probe staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
            self.queue.submit([encoder.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.wait()?;
            rx.recv()
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            let words = slice
                .get_mapped_range()
                .map_err(|e| e.to_string())?
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            staging.unmap();
            Ok(words)
        })
    }

    /// The f32s of `buffer`.
    pub(super) fn read_f32(&self, buffer: &wgpu::Buffer) -> Result<Vec<f32>, String> {
        Ok(self.read(buffer)?.into_iter().map(f32::from_bits).collect())
    }

    /// The rate of `work` units a dispatch of `kernel`: dispatched once to
    /// warm up, then as many times in one pass as take about `target`
    /// (at most 256): units per second.
    pub(super) fn rate(
        &self,
        kernel: &wgpu::ComputePipeline,
        buffers: &[Bound],
        groups: [u32; 3],
        work: f64,
        target: Duration,
    ) -> Result<f64, String> {
        let once = self.run(kernel, buffers, groups, 1)?;
        let times = (target.as_secs_f64() / once.as_secs_f64().max(1e-6)).clamp(1.0, 256.0) as u32;
        let took = self.run(kernel, buffers, groups, times)?;
        Ok(work * f64::from(times) / took.as_secs_f64().max(1e-9))
    }
}

/// How the probe's buffers are used: bound for storage, copied to and
/// from.
fn usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

/// The f32s as words.
pub(super) fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A small deterministic generator of words (xorshift), for the probes'
/// inputs.
pub(super) struct Noise(u64);

impl Noise {
    pub(super) fn new(seed: u64) -> Noise {
        Noise(seed.max(1))
    }

    pub(super) fn word(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    /// A value in `[0, 1)`.
    pub(super) fn unit(&mut self) -> f64 {
        f64::from(self.word()) / 4_294_967_296.0
    }
}
