//! Time on the GPU, for profiles. While profiling is on ([`start`]), the
//! GPU stamps the work made through [`crate::pace`] (timestamp queries,
//! where the adapter has them), each submission is waited for at once,
//! and the times are added up by program ([`take`]): each kernel's, every
//! dispatch in a compute pass of its own ([`Detail::Kernels`]), or each
//! submission's as a whole, the GPU's time between its dispatches too
//! ([`Detail::Submissions`]). Off (the default), nothing is stamped and
//! nothing waits for it.
//!
//! A pass is stamped where it starts, once the GPU is done with what came
//! before it (and an empty pass after the last): a dispatch's time is from
//! its pass's start to the next one's, what it costs where it stands, its
//! wait for the one before to drain and its own start included. (Where a
//! pass ends is not stamped: some drivers stamp it as soon as its work is
//! started.)

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// 0: off; else a [`Detail`] and one.
static ON: AtomicU8 = AtomicU8::new(0);
/// Each dispatch's GPU time in ns, by program, kernel and workgroups.
static TIMES: Mutex<Vec<(Key, Vec<f64>)>> = Mutex::new(Vec::new());

/// A program, a kernel and its workgroups: dispatches of the same work.
type Key = (&'static str, &'static str, [u32; 3]);

/// What a submission stamped as a whole is filed under.
const WHOLE: (&str, [u32; 3]) = ("submissions", [0; 3]);

/// What a profile times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    /// Each dispatch, by its kernel.
    Kernels,
    /// Each submission as a whole (as `submissions`).
    Submissions,
}

/// Starts profiling (what was collected before is dropped).
pub fn start(detail: Detail) {
    times().clear();
    ON.store(detail as u8 + 1, Ordering::Release);
}

/// What profiling times now, if it is on.
fn detail() -> Option<Detail> {
    match ON.load(Ordering::Acquire) {
        1 => Some(Detail::Kernels),
        2 => Some(Detail::Submissions),
        _ => None,
    }
}

/// The middle value of `values` (sorted on the way).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// Stops profiling: what was collected since [`start`], each kernel's
/// dispatches of the same workgroups taken together.
pub fn take() -> Profile {
    ON.store(0, Ordering::Release);
    let mut kernels: Vec<KernelTime> = Vec::new();
    let mut shapes = Vec::new();
    for ((program, kernel, groups), mut took) in times().drain(..) {
        let calls = took.len() as u64;
        let ns = |v: f64| Duration::from_nanos(v as u64);
        let gpu = ns(took.iter().sum());
        let typical = ns(median(&mut took) * calls as f64);
        shapes.push(Shape {
            program,
            kernel,
            groups,
            calls,
            typical,
        });
        match kernels
            .iter_mut()
            .find(|k| (k.program, k.kernel) == (program, kernel))
        {
            Some(k) => {
                k.calls += calls;
                k.gpu += gpu;
                k.typical += typical;
            }
            None => kernels.push(KernelTime {
                program,
                kernel,
                calls,
                gpu,
                typical,
            }),
        }
    }
    kernels.sort_by_key(|k| std::cmp::Reverse(k.typical));
    shapes.sort_by_key(|s| std::cmp::Reverse(s.typical));
    Profile { kernels, shapes }
}

fn times() -> std::sync::MutexGuard<'static, Vec<(Key, Vec<f64>)>> {
    TIMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// What a profile found: each kernel's dispatches and GPU time, and each
/// kernel's of the same workgroups, the longest first.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub kernels: Vec<KernelTime>,
    pub shapes: Vec<Shape>,
}

/// A kernel's dispatches of the same workgroups in a [`Profile`]: how many,
/// and their GPU time, typically (see [`KernelTime`]).
#[derive(Clone, Debug)]
pub struct Shape {
    pub program: &'static str,
    pub kernel: &'static str,
    pub groups: [u32; 3],
    pub calls: u64,
    pub typical: Duration,
}

/// One kernel of one program in a [`Profile`]. Where the GPU is shared
/// with other work (other processes', the window's), a dispatch may wait
/// for it: the typical times leave that out (each of the kernel's
/// dispatches of the same workgroups taken as long as their median).
#[derive(Clone, Debug)]
pub struct KernelTime {
    /// The program's label (`jodel-gpu asr`, ...).
    pub program: &'static str,
    /// The kernel's entry point.
    pub kernel: &'static str,
    pub calls: u64,
    /// Its dispatches' GPU time, all of it, and typically.
    pub gpu: Duration,
    pub typical: Duration,
}

impl Profile {
    /// The GPU time of every kernel together, typically (see
    /// [`KernelTime`]).
    pub fn total(&self) -> Duration {
        self.kernels.iter().map(|k| k.typical).sum()
    }

    /// The GPU time of every kernel together, all of it.
    pub fn all(&self) -> Duration {
        self.kernels.iter().map(|k| k.gpu).sum()
    }

    /// The dispatches of every kernel together.
    pub fn calls(&self) -> u64 {
        self.kernels.iter().map(|k| k.calls).sum()
    }
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = self.total().as_secs_f64().max(1e-12);
        writeln!(
            f,
            "GPU {:.1} ms typically ({:.1} ms in all) in {} dispatches",
            total * 1e3,
            self.all().as_secs_f64() * 1e3,
            self.calls(),
        )?;
        for k in &self.kernels {
            let ms = k.typical.as_secs_f64() * 1e3;
            let each = |d: Duration| d.as_secs_f64() * 1e6 / k.calls.max(1) as f64;
            writeln!(
                f,
                "  {:>9.2} ms {:>5.1} %  {:>7} × {:>8.1} µs (in all {:.1} µs)  {} {}",
                ms,
                100.0 * ms / 1e3 / total,
                k.calls,
                each(k.typical),
                each(k.gpu),
                k.program,
                k.kernel
            )?;
        }
        Ok(())
    }
}

/// The stamps of one submission: where each dispatch's pass starts (or the
/// one pass of all of them) and where an empty pass after them starts,
/// resolved into a buffer and read back once the submission is done.
pub(crate) struct Stamps {
    program: &'static str,
    kernels: Vec<(&'static str, [u32; 3])>,
    set: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    read: wgpu::Buffer,
}

impl Stamps {
    /// Stamps for `kernels`' dispatches (each its kernel and workgroups)
    /// in that order, where profiling is on and the device stamps (else
    /// `None`).
    pub(crate) fn new(
        device: &wgpu::Device,
        program: &'static str,
        kernels: impl Iterator<Item = (&'static str, [u32; 3])>,
    ) -> Option<Stamps> {
        let detail = detail()?;
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let kernels: Vec<(&'static str, [u32; 3])> = match detail {
            Detail::Kernels => kernels.collect(),
            Detail::Submissions => vec![WHOLE],
        };
        let count = kernels.len() as u32 + 1;
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(program),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let size = u64::from(count) * 8;
        let resolved = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(program),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(program),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Stamps {
            program,
            kernels,
            set,
            resolved,
            read,
        })
    }

    /// Whether the submission is stamped as a whole (one pass).
    pub(crate) fn whole(&self) -> bool {
        self.kernels == [WHOLE]
    }

    /// The stamp of pass `i`'s start (0: the whole's).
    pub(crate) fn writes(&self, i: usize) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(i as u32),
            end_of_pass_write_index: None,
        }
    }

    /// After the passes: the empty pass's stamp, then the stamps resolved
    /// and copied to where they are read.
    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.kernels.len() as u32 + 1;
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.program),
            timestamp_writes: Some(self.writes(self.kernels.len())),
        }));
        encoder.resolve_query_set(&self.set, 0..count, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.read, 0, u64::from(count) * 8);
    }

    /// Once the submission is done: each dispatch's time added to its
    /// kernel's.
    pub(crate) fn add(self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), String> {
        let slice = self.read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let ticks: Vec<u64> = slice
            .get_mapped_range()
            .map_err(|e| e.to_string())?
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
            .collect();
        let period = f64::from(queue.get_timestamp_period());
        let mut all = times();
        for (&(kernel, groups), pair) in self.kernels.iter().zip(ticks.windows(2)) {
            let took = pair[1].saturating_sub(pair[0]) as f64 * period;
            let key = (self.program, kernel, groups);
            match all.iter_mut().find(|(k, _)| *k == key) {
                Some((_, samples)) => samples.push(took),
                None => all.push((key, vec![took])),
            }
        }
        Ok(())
    }
}
