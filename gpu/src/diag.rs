//! What the GPU side finds on this machine, for a diagnostic report
//! ([`report`]), and where a separator's chunk spends its time
//! ([`Profile`]).

use std::time::{Duration, Instant};

use jodel_rhythm::stems::{Separator, DIM, FRAME, HIDDEN, STEMS};

use crate::engine::{bytes, guarded, Kernels};
use crate::separator::{self, Buf, Inner, Kernel, KV, MAX_FRAMES};
use crate::{device, net, setting_up, GpuSeparator};

/// What the GPU side finds on this machine, as lines for a diagnostic
/// report: every adapter `wgpu` sees on every backend with the limits the
/// separator needs, the one it would use (or why none), the card's memory
/// budget and what is used of it ([`crate::memory`]), whether it sets
/// up (or the error), how long a 6 s chunk of silence takes on it, and
/// how that time splits (its profile: each kernel on its own, the matrix
/// multiplies by layer, the upload and the readback, whole chunks in a
/// row); then the beat network's. Sets up a separator of its own (not
/// [`GpuSeparator::shared`]'s). Lists the adapters again, on demand only.
pub fn report() -> Vec<String> {
    let _turn = setting_up();
    let mut lines = device::report();
    lines.extend(crate::memory::report());
    lines.extend(separator_report());
    lines.extend(net::report());
    lines
}

/// The separator's lines of [`report`].
fn separator_report() -> Vec<String> {
    let started = Instant::now();
    let separator = match std::panic::catch_unwind(GpuSeparator::new) {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return vec![format!("separator set up: failed: {e}")],
        Err(_) => return vec!["separator set up: panicked".into()],
    };
    let mut lines = vec![format!(
        "separator set up: yes, in {} ms",
        started.elapsed().as_millis()
    )];
    let stft = vec![0.0f32; MAX_FRAMES * FRAME];
    let mut times = Vec::new();
    for _ in 0..2 {
        let started = Instant::now();
        match separator.masks(MAX_FRAMES, &stft) {
            Ok(_) => times.push(started.elapsed().as_millis()),
            Err(e) => {
                lines.push(format!("a 6 s chunk: failed: {e}"));
                return lines;
            }
        }
    }
    lines.push(format!(
        "a 6 s chunk of silence: {} ms, then {} ms",
        times[0], times[1]
    ));
    match separator.profile(MAX_FRAMES, &stft) {
        Ok(profile) => lines.extend(profile.lines()),
        Err(e) => lines.push(format!("profile: failed: {e}")),
    }
    lines
}

/// How long the parts of one chunk take on the GPU
/// ([`GpuSeparator::profile`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Profile {
    /// Frames in the chunk.
    frames: usize,
    /// Sending the chunk's STFT up (and waiting for it).
    upload: Duration,
    /// Copying the masks back and reading them.
    readback: Duration,
    /// Whole chunks one after the other, as the separator runs them.
    chunks: Vec<Duration>,
    /// Each kind of dispatch on its own: (name, dispatches, workgroups,
    /// time, the best of three).
    parts: Vec<(String, usize, u64, Duration)>,
}

impl Profile {
    /// The profile as lines for a report.
    pub(crate) fn lines(&self) -> Vec<String> {
        let ms = |d: Duration| format!("{:.1} ms", d.as_secs_f64() * 1e3);
        let mut lines = vec![
            format!(
                "profile of one chunk ({} frames); each part on its own, the best of three:",
                self.frames
            ),
            format!(
                "  upload {}, readback {}",
                ms(self.upload),
                ms(self.readback)
            ),
            format!(
                "  whole chunks in a row: {}",
                self.chunks
                    .iter()
                    .map(|&d| ms(d))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ];
        let width = self.parts.iter().map(|p| p.0.len()).max().unwrap_or(0);
        let mut kernels = Duration::ZERO;
        for (name, dispatches, groups, time) in &self.parts {
            if !name.starts_with("  ") {
                kernels += *time;
            }
            lines.push(format!(
                "  {name:width$}  {dispatches:5} dispatches {groups:9} workgroups  {}",
                ms(*time)
            ));
        }
        lines.push(format!("  the kernels' times added up: {}", ms(kernels)));
        lines
    }
}

impl GpuSeparator {
    /// Times the parts of a chunk of `frames` frames of `stft` (see
    /// [`Profile`]): for finding what is slow on a GPU.
    pub(crate) fn profile(&self, frames: usize, stft: &[f32]) -> Result<Profile, String> {
        separator::check(frames, stft)?;
        self.run(|inner| {
            let device = inner.model.program.device.clone();
            guarded(&device, || profile(inner, frames, stft))
        })
    }
}

/// What a matrix multiply of the plan computes, by its shape.
fn matmul_part(params: &[usize; 12]) -> &'static str {
    let (n, k) = (params[1], params[2]);
    match (n, k) {
        (DIM, DIM) => "  matmul q, o (attention)",
        (KV, DIM) => "  matmul k, v (attention)",
        (HIDDEN, DIM) => "  matmul gate, up (feed-forward)",
        (DIM, HIDDEN) => "  matmul down (feed-forward)",
        (DIM, _) => "  matmul band split",
        _ => "  matmul masks",
    }
}

/// The workgroups of the dispatches `picked` of `ops`.
fn workgroups(ops: &[separator::Op], picked: &[usize]) -> u64 {
    picked
        .iter()
        .map(|&i| ops[i].groups.iter().map(|&g| u64::from(g)).product::<u64>())
        .sum()
}

fn profile(inner: &mut Inner, frames: usize, stft: &[f32]) -> Result<Profile, String> {
    let mut profile = Profile {
        frames,
        ..Profile::default()
    };
    // Whole chunks first, on a GPU that has not run much yet.
    for _ in 0..5 {
        let started = Instant::now();
        inner.run(frames, stft)?;
        profile.chunks.push(started.elapsed());
    }
    let model = &inner.model;
    let program = &model.program;
    let started = Instant::now();
    program
        .queue
        .write_buffer(&model.buffers[&Buf::Stft], 0, &bytes(stft));
    program.queue.submit([]);
    program.wait()?;
    profile.upload = started.elapsed();
    let size = STEMS * frames * FRAME;
    let started = Instant::now();
    let mut encoder = program
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodel-gpu profile"),
        });
    encoder.copy_buffer_to_buffer(
        &model.buffers[&Buf::Masks],
        0,
        &model.staging,
        0,
        (size * 4) as u64,
    );
    program.queue.submit([encoder.finish()]);
    let read: f64 = device::read_back(&program.device, &model.staging, size)?
        .iter()
        .map(|&v| f64::from(v))
        .sum();
    std::hint::black_box(read);
    profile.readback = started.elapsed();
    // Each kernel on its own, the matrix multiplies also by layer.
    let ops = inner.ops(frames);
    for &kernel in Kernel::ALL {
        let picked: Vec<usize> = (0..ops.len())
            .filter(|&i| ops[i].kernel == kernel)
            .collect();
        if picked.is_empty() {
            continue;
        }
        let time = model.time(frames, &|i| ops[i].kernel == kernel)?;
        profile.parts.push((
            kernel.entry().to_owned(),
            picked.len(),
            workgroups(&ops, &picked),
            time,
        ));
        if kernel == Kernel::Matmul {
            let mut names: Vec<&'static str> = picked
                .iter()
                .map(|&i| matmul_part(&ops[i].params))
                .collect();
            names.sort_unstable();
            names.dedup();
            for name in names {
                let mine: Vec<usize> = picked
                    .iter()
                    .copied()
                    .filter(|&i| matmul_part(&ops[i].params) == name)
                    .collect();
                let time = model.time(frames, &|i| {
                    ops[i].kernel == Kernel::Matmul && matmul_part(&ops[i].params) == name
                })?;
                profile
                    .parts
                    .push((name.to_owned(), mine.len(), workgroups(&ops, &mine), time));
            }
        }
    }
    Ok(profile)
}
