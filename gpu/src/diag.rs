//! What the GPU side finds on this machine, for a diagnostic report
//! ([`report`]), and where a separator's chunk spends its time
//! ([`Profile`]).

use std::time::{Duration, Instant};

use jodel_rhythm::stems::{Separator, FRAME, STEMS};

use crate::engine::{bytes, guarded};
use crate::separator::{self, Buf, Net, MAX_FRAMES};
use crate::{device, net, profile, setting_up, GpuSeparator};

/// What the GPU side finds on this machine, as lines for a diagnostic
/// report: every adapter `wgpu` sees with the limits the separator needs, the one it would use (or why none), the card's memory
/// budget and what is used of it ([`crate::memory`]), whether it sets
/// up (or the error), how long a 6 s chunk of silence takes on it, and
/// how that time splits (its profile: the upload and the readback, whole
/// chunks in a row, and each kernel's GPU time, by its workgroups too:
/// [`crate::profile`]); then the beat network's. Sets up a separator of
/// its own (not [`GpuSeparator::shared`]'s), its program labelled
/// [`DIAG`]. Lists the adapters again, on demand only. For a process of
/// its own (the player's `--diag`): while it profiles, every network's
/// submissions in the process are stamped and waited for at once.
pub fn report() -> Vec<String> {
    let _turn = setting_up();
    let mut lines = device::report();
    lines.extend(crate::memory::report());
    lines.extend(separator_report());
    lines.extend(net::report());
    lines
}

/// The label of the report's separator's program, its kernels told apart
/// in the profile.
const DIAG: &str = "jodel-gpu diag";

/// The separator's lines of [`report`].
fn separator_report() -> Vec<String> {
    let started = Instant::now();
    let separator = match std::panic::catch_unwind(|| GpuSeparator::labelled(DIAG)) {
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
    /// A chunk's kernels on the GPU ([`crate::profile`]), the report's
    /// separator's only.
    kernels: profile::Profile,
}

/// The kernels' dispatches of the same workgroups listed, the longest
/// first.
const SHAPES: usize = 12;

impl Profile {
    /// The profile as lines for a report.
    pub(crate) fn lines(&self) -> Vec<String> {
        let ms = |d: Duration| format!("{:.1} ms", d.as_secs_f64() * 1e3);
        let mut lines = vec![
            format!("profile of one chunk ({} frames):", self.frames),
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
        lines.extend(self.kernels.to_string().lines().map(|l| format!("  {l}")));
        lines.push("  the longest by workgroups:".into());
        for s in self.kernels.shapes.iter().take(SHAPES) {
            lines.push(format!(
                "  {:>9}  {:>5} dispatches  {} {:?}",
                ms(s.typical),
                s.calls,
                s.kernel,
                s.groups
            ));
        }
        lines
    }
}

impl GpuSeparator {
    /// Times the parts of a chunk of `frames` frames of `stft` (see
    /// [`Profile`]): for finding what is slow on a GPU.
    pub(crate) fn profile(&self, frames: usize, stft: &[f32]) -> Result<Profile, String> {
        separator::check(frames, stft)?;
        self.run(|net| profile(net, frames, stft))
    }
}

fn profile(net: &mut Net, frames: usize, stft: &[f32]) -> Result<Profile, String> {
    let mut chunks = Vec::new();
    // Whole chunks first, on a GPU that has not run much yet.
    for _ in 0..5 {
        let started = Instant::now();
        net.run(frames, stft)?;
        chunks.push(started.elapsed());
    }
    let model = &net.model;
    let program = &model.program;
    let (upload, readback) = guarded(&program.device, || {
        let started = Instant::now();
        program
            .queue
            .write_buffer(&model.buffers[&Buf::Stft], 0, &bytes(stft));
        program.queue.submit([]);
        program.wait()?;
        let upload = started.elapsed();
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
        Ok((upload, started.elapsed()))
    })?;
    // Each kernel's dispatches, each in a pass of its own.
    profile::start(profile::Detail::Kernels);
    let ran = net.run(frames, stft);
    let mut kernels = profile::take();
    ran?;
    kernels.kernels.retain(|k| k.program == DIAG);
    kernels.shapes.retain(|s| s.program == DIAG);
    Ok(Profile {
        frames,
        upload,
        readback,
        chunks,
        kernels,
    })
}
