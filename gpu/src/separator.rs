//! The stem separator of [`jodel_rhythm::stems`] on the GPU
//! ([`GpuSeparator`]): the CPU separator's network, its weights packed as
//! the CPU model holds them ([`CpuSeparator::visit`]), giving the same
//! masks to within rounding. A chunk runs entirely on the GPU: its STFT
//! goes up, the band split, the 16 transformers and the mask layers run as
//! one list of dispatches of five kernels (`kernels.wgsl`: a tiled matrix
//! multiply, which also takes the feed-forward's gated GELU, RMSNorm, the
//! rotary positions, attention with an online softmax and a transpose),
//! and the masks come back.
//!
//! The process keeps two ([`GpuSeparator::shared`] and
//! [`GpuSeparator::background`]), each set up when first needed and again
//! when its device is lost (`engine::Kept`).

use std::sync::OnceLock;

use jodel_rhythm::nn::layers::HEAD;
use jodel_rhythm::stems::{
    self, CpuSeparator, Separator, BAND_BINS, DEPTH, DIM, FRAME, HEADS, HIDDEN, KV_HEADS,
    REGISTERS, STEMS,
};

use crate::engine::{
    self, guarded, kernels_source, params, rotary, spread, Access, After, Angles, Kept, Kernels,
    Model, Mul, OnDevice, Program, SetUp,
};
use crate::pace::{self, Pieces};
use crate::packer::Named;

/// The most frames a chunk may have (6 s at 44.1 kHz, hop 512).
pub(crate) const MAX_FRAMES: usize = stems::CHUNK / 512 + 1;
/// The keys' width, and the values'.
pub(crate) const KV: usize = KV_HEADS * HEAD;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Matmul,
    Norm,
    Rope,
    Attention,
    Transpose,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Matmul,
        Kernel::Norm,
        Kernel::Rope,
        Kernel::Attention,
        Kernel::Transpose,
    ];
    const PARAM_BYTES: u64 = 48;
    const EXTRA: &'static [Access] = &[];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Norm => "rmsnorm",
            Kernel::Rope => "rope",
            Kernel::Attention => "attention",
            Kernel::Transpose => "transpose",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Buf {
    Weights,
    Table,
    Stft,
    Band,
    Grid,
    ByBand,
    Normed,
    Queries,
    KeysValues,
    Attended,
    Gate,
    Final,
    Masks,
}

/// The weights as one buffer's contents, by name, as the CPU separator
/// holds them; but each feed-forward's gate and up (`NAME.gate.w`,
/// `NAME.up.w`, each `[DIM][HIDDEN]`) as one matrix of their columns in
/// pairs (`NAME.gate_up.w`, `[DIM][2 · HIDDEN]`: gate `j` at `2j`, its up
/// next to it), for the matrix multiply that takes their gated GELU.
fn pack() -> Result<Named, String> {
    let cpu = CpuSeparator::built_in()?;
    let mut gates: Vec<(String, Vec<f32>)> = Vec::new();
    let mut named = Named::default();
    cpu.visit(&mut |name, values| {
        if let Some(layer) = name.strip_suffix(".gate.w") {
            gates.push((layer.to_owned(), values.to_vec()));
        } else if let Some(layer) = name.strip_suffix(".up.w") {
            let at = gates.iter().position(|(l, _)| l == layer);
            let (_, gate) = gates.swap_remove(at.expect("a gate before its up"));
            let paired: Vec<f32> = gate
                .chunks(HIDDEN)
                .zip(values.chunks(HIDDEN))
                .flat_map(|(g, u)| g.iter().zip(u).flat_map(|(&g, &u)| [g, u]))
                .collect();
            named.put(&format!("{layer}.gate_up.w"), &paired);
        } else {
            named.put(name, values);
        }
    });
    if gates.is_empty() {
        Ok(named)
    } else {
        Err("a feed-forward gate without its up".into())
    }
}

/// One dispatch.
pub(crate) type Op = engine::Op<Kernel, [Buf; 3], 12>;

/// The dispatches of a chunk of `frames` frames, the weights where `w`
/// says.
pub(crate) fn plan(w: &Named, frames: usize) -> Vec<Op> {
    let bands = BAND_BINS.len();
    let (rows, cols) = (frames + REGISTERS, bands + REGISTERS);
    let tokens = rows * cols;
    let mut ops = Vec::new();
    let matmul = |ops: &mut Vec<Op>, mul: Mul<Buf>| {
        ops.push(mul.tiled(Kernel::Matmul, Buf::Weights));
    };
    let norm = |ops: &mut Vec<Op>,
                (x, x_off, ldx): (Buf, usize, usize),
                (y, ldy): (Buf, usize),
                rows: usize,
                dim: usize,
                weight: usize| {
        ops.push(Op {
            kernel: Kernel::Norm,
            buffers: [x, Buf::Weights, y],
            params: params(&[rows, dim, x_off, ldx, weight, 0, ldy]),
            // 32 invocations a row.
            groups: spread(32 * rows),
        });
    };
    let transpose = |ops: &mut Vec<Op>, from: Buf, to: Buf, rows: usize, cols: usize| {
        ops.push(Op {
            kernel: Kernel::Transpose,
            buffers: [from, Buf::Weights, to],
            params: params(&[rows, cols]),
            groups: spread(rows * cols * DIM),
        });
    };
    // A transformer layer (`name`) on `x`: sequences of `len` tokens, the
    // first `turned` positions turned.
    let block = |ops: &mut Vec<Op>, x: Buf, name: &str, len: usize, turned: usize| {
        let at = |what: &str| w.at(&format!("{name}.{what}"));
        let dense = |from, to, (n, k), what: &str, after| {
            Mul::dense(from, to, (tokens, n, k), at(what), None, after)
        };
        norm(
            ops,
            (x, 0, DIM),
            (Buf::Normed, DIM),
            tokens,
            DIM,
            at("attention_norm"),
        );
        matmul(
            ops,
            dense(Buf::Normed, Buf::Queries, (DIM, DIM), "q.w", After::Store),
        );
        // The keys and the values side by side, a token's in a row of DIM.
        for (what, column) in [("k.w", 0), ("v.w", KV)] {
            matmul(
                ops,
                Mul {
                    to: (Buf::KeysValues, column, DIM),
                    ..dense(Buf::Normed, Buf::KeysValues, (KV, DIM), what, After::Store)
                },
            );
        }
        for (buf, heads) in [(Buf::Queries, HEADS), (Buf::KeysValues, KV_HEADS)] {
            ops.push(Op {
                kernel: Kernel::Rope,
                buffers: [Buf::Table, Buf::Weights, buf],
                params: params(&[tokens, len, turned, heads, DIM]),
                groups: spread(tokens * heads * 16),
            });
        }
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Queries, Buf::KeysValues, Buf::Attended],
            params: params(&[len]),
            groups: [len.div_ceil(64) as u32, HEADS as u32, (tokens / len) as u32],
        });
        matmul(ops, dense(Buf::Attended, x, (DIM, DIM), "o.w", After::Add));
        norm(
            ops,
            (x, 0, DIM),
            (Buf::Normed, DIM),
            tokens,
            DIM,
            at("mlp_norm"),
        );
        matmul(
            ops,
            Mul {
                to: (Buf::Gate, 0, HIDDEN),
                ..dense(
                    Buf::Normed,
                    Buf::Gate,
                    (2 * HIDDEN, DIM),
                    "gate_up.w",
                    After::GeGlu,
                )
            },
        );
        matmul(
            ops,
            dense(Buf::Gate, x, (DIM, HIDDEN), "down.w", After::Add),
        );
    };
    // The band split, into the grid (cleared, registers copied in before).
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        let width = 4 * bins;
        norm(
            &mut ops,
            (Buf::Stft, 4 * start, FRAME),
            (Buf::Band, width),
            frames,
            width,
            w.at(&format!("split.{b}.norm")),
        );
        matmul(
            &mut ops,
            Mul {
                to: (Buf::Grid, b * DIM, cols * DIM),
                ..Mul::dense(
                    Buf::Band,
                    Buf::Grid,
                    (frames, DIM, width),
                    w.at(&format!("split.{b}.w")),
                    Some(w.at(&format!("split.{b}.b"))),
                    After::Store,
                )
            },
        );
        start += bins;
    }
    for l in 0..DEPTH {
        transpose(&mut ops, Buf::Grid, Buf::ByBand, rows, cols);
        block(&mut ops, Buf::ByBand, &format!("{l}.0"), rows, frames);
        transpose(&mut ops, Buf::ByBand, Buf::Grid, cols, rows);
        block(&mut ops, Buf::Grid, &format!("{l}.1"), cols, bands);
    }
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        norm(
            &mut ops,
            (Buf::Grid, b * DIM, cols * DIM),
            (Buf::Final, DIM),
            frames,
            DIM,
            w.at("final_norm"),
        );
        for s in 0..STEMS {
            matmul(
                &mut ops,
                Mul {
                    to: (Buf::Masks, s * frames * FRAME + 4 * start, FRAME),
                    ..Mul::dense(
                        Buf::Final,
                        Buf::Masks,
                        (frames, 4 * bins, DIM),
                        w.at(&format!("mask.{s}.{b}.w")),
                        Some(w.at(&format!("mask.{s}.{b}.b"))),
                        After::Store,
                    )
                },
            );
        }
        start += bins;
    }
    ops
}

/// Why `frames` frames of `stft` are not a chunk, if they are not.
pub(crate) fn check(frames: usize, stft: &[f32]) -> Result<(), String> {
    if frames == 0 || frames > MAX_FRAMES || stft.len() != frames * FRAME {
        return Err(format!("{frames} frames, {} values", stft.len()));
    }
    Ok(())
}

/// The separator on its device, and where its weights are there.
pub(crate) struct Inner {
    pub(crate) model: Model<Kernel, Buf>,
    weights: Named,
}

impl OnDevice for Inner {
    fn lost(&self) -> bool {
        self.model.program.lost()
    }
}

/// The separator network on the GPU.
pub struct GpuSeparator {
    kept: Kept<Inner>,
    /// What its runs cost, read without waiting for one (the same count
    /// across setups).
    counts: pace::Counts,
}

/// What sets the separator up on the best GPU, its runs paced as `pieces`
/// say.
fn set_up(pieces: Pieces) -> SetUp<Inner> {
    Box::new(move || {
        crate::device::open("jodel-gpu", |device, queue, limits| {
            Inner::new(device, queue, limits, pieces.clone())
        })
    })
}

impl GpuSeparator {
    /// The separator on the best GPU, set up on first use (and again once
    /// its device is lost); `None` without a usable one (a software
    /// renderer does not count), or while setting it up fails (it is tried
    /// again a while later).
    pub fn shared() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<GpuSeparator> = OnceLock::new();
        let gpu = GPU.get_or_init(|| GpuSeparator::kept(Pieces::foreground()));
        gpu.kept.usable().then_some(gpu)
    }

    /// A separator of its own for background work (analyses, getting
    /// lyrics timed), so that it never waits for, nor holds up, the stems
    /// of the track playing ([`GpuSeparator::shared`]'s); that one where a
    /// second does not fit on the GPU.
    pub fn background() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<GpuSeparator> = OnceLock::new();
        let gpu = GPU.get_or_init(|| GpuSeparator::kept(Pieces::resting()));
        gpu.kept
            .usable()
            .then_some(gpu)
            .or_else(GpuSeparator::shared)
    }

    /// The separator for the calling thread's work: the background one
    /// ([`GpuSeparator::background`]) on a thread marked as doing
    /// background work ([`pace::background`]), else the shared one.
    pub fn for_this_thread() -> Option<&'static GpuSeparator> {
        if pace::is_background() {
            GpuSeparator::background()
        } else {
            GpuSeparator::shared()
        }
    }

    /// What its GPU work cost so far ([`pace::Tally::since`] an earlier
    /// tally says what was done between), without waiting for its work.
    pub fn tally(&self) -> pace::Tally {
        self.counts.tally()
    }

    /// Sets up a separator of its own on the best GPU, its work not
    /// resting (as [`GpuSeparator::shared`]'s).
    pub fn new() -> Result<GpuSeparator, String> {
        let pieces = Pieces::foreground();
        Ok(GpuSeparator {
            counts: pieces.counts(),
            kept: Kept::set_up_now(set_up(pieces))?,
        })
    }

    /// A separator whose runs are paced as `pieces` say, set up when first
    /// used.
    fn kept(pieces: Pieces) -> GpuSeparator {
        GpuSeparator {
            counts: pieces.counts(),
            kept: Kept::new(set_up(pieces)),
        }
    }

    /// `work` on the separator on its device ([`Kept::run`]).
    pub(crate) fn run<R>(
        &self,
        work: impl FnMut(&mut Inner) -> Result<R, String>,
    ) -> Result<R, String> {
        self.kept.run(work)
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        pieces: Pieces,
    ) -> Result<Inner, String> {
        let mut weights = pack()?;
        let values = weights.take_values();
        let tokens = (MAX_FRAMES + REGISTERS) * (BAND_BINS.len() + REGISTERS);
        let widest = BAND_BINS.iter().copied().max().unwrap_or(0);
        let sizes = [
            (Buf::Weights, values.len()),
            (Buf::Table, (MAX_FRAMES + REGISTERS) * 32),
            (Buf::Stft, MAX_FRAMES * FRAME),
            (Buf::Band, MAX_FRAMES * 4 * widest),
            (Buf::Grid, tokens * DIM),
            (Buf::ByBand, tokens * DIM),
            (Buf::Normed, tokens * DIM),
            (Buf::Queries, tokens * DIM),
            (Buf::KeysValues, tokens * DIM),
            (Buf::Attended, tokens * DIM),
            (Buf::Gate, tokens * HIDDEN),
            (Buf::Final, MAX_FRAMES * DIM),
            (Buf::Masks, STEMS * MAX_FRAMES * FRAME),
        ];
        let program = Program::new(device, queue, limits, "jodel-gpu", &kernels_source(&[]));
        let model = Model::new(program, &sizes, STEMS * MAX_FRAMES * FRAME, pieces)?;
        model.write(Buf::Weights, &values);
        model.write(
            Buf::Table,
            &rotary(0..MAX_FRAMES + REGISTERS, engine::HEAD_32, Angles::F32),
        );
        Ok(Inner { model, weights })
    }

    /// The masks of a chunk of `frames` frames of `stft`.
    fn masks(&mut self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        check(frames, stft)?;
        let device = self.model.program.device.clone();
        guarded(&device, || self.run(frames, stft))
    }

    /// The dispatches of a chunk of `frames` frames (as its plan records
    /// them).
    pub(crate) fn ops(&self, frames: usize) -> Vec<Op> {
        plan(&self.weights, frames)
    }

    /// One chunk through its plan (recorded on first use), in short
    /// submissions ([`pace::dispatch`]): the masks.
    pub(crate) fn run(&mut self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.model.write(Buf::Stft, stft);
        let rows = frames + REGISTERS;
        let cols = BAND_BINS.len() + REGISTERS;
        let (weights, registers) = (&self.weights, self.weights.at("registers"));
        self.model.run(
            frames,
            || plan(weights, frames),
            |encoder, buffers| {
                let grid = &buffers[&Buf::Grid];
                encoder.clear_buffer(grid, 0, Some((rows * cols * DIM * 4) as u64));
                for i in 0..REGISTERS {
                    let from = (registers + i * REGISTERS * DIM) * 4;
                    let to = ((frames + i) * cols + BAND_BINS.len()) * DIM * 4;
                    encoder.copy_buffer_to_buffer(
                        &buffers[&Buf::Weights],
                        from as u64,
                        grid,
                        to as u64,
                        (REGISTERS * DIM * 4) as u64,
                    );
                }
            },
            (Buf::Masks, STEMS * frames * FRAME),
        )
    }
}

impl Separator for GpuSeparator {
    fn name(&self) -> String {
        self.kept.name()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.run(|inner| inner.masks(frames, stft))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separator whose device is lost sets up again and separates as
    /// before; skipped (with a note) where there is no usable GPU.
    #[test]
    fn a_separator_whose_device_is_lost_sets_up_again() {
        let Ok(gpu) = GpuSeparator::new() else {
            eprintln!("no usable GPU: skipped");
            return;
        };
        let frames = 40;
        let stft: Vec<f32> = (0..frames * FRAME)
            .map(|i| ((i % 97) as f32 * 0.37).sin())
            .collect();
        let before = gpu.masks(frames, &stft).unwrap();
        gpu.run(|inner| {
            inner.model.program.device.destroy();
            Ok(())
        })
        .unwrap();
        assert_eq!(gpu.masks(frames, &stft).unwrap(), before);
        assert!(gpu.tally().submissions >= 2);
    }
}
