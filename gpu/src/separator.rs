//! The stem separator of [`jodel_rhythm::stems`] on the GPU
//! ([`GpuSeparator`]): the same network as the built-in CPU separator, on
//! the same weights, giving the same masks to within rounding. A chunk
//! runs entirely on the GPU: its STFT goes up, the band split, the 16
//! transformers and the mask layers run as one list of dispatches of six
//! kernels (`kernels.wgsl`: a tiled matrix multiply, RMSNorm, the rotary
//! positions, attention with an online softmax, the gated GELU and a
//! transpose), and the masks come back.

use std::sync::{Mutex, OnceLock};

use jodel_rhythm::stems::{self, Separator, BAND_BINS, DIM, STEMS};

use crate::engine::{
    self, guarded, kernels_source, params, rotary_table, spread, Access, Kernels, Model, Program,
};
use crate::pace::{self, Pieces};
use crate::{packer, Turned};

/// The most frames a chunk may have (6 s at 44.1 kHz, hop 512).
pub(crate) const MAX_FRAMES: usize = stems::CHUNK / 512 + 1;
/// Values per STFT frame: bins × channels × (real, imaginary).
pub(crate) const FRAME: usize = 1025 * 4;
const REGISTERS: usize = 4;
pub(crate) const HIDDEN: usize = 768;
const DEPTH: usize = 8;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Matmul,
    Norm,
    Rope,
    Attention,
    Geglu,
    Transpose,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Matmul,
        Kernel::Norm,
        Kernel::Rope,
        Kernel::Attention,
        Kernel::Geglu,
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
            Kernel::Geglu => "geglu",
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
    Up,
    Final,
    Masks,
}

/// One weight layer's place in the weights buffer.
#[derive(Clone, Copy, Debug, Default)]
struct Layer {
    weight: u32,
    bias: u32,
}

/// One transformer layer's weights.
#[derive(Clone, Copy, Debug, Default)]
struct BlockAt {
    attention_norm: u32,
    q: u32,
    k: u32,
    v: u32,
    o: u32,
    mlp_norm: u32,
    gate: u32,
    up: u32,
    down: u32,
}

/// Where everything is in the weights buffer (in f32s).
pub(crate) struct Offsets {
    split_norm: Vec<u32>,
    split: Vec<Layer>,
    blocks: Vec<[BlockAt; 2]>,
    registers: u32,
    final_norm: u32,
    masks: Vec<Vec<Layer>>,
}

/// The weights as one buffer's contents, the layers transposed to
/// [input][output].
fn pack_weights() -> Result<(Vec<f32>, Offsets), String> {
    let mut w = stems::weights();
    let mut data = packer::Packer::default();
    let mut put = |values: Vec<f32>| -> u32 { data.push(&values) as u32 };
    let linear = |w: &mut jodel_rhythm::nn::weights::Weights,
                  put: &mut dyn FnMut(Vec<f32>) -> u32,
                  name: &str,
                  inputs: usize,
                  outputs: usize,
                  bias: bool|
     -> Result<Layer, String> {
        let raw = w.take(&format!("{name}.weight"), &[outputs, inputs])?;
        let weight = put(packer::transposed(&raw, outputs, inputs));
        let bias = if bias {
            put(w.take(&format!("{name}.bias"), &[outputs])?)
        } else {
            0
        };
        Ok(Layer { weight, bias })
    };
    let mut split_norm = Vec::new();
    let mut split = Vec::new();
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        let name = format!("model.band_split.to_features.{b}");
        split_norm.push(put(w.take(&format!("{name}.0.weight"), &[4 * bins])?));
        split.push(linear(
            &mut w,
            &mut put,
            &format!("{name}.1"),
            4 * bins,
            DIM,
            true,
        )?);
    }
    let mut blocks = Vec::new();
    for l in 0..DEPTH {
        let mut pair = [BlockAt::default(); 2];
        for (which, at) in pair.iter_mut().enumerate() {
            let name = format!("model.layers.{l}.{which}.layers.0");
            let kv = 128;
            *at = BlockAt {
                attention_norm: put(w.take(&format!("{name}.input_layernorm.weight"), &[DIM])?),
                q: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.q_proj"),
                    DIM,
                    DIM,
                    false,
                )?
                .weight,
                k: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.k_proj"),
                    DIM,
                    kv,
                    false,
                )?
                .weight,
                v: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.v_proj"),
                    DIM,
                    kv,
                    false,
                )?
                .weight,
                o: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.self_attn.o_proj"),
                    DIM,
                    DIM,
                    false,
                )?
                .weight,
                mlp_norm: put(w.take(&format!("{name}.post_attention_layernorm.weight"), &[DIM])?),
                gate: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.gate_proj"),
                    DIM,
                    HIDDEN,
                    false,
                )?
                .weight,
                up: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.up_proj"),
                    DIM,
                    HIDDEN,
                    false,
                )?
                .weight,
                down: linear(
                    &mut w,
                    &mut put,
                    &format!("{name}.mlp.down_proj"),
                    HIDDEN,
                    DIM,
                    false,
                )?
                .weight,
            };
        }
        blocks.push(pair);
    }
    let registers = put(w.take("model.register_tokens", &[REGISTERS, REGISTERS, DIM])?);
    let final_norm = put(w.take("model.final_norm.weight", &[DIM])?);
    let mut masks = Vec::new();
    for s in 0..STEMS {
        let mut bands = Vec::new();
        for (b, &bins) in BAND_BINS.iter().enumerate() {
            let name = format!("mask_estimators.{s}.to_freq_mlps.{b}");
            bands.push(linear(&mut w, &mut put, &name, DIM, 4 * bins, true)?);
        }
        masks.push(bands);
    }
    if !w.left().is_empty() {
        return Err(format!("unused weights: {:?}", w.left()));
    }
    Ok((
        data.data,
        Offsets {
            split_norm,
            split,
            blocks,
            registers,
            final_norm,
            masks,
        },
    ))
}

/// One dispatch.
pub(crate) type Op = engine::Op<Kernel, [Buf; 3], 12>;

/// The dispatches of a chunk of `frames` frames.
pub(crate) fn plan(o: &Offsets, frames: usize) -> Vec<Op> {
    let bands = BAND_BINS.len();
    let (rows, cols) = (frames + REGISTERS, bands + REGISTERS);
    let tokens = rows * cols;
    let mut ops = Vec::new();
    let matmul = |ops: &mut Vec<Op>,
                  a: Buf,
                  c: Buf,
                  m: usize,
                  n: usize,
                  k: usize,
                  a_off: usize,
                  lda: usize,
                  w: u32,
                  c_off: usize,
                  ldc: usize,
                  bias: Option<u32>,
                  add: bool| {
        let flags = usize::from(bias.is_some()) | (usize::from(add) << 1);
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [a, Buf::Weights, c],
            params: params(&[
                m,
                n,
                k,
                a_off,
                lda,
                w as usize,
                c_off,
                ldc,
                bias.unwrap_or(0) as usize,
                flags,
            ]),
            groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
        });
    };
    let norm = |ops: &mut Vec<Op>,
                x: Buf,
                y: Buf,
                rows: usize,
                dim: usize,
                x_off: usize,
                ldx: usize,
                w: u32,
                ldy: usize| {
        ops.push(Op {
            kernel: Kernel::Norm,
            buffers: [x, Buf::Weights, y],
            params: params(&[rows, dim, x_off, ldx, w as usize, 0, ldy]),
            groups: spread(rows),
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
    let block = |ops: &mut Vec<Op>, x: Buf, at: &BlockAt, len: usize, turned: usize| {
        norm(
            ops,
            x,
            Buf::Normed,
            tokens,
            DIM,
            0,
            DIM,
            at.attention_norm,
            DIM,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Queries,
            tokens,
            DIM,
            DIM,
            0,
            DIM,
            at.q,
            0,
            DIM,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::KeysValues,
            tokens,
            128,
            DIM,
            0,
            DIM,
            at.k,
            0,
            DIM,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::KeysValues,
            tokens,
            128,
            DIM,
            0,
            DIM,
            at.v,
            128,
            DIM,
            None,
            false,
        );
        for (buf, heads) in [(Buf::Queries, 8), (Buf::KeysValues, 4)] {
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
            groups: [len.div_ceil(64) as u32, 8, (tokens / len) as u32],
        });
        matmul(
            ops,
            Buf::Attended,
            x,
            tokens,
            DIM,
            DIM,
            0,
            DIM,
            at.o,
            0,
            DIM,
            None,
            true,
        );
        norm(ops, x, Buf::Normed, tokens, DIM, 0, DIM, at.mlp_norm, DIM);
        matmul(
            ops,
            Buf::Normed,
            Buf::Gate,
            tokens,
            HIDDEN,
            DIM,
            0,
            DIM,
            at.gate,
            0,
            HIDDEN,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Up,
            tokens,
            HIDDEN,
            DIM,
            0,
            DIM,
            at.up,
            0,
            HIDDEN,
            None,
            false,
        );
        ops.push(Op {
            kernel: Kernel::Geglu,
            buffers: [Buf::Up, Buf::Weights, Buf::Gate],
            params: params(&[tokens * HIDDEN]),
            groups: spread(tokens * HIDDEN),
        });
        matmul(
            ops,
            Buf::Gate,
            x,
            tokens,
            DIM,
            HIDDEN,
            0,
            HIDDEN,
            at.down,
            0,
            DIM,
            None,
            true,
        );
    };
    // The band split, into the grid (cleared, registers copied in before).
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        let width = 4 * bins;
        norm(
            &mut ops,
            Buf::Stft,
            Buf::Band,
            frames,
            width,
            4 * start,
            FRAME,
            o.split_norm[b],
            width,
        );
        matmul(
            &mut ops,
            Buf::Band,
            Buf::Grid,
            frames,
            DIM,
            width,
            0,
            width,
            o.split[b].weight,
            b * DIM,
            cols * DIM,
            Some(o.split[b].bias),
            false,
        );
        start += bins;
    }
    for pair in &o.blocks {
        transpose(&mut ops, Buf::Grid, Buf::ByBand, rows, cols);
        block(&mut ops, Buf::ByBand, &pair[0], rows, frames);
        transpose(&mut ops, Buf::ByBand, Buf::Grid, cols, rows);
        block(&mut ops, Buf::Grid, &pair[1], cols, bands);
    }
    let mut start = 0;
    for (b, &bins) in BAND_BINS.iter().enumerate() {
        norm(
            &mut ops,
            Buf::Grid,
            Buf::Final,
            frames,
            DIM,
            b * DIM,
            cols * DIM,
            o.final_norm,
            DIM,
        );
        for (s, stem) in o.masks.iter().enumerate() {
            let c_off = s * frames * FRAME + 4 * start;
            matmul(
                &mut ops,
                Buf::Final,
                Buf::Masks,
                frames,
                4 * bins,
                DIM,
                0,
                DIM,
                stem[b].weight,
                c_off,
                FRAME,
                Some(stem[b].bias),
                false,
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
    offsets: Offsets,
}

/// The separator network on the GPU.
pub struct GpuSeparator {
    name: String,
    inner: Turned<Mutex<Inner>>,
    /// What its runs cost, read without waiting for one.
    counts: pace::Counts,
}

/// A separator whose runs are paced as `pieces` say; `None` if setting
/// it up failed in any way.
fn set_up(pieces: Pieces) -> Option<GpuSeparator> {
    std::panic::catch_unwind(|| GpuSeparator::with_pieces(pieces))
        .ok()?
        .ok()
}

impl GpuSeparator {
    /// The separator on the best GPU, set up on first use; `None` without
    /// a usable one (a software renderer does not count), or if setting it
    /// up failed in any way.
    pub fn shared() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<Option<GpuSeparator>> = OnceLock::new();
        GPU.get_or_init(|| set_up(Pieces::foreground())).as_ref()
    }

    /// A separator of its own for background work (analyses, getting
    /// lyrics timed), so that it never waits for, nor holds up, the stems
    /// of the track playing ([`GpuSeparator::shared`]'s); that one where a
    /// second does not fit on the GPU.
    pub fn background() -> Option<&'static GpuSeparator> {
        static GPU: OnceLock<Option<GpuSeparator>> = OnceLock::new();
        GPU.get_or_init(|| set_up(Pieces::resting()))
            .as_ref()
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

    /// Sets up the separator on the best GPU, its work not resting (as
    /// [`GpuSeparator::shared`]'s).
    pub fn new() -> Result<GpuSeparator, String> {
        GpuSeparator::with_pieces(Pieces::foreground())
    }

    /// Sets up the separator on the best GPU, its runs paced as `pieces`
    /// say.
    fn with_pieces(pieces: Pieces) -> Result<GpuSeparator, String> {
        let (inner, name) = crate::device::open("jodel-gpu", |device, queue, limits| {
            Inner::new(device, queue, limits, pieces)
        })?;
        Ok(GpuSeparator {
            name,
            counts: inner.model.pieces.counts(),
            inner: Turned::new(Mutex::new(inner)),
        })
    }

    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, Inner>, String> {
        self.inner
            .lock()
            .map_err(|_| "the GPU separator failed before".to_owned())
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        pieces: Pieces,
    ) -> Result<Inner, String> {
        let (weights, offsets) = pack_weights()?;
        let tokens = (MAX_FRAMES + REGISTERS) * (BAND_BINS.len() + REGISTERS);
        let sizes = [
            (Buf::Weights, weights.len()),
            (Buf::Table, (MAX_FRAMES + REGISTERS) * 32),
            (Buf::Stft, MAX_FRAMES * FRAME),
            (Buf::Band, MAX_FRAMES * 4 * 129),
            (Buf::Grid, tokens * DIM),
            (Buf::ByBand, tokens * DIM),
            (Buf::Normed, tokens * DIM),
            (Buf::Queries, tokens * DIM),
            (Buf::KeysValues, tokens * DIM),
            (Buf::Attended, tokens * DIM),
            (Buf::Gate, tokens * HIDDEN),
            (Buf::Up, tokens * HIDDEN),
            (Buf::Final, MAX_FRAMES * DIM),
            (Buf::Masks, STEMS * MAX_FRAMES * FRAME),
        ];
        let program = Program::new(device, queue, limits, "jodel-gpu", &kernels_source(&[]));
        let model = Model::new(program, &sizes, STEMS * MAX_FRAMES * FRAME, pieces)?;
        model.write(Buf::Weights, &weights);
        model.write(Buf::Table, &rotary_table(MAX_FRAMES + REGISTERS));
        Ok(Inner { model, offsets })
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
        plan(&self.offsets, frames)
    }

    /// One chunk through its plan (recorded on first use), in short
    /// submissions ([`pace::dispatch`]): the masks.
    pub(crate) fn run(&mut self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.model.write(Buf::Stft, stft);
        let rows = frames + REGISTERS;
        let cols = BAND_BINS.len() + REGISTERS;
        let (offsets, registers) = (&self.offsets, self.offsets.registers as usize);
        self.model.run(
            frames,
            || plan(offsets, frames),
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
        self.name.clone()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        self.lock()?.masks(frames, stft)
    }
}
