//! The beat network of [`jodel_rhythm::net`] (Beat This!, small) on the
//! GPU, with the same built-in weights: [`GpuNetwork`] is a
//! [`NetRunner`], so the analysis runs a piece's layers here and only the
//! two logits per frame come back.
//!
//! A piece is one list of dispatches: the spectrogram in, the stem's
//! convolution (as patches through the tiled matrix multiply), the three
//! blocks (attention and feed-forward across the bands of each frame, then
//! across the frames of each band, a convolution halving the bands), the
//! projection, six transformer layers, the final RMSNorm and the head.
//! The matrix multiply, RMSNorm and rotary kernels are the separator's
//! (`kernels.wgsl`, in the beat network's modes); the rest are in
//! `net.wgsl`.

use std::sync::{Mutex, OnceLock};

use jodel_rhythm::net::mel::BANDS;
use jodel_rhythm::net::{built_in_weights, NetRunner, PIECE_FRAMES};
use jodel_rhythm::nn::weights::Weights;

use crate::engine::{
    self, guarded, kernels_source, params, rotary_table, spread, Access, Kernels, Model, Program,
};
use crate::pace::{self, Pieces};
use crate::Turned;

/// The front end's channels after the stem, and the transformer's width.
const STEM: usize = 32;
const DIM: usize = 128;
const LAYERS: usize = 6;
/// A head's width.
const HEAD: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Norm,
    Rope,
    In,
    Patches,
    AffineGelu,
    Gelu,
    Transpose,
    Flatten,
    Attention,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Matmul,
        Kernel::Norm,
        Kernel::Rope,
        Kernel::In,
        Kernel::Patches,
        Kernel::AffineGelu,
        Kernel::Gelu,
        Kernel::Transpose,
        Kernel::Flatten,
        Kernel::Attention,
    ];
    const PARAM_BYTES: u64 = 48;
    const EXTRA: &'static [Access] = &[];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Norm => "rmsnorm",
            Kernel::Rope => "rope",
            Kernel::In => "net_in",
            Kernel::Patches => "net_patches",
            Kernel::AffineGelu => "net_affine_gelu",
            Kernel::Gelu => "net_gelu",
            Kernel::Transpose => "net_transpose",
            Kernel::Flatten => "net_flatten",
            Kernel::Attention => "net_attention",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Weights,
    Table,
    Spect,
    X,
    Y,
    Normed,
    Qkv,
    Gates,
    Attended,
    Hidden,
    Patches,
    Logits,
}

/// An attention's weights (offsets in f32s).
#[derive(Clone, Copy, Debug, Default)]
struct AttnAt {
    norm: u32,
    qkv: u32,
    gates: u32,
    gates_bias: u32,
    out: u32,
}

/// A feed-forward step's weights.
#[derive(Clone, Copy, Debug, Default)]
struct FfAt {
    norm: u32,
    up: u32,
    up_bias: u32,
    down: u32,
    down_bias: u32,
}

/// A front-end block's weights.
#[derive(Clone, Copy, Debug, Default)]
struct BlockAt {
    attn_f: AttnAt,
    ff_f: FfAt,
    attn_t: AttnAt,
    ff_t: FfAt,
    conv: u32,
    scale: u32,
    shift: u32,
}

/// Where everything is in the weights buffer.
#[derive(Clone, Debug, Default)]
struct Offsets {
    in_scale: u32,
    in_shift: u32,
    stem: u32,
    stem_scale: u32,
    stem_shift: u32,
    blocks: Vec<BlockAt>,
    project: u32,
    project_bias: u32,
    layers: Vec<(AttnAt, FfAt)>,
    norm: u32,
    head: u32,
    head_bias: u32,
}

/// Takes the weights out of the file into one buffer's contents.
struct Packer {
    w: Weights,
    data: Vec<f32>,
}

impl Packer {
    fn put(&mut self, values: Vec<f32>) -> u32 {
        let at = self.data.len() as u32;
        self.data.extend(values);
        at
    }

    fn vector(&mut self, name: &str, dim: usize) -> Result<u32, String> {
        let v = self.w.take(name, &[dim])?;
        Ok(self.put(v))
    }

    /// A layer, transposed to [input][output]; and its bias (0 without).
    fn linear(
        &mut self,
        name: &str,
        inputs: usize,
        outputs: usize,
        bias: bool,
    ) -> Result<(u32, u32), String> {
        let raw = self.w.take(&format!("{name}.weight"), &[outputs, inputs])?;
        let mut t = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                t[i * outputs + o] = raw[o * inputs + i];
            }
        }
        let weight = self.put(t);
        let bias = if bias {
            self.vector(&format!("{name}.bias"), outputs)?
        } else {
            0
        };
        Ok((weight, bias))
    }

    /// A convolution of `kf` bands by 3 frames as a layer over its patches:
    /// [out][in][kf][kt] to [(kf, kt, in)][out].
    fn conv(
        &mut self,
        name: &str,
        inputs: usize,
        outputs: usize,
        kf: usize,
    ) -> Result<u32, String> {
        let raw = self
            .w
            .take(&format!("{name}.weight"), &[outputs, inputs, kf, 3])?;
        let mut t = vec![0.0f32; raw.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                for f in 0..kf {
                    for k in 0..3 {
                        t[((f * 3 + k) * inputs + i) * outputs + o] =
                            raw[((o * inputs + i) * kf + f) * 3 + k];
                    }
                }
            }
        }
        Ok(self.put(t))
    }

    /// A per-channel scale and shift.
    fn affine(&mut self, name: &str, dim: usize) -> Result<(u32, u32), String> {
        let scale = self.vector(&format!("{name}.scale"), dim)?;
        let shift = self.vector(&format!("{name}.shift"), dim)?;
        Ok((scale, shift))
    }

    fn attention(&mut self, name: &str, dim: usize) -> Result<AttnAt, String> {
        let heads = dim / HEAD;
        let norm = self.vector(&format!("{name}.norm.gamma"), dim)?;
        let (qkv, _) = self.linear(&format!("{name}.to_qkv"), dim, 3 * dim, false)?;
        let (gates, gates_bias) = self.linear(&format!("{name}.to_gates"), dim, heads, true)?;
        let (out, _) = self.linear(&format!("{name}.to_out.0"), dim, dim, false)?;
        Ok(AttnAt {
            norm,
            qkv,
            gates,
            gates_bias,
            out,
        })
    }

    fn feed_forward(&mut self, name: &str, dim: usize) -> Result<FfAt, String> {
        let norm = self.vector(&format!("{name}.net.0.gamma"), dim)?;
        let (up, up_bias) = self.linear(&format!("{name}.net.1"), dim, 4 * dim, true)?;
        let (down, down_bias) = self.linear(&format!("{name}.net.4"), 4 * dim, dim, true)?;
        Ok(FfAt {
            norm,
            up,
            up_bias,
            down,
            down_bias,
        })
    }
}

/// The weights as one buffer's contents and where each tensor is; every
/// tensor of the file is used.
fn pack_weights() -> Result<(Vec<f32>, Offsets), String> {
    let mut p = Packer {
        w: built_in_weights()?,
        data: Vec::with_capacity(2_000_000),
    };
    let mut o = Offsets::default();
    (o.in_scale, o.in_shift) = p.affine("frontend.stem.bn1d", BANDS)?;
    o.stem = p.conv("frontend.stem.conv2d", 1, STEM, 4)?;
    (o.stem_scale, o.stem_shift) = p.affine("frontend.stem.bn2d", STEM)?;
    let mut c = STEM;
    for b in 0..3 {
        let n = format!("frontend.blocks.{b}");
        let attn_f = p.attention(&format!("{n}.partial.attnF"), c)?;
        let ff_f = p.feed_forward(&format!("{n}.partial.ffF"), c)?;
        let attn_t = p.attention(&format!("{n}.partial.attnT"), c)?;
        let ff_t = p.feed_forward(&format!("{n}.partial.ffT"), c)?;
        let conv = p.conv(&format!("{n}.conv2d"), c, 2 * c, 2)?;
        let (scale, shift) = p.affine(&format!("{n}.norm"), 2 * c)?;
        o.blocks.push(BlockAt {
            attn_f,
            ff_f,
            attn_t,
            ff_t,
            conv,
            scale,
            shift,
        });
        c *= 2;
    }
    let bands_out = BANDS / 4 / 8;
    (o.project, o.project_bias) = p.linear("frontend.linear", c * bands_out, DIM, true)?;
    for l in 0..LAYERS {
        let n = format!("transformer_blocks.layers.{l}");
        let attn = p.attention(&format!("{n}.0"), DIM)?;
        let ff = p.feed_forward(&format!("{n}.1"), DIM)?;
        o.layers.push((attn, ff));
    }
    o.norm = p.vector("transformer_blocks.norm.gamma", DIM)?;
    (o.head, o.head_bias) = p.linear("task_heads.beat_downbeat_lin", DIM, 2, true)?;
    if !p.w.left().is_empty() {
        return Err(format!("unused weights: {:?}", p.w.left()));
    }
    Ok((p.data, o))
}

/// One dispatch.
type Op = engine::Op<Kernel, [Buf; 3], 12>;

/// The dispatches of a piece of `frames` frames.
fn plan(o: &Offsets, frames: usize) -> Vec<Op> {
    let f = frames;
    let mut ops: Vec<Op> = Vec::new();
    let elementwise = |ops: &mut Vec<Op>, kernel, buffers, p: &[usize], count: usize| {
        ops.push(Op {
            kernel,
            buffers,
            params: params(p),
            groups: spread(count),
        });
    };
    // dst (+)= a · weights, see the kernel.
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        ops: &mut Vec<Op>,
        a: Buf,
        c: Buf,
        m: usize,
        n: usize,
        k: usize,
        w: u32,
        bias: Option<u32>,
        add: bool,
    ) {
        let flags = usize::from(bias.is_some()) | (usize::from(add) << 1);
        ops.push(Op {
            kernel: Kernel::Matmul,
            buffers: [a, Buf::Weights, c],
            params: params(&[
                m,
                n,
                k,
                0,
                k,
                w as usize,
                0,
                n,
                bias.unwrap_or(0) as usize,
                flags,
            ]),
            groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
        });
    }
    fn norm(ops: &mut Vec<Op>, x: Buf, y: Buf, rows: usize, dim: usize, w: u32) {
        ops.push(Op {
            kernel: Kernel::Norm,
            buffers: [x, Buf::Weights, y],
            params: params(&[rows, dim, 0, dim, w as usize, 0, dim, 1]),
            groups: spread(rows),
        });
    }
    // Attention over sequences of `len` tokens of `dim`, added to `x`.
    fn attention(ops: &mut Vec<Op>, x: Buf, tokens: usize, len: usize, dim: usize, at: AttnAt) {
        let heads = dim / HEAD;
        norm(ops, x, Buf::Normed, tokens, dim, at.norm);
        matmul(
            ops,
            Buf::Normed,
            Buf::Qkv,
            tokens,
            3 * dim,
            dim,
            at.qkv,
            None,
            false,
        );
        matmul(
            ops,
            Buf::Normed,
            Buf::Gates,
            tokens,
            heads,
            dim,
            at.gates,
            Some(at.gates_bias),
            false,
        );
        for col in [0, dim] {
            ops.push(Op {
                kernel: Kernel::Rope,
                buffers: [Buf::Table, Buf::Weights, Buf::Qkv],
                params: params(&[tokens, len, len, heads, 3 * dim, col, 1]),
                groups: spread(tokens * heads * 16),
            });
        }
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Qkv, Buf::Gates, Buf::Attended],
            params: params(&[len, dim, heads]),
            groups: [len.div_ceil(64) as u32, heads as u32, (tokens / len) as u32],
        });
        matmul(ops, Buf::Attended, x, tokens, dim, dim, at.out, None, true);
    }
    // The feed-forward step, added to `x`.
    fn feed_forward(ops: &mut Vec<Op>, x: Buf, tokens: usize, dim: usize, at: FfAt) {
        norm(ops, x, Buf::Normed, tokens, dim, at.norm);
        matmul(
            ops,
            Buf::Normed,
            Buf::Hidden,
            tokens,
            4 * dim,
            dim,
            at.up,
            Some(at.up_bias),
            false,
        );
        let count = tokens * 4 * dim;
        ops.push(Op {
            kernel: Kernel::Gelu,
            buffers: [Buf::Table, Buf::Weights, Buf::Hidden],
            params: params(&[count]),
            groups: spread(count),
        });
        matmul(
            ops,
            Buf::Hidden,
            x,
            tokens,
            dim,
            4 * dim,
            at.down,
            Some(at.down_bias),
            true,
        );
    }
    // The stem: each band scaled, patches of 4 bands by 3 frames, to 32
    // channels, scaled, GELU.
    elementwise(
        &mut ops,
        Kernel::In,
        [Buf::Spect, Buf::Weights, Buf::X],
        &[f, BANDS, o.in_scale as usize, o.in_shift as usize],
        f * BANDS,
    );
    let mut bands = BANDS / 4;
    elementwise(
        &mut ops,
        Kernel::Patches,
        [Buf::X, Buf::Weights, Buf::Patches],
        &[f, 1, 4, bands],
        bands * f * 12,
    );
    matmul(
        &mut ops,
        Buf::Patches,
        Buf::Y,
        bands * f,
        STEM,
        12,
        o.stem,
        None,
        false,
    );
    elementwise(
        &mut ops,
        Kernel::AffineGelu,
        [Buf::Table, Buf::Weights, Buf::Y],
        &[
            bands * f * STEM,
            STEM,
            o.stem_scale as usize,
            o.stem_shift as usize,
        ],
        bands * f * STEM,
    );
    // The blocks, the activations in `cur` as [band][frame][channel].
    let (mut cur, mut other) = (Buf::Y, Buf::X);
    let mut c = STEM;
    for block in &o.blocks {
        let tokens = bands * f;
        // Across the bands of each frame: [frame][band][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [cur, Buf::Weights, other],
            &[bands, f, c],
            tokens * c,
        );
        attention(&mut ops, other, tokens, bands, c, block.attn_f);
        feed_forward(&mut ops, other, tokens, c, block.ff_f);
        // Across the frames of each band: back to [band][frame][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [other, Buf::Weights, cur],
            &[f, bands, c],
            tokens * c,
        );
        attention(&mut ops, cur, tokens, f, c, block.attn_t);
        feed_forward(&mut ops, cur, tokens, c, block.ff_t);
        // Patches of 2 bands by 3 frames, to twice the channels.
        let out_bands = bands / 2;
        elementwise(
            &mut ops,
            Kernel::Patches,
            [cur, Buf::Weights, Buf::Patches],
            &[f, c, 2, out_bands],
            out_bands * f * 6 * c,
        );
        matmul(
            &mut ops,
            Buf::Patches,
            other,
            out_bands * f,
            2 * c,
            6 * c,
            block.conv,
            None,
            false,
        );
        let count = out_bands * f * 2 * c;
        elementwise(
            &mut ops,
            Kernel::AffineGelu,
            [Buf::Table, Buf::Weights, other],
            &[count, 2 * c, block.scale as usize, block.shift as usize],
            count,
        );
        std::mem::swap(&mut cur, &mut other);
        bands = out_bands;
        c *= 2;
    }
    // Each frame's channels × bands, channel-major, to the width.
    elementwise(
        &mut ops,
        Kernel::Flatten,
        [cur, Buf::Weights, Buf::Patches],
        &[f, bands, c],
        f * c * bands,
    );
    matmul(
        &mut ops,
        Buf::Patches,
        other,
        f,
        DIM,
        c * bands,
        o.project,
        Some(o.project_bias),
        false,
    );
    for &(attn, ff) in &o.layers {
        attention(&mut ops, other, f, f, DIM, attn);
        feed_forward(&mut ops, other, f, DIM, ff);
    }
    norm(&mut ops, other, Buf::Normed, f, DIM, o.norm);
    matmul(
        &mut ops,
        Buf::Normed,
        Buf::Logits,
        f,
        2,
        DIM,
        o.head,
        Some(o.head_bias),
        false,
    );
    ops
}

/// The network on its device, and where its weights are there.
struct Inner {
    model: Model<Kernel, Buf>,
    offsets: Offsets,
}

/// The beat network on the GPU.
pub struct GpuNetwork {
    name: String,
    inner: Turned<Mutex<Inner>>,
    /// What its runs cost, read without waiting for one.
    counts: pace::Counts,
}

impl GpuNetwork {
    /// The network on the best GPU, set up on first use; `None` without a
    /// usable one, or if setting it up failed.
    pub fn shared() -> Option<&'static GpuNetwork> {
        static NET: OnceLock<Option<GpuNetwork>> = OnceLock::new();
        NET.get_or_init(|| std::panic::catch_unwind(GpuNetwork::new).ok()?.ok())
            .as_ref()
    }

    /// What its GPU work cost so far ([`pace::Tally::since`] an earlier
    /// tally says what was done between), without waiting for its work.
    pub fn tally(&self) -> pace::Tally {
        self.counts.tally()
    }

    /// Sets the network up on the best GPU.
    pub fn new() -> Result<GpuNetwork, String> {
        let (inner, name) = crate::device::open("jodel-gpu network", |device, queue, limits| {
            Inner::new(device, queue, limits)
        })?;
        Ok(GpuNetwork {
            name,
            counts: inner.model.pieces.counts(),
            inner: Turned::new(Mutex::new(inner)),
        })
    }
}

/// Lines for a diagnostic report: whether the network sets up on the GPU
/// (a network of its own, not [`GpuNetwork::shared`]'s), and how long a
/// 30 s piece takes on it.
pub(crate) fn report() -> Vec<String> {
    let started = std::time::Instant::now();
    let network = match std::panic::catch_unwind(GpuNetwork::new) {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return vec![format!("beat network set up: failed: {e}")],
        Err(_) => return vec!["beat network set up: panicked".into()],
    };
    let mut lines = vec![format!(
        "beat network set up: yes, in {} ms",
        started.elapsed().as_millis()
    )];
    // A 30 s piece of a quiet spectrogram.
    let spect = vec![-5.0f32; PIECE_FRAMES * BANDS];
    let mut times = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        if let Err(e) = network.forward(&spect, PIECE_FRAMES) {
            lines.push(format!("a 30 s piece of the beat network: failed: {e}"));
            return lines;
        }
        times.push(started.elapsed().as_millis());
    }
    lines.push(format!(
        "a 30 s piece of the beat network: {} ms, then {} ms",
        times[0], times[1]
    ));
    lines
}

impl NetRunner for GpuNetwork {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn forward(&self, spect: &[f32], frames: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU network failed before".to_owned())?;
        if frames == 0 || frames > PIECE_FRAMES || spect.len() != frames * BANDS {
            return Err(format!("{frames} frames, {} values", spect.len()));
        }
        let device = inner.model.program.device.clone();
        guarded(&device, || inner.run(frames, spect))
    }
}

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
    ) -> Result<Inner, String> {
        let (weights, offsets) = pack_weights()?;
        let f = PIECE_FRAMES;
        let wide = 4 * STEM * BANDS / 4 * f;
        let sizes = [
            (Buf::Weights, weights.len()),
            (Buf::Table, f * 32),
            (Buf::Spect, f * BANDS),
            (Buf::X, wide / 4),
            (Buf::Y, wide / 4),
            (Buf::Normed, wide / 4),
            (Buf::Qkv, 3 * wide / 4),
            (Buf::Gates, BANDS / 4 * f),
            (Buf::Attended, wide / 4),
            (Buf::Hidden, wide),
            (Buf::Patches, 3 * wide / 4),
            (Buf::Logits, 2 * f),
        ];
        let program = Program::new(
            device,
            queue,
            limits,
            "jodel-gpu network",
            &kernels_source(&[include_str!("net.wgsl")]),
        );
        let model = Model::new(program, &sizes, 2 * f, Pieces::resting())?;
        model.write(Buf::Weights, &weights);
        model.write(Buf::Table, &rotary_table(f));
        Ok(Inner { model, offsets })
    }

    /// One piece through its plan (recorded on first use): the beat and
    /// downbeat logits.
    fn run(&mut self, frames: usize, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.model.write(Buf::Spect, spect);
        let offsets = &self.offsets;
        let out = self.model.run(
            frames,
            || plan(offsets, frames),
            |_, _| {},
            (Buf::Logits, 2 * frames),
        )?;
        let downbeat: Vec<f32> = out.chunks(2).map(|o| o[1]).collect();
        let beat = out.chunks(2).map(|o| o[0] + o[1]).collect();
        Ok((beat, downbeat))
    }
}
