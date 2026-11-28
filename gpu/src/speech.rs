//! The speech model of [`jodel_lyrics::speech`] (wav2vec2-base-960h) on the
//! GPU: the same network, weights and pieces as the CPU's, as one list of
//! dispatches a piece (`speech.wgsl` with `shared.wgsl`'s matrix multiply
//! and GELU), giving the same logits to within rounding.

use std::collections::HashMap;
use std::sync::Mutex;

use jodel_lyrics::aligner::Acoustic;
use jodel_lyrics::ctc::Emissions;
use jodel_lyrics::speech::{
    emissions_in_pieces, Levels, Wav2Vec2, CONTEXT_SECONDS, CONV, HOP, PIECE_SECONDS, RECEPTIVE,
};
use jodel_lyrics::{Maker, VOCALS_RATE};

use crate::engine::{
    buffers_footprint, params, shared_source, spread, Access, Input, Kernels, Model, ModelOp,
    OnDevice, Output, PackedNet, Program, Tiled, TiledMul, PARAM_WORDS_BYTES,
};
use crate::pace::Pieces;
use crate::packer::Named;
use crate::{Footprint, Turned};

const CHANNELS: usize = 512;
const DIM: usize = 768;
const QKV: usize = 3 * DIM;
const FF: usize = 3072;
const LAYERS: usize = 12;
const POS: usize = 128 * 48;
const TOKENS: usize = 32;

/// The kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kernel {
    Matmul,
    Conv0,
    ChanNormGelu,
    Gelu,
    LayerNorm,
    PosPatches,
    PosAdd,
    Attention,
}

impl Kernels for Kernel {
    const ALL: &'static [Kernel] = &[
        Kernel::Matmul,
        Kernel::Conv0,
        Kernel::ChanNormGelu,
        Kernel::Gelu,
        Kernel::LayerNorm,
        Kernel::PosPatches,
        Kernel::PosAdd,
        Kernel::Attention,
    ];
    const PARAM_BYTES: u64 = PARAM_WORDS_BYTES;
    const EXTRA: &'static [Access] = &[];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Conv0 => "sp_conv0",
            Kernel::ChanNormGelu => "sp_chan_norm_gelu",
            Kernel::Gelu => "gelu_in_place",
            Kernel::LayerNorm => "sp_layernorm",
            Kernel::PosPatches => "sp_pos_patches",
            Kernel::PosAdd => "sp_pos_add",
            Kernel::Attention => "sp_attention",
        }
    }
}

/// The buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Buf {
    Weights,
    Audio,
    ConvA,
    ConvB,
    Stats,
    Hidden,
    Qkv,
    Attended,
    Up,
    Patches,
    Pos,
    Logits,
}

/// The most samples a piece has: its frames and the context either side.
fn longest_piece() -> usize {
    let frames_of = |s: f64| (s * f64::from(VOCALS_RATE)) as usize / HOP;
    let frames = frames_of(PIECE_SECONDS) + 2 * frames_of(CONTEXT_SECONDS);
    (frames - 1) * HOP + RECEPTIVE
}

/// The frames after each convolution of `samples` samples.
fn conv_frames(samples: usize) -> [usize; 7] {
    let mut out = [0; 7];
    let mut n = samples;
    for (l, &(k, s)) in CONV.iter().enumerate() {
        n = if n < k { 0 } else { (n - k) / s + 1 };
        out[l] = n;
    }
    out
}

/// The weights as one buffer's contents, by name (each layer's queries,
/// keys and values side by side, `{l}.qkv.w` and `{l}.qkv.b`).
fn pack(model: &Wav2Vec2) -> Named {
    let mut named = Named::default();
    let mut parts: HashMap<String, Vec<f32>> = HashMap::new();
    model.visit(&mut |name, values| {
        // Each layer's queries, keys and values go side by side (below).
        let fused = [".q.", ".k.", ".v."].iter().any(|p| name.contains(p));
        if fused {
            parts.insert(name.to_owned(), values.to_vec());
        } else {
            named.put(name, values);
        }
    });
    for l in 0..LAYERS {
        let mut take = |what: &str| parts.remove(&format!("{l}.{what}")).unwrap_or_default();
        let (q, k, v) = (take("q.w"), take("k.w"), take("v.w"));
        let mut qkv = Vec::with_capacity(3 * DIM * DIM);
        for i in 0..DIM {
            for w in [&q, &k, &v] {
                qkv.extend_from_slice(&w[i * DIM..(i + 1) * DIM]);
            }
        }
        named.put(&format!("{l}.qkv.w"), &qkv);
        let b: Vec<f32> = ["q.b", "k.b", "v.b"].iter().flat_map(|b| take(b)).collect();
        named.put(&format!("{l}.qkv.b"), &b);
    }
    named
}

/// One dispatch.
type Op = ModelOp<Kernel, Buf>;

/// The model on its device.
type Net = PackedNet<Kernel, Buf>;

/// `mul` through the tiled matrix multiply.
fn matmul(ops: &mut Vec<Op>, mul: TiledMul<Buf>) {
    ops.push(mul.tiled(Kernel::Matmul, Buf::Weights));
}

/// An elementwise or per-row kernel over `count` invocations.
fn each(ops: &mut Vec<Op>, kernel: Kernel, buffers: [Buf; 3], count: usize, values: &[usize]) {
    ops.push(Op {
        kernel,
        buffers,
        params: params(values),
        groups: spread(count),
        cost: crate::pace::Cost::UNESTIMATED,
    });
}

/// The dispatches of a piece of `samples` samples, the weights where `at`
/// says.
fn plan(at: &Named, samples: usize) -> Vec<Op> {
    let w = |name: &str| at.at(name);
    let frames = conv_frames(samples);
    let mut ops = Vec::new();
    let t0 = frames[0];
    each(
        &mut ops,
        Kernel::Conv0,
        [Buf::Audio, Buf::Weights, Buf::ConvA],
        t0 * CHANNELS,
        &[t0, w("conv0")],
    );
    // The GroupNorm at the whole voice's statistics, written to `Stats`
    // before each piece.
    each(
        &mut ops,
        Kernel::ChanNormGelu,
        [Buf::Stats, Buf::Weights, Buf::ConvA],
        t0 * CHANNELS,
        &[t0 * CHANNELS, CHANNELS, w("gn.w"), w("gn.b")],
    );
    let (mut from, mut to) = (Buf::ConvA, Buf::ConvB);
    for (l, &(k, s)) in CONV.iter().enumerate().skip(1) {
        let m = frames[l];
        // Each frame's patch of `k` frames, `s` frames on from the last.
        matmul(
            &mut ops,
            TiledMul {
                from: (from, 0, s * CHANNELS),
                ..TiledMul::dense(
                    from,
                    to,
                    (m, CHANNELS, k * CHANNELS),
                    w(&format!("conv{l}")),
                    None,
                    Tiled::Store,
                )
            },
        );
        each(
            &mut ops,
            Kernel::Gelu,
            [Buf::Stats, Buf::Weights, to],
            m * CHANNELS,
            &[m * CHANNELS],
        );
        std::mem::swap(&mut from, &mut to);
    }
    let f = frames[6];
    let norm = |ops: &mut Vec<Op>, buf: Buf, dim: usize, name: &str| {
        // 32 invocations a row.
        each(
            ops,
            Kernel::LayerNorm,
            [Buf::Stats, Buf::Weights, buf],
            32 * f,
            &[f, dim, w(&format!("{name}.w")), w(&format!("{name}.b"))],
        );
    };
    // A layer `name` of `f` rows, its bias with it.
    let layer = |from, to, (n, k), name: &str, after| {
        TiledMul::dense(
            from,
            to,
            (f, n, k),
            w(&format!("{name}.w")),
            Some(w(&format!("{name}.b"))),
            after,
        )
    };
    norm(&mut ops, from, CHANNELS, "fp.ln");
    matmul(
        &mut ops,
        layer(from, Buf::Hidden, (DIM, CHANNELS), "proj", Tiled::Store),
    );
    for g in 0..16 {
        each(
            &mut ops,
            Kernel::PosPatches,
            [Buf::Hidden, Buf::Weights, Buf::Patches],
            f * POS,
            &[f, g],
        );
        // Each group's 48 channels, into its columns of the width.
        matmul(
            &mut ops,
            TiledMul {
                to: (Buf::Pos, g * 48, DIM),
                ..TiledMul::dense(
                    Buf::Patches,
                    Buf::Pos,
                    (f, 48, POS),
                    w(&format!("pos{g}")),
                    None,
                    Tiled::Store,
                )
            },
        );
    }
    each(
        &mut ops,
        Kernel::PosAdd,
        [Buf::Pos, Buf::Weights, Buf::Hidden],
        f * DIM,
        &[f * DIM, w("pos.b")],
    );
    norm(&mut ops, Buf::Hidden, DIM, "enc.ln");
    for l in 0..LAYERS {
        let n = |what: &str| format!("{l}.{what}");
        matmul(
            &mut ops,
            layer(Buf::Hidden, Buf::Qkv, (QKV, DIM), &n("qkv"), Tiled::Store),
        );
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Qkv, Buf::Weights, Buf::Attended],
            params: params(&[f]),
            groups: [f.div_ceil(64) as u32, 12, 1],
            cost: crate::pace::Cost::UNESTIMATED,
        });
        matmul(
            &mut ops,
            layer(Buf::Attended, Buf::Hidden, (DIM, DIM), &n("o"), Tiled::Add),
        );
        norm(&mut ops, Buf::Hidden, DIM, &n("ln"));
        matmul(
            &mut ops,
            layer(Buf::Hidden, Buf::Up, (FF, DIM), &n("up"), Tiled::Store),
        );
        each(
            &mut ops,
            Kernel::Gelu,
            [Buf::Stats, Buf::Weights, Buf::Up],
            f * FF,
            &[f * FF],
        );
        matmul(
            &mut ops,
            layer(Buf::Up, Buf::Hidden, (DIM, FF), &n("down"), Tiled::Add),
        );
        norm(&mut ops, Buf::Hidden, DIM, &n("fln"));
    }
    matmul(
        &mut ops,
        layer(
            Buf::Hidden,
            Buf::Logits,
            (TOKENS, DIM),
            "head",
            Tiled::Store,
        ),
    );
    ops
}

/// wav2vec2-base-960h on the GPU.
pub struct GpuSpeech {
    name: String,
    /// The first convolution's weights, for a voice's [`Levels`].
    conv0: Vec<f32>,
    net: Turned<Mutex<Net>>,
}

impl GpuSpeech {
    /// The model on the best GPU, from the CPU's (its weights as they are).
    pub fn new(model: &Wav2Vec2) -> Result<GpuSpeech, String> {
        let (net, name) = crate::device::open("jodel-gpu speech", |device, queue, limits| {
            on(device, queue, limits, model)
        })?;
        let mut conv0 = Vec::new();
        model.visit(&mut |name, values| {
            if name == "conv0" {
                conv0 = values.to_vec();
            }
        });
        Ok(GpuSpeech {
            name,
            conv0,
            net: Turned::new(Mutex::new(net)),
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The logits (`frames × 32`) of a voice at 16 kHz heard on its own, in
    /// one piece (at most a piece and its context long): normalised at its
    /// own levels, then the network.
    pub fn logits(&self, voice: &[f32]) -> Result<Vec<f32>, String> {
        self.logits_at(voice, &Levels::of(&self.conv0, voice))
    }

    /// The logits (`frames × 32`) of a piece of a voice at 16 kHz (at most
    /// a piece and its context long), heard at the whole voice's `levels`.
    pub fn logits_at(&self, piece: &[f32], levels: &Levels) -> Result<Vec<f32>, String> {
        if piece.len() > longest_piece() {
            return Err(format!("{} samples, longer than a piece", piece.len()));
        }
        if Wav2Vec2::frames_of(piece.len()) == 0 {
            return Ok(Vec::new());
        }
        let mut net = self
            .net
            .lock()
            .map_err(|_| "the GPU speech model failed before".to_owned())?;
        let x = levels.normalise(piece);
        net.run(&x, &levels.channels(&x))
    }
}

impl OnDevice for GpuSpeech {
    fn lost(&self) -> bool {
        self.net.lock().map_or(true, |net| net.lost())
    }
}

/// The floats of each of the model's buffers, its weights `weights` of
/// them, and the floats read back.
fn sizes(weights: usize) -> ([(Buf, usize); 12], usize) {
    let samples = longest_piece();
    let frames = conv_frames(samples);
    let f = frames[6];
    let sizes = [
        (Buf::Weights, weights),
        (Buf::Audio, samples),
        (Buf::ConvA, frames[0] * CHANNELS),
        (Buf::ConvB, frames[1] * CHANNELS),
        (Buf::Stats, 2 * CHANNELS),
        (Buf::Hidden, f * DIM),
        (Buf::Qkv, f * QKV),
        (Buf::Attended, f * DIM),
        (Buf::Up, f * FF),
        (Buf::Patches, f * POS),
        (Buf::Pos, f * DIM),
        (Buf::Logits, f * TOKENS),
    ];
    (sizes, f * TOKENS)
}

impl GpuSpeech {
    /// What the model takes on the GPU, from its shapes, its checkpoint
    /// of `checkpoint` bytes (its f32 weights, at most): every buffer it
    /// keeps (its weights and a piece's room) and the one it reads back
    /// through; and its weights once more while they go up.
    pub fn footprint(checkpoint: u64) -> Footprint {
        let (sizes, read) = sizes((checkpoint / 4) as usize);
        buffers_footprint(&sizes, read, checkpoint)
    }
}

impl Acoustic for GpuSpeech {
    fn maker(&self) -> Maker {
        jodel_lyrics::speech::maker()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        let levels = Levels::of(&self.conv0, vocals_16k);
        emissions_in_pieces(vocals_16k, |piece| self.logits_at(piece, &levels).ok())
    }
}

/// The model on a device, from the CPU's.
fn on(
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: &wgpu::Limits,
    model: &Wav2Vec2,
) -> Result<Net, String> {
    let mut weights = pack(model);
    let values = weights.take_values();
    let (sizes, read) = sizes(values.len());
    let program = Program::new(
        device,
        queue,
        limits,
        "jodel-gpu speech",
        &shared_source(&[include_str!("speech.wgsl")]),
    );
    let writes = [(Buf::Weights, &values[..])];
    let model = Model::new(program, &sizes, &writes, read, Pieces::resting())?;
    Ok(PackedNet { model, weights })
}

impl Net {
    /// One piece through its plan (recorded on first use), its first
    /// convolution's channels at `channels` (each one's mean and 1/σ): the
    /// logits.
    fn run(&mut self, x: &[f32], channels: &[f32]) -> Result<Vec<f32>, String> {
        let frames = conv_frames(x.len())[6];
        let weights = &self.weights;
        self.model.run(
            x.len(),
            || plan(weights, x.len()),
            &[
                Input {
                    buf: Buf::Audio,
                    at: 0,
                    values: x,
                },
                Input {
                    buf: Buf::Stats,
                    at: 0,
                    values: channels,
                },
            ],
            |_, _| {},
            Output {
                buf: Buf::Logits,
                first: 0,
                count: frames * TOKENS,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_piece_fits_the_frames() {
        let frames = conv_frames(longest_piece());
        assert_eq!(frames[6], Wav2Vec2::frames_of(longest_piece()));
        assert_eq!(frames[6], 1700);
    }
}
