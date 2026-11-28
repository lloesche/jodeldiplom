//! The beat network of [`jodel_rhythm::net`] (Beat This!, small) on the
//! GPU, its weights packed as the CPU network holds them
//! ([`Network::visit`]): [`GpuNetwork`] is a [`NetRunner`], so the analysis
//! runs a piece's layers here and only the two logits per frame come back.
//!
//! A piece is one list of dispatches: the spectrogram in, the stem's
//! convolution (as patches through the tiled matrix multiply), the three
//! blocks (attention and feed-forward across the bands of each frame, then
//! across the frames of each band, a convolution halving the bands), the
//! projection, six transformer layers, the final RMSNorm and the head.
//! The matrix multiply, RMSNorm, rotary, transpose and GELU kernels are
//! the shared ones (`shared.wgsl`, in the beat network's modes); the rest
//! are in `net.wgsl`. The process keeps one ([`GpuNetwork::shared`]), set
//! up when first needed and again when its device is lost
//! (`engine::KeptNet`).

use std::sync::{Arc, OnceLock};

use jodel_rhythm::net::mel::BANDS;
use jodel_rhythm::net::{NetRunner, Network, BLOCKS, DIM, LAYERS, PIECE_FRAMES, STEM};
use jodel_rhythm::nn::layers::HEAD;

use crate::engine::{
    self, buffers_footprint, params, rotary, shared_source, spread, Access, Angles, Input, KeptNet,
    Kernels, Model, ModelOp, Output, PackedNet, Program, Tiled, TiledMul, PARAM_WORDS_BYTES,
};
use crate::pace::{self, Pieces};
use crate::packer::Named;
use crate::{Footprint, Resident};

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
    const PARAM_BYTES: u64 = PARAM_WORDS_BYTES;
    const EXTRA: &'static [Access] = &[];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Norm => "rmsnorm",
            Kernel::Rope => "rope",
            Kernel::In => "net_in",
            Kernel::Patches => "net_patches",
            Kernel::AffineGelu => "net_affine_gelu",
            Kernel::Gelu => "gelu_in_place",
            Kernel::Transpose => "transpose",
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

/// The weights as one buffer's contents, by name, as the CPU network
/// holds them.
fn pack() -> Result<Named, String> {
    let cpu = Network::built_in()?;
    Ok(Named::visited(|f| cpu.visit(f)))
}

/// One dispatch.
type Op = ModelOp<Kernel, Buf>;

/// The network on its device.
type Net = PackedNet<Kernel, Buf>;

/// `dst` (+)= `from` · the weights `w` (+ the bias `bias`), `m` rows of `k`
/// to `n`, each matrix whole.
fn matmul(
    ops: &mut Vec<Op>,
    (from, to): (Buf, Buf),
    (m, n, k): (usize, usize, usize),
    (w, bias): (usize, Option<usize>),
    after: Tiled,
) {
    ops.push(
        TiledMul::dense(from, to, (m, n, k), w, bias, after).tiled(Kernel::Matmul, Buf::Weights),
    );
}

/// The network's RMSNorm (its mode 1) of `rows` rows of `dim` from `x` to
/// `y`, its gamma at `w`.
fn norm(ops: &mut Vec<Op>, x: Buf, y: Buf, rows: usize, dim: usize, w: usize) {
    ops.push(Op {
        kernel: Kernel::Norm,
        buffers: [x, Buf::Weights, y],
        params: params(&[rows, dim, 0, dim, w, 0, dim, 1]),
        // 32 invocations a row.
        groups: spread(32 * rows),
        cost: crate::pace::Cost::UNESTIMATED,
    });
}

/// The attention `name` over sequences of `len` tokens of `dim`, added to
/// `x`.
fn attention(
    ops: &mut Vec<Op>,
    w: &Named,
    name: &str,
    x: Buf,
    (tokens, len, dim): (usize, usize, usize),
) {
    let at = |what: &str| w.at(&format!("{name}.{what}"));
    let heads = dim / HEAD;
    norm(ops, x, Buf::Normed, tokens, dim, at("norm"));
    matmul(
        ops,
        (Buf::Normed, Buf::Qkv),
        (tokens, 3 * dim, dim),
        (at("qkv.w"), None),
        Tiled::Store,
    );
    matmul(
        ops,
        (Buf::Normed, Buf::Gates),
        (tokens, heads, dim),
        (at("gates.w"), Some(at("gates.b"))),
        Tiled::Store,
    );
    for col in [0, dim] {
        ops.push(Op {
            kernel: Kernel::Rope,
            buffers: [Buf::Table, Buf::Weights, Buf::Qkv],
            params: params(&[tokens, len, len, heads, 3 * dim, col, 1]),
            groups: spread(tokens * heads * 16),
            cost: crate::pace::Cost::UNESTIMATED,
        });
    }
    ops.push(Op {
        kernel: Kernel::Attention,
        buffers: [Buf::Qkv, Buf::Gates, Buf::Attended],
        params: params(&[len, dim, heads]),
        groups: [len.div_ceil(64) as u32, heads as u32, (tokens / len) as u32],
        cost: crate::pace::Cost::UNESTIMATED,
    });
    matmul(
        ops,
        (Buf::Attended, x),
        (tokens, dim, dim),
        (at("out.w"), None),
        Tiled::Add,
    );
}

/// The feed-forward step `name` on `tokens` tokens of `dim`, added to `x`.
fn feed_forward(ops: &mut Vec<Op>, w: &Named, name: &str, x: Buf, (tokens, dim): (usize, usize)) {
    let at = |what: &str| w.at(&format!("{name}.{what}"));
    norm(ops, x, Buf::Normed, tokens, dim, at("norm"));
    matmul(
        ops,
        (Buf::Normed, Buf::Hidden),
        (tokens, 4 * dim, dim),
        (at("up.w"), Some(at("up.b"))),
        Tiled::Store,
    );
    let count = tokens * 4 * dim;
    ops.push(Op {
        kernel: Kernel::Gelu,
        buffers: [Buf::Table, Buf::Weights, Buf::Hidden],
        params: params(&[count]),
        groups: spread(count),
        cost: crate::pace::Cost::UNESTIMATED,
    });
    matmul(
        ops,
        (Buf::Hidden, x),
        (tokens, dim, 4 * dim),
        (at("down.w"), Some(at("down.b"))),
        Tiled::Add,
    );
}

/// The dispatches of a piece of `frames` frames, the weights where `w`
/// says.
fn plan(w: &Named, frames: usize) -> Vec<Op> {
    let f = frames;
    let mut ops: Vec<Op> = Vec::new();
    let elementwise = |ops: &mut Vec<Op>, kernel, buffers, p: &[usize], count: usize| {
        ops.push(Op {
            kernel,
            buffers,
            params: params(p),
            groups: spread(count),
            cost: crate::pace::Cost::UNESTIMATED,
        });
    };
    // The stem: each band scaled, patches of 4 bands by 3 frames, to 32
    // channels, scaled, GELU.
    elementwise(
        &mut ops,
        Kernel::In,
        [Buf::Spect, Buf::Weights, Buf::X],
        &[f, BANDS, w.at("in.scale"), w.at("in.shift")],
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
        (Buf::Patches, Buf::Y),
        (bands * f, STEM, 12),
        (w.at("stem.w"), None),
        Tiled::Store,
    );
    elementwise(
        &mut ops,
        Kernel::AffineGelu,
        [Buf::Table, Buf::Weights, Buf::Y],
        &[
            bands * f * STEM,
            STEM,
            w.at("stem.scale"),
            w.at("stem.shift"),
        ],
        bands * f * STEM,
    );
    // The blocks, the activations in `cur` as [band][frame][channel].
    let (mut cur, mut other) = (Buf::Y, Buf::X);
    let mut c = STEM;
    for b in 0..BLOCKS {
        let tokens = bands * f;
        // Across the bands of each frame: [frame][band][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [cur, Buf::Weights, other],
            &[bands, f, c],
            tokens * c,
        );
        attention(
            &mut ops,
            w,
            &format!("{b}.attn_f"),
            other,
            (tokens, bands, c),
        );
        feed_forward(&mut ops, w, &format!("{b}.ff_f"), other, (tokens, c));
        // Across the frames of each band: back to [band][frame][channel].
        elementwise(
            &mut ops,
            Kernel::Transpose,
            [other, Buf::Weights, cur],
            &[f, bands, c],
            tokens * c,
        );
        attention(&mut ops, w, &format!("{b}.attn_t"), cur, (tokens, f, c));
        feed_forward(&mut ops, w, &format!("{b}.ff_t"), cur, (tokens, c));
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
            (Buf::Patches, other),
            (out_bands * f, 2 * c, 6 * c),
            (w.at(&format!("{b}.conv.w")), None),
            Tiled::Store,
        );
        let count = out_bands * f * 2 * c;
        elementwise(
            &mut ops,
            Kernel::AffineGelu,
            [Buf::Table, Buf::Weights, other],
            &[
                count,
                2 * c,
                w.at(&format!("{b}.conv.scale")),
                w.at(&format!("{b}.conv.shift")),
            ],
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
        (Buf::Patches, other),
        (f, DIM, c * bands),
        (w.at("project.w"), Some(w.at("project.b"))),
        Tiled::Store,
    );
    for l in 0..LAYERS {
        attention(&mut ops, w, &format!("layer{l}.attn"), other, (f, f, DIM));
        feed_forward(&mut ops, w, &format!("layer{l}.ff"), other, (f, DIM));
    }
    norm(&mut ops, other, Buf::Normed, f, DIM, w.at("norm"));
    matmul(
        &mut ops,
        (Buf::Normed, Buf::Logits),
        (f, 2, DIM),
        (w.at("head.w"), Some(w.at("head.b"))),
        Tiled::Store,
    );
    ops
}

/// The beat network on the GPU.
pub struct GpuNetwork {
    net: KeptNet<Kernel, Buf>,
}

/// The network set up on the best GPU (with its name), its runs paced as
/// `pieces` say.
fn open(pieces: Pieces) -> Result<(Net, String), String> {
    crate::device::open("jodel-gpu network", |device, queue, limits| {
        on(device, queue, limits, pieces)
    })
}

/// The floats of each of the network's buffers, its weights `weights` of
/// them.
fn sizes(weights: usize) -> [(Buf, usize); 12] {
    let f = PIECE_FRAMES;
    let wide = 4 * STEM * BANDS / 4 * f;
    [
        (Buf::Weights, weights),
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
    ]
}

/// The floats a piece's logits are read back through.
const READ: usize = 2 * PIECE_FRAMES;

/// What the network takes on the GPU, from its shapes: every buffer it
/// keeps while it is set up and the one it reads back through; and its
/// weights once more while they go up. Its weights are counted as packed
/// (built in: packed on the processor for this, once a process, as a
/// scheduler first asks).
fn footprint() -> Option<Footprint> {
    let weights = pack().ok()?.take_values().len();
    Some(buffers_footprint(&sizes(weights), READ, weights as u64 * 4))
}

impl GpuNetwork {
    /// The network on the best GPU, set up on first use (and again once
    /// its device is lost); `None` without a usable one, or while setting
    /// it up fails (it is tried again a while later).
    pub fn shared() -> Option<&'static GpuNetwork> {
        let net = GpuNetwork::kept_shared();
        net.net.usable().then_some(net)
    }

    /// The shared network as a scheduler keeps it, set up or not.
    pub fn shared_resident() -> Arc<dyn Resident> {
        GpuNetwork::kept_shared().resident()
    }

    /// The network as a scheduler keeps it: its footprint, and its
    /// release.
    pub fn resident(&self) -> Arc<dyn Resident> {
        self.net.resident()
    }

    /// The shared network, set up or not.
    fn kept_shared() -> &'static GpuNetwork {
        static NET: OnceLock<GpuNetwork> = OnceLock::new();
        NET.get_or_init(|| GpuNetwork {
            net: KeptNet::new(Pieces::resting(), footprint, open),
        })
    }

    /// What its GPU work cost so far ([`pace::Tally::since`] an earlier
    /// tally says what was done between), without waiting for its work.
    pub fn tally(&self) -> pace::Tally {
        self.net.tally()
    }

    /// Sets up a network of its own on the best GPU.
    pub fn new() -> Result<GpuNetwork, String> {
        Ok(GpuNetwork {
            net: KeptNet::set_up_now(Pieces::resting(), footprint, open)?,
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
        self.net.name()
    }

    fn forward(&self, spect: &[f32], frames: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        if frames == 0 || frames > PIECE_FRAMES || spect.len() != frames * BANDS {
            return Err(format!("{frames} frames, {} values", spect.len()));
        }
        self.net.run(|net| net.run(frames, spect))
    }
}

/// The network on a device, its runs paced as `pieces` say.
fn on(
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: &wgpu::Limits,
    pieces: Pieces,
) -> Result<Net, String> {
    let mut weights = pack()?;
    let values = weights.take_values();
    let f = PIECE_FRAMES;
    let program = Program::new(
        device,
        queue,
        limits,
        "jodel-gpu network",
        &shared_source(&[include_str!("net.wgsl")]),
    );
    let table = rotary(0..f, engine::HEAD_32, Angles::F32);
    let writes = [(Buf::Weights, &values[..]), (Buf::Table, &table[..])];
    let model = Model::new(program, &sizes(values.len()), &writes, READ, pieces)?;
    Ok(PackedNet { model, weights })
}

impl Net {
    /// One piece through its plan (recorded on first use): the beat and
    /// downbeat logits.
    fn run(&mut self, frames: usize, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let weights = &self.weights;
        let out = self.model.run(
            frames,
            || plan(weights, frames),
            &[Input {
                buf: Buf::Spect,
                at: 0,
                values: spect,
            }],
            |_, _| {},
            Output {
                buf: Buf::Logits,
                first: 0,
                count: 2 * frames,
            },
        )?;
        let downbeat: Vec<f32> = out.chunks(2).map(|o| o[1]).collect();
        let beat = out.chunks(2).map(|o| o[0] + o[1]).collect();
        Ok((beat, downbeat))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A network whose device is lost sets up again and gives the same
    /// logits; skipped (with a note) where there is no usable GPU.
    #[test]
    fn a_network_whose_device_is_lost_sets_up_again() {
        let Ok(network) = GpuNetwork::new() else {
            eprintln!("no usable GPU: skipped");
            return;
        };
        let spect: Vec<f32> = (0..300 * BANDS)
            .map(|i| -5.0 + (i % 13) as f32 * 0.4)
            .collect();
        let before = network.forward(&spect, 300).unwrap();
        network
            .net
            .run(|net| {
                net.model.program.device.destroy();
                Ok(())
            })
            .unwrap();
        assert_eq!(network.forward(&spect, 300).unwrap(), before);
    }
}
