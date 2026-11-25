//! The speech model of [`jodel_lyrics::speech`] (wav2vec2-base-960h) on the
//! GPU: the same network, weights and pieces as the CPU's, as one list of
//! dispatches a piece (`speech.wgsl` with `kernels.wgsl`'s matrix
//! multiply), giving the same logits to within rounding.

use std::collections::HashMap;
use std::sync::Mutex;

use jodel_lyrics::aligner::Acoustic;
use jodel_lyrics::ctc::Emissions;
use jodel_lyrics::speech::{
    emissions_in_pieces, Levels, Wav2Vec2, CONTEXT_SECONDS, CONV, HOP, PIECE_SECONDS, RECEPTIVE,
};
use jodel_lyrics::{Maker, VOCALS_RATE};

use crate::engine::{
    self, guarded, kernels_source, params, spread, Access, Kernels, Model, Program,
};
use crate::pace::Pieces;
use crate::packer::Packer;
use crate::Turned;

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
    const PARAM_BYTES: u64 = 48;
    const EXTRA: &'static [Access] = &[];

    fn entry(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::Conv0 => "sp_conv0",
            Kernel::ChanNormGelu => "sp_chan_norm_gelu",
            Kernel::Gelu => "sp_gelu",
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

/// The weights as one buffer's contents, and where each is (in f32s).
fn pack(model: &Wav2Vec2) -> (Vec<f32>, HashMap<String, u32>) {
    let mut data = Packer::default();
    let mut at = HashMap::new();
    let mut parts: HashMap<String, Vec<f32>> = HashMap::new();
    model.visit(&mut |name, values| {
        // Each layer's queries, keys and values go side by side (below).
        let fused = [".q.", ".k.", ".v."].iter().any(|p| name.contains(p));
        if fused {
            parts.insert(name.to_owned(), values.to_vec());
        } else {
            at.insert(name.to_owned(), data.push(values) as u32);
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
        at.insert(format!("{l}.qkv.w"), data.push(&qkv) as u32);
        let b: Vec<f32> = ["q.b", "k.b", "v.b"].iter().flat_map(|b| take(b)).collect();
        at.insert(format!("{l}.qkv.b"), data.push(&b) as u32);
    }
    (data.data, at)
}

/// One dispatch.
type Op = engine::Op<Kernel, [Buf; 3], 12>;

/// `dst` (`m × n` at `c_off`, rows `ldc` apart) (+)= `src` rows (`k` values
/// from `a_off`, `lda` apart) times the weight at `w` (+ the bias at `bias`).
#[allow(clippy::too_many_arguments)]
fn matmul(
    ops: &mut Vec<Op>,
    src: Buf,
    dst: Buf,
    (m, n, k): (usize, usize, usize),
    lda: usize,
    w: u32,
    (c_off, ldc): (usize, usize),
    bias: Option<u32>,
    add: bool,
) {
    let flags = usize::from(bias.is_some()) | (usize::from(add) << 1);
    ops.push(Op {
        kernel: Kernel::Matmul,
        buffers: [src, Buf::Weights, dst],
        params: params(&[
            m,
            n,
            k,
            0,
            lda,
            w as usize,
            c_off,
            ldc,
            bias.unwrap_or(0) as usize,
            flags,
        ]),
        groups: [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1],
    });
}

/// An elementwise or per-row kernel over `count` invocations.
fn each(ops: &mut Vec<Op>, kernel: Kernel, buffers: [Buf; 3], count: usize, values: &[usize]) {
    ops.push(Op {
        kernel,
        buffers,
        params: params(values),
        groups: spread(count),
    });
}

/// The dispatches of a piece of `samples` samples.
fn plan(at: &HashMap<String, u32>, samples: usize) -> Vec<Op> {
    let w = |name: &str| at[name] as usize;
    let wu = |name: &str| at[name];
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
        matmul(
            &mut ops,
            from,
            to,
            (m, CHANNELS, k * CHANNELS),
            s * CHANNELS,
            wu(&format!("conv{l}")),
            (0, CHANNELS),
            None,
            false,
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
        each(
            ops,
            Kernel::LayerNorm,
            [Buf::Stats, Buf::Weights, buf],
            f,
            &[f, dim, w(&format!("{name}.w")), w(&format!("{name}.b"))],
        );
    };
    norm(&mut ops, from, CHANNELS, "fp.ln");
    matmul(
        &mut ops,
        from,
        Buf::Hidden,
        (f, DIM, CHANNELS),
        CHANNELS,
        wu("proj.w"),
        (0, DIM),
        Some(wu("proj.b")),
        false,
    );
    for g in 0..16 {
        each(
            &mut ops,
            Kernel::PosPatches,
            [Buf::Hidden, Buf::Weights, Buf::Patches],
            f * POS,
            &[f, g],
        );
        matmul(
            &mut ops,
            Buf::Patches,
            Buf::Pos,
            (f, 48, POS),
            POS,
            wu(&format!("pos{g}")),
            (g * 48, DIM),
            None,
            false,
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
        let n = |what: &str| wu(&format!("{l}.{what}"));
        matmul(
            &mut ops,
            Buf::Hidden,
            Buf::Qkv,
            (f, QKV, DIM),
            DIM,
            n("qkv.w"),
            (0, QKV),
            Some(n("qkv.b")),
            false,
        );
        ops.push(Op {
            kernel: Kernel::Attention,
            buffers: [Buf::Qkv, Buf::Weights, Buf::Attended],
            params: params(&[f]),
            groups: [f.div_ceil(64) as u32, 12, 1],
        });
        matmul(
            &mut ops,
            Buf::Attended,
            Buf::Hidden,
            (f, DIM, DIM),
            DIM,
            n("o.w"),
            (0, DIM),
            Some(n("o.b")),
            true,
        );
        norm(&mut ops, Buf::Hidden, DIM, &format!("{l}.ln"));
        matmul(
            &mut ops,
            Buf::Hidden,
            Buf::Up,
            (f, FF, DIM),
            DIM,
            n("up.w"),
            (0, FF),
            Some(n("up.b")),
            false,
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
            Buf::Up,
            Buf::Hidden,
            (f, DIM, FF),
            FF,
            n("down.w"),
            (0, DIM),
            Some(n("down.b")),
            true,
        );
        norm(&mut ops, Buf::Hidden, DIM, &format!("{l}.fln"));
    }
    matmul(
        &mut ops,
        Buf::Hidden,
        Buf::Logits,
        (f, TOKENS, DIM),
        DIM,
        wu("head.w"),
        (0, TOKENS),
        Some(wu("head.b")),
        false,
    );
    ops
}

/// The model on its device, and where its weights are there.
struct Inner {
    model: Model<Kernel, Buf>,
    at: HashMap<String, u32>,
}

/// wav2vec2-base-960h on the GPU.
pub struct GpuSpeech {
    name: String,
    /// The first convolution's weights, for a voice's [`Levels`].
    conv0: Vec<f32>,
    inner: Turned<Mutex<Inner>>,
}

impl GpuSpeech {
    /// The model on the best GPU, from the CPU's (its weights as they are).
    pub fn new(model: &Wav2Vec2) -> Result<GpuSpeech, String> {
        let (inner, name) = crate::device::open("jodel-gpu speech", |device, queue, limits| {
            Inner::new(device, queue, limits, model)
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
            inner: Turned::new(Mutex::new(inner)),
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
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "the GPU speech model failed before".to_owned())?;
        let device = inner.model.program.device.clone();
        let x = levels.normalise(piece);
        guarded(&device, || inner.run(&x, &levels.channels(&x)))
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

impl Inner {
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: &wgpu::Limits,
        model: &Wav2Vec2,
    ) -> Result<Inner, String> {
        let (weights, at) = pack(model);
        let samples = longest_piece();
        let frames = conv_frames(samples);
        let f = frames[6];
        let sizes = [
            (Buf::Weights, weights.len()),
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
        let program = Program::new(
            device,
            queue,
            limits,
            "jodel-gpu speech",
            &kernels_source(&[include_str!("speech.wgsl")]),
        );
        let model = Model::new(program, &sizes, f * TOKENS, Pieces::resting())?;
        model.write(Buf::Weights, &weights);
        Ok(Inner { model, at })
    }

    /// One piece through its plan (recorded on first use), its first
    /// convolution's channels at `channels` (each one's mean and 1/σ): the
    /// logits.
    fn run(&mut self, x: &[f32], channels: &[f32]) -> Result<Vec<f32>, String> {
        self.model.write(Buf::Audio, x);
        self.model.write(Buf::Stats, channels);
        let frames = conv_frames(x.len())[6];
        let at = &self.at;
        self.model.run(
            x.len(),
            || plan(at, x.len()),
            |_, _| {},
            (Buf::Logits, frames * TOKENS),
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
