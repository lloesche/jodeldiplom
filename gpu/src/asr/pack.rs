//! The packed model laid out for the GPU: its matrices one after the other
//! in one buffer of u32s, its other values (and the encoder's positions
//! and the rotary table) in one of f32s, and where each tensor went,
//! looked up by name once ([`Layout`]).

use std::collections::HashMap;

use jodel_lyrics::asr::packed::{PackedModel, Tensor};
use jodel_lyrics::asr::{
    piece_steps, sinusoids, ENC_DIM, ENC_LAYERS, HEAD, LAYERS, PIECE, ROPE_BASE,
};

/// How a matrix's values are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    /// Groups of 32 at 4 bits, a half-precision scale and offset each.
    Q4,
    /// The same at 8 bits.
    Q8,
    /// Halves.
    F16,
}

impl Format {
    /// The kernels' flag for it ([`super::ops`]).
    pub(super) fn flag(self) -> usize {
        match self {
            Format::Q4 => 0,
            Format::Q8 => 8,
            Format::F16 => 16,
        }
    }

    /// Its bits a value.
    pub(super) fn bits(self) -> usize {
        match self {
            Format::Q4 => 4,
            Format::Q8 => 8,
            Format::F16 => 16,
        }
    }
}

/// A matrix: where it is (u32s into the weights) and how it is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Matrix {
    pub(super) at: usize,
    pub(super) format: Format,
}

/// One encoder layer's tensors (offsets of values in f32s).
pub(super) struct EncLayer {
    pub(super) ln: (usize, usize),
    pub(super) qkv: (Matrix, usize),
    pub(super) o: (Matrix, usize),
    pub(super) fln: (usize, usize),
    pub(super) up: (Matrix, usize),
    pub(super) down: (Matrix, usize),
}

/// One decoder layer's tensors.
pub(super) struct DecLayer {
    pub(super) norm: usize,
    pub(super) qkv: Matrix,
    pub(super) q_norm: usize,
    pub(super) k_norm: usize,
    pub(super) o: Matrix,
    pub(super) post_norm: usize,
    pub(super) gate_up: Matrix,
    pub(super) down: Matrix,
}

/// Where every tensor went. A matrix with its bias: `(matrix, bias)`; a
/// norm with its bias: `(weights, bias)`.
pub(super) struct Layout {
    pub(super) conv1: (usize, usize),
    pub(super) conv2: (Matrix, usize),
    pub(super) conv3: (Matrix, usize),
    pub(super) conv_out: Matrix,
    pub(super) positions: usize,
    pub(super) enc: Vec<EncLayer>,
    pub(super) post: (usize, usize),
    pub(super) proj1: (Matrix, usize),
    pub(super) proj2: (Matrix, usize),
    pub(super) embed: Matrix,
    pub(super) dec: Vec<DecLayer>,
    pub(super) norm: usize,
    pub(super) head: Matrix,
    pub(super) rope: usize,
}

/// The packed model's tensors laid out.
pub(super) struct Packed {
    pub(super) weights: Vec<u32>,
    pub(super) small: Vec<f32>,
    pub(super) layout: Layout,
    /// Each matrix: its name, where it is, its columns.
    pub(super) matrices: Vec<(String, usize, usize)>,
}

/// The packed model's tensors laid out for the GPU, `rope` positions of
/// rotary angles; an error names a tensor the model lacks.
pub(super) fn pack(model: &PackedModel, rope: usize) -> Result<Packed, String> {
    let mut weights = Vec::new();
    let mut small = Vec::new();
    let mut values = HashMap::new();
    let mut held: HashMap<String, Matrix> = HashMap::new();
    let mut matrices = Vec::new();
    for (name, t) in &model.tensors {
        let (data, cols, format) = match t {
            Tensor::Values(v) => {
                values.insert(name.clone(), small.len());
                small.extend_from_slice(v);
                continue;
            }
            Tensor::Q4 { data, cols, .. } => (data, cols, Format::Q4),
            Tensor::Q8 { data, cols, .. } => (data, cols, Format::Q8),
            Tensor::F16 { data, cols, .. } => (data, cols, Format::F16),
        };
        held.insert(
            name.clone(),
            Matrix {
                at: weights.len(),
                format,
            },
        );
        matrices.push((name.clone(), weights.len(), *cols));
        weights.extend_from_slice(data);
    }
    let positions = small.len();
    small.extend(sinusoids(piece_steps(PIECE), ENC_DIM));
    let rope_at = small.len();
    let half = HEAD / 2;
    for pos in 0..rope {
        for i in 0..half {
            let a = pos as f64 / ROPE_BASE.powf(2.0 * i as f64 / HEAD as f64);
            small.push(a.cos() as f32);
            small.push(a.sin() as f32);
        }
    }
    let v = |name: &str| {
        values
            .get(name)
            .copied()
            .ok_or_else(|| format!("the model has no {name}"))
    };
    let m = |name: &str| {
        held.get(name)
            .copied()
            .ok_or_else(|| format!("the model has no {name}"))
    };
    let with_bias =
        |name: &str| Ok::<_, String>((m(&format!("{name}.w"))?, v(&format!("{name}.b"))?));
    let enc = (0..ENC_LAYERS)
        .map(|l| {
            let n = |what: &str| format!("enc{l}.{what}");
            Ok(EncLayer {
                ln: (v(&n("ln.w"))?, v(&n("ln.b"))?),
                qkv: with_bias(&n("qkv"))?,
                o: with_bias(&n("o"))?,
                fln: (v(&n("fln.w"))?, v(&n("fln.b"))?),
                up: with_bias(&n("up"))?,
                down: with_bias(&n("down"))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let dec = (0..LAYERS)
        .map(|l| {
            let n = |what: &str| format!("dec{l}.{what}");
            Ok(DecLayer {
                norm: v(&n("norm"))?,
                qkv: m(&n("qkv.w"))?,
                q_norm: v(&n("q_norm"))?,
                k_norm: v(&n("k_norm"))?,
                o: m(&n("o.w"))?,
                post_norm: v(&n("post_norm"))?,
                gate_up: m(&n("gate_up.w"))?,
                down: m(&n("down.w"))?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let embed = m("embed")?;
    let layout = Layout {
        conv1: (v("conv1.w")?, v("conv1.b")?),
        conv2: with_bias("conv2")?,
        conv3: with_bias("conv3")?,
        conv_out: m("conv_out.w")?,
        positions,
        enc,
        post: (v("post.w")?, v("post.b")?),
        proj1: with_bias("proj1")?,
        proj2: with_bias("proj2")?,
        embed,
        dec,
        norm: v("norm")?,
        // A tied head is the embeddings.
        head: m("head").unwrap_or(embed),
        rope: rope_at,
    };
    Ok(Packed {
        weights,
        small,
        layout,
        matrices,
    })
}
