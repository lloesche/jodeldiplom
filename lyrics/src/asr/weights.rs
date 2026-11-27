//! The recogniser's weights: by name, read from its checkpoint as a GPU's
//! version takes them ([`checkpoint_tensors`]), and in the CPU's model in
//! place, rounded as a quantised copy would hold them
//! ([`Qwen3Asr::round_weights`]).

use jodel_tensors::SafeTensors;

use super::{band_major, quant, Qwen3Asr, BANDS, CONV, DIM, ENC_DIM, ENC_FF, ENC_LAYERS};
use super::{FF, HEAD, HEADS, KV_HEADS, LAYERS, VOCAB};
use crate::nn::{tap_major, Linear};
use jodel_tensors::kernels::transposed;

/// How a tensor of [`checkpoint_tensors`] is read from the checkpoint.
pub(super) enum Read {
    /// These tensors' values one after another (a norm's weights, a layer's
    /// biases stacked), each of its length.
    Values(Vec<(String, usize)>),
    /// These matrices (`[rows][columns]` as the checkpoint has them, a row
    /// an output) stacked by rows, each with its rows; and their columns.
    Rows(Vec<(String, usize)>, usize),
    /// A 3×3 convolution's weight, with its inputs and outputs: its rows
    /// tap-major ([`tap_major`]).
    Taps(String, usize, usize),
    /// The projection of the encoder's flattened steps: its rows band-major
    /// ([`band_major`]).
    Bands(String),
}

/// A tensor's values, and its shape when it is a matrix (`(rows, columns)`).
pub(super) type Shaped = (Vec<f32>, Option<(usize, usize)>);

impl Read {
    /// The tensor from `t`.
    pub(super) fn from(&self, t: &SafeTensors) -> Result<Shaped, String> {
        Ok(match self {
            Read::Values(parts) => {
                let mut values = Vec::new();
                for (name, len) in parts {
                    values.extend(t.get(name, &[*len])?);
                }
                (values, None)
            }
            Read::Rows(parts, columns) => {
                let mut values = Vec::new();
                for (name, rows) in parts {
                    values.extend(t.get(name, &[*rows, *columns])?);
                }
                let rows = values.len() / columns;
                (values, Some((rows, *columns)))
            }
            Read::Taps(name, inputs, outputs) => {
                let raw = t.get(name, &[*outputs, *inputs, 3, 3])?;
                (
                    tap_major(&raw, *inputs, *outputs),
                    Some((*outputs, 9 * inputs)),
                )
            }
            Read::Bands(name) => {
                let raw = t.get(name, &[ENC_DIM, CONV * BANDS])?;
                (band_major(&raw), Some((ENC_DIM, CONV * BANDS)))
            }
        })
    }
}

/// The recogniser's tensors by name, in order, as a GPU's version takes
/// them, and how each is read from the checkpoint. A matrix is in rows, an
/// output each, its inputs along it:
///
/// - `conv1.w` `[480][9]` (3 bands × 3 steps), `conv1.b`;
/// - `conv2.w`, `conv3.w` `[480][9·480]` (tap-major inputs), their `.b`;
/// - `conv_out.w` `[896][16·480]` (band-major inputs);
/// - per encoder layer `l`: `enc{l}.ln.w`, `.b`; `enc{l}.qkv.w`
///   `[3·896][896]` (queries, keys, values), `enc{l}.qkv.b`;
///   `enc{l}.o.w`, `.b`; `enc{l}.fln.w`, `.b`; `enc{l}.up.w` `[3584][896]`,
///   `.b`; `enc{l}.down.w`, `.b`;
/// - `post.w`, `post.b`; `proj1.w`, `.b`; `proj2.w` `[1024][896]`, `.b`;
/// - `embed` and `head` `[151936][1024]`;
/// - per decoder layer `l`: `dec{l}.norm`; `dec{l}.qkv.w` `[4096][1024]`
///   (2048 queries, 1024 keys, 1024 values); `dec{l}.o.w` `[1024][2048]`;
///   `dec{l}.q_norm`, `dec{l}.k_norm` (128); `dec{l}.post_norm`;
///   `dec{l}.gate_up.w` `[2·3072][1024]` (gates, then ups); `dec{l}.down.w`;
/// - `norm`.
pub(super) fn checkpoint_tensors() -> Vec<(String, Read)> {
    let mut t = Table(Vec::new());
    let a = "thinker.audio_tower";
    for (name, from, inputs) in [
        ("conv1", "conv2d1", 1),
        ("conv2", "conv2d2", CONV),
        ("conv3", "conv2d3", CONV),
    ] {
        let from = format!("{a}.{from}");
        t.push(
            format!("{name}.w"),
            Read::Taps(format!("{from}.weight"), inputs, CONV),
        );
        t.values(format!("{name}.b"), format!("{from}.bias"), CONV);
    }
    t.push(
        "conv_out.w".into(),
        Read::Bands(format!("{a}.conv_out.weight")),
    );
    for l in 0..ENC_LAYERS {
        let n = format!("{a}.layers.{l}");
        let part = |p: &str, outputs| (format!("{n}.{p}"), outputs);
        let (q, k, v) = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj");
        let qkv = [part(q, ENC_DIM), part(k, ENC_DIM), part(v, ENC_DIM)];
        t.norm(&format!("enc{l}.ln"), &format!("{n}.self_attn_layer_norm"));
        t.linear(&format!("enc{l}.qkv"), &qkv, ENC_DIM, true);
        t.linear(
            &format!("enc{l}.o"),
            &[part("self_attn.out_proj", ENC_DIM)],
            ENC_DIM,
            true,
        );
        t.norm(&format!("enc{l}.fln"), &format!("{n}.final_layer_norm"));
        t.linear(&format!("enc{l}.up"), &[part("fc1", ENC_FF)], ENC_DIM, true);
        t.linear(
            &format!("enc{l}.down"),
            &[part("fc2", ENC_DIM)],
            ENC_FF,
            true,
        );
    }
    t.norm("post", &format!("{a}.ln_post"));
    t.linear("proj1", &[(format!("{a}.proj1"), ENC_DIM)], ENC_DIM, true);
    t.linear("proj2", &[(format!("{a}.proj2"), DIM)], ENC_DIM, true);
    let m = "thinker.model";
    let table = |name: String| Read::Rows(vec![(name, VOCAB)], DIM);
    t.push("embed".into(), table(format!("{m}.embed_tokens.weight")));
    t.push("head".into(), table("thinker.lm_head.weight".into()));
    for l in 0..LAYERS {
        let n = format!("{m}.layers.{l}");
        let part = |p: &str, outputs| (format!("{n}.{p}"), outputs);
        let weight = |p: &str| format!("{n}.{p}.weight");
        let (q, k, v) = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj");
        let qkv = [
            part(q, HEADS * HEAD),
            part(k, KV_HEADS * HEAD),
            part(v, KV_HEADS * HEAD),
        ];
        let gate_up = [part("mlp.gate_proj", FF), part("mlp.up_proj", FF)];
        t.values(format!("dec{l}.norm"), weight("input_layernorm"), DIM);
        t.linear(&format!("dec{l}.qkv"), &qkv, DIM, false);
        t.linear(
            &format!("dec{l}.o"),
            &[part("self_attn.o_proj", DIM)],
            HEADS * HEAD,
            false,
        );
        t.values(format!("dec{l}.q_norm"), weight("self_attn.q_norm"), HEAD);
        t.values(format!("dec{l}.k_norm"), weight("self_attn.k_norm"), HEAD);
        t.values(
            format!("dec{l}.post_norm"),
            weight("post_attention_layernorm"),
            DIM,
        );
        t.linear(&format!("dec{l}.gate_up"), &gate_up, DIM, false);
        t.linear(
            &format!("dec{l}.down"),
            &[part("mlp.down_proj", DIM)],
            FF,
            false,
        );
    }
    t.values("norm".into(), format!("{m}.norm.weight"), DIM);
    t.0
}

/// The tensors of [`checkpoint_tensors`] as they are listed.
struct Table(Vec<(String, Read)>);

impl Table {
    fn push(&mut self, name: String, read: Read) {
        self.0.push((name, read));
    }

    /// The tensor `from`'s values, `len` of them.
    fn values(&mut self, name: String, from: String, len: usize) {
        self.push(name, Read::Values(vec![(from, len)]));
    }

    /// A LayerNorm of the encoder's width: `{from}.weight` and `.bias`.
    fn norm(&mut self, name: &str, from: &str) {
        self.values(format!("{name}.w"), format!("{from}.weight"), ENC_DIM);
        self.values(format!("{name}.b"), format!("{from}.bias"), ENC_DIM);
    }

    /// Linear layers of `inputs` stacked: each part's `{part}.weight` with
    /// its outputs, and its `{part}.bias` when `biased`.
    fn linear(&mut self, name: &str, parts: &[(String, usize)], inputs: usize, biased: bool) {
        let of = |what: &str| {
            parts
                .iter()
                .map(|(p, o)| (format!("{p}.{what}"), *o))
                .collect()
        };
        self.push(format!("{name}.w"), Read::Rows(of("weight"), inputs));
        if biased {
            self.push(format!("{name}.b"), Read::Values(of("bias")));
        }
    }
}

impl Qwen3Asr {
    /// Rounds the large weights (every linear layer, the convolutions after
    /// the first, the embeddings and the head) as a quantised copy would
    /// hold them ([`quant::round_groups`]): groups of `group` along each
    /// output's inputs, `bits` a weight, a scale a group, and a minimum too
    /// unless `symmetric`. For measuring what quantising costs.
    pub fn round_weights(&mut self, bits: u32, group: usize, symmetric: bool) {
        // A matrix of `[input][output]`, as the layers compute with them,
        // rounded along its outputs' rows.
        let round_input_major = |w: &mut [f32], inputs: usize, outputs: usize| {
            let mut rows = transposed(w, inputs, outputs);
            quant::round_groups(&mut rows, bits, group, symmetric);
            w.copy_from_slice(&transposed(&rows, outputs, inputs));
        };
        let mut linears: Vec<&mut Linear> =
            vec![&mut self.conv_out, &mut self.proj1, &mut self.proj2];
        for l in &mut self.encoder {
            linears.extend([
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.up,
                &mut l.down,
            ]);
        }
        for l in &mut self.decoder {
            linears.extend([
                &mut l.q,
                &mut l.k,
                &mut l.v,
                &mut l.o,
                &mut l.gate,
                &mut l.up,
                &mut l.down,
            ]);
        }
        for l in linears {
            round_input_major(&mut l.weight, l.inputs, l.outputs);
        }
        for c in [&mut self.conv2, &mut self.conv3] {
            round_input_major(&mut c.weight, 9 * c.inputs, c.outputs);
        }
        // `[output][input]`: a row an output already.
        for w in [&mut self.embed, &mut self.head] {
            quant::round_groups(w, bits, group, symmetric);
        }
    }
}
