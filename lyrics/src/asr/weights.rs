//! The recogniser's weights handed out: by name, as a GPU's version takes
//! them ([`Qwen3Asr::visit`]), and in place, rounded as a quantised copy
//! would hold them ([`Qwen3Asr::round_weights`]).

use super::{quant, Qwen3Asr, Visitor, DIM, VOCAB};
use crate::nn::{transposed, Conv, LayerNorm, Linear};

/// How a matrix's values lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// `[input][output]`, as the layers compute with them.
    InputMajor,
    /// `[output][input]`: a row an output.
    OutputMajor,
}

/// What [`Qwen3Asr::visit_matrices_mut`] hands each matrix to: its values,
/// its inputs and outputs, and how they lie.
type MatrixVisitor<'a> = dyn FnMut(&mut [f32], usize, usize, Layout) + 'a;

impl Qwen3Asr {
    /// Hands each tensor to `f` by name, with its shape when it is a matrix
    /// (`(rows, columns)`: a row an output, its inputs along it), for a
    /// GPU's version:
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
    pub fn visit(&self, f: &mut Visitor) {
        /// Stacked linear layers' weights as rows, and their biases.
        fn rows(parts: &[&Linear]) -> (Vec<f32>, usize, usize, Vec<f32>) {
            let inputs = parts[0].inputs;
            let outputs: usize = parts.iter().map(|l| l.outputs).sum();
            let mut w = Vec::with_capacity(inputs * outputs);
            let mut b = Vec::new();
            for l in parts {
                w.extend(transposed(&l.weight, inputs, l.outputs));
                b.extend(l.bias.clone().unwrap_or_default());
            }
            (w, outputs, inputs, b)
        }
        let linear = |f: &mut Visitor, name: &str, parts: &[&Linear]| {
            let (w, o, i, b) = rows(parts);
            f(&format!("{name}.w"), &w, Some((o, i)));
            if !b.is_empty() {
                f(&format!("{name}.b"), &b, None);
            }
        };
        let conv = |f: &mut Visitor, name: &str, c: &Conv| {
            let (width, outputs) = (9 * c.inputs, c.outputs);
            let w = transposed(&c.weight, width, outputs);
            f(&format!("{name}.w"), &w, Some((outputs, width)));
            f(&format!("{name}.b"), &c.bias, None);
        };
        conv(f, "conv1", &self.conv1);
        conv(f, "conv2", &self.conv2);
        conv(f, "conv3", &self.conv3);
        linear(f, "conv_out", &[&self.conv_out]);
        let norm = |f: &mut Visitor, name: &str, n: &LayerNorm| {
            f(&format!("{name}.w"), &n.weight, None);
            f(&format!("{name}.b"), &n.bias, None);
        };
        for (i, l) in self.encoder.iter().enumerate() {
            norm(f, &format!("enc{i}.ln"), &l.norm);
            linear(f, &format!("enc{i}.qkv"), &[&l.q, &l.k, &l.v]);
            linear(f, &format!("enc{i}.o"), &[&l.o]);
            norm(f, &format!("enc{i}.fln"), &l.final_norm);
            linear(f, &format!("enc{i}.up"), &[&l.up]);
            linear(f, &format!("enc{i}.down"), &[&l.down]);
        }
        norm(f, "post", &self.post_norm);
        linear(f, "proj1", &[&self.proj1]);
        linear(f, "proj2", &[&self.proj2]);
        f("embed", &self.embed, Some((VOCAB, DIM)));
        f("head", &self.head, Some((VOCAB, DIM)));
        for (i, l) in self.decoder.iter().enumerate() {
            f(&format!("dec{i}.norm"), &l.norm, None);
            linear(f, &format!("dec{i}.qkv"), &[&l.q, &l.k, &l.v]);
            linear(f, &format!("dec{i}.o"), &[&l.o]);
            f(&format!("dec{i}.q_norm"), &l.q_norm, None);
            f(&format!("dec{i}.k_norm"), &l.k_norm, None);
            f(&format!("dec{i}.post_norm"), &l.post_norm, None);
            linear(f, &format!("dec{i}.gate_up"), &[&l.gate, &l.up]);
            linear(f, &format!("dec{i}.down"), &[&l.down]);
        }
        f("norm", &self.norm, None);
    }

    /// Hands each large weight matrix to `f` to change in place: every
    /// linear layer, the convolutions after the first, the embeddings and
    /// the head; with its inputs, its outputs and how its values lie.
    fn visit_matrices_mut(&mut self, f: &mut MatrixVisitor) {
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
            f(&mut l.weight, l.inputs, l.outputs, Layout::InputMajor);
        }
        for c in [&mut self.conv2, &mut self.conv3] {
            f(&mut c.weight, 9 * c.inputs, c.outputs, Layout::InputMajor);
        }
        f(&mut self.embed, DIM, VOCAB, Layout::OutputMajor);
        f(&mut self.head, DIM, VOCAB, Layout::OutputMajor);
    }

    /// Rounds the large weights (every linear layer, the convolutions after
    /// the first, the embeddings and the head) as a quantised copy would
    /// hold them ([`quant::round_groups`]): groups of `group` along each
    /// output's inputs, `bits` a weight, a scale a group, and a minimum too
    /// unless `symmetric`. For measuring what quantising costs.
    pub fn round_weights(&mut self, bits: u32, group: usize, symmetric: bool) {
        self.visit_matrices_mut(&mut |w, inputs, outputs, layout| match layout {
            Layout::OutputMajor => quant::round_groups(w, bits, group, symmetric),
            Layout::InputMajor => {
                let mut rows = transposed(w, inputs, outputs);
                quant::round_groups(&mut rows, bits, group, symmetric);
                w.copy_from_slice(&transposed(&rows, outputs, inputs));
            }
        });
    }
}
