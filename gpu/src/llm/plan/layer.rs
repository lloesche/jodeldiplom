//! A plan's layers: a layer's dispatches for a prompt's rows or a step's
//! row ([`Plan::layer`]), and from them a step ([`Plan::step`]) and a
//! prompt's chunk ([`Plan::chunk`]).

use super::super::{head_of, shares, Buf, DIM, HEADS, KV_HEADS, LAYERS, OWN_KV, VOCAB};
use super::{Op, Plan, CHUNK, NONE};
use crate::engine::After;

impl Plan<'_> {
    /// Layer `l` for `rows` rows from position `first` (a prompt's, through
    /// flash attention), or for one at the decoder's (`None`, through the
    /// step's attention); its rows in `X` from row `x_row` (and the
    /// per-layer projection's from that row); the rows end written for the
    /// next layer's matrices (the head's after the last) where `next`.
    fn layer(
        &self,
        ops: &mut Vec<Op>,
        l: usize,
        (rows, x_row): (usize, usize),
        first: Option<usize>,
        next: bool,
    ) {
        let layer = &self.layout.layers[l];
        let w = Buf::Layers(layer.group);
        let kv = if shares(l) { 0 } else { KV_HEADS };
        let qkv = (HEADS + 2 * kv) * head_of(l);
        let qa = self.sizes.qa;
        for (m, row) in self.qkv(l) {
            self.matmul(ops, (rows, qa), (m, w), (Buf::H, row, qkv), After::Store);
        }
        self.prep(ops, l, rows, first, qkv);
        match first {
            None => self.attend_step(ops, l),
            Some(first) => {
                self.flash(ops, l, rows, first);
                self.quant(ops, rows, layer.o, None);
            }
        }
        let o = (layer.placed(layer.o), w);
        self.matmul(ops, (rows, qa), o, (Buf::H, 0, DIM), After::Store);
        self.residual(
            ops,
            (rows, x_row),
            Some(layer.post_attn),
            None,
            Some((layer.pre_ff, layer.gate_up)),
        );
        let gate_up = (layer.placed(layer.gate_up), w);
        let from = self.rows_for(ops, rows, gate_up, After::GeGlu, (layer.down, None));
        let down = (layer.placed(layer.down), w);
        self.matmul(ops, (rows, from), down, (Buf::H, 0, DIM), After::Store);
        self.residual(
            ops,
            (rows, x_row),
            Some(layer.post_ff),
            None,
            Some((NONE, layer.gate)),
        );
        self.per_layer(ops, l, (rows, x_row), first);
        let gate = (layer.placed(layer.gate), w);
        let times = Some(self.sizes.pli);
        let from = self.rows_for(ops, rows, gate, After::Gelu, (layer.proj, times));
        let proj = (layer.placed(layer.proj), w);
        self.matmul(ops, (rows, from), proj, (Buf::H, 0, DIM), After::Store);
        let out = next.then(|| {
            self.layout
                .layers
                .get(l + 1)
                .map_or((self.layout.final_norm, self.layout.embed), |n| {
                    (n.input_norm, n.q)
                })
        });
        self.residual(
            ops,
            (rows, x_row),
            Some(layer.post_ple),
            Some(layer.scalar),
            out,
        );
    }

    /// The layers from `from` for the row `x_row` of `X` at the decoder's
    /// position (written for `from`'s matrices already), then the head and
    /// the next token.
    fn last_row(&self, ops: &mut Vec<Op>, from: usize, x_row: usize) {
        for l in from..LAYERS {
            self.layer(ops, l, (1, x_row), None, true);
        }
        self.matmul(
            ops,
            (1, self.sizes.qa),
            (self.layout.embed, Buf::Shared),
            (Buf::Logits, 0, VOCAB),
            After::Argmax { stride: 0 },
        );
        self.argmax(ops);
    }

    /// A step: the token at the decoder's position through every layer,
    /// the next one chosen and put after it, the position moved on.
    pub(in crate::llm) fn step(&self) -> Vec<Op> {
        let mut ops = Vec::new();
        self.embed(&mut ops, 1, None);
        self.last_row(&mut ops, 0, 0);
        ops
    }

    /// A prompt's chunk: `rows` rows from position `first` through the
    /// layers with keys and values of their own; where it is the `last`,
    /// its last row through the rest, and the first token chosen (the
    /// decoder's position must be that row's).
    pub(in crate::llm) fn chunk(&self, first: usize, rows: usize, last: bool) -> Vec<Op> {
        assert!((1..=CHUNK).contains(&rows), "{rows} rows in a chunk");
        let mut ops = Vec::new();
        self.embed(&mut ops, rows, Some(first));
        for l in 0..OWN_KV {
            self.layer(&mut ops, l, (rows, 0), Some(first), l + 1 < OWN_KV);
        }
        if last {
            let next = &self.layout.layers[OWN_KV];
            self.residual(
                &mut ops,
                (1, rows - 1),
                None,
                None,
                Some((next.input_norm, next.q)),
            );
            self.last_row(&mut ops, OWN_KV, rows - 1);
        }
        ops
    }
}
