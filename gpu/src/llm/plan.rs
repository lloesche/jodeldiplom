//! The decoder's dispatches as they are planned: a prompt's rows a chunk
//! at a time ([`Plan::chunk`]), and a step ([`Plan::step`]), whose
//! dispatches never change (every kernel of a step reads the position it
//! decodes at from the GPU, and the last moves it on), so a step is
//! recorded once and run again and again.
//!
//! A prompt's chunk runs the layers that have keys and values of their own
//! over all its rows, through the tiled matrix multiplies and flash
//! attention; the layers after them only need the prompt's last row (they
//! write no keys or values), which goes through them as a step does.
//!
//! Every matrix multiply is the engine's [`Mul`]: from the rows written
//! for it in `ints` (at 8 bits or as f32s, [`Precision`]), through a 4-bit
//! matrix, to its products as [`After`] says.

use super::pack::{Layout, Matrix};
use super::{
    head_of, is_global, kv_of, shares, Buf, Kernel, DIM, EMBED_SCALE, FF, HEADS, KV_HEADS, LAYERS,
    OWN_KV, PER_LAYER, PER_LAYERS, PER_LAYER_EMBED_SCALE, PER_LAYER_PROJECTION_SCALE, VOCAB,
    WINDOW,
};
use crate::engine::{self, params, spread, After, Mul};

/// One dispatch: its kernel, its buffers (src, dst and wq: aux is always
/// the small values, `ints` the ints), its numbers, its workgroups.
pub(super) type Op = engine::Op<Kernel, Bind, 16>;

/// The buffers a dispatch binds besides the small values and the ints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Bind {
    pub(super) src: Buf,
    pub(super) dst: Buf,
    pub(super) wq: Buf,
}

/// The rows of a prompt a chunk takes at most.
pub(super) const CHUNK: usize = 512;
/// The splits of a step's attention (`attend`, each taking every so
/// many chunks of 64 positions): a sliding layer's window is 8 chunks.
const LOCAL_SPLITS: usize = 8;
const GLOBAL_SPLITS: usize = 16;
/// The head's workgroups (16 rows each).
const HEAD_GROUPS: usize = VOCAB / 16;
/// The most positions of a context.
pub(super) const MOST_POSITIONS: usize = 16_384;
/// The words before the tokens in `ints` (the state).
pub(super) const TOKENS: usize = 16;
/// No norm (`residual`), no other rows (`quant`).
const NONE: usize = u32::MAX as usize;
/// `residual`'s flags.
const ADD: usize = 1;
const SCALE: usize = 2;
const OUT: usize = 4;

/// How a matrix multiply takes its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// At 8 bits, a block of 32 a scale, multiplied four at a time with
    /// `dot4I8Packed` (where the GPU has it natively).
    Q8,
    /// As f32s (where `dot4I8Packed` would be emulated, as on Metal).
    Plain,
}

impl Precision {
    /// The producers' mode number.
    fn mode(self) -> usize {
        match self {
            Precision::Q8 => 1,
            Precision::Plain => 0,
        }
    }
}

/// A 4-bit matrix's place for the engine (one format: no flag).
impl engine::Matrix for Matrix {
    fn at(self) -> usize {
        self.at
    }

    fn flag(self) -> usize {
        0
    }
}

/// The sizes of the buffers that depend on the context: f32s (or u32s)
/// each, and where things are in them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sizes {
    /// The cache's positions.
    pub(super) positions: usize,
    pub(super) cache: usize,
    /// Where the queries of the rows under way are in the cache.
    pub(super) queries: usize,
    pub(super) rope: usize,
    pub(super) x: usize,
    /// A matrix multiply's products, and the per-layer inputs after them
    /// (at `pli`).
    pub(super) h: usize,
    pub(super) pli: usize,
    pub(super) per_layer: usize,
    pub(super) parts: usize,
    pub(super) ints: usize,
    /// Where in `ints` a matrix multiply's rows and their scales are, and
    /// the head's workgroups' largest.
    pub(super) qa: usize,
    pub(super) qs: usize,
    pub(super) best: usize,
}

/// The words a layer with its own keys and values keeps a position.
fn kv_words(l: usize) -> usize {
    2 * KV_HEADS * head_of(l)
}

impl Sizes {
    /// The sizes for a cache of `positions`.
    pub(super) fn new(positions: usize) -> Sizes {
        let cache: usize = (0..OWN_KV).map(|l| kv_words(l) * positions).sum();
        let qa = TOKENS + positions;
        let qs = qa + CHUNK * FF;
        let best = qs + CHUNK * FF / 32 * 2;
        Sizes {
            positions,
            cache: cache + CHUNK * HEADS * head_of(5),
            queries: cache,
            rope: positions * (head_of(0) / 2 + head_of(5) / 2) * 2,
            x: CHUNK * DIM,
            h: CHUNK * (FF + PER_LAYER),
            pli: CHUNK * FF,
            per_layer: CHUNK * PER_LAYERS,
            parts: HEADS * GLOBAL_SPLITS * (head_of(5) + 2),
            ints: best + HEAD_GROUPS * 2,
            qa,
            qs,
            best,
        }
    }

    /// Every buffer and its words (the vocabulary's logits too).
    pub(super) fn buffers(&self) -> [(Buf, usize); 9] {
        [
            (Buf::Cache, self.cache),
            (Buf::Rope, self.rope),
            (Buf::X, self.x),
            (Buf::H, self.h),
            (Buf::PerLayer, self.per_layer),
            (Buf::Parts, self.parts),
            (Buf::Logits, VOCAB),
            (Buf::Ints, self.ints),
            (Buf::Spare, 1),
        ]
    }

    /// Where layer `l`'s keys and values are in the cache (a layer that
    /// shares another's: that one's).
    fn kv(&self, l: usize) -> (usize, usize) {
        let l = kv_of(l);
        let at: usize = (0..l).map(|e| kv_words(e) * self.positions).sum();
        (at, at + kv_words(l) / 2 * self.positions)
    }

    /// Where the rotary table of a layer's kind is.
    fn rope_at(&self, l: usize) -> usize {
        if is_global(l) {
            self.positions * head_of(0)
        } else {
            0
        }
    }

    /// The rotary tables: for each position, a layer kind's cosine and sine
    /// of each pair (a sliding layer's 128 pairs, then a global layer's
    /// 256, of which the first 64 turn), as f32 bits.
    pub(super) fn rope(&self) -> Vec<u32> {
        let mut table = Vec::with_capacity(self.rope);
        for global in [false, true] {
            let d = head_of(if global { 5 } else { 0 });
            let (base, turned) = if global {
                (super::GLOBAL_BASE, super::GLOBAL_TURNED)
            } else {
                (super::LOCAL_BASE, d / 2)
            };
            // As the reference computes them: the frequencies in f32, the
            // angles a position times them in f32.
            let freq: Vec<f32> = (0..d / 2)
                .map(|i| {
                    if i < turned {
                        1.0 / base.powf((2 * i) as f32 / d as f32)
                    } else {
                        0.0
                    }
                })
                .collect();
            for pos in 0..self.positions {
                for &f in &freq {
                    let angle = f64::from(pos as f32 * f);
                    table.push((angle.cos() as f32).to_bits());
                    table.push((angle.sin() as f32).to_bits());
                }
            }
        }
        table
    }
}

/// What a plan needs: where the weights are, the context's sizes, how
/// rows are multiplied.
pub(super) struct Plan<'a> {
    pub(super) layout: &'a Layout,
    pub(super) sizes: Sizes,
    pub(super) precision: Precision,
}

impl Plan<'_> {
    /// `rows` rows written for a matrix multiply through `m` (in `wq`) to
    /// `dst` (from `at`, rows `ld` apart), as `after` says: a GEMV for a
    /// row, else the tiled GEMM.
    fn matmul(
        &self,
        ops: &mut Vec<Op>,
        rows: usize,
        (m, wq): (Matrix, Buf),
        (dst, at, ld): (Buf, usize, usize),
        after: After,
    ) {
        let mul = Mul {
            from: (Buf::Ints, self.sizes.qa, m.cols),
            to: (dst, at, ld),
            m: rows,
            n: m.rows,
            k: m.cols,
            w: m,
            bias: None,
            after,
        };
        let mut numbers = mul.numbers().to_vec();
        numbers.extend([self.sizes.qs, self.sizes.best]);
        let bind = Bind {
            src: Buf::Small,
            dst,
            wq,
        };
        let q8 = self.precision == Precision::Q8;
        if rows == 1 {
            assert_eq!(m.rows % 16, 0, "a GEMV takes rows in sixteens");
            ops.push(Op {
                kernel: if q8 { Kernel::GemvQ8 } else { Kernel::GemvF32 },
                buffers: bind,
                params: params(&numbers),
                groups: [(m.rows / 16) as u32, 1, 1],
            });
        } else {
            assert_eq!(m.rows % 64, 0, "a GEMM takes columns in tiles of 64");
            assert!(
                matches!(after, After::Store | After::Gelu | After::GeGlu),
                "a GEMM writes, takes GELU or gates"
            );
            ops.push(Op {
                kernel: if q8 { Kernel::GemmQ8 } else { Kernel::GemmF32 },
                buffers: bind,
                params: params(&numbers),
                groups: [(m.rows / 64) as u32, rows.div_ceil(64) as u32, 1],
            });
        }
    }

    /// `rows` rows of `width` values of `H` (from the start) written for a
    /// matrix multiply, each times `H`'s at the same place of the rows at
    /// `times` (`width` apart) where that is given.
    fn quant(&self, ops: &mut Vec<Op>, rows: usize, width: usize, times: Option<usize>) {
        ops.push(Op {
            kernel: Kernel::Quant,
            buffers: Bind {
                src: Buf::H,
                dst: Buf::Spare,
                wq: Buf::Rope,
            },
            params: params(&[
                rows,
                width,
                0,
                width,
                self.sizes.qa,
                self.sizes.qs,
                self.precision.mode(),
                times.unwrap_or(NONE),
                width,
            ]),
            groups: spread(rows * width / 32),
        });
    }

    /// The residual stream's step for `rows` rows of `X` from row `x_row`:
    /// `H`'s rows RMS-normed by `add` and added, the rows scaled by
    /// `scale`, then written for a matrix multiply where `out` (normed by
    /// it, or as they are where it is `NONE`).
    fn residual(
        &self,
        ops: &mut Vec<Op>,
        (rows, x_row): (usize, usize),
        add: Option<usize>,
        scale: Option<usize>,
        out: Option<usize>,
    ) {
        let flags = add.map_or(0, |_| ADD) | scale.map_or(0, |_| SCALE) | out.map_or(0, |_| OUT);
        ops.push(Op {
            kernel: Kernel::Residual,
            buffers: Bind {
                src: Buf::H,
                dst: Buf::X,
                wq: Buf::Rope,
            },
            params: params(&[
                DIM,
                0,
                0,
                DIM,
                add.unwrap_or(0),
                scale.unwrap_or(0),
                out.unwrap_or(NONE),
                self.sizes.qa,
                self.sizes.qs,
                self.precision.mode(),
                flags,
                x_row,
            ]),
            groups: [rows as u32, 1, 1],
        });
    }

    /// The embeddings of `rows` tokens into `X`, written for the per-layer
    /// projection, through it, then normed for layer 0: from position
    /// `first`, or the decoder's (`None`).
    fn embed(&self, ops: &mut Vec<Op>, rows: usize, first: Option<usize>) {
        let (stated, first) = (first.is_none(), first.unwrap_or(0));
        ops.push(Op {
            kernel: Kernel::Embed,
            buffers: Bind {
                src: Buf::Small,
                dst: Buf::X,
                wq: Buf::Shared,
            },
            params: params(&[
                rows,
                first,
                usize::from(stated),
                0,
                self.layout.embed.at,
                EMBED_SCALE.to_bits() as usize,
                DIM,
                VOCAB,
            ]),
            groups: spread(rows * DIM),
        });
        self.residual(ops, (rows, 0), None, None, Some(NONE));
        self.matmul(
            ops,
            rows,
            (self.layout.per_layer, Buf::Shared),
            (Buf::PerLayer, 0, PER_LAYERS),
            After::Store,
        );
        self.residual(
            ops,
            (rows, 0),
            None,
            None,
            Some(self.layout.layers[0].input_norm),
        );
    }

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
        let d = head_of(l);
        let (stated, first_pos) = (first.is_none(), first.unwrap_or(0));
        let kv = if shares(l) { 0 } else { KV_HEADS };
        let qkv = layer.qkv.rows;
        let window = if is_global(l) { 0 } else { WINDOW };
        let (k_at, v_at) = self.sizes.kv(l);
        let positions = self.sizes.positions;
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.qkv), w),
            (Buf::H, 0, qkv),
            After::Store,
        );
        ops.push(Op {
            kernel: Kernel::Prep,
            buffers: Bind {
                src: Buf::H,
                dst: Buf::Cache,
                wq: Buf::Rope,
            },
            params: params(&[
                rows,
                first_pos,
                usize::from(stated),
                d,
                0,
                qkv,
                layer.q_norm,
                layer.k_norm.unwrap_or(0),
                self.sizes.queries,
                k_at,
                v_at,
                positions,
                self.sizes.rope_at(l),
                kv,
                HEADS,
            ]),
            groups: [rows as u32, (HEADS + 2 * kv) as u32, 1],
        });
        if stated {
            let splits = if window > 0 {
                LOCAL_SPLITS
            } else {
                GLOBAL_SPLITS
            };
            ops.push(Op {
                kernel: Kernel::Attend,
                buffers: Bind {
                    src: Buf::Cache,
                    dst: Buf::Parts,
                    wq: Buf::Rope,
                },
                params: params(&[
                    d,
                    window,
                    self.sizes.queries,
                    k_at,
                    v_at,
                    positions,
                    0,
                    splits,
                ]),
                groups: [splits as u32, KV_HEADS as u32, 1],
            });
            ops.push(Op {
                kernel: Kernel::Join,
                buffers: Bind {
                    src: Buf::Parts,
                    dst: Buf::Spare,
                    wq: Buf::Rope,
                },
                params: params(&[
                    d,
                    splits,
                    0,
                    self.sizes.qa,
                    self.sizes.qs,
                    self.precision.mode(),
                    HEADS,
                ]),
                groups: [1, 1, 1],
            });
        } else {
            let per_block = 256 / (d / 16);
            ops.push(Op {
                kernel: Kernel::Flash,
                buffers: Bind {
                    src: Buf::Cache,
                    dst: Buf::H,
                    wq: Buf::Rope,
                },
                params: params(&[
                    rows,
                    first_pos,
                    d,
                    window,
                    self.sizes.queries,
                    k_at,
                    v_at,
                    positions,
                    0,
                    HEADS,
                    HEADS / KV_HEADS,
                ]),
                groups: [rows.div_ceil(per_block) as u32, HEADS as u32, 1],
            });
            self.quant(ops, rows, HEADS * d, None);
        }
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.o), w),
            (Buf::H, 0, DIM),
            After::Store,
        );
        self.residual(
            ops,
            (rows, x_row),
            Some(layer.post_attn),
            None,
            Some(layer.pre_ff),
        );
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.gate_up), w),
            (Buf::H, 0, FF),
            After::GeGlu,
        );
        self.quant(ops, rows, FF, None);
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.down), w),
            (Buf::H, 0, DIM),
            After::Store,
        );
        self.residual(ops, (rows, x_row), Some(layer.post_ff), None, Some(NONE));
        ops.push(Op {
            kernel: Kernel::PerLayer,
            buffers: Bind {
                src: Buf::PerLayer,
                dst: Buf::H,
                wq: w,
            },
            params: params(&[
                rows,
                first_pos,
                usize::from(stated),
                0,
                PER_LAYERS,
                l * PER_LAYER,
                self.layout.per_layer_norm,
                layer.placed(layer.ple).at,
                self.sizes.pli,
                PER_LAYER,
                x_row,
                PER_LAYER_PROJECTION_SCALE.to_bits() as usize,
                PER_LAYER_EMBED_SCALE.to_bits() as usize,
                VOCAB,
            ]),
            groups: [rows as u32, 1, 1],
        });
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.gate), w),
            (Buf::H, 0, PER_LAYER),
            After::Gelu,
        );
        self.quant(ops, rows, PER_LAYER, Some(self.sizes.pli));
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.proj), w),
            (Buf::H, 0, DIM),
            After::Store,
        );
        let out = next.then(|| {
            self.layout
                .layers
                .get(l + 1)
                .map_or(self.layout.final_norm, |n| n.input_norm)
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
            1,
            (self.layout.embed, Buf::Shared),
            (Buf::Logits, 0, VOCAB),
            After::Argmax { stride: 0 },
        );
        ops.push(Op {
            kernel: Kernel::Argmax,
            buffers: Bind {
                src: Buf::Small,
                dst: Buf::Spare,
                wq: Buf::Rope,
            },
            params: params(&[HEAD_GROUPS, self.sizes.best, self.sizes.positions]),
            groups: [1, 1, 1],
        });
    }

    /// A step: the token at the decoder's position through every layer,
    /// the next one chosen and put after it, the position moved on.
    pub(super) fn step(&self) -> Vec<Op> {
        let mut ops = Vec::new();
        self.embed(&mut ops, 1, None);
        self.last_row(&mut ops, 0, 0);
        ops
    }

    /// A prompt's chunk: `rows` rows from position `first` through the
    /// layers with keys and values of their own; where it is the `last`,
    /// its last row through the rest, and the first token chosen (the
    /// decoder's position must be that row's).
    pub(super) fn chunk(&self, first: usize, rows: usize, last: bool) -> Vec<Op> {
        assert!((1..=CHUNK).contains(&rows), "{rows} rows in a chunk");
        let mut ops = Vec::new();
        self.embed(&mut ops, rows, Some(first));
        for l in 0..OWN_KV {
            self.layer(&mut ops, l, (rows, 0), Some(first), l + 1 < OWN_KV);
        }
        if last {
            self.residual(
                &mut ops,
                (1, rows - 1),
                None,
                None,
                Some(self.layout.layers[OWN_KV].input_norm),
            );
            self.last_row(&mut ops, OWN_KV, rows - 1);
        }
        ops
    }
}
