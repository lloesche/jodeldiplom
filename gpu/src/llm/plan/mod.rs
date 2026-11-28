//! The decoder's dispatches as they are planned: a prompt's rows a chunk
//! at a time ([`Plan::chunk`]), and a step ([`Plan::step`]), whose
//! dispatches never change (every kernel of a step reads the position it
//! decodes at from the GPU, and the last moves it on), so a step is
//! recorded once and run again and again; and the room a context takes
//! ([`ContextPlan`]: what [`super::Gemma::planned`] says and
//! [`super::Gemma::reserve`] makes).
//!
//! A prompt's chunk runs the layers that have keys and values of their own
//! over all its rows, through the tiled matrix multiplies and flash
//! attention; the layers after them only need the prompt's last row (they
//! write no keys or values), which goes through them as a step does. The
//! layers, the steps and the chunks are planned in [`layer`]; their
//! dispatches' helpers are here.
//!
//! Every matrix multiply is the engine's [`Mul`]: from the rows written
//! for it in `ints` (at 8 bits or as f32s: [`Plan::mode`]), through a
//! matrix as the file holds it, to its products as [`After`] says. Each
//! kernel's numbers are written by one helper here, by name
//! ([`super::kernels::named`]: the kernels read them by the same names,
//! `prep_rows()`).
//!
//! The room a context takes is in [`sizes`], what each dispatch costs and
//! how a prompt's attention is cut in [`cost`].

mod cost;
mod layer;
mod sizes;

use super::kernels::{
    fields, named, Kernel, ADD, FLASH_LANES, GEMM_TILE, GEMV_ROWS, MOST_SPLITS, NONE, OUT, Q8,
    Q8_BLOCK, Q8_KINDS, SCALE,
};
use super::model::{Layout, Matrix};
use super::{
    head_of, is_global, shares, Buf, DIM, EMBED_SCALE, FF, HEADS, KV_HEADS, PER_LAYER, PER_LAYERS,
    PER_LAYER_EMBED_SCALE, PER_LAYER_PROJECTION_SCALE, WINDOW,
};
use crate::engine::{self, params, spread, After, Mul};
use crate::pace::{Cost, PIECE};
use cost::{flash_spans, span_flops, Rates, M4_MAX_F32, RTX_5080_Q8};
use sizes::HEAD_GROUPS;
pub(super) use sizes::{ContextPlan, Sizes, STAGING_WORDS};
pub use sizes::{LEAST_CONTEXT, MOST_CONTEXT};

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

impl Bind {
    /// `src` read and `dst` written, no weights (the rotary tables bound
    /// in their place, unread).
    fn rows(src: Buf, dst: Buf) -> Bind {
        Bind {
            src,
            dst,
            wq: Buf::Rope,
        }
    }
}

/// The rows of a prompt a chunk takes at most.
pub(super) const CHUNK: usize = 512;

/// The splits of a step's attention (`attend`, taking chunks of 64
/// positions in turn): a sliding layer's window is 8 chunks, a split each.
/// A global layer's are 16, and one more for every so many chunks
/// ([`super::Shape::per`]) where that is more as its positions grow, up
/// to [`MOST_SPLITS`] (the workgroups dispatched): the long contexts'
/// chunks spread over the GPU.
const LOCAL_SPLITS: usize = 8;
const GLOBAL_SPLITS: usize = 16;
// The splits at work never more than those dispatched (WGSL's `clamp` is
// left open where its low bound is over its high one).
const _: () = assert!(GLOBAL_SPLITS <= MOST_SPLITS && LOCAL_SPLITS <= MOST_SPLITS);
/// A global layer's chunks for one more split where the plans are kept
/// token for token as they were (Vulkan): 18, so up to the 18 432
/// positions a context had at most before its splits are the 16 they
/// were. A choice for the same tokens, not for the GPU: more splits would
/// be faster there too past a few thousand positions.
pub(super) const KEPT_CHUNKS_A_SPLIT: usize = 18;
/// A global layer's chunks for one more split where the GPU takes many
/// workgroups to fill and was never held to the sums before (Apple's):
/// 2, so a context of 12K positions has 94 splits, 188 workgroups.
pub(super) const SPREAD_CHUNKS_A_SPLIT: usize = 2;
/// A matrix's place for the engine (its kind is the kernel's: no flag).
impl engine::Matrix for Matrix {
    fn at(self) -> usize {
        self.at
    }

    fn flag(self) -> usize {
        0
    }
}

/// What a plan needs: where the weights are, the context's sizes, and how
/// the decoder is shaped for its GPU.
pub(super) struct Plan<'a> {
    pub(super) layout: &'a Layout,
    pub(super) sizes: Sizes,
    pub(super) shape: super::Shape,
}

impl Plan<'_> {
    /// How the rows for a matrix multiply through `m` are written: at 8
    /// bits ([`Q8`]) where it is quantised and the plan takes rows so, else
    /// as f32s (0).
    fn mode(&self, m: Matrix) -> usize {
        if self.shape.q8 && Q8_KINDS.contains(&m.kind) {
            Q8
        } else {
            0
        }
    }

    /// `rows` rows written for a matrix multiply (from `from` in `ints`)
    /// through `m` (in `wq`) to `dst` (from `at`, rows `ld` apart), as
    /// `after` says: a GEMV for a row, else the tiled GEMM; each of `m`'s
    /// kind, on its path.
    fn matmul(
        &self,
        ops: &mut Vec<Op>,
        (rows, from): (usize, usize),
        (m, wq): (Matrix, Buf),
        to: (Buf, usize, usize),
        after: After,
    ) {
        self.mul(ops, (rows, from), (m, wq), to, after, (NONE, NONE));
    }

    /// `rows` rows (at the rows' place) through `m` (in `wq`, as `after`
    /// says) made the rows for a matrix multiply through `next`, each times
    /// `H`'s at `times` where that is given; where in `ints` they start. A
    /// step's row where `next` takes f32s is one GEMV that writes them
    /// itself, after the row it reads ([`Plan::rows_after`]: no workgroup
    /// writes over inputs another still reads); else the products go to
    /// `H` and `quant` writes the rows from there.
    fn rows_for(
        &self,
        ops: &mut Vec<Op>,
        rows: usize,
        (m, wq): (Matrix, Buf),
        after: After,
        (next, times): (Matrix, Option<usize>),
    ) -> usize {
        let qa = self.sizes.qa;
        if rows == 1 && self.mode(next) != Q8 {
            let into = self.rows_after();
            let to = (Buf::H, 0, m.rows);
            self.mul(
                ops,
                (1, qa),
                (m, wq),
                to,
                after,
                (into, times.unwrap_or(NONE)),
            );
            return into;
        }
        let to = (Buf::H, 0, next.cols);
        self.matmul(ops, (rows, qa), (m, wq), to, after);
        self.quant(ops, rows, next, times);
        qa
    }

    /// Where a step's GEMV writes the next one's rows ([`Plan::rows_for`]):
    /// past the widest row at the rows' place.
    fn rows_after(&self) -> usize {
        const _: () = assert!(DIM <= FF && PER_LAYER <= FF && CHUNK >= 2);
        self.sizes.qa + FF
    }

    /// [`Plan::matmul`], a GEMV's products written `into` that place in
    /// `ints` as the rows for the next matrix multiply, each times dst's at
    /// `times` (NONE: to dst, and none; [`Plan::rows_for`]). A GEMV costs
    /// its weights read (and its products written); a GEMM is cut into
    /// slices of whole tiles of rows where it would take longer than a
    /// pace's piece at its path's rate (a chunk's gates and ups through the
    /// f32 path some 54 GFLOP, 12 ms on the M4 Max), each slice its rows'
    /// place in `ints` and in `dst`, costing its FLOPs (its rows in whole
    /// tiles).
    fn mul(
        &self,
        ops: &mut Vec<Op>,
        (rows, from): (usize, usize),
        (m, wq): (Matrix, Buf),
        (dst, at, ld): (Buf, usize, usize),
        after: After,
        (into, times): (usize, usize),
    ) {
        assert!(
            rows == 1 || (into, times) == (NONE, NONE),
            "a GEMM writes its products to dst"
        );
        let q8 = self.mode(m) == Q8;
        // The words a row written for it takes in `ints`, and its scales.
        let (row_words, row_scales) = if q8 {
            (m.cols / 4, m.cols / Q8_BLOCK * 2)
        } else {
            (m.cols, 0)
        };
        let bind = Bind {
            src: Buf::Small,
            dst,
            wq,
        };
        // The rows from `r0`, `n` of them.
        let numbers = |r0: usize, n: usize| {
            let mul = Mul {
                from: (Buf::Ints, from + r0 * row_words, m.cols),
                to: (dst, at + r0 * ld, ld),
                m: n,
                n: m.rows,
                k: m.cols,
                w: m,
                bias: None,
                after,
            };
            let mut numbers = mul.numbers().to_vec();
            let qs = self.sizes.qs + r0 * row_scales;
            numbers.extend([qs, self.sizes.best, into, times]);
            debug_assert_eq!(
                numbers.len(),
                fields("mm").len(),
                "a matrix multiply's numbers"
            );
            params(&numbers)
        };
        if rows == 1 {
            assert_eq!(m.rows % GEMV_ROWS, 0, "a GEMV takes rows in sixteens");
            ops.push(Op {
                kernel: if q8 {
                    Kernel::GemvQ8(m.kind)
                } else {
                    Kernel::GemvF32(m.kind)
                },
                buffers: bind,
                params: numbers(0, 1),
                groups: [(m.rows / GEMV_ROWS) as u32, 1, 1],
                cost: Cost::traffic((m.bytes() + m.rows * 8) as f64, self.rates().weights),
            });
            return;
        }
        assert_eq!(m.rows % GEMM_TILE, 0, "a GEMM takes columns in tiles of 64");
        assert!(
            matches!(after, After::Store | After::Gelu | After::GeGlu),
            "a GEMM writes, takes GELU or gates"
        );
        let tiles = rows.div_ceil(GEMM_TILE);
        let tile = 2.0 * (GEMM_TILE * m.rows * m.cols) as f64;
        let budget = self.rates().gemm * PIECE.as_secs_f64();
        let per = ((budget / tile) as usize).clamp(1, tiles);
        for t0 in (0..tiles).step_by(per) {
            let r0 = t0 * GEMM_TILE;
            let n = (per * GEMM_TILE).min(rows - r0);
            ops.push(Op {
                kernel: if q8 {
                    Kernel::GemmQ8(m.kind)
                } else {
                    Kernel::GemmF32(m.kind)
                },
                buffers: bind,
                params: numbers(r0, n),
                groups: [(m.rows / GEMM_TILE) as u32, n.div_ceil(GEMM_TILE) as u32, 1],
                cost: Cost::work(tile * n.div_ceil(GEMM_TILE) as f64, self.rates().gemm),
            });
        }
    }

    /// The rates its path's dispatches are estimated and cut at.
    fn rates(&self) -> &'static Rates {
        if self.shape.q8 {
            &RTX_5080_Q8
        } else {
            &M4_MAX_F32
        }
    }

    /// `bytes` moved at its path's bandwidth.
    fn moving(&self, bytes: usize) -> Cost {
        Cost::traffic(bytes as f64, self.rates().bytes)
    }

    /// `rows` rows of `H` (from its start, as wide as `m`'s inputs) written
    /// for a matrix multiply through `m`, each times `H`'s at the same place
    /// of the rows at `times` (as wide) where that is given.
    fn quant(&self, ops: &mut Vec<Op>, rows: usize, m: Matrix, times: Option<usize>) {
        let width = m.cols;
        ops.push(Op {
            kernel: Kernel::Quant,
            buffers: Bind::rows(Buf::H, Buf::Spare),
            params: named(
                "quant",
                &[
                    ("rows", rows),
                    ("width", width),
                    ("stride", width),
                    ("qa", self.sizes.qa),
                    ("qs", self.sizes.qs),
                    ("mode", self.mode(m)),
                    ("times", times.unwrap_or(NONE)),
                    ("times_stride", width),
                ],
            ),
            groups: spread(rows * width / Q8_BLOCK),
            // Its rows read, and written as words with their scales.
            cost: self.moving(rows * width * 12),
        });
    }

    /// The residual stream's step for `rows` rows of `X` from row `x_row`:
    /// `H`'s rows RMS-normed by `add` and added, the rows scaled by
    /// `scale`, then written for a matrix multiply through the matrix of
    /// `out` (normed by its norm, or as they are where that is `NONE`).
    fn residual(
        &self,
        ops: &mut Vec<Op>,
        (rows, x_row): (usize, usize),
        add: Option<usize>,
        scale: Option<usize>,
        out: Option<(usize, Matrix)>,
    ) {
        let flags = add.map_or(0, |_| ADD) | scale.map_or(0, |_| SCALE) | out.map_or(0, |_| OUT);
        ops.push(Op {
            kernel: Kernel::Residual,
            buffers: Bind::rows(Buf::H, Buf::X),
            params: named(
                "residual",
                &[
                    ("width", DIM),
                    ("stride", DIM),
                    ("add", add.unwrap_or(0)),
                    ("scalar", scale.unwrap_or(0)),
                    ("norm", out.map_or(NONE, |o| o.0)),
                    ("qa", self.sizes.qa),
                    ("qs", self.sizes.qs),
                    ("mode", out.map_or(0, |o| self.mode(o.1))),
                    ("flags", flags),
                    ("x_row", x_row),
                ],
            ),
            groups: [rows as u32, 1, 1],
            // A row of `H` and of `X` read, `X`'s written, and the rows for
            // the matrix.
            cost: self.moving(rows * DIM * 16),
        });
    }

    /// The embeddings of `rows` tokens into `X`, written for the per-layer
    /// projection, through it, then normed for layer 0: from position
    /// `first`, or the decoder's (`None`).
    fn embed(&self, ops: &mut Vec<Op>, rows: usize, first: Option<usize>) {
        let (stated, first) = (first.is_none(), first.unwrap_or(0));
        let embed = self.layout.embed;
        ops.push(Op {
            kernel: Kernel::Embed,
            buffers: Bind {
                src: Buf::Small,
                dst: Buf::X,
                wq: Buf::Shared,
            },
            params: named(
                "embed",
                &[
                    ("rows", rows),
                    ("first", first),
                    ("stated", usize::from(stated)),
                    ("at", embed.at),
                    ("scale", EMBED_SCALE.to_bits() as usize),
                    ("width", DIM),
                    ("kind", embed.kind.id() as usize),
                ],
            ),
            groups: spread(rows * DIM),
            // Each row's embedding read, and written.
            cost: self.moving(rows * DIM * 8),
        });
        let per_layer = self.layout.per_layer;
        self.residual(ops, (rows, 0), None, None, Some((NONE, per_layer)));
        self.matmul(
            ops,
            (rows, self.sizes.qa),
            (per_layer, Buf::Shared),
            (Buf::PerLayer, 0, PER_LAYERS),
            After::Store,
        );
        let first = &self.layout.layers[0];
        self.residual(
            ops,
            (rows, 0),
            None,
            None,
            Some((first.input_norm, first.q)),
        );
    }

    /// Layer `l`'s queries, keys and values (as it has them), each run of
    /// them of one kind one matrix, where each starts in a row of them.
    fn qkv(&self, l: usize) -> Vec<(Matrix, usize)> {
        let layer = &self.layout.layers[l];
        let mut parts = vec![layer.placed(layer.q)];
        if let Some((k, v)) = layer.kv {
            parts.extend([layer.placed(k), layer.placed(v)]);
        }
        let mut out: Vec<(Matrix, usize)> = Vec::new();
        let mut row = 0;
        for m in parts {
            let joined = out.last().and_then(|&(last, _)| last.and(m));
            match (joined, out.last_mut()) {
                (Some(joined), Some(last)) => last.0 = joined,
                _ => out.push((m, row)),
            }
            row += m.rows;
        }
        // Their rows are written once, for the queries' mode (the next
        // layer's `q` in `residual`): every part takes them so.
        debug_assert!(out.iter().all(|(m, _)| self.mode(*m) == self.mode(layer.q)));
        out
    }

    /// Layer `l`'s queries, keys and values of `rows` rows (in `H`, `qkv`
    /// apart) normed and turned into their places by the cache: at
    /// position `first`, or the decoder's (`None`).
    fn prep(&self, ops: &mut Vec<Op>, l: usize, rows: usize, first: Option<usize>, qkv: usize) {
        let layer = &self.layout.layers[l];
        let kv = if shares(l) { 0 } else { KV_HEADS };
        let at = self.sizes.kv(l);
        let (rope, turned) = self.sizes.rope_of(l);
        ops.push(Op {
            kernel: Kernel::Prep,
            buffers: Bind::rows(Buf::H, at.buf),
            params: named(
                "prep",
                &[
                    ("rows", rows),
                    ("first", first.unwrap_or(0)),
                    ("stated", usize::from(first.is_none())),
                    ("d", head_of(l)),
                    ("stride", qkv),
                    ("q_norm", layer.q_norm),
                    // Unread where the layer has no keys of its own (kv 0).
                    ("k_norm", layer.k_norm.unwrap_or(0)),
                    ("queries", at.queries),
                    ("keys", at.keys),
                    ("values", at.values),
                    ("positions", at.places),
                    ("rope", rope),
                    ("kv", kv),
                    ("heads", HEADS),
                    ("mask", at.mask),
                    ("turned", turned),
                ],
            ),
            groups: [rows as u32, (HEADS + 2 * kv) as u32, 1],
            // Each head's values read, and written to its place.
            cost: self.moving(rows * (HEADS + 2 * kv) * head_of(l) * 8),
        });
    }

    /// Layer `l`'s attention for the row at the decoder's position (in
    /// splits, then joined), written for its output's matrix.
    fn attend_step(&self, ops: &mut Vec<Op>, l: usize) {
        let d = head_of(l);
        // The workgroups dispatched, the least at work, the chunks for one
        // more.
        let (window, splits, least, per) = if is_global(l) {
            (0, MOST_SPLITS, GLOBAL_SPLITS, self.shape.per)
        } else {
            (WINDOW, LOCAL_SPLITS, LOCAL_SPLITS, 1)
        };
        let at = self.sizes.kv(l);
        ops.push(Op {
            kernel: Kernel::Attend,
            buffers: Bind::rows(at.buf, Buf::Parts),
            params: named(
                "attend",
                &[
                    ("d", d),
                    ("window", window),
                    ("queries", at.queries),
                    ("keys", at.keys),
                    ("values", at.values),
                    ("positions", at.places),
                    ("splits", splits),
                    ("mask", at.mask),
                    ("least", least),
                    ("per", per),
                ],
            ),
            groups: [splits as u32, KV_HEADS as u32, 1],
            // Its keys and values read: the window's, or (it reads the
            // decoder's position from the GPU) the whole context's at most.
            cost: self.moving(
                2 * KV_HEADS
                    * d
                    * if window > 0 {
                        window
                    } else {
                        self.sizes.positions
                    }
                    * 4,
            ),
        });
        ops.push(Op {
            kernel: Kernel::Join,
            buffers: Bind::rows(Buf::Parts, Buf::Spare),
            params: named(
                "join",
                &[
                    ("d", d),
                    ("splits", splits),
                    ("qa", self.sizes.qa),
                    ("qs", self.sizes.qs),
                    ("mode", self.mode(self.layout.layers[l].o)),
                    ("heads", HEADS),
                ],
            ),
            groups: [HEADS as u32, 1, 1],
            // Each split's sums read.
            cost: self.moving(HEADS * splits * (d + 2) * 4),
        });
    }

    /// Layer `l`'s attention for `rows` rows from position `first` (a
    /// prompt's), into `H`: its blocks' walks over their keys in spans of
    /// about a pace's piece at its path's rate ([`flash_spans`]), so that
    /// no dispatch keeps the GPU from the window long, the sums carried
    /// in `H` past the output ([`Sizes::carried`]).
    fn flash(&self, ops: &mut Vec<Op>, l: usize, rows: usize, first: usize) {
        debug_assert!(rows <= CHUNK, "the sums it carries are a chunk's");
        let d = head_of(l);
        let window = if is_global(l) { 0 } else { WINDOW };
        let at = self.sizes.kv(l);
        let blocks = rows.div_ceil(FLASH_LANES / (d / 16));
        let rate = self.rates().flash;
        for (tile0, tiles) in flash_spans((d, window), (rows, first), rate) {
            ops.push(Op {
                kernel: Kernel::Flash,
                buffers: Bind::rows(at.buf, Buf::H),
                params: named(
                    "flash",
                    &[
                        ("rows", rows),
                        ("first", first),
                        ("d", d),
                        ("window", window),
                        ("queries", at.queries),
                        ("keys", at.keys),
                        ("values", at.values),
                        ("positions", at.places),
                        ("heads", HEADS),
                        ("group", HEADS / KV_HEADS),
                        ("mask", at.mask),
                        ("tile0", tile0),
                        ("tiles", tiles),
                        ("state", self.sizes.carried),
                    ],
                ),
                groups: [blocks as u32, HEADS as u32, 1],
                cost: Cost::work(span_flops((d, window), (rows, first), (tile0, tiles)), rate),
            });
        }
    }

    /// Layer `l`'s per-layer inputs of `rows` rows from position `first`
    /// (or the decoder's, `None`), the projection's from row `x_row`, into
    /// `H` after the products (at `pli`).
    fn per_layer(
        &self,
        ops: &mut Vec<Op>,
        l: usize,
        (rows, x_row): (usize, usize),
        first: Option<usize>,
    ) {
        let layer = &self.layout.layers[l];
        ops.push(Op {
            kernel: Kernel::PerLayer,
            buffers: Bind {
                src: Buf::PerLayer,
                dst: Buf::H,
                wq: Buf::Layers(layer.group),
            },
            params: named(
                "per_layer",
                &[
                    ("rows", rows),
                    ("first", first.unwrap_or(0)),
                    ("stated", usize::from(first.is_none())),
                    ("stride", PER_LAYERS),
                    ("slice", l * PER_LAYER),
                    ("norm", self.layout.per_layer_norm),
                    ("at", layer.placed(layer.ple).at),
                    ("out", self.sizes.pli),
                    ("width", PER_LAYER),
                    ("x_row", x_row),
                    ("scale", PER_LAYER_PROJECTION_SCALE.to_bits() as usize),
                    ("embed_scale", PER_LAYER_EMBED_SCALE.to_bits() as usize),
                    ("kind", layer.ple.kind.id() as usize),
                ],
            ),
            groups: [rows as u32, 1, 1],
            // Its inputs read, normed and written beside the products.
            cost: self.moving(rows * PER_LAYER * 16),
        });
    }

    /// The next token from the head's workgroups' largest, put after the
    /// decoder's position, which moves on.
    fn argmax(&self, ops: &mut Vec<Op>) {
        ops.push(Op {
            kernel: Kernel::Argmax,
            buffers: Bind::rows(Buf::Small, Buf::Spare),
            params: named(
                "argmax",
                &[
                    ("groups", HEAD_GROUPS),
                    ("best", self.sizes.best),
                    ("positions", self.sizes.positions),
                ],
            ),
            groups: [1, 1, 1],
            // The head's workgroups' largest read.
            cost: self.moving(HEAD_GROUPS * 8),
        });
    }
}
