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
//! write no keys or values), which goes through them as a step does.
//!
//! Every matrix multiply is the engine's [`Mul`]: from the rows written
//! for it in `ints` (at 8 bits or as f32s: [`Plan::mode`]), through a
//! matrix as the file holds it, to its products as [`After`] says. Each
//! kernel's numbers are written by one helper here, by name
//! ([`super::kernels::named`]: the kernels read them by the same names,
//! `prep_rows()`).

use super::kernels::{
    fields, named, Kernel, ADD, FLASH_LANES, GEMM_TILE, GEMV_ROWS, MOST_SPLITS, NONE, OUT, Q8,
    Q8_BLOCK, Q8_KINDS, SCALE, TOKENS,
};
use super::model::{Layout, Matrix};
use super::{
    head_of, is_global, kv_of, shares, Buf, DIM, EMBED_SCALE, FF, GLOBAL_BASE, GLOBAL_HEAD,
    GLOBAL_TURNED, HEADS, KV_HEADS, LAYERS, LOCAL_BASE, LOCAL_HEAD, OWN_KV, PER_LAYER, PER_LAYERS,
    PER_LAYER_EMBED_SCALE, PER_LAYER_PROJECTION_SCALE, STEPS_A_READ, VOCAB, WINDOW,
};
use crate::engine::{self, params, spread, After, Mul, PARAM_STRIDE};

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
/// The head's workgroups.
const HEAD_GROUPS: usize = VOCAB / GEMV_ROWS;
/// The most positions of a context: the 128K the model was trained for
/// (its global layers' rotary positions go no further).
const MOST_POSITIONS: usize = 131_072;
/// The positions past a context the steps may run (those made after a
/// stop, before it is read: two runs of steps).
const PAST: usize = 2 * STEPS_A_READ;
/// The largest context [`super::Gemma::reserve`] takes (a prompt and what
/// it writes): the most positions, less those the steps may run past it.
pub const MOST_CONTEXT: usize = MOST_POSITIONS - PAST;
/// The positions a context is chosen in ([`ContextPlan::fitting`]), and
/// the least chosen: the most there was room for before the sliding
/// layers kept rings (every layer's keys and values in one buffer), so a
/// plan that fitted then fits on any card.
const STEP: usize = 2048;
const LEAST_POSITIONS: usize = 9 * STEP;
/// The least context [`super::Gemma::fitting`] chooses.
pub const LEAST_CONTEXT: usize = LEAST_POSITIONS - PAST;
/// A sliding layer's ring of keys and values: the places of its window
/// and of a chunk's rows (a chunk writes over none of the positions its
/// first row still reads), a power of two (a position's place is its low
/// bits).
pub(super) const RING: usize = 1024;
const _: () = assert!(RING.is_power_of_two() && RING >= WINDOW + CHUNK);
/// The layers with keys and values of their own, of each kind.
const SLIDING: usize = OWN_KV - GLOBALS;
const GLOBALS: usize = {
    let (mut l, mut n) = (0, 0);
    while l < OWN_KV {
        n += is_global(l) as usize;
        l += 1;
    }
    n
};
/// The words of a staging buffer (the logits, or a run's tokens).
pub(super) const STAGING_WORDS: usize = VOCAB;
const _: () = assert!(VOCAB >= STEPS_A_READ);

/// A matrix's place for the engine (its kind is the kernel's: no flag).
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
    /// The cache's positions (a global layer's places).
    pub(super) positions: usize,
    /// The sliding layers' rings and the queries under way of their rows
    /// ([`Buf::Ring`]); a global layer's keys and values and the queries
    /// under way of its rows (each [`Buf::Global`]).
    pub(super) ring: usize,
    pub(super) global: usize,
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

/// Where a layer's keys and values are: their buffer, where its keys, its
/// values and the queries under way of its rows start there, the places
/// a key's value has, and the mask that takes a position to its place.
#[derive(Clone, Copy, Debug)]
struct Kv {
    buf: Buf,
    keys: usize,
    values: usize,
    queries: usize,
    places: usize,
    mask: usize,
}

impl Sizes {
    /// The sizes for a cache of `positions`.
    pub(super) fn new(positions: usize) -> Sizes {
        let qa = TOKENS + positions;
        let qs = qa + CHUNK * FF;
        let best = qs + CHUNK * FF / Q8_BLOCK * 2;
        Sizes {
            positions,
            ring: SLIDING * 2 * KV_HEADS * LOCAL_HEAD * RING + CHUNK * HEADS * LOCAL_HEAD,
            global: 2 * KV_HEADS * GLOBAL_HEAD * positions + CHUNK * HEADS * GLOBAL_HEAD,
            rope: positions * (LOCAL_HEAD / 2 + GLOBAL_TURNED) * 2,
            x: CHUNK * DIM,
            h: CHUNK * (FF + PER_LAYER),
            pli: CHUNK * FF,
            per_layer: CHUNK * PER_LAYERS,
            parts: HEADS * MOST_SPLITS * (GLOBAL_HEAD + 2),
            ints: best + HEAD_GROUPS * 2,
            qa,
            qs,
            best,
        }
    }

    /// Every buffer and its words (the vocabulary's logits too).
    pub(super) fn buffers(&self) -> Vec<(Buf, usize)> {
        let mut all = vec![(Buf::Ring, self.ring)];
        all.extend((0..GLOBALS).map(|g| (Buf::Global(g), self.global)));
        all.extend([
            (Buf::Rope, self.rope),
            (Buf::X, self.x),
            (Buf::H, self.h),
            (Buf::PerLayer, self.per_layer),
            (Buf::Parts, self.parts),
            (Buf::Logits, VOCAB),
            (Buf::Ints, self.ints),
            (Buf::Spare, 1),
        ]);
        all
    }

    /// Where layer `l`'s keys and values are (a layer that shares
    /// another's: that one's): a sliding layer's in its ring, a global
    /// layer's in a buffer of its own.
    fn kv(&self, l: usize) -> Kv {
        let l = kv_of(l);
        let words = kv_words(l);
        // The layers of its kind before it with keys and values of their own.
        let before = (0..l).filter(|&e| is_global(e) == is_global(l)).count();
        if is_global(l) {
            Kv {
                buf: Buf::Global(before),
                keys: 0,
                values: words / 2 * self.positions,
                queries: words * self.positions,
                places: self.positions,
                mask: u32::MAX as usize,
            }
        } else {
            let at = before * words * RING;
            Kv {
                buf: Buf::Ring,
                keys: at,
                values: at + words / 2 * RING,
                queries: SLIDING * words * RING,
                places: RING,
                mask: RING - 1,
            }
        }
    }

    /// Where the rotary table of a layer's kind is, and the pairs it turns.
    fn rope_of(&self, l: usize) -> (usize, usize) {
        if is_global(l) {
            (self.positions * LOCAL_HEAD, GLOBAL_TURNED)
        } else {
            (0, LOCAL_HEAD / 2)
        }
    }

    /// The rotary tables: for each position, a layer kind's cosine and sine
    /// of each pair it turns (a sliding layer's 128 pairs, then the first
    /// 64 of a global layer's 256: the others stay).
    pub(super) fn rope(&self) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.rope);
        for (dim, base, turned) in [
            (LOCAL_HEAD, LOCAL_BASE, LOCAL_HEAD / 2),
            (GLOBAL_HEAD, GLOBAL_BASE, GLOBAL_TURNED),
        ] {
            // As the reference computes them: the frequencies in f32, the
            // angles a position times them in f32.
            let freqs = engine::Freqs {
                base: f64::from(base),
                dim,
                turned,
            };
            // 1024 positions at a time, each position's pairs turned kept
            // (`engine::rotary` writes every pair of each position: in
            // pieces, its whole-width table of 131 072 positions is never
            // held).
            for from in (0..self.positions).step_by(1024) {
                let to = (from + 1024).min(self.positions);
                let some = engine::rotary(from..to, freqs, engine::Angles::F32ThenF64);
                table.extend(some.chunks(dim).flat_map(|p| &p[..2 * turned]));
            }
        }
        table
    }
}

/// The room for a context of `context` positions (a prompt and what it
/// writes) as planned: its buffers' sizes (for the context and the steps
/// run past a stop), and the dispatches of a step and of a prompt of the
/// whole context.
pub(super) struct ContextPlan {
    pub(super) context: usize,
    pub(super) sizes: Sizes,
    step: usize,
    prompt: usize,
}

impl ContextPlan {
    /// The room for `context` positions with the weights of `layout`; an
    /// error past the most a context may have.
    pub(super) fn new(layout: &Layout, context: usize) -> Result<ContextPlan, String> {
        if context > MOST_CONTEXT {
            return Err(format!(
                "a context of {context}: at most {MOST_CONTEXT} positions"
            ));
        }
        let sizes = Sizes::new(context + PAST);
        // Either path's dispatches are as many, and a chunk's are as many
        // wherever it is and however many rows it has (the last's more).
        let plan = Plan {
            layout,
            sizes,
            shape: super::Shape {
                q8: true,
                per: KEPT_CHUNKS_A_SPLIT,
            },
        };
        let step = plan.step().len();
        let chunks = context.div_ceil(CHUNK);
        let prompt = match chunks {
            0 => 0,
            _ => (chunks - 1) * plan.chunk(0, CHUNK, false).len() + plan.chunk(0, 1, true).len(),
        };
        Ok(ContextPlan {
            context,
            sizes,
            step,
            prompt,
        })
    }

    /// The bytes held while it is reserved: its buffers, the recorded
    /// step's numbers, the two staging buffers.
    pub(super) fn held_bytes(&self) -> u64 {
        let words: usize = self.sizes.buffers().iter().map(|b| b.1.max(1)).sum();
        words as u64 * 4 + self.step as u64 * PARAM_STRIDE + 2 * STAGING_WORDS as u64 * 4
    }

    /// The bytes at their most: with a prompt of the whole context's
    /// numbers recorded too.
    pub(super) fn peak_bytes(&self) -> u64 {
        self.held_bytes() + self.prompt as u64 * PARAM_STRIDE
    }

    /// The largest context whose room at its most, with the `weights`
    /// bytes beside it, takes at most `room` bytes, each of its buffers
    /// bound within `binding` bytes, and that is no larger than `need`
    /// takes (rounded up): whole steps of [`STEP`] positions (less those the
    /// steps may run past it), from [`LEAST_CONTEXT`] to [`MOST_CONTEXT`];
    /// the least where none fits, or no room is told.
    pub(super) fn fitting(
        layout: &Layout,
        weights: u64,
        (room, binding): (Option<u64>, u64),
        need: usize,
    ) -> usize {
        let context = |steps: usize| steps * STEP - PAST;
        let fits = |steps: usize| {
            let Ok(plan) = ContextPlan::new(layout, context(steps)) else {
                return false;
            };
            let bound = plan
                .sizes
                .buffers()
                .iter()
                .all(|b| b.1 as u64 * 4 <= binding);
            bound && room.is_some_and(|room| weights + plan.peak_bytes() <= room)
        };
        // The room grows with the context: the most steps that fit.
        let least_steps = LEAST_POSITIONS / STEP;
        let needed = (need + PAST).div_ceil(STEP).max(least_steps);
        let (mut least, mut most) = (least_steps, (MOST_POSITIONS / STEP).min(needed));
        while least < most {
            let mid = (least + most).div_ceil(2);
            if fits(mid) {
                least = mid;
            } else {
                most = mid - 1;
            }
        }
        context(least)
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

    /// `rows` rows written for a matrix multiply through `m` (in `wq`) to
    /// `dst` (from `at`, rows `ld` apart), as `after` says: a GEMV for a
    /// row, else the tiled GEMM; each of `m`'s kind, on its path.
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
        debug_assert_eq!(
            numbers.len(),
            fields("mm").len(),
            "a matrix multiply's numbers"
        );
        let bind = Bind {
            src: Buf::Small,
            dst,
            wq,
        };
        let q8 = self.mode(m) == Q8;
        if rows == 1 {
            assert_eq!(m.rows % GEMV_ROWS, 0, "a GEMV takes rows in sixteens");
            ops.push(Op {
                kernel: if q8 {
                    Kernel::GemvQ8(m.kind)
                } else {
                    Kernel::GemvF32(m.kind)
                },
                buffers: bind,
                params: params(&numbers),
                groups: [(m.rows / GEMV_ROWS) as u32, 1, 1],
            });
        } else {
            assert_eq!(m.rows % GEMM_TILE, 0, "a GEMM takes columns in tiles of 64");
            assert!(
                matches!(after, After::Store | After::Gelu | After::GeGlu),
                "a GEMM writes, takes GELU or gates"
            );
            ops.push(Op {
                kernel: if q8 {
                    Kernel::GemmQ8(m.kind)
                } else {
                    Kernel::GemmF32(m.kind)
                },
                buffers: bind,
                params: params(&numbers),
                groups: [
                    (m.rows / GEMM_TILE) as u32,
                    rows.div_ceil(GEMM_TILE) as u32,
                    1,
                ],
            });
        }
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
        });
        let per_layer = self.layout.per_layer;
        self.residual(ops, (rows, 0), None, None, Some((NONE, per_layer)));
        self.matmul(
            ops,
            rows,
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
        });
    }

    /// Layer `l`'s attention for `rows` rows from position `first` (a
    /// prompt's), into `H`.
    fn flash(&self, ops: &mut Vec<Op>, l: usize, rows: usize, first: usize) {
        let d = head_of(l);
        let window = if is_global(l) { 0 } else { WINDOW };
        let at = self.sizes.kv(l);
        let per_block = FLASH_LANES / (d / 16);
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
                ],
            ),
            groups: [rows.div_ceil(per_block) as u32, HEADS as u32, 1],
        });
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
        });
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
        let kv = if shares(l) { 0 } else { KV_HEADS };
        let qkv = (HEADS + 2 * kv) * head_of(l);
        for (m, row) in self.qkv(l) {
            self.matmul(ops, rows, (m, w), (Buf::H, row, qkv), After::Store);
        }
        self.prep(ops, l, rows, first, qkv);
        match first {
            None => self.attend_step(ops, l),
            Some(first) => {
                self.flash(ops, l, rows, first);
                self.quant(ops, rows, layer.o, None);
            }
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
            Some((layer.pre_ff, layer.gate_up)),
        );
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.gate_up), w),
            (Buf::H, 0, FF),
            After::GeGlu,
        );
        self.quant(ops, rows, layer.down, None);
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.down), w),
            (Buf::H, 0, DIM),
            After::Store,
        );
        self.residual(
            ops,
            (rows, x_row),
            Some(layer.post_ff),
            None,
            Some((NONE, layer.gate)),
        );
        self.per_layer(ops, l, (rows, x_row), first);
        self.matmul(
            ops,
            rows,
            (layer.placed(layer.gate), w),
            (Buf::H, 0, PER_LAYER),
            After::Gelu,
        );
        self.quant(ops, rows, layer.proj, Some(self.sizes.pli));
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
            1,
            (self.layout.embed, Buf::Shared),
            (Buf::Logits, 0, VOCAB),
            After::Argmax { stride: 0 },
        );
        self.argmax(ops);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Either path's step and chunks are as many dispatches (the room is
    /// planned on one); a position more of context holds the keys and
    /// values of each global layer with its own, its rows of both rotary
    /// tables (a global layer's pairs turned only) and its token, and
    /// nothing else (the sliding layers' rings stay as they are); a
    /// prompt's numbers are recorded on top at the most.
    #[test]
    fn the_room_is_as_planned_on_either_path() {
        let layout = super::super::model::tests::gemma();
        let context = 2 * CHUNK + 100;
        let room = ContextPlan::new(&layout, context).unwrap();
        let [fast, full] = [true, false].map(|q8| Plan {
            layout: &layout,
            sizes: room.sizes,
            shape: super::super::Shape {
                q8,
                per: SPREAD_CHUNKS_A_SPLIT,
            },
        });
        assert_eq!(fast.step().len(), full.step().len());
        let mut prompt = 0;
        for (first, rows, last) in [
            (0, CHUNK, false),
            (CHUNK, CHUNK, false),
            (2 * CHUNK, 100, true),
        ] {
            let n = full.chunk(first, rows, last).len();
            assert_eq!(fast.chunk(first, rows, last).len(), n, "chunk at {first}");
            prompt += n as u64;
        }
        let more = ContextPlan::new(&layout, context + 1).unwrap();
        let kv: usize = (0..OWN_KV)
            .filter(|&l| is_global(l))
            .map(|l| 2 * KV_HEADS * head_of(l))
            .sum();
        let a_position = (kv + LOCAL_HEAD + 2 * GLOBAL_TURNED + 1) as u64 * 4;
        assert_eq!(more.held_bytes() - room.held_bytes(), a_position);
        assert_eq!(room.peak_bytes(), room.held_bytes() + prompt * PARAM_STRIDE);
        assert!(ContextPlan::new(&layout, MOST_CONTEXT + 1).is_err());
        // The largest context's buffers each within the 2 GB of a binding
        // as wgpu has it.
        let binding = (1 << 31) - 4;
        let most = ContextPlan::new(&layout, MOST_CONTEXT).unwrap();
        assert!(most.sizes.buffers().iter().all(|b| b.1 * 4 <= binding));
    }

    /// A sliding layer's ring takes each position to its own place for as
    /// long as its window and a chunk's rows read it; a global layer's
    /// every position; the rotary tables hold each position's turned
    /// pairs.
    #[test]
    fn a_ring_keeps_a_window_and_a_chunk() {
        let sizes = Sizes::new(5000);
        let sliding = sizes.kv(0);
        assert_eq!((sliding.buf, sliding.places), (Buf::Ring, RING));
        let reads = |pos: usize| (pos + 1).saturating_sub(WINDOW)..=pos;
        for first in [0, 300, 1000, 2047] {
            // A chunk's rows written, then every row's window read.
            let written: Vec<usize> = (first..first + CHUNK).collect();
            for row in &written {
                for p in reads(*row) {
                    let clash = written
                        .iter()
                        .any(|&w| w != p && w & sliding.mask == p & sliding.mask);
                    assert!(!clash, "position {p} overwritten in the chunk from {first}");
                }
            }
        }
        let global = sizes.kv(5);
        assert_eq!((global.buf, global.places), (Buf::Global(0), 5000));
        assert_eq!(4999 & global.mask, 4999);
        // The shared layers read the last of their kind's.
        assert_eq!(sizes.kv(41).buf, Buf::Global(GLOBALS - 1));
        assert_eq!(sizes.kv(40).keys, sizes.kv(22).keys);
        assert_eq!(sizes.rope().len(), sizes.rope);
    }

    /// The context chosen is the largest whose room fits, in whole steps,
    /// within the least and the most and no larger than the need (rounded
    /// up to a step); a binding too small for a global layer's buffer
    /// holds it back too.
    #[test]
    fn the_context_is_the_largest_that_fits() {
        let layout = super::super::model::tests::gemma();
        let weights = 1 << 30;
        let peak = |c: usize| weights + ContextPlan::new(&layout, c).unwrap().peak_bytes();
        let binding = (1 << 31) - 4;
        let at = |positions: usize| positions - PAST;
        let fit =
            |room, binding, need| ContextPlan::fitting(&layout, weights, (room, binding), need);
        assert_eq!(fit(None, binding, MOST_CONTEXT), LEAST_CONTEXT);
        assert_eq!(fit(Some(0), binding, MOST_CONTEXT), LEAST_CONTEXT);
        assert_eq!(fit(Some(u64::MAX), binding, MOST_CONTEXT), MOST_CONTEXT);
        let room = peak(at(20 * STEP));
        assert_eq!(fit(Some(room), binding, MOST_CONTEXT), at(20 * STEP));
        assert_eq!(fit(Some(room - 1), binding, MOST_CONTEXT), at(19 * STEP));
        // The need rounded up to a step; never under the least.
        assert_eq!(
            fit(Some(u64::MAX), binding, at(12 * STEP) + 1),
            at(13 * STEP)
        );
        assert_eq!(fit(Some(u64::MAX), binding, at(12 * STEP)), at(12 * STEP));
        assert_eq!(fit(Some(u64::MAX), binding, 1000), LEAST_CONTEXT);
        // A binding only as large as a global layer's buffer at 32 steps
        // (520 MiB: its keys and values, and its queries).
        let small = (Sizes::new(32 * STEP).global * 4) as u64;
        assert_eq!(fit(Some(u64::MAX), small, MOST_CONTEXT), at(32 * STEP));
    }
}
