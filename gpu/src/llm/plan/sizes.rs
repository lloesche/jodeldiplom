//! The room a context takes ([`ContextPlan`]): the buffers that depend on
//! it ([`Sizes`]) and where things are in them, and the dispatches a step
//! and a prompt of the whole context record.

use super::super::kernels::{Sums, GEMV_ROWS, MOST_SPLITS, Q8_BLOCK, TOKENS};
use super::super::model::Layout;
use super::super::{
    head_of, is_global, kv_of, Buf, Shape, DIM, FF, GLOBAL_BASE, GLOBAL_HEAD, GLOBAL_TURNED, HEADS,
    KV_HEADS, LOCAL_BASE, LOCAL_HEAD, OWN_KV, PER_LAYER, PER_LAYERS, STEPS_A_READ, VOCAB, WINDOW,
};
use super::{Op, Plan, CHUNK, KEPT_CHUNKS_A_SPLIT};
use crate::engine::{self, PARAM_STRIDE};

/// The head's workgroups.
pub(in crate::llm) const HEAD_GROUPS: usize = VOCAB / GEMV_ROWS;
/// Where in `H` a prompt's attention carries its sums from one span of its
/// keys to the next ([`Sizes::carried`]): past a chunk's output (every
/// head's values of every row, a global layer's the widest).
const CARRIED: usize = CHUNK * HEADS * GLOBAL_HEAD;
// The sums it carries (a row's heads, each its largest, its total and its
// values) fit in `H` before the per-layer inputs.
const _: () = assert!(CARRIED + CHUNK * HEADS * (GLOBAL_HEAD + 2) <= CHUNK * FF);
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
pub(in crate::llm) const RING: usize = 1024;
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
pub(in crate::llm) const STAGING_WORDS: usize = VOCAB;
const _: () = assert!(VOCAB >= STEPS_A_READ);

/// The sizes of the buffers that depend on the context: f32s (or u32s)
/// each, and where things are in them.
#[derive(Clone, Copy, Debug)]
pub(in crate::llm) struct Sizes {
    /// The cache's positions (a global layer's places).
    pub(in crate::llm) positions: usize,
    /// The sliding layers' rings and the queries under way of their rows
    /// ([`Buf::Ring`]); a global layer's keys and values and the queries
    /// under way of its rows (each [`Buf::Global`]).
    pub(in crate::llm) ring: usize,
    pub(in crate::llm) global: usize,
    pub(in crate::llm) rope: usize,
    pub(in crate::llm) x: usize,
    /// A matrix multiply's products, and the per-layer inputs after them
    /// (at `pli`).
    pub(in crate::llm) h: usize,
    pub(in crate::llm) pli: usize,
    /// Where in `H` a prompt's attention carries its sums from one span of
    /// its keys to the next (`flash`'s `state`): past a chunk's output,
    /// before the per-layer inputs. Only a prompt's chunks carry them; a
    /// step's one-row multiplies write the next one's rows in `ints`
    /// (`Plan::rows_after`), so the two never share a place.
    pub(in crate::llm) carried: usize,
    pub(in crate::llm) per_layer: usize,
    pub(in crate::llm) parts: usize,
    pub(in crate::llm) ints: usize,
    /// Where in `ints` a matrix multiply's rows and their scales are, and
    /// the head's workgroups' largest.
    pub(in crate::llm) qa: usize,
    pub(in crate::llm) qs: usize,
    pub(in crate::llm) best: usize,
}

/// The words a layer with its own keys and values keeps a position.
fn kv_words(l: usize) -> usize {
    2 * KV_HEADS * head_of(l)
}

/// Where a layer's keys and values are: their buffer, where its keys, its
/// values and the queries under way of its rows start there, the places
/// a key's value has, and the mask that takes a position to its place.
#[derive(Clone, Copy, Debug)]
pub(super) struct Kv {
    pub(super) buf: Buf,
    pub(super) keys: usize,
    pub(super) values: usize,
    pub(super) queries: usize,
    pub(super) places: usize,
    pub(super) mask: usize,
}

impl Sizes {
    /// The sizes for a cache of `positions`.
    pub(in crate::llm) fn new(positions: usize) -> Sizes {
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
            carried: CARRIED,
            per_layer: CHUNK * PER_LAYERS,
            parts: HEADS * MOST_SPLITS * (GLOBAL_HEAD + 2),
            ints: best + HEAD_GROUPS * 2,
            qa,
            qs,
            best,
        }
    }

    /// Every buffer and its words (the vocabulary's logits too).
    pub(in crate::llm) fn buffers(&self) -> Vec<(Buf, usize)> {
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
    pub(super) fn kv(&self, l: usize) -> Kv {
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
    pub(super) fn rope_of(&self, l: usize) -> (usize, usize) {
        if is_global(l) {
            (self.positions * LOCAL_HEAD, GLOBAL_TURNED)
        } else {
            (0, LOCAL_HEAD / 2)
        }
    }

    /// The rotary tables: for each position, a layer kind's cosine and sine
    /// of each pair it turns (a sliding layer's 128 pairs, then the first
    /// 64 of a global layer's 256: the others stay).
    pub(in crate::llm) fn rope(&self) -> Vec<f32> {
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
pub(in crate::llm) struct ContextPlan {
    pub(in crate::llm) context: usize,
    pub(in crate::llm) sizes: Sizes,
    step: usize,
    prompt: usize,
}

impl ContextPlan {
    /// The room for `context` positions with the weights of `layout`; an
    /// error past the most a context may have.
    pub(in crate::llm) fn new(layout: &Layout, context: usize) -> Result<ContextPlan, String> {
        if context > MOST_CONTEXT {
            return Err(format!(
                "a context of {context}: at most {MOST_CONTEXT} positions"
            ));
        }
        let sizes = Sizes::new(context + PAST);
        // Each piece as the more dispatches of the two paths; the chunks
        // before the last each as many as the deepest of them (a chunk's
        // attention is cut finer the deeper it is, `flash_spans`, and the
        // f32 path's matrix multiplies more than the 8-bit path's).
        let plans = [true, false].map(|q8| Plan {
            layout,
            sizes,
            shape: Shape {
                q8,
                per: KEPT_CHUNKS_A_SPLIT,
                sums: Sums::Shared,
            },
        });
        let most =
            |ops: &dyn Fn(&Plan) -> Vec<Op>| plans.iter().map(|p| ops(p).len()).fold(0, usize::max);
        let step = most(&|p| p.step());
        let chunks = context.div_ceil(CHUNK);
        let prompt = match chunks {
            0 => 0,
            _ => {
                let last = (chunks - 1) * CHUNK;
                let deep = last.saturating_sub(CHUNK);
                let deepest = most(&|p| p.chunk(deep, CHUNK, false));
                let tail = most(&|p| p.chunk(last, context - last, true));
                (chunks - 1) * deepest + tail
            }
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
    pub(in crate::llm) fn held_bytes(&self) -> u64 {
        let words: usize = self.sizes.buffers().iter().map(|b| b.1.max(1)).sum();
        words as u64 * 4 + self.step as u64 * PARAM_STRIDE + 2 * STAGING_WORDS as u64 * 4
    }

    /// The bytes at their most: with a prompt of the whole context's
    /// numbers recorded too.
    pub(in crate::llm) fn peak_bytes(&self) -> u64 {
        self.held_bytes() + self.prompt as u64 * PARAM_STRIDE
    }

    /// The largest context whose room at its most, with the `weights`
    /// bytes beside it, takes at most `room` bytes, each of its buffers
    /// bound within `binding` bytes, and that is no larger than `need`
    /// takes (rounded up): whole steps of [`STEP`] positions (less those the
    /// steps may run past it), from [`LEAST_CONTEXT`] to [`MOST_CONTEXT`];
    /// the least where none fits, or no room is told.
    pub(in crate::llm) fn fitting(
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

#[cfg(test)]
mod tests {
    use super::super::super::model::tests::gemma;
    use super::super::super::Shape;
    use super::super::SPREAD_CHUNKS_A_SPLIT;
    use super::*;

    /// The room holds the more dispatches of the two paths for a step and
    /// for each chunk (a chunk deeper as many or more); a position more of
    /// context holds the keys and
    /// values of each global layer with its own, its rows of both rotary
    /// tables (a global layer's pairs turned only) and its token, and
    /// nothing else (the sliding layers' rings stay as they are); a
    /// prompt's numbers are recorded on top at the most.
    #[test]
    fn the_room_is_as_planned_on_either_path() {
        let layout = gemma();
        let context = 2 * CHUNK + 100;
        let room = ContextPlan::new(&layout, context).unwrap();
        let [fast, full] = [true, false].map(|q8| Plan {
            layout: &layout,
            sizes: room.sizes,
            shape: Shape {
                q8,
                per: SPREAD_CHUNKS_A_SPLIT,
                sums: Sums::Shared,
            },
        });
        // Each piece planned as the more of the two paths', the chunks
        // before the last each as the deepest of them.
        let most = |ops: &dyn Fn(&Plan) -> usize| ops(&fast).max(ops(&full));
        assert_eq!(room.step, most(&|p| p.step().len()));
        for p in [&fast, &full] {
            let shallow = p.chunk(0, CHUNK, false).len();
            assert!(shallow <= p.chunk(CHUNK, CHUNK, false).len());
        }
        let deepest = most(&|p| p.chunk(CHUNK, CHUNK, false).len());
        let last = most(&|p| p.chunk(2 * CHUNK, 100, true).len());
        let prompt = (2 * deepest + last) as u64;
        assert_eq!(room.prompt as u64, prompt);
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
        let layout = gemma();
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
