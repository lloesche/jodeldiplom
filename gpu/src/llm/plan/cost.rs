//! What a dispatch of the decoder costs: the rates its costs are
//! estimated at ([`Rates`], each path's measured), and a prompt's
//! attention cut into spans of its keys of about a pace's piece
//! ([`flash_spans`]).

use super::super::kernels::FLASH_LANES;
use super::super::HEADS;
use crate::pace::{M4_MAX_BANDWIDTH, PIECE, RTX_5080_BANDWIDTH};

/// The rates a path's dispatches are estimated at (their [`crate::pace::Cost`],
/// what a pace sizes its submissions by), and the long ones cut by (a
/// matrix multiply by rows, a prompt's attention by keys: [`flash_spans`]):
/// a matrix multiply's and the prompt's attention's FLOP/s (two a
/// multiply-add, four a value of a key, as the plan counts them), the bytes
/// a second a one-row multiply reads its weights at, and the rest move
/// their values at (an estimate: `pace::cost`).
pub(super) struct Rates {
    pub(super) gemm: f64,
    pub(super) flash: f64,
    pub(super) weights: f64,
    pub(super) bytes: f64,
}

/// The f32 path's (Metal's): the M4 Max's, from a prompt of 12 209 tokens
/// and 512 steps after it (`gemm_f32_q4k` and `_q6k` 55 TFLOP in 11.6 s;
/// the flash attention 5.9 TFLOP in 17.9 s; `gemv_f32_q4k` 0.80 TB of
/// weights in 6.0 s). Any GPU's f32 path is cut by them (the 5080's
/// `Precision::Full` too: cut more than it needs).
pub(super) const M4_MAX_F32: Rates = Rates {
    gemm: 4.5e12,
    flash: 0.33e12,
    weights: 130e9,
    bytes: M4_MAX_BANDWIDTH,
};

/// The 8-bit path's: the RTX 5080's, from a prompt of 2,779 tokens and 128
/// steps after it (`gemm_q8_q4k` and `_q6k` 12.7 TFLOP in 0.49 s; the flash
/// attention 0.47 TFLOP in 0.87 s; `gemv_q8_q4k` 0.20 TB of weights in
/// 0.46 s).
pub(super) const RTX_5080_Q8: Rates = Rates {
    gemm: 26e12,
    flash: 0.5e12,
    weights: 435e9,
    bytes: RTX_5080_BANDWIDTH,
};

/// The walks over their keys of a prompt's attention of `rows` rows from
/// position `first`, heads of `d` values and a window of `window` (0:
/// none): each block of queries' tiles of 8 keys, from the first its
/// window takes to its last row's own (`flash`).
fn walks((d, window): (usize, usize), (rows, first): (usize, usize)) -> Vec<usize> {
    let per_block = FLASH_LANES / (d / 16);
    (0..rows.div_ceil(per_block))
        .map(|b| {
            let block_first = first + b * per_block;
            let block_last = first + ((b + 1) * per_block).min(rows) - 1;
            let lo = if window > 0 {
                (block_first + 1).max(window) - window
            } else {
                0
            };
            (block_last + 8 - lo) / 8
        })
        .collect()
}

/// The FLOPs of a tile of a block's walk with heads of `d` values: its
/// queries, their heads, 8 keys, four FLOPs a value of a key.
fn tile_flops(d: usize) -> f64 {
    (FLASH_LANES / (d / 16) * HEADS * 8 * 4 * d) as f64
}

/// The FLOPs of the span of `n` tiles from `t0` of every block's walk
/// ([`walks`]) of a prompt's attention of `kind` (`d`, `window`) over
/// `rows` rows from `first`.
pub(super) fn span_flops(kind: (usize, usize), at: (usize, usize), (t0, n): (usize, usize)) -> f64 {
    let tiles: usize = (walks(kind, at).iter())
        .map(|&w| w.min(t0 + n).saturating_sub(t0))
        .sum();
    tiles as f64 * tile_flops(kind.0)
}

/// A prompt's attention of `kind` (`d`, `window`) over `rows` rows from
/// position `first` cut into spans of its blocks' walks over their keys
/// (`flash`'s `tile0` and `tiles`): as many tiles a span as keep its FLOPs
/// within a pace's piece ([`PIECE`]) at `rate`, one at least. Every block
/// takes every span (all its workgroups at once: fewer blocks would not
/// make a dispatch shorter, each workgroup walking its keys in turn; the
/// slices of 16 blocks this replaced took 37 to 60 ms deep in a story's
/// prompt on the M4 Max), their sums carried from one span to the next,
/// the same however it is cut.
pub(super) fn flash_spans(
    kind: (usize, usize),
    at: (usize, usize),
    rate: f64,
) -> Vec<(usize, usize)> {
    let budget = rate * PIECE.as_secs_f64();
    let walks = walks(kind, at);
    let longest = walks.iter().copied().max().unwrap_or(0);
    // A tile's FLOPs: those of the blocks whose walk reaches it.
    let tile = |t: usize| walks.iter().filter(|&&w| w > t).count() as f64 * tile_flops(kind.0);
    let mut spans = Vec::new();
    let mut t0 = 0;
    while t0 < longest {
        let (mut t1, mut sum) = (t0 + 1, tile(t0));
        while t1 < longest && sum + tile(t1) <= budget {
            sum += tile(t1);
            t1 += 1;
        }
        spans.push((t0, t1 - t0));
        t0 = t1;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::Sums;
    use super::super::super::model::tests::gemma;
    use super::super::super::{Shape, GLOBAL_HEAD, LOCAL_HEAD, OWN_KV, WINDOW};
    use super::super::SPREAD_CHUNKS_A_SPLIT;
    use super::super::{ContextPlan, Kernel, Plan, CHUNK, KEPT_CHUNKS_A_SPLIT};
    use super::*;

    /// A prompt's attention is cut into spans that take every tile of
    /// every block's walk over its keys once, in order, each within a
    /// piece's FLOPs but where it is one tile; a short one stays whole, and
    /// a chunk deep into a long context is cut finer the deeper it is, on
    /// either path.
    #[test]
    fn a_long_attention_is_cut_into_spans_of_its_keys() {
        let rate = M4_MAX_F32.flash;
        let budget = rate * PIECE.as_secs_f64();
        for kind in [(LOCAL_HEAD, WINDOW), (GLOBAL_HEAD, 0)] {
            for at in [(7, 0), (CHUNK, 0), (CHUNK, 8 * CHUNK), (100, 30 * CHUNK)] {
                let spans = flash_spans(kind, at, rate);
                let mut next = 0;
                for &(t0, n) in &spans {
                    assert_eq!(t0, next, "{kind:?}, {at:?}");
                    let work = span_flops(kind, at, (t0, n));
                    assert!(work <= budget || n == 1, "{kind:?}, {at:?}: {work}");
                    next += n;
                }
                let walks = walks(kind, at);
                assert_eq!(Some(&next), walks.iter().max(), "{kind:?}, {at:?}");
                let whole = span_flops(kind, at, (0, next));
                let sum: f64 = spans.iter().map(|&s| span_flops(kind, at, s)).sum();
                assert_eq!(sum, whole, "{kind:?}, {at:?}");
            }
            assert_eq!(flash_spans(kind, (7, 0), rate).len(), 1);
        }
        let spans = |first| flash_spans((GLOBAL_HEAD, 0), (CHUNK, first), rate).len();
        assert!(spans(4 * CHUNK) >= 4 && spans(16 * CHUNK) > 2 * spans(4 * CHUNK));
        let layout = gemma();
        let sizes = ContextPlan::new(&layout, 6 * CHUNK).unwrap().sizes;
        let [fast, full] = [true, false].map(|q8| Plan {
            layout: &layout,
            sizes,
            shape: Shape {
                q8,
                per: KEPT_CHUNKS_A_SPLIT,
                sums: Sums::Shared,
            },
        });
        let flashes = |plan: &Plan| {
            let ops = plan.chunk(4 * CHUNK, CHUNK, false);
            ops.iter().filter(|op| op.kernel == Kernel::Flash).count()
        };
        assert!(flashes(&fast) > OWN_KV, "{}", flashes(&fast));
        assert!(flashes(&full) > OWN_KV, "{}", flashes(&full));
    }

    /// A story's prompt of some 9,300 tokens on either path (its last
    /// chunks' global attention the longest dispatches before they were cut
    /// by their keys, 37 to 60 ms on the M4 Max), paced by its dispatches'
    /// costs, each taking its estimate and again off in unlike ways
    /// ([`crate::pace::within_two_pieces`]): from the first, no submission
    /// takes more than two pieces, or than its one dispatch where that is
    /// longer, and no dispatch is estimated at more than a piece and a half.
    /// The longest dispatches are printed (`--nocapture`).
    #[test]
    fn a_prompts_submissions_are_short_from_the_first() {
        let layout = gemma();
        let n = 9300;
        let sizes = ContextPlan::new(&layout, n + 64).unwrap().sizes;
        for q8 in [false, true] {
            let plan = Plan {
                layout: &layout,
                sizes,
                shape: Shape {
                    q8,
                    per: SPREAD_CHUNKS_A_SPLIT,
                    sums: Sums::Shared,
                },
            };
            let mut ops = Vec::new();
            for first in (0..n).step_by(CHUNK) {
                let rows = CHUNK.min(n - first);
                ops.extend(plan.chunk(first, rows, first + rows == n));
            }
            let costs: Vec<_> = ops.iter().map(|op| op.cost).collect();
            let mut longest: Vec<(f64, usize)> = costs.iter().map(|c| c.seconds).zip(0..).collect();
            longest.sort_by(|a, b| b.0.total_cmp(&a.0));
            let dear: Vec<String> = (longest.iter().take(4))
                .map(|&(t, i)| format!("{:?} {:.1} ms", ops[i].kernel, t * 1e3))
                .collect();
            eprintln!("q8 {q8}: the longest dispatches {}", dear.join(", "));
            let piece = PIECE.as_secs_f64();
            assert!(longest[0].0 <= 1.5 * piece, "q8 {q8}: {}", dear.join(", "));
            crate::pace::within_two_pieces(&format!("q8 {q8}"), &costs);
        }
    }
}
