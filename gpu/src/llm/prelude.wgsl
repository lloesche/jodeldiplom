// The language model's prelude: each kernel reads `src` and `aux` (the
// norms and scalars) and writes `dst`, all f32; `p` holds its numbers
// (named per kernel); `wq` holds 4-bit matrices (a buffer of weights, or
// the rotary table as f32 bits), read two words at a time; `ints` holds
// the decoder's state, the tokens, and the rows a matrix multiplies (at 8
// bits with their scales, or as f32 bits on the plain path).
//
// `ints`: the position the next step decodes at word 0, the tokens from
// word TOKENS on (a token at its position).

struct Params {
    f0: u32, f1: u32, f2: u32, f3: u32,
    f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32,
    f12: u32, f13: u32, f14: u32, f15: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> wq: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> ints: array<u32>;

const TOKENS: u32 = 16u;
const EPS: f32 = 1e-6;
// The int8 rows and their scales (the q8 path), or the rows as f32s.
const Q8: u32 = 1u;

// Word `i` of `wq`.
fn word(i: u32) -> u32 {
    let v = wq[i >> 1u];
    return select(v.x, v.y, (i & 1u) == 1u);
}

// A group's scale (x) and offset (y), from its scales word.
fn scale_of(i: u32) -> vec2<f32> {
    return unpack2x16float(word(i));
}

// Value `col` of row `row` of the 4-bit matrix at `at` (`cols` inputs a
// row, `rows` rows).
fn q4_value(at: u32, rows: u32, cols: u32, row: u32, col: u32) -> f32 {
    let groups = cols / 32u;
    let g = row * groups + col / 32u;
    let j = col % 32u;
    let nibble = (word(at + g * 4u + j / 8u) >> (4u * (j % 8u))) & 15u;
    let s = scale_of(at + rows * groups * 4u + g);
    return s.y + f32(nibble) * s.x;
}

// The first position of a kernel's rows: the decoder's (`ints`' word 0)
// where `stated`, else `first`.
fn first_position(stated: u32, first: u32) -> u32 {
    if (stated != 0u) {
        return ints[0];
    }
    return first;
}

// GELU with tanh (PyTorch's `approximate="tanh"`).
fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

// Signed bytes (each -127 to 127) of four values, the first lowest.
fn pack_i8(a: i32, b: i32, c: i32, d: i32) -> u32 {
    return (u32(a) & 255u) | ((u32(b) & 255u) << 8u) | ((u32(c) & 255u) << 16u) | ((u32(d) & 255u) << 24u);
}

// No norm, no other rows.
const NONE: u32 = 0xFFFFFFFFu;

// Values a workgroup writes for a matrix multiply (a row of the residual
// stream, a step's attention), staged.
var<workgroup> staged: array<f32, 4096>;

// Value `i` of a block: from `staged` (at `at`), or from src (at `at`, times
// src's at `times` where that is not NONE).
fn block_value(from_src: bool, at: u32, times: u32, i: u32) -> f32 {
    if (!from_src) {
        return staged[at + i];
    }
    var v = src[at + i];
    if (times != NONE) {
        v *= src[times + i];
    }
    return v;
}

// `x` at 8 bits by the scale `d`.
fn round8(x: f32, d: f32) -> i32 {
    if (d > 0.0) {
        return i32(clamp(round(x / d), -127.0, 127.0));
    }
    return 0;
}

// 32 values (`block_value`'s) of row `row` written for a matrix multiply as
// block `block` of `blocks` a row: on the q8 path rounded to 8 bits by the
// block's largest (its scale and the scaled sum of its values kept, at
// `qs`), the even values of each eight in one word and the odd ones in the
// next (as a 4-bit matrix's word holds them), at `qa`; else as f32 bits.
fn put_block(from_src: bool, at: u32, times: u32, row: u32, block: u32, blocks: u32, qa: u32, qs: u32, mode: u32) {
    let b = row * blocks + block;
    if (mode != Q8) {
        for (var i = 0u; i < 32u; i++) {
            ints[qa + b * 32u + i] = bitcast<u32>(block_value(from_src, at, times, i));
        }
        return;
    }
    var most = 0.0;
    for (var i = 0u; i < 32u; i++) {
        most = max(most, abs(block_value(from_src, at, times, i)));
    }
    let d = most / 127.0;
    var sum = 0;
    for (var w = 0u; w < 4u; w++) {
        let o = w * 8u;
        let q0 = round8(block_value(from_src, at, times, o), d);
        let q1 = round8(block_value(from_src, at, times, o + 1u), d);
        let q2 = round8(block_value(from_src, at, times, o + 2u), d);
        let q3 = round8(block_value(from_src, at, times, o + 3u), d);
        let q4 = round8(block_value(from_src, at, times, o + 4u), d);
        let q5 = round8(block_value(from_src, at, times, o + 5u), d);
        let q6 = round8(block_value(from_src, at, times, o + 6u), d);
        let q7 = round8(block_value(from_src, at, times, o + 7u), d);
        ints[qa + b * 8u + 2u * w] = pack_i8(q0, q2, q4, q6);
        ints[qa + b * 8u + 2u * w + 1u] = pack_i8(q1, q3, q5, q7);
        sum += q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7;
    }
    ints[qs + b * 2u] = bitcast<u32>(d);
    ints[qs + b * 2u + 1u] = bitcast<u32>(d * f32(sum));
}
