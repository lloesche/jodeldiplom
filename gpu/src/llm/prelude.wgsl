// The language model's prelude (after the constants it shares with the
// plan: TOKENS, NONE, the flags, the kinds): each kernel reads `src` and
// `aux` (the norms and scalars) and writes `dst`, all f32; `p` holds its
// numbers (named per kernel); `wq` holds the GGUF's matrices (a buffer of
// weights, `blocks.wgsl`, or the rotary table as f32 bits), read two words
// at a time; `ints` holds the decoder's state, the tokens, and the rows a
// matrix multiplies (at 8 bits with their scales, mode Q8, or as f32 bits
// on the plain path, mode 0).
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

const EPS: f32 = 1e-6;

// Word `i` of `wq`.
fn word(i: u32) -> u32 {
    let v = wq[i >> 1u];
    return select(v.x, v.y, (i & 1u) == 1u);
}

// The first position of a kernel's rows: the decoder's (`ints`' word 0)
// where `stated`, else `first`.
fn first_position(stated: u32, first: u32) -> u32 {
    if (stated != 0u) {
        return ints[0];
    }
    return first;
}

// tanh of `x` held to ±10 first: Metal's fast tanh goes through exp(2x)
// and gives NaN (inf / inf) for large `x`; past ±9.1 tanh is ±1 in f32
// anyway, so elsewhere nothing changes.
fn tanh_held(x: f32) -> f32 {
    return tanh(clamp(x, -10.0, 10.0));
}

// exp of `x` held to -87 first: Metal's fast exp makes an integer of the
// exponent's whole part, which overflows (and gives garbage, not 0) for a
// score against the -1e30 that stands for none; exp(-87) is 1.6e-38, so
// what it weighs is gone either way.
fn exp_held(x: f32) -> f32 {
    return exp(max(x, -87.0));
}

// [`exp_held`] of four.
fn exp_held4(x: vec4<f32>) -> vec4<f32> {
    return exp(max(x, vec4<f32>(-87.0)));
}

// GELU with tanh (PyTorch's `approximate="tanh"`).
fn gelu_tanh(x: f32) -> f32 {
    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh_held(inner));
}

// Signed bytes (each -127 to 127) of four values, the first lowest.
fn pack_i8(a: i32, b: i32, c: i32, d: i32) -> u32 {
    return (u32(a) & 255u) | ((u32(b) & 255u) << 8u) | ((u32(c) & 255u) << 16u) | ((u32(d) & 255u) << 24u);
}

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
// `qs`), four to a word in order, at `qa`; else as f32 bits.
fn put_block(from_src: bool, at: u32, times: u32, row: u32, block: u32, blocks: u32, qa: u32, qs: u32, mode: u32) {
    let b = row * blocks + block;
    if (mode != Q8) {
        for (var i = 0u; i < Q8_BLOCK; i++) {
            ints[qa + b * Q8_BLOCK + i] = bitcast<u32>(block_value(from_src, at, times, i));
        }
        return;
    }
    var most = 0.0;
    for (var i = 0u; i < Q8_BLOCK; i++) {
        most = max(most, abs(block_value(from_src, at, times, i)));
    }
    let d = most / 127.0;
    var sum = 0;
    for (var w = 0u; w < Q8_BLOCK / 4u; w++) {
        let o = w * 4u;
        let q0 = round8(block_value(from_src, at, times, o), d);
        let q1 = round8(block_value(from_src, at, times, o + 1u), d);
        let q2 = round8(block_value(from_src, at, times, o + 2u), d);
        let q3 = round8(block_value(from_src, at, times, o + 3u), d);
        ints[qa + b * (Q8_BLOCK / 4u) + w] = pack_i8(q0, q1, q2, q3);
        sum += q0 + q1 + q2 + q3;
    }
    ints[qs + b * 2u] = bitcast<u32>(d);
    ints[qs + b * 2u + 1u] = bitcast<u32>(d * f32(sum));
}
