// What the two matrix multiplies (`gemm.wgsl` on f32 multiply-adds,
// `gemm_coop.wgsl` on cooperative matrices) share: C (m × n, from c_off,
// ldc apart) = A (m × k) times Wᵀ (n rows of k, each row ldw values
// apart, zeros after k), then the tail (`mm_put`). A is f32 rows (from
// a_off, lda apart: both multiples of 4) or a convolution's patches read
// where they lie (`mm_a4`); W is FP8 times a scale per 64 rows and 128 of
// k, bf16, or f32.
//
// Numbers: the engine's `Mul`'s ten (n0 m, n1 n, n2 k, n3 a_off, n4 lda,
// n5 W's first vector of four words, n6 c_off, n7 ldc, n8 the bias's
// offset in `small`, n9 its flags: `MM_*`), then n10 the scales' offset in
// `small`, n11 the scales a row group has, n12 the gates' offset in
// `src2`, n13 ldw (a multiple of 32), n27 `MX_*`. A convolution's: n14 the
// input's rows, n15 its columns, n16 its channels (a multiple of 4), n17
// the output's columns, n18 the kernel's size, n19 its stride, n20 its
// padding, n21 1 where the input is first doubled (nearest), n22 the group
// norm's means and inverse spreads in `src2` (a pair per group), n23 its
// weights' and n24 its biases' offsets in `small`, n25 its channels a
// group, n26 `GN_*`.

// The engine's flags: the bias added, the products added to what is
// there, and SwiGLU (here each tile's 128 columns as 64 gates and their
// 64 ups, silu(gate) · up written at column tile/2 + j); W's values bf16
// or f32 (else FP8).
const MM_BIAS: u32 = 1u;
const MM_ADD: u32 = 2u;
const MM_SWIGLU: u32 = 64u;
const MM_BF16: u32 = 4096u;
const MM_F32: u32 = 8192u;
// Patches of a convolution for A; the products added times a gate per
// column.
const MX_CONV: u32 = 1u;
const MX_GATED: u32 = 2u;
// A convolution's input normalized by its groups first, then SiLU.
const GN_NORM: u32 = 1u;
const GN_SILU: u32 = 2u;

// The tail of one value: the bias, then written, added or gated.
fn mm_put(row: u32, col: u32, v0: f32) {
    if (row >= p.n0 || col >= p.n1) {
        return;
    }
    var v = v0;
    if ((p.n9 & MM_BIAS) != 0u) {
        v += small[p.n8 + col];
    }
    let at = p.n6 + row * p.n7 + col;
    if ((p.n27 & MX_GATED) != 0u) {
        dst_put(at, dst_at(at) + src2[p.n12 + col] * v);
    } else if ((p.n9 & MM_ADD) != 0u) {
        dst_put(at, dst_at(at) + v);
    } else {
        dst_put(at, v);
    }
}

// Four neighbouring columns of one row through the tail.
fn mm_put4(row: u32, col: u32, v: vec4<f32>) {
    mm_put(row, col, v.x);
    mm_put(row, col + 1u, v.y);
    mm_put(row, col + 2u, v.z);
    mm_put(row, col + 3u, v.w);
}

// Four gates `g` and their ups `u` of one row: silu(g) · u written at
// column `col` (of four).
fn mm_swiglu4(row: u32, col: u32, g: vec4<f32>, u: vec4<f32>) {
    if (row >= p.n0) {
        return;
    }
    let at = p.n6 + row * p.n7 + col;
    dst[at >> 2u] = vec4<f32>(silu(g.x), silu(g.y), silu(g.z), silu(g.w)) * u;
}

// A's values k0.. k0 + 3 of row `row` (k0 a multiple of 4; zeros past m
// and k).
fn mm_a4(row: u32, k0: u32) -> vec4<f32> {
    if (row >= p.n0 || k0 >= p.n2) {
        return vec4<f32>(0.0);
    }
    if ((p.n27 & MX_CONV) == 0u) {
        return src[(p.n3 + row * p.n4 + k0) >> 2u];
    }
    // The output pixel, the kernel's tap and the input's channels.
    let cin = p.n16;
    let oy = row / p.n17;
    let ox = row % p.n17;
    let tap = k0 / cin;
    let ci = k0 % cin;
    let ky = tap / p.n18;
    let kx = tap % p.n18;
    let up = p.n21;
    let iy = i32(oy * p.n19 + ky) - i32(p.n20);
    let ix = i32(ox * p.n19 + kx) - i32(p.n20);
    if (iy < 0 || ix < 0 || u32(iy) >= (p.n14 << up) || u32(ix) >= (p.n15 << up)) {
        return vec4<f32>(0.0);
    }
    let sy = u32(iy) >> up;
    let sx = u32(ix) >> up;
    var v = src[(p.n3 + (sy * p.n15 + sx) * cin + ci) >> 2u];
    if ((p.n26 & GN_NORM) != 0u) {
        let g = ci / p.n25;
        let mean = src2[p.n22 + 2u * g];
        let rstd = src2[p.n22 + 2u * g + 1u];
        let w = vec4<f32>(small[p.n23 + ci], small[p.n23 + ci + 1u], small[p.n23 + ci + 2u], small[p.n23 + ci + 3u]);
        let b = vec4<f32>(small[p.n24 + ci], small[p.n24 + ci + 1u], small[p.n24 + ci + 2u], small[p.n24 + ci + 3u]);
        v = (v - mean) * rstd * w + b;
        if ((p.n26 & GN_SILU) != 0u) {
            v = v / (1.0 + exp(-v));
        }
    }
    return v;
}

// The scale of W's row `col` at k0 (FP8).
fn mm_scale(col: u32, k0: u32) -> f32 {
    return small[p.n10 + (col / 64u) * p.n11 + k0 / 128u];
}

// Eight values of W's row `col` from k0 (a multiple of 8; zeros past n).
struct W8 {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn mm_w8(col: u32, k0: u32) -> W8 {
    if (col >= p.n1) {
        return W8(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let i = col * p.n13 + k0;
    if ((p.n9 & MM_BF16) != 0u) {
        let w = wts[p.n5 + i / 8u];
        return W8(vec4<f32>(bf16x2(w.x), bf16x2(w.y)), vec4<f32>(bf16x2(w.z), bf16x2(w.w)));
    }
    if ((p.n9 & MM_F32) != 0u) {
        return W8(bitcast<vec4<f32>>(wts[p.n5 + i / 4u]), bitcast<vec4<f32>>(wts[p.n5 + i / 4u + 1u]));
    }
    let w = wts[p.n5 + i / 16u];
    let pair = select(w.xy, w.zw, (i & 8u) != 0u);
    let s = mm_scale(col, k0);
    return W8(e4m3x4(pair.x) * s, e4m3x4(pair.y) * s);
}
