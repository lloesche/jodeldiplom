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
// offset in `small`, n9 its flags: `MM_*`), then the plan's own, read
// through the functions the plan declares (`plan::MM_NUMBERS`): the
// scales' offset in `small` (`mm_scales`), the scales a row group has
// (`mm_per`), the gates' offset in `src2` (`mm_gates`), ldw (`mm_ldw`, a
// multiple of 32), `MX_*` (`mx_flags`). A convolution's: the input's rows
// and columns (`cv_h`, `cv_w`), its channels (`cv_c`, a multiple of 4),
// the output's columns (`cv_ow`), the kernel's size, stride and padding
// (`cv_size`, `cv_stride`, `cv_pad`), 1 where the input is first doubled
// (`cv_up`, nearest); the group norm's means and inverse spreads in
// `src2` (`gn_at`, a pair per group), its weights' and biases' offsets in
// `small` (`gn_weight`, `gn_bias`), its channels a group (`gn_group`),
// `GN_*` (`gn_flags`).
//
// The flags (declared by the plan): the engine's (the bias added, the
// products added to what is there, and SwiGLU: here each tile's 128
// columns as 64 gates and their 64 ups, silu(gate) · up written at column
// tile/2 + j); W's values bf16 or f32 (else FP8); a convolution's patches
// for A, the products added times a gate per column; a convolution's input
// normalized by its groups first, then SiLU.

// The tail of one value: the bias, then written, added or gated.
fn mm_put(row: u32, col: u32, v0: f32) {
    if (row >= mm_m() || col >= mm_n()) {
        return;
    }
    var v = v0;
    if ((mm_flags() & MM_BIAS) != 0u) {
        v += small[mm_bias() + col];
    }
    let at = mm_c() + row * mm_ldc() + col;
    if ((mx_flags() & MX_GATED) != 0u) {
        dst_put(at, dst_at(at) + src2[mm_gates() + col] * v);
    } else if ((mm_flags() & MM_ADD) != 0u) {
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
    if (row >= mm_m()) {
        return;
    }
    let at = mm_c() + row * mm_ldc() + col;
    dst_put4(at >> 2u, vec4<f32>(silu(g.x), silu(g.y), silu(g.z), silu(g.w)) * u);
}

// A's values k0.. k0 + 3 of row `row` (k0 a multiple of 4; zeros past m
// and k).
fn mm_a4(row: u32, k0: u32) -> vec4<f32> {
    if (row >= mm_m() || k0 >= mm_k()) {
        return vec4<f32>(0.0);
    }
    if ((mx_flags() & MX_CONV) == 0u) {
        return src[(mm_a() + row * mm_lda() + k0) >> 2u];
    }
    // The output pixel, the kernel's tap and the input's channels.
    let cin = cv_c();
    let oy = row / cv_ow();
    let ox = row % cv_ow();
    let tap = k0 / cin;
    let ci = k0 % cin;
    let ky = tap / cv_size();
    let kx = tap % cv_size();
    let up = cv_up();
    let iy = i32(oy * cv_stride() + ky) - i32(cv_pad());
    let ix = i32(ox * cv_stride() + kx) - i32(cv_pad());
    if (iy < 0 || ix < 0 || u32(iy) >= (cv_h() << up) || u32(ix) >= (cv_w() << up)) {
        return vec4<f32>(0.0);
    }
    let sy = u32(iy) >> up;
    let sx = u32(ix) >> up;
    var v = src[(mm_a() + (sy * cv_w() + sx) * cin + ci) >> 2u];
    if ((gn_flags() & GN_NORM) != 0u) {
        let g = ci / gn_group();
        let mean = src2[gn_at() + 2u * g];
        let rstd = src2[gn_at() + 2u * g + 1u];
        let w = vec4<f32>(small[gn_weight() + ci], small[gn_weight() + ci + 1u], small[gn_weight() + ci + 2u], small[gn_weight() + ci + 3u]);
        let b = vec4<f32>(small[gn_bias() + ci], small[gn_bias() + ci + 1u], small[gn_bias() + ci + 2u], small[gn_bias() + ci + 3u]);
        v = (v - mean) * rstd * w + b;
        if ((gn_flags() & GN_SILU) != 0u) {
            v = v / (1.0 + exp(min(-v, vec4<f32>(87.0))));
        }
    }
    return v;
}

// The scale of W's row `col` at k0 (FP8).
fn mm_scale(col: u32, k0: u32) -> f32 {
    return small[mm_scales() + (col / 64u) * mm_per() + k0 / 128u];
}

// Eight values of W's row `col` from k0 (a multiple of 8; zeros past n).
struct W8 {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn mm_w8(col: u32, k0: u32) -> W8 {
    if (col >= mm_n()) {
        return W8(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let i = col * mm_ldw() + k0;
    if ((mm_flags() & MM_BF16) != 0u) {
        let w = wts[mm_w() + i / 8u];
        return W8(vec4<f32>(bf16x2(w.x), bf16x2(w.y)), vec4<f32>(bf16x2(w.z), bf16x2(w.w)));
    }
    if ((mm_flags() & MM_F32) != 0u) {
        return W8(bitcast<vec4<f32>>(wts[mm_w() + i / 4u]), bitcast<vec4<f32>>(wts[mm_w() + i / 4u + 1u]));
    }
    let w = wts[mm_w() + i / 16u];
    let pair = select(w.xy, w.zw, (i & 8u) != 0u);
    let s = mm_scale(col, k0);
    return W8(e4m3x4(pair.x) * s, e4m3x4(pair.y) * s);
}
