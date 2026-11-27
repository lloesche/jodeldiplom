// What every model's kernels share, compiled after the model's own prelude
// (its parameters and bindings) and before its kernels: nothing here reads
// a binding.

// The invocation's place in a one-dimensional job spread over a grid of
// workgroups of 256 (a dispatch has at most 65535 groups per side).
fn linear_id(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

var<workgroup> red64: array<f32, 64>;

// The sum of `v` over a workgroup of 64 (every invocation calls it, and
// gets it), in a tree.
fn sum64(t: u32, v: f32) -> f32 {
    red64[t] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (t < s) {
            red64[t] += red64[t + s];
        }
        workgroupBarrier();
    }
    let total = red64[0];
    workgroupBarrier();
    return total;
}

// The largest `v` over a workgroup of 64, as sum64.
fn max64(t: u32, v: f32) -> f32 {
    red64[t] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (t < s) {
            red64[t] = max(red64[t], red64[t + s]);
        }
        workgroupBarrier();
    }
    let most = red64[0];
    workgroupBarrier();
    return most;
}

var<workgroup> red256: array<f32, 256>;

// The sum of `v` over a workgroup of 256 (every invocation calls it, and
// gets it), in a tree.
fn sum256(li: u32, v: f32) -> f32 {
    workgroupBarrier();
    red256[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red256[li] += red256[li + s];
        }
        workgroupBarrier();
    }
    return red256[0];
}

// The tiled matrix multiplies' tiles: 64 × 64 through k in steps of 16;
// each invocation keeps 4 rows × 4 columns in four vectors. The tiles hold
// vectors: a step of k reads one of A (its four rows) and one of B (its
// four columns) for 16 multiply-adds, and the 16 steps are written out (no
// loop for the compiler to unroll, which Apple's does not always do with
// the loops wgpu writes for Metal). A is stored [k][row / 4], B [k][col / 4].
var<workgroup> tile_a: array<vec4<f32>, 256>;
var<workgroup> tile_b: array<vec4<f32>, 256>;

// The four rows × four columns an invocation sums.
struct Acc {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    r3: vec4<f32>,
}

// One step of k (kk of the tile's 16) for the invocation at (lx, ly).
fn mm_step(acc: Acc, kk: u32, lx: u32, ly: u32) -> Acc {
    let a = tile_a[kk * 16u + ly];
    let b = tile_b[kk * 16u + lx];
    return Acc(
        fma(vec4<f32>(a.x), b, acc.r0),
        fma(vec4<f32>(a.y), b, acc.r1),
        fma(vec4<f32>(a.z), b, acc.r2),
        fma(vec4<f32>(a.w), b, acc.r3),
    );
}

// The 16 steps of a loaded tile pair, written out.
fn mm_tile(start: Acc, lx: u32, ly: u32) -> Acc {
    var acc = start;
    acc = mm_step(acc, 0u, lx, ly);
    acc = mm_step(acc, 1u, lx, ly);
    acc = mm_step(acc, 2u, lx, ly);
    acc = mm_step(acc, 3u, lx, ly);
    acc = mm_step(acc, 4u, lx, ly);
    acc = mm_step(acc, 5u, lx, ly);
    acc = mm_step(acc, 6u, lx, ly);
    acc = mm_step(acc, 7u, lx, ly);
    acc = mm_step(acc, 8u, lx, ly);
    acc = mm_step(acc, 9u, lx, ly);
    acc = mm_step(acc, 10u, lx, ly);
    acc = mm_step(acc, 11u, lx, ly);
    acc = mm_step(acc, 12u, lx, ly);
    acc = mm_step(acc, 13u, lx, ly);
    acc = mm_step(acc, 14u, lx, ly);
    acc = mm_step(acc, 15u, lx, ly);
    return acc;
}

// GELU with the error function (as PyTorch's default), the error function
// as Abramowitz and Stegun's 7.1.26 (to 1.5e-7).
fn gelu(x: f32) -> f32 {
    let z = abs(x) * 0.70710678;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign(x) * (1.0 - poly * exp(-z * z));
    return 0.5 * x * (1.0 + erf);
}

// A head of 64 values as sixteen vectors, each a variable of its own: the
// attention kernels hold a query and its weighted values so (an array
// indexed in a loop may be kept in memory rather than registers on Apple's
// GPUs, from the loops wgpu writes for Metal), each written out in the
// order a loop over the vectors had.
struct H64 {
    v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, v4: vec4<f32>, v5: vec4<f32>, v6: vec4<f32>, v7: vec4<f32>, v8: vec4<f32>, v9: vec4<f32>, v10: vec4<f32>, v11: vec4<f32>, v12: vec4<f32>, v13: vec4<f32>, v14: vec4<f32>, v15: vec4<f32>,
}

fn h64_zero() -> H64 {
    let z = vec4<f32>(0.0);
    return H64(z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z);
}

// The two heads' dot product, the vectors' in their order.
fn h64_dot(a: H64, b: H64) -> f32 {
    var s = 0.0;
    s += dot(a.v0, b.v0);
    s += dot(a.v1, b.v1);
    s += dot(a.v2, b.v2);
    s += dot(a.v3, b.v3);
    s += dot(a.v4, b.v4);
    s += dot(a.v5, b.v5);
    s += dot(a.v6, b.v6);
    s += dot(a.v7, b.v7);
    s += dot(a.v8, b.v8);
    s += dot(a.v9, b.v9);
    s += dot(a.v10, b.v10);
    s += dot(a.v11, b.v11);
    s += dot(a.v12, b.v12);
    s += dot(a.v13, b.v13);
    s += dot(a.v14, b.v14);
    s += dot(a.v15, b.v15);
    return s;
}

// a · k.
fn h64_scale(a: H64, k: f32) -> H64 {
    return H64(a.v0 * k, a.v1 * k, a.v2 * k, a.v3 * k, a.v4 * k, a.v5 * k, a.v6 * k, a.v7 * k, a.v8 * k, a.v9 * k, a.v10 * k, a.v11 * k, a.v12 * k, a.v13 * k, a.v14 * k, a.v15 * k);
}

// a / k.
fn h64_div(a: H64, k: f32) -> H64 {
    return H64(a.v0 / k, a.v1 / k, a.v2 / k, a.v3 / k, a.v4 / k, a.v5 / k, a.v6 / k, a.v7 / k, a.v8 / k, a.v9 / k, a.v10 / k, a.v11 / k, a.v12 / k, a.v13 / k, a.v14 / k, a.v15 / k);
}

// a · fade + b (the online softmax's new largest score).
fn h64_fade(a: H64, fade: f32, b: H64) -> H64 {
    return H64(a.v0 * fade + b.v0, a.v1 * fade + b.v1, a.v2 * fade + b.v2, a.v3 * fade + b.v3, a.v4 * fade + b.v4, a.v5 * fade + b.v5, a.v6 * fade + b.v6, a.v7 * fade + b.v7, a.v8 * fade + b.v8, a.v9 * fade + b.v9, a.v10 * fade + b.v10, a.v11 * fade + b.v11, a.v12 * fade + b.v12, a.v13 * fade + b.v13, a.v14 * fade + b.v14, a.v15 * fade + b.v15);
}

// A query's online softmax over heads of 64 so far: the largest score,
// the sum of exp(score - largest), and the values so weighted.
struct Soft64 {
    best: f32,
    total: f32,
    acc: H64,
}

fn soft64() -> Soft64 {
    return Soft64(-1e30, 0.0, h64_zero());
}

// The softmax with a key's score s and its values v taken in.
fn soft64_take(st: Soft64, s: f32, v: H64) -> Soft64 {
    if (s > st.best) {
        // A new maximum: rescale what was summed so far.
        let fade = exp(st.best - s);
        return Soft64(s, st.total * fade + 1.0, h64_fade(st.acc, fade, v));
    }
    let e = exp(s - st.best);
    return Soft64(st.best, st.total + e, h64_add(st.acc, e, v));
}

// a + e · b, fused (the online softmax's other scores).
fn h64_add(a: H64, e: f32, b: H64) -> H64 {
    let k = vec4<f32>(e);
    return H64(fma(k, b.v0, a.v0), fma(k, b.v1, a.v1), fma(k, b.v2, a.v2), fma(k, b.v3, a.v3), fma(k, b.v4, a.v4), fma(k, b.v5, a.v5), fma(k, b.v6, a.v6), fma(k, b.v7, a.v7), fma(k, b.v8, a.v8), fma(k, b.v9, a.v9), fma(k, b.v10, a.v10), fma(k, b.v11, a.v11), fma(k, b.v12, a.v12), fma(k, b.v13, a.v13), fma(k, b.v14, a.v14), fma(k, b.v15, a.v15));
}
