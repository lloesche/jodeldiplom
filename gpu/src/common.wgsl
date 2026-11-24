// What every model's kernels share, compiled after the model's own prelude
// (its parameters and bindings) and before its kernels: nothing here reads
// a binding.

// The invocation's place in a one-dimensional job spread over a grid of
// workgroups of 256 (a dispatch has at most 65535 groups per side).
fn linear_id(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
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
