// The prelude of the picture models' kernels (the text encoder, the
// transformer and the autoencoder of FLUX.2 klein). Activations are f32,
// read and written four at a time where they lie side by side (`src`,
// `dst`); matrices are FP8 (e4m3, four a word) or bf16 (two a word) in
// `wts`, read sixteen or eight values at a time; `small` holds the f32
// tensors (norms, biases, scales, tables) and `src2` a second input. `p`
// holds a dispatch's numbers, named per kernel.

struct Params {
    n0: u32, n1: u32, n2: u32, n3: u32,
    n4: u32, n5: u32, n6: u32, n7: u32,
    n8: u32, n9: u32, n10: u32, n11: u32,
    n12: u32, n13: u32, n14: u32, n15: u32,
    n16: u32, n17: u32, n18: u32, n19: u32,
    n20: u32, n21: u32, n22: u32, n23: u32,
    n24: u32, n25: u32, n26: u32, n27: u32,
    n28: u32, n29: u32, n30: u32, n31: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wts: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> small: array<f32>;
@group(0) @binding(5) var<storage, read> src2: array<f32>;

// The f32 at `i` of `src` (or `dst`), and one written to `dst`.
fn src_at(i: u32) -> f32 {
    return src[i >> 2u][i & 3u];
}

fn dst_at(i: u32) -> f32 {
    return dst[i >> 2u][i & 3u];
}

fn dst_put(i: u32, v: f32) {
    dst[i >> 2u][i & 3u] = v;
}

// The invocation's place in a one-dimensional job spread over a grid of
// workgroups of 256.
fn flat_id(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

// One FP8 (e4m3) byte's value: its exponent rebased from 7 to f32's 127
// (a zero exponent is a subnormal, m · 2^-9).
fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let normal = bitcast<f32>(((b & 0x7fu) << 20u) + (120u << 23u));
    let v = select(normal, f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 0x80u) != 0u);
}

// A word's four FP8 values, its lowest byte first.
fn e4m3x4(w: u32) -> vec4<f32> {
    return vec4<f32>(e4m3(w & 0xffu), e4m3((w >> 8u) & 0xffu), e4m3((w >> 16u) & 0xffu), e4m3(w >> 24u));
}

// A word's two bf16 values, its low half first.
fn bf16x2(w: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}
