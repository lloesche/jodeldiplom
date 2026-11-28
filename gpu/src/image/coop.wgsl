// What the two tensor-core kernels (`gemm_coop.wgsl`, `attention_coop.wgsl`)
// share: their workgroup memory, and f16 written into it. They are written
// on tiles of 16 × 16: A (16 rows × 16 of k, f16), B (16 of k × 16 columns,
// f16) and C (16 × 16 f32 sums), loaded from and stored to these two arrays
// by the tile's layer (`coop16.wgsl`: one cooperative matrix of
// 16 × 16 × 16 a tile; `coop8.wgsl`: four of 8 × 8 × 8, where the GPU has
// only those, as Metal does). Both layers name the same types (`CmA`,
// `CmB`, `CmC`) and the multiply's functions (`cm_ld_at`, `cm_ld_b`,
// `cm_mma`, `cm_st_c`, `cm_ld_c`; on 16 × 16 `cm_mma` is the matrices' own
// product, written in its place), so the multiply is the same on both; the
// attention on 16 × 16 is written on these tiles too, Apple's 8 × 8 has
// one of its own (`attention8.wgsl`, `Tiles::attention`).
// The layer also sets the step of k the multiply takes (`CM_K`: 64, or 32
// where the workgroup memory is Apple's 32 KB), so the size of `cm_h`
// (`CM_H`): on 8 × 8 the multiply takes 20 KB of f16 and 9 KB of f32; on
// 16 × 16, 36 KB of f16, and the attention 1.5 KB more of its own
// (`the_kernels_validate` sums each kernel's against `Tiles::memory`).

// A row of the multiply's tiles (the step and 8 of padding), and the f16
// tiles' size: A's 128 rows and W's.
const CM_S: u32 = CM_K + 8u;
const CM_H: u32 = 256u * CM_S;

// The f16 tiles: the multiply's A (from 0) and W (from `CM_B`), 128 rows
// of `CM_S`; the attention's queries, keys (then the exps) and values.
var<workgroup> cm_h: array<f16, CM_H>;
// Each subgroup's 16 × 16 of f32 (sums on their way out, scores, a
// rescale), then 256 zeros (`CM_ZERO`) a tile of C starts from.
var<workgroup> cm_f: array<f32, 2304>;
const CM_ZERO: u32 = 2048u;

// The zeros written, one an invocation (a workgroup of 256): every
// invocation calls it, and a barrier follows before [`cm_zero`].
fn cm_zero_init(li: u32) {
    cm_f[CM_ZERO + li] = 0.0;
}

// A tile of C of zeros, loaded from the zeros (not WGSL's zero value of a
// cooperative matrix, which Metal's writer leaves to the type's default
// constructor).
fn cm_zero() -> CmC {
    return cm_ld_c(CM_ZERO, 16u);
}

// One f32 to f16, clamped to its range.
fn cm_half(v: f32) -> f16 {
    return f16(clamp(v, -65504.0, 65504.0));
}

// Four f32s written at `at` as f16s, each clamped to f16's range first
// (`at` a multiple of 4: whole words, one invocation's).
fn cm_put_held4(at: u32, v: vec4<f32>) {
    cm_h[at] = cm_half(v.x);
    cm_h[at + 1u] = cm_half(v.y);
    cm_h[at + 2u] = cm_half(v.z);
    cm_h[at + 3u] = cm_half(v.w);
}
