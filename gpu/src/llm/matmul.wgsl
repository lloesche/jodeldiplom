// The language model's matrix multiplies: a row (a token being decoded)
// through a matrix (`gemv_*`), and many rows (a prompt) through one
// (`gemm_*`), each written for every kind of matrix it takes
// (`matmul_q8.wgsl`, `matmul_plain.wgsl`: `_q4k`, `_q6k`, …). On the q8
// path the rows are at 8 bits and the weights read as signed bytes, four
// multiply-adds an instruction with dot4I8Packed (16 values are
// `d_a·s·Σ q_w·q_a`, and each 32 adds `m·d_a·Σ q_a`: the row's scale, the
// weights' scale and minimum); on the plain one the rows are f32s and the
// weights read as f32s.
//
// Each takes the engine's numbers of a matrix multiply (`Mul::numbers`:
// m, n, k, the rows' place in `ints` and their values a row, the matrix's
// place in `wq` (in words), where the products go and their rows' stride,
// no bias, the flags), then the rows' scales in `ints` and (the head)
// where its workgroups' largest go. What becomes of the products, as the
// flags say (`After`): written; GELU taken; the rows taken in pairs, gate
// and up, gelu(gate) · up written at the pair's index; or, the head,
// capped (LOGIT_CAP · tanh(x / LOGIT_CAP)) and written, each workgroup's
// largest kept with its row (the flags GELU, GEGLU, ARGMAX the engine's).

// The products of 16 signed bytes and 16 others, four to a word each.
fn dot16(w: vec4<u32>, x: vec4<u32>) -> i32 {
    return dot4I8Packed(w.x, x.x) + dot4I8Packed(w.y, x.y) + dot4I8Packed(w.z, x.z) + dot4I8Packed(w.w, x.w);
}

// A row through a matrix: a workgroup takes 16 rows, eight slots of two,
// each slot's 32 invocations taking every 32nd 16 (q8) or 8 (plain) of
// its rows' inputs (neighbours reading side by side), their sums then
// added in their order.
var<workgroup> gemv_sum0: array<f32, 256>;
var<workgroup> gemv_sum1: array<f32, 256>;
var<workgroup> best_value: array<f32, 8>;
var<workgroup> best_row: array<u32, 8>;

// The slots' sums added in their order, and the tail.
fn gemv_finish(lid: u32, group: u32, r0: u32, acc0: f32, acc1: f32) {
    gemv_sum0[lid] = acc0;
    gemv_sum1[lid] = acc1;
    workgroupBarrier();
    let slot = lid / 32u;
    let flags = mm_flags();
    if (lid % 32u == 0u) {
        var v0 = 0.0;
        var v1 = 0.0;
        for (var i = 0u; i < 32u; i++) {
            v0 += gemv_sum0[lid + i];
            v1 += gemv_sum1[lid + i];
        }
        let out = mm_c();
        if ((flags & GEGLU) != 0u) {
            dst[out + r0 / 2u] = gelu_tanh(v0) * v1;
        } else if ((flags & GELU) != 0u) {
            dst[out + r0] = gelu_tanh(v0);
            dst[out + r0 + 1u] = gelu_tanh(v1);
        } else if ((flags & ARGMAX) != 0u) {
            let c0 = tanh_held(v0 / LOGIT_CAP) * LOGIT_CAP;
            let c1 = tanh_held(v1 / LOGIT_CAP) * LOGIT_CAP;
            dst[out + r0] = c0;
            dst[out + r0 + 1u] = c1;
            // The first of equals.
            let second = c1 > c0;
            best_value[slot] = select(c0, c1, second);
            best_row[slot] = select(r0, r0 + 1u, second);
        } else {
            dst[out + r0] = v0;
            dst[out + r0 + 1u] = v1;
        }
    }
    if ((flags & ARGMAX) != 0u) {
        workgroupBarrier();
        if (lid == 0u) {
            var v = best_value[0];
            var r = best_row[0];
            for (var i = 1u; i < 8u; i++) {
                if (best_value[i] > v) {
                    v = best_value[i];
                    r = best_row[i];
                }
            }
            ints[mm_best() + group * 2u] = bitcast<u32>(v);
            ints[mm_best() + group * 2u + 1u] = r;
        }
    }
}

// Many rows through a matrix: tiles of 64 rows × 64 columns, each
// invocation 4 rows (16 apart) × 4 columns (two pairs: 2x, 2x + 1 and
// 32 on), 32 inputs at a time.
var<workgroup> tile_rows: array<vec4<u32>, 128>;
var<workgroup> tile_cols: array<vec4<u32>, 128>;
var<workgroup> tile_row_scale: array<vec2<f32>, 64>;
// Each 16 of a column's 32: its scale and minimum (the second's none).
var<workgroup> tile_col_scale: array<vec2<f32>, 128>;

// The invocation's four columns in the tile.
fn tile_col(tx: u32, j: u32) -> u32 {
    return 2u * tx + (j & 1u) + 32u * (j >> 1u);
}

// A row of the tile (`r`) through the invocation's four columns, their
// 16s' scales `lo` and `hi` and their minimums `m`.
fn gemm_row_q8(acc: vec4<f32>, r: u32, tx: u32, lo: vec4<f32>, hi: vec4<f32>, m: vec4<f32>) -> vec4<f32> {
    let a = tile_rows[r * 2u];
    let b = tile_rows[r * 2u + 1u];
    let s = tile_row_scale[r];
    let c0 = tile_col(tx, 0u) * 2u;
    let c1 = tile_col(tx, 1u) * 2u;
    let c2 = tile_col(tx, 2u) * 2u;
    let c3 = tile_col(tx, 3u) * 2u;
    let first = vec4<f32>(
        f32(dot16(tile_cols[c0], a)),
        f32(dot16(tile_cols[c1], a)),
        f32(dot16(tile_cols[c2], a)),
        f32(dot16(tile_cols[c3], a)),
    );
    let second = vec4<f32>(
        f32(dot16(tile_cols[c0 + 1u], b)),
        f32(dot16(tile_cols[c1 + 1u], b)),
        f32(dot16(tile_cols[c2 + 1u], b)),
        f32(dot16(tile_cols[c3 + 1u], b)),
    );
    return acc + s.x * (lo * first + hi * second) + m * s.y;
}

// A tile's rows of 32 values from `g` at 8 bits (invocations 0 to 127,
// half a row each; the first 64 a row's scales too).
fn gemm_rows_q8(lid: u32, row0: u32, g: u32, groups: u32) {
    let m = mm_m();
    let row = row0 + lid / 2u;
    var v = vec4<u32>(0u);
    if (row < m) {
        let a = mm_a() + (row * groups + g) * 8u + (lid % 2u) * 4u;
        v = vec4<u32>(ints[a], ints[a + 1u], ints[a + 2u], ints[a + 3u]);
    }
    tile_rows[lid] = v;
    if (lid < 64u) {
        var s = vec2<f32>(0.0);
        if (row0 + lid < m) {
            let at = mm_qs() + ((row0 + lid) * groups + g) * 2u;
            s = vec2<f32>(bitcast<f32>(ints[at]), bitcast<f32>(ints[at + 1u]));
        }
        tile_row_scale[lid] = s;
    }
}

// The tile's products of the 32 inputs loaded (`tile_rows`, `tile_cols`).
fn gemm_step_q8(lid: u32, acc: ptr<function, array<vec4<f32>, 4>>) {
    let tx = lid % 16u;
    let ty = lid / 16u;
    let c0 = tile_col(tx, 0u) * 2u;
    let c1 = tile_col(tx, 1u) * 2u;
    let c2 = tile_col(tx, 2u) * 2u;
    let c3 = tile_col(tx, 3u) * 2u;
    let w0 = tile_col_scale[c0];
    let w1 = tile_col_scale[c1];
    let w2 = tile_col_scale[c2];
    let w3 = tile_col_scale[c3];
    let lo = vec4<f32>(w0.x, w1.x, w2.x, w3.x);
    let hi = vec4<f32>(tile_col_scale[c0 + 1u].x, tile_col_scale[c1 + 1u].x, tile_col_scale[c2 + 1u].x, tile_col_scale[c3 + 1u].x);
    let m = vec4<f32>(w0.y, w1.y, w2.y, w3.y);
    (*acc)[0] = gemm_row_q8((*acc)[0], ty, tx, lo, hi, m);
    (*acc)[1] = gemm_row_q8((*acc)[1], ty + 16u, tx, lo, hi, m);
    (*acc)[2] = gemm_row_q8((*acc)[2], ty + 32u, tx, lo, hi, m);
    (*acc)[3] = gemm_row_q8((*acc)[3], ty + 48u, tx, lo, hi, m);
}

// The plain path's tiles as f32s, inputs by rows.
var<workgroup> tile_x: array<f32, 2048>;
var<workgroup> tile_w: array<f32, 2048>;

// A tile's rows of 32 values from `g` as f32s (each invocation 8).
fn gemm_rows_f32(lid: u32, row0: u32, g: u32) {
    let m = mm_m();
    let k = mm_k();
    for (var s = 0u; s < 8u; s++) {
        let idx = lid + 256u * s;
        let r = idx / 32u;
        let kk = idx % 32u;
        var v = 0.0;
        if (row0 + r < m) {
            v = bitcast<f32>(ints[mm_a() + (row0 + r) * k + g * 32u + kk]);
        }
        tile_x[kk * 64u + r] = v;
    }
}

// 8 of a column's 32 values (from `8 · part`) into the tile.
fn gemm_put_w(c: u32, part: u32, e: Eight) {
    let at = part * 8u * 64u + c;
    tile_w[at] = e.a.x;
    tile_w[at + 64u] = e.a.y;
    tile_w[at + 128u] = e.a.z;
    tile_w[at + 192u] = e.a.w;
    tile_w[at + 256u] = e.b.x;
    tile_w[at + 320u] = e.b.y;
    tile_w[at + 384u] = e.b.z;
    tile_w[at + 448u] = e.b.w;
}

// The tile's products of the 32 inputs loaded (`tile_x`, `tile_w`).
fn gemm_step_f32(lid: u32, acc: ptr<function, array<vec4<f32>, 4>>) {
    let tx = lid % 16u;
    let ty = lid / 16u;
    let c0 = tile_col(tx, 0u);
    let c1 = tile_col(tx, 1u);
    let c2 = tile_col(tx, 2u);
    let c3 = tile_col(tx, 3u);
    for (var kk = 0u; kk < 32u; kk++) {
        let w = vec4<f32>(tile_w[kk * 64u + c0], tile_w[kk * 64u + c1], tile_w[kk * 64u + c2], tile_w[kk * 64u + c3]);
        (*acc)[0] += tile_x[kk * 64u + ty] * w;
        (*acc)[1] += tile_x[kk * 64u + ty + 16u] * w;
        (*acc)[2] += tile_x[kk * 64u + ty + 32u] * w;
        (*acc)[3] += tile_x[kk * 64u + ty + 48u] * w;
    }
}

// The tile's products written as the flags say.
fn gemm_finish(lid: u32, row0: u32, col0: u32, acc: array<vec4<f32>, 4>) {
    let tx = lid % 16u;
    let ty = lid / 16u;
    gemm_put(row0 + ty, col0 + 2u * tx, acc[0]);
    gemm_put(row0 + ty + 16u, col0 + 2u * tx, acc[1]);
    gemm_put(row0 + ty + 32u, col0 + 2u * tx, acc[2]);
    gemm_put(row0 + ty + 48u, col0 + 2u * tx, acc[3]);
}

// A row's four products (columns `col`, `col + 1`, `col + 32`,
// `col + 33`) as the flags say (the head takes one row: a GEMV).
fn gemm_put(row: u32, col: u32, v: vec4<f32>) {
    if (row >= mm_m()) {
        return;
    }
    let out = mm_c() + row * mm_ldc();
    let flags = mm_flags();
    if ((flags & GEGLU) != 0u) {
        dst[out + col / 2u] = gelu_tanh(v.x) * v.y;
        dst[out + col / 2u + 16u] = gelu_tanh(v.z) * v.w;
        return;
    }
    var w = v;
    if ((flags & GELU) != 0u) {
        w = vec4<f32>(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w));
    }
    dst[out + col] = w.x;
    dst[out + col + 1u] = w.y;
    dst[out + col + 32u] = w.z;
    dst[out + col + 33u] = w.w;
}
