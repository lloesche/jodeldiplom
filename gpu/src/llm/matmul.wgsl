// The language model's matrix multiplies: a row (a token being decoded)
// through a 4-bit matrix (`gemv_*`), and many rows (a prompt) through one
// (`gemm_*`); each on the q8 path (the rows at 8 bits, four multiply-adds
// an instruction with dot4I8Packed: a group of 32 is
// `d_w·d_a·Σ q_w·q_a + m_w·d_a·Σ q_a`, the weights' scale and offset with
// the row's scale and its block's scaled sum) or the plain one (the rows
// as f32s, the weights unpacked to f32s).
//
// Each takes the engine's numbers of a matrix multiply (`Mul::numbers`:
// m, n, k, the rows' place in `ints` and their values a row, the matrix's
// place in `wq`, where the products go and their rows' stride, no bias,
// the flags), then the rows' scales in `ints` and (the head) where its
// workgroups' largest go. What becomes of the products, as the flags say
// (`After`): written; GELU taken; the rows taken in pairs, gate and up,
// gelu(gate) · up written at the pair's index; or, the head, capped
// (30 · tanh(x / 30)) and written, each workgroup's largest kept with its
// row.

const GELU: u32 = 4u;
const ARGMAX: u32 = 128u;
const GEGLU: u32 = 512u;
const LOW: u32 = 0x0F0F0F0Fu;

// The products of a 4-bit half-group (two words, 16 values) and 16 int8
// values in four words (even and odd values of each eight).
fn dot16(w: vec2<u32>, x0: u32, x1: u32, x2: u32, x3: u32) -> i32 {
    return dot4I8Packed(w.x & LOW, x0) + dot4I8Packed((w.x >> 4u) & LOW, x1)
        + dot4I8Packed(w.y & LOW, x2) + dot4I8Packed((w.y >> 4u) & LOW, x3);
}

// A row through a matrix: a workgroup takes 16 rows, eight slots of two,
// each slot's 32 invocations taking every 32nd half-group of its rows'
// inputs (the half-groups neighbours read side by side), their sums then
// added in their order.
var<workgroup> gemv_sum0: array<f32, 256>;
var<workgroup> gemv_sum1: array<f32, 256>;
var<workgroup> best_value: array<f32, 8>;
var<workgroup> best_row: array<u32, 8>;

@compute @workgroup_size(256)
fn gemv_q8(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = p.f1;
    let groups = p.f2 / 32u;
    let r0 = wid.x * 16u + (lid / 32u) * 2u;
    let nib0 = p.f5 + r0 * groups * 4u;
    let nib1 = nib0 + groups * 4u;
    let sc0 = p.f5 + n * groups * 4u + r0 * groups;
    let sc1 = sc0 + groups;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var h = lid % 32u; h < 2u * groups; h += 32u) {
        let g = h >> 1u;
        let half = h & 1u;
        let a = p.f3 + g * 8u + half * 4u;
        let x0 = ints[a];
        let x1 = ints[a + 1u];
        let x2 = ints[a + 2u];
        let x3 = ints[a + 3u];
        let da = bitcast<f32>(ints[p.f10 + g * 2u]);
        // The offset's term once a group.
        let sa = select(0.0, bitcast<f32>(ints[p.f10 + g * 2u + 1u]), half == 0u);
        let w0 = wq[(nib0 + g * 4u + half * 2u) >> 1u];
        let w1 = wq[(nib1 + g * 4u + half * 2u) >> 1u];
        let s0 = scale_of(sc0 + g);
        let s1 = scale_of(sc1 + g);
        acc0 += s0.x * da * f32(dot16(w0, x0, x1, x2, x3)) + s0.y * sa;
        acc1 += s1.x * da * f32(dot16(w1, x0, x1, x2, x3)) + s1.y * sa;
    }
    gemv_finish(lid, wid.x, r0, acc0, acc1);
}

// gemv_q8 on the plain path.
@compute @workgroup_size(256)
fn gemv_f32(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = p.f1;
    let groups = p.f2 / 32u;
    let r0 = wid.x * 16u + (lid / 32u) * 2u;
    let nib0 = p.f5 + r0 * groups * 4u;
    let nib1 = nib0 + groups * 4u;
    let sc0 = p.f5 + n * groups * 4u + r0 * groups;
    let sc1 = sc0 + groups;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var h = lid % 32u; h < 2u * groups; h += 32u) {
        let g = h >> 1u;
        let half = h & 1u;
        let a = p.f3 + h * 16u;
        let w0 = wq[(nib0 + g * 4u + half * 2u) >> 1u];
        let w1 = wq[(nib1 + g * 4u + half * 2u) >> 1u];
        var q0 = 0.0;
        var q1 = 0.0;
        var sum = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let xa = bitcast<f32>(ints[a + i]);
            let xb = bitcast<f32>(ints[a + 8u + i]);
            let shift = 4u * i;
            q0 += f32((w0.x >> shift) & 15u) * xa + f32((w0.y >> shift) & 15u) * xb;
            q1 += f32((w1.x >> shift) & 15u) * xa + f32((w1.y >> shift) & 15u) * xb;
            sum += xa + xb;
        }
        let s0 = scale_of(sc0 + g);
        let s1 = scale_of(sc1 + g);
        acc0 += s0.x * q0 + s0.y * sum;
        acc1 += s1.x * q1 + s1.y * sum;
    }
    gemv_finish(lid, wid.x, r0, acc0, acc1);
}

// The slots' sums added in their order, and the tail.
fn gemv_finish(lid: u32, group: u32, r0: u32, acc0: f32, acc1: f32) {
    gemv_sum0[lid] = acc0;
    gemv_sum1[lid] = acc1;
    workgroupBarrier();
    let slot = lid / 32u;
    let flags = p.f9;
    if (lid % 32u == 0u) {
        var v0 = 0.0;
        var v1 = 0.0;
        for (var i = 0u; i < 32u; i++) {
            v0 += gemv_sum0[lid + i];
            v1 += gemv_sum1[lid + i];
        }
        let out = p.f6;
        if ((flags & GEGLU) != 0u) {
            dst[out + r0 / 2u] = gelu_tanh(v0) * v1;
        } else if ((flags & GELU) != 0u) {
            dst[out + r0] = gelu_tanh(v0);
            dst[out + r0 + 1u] = gelu_tanh(v1);
        } else if ((flags & ARGMAX) != 0u) {
            let c0 = tanh(v0 / 30.0) * 30.0;
            let c1 = tanh(v1 / 30.0) * 30.0;
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
            ints[p.f11 + group * 2u] = bitcast<u32>(v);
            ints[p.f11 + group * 2u + 1u] = r;
        }
    }
}

// Many rows through a matrix: tiles of 64 rows × 64 columns, each
// invocation 4 rows (16 apart) × 4 columns (two pairs: 2x, 2x + 1 and
// 32 on), a group of 32 inputs at a time.
var<workgroup> tile_rows: array<vec4<u32>, 128>;
var<workgroup> tile_cols: array<vec4<u32>, 128>;
var<workgroup> tile_row_scale: array<vec2<f32>, 64>;
var<workgroup> tile_col_scale: array<vec2<f32>, 64>;

// The products of 32 int8 values (`a`, `b`) and 32 unpacked 4-bit ones
// (`w`, `x`), each eight's even and odd values in alternate words.
fn dot32(a: vec4<u32>, b: vec4<u32>, w: vec4<u32>, x: vec4<u32>) -> f32 {
    let s = dot4I8Packed(a.x, w.x) + dot4I8Packed(a.y, w.y) + dot4I8Packed(a.z, w.z) + dot4I8Packed(a.w, w.w)
        + dot4I8Packed(b.x, x.x) + dot4I8Packed(b.y, x.y) + dot4I8Packed(b.z, x.z) + dot4I8Packed(b.w, x.w);
    return f32(s);
}

// The invocation's four columns in the tile.
fn tile_col(tx: u32, j: u32) -> u32 {
    return 2u * tx + (j & 1u) + 32u * (j >> 1u);
}

// A row of the tile (`r`) through the invocation's four columns.
fn gemm_row_q8(acc: vec4<f32>, r: u32, tx: u32, sw: vec4<f32>, mw: vec4<f32>) -> vec4<f32> {
    let a = tile_rows[r * 2u];
    let b = tile_rows[r * 2u + 1u];
    let s = tile_row_scale[r];
    let c0 = tile_col(tx, 0u);
    let c1 = tile_col(tx, 1u);
    let c2 = tile_col(tx, 2u);
    let c3 = tile_col(tx, 3u);
    let i = vec4<f32>(
        dot32(a, b, tile_cols[c0 * 2u], tile_cols[c0 * 2u + 1u]),
        dot32(a, b, tile_cols[c1 * 2u], tile_cols[c1 * 2u + 1u]),
        dot32(a, b, tile_cols[c2 * 2u], tile_cols[c2 * 2u + 1u]),
        dot32(a, b, tile_cols[c3 * 2u], tile_cols[c3 * 2u + 1u]),
    );
    return acc + sw * s.x * i + mw * s.y;
}

@compute @workgroup_size(256)
fn gemm_q8(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = p.f0;
    let n = p.f1;
    let groups = p.f2 / 32u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let tx = lid % 16u;
    let ty = lid / 16u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var g = 0u; g < groups; g++) {
        if (lid < 128u) {
            let row = row0 + lid / 2u;
            var v = vec4<u32>(0u);
            if (row < m) {
                let a = p.f3 + (row * groups + g) * 8u + (lid % 2u) * 4u;
                v = vec4<u32>(ints[a], ints[a + 1u], ints[a + 2u], ints[a + 3u]);
            }
            tile_rows[lid] = v;
        } else if (lid < 192u) {
            let c = lid - 128u;
            let at = (p.f5 + ((col0 + c) * groups + g) * 4u) >> 1u;
            let w01 = wq[at];
            let w23 = wq[at + 1u];
            tile_cols[c * 2u] = vec4<u32>(w01.x & LOW, (w01.x >> 4u) & LOW, w01.y & LOW, (w01.y >> 4u) & LOW);
            tile_cols[c * 2u + 1u] = vec4<u32>(w23.x & LOW, (w23.x >> 4u) & LOW, w23.y & LOW, (w23.y >> 4u) & LOW);
        } else {
            let i = lid - 192u;
            let row = row0 + i;
            var s = vec2<f32>(0.0);
            if (row < m) {
                let at = p.f10 + (row * groups + g) * 2u;
                s = vec2<f32>(bitcast<f32>(ints[at]), bitcast<f32>(ints[at + 1u]));
            }
            tile_row_scale[i] = s;
            tile_col_scale[i] = scale_of(p.f5 + n * groups * 4u + (col0 + i) * groups + g);
        }
        workgroupBarrier();
        let w0 = tile_col_scale[tile_col(tx, 0u)];
        let w1 = tile_col_scale[tile_col(tx, 1u)];
        let w2 = tile_col_scale[tile_col(tx, 2u)];
        let w3 = tile_col_scale[tile_col(tx, 3u)];
        let sw = vec4<f32>(w0.x, w1.x, w2.x, w3.x);
        let mw = vec4<f32>(w0.y, w1.y, w2.y, w3.y);
        acc0 = gemm_row_q8(acc0, ty, tx, sw, mw);
        acc1 = gemm_row_q8(acc1, ty + 16u, tx, sw, mw);
        acc2 = gemm_row_q8(acc2, ty + 32u, tx, sw, mw);
        acc3 = gemm_row_q8(acc3, ty + 48u, tx, sw, mw);
        workgroupBarrier();
    }
    gemm_put(row0 + ty, col0 + 2u * tx, acc0);
    gemm_put(row0 + ty + 16u, col0 + 2u * tx, acc1);
    gemm_put(row0 + ty + 32u, col0 + 2u * tx, acc2);
    gemm_put(row0 + ty + 48u, col0 + 2u * tx, acc3);
}

// gemm_q8 on the plain path: the tiles as f32s, inputs by rows.
var<workgroup> tile_a: array<f32, 2048>;
var<workgroup> tile_w: array<f32, 2048>;

@compute @workgroup_size(256)
fn gemm_f32(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = p.f0;
    let n = p.f1;
    let k = p.f2;
    let groups = k / 32u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let tx = lid % 16u;
    let ty = lid / 16u;
    let c0 = tile_col(tx, 0u);
    let c1 = tile_col(tx, 1u);
    let c2 = tile_col(tx, 2u);
    let c3 = tile_col(tx, 3u);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var g = 0u; g < groups; g++) {
        for (var s = 0u; s < 8u; s++) {
            let idx = lid + 256u * s;
            let r = idx / 32u;
            let kk = idx % 32u;
            var v = 0.0;
            if (row0 + r < m) {
                v = bitcast<f32>(ints[p.f3 + (row0 + r) * k + g * 32u + kk]);
            }
            tile_a[kk * 64u + r] = v;
        }
        let c = lid / 4u;
        let part = lid % 4u;
        let grp = (col0 + c) * groups + g;
        let nibbles = word(p.f5 + grp * 4u + part);
        let sc = scale_of(p.f5 + n * groups * 4u + grp);
        for (var j = 0u; j < 8u; j++) {
            tile_w[(part * 8u + j) * 64u + c] = sc.y + f32((nibbles >> (4u * j)) & 15u) * sc.x;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let w = vec4<f32>(tile_w[kk * 64u + c0], tile_w[kk * 64u + c1], tile_w[kk * 64u + c2], tile_w[kk * 64u + c3]);
            acc0 += tile_a[kk * 64u + ty] * w;
            acc1 += tile_a[kk * 64u + ty + 16u] * w;
            acc2 += tile_a[kk * 64u + ty + 32u] * w;
            acc3 += tile_a[kk * 64u + ty + 48u] * w;
        }
        workgroupBarrier();
    }
    gemm_put(row0 + ty, col0 + 2u * tx, acc0);
    gemm_put(row0 + ty + 16u, col0 + 2u * tx, acc1);
    gemm_put(row0 + ty + 32u, col0 + 2u * tx, acc2);
    gemm_put(row0 + ty + 48u, col0 + 2u * tx, acc3);
}

// A row's four products (columns `col`, `col + 1`, `col + 32`,
// `col + 33`) as the flags say (the head takes one row: a GEMV).
fn gemm_put(row: u32, col: u32, v: vec4<f32>) {
    if (row >= p.f0) {
        return;
    }
    let out = p.f6 + row * p.f7;
    let flags = p.f9;
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
