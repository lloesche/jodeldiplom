// The separator's kernels. Every one reads `src` and `aux` and writes
// `dst`; `p` holds its numbers (named per kernel below). Buffers hold f32,
// tokens of features laid out one after the other.

struct Params {
    f0: u32, f1: u32, f2: u32, f3: u32,
    f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// The invocation's place in a one-dimensional job spread over a grid of
// workgroups of 256 (a dispatch has at most 65535 groups per side).
fn linear_id(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 256u;
}

// ---- dst[c_off + r*ldc + c] (+)= Σk src[a_off + r*lda + k] · aux[w_off + k*n + c] (+ aux[bias_off + c])
// f0 m, f1 n, f2 k, f3 a_off, f4 lda, f5 w_off, f6 c_off, f7 ldc, f8 bias_off,
// f9 flags (1: add the bias, 2: add to dst). Tiles of 64 × 64 through k in
// steps of 16; each invocation keeps 4 rows × 4 columns in four vectors.
// The tiles hold vectors: a step of k reads one of A (its four rows) and
// one of B (its four columns) for 16 multiply-adds, and the 16 steps are
// written out (no loop for the compiler to unroll, which Apple's does not
// always do with the loops wgpu writes for Metal).
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

// A's value at row r, column ka (0 outside the matrix).
fn a_at(r: u32, ka: u32) -> f32 {
    if (r < p.f0 && ka < p.f2) {
        return src[p.f3 + r * p.f4 + ka];
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = p.f1;
    let k = p.f2;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let t = lid.y * 16u + lid.x;
    var acc = Acc(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    // This invocation's loads: A rows 4·(t/16).. at k t%16 (neighbours read
    // neighbouring k of a row); B k t/16, columns (t%16)*4..
    let ar = row0 + (t / 16u) * 4u;
    let aq = t % 16u;
    let bk = t / 16u;
    let bc = col0 + (t % 16u) * 4u;
    for (var k0 = 0u; k0 < k; k0 += 16u) {
        let ka = k0 + aq;
        // Stored [k][row / 4], four rows in a vector.
        tile_a[aq * 16u + t / 16u] = vec4<f32>(
            a_at(ar, ka),
            a_at(ar + 1u, ka),
            a_at(ar + 2u, ka),
            a_at(ar + 3u, ka),
        );
        let kb = k0 + bk;
        var b = vec4<f32>(0.0);
        if (kb < k) {
            let at = p.f5 + kb * n + bc;
            if (bc + 3u < n) {
                b = vec4<f32>(aux[at], aux[at + 1u], aux[at + 2u], aux[at + 3u]);
            } else {
                for (var j = 0u; j < 4u; j++) {
                    if (bc + j < n) {
                        b[j] = aux[at + j];
                    }
                }
            }
        }
        tile_b[t] = b;
        workgroupBarrier();
        let lx = lid.x;
        let ly = lid.y;
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
        workgroupBarrier();
    }
    let r0 = row0 + lid.y * 4u;
    let c0 = col0 + lid.x * 4u;
    store_row(r0, c0, acc.r0);
    store_row(r0 + 1u, c0, acc.r1);
    store_row(r0 + 2u, c0, acc.r2);
    store_row(r0 + 3u, c0, acc.r3);
}

// Writes four outputs of row r from column c0 on (those inside the matrix).
fn store_row(r: u32, c0: u32, v: vec4<f32>) {
    if (r >= p.f0) {
        return;
    }
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < p.f1) {
            var x = v[j];
            if ((p.f9 & 1u) != 0u) {
                x += aux[p.f8 + c];
            }
            let at = p.f6 + r * p.f7 + c;
            if ((p.f9 & 2u) != 0u) {
                dst[at] += x;
            } else {
                dst[at] = x;
            }
        }
    }
}

// ---- RMSNorm: dst[y_off + r*ldy + d] = x / sqrt(mean(x²) + 1e-6) · aux[w_off + d]
// with x = src[x_off + r*ldx + d]; with f7 = 1 instead x · √dim / max(|x|, 1e-12)
// · aux[w_off + d] (the beat network's). f0 rows, f1 dim, f2 x_off, f3 ldx,
// f4 w_off, f5 y_off, f6 ldy, f7 mode. One row per invocation.
@compute @workgroup_size(256)
fn rmsnorm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = linear_id(gid, groups);
    if (r >= p.f0) {
        return;
    }
    let dim = p.f1;
    let x = p.f2 + r * p.f3;
    var sum = 0.0;
    for (var d = 0u; d < dim; d++) {
        let v = src[x + d];
        sum += v * v;
    }
    var k = inverseSqrt(sum / f32(dim) + 1e-6);
    if (p.f7 == 1u) {
        k = sqrt(f32(dim)) / max(sqrt(sum), 1e-12);
    }
    let y = p.f5 + r * p.f6;
    for (var d = 0u; d < dim; d++) {
        dst[y + d] = src[x + d] * k * aux[p.f4 + d];
    }
}

// ---- Rotary positions, in place in dst, halves paired: value i and
// i + 16 of a head turn by pair i's angle, from the table in src
// ([position][pair][cos, sin]); with f6 = 1 neighbours paired instead
// (values 2i and 2i + 1, the beat network's). f0 tokens, f1 len (positions
// run 0..len along each sequence), f2 turned (positions from it on stay),
// f3 heads, f4 width (values per token), f5 col_off (the first head's
// column), f6 mode.
@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = linear_id(gid, groups);
    let per_token = p.f3 * 16u;
    if (idx >= p.f0 * per_token) {
        return;
    }
    let token = idx / per_token;
    let h = (idx % per_token) / 16u;
    let i = idx % 16u;
    let position = token % p.f1;
    if (position >= p.f2) {
        return;
    }
    let base = token * p.f4 + p.f5 + h * 32u;
    var ia = base + i;
    var ib = base + i + 16u;
    if (p.f6 == 1u) {
        ia = base + 2u * i;
        ib = ia + 1u;
    }
    let a = dst[ia];
    let b = dst[ib];
    let c = src[(position * 16u + i) * 2u];
    let s = src[(position * 16u + i) * 2u + 1u];
    dst[ia] = a * c - b * s;
    dst[ib] = b * c + a * s;
}

// ---- Attention, 8 query heads on 4 key/value heads of 32. src: queries
// [token][256]; aux: keys and values [token][256] (keys in columns 0..128,
// values in 128..256); dst: [token][256]. f0 len (tokens per sequence).
// Workgroups: (queries / 64, heads, sequences); an online softmax over
// tiles of 64 keys, so the scores are never stored. A head is held as
// eight vectors of four, each a variable of its own (q0…q7, the values'
// sum a0…a7): an array indexed in a loop may be kept in memory rather
// than registers (Apple's GPUs, from the loops wgpu writes for Metal),
// and this loop runs for every query and key.
var<workgroup> keys: array<vec4<f32>, 512>;
var<workgroup> vals: array<vec4<f32>, 512>;

fn src4(i: u32) -> vec4<f32> {
    return vec4<f32>(src[i], src[i + 1u], src[i + 2u], src[i + 3u]);
}

fn store4(i: u32, v: vec4<f32>) {
    dst[i] = v.x;
    dst[i + 1u] = v.y;
    dst[i + 2u] = v.z;
    dst[i + 3u] = v.w;
}

@compute @workgroup_size(64)
fn attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let seq = wid.z;
    let h = wid.y;
    let kv = h / 2u;
    let qp = wid.x * 64u + lid.x;
    let valid = qp < len;
    let scale = inverseSqrt(32.0);
    var q0 = vec4<f32>(0.0);
    var q1 = vec4<f32>(0.0);
    var q2 = vec4<f32>(0.0);
    var q3 = vec4<f32>(0.0);
    var q4 = vec4<f32>(0.0);
    var q5 = vec4<f32>(0.0);
    var q6 = vec4<f32>(0.0);
    var q7 = vec4<f32>(0.0);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);
    if (valid) {
        let at = (seq * len + qp) * 256u + h * 32u;
        q0 = src4(at) * scale;
        q1 = src4(at + 4u) * scale;
        q2 = src4(at + 8u) * scale;
        q3 = src4(at + 12u) * scale;
        q4 = src4(at + 16u) * scale;
        q5 = src4(at + 20u) * scale;
        q6 = src4(at + 24u) * scale;
        q7 = src4(at + 28u) * scale;
    }
    var best = -1e30;
    var total = 0.0;
    for (var k0 = 0u; k0 < len; k0 += 64u) {
        let kp = k0 + lid.x;
        if (kp < len) {
            let at = (seq * len + kp) * 256u + kv * 32u;
            for (var d = 0u; d < 8u; d++) {
                let i = at + 4u * d;
                keys[lid.x * 8u + d] = vec4<f32>(aux[i], aux[i + 1u], aux[i + 2u], aux[i + 3u]);
                let j = i + 128u;
                vals[lid.x * 8u + d] = vec4<f32>(aux[j], aux[j + 1u], aux[j + 2u], aux[j + 3u]);
            }
        }
        workgroupBarrier();
        let count = min(64u, len - k0);
        if (valid) {
            for (var j = 0u; j < count; j++) {
                let kb = j * 8u;
                // In the order the loop over the vectors had.
                var s = dot(q0, keys[kb]);
                s += dot(q1, keys[kb + 1u]);
                s += dot(q2, keys[kb + 2u]);
                s += dot(q3, keys[kb + 3u]);
                s += dot(q4, keys[kb + 4u]);
                s += dot(q5, keys[kb + 5u]);
                s += dot(q6, keys[kb + 6u]);
                s += dot(q7, keys[kb + 7u]);
                if (s > best) {
                    // A new maximum: rescale what was summed so far.
                    let fade = exp(best - s);
                    total = total * fade + 1.0;
                    a0 = a0 * fade + vals[kb];
                    a1 = a1 * fade + vals[kb + 1u];
                    a2 = a2 * fade + vals[kb + 2u];
                    a3 = a3 * fade + vals[kb + 3u];
                    a4 = a4 * fade + vals[kb + 4u];
                    a5 = a5 * fade + vals[kb + 5u];
                    a6 = a6 * fade + vals[kb + 6u];
                    a7 = a7 * fade + vals[kb + 7u];
                    best = s;
                } else {
                    let e = vec4<f32>(exp(s - best));
                    total += e.x;
                    a0 = fma(e, vals[kb], a0);
                    a1 = fma(e, vals[kb + 1u], a1);
                    a2 = fma(e, vals[kb + 2u], a2);
                    a3 = fma(e, vals[kb + 3u], a3);
                    a4 = fma(e, vals[kb + 4u], a4);
                    a5 = fma(e, vals[kb + 5u], a5);
                    a6 = fma(e, vals[kb + 6u], a6);
                    a7 = fma(e, vals[kb + 7u], a7);
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let at = (seq * len + qp) * 256u + h * 32u;
        store4(at, a0 / total);
        store4(at + 4u, a1 / total);
        store4(at + 8u, a2 / total);
        store4(at + 12u, a3 / total);
        store4(at + 16u, a4 / total);
        store4(at + 20u, a5 / total);
        store4(at + 24u, a6 / total);
        store4(at + 28u, a7 / total);
    }
}

// ---- Gated GELU, in place: dst[i] = gelu(dst[i]) · src[i]. f0 count.
// GELU with the error function as Abramowitz and Stegun's 7.1.26.
fn gelu(x: f32) -> f32 {
    let z = abs(x) * 0.70710678;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign(x) * (1.0 - poly * exp(-z * z));
    return 0.5 * x * (1.0 + erf);
}

@compute @workgroup_size(256)
fn geglu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    dst[i] = gelu(dst[i]) * src[i];
}

// ---- dst = src as tokens of 256 in [cols][rows] from [rows][cols].
// f0 rows, f1 cols.
@compute @workgroup_size(256)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let rows = p.f0;
    let cols = p.f1;
    if (i >= rows * cols * 256u) {
        return;
    }
    let d = i % 256u;
    let token = i / 256u;
    let r = token / cols;
    let c = token % cols;
    dst[(c * rows + r) * 256u + d] = src[i];
}
