// The kernels and helpers the separator, the beat network, the speech
// model and the recogniser share, compiled after `bindings.wgsl` (the
// bindings) and `common.wgsl` (GELU, the heads of 64): the tiled matrix
// multiply, RMSNorm, the rotary positions, a transpose and GELU in place;
// a row's sum over its lanes; the attention kernels' tiles of keys and
// values and their readers.

// ---- dst[c_off + r*ldc + c] (+)= Σk src[a_off + r*lda + k] · aux[w_off + k*n + c] (+ aux[bias_off + c])
// f0 m, f1 n, f2 k, f3 a_off, f4 lda, f5 w_off, f6 c_off, f7 ldc, f8 bias_off,
// f9 flags (1: add the bias, 2: add to dst, 512: the columns are gates and
// ups in pairs (gate j at 2j, its up at 2j + 1), each pair out as
// gelu(gate) · up at column j). Tiles of 128 × 64 through k in steps of
// 16, the next step's values loaded into registers while one is multiplied
// from workgroup memory; each invocation keeps 8 rows × 4 columns, each
// output's products added in the order of k. The tiles hold vectors: a
// step of k reads two of A (eight rows) and one of B (four columns) for
// 32 multiply-adds, each kept in a variable of its own (no array indexed
// in a loop: Apple's compiler may keep one in memory). A is stored [k][row
// / 4], B [k][col / 4].
var<workgroup> mm_a: array<vec4<f32>, 512>;
var<workgroup> mm_b: array<vec4<f32>, 256>;

// A's value at row r, column ka (0 outside the matrix).
fn a_at(r: u32, ka: u32) -> f32 {
    if (r < p.f0 && ka < p.f2) {
        return src[p.f3 + r * p.f4 + ka];
    }
    return 0.0;
}

// A's four rows from r at ka.
fn mm_rows4(r: u32, ka: u32) -> vec4<f32> {
    return vec4<f32>(a_at(r, ka), a_at(r + 1u, ka), a_at(r + 2u, ka), a_at(r + 3u, ka));
}

// B's four columns from c at row kb (0 outside the matrix).
fn mm_cols4(kb: u32, c: u32) -> vec4<f32> {
    let n = p.f1;
    var b = vec4<f32>(0.0);
    if (kb < p.f2) {
        let at = p.f5 + kb * n + c;
        if (c + 3u < n) {
            b = vec4<f32>(aux[at], aux[at + 1u], aux[at + 2u], aux[at + 3u]);
        } else {
            for (var j = 0u; j < 4u; j++) {
                if (c + j < n) {
                    b[j] = aux[at + j];
                }
            }
        }
    }
    return b;
}

// An invocation's share of a step's tiles: A's rows 4·q.. and 4·(q + 16)..
// at its k, and B's four columns at its row.
struct MmLoads {
    a0: vec4<f32>,
    a1: vec4<f32>,
    b: vec4<f32>,
}

fn mm_loads(k0: u32, row0: u32, col0: u32, t: u32) -> MmLoads {
    let ka = k0 + t % 16u;
    let r = row0 + 4u * (t / 16u);
    return MmLoads(mm_rows4(r, ka), mm_rows4(r + 64u, ka), mm_cols4(k0 + t / 16u, col0 + 4u * (t % 16u)));
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = p.f2;
    let row0 = wid.y * 128u;
    let col0 = wid.x * 64u;
    let t = lid.y * 16u + lid.x;
    var r0 = vec4<f32>(0.0);
    var r1 = vec4<f32>(0.0);
    var r2 = vec4<f32>(0.0);
    var r3 = vec4<f32>(0.0);
    var r4 = vec4<f32>(0.0);
    var r5 = vec4<f32>(0.0);
    var r6 = vec4<f32>(0.0);
    var r7 = vec4<f32>(0.0);
    var next = mm_loads(0u, row0, col0, t);
    for (var k0 = 0u; k0 < k; k0 += 16u) {
        mm_a[(t % 16u) * 32u + t / 16u] = next.a0;
        mm_a[(t % 16u) * 32u + t / 16u + 16u] = next.a1;
        mm_b[t] = next.b;
        workgroupBarrier();
        if (k0 + 16u < k) {
            next = mm_loads(k0 + 16u, row0, col0, t);
        }
        for (var kk = 0u; kk < 16u; kk++) {
            let a = mm_a[kk * 32u + 2u * lid.y];
            let a2 = mm_a[kk * 32u + 2u * lid.y + 1u];
            let b = mm_b[kk * 16u + lid.x];
            r0 = fma(vec4<f32>(a.x), b, r0);
            r1 = fma(vec4<f32>(a.y), b, r1);
            r2 = fma(vec4<f32>(a.z), b, r2);
            r3 = fma(vec4<f32>(a.w), b, r3);
            r4 = fma(vec4<f32>(a2.x), b, r4);
            r5 = fma(vec4<f32>(a2.y), b, r5);
            r6 = fma(vec4<f32>(a2.z), b, r6);
            r7 = fma(vec4<f32>(a2.w), b, r7);
        }
        workgroupBarrier();
    }
    let rr = row0 + lid.y * 8u;
    let cc = col0 + lid.x * 4u;
    store_row(rr, cc, r0);
    store_row(rr + 1u, cc, r1);
    store_row(rr + 2u, cc, r2);
    store_row(rr + 3u, cc, r3);
    store_row(rr + 4u, cc, r4);
    store_row(rr + 5u, cc, r5);
    store_row(rr + 6u, cc, r6);
    store_row(rr + 7u, cc, r7);
}

// Writes four outputs of row r from column c0 on (those inside the
// matrix), or their two pairs' gated products.
fn store_row(r: u32, c0: u32, v: vec4<f32>) {
    if (r >= p.f0) {
        return;
    }
    if ((p.f9 & 512u) != 0u) {
        let at = p.f6 + r * p.f7 + c0 / 2u;
        if (c0 + 1u < p.f1) {
            dst[at] = gelu(v.x) * v.y;
        }
        if (c0 + 3u < p.f1) {
            dst[at + 1u] = gelu(v.z) * v.w;
        }
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

// A row's sum over its 32 invocations (`t`'s row: the invocations of the
// workgroup's eight rows of 32 lanes, a row's neighbours), each giving
// `v`: in a tree (16 apart, then 8, ..., 1). Every invocation of the
// workgroup takes part (in uniform control flow).
var<workgroup> lane_sums: array<f32, 256>;

fn lane_sum(t: u32, v: f32) -> f32 {
    lane_sums[t] = v;
    workgroupBarrier();
    for (var half = 16u; half > 0u; half /= 2u) {
        if (t % 32u < half) {
            lane_sums[t] += lane_sums[t + half];
        }
        workgroupBarrier();
    }
    let total = lane_sums[t - t % 32u];
    workgroupBarrier();
    return total;
}

// ---- RMSNorm: dst[y_off + r*ldy + d] = x / sqrt(mean(x²) + 1e-6) · aux[w_off + d]
// with x = src[x_off + r*ldx + d]; with f7 = 1 instead x · √dim / max(|x|, 1e-12)
// · aux[w_off + d] (the beat network's). f0 rows, f1 dim, f2 x_off, f3 ldx,
// f4 w_off, f5 y_off, f6 ldy, f7 mode. A row is 32 invocations (32 a
// workgroup's invocations from linear_id: 32 · rows of them), each lane
// the values lane, lane + 32, ... (neighbouring lanes reading neighbouring
// values).
@compute @workgroup_size(256)
fn rmsnorm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = linear_id(gid, groups) / 32u;
    let lane = lid.x % 32u;
    let live = r < p.f0;
    let dim = p.f1;
    let x = p.f2 + r * p.f3;
    var sum = 0.0;
    if (live) {
        for (var d = lane; d < dim; d += 32u) {
            let v = src[x + d];
            sum += v * v;
        }
    }
    let total = lane_sum(lid.x, sum);
    if (!live) {
        return;
    }
    var k = inverseSqrt(total / f32(dim) + 1e-6);
    if (p.f7 == 1u) {
        k = sqrt(f32(dim)) / max(sqrt(total), 1e-12);
    }
    let y = p.f5 + r * p.f6;
    for (var d = lane; d < dim; d += 32u) {
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

// ---- The attention kernels' tiles: keys and values of a tile of a
// sequence, as vectors of four (64 keys of heads of 32, or 32 of heads of
// 64), and four values of src, aux or dst at a time.
var<workgroup> keys: array<vec4<f32>, 512>;
var<workgroup> vals: array<vec4<f32>, 512>;

fn aux4(i: u32) -> vec4<f32> {
    return vec4<f32>(aux[i], aux[i + 1u], aux[i + 2u], aux[i + 3u]);
}

fn src4(i: u32) -> vec4<f32> {
    return vec4<f32>(src[i], src[i + 1u], src[i + 2u], src[i + 3u]);
}

fn store4(i: u32, v: vec4<f32>) {
    dst[i] = v.x;
    dst[i + 1u] = v.y;
    dst[i + 2u] = v.z;
    dst[i + 3u] = v.w;
}

// A head of 64 values from the tile of keys (its 16 vectors from b).
fn key64(b: u32) -> H64 {
    return H64(keys[b + 0u], keys[b + 1u], keys[b + 2u], keys[b + 3u], keys[b + 4u], keys[b + 5u], keys[b + 6u], keys[b + 7u], keys[b + 8u], keys[b + 9u], keys[b + 10u], keys[b + 11u], keys[b + 12u], keys[b + 13u], keys[b + 14u], keys[b + 15u]);
}

// Its values from the tile of values.
fn val64(b: u32) -> H64 {
    return H64(vals[b + 0u], vals[b + 1u], vals[b + 2u], vals[b + 3u], vals[b + 4u], vals[b + 5u], vals[b + 6u], vals[b + 7u], vals[b + 8u], vals[b + 9u], vals[b + 10u], vals[b + 11u], vals[b + 12u], vals[b + 13u], vals[b + 14u], vals[b + 15u]);
}

// 64 values of src from at.
fn src64(at: u32) -> H64 {
    return H64(src4(at + 0u * 4u), src4(at + 1u * 4u), src4(at + 2u * 4u), src4(at + 3u * 4u), src4(at + 4u * 4u), src4(at + 5u * 4u), src4(at + 6u * 4u), src4(at + 7u * 4u), src4(at + 8u * 4u), src4(at + 9u * 4u), src4(at + 10u * 4u), src4(at + 11u * 4u), src4(at + 12u * 4u), src4(at + 13u * 4u), src4(at + 14u * 4u), src4(at + 15u * 4u));
}

// 64 values to dst from at.
fn store64(at: u32, v: H64) {
    store4(at + 0u, v.v0);
    store4(at + 4u, v.v1);
    store4(at + 8u, v.v2);
    store4(at + 12u, v.v3);
    store4(at + 16u, v.v4);
    store4(at + 20u, v.v5);
    store4(at + 24u, v.v6);
    store4(at + 28u, v.v7);
    store4(at + 32u, v.v8);
    store4(at + 36u, v.v9);
    store4(at + 40u, v.v10);
    store4(at + 44u, v.v11);
    store4(at + 48u, v.v12);
    store4(at + 52u, v.v13);
    store4(at + 56u, v.v14);
    store4(at + 60u, v.v15);
}

// A query of a head of 32 (scaled), as eight vectors.
struct Q32 {
    q0: vec4<f32>, q1: vec4<f32>, q2: vec4<f32>, q3: vec4<f32>,
    q4: vec4<f32>, q5: vec4<f32>, q6: vec4<f32>, q7: vec4<f32>,
}

// A query's online softmax so far: the largest score, the sum of
// exp(score - largest), and the values so weighted, eight vectors.
struct Soft32 {
    best: f32,
    total: f32,
    a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>,
    a4: vec4<f32>, a5: vec4<f32>, a6: vec4<f32>, a7: vec4<f32>,
}

// The query's score against the tile's key at kb, in the order the loop
// over the vectors had.
fn att_score(q: Q32, kb: u32) -> f32 {
    var s = dot(q.q0, keys[kb]);
    s += dot(q.q1, keys[kb + 1u]);
    s += dot(q.q2, keys[kb + 2u]);
    s += dot(q.q3, keys[kb + 3u]);
    s += dot(q.q4, keys[kb + 4u]);
    s += dot(q.q5, keys[kb + 5u]);
    s += dot(q.q6, keys[kb + 6u]);
    s += dot(q.q7, keys[kb + 7u]);
    return s;
}

// The softmax with the score s of the tile's key at kb taken in.
fn att_take(st: Soft32, s: f32, kb: u32) -> Soft32 {
    var o = st;
    if (s > st.best) {
        // A new maximum: rescale what was summed so far.
        let fade = exp(st.best - s);
        o.total = st.total * fade + 1.0;
        o.a0 = st.a0 * fade + vals[kb];
        o.a1 = st.a1 * fade + vals[kb + 1u];
        o.a2 = st.a2 * fade + vals[kb + 2u];
        o.a3 = st.a3 * fade + vals[kb + 3u];
        o.a4 = st.a4 * fade + vals[kb + 4u];
        o.a5 = st.a5 * fade + vals[kb + 5u];
        o.a6 = st.a6 * fade + vals[kb + 6u];
        o.a7 = st.a7 * fade + vals[kb + 7u];
        o.best = s;
    } else {
        let e = vec4<f32>(exp(s - st.best));
        o.total = st.total + e.x;
        o.a0 = fma(e, vals[kb], st.a0);
        o.a1 = fma(e, vals[kb + 1u], st.a1);
        o.a2 = fma(e, vals[kb + 2u], st.a2);
        o.a3 = fma(e, vals[kb + 3u], st.a3);
        o.a4 = fma(e, vals[kb + 4u], st.a4);
        o.a5 = fma(e, vals[kb + 5u], st.a5);
        o.a6 = fma(e, vals[kb + 6u], st.a6);
        o.a7 = fma(e, vals[kb + 7u], st.a7);
    }
    return o;
}

// ---- [a][b][c] to [b][a][c]. f0 a, f1 b, f2 c.
@compute @workgroup_size(256)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let a = p.f0;
    let b = p.f1;
    let c = p.f2;
    if (i >= a * b * c) {
        return;
    }
    let ch = i % c;
    let token = i / c;
    let ia = token / b;
    let jb = token % b;
    dst[(jb * a + ia) * c + ch] = src[i];
}

// ---- In place: dst[i] = gelu(dst[i]). f0 count.
@compute @workgroup_size(256)
fn gelu_in_place(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    dst[i] = gelu(dst[i]);
}
