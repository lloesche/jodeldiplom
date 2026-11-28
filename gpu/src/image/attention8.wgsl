// Attention on Apple's simdgroup matrices of 8 × 8 (`matrix8.wgsl`; the
// numbers of `attention.wgsl`), shaped as MLX's steel attention is for
// heads of 128: a workgroup of four subgroups of 32 takes 32 queries of a
// head, each subgroup 8 of them, and the keys and values come 16 at a
// time. A subgroup keeps only its 8 rows' 128 sums as sixteen f32
// matrices (32 values an invocation); the queries stay in workgroup
// memory and are read as they are multiplied. (Holding 16 rows, the
// queries and the sums as 32 matrices each, the 16 × 16 kernel's shape on
// 8 × 8 matrices took 3.3 s a dispatch on the M4 Max and gave nothing
// right.)
//
// Each step: the keys as they are (read by columns, as Kᵀ), the values
// by dims (`a8_pair`: two keys of one dim a word, so they too are read by
// columns); a subgroup's 8 × 16 scores stored by columns to its part of
// `a8_f`; each invocation then takes four scores of a row (row lane % 8,
// keys 4·(lane / 8)..), the row's largest is found through `a8_part`, and
// the online softmax goes on as the 16 × 16 kernel's: the largest moved
// only when a tile's is more than 8 above it (the exps stay below e⁸, in
// f16's range), the sums rescaled (through `a8_f`) only then; the exps
// written as f16 rows (`a8_p`), read by rows for their product with the
// values.

// `a8_h` (f16): the queries' 32 rows of `A8_QS` (128 values and 8 of
// padding), the keys' 16 rows of `A8_QS` from `A8_K`, the values by dims
// (128 rows of `A8_VS`: 16 keys and 8 of padding) from `A8_V`, each
// subgroup's 8 rows of exps (`A8_PS`: 16 keys and 8 of padding) from `A8_P`.
const A8_QS: u32 = 136u;
const A8_VS: u32 = 24u;
const A8_PS: u32 = 24u;
const A8_K: u32 = 32u * A8_QS;
const A8_V: u32 = A8_K + 16u * A8_QS;
const A8_P: u32 = A8_V + 128u * A8_VS;
const A8_H: u32 = A8_P + 4u * 8u * A8_PS;
// `a8_f` (f32): each subgroup's 8 × 16 scores (and an 8 × 8 of sums on
// their way out or rescaled), then 64 zeros a matrix of sums starts from.
const A8_ZERO: u32 = 4u * 128u;
// Every tile starts on a whole 16 bytes (8 f16s), and its rows too.
const_assert A8_K % 8u == 0u && A8_V % 8u == 0u && A8_P % 8u == 0u;
const_assert A8_QS % 8u == 0u && A8_VS % 8u == 0u && A8_PS % 8u == 0u;

var<workgroup> a8_h: array<f16, A8_H>;
var<workgroup> a8_f: array<f32, A8_ZERO + 64u>;
// Each invocation's largest score of the tile (then its share of its
// row's sum), each row's last rescale and its sum, and whether any row
// moved its largest this tile.
var<workgroup> a8_part: array<f32, 128>;
var<workgroup> a8_fade: array<f32, 32>;
var<workgroup> a8_sum: array<f32, 32>;
var<workgroup> a8_flag: u32;

// Four f32s written at `at` of `a8_h` as f16s (a multiple of 4: whole
// words, one invocation's).
fn a8_put4(at: u32, v: vec4<f32>) {
    a8_h[at] = f16(v.x);
    a8_h[at + 1u] = f16(v.y);
    a8_h[at + 2u] = f16(v.z);
    a8_h[at + 3u] = f16(v.w);
}

// Two keys' values of one dim written at `at` (even: a word, one
// invocation's).
fn a8_pair(at: u32, lo: f32, hi: f32) {
    a8_h[at] = f16(lo);
    a8_h[at + 1u] = f16(hi);
}

// A matrix of sums of zeros.
fn a8_zero() -> M8c {
    return m8_ld_c(A8_ZERO, 8u);
}

// The subgroup's 8 queries (from `qa`) times 8 keys (from row `k` of the
// keys' tile): their scores.
fn a8_scores(qa: u32, k: u32) -> M8c {
    var s = a8_zero();
    for (var j = 0u; j < 16u; j++) {
        s = m8_mma(m8_ld_a(qa + 8u * j, A8_QS), m8_ld_b(A8_K + k * A8_QS + 8u * j, A8_QS), s);
    }
    return s;
}

// The largest of `top` and the four scores of keys k.. that `query` sees.
fn a8_top4(top: f32, s: vec4<f32>, query: u32, k: u32) -> f32 {
    var t = top;
    if (fa_sees(query, k)) {
        t = max(t, s.x);
    }
    if (fa_sees(query, k + 1u)) {
        t = max(t, s.y);
    }
    if (fa_sees(query, k + 2u)) {
        t = max(t, s.z);
    }
    if (fa_sees(query, k + 3u)) {
        t = max(t, s.w);
    }
    return t;
}

// exp(score - best) of four scores of keys k.. (0 where `query` does not
// see the key).
fn a8_exp4(s: vec4<f32>, query: u32, k: u32, best: f32) -> vec4<f32> {
    var e = vec4<f32>(0.0);
    if (fa_sees(query, k)) {
        e.x = exp_held(s.x - best);
    }
    if (fa_sees(query, k + 1u)) {
        e.y = exp_held(s.y - best);
    }
    if (fa_sees(query, k + 2u)) {
        e.z = exp_held(s.z - best);
    }
    if (fa_sees(query, k + 3u)) {
        e.w = exp_held(s.w - best);
    }
    return e;
}

// One of the sums' matrices scaled per row by `a8_fade`, through the
// subgroup's part of `a8_f` (by columns: value (r, c) at c·8 + r; two an
// invocation).
fn a8_rescale(o: M8c, sg: u32, lane: u32) -> M8c {
    let at = sg * 128u;
    m8_st_c(o, at, 8u);
    workgroupBarrier();
    for (var j = 0u; j < 2u; j++) {
        let e = lane * 2u + j;
        a8_f[at + e] *= a8_fade[sg * 8u + e % 8u];
    }
    workgroupBarrier();
    let out = m8_ld_c(at, 8u);
    workgroupBarrier();
    return out;
}

// The exps' product with the values' 8 dims from 8·`m`, added to `o`.
fn a8_pv(o: M8c, p0: M8a, p1: M8a, m: u32) -> M8c {
    let v = A8_V + 8u * m * A8_VS;
    let a = m8_mma(p0, m8_ld_b(v, A8_VS), o);
    return m8_mma(p1, m8_ld_b(v + 8u, A8_VS), a);
}

// One of the sums' matrices (dims 8·`m`..) divided by the rows' sums
// (`a8_sum`) and written out.
fn a8_out(o: M8c, sg: u32, lane: u32, q0: u32, h: u32, m: u32) {
    let at = sg * 128u;
    m8_st_c(o, at, 8u);
    workgroupBarrier();
    for (var j = 0u; j < 2u; j++) {
        let e = lane * 2u + j;
        let r = e % 8u;
        let query = q0 + sg * 8u + r;
        if (query < p.n0) {
            dst_put(p.n7 + query * p.n8 + h * 128u + 8u * m + e / 8u, a8_f[at + e] / a8_sum[sg * 8u + r]);
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(128)
fn attention8(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let kvh = h / p.n3;
    let q0 = (wg.x + p.n12) * 32u;
    let sg = li / 32u;
    let lane = li % 32u;
    // This invocation's row and its four keys of each 16.
    let r = lane % 8u;
    let g = lane / 8u;
    let query = q0 + sg * 8u + r;
    let scale = bitcast<f32>(p.n9);
    if (li < 64u) {
        a8_f[A8_ZERO + li] = 0.0;
    }
    if (li == 0u) {
        a8_flag = 0u;
    }
    // The queries, scaled: 32 rows of 32 vectors, 8 an invocation.
    for (var j = 0u; j < 8u; j++) {
        let e = li + 128u * j;
        let row = e / 32u;
        var v = vec4<f32>(0.0);
        if (q0 + row < p.n0) {
            v = src[(p.n4 + (h * p.n0 + q0 + row) * 128u) / 4u + e % 32u] * scale;
        }
        a8_put4(row * A8_QS + 4u * (e % 32u), v);
    }
    workgroupBarrier();
    var o0 = a8_zero(); var o1 = a8_zero(); var o2 = a8_zero(); var o3 = a8_zero();
    var o4 = a8_zero(); var o5 = a8_zero(); var o6 = a8_zero(); var o7 = a8_zero();
    var o8 = a8_zero(); var o9 = a8_zero(); var o10 = a8_zero(); var o11 = a8_zero();
    var o12 = a8_zero(); var o13 = a8_zero(); var o14 = a8_zero(); var o15 = a8_zero();
    // This invocation's row: its largest score so far, its share of the
    // row's sum.
    var best = -1.0e30;
    var total = 0.0;
    let kbase = p.n5 + kvh * p.n1 * 128u;
    let vbase = p.n6 + kvh * p.n1 * 128u;
    let tiles = (p.n1 + 15u) / 16u;
    let qa = sg * 8u * A8_QS;
    let sa = sg * 128u;
    let pa = A8_P + sg * 8u * A8_PS;
    for (var t = 0u; t < tiles; t++) {
        workgroupBarrier();
        // The keys: 16 rows of 32 vectors, 4 an invocation.
        for (var j = 0u; j < 4u; j++) {
            let e = li + 128u * j;
            let key = t * 16u + e / 32u;
            var kv = vec4<f32>(0.0);
            if (key < p.n1) {
                kv = src[(kbase + key * 128u) / 4u + e % 32u];
            }
            a8_put4(A8_K + (e / 32u) * A8_QS + 4u * (e % 32u), kv);
        }
        // The values by dims: keys 2·i and 2·i + 1 (i = e / 32) of dims
        // 4·(e % 32)..; 2 an invocation.
        for (var j = 0u; j < 2u; j++) {
            let e = li + 128u * j;
            let i = e / 32u;
            let dq = e % 32u;
            let k0 = t * 16u + 2u * i;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (k0 < p.n1) {
                v0 = src[(vbase + k0 * 128u) / 4u + dq];
            }
            if (k0 + 1u < p.n1) {
                v1 = src[(vbase + (k0 + 1u) * 128u) / 4u + dq];
            }
            let at = A8_V + 4u * dq * A8_VS + 2u * i;
            a8_pair(at, v0.x, v1.x);
            a8_pair(at + A8_VS, v0.y, v1.y);
            a8_pair(at + 2u * A8_VS, v0.z, v1.z);
            a8_pair(at + 3u * A8_VS, v0.w, v1.w);
        }
        workgroupBarrier();
        // The scores, by columns: key c of row r at sa + c·8 + r.
        m8_st_c(a8_scores(qa, 0u), sa, 8u);
        m8_st_c(a8_scores(qa, 8u), sa + 64u, 8u);
        workgroupBarrier();
        let sk = sa + 32u * g + r;
        let s = vec4<f32>(a8_f[sk], a8_f[sk + 8u], a8_f[sk + 16u], a8_f[sk + 24u]);
        let k0 = t * 16u + 4u * g;
        a8_part[li] = a8_top4(-1.0e30, s, query, k0);
        workgroupBarrier();
        let rb = sg * 32u + r;
        let top = max(max(a8_part[rb], a8_part[rb + 8u]), max(a8_part[rb + 16u], a8_part[rb + 24u]));
        var fade = 1.0;
        if (top > best + 8.0) {
            fade = exp_held(best - top);
            best = top;
            if (total > 0.0) {
                a8_flag = 1u;
            }
        }
        let e = a8_exp4(s, query, k0, best);
        total = total * fade + e.x + e.y + e.z + e.w;
        a8_put4(pa + r * A8_PS + 4u * g, e);
        if (g == 0u) {
            a8_fade[sg * 8u + r] = fade;
        }
        if (workgroupUniformLoad(&a8_flag) != 0u) {
            o0 = a8_rescale(o0, sg, lane); o1 = a8_rescale(o1, sg, lane);
            o2 = a8_rescale(o2, sg, lane); o3 = a8_rescale(o3, sg, lane);
            o4 = a8_rescale(o4, sg, lane); o5 = a8_rescale(o5, sg, lane);
            o6 = a8_rescale(o6, sg, lane); o7 = a8_rescale(o7, sg, lane);
            o8 = a8_rescale(o8, sg, lane); o9 = a8_rescale(o9, sg, lane);
            o10 = a8_rescale(o10, sg, lane); o11 = a8_rescale(o11, sg, lane);
            o12 = a8_rescale(o12, sg, lane); o13 = a8_rescale(o13, sg, lane);
            o14 = a8_rescale(o14, sg, lane); o15 = a8_rescale(o15, sg, lane);
            if (li == 0u) {
                a8_flag = 0u;
            }
            workgroupBarrier();
        }
        // The exps' product with the values: 8 queries × 16 keys times 16
        // keys × 128.
        let p0 = m8_ld_a(pa, A8_PS);
        let p1 = m8_ld_a(pa + 8u, A8_PS);
        o0 = a8_pv(o0, p0, p1, 0u); o1 = a8_pv(o1, p0, p1, 1u);
        o2 = a8_pv(o2, p0, p1, 2u); o3 = a8_pv(o3, p0, p1, 3u);
        o4 = a8_pv(o4, p0, p1, 4u); o5 = a8_pv(o5, p0, p1, 5u);
        o6 = a8_pv(o6, p0, p1, 6u); o7 = a8_pv(o7, p0, p1, 7u);
        o8 = a8_pv(o8, p0, p1, 8u); o9 = a8_pv(o9, p0, p1, 9u);
        o10 = a8_pv(o10, p0, p1, 10u); o11 = a8_pv(o11, p0, p1, 11u);
        o12 = a8_pv(o12, p0, p1, 12u); o13 = a8_pv(o13, p0, p1, 13u);
        o14 = a8_pv(o14, p0, p1, 14u); o15 = a8_pv(o15, p0, p1, 15u);
    }
    // Each row's sum: its four invocations' shares.
    workgroupBarrier();
    a8_part[li] = total;
    workgroupBarrier();
    if (g == 0u) {
        let rb = sg * 32u + r;
        a8_sum[sg * 8u + r] = (a8_part[rb] + a8_part[rb + 8u]) + (a8_part[rb + 16u] + a8_part[rb + 24u]);
    }
    workgroupBarrier();
    a8_out(o0, sg, lane, q0, h, 0u); a8_out(o1, sg, lane, q0, h, 1u);
    a8_out(o2, sg, lane, q0, h, 2u); a8_out(o3, sg, lane, q0, h, 3u);
    a8_out(o4, sg, lane, q0, h, 4u); a8_out(o5, sg, lane, q0, h, 5u);
    a8_out(o6, sg, lane, q0, h, 6u); a8_out(o7, sg, lane, q0, h, 7u);
    a8_out(o8, sg, lane, q0, h, 8u); a8_out(o9, sg, lane, q0, h, 9u);
    a8_out(o10, sg, lane, q0, h, 10u); a8_out(o11, sg, lane, q0, h, 11u);
    a8_out(o12, sg, lane, q0, h, 12u); a8_out(o13, sg, lane, q0, h, 13u);
    a8_out(o14, sg, lane, q0, h, 14u); a8_out(o15, sg, lane, q0, h, 15u);
}
