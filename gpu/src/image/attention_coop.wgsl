// Attention on cooperative matrices of 16 × 16 (the numbers of
// `attention.wgsl`; on 8 × 8 `attention8.wgsl`): a workgroup takes
// 128 queries of a head, each of its eight subgroups 16 of
// them, the queries (scaled) held as eight tiles of 16 × 16 in f16
// (`coop.wgsl`); keys and values come 32 at a time as f16, shared by all
// eight. Each subgroup's scores of 16 queries × 32 keys go through its
// 16 × 16 of `cm_f`, the first 16 keys' then the next 16's, into
// registers, two invocations a row (16 keys each): the row's largest score
// so far, each half's sum of exp(score - largest), and exp(score -
// largest) as f16 for the values' product, written over the keys (no
// longer read by then). The largest is moved only when a tile's is more
// than 8 above it (the exps stay below e⁸, in f16's range), and only then
// are the sums rescaled (through `cm_f`), so most tiles add their products
// straight into the eight f32 tiles of a row's 128 values.

// A row of a head in `cm_h`: 128 values and 8 of padding. Half the
// queries' tile at a time, then the keys' (from 0; then the exps, 16 × 32
// a subgroup) and the values' (from `FC_V`).
const FC_S: u32 = 136u;
const FC_V: u32 = 4352u;
// The exps (eight subgroups' 16 × 32) end before the values, and the
// values (32 rows) within `cm_h`.
const_assert 8u * 512u <= FC_V;
const_assert FC_V + 32u * FC_S <= CM_H;
// Each invocation's largest score of the tile (then its half-row's sum),
// and each query's last rescale.
var<workgroup> fc_part: array<f32, 256>;
var<workgroup> fc_fade: array<f32, 128>;
var<workgroup> fc_flag: u32;

// Four f32s written at `at` of `cm_h` as f16s (a multiple of 4: whole
// words, one invocation's).
fn fc_put4(at: u32, v: vec4<f32>) {
    cm_h[at] = f16(v.x);
    cm_h[at + 1u] = f16(v.y);
    cm_h[at + 2u] = f16(v.z);
    cm_h[at + 3u] = f16(v.w);
}

// One of the sums' tiles scaled per row by `fc_fade`.
fn fc_rescale(o: CmC, sg: u32, lane: u32) -> CmC {
    let at = sg * 256u;
    cm_st_c(o, at, 16u);
    workgroupBarrier();
    // Column-major: value (r, c) at c·16 + r; eight a lane.
    for (var j = 0u; j < 8u; j++) {
        let e = lane * 8u + j;
        cm_f[at + e] *= fc_fade[sg * 16u + e % 16u];
    }
    workgroupBarrier();
    let out = cm_ld_c(at, 16u);
    workgroupBarrier();
    return out;
}

// One of the sums' tiles (values 16·d..) divided by the rows' sums (in
// `fc_part`) and written out.
fn fc_out(o: CmC, sg: u32, lane: u32, q0: u32, h: u32, d: u32) {
    let at = sg * 256u;
    cm_st_c(o, at, 16u);
    workgroupBarrier();
    for (var j = 0u; j < 8u; j++) {
        let e = lane * 8u + j;
        let r = e % 16u;
        let query = q0 + sg * 16u + r;
        if (query < p.n0) {
            dst_put(p.n7 + query * p.n8 + h * 128u + d * 16u + e / 16u, cm_f[at + e] / fc_part[sg * 32u + r]);
        }
    }
    workgroupBarrier();
}

// The queries `half` (64 rows from q0 + 64·half), scaled, as f16 into
// `cm_h`: 64 × 32 vectors, 8 an invocation.
fn fc_queries(li: u32, h: u32, q0: u32, scale: f32) {
    for (var j = 0u; j < 8u; j++) {
        let e = li + 256u * j;
        let r = e / 32u;
        var v = vec4<f32>(0.0);
        if (q0 + r < p.n0) {
            v = src[(p.n4 + (h * p.n0 + q0 + r) * 128u) / 4u + e % 32u] * scale;
        }
        fc_put4(r * FC_S + 4u * (e % 32u), v);
    }
}

// A subgroup's 16 queries' 128 values as eight tiles, from `cm_h`'s row
// `row`.
struct Qf {
    q0: CmA, q1: CmA, q2: CmA, q3: CmA,
    q4: CmA, q5: CmA, q6: CmA, q7: CmA,
}

fn fc_qf(row: u32) -> Qf {
    let a = row * FC_S;
    return Qf(
        cm_ld_at(a, FC_S),
        cm_ld_at(a + 16u, FC_S),
        cm_ld_at(a + 32u, FC_S),
        cm_ld_at(a + 48u, FC_S),
        cm_ld_at(a + 64u, FC_S),
        cm_ld_at(a + 80u, FC_S),
        cm_ld_at(a + 96u, FC_S),
        cm_ld_at(a + 112u, FC_S),
    );
}

// The scores of the subgroup's queries and 16 keys (from row `k` of the
// keys' tile).
fn fc_scores(q: Qf, k: u32) -> CmC {
    let b = k * FC_S;
    var s = cm_zero();
    s = cm_mma(q.q0, cm_ld_b(b, FC_S), s);
    s = cm_mma(q.q1, cm_ld_b(b + 16u, FC_S), s);
    s = cm_mma(q.q2, cm_ld_b(b + 32u, FC_S), s);
    s = cm_mma(q.q3, cm_ld_b(b + 48u, FC_S), s);
    s = cm_mma(q.q4, cm_ld_b(b + 64u, FC_S), s);
    s = cm_mma(q.q5, cm_ld_b(b + 80u, FC_S), s);
    s = cm_mma(q.q6, cm_ld_b(b + 96u, FC_S), s);
    s = cm_mma(q.q7, cm_ld_b(b + 112u, FC_S), s);
    return s;
}

// A row's 16 scores of a half-tile of keys, four by four.
struct Row16 {
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
    s3: vec4<f32>,
}

// The scores' tile `s` through the subgroup's 16 × 16 of `cm_f` (by rows),
// row `row`'s taken by the invocations of `half`, the others keeping
// `kept`.
fn fc_row(s: CmC, sg: u32, row: u32, half: u32, mine: u32, kept: Row16) -> Row16 {
    let at = sg * 256u;
    coopStoreT(s, &cm_f[at], 16u);
    workgroupBarrier();
    var out = kept;
    if (half == mine) {
        let r = at + row * 16u;
        out = Row16(
            vec4<f32>(cm_f[r], cm_f[r + 1u], cm_f[r + 2u], cm_f[r + 3u]),
            vec4<f32>(cm_f[r + 4u], cm_f[r + 5u], cm_f[r + 6u], cm_f[r + 7u]),
            vec4<f32>(cm_f[r + 8u], cm_f[r + 9u], cm_f[r + 10u], cm_f[r + 11u]),
            vec4<f32>(cm_f[r + 12u], cm_f[r + 13u], cm_f[r + 14u], cm_f[r + 15u]),
        );
    }
    workgroupBarrier();
    return out;
}

// The largest of `top` and the four scores of keys k.. that `query` sees.
fn fc_top4(top: f32, s: vec4<f32>, query: u32, k: u32) -> f32 {
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
// see the key), written as f16 at `at` and added to `total`, in order.
fn fc_exp4(total: f32, s: vec4<f32>, query: u32, k: u32, best: f32, at: u32) -> f32 {
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
    fc_put4(at, e);
    return total + e.x + e.y + e.z + e.w;
}

// The exps' product with the values' 16 columns from `d`, added to `o`.
fn fc_pv(o: CmC, p0: CmA, p1: CmA, d: u32) -> CmC {
    let v = FC_V + d;
    let a = cm_mma(p0, coopLoadT<CmB>(&cm_h[v], FC_S), o);
    return cm_mma(p1, coopLoadT<CmB>(&cm_h[v + 16u * FC_S], FC_S), a);
}

@compute @workgroup_size(256)
fn attention_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let kvh = h / p.n3;
    let q0 = (wg.x + p.n12) * 128u;
    let sg = li / 32u;
    let lane = li % 32u;
    let row = lane % 16u;
    let half = lane / 16u;
    let query = q0 + sg * 16u + row;
    let scale = bitcast<f32>(p.n9);
    cm_zero_init(li);
    // The queries, half at a time through `cm_h` (the second half's
    // subgroups load the first's rows too, then their own: no cooperative
    // matrix starts as WGSL's zero value).
    fc_queries(li, h, q0, scale);
    if (li == 0u) {
        fc_flag = 0u;
    }
    workgroupBarrier();
    var q = fc_qf((sg % 4u) * 16u);
    workgroupBarrier();
    fc_queries(li, h, q0 + 64u, scale);
    workgroupBarrier();
    if (sg >= 4u) {
        q = fc_qf((sg - 4u) * 16u);
    }
    var o0 = cm_zero(); var o1 = cm_zero(); var o2 = cm_zero(); var o3 = cm_zero();
    var o4 = cm_zero(); var o5 = cm_zero(); var o6 = cm_zero(); var o7 = cm_zero();
    // This invocation's row: its largest score so far, its half's sum.
    var best = -1.0e30;
    var total = 0.0;
    let kbase = p.n5 + kvh * p.n1 * 128u;
    let vbase = p.n6 + kvh * p.n1 * 128u;
    let tiles = (p.n1 + 31u) / 32u;
    // Its 16 exps in the subgroup's 16 × 32 (over the keys).
    let pat = sg * 512u + row * 32u + half * 16u;
    for (var t = 0u; t < tiles; t++) {
        workgroupBarrier();
        // Keys and values: 32 × 32 vectors each, 4 an invocation.
        for (var j = 0u; j < 4u; j++) {
            let e = li + 256u * j;
            let key = t * 32u + e / 32u;
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (key < p.n1) {
                kv = src[(kbase + key * 128u) / 4u + e % 32u];
                vv = src[(vbase + key * 128u) / 4u + e % 32u];
            }
            let kat = (e / 32u) * FC_S + 4u * (e % 32u);
            fc_put4(kat, kv);
            fc_put4(FC_V + kat, vv);
        }
        workgroupBarrier();
        // The row's 16 scores of this invocation's half of the keys; past
        // the second store's barrier no subgroup reads the keys again.
        var sc = fc_row(fc_scores(q, 0u), sg, row, half, 0u, Row16(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)));
        sc = fc_row(fc_scores(q, 16u), sg, row, half, 1u, sc);
        let k0 = t * 32u + half * 16u;
        var top = fc_top4(-1.0e30, sc.s0, query, k0);
        top = fc_top4(top, sc.s1, query, k0 + 4u);
        top = fc_top4(top, sc.s2, query, k0 + 8u);
        top = fc_top4(top, sc.s3, query, k0 + 12u);
        fc_part[li] = top;
        workgroupBarrier();
        top = max(top, fc_part[li ^ 16u]);
        var fade = 1.0;
        if (top > best + 8.0) {
            fade = exp_held(best - top);
            best = top;
            if (total > 0.0) {
                fc_flag = 1u;
            }
        }
        total *= fade;
        total = fc_exp4(total, sc.s0, query, k0, best, pat);
        total = fc_exp4(total, sc.s1, query, k0 + 4u, best, pat + 4u);
        total = fc_exp4(total, sc.s2, query, k0 + 8u, best, pat + 8u);
        total = fc_exp4(total, sc.s3, query, k0 + 12u, best, pat + 12u);
        if (half == 0u) {
            fc_fade[sg * 16u + row] = fade;
        }
        if (workgroupUniformLoad(&fc_flag) != 0u) {
            o0 = fc_rescale(o0, sg, lane);
            o1 = fc_rescale(o1, sg, lane);
            o2 = fc_rescale(o2, sg, lane);
            o3 = fc_rescale(o3, sg, lane);
            o4 = fc_rescale(o4, sg, lane);
            o5 = fc_rescale(o5, sg, lane);
            o6 = fc_rescale(o6, sg, lane);
            o7 = fc_rescale(o7, sg, lane);
            if (li == 0u) {
                fc_flag = 0u;
            }
            workgroupBarrier();
        }
        // The values' product: 16 queries × 32 keys times 32 keys × 128.
        let pa = sg * 512u;
        let p0 = cm_ld_at(pa, 32u);
        let p1 = cm_ld_at(pa + 16u, 32u);
        o0 = fc_pv(o0, p0, p1, 0u);
        o1 = fc_pv(o1, p0, p1, 16u);
        o2 = fc_pv(o2, p0, p1, 32u);
        o3 = fc_pv(o3, p0, p1, 48u);
        o4 = fc_pv(o4, p0, p1, 64u);
        o5 = fc_pv(o5, p0, p1, 80u);
        o6 = fc_pv(o6, p0, p1, 96u);
        o7 = fc_pv(o7, p0, p1, 112u);
    }
    // Each row's sum: its two halves'.
    workgroupBarrier();
    fc_part[li] = total;
    workgroupBarrier();
    let all = total + fc_part[li ^ 16u];
    workgroupBarrier();
    if (half == 0u) {
        fc_part[sg * 32u + row] = all;
    }
    workgroupBarrier();
    fc_out(o0, sg, lane, q0, h, 0u);
    fc_out(o1, sg, lane, q0, h, 1u);
    fc_out(o2, sg, lane, q0, h, 2u);
    fc_out(o3, sg, lane, q0, h, 3u);
    fc_out(o4, sg, lane, q0, h, 4u);
    fc_out(o5, sg, lane, q0, h, 5u);
    fc_out(o6, sg, lane, q0, h, 6u);
    fc_out(o7, sg, lane, q0, h, 7u);
}
