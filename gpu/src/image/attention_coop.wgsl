// Attention on cooperative matrices (the numbers of `attention.wgsl`): a
// workgroup takes 128 queries of a head, each of its eight subgroups 16 of
// them, the queries (scaled) held as eight 16 × 16 f16 matrices; keys and
// values come 32 at a time as f16, shared by all eight. The scores of a
// subgroup's 16 × 32 go through workgroup memory, two invocations a row
// (16 keys each): the row's largest score so far, each half's sum of
// exp(score - largest), and exp(score - largest) as f16 for the values'
// product. The largest is moved only when a tile's is more than 8 above it
// (the exps stay below e⁸, in f16's range), and only then are the sums
// rescaled (through workgroup memory), so most tiles add their products
// straight into the eight f32 matrices of a row's 128 values.

// A row of a head in workgroup memory: 128 values and 8 of padding.
const FC_S: u32 = 136u;
// Half the queries' tile at a time, then the keys' (from 0) and the
// values' (from 4352).
var<workgroup> fc_kv: array<f16, 8704>;
// Each subgroup's 16 × 32 scores (and a 16 × 16 of sums when rescaled).
var<workgroup> fc_s: array<f32, 4096>;
// Each subgroup's 16 × 32 exps.
var<workgroup> fc_p: array<f16, 4096>;
// Each invocation's largest score of the tile (then its half-row's sum),
// and each query's last rescale.
var<workgroup> fc_part: array<f32, 256>;
var<workgroup> fc_fade: array<f32, 128>;
var<workgroup> fc_flag: u32;

fn fc_put4(at: u32, v: vec4<f32>) {
    fc_kv[at] = f16(v.x);
    fc_kv[at + 1u] = f16(v.y);
    fc_kv[at + 2u] = f16(v.z);
    fc_kv[at + 3u] = f16(v.w);
}

// One of the sums' 16 × 16 scaled per row by `fc_fade`.
fn fc_rescale(o: coop_mat16x16<f32, C>, sg: u32, lane: u32) -> coop_mat16x16<f32, C> {
    let at = sg * 512u;
    coopStore(o, &fc_s[at], 16u);
    workgroupBarrier();
    // Column-major: value (r, c) at c·16 + r; eight a lane.
    for (var j = 0u; j < 8u; j++) {
        let e = lane * 8u + j;
        fc_s[at + e] *= fc_fade[sg * 16u + e % 16u];
    }
    workgroupBarrier();
    let out = coopLoad<coop_mat16x16<f32, C>>(&fc_s[at], 16u);
    workgroupBarrier();
    return out;
}

// One of the sums' 16 × 16 (values 16·d..) divided by the rows' sums (in
// `fc_part`) and written out.
fn fc_out(o: coop_mat16x16<f32, C>, sg: u32, lane: u32, q0: u32, h: u32, d: u32) {
    let at = sg * 512u;
    coopStore(o, &fc_s[at], 16u);
    workgroupBarrier();
    for (var j = 0u; j < 8u; j++) {
        let e = lane * 8u + j;
        let r = e % 16u;
        let query = q0 + sg * 16u + r;
        if (query < p.n0) {
            dst_put(p.n7 + query * p.n8 + h * 128u + d * 16u + e / 16u, fc_s[at + e] / fc_part[sg * 32u + r]);
        }
    }
    workgroupBarrier();
}

// The queries `half` (64 rows from q0 + 64·half), scaled, as f16 into
// `fc_kv`: 64 × 32 vectors, 8 an invocation.
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

// A subgroup's 16 queries' 128 values as eight matrices, from `fc_kv`'s
// row `row`.
struct Qf {
    q0: coop_mat16x16<f16, A>, q1: coop_mat16x16<f16, A>, q2: coop_mat16x16<f16, A>, q3: coop_mat16x16<f16, A>,
    q4: coop_mat16x16<f16, A>, q5: coop_mat16x16<f16, A>, q6: coop_mat16x16<f16, A>, q7: coop_mat16x16<f16, A>,
}

fn fc_qf(row: u32) -> Qf {
    let a = row * FC_S;
    return Qf(
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 16u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 32u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 48u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 64u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 80u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 96u], FC_S),
        coopLoadT<coop_mat16x16<f16, A>>(&fc_kv[a + 112u], FC_S),
    );
}

// The scores of the subgroup's queries and 16 keys (from row `k` of the
// keys' tile).
fn fc_scores(q: Qf, k: u32) -> coop_mat16x16<f32, C> {
    let b = k * FC_S;
    var s = coop_mat16x16<f32, C>();
    s = coopMultiplyAdd(q.q0, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b], FC_S), s);
    s = coopMultiplyAdd(q.q1, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 16u], FC_S), s);
    s = coopMultiplyAdd(q.q2, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 32u], FC_S), s);
    s = coopMultiplyAdd(q.q3, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 48u], FC_S), s);
    s = coopMultiplyAdd(q.q4, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 64u], FC_S), s);
    s = coopMultiplyAdd(q.q5, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 80u], FC_S), s);
    s = coopMultiplyAdd(q.q6, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 96u], FC_S), s);
    s = coopMultiplyAdd(q.q7, coopLoad<coop_mat16x16<f16, B>>(&fc_kv[b + 112u], FC_S), s);
    return s;
}

// The exps' product with the values' 16 columns from `d`, added to `o`.
fn fc_pv(o: coop_mat16x16<f32, C>, p0: coop_mat16x16<f16, A>, p1: coop_mat16x16<f16, A>, d: u32) -> coop_mat16x16<f32, C> {
    let v = 4352u + d;
    let a = coopMultiplyAdd(p0, coopLoadT<coop_mat16x16<f16, B>>(&fc_kv[v], FC_S), o);
    return coopMultiplyAdd(p1, coopLoadT<coop_mat16x16<f16, B>>(&fc_kv[v + 16u * FC_S], FC_S), a);
}

@compute @workgroup_size(256)
fn attention16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let h = wg.y;
    let kvh = h / p.n3;
    let q0 = wg.x * 128u;
    let sg = li / 32u;
    let lane = li % 32u;
    let row = lane % 16u;
    let half = lane / 16u;
    let query = q0 + sg * 16u + row;
    let scale = bitcast<f32>(p.n9);
    // The queries, half at a time through `fc_kv`.
    var q = Qf(coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>(),
        coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>(), coop_mat16x16<f16, A>());
    fc_queries(li, h, q0, scale);
    if (li == 0u) {
        fc_flag = 0u;
    }
    workgroupBarrier();
    if (sg < 4u) {
        q = fc_qf(sg * 16u);
    }
    workgroupBarrier();
    fc_queries(li, h, q0 + 64u, scale);
    workgroupBarrier();
    if (sg >= 4u) {
        q = fc_qf((sg - 4u) * 16u);
    }
    var o0 = coop_mat16x16<f32, C>(); var o1 = coop_mat16x16<f32, C>();
    var o2 = coop_mat16x16<f32, C>(); var o3 = coop_mat16x16<f32, C>();
    var o4 = coop_mat16x16<f32, C>(); var o5 = coop_mat16x16<f32, C>();
    var o6 = coop_mat16x16<f32, C>(); var o7 = coop_mat16x16<f32, C>();
    // This invocation's row: its largest score so far, its half's sum.
    var best = -1.0e30;
    var total = 0.0;
    let kbase = p.n5 + kvh * p.n1 * 128u;
    let vbase = p.n6 + kvh * p.n1 * 128u;
    let tiles = (p.n1 + 31u) / 32u;
    let at = sg * 512u + row * 32u + half * 16u;
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
            fc_put4(4352u + kat, vv);
        }
        workgroupBarrier();
        // Row-major, 32 a row.
        coopStoreT(fc_scores(q, 0u), &fc_s[sg * 512u], 32u);
        coopStoreT(fc_scores(q, 16u), &fc_s[sg * 512u + 16u], 32u);
        workgroupBarrier();
        var top = -1.0e30;
        let k0 = t * 32u + half * 16u;
        for (var j = 0u; j < 16u; j++) {
            if (fa_sees(query, k0 + j)) {
                top = max(top, fc_s[at + j]);
            }
        }
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
        for (var j = 0u; j < 16u; j++) {
            var e = 0.0;
            if (fa_sees(query, k0 + j)) {
                e = exp_held(fc_s[at + j] - best);
            }
            total += e;
            fc_p[at + j] = f16(e);
        }
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
        let p0 = coopLoadT<coop_mat16x16<f16, A>>(&fc_p[pa], 32u);
        let p1 = coopLoadT<coop_mat16x16<f16, A>>(&fc_p[pa + 16u], 32u);
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
