// The matrix multiply on cooperative matrices (tensor cores): the numbers
// and tails of `tail.wgsl`, A rounded to f16 (clamped to its range) and
// W's values (times their scales) to f16, multiplied 16 × 16 × 16 at a
// time into f32 sums. Tiles of 128 rows × 128 columns through k in steps
// of 64, the next step's values read while this one is multiplied. The
// eight subgroups of 32 each keep 32 rows × 64 columns: rows 32·(sg /
// 2).., columns 32·(sg % 2).. and 64 more (so a SwiGLU tile's gate and its
// up are in the same subgroup).

const CM_K: u32 = 64u;
// A row of a tile in workgroup memory: 64 values and 8 of padding.
const CM_S: u32 = 72u;

var<workgroup> cm_a: array<f16, 9216>;
var<workgroup> cm_b: array<f16, 9216>;
// Each subgroup's 16 × 16 for the tail.
var<workgroup> cm_c: array<f32, 2048>;

fn cm_half(v: f32) -> f16 {
    return f16(clamp(v, -65504.0, 65504.0));
}

fn cm_putb(at: u32, v: vec4<f32>) {
    cm_b[at] = cm_half(v.x);
    cm_b[at + 1u] = cm_half(v.y);
    cm_b[at + 2u] = cm_half(v.z);
    cm_b[at + 3u] = cm_half(v.w);
}

fn cm_puta(at: u32, v: vec4<f32>) {
    cm_a[at] = cm_half(v.x);
    cm_a[at + 1u] = cm_half(v.y);
    cm_a[at + 2u] = cm_half(v.z);
    cm_a[at + 3u] = cm_half(v.w);
}

// A row's 8 values of one 16 × 16 of a subgroup's sums (row lane % 16,
// columns 8·(lane / 16)..), through workgroup memory.
struct Cm8 {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn cm_read(c: coop_mat16x16<f32, C>, sg: u32, lane: u32) -> Cm8 {
    coopStore(c, &cm_c[sg * 256u], 16u);
    workgroupBarrier();
    let base = sg * 256u + 8u * (lane / 16u) * 16u + lane % 16u;
    let out = Cm8(
        vec4<f32>(cm_c[base], cm_c[base + 16u], cm_c[base + 32u], cm_c[base + 48u]),
        vec4<f32>(cm_c[base + 64u], cm_c[base + 80u], cm_c[base + 96u], cm_c[base + 112u]),
    );
    workgroupBarrier();
    return out;
}

// One 16 × 16 of sums through the tail at (row, col).
fn cm_tail(c: coop_mat16x16<f32, C>, sg: u32, lane: u32, row: u32, col: u32) {
    let v = cm_read(c, sg, lane);
    let r = row + lane % 16u;
    let c0 = col + 8u * (lane / 16u);
    mm_put4(r, c0, v.lo);
    mm_put4(r, c0 + 4u, v.hi);
}

// A gate's 16 × 16 and its up's: silu(gate) · up of each value, at
// column tile/2 + the gate's column.
fn cm_swiglu(g: coop_mat16x16<f32, C>, u: coop_mat16x16<f32, C>, sg: u32, lane: u32, row: u32, col: u32) {
    let gv = cm_read(g, sg, lane);
    let uv = cm_read(u, sg, lane);
    let r = row + lane % 16u;
    let c = col + 8u * (lane / 16u);
    mm_swiglu4(r, c, gv.lo, uv.lo);
    mm_swiglu4(r, c + 4u, gv.hi, uv.hi);
}

@compute @workgroup_size(256)
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let sg = li / 32u;
    let lane = li % 32u;
    let wm = 32u * (sg / 2u);
    let wn = 32u * (sg % 2u);
    let row0 = wg.y * 128u;
    let col0 = wg.x * 128u;
    // Each invocation reads four values of eight rows of A a step (rows
    // li/16 + 16·i, values 4·(li%16)..) and 32 of one row of W (li/2,
    // values 32·(li%2)..).
    let ar = li / 16u;
    let aq = 4u * (li % 16u);
    let wc = li / 2u;
    let wq = 32u * (li % 2u);
    var c00 = coop_mat16x16<f32, C>(); var c01 = coop_mat16x16<f32, C>();
    var c02 = coop_mat16x16<f32, C>(); var c03 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>(); var c11 = coop_mat16x16<f32, C>();
    var c12 = coop_mat16x16<f32, C>(); var c13 = coop_mat16x16<f32, C>();
    var na0 = mm_a4(row0 + ar, aq);
    var na1 = mm_a4(row0 + ar + 16u, aq);
    var na2 = mm_a4(row0 + ar + 32u, aq);
    var na3 = mm_a4(row0 + ar + 48u, aq);
    var na4 = mm_a4(row0 + ar + 64u, aq);
    var na5 = mm_a4(row0 + ar + 80u, aq);
    var na6 = mm_a4(row0 + ar + 96u, aq);
    var na7 = mm_a4(row0 + ar + 112u, aq);
    var nw0 = mm_w8(col0 + wc, wq);
    var nw1 = mm_w8(col0 + wc, wq + 8u);
    var nw2 = mm_w8(col0 + wc, wq + 16u);
    var nw3 = mm_w8(col0 + wc, wq + 24u);
    let steps = (p.n2 + CM_K - 1u) / CM_K;
    for (var s = 0u; s < steps; s++) {
        workgroupBarrier();
        cm_puta(ar * CM_S + aq, na0);
        cm_puta((ar + 16u) * CM_S + aq, na1);
        cm_puta((ar + 32u) * CM_S + aq, na2);
        cm_puta((ar + 48u) * CM_S + aq, na3);
        cm_puta((ar + 64u) * CM_S + aq, na4);
        cm_puta((ar + 80u) * CM_S + aq, na5);
        cm_puta((ar + 96u) * CM_S + aq, na6);
        cm_puta((ar + 112u) * CM_S + aq, na7);
        let bw = wc * CM_S + wq;
        cm_putb(bw, nw0.lo);
        cm_putb(bw + 4u, nw0.hi);
        cm_putb(bw + 8u, nw1.lo);
        cm_putb(bw + 12u, nw1.hi);
        cm_putb(bw + 16u, nw2.lo);
        cm_putb(bw + 20u, nw2.hi);
        cm_putb(bw + 24u, nw3.lo);
        cm_putb(bw + 28u, nw3.hi);
        workgroupBarrier();
        if (s + 1u < steps) {
            let k1 = (s + 1u) * CM_K;
            na0 = mm_a4(row0 + ar, k1 + aq);
            na1 = mm_a4(row0 + ar + 16u, k1 + aq);
            na2 = mm_a4(row0 + ar + 32u, k1 + aq);
            na3 = mm_a4(row0 + ar + 48u, k1 + aq);
            na4 = mm_a4(row0 + ar + 64u, k1 + aq);
            na5 = mm_a4(row0 + ar + 80u, k1 + aq);
            na6 = mm_a4(row0 + ar + 96u, k1 + aq);
            na7 = mm_a4(row0 + ar + 112u, k1 + aq);
            nw0 = mm_w8(col0 + wc, k1 + wq);
            nw1 = mm_w8(col0 + wc, k1 + wq + 8u);
            nw2 = mm_w8(col0 + wc, k1 + wq + 16u);
            nw3 = mm_w8(col0 + wc, k1 + wq + 24u);
        }
        for (var kk = 0u; kk < CM_K; kk += 16u) {
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&cm_a[wm * CM_S + kk], CM_S);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&cm_a[(wm + 16u) * CM_S + kk], CM_S);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&cm_b[wn * CM_S + kk], CM_S);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&cm_b[(wn + 16u) * CM_S + kk], CM_S);
            let b2 = coopLoad<coop_mat16x16<f16, B>>(&cm_b[(wn + 64u) * CM_S + kk], CM_S);
            let b3 = coopLoad<coop_mat16x16<f16, B>>(&cm_b[(wn + 80u) * CM_S + kk], CM_S);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c02 = coopMultiplyAdd(a0, b2, c02);
            c03 = coopMultiplyAdd(a0, b3, c03);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
            c12 = coopMultiplyAdd(a1, b2, c12);
            c13 = coopMultiplyAdd(a1, b3, c13);
        }
    }
    let r0 = row0 + wm;
    let r1 = r0 + 16u;
    if ((p.n9 & MM_SWIGLU) != 0u) {
        let oc = col0 / 2u + wn;
        cm_swiglu(c00, c02, sg, lane, r0, oc);
        cm_swiglu(c01, c03, sg, lane, r0, oc + 16u);
        cm_swiglu(c10, c12, sg, lane, r1, oc);
        cm_swiglu(c11, c13, sg, lane, r1, oc + 16u);
        return;
    }
    let ca = col0 + wn;
    cm_tail(c00, sg, lane, r0, ca);
    cm_tail(c01, sg, lane, r0, ca + 16u);
    cm_tail(c02, sg, lane, r0, ca + 64u);
    cm_tail(c03, sg, lane, r0, ca + 80u);
    cm_tail(c10, sg, lane, r1, ca);
    cm_tail(c11, sg, lane, r1, ca + 16u);
    cm_tail(c12, sg, lane, r1, ca + 64u);
    cm_tail(c13, sg, lane, r1, ca + 80u);
}
