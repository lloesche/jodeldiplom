// The matrix multiply on cooperative matrices (tensor cores): the numbers
// and tails of `tail.wgsl`, A rounded to f16 (clamped to its range) and
// W's values (times their scales) to f16, multiplied a tile of 16 × 16 ×
// 16 at a time into f32 sums (`coop.wgsl`). Tiles of 128 rows × 128
// columns through k in steps of `CM_K` (the layer's: 64, or 32 in 32 KB of
// workgroup memory), the next step's values read while this one is
// multiplied, and rounded to f16 only as they are written to the tiles
// (rounded as they were read, the wait for each read came before the
// multiplies instead of after them: 26 % longer on the 5080). The eight
// subgroups of 32 each keep 32 rows × 64 columns: rows 32·(sg / 2)..,
// columns 32·(sg % 2).. and 64 more (so a SwiGLU tile's gate and its up
// are in the same subgroup).

// W's tile in `cm_h`, after A's 128 rows.
const CM_B: u32 = 128u * CM_S;
// The invocations that read a row of A's tile (four values each), and
// the rows between an invocation's: each reads 128 / `CM_R` rows (4, or 8
// where a step is 64), and `CM_K` / 2 values of one row of W.
const CM_Q: u32 = CM_K / 4u;
const CM_R: u32 = 256u / CM_Q;

// Eight of W's values written to its tile as f16.
fn cm_put_w(at: u32, w: W8) {
    cm_put_held4(at, w.lo);
    cm_put_held4(at + 4u, w.hi);
}

// A row's 8 values of one tile of a subgroup's sums (row lane % 16,
// columns 8·(lane / 16)..), through its 16 × 16 of `cm_f`.
struct Cm8 {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn cm_read(c: CmC, sg: u32, lane: u32) -> Cm8 {
    cm_st_c(c, sg * 256u, 16u);
    workgroupBarrier();
    let base = sg * 256u + 8u * (lane / 16u) * 16u + lane % 16u;
    let out = Cm8(
        vec4<f32>(cm_f[base], cm_f[base + 16u], cm_f[base + 32u], cm_f[base + 48u]),
        vec4<f32>(cm_f[base + 64u], cm_f[base + 80u], cm_f[base + 96u], cm_f[base + 112u]),
    );
    workgroupBarrier();
    return out;
}

// One tile of sums through the tail at (row, col).
fn cm_tail(c: CmC, sg: u32, lane: u32, row: u32, col: u32) {
    let v = cm_read(c, sg, lane);
    let r = row + lane % 16u;
    let c0 = col + 8u * (lane / 16u);
    mm_put4(r, c0, v.lo);
    mm_put4(r, c0 + 4u, v.hi);
}

// A gate's tile and its up's: silu(gate) · up of each value, at column
// tile/2 + the gate's column.
fn cm_swiglu(g: CmC, u: CmC, sg: u32, lane: u32, row: u32, col: u32) {
    let gv = cm_read(g, sg, lane);
    let uv = cm_read(u, sg, lane);
    let r = row + lane % 16u;
    let c = col + 8u * (lane / 16u);
    mm_swiglu4(r, c, gv.lo, uv.lo);
    mm_swiglu4(r, c + 4u, gv.hi, uv.hi);
}

@compute @workgroup_size(256)
fn gemm_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let sg = li / 32u;
    let lane = li % 32u;
    let wm = 32u * (sg / 2u);
    let wn = 32u * (sg % 2u);
    let row0 = wg.y * 128u;
    let col0 = wg.x * 128u;
    // Each invocation reads four values of rows li/CM_Q + CM_R·i of A a
    // step (values 4·(li%CM_Q)..) and CM_K/2 of one row of W (li/2, values
    // (CM_K/2)·(li%2)..); the reads past four rows and 16 of W only where
    // a step is 64. `CM_K > 32u` is a constant: each layer's kernel keeps
    // one side of these branches. They are not loops over arrays: an array
    // indexed in a loop may be kept in memory on Apple's GPUs (common.wgsl,
    // `H64`), and this is the code the Mac has run.
    let ar = li / CM_Q;
    let aq = 4u * (li % CM_Q);
    let wc = li / 2u;
    let wq = (CM_K / 2u) * (li % 2u);
    cm_zero_init(li);
    workgroupBarrier();
    var c00 = cm_zero(); var c01 = cm_zero(); var c02 = cm_zero(); var c03 = cm_zero();
    var c10 = cm_zero(); var c11 = cm_zero(); var c12 = cm_zero(); var c13 = cm_zero();
    var na0 = mm_a4(row0 + ar, aq);
    var na1 = mm_a4(row0 + ar + CM_R, aq);
    var na2 = mm_a4(row0 + ar + 2u * CM_R, aq);
    var na3 = mm_a4(row0 + ar + 3u * CM_R, aq);
    var nw0 = mm_w8(col0 + wc, wq);
    var nw1 = mm_w8(col0 + wc, wq + 8u);
    var na4 = na0; var na5 = na0; var na6 = na0; var na7 = na0;
    var nw2 = nw0; var nw3 = nw0;
    if (CM_K > 32u) {
        na4 = mm_a4(row0 + ar + 4u * CM_R, aq);
        na5 = mm_a4(row0 + ar + 5u * CM_R, aq);
        na6 = mm_a4(row0 + ar + 6u * CM_R, aq);
        na7 = mm_a4(row0 + ar + 7u * CM_R, aq);
        nw2 = mm_w8(col0 + wc, wq + 16u);
        nw3 = mm_w8(col0 + wc, wq + 24u);
    }
    let steps = (mm_k() + CM_K - 1u) / CM_K;
    for (var s = 0u; s < steps; s++) {
        workgroupBarrier();
        cm_put_held4(ar * CM_S + aq, na0);
        cm_put_held4((ar + CM_R) * CM_S + aq, na1);
        cm_put_held4((ar + 2u * CM_R) * CM_S + aq, na2);
        cm_put_held4((ar + 3u * CM_R) * CM_S + aq, na3);
        let bw = CM_B + wc * CM_S + wq;
        cm_put_w(bw, nw0);
        cm_put_w(bw + 8u, nw1);
        if (CM_K > 32u) {
            cm_put_held4((ar + 4u * CM_R) * CM_S + aq, na4);
            cm_put_held4((ar + 5u * CM_R) * CM_S + aq, na5);
            cm_put_held4((ar + 6u * CM_R) * CM_S + aq, na6);
            cm_put_held4((ar + 7u * CM_R) * CM_S + aq, na7);
            cm_put_w(bw + 16u, nw2);
            cm_put_w(bw + 24u, nw3);
        }
        workgroupBarrier();
        if (s + 1u < steps) {
            let k1 = (s + 1u) * CM_K;
            na0 = mm_a4(row0 + ar, k1 + aq);
            na1 = mm_a4(row0 + ar + CM_R, k1 + aq);
            na2 = mm_a4(row0 + ar + 2u * CM_R, k1 + aq);
            na3 = mm_a4(row0 + ar + 3u * CM_R, k1 + aq);
            nw0 = mm_w8(col0 + wc, k1 + wq);
            nw1 = mm_w8(col0 + wc, k1 + wq + 8u);
            if (CM_K > 32u) {
                na4 = mm_a4(row0 + ar + 4u * CM_R, k1 + aq);
                na5 = mm_a4(row0 + ar + 5u * CM_R, k1 + aq);
                na6 = mm_a4(row0 + ar + 6u * CM_R, k1 + aq);
                na7 = mm_a4(row0 + ar + 7u * CM_R, k1 + aq);
                nw2 = mm_w8(col0 + wc, k1 + wq + 16u);
                nw3 = mm_w8(col0 + wc, k1 + wq + 24u);
            }
        }
        for (var kk = 0u; kk < CM_K; kk += 16u) {
            let a0 = cm_ld_at(wm * CM_S + kk, CM_S);
            let a1 = cm_ld_at((wm + 16u) * CM_S + kk, CM_S);
            let b0 = cm_ld_b(CM_B + wn * CM_S + kk, CM_S);
            let b1 = cm_ld_b(CM_B + (wn + 16u) * CM_S + kk, CM_S);
            let b2 = cm_ld_b(CM_B + (wn + 64u) * CM_S + kk, CM_S);
            let b3 = cm_ld_b(CM_B + (wn + 80u) * CM_S + kk, CM_S);
            c00 = cm_mma(a0, b0, c00);
            c01 = cm_mma(a0, b1, c01);
            c02 = cm_mma(a0, b2, c02);
            c03 = cm_mma(a0, b3, c03);
            c10 = cm_mma(a1, b0, c10);
            c11 = cm_mma(a1, b1, c11);
            c12 = cm_mma(a1, b2, c12);
            c13 = cm_mma(a1, b3, c13);
        }
    }
    let r0 = row0 + wm;
    let r1 = r0 + 16u;
    if ((mm_flags() & MM_SWIGLU) != 0u) {
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
