// The matrix multiply on f32 multiply-adds (its numbers and tails in
// `tail.wgsl`). Tiles of 128 rows × 128 columns through k in steps of 16,
// the next step's values read while this one is multiplied; each of 256
// invocations keeps 8 rows × 8 columns (rows ty·4.. and 64 + ty·4..,
// columns likewise).

// A's tile as [16 k][128 rows] and W's as [16 k][128 columns], four
// values a vector.
var<workgroup> gemm_tile_a: array<vec4<f32>, 512>;
var<workgroup> gemm_tile_b: array<vec4<f32>, 512>;

fn gemm_f32(wg: vec3<u32>, li: u32) {
    let tx = li % 16u;
    let ty = li / 16u;
    let row0 = wg.y * 128u;
    let col0 = wg.x * 128u;
    // Each invocation reads four values of two rows of A a step (rows
    // li/4 and 64 more, values 4·(li%4)..) and eight of one row of W
    // (li/2, values 8·(li%2)..).
    let ar = li / 4u;
    let aq = 4u * (li % 4u);
    let wc = li / 2u;
    let wh = 8u * (li % 2u);
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0); var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0); var acc5 = vec4<f32>(0.0); var acc6 = vec4<f32>(0.0); var acc7 = vec4<f32>(0.0);
    var acc8 = vec4<f32>(0.0); var acc9 = vec4<f32>(0.0); var acc10 = vec4<f32>(0.0); var acc11 = vec4<f32>(0.0);
    var acc12 = vec4<f32>(0.0); var acc13 = vec4<f32>(0.0); var acc14 = vec4<f32>(0.0); var acc15 = vec4<f32>(0.0);
    var na0 = mm_a4(row0 + ar, aq);
    var na1 = mm_a4(row0 + ar + 64u, aq);
    var nw = mm_w8(col0 + wc, wh);
    let steps = (mm_k() + 15u) / 16u;
    let ra = ar / 4u;
    let ca = ar % 4u;
    let rb = wc / 4u;
    let cb = wc % 4u;
    for (var s = 0u; s < steps; s++) {
        workgroupBarrier();
        // Transposed: value j of A's four is k = aq + j.
        gemm_tile_a[(aq + 0u) * 32u + ra][ca] = na0.x;
        gemm_tile_a[(aq + 1u) * 32u + ra][ca] = na0.y;
        gemm_tile_a[(aq + 2u) * 32u + ra][ca] = na0.z;
        gemm_tile_a[(aq + 3u) * 32u + ra][ca] = na0.w;
        gemm_tile_a[(aq + 0u) * 32u + ra + 16u][ca] = na1.x;
        gemm_tile_a[(aq + 1u) * 32u + ra + 16u][ca] = na1.y;
        gemm_tile_a[(aq + 2u) * 32u + ra + 16u][ca] = na1.z;
        gemm_tile_a[(aq + 3u) * 32u + ra + 16u][ca] = na1.w;
        gemm_tile_b[(wh + 0u) * 32u + rb][cb] = nw.lo.x;
        gemm_tile_b[(wh + 1u) * 32u + rb][cb] = nw.lo.y;
        gemm_tile_b[(wh + 2u) * 32u + rb][cb] = nw.lo.z;
        gemm_tile_b[(wh + 3u) * 32u + rb][cb] = nw.lo.w;
        gemm_tile_b[(wh + 4u) * 32u + rb][cb] = nw.hi.x;
        gemm_tile_b[(wh + 5u) * 32u + rb][cb] = nw.hi.y;
        gemm_tile_b[(wh + 6u) * 32u + rb][cb] = nw.hi.z;
        gemm_tile_b[(wh + 7u) * 32u + rb][cb] = nw.hi.w;
        workgroupBarrier();
        if (s + 1u < steps) {
            let k1 = (s + 1u) * 16u;
            na0 = mm_a4(row0 + ar, k1 + aq);
            na1 = mm_a4(row0 + ar + 64u, k1 + aq);
            nw = mm_w8(col0 + wc, k1 + wh);
        }
        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = gemm_tile_a[kk * 32u + ty];
            let a1 = gemm_tile_a[kk * 32u + ty + 16u];
            let b0 = gemm_tile_b[kk * 32u + tx];
            let b1 = gemm_tile_b[kk * 32u + tx + 16u];
            acc0 = fma(vec4<f32>(a0.x), b0, acc0);
            acc1 = fma(vec4<f32>(a0.x), b1, acc1);
            acc2 = fma(vec4<f32>(a0.y), b0, acc2);
            acc3 = fma(vec4<f32>(a0.y), b1, acc3);
            acc4 = fma(vec4<f32>(a0.z), b0, acc4);
            acc5 = fma(vec4<f32>(a0.z), b1, acc5);
            acc6 = fma(vec4<f32>(a0.w), b0, acc6);
            acc7 = fma(vec4<f32>(a0.w), b1, acc7);
            acc8 = fma(vec4<f32>(a1.x), b0, acc8);
            acc9 = fma(vec4<f32>(a1.x), b1, acc9);
            acc10 = fma(vec4<f32>(a1.y), b0, acc10);
            acc11 = fma(vec4<f32>(a1.y), b1, acc11);
            acc12 = fma(vec4<f32>(a1.z), b0, acc12);
            acc13 = fma(vec4<f32>(a1.z), b1, acc13);
            acc14 = fma(vec4<f32>(a1.w), b0, acc14);
            acc15 = fma(vec4<f32>(a1.w), b1, acc15);
        }
    }
    let r0 = row0 + ty * 4u;
    let r1 = r0 + 64u;
    let c0 = col0 + tx * 4u;
    let c1 = c0 + 64u;
    if ((mm_flags() & MM_SWIGLU) != 0u) {
        // Gates in columns 0..63 of the tile, their ups 64 further.
        let oc = col0 / 2u + tx * 4u;
        mm_swiglu4(r0, oc, acc0, acc1);
        mm_swiglu4(r0 + 1u, oc, acc2, acc3);
        mm_swiglu4(r0 + 2u, oc, acc4, acc5);
        mm_swiglu4(r0 + 3u, oc, acc6, acc7);
        mm_swiglu4(r1, oc, acc8, acc9);
        mm_swiglu4(r1 + 1u, oc, acc10, acc11);
        mm_swiglu4(r1 + 2u, oc, acc12, acc13);
        mm_swiglu4(r1 + 3u, oc, acc14, acc15);
        return;
    }
    mm_put4(r0, c0, acc0);
    mm_put4(r0, c1, acc1);
    mm_put4(r0 + 1u, c0, acc2);
    mm_put4(r0 + 1u, c1, acc3);
    mm_put4(r0 + 2u, c0, acc4);
    mm_put4(r0 + 2u, c1, acc5);
    mm_put4(r0 + 3u, c0, acc6);
    mm_put4(r0 + 3u, c1, acc7);
    mm_put4(r1, c0, acc8);
    mm_put4(r1, c1, acc9);
    mm_put4(r1 + 1u, c0, acc10);
    mm_put4(r1 + 1u, c1, acc11);
    mm_put4(r1 + 2u, c0, acc12);
    mm_put4(r1 + 2u, c1, acc13);
    mm_put4(r1 + 3u, c0, acc14);
    mm_put4(r1 + 3u, c1, acc15);
}

@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    gemm_f32(wg, li);
}
