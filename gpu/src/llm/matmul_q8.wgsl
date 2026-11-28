// The q8 path's matrix multiplies of one kind of matrix (`KIND`, written
// out for each kind by the program's source: `half_KIND` reads it).

@compute @workgroup_size(256)
fn gemv_q8_KIND(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let rb = row_bytes_KIND(k);
    let r0 = wid.x * GEMV_ROWS + (lid / 32u) * 2u;
    let row0 = mm_w() * 4u + r0 * rb;
    let row1 = row0 + rb;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var h = lid % 32u; h < k / 16u; h += 32u) {
        let g = h >> 1u;
        let a = mm_a() + h * 4u;
        let x = vec4<u32>(ints[a], ints[a + 1u], ints[a + 2u], ints[a + 3u]);
        let da = bitcast<f32>(ints[mm_qs() + g * 2u]);
        let sa = bitcast<f32>(ints[mm_qs() + g * 2u + 1u]);
        let w0 = half_KIND(row0, h);
        let w1 = half_KIND(row1, h);
        acc0 += w0.s * da * f32(dot16(w0.q, x)) + w0.m * sa;
        acc1 += w1.s * da * f32(dot16(w1.q, x)) + w1.m * sa;
    }
    gemv_finish(lid, wid.x, r0, gemv_sums(lid, acc0, acc1));
}

@compute @workgroup_size(256)
fn gemm_q8_KIND(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let groups = k / Q8_BLOCK;
    let rb = row_bytes_KIND(k);
    let row0 = wid.y * GEMM_TILE;
    let col0 = wid.x * GEMM_TILE;
    var acc = array<vec4<f32>, 4>();
    // Invocations 0 to 127 load the rows, the others each 16 of a column.
    let half = lid & 127u;
    let row = mm_w() * 4u + (col0 + half / 2u) * rb;
    for (var g = 0u; g < groups; g++) {
        if (lid < 128u) {
            gemm_rows_q8(lid, row0, g, groups);
        } else {
            let w = half_KIND(row, 2u * g + (half & 1u));
            tile_cols[half] = w.q;
            tile_col_scale[half] = vec2<f32>(w.s, w.m);
        }
        workgroupBarrier();
        gemm_step_q8(lid, &acc);
        workgroupBarrier();
    }
    gemm_finish(lid, row0, col0, acc);
}
