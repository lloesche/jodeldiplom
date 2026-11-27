// The plain path's matrix multiplies of one kind of matrix (`KIND`,
// written out for each kind by the program's source: `eight_KIND` reads
// it).

@compute @workgroup_size(256)
fn gemv_f32_KIND(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let rb = row_bytes_KIND(k);
    let r0 = wid.x * GEMV_ROWS + (lid / 32u) * 2u;
    let row0 = mm_w() * 4u + r0 * rb;
    let row1 = row0 + rb;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var e = lid % 32u; e < k / 8u; e += 32u) {
        let a = mm_a() + e * 8u;
        let xa = bitcast<vec4<f32>>(vec4<u32>(ints[a], ints[a + 1u], ints[a + 2u], ints[a + 3u]));
        let xb = bitcast<vec4<f32>>(vec4<u32>(ints[a + 4u], ints[a + 5u], ints[a + 6u], ints[a + 7u]));
        let w0 = eight_KIND(row0, e);
        let w1 = eight_KIND(row1, e);
        acc0 += dot(w0.a, xa) + dot(w0.b, xb);
        acc1 += dot(w1.a, xa) + dot(w1.b, xb);
    }
    gemv_finish(lid, wid.x, r0, acc0, acc1);
}

@compute @workgroup_size(256)
fn gemm_f32_KIND(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let groups = k / 32u;
    let rb = row_bytes_KIND(k);
    let row0 = wid.y * GEMM_TILE;
    let col0 = wid.x * GEMM_TILE;
    let c = lid / 4u;
    let part = lid % 4u;
    let row = mm_w() * 4u + (col0 + c) * rb;
    var acc = array<vec4<f32>, 4>();
    for (var g = 0u; g < groups; g++) {
        gemm_rows_f32(lid, row0, g);
        gemm_put_w(c, part, eight_KIND(row, g * 4u + part));
        workgroupBarrier();
        gemm_step_f32(lid, &acc);
        workgroupBarrier();
    }
    gemm_finish(lid, row0, col0, acc);
}
