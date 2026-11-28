// The plain path's many rows through a matrix of one kind (`KIND`,
// written out for each kind by the program's source: `eight_KIND` reads
// it).

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
