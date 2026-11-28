// A row through a matrix of one kind on the plain path (`KIND`, written
// out by the program's source for each kind without a kernel of its own
// in `gemv_kquant.wgsl`: `eight_KIND` reads it): each slot's 32
// invocations take every 32nd 8 of its two rows' inputs.

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
        let xa = ints4(a);
        let xb = ints4(a + 4u);
        let w0 = eight_KIND(row0, e);
        let w1 = eight_KIND(row1, e);
        acc0 += dot(w0.a, xa) + dot(w0.b, xb);
        acc1 += dot(w1.a, xa) + dot(w1.b, xb);
    }
    gemv_finish(lid, wid.x, r0, gemv_sums_f32(lid, acc0, acc1));
}
