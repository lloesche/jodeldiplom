// The f32 GEMVs' slot sums on the 8-bit path (its F32 and BF16 matrices'
// sums as they always were) and on a GPU without subgroups of 32: through
// workgroup memory, as the q8 GEMVs'.
fn gemv_sums_f32(lid: u32, acc0: f32, acc1: f32) -> vec2<f32> {
    return gemv_sums(lid, acc0, acc1);
}
