// The f32 GEMVs' slot sums on the f32 path, on a GPU whose subgroups are
// 32 wide (a slot is one): each slot's two sums added across it, on all of
// its invocations.
fn gemv_sums_f32(lid: u32, acc0: f32, acc1: f32) -> vec2<f32> {
    return vec2<f32>(subgroupAdd(acc0), subgroupAdd(acc1));
}
