// The tiles of `coop.wgsl` as four cooperative matrices of 8 × 8 × 8 each
// (in f16 with f32 sums: Metal's simdgroup matrices, a subgroup of 32 to a
// matrix), the quarter at rows 8·i.., columns 8·j.. as `mij`; the
// multiply's loads, stores and product as `coop16.wgsl`'s, a quarter at a
// time (the attention takes its 8 × 8 matrices as they are:
// `matrix8.wgsl`, `attention8.wgsl`). A quarter by rows starts 8·i rows
// and 8·j values on, by columns 8·j columns and 8·i values on.

// The multiply's step of k (`coop.wgsl` sizes its tiles by it): 32, so
// the f16 tiles take 20 KB and a kernel stays within 32 KB.
const CM_K: u32 = 32u;

struct CmA {
    m00: coop_mat8x8<f16, A>, m01: coop_mat8x8<f16, A>,
    m10: coop_mat8x8<f16, A>, m11: coop_mat8x8<f16, A>,
}

struct CmB {
    m00: coop_mat8x8<f16, B>, m01: coop_mat8x8<f16, B>,
    m10: coop_mat8x8<f16, B>, m11: coop_mat8x8<f16, B>,
}

struct CmC {
    m00: coop_mat8x8<f32, C>, m01: coop_mat8x8<f32, C>,
    m10: coop_mat8x8<f32, C>, m11: coop_mat8x8<f32, C>,
}

// A by rows from `cm_h`.
fn cm_ld_at(at: u32, stride: u32) -> CmA {
    let d = 8u * stride;
    return CmA(
        coopLoadT<coop_mat8x8<f16, A>>(&cm_h[at], stride),
        coopLoadT<coop_mat8x8<f16, A>>(&cm_h[at + 8u], stride),
        coopLoadT<coop_mat8x8<f16, A>>(&cm_h[at + d], stride),
        coopLoadT<coop_mat8x8<f16, A>>(&cm_h[at + d + 8u], stride),
    );
}

// B by columns from `cm_h`.
fn cm_ld_b(at: u32, stride: u32) -> CmB {
    let d = 8u * stride;
    return CmB(
        coopLoad<coop_mat8x8<f16, B>>(&cm_h[at], stride),
        coopLoad<coop_mat8x8<f16, B>>(&cm_h[at + d], stride),
        coopLoad<coop_mat8x8<f16, B>>(&cm_h[at + 8u], stride),
        coopLoad<coop_mat8x8<f16, B>>(&cm_h[at + d + 8u], stride),
    );
}

// A · B + C: each quarter of C plus its row of A's quarters times its
// column of B's.
fn cm_mma(a: CmA, b: CmB, c: CmC) -> CmC {
    return CmC(
        coopMultiplyAdd(a.m01, b.m10, coopMultiplyAdd(a.m00, b.m00, c.m00)),
        coopMultiplyAdd(a.m01, b.m11, coopMultiplyAdd(a.m00, b.m01, c.m01)),
        coopMultiplyAdd(a.m11, b.m10, coopMultiplyAdd(a.m10, b.m00, c.m10)),
        coopMultiplyAdd(a.m11, b.m11, coopMultiplyAdd(a.m10, b.m01, c.m11)),
    );
}

// C by columns to `cm_f`.
fn cm_st_c(c: CmC, at: u32, stride: u32) {
    let d = 8u * stride;
    coopStore(c.m00, &cm_f[at], stride);
    coopStore(c.m01, &cm_f[at + d], stride);
    coopStore(c.m10, &cm_f[at + 8u], stride);
    coopStore(c.m11, &cm_f[at + d + 8u], stride);
}

// C by columns from `cm_f`.
fn cm_ld_c(at: u32, stride: u32) -> CmC {
    let d = 8u * stride;
    return CmC(
        coopLoad<coop_mat8x8<f32, C>>(&cm_f[at], stride),
        coopLoad<coop_mat8x8<f32, C>>(&cm_f[at + d], stride),
        coopLoad<coop_mat8x8<f32, C>>(&cm_f[at + 8u], stride),
        coopLoad<coop_mat8x8<f32, C>>(&cm_f[at + d + 8u], stride),
    );
}
