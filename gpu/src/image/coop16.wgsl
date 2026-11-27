// The tiles of `coop.wgsl` as cooperative matrices of 16 × 16 × 16 (in
// f16 with f32 sums: NVIDIA's tensor cores through Vulkan). Each load and
// store takes a tile's first place in `cm_h` or `cm_f` and the stride of
// its rows (by rows: value (r, c) at at + r · stride + c) or of its columns
// (by columns: at at + c · stride + r).

// The multiply's step of k (`coop.wgsl` sizes its tiles by it).
const CM_K: u32 = 64u;

alias CmA = coop_mat16x16<f16, A>;
alias CmB = coop_mat16x16<f16, B>;
alias CmC = coop_mat16x16<f32, C>;

// A by rows from `cm_h`.
fn cm_ld_at(at: u32, stride: u32) -> CmA {
    return coopLoadT<CmA>(&cm_h[at], stride);
}

// B by columns from `cm_h`.
fn cm_ld_b(at: u32, stride: u32) -> CmB {
    return coopLoad<CmB>(&cm_h[at], stride);
}

// A · B + C (`cm_mma`) is the matrices' own `coopMultiplyAdd`, written
// into the kernels in its place (`Tiles::kernel`): made through a function
// of its own, the 5080's multiply took 3 % longer.

// C by columns to `cm_f`.
fn cm_st_c(c: CmC, at: u32, stride: u32) {
    coopStore(c, &cm_f[at], stride);
}

// C by columns from `cm_f`.
fn cm_ld_c(at: u32, stride: u32) -> CmC {
    return coopLoad<CmC>(&cm_f[at], stride);
}
