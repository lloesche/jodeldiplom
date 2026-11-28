// The matrices of Apple's attention (`attention8.wgsl`): Metal's
// simdgroup matrices of 8 × 8 (a subgroup of 32 holds one), A (8 rows × 8
// of k, f16), B (8 of k × 8 columns, f16) and C (8 × 8 f32 sums), loaded
// from and stored to the attention's two arrays (`a8_h` in f16, `a8_f` in
// f32) at a first place and the stride of its rows (by rows: value (r, c)
// at at + r · stride + c) or of its columns (by columns: at at + c ·
// stride + r). Only what the probe checks right on Metal (section 8): A by
// rows, B and C by columns, C stored by columns, and their product.
// `tests.rs` holds the attention to the processor through the same
// functions written out on f32s (`matrix8_emulated.wgsl`), where no GPU
// has these matrices.

alias M8a = coop_mat8x8<f16, A>;
alias M8b = coop_mat8x8<f16, B>;
alias M8c = coop_mat8x8<f32, C>;

// A by rows from `a8_h`.
fn m8_ld_a(at: u32, stride: u32) -> M8a {
    return coopLoadT<M8a>(&a8_h[at], stride);
}

// B by columns from `a8_h`.
fn m8_ld_b(at: u32, stride: u32) -> M8b {
    return coopLoad<M8b>(&a8_h[at], stride);
}

// A · B + C.
fn m8_mma(a: M8a, b: M8b, c: M8c) -> M8c {
    return coopMultiplyAdd(a, b, c);
}

// C by columns to `a8_f`.
fn m8_st_c(c: M8c, at: u32, stride: u32) {
    coopStore(c, &a8_f[at], stride);
}

// C by columns from `a8_f`.
fn m8_ld_c(at: u32, stride: u32) -> M8c {
    return coopLoad<M8c>(&a8_f[at], stride);
}
