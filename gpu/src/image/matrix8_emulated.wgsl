// `matrix8.wgsl`'s matrices written out on f32s, for the tests only: each
// invocation holds a whole matrix (by columns, value (r, c) at c · 8 + r),
// so a load reads all of it, a product is 512 multiply-adds, and a store
// writes all of it (every invocation of the subgroup the same values).
// The attention's code around them runs as it does on Metal, so a GPU
// without 8 × 8 matrices (the 5080) holds it to the processor.

struct M8 {
    v: array<f32, 64>,
}

alias M8a = M8;
alias M8b = M8;
alias M8c = M8;

fn m8_ld_a(at: u32, stride: u32) -> M8 {
    var m: M8;
    for (var c = 0u; c < 8u; c++) {
        for (var r = 0u; r < 8u; r++) {
            m.v[c * 8u + r] = f32(a8_h[at + r * stride + c]);
        }
    }
    return m;
}

fn m8_ld_b(at: u32, stride: u32) -> M8 {
    var m: M8;
    for (var c = 0u; c < 8u; c++) {
        for (var r = 0u; r < 8u; r++) {
            m.v[c * 8u + r] = f32(a8_h[at + c * stride + r]);
        }
    }
    return m;
}

fn m8_mma(a: M8, b: M8, c: M8) -> M8 {
    var d = c;
    for (var col = 0u; col < 8u; col++) {
        for (var r = 0u; r < 8u; r++) {
            var s = d.v[col * 8u + r];
            for (var k = 0u; k < 8u; k++) {
                s += a.v[k * 8u + r] * b.v[col * 8u + k];
            }
            d.v[col * 8u + r] = s;
        }
    }
    return d;
}

fn m8_st_c(c: M8, at: u32, stride: u32) {
    for (var col = 0u; col < 8u; col++) {
        for (var r = 0u; r < 8u; r++) {
            a8_f[at + col * stride + r] = c.v[col * 8u + r];
        }
    }
}

fn m8_ld_c(at: u32, stride: u32) -> M8 {
    var m: M8;
    for (var c = 0u; c < 8u; c++) {
        for (var r = 0u; r < 8u; r++) {
            m.v[c * 8u + r] = a8_f[at + c * stride + r];
        }
    }
    return m;
}
