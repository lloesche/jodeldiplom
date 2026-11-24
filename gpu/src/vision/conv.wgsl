// The vision networks' convolution (and linear layers), compiled after
// `prelude.wgsl` and `../common.wgsl`.

// ---- A convolution as a matrix multiply, its patches read where they lie:
// dst[c_off + r·ldc + col] (+)= Σk A[r][k] · aux[w_off + k·n + col] (+ bias),
// row r a pixel of the output (view, y, x), k = (ky·size + kx)·cin + ci, A
// the input at (y·stride + ky - pad, x·stride + kx - pad), channel ci (0
// outside). A linear layer is a 1 × 1 convolution over a row of pixels.
// f0 rows (views · out_h · out_w), f1 n (columns), f2 k (size² · cin),
// f3 a_off, f4 w_off, f5 bias_off, f6 c_off, f7 ldc, f8 flags, f9 cin,
// f10 lda (a pixel's stride in the input), f11 in_w, f12 in_h (as the
// convolution sees them), f13 out_w, f14 out_h, f15 size | stride << 8 |
// pad << 16 | up << 24 (up: the output's pixels spread over up × up, each
// column (dy·up + dx)·n/up² + c; a transposed convolution with its stride
// as its size), f16 the first output row (a band of the output: its rows
// counted from there, stored from c_off), f17 the input's first row (a band
// of the input: rows above and below it read as 0, as the picture's edge).
// Flags: 1 add the bias, 2 add to dst, 4 ReLU on the input, 8 GELU on the
// output, 16 ReLU on the output, 32 the input at half its size (each pixel
// read for two by two: nearest upsampling on the way in).
// Tiles of 64 × 64 through k in steps of 16; each invocation keeps 4 rows ×
// 4 columns in four vectors (`common.wgsl`'s tiles).

// Where row r's pixel reads from: its view's start in the input (in
// pixels), and its first input row and column (before ky, kx).
struct RowAt {
    ok: bool,
    base: i32,
    y0: i32,
    x0: i32,
}

fn row_at(r: u32) -> RowAt {
    if (r >= p.f0) {
        return RowAt(false, 0, 0, 0);
    }
    let per = p.f13 * p.f14;
    let view = r / per;
    let rem = r % per;
    let oy = rem / p.f13 + p.f16;
    let ox = rem % p.f13;
    let stride = i32((p.f15 >> 8u) & 255u);
    let pad = i32((p.f15 >> 16u) & 255u);
    var real_h = p.f12;
    var real_w = p.f11;
    if ((p.f8 & 32u) != 0u) {
        real_h = p.f12 / 2u;
        real_w = p.f11 / 2u;
    }
    return RowAt(true, i32(view * real_h * real_w), i32(oy) * stride - pad - i32(p.f17), i32(ox) * stride - pad);
}

// A at row `at`, k (ky, kx, ci).
fn a_at(at: RowAt, ky: u32, kx: u32, ci: u32, valid: bool) -> f32 {
    if (!at.ok || !valid) {
        return 0.0;
    }
    let y = at.y0 + i32(ky);
    let x = at.x0 + i32(kx);
    if (y < 0 || x < 0 || y >= i32(p.f12) || x >= i32(p.f11)) {
        return 0.0;
    }
    var yy = u32(y);
    var xx = u32(x);
    var w = p.f11;
    if ((p.f8 & 32u) != 0u) {
        yy = yy / 2u;
        xx = xx / 2u;
        w = p.f11 / 2u;
    }
    let pixel = u32(at.base) + yy * w + xx;
    var v = src[p.f3 + pixel * p.f10 + ci];
    if ((p.f8 & 4u) != 0u) {
        v = max(v, 0.0);
    }
    return v;
}

fn conv_store(r: u32, col: u32, value: f32) {
    if (r >= p.f0 || col >= p.f1) {
        return;
    }
    var v = value;
    let up = p.f15 >> 24u;
    var c = col;
    var at = p.f6 + r * p.f7 + col;
    if (up > 1u) {
        let n = p.f1 / (up * up);
        let d = col / n;
        c = col % n;
        let per = p.f13 * p.f14;
        let view = r / per;
        let rem = r % per;
        let y = (rem / p.f13) * up + d / up;
        let x = (rem % p.f13) * up + d % up;
        at = p.f6 + ((view * p.f14 * up + y) * p.f13 * up + x) * p.f7 + c;
    }
    if ((p.f8 & 1u) != 0u) {
        v += aux[p.f5 + c];
    }
    if ((p.f8 & 2u) != 0u) {
        v += dst[at];
    }
    if ((p.f8 & 8u) != 0u) {
        v = gelu(v);
    }
    if ((p.f8 & 16u) != 0u) {
        v = max(v, 0.0);
    }
    dst[at] = v;
}

@compute @workgroup_size(16, 16)
fn conv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = p.f1;
    let k = p.f2;
    let cin = p.f9;
    let size = p.f15 & 255u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let t = lid.y * 16u + lid.x;
    var acc = Acc(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    // This invocation's loads: A rows 4·(t/16).. at k t%16; B k t/16,
    // columns (t%16)*4..
    let ar = row0 + (t / 16u) * 4u;
    let at0 = row_at(ar);
    let at1 = row_at(ar + 1u);
    let at2 = row_at(ar + 2u);
    let at3 = row_at(ar + 3u);
    let aq = t % 16u;
    let bk = t / 16u;
    let bc = col0 + (t % 16u) * 4u;
    for (var k0 = 0u; k0 < k; k0 += 16u) {
        let ka = k0 + aq;
        let valid = ka < k;
        let kpos = ka / cin;
        let ci = ka % cin;
        let ky = kpos / size;
        let kx = kpos % size;
        tile_a[aq * 16u + t / 16u] = vec4<f32>(
            a_at(at0, ky, kx, ci, valid),
            a_at(at1, ky, kx, ci, valid),
            a_at(at2, ky, kx, ci, valid),
            a_at(at3, ky, kx, ci, valid),
        );
        let kb = k0 + bk;
        var b = vec4<f32>(0.0);
        if (kb < k) {
            let w = p.f4 + kb * n + bc;
            if (bc + 3u < n) {
                b = vec4<f32>(aux[w], aux[w + 1u], aux[w + 2u], aux[w + 3u]);
            } else {
                for (var j = 0u; j < 4u; j++) {
                    if (bc + j < n) {
                        b[j] = aux[w + j];
                    }
                }
            }
        }
        tile_b[t] = b;
        workgroupBarrier();
        acc = mm_tile(acc, lid.x, lid.y);
        workgroupBarrier();
    }
    let r0 = row0 + lid.y * 4u;
    let c0 = col0 + lid.x * 4u;
    let rows = array<vec4<f32>, 4>(acc.r0, acc.r1, acc.r2, acc.r3);
    for (var i = 0u; i < 4u; i++) {
        let v = rows[i];
        conv_store(r0 + i, c0, v.x);
        conv_store(r0 + i, c0 + 1u, v.y);
        conv_store(r0 + i, c0 + 2u, v.z);
        conv_store(r0 + i, c0 + 3u, v.w);
    }
}
