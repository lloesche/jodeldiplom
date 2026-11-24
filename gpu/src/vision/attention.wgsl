// ---- Attention as two batched matrix multiplies around a softmax, compiled
// after conv.wgsl (its bindings and tiles): the scores q · kᵀ · scale for a
// chunk of queries ([`bmm`]), each row's softmax (Swin's relative positions
// and mask added first) in place ([`softmax`]), then the scores times the
// values ([`bmm`] again).

// ---- A batched matrix multiply: for each z = group · heads + head,
// C[z] = A[z] (m × k) · B[z] (k × n) · scale, A from src, B from src2, C to
// dst: A at a_off + g·a_gs + h·a_hs + r·lda + kk, B at b_off + g·b_gs +
// h·b_hs + kk·ldbk + c·ldbn, C at c_off + g·c_gs + h·c_hs + r·ldc + c.
// f0 m, f1 n, f2 k, f3 heads, f4 a_off, f5 a_gs, f6 a_hs, f7 lda, f8 b_off,
// f9 b_gs, f10 b_hs, f11 ldbk, f12 ldbn, f13 c_off, f14 c_gs, f15 c_hs, f16
// ldc, f17 scale (f32 bits). Tiles of 64 × 64 as `conv`.
@compute @workgroup_size(16, 16)
fn bmm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = p.f0;
    let n = p.f1;
    let k = p.f2;
    let g = wid.z / p.f3;
    let h = wid.z % p.f3;
    let a_base = p.f4 + g * p.f5 + h * p.f6;
    let b_base = p.f8 + g * p.f9 + h * p.f10;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let t = lid.y * 16u + lid.x;
    var acc = Acc(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    let ar = row0 + (t / 16u) * 4u;
    let aq = t % 16u;
    let bk = t / 16u;
    let bc = col0 + (t % 16u) * 4u;
    for (var k0 = 0u; k0 < k; k0 += 16u) {
        let ka = k0 + aq;
        var a = vec4<f32>(0.0);
        if (ka < k) {
            for (var j = 0u; j < 4u; j++) {
                if (ar + j < m) {
                    a[j] = src[a_base + (ar + j) * p.f7 + ka];
                }
            }
        }
        tile_a[aq * 16u + t / 16u] = a;
        let kb = k0 + bk;
        var b = vec4<f32>(0.0);
        if (kb < k) {
            for (var j = 0u; j < 4u; j++) {
                if (bc + j < n) {
                    b[j] = src2[b_base + kb * p.f11 + (bc + j) * p.f12];
                }
            }
        }
        tile_b[t] = b;
        workgroupBarrier();
        acc = mm_tile(acc, lid.x, lid.y);
        workgroupBarrier();
    }
    let scale = bitcast<f32>(p.f17);
    let c_base = p.f13 + g * p.f14 + h * p.f15;
    let r0 = row0 + lid.y * 4u;
    let c0 = col0 + lid.x * 4u;
    let rows = array<vec4<f32>, 4>(acc.r0, acc.r1, acc.r2, acc.r3);
    for (var i = 0u; i < 4u; i++) {
        let r = r0 + i;
        if (r < m) {
            for (var j = 0u; j < 4u; j++) {
                if (c0 + j < n) {
                    dst[c_base + r * p.f16 + c0 + j] = rows[i][j] * scale;
                }
            }
        }
    }
}

// The region of the padded picture a window's token is in (for the mask of
// shifted windows): 0, 1 or 2 down and across, as Swin numbers them.
// win: size | shift << 8 | windows across << 16 | windows down << 24.
fn swin_region(win: u32, g: u32, l: u32) -> u32 {
    let s = win & 255u;
    let shift = (win >> 8u) & 255u;
    let nwx = (win >> 16u) & 255u;
    let nwy = win >> 24u;
    let w = g % (nwx * nwy);
    let py = (w / nwx) * s + l / s;
    let px = (w % nwx) * s + l % s;
    let hp = nwy * s;
    let wp = nwx * s;
    var ry = 0u;
    if (py >= hp - shift) {
        ry = 2u;
    } else if (py >= hp - s) {
        ry = 1u;
    }
    var rx = 0u;
    if (px >= wp - shift) {
        rx = 2u;
    } else if (px >= wp - s) {
        rx = 1u;
    }
    return ry * 3u + rx;
}

// ---- Each row's softmax in place, a workgroup a row: rows of n scores from
// f2, row r of query i = r % nq of z = r / nq (group g0 + z / heads, head z
// % heads). With f5 = 1, Swin's bias first: the relative positions' table at
// f7 (a row of `heads` per relative position) and, when shifted, -100 where
// the two tokens' regions differ. f0 rows, f1 n, f2 off, f3 nq, f4 heads,
// f5 bias, f6 Swin's window (as swin_region), f8 g0.
@compute @workgroup_size(64)
fn softmax(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = wid.x + wid.y * groups.x;
    let t = lid.x;
    let live = r < p.f0;
    let rr = min(r, p.f0 - 1u);
    let n = p.f1;
    let base = p.f2 + rr * n;
    let i = rr % p.f3;
    let z = rr / p.f3;
    let g = p.f8 + z / p.f4;
    let head = z % p.f4;
    let s = p.f6 & 255u;
    let shift = (p.f6 >> 8u) & 255u;
    var best = -3.0e38;
    for (var j = t; j < n; j += 64u) {
        var v = dst[base + j];
        if (p.f5 == 1u) {
            let rel = (i / s + s - 1u - j / s) * (2u * s - 1u) + (i % s + s - 1u - j % s);
            v += aux[p.f7 + rel * p.f4 + head];
            if (shift > 0u && swin_region(p.f6, g, i) != swin_region(p.f6, g, j)) {
                v -= 100.0;
            }
            if (live) {
                dst[base + j] = v;
            }
        }
        best = max(best, v);
    }
    red[t] = best;
    workgroupBarrier();
    for (var w = 32u; w > 0u; w >>= 1u) {
        if (t < w) {
            red[t] = max(red[t], red[t + w]);
        }
        workgroupBarrier();
    }
    let most = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = t; j < n; j += 64u) {
        sum += exp(dst[base + j] - most);
    }
    let total = reduce_sum(t, sum);
    if (!live) {
        return;
    }
    let inv = 1.0 / total;
    for (var j = t; j < n; j += 64u) {
        dst[base + j] = exp(dst[base + j] - most) * inv;
    }
}
