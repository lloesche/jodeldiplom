// The vision networks' smaller kernels, compiled after `prelude.wgsl`
// and `../common.wgsl`.

// ---- LayerNorm, a workgroup a row: row r's dim values (at off + (r /
// per)·outer + (r % per)·inner, so a row may be a head's slice of a token)
// to zero mean and unit variance, times aux[w_off + d] plus aux[b_off + d].
// f0 rows, f1 dim, f2 src_off, f3 src_outer, f4 src_inner, f5 per, f6
// dst_off, f7 dst_outer, f8 dst_inner, f9 w_off, f10 b_off, f11 eps (f32
// bits), f12 flags (1: the rows are read from dst, in place).
var<workgroup> red: array<f32, 64>;

fn reduce_sum(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (t < s) {
            red[t] += red[t + s];
        }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

fn ln_read(at: u32) -> f32 {
    if ((p.f12 & 1u) != 0u) {
        return dst[at];
    }
    return src[at];
}

@compute @workgroup_size(64)
fn layernorm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = wid.x + wid.y * groups.x;
    let t = lid.x;
    let live = r < p.f0;
    let dim = p.f1;
    let rr = min(r, p.f0 - 1u);
    let row_in = p.f2 + (rr / p.f5) * p.f3 + (rr % p.f5) * p.f4;
    let row_out = p.f6 + (rr / p.f5) * p.f7 + (rr % p.f5) * p.f8;
    var sum = 0.0;
    for (var d = t; d < dim; d += 64u) {
        sum += ln_read(row_in + d);
    }
    let mean = reduce_sum(t, sum) / f32(dim);
    var squares = 0.0;
    for (var d = t; d < dim; d += 64u) {
        let v = ln_read(row_in + d) - mean;
        squares += v * v;
    }
    let inv = inverseSqrt(reduce_sum(t, squares) / f32(dim) + bitcast<f32>(p.f11));
    if (!live) {
        return;
    }
    for (var d = t; d < dim; d += 64u) {
        dst[row_out + d] = (ln_read(row_in + d) - mean) * inv * aux[p.f9 + d] + aux[p.f10 + d];
    }
}

// ---- In place over f0 values from f2: f1 = 0 GELU, 1 ReLU, 2 sigmoid.
@compute @workgroup_size(256)
fn act(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    let at = p.f2 + i;
    let v = dst[at];
    switch p.f1 {
        case 0u: { dst[at] = gelu(v); }
        case 1u: { dst[at] = max(v, 0.0); }
        default: { dst[at] = 1.0 / (1.0 + exp(-v)); }
    }
}

// ---- A channel's mean and spread over a chunk of a view's pixels (for an
// instance norm): partial[((view · total + first + chunk) · C + c) · 2] =
// the chunk's mean, and + 1 the sum of its squared differences from it.
// f0 pixels a view (in this pass), f1 C, f2 views, f3 chunk (pixels), f4
// src_off, f5 first (the pass's first chunk, a band of a picture counting
// on from the bands before), f6 total (chunks a view in all).
// A workgroup is a chunk and 64 channels.
@compute @workgroup_size(64)
fn chan_sums(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let c = wid.y * 64u + lid.x;
    let chunks = (p.f0 + p.f3 - 1u) / p.f3;
    let view = wid.x / chunks;
    let chunk = wid.x % chunks;
    if (c >= p.f1 || view >= p.f2) {
        return;
    }
    let first = chunk * p.f3;
    let last = min(first + p.f3, p.f0);
    var sum = 0.0;
    for (var i = first; i < last; i++) {
        sum += src[p.f4 + (view * p.f0 + i) * p.f1 + c];
    }
    let mean = sum / f32(last - first);
    var spread = 0.0;
    for (var i = first; i < last; i++) {
        let d = src[p.f4 + (view * p.f0 + i) * p.f1 + c] - mean;
        spread += d * d;
    }
    let at = ((view * p.f6 + p.f5 + chunk) * p.f1 + c) * 2u;
    dst[at] = mean;
    dst[at + 1u] = spread;
}

// ---- The chunks put together (Chan's way: each chunk's mean and spread
// merged in turn): stats[(view · C + c) · 2] = the mean, + 1 = 1/√(variance
// + eps). f0 pixels a view (in all), f1 C, f2 views, f3 chunk, f4 eps (f32
// bits).
@compute @workgroup_size(256)
fn chan_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f1 * p.f2) {
        return;
    }
    let view = i / p.f1;
    let c = i % p.f1;
    let chunks = (p.f0 + p.f3 - 1u) / p.f3;
    var n = 0.0;
    var mean = 0.0;
    var spread = 0.0;
    for (var k = 0u; k < chunks; k++) {
        let at = ((view * chunks + k) * p.f1 + c) * 2u;
        let nb = f32(min(p.f3, p.f0 - k * p.f3));
        let d = src[at] - mean;
        let total = n + nb;
        mean += d * nb / total;
        spread += src[at + 1u] + d * d * n * nb / total;
        n = total;
    }
    dst[i * 2u] = mean;
    dst[i * 2u + 1u] = inverseSqrt(spread / n + bitcast<f32>(p.f4));
}

// ---- In place: dst[off + (view · P + i) · C + c] = (x - mean) / σ, then GELU
// if f5. f0 pixels a view, f1 C, f2 views, f4 dst_off, f5 GELU.
@compute @workgroup_size(256)
fn chan_apply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * p.f1 * p.f2) {
        return;
    }
    let c = i % p.f1;
    let view = i / (p.f0 * p.f1);
    let s = (view * p.f1 + c) * 2u;
    var v = (dst[p.f4 + i] - src[s]) * src[s + 1u];
    if (p.f5 != 0u) {
        v = gelu(v);
    }
    dst[p.f4 + i] = v;
}

// ---- Resizing, as PyTorch's interpolate: dst (+)= src at out's size.
// f0 views, f1 C, f2 in_h, f3 in_w, f4 out_h, f5 out_w, f6 src_off, f7
// dst_off, f8 mode (0 bilinear, 1 bilinear with the corners aligned, 2
// nearest), f9 add, f10 src pixel stride, f11 dst pixel stride.
fn src_coord(o: u32, n_in: u32, n_out: u32, mode: u32) -> f32 {
    if (mode == 1u) {
        if (n_out <= 1u) {
            return 0.0;
        }
        return f32(o) * f32(n_in - 1u) / f32(n_out - 1u);
    }
    return max((f32(o) + 0.5) * f32(n_in) / f32(n_out) - 0.5, 0.0);
}

@compute @workgroup_size(256)
fn resize(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let c_n = p.f1;
    let per = p.f4 * p.f5;
    if (i >= p.f0 * per * c_n) {
        return;
    }
    let c = i % c_n;
    let pix = i / c_n;
    let view = pix / per;
    let oy = (pix % per) / p.f5;
    let ox = pix % p.f5;
    let base = p.f6 + view * p.f2 * p.f3 * p.f10 + c;
    var v = 0.0;
    if (p.f8 == 2u) {
        let y = min(u32(floor(f32(oy) * f32(p.f2) / f32(p.f4))), p.f2 - 1u);
        let x = min(u32(floor(f32(ox) * f32(p.f3) / f32(p.f5))), p.f3 - 1u);
        v = src[base + (y * p.f3 + x) * p.f10];
    } else {
        let sy = src_coord(oy, p.f2, p.f4, p.f8);
        let sx = src_coord(ox, p.f3, p.f5, p.f8);
        let y0 = min(u32(floor(sy)), p.f2 - 1u);
        let x0 = min(u32(floor(sx)), p.f3 - 1u);
        let y1 = min(y0 + 1u, p.f2 - 1u);
        let x1 = min(x0 + 1u, p.f3 - 1u);
        let ly = sy - f32(y0);
        let lx = sx - f32(x0);
        let a = src[base + (y0 * p.f3 + x0) * p.f10];
        let b = src[base + (y0 * p.f3 + x1) * p.f10];
        let cc = src[base + (y1 * p.f3 + x0) * p.f10];
        let d = src[base + (y1 * p.f3 + x1) * p.f10];
        v = (1.0 - ly) * ((1.0 - lx) * a + lx * b) + ly * ((1.0 - lx) * cc + lx * d);
    }
    let at = p.f7 + (view * per + oy * p.f5 + ox) * p.f11 + c;
    if (p.f9 != 0u) {
        v += dst[at];
    }
    dst[at] = v;
}

// ---- Adaptive average pooling, as PyTorch's: out cell (i, j) the mean of
// rows ⌊i·in/out⌋..⌈(i+1)·in/out⌉ and the same for columns.
// f0 views, f1 C, f2 in_h, f3 in_w, f4 out_h, f5 out_w, f6 src_off, f7
// dst_off, f10 src pixel stride, f11 dst pixel stride.
@compute @workgroup_size(256)
fn pool(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let per = p.f4 * p.f5;
    if (i >= p.f0 * per * p.f1) {
        return;
    }
    let c = i % p.f1;
    let pix = i / p.f1;
    let view = pix / per;
    let oy = (pix % per) / p.f5;
    let ox = pix % p.f5;
    let y0 = (oy * p.f2) / p.f4;
    let y1 = ((oy + 1u) * p.f2 + p.f4 - 1u) / p.f4;
    let x0 = (ox * p.f3) / p.f5;
    let x1 = ((ox + 1u) * p.f3 + p.f5 - 1u) / p.f5;
    let base = p.f6 + view * p.f2 * p.f3 * p.f10 + c;
    var sum = 0.0;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            sum += src[base + (y * p.f3 + x) * p.f10];
        }
    }
    dst[p.f7 + (view * per + oy * p.f5 + ox) * p.f11 + c] = sum / f32((y1 - y0) * (x1 - x0));
}

// ---- A strided copy: dst[d_off + i0·d0 + i1·d1 + i2·d2 + i3] (+)=
// src[s_off + i0·s0 + i1·s1 + i2·s2 + i3]. f0..f3 the counts (n3 innermost),
// f4 s_off, f5..f7 s0..s2, f8 d_off, f9..f11 d0..d2, f12 add.
@compute @workgroup_size(256)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * p.f1 * p.f2 * p.f3) {
        return;
    }
    let i3 = i % p.f3;
    let r2 = i / p.f3;
    let i2 = r2 % p.f2;
    let r1 = r2 / p.f2;
    let i1 = r1 % p.f1;
    let i0 = r1 / p.f1;
    let v = src[p.f4 + i0 * p.f5 + i1 * p.f6 + i2 * p.f7 + i3];
    let at = p.f8 + i0 * p.f9 + i1 * p.f10 + i2 * p.f11 + i3;
    if (p.f12 != 0u) {
        dst[at] += v;
    } else {
        dst[at] = v;
    }
}

// ---- Shifted windows (Swin). The windows of a view's image (H × W × C)
// padded to Hp × Wp and rolled back by `shift`: dst[((view · nW + w) · S² +
// l) · C + c], zero where the padding is. f0 views, f1 H, f2 W, f3 C, f4
// size S, f5 shift, f6 Hp, f7 Wp.
@compute @workgroup_size(256)
fn win_part(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let s = p.f4;
    let nwx = p.f7 / s;
    let per_view = p.f6 * p.f7;
    if (i >= p.f0 * per_view * p.f3) {
        return;
    }
    let c = i % p.f3;
    let tok = i / p.f3;
    let view = tok / per_view;
    let in_view = tok % per_view;
    let w = in_view / (s * s);
    let l = in_view % (s * s);
    let py = (w / nwx) * s + l / s;
    let px = (w % nwx) * s + l % s;
    let y = (py + p.f5) % p.f6;
    let x = (px + p.f5) % p.f7;
    var v = 0.0;
    if (y < p.f1 && x < p.f2) {
        v = src[((view * p.f1 + y) * p.f2 + x) * p.f3 + c];
    }
    dst[i] = v;
}

// ---- The windows back into the image, added to it: dst[(view, y, x, c)]
// += src[the window token at (y, x) rolled by `shift`]. f0 views, f1 H, f2
// W, f3 C, f4 S, f5 shift, f6 Hp, f7 Wp.
@compute @workgroup_size(256)
fn win_back(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let s = p.f4;
    if (i >= p.f0 * p.f1 * p.f2 * p.f3) {
        return;
    }
    let c = i % p.f3;
    let pix = i / p.f3;
    let view = pix / (p.f1 * p.f2);
    let y = (pix / p.f2) % p.f1;
    let x = pix % p.f2;
    let py = (y + p.f6 - p.f5) % p.f6;
    let px = (x + p.f7 - p.f5) % p.f7;
    let nwx = p.f7 / s;
    let w = (py / s) * nwx + px / s;
    let l = (py % s) * s + px % s;
    dst[i] += src[((view * (p.f6 * p.f7) + w * s * s + l) * p.f3 + c)];
}

// ---- 2-D rotary positions on the queries and keys of a qkv buffer, in
// place: a head's first half turned by its token's row, the second by its
// column, each half's two quarters as a pair, the angle's cosine and sine
// from aux's table (a position's 16 cosines then 16 sines, from f6).
// Token 0 is at (0, 0); token t at its patch's row and column plus one, or
// with f5 = 1 every patch at (1, 1). f0 tokens, f1 heads, f2 the row
// stride, f3 the grid's columns, f4 C (the keys' offset), f5 mode, f6 the
// table. The heads are 64 wide.
@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    // A token's heads, queries and keys, halves, 16 pairs each.
    if (i >= p.f0 * p.f1 * 2u * 2u * 16u) {
        return;
    }
    let j = i % 16u;
    let half = (i / 16u) % 2u;
    let which = (i / 32u) % 2u;
    let head = (i / 64u) % p.f1;
    let t = i / (64u * p.f1);
    var pos = 0u;
    if (t > 0u) {
        if (p.f5 == 1u) {
            pos = 1u;
        } else if (half == 0u) {
            pos = (t - 1u) / p.f3 + 1u;
        } else {
            pos = (t - 1u) % p.f3 + 1u;
        }
    }
    let at = t * p.f2 + which * p.f4 + head * 64u + half * 32u + j;
    let cs = aux[p.f6 + pos * 32u + j];
    let sn = aux[p.f6 + pos * 32u + 16u + j];
    let a = dst[at];
    let b = dst[at + 16u];
    dst[at] = a * cs - b * sn;
    dst[at + 16u] = b * cs + a * sn;
}

// ---- A table into dst: dst[d_off + i] = (or +=, f3 = 1) aux[a_off + i % period].
// f0 count, f1 d_off, f2 a_off, f3 mode, f4 period.
@compute @workgroup_size(256)
fn table(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    let v = aux[p.f2 + i % p.f4];
    if (p.f3 == 1u) {
        dst[p.f1 + i] += v;
    } else {
        dst[p.f1 + i] = v;
    }
}

// ---- Four views (the quarters of a 2h × 2w picture, in rows) each times
// a one-channel map of h × w over the whole picture, read at half the
// picture's resolution (nearest): dst[(v, y, x, c)] *= src[m_off + map at
// ((qy·h + y) / 2, (qx·w + x) / 2)]. f0 h, f1 w, f2 C, f3 m_off, f4 d_off.
@compute @workgroup_size(256)
fn mulmap(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let h = p.f0;
    let w = p.f1;
    if (i >= 4u * h * w * p.f2) {
        return;
    }
    let pix = i / p.f2;
    let v = pix / (h * w);
    let y = (pix / w) % h;
    let x = pix % w;
    let my = ((v / 2u) * h + y) / 2u;
    let mx = ((v % 2u) * w + x) / 2u;
    dst[p.f4 + i] *= src[p.f3 + my * w + mx];
}
