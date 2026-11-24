// The speech recogniser's kernels (Qwen3-ASR-0.6B), compiled after
// kernels.wgsl: they share its bindings (src, aux, dst, p) and helpers
// (linear_id, gelu), and add the 4-bit weights, `qw`.
//
// A 4-bit matrix ([n][k], a row an output) is groups of 32 values along
// its rows, each 5 u32s: four of nibbles (value j in word j / 8, bits
// 4·(j % 8)), then the scale (low half) and offset (high half) as halves;
// a value is offset + nibble · scale (jodel_lyrics::asr::quant).

@group(0) @binding(4) var<storage, read> qw: array<u32>;

// ---- dst[c_off + r*ldc + c] (+)= Σk src[a_off + r*lda + k] · W[c][k]
// (+ aux[bias_off + c]), then GELU if asked. W 4-bit at qw[w_off]. f0 m,
// f1 n, f2 k (whole groups), f3 a_off, f4 lda, f5 w_off, f6 c_off, f7 ldc,
// f8 bias_off, f9 flags (1: add the bias, 2: add to dst, 4: GELU, 8: the
// matrix at 8 bits: groups of 9 u32s, 8 of bytes, value j in word j / 4,
// bits 8·(j % 4), then the scale and offset; 16: at 16 bits, groups of 16
// u32s of two halves, value j in word j / 2, the low half first). Tiles
// of 128 × 64 through k a group (32) at a time, held as vectors of four
// rows and of four columns, the next group preloaded into registers while
// one is multiplied; each invocation keeps 8 rows × 4 columns, each output's
// products added in the order of k.
var<workgroup> qa_a: array<vec4<f32>, 1024>;
var<workgroup> qa_b: array<vec4<f32>, 512>;

fn qa_a_at(r: u32, ka: u32) -> f32 {
    if (r < p.f0) {
        return src[p.f3 + r * p.f4 + ka];
    }
    return 0.0;
}

// The weight of column c (all n columns' groups apart) at value j of group
// g, as a value (0 past the last column).
fn qa_w_at(c: u32, groups: u32, g: u32, j: u32) -> f32 {
    if (c >= p.f1) {
        return 0.0;
    }
    if ((p.f9 & 16u) != 0u) {
        let v = unpack2x16float(qw[p.f5 + (c * groups + g) * 16u + j / 2u]);
        return select(v.x, v.y, (j & 1u) == 1u);
    }
    if ((p.f9 & 8u) != 0u) {
        let base = p.f5 + (c * groups + g) * 9u;
        let so = unpack2x16float(qw[base + 8u]);
        return so.y + f32((qw[base + j / 4u] >> (8u * (j % 4u))) & 255u) * so.x;
    }
    let base = p.f5 + (c * groups + g) * 5u;
    let so = unpack2x16float(qw[base + 4u]);
    return so.y + f32((qw[base + j / 8u] >> (4u * (j % 8u))) & 15u) * so.x;
}

// An invocation's share of a group's tiles (in registers, apart, so no
// compiler takes them through memory): four vectors of A (rows 4·q.. at
// k, q = aq + 8·h for h = 0..3) and two of B (columns c.. at values j and
// j + 1).
struct QaLoads {
    a0: vec4<f32>,
    a1: vec4<f32>,
    a2: vec4<f32>,
    a3: vec4<f32>,
    b0: vec4<f32>,
    b1: vec4<f32>,
}

fn qa_rows4(r: u32, k: u32) -> vec4<f32> {
    return vec4<f32>(qa_a_at(r, k), qa_a_at(r + 1u, k), qa_a_at(r + 2u, k), qa_a_at(r + 3u, k));
}

fn qa_cols4(c: u32, groups: u32, g: u32, j: u32) -> vec4<f32> {
    return vec4<f32>(qa_w_at(c, groups, g, j), qa_w_at(c + 1u, groups, g, j),
                     qa_w_at(c + 2u, groups, g, j), qa_w_at(c + 3u, groups, g, j));
}

fn qa_loads(g: u32, row0: u32, aq: u32, ak: u32, c: u32, groups: u32, j: u32) -> QaLoads {
    let k = g * 32u + ak;
    let r = row0 + 4u * aq;
    return QaLoads(qa_rows4(r, k), qa_rows4(r + 32u, k), qa_rows4(r + 64u, k), qa_rows4(r + 96u, k),
                   qa_cols4(c, groups, g, j), qa_cols4(c, groups, g, j + 1u));
}

@compute @workgroup_size(16, 16)
fn qa_matmul(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = p.f2;
    let groups = k / 32u;
    let row0 = wid.y * 128u;
    let col0 = wid.x * 64u;
    let t = lid.y * 16u + lid.x;
    var r0 = vec4<f32>(0.0);
    var r1 = vec4<f32>(0.0);
    var r2 = vec4<f32>(0.0);
    var r3 = vec4<f32>(0.0);
    var r4 = vec4<f32>(0.0);
    var r5 = vec4<f32>(0.0);
    var r6 = vec4<f32>(0.0);
    var r7 = vec4<f32>(0.0);
    // Loads: A at k t % 32 for the four rows 4·q.., q = t / 32 + 8·h; B
    // the four columns 4·(t / 16).. at values 2·(t % 16) and the next.
    let ak = t % 32u;
    let aq = t / 32u;
    let bq = t / 16u;
    let bj = 2u * (t % 16u);
    let c = col0 + 4u * bq;
    // The next group's values loaded into registers while this one is
    // multiplied from workgroup memory (one tile there, not two).
    var next = qa_loads(0u, row0, aq, ak, c, groups, bj);
    for (var g = 0u; g < groups; g++) {
        qa_a[ak * 32u + aq] = next.a0;
        qa_a[ak * 32u + aq + 8u] = next.a1;
        qa_a[ak * 32u + aq + 16u] = next.a2;
        qa_a[ak * 32u + aq + 24u] = next.a3;
        qa_b[bj * 16u + bq] = next.b0;
        qa_b[(bj + 1u) * 16u + bq] = next.b1;
        workgroupBarrier();
        if (g + 1u < groups) {
            next = qa_loads(g + 1u, row0, aq, ak, c, groups, bj);
        }
        for (var kk = 0u; kk < 32u; kk++) {
            let a = qa_a[kk * 32u + 2u * lid.y];
            let a2 = qa_a[kk * 32u + 2u * lid.y + 1u];
            let b = qa_b[kk * 16u + lid.x];
            r0 = fma(vec4<f32>(a.x), b, r0);
            r1 = fma(vec4<f32>(a.y), b, r1);
            r2 = fma(vec4<f32>(a.z), b, r2);
            r3 = fma(vec4<f32>(a.w), b, r3);
            r4 = fma(vec4<f32>(a2.x), b, r4);
            r5 = fma(vec4<f32>(a2.y), b, r5);
            r6 = fma(vec4<f32>(a2.z), b, r6);
            r7 = fma(vec4<f32>(a2.w), b, r7);
        }
        workgroupBarrier();
    }
    let rr = row0 + lid.y * 8u;
    let cc = col0 + lid.x * 4u;
    qa_store(rr, cc, r0);
    qa_store(rr + 1u, cc, r1);
    qa_store(rr + 2u, cc, r2);
    qa_store(rr + 3u, cc, r3);
    qa_store(rr + 4u, cc, r4);
    qa_store(rr + 5u, cc, r5);
    qa_store(rr + 6u, cc, r6);
    qa_store(rr + 7u, cc, r7);
}

fn qa_store(r: u32, c0: u32, v: vec4<f32>) {
    if (r >= p.f0) {
        return;
    }
    for (var j = 0u; j < 4u; j++) {
        let c = c0 + j;
        if (c < p.f1) {
            var x = v[j];
            if ((p.f9 & 1u) != 0u) {
                x += aux[p.f8 + c];
            }
            if ((p.f9 & 4u) != 0u) {
                x = gelu(x);
            }
            let at = p.f6 + r * p.f7 + c;
            if ((p.f9 & 2u) != 0u) {
                dst[at] += x;
            } else {
                dst[at] = x;
            }
        }
    }
}

// ---- The first convolution (one input channel), then GELU:
// dst[((img·bo + b)·so + s)·480 + c] = gelu(aux[b_off + c] + Σ taps
// aux[w_off + c·9 + 3kb + ks] · src[(img·128 + 2b + kb - 1)·width + 2s + ks - 1])
// (zero outside), img counted from f5. f0 images, f1 width (steps in), f2
// so (steps out), f3 w_off, f4 b_off, f5 first image; 64 bands out.
@compute @workgroup_size(256)
fn qa_conv1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let so = p.f2;
    if (i >= p.f0 * 64u * so * 480u) {
        return;
    }
    let c = i % 480u;
    let rest = i / 480u;
    let s = rest % so;
    let b = (rest / so) % 64u;
    let img = rest / (so * 64u);
    var sum = aux[p.f4 + c];
    for (var kb = 0u; kb < 3u; kb++) {
        let ib = i32(2u * b + kb) - 1;
        if (ib < 0 || ib >= 128) {
            continue;
        }
        for (var ks = 0u; ks < 3u; ks++) {
            let is = i32(2u * s + ks) - 1;
            if (is < 0 || u32(is) >= p.f1) {
                continue;
            }
            sum += aux[p.f3 + c * 9u + 3u * kb + ks] * src[((p.f5 + img) * 128u + u32(ib)) * p.f1 + u32(is)];
        }
    }
    dst[i] = gelu(sum);
}

// ---- The windows of a 3×3 convolution with stride 2 over 480 channels
// side by side: dst[r·4320 + (3kb + ks)·480 + c] = src[((img·bands + ib)·steps
// + is)·480 + c] (zero outside), r = (img·bo + b)·so + s, ib = 2b + kb - 1,
// is = 2s + ks - 1. f0 images, f1 bands, f2 steps, f3 bo, f4 so.
@compute @workgroup_size(256)
fn qa_patches(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let bo = p.f3;
    let so = p.f4;
    if (i >= p.f0 * bo * so * 4320u) {
        return;
    }
    let c = i % 480u;
    let tap = (i / 480u) % 9u;
    let r = i / 4320u;
    let s = r % so;
    let b = (r / so) % bo;
    let img = r / (so * bo);
    let ib = i32(2u * b + tap / 3u) - 1;
    let is = i32(2u * s + tap % 3u) - 1;
    var v = 0.0;
    if (ib >= 0 && u32(ib) < p.f1 && is >= 0 && u32(is) < p.f2) {
        v = src[((img * p.f1 + u32(ib)) * p.f2 + u32(is)) * 480u + c];
    }
    dst[i] = v;
}

// ---- Each step's 16 bands side by side: dst[(img·so + s)·7680 + b·480 +
// c] = src[((img·16 + b)·so + s)·480 + c]. f0 images, f1 so.
@compute @workgroup_size(256)
fn qa_flatten(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let so = p.f1;
    if (i >= p.f0 * so * 7680u) {
        return;
    }
    let c = i % 480u;
    let b = (i / 480u) % 16u;
    let row = i / 7680u;
    let s = row % so;
    let img = row / so;
    dst[i] = src[((img * 16u + b) * so + s) * 480u + c];
}

// ---- The pieces' steps into the encoder's rows, with their positions:
// image first + img (of a sequence's `per`, its steps `len` rows apart
// from the next's), step s to row (seq·len + j·so + s), as
// dst[row·896 + d] = src[(img·so + s)·896 + d] + aux[pos_off + s·896 + d],
// for the steps kept (all but each sequence's last image's past `keep`).
// f0 images, f1 so, f2 keep, f3 first, f4 pos_off, f5 per, f6 len.
@compute @workgroup_size(256)
fn qa_place(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let so = p.f1;
    if (i >= p.f0 * so * 896u) {
        return;
    }
    let d = i % 896u;
    let row = i / 896u;
    let s = row % so;
    let img = p.f3 + row / so;
    let seq = img / p.f5;
    let j = img % p.f5;
    if (j + 1u == p.f5 && s >= p.f2) {
        return;
    }
    dst[(seq * p.f6 + j * so + s) * 896u + d] = src[i] + aux[p.f4 + s * 896u + d];
}

// ---- LayerNorm, src to dst (eps 1e-5), a workgroup a row (wid.x +
// wid.y·nw.x): the row read into workgroup memory together, its sums
// taken in order by one invocation. f0 rows, f1 dim (at most 1024), f2
// w_off, f3 b_off.
var<workgroup> qa_row: array<f32, 1024>;
var<workgroup> qa_stats: vec2<f32>;

@compute @workgroup_size(64)
fn qa_layernorm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let r = wid.x + wid.y * nw.x;
    let dim = p.f1;
    let x = r * dim;
    let valid = r < p.f0;
    if (valid) {
        for (var d = lid.x; d < dim; d += 64u) {
            qa_row[d] = src[x + d];
        }
    }
    workgroupBarrier();
    if (valid && lid.x == 0u) {
        var sum = 0.0;
        for (var d = 0u; d < dim; d++) {
            sum += qa_row[d];
        }
        let mean = sum / f32(dim);
        var squares = 0.0;
        for (var d = 0u; d < dim; d++) {
            let v = qa_row[d] - mean;
            squares += v * v;
        }
        qa_stats = vec2<f32>(mean, inverseSqrt(squares / f32(dim) + 1e-5));
    }
    workgroupBarrier();
    if (valid) {
        let mean = qa_stats.x;
        let k = qa_stats.y;
        for (var d = lid.x; d < dim; d += 64u) {
            dst[x + d] = (qa_row[d] - mean) * k * aux[p.f2 + d] + aux[p.f3 + d];
        }
    }
}

// ---- The encoder's attention, 14 heads of 64, within windows: src
// queries, keys and values side by side [step][2688]; dst [step][896]. A
// query attends to the keys of its window (steps from ⌊q / window⌋·window,
// window long), within its sequence (wid.z, len steps each). f0 len, f1
// window. Workgroups (queries / 64, 14, sequences); tiles of
// 32 keys over the workgroup's windows, an online softmax.
var<workgroup> qa_keys: array<vec4<f32>, 512>;
var<workgroup> qa_vals: array<vec4<f32>, 512>;

fn qa_src4(i: u32) -> vec4<f32> {
    return vec4<f32>(src[i], src[i + 1u], src[i + 2u], src[i + 3u]);
}

fn qa_aux4(i: u32) -> vec4<f32> {
    return vec4<f32>(aux[i], aux[i + 1u], aux[i + 2u], aux[i + 3u]);
}

fn qa_store4(i: u32, v: vec4<f32>) {
    dst[i] = v.x;
    dst[i + 1u] = v.y;
    dst[i + 2u] = v.z;
    dst[i + 3u] = v.w;
}

@compute @workgroup_size(64)
fn qa_enc_attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let window = p.f1;
    let h = wid.y;
    let o = wid.z * len;
    let first = wid.x * 64u;
    let qp = first + lid.x;
    let valid = qp < len;
    let w0 = (qp / window) * window;
    let w1 = min(w0 + window, len);
    var q: array<vec4<f32>, 16>;
    var acc: array<vec4<f32>, 16>;
    for (var d = 0u; d < 16u; d++) {
        acc[d] = vec4<f32>(0.0);
        q[d] = vec4<f32>(0.0);
    }
    if (valid) {
        let at = (o + qp) * 2688u + h * 64u;
        for (var d = 0u; d < 16u; d++) {
            q[d] = qa_src4(at + 4u * d) * 0.125;
        }
    }
    var best = -1e30;
    var total = 0.0;
    // The keys any query here may see.
    let last = min(first + 63u, len - 1u);
    let k_from = (first / window) * window;
    let k_to = min((last / window) * window + window, len);
    for (var k0 = k_from; k0 < k_to; k0 += 32u) {
        for (var r = 0u; r < 8u; r++) {
            let slot = lid.x * 8u + r;
            let kp = k0 + slot / 16u;
            if (kp < len) {
                let at = (o + kp) * 2688u + 896u + h * 64u + 4u * (slot % 16u);
                qa_keys[slot] = qa_src4(at);
                qa_vals[slot] = qa_src4(at + 896u);
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var j = 0u; j < 32u; j++) {
                let kp = k0 + j;
                if (kp < w0 || kp >= w1) {
                    continue;
                }
                let kb = j * 16u;
                var s = 0.0;
                for (var d = 0u; d < 16u; d++) {
                    s += dot(q[d], qa_keys[kb + d]);
                }
                if (s > best) {
                    let fade = exp(best - s);
                    total = total * fade + 1.0;
                    for (var d = 0u; d < 16u; d++) {
                        acc[d] = acc[d] * fade + qa_vals[kb + d];
                    }
                    best = s;
                } else {
                    let e = exp(s - best);
                    total += e;
                    for (var d = 0u; d < 16u; d++) {
                        acc[d] = fma(vec4<f32>(e), qa_vals[kb + d], acc[d]);
                    }
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let at = (o + qp) * 896u + h * 64u;
        for (var d = 0u; d < 16u; d++) {
            qa_store4(at + 4u * d, acc[d] / total);
        }
    }
}

// ---- RMSNorm (eps 1e-6), src to dst, a workgroup a row: dst[r·dim + d]
// = src[a_off + r·lda + d] · k · aux[w_off + d]. f0 rows, f1 dim, f2 w_off,
// f3 a_off, f4 lda. Workgroups: one a row (wid.x + wid.y·nw.x).
var<workgroup> qa_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn qa_rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let r = wid.x + wid.y * nw.x;
    let dim = p.f1;
    let x = p.f3 + r * p.f4;
    var sum = 0.0;
    if (r < p.f0) {
        for (var d = lid.x; d < dim; d += 256u) {
            let v = src[x + d];
            sum += v * v;
        }
    }
    qa_sum[lid.x] = sum;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half /= 2u) {
        if (lid.x < half) {
            qa_sum[lid.x] += qa_sum[lid.x + half];
        }
        workgroupBarrier();
    }
    if (r >= p.f0) {
        return;
    }
    let k = inverseSqrt(qa_sum[0] / f32(dim) + 1e-6);
    for (var d = lid.x; d < dim; d += 256u) {
        dst[r * dim + d] = src[x + d] * k * aux[p.f2 + d];
    }
}
// ---- In place on heads of 128 in dst: each head RMS-normed (eps 1e-6),
// scaled by aux[w_off + d], then turned by its row's rotary angles (halves
// paired: values i and i + 64 by angle i, the table in aux at rope_off,
// [position][64][cos, sin]). An invocation a head of a row. f0 rows, f1
// width (values a row), f2 col_off, f3 heads, f4 w_off, f5 position of
// row 0, f6 rope_off, f7 rows a sequence (its positions from f5 again; 0:
// one sequence).
@compute @workgroup_size(256)
fn qa_head_norm_rope(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * p.f3) {
        return;
    }
    let r = i / p.f3;
    let h = i % p.f3;
    let at = r * p.f1 + p.f2 + h * 128u;
    var sum = 0.0;
    for (var d = 0u; d < 128u; d++) {
        let v = dst[at + d];
        sum += v * v;
    }
    let k = inverseSqrt(sum / 128.0 + 1e-6);
    let table = p.f6 + (p.f5 + select(r, r % p.f7, p.f7 != 0u)) * 128u;
    for (var d = 0u; d < 64u; d++) {
        let a = dst[at + d] * k * aux[p.f4 + d];
        let b = dst[at + d + 64u] * k * aux[p.f4 + d + 64u];
        let c = aux[table + 2u * d];
        let s = aux[table + 2u * d + 1u];
        dst[at + d] = a * c - b * s;
        dst[at + d + 64u] = b * c + a * s;
    }
}

// ---- dst[d_off + r·ldd + j] = src[s_off + r·lds + j] for j < width.
// f0 rows, f1 width, f2 s_off, f3 lds, f4 d_off, f5 ldd; f6 rows a block
// (0: one block), f7 blocks apart in dst (row r at d_off + ⌊r / f6⌋·f7 +
// (r % f6)·ldd).
@compute @workgroup_size(256)
fn qa_copy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * p.f1) {
        return;
    }
    let r = i / p.f1;
    let j = i % p.f1;
    var at = p.f4 + r * p.f5 + j;
    if (p.f6 != 0u) {
        at = p.f4 + (r / p.f6) * p.f7 + (r % p.f6) * p.f5 + j;
    }
    dst[at] = src[p.f2 + r * p.f3 + j];
}

// ---- The decoder's attention, causal, 16 query heads on 8 key/value
// heads of 128: src queries [row][4096] (columns 0..2048); aux the layer's
// cache from cache_off, [position][2048] (keys 0..1024, values
// 1024..2048); dst [row][2048]. Row r is position start + r and sees
// positions 0..=start + r. f0 rows, f1 start, f2 cache_off, f3 the next
// sequence's cache after (sequence wid.z: rows from wid.z·rows). Workgroups
// (rows / 64, 16, sequences); tiles of 8 positions, an online softmax.
@compute @workgroup_size(64)
fn qa_dec_attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let rows = p.f0;
    let start = p.f1;
    let h = wid.y;
    let kv = h / 2u;
    let o = wid.z * rows;
    let cache = p.f2 + wid.z * p.f3;
    let first = wid.x * 64u;
    let r = first + lid.x;
    let valid = r < rows;
    let pos = start + r;
    var q: array<vec4<f32>, 32>;
    var acc: array<vec4<f32>, 32>;
    for (var d = 0u; d < 32u; d++) {
        acc[d] = vec4<f32>(0.0);
        q[d] = vec4<f32>(0.0);
    }
    if (valid) {
        let at = (o + r) * 4096u + h * 128u;
        let scale = inverseSqrt(128.0);
        for (var d = 0u; d < 32u; d++) {
            q[d] = qa_src4(at + 4u * d) * scale;
        }
    }
    var best = -1e30;
    var total = 0.0;
    let seen = start + min(first + 64u, rows);
    for (var k0 = 0u; k0 < seen; k0 += 8u) {
        // 64 invocations load 8 positions of 32 vectors: 4 each.
        for (var x = 0u; x < 4u; x++) {
            let slot = lid.x * 4u + x;
            let kp = k0 + slot / 32u;
            if (kp < seen) {
                let at = cache + kp * 2048u + kv * 128u + 4u * (slot % 32u);
                qa_keys[slot] = qa_aux4(at);
                qa_vals[slot] = qa_aux4(at + 1024u);
            }
        }
        workgroupBarrier();
        if (valid) {
            for (var j = 0u; j < 8u; j++) {
                let kp = k0 + j;
                if (kp > pos) {
                    break;
                }
                let kb = j * 32u;
                var s = 0.0;
                for (var d = 0u; d < 32u; d++) {
                    s += dot(q[d], qa_keys[kb + d]);
                }
                if (s > best) {
                    let fade = exp(best - s);
                    total = total * fade + 1.0;
                    for (var d = 0u; d < 32u; d++) {
                        acc[d] = acc[d] * fade + qa_vals[kb + d];
                    }
                    best = s;
                } else {
                    let e = exp(s - best);
                    total += e;
                    for (var d = 0u; d < 32u; d++) {
                        acc[d] = fma(vec4<f32>(e), qa_vals[kb + d], acc[d]);
                    }
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let at = (o + r) * 2048u + h * 128u;
        for (var d = 0u; d < 32u; d++) {
            qa_store4(at + 4u * d, acc[d] / total);
        }
    }
}

// ---- SwiGLU: dst[r·3072 + j] = silu(src[r·6144 + j]) · src[r·6144 +
// 3072 + j]. f0 rows.
@compute @workgroup_size(256)
fn qa_swiglu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * 3072u) {
        return;
    }
    let r = i / 3072u;
    let j = i % 3072u;
    let g = src[r * 6144u + j];
    dst[i] = g / (1.0 + exp(-g)) * src[r * 6144u + 3072u + j];
}

// ---- Token embeddings: dst[(row0 + r)·1024 + d] = the 4-bit table's row
// u32(src[id_off + r]) (in sequences of f5 rows, f6 ids apart: row r's id
// at id_off + ⌊r / f5⌋·f6 + r % f5; f5 0: one). f0 rows, f1 id_off, f2
// w_off, f3 row0, f4 its bits
// (4, 8 or 16).
@compute @workgroup_size(256)
fn qa_embed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * 1024u) {
        return;
    }
    let r = i / 1024u;
    let d = i % 1024u;
    let id = u32(src[p.f1 + select(r, (r / p.f5) * p.f6 + r % p.f5, p.f5 != 0u)]);
    let j = d % 32u;
    if (p.f4 == 16u) {
        let v = unpack2x16float(qw[p.f2 + (id * 32u + d / 32u) * 16u + j / 2u]);
        dst[(p.f3 + r) * 1024u + d] = select(v.x, v.y, (j & 1u) == 1u);
    } else if (p.f4 == 8u) {
        let base = p.f2 + (id * 32u + d / 32u) * 9u;
        let so = unpack2x16float(qw[base + 8u]);
        let q = (qw[base + j / 4u] >> (8u * (j % 4u))) & 255u;
        dst[(p.f3 + r) * 1024u + d] = so.y + f32(q) * so.x;
    } else {
        let base = p.f2 + (id * 32u + d / 32u) * 5u;
        let so = unpack2x16float(qw[base + 4u]);
        let nib = (qw[base + j / 8u] >> (4u * (j % 8u))) & 15u;
        dst[(p.f3 + r) * 1024u + d] = so.y + f32(nib) * so.x;
    }
}

// ---- Calibration: each input column's sum of squares added up, dst[s_off +
// c] += Σr src[a_off + r·lda + c]², for the columns c < k of m rows (a matrix
// multiply's input). f0 m, f1 k, f2 a_off, f3 lda, f4 s_off.
@compute @workgroup_size(256)
fn qa_colsq(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = linear_id(gid, groups);
    if (c >= p.f1) {
        return;
    }
    var sum = 0.0;
    for (var r = 0u; r < p.f0; r++) {
        let v = src[p.f2 + r * p.f3 + c];
        sum += v * v;
    }
    dst[p.f4 + c] += sum;
}
