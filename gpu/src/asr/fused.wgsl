// The fused kernels: a few rows through a 4-bit matrix (`qa_rows`, also
// the encoder's when a piece has one row) and the rest of the decoder's
// steps for a batch of sequences, a row each (compiled after kernels.wgsl
// and asr.wgsl: their bindings, `qw` and helpers). A row's
// values do not depend on how many rows go with it: each is summed in one
// fixed order, the order the decoder's steps have always had (a row at a
// time, a kernel each):
// - a matrix's output column: each of 32 lanes adds up its groups of 32
//   values (lane, lane + 32, ...), each group as its products' sum times
//   the scale plus its inputs' sum times the offset (at 4 and 8 bits; the
//   products alone at 16), then the lanes in a tree (16 apart, then 8, 4,
//   2, 1);
// - the attention: in chunks of 128 positions, each chunk's largest score,
//   its sum of exp(score - largest) and its values so weighted, then the
//   chunks put together by their largest.
//
// A second buffer to write, `dst2`, for the kernels that write two.

@group(0) @binding(5) var<storage, read_write> dst2: array<f32>;

// A group's 32 inputs, four at a time.
struct X8 {
    x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
    x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>,
}

fn st_lo(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(w & 15u), f32((w >> 4u) & 15u), f32((w >> 8u) & 15u), f32((w >> 12u) & 15u));
}

fn st_hi(w: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> 16u) & 15u), f32((w >> 20u) & 15u), f32((w >> 24u) & 15u), f32(w >> 28u));
}

fn st_bytes(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u), f32(w >> 24u));
}

fn st_halves(a: u32, b: u32) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(a), unpack2x16float(b));
}

// A group's inputs added up as a column adds them beside the products (at
// 4 and 8 bits; none at 16).
fn st_total(fmt: u32, x: X8) -> f32 {
    var total = 0.0;
    if (fmt == 8u) {
        total += x.x0.x + x.x0.y + x.x0.z + x.x0.w;
        total += x.x1.x + x.x1.y + x.x1.z + x.x1.w;
        total += x.x2.x + x.x2.y + x.x2.z + x.x2.w;
        total += x.x3.x + x.x3.y + x.x3.z + x.x3.w;
        total += x.x4.x + x.x4.y + x.x4.z + x.x4.w;
        total += x.x5.x + x.x5.y + x.x5.z + x.x5.w;
        total += x.x6.x + x.x6.y + x.x6.z + x.x6.w;
        total += x.x7.x + x.x7.y + x.x7.z + x.x7.w;
    } else if (fmt == 4u) {
        total += x.x0.x + x.x0.y + x.x0.z + x.x0.w + x.x1.x + x.x1.y + x.x1.z + x.x1.w;
        total += x.x2.x + x.x2.y + x.x2.z + x.x2.w + x.x3.x + x.x3.y + x.x3.z + x.x3.w;
        total += x.x4.x + x.x4.y + x.x4.z + x.x4.w + x.x5.x + x.x5.y + x.x5.z + x.x5.w;
        total += x.x6.x + x.x6.y + x.x6.z + x.x6.w + x.x7.x + x.x7.y + x.x7.z + x.x7.w;
    }
    return total;
}

// Group g of column c of the matrix at w_off (f5; `groups` a column) in
// the format `fmt` (8: 8 bits, 16: halves, else 4 bits) for qa_rows's
// rows, added to `s` (a row each) as a column adds a group: a word at a
// time, each row's inputs from workgroup memory (st_staged's places), its
// total from st_tot.
fn st_group(fmt: u32, s: vec4<f32>, rows: u32, c: u32, groups: u32, g: u32, lane: u32) -> vec4<f32> {
    let x = lane * 8u;
    var out = s;
    if (fmt == 16u) {
        let b = p.f5 + (c * groups + g) * 16u;
        for (var q = 0u; q < 8u; q++) {
            let h = st_halves(qw[b + 2u * q], qw[b + 2u * q + 1u]);
            let a = x + ((q + lane) & 7u);
            out.x += dot(st_x[a], h);
            if (rows > 1u) {
                out.y += dot(st_x[256u + a], h);
            }
            if (rows > 2u) {
                out.z += dot(st_x[512u + a], h);
            }
            if (rows > 3u) {
                out.w += dot(st_x[768u + a], h);
            }
        }
        return out;
    }
    var d = vec4<f32>(0.0);
    var so = vec2<f32>(0.0);
    if (fmt == 8u) {
        let b = p.f5 + (c * groups + g) * 9u;
        so = unpack2x16float(qw[b + 8u]);
        for (var q = 0u; q < 8u; q++) {
            let v = st_bytes(qw[b + q]);
            let a = x + ((q + lane) & 7u);
            d.x += dot(st_x[a], v);
            if (rows > 1u) {
                d.y += dot(st_x[256u + a], v);
            }
            if (rows > 2u) {
                d.z += dot(st_x[512u + a], v);
            }
            if (rows > 3u) {
                d.w += dot(st_x[768u + a], v);
            }
        }
    } else {
        let b = p.f5 + (c * groups + g) * 5u;
        so = unpack2x16float(qw[b + 4u]);
        for (var q = 0u; q < 4u; q++) {
            let word = qw[b + q];
            let n0 = st_lo(word);
            let n1 = st_hi(word);
            let a0 = x + ((2u * q + lane) & 7u);
            let a1 = x + ((2u * q + 1u + lane) & 7u);
            d.x += dot(st_x[a0], n0) + dot(st_x[a1], n1);
            if (rows > 1u) {
                d.y += dot(st_x[256u + a0], n0) + dot(st_x[256u + a1], n1);
            }
            if (rows > 2u) {
                d.z += dot(st_x[512u + a0], n0) + dot(st_x[512u + a1], n1);
            }
            if (rows > 3u) {
                d.w += dot(st_x[768u + a0], n0) + dot(st_x[768u + a1], n1);
            }
        }
    }
    out.x = out.x + (d.x * so.x + st_tot[lane] * so.y);
    if (rows > 1u) {
        out.y = out.y + (d.y * so.x + st_tot[32u + lane] * so.y);
    }
    if (rows > 2u) {
        out.z = out.z + (d.z * so.x + st_tot[64u + lane] * so.y);
    }
    if (rows > 3u) {
        out.w = out.w + (d.w * so.x + st_tot[96u + lane] * so.y);
    }
    return out;
}

// ---- Up to 4 rows (a block, wid.x) through a matrix, eight output
// columns at a time, `rounds` (flags >> 16, at least 1) of them a
// workgroup (wid.y), each column summed in its order: dst[c_off + r·ldc + c]
// (+)= Σk x[r][k] · W[c][k] (+ bias), GELU after if asked. Parameters as
// qa_matmul's, and f10 norm_off, f11 as the flags say. The rows go through
// workgroup memory 1024 values at a time, a lane's group (eight vectors)
// at 8·lane, turned by lane so the lanes read apart; each group's inputs
// added up once for the workgroup. More than one round only where k is at
// most 1024 (the rows in workgroup memory once). Flags: as qa_matmul's, and
// - 32: each row RMS-normed first (as qa_rmsnorm, by aux[norm_off + k]);
// - 64: the columns are gates and ups (the first n / 2 and the rest),
//   each pair out as silu(gate) · up (as qa_swiglu) at column j of n / 2,
//   four a round;
// - 128: each row's largest value of the workgroup's and its column (the
//   first of equals) to dst2[row·f11 + 2·wid.y..], for qa_argmax_embed;
// - 256: the rows are qa_dec_attend's chunks (src from a_off, lda apart:
//   [16 heads][8 chunks][130]), f11 of them, each head's put together (as
//   the attention's).
var<workgroup> st_x: array<vec4<f32>, 1024>;
var<workgroup> st_tot: array<f32, 128>;
var<workgroup> st_r: array<vec4<f32>, 328>;
var<workgroup> st_out: array<f32, 256>;
var<workgroup> st_best: array<f32, 32>;
var<workgroup> st_at: array<u32, 32>;

// The last three levels of a tree (4 apart, then 2, 1) over the eight
// values of st_r from b, `apart` apart.
fn st_eight(b: u32, apart: u32) -> vec4<f32> {
    let e0 = st_r[b];
    let e1 = st_r[b + apart];
    let e2 = st_r[b + 2u * apart];
    let e3 = st_r[b + 3u * apart];
    let e4 = st_r[b + 4u * apart];
    let e5 = st_r[b + 5u * apart];
    let e6 = st_r[b + 6u * apart];
    let e7 = st_r[b + 7u * apart];
    return ((e0 + e4) + (e2 + e6)) + ((e1 + e5) + (e3 + e7));
}

// Row r's group of the lane, from workgroup memory.
fn st_staged(r: u32, lane: u32) -> X8 {
    let b = r * 256u + lane * 8u;
    return X8(st_x[b + (lane & 7u)], st_x[b + ((1u + lane) & 7u)],
              st_x[b + ((2u + lane) & 7u)], st_x[b + ((3u + lane) & 7u)],
              st_x[b + ((4u + lane) & 7u)], st_x[b + ((5u + lane) & 7u)],
              st_x[b + ((6u + lane) & 7u)], st_x[b + ((7u + lane) & 7u)]);
}

// Value k of a row of qa_dec_attend's chunks from a, the chunks put together.
fn st_joined(a: u32, k: u32) -> f32 {
    let base = a + (k / 128u) * 8u * 130u;
    let d = k % 128u;
    var top = -1e30;
    for (var c = 0u; c < p.f11; c++) {
        top = max(top, src[base + c * 130u]);
    }
    var total = 0.0;
    var acc = 0.0;
    for (var c = 0u; c < p.f11; c++) {
        let at = base + c * 130u;
        let kc = exp(src[at] - top);
        total += kc * src[at + 1u];
        acc += kc * src[at + 2u + d];
    }
    return acc / total;
}

// Four values of a row from a (k, k + 1, ...), put together if asked.
fn st_in4(a: u32, k: u32) -> vec4<f32> {
    if ((p.f9 & 256u) != 0u) {
        return vec4<f32>(st_joined(a, k), st_joined(a, k + 1u), st_joined(a, k + 2u), st_joined(a, k + 3u));
    }
    return qa_src4(a + k);
}

// The output column of the invocations of `slot` in round `round`.
fn st_column(round: u32, slot: u32, rounds: u32, wy: u32) -> u32 {
    if ((p.f9 & 64u) != 0u) {
        return (wy * rounds + round) * 4u + slot % 4u + select(0u, p.f1 / 2u, slot >= 4u);
    }
    return (wy * rounds + round) * 8u + slot;
}

@compute @workgroup_size(256)
fn qa_rows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let r0 = wid.x * 4u;
    let rows = min(p.f0 - r0, 4u);
    let fmt = select(select(4u, 8u, (p.f9 & 8u) != 0u), 16u, (p.f9 & 16u) != 0u);
    let norm = (p.f9 & 32u) != 0u;
    let swiglu = (p.f9 & 64u) != 0u;
    let rounds = max(p.f9 >> 16u, 1u);
    var ks = vec4<f32>(1.0);
    if (norm) {
        var sums = vec4<f32>(0.0);
        for (var r = 0u; r < rows; r++) {
            let x = p.f3 + (r0 + r) * p.f4;
            var sum = 0.0;
            for (var d = t; d < p.f2; d += 256u) {
                let v = src[x + d];
                sum += v * v;
            }
            sums[r] = sum;
        }
        // qa_rmsnorm's tree (128 apart, then 64, ..., 1), a few levels at
        // a time.
        st_r[t] = sums;
        workgroupBarrier();
        if (t < 64u) {
            st_r[256u + t] = (st_r[t] + st_r[t + 128u]) + (st_r[t + 64u] + st_r[t + 192u]);
        }
        workgroupBarrier();
        if (t < 8u) {
            st_r[320u + t] = st_eight(256u + t, 8u);
        }
        workgroupBarrier();
        ks = inverseSqrt(st_eight(320u, 1u) / f32(p.f2) + vec4<f32>(1e-6));
        workgroupBarrier();
    }
    let lane = t % 32u;
    let slot = t / 32u;
    let groups = p.f2 / 32u;
    // The first round's sums, through the chunks before the last.
    var s = vec4<f32>(0.0);
    var j = 0u;
    loop {
        // The rows' next 1024 values (fewer at the end), four at a time.
        let n = min(32u, groups - j) * 8u;
        for (var e = t; e < 1024u; e += 256u) {
            let r = e / 256u;
            let q = e % 256u;
            if (r < rows && q < n) {
                let k = j * 32u + q * 4u;
                var v = st_in4(p.f3 + (r0 + r) * p.f4, k);
                if (norm) {
                    let kr = ks[r];
                    let w = p.f10 + k;
                    v = vec4<f32>(v.x * kr * aux[w], v.y * kr * aux[w + 1u], v.z * kr * aux[w + 2u], v.w * kr * aux[w + 3u]);
                }
                let g = q / 8u;
                st_x[r * 256u + g * 8u + (((q % 8u) + g) & 7u)] = v;
            }
        }
        workgroupBarrier();
        if (t < 128u && t / 32u < rows && j + lane < groups) {
            st_tot[t] = st_total(fmt, st_staged(t / 32u, lane));
        }
        workgroupBarrier();
        if (j + 32u >= groups) {
            break;
        }
        let c = st_column(0u, slot, rounds, wid.y);
        if (c < p.f1) {
            s = st_group(fmt, s, rows, c, groups, j + lane, lane);
        }
        workgroupBarrier();
        j += 32u;
    }
    // The last chunk's columns, round by round.
    for (var round = 0u; round < rounds; round++) {
        let c = st_column(round, slot, rounds, wid.y);
        var sc = select(vec4<f32>(0.0), s, round == 0u);
        if (c < p.f1 && j + lane < groups) {
            sc = st_group(fmt, sc, rows, c, groups, j + lane, lane);
        }
        // Each column's lanes in their tree (16 apart, then 8, ...,
        // 1), the four rows at once, a few levels at a time.
        st_r[slot * 33u + lane] = sc;
        workgroupBarrier();
        if (lane < 8u) {
            let b = slot * 33u + lane;
            st_r[264u + slot * 8u + lane] = (st_r[b] + st_r[b + 16u]) + (st_r[b + 8u] + st_r[b + 24u]);
        }
        workgroupBarrier();
        if (swiglu) {
            if (t < 16u) {
                let jj = t / 4u;
                let r = t % 4u;
                let col = (wid.y * rounds + round) * 4u + jj;
                if (r < rows && col < p.f1 / 2u) {
                    let g = st_eight(264u + jj * 8u, 1u)[r];
                    dst[p.f6 + (r0 + r) * p.f7 + col] = g / (1.0 + exp(-g)) * st_eight(264u + (4u + jj) * 8u, 1u)[r];
                }
            }
        } else if (t < 32u) {
            let r = t % 4u;
            let col = (wid.y * rounds + round) * 8u + t / 4u;
            if (r < rows && col < p.f1) {
                var x = st_eight(264u + (t / 4u) * 8u, 1u)[r];
                if ((p.f9 & 1u) != 0u) {
                    x += aux[p.f8 + col];
                }
                if ((p.f9 & 4u) != 0u) {
                    x = gelu(x);
                }
                let o = p.f6 + (r0 + r) * p.f7 + col;
                if ((p.f9 & 2u) != 0u) {
                    dst[o] += x;
                } else {
                    dst[o] = x;
                }
                // (A matrix whose largest is asked for is written as it is.)
                st_out[round * 32u + t] = x;
            }
        }
    }
    // The largest (the first of equals): each round's, then the rounds'.
    if ((p.f9 & 128u) != 0u) {
        workgroupBarrier();
        if (t < 32u) {
            let r = t % 4u;
            let round = t / 4u;
            var best = -3.0e38;
            var at = 0u;
            for (var sl = 0u; sl < 8u; sl++) {
                let col = (wid.y * rounds + round) * 8u + sl;
                if (round < rounds && col < p.f1) {
                    let v = st_out[round * 32u + sl * 4u + r];
                    if (v > best) {
                        best = v;
                        at = col;
                    }
                }
            }
            st_best[t] = best;
            st_at[t] = at;
        }
        workgroupBarrier();
        if (t < rows) {
            var best = -3.0e38;
            var at = 0u;
            for (var round = 0u; round < rounds; round++) {
                let v = st_best[round * 4u + t];
                if (v > best) {
                    best = v;
                    at = st_at[round * 4u + t];
                }
            }
            dst2[(r0 + t) * p.f11 + 2u * wid.y] = best;
            dst2[(r0 + t) * p.f11 + 2u * wid.y + 1u] = f32(at);
        }
    }
}

// ---- One step's attention for one row (wid.x), one key/value head
// (wid.y, its two query heads with it) and one chunk of 128 positions
// (wid.z) of 0..=pos: the row's queries and key normed and turned (as
// qa_head_norm_rope), the key and value into the cache (by the chunk with
// pos), and the chunk's part of each query head's attention (qa_rows puts
// the chunks together). src the rows' queries, keys and values
// [row][4096]; dst the caches, a row's `seq` apart,
// [layer][position][2048] (keys, then values), and the chunks at part_off,
// [row][head][8][130] (the chunk's largest score, its sum of exp(score -
// largest) and its values so weighted). f0 pos, f1 cache_off (the
// layer's), f2 seq, f6 q_norm_off, f7 k_norm_off, f8 rope_off, f9
// part_off.
var<workgroup> sd_v: array<f32, 512>;
var<workgroup> sd_red: array<f32, 512>;
var<workgroup> sd_k: array<f32, 4>;

fn sd_v4(i: u32) -> vec4<f32> {
    return vec4<f32>(sd_v[i], sd_v[i + 1u], sd_v[i + 2u], sd_v[i + 3u]);
}

fn sd_dst4(i: u32) -> vec4<f32> {
    return vec4<f32>(dst[i], dst[i + 1u], dst[i + 2u], dst[i + 3u]);
}

@compute @workgroup_size(256)
fn qa_dec_attend(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let row = wid.x;
    let kv = wid.y;
    let chunk = wid.z;
    let pos = p.f0;
    let qo = row * 4096u;
    sd_v[t] = src[qo + kv * 256u + t];
    if (t < 128u) {
        sd_v[256u + t] = src[qo + 2048u + kv * 128u + t];
    } else {
        sd_v[256u + t] = src[qo + 3072u + kv * 128u + t - 128u];
    }
    workgroupBarrier();
    // The two query heads and the key normed (one invocation each), then
    // turned.
    if (t < 3u) {
        var squares = 0.0;
        for (var d = 0u; d < 128u; d++) {
            let v = sd_v[t * 128u + d];
            squares += v * v;
        }
        sd_k[t] = inverseSqrt(squares / 128.0 + 1e-6);
    }
    workgroupBarrier();
    if (t < 192u) {
        let h = t / 64u;
        let d = t % 64u;
        let at = h * 128u;
        let hk = sd_k[h];
        let nw = select(p.f6, p.f7, h == 2u);
        let a = sd_v[at + d] * hk * aux[nw + d];
        let b = sd_v[at + d + 64u] * hk * aux[nw + d + 64u];
        let table = p.f8 + pos * 128u;
        let c = aux[table + 2u * d];
        let s = aux[table + 2u * d + 1u];
        sd_v[at + d] = a * c - b * s;
        sd_v[at + d + 64u] = b * c + a * s;
    }
    workgroupBarrier();
    let cache = p.f1 + row * p.f2 + kv * 128u;
    if (chunk == pos / 128u) {
        if (t < 128u) {
            dst[cache + pos * 2048u + t] = sd_v[256u + t];
        } else {
            dst[cache + pos * 2048u + 1024u + t - 128u] = sd_v[384u + t - 128u];
        }
    }
    // A query head a half of the workgroup: a position, then a value, an
    // invocation.
    let hb = t / 128u;
    let i = t % 128u;
    let base = hb * 128u;
    let n = pos + 1u;
    let j = chunk * 128u + i;
    var s = -1e30;
    if (j < n) {
        var dots = 0.0;
        if (j == pos) {
            for (var d = 0u; d < 128u; d += 4u) {
                dots += dot(sd_v4(base + d), sd_v4(256u + d));
            }
        } else {
            let at = cache + j * 2048u;
            for (var d = 0u; d < 128u; d += 4u) {
                dots += dot(sd_v4(base + d), sd_dst4(at + d));
            }
        }
        s = dots * inverseSqrt(128.0);
    }
    sd_red[t] = s;
    workgroupBarrier();
    for (var half = 64u; half > 0u; half /= 2u) {
        if (i < half) {
            sd_red[t] = max(sd_red[t], sd_red[t + half]);
        }
        workgroupBarrier();
    }
    let m = sd_red[base];
    workgroupBarrier();
    var e = 0.0;
    if (j < n) {
        e = exp(s - m);
    }
    sd_red[256u + t] = e;
    sd_red[t] = e;
    workgroupBarrier();
    for (var half = 64u; half > 0u; half /= 2u) {
        if (i < half) {
            sd_red[t] += sd_red[t + half];
        }
        workgroupBarrier();
    }
    var acc = 0.0;
    let count = min(128u, n - chunk * 128u);
    for (var q = 0u; q < count; q++) {
        let jj = chunk * 128u + q;
        var v = 0.0;
        if (jj == pos) {
            v = sd_v[384u + i];
        } else {
            v = dst[cache + jj * 2048u + 1024u + i];
        }
        acc += sd_red[256u + base + q] * v;
    }
    let out = p.f9 + ((row * 16u + 2u * kv + hb) * 8u + chunk) * 130u;
    if (i == 0u) {
        dst[out] = m;
        dst[out + 1u] = sd_red[base];
    }
    dst[out + 2u + i] = acc;
}

// ---- Each row's next token, the first of its largest logits (from
// qa_rows's largest of each workgroup: src [row][f0][value, column]): its
// id to dst[wid.x·f2 + f1] and its embedding (as qa_embed) to
// dst2[wid.x·1024..]. f0 the workgroups a row, f1 the token's place, f2
// places a row, f3 w_off, f4 its bits (4, 8 or 16).
var<workgroup> qa_best: array<f32, 256>;
var<workgroup> qa_at: array<u32, 256>;

@compute @workgroup_size(256)
fn qa_argmax_embed(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let off = wid.x * 2u * p.f0;
    var best = -3.0e38;
    var at = 0u;
    for (var i = lid.x; i < p.f0; i += 256u) {
        let v = src[off + 2u * i];
        if (v > best) {
            best = v;
            at = u32(src[off + 2u * i + 1u]);
        }
    }
    qa_best[lid.x] = best;
    qa_at[lid.x] = at;
    workgroupBarrier();
    for (var half = 128u; half > 0u; half /= 2u) {
        if (lid.x < half) {
            let o = lid.x + half;
            if (qa_best[o] > qa_best[lid.x] || (qa_best[o] == qa_best[lid.x] && qa_at[o] < qa_at[lid.x])) {
                qa_best[lid.x] = qa_best[o];
                qa_at[lid.x] = qa_at[o];
            }
        }
        workgroupBarrier();
    }
    let id = qa_at[0];
    if (lid.x == 0u) {
        dst[wid.x * p.f2 + p.f1] = f32(id);
    }
    for (var d = lid.x; d < 1024u; d += 256u) {
        let j = d % 32u;
        var v = 0.0;
        if (p.f4 == 16u) {
            let h = unpack2x16float(qw[p.f3 + (id * 32u + d / 32u) * 16u + j / 2u]);
            v = select(h.x, h.y, (j & 1u) == 1u);
        } else if (p.f4 == 8u) {
            let b = p.f3 + (id * 32u + d / 32u) * 9u;
            let so = unpack2x16float(qw[b + 8u]);
            let q = (qw[b + j / 4u] >> (8u * (j % 4u))) & 255u;
            v = so.y + f32(q) * so.x;
        } else {
            let b = p.f3 + (id * 32u + d / 32u) * 5u;
            let so = unpack2x16float(qw[b + 4u]);
            let nib = (qw[b + j / 8u] >> (4u * (j % 8u))) & 15u;
            v = so.y + f32(nib) * so.x;
        }
        dst2[wid.x * 1024u + d] = v;
    }
}
