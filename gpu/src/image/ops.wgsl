// The picture models' smaller kernels: norms, the heads' preparation for
// attention, the sampler's step, copies, the latents' patches, the group
// norm's statistics, a softmax and a transpose.

var<workgroup> red: array<f32, 256>;

// The sum of `v` over the workgroup of 256 (every invocation gets it).
fn sum256(li: u32, v: f32) -> f32 {
    workgroupBarrier();
    red[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    return red[0];
}

// RMSNorm of a row (its weights in `small`): n0 rows, n1 width (a
// multiple of 4), n2 src's offset, n3 its stride, n4 dst's offset, n5 its
// stride, n6 the weights' offset, n7 epsilon's bits.
@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x;
    let d4 = p.n1 / 4u;
    let xi = (p.n2 + row * p.n3) / 4u;
    var ss = 0.0;
    for (var i = li; i < d4; i += 256u) {
        let v = src[xi + i];
        ss += dot(v, v);
    }
    let total = sum256(li, ss);
    let r = inverseSqrt(total / f32(p.n1) + bitcast<f32>(p.n7));
    let yi = (p.n4 + row * p.n5) / 4u;
    for (var i = li; i < d4; i += 256u) {
        let w = p.n6 + 4u * i;
        dst[yi + i] = src[xi + i] * r * vec4<f32>(small[w], small[w + 1u], small[w + 2u], small[w + 3u]);
    }
}

// LayerNorm of a row (no weights), then (1 + scale) · x + shift per column
// (both in `src2`): n0 rows, n1 width, n2 src's offset, n3 its stride, n4
// dst's offset, n5 its stride, n6 the shifts' offset, n7 the scales', n8
// epsilon's bits.
@compute @workgroup_size(256)
fn layernorm_mod(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row = wg.x;
    let d4 = p.n1 / 4u;
    let xi = (p.n2 + row * p.n3) / 4u;
    var s = 0.0;
    for (var i = li; i < d4; i += 256u) {
        let v = src[xi + i];
        s += v.x + v.y + v.z + v.w;
    }
    let mean = sum256(li, s) / f32(p.n1);
    var ss = 0.0;
    for (var i = li; i < d4; i += 256u) {
        let v = src[xi + i] - vec4<f32>(mean);
        ss += dot(v, v);
    }
    let r = inverseSqrt(sum256(li, ss) / f32(p.n1) + bitcast<f32>(p.n8));
    let yi = (p.n4 + row * p.n5) / 4u;
    for (var i = li; i < d4; i += 256u) {
        let c = 4u * i;
        let shift = vec4<f32>(src2[p.n6 + c], src2[p.n6 + c + 1u], src2[p.n6 + c + 2u], src2[p.n6 + c + 3u]);
        let scale = vec4<f32>(src2[p.n7 + c], src2[p.n7 + c + 1u], src2[p.n7 + c + 2u], src2[p.n7 + c + 3u]);
        dst[yi + i] = (src[xi + i] - vec4<f32>(mean)) * r * (1.0 + scale) + shift;
    }
}

// One head of one token (a workgroup of 64, each a pair of values) from
// its row to the heads' layout ([head][token][128]): RMSNorm over the head
// where asked (its weights in `small`), then the rotary positions (pairs
// side by side, or halves 64 apart; each pair's cosine and sine per token
// in `src2`). n0 tokens, n1 heads, n2 the first value's offset in src, n3
// src's stride, n4 dst's offset, n5 the tokens of dst's layout, n6 the
// first token's place in it, n7 the norm's weights (or 0xffffffff), n8 the
// rotation (0 none, 1 pairs, 2 halves), n9 the table's offset, n10
// epsilon's bits.
var<workgroup> qk_red: array<f32, 64>;

@compute @workgroup_size(64)
fn qk_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    let h = wg.y;
    let base = p.n2 + t * p.n3 + h * 128u;
    let halves = p.n8 == 2u;
    let i0 = select(2u * li, li, halves);
    let i1 = select(2u * li + 1u, li + 64u, halves);
    var a = src_at(base + i0);
    var b = src_at(base + i1);
    if (p.n7 != 0xffffffffu) {
        qk_red[li] = a * a + b * b;
        workgroupBarrier();
        for (var s = 32u; s > 0u; s >>= 1u) {
            if (li < s) {
                qk_red[li] += qk_red[li + s];
            }
            workgroupBarrier();
        }
        let r = inverseSqrt(qk_red[0] / 128.0 + bitcast<f32>(p.n10));
        a = a * r * small[p.n7 + i0];
        b = b * r * small[p.n7 + i1];
    }
    let place = p.n6 + t;
    if (p.n8 != 0u) {
        let at = p.n9 + (place * 64u + li) * 2u;
        let c = src2[at];
        let s = src2[at + 1u];
        let ra = a * c - b * s;
        let rb = b * c + a * s;
        a = ra;
        b = rb;
    }
    let out = p.n4 + (h * p.n5 + place) * 128u;
    dst_put(out + i0, a);
    dst_put(out + i1, b);
}

// The sampler's step: dst (n1..) += dt · src (n2..), n0 values, n3 dt's bits.
@compute @workgroup_size(256)
fn euler(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = flat_id(vec3<u32>(wg.x * 256u + li, wg.y, 0u), groups);
    if (i >= p.n0) {
        return;
    }
    dst_put(p.n1 + i, dst_at(p.n1 + i) + bitcast<f32>(p.n3) * src_at(p.n2 + i));
}

// SiLU: dst (n2..) = silu(src (n1..)), n0 values.
@compute @workgroup_size(256)
fn act(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = flat_id(vec3<u32>(wg.x * 256u + li, wg.y, 0u), groups);
    if (i >= p.n0) {
        return;
    }
    dst_put(p.n2 + i, silu(src_at(p.n1 + i)));
}

// Rows copied: n0 rows of n1 values (multiples of 4) from src (n2, n3
// apart) to dst (n4, n5 apart).
@compute @workgroup_size(256)
fn copy(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = flat_id(vec3<u32>(wg.x * 256u + li, wg.y, 0u), groups);
    let w4 = p.n1 / 4u;
    if (i >= p.n0 * w4) {
        return;
    }
    let r = i / w4;
    let c = i % w4;
    dst[(p.n4 + r * p.n5) / 4u + c] = src[(p.n2 + r * p.n3) / 4u + c];
}

// The autoencoder's latents (2·n0 × 2·n1 pixels of n3 channels, the first
// 32 taken, from n2) as tokens of 2 × 2 patches (channel c·4 + dy·2 + dx),
// normalized by the batch norm's means (n5) and spreads (n6) in `small`,
// to dst (n4). One invocation a token's channel.
@compute @workgroup_size(256)
fn patchify(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = flat_id(vec3<u32>(wg.x * 256u + li, wg.y, 0u), groups);
    if (i >= p.n0 * p.n1 * 128u) {
        return;
    }
    let token = i / 128u;
    let ch = i % 128u;
    let y = token / p.n1;
    let x = token % p.n1;
    let c = ch / 4u;
    let dy = (ch / 2u) % 2u;
    let dx = ch % 2u;
    let v = src_at(p.n2 + ((2u * y + dy) * 2u * p.n1 + 2u * x + dx) * p.n3 + c);
    dst_put(p.n4 + i, (v - small[p.n5 + ch]) / small[p.n6 + ch]);
}

// The reverse: tokens (n0 × n1 of 128 from n2) back to latents of 32
// channels (2·n0 × 2·n1, to n3), times the spreads (n5) plus the means
// (n4). One invocation a latent value.
@compute @workgroup_size(256)
fn unpatchify(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = flat_id(vec3<u32>(wg.x * 256u + li, wg.y, 0u), groups);
    if (i >= p.n0 * p.n1 * 128u) {
        return;
    }
    let c = i % 32u;
    let px = i / 32u;
    let w2 = 2u * p.n1;
    let yy = px / w2;
    let xx = px % w2;
    let ch = c * 4u + (yy % 2u) * 2u + xx % 2u;
    let token = (yy / 2u) * p.n1 + xx / 2u;
    dst_put(p.n3 + i, src_at(p.n2 + token * 128u + ch) * small[p.n5 + ch] + small[p.n4 + ch]);
}

// The group norm's statistics, first a chunk of pixels at a time: each
// invocation sums a channel quad over every 256/(n1/4)-th pixel of the
// chunk, then each group's count, mean and squared deviations are merged
// (Chan's formula) and written to dst (n5 + (chunk · 32 + group) · 3).
// n0 pixels, n1 channels (n1/4 divides 256), n2 src's offset, n3 channels
// a group (a multiple of 4), n4 pixels a chunk.
var<workgroup> gn_n: array<f32, 256>;
var<workgroup> gn_mean: array<f32, 256>;
var<workgroup> gn_m2: array<f32, 256>;

fn gn_merge(n: f32, mean: f32, m2: f32, n2: f32, mean2: f32, m22: f32) -> vec3<f32> {
    let total = n + n2;
    if (total == 0.0) {
        return vec3<f32>(0.0);
    }
    let d = mean2 - mean;
    return vec3<f32>(total, mean + d * n2 / total, m2 + m22 + d * d * n * n2 / total);
}

@compute @workgroup_size(256)
fn gn_stats(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let quads = p.n1 / 4u;
    let lanes = 256u / quads;
    let q = li % quads;
    let lane = li / quads;
    let first = wg.x * p.n4;
    let last = min(first + p.n4, p.n0);
    var n = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    for (var px = first + lane; px < last; px += lanes) {
        let v = src[(p.n2 + px * p.n1) / 4u + q];
        // Welford over the four values.
        for (var j = 0u; j < 4u; j++) {
            n += 1.0;
            let d = v[j] - mean;
            mean += d / n;
            m2 += d * (v[j] - mean);
        }
    }
    gn_n[li] = n;
    gn_mean[li] = mean;
    gn_m2[li] = m2;
    workgroupBarrier();
    if (li < 32u) {
        // Group li's quads: q from li·(n3/4), for every lane.
        let per = p.n3 / 4u;
        var acc = vec3<f32>(0.0);
        for (var l = 0u; l < lanes; l++) {
            for (var k = 0u; k < per; k++) {
                let t = l * quads + li * per + k;
                acc = gn_merge(acc.x, acc.y, acc.z, gn_n[t], gn_mean[t], gn_m2[t]);
            }
        }
        let at = p.n5 + (wg.x * 32u + li) * 3u;
        dst_put(at, acc.x);
        dst_put(at + 1u, acc.y);
        dst_put(at + 2u, acc.z);
    }
}

// The chunks merged: each group's mean and 1/√(variance + ε) to dst (n2 +
// 2·group). n0 chunks, n1 the chunks' offset in src, n3 epsilon's bits.
@compute @workgroup_size(32)
fn gn_reduce(@builtin(local_invocation_index) li: u32) {
    var acc = vec3<f32>(0.0);
    for (var c = 0u; c < p.n0; c++) {
        let at = p.n1 + (c * 32u + li) * 3u;
        acc = gn_merge(acc.x, acc.y, acc.z, src_at(at), src_at(at + 1u), src_at(at + 2u));
    }
    dst_put(p.n2 + 2u * li, acc.y);
    dst_put(p.n2 + 2u * li + 1u, inverseSqrt(acc.z / acc.x + bitcast<f32>(p.n3)));
}

// A softmax over each row, in place in dst, its values scaled first: n0
// rows, n1 values a row, n2 the offset, n3 the stride, n4 the scale's bits.
@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let at = p.n2 + wg.x * p.n3;
    let scale = bitcast<f32>(p.n4);
    var best = -3.0e38;
    for (var i = li; i < p.n1; i += 256u) {
        best = max(best, dst_at(at + i) * scale);
    }
    workgroupBarrier();
    red[li] = best;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] = max(red[li], red[li + s]);
        }
        workgroupBarrier();
    }
    let top = red[0];
    var total = 0.0;
    for (var i = li; i < p.n1; i += 256u) {
        let e = exp(dst_at(at + i) * scale - top);
        dst_put(at + i, e);
        total += e;
    }
    let all = sum256(li, total);
    for (var i = li; i < p.n1; i += 256u) {
        dst_put(at + i, dst_at(at + i) / all);
    }
}

// dst (n4, rows n5 apart) = the transpose of src (n0 rows × n1 columns
// from n2, n3 apart), 16 × 16 at a time.
var<workgroup> tr_tile: array<f32, 272>;

@compute @workgroup_size(16, 16)
fn transpose(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wg.y * 16u + lid.y;
    let c = wg.x * 16u + lid.x;
    if (r < p.n0 && c < p.n1) {
        tr_tile[lid.y * 17u + lid.x] = src_at(p.n2 + r * p.n3 + c);
    }
    workgroupBarrier();
    let orow = wg.x * 16u + lid.y;
    let ocol = wg.y * 16u + lid.x;
    if (orow < p.n1 && ocol < p.n0) {
        dst_put(p.n4 + orow * p.n5 + ocol, tr_tile[lid.x * 17u + lid.y]);
    }
}
