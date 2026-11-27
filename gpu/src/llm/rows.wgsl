// The language model's kernels a row at a time: the embeddings in, a row
// written for a matrix multiply, the residual stream's step between them,
// the per-layer inputs, and the next token.

// Each row's token's embeddings, scaled: an invocation a value. Numbers:
// rows, the first position, whether it is the decoder's, where they go,
// the embeddings in `wq`, the scale (f32 bits), the width, the tokens.
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * 256u;
    let width = p.f6;
    if (i >= p.f0 * width) {
        return;
    }
    let row = i / width;
    let token = ints[TOKENS + first_position(p.f2, p.f1) + row];
    dst[p.f3 + i] = q4_value(p.f4, p.f7, width, token, i % width) * bitcast<f32>(p.f5);
}

// Rows of src written for a matrix multiply (`put_block`), each value
// times src's at the same place of other rows where those are given: an
// invocation a block of 32. Numbers: rows, their values, their place,
// their stride, the rows' values and scales in `ints`, the path, the
// other rows' place (NONE: none) and stride.
@compute @workgroup_size(256)
fn quant(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = gid.x + gid.y * groups.x * 256u;
    let blocks = p.f1 / 32u;
    if (i >= p.f0 * blocks) {
        return;
    }
    let row = i / blocks;
    let block = i % blocks;
    let at = p.f2 + row * p.f3 + block * 32u;
    var times = NONE;
    if (p.f7 != NONE) {
        times = p.f7 + row * p.f8 + block * 32u;
    }
    put_block(true, at, times, row, block, blocks, p.f4, p.f5, p.f6);
}

// The residual stream's step between matrices, a workgroup a row of
// dst: src's row RMS-normed and added (ADD), the row scaled (SCALE), then
// written for a matrix multiply (OUT), RMS-normed first where it has a
// norm. Numbers: the width (at most 4096), the rows' place in dst, src's
// place and stride, the added norm, the scalar, the written norm (NONE:
// none), the row's values and scales in `ints`, the path, the flags, the
// first row in dst.
const ADD: u32 = 1u;
const SCALE: u32 = 2u;
const OUT: u32 = 4u;

@compute @workgroup_size(256)
fn residual(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    let k = p.f0;
    let flags = p.f10;
    let x = p.f1 + (p.f11 + r) * k;
    let h = p.f2 + r * p.f3;
    for (var i = lid; i < k; i += 256u) {
        staged[i] = dst[x + i];
    }
    if ((flags & ADD) != 0u) {
        var ss = 0.0;
        for (var i = lid; i < k; i += 256u) {
            ss += src[h + i] * src[h + i];
        }
        let inv = pow(wg_sum(lid, ss) / f32(k) + EPS, -0.5);
        for (var i = lid; i < k; i += 256u) {
            staged[i] = staged[i] + src[h + i] * inv * aux[p.f4 + i];
        }
    }
    if ((flags & SCALE) != 0u) {
        let s = aux[p.f5];
        for (var i = lid; i < k; i += 256u) {
            staged[i] = staged[i] * s;
        }
    }
    if ((flags & (ADD | SCALE)) != 0u) {
        for (var i = lid; i < k; i += 256u) {
            dst[x + i] = staged[i];
        }
    }
    if ((flags & OUT) == 0u) {
        return;
    }
    if (p.f6 != NONE) {
        var ss = 0.0;
        for (var i = lid; i < k; i += 256u) {
            ss += staged[i] * staged[i];
        }
        let inv = pow(wg_sum(lid, ss) / f32(k) + EPS, -0.5);
        for (var i = lid; i < k; i += 256u) {
            staged[i] = staged[i] * inv * aux[p.f6 + i];
        }
    }
    workgroupBarrier();
    let blocks = k / 32u;
    for (var b = lid; b < blocks; b += 256u) {
        put_block(false, b * 32u, NONE, r, b, blocks, p.f7, p.f8, p.f9);
    }
}

// The per-layer inputs of rows, a workgroup a row, an invocation a value:
// the layer's slice of the embeddings' projection scaled and RMS-normed,
// plus the token's per-layer embedding scaled, times 1/√2. Numbers: rows,
// the first position, whether it is the decoder's, the projection's
// place (src) and stride, the layer's slice in it, the norm, the per-layer
// embeddings in `wq`, where they go, the width, the projection's first
// row, its scale and the embeddings' (f32 bits), the tokens.
@compute @workgroup_size(256)
fn per_layer(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    let width = p.f9;
    var v = 0.0;
    if (lid < width) {
        v = src[p.f3 + (p.f10 + r) * p.f4 + p.f5 + lid] * bitcast<f32>(p.f11);
    }
    let inv = pow(wg_sum(lid, v * v) / f32(width) + EPS, -0.5);
    if (lid >= width) {
        return;
    }
    let token = ints[TOKENS + first_position(p.f2, p.f1) + r];
    let e = q4_value(p.f7, p.f13, width, token, lid) * bitcast<f32>(p.f12);
    dst[p.f8 + r * width + lid] = (v * inv * aux[p.f6 + lid] + e) * 0.70710678;
}

// The next token: the largest of the head's workgroups' largest (the
// first of equals), put after the decoder's position, which moves on to
// it (not past the cache's last). Numbers: the workgroups, their largest
// in `ints`, the cache's positions.
var<workgroup> next_value: array<f32, 256>;
var<workgroup> next_token: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_index) lid: u32) {
    var v = -3.4e38;
    var t = 0xFFFFFFFFu;
    for (var j = lid; j < p.f0; j += 256u) {
        let x = bitcast<f32>(ints[p.f1 + 2u * j]);
        let i = ints[p.f1 + 2u * j + 1u];
        if (x > v || (x == v && i < t)) {
            v = x;
            t = i;
        }
    }
    next_value[lid] = v;
    next_token[lid] = t;
    for (var s = 128u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if (lid < s) {
            let x = next_value[lid + s];
            let i = next_token[lid + s];
            if (x > next_value[lid] || (x == next_value[lid] && i < next_token[lid])) {
                next_value[lid] = x;
                next_token[lid] = i;
            }
        }
    }
    if (lid == 0u) {
        let pos = ints[0];
        if (pos + 1u < p.f2) {
            ints[TOKENS + pos + 1u] = next_token[0];
            ints[0] = pos + 1u;
        }
    }
}
