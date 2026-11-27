// The language model's kernels a row at a time: the embeddings in, a row
// written for a matrix multiply, the residual stream's step between them,
// the per-layer inputs, and the next token.

// Each row's token's embeddings, scaled, from dst's start: an invocation a
// value. The embeddings are in `wq` (`at` in words), the scale is f32
// bits.
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let width = embed_width();
    if (i >= embed_rows() * width) {
        return;
    }
    let row = i / width;
    let token = ints[TOKENS + first_position(embed_stated(), embed_first()) + row];
    dst[i] = value(embed_kind(), embed_at() * 4u + token * row_bytes(embed_kind(), width), i % width) * bitcast<f32>(embed_scale());
}

// Rows of src (from its start) written for a matrix multiply
// (`put_block`), each value times src's at the same place of other rows
// where those are given (`times`; NONE: none): an invocation a block of
// Q8_BLOCK. The rows' values and scales go in `ints`.
@compute @workgroup_size(256)
fn quant(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let width = quant_width();
    let blocks = width / Q8_BLOCK;
    let n = quant_rows() * blocks;
    if (i >= n) {
        return;
    }
    if (quant_mode() != Q8) {
        // The f32 path: the rows' values as they are (`put_block`'s
        // layout, a row after another), invocation i taking values i,
        // i + n, i + 2n, …, so neighbours write neighbouring words
        // (`put_row_f32` says why).
        for (var j = 0u; j < Q8_BLOCK; j++) {
            let v = i + j * n;
            let row = v / width;
            let col = v % width;
            var x = src[row * quant_stride() + col];
            if (quant_times() != NONE) {
                x *= src[quant_times() + row * quant_times_stride() + col];
            }
            ints[quant_qa() + v] = bitcast<u32>(x);
        }
        return;
    }
    let row = i / blocks;
    let block = i % blocks;
    let at = row * quant_stride() + block * Q8_BLOCK;
    var times = NONE;
    if (quant_times() != NONE) {
        times = quant_times() + row * quant_times_stride() + block * Q8_BLOCK;
    }
    put_block(true, at, times, row, block, blocks, quant_qa(), quant_qs());
}

// The residual stream's step between matrices, a workgroup a row of
// dst: src's row RMS-normed and added (ADD), the row scaled (SCALE), then
// written for a matrix multiply (OUT), RMS-normed first where it has a
// norm (NONE: none). The width is at most 4096; src's stride is from its
// start; the row's values and scales go in `ints`.
@compute @workgroup_size(256)
fn residual(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    let k = residual_width();
    let flags = residual_flags();
    let x = (residual_x_row() + r) * k;
    let h = r * residual_stride();
    for (var i = lid; i < k; i += 256u) {
        staged[i] = dst[x + i];
    }
    if ((flags & ADD) != 0u) {
        var ss = 0.0;
        for (var i = lid; i < k; i += 256u) {
            ss += src[h + i] * src[h + i];
        }
        let inv = pow(sum256(lid, ss) / f32(k) + EPS, -0.5);
        for (var i = lid; i < k; i += 256u) {
            staged[i] = staged[i] + src[h + i] * inv * aux[residual_add() + i];
        }
    }
    if ((flags & SCALE) != 0u) {
        let s = aux[residual_scalar()];
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
    if (residual_norm() != NONE) {
        var ss = 0.0;
        for (var i = lid; i < k; i += 256u) {
            ss += staged[i] * staged[i];
        }
        let inv = pow(sum256(lid, ss) / f32(k) + EPS, -0.5);
        for (var i = lid; i < k; i += 256u) {
            staged[i] = staged[i] * inv * aux[residual_norm() + i];
        }
    }
    workgroupBarrier();
    if (residual_mode() != Q8) {
        put_row_f32(lid, r, k, residual_qa());
        return;
    }
    let blocks = k / Q8_BLOCK;
    for (var b = lid; b < blocks; b += 256u) {
        put_block(false, b * Q8_BLOCK, NONE, r, b, blocks, residual_qa(), residual_qs());
    }
}

// The per-layer inputs of rows, a workgroup a row, an invocation a value:
// the layer's slice of the embeddings' projection scaled and RMS-normed,
// plus the token's per-layer embedding scaled, times 1/√2. The projection
// is src (its stride from its start), the per-layer embeddings are in
// `wq`, the scales are f32 bits.
@compute @workgroup_size(256)
fn per_layer(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    let width = per_layer_width();
    var v = 0.0;
    if (lid < width) {
        v = src[(per_layer_x_row() + r) * per_layer_stride() + per_layer_slice() + lid] * bitcast<f32>(per_layer_scale());
    }
    let inv = pow(sum256(lid, v * v) / f32(width) + EPS, -0.5);
    if (lid >= width) {
        return;
    }
    let token = ints[TOKENS + first_position(per_layer_stated(), per_layer_first()) + r];
    let e = value(per_layer_kind(), per_layer_at() * 4u + token * row_bytes(per_layer_kind(), width), lid) * bitcast<f32>(per_layer_embed_scale());
    dst[per_layer_out() + r * width + lid] = (v * inv * aux[per_layer_norm() + lid] + e) * HALF_SQRT2;
}

// The next token: the largest of the head's workgroups' largest (the
// first of equals), put after the decoder's position, which moves on to
// it (not past the cache's last). The workgroups' largest are in `ints`.
var<workgroup> next_value: array<f32, 256>;
var<workgroup> next_token: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_index) lid: u32) {
    var v = -3.4e38;
    var t = 0xFFFFFFFFu;
    for (var j = lid; j < argmax_groups(); j += 256u) {
        let x = bitcast<f32>(ints[argmax_best() + 2u * j]);
        let i = ints[argmax_best() + 2u * j + 1u];
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
        if (pos + 1u < argmax_positions()) {
            ints[TOKENS + pos + 1u] = next_token[0];
            ints[0] = pos + 1u;
        }
    }
}
