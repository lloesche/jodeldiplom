// The speech model's kernels (wav2vec2-base-960h), compiled after
// kernels.wgsl: they share its bindings (src, aux, dst, p) and helpers
// (linear_id, gelu, src4, store4, the workgroup arrays keys and vals), and
// the model also uses its matmul (the convolutions too: a frame's window is
// a contiguous run of the frames before, read with lda = stride · 512).

// ---- The first convolution: dst[t·512 + c] = Σj aux[w_off + c·10 + j] ·
// src[5t + j]. f0 frames, f1 w_off.
@compute @workgroup_size(256)
fn sp_conv0(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0 * 512u) {
        return;
    }
    let t = i / 512u;
    let c = i % 512u;
    let w = p.f1 + c * 10u;
    let x = 5u * t;
    var sum = 0.0;
    for (var j = 0u; j < 10u; j++) {
        sum += aux[w + j] * src[x + j];
    }
    dst[i] = sum;
}

// ---- In place: dst[i] = gelu((dst[i] - mean) / σ · aux[g_off + c] +
// aux[b_off + c]), c = i % channels, the statistics in src (each channel's
// mean and 1/σ over the whole voice, side by side: the GroupNorm of one group
// a channel). f0 count, f1 channels, f2 g_off, f3 b_off.
@compute @workgroup_size(256)
fn sp_chan_norm_gelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    let c = i % p.f1;
    let v = (dst[i] - src[2u * c]) * src[2u * c + 1u];
    dst[i] = gelu(v * aux[p.f2 + c] + aux[p.f3 + c]);
}

// ---- In place: dst[i] = gelu(dst[i]). f0 count.
@compute @workgroup_size(256)
fn sp_gelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    dst[i] = gelu(dst[i]);
}

// ---- LayerNorm in place: each row of dim values to zero mean and unit
// variance (eps 1e-5), times aux[w_off + d] plus aux[b_off + d]. f0 rows,
// f1 dim, f2 w_off, f3 b_off. A row is 32 invocations, as rmsnorm's.
@compute @workgroup_size(256)
fn sp_layernorm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = linear_id(gid, groups) / 32u;
    let lane = lid.x % 32u;
    let live = r < p.f0;
    let dim = p.f1;
    let x = r * dim;
    var sum = 0.0;
    if (live) {
        for (var d = lane; d < dim; d += 32u) {
            sum += dst[x + d];
        }
    }
    let mean = lane_sum(lid.x, sum) / f32(dim);
    var squares = 0.0;
    if (live) {
        for (var d = lane; d < dim; d += 32u) {
            let v = dst[x + d] - mean;
            squares += v * v;
        }
    }
    let k = inverseSqrt(lane_sum(lid.x, squares) / f32(dim) + 1e-5);
    if (!live) {
        return;
    }
    for (var d = lane; d < dim; d += 32u) {
        dst[x + d] = (dst[x + d] - mean) * k * aux[p.f2 + d] + aux[p.f3 + d];
    }
}

// ---- The positional convolution's patches for one group: dst[t·6144 +
// j·48 + i] = src[(t + j - 64)·768 + g·48 + i], zero past the ends.
// f0 frames, f1 group.
@compute @workgroup_size(256)
fn sp_pos_patches(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let frames = p.f0;
    if (i >= frames * 6144u) {
        return;
    }
    let t = i / 6144u;
    let r = i % 6144u;
    let j = r / 48u;
    let ch = r % 48u;
    var v = 0.0;
    if (t + j >= 64u && t + j - 64u < frames) {
        v = src[(t + j - 64u) * 768u + p.f1 * 48u + ch];
    }
    dst[i] = v;
}

// ---- The positional convolution added: dst[i] += gelu(src[i] +
// aux[b_off + i % 768]). f0 count, f1 b_off.
@compute @workgroup_size(256)
fn sp_pos_add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    dst[i] += gelu(src[i] + aux[p.f1 + i % 768u]);
}

// ---- Attention, 12 heads of 64. src: queries, keys and values side by
// side, [token][2304]; dst: [token][768]. f0 len. Workgroups (queries / 64,
// heads); an online softmax over tiles of 32 keys (the workgroup arrays
// hold 32 keys of 16 vectors, and their values).
// A key of 64 values from the tile (its 16 vectors from b).
fn sp_key64(b: u32) -> H64 {
    return H64(keys[b + 0u], keys[b + 1u], keys[b + 2u], keys[b + 3u], keys[b + 4u], keys[b + 5u], keys[b + 6u], keys[b + 7u], keys[b + 8u], keys[b + 9u], keys[b + 10u], keys[b + 11u], keys[b + 12u], keys[b + 13u], keys[b + 14u], keys[b + 15u]);
}

// Its values from the tile.
fn sp_val64(b: u32) -> H64 {
    return H64(vals[b + 0u], vals[b + 1u], vals[b + 2u], vals[b + 3u], vals[b + 4u], vals[b + 5u], vals[b + 6u], vals[b + 7u], vals[b + 8u], vals[b + 9u], vals[b + 10u], vals[b + 11u], vals[b + 12u], vals[b + 13u], vals[b + 14u], vals[b + 15u]);
}

// 64 values of src from at.
fn sp_src64(at: u32) -> H64 {
    return H64(src4(at + 0u * 4u), src4(at + 1u * 4u), src4(at + 2u * 4u), src4(at + 3u * 4u), src4(at + 4u * 4u), src4(at + 5u * 4u), src4(at + 6u * 4u), src4(at + 7u * 4u), src4(at + 8u * 4u), src4(at + 9u * 4u), src4(at + 10u * 4u), src4(at + 11u * 4u), src4(at + 12u * 4u), src4(at + 13u * 4u), src4(at + 14u * 4u), src4(at + 15u * 4u));
}

// 64 values to dst from at.
fn sp_store64(at: u32, v: H64) {
    store4(at + 0u, v.v0);
    store4(at + 4u, v.v1);
    store4(at + 8u, v.v2);
    store4(at + 12u, v.v3);
    store4(at + 16u, v.v4);
    store4(at + 20u, v.v5);
    store4(at + 24u, v.v6);
    store4(at + 28u, v.v7);
    store4(at + 32u, v.v8);
    store4(at + 36u, v.v9);
    store4(at + 40u, v.v10);
    store4(at + 44u, v.v11);
    store4(at + 48u, v.v12);
    store4(at + 52u, v.v13);
    store4(at + 56u, v.v14);
    store4(at + 60u, v.v15);
}

@compute @workgroup_size(64)
fn sp_attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let h = wid.y;
    let qp = wid.x * 64u + lid.x;
    let valid = qp < len;
    var q = h64_zero();
    var st = soft64();
    if (valid) {
        q = h64_scale(sp_src64(qp * 2304u + h * 64u), 0.125);
    }
    for (var k0 = 0u; k0 < len; k0 += 32u) {
        // 64 invocations load 32 keys and values of 16 vectors: 8 each.
        for (var r = 0u; r < 8u; r++) {
            let slot = lid.x * 8u + r;
            let kp = k0 + slot / 16u;
            if (kp < len) {
                let at = kp * 2304u + 768u + h * 64u + 4u * (slot % 16u);
                keys[slot] = src4(at);
                vals[slot] = src4(at + 768u);
            }
        }
        workgroupBarrier();
        let count = min(32u, len - k0);
        if (valid) {
            // Four keys' scores at a time (four sums under way together),
            // then taken in in their order.
            var j = 0u;
            for (; j + 4u <= count; j += 4u) {
                let kb = j * 16u;
                let s0 = h64_dot(q, sp_key64(kb));
                let s1 = h64_dot(q, sp_key64(kb + 16u));
                let s2 = h64_dot(q, sp_key64(kb + 32u));
                let s3 = h64_dot(q, sp_key64(kb + 48u));
                st = soft64_take(st, s0, sp_val64(kb));
                st = soft64_take(st, s1, sp_val64(kb + 16u));
                st = soft64_take(st, s2, sp_val64(kb + 32u));
                st = soft64_take(st, s3, sp_val64(kb + 48u));
            }
            for (; j < count; j++) {
                let kb = j * 16u;
                st = soft64_take(st, h64_dot(q, sp_key64(kb)), sp_val64(kb));
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        sp_store64(qp * 768u + h * 64u, h64_div(st.acc, st.total));
    }
}
