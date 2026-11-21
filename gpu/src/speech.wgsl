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

// ---- LayerNorm in place, one row an invocation: each row of dim values
// to zero mean and unit variance (eps 1e-5), times aux[w_off + d] plus
// aux[b_off + d]. f0 rows, f1 dim, f2 w_off, f3 b_off.
@compute @workgroup_size(256)
fn sp_layernorm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let r = linear_id(gid, groups);
    if (r >= p.f0) {
        return;
    }
    let dim = p.f1;
    let x = r * dim;
    var sum = 0.0;
    for (var d = 0u; d < dim; d++) {
        sum += dst[x + d];
    }
    let mean = sum / f32(dim);
    var squares = 0.0;
    for (var d = 0u; d < dim; d++) {
        let v = dst[x + d] - mean;
        squares += v * v;
    }
    let k = inverseSqrt(squares / f32(dim) + 1e-5);
    for (var d = 0u; d < dim; d++) {
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
@compute @workgroup_size(64)
fn sp_attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let h = wid.y;
    let qp = wid.x * 64u + lid.x;
    let valid = qp < len;
    var q: array<vec4<f32>, 16>;
    var acc: array<vec4<f32>, 16>;
    for (var d = 0u; d < 16u; d++) {
        acc[d] = vec4<f32>(0.0);
        q[d] = vec4<f32>(0.0);
    }
    if (valid) {
        let at = qp * 2304u + h * 64u;
        for (var d = 0u; d < 16u; d++) {
            q[d] = src4(at + 4u * d) * 0.125;
        }
    }
    var best = -1e30;
    var total = 0.0;
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
            for (var j = 0u; j < count; j++) {
                let kb = j * 16u;
                var s = 0.0;
                for (var d = 0u; d < 16u; d++) {
                    s += dot(q[d], keys[kb + d]);
                }
                if (s > best) {
                    // A new maximum: rescale what was summed so far.
                    let fade = exp(best - s);
                    total = total * fade + 1.0;
                    for (var d = 0u; d < 16u; d++) {
                        acc[d] = acc[d] * fade + vals[kb + d];
                    }
                    best = s;
                } else {
                    let e = exp(s - best);
                    total += e;
                    for (var d = 0u; d < 16u; d++) {
                        acc[d] = fma(vec4<f32>(e), vals[kb + d], acc[d]);
                    }
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let at = qp * 768u + h * 64u;
        for (var d = 0u; d < 16u; d++) {
            store4(at + 4u * d, acc[d] / total);
        }
    }
}
