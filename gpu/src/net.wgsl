// The beat network's kernels (Beat This!, small), compiled after
// kernels.wgsl: they share its bindings (src, aux, dst, p) and helpers
// (linear_id, gelu, src4, store4, the workgroup arrays keys and vals), and
// the network also uses its matmul, rmsnorm (f7 = 1) and rope (f6 = 1).
// Activations are [band][frame][channel] in the front end, tokens of the
// width in the transformer.

// ---- The spectrogram into the front end: dst[f·frames + t] =
// src[t·bands + f] · aux[scale_off + f] + aux[shift_off + f].
// f0 frames, f1 bands, f2 scale_off, f3 shift_off.
@compute @workgroup_size(256)
fn net_in(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let frames = p.f0;
    let bands = p.f1;
    if (i >= frames * bands) {
        return;
    }
    let f = i / frames;
    let t = i % frames;
    dst[i] = src[t * bands + f] * aux[p.f2 + f] + aux[p.f3 + f];
}

// ---- A convolution's patches: per output band and frame, kf bands by 3
// frames of c channels ([f][dt][channel]), the frames padded with zeros.
// src [band][frame][c] → dst [out band][frame][kf·3·c].
// f0 frames, f1 c, f2 kf, f3 out bands.
@compute @workgroup_size(256)
fn net_patches(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let frames = p.f0;
    let c = p.f1;
    let kf = p.f2;
    let size = kf * 3u * c;
    if (i >= p.f3 * frames * size) {
        return;
    }
    let row = i / size;
    let col = i % size;
    let fo = row / frames;
    let t = row % frames;
    let f = col / (3u * c);
    let dt = (col % (3u * c)) / c;
    let ch = col % c;
    let band = fo * kf + f;
    // The frame t + dt - 1, inside the piece.
    var v = 0.0;
    if (t + dt >= 1u && t + dt - 1u < frames) {
        v = src[(band * frames + t + dt - 1u) * c + ch];
    }
    dst[i] = v;
}

// ---- In place: dst[i] = gelu(dst[i] · aux[scale_off + i % dim] +
// aux[shift_off + i % dim]). f0 count, f1 dim, f2 scale_off, f3 shift_off.
@compute @workgroup_size(256)
fn net_affine_gelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    let ch = i % p.f1;
    dst[i] = gelu(dst[i] * aux[p.f2 + ch] + aux[p.f3 + ch]);
}

// ---- In place: dst[i] = gelu(dst[i]). f0 count.
@compute @workgroup_size(256)
fn net_gelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    if (i >= p.f0) {
        return;
    }
    dst[i] = gelu(dst[i]);
}

// ---- [a][b][c] to [b][a][c]. f0 a, f1 b, f2 c.
@compute @workgroup_size(256)
fn net_transpose(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let a = p.f0;
    let b = p.f1;
    let c = p.f2;
    if (i >= a * b * c) {
        return;
    }
    let ch = i % c;
    let token = i / c;
    let ia = token / b;
    let jb = token % b;
    dst[(jb * a + ia) * c + ch] = src[i];
}

// ---- The front end's end: each frame's channels × bands, channel-major.
// src [band][frame][c] → dst [frame][c·bands]. f0 frames, f1 bands, f2 c.
@compute @workgroup_size(256)
fn net_flatten(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = linear_id(gid, groups);
    let frames = p.f0;
    let bands = p.f1;
    let c = p.f2;
    if (i >= frames * c * bands) {
        return;
    }
    let t = i / (c * bands);
    let ch = (i % (c * bands)) / bands;
    let f = i % bands;
    dst[i] = src[(f * frames + t) * c + ch];
}

// ---- Attention with a sigmoid gate per head, heads of 32. src: [token]
// [3·dim], the queries, keys and values side by side (queries and keys
// rotated before); aux: the gates' logits [token][heads]; dst [token][dim].
// f0 len (tokens per sequence), f1 dim, f2 heads. Workgroups: (queries /
// 64, heads, sequences); an online softmax over tiles of 64 keys, a head
// held as eight vectors, each a variable (see attention).
@compute @workgroup_size(64)
fn net_attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let dim = p.f1;
    let heads = p.f2;
    let width = 3u * dim;
    let seq = wid.z;
    let h = wid.y;
    let qp = wid.x * 64u + lid.x;
    let valid = qp < len;
    let scale = inverseSqrt(32.0);
    let z = vec4<f32>(0.0);
    var q = Q32(z, z, z, z, z, z, z, z);
    if (valid) {
        let at = (seq * len + qp) * width + h * 32u;
        q = Q32(src4(at) * scale, src4(at + 4u) * scale, src4(at + 8u) * scale, src4(at + 12u) * scale,
                src4(at + 16u) * scale, src4(at + 20u) * scale, src4(at + 24u) * scale, src4(at + 28u) * scale);
    }
    var st = Soft32(-1e30, 0.0, z, z, z, z, z, z, z, z);
    for (var k0 = 0u; k0 < len; k0 += 64u) {
        let kp = k0 + lid.x;
        if (kp < len) {
            let at = (seq * len + kp) * width + dim + h * 32u;
            for (var d = 0u; d < 8u; d++) {
                keys[lid.x * 8u + d] = src4(at + 4u * d);
                vals[lid.x * 8u + d] = src4(at + dim + 4u * d);
            }
        }
        workgroupBarrier();
        let count = min(64u, len - k0);
        if (valid) {
            // Four keys' scores at a time, then taken in in their order (as
            // attention's).
            var j = 0u;
            for (; j + 4u <= count; j += 4u) {
                let kb = j * 8u;
                let s0 = att_score(q, kb);
                let s1 = att_score(q, kb + 8u);
                let s2 = att_score(q, kb + 16u);
                let s3 = att_score(q, kb + 24u);
                st = att_take(st, s0, kb);
                st = att_take(st, s1, kb + 8u);
                st = att_take(st, s2, kb + 16u);
                st = att_take(st, s3, kb + 24u);
            }
            for (; j < count; j++) {
                st = att_take(st, att_score(q, j * 8u), j * 8u);
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        let token = seq * len + qp;
        let gate = 1.0 / (1.0 + exp(-aux[token * heads + h]));
        let k = gate / st.total;
        let at = token * dim + h * 32u;
        store4(at, st.a0 * k);
        store4(at + 4u, st.a1 * k);
        store4(at + 8u, st.a2 * k);
        store4(at + 12u, st.a3 * k);
        store4(at + 16u, st.a4 * k);
        store4(at + 20u, st.a5 * k);
        store4(at + 24u, st.a6 * k);
        store4(at + 28u, st.a7 * k);
    }
}
