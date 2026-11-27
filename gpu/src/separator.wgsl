// The separator's own kernel, compiled after `shared.wgsl`: its
// attention (its matrix multiplies, norms, rotary positions and
// transposes are the shared ones).

// ---- Attention, 8 query heads on 4 key/value heads of 32. src: queries
// [token][256]; aux: keys and values [token][256] (keys in columns 0..128,
// values in 128..256); dst: [token][256]. f0 len (tokens per sequence).
// Workgroups: (queries / 64, heads, sequences); an online softmax over
// tiles of 64 keys (`keys`, `vals`), so the scores are never stored. A
// head is held as eight vectors of four, each a variable of its own
// (`Q32`, `Soft32`): an array indexed in a loop may be kept in memory
// rather than registers (Apple's GPUs, from the loops wgpu writes for
// Metal), and this loop runs for every query and key.
@compute @workgroup_size(64)
fn attention(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let len = p.f0;
    let seq = wid.z;
    let h = wid.y;
    let kv = h / 2u;
    let qp = wid.x * 64u + lid.x;
    let valid = qp < len;
    let scale = inverseSqrt(32.0);
    let z = vec4<f32>(0.0);
    var q = Q32(z, z, z, z, z, z, z, z);
    if (valid) {
        let at = (seq * len + qp) * 256u + h * 32u;
        q = Q32(src4(at) * scale, src4(at + 4u) * scale, src4(at + 8u) * scale, src4(at + 12u) * scale,
                src4(at + 16u) * scale, src4(at + 20u) * scale, src4(at + 24u) * scale, src4(at + 28u) * scale);
    }
    var st = Soft32(-1e30, 0.0, z, z, z, z, z, z, z, z);
    for (var k0 = 0u; k0 < len; k0 += 64u) {
        let kp = k0 + lid.x;
        if (kp < len) {
            let at = (seq * len + kp) * 256u + kv * 32u;
            for (var d = 0u; d < 8u; d++) {
                let i = at + 4u * d;
                keys[lid.x * 8u + d] = vec4<f32>(aux[i], aux[i + 1u], aux[i + 2u], aux[i + 3u]);
                let j = i + 128u;
                vals[lid.x * 8u + d] = vec4<f32>(aux[j], aux[j + 1u], aux[j + 2u], aux[j + 3u]);
            }
        }
        workgroupBarrier();
        let count = min(64u, len - k0);
        if (valid) {
            // Four keys' scores at a time (four sums under way together),
            // then taken in in their order.
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
        let at = (seq * len + qp) * 256u + h * 32u;
        store4(at, st.a0 / st.total);
        store4(at + 4u, st.a1 / st.total);
        store4(at + 8u, st.a2 / st.total);
        store4(at + 12u, st.a3 / st.total);
        store4(at + 16u, st.a4 / st.total);
        store4(at + 20u, st.a5 / st.total);
        store4(at + 24u, st.a6 / st.total);
        store4(at + 28u, st.a7 / st.total);
    }
}
