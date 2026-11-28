// Attention over heads of 128 laid out [head][token][128] (flash
// attention: an online softmax over tiles of keys, the scores never
// stored). The numbers, for every attention kernel (this one on f32
// multiply-adds, `attention_coop.wgsl` and `attention8.wgsl` on cooperative
// matrices): n0 queries, n1 keys, n2
// heads, n3 query heads a key head serves, n4 the queries' offset, n5 the
// keys', n6 the values', n7 the output's (token-major: head h's 128
// values at h·128 of a row), n8 its stride, n9 the scale's bits, n10 1 for
// causal, n11 the keys that count (those from it on are left out), n12 the
// dispatch's first block of queries (a dispatch cut into slices).
//
// Here a workgroup takes 64 queries of a head, two invocations a query,
// each with half of the head (64 values) in registers; keys come 16 at a
// time, each half's dot product shared through workgroup memory.

var<workgroup> fa_k: array<vec4<f32>, 512>;
var<workgroup> fa_v: array<vec4<f32>, 512>;
var<workgroup> fa_s: array<f32, 2048>;

// 64 values (16 vectors) from `at` of src (a multiple of 4).
fn fa_src(at: u32) -> H64 {
    let i = at / 4u;
    return H64(src[i], src[i + 1u], src[i + 2u], src[i + 3u], src[i + 4u], src[i + 5u], src[i + 6u], src[i + 7u], src[i + 8u], src[i + 9u], src[i + 10u], src[i + 11u], src[i + 12u], src[i + 13u], src[i + 14u], src[i + 15u]);
}

fn fa_kh(at: u32) -> H64 {
    return H64(fa_k[at], fa_k[at + 1u], fa_k[at + 2u], fa_k[at + 3u], fa_k[at + 4u], fa_k[at + 5u], fa_k[at + 6u], fa_k[at + 7u], fa_k[at + 8u], fa_k[at + 9u], fa_k[at + 10u], fa_k[at + 11u], fa_k[at + 12u], fa_k[at + 13u], fa_k[at + 14u], fa_k[at + 15u]);
}

fn fa_vh(at: u32) -> H64 {
    return H64(fa_v[at], fa_v[at + 1u], fa_v[at + 2u], fa_v[at + 3u], fa_v[at + 4u], fa_v[at + 5u], fa_v[at + 6u], fa_v[at + 7u], fa_v[at + 8u], fa_v[at + 9u], fa_v[at + 10u], fa_v[at + 11u], fa_v[at + 12u], fa_v[at + 13u], fa_v[at + 14u], fa_v[at + 15u]);
}

// Whether query `q` sees key `k`.
fn fa_sees(q: u32, k: u32) -> bool {
    return k < p.n11 && (p.n10 == 0u || k <= q);
}

@compute @workgroup_size(128)
fn attention(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    attention_f32(wg, li);
}

fn attention_f32(wg: vec3<u32>, li: u32) {
    let h = wg.y;
    let kvh = h / p.n3;
    let query = (wg.x + p.n12) * 64u + li / 2u;
    let half = li % 2u;
    let live = query < p.n0;
    var q = h64_zero();
    if (live) {
        q = h64_scale(fa_src(p.n4 + (h * p.n0 + query) * 128u + half * 64u), bitcast<f32>(p.n9));
    }
    var o = h64_zero();
    var best = -1.0e30;
    var total = 0.0;
    let tiles = (p.n1 + 15u) / 16u;
    let kbase = p.n5 + kvh * p.n1 * 128u;
    let vbase = p.n6 + kvh * p.n1 * 128u;
    for (var t = 0u; t < tiles; t++) {
        workgroupBarrier();
        for (var j = 0u; j < 4u; j++) {
            let e = li + 128u * j;
            let key = t * 16u + e / 32u;
            var kv = vec4<f32>(0.0);
            var vv = vec4<f32>(0.0);
            if (key < p.n1) {
                kv = src[(kbase + key * 128u) / 4u + e % 32u];
                vv = src[(vbase + key * 128u) / 4u + e % 32u];
            }
            fa_k[e] = kv;
            fa_v[e] = vv;
        }
        workgroupBarrier();
        for (var j = 0u; j < 16u; j++) {
            fa_s[li * 16u + j] = h64_dot(q, fa_kh(j * 32u + half * 16u));
        }
        workgroupBarrier();
        let mine = li * 16u;
        let other = (li ^ 1u) * 16u;
        var top = best;
        for (var j = 0u; j < 16u; j++) {
            if (fa_sees(query, t * 16u + j)) {
                top = max(top, fa_s[mine + j] + fa_s[other + j]);
            }
        }
        let fade = exp_held(best - top);
        o = h64_scale(o, fade);
        total *= fade;
        best = top;
        for (var j = 0u; j < 16u; j++) {
            if (fa_sees(query, t * 16u + j)) {
                let e = exp_held(fa_s[mine + j] + fa_s[other + j] - best);
                total += e;
                o = h64_add(o, e, fa_vh(j * 32u + half * 16u));
            }
        }
    }
    if (!live) {
        return;
    }
    o = h64_div(o, total);
    let out = (p.n7 + query * p.n8 + h * 128u + half * 64u) / 4u;
    dst_put4(out, o.v0); dst_put4(out + 1u, o.v1); dst_put4(out + 2u, o.v2); dst_put4(out + 3u, o.v3);
    dst_put4(out + 4u, o.v4); dst_put4(out + 5u, o.v5); dst_put4(out + 6u, o.v6); dst_put4(out + 7u, o.v7);
    dst_put4(out + 8u, o.v8); dst_put4(out + 9u, o.v9); dst_put4(out + 10u, o.v10); dst_put4(out + 11u, o.v11);
    dst_put4(out + 12u, o.v12); dst_put4(out + 13u, o.v13); dst_put4(out + 14u, o.v14); dst_put4(out + 15u, o.v15);
}
