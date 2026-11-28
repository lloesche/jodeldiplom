// The language model's attention. The cache keeps, for each layer with
// keys and values of its own, its keys with each value the positions'
// side by side (`[kv head][value][place]`, so that a step's scores read
// neighbouring positions in neighbouring invocations) and its values a
// position's side by side (`[kv head][place][value]`); the queries of the
// rows under way are kept in the same buffer (`[row][head][value]`). A
// position's place is the position masked (`mask`): a sliding layer's
// keys and values are a ring, a place the position's low bits (its window
// and a chunk's rows fit in it), a global layer's a place a position (all
// bits). There is no scaling of the scores (the queries and keys are
// normed) and no cap.

// The queries, keys and values of rows normed and turned: a workgroup a
// row and head (queries, then keys, then values), an invocation a pair of
// values (`i`, `i + D/2`, turned together). The queries go to their place
// by the cache, the keys and values into it at their positions' places.
// The rows' stride in src is from its start; the rotary table is `wq` (a
// cosine and sine for each pair turned and position: the first `turned`
// pairs turn, the rest stay as they are); `kv` is 0 where the layer
// shares another's keys and values.
@compute @workgroup_size(256)
fn prep(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let slot = wid.y;
    let d = prep_d();
    let half = d / 2u;
    let heads = prep_heads();
    let kv = prep_kv();
    let positions = prep_positions();
    let pos = first_position(prep_stated(), prep_first()) + row;
    let place = pos & prep_mask();
    let turned = prep_turned();
    let base = row * prep_stride() + slot * d;
    var x = 0.0;
    var y = 0.0;
    if (lid < half) {
        x = src[base + lid];
        y = src[base + lid + half];
    }
    let total = sum256(lid, x * x + y * y);
    let inv = pow(total / f32(d) + EPS, -0.5);
    if (lid >= half) {
        return;
    }
    var a = x * inv;
    var b = y * inv;
    if (slot < heads + kv) {
        let w = select(prep_k_norm(), prep_q_norm(), slot < heads) + lid;
        a *= aux[w];
        b *= aux[w + half];
        if (lid < turned) {
            let t = prep_rope() + (pos * turned + lid) * 2u;
            let c = bitcast<f32>(word(t));
            let s = bitcast<f32>(word(t + 1u));
            let ta = a * c - b * s;
            let tb = b * c + a * s;
            a = ta;
            b = tb;
        }
    }
    if (slot < heads) {
        let q = prep_queries() + (row * heads + slot) * d;
        dst[q + lid] = a;
        dst[q + lid + half] = b;
    } else if (slot < heads + kv) {
        let k = prep_keys() + (slot - heads) * d * positions;
        dst[k + lid * positions + place] = a;
        dst[k + (lid + half) * positions + place] = b;
    } else {
        let v = prep_values() + ((slot - heads - kv) * positions + place) * d;
        dst[v + lid] = a;
        dst[v + lid + half] = b;
    }
}

// A step's attention, split: a workgroup a split and key/value head (its
// four query heads together), the splits taking chunks of 64 positions in
// turn (each position at its place: `mask`) with an online softmax. A
// chunk's scores: an invocation a position and quarter of the values, a
// key's value read once for the four heads; its weighted values: an
// invocation one or two values of every head. Each split's largest score,
// its sum of exp(score − largest) and its values so weighted are kept for
// `join`. The positions are the decoder's own back to the window (all of
// them where there is none). The window is 0 for none; the queries are in
// src; the splits are written from dst's start, `splits` of them (the
// workgroups dispatched): the `least` at work, or one for every `per` of
// the chunks where that is more; those past them write only their totals,
// 0 (so the join leaves them out).
// What the four heads' invocations write each for its own head is in
// f32s, never one lane of a shared vec4: on Apple's GPUs a lane's store
// to workgroup memory may write the whole vector back, over the others'.
var<workgroup> step_q: array<vec4<f32>, 512>;
var<workgroup> step_s: array<vec4<f32>, 256>;
var<workgroup> step_p: array<f32, 256>;
var<workgroup> step_best: array<f32, 4>;
var<workgroup> step_total: array<f32, 4>;
var<workgroup> step_fade: array<f32, 4>;
var<workgroup> step_next: array<f32, 4>;
var<workgroup> step_pos: u32;

// The positions a step's attention at `pos` takes (back to `window`, 0:
// none): its first, and its chunks of 64.
fn step_span(pos: u32, window: u32) -> vec2<u32> {
    var lo = 0u;
    if (window > 0u) {
        lo = max(pos + 1u, window) - window;
    }
    return vec2<u32>(lo, (pos - lo) / 64u + 1u);
}

// The four heads' weighted sums of a value over the chunk's `count`
// positions from `first`, each at its place (`mask`) of the values from
// `v` (place 0's value), places `d` apart.
fn step_values(v: u32, first: u32, mask: u32, d: u32, count: u32) -> vec4<f32> {
    var o = vec4<f32>(0.0);
    var j = 0u;
    for (; j + 4u <= count; j += 4u) {
        let a = src[v + ((first + j) & mask) * d];
        let b = src[v + ((first + j + 1u) & mask) * d];
        let c = src[v + ((first + j + 2u) & mask) * d];
        let e = src[v + ((first + j + 3u) & mask) * d];
        o += step_p4(j) * a + step_p4(j + 1u) * b + step_p4(j + 2u) * c + step_p4(j + 3u) * e;
    }
    for (; j < count; j++) {
        o += step_p4(j) * src[v + ((first + j) & mask) * d];
    }
    return o;
}

// Position `j`'s weights of the four heads.
fn step_p4(j: u32) -> vec4<f32> {
    return vec4<f32>(step_p[4u * j], step_p[4u * j + 1u], step_p[4u * j + 2u], step_p[4u * j + 3u]);
}

// The four heads' fades.
fn step_fade4() -> vec4<f32> {
    return vec4<f32>(step_fade[0], step_fade[1], step_fade[2], step_fade[3]);
}

@compute @workgroup_size(256)
fn attend(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = attend_d();
    let window = attend_window();
    let positions = attend_positions();
    let mask = attend_mask();
    let split = wid.x;
    let splits = attend_splits();
    let g = wid.y;
    if (lid == 0u) {
        step_pos = ints[0];
    }
    if (lid < 4u) {
        step_best[lid] = -1e30;
        step_total[lid] = 0.0;
    }
    // The same in every invocation: the chunks' loop is uniform.
    let pos = workgroupUniformLoad(&step_pos);
    let span = step_span(pos, window);
    let lo = span.x;
    let chunks = span.y;
    // The splits at work take every `used`th chunk; one past them writes
    // its heads' totals as 0 alone (the join leaves it out).
    let per = attend_per();
    let used = clamp((chunks + per - 1u) / per, attend_least(), splits);
    let stride = d + 2u;
    let out = (g * 4u * splits + split) * stride;
    let head = splits * stride;
    if (split >= used) {
        if (lid < 4u) {
            dst[out + lid * head + 1u] = 0.0;
        }
        return;
    }
    // The queries, a value's four heads a vector.
    for (var i = lid; i < d; i += 256u) {
        let q = attend_queries() + g * 4u * d + i;
        step_q[i] = vec4<f32>(src[q], src[q + d], src[q + 2u * d], src[q + 3u * d]);
    }
    let j = lid % 64u;
    let quarter = lid / 64u;
    let part = d / 4u;
    let keys = attend_keys() + g * d * positions;
    // The values this invocation sums: `lid` (and `lid + 256` for heads of
    // 512).
    var oa = vec4<f32>(0.0);
    var ob = vec4<f32>(0.0);
    for (var c = split; c < chunks; c += used) {
        let first = lo + c * 64u;
        let count = min(64u, pos - first + 1u);
        workgroupBarrier();
        var s = vec4<f32>(0.0);
        if (j < count) {
            let k = keys + ((first + j) & mask);
            let e0 = quarter * part;
            for (var e = e0; e < e0 + part; e += 4u) {
                let a = src[k + e * positions];
                let b = src[k + (e + 1u) * positions];
                let cc = src[k + (e + 2u) * positions];
                let dd = src[k + (e + 3u) * positions];
                s += step_q[e] * a + step_q[e + 1u] * b + step_q[e + 2u] * cc + step_q[e + 3u] * dd;
            }
        }
        step_s[lid] = s;
        workgroupBarrier();
        if (lid < 64u) {
            step_s[lid] = step_s[lid] + step_s[lid + 64u] + step_s[lid + 128u] + step_s[lid + 192u];
        }
        workgroupBarrier();
        // Each head's largest score so far, then each position's weight of
        // each head an invocation (the exps at once), then each head's
        // total in the positions' order.
        if (lid < 4u) {
            var best = step_best[lid];
            for (var i = 0u; i < count; i++) {
                best = max(best, step_s[i][lid]);
            }
            step_next[lid] = best;
        }
        workgroupBarrier();
        let ph = lid / 64u;
        let pi = lid % 64u;
        if (pi < count) {
            step_p[4u * pi + ph] = exp_held(step_s[pi][ph] - step_next[ph]);
        }
        workgroupBarrier();
        if (lid < 4u) {
            let best = step_next[lid];
            let fade = exp_held(step_best[lid] - best);
            var total = step_total[lid] * fade;
            for (var i = 0u; i < count; i++) {
                total += step_p[4u * i + lid];
            }
            step_best[lid] = best;
            step_total[lid] = total;
            step_fade[lid] = fade;
        }
        workgroupBarrier();
        let v = attend_values() + g * positions * d;
        let fade = step_fade4();
        if (lid < d) {
            oa = oa * fade + step_values(v + lid, first, mask, d, count);
        }
        if (lid + 256u < d) {
            ob = ob * fade + step_values(v + lid + 256u, first, mask, d, count);
        }
    }
    workgroupBarrier();
    for (var h = 0u; h < 4u; h++) {
        if (lid < d) {
            dst[out + h * head + 2u + lid] = oa[h];
        }
        if (lid + 256u < d) {
            dst[out + h * head + 2u + lid + 256u] = ob[h];
        }
    }
    if (lid < 4u) {
        dst[out + lid * head] = step_best[lid];
        dst[out + lid * head + 1u] = step_total[lid];
    }
}

// The splits of a head put together (exp(largest − all's largest) each)
// into the step's attention, written for the output matrix (in blocks of
// Q8_BLOCK, as `put_block`): a workgroup a head. The splits are src from
// its start, `splits` a head; one whose total is 0 took no position (an
// idle one) and is left out. The row's values and scales go in `ints`.
var<workgroup> join_weight: array<f32, MOST_SPLITS>;
var<workgroup> join_total: f32;

@compute @workgroup_size(256)
fn join(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = join_d();
    let splits = join_splits();
    let h = wid.x;
    let stride = d + 2u;
    let first = h * splits * stride;
    if (lid == 0u) {
        var best = -1e30;
        for (var c = 0u; c < splits; c++) {
            let at = first + c * stride;
            if (src[at + 1u] > 0.0) {
                best = max(best, src[at]);
            }
        }
        var total = 0.0;
        for (var c = 0u; c < splits; c++) {
            let at = first + c * stride;
            var w = 0.0;
            if (src[at + 1u] > 0.0) {
                w = exp_held(src[at] - best);
            }
            join_weight[c] = w;
            total += w * src[at + 1u];
        }
        join_total = total;
    }
    workgroupBarrier();
    for (var i = lid; i < d; i += 256u) {
        var o = 0.0;
        for (var c = 0u; c < splits; c++) {
            let w = join_weight[c];
            if (w > 0.0) {
                o += w * src[first + c * stride + 2u + i];
            }
        }
        staged[i] = o / join_total;
    }
    workgroupBarrier();
    if (join_mode() != Q8) {
        put_row_f32(lid, h, d, join_qa());
        return;
    }
    let blocks = d / Q8_BLOCK;
    for (var b = lid; b < blocks; b += 256u) {
        let all = join_heads() * blocks;
        put_block(false, b * Q8_BLOCK, NONE, 0u, h * blocks + b, all, join_qa(), join_qs());
    }
}

// A prompt's attention, flash: a workgroup a head and block of queries
// (FLASH_LANES / (D/16) of them), each query's D/16 invocations holding 16
// of its values' weighted sums, over tiles of 8 keys (in workgroup memory,
// the keys, then the values) with an online softmax. Its keys are its own
// position's and those before it, back to the window (0: none), each at
// its place (`mask`). The queries are in src; `group` is the query heads
// a key/value head; the output is written from dst's start, a row's heads
// side by side. A dispatch takes a span of each block's walk over its
// keys: its tiles from `tile0`, `tiles` of them (the plan cuts a long
// attention so: a workgroup walks its keys one tile after another, so a
// dispatch takes as long as its longest walk, however few blocks it
// takes). A block's largest score, its total and its weighted sums so far
// are carried from one span to the next in dst from `state`
// (`[row][head]`, the largest and the total then the D sums), each as it
// was, so the sums are those of one dispatch bit for bit; the span that
// ends a block's walk writes its output. A block whose walk ended before
// the span does nothing.
// The tile in f32s, written a value an invocation and read four at a time
// ([`flash4`]): never one lane of a shared vec4 written (on Apple's GPUs
// that may write the whole vector back, over its neighbours' values).
var<workgroup> flash_tile: array<f32, 4096>;
var<workgroup> flash_part: array<f32, 256>;

// The tile's vector `i` (values 4i to 4i + 3).
fn flash4(i: u32) -> vec4<f32> {
    let b = 4u * i;
    return vec4<f32>(flash_tile[b], flash_tile[b + 1u], flash_tile[b + 2u], flash_tile[b + 3u]);
}

// Σ over the tile's 8 keys of p (pa: the first four, pb: the rest) times
// the tile's vector `at` of key 0 (keys `d4` vectors apart).
fn tile_mix(at: u32, d4: u32, pa: vec4<f32>, pb: vec4<f32>) -> vec4<f32> {
    var o = pa.x * flash4(at);
    o += pa.y * flash4(at + d4);
    o += pa.z * flash4(at + 2u * d4);
    o += pa.w * flash4(at + 3u * d4);
    o += pb.x * flash4(at + 4u * d4);
    o += pb.y * flash4(at + 5u * d4);
    o += pb.z * flash4(at + 6u * d4);
    o += pb.w * flash4(at + 7u * d4);
    return o;
}

@compute @workgroup_size(FLASH_LANES)
fn flash(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let rows = flash_rows();
    let d = flash_d();
    let window = flash_window();
    let positions = flash_positions();
    let mask = flash_mask();
    let heads = flash_heads();
    let lanes = d / 16u;
    let per_block = FLASH_LANES / lanes;
    let qi = lid / lanes;
    let lane = lid % lanes;
    let h = wid.y;
    let g = h / flash_group();
    let block = wid.x;
    let row = block * per_block + qi;
    let real = row < rows;
    let pos = flash_first() + row;
    let block_first = flash_first() + block * per_block;
    let block_last = flash_first() + min(block * per_block + per_block, rows) - 1u;
    var lo = 0u;
    if (window > 0u) {
        lo = max(block_first + 1u, window) - window;
    }
    // The block's walk over its keys, a tile of 8 a turn, and the span of
    // it this dispatch takes (the same in every invocation: its loop is
    // uniform).
    let walk = (block_last + 8u - lo) / 8u;
    let t0 = flash_tile0();
    if (t0 >= walk) {
        return;
    }
    let t1 = t0 + min(walk - t0, flash_tiles());
    let keys = flash_keys() + g * d * positions;
    let values = flash_values() + g * positions * d;
    let d4 = d / 4u;
    let carried = flash_state() + (row * heads + h) * (d + 2u);
    let sums = carried + 2u + lane * 16u;
    var best = -1e30;
    var total = 0.0;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    if (t0 > 0u && real) {
        best = dst[carried];
        total = dst[carried + 1u];
        o0 = vec4<f32>(dst[sums], dst[sums + 1u], dst[sums + 2u], dst[sums + 3u]);
        o1 = vec4<f32>(dst[sums + 4u], dst[sums + 5u], dst[sums + 6u], dst[sums + 7u]);
        o2 = vec4<f32>(dst[sums + 8u], dst[sums + 9u], dst[sums + 10u], dst[sums + 11u]);
        o3 = vec4<f32>(dst[sums + 12u], dst[sums + 13u], dst[sums + 14u], dst[sums + 15u]);
    }
    // Every invocation has read the sums carried before the query's first
    // writes the new ones (after the walk): the loop's barriers order only
    // workgroup memory.
    storageBarrier();
    for (var tile = t0; tile < t1; tile++) {
        let k0 = lo + 8u * tile;
        // The keys: neighbours take neighbouring positions.
        for (var i = lid; i < 8u * d; i += FLASH_LANES) {
            let j = i % 8u;
            let e = i / 8u;
            var v = 0.0;
            if (k0 + j <= block_last) {
                v = src[keys + e * positions + ((k0 + j) & mask)];
            }
            flash_tile[j * d + e] = v;
        }
        workgroupBarrier();
        // Each invocation a key's score over 128 of the values.
        let j = lane % 8u;
        let part = lane / 8u;
        var s = 0.0;
        if (real) {
            let q = flash_queries() + (row * heads + h) * d + part * 128u;
            let t = (j * d + part * 128u) / 4u;
            for (var i = 0u; i < 32u; i++) {
                let qv = vec4<f32>(src[q + 4u * i], src[q + 4u * i + 1u], src[q + 4u * i + 2u], src[q + 4u * i + 3u]);
                s += dot(qv, flash4(t + i));
            }
        }
        flash_part[lid] = s;
        workgroupBarrier();
        var sa = vec4<f32>(0.0);
        var sb = vec4<f32>(0.0);
        let pb0 = qi * lanes;
        for (var pp = 0u; pp < lanes / 8u; pp++) {
            let at = pb0 + pp * 8u;
            sa += vec4<f32>(flash_part[at], flash_part[at + 1u], flash_part[at + 2u], flash_part[at + 3u]);
            sb += vec4<f32>(flash_part[at + 4u], flash_part[at + 5u], flash_part[at + 6u], flash_part[at + 7u]);
        }
        let ka = vec4<u32>(k0, k0 + 1u, k0 + 2u, k0 + 3u);
        let kb = ka + vec4<u32>(4u);
        var oka = (ka <= vec4<u32>(pos)) & vec4<bool>(real);
        var okb = (kb <= vec4<u32>(pos)) & vec4<bool>(real);
        if (window > 0u) {
            oka = oka & (vec4<u32>(pos) - ka < vec4<u32>(window));
            okb = okb & (vec4<u32>(pos) - kb < vec4<u32>(window));
        }
        let ma = select(vec4<f32>(-1e30), sa, oka);
        let mb = select(vec4<f32>(-1e30), sb, okb);
        let m4 = max(ma, mb);
        let next = max(best, max(max(m4.x, m4.y), max(m4.z, m4.w)));
        let fade = exp_held(best - next);
        let pa = select(vec4<f32>(0.0), exp_held4(sa - vec4<f32>(next)), oka);
        let pb = select(vec4<f32>(0.0), exp_held4(sb - vec4<f32>(next)), okb);
        total = total * fade + dot(pa, vec4<f32>(1.0)) + dot(pb, vec4<f32>(1.0));
        best = next;
        workgroupBarrier();
        // The values: neighbours take neighbouring values.
        for (var i = lid; i < 8u * d; i += FLASH_LANES) {
            let jj = i / d;
            var v = 0.0;
            if (k0 + jj <= block_last) {
                v = src[values + ((k0 + jj) & mask) * d + i % d];
            }
            flash_tile[i] = v;
        }
        workgroupBarrier();
        let at = lane * 4u;
        o0 = o0 * fade + tile_mix(at, d4, pa, pb);
        o1 = o1 * fade + tile_mix(at + 1u, d4, pa, pb);
        o2 = o2 * fade + tile_mix(at + 2u, d4, pa, pb);
        o3 = o3 * fade + tile_mix(at + 3u, d4, pa, pb);
        workgroupBarrier();
    }
    if (!real) {
        return;
    }
    // The walk ended: the output. Else the sums so far, for the next span
    // (each a whole f32; the largest and the total, the same in each of
    // the query's invocations, by its first).
    var out = sums;
    var o = array<vec4<f32>, 4>(o0, o1, o2, o3);
    if (t1 == walk) {
        out = row * heads * d + h * d + lane * 16u;
        o = array<vec4<f32>, 4>(o0 / total, o1 / total, o2 / total, o3 / total);
    } else if (lane == 0u) {
        dst[carried] = best;
        dst[carried + 1u] = total;
    }
    for (var i = 0u; i < 4u; i++) {
        dst[out + 4u * i] = o[i].x;
        dst[out + 4u * i + 1u] = o[i].y;
        dst[out + 4u * i + 2u] = o[i].z;
        dst[out + 4u * i + 3u] = o[i].w;
    }
}
