// The weights as the GGUF holds them, read where they lie in `wq` (a
// matrix from a multiple of 16 bytes, its rows one after another, each
// row its blocks):
//
// - Q4_K: 256 values in 144 bytes: the scale and the minimum's scale (two
//   halves), 12 bytes of 6-bit scales and minimums (one each for each 32
//   values), 128 bytes of 4-bit values (byte l of each 32 holding value l
//   of the 32 at 2c in its low nibble and of the 32 at 2c + 1 in its high
//   one): value = d·sc·q − dmin·m. A block is 36 words, so it starts on a
//   pair of them, read a pair at a time.
// - Q5_K: as Q4_K, with 32 bytes of fifth bits (byte l: value l of each
//   32, bit 32-group) after the scales: 176 bytes.
// - Q6_K: 256 values in 210 bytes: 128 bytes of the low four bits, 64 of
//   the high two, 16 signed 8-bit scales (one each for each 16 values), the
//   scale (a half): value = d·sc·(q − 32). A block starts halfway into a
//   word where it follows an odd number of them.
// - BF16 and F32: the values.
//
// For the q8 path a row is read 16 values at a time (`half_*`), as signed
// 8-bit values four to a word in order, with their scale and the
// minimum's scale (the latter in the first 16 of each 32 only: it is
// taken once with the 32 rows' values' sum); for the plain path 8 at a
// time as f32s (`eight_*`); for an embedding one value at a time
// (`value`). A kind is named by its GGML type number (F32, BF16, Q4K,
// Q5K, Q6K: the constants given before the prelude).

const LOW: u32 = 0x0F0F0F0Fu;

// The 16 bits at byte `b` (even) of `wq`.
fn half_at(b: u32) -> u32 {
    return (word(b >> 2u) >> ((b & 2u) * 8u)) & 0xFFFFu;
}

// Four words from byte `b` (even) of `wq`, the first lowest: read
// as they lie, or from the words around them where `b` is halfway into
// one.
fn words_at(b: u32) -> vec4<u32> {
    let i = b >> 2u;
    let w = vec4<u32>(word(i), word(i + 1u), word(i + 2u), word(i + 3u));
    if ((b & 2u) == 0u) {
        return w;
    }
    let next = vec4<u32>(w.yzw, word(i + 4u));
    return (w >> vec4<u32>(16u)) | (next << vec4<u32>(16u));
}

// Words `i` and `i + 1` (`i` even) of `wq`.
fn pair(i: u32) -> vec2<u32> {
    return wq[i >> 1u];
}

// A word's four bytes as f32s, the first first.
fn bytes_f32(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(w, w >> 8u, w >> 16u, w >> 24u) & vec4<u32>(255u));
}

// 16 values of a row for the q8 path.
struct Half {
    q: vec4<u32>,
    s: f32,
    m: f32,
}

// 8 values of a row, in order.
struct Eight {
    a: vec4<f32>,
    b: vec4<f32>,
}

// The bytes of a row of `cols` values of `kind` (each kind's
// `row_bytes_KIND` is given before the prelude).
fn row_bytes(kind: u32, cols: u32) -> u32 {
    switch kind {
        case Q4K: {
            return row_bytes_q4k(cols);
        }
        case Q5K: {
            return row_bytes_q5k(cols);
        }
        case Q6K: {
            return row_bytes_q6k(cols);
        }
        case BF16: {
            return row_bytes_bf16(cols);
        }
        default: {
            return row_bytes_f32(cols);
        }
    }
}

// The scale and minimum (d·sc, dmin·m) of the 32 values at `s` of a Q4_K
// or Q5_K block from its first four words (d and dmin, the 12 bytes of
// scales and minimums).
fn k_scale_min(head: vec4<u32>, s: u32) -> vec2<f32> {
    let dm = unpack2x16float(head.x);
    let j = (s & 3u) * 8u;
    let a = head.y >> j;
    let b = head.z >> j;
    var sc = a & 63u;
    var m = b & 63u;
    if (s >= 4u) {
        let c = head.w >> j;
        sc = (c & 15u) | (((a >> 6u) & 3u) << 4u);
        m = ((c >> 4u) & 15u) | (((b >> 6u) & 3u) << 4u);
    }
    return vec2<f32>(dm.x * f32(sc), dm.y * f32(m));
}

// Where a Q4_K row's 32 values `g` are (the row at byte `row`): its
// block's first word, the 32's place in the block, the first word of its
// nibbles.
fn q4k_block(row: u32, g: u32) -> vec4<u32> {
    let w = (row >> 2u) + (g >> 3u) * (Q4K_BYTES / 4u);
    return vec4<u32>(w, (g & 7u), w + 4u + ((g & 7u) >> 1u) * 8u, 0u);
}

// Values 16h to 16h + 15 of the Q4_K row at byte `row`.
fn half_q4k(row: u32, h: u32) -> Half {
    let at = q4k_block(row, h >> 1u);
    let s = at.y;
    let sm = k_scale_min(vec4<u32>(pair(at.x), pair(at.x + 2u)), s);
    let v = at.z + (h & 1u) * 4u;
    let q = (vec4<u32>(pair(v), pair(v + 2u)) >> vec4<u32>((s & 1u) * 4u)) & vec4<u32>(LOW);
    return Half(q, sm.x, select(0.0, -sm.y, (h & 1u) == 0u));
}

// Values 8e to 8e + 7 of the Q4_K row at byte `row`.
fn eight_q4k(row: u32, e: u32) -> Eight {
    let at = q4k_block(row, e >> 2u);
    let s = at.y;
    let sm = k_scale_min(vec4<u32>(pair(at.x), pair(at.x + 2u)), s);
    let q = (pair(at.z + (e & 3u) * 2u) >> vec2<u32>((s & 1u) * 4u)) & vec2<u32>(LOW);
    return Eight(sm.x * bytes_f32(q.x) - sm.y, sm.x * bytes_f32(q.y) - sm.y);
}

// Value `col` of the Q5_K row at byte `row`.
fn value_q5k(row: u32, col: u32) -> f32 {
    let w = (row >> 2u) + (col >> 8u) * (Q5K_BYTES / 4u);
    let s = (col >> 5u) & 7u;
    let l = col & 31u;
    let sm = k_scale_min(vec4<u32>(word(w), word(w + 1u), word(w + 2u), word(w + 3u)), s);
    let low = (word(w + 12u + (s >> 1u) * 8u + (l >> 2u)) >> ((l & 3u) * 8u + (s & 1u) * 4u)) & 15u;
    let high = (word(w + 4u + (l >> 2u)) >> ((l & 3u) * 8u + s)) & 1u;
    return sm.x * f32(low | (high << 4u)) - sm.y;
}

// Four 6-bit values (0 to 63) from their low nibbles (`low`, shifted to
// them) and high two bits (`high`, shifted to them).
fn q6_bytes(low: vec4<u32>, high: vec4<u32>) -> vec4<u32> {
    return (low & vec4<u32>(LOW)) | ((high & vec4<u32>(0x03030303u)) << vec4<u32>(4u));
}

// The scale (d·sc) of the 16 values at `i` of the Q6_K block at byte `b`.
fn q6_scale(b: u32, i: u32) -> f32 {
    let d = unpack2x16float(half_at(b + 208u)).x;
    let at = b + 192u + i;
    let byte = (word(at >> 2u) >> ((at & 3u) * 8u)) & 255u;
    return d * f32(bitcast<i32>(byte << 24u) >> 24u);
}

// Where a Q6_K row's 16 or 8 values from `16h` or `8e` are (its 32 values
// `g`, the row at byte `row`, `part` 16 or 8 of them in): the block, its
// low nibbles and their shift, its high bits and their shift.
fn q6k_at(row: u32, g: u32, part: u32) -> vec4<u32> {
    let b = row + (g >> 3u) * Q6K_BYTES;
    let s = g & 7u;
    return vec4<u32>(
        b,
        b + (s >> 2u) * 64u + (s & 1u) * 32u + part,
        b + 128u + (s >> 2u) * 32u + part,
        s,
    );
}

// Values 16h to 16h + 15 of the Q6_K row at byte `row`.
fn half_q6k(row: u32, h: u32) -> Half {
    let at = q6k_at(row, h >> 1u, (h & 1u) * 16u);
    let s = at.w;
    let q = q6_bytes(words_at(at.y) >> vec4<u32>(((s >> 1u) & 1u) * 4u), words_at(at.z) >> vec4<u32>((s & 3u) * 2u));
    // Each byte less 32 as a signed byte: q + 96 is 96 to 159, its top bit
    // flipped q − 32's two's complement.
    let signed = (q + vec4<u32>(0x60606060u)) ^ vec4<u32>(0x80808080u);
    return Half(signed, q6_scale(at.x, 2u * s + (h & 1u)), 0.0);
}

// Values 8e to 8e + 7 of the Q6_K row at byte `row`.
fn eight_q6k(row: u32, e: u32) -> Eight {
    let at = q6k_at(row, e >> 2u, (e & 3u) * 8u);
    let s = at.w;
    let q = q6_bytes(words_at(at.y) >> vec4<u32>(((s >> 1u) & 1u) * 4u), words_at(at.z) >> vec4<u32>((s & 3u) * 2u));
    let d = q6_scale(at.x, 2u * s + ((e & 3u) >> 1u));
    return Eight(d * (bytes_f32(q.x) - 32.0), d * (bytes_f32(q.y) - 32.0));
}

// Values 8e to 8e + 7 of the BF16 row at byte `row`.
fn eight_bf16(row: u32, e: u32) -> Eight {
    let at = (row >> 2u) + e * 4u;
    let w = vec4<u32>(pair(at), pair(at + 2u));
    let even = bitcast<vec4<f32>>(w << vec4<u32>(16u));
    let odd = bitcast<vec4<f32>>(w & vec4<u32>(0xFFFF0000u));
    return Eight(vec4<f32>(even.x, odd.x, even.y, odd.y), vec4<f32>(even.z, odd.z, even.w, odd.w));
}

// Values 8e to 8e + 7 of the F32 row at byte `row`.
fn eight_f32(row: u32, e: u32) -> Eight {
    let at = (row >> 2u) + e * 8u;
    let a = vec4<u32>(pair(at), pair(at + 2u));
    let b = vec4<u32>(pair(at + 4u), pair(at + 6u));
    return Eight(bitcast<vec4<f32>>(a), bitcast<vec4<f32>>(b));
}

// Value `col` of the row at byte `row` of a matrix of `kind`.
fn value(kind: u32, row: u32, col: u32) -> f32 {
    if (kind == Q5K) {
        return value_q5k(row, col);
    }
    var e: Eight;
    switch kind {
        case Q4K: {
            e = eight_q4k(row, col >> 3u);
        }
        case Q6K: {
            e = eight_q6k(row, col >> 3u);
        }
        case BF16: {
            e = eight_bf16(row, col >> 3u);
        }
        default: {
            e = eight_f32(row, col >> 3u);
        }
    }
    let v = select(e.a, e.b, (col & 4u) != 0u);
    return v[col & 3u];
}
