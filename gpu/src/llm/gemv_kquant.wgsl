// A row through a Q4_K or a Q6_K matrix on the plain path (the decoder's
// steps on Metal), shaped as llama.cpp's Metal kernels shape theirs
// (`kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_q6_K_f32`): a workgroup takes
// 16 rows, eight slots of 32 invocations (a SIMD group each where
// subgroups are 32 wide) two rows each. Each invocation holds a run of a
// block's inputs (Q4_K: 32, four runs of 8; Q6_K: 16, four runs of 4)
// and takes it through both rows, reading the weights a word at a time,
// decoding the run's scales once and masking the 4-bit and 6-bit values
// where they lie in their words (no shifts: a value masked in byte j of
// its word is 256^j times itself, and the products are summed back by
// those places, `PLACES`; exact, as the places are powers of two). A
// slot's invocations take a Q4_K row's blocks four at a time (eight
// invocations a block), a Q6_K row's two at a time (16 a block); the
// slot's sums are added at the end (`gemv_sums_f32`).

// The low and the high nibbles of a word's four bytes; its bytes; what a
// value masked in each byte is scaled back by.
const NIBBLES: vec4<u32> = vec4<u32>(0xFu, 0xF00u, 0xF0000u, 0xF000000u);
const HIGH_NIBBLES: vec4<u32> = vec4<u32>(0xF0u, 0xF000u, 0xF00000u, 0xF0000000u);
const BYTES: vec4<u32> = vec4<u32>(0xFFu, 0xFF00u, 0xFF0000u, 0xFF000000u);
const PLACES: vec4<f32> = vec4<f32>(1.0, 1.0 / 256.0, 1.0 / 65536.0, 1.0 / 16777216.0);
// Q6_K's offset (32) in each byte's place.
const SIX_OFFSET: vec4<f32> = vec4<f32>(32.0, 8192.0, 2097152.0, 536870912.0);

// A Q4_K run's 32 inputs: 8 each of the block's 32s 2iq, 2iq + 1, 2iq + 4
// and 2iq + 5 (from the run's place, 8ir), and each 8's sum (for the
// minimums).
struct Run32 {
    a0: vec4<f32>,
    a1: vec4<f32>,
    b0: vec4<f32>,
    b1: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    d0: vec4<f32>,
    d1: vec4<f32>,
    sums: vec4<f32>,
}

// The Q4_K run's inputs from word `a` of `ints`.
fn q4k_inputs(a: u32) -> Run32 {
    let a0 = ints4(a);
    let a1 = ints4(a + 4u);
    let b0 = ints4(a + 32u);
    let b1 = ints4(a + 36u);
    let c0 = ints4(a + 128u);
    let c1 = ints4(a + 132u);
    let d0 = ints4(a + 160u);
    let d1 = ints4(a + 164u);
    let one = vec4<f32>(1.0);
    let sums = vec4<f32>(dot(a0 + a1, one), dot(b0 + b1, one), dot(c0 + c1, one), dot(d0 + d1, one));
    return Run32(a0, a1, b0, b1, c0, c1, d0, d1, sums);
}

// Eight values masked in the words `q` (by `mask`, in their places) times
// the inputs `lo` (the first word's) and `hi`, summed back to their places.
fn masked8(q: vec2<u32>, mask: vec4<u32>, lo: vec4<f32>, hi: vec4<f32>) -> f32 {
    let p = vec4<f32>(vec4<u32>(q.x) & mask) * lo + vec4<f32>(vec4<u32>(q.y) & mask) * hi;
    return dot(p, PLACES);
}

// The run (`iq`, `ir`) of the Q4_K block at word `w` through its inputs
// `x`: its four 32s' scales and minimums (6 bits each, `k_scale_min`'s
// unpacking two bytes at a time), their 8 values each from the nibbles'
// 32-byte quarters iq (32s 2iq, 2iq + 1) and iq + 2 (2iq + 4, 2iq + 5),
// bytes 8ir to 8ir + 7.
fn q4k_run(w: u32, iq: u32, ir: u32, x: Run32) -> f32 {
    let head = vec4<u32>(pair(w), pair(w + 2u));
    let dm = unpack2x16float(head.x);
    let at = iq * 16u;
    let s0 = (head.y >> at) & 0xFFFFu;
    let s1 = (head.z >> at) & 0xFFFFu;
    let s2 = (head.w >> at) & 0xFFFFu;
    let sc_lo = s0 & 0x3F3Fu;
    let m_lo = s1 & 0x3F3Fu;
    let sc_hi = (s2 & 0x0F0Fu) | ((s0 & 0xC0C0u) >> 2u);
    let m_hi = ((s2 >> 4u) & 0x0F0Fu) | ((s1 & 0xC0C0u) >> 2u);
    let sc = vec4<f32>(vec4<u32>(sc_lo, sc_lo >> 8u, sc_hi, sc_hi >> 8u) & vec4<u32>(255u));
    let m = vec4<f32>(vec4<u32>(m_lo, m_lo >> 8u, m_hi, m_hi >> 8u) & vec4<u32>(255u));
    let q1 = pair(w + 4u + iq * 8u + ir * 2u);
    let q2 = pair(w + 20u + iq * 8u + ir * 2u);
    let sums = vec4<f32>(
        masked8(q1, NIBBLES, x.a0, x.a1),
        masked8(q1, HIGH_NIBBLES, x.b0, x.b1) / 16.0,
        masked8(q2, NIBBLES, x.c0, x.c1),
        masked8(q2, HIGH_NIBBLES, x.d0, x.d1) / 16.0,
    );
    return dm.x * dot(sc, sums) - dm.y * dot(m, x.sums);
}

@compute @workgroup_size(256)
fn gemv_f32_q4k(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let words = row_bytes_q4k(k) / 4u;
    let r0 = wid.x * GEMV_ROWS + (lid / 32u) * 2u;
    let lane = lid % 32u;
    let iq = (lane / 4u) & 1u;
    let ir = lane & 3u;
    let row0 = mm_w() + r0 * words;
    let row1 = row0 + words;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var b = lane / 8u; b < k / 256u; b += 4u) {
        let x = q4k_inputs(mm_a() + b * 256u + iq * 64u + ir * 8u);
        let w = b * (Q4K_BYTES / 4u);
        acc0 += q4k_run(row0 + w, iq, ir, x);
        acc1 += q4k_run(row1 + w, iq, ir, x);
    }
    gemv_finish(lid, wid.x, r0, gemv_sums_f32(lid, acc0, acc1));
}

// A Q6_K run's 16 inputs: 4 each from its place, and 32, 64 and 96 on.
struct Run16 {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
}

// Four 6-bit values (each in its byte of `v`) less 32, times `x`, summed.
fn q6_dot(v: u32, x: vec4<f32>) -> f32 {
    return dot((vec4<f32>(vec4<u32>(v) & BYTES) - SIX_OFFSET) * x, PLACES);
}

// The run (`ip`, `il`; its 16s' scales from `si`) of the Q6_K block at
// byte `b` through its inputs `x`: values 128ip + 4il + j (j = 0 to 3),
// and 32, 64 and 96 on, from the low nibbles' bytes 64ip + 4il (low and
// high nibbles) and 32 on, and the high bits' bytes 32ip + 4il (two bits
// each of the four).
fn q6k_run(b: u32, ip: u32, il: u32, si: u32, x: Run16) -> f32 {
    let ql = b + 64u * ip + 4u * il;
    let q1 = bytes4(ql);
    let q2 = bytes4(ql + 32u);
    let qh = bytes4(b + 128u + 32u * ip + 4u * il);
    let v0 = (q1 & LOW) | ((qh & 0x03030303u) << 4u);
    let v1 = (q2 & LOW) | ((qh & 0x0C0C0C0Cu) << 2u);
    let v2 = ((q1 >> 4u) & LOW) | (qh & 0x30303030u);
    let v3 = ((q2 >> 4u) & LOW) | ((qh & 0xC0C0C0C0u) >> 2u);
    // The signed scales si, si + 2, si + 4 and si + 6.
    let at = b + 192u + (si & 8u);
    let s = vec2<u32>(bytes4(at), bytes4(at + 4u)) >> vec2<u32>(8u * (si & 1u));
    let sc = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(s.x << 24u, s.x << 8u, s.y << 24u, s.y << 8u)) >> vec4<u32>(24u));
    let d = unpack2x16float(half_at(b + 208u)).x;
    let sums = vec4<f32>(q6_dot(v0, x.a), q6_dot(v1, x.b), q6_dot(v2, x.c), q6_dot(v3, x.d));
    return d * dot(sc, sums);
}

@compute @workgroup_size(256)
fn gemv_f32_q6k(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = mm_k();
    let rb = row_bytes_q6k(k);
    let r0 = wid.x * GEMV_ROWS + (lid / 32u) * 2u;
    let lane = lid % 32u;
    let t = lane / 2u;
    let ip = t / 8u;
    let il = t % 8u;
    let si = 8u * ip + il / 4u;
    let row0 = mm_w() * 4u + r0 * rb;
    let row1 = row0 + rb;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var b = lane & 1u; b < k / 256u; b += 2u) {
        let a = mm_a() + b * 256u + 128u * ip + 4u * il;
        let x = Run16(ints4(a), ints4(a + 32u), ints4(a + 64u), ints4(a + 96u));
        let at = b * Q6K_BYTES;
        acc0 += q6k_run(row0 + at, ip, il, si, x);
        acc1 += q6k_run(row1 + at, ip, il, si, x);
    }
    gemv_finish(lid, wid.x, r0, gemv_sums_f32(lid, acc0, acc1));
}
