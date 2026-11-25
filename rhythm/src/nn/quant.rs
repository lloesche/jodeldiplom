//! Weights at fewer bits, read back: three block layouts of the GGML file
//! format, and half precision.
//!
//! - Q4_K, about 4.5 bits a value. A row of a matrix (its length a
//!   multiple of [`BLOCK`]) is blocks of 256 values, each 144 bytes:
//!   `d` and `dmin`, IEEE halves (4 bytes); eight 6-bit scales and eight
//!   6-bit minimums, one each for the block's eight runs of 32, packed in
//!   12 bytes; 256 nibbles (128 bytes): for each 64 values, 32 bytes, the
//!   first 32 values in the low nibbles, the next 32 in the high. A value
//!   is `d·scale·q − dmin·min`, `q` its nibble.
//! - Q5_K, 5.5 bits a value: as Q4_K, with a fifth bit to each level, so
//!   `q` is 0 to 31. A block is 176 bytes: `d`, `dmin` and the scales as
//!   in Q4_K (16 bytes), then the fifth bits (32 bytes: bit `2c` of byte
//!   `l` for value `64c + l`, bit `2c + 1` for value `64c + 32 + l`), then
//!   the low nibbles as in Q4_K (128 bytes).
//! - Q8_0, 8.5 bits a value: a row (a multiple of [`Q8_BLOCK`]) is blocks
//!   of 32 values, each a half `d` and 32 signed bytes `q`, a value `d·q`.
//!
//! The layout is named here once (the byte ranges, [`fifth_bit`]) for
//! this reader and for the writer. Only reading is in the library; the
//! packing (fitting the blocks, by GPTQ for the built-in pitch weights) is
//! the tool's: `rhythm/tools/src/pack/` (the `pitch_pack` binary of
//! `jodel-rhythm-tools`).

use std::ops::Range;

/// Values a K block holds.
pub const BLOCK: usize = 256;
/// Bytes a Q4_K block takes.
pub const BLOCK_BYTES: usize = 144;
/// Bytes a Q5_K block takes.
pub const Q5_BLOCK_BYTES: usize = 176;
/// Values a Q8_0 block holds.
pub const Q8_BLOCK: usize = 32;
/// Bytes a Q8_0 block takes.
pub const Q8_BLOCK_BYTES: usize = 34;
/// Values a run (one scale and minimum) holds.
pub const RUN: usize = 32;

/// A K block's `d`, a half.
pub const D: Range<usize> = 0..2;
/// A K block's `dmin`, a half.
pub const DMIN: Range<usize> = 2..4;
/// A K block's eight 6-bit scales and eight 6-bit minimums: run `j < 4`'s
/// in the low six bits of bytes `j` and `j + 4`; run `j ≥ 4`'s low four
/// bits in byte `j + 4` (the scale's low, the minimum's high nibble), their
/// top two in the top bits of bytes `j − 4` and `j`.
pub const SCALES: Range<usize> = 4..16;
/// A Q5_K block's fifth bits ([`fifth_bit`]).
pub const FIFTH_BITS: Range<usize> = 16..48;
/// A Q8_0 block's `d`, a half; its 32 signed levels follow.
pub const Q8_D: Range<usize> = 0..2;

/// The bits of a level of a K block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    /// Q4_K.
    Four,
    /// Q5_K.
    Five,
}

impl Bits {
    /// Bytes a block takes.
    pub fn block_bytes(self) -> usize {
        match self {
            Bits::Four => BLOCK_BYTES,
            Bits::Five => Q5_BLOCK_BYTES,
        }
    }

    /// The highest level.
    pub fn top(self) -> u8 {
        match self {
            Bits::Four => 15,
            Bits::Five => 31,
        }
    }

    /// A block's low nibbles (two values a byte: for each 64 values, 32
    /// bytes, the first 32 values in the low nibbles, the next 32 in the
    /// high).
    pub fn nibbles(self) -> Range<usize> {
        match self {
            Bits::Four => SCALES.end..BLOCK_BYTES,
            Bits::Five => FIFTH_BITS.end..Q5_BLOCK_BYTES,
        }
    }
}

/// Where value `i`'s fifth bit lies in [`FIFTH_BITS`]: the byte and the
/// bit in it.
pub fn fifth_bit(i: usize) -> (usize, u32) {
    let (c, half) = (i / 64, (i % 64) / 32);
    (i % 32, (2 * c + half) as u32)
}

/// A little-endian half at `at` of `bytes`.
fn half_at(bytes: &[u8], at: Range<usize>) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at.start], bytes[at.start + 1]]))
}

/// Half precision to single.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// The scale and minimum of run `j` from the 12 packed bytes ([`SCALES`]).
fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One block's 256 values from its bytes, its levels of `bits`.
fn dequantize_block(block: &[u8], bits: Bits, out: &mut [f32]) {
    let d = half_at(block, D);
    let dmin = half_at(block, DMIN);
    let packed = &block[SCALES];
    let qh = (bits == Bits::Five).then(|| &block[FIFTH_BITS]);
    let qs = &block[bits.nibbles()];
    let high = |i: usize| -> u8 {
        qh.map_or(0, |h| {
            let (at, bit) = fifth_bit(i);
            ((h[at] >> bit) & 1) << 4
        })
    };
    for (c, chunk) in qs.chunks_exact(32).enumerate() {
        let (s1, m1) = scale_min(2 * c, packed);
        let (s2, m2) = scale_min(2 * c + 1, packed);
        let (d1, b1) = (d * f32::from(s1), dmin * f32::from(m1));
        let (d2, b2) = (d * f32::from(s2), dmin * f32::from(m2));
        for (l, &q) in chunk.iter().enumerate() {
            let (i, k) = (c * 64 + l, c * 64 + 32 + l);
            out[i] = d1 * f32::from((q & 0xf) | high(i)) - b1;
            out[k] = d2 * f32::from((q >> 4) | high(k)) - b2;
        }
    }
}

/// The values of packed K blocks of `bits`.
pub fn dequantize(packed: &[u8], bits: Bits) -> Vec<f32> {
    let bytes = bits.block_bytes();
    let mut out = vec![0.0f32; packed.len() / bytes * BLOCK];
    for (block, o) in packed.chunks_exact(bytes).zip(out.chunks_exact_mut(BLOCK)) {
        dequantize_block(block, bits, o);
    }
    out
}

/// The values of packed Q8_0 blocks.
pub fn dequantize_q8(packed: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() / Q8_BLOCK_BYTES * Q8_BLOCK);
    for block in packed.chunks_exact(Q8_BLOCK_BYTES) {
        let d = half_at(block, Q8_D);
        out.extend(block[Q8_D.end..].iter().map(|&q| d * f32::from(q as i8)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    /// A block written by hand: `d` 1, `dmin` ½; run `j`'s scale `j + 1`
    /// and minimum `2j + 1` (the last four runs' sixth bits too); value
    /// `i`'s low nibble `i % 16`, and its fifth bit (Q5_K) where `i` is odd.
    fn block(bits: Bits) -> Vec<u8> {
        let mut b = vec![0u8; bits.block_bytes()];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        for j in 0..8u8 {
            let (s, m) = (j + 1 + if j >= 4 { 32 } else { 0 }, 2 * j + 1);
            let j = usize::from(j);
            if j < 4 {
                (b[4 + j], b[8 + j]) = (s, m);
            } else {
                b[8 + j] = (s & 0xf) | ((m & 0xf) << 4);
                b[j] |= (s >> 4) << 6;
                b[4 + j] |= (m >> 4) << 6;
            }
        }
        let qs = if bits == Bits::Five {
            // Byte `l` holds the fifth bits of the values `l` past each 32.
            for (l, h) in b[16..48].iter_mut().enumerate() {
                *h = if l % 2 == 1 { 0xff } else { 0 };
            }
            48
        } else {
            16
        };
        for (l, q) in b[qs..].iter_mut().enumerate() {
            *q = (l % 16) as u8 | (((l + 32) % 16) as u8) << 4;
        }
        b
    }

    #[test]
    fn k_blocks_read_as_written() {
        for bits in [Bits::Four, Bits::Five] {
            let out = dequantize(&block(bits), bits);
            assert_eq!(out.len(), BLOCK);
            for (i, &v) in out.iter().enumerate() {
                let j = i / RUN;
                let scale = (j + 1 + if j >= 4 { 32 } else { 0 }) as f32;
                let min = (2 * j + 1) as f32;
                let fifth = if bits == Bits::Five && i % 2 == 1 {
                    16
                } else {
                    0
                };
                let q = (i % 16 + fifth) as f32;
                assert_eq!(v, scale * q - 0.5 * min, "{bits:?} value {i}");
            }
        }
    }

    #[test]
    fn q8_blocks_read_as_written() {
        let mut b = 0x3800u16.to_le_bytes().to_vec();
        b.extend((0..32).map(|q: i8| (q - 16) as u8));
        let out = dequantize_q8(&b);
        let want: Vec<f32> = (0..32).map(|q| 0.5 * (q - 16) as f32).collect();
        assert_eq!(out, want);
    }
}
