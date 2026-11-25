//! Writing a packed weights file (`BTQ1`, which
//! `jodel_rhythm::nn::weights` reads): each tensor at full or half
//! precision, or in blocks of fewer bits fitted here (`fit`, `gptq`), in
//! the layout `jodel_rhythm::nn::quant` names.

mod fit;
mod gptq;

use jodel_rhythm::nn::quant::{f16_to_f32, BLOCK, Q8_BLOCK, Q8_BLOCK_BYTES, Q8_D};
use jodel_rhythm::nn::weights::Kind;

use fit::Fit;

/// A tensor to pack ([`pack`]).
pub struct PackedTensor {
    /// Its name.
    pub name: String,
    /// Its shape, rows first.
    pub shape: Vec<usize>,
    /// Its values, row-major.
    pub values: Vec<f32>,
    /// How to store it.
    pub kind: Kind,
    /// For K blocks: how much each column's error counts.
    pub importance: Option<Vec<f32>>,
    /// For K blocks: whether to refine the fitted scales.
    pub refine: bool,
    /// For K blocks: the mean product of each two columns' inputs
    /// (`[cols][cols]`), to pack by GPTQ (`gptq::quantize_gptq`).
    pub gram: Option<Vec<f32>>,
}

/// A packed file of `tensors`: each its name, shape, and its values
/// stored as its kind, K blocks fitted by its importance (one a column of
/// its rows) where given, refined where asked, and by GPTQ on its gram
/// where given.
pub fn pack(tensors: &[PackedTensor]) -> Result<Vec<u8>, String> {
    let mut out = b"BTQ1".to_vec();
    out.extend((tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let n: usize = t.shape.iter().product();
        if n != t.values.len() {
            return Err(format!(
                "{}: {} values for {:?}",
                t.name,
                t.values.len(),
                t.shape
            ));
        }
        out.extend((t.name.len() as u16).to_le_bytes());
        out.extend(t.name.as_bytes());
        out.push(t.shape.len() as u8);
        for &d in &t.shape {
            out.extend((d as u32).to_le_bytes());
        }
        let cols = n / t.shape.first().copied().unwrap_or(1).max(1);
        let whole = |block: usize| {
            if cols % block == 0 {
                Ok(())
            } else {
                Err(format!("{}: rows of {cols} are not whole blocks", t.name))
            }
        };
        let data: Vec<u8> = if let Some(bits) = t.kind.bits() {
            whole(BLOCK)?;
            if t.importance.as_ref().is_some_and(|i| i.len() != cols) {
                return Err(format!("{}: not an importance a column", t.name));
            }
            let fit = Fit {
                importance: t.importance.as_deref(),
                refine: t.refine,
            };
            match &t.gram {
                Some(g) => gptq::quantize_gptq(&t.values, cols, bits, g, fit)
                    .map_err(|e| format!("{}: {e}", t.name))?,
                None => fit::quantize(&t.values, cols, bits, fit),
            }
        } else {
            match t.kind {
                Kind::F32 => t.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Kind::F16 => t
                    .values
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect(),
                Kind::Q8 => {
                    whole(Q8_BLOCK)?;
                    quantize_q8(&t.values)
                }
                Kind::Q4k | Kind::Q5k => unreachable!("K blocks have bits"),
            }
        };
        out.push(t.kind as u8);
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
    }
    Ok(out)
}

/// Single precision to half, rounding to the nearest (ties to even).
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): the mantissa with its leading one, shifted.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1; // may carry into the exponent, as it should
    }
    sign | v as u16
}

/// `values` (a multiple of [`Q8_BLOCK`]) packed as Q8_0 blocks: each
/// block's step its largest magnitude over 127.
fn quantize_q8(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % Q8_BLOCK, 0, "not whole Q8_0 blocks");
    let mut out = vec![0u8; values.len() / Q8_BLOCK * Q8_BLOCK_BYTES];
    for (x, block) in values
        .chunks_exact(Q8_BLOCK)
        .zip(out.chunks_exact_mut(Q8_BLOCK_BYTES))
    {
        let top = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = f32_to_f16(top / 127.0);
        let step = f16_to_f32(d);
        block[Q8_D].copy_from_slice(&d.to_le_bytes());
        for (&v, q) in x.iter().zip(&mut block[Q8_D.end..]) {
            let level = if step > 0.0 {
                (v / step).round_ties_even().clamp(-127.0, 127.0)
            } else {
                0.0
            };
            *q = level as i8 as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_rhythm::nn::quant::dequantize_q8;
    use jodel_rhythm::nn::weights::Weights;

    #[test]
    fn halves_round_to_nearest() {
        for x in [
            0.0f32, 1.0, -2.0, 0.333_333, 65504.0, 6.0e-8, 1.0e-5, -2.375,
        ] {
            let back = f16_to_f32(f32_to_f16(x));
            assert!(
                (back - x).abs() <= x.abs() / 1024.0 + 6.0e-8,
                "{x} → {back}"
            );
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0e6), 0x7c00);
    }

    /// Q8_0 keeps each value within half a step of its block's.
    #[test]
    fn eight_bit_blocks() {
        let x: Vec<f32> = (0..BLOCK * 4)
            .map(|i| (i as f32 * 0.37).sin() * if i < 512 { 0.05 } else { 0.2 })
            .collect();
        let packed = quantize_q8(&x);
        assert_eq!(packed.len(), x.len() / Q8_BLOCK * Q8_BLOCK_BYTES);
        let back = dequantize_q8(&packed);
        for (a, b) in x.chunks(Q8_BLOCK).zip(back.chunks(Q8_BLOCK)) {
            let step = a.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            for (p, q) in a.iter().zip(b) {
                assert!((p - q).abs() <= step * 0.51 + 1e-6, "{p} → {q}");
            }
        }
        assert!(dequantize_q8(&quantize_q8(&[0.0; 32]))
            .iter()
            .all(|&v| v == 0.0));
    }

    fn tensor(name: &str, shape: Vec<usize>, values: Vec<f32>, kind: Kind) -> PackedTensor {
        PackedTensor {
            name: name.into(),
            shape,
            values,
            kind,
            importance: None,
            refine: false,
            gram: None,
        }
    }

    #[test]
    fn packed_files_read_back() {
        let values: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let packed = pack(&[
            tensor("a", vec![2, 256], values.clone(), Kind::Q4k),
            tensor("b", vec![3], vec![1.0, -2.5, 0.125], Kind::F16),
            tensor("c", vec![1], vec![0.1], Kind::F32),
        ])
        .unwrap();
        let mut w = Weights::parse(&packed).unwrap();
        let a = w.take("a", &[2, 256]).unwrap();
        // Half a step of 15 over a run's range of 2, and a little.
        assert!(a.iter().zip(&values).all(|(x, y)| (x - y).abs() < 0.08));
        assert_eq!(w.take("b", &[3]).unwrap(), vec![1.0, -2.5, 0.125]);
        assert_eq!(w.take("c", &[1]).unwrap(), vec![0.1]);
        assert!(w.left().is_empty());
    }

    /// Q5_K (with importance, refined) and Q8_0 tensors read back closer
    /// than Q4_K; a wrong importance or partial blocks are refused.
    #[test]
    fn finer_kinds_read_back() {
        let values: Vec<f32> = (0..768).map(|i| (i as f32 * 0.1).sin()).collect();
        let t = |kind, importance: Option<Vec<f32>>, shape: Vec<usize>| PackedTensor {
            importance,
            refine: true,
            ..tensor("a", shape, values.clone(), kind)
        };
        let worst = |kind, importance| {
            let packed = pack(&[t(kind, importance, vec![3, 256])]).unwrap();
            let a = Weights::parse(&packed)
                .unwrap()
                .take("a", &[3, 256])
                .unwrap();
            a.iter()
                .zip(&values)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        let imp: Vec<f32> = (0..256).map(|i| 0.5 + (i % 3) as f32).collect();
        let (four, five, weighed, eight) = (
            worst(Kind::Q4k, None),
            worst(Kind::Q5k, None),
            worst(Kind::Q5k, Some(imp.clone())),
            worst(Kind::Q8, None),
        );
        assert!(
            four < 0.08 && five < 0.04 && weighed < 0.08 && eight < 0.005,
            "{four} {five} {weighed} {eight}"
        );
        assert!(pack(&[t(Kind::Q5k, Some(vec![1.0; 255]), vec![3, 256])]).is_err());
        assert!(pack(&[t(Kind::Q8, None, vec![24, 32])]).is_ok());
        assert!(pack(&[t(Kind::Q8, None, vec![32, 24])]).is_err());
    }
}
