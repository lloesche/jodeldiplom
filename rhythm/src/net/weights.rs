//! The weights file: named tensors (see `tools/convert_beat_this.py`).
//!
//! Two layouts:
//!
//! - `BTW1`: the element width (u32: 2 halves, 4 singles), the count of
//!   tensors (u32), then each: its name (u16 length, UTF-8), its rank (u8)
//!   and dimensions (u32 each), its elements.
//! - `BTQ1`, packed: the count of tensors (u32), then each: its name, rank
//!   and dimensions as above, its kind (u8: [`Kind`]), the length of its
//!   data in bytes (u32), the data. A [`Kind::Q4k`] or [`Kind::Q5k`]
//!   tensor's rows (the first dimension; the rest make a row) are whole
//!   blocks of 256 values, a [`Kind::Q8`] tensor's whole blocks of 32 (see
//!   [`super::q4k`]).
//!
//! Little-endian throughout. Every tensor is read into single precision.

use std::collections::HashMap;

use super::q4k::{self, Bits, Fit};

/// Named tensors, in single precision.
pub struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

/// How a tensor of a packed file is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Single precision.
    F32 = 0,
    /// Half precision.
    F16 = 1,
    /// Q4_K blocks ([`super::q4k`]).
    Q4k = 2,
    /// Q5_K blocks ([`super::q4k`]).
    Q5k = 3,
    /// Q8_0 blocks ([`super::q4k`]).
    Q8 = 4,
}

/// Half precision to single.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
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

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or("the weights file ends early")?;
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

impl Weights {
    /// Reads a weights file, either layout.
    pub fn parse(bytes: &[u8]) -> Result<Weights, String> {
        let mut c = Cursor { bytes, at: 0 };
        match c.take(4)? {
            b"BTW1" => {}
            b"BTQ1" => return Weights::parse_packed(c),
            _ => return Err("not a weights file".into()),
        }
        let width = c.u32()? as usize;
        if width != 2 && width != 4 {
            return Err(format!("elements of {width} bytes"));
        }
        let count = c.u32()?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let len = usize::from(c.u16()?);
            let name = std::str::from_utf8(c.take(len)?)
                .map_err(|_| "a tensor name is not UTF-8")?
                .to_owned();
            let rank = usize::from(c.take(1)?[0]);
            let shape = (0..rank)
                .map(|_| c.u32().map(|d| d as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let n: usize = shape.iter().product();
            let raw = c.take(n * width)?;
            let data = if width == 2 {
                raw.chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect()
            } else {
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            };
            tensors.insert(name, (shape, data));
        }
        if c.at != bytes.len() {
            return Err("the weights file has trailing bytes".into());
        }
        Ok(Weights { tensors })
    }

    fn parse_packed(mut c: Cursor) -> Result<Weights, String> {
        let count = c.u32()?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let len = usize::from(c.u16()?);
            let name = std::str::from_utf8(c.take(len)?)
                .map_err(|_| "a tensor name is not UTF-8")?
                .to_owned();
            let rank = usize::from(c.take(1)?[0]);
            let shape = (0..rank)
                .map(|_| c.u32().map(|d| d as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let n: usize = shape.iter().product();
            let kind = c.take(1)?[0];
            let bytes = c.u32()? as usize;
            let raw = c.take(bytes)?;
            let data: Vec<f32> = match kind {
                0 => raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                1 => raw
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                    .collect(),
                2 | 3 => {
                    let bits = if kind == 2 { Bits::Four } else { Bits::Five };
                    if raw.len() % bits.block_bytes() != 0 {
                        return Err(format!("{name}: not whole blocks"));
                    }
                    q4k::dequantize(raw, bits)
                }
                4 => {
                    if raw.len() % q4k::Q8_BLOCK_BYTES != 0 {
                        return Err(format!("{name}: not whole blocks"));
                    }
                    q4k::dequantize_q8(raw)
                }
                k => return Err(format!("{name}: unknown kind {k}")),
            };
            if data.len() != n {
                return Err(format!("{name}: {} values for {shape:?}", data.len()));
            }
            tensors.insert(name, (shape, data));
        }
        if c.at != c.bytes.len() {
            return Err("the weights file has trailing bytes".into());
        }
        Ok(Weights { tensors })
    }

    /// A packed file (`BTQ1`) of `tensors`: each its name, shape, and its
    /// values stored as `kind`, K blocks fitted by `importance` (one a
    /// column of its rows) where given, refined where asked, and by GPTQ
    /// on `gram` where given.
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
            let data: Vec<u8> = match t.kind {
                Kind::F32 => t.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                Kind::F16 => t
                    .values
                    .iter()
                    .flat_map(|&v| q4k::f32_to_f16(v).to_le_bytes())
                    .collect(),
                Kind::Q4k | Kind::Q5k => {
                    let bits = if t.kind == Kind::Q4k {
                        Bits::Four
                    } else {
                        Bits::Five
                    };
                    let cols = n / t.shape.first().copied().unwrap_or(1).max(1);
                    if cols % q4k::BLOCK != 0 {
                        return Err(format!("{}: rows of {cols} are not whole blocks", t.name));
                    }
                    if t.importance.as_ref().is_some_and(|i| i.len() != cols) {
                        return Err(format!("{}: not an importance a column", t.name));
                    }
                    let fit = Fit {
                        importance: t.importance.as_deref(),
                        refine: t.refine,
                    };
                    match &t.gram {
                        Some(g) => q4k::quantize_gptq(&t.values, cols, bits, g, fit)
                            .map_err(|e| format!("{}: {e}", t.name))?,
                        None => q4k::quantize(&t.values, cols, bits, fit),
                    }
                }
                Kind::Q8 => {
                    let cols = n / t.shape.first().copied().unwrap_or(1).max(1);
                    if cols % q4k::Q8_BLOCK != 0 {
                        return Err(format!("{}: rows of {cols} are not whole blocks", t.name));
                    }
                    q4k::quantize_q8(&t.values)
                }
            };
            out.push(t.kind as u8);
            out.extend((data.len() as u32).to_le_bytes());
            out.extend(data);
        }
        Ok(out)
    }

    /// Every tensor's name and shape, by name.
    pub fn shapes(&self) -> Vec<(&str, &[usize])> {
        let mut all: Vec<(&str, &[usize])> = self
            .tensors
            .iter()
            .map(|(k, (s, _))| (k.as_str(), s.as_slice()))
            .collect();
        all.sort_unstable();
        all
    }

    /// Tensor `name`'s values, left in place.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|(_, v)| v.as_slice())
    }

    /// Takes out tensor `name`, which must have `shape`.
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (have, data) = self
            .tensors
            .remove(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        if have != shape {
            return Err(format!("{name} is {have:?}, not {shape:?}"));
        }
        Ok(data)
    }

    /// The names not taken yet.
    pub fn left(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

/// A tensor to pack ([`Weights::pack`]).
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
    /// (`[cols][cols]`), to pack by GPTQ ([`q4k::quantize_gptq`]).
    pub gram: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_files_read_back() {
        let values: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let packed = Weights::pack(&[
            PackedTensor {
                name: "a".into(),
                shape: vec![2, 256],
                values: values.clone(),
                kind: Kind::Q4k,
                importance: None,
                refine: false,
                gram: None,
            },
            PackedTensor {
                name: "b".into(),
                shape: vec![3],
                values: vec![1.0, -2.5, 0.125],
                kind: Kind::F16,
                importance: None,
                refine: false,
                gram: None,
            },
            PackedTensor {
                name: "c".into(),
                shape: vec![1],
                values: vec![0.1],
                kind: Kind::F32,
                importance: None,
                refine: false,
                gram: None,
            },
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
        let tensor = |kind, importance: Option<Vec<f32>>, shape: Vec<usize>| PackedTensor {
            name: "a".into(),
            shape,
            values: values.clone(),
            kind,
            importance,
            refine: true,
            gram: None,
        };
        let worst = |kind, importance| {
            let packed = Weights::pack(&[tensor(kind, importance, vec![3, 256])]).unwrap();
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
        assert!(Weights::pack(&[tensor(Kind::Q5k, Some(vec![1.0; 255]), vec![3, 256])]).is_err());
        assert!(Weights::pack(&[tensor(Kind::Q8, None, vec![24, 32])]).is_ok());
        assert!(Weights::pack(&[tensor(Kind::Q8, None, vec![32, 24])]).is_err());
    }

    #[test]
    fn half_precision_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }
}
