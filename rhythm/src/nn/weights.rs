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
//!   [`super::quant`]). The `pitch_pack` tool (`rhythm/tools`) writes
//!   them.
//!
//! Little-endian throughout. Every tensor is read into single precision.

use std::collections::HashMap;

use super::quant::{self, f16_to_f32, Bits};

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
    /// Q4_K blocks ([`super::quant`]).
    Q4k = 2,
    /// Q5_K blocks ([`super::quant`]).
    Q5k = 3,
    /// Q8_0 blocks ([`super::quant`]).
    Q8 = 4,
}

impl Kind {
    /// The kind stored as `byte`.
    fn of(byte: u8) -> Option<Kind> {
        [Kind::F32, Kind::F16, Kind::Q4k, Kind::Q5k, Kind::Q8]
            .into_iter()
            .find(|&k| k as u8 == byte)
    }

    /// The bits of a K block kind's levels; `None` for the others.
    pub fn bits(self) -> Option<Bits> {
        match self {
            Kind::Q4k => Some(Bits::Four),
            Kind::Q5k => Some(Bits::Five),
            Kind::F32 | Kind::F16 | Kind::Q8 => None,
        }
    }
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

    /// A tensor's name and shape.
    fn head(&mut self) -> Result<(String, Vec<usize>), String> {
        let len = usize::from(self.u16()?);
        let name = std::str::from_utf8(self.take(len)?)
            .map_err(|_| "a tensor name is not UTF-8")?
            .to_owned();
        let rank = usize::from(self.take(1)?[0]);
        let shape = (0..rank)
            .map(|_| self.u32().map(|d| d as usize))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((name, shape))
    }

    /// The end of the file.
    fn end(&self) -> Result<(), String> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err("the weights file has trailing bytes".into())
        }
    }
}

fn singles(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn halves(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
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
            let (name, shape) = c.head()?;
            let n: usize = shape.iter().product();
            let raw = c.take(n * width)?;
            let data = if width == 2 {
                halves(raw)
            } else {
                singles(raw)
            };
            tensors.insert(name, (shape, data));
        }
        c.end()?;
        Ok(Weights { tensors })
    }

    fn parse_packed(mut c: Cursor) -> Result<Weights, String> {
        let count = c.u32()?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let (name, shape) = c.head()?;
            let n: usize = shape.iter().product();
            let byte = c.take(1)?[0];
            let kind = Kind::of(byte).ok_or_else(|| format!("{name}: unknown kind {byte}"))?;
            let bytes = c.u32()? as usize;
            let raw = c.take(bytes)?;
            let whole = |block: usize| {
                if raw.len() % block == 0 {
                    Ok(())
                } else {
                    Err(format!("{name}: not whole blocks"))
                }
            };
            let data = if let Some(bits) = kind.bits() {
                whole(bits.block_bytes())?;
                quant::dequantize(raw, bits)
            } else {
                match kind {
                    Kind::F32 => singles(raw),
                    Kind::F16 => halves(raw),
                    Kind::Q8 => {
                        whole(quant::Q8_BLOCK_BYTES)?;
                        quant::dequantize_q8(raw)
                    }
                    Kind::Q4k | Kind::Q5k => unreachable!("K blocks have bits"),
                }
            };
            if data.len() != n {
                return Err(format!("{name}: {} values for {shape:?}", data.len()));
            }
            tensors.insert(name, (shape, data));
        }
        c.end()?;
        Ok(Weights { tensors })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A packed file of `tensors` (name, shape, kind byte, data).
    fn packed(tensors: &[(&str, &[u32], u8, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"BTQ1".to_vec();
        out.extend((tensors.len() as u32).to_le_bytes());
        for (name, shape, kind, data) in tensors {
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(name.as_bytes());
            out.push(shape.len() as u8);
            out.extend(shape.iter().flat_map(|d| d.to_le_bytes()));
            out.push(*kind);
            out.extend((data.len() as u32).to_le_bytes());
            out.extend(data);
        }
        out
    }

    /// A Q8_0 block: `d` ½, the levels -16 to 15.
    fn q8_block() -> Vec<u8> {
        let mut b = 0x3800u16.to_le_bytes().to_vec();
        b.extend((0..32).map(|q: i8| (q - 16) as u8));
        b
    }

    #[test]
    fn packed_files_read_back() {
        let file = packed(&[
            (
                "a",
                &[3],
                0,
                [1.0f32, -2.5, 0.125]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            (
                "b",
                &[2],
                1,
                [0x3c00u16, 0xc000]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            ("c", &[1, 32], 4, q8_block()),
        ]);
        let mut w = Weights::parse(&file).unwrap();
        assert_eq!(
            w.shapes(),
            vec![("a", &[3][..]), ("b", &[2]), ("c", &[1, 32])]
        );
        assert_eq!(w.take("a", &[3]).unwrap(), vec![1.0, -2.5, 0.125]);
        assert_eq!(w.take("b", &[2]).unwrap(), vec![1.0, -2.0]);
        assert!(w.take("c", &[32]).is_err());
        assert!(w.left().is_empty());
    }

    #[test]
    fn broken_files_are_refused() {
        let q8 = |shape: &'static [u32], data: Vec<u8>| packed(&[("c", shape, 4, data)]);
        let mut cut = q8_block();
        cut.pop();
        for (file, why) in [
            (q8(&[1, 32], cut), "not whole blocks"),
            (q8(&[2, 32], q8_block()), "32 values for [2, 32]"),
            (packed(&[("c", &[1], 9, vec![0; 4])]), "unknown kind 9"),
            (
                [q8(&[1, 32], q8_block()), vec![0]].concat(),
                "trailing bytes",
            ),
            (q8(&[1, 32], q8_block())[..40].to_vec(), "ends early"),
            (b"BTX1".to_vec(), "not a weights file"),
        ] {
            let err = Weights::parse(&file).err().unwrap();
            assert!(err.contains(why), "{err}");
        }
    }
}
