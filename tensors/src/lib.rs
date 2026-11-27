//! The networks' numeric base, depending on nothing: a `.safetensors`
//! file's tensors as f32 or as stored ([`SafeTensors`]), the checkpoints
//! the speech models (`jodel-lyrics`) and the story scenes' depth and matte
//! models and the picture model (`jodel-gpu`) are read from, as published
//! (a checkpoint in several files through its index: [`Shards`]; either
//! read by name: [`Tensors`]); a GGUF file's metadata and tensors as it
//! holds them ([`gguf`], the story's language model's); byte-pair encoding
//! over a vocabulary's ids ([`bpe`], the story's tokenisers'); half floats
//! ([`half`]); a small JSON reader ([`json`]); and what the CPU kernels
//! share ([`kernels`], [`fast!`]).
//!
//! A `.safetensors` file is an 8-byte length, a JSON header naming each
//! tensor's type, shape and place, and the data.

pub mod bpe;
pub mod gguf;
pub mod half;
pub mod json;
pub mod kernels;

use std::collections::HashMap;
use std::path::Path;

use half::{bf16_to_f32, f16_to_f32};
use json::Json;

/// How a tensor's values are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    F16,
    /// FP8 with 4 bits of exponent and 3 of mantissa (`F8_E4M3`).
    F8E4M3,
    /// Any other type, by its name. The file still opens and such a tensor
    /// fails only when it is read ([`SafeTensors::get`]): published
    /// checkpoints carry integer buffers (position ids, step counters)
    /// beside the weights, and a reader takes only the weights it names.
    Other(String),
}

impl DType {
    fn of(name: &str) -> DType {
        match name {
            "F32" => DType::F32,
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            "F8_E4M3" => DType::F8E4M3,
            other => DType::Other(other.to_owned()),
        }
    }
}

/// Tensors read by name: a checkpoint's file ([`SafeTensors`]), or its
/// files together ([`Shards`]).
pub trait Tensors {
    /// Tensor `name` as stored: its type, its shape and its bytes.
    fn raw(&self, name: &str) -> Result<(&DType, &[usize], &[u8]), String>;
    /// Tensor `name` with `shape`, as f32 (stored as f32, bf16 or f16).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String>;
}

/// A checkpoint in several files with an index
/// (`model.safetensors.index.json`: each tensor's file), those of its
/// files that hold the tensors asked for opened.
pub struct Shards {
    files: Vec<SafeTensors>,
}

impl Shards {
    /// The files of the checkpoint in `dir` (its index the file `index`)
    /// that hold `names`; an error where the index puts one in a file not
    /// among `files` (the files the caller knows the checkpoint by).
    pub fn open(
        dir: &Path,
        index: &str,
        names: &[String],
        files: &[&str],
    ) -> Result<Shards, String> {
        let at = dir.join(index);
        let text = std::fs::read_to_string(&at).map_err(|e| format!("{}: {e}", at.display()))?;
        let map = Json::parse(&text)
            .and_then(|j| j.get("weight_map").cloned())
            .ok_or("no weight map in the index")?;
        let mut held: Vec<&str> = names
            .iter()
            .map(|n| {
                let file = map
                    .get(n)
                    .and_then(Json::as_str)
                    .ok_or_else(|| format!("{n} is in no file of the index"))?;
                files
                    .iter()
                    .copied()
                    .find(|&f| f == file)
                    .ok_or_else(|| format!("{n} is in {file}, not a file of the checkpoint"))
            })
            .collect::<Result<_, String>>()?;
        held.sort_unstable();
        held.dedup();
        Ok(Shards {
            files: held
                .iter()
                .map(|f| SafeTensors::open(&dir.join(f)))
                .collect::<Result<_, _>>()?,
        })
    }

    /// The file that holds tensor `name`.
    fn file(&self, name: &str) -> Result<&SafeTensors, String> {
        self.files
            .iter()
            .find(|f| f.has(name))
            .ok_or_else(|| format!("no tensor {name}"))
    }
}

impl Tensors for Shards {
    fn raw(&self, name: &str) -> Result<(&DType, &[usize], &[u8]), String> {
        self.file(name)?.raw(name)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        self.file(name)?.get(name, shape)
    }
}

impl Tensors for SafeTensors {
    fn raw(&self, name: &str) -> Result<(&DType, &[usize], &[u8]), String> {
        SafeTensors::raw(self, name)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        SafeTensors::get(self, name, shape)
    }
}

/// One tensor's entry in the header: its type, shape and bytes in the
/// file (within it, checked when the header is read).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    bytes: std::ops::Range<usize>,
}

/// A `.safetensors` file's tensors.
pub struct SafeTensors {
    bytes: Vec<u8>,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    /// Reads `path`.
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        SafeTensors::parse(bytes)
    }

    /// The tensors in `bytes`. A header that does not parse, or places a
    /// tensor outside the file, is an error.
    pub fn parse(bytes: Vec<u8>) -> Result<SafeTensors, String> {
        let len = bytes
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ok_or("not a safetensors file")?;
        let data_start = usize::try_from(len)
            .ok()
            .and_then(|len| len.checked_add(8))
            .filter(|&end| end <= bytes.len())
            .ok_or(format!(
                "a safetensors header of {len} bytes in a file of {}",
                bytes.len()
            ))?;
        let text = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|_| "a safetensors header that is not UTF-8")?;
        let entries = header(text, data_start, bytes.len())
            .map_err(|e| format!("a broken safetensors header: {e}"))?;
        Ok(SafeTensors { entries, bytes })
    }

    /// Whether the file has a tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Tensor `name` as stored: its type, its shape and its bytes.
    pub fn raw(&self, name: &str) -> Result<(&DType, &[usize], &[u8]), String> {
        let t = self.entries.get(name).ok_or(format!("no tensor {name}"))?;
        Ok((&t.dtype, &t.shape, &self.bytes[t.bytes.clone()]))
    }

    /// Tensor `name` with `shape`, as f32 (stored as f32, bf16 or f16).
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.entries.get(name).ok_or(format!("no tensor {name}"))?;
        fn bits(b: &[u8]) -> u16 {
            u16::from_le_bytes([b[0], b[1]])
        }
        let (size, value): (usize, fn(&[u8]) -> f32) = match &t.dtype {
            DType::F32 => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
            DType::Bf16 => (2, |b| bf16_to_f32(bits(b))),
            DType::F16 => (2, |b| f16_to_f32(bits(b))),
            other => return Err(format!("{name} is {other:?}, not f32, bf16 or f16")),
        };
        if t.shape != shape {
            return Err(format!("{name} is {:?}, not {shape:?}", t.shape));
        }
        let raw = &self.bytes[t.bytes.clone()];
        let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if count.and_then(|n| n.checked_mul(size)) != Some(raw.len()) {
            return Err(format!("{name} has the wrong length"));
        }
        Ok(raw.chunks_exact(size).map(value).collect())
    }
}

/// The header's tensors, their data from `data_start` in a file of `len`
/// bytes; an error naming the tensor and what is wrong where its entry
/// is broken or places it outside the file.
fn header(text: &str, data_start: usize, len: usize) -> Result<HashMap<String, Entry>, String> {
    let Some(Json::Object(fields)) = Json::parse(text) else {
        return Err("not a JSON object".into());
    };
    // Whole numbers a `usize` holds, not saturated casts.
    let whole =
        |n: f64| (n >= 0.0 && n.fract() == 0.0 && n < usize::MAX as f64).then_some(n as usize);
    let numbers = |v: Option<&Json>| -> Option<Vec<usize>> {
        v?.as_array()?
            .iter()
            .map(|i| i.as_f64().and_then(whole))
            .collect()
    };
    let mut out = HashMap::new();
    for (name, v) in fields {
        if name == "__metadata__" {
            continue;
        }
        if !matches!(v, Json::Object(_)) {
            return Err(format!("{name}: not an object"));
        }
        let Some(dtype) = v.get("dtype").and_then(Json::as_str) else {
            return Err(format!("{name}: no dtype"));
        };
        let shape = numbers(v.get("shape"))
            .ok_or_else(|| format!("{name}: a shape that is not whole numbers"))?;
        let offsets = numbers(v.get("data_offsets"));
        let Some([from, to]) = offsets.as_deref() else {
            return Err(format!(
                "{name}: data offsets that are not two whole numbers"
            ));
        };
        let bytes = data_start
            .checked_add(*from)
            .zip(data_start.checked_add(*to))
            .map(|(a, b)| a..b)
            .filter(|b| b.start <= b.end && b.end <= len)
            .ok_or_else(|| {
                format!("{name}: data offsets {from}..{to} outside the file's {len} bytes")
            })?;
        out.insert(
            name,
            Entry {
                dtype: DType::of(dtype),
                shape,
                bytes,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of `tensors` (name, dtype, shape, bytes).
    fn file(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        let mut entries = vec![r#""__metadata__":{"format":"pt"}"#.to_owned()];
        for (name, dtype, shape, bytes) in tensors {
            let shape: Vec<String> = shape.iter().map(ToString::to_string).collect();
            entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{},{}]}}"#,
                shape.join(","),
                data.len(),
                data.len() + bytes.len()
            ));
            data.extend(bytes);
        }
        let header = format!("{{{}}}", entries.join(", "));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header.as_bytes());
        out.extend(data);
        out
    }

    #[test]
    fn tensors_come_back_as_f32_from_each_type() {
        let f32s: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // 1.5 and -2.0 in half precision.
        let f16: Vec<u8> = [0x3e00u16, 0xc000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = SafeTensors::parse(file(&[
            ("a.weight", "F32", &[1, 2], f32s),
            ("b", "BF16", &[2], bf16),
            ("c", "F16", &[2], f16),
        ]))
        .unwrap();
        assert!(t.has("a.weight") && !t.has("d"));
        assert_eq!(t.get("a.weight", &[1, 2]).unwrap(), vec![1.5, -2.0]);
        assert_eq!(t.get("b", &[2]).unwrap(), vec![1.5, -2.0]);
        assert_eq!(t.get("c", &[2]).unwrap(), vec![1.5, -2.0]);
        assert!(t.get("a.weight", &[2]).is_err());
        assert!(t.get("d", &[2]).is_err());
        assert!(SafeTensors::parse(b"short".to_vec()).is_err());
    }

    #[test]
    fn a_tensor_comes_back_as_stored() {
        let t = SafeTensors::parse(file(&[
            ("w", "F8_E4M3", &[2, 2], vec![0x38, 0xb8, 0x40, 0x00]),
            ("s", "F32", &[], 0.5f32.to_le_bytes().to_vec()),
        ]))
        .unwrap();
        let (dtype, shape, bytes) = t.raw("w").unwrap();
        assert_eq!(
            (dtype, shape, bytes),
            (&DType::F8E4M3, &[2, 2][..], &[0x38, 0xb8, 0x40, 0x00][..])
        );
        let (dtype, shape, bytes) = t.raw("s").unwrap();
        assert_eq!((dtype, shape.len(), bytes.len()), (&DType::F32, 0, 4));
        assert!(t.raw("x").is_err());
        // Read as f32, the FP8 tensor is refused.
        assert!(t.get("w", &[2, 2]).is_err());
    }

    /// A file whose header places a tensor at `offsets` (four bytes of
    /// data follow it).
    fn placed(offsets: &str) -> Vec<u8> {
        let header = format!(r#"{{"a":{{"dtype":"F32","shape":[1],"data_offsets":{offsets}}}}}"#);
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header.as_bytes());
        out.extend(1.5f32.to_le_bytes());
        out
    }

    /// A corrupt header is an error when the file is read, never a panic
    /// or a wrapped offset: offsets past the file or past what a `usize`
    /// holds, backwards, negative or fractional; a header longer than the
    /// file; an unsupported type fails only when that tensor is read.
    #[test]
    fn a_corrupt_header_is_an_error() {
        assert_eq!(
            SafeTensors::parse(placed("[0,4]")).unwrap().get("a", &[1]),
            Ok(vec![1.5])
        );
        for (offsets, why) in [
            ("[0,5]", "a: data offsets 0..5 outside the file's"),
            ("[4,0]", "a: data offsets 4..0 outside"),
            (
                "[1e300,1e300]",
                "a: data offsets that are not two whole numbers",
            ),
            (
                "[18446744073709551615,18446744073709551616]",
                "a: data offsets that are not two whole numbers",
            ),
            ("[-4,0]", "a: data offsets that are not two whole numbers"),
            ("[0.5,4]", "a: data offsets that are not two whole numbers"),
            ("[0]", "a: data offsets that are not two whole numbers"),
        ] {
            let err = SafeTensors::parse(placed(offsets)).err().unwrap();
            assert!(err.contains(why), "{offsets}: {err}");
        }
        let mut long = placed("[0,4]");
        long[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        let err = SafeTensors::parse(long).err().unwrap();
        assert!(
            err.contains("header of 18446744073709551615 bytes"),
            "{err}"
        );
        let ints = file(&[("i", "I64", &[1], vec![0; 8])]);
        let t = SafeTensors::parse(ints).unwrap();
        assert!(t.get("i", &[1]).unwrap_err().contains("I64"));
    }
}
