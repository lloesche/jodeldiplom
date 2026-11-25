//! A `.safetensors` file's tensors as f32 ([`SafeTensors`]): the
//! checkpoints the speech models (`jodel-lyrics`) and the covers' depth and
//! matte models (`jodel-gpu`) are read from, as published.
//!
//! The file is an 8-byte length, a JSON header naming each tensor's type,
//! shape and place, and the data. Only the header's own small part of JSON
//! is read here: an object of objects with strings, numbers and arrays of
//! numbers (and a `__metadata__` object of strings, skipped).

use std::collections::HashMap;
use std::path::Path;

/// How a tensor's values are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
enum DType {
    F32,
    Bf16,
    F16,
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
            other => DType::Other(other.to_owned()),
        }
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

    /// Tensor `name` with `shape`, as f32 (stored as f32, bf16 or f16).
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.entries.get(name).ok_or(format!("no tensor {name}"))?;
        fn half(b: &[u8]) -> u16 {
            u16::from_le_bytes([b[0], b[1]])
        }
        let (size, value): (usize, fn(&[u8]) -> f32) = match &t.dtype {
            DType::F32 => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
            DType::Bf16 => (2, |b| f32::from_bits(u32::from(half(b)) << 16)),
            DType::F16 => (2, |b| f16_to_f32(half(b))),
            DType::Other(other) => return Err(format!("{name} is {other}, not f32, bf16 or f16")),
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

/// An IEEE half-precision number as f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: normalise.
            let shift = f.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((f << (shift + 1)) & 0x3ff) << 13
        }
        (31, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// A value of the header's JSON.
enum Value {
    Text(String),
    Number(f64),
    List(Vec<Value>),
    Object(Vec<(String, Value)>),
    Other,
}

/// Reads JSON from a position, as far as the header needs.
struct Reader<'a> {
    s: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn space(&mut self) {
        while self.s.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Option<()> {
        self.space();
        (self.s.get(self.at) == Some(&c)).then(|| self.at += 1)
    }

    fn text(&mut self) -> Option<String> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.at)?;
            self.at += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.s.get(self.at)?;
                    self.at += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'u' => {
                            let hex =
                                std::str::from_utf8(self.s.get(self.at..self.at + 4)?).ok()?;
                            self.at += 4;
                            let ch =
                                char::from_u32(u32::from_str_radix(hex, 16).ok()?).unwrap_or('?');
                            out.extend(ch.to_string().as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                other => out.push(other),
            }
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.space();
        match *self.s.get(self.at)? {
            b'"' => self.text().map(Value::Text),
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                if self.eat(b']').is_some() {
                    return Some(Value::List(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(b']').is_some() {
                        return Some(Value::List(items));
                    }
                    self.eat(b',')?;
                }
            }
            b'{' => {
                self.at += 1;
                let mut fields = Vec::new();
                if self.eat(b'}').is_some() {
                    return Some(Value::Object(fields));
                }
                loop {
                    let key = self.text()?;
                    self.eat(b':')?;
                    fields.push((key, self.value()?));
                    if self.eat(b'}').is_some() {
                        return Some(Value::Object(fields));
                    }
                    self.eat(b',')?;
                }
            }
            b't' | b'f' | b'n' => {
                while self.s.get(self.at).is_some_and(u8::is_ascii_alphabetic) {
                    self.at += 1;
                }
                Some(Value::Other)
            }
            _ => {
                let start = self.at;
                while self
                    .s
                    .get(self.at)
                    .is_some_and(|c| c.is_ascii_digit() || b"+-.eE".contains(c))
                {
                    self.at += 1;
                }
                std::str::from_utf8(&self.s[start..self.at])
                    .ok()?
                    .parse()
                    .ok()
                    .map(Value::Number)
            }
        }
    }
}

/// The header's tensors, their data from `data_start` in a file of `len`
/// bytes; an error naming the tensor and what is wrong where its entry
/// is broken or places it outside the file.
fn header(text: &str, data_start: usize, len: usize) -> Result<HashMap<String, Entry>, String> {
    let mut r = Reader {
        s: text.as_bytes(),
        at: 0,
    };
    let Some(Value::Object(fields)) = r.value() else {
        return Err("not a JSON object".into());
    };
    // Whole numbers a `usize` holds, not saturated casts.
    let whole =
        |n: f64| (n >= 0.0 && n.fract() == 0.0 && n < usize::MAX as f64).then_some(n as usize);
    let numbers = |v: Option<&Value>| -> Option<Vec<usize>> {
        match v? {
            Value::List(items) => items
                .iter()
                .map(|i| match i {
                    Value::Number(n) => whole(*n),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let mut out = HashMap::new();
    for (name, v) in fields {
        if name == "__metadata__" {
            continue;
        }
        let Value::Object(f) = v else {
            return Err(format!("{name}: not an object"));
        };
        let get = |k: &str| f.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let Some(Value::Text(dtype)) = get("dtype") else {
            return Err(format!("{name}: no dtype"));
        };
        let shape = numbers(get("shape"))
            .ok_or_else(|| format!("{name}: a shape that is not whole numbers"))?;
        let offsets = numbers(get("data_offsets"));
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
