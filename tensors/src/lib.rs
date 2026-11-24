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

/// One tensor's entry in the header.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    from: usize,
    to: usize,
}

/// A `.safetensors` file's tensors.
pub struct SafeTensors {
    bytes: Vec<u8>,
    entries: HashMap<String, Entry>,
    data_start: usize,
}

impl SafeTensors {
    /// Reads `path`.
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        SafeTensors::parse(bytes)
    }

    /// The tensors in `bytes`.
    pub fn parse(bytes: Vec<u8>) -> Result<SafeTensors, String> {
        let len = bytes
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or("not a safetensors file")?;
        let text = bytes
            .get(8..8usize.saturating_add(len))
            .and_then(|h| std::str::from_utf8(h).ok())
            .ok_or("a broken safetensors header")?;
        let entries = header(text).ok_or("a broken safetensors header")?;
        Ok(SafeTensors {
            data_start: 8 + len,
            entries,
            bytes,
        })
    }

    /// Whether the file has a tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Tensor `name` with `shape`, as f32 (stored as f32, bf16 or f16).
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.entries.get(name).ok_or(format!("no tensor {name}"))?;
        let size = match t.dtype.as_str() {
            "F32" => 4,
            "BF16" | "F16" => 2,
            other => return Err(format!("{name} is {other}, not f32, bf16 or f16")),
        };
        if t.shape != shape {
            return Err(format!("{name} is {:?}, not {shape:?}", t.shape));
        }
        let raw = self
            .bytes
            .get(self.data_start + t.from..self.data_start + t.to)
            .ok_or(format!("{name} is past the file's end"))?;
        if raw.len() != size * shape.iter().product::<usize>() {
            return Err(format!("{name} has the wrong length"));
        }
        Ok(match t.dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            _ => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        })
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

/// The header's tensors.
fn header(text: &str) -> Option<HashMap<String, Entry>> {
    let mut r = Reader {
        s: text.as_bytes(),
        at: 0,
    };
    let Value::Object(fields) = r.value()? else {
        return None;
    };
    let numbers = |v: &Value| -> Option<Vec<usize>> {
        match v {
            Value::List(items) => items
                .iter()
                .map(|i| match i {
                    Value::Number(n) => Some(*n as usize),
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
        let Value::Object(f) = v else { return None };
        let get = |k: &str| f.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let Some(Value::Text(dtype)) = get("dtype") else {
            return None;
        };
        let shape = numbers(get("shape")?)?;
        let offsets = numbers(get("data_offsets")?)?;
        let [from, to] = offsets[..] else { return None };
        out.insert(
            name,
            Entry {
                dtype: dtype.clone(),
                shape,
                from,
                to,
            },
        );
    }
    Some(out)
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
}
