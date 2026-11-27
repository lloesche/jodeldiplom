//! A GGUF file (llama.cpp's format, as quantised models are published):
//! its metadata ([`Value`]s by key) and its tensors' places
//! ([`Tensor`]), read from its header ([`Gguf::open`]); a tensor's bytes
//! are read when asked ([`Gguf::read`]), as the file holds them.
//!
//! The file: `GGUF`, its version (2 or 3), the number of tensors and of
//! metadata entries (u64s), each entry (its key, a u32 type, its value),
//! each tensor's name, dimensions (the first the one its rows run along),
//! type and offset; then, from the next multiple of the alignment
//! (`general.alignment`, else 32), the tensors' data, each at its offset
//! from there. Strings are a u64 length and UTF-8; everything is little
//! endian.
//!
//! A broken or hostile file is an error, never a panic or an allocation
//! much past the file's size: lengths are checked against the bytes left,
//! sizes are checked for overflow, arrays hold their values as typed as
//! they are stored (and arrays of arrays, which llama.cpp never writes,
//! are refused), and a key or a tensor's name given twice is refused.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// How a tensor's values are held: as floats, or in llama.cpp's blocks of
/// quantised values along its rows ([`Kind::block`]). The types this
/// reader knows (a file with another is refused).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    F32,
    Bf16,
    /// 256 values in 144 bytes: a half scale and minimum, eight 6-bit
    /// scales and minimums (of 32 values each), 4-bit values.
    Q4K,
    /// As `Q4K` with a fifth bit a value: 176 bytes.
    Q5K,
    /// 256 values in 210 bytes: 6-bit values, 16 8-bit scales (of 16
    /// values each), a half scale.
    Q6K,
}

/// A kind's facts: its GGML type number, its name as llama.cpp writes it,
/// the values a block holds and its bytes.
struct KindInfo {
    kind: Kind,
    id: u32,
    name: &'static str,
    values: usize,
    bytes: usize,
}

/// Every kind this reader knows.
const KINDS: [KindInfo; 5] = [
    KindInfo {
        kind: Kind::F32,
        id: 0,
        name: "F32",
        values: 1,
        bytes: 4,
    },
    KindInfo {
        kind: Kind::Q4K,
        id: 12,
        name: "Q4_K",
        values: 256,
        bytes: 144,
    },
    KindInfo {
        kind: Kind::Q5K,
        id: 13,
        name: "Q5_K",
        values: 256,
        bytes: 176,
    },
    KindInfo {
        kind: Kind::Q6K,
        id: 14,
        name: "Q6_K",
        values: 256,
        bytes: 210,
    },
    KindInfo {
        kind: Kind::Bf16,
        id: 30,
        name: "BF16",
        values: 1,
        bytes: 2,
    },
];

impl Kind {
    /// Every kind this reader knows.
    pub const ALL: [Kind; KINDS.len()] = {
        let mut all = [Kind::F32; KINDS.len()];
        let mut i = 0;
        while i < all.len() {
            all[i] = KINDS[i].kind;
            i += 1;
        }
        all
    };

    fn info(self) -> &'static KindInfo {
        KINDS
            .iter()
            .find(|i| i.kind == self)
            .expect("every kind in the table")
    }

    /// The kind of GGML's type number `id`.
    pub fn of(id: u32) -> Option<Kind> {
        KINDS.iter().find(|i| i.id == id).map(|i| i.kind)
    }

    /// Its GGML type number.
    pub fn id(self) -> u32 {
        self.info().id
    }

    /// Its name, as llama.cpp writes it.
    pub fn name(self) -> &'static str {
        self.info().name
    }

    /// Its blocks: the values a block holds and its bytes.
    pub fn block(self) -> (usize, usize) {
        let i = self.info();
        (i.values, i.bytes)
    }

    /// The bytes of `values` values (whole blocks, else `None`; `None`
    /// too where the bytes would overflow).
    pub fn bytes(self, values: usize) -> Option<usize> {
        let (n, bytes) = self.block();
        if values % n != 0 {
            return None;
        }
        (values / n).checked_mul(bytes)
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Array),
}

/// A metadata array, its values held as they are stored.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
}

impl Value {
    /// It as a whole number (of any width or sign).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::Uint(n) => Some(n),
            Value::Int(n) => u64::try_from(n).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl Array {
    /// Its strings, where it is of strings.
    pub fn strings(&self) -> Option<&[String]> {
        match self {
            Array::String(s) => Some(s),
            _ => None,
        }
    }

    /// Its values as i64s, where it is of whole numbers.
    pub fn ints(&self) -> Option<Vec<i64>> {
        Some(match self {
            Array::U8(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::I8(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::U16(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::I16(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::U32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::I32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            Array::I64(v) => v.clone(),
            Array::U64(v) => v
                .iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<_>>()?,
            Array::F32(_) | Array::F64(_) | Array::Bool(_) | Array::String(_) => return None,
        })
    }
}

/// A tensor's place: its name, its dimensions (the first along its rows),
/// its kind, and where its bytes are in the file (as its header checked
/// them: only a [`Gguf`] makes one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    name: String,
    dims: Vec<usize>,
    kind: Kind,
    offset: u64,
    bytes: usize,
}

impl Tensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its dimensions, the first the one its rows run along.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The bytes it takes in the file.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The values of a row (its first dimension; a tensor has one at
    /// least).
    pub fn cols(&self) -> usize {
        self.dims[0]
    }

    /// Its rows (the product of the other dimensions).
    pub fn rows(&self) -> usize {
        self.dims[1..].iter().product()
    }

    /// The bytes of a row.
    pub fn row_bytes(&self) -> usize {
        self.kind
            .bytes(self.cols())
            .expect("rows of whole blocks (checked when read)")
    }
}

/// A GGUF file: its metadata, its tensors, and the file to read their
/// bytes from.
pub struct Gguf<R = std::fs::File> {
    metadata: HashMap<String, Value>,
    tensors: Vec<Tensor>,
    by_name: HashMap<String, usize>,
    file: R,
}

impl Gguf {
    /// The file at `path`, its header read.
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let at = |e: std::io::Error| format!("{}: {e}", path.display());
        let file = std::fs::File::open(path).map_err(at)?;
        let len = file.metadata().map_err(at)?.len();
        Gguf::from_reader(file, len).map_err(|e| format!("{}: {e}", path.display()))
    }
}

impl<R> Gguf<R> {
    /// Tensor `name`.
    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| format!("no tensor {name}"))
    }

    /// Every tensor, in the file's order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// Metadata `key`.
    pub fn get(&self, key: &str) -> Result<&Value, String> {
        self.metadata
            .get(key)
            .ok_or_else(|| format!("no metadata {key}"))
    }

    /// Every metadata entry.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
}

impl<R: Read + Seek> Gguf<R> {
    /// The file `file` (`len` bytes), its header read. A header that does
    /// not parse, a type this reader does not know, a key or name given
    /// twice, or a tensor outside the file or not of whole blocks is an
    /// error.
    pub fn from_reader(mut file: R, len: u64) -> Result<Gguf<R>, String> {
        let mut r = Header {
            file: std::io::BufReader::with_capacity(1 << 20, &mut file),
            at: 0,
            len,
        };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("GGUF version {version}, not 2 or 3"));
        }
        let tensors = r.u64()?;
        let entries = r.u64()?;
        let mut metadata = HashMap::new();
        for _ in 0..entries {
            let key = r.string()?;
            let kind = r.u32()?;
            let value = r.value(kind)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(format!("metadata {key} twice"));
            }
        }
        let mut infos = Vec::new();
        for _ in 0..tensors {
            let name = r.string()?;
            let n = r.u32()?;
            if !(1..=4).contains(&n) {
                return Err(format!("{name}: {n} dimensions"));
            }
            let dims = (0..n)
                .map(|_| r.u64().and_then(size))
                .collect::<Result<Vec<usize>, String>>()?;
            let id = r.u32()?;
            let kind = Kind::of(id).ok_or_else(|| format!("{name}: GGML type {id}"))?;
            let offset = r.u64()?;
            infos.push((name, dims, kind, offset));
        }
        let align = match metadata.get("general.alignment") {
            Some(v) => v.as_u64().filter(|&a| a > 0).ok_or("a broken alignment")?,
            None => 32,
        };
        let data = r.at.div_ceil(align) * align;
        let mut out = Vec::with_capacity(infos.len());
        let mut by_name = HashMap::new();
        for (name, dims, kind, offset) in infos {
            let values = dims
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| format!("{name}: too many values"))?;
            if dims[0] % kind.block().0 != 0 {
                return Err(format!("{name}: rows not of whole {} blocks", kind.name()));
            }
            let bytes = kind
                .bytes(values)
                .ok_or_else(|| format!("{name}: too many bytes"))?;
            let start = data
                .checked_add(offset)
                .filter(|s| offset % align == 0 && s.saturating_add(bytes as u64) <= len)
                .ok_or_else(|| format!("{name}: its data outside the file or not aligned"))?;
            if by_name.insert(name.clone(), out.len()).is_some() {
                return Err(format!("tensor {name} twice"));
            }
            out.push(Tensor {
                name,
                dims,
                kind,
                offset: start,
                bytes,
            });
        }
        Ok(Gguf {
            metadata,
            tensors: out,
            by_name,
            file,
        })
    }

    /// The bytes `range` (within its bytes) of tensor `name` (its place as
    /// this file's header gives it).
    pub fn read(&mut self, name: &str, range: std::ops::Range<usize>) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(name)?;
        if range.start > range.end || range.end > tensor.bytes {
            return Err(format!("bytes {range:?} of {name}"));
        }
        let at = tensor.offset + range.start as u64;
        let io = |e: std::io::Error| format!("{name}: {e}");
        self.file.seek(SeekFrom::Start(at)).map_err(io)?;
        let mut out = vec![0u8; range.len()];
        self.file.read_exact(&mut out).map_err(io)?;
        Ok(out)
    }
}

/// `n` as a size here.
fn size(n: u64) -> Result<usize, String> {
    usize::try_from(n).map_err(|_| format!("a size of {n}"))
}

/// The header being read: where in the file, and its length.
struct Header<R> {
    file: R,
    at: u64,
    len: u64,
}

impl<R: Read> Header<R> {
    fn take(&mut self, n: u64) -> Result<Vec<u8>, String> {
        if n > self.len.saturating_sub(self.at) {
            return Err("a header past the file's end".into());
        }
        let mut out = vec![0u8; n as usize];
        self.file
            .read_exact(&mut out)
            .map_err(|e| format!("its header: {e}"))?;
        self.at += n;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N as u64)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, String> {
        self.array().map(u64::from_le_bytes)
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        String::from_utf8(self.take(n)?).map_err(|_| "a string that is not UTF-8".to_owned())
    }

    /// `n` values each read by `one`, `bytes` (at least) each: refused
    /// where they would run past the file.
    fn many<T>(
        &mut self,
        n: u64,
        bytes: u64,
        mut one: impl FnMut(&mut Self) -> Result<T, String>,
    ) -> Result<Vec<T>, String> {
        if n.saturating_mul(bytes) > self.len.saturating_sub(self.at) {
            return Err("an array past the file's end".into());
        }
        (0..n).map(|_| one(self)).collect()
    }

    /// An array: its values' type, its length, its values.
    fn list(&mut self) -> Result<Array, String> {
        let kind = self.u32()?;
        let n = self.u64()?;
        Ok(match kind {
            0 => Array::U8(self.many(n, 1, |r| Ok(r.array::<1>()?[0]))?),
            1 => Array::I8(self.many(n, 1, |r| Ok(r.array::<1>()?[0] as i8))?),
            2 => Array::U16(self.many(n, 2, |r| r.array().map(u16::from_le_bytes))?),
            3 => Array::I16(self.many(n, 2, |r| r.array().map(i16::from_le_bytes))?),
            4 => Array::U32(self.many(n, 4, Self::u32)?),
            5 => Array::I32(self.many(n, 4, |r| r.array().map(i32::from_le_bytes))?),
            6 => Array::F32(self.many(n, 4, |r| r.array().map(f32::from_le_bytes))?),
            7 => Array::Bool(self.many(n, 1, |r| Ok(r.array::<1>()?[0] != 0))?),
            8 => Array::String(self.many(n, 8, Self::string)?),
            9 => return Err("an array of arrays".into()),
            10 => Array::U64(self.many(n, 8, Self::u64)?),
            11 => Array::I64(self.many(n, 8, |r| r.array().map(i64::from_le_bytes))?),
            12 => Array::F64(self.many(n, 8, |r| r.array().map(f64::from_le_bytes))?),
            other => return Err(format!("an array of type {other}")),
        })
    }

    /// A value of GGUF's type `kind`.
    fn value(&mut self, kind: u32) -> Result<Value, String> {
        Ok(match kind {
            0 => Value::Uint(u64::from(self.array::<1>()?[0])),
            1 => Value::Int(i64::from(self.array::<1>()?[0] as i8)),
            2 => Value::Uint(u64::from(u16::from_le_bytes(self.array()?))),
            3 => Value::Int(i64::from(i16::from_le_bytes(self.array()?))),
            4 => Value::Uint(u64::from(self.u32()?)),
            5 => Value::Int(i64::from(i32::from_le_bytes(self.array()?))),
            6 => Value::Float(f64::from(f32::from_le_bytes(self.array()?))),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => Value::Array(self.list()?),
            10 => Value::Uint(self.u64()?),
            11 => Value::Int(i64::from_le_bytes(self.array()?)),
            12 => Value::Float(f64::from_le_bytes(self.array()?)),
            other => return Err(format!("a metadata value of type {other}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// What a small file holds: its version, the alignment, and its second
    /// tensor's type, offset and dimensions.
    struct Small {
        version: u32,
        align: u32,
        m_type: u32,
        m_offset: u64,
        m_dims: [u64; 2],
    }

    const SMALL: Small = Small {
        version: 3,
        align: 64,
        m_type: 12,
        m_offset: 64,
        m_dims: [256, 2],
    };

    /// A small file: metadata of each kind, an alignment, and two tensors
    /// (an f32 vector, a matrix of 2 rows of 256 of `s.m_type`, a Q4_K's
    /// bytes).
    fn file(s: Small) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(s.version.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend(5u64.to_le_bytes());
        string(&mut out, "general.alignment");
        out.extend(4u32.to_le_bytes());
        out.extend(s.align.to_le_bytes());
        string(&mut out, "a.name");
        out.extend(8u32.to_le_bytes());
        string(&mut out, "tiny");
        string(&mut out, "a.ints");
        out.extend(9u32.to_le_bytes());
        out.extend(5u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        out.extend((-3i32).to_le_bytes());
        out.extend(7i32.to_le_bytes());
        string(&mut out, "a.words");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend(2u64.to_le_bytes());
        string(&mut out, "x y");
        string(&mut out, "▁z");
        string(&mut out, "a.eps");
        out.extend(6u32.to_le_bytes());
        out.extend(0.5f32.to_le_bytes());
        string(&mut out, "v");
        out.extend(1u32.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        string(&mut out, "m");
        out.extend(2u32.to_le_bytes());
        out.extend(s.m_dims[0].to_le_bytes());
        out.extend(s.m_dims[1].to_le_bytes());
        out.extend(s.m_type.to_le_bytes());
        out.extend(s.m_offset.to_le_bytes());
        let align = (s.align as usize).max(1);
        out.resize(out.len().div_ceil(align) * align, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            out.extend(v.to_le_bytes());
        }
        out.resize(out.len().div_ceil(align) * align, 0);
        out.extend((0..288).map(|i| i as u8));
        out
    }

    fn open(bytes: Vec<u8>) -> Result<Gguf<Cursor<Vec<u8>>>, String> {
        let len = bytes.len() as u64;
        Gguf::from_reader(Cursor::new(bytes), len)
    }

    #[test]
    fn a_small_file_reads_back() {
        let mut g = open(file(SMALL)).unwrap();
        assert_eq!(g.get("a.name").unwrap().as_str(), Some("tiny"));
        assert_eq!(
            g.get("a.ints").unwrap(),
            &Value::Array(Array::I32(vec![-3, 7]))
        );
        let words = g.get("a.words").unwrap().as_array().unwrap();
        assert_eq!(words.strings().unwrap()[1], "▁z");
        assert_eq!(g.get("a.eps").unwrap(), &Value::Float(0.5));
        let v = g.tensor("v").unwrap();
        assert_eq!(
            (v.kind(), v.cols(), v.rows(), v.bytes()),
            (Kind::F32, 4, 1, 16)
        );
        assert_eq!(g.read("v", 4..8).unwrap(), 2.0f32.to_le_bytes());
        let m = g.tensor("m").unwrap();
        assert_eq!(
            (m.kind(), m.cols(), m.rows(), m.row_bytes()),
            (Kind::Q4K, 256, 2, 144)
        );
        let bytes = g.read("m", 0..288).unwrap();
        assert_eq!(bytes, (0..288).map(|i| i as u8).collect::<Vec<_>>());
        assert!(g.read("m", 0..289).is_err());
        assert!(g.read("w", 0..1).is_err());
        assert!(g.tensor("w").is_err());
        assert_eq!(g.tensors().len(), 2);
    }

    /// The error `bytes` gives, which must hold `why`.
    fn refused(bytes: Vec<u8>, why: &str) {
        let err = open(bytes).err().expect("refused");
        assert!(err.contains(why), "{err} (want {why})");
    }

    #[test]
    fn a_broken_file_is_an_error() {
        let good = file(SMALL);
        refused(good[..good.len() - 1].to_vec(), "m: its data outside");
        assert!(open(good[..40].to_vec()).is_err());
        let mut bad = good.clone();
        bad[0] = b'X';
        refused(bad, "not a GGUF file");
        // A Q4_0 (2), which it does not know.
        refused(file(Small { m_type: 2, ..SMALL }), "m: GGML type 2");
        refused(
            file(Small {
                m_offset: 65,
                ..SMALL
            }),
            "not aligned",
        );
        refused(file(Small { align: 0, ..SMALL }), "a broken alignment");
        refused(
            file(Small {
                version: 1,
                ..SMALL
            }),
            "version 1",
        );
        refused(
            file(Small {
                version: 4,
                ..SMALL
            }),
            "version 4",
        );
        refused(
            file(Small {
                m_dims: [256, u64::MAX / 2],
                ..SMALL
            }),
            "m: too many values",
        );
        // As many values as a size holds, as f32s four times the bytes.
        refused(
            file(Small {
                m_type: 0,
                m_dims: [4, u64::MAX / 4],
                ..SMALL
            }),
            "m: too many bytes",
        );
        refused(
            file(Small {
                m_dims: [100, 2],
                ..SMALL
            }),
            "rows not of whole Q4_K blocks",
        );
    }

    /// The small file with its "a.ints" entry's array rewritten: its type
    /// and its length.
    fn with_ints(kind: u32, n: u64) -> Vec<u8> {
        let mut bytes = file(SMALL);
        let key = b"a.ints";
        let at = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len() + 4;
        bytes[at..at + 4].copy_from_slice(&kind.to_le_bytes());
        bytes[at + 4..at + 12].copy_from_slice(&n.to_le_bytes());
        bytes
    }

    #[test]
    fn a_hostile_header_is_an_error() {
        refused(with_ints(5, 1 << 40), "an array past the file's end");
        refused(with_ints(9, 2), "an array of arrays");
        // "a.name" given twice: the second key renamed to it.
        let mut twice = file(SMALL);
        let at = twice.windows(6).position(|w| w == b"a.ints").unwrap();
        twice[at..at + 6].copy_from_slice(b"a.name");
        refused(twice, "metadata a.name twice");
        // Tensor "v" twice: "m" renamed to it.
        let mut twice = file(SMALL);
        let at = twice
            .windows(9)
            .position(|w| w == [1, 0, 0, 0, 0, 0, 0, 0, b'm'])
            .unwrap();
        twice[at + 8] = b'v';
        refused(twice, "tensor v twice");
    }
}
