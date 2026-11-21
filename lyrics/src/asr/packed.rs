//! The recogniser as one file of our own: its large matrices already at 4
//! bits ([`super::quant`]), the rest as f32, and the vocabulary; about
//! 0.5 GB against the checkpoint's 1.9 GB, read in a moment. The GPU's
//! version runs from it ([`PackedModel`]); [`PackedModel::from_model`]
//! makes it from the CPU's.
//!
//! The file: `JDLASR` and a version byte and a reserved one, then the
//! entries, each its name (u16 length, UTF-8), its kind (u8: 0 values, 1 a
//! 4-bit matrix, 2 an 8-bit one, 3 the vocabulary, 4 a matrix of halves), rows and columns (u32
//! each, 0 for values), the count of its 4-byte words or bytes (u64), and
//! those. Little-endian throughout. It ends with the FNV-1a hash (u64) of
//! all before it.

use std::path::Path;

use super::{quant, text, Qwen3Asr};

const MAGIC: &[u8; 6] = b"JDLASR";
/// The file's version: 2 since the player weights the 4-bit rounding by the
/// built-in calibration statistics ([`quant::Statistics::built_in`]), so a
/// file packed without them is packed again.
pub const VERSION: u8 = 2;

/// A tensor of the packed model.
#[derive(Clone, Debug, PartialEq)]
pub enum Tensor {
    /// f32 values (biases, norms, the first convolution).
    Values(Vec<f32>),
    /// A matrix at 4 bits (`rows × cols`, [`quant::quantize`]'s layout).
    Q4 {
        /// Rows (outputs).
        rows: usize,
        /// Columns (inputs).
        cols: usize,
        /// The groups.
        data: Vec<u32>,
    },
    /// A matrix at 8 bits ([`quant::quantize8`]'s layout).
    Q8 {
        /// Rows (outputs).
        rows: usize,
        /// Columns (inputs).
        cols: usize,
        /// The groups.
        data: Vec<u32>,
    },
    /// A matrix as halves ([`quant::to_halves`]'s layout): the checkpoint's
    /// values, for a reference.
    F16 {
        /// Rows (outputs).
        rows: usize,
        /// Columns (inputs).
        cols: usize,
        /// Two halves a word.
        data: Vec<u32>,
    },
}

/// The recogniser's tensors, named as [`Qwen3Asr::visit`] names them, and
/// its vocabulary. Without a `head`, the head is the embeddings (the
/// checkpoint ties them).
#[derive(Clone, Debug)]
pub struct PackedModel {
    /// The tensors in [`Qwen3Asr::visit`]'s order.
    pub tensors: Vec<(String, Tensor)>,
    /// The vocabulary.
    pub vocab: text::Vocab,
}

/// Matrix `name`'s columns' weights from `statistics`: their mean squares,
/// none under a thousandth of their mean (a column never heard still
/// counts a little). `None` when it has none for it.
fn importance(statistics: &quant::Statistics, name: &str, cols: usize) -> Option<Vec<f32>> {
    let ms = statistics.mean_squares(name).filter(|m| m.len() == cols)?;
    let mean = ms.iter().sum::<f32>() / cols as f32;
    (mean > 0.0).then(|| ms.iter().map(|&m| m.max(mean * 1e-3)).collect())
}

fn fnv(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

impl PackedModel {
    /// The CPU model's tensors, its matrices quantised to the bits `bits`
    /// gives each by name (4, 8 or 16; the first convolution stays f32), the
    /// 4-bit ones weighted by `statistics` where it has them
    /// ([`quant::quantize_weighted`]).
    pub fn from_model(
        model: &Qwen3Asr,
        bits: impl Fn(&str) -> u32,
        statistics: Option<&quant::Statistics>,
    ) -> PackedModel {
        PackedModel::from_model_with(model, bits, statistics, |_, _| {})
    }

    /// [`PackedModel::from_model`], telling `progress` the tensors packed so
    /// far and in all after each.
    pub fn from_model_with(
        model: &Qwen3Asr,
        bits: impl Fn(&str) -> u32,
        statistics: Option<&quant::Statistics>,
        mut progress: impl FnMut(usize, usize),
    ) -> PackedModel {
        let mut all = 0;
        model.visit(&mut |_, _, _| all += 1);
        let mut tensors = Vec::new();
        model.visit(&mut |name, values, shape| {
            let t = match shape {
                Some((rows, cols)) if name != "conv1.w" => match bits(name) {
                    16 => Tensor::F16 {
                        rows,
                        cols,
                        data: quant::to_halves(values, rows, cols),
                    },
                    8 => Tensor::Q8 {
                        rows,
                        cols,
                        data: quant::quantize8(values, rows, cols),
                    },
                    _ => Tensor::Q4 {
                        rows,
                        cols,
                        data: match statistics.and_then(|s| importance(s, name, cols)) {
                            Some(imp) => quant::quantize_weighted(values, rows, cols, &imp),
                            None => quant::quantize(values, rows, cols, false),
                        },
                    },
                },
                _ => Tensor::Values(values.to_vec()),
            };
            tensors.push((name.to_owned(), t));
            progress(tensors.len(), all);
        });
        // The checkpoint ties the head to the embeddings: one copy, while
        // they round alike (weighted, only the head has statistics: two).
        let find = |n: &str| tensors.iter().position(|(name, _)| name == n);
        if let (Some(e), Some(h)) = (find("embed"), find("head")) {
            if tensors[e].1 == tensors[h].1 {
                tensors.remove(h);
            }
        }
        PackedModel {
            tensors,
            vocab: model.vocab().clone(),
        }
    }

    /// The file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[VERSION, 0]);
        let mut entry =
            |name: &str, kind: u8, rows: usize, cols: usize, words: &[u8], count: usize| {
                out.extend_from_slice(&(name.len() as u16).to_le_bytes());
                out.extend_from_slice(name.as_bytes());
                out.push(kind);
                out.extend_from_slice(&(rows as u32).to_le_bytes());
                out.extend_from_slice(&(cols as u32).to_le_bytes());
                out.extend_from_slice(&(count as u64).to_le_bytes());
                out.extend_from_slice(words);
            };
        for (name, t) in &self.tensors {
            match t {
                Tensor::Values(v) => {
                    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    entry(name, 0, 0, 0, &b, v.len());
                }
                Tensor::Q4 { rows, cols, data }
                | Tensor::Q8 { rows, cols, data }
                | Tensor::F16 { rows, cols, data } => {
                    let kind = match t {
                        Tensor::Q4 { .. } => 1,
                        Tensor::Q8 { .. } => 2,
                        _ => 4,
                    };
                    let b: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                    entry(name, kind, *rows, *cols, &b, data.len());
                }
            }
        }
        let v = self.vocab.to_bytes();
        entry("vocab", 3, 0, 0, &v, v.len());
        let hash = fnv(&out);
        out.extend_from_slice(&hash.to_le_bytes());
        out
    }

    /// Writes the file.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        // Whole or not at all, and durable.
        durable::write_atomic(path, &self.to_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Whether the file at `path` is a packed recogniser of this version
    /// (from its first bytes only).
    pub fn is_current(path: &Path) -> bool {
        use std::io::Read;
        let mut head = [0u8; 8];
        std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut head))
            .is_ok()
            && &head[..6] == MAGIC
            && head[6] == VERSION
    }

    /// Reads the file at `path`.
    pub fn open(path: &Path) -> Result<PackedModel, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        PackedModel::from_bytes(&data)
    }

    /// From the file's bytes.
    pub fn from_bytes(data: &[u8]) -> Result<PackedModel, String> {
        let broken = || "not a packed recogniser, or a broken one".to_owned();
        if data.len() < 16 || &data[..6] != MAGIC {
            return Err(broken());
        }
        if data[6] != VERSION {
            return Err(format!("a packed recogniser of version {}", data[6]));
        }
        let (body, tail) = data.split_at(data.len() - 8);
        if fnv(body) != u64::from_le_bytes(tail.try_into().unwrap()) {
            return Err("the packed recogniser is damaged".into());
        }
        let mut at = 8;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = body.get(at..at + n).ok_or_else(broken)?;
            at += n;
            Ok(s)
        };
        let mut tensors = Vec::new();
        let mut vocab = None;
        while let Ok(len) = take(2) {
            let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
            let name = String::from_utf8(take(len)?.to_vec()).map_err(|_| broken())?;
            let kind = take(1)?[0];
            let rows = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let cols = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let count = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
            let words = |b: &[u8]| -> Vec<u32> {
                b.chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            };
            let t = match kind {
                0 => Tensor::Values(
                    take(4 * count)?
                        .chunks_exact(4)
                        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                        .collect(),
                ),
                1 => Tensor::Q4 {
                    rows,
                    cols,
                    data: words(take(4 * count)?),
                },
                2 => Tensor::Q8 {
                    rows,
                    cols,
                    data: words(take(4 * count)?),
                },
                3 => {
                    vocab = Some(text::Vocab::from_bytes(take(count)?)?);
                    continue;
                }
                4 => Tensor::F16 {
                    rows,
                    cols,
                    data: words(take(4 * count)?),
                },
                _ => return Err(broken()),
            };
            tensors.push((name, t));
        }
        Ok(PackedModel {
            tensors,
            vocab: vocab.ok_or("the packed recogniser has no vocabulary")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_reads_back_and_a_damaged_one_is_refused() {
        let vocab = text::Vocab::parse(
            r#"{"a": 0, "b": 1}"#,
            r#"{"added_tokens_decoder": {"2": {"content": "<x>", "special": true}}}"#,
        )
        .unwrap();
        let q = quant::quantize(&[0.5; 64], 2, 32, false);
        let model = PackedModel {
            tensors: vec![
                ("bias".into(), Tensor::Values(vec![1.0, -2.0])),
                (
                    "w".into(),
                    Tensor::Q4 {
                        rows: 2,
                        cols: 32,
                        data: q,
                    },
                ),
            ],
            vocab,
        };
        let bytes = model.to_bytes();
        let back = PackedModel::from_bytes(&bytes).unwrap();
        assert_eq!(back.tensors, model.tensors);
        assert_eq!(back.vocab.decode(&[0, 1, 2]), "ab");
        let mut bad = bytes.clone();
        bad[20] ^= 1;
        assert!(PackedModel::from_bytes(&bad).is_err());
    }
}
