//! The model packed for the GPU, and its file.
//!
//! Every matrix is held at 4 bits: each row's inputs in groups of 32, each
//! group's values `offset + nibble · scale` with its scale and offset
//! fitted to it (`jodel_lyrics::asr::quant::quantize`, as the speech
//! recogniser's). A matrix is two planes in its buffer, each from a
//! multiple of 4 words: the nibbles (four words a group, value `j` in word
//! `j / 8` at bits `4·(j % 8)`, so a word's low nibbles are its even values
//! and its high ones its odd values), then the scales (a word a group, the
//! scale and offset as halves).
//!
//! The weights go in 15 buffers: the embeddings (the head too: they are
//! tied) and the per-layer projection in one, and the layers three to a
//! buffer ([`LAYERS_A_BUFFER`]), each layer's matrices with its slice of
//! the per-layer embeddings. The norms and layer scalars are f32s in a
//! small one ([`Layout`] says where everything is, from the architecture
//! alone).
//!
//! The file: `JDLGEMMA`, a version, then sections, each its length in
//! bytes (u64) and its bytes: the tokeniser, the small values, the shared
//! matrices, each layer's.

use std::io::{Read, Write};
use std::path::Path;

use jodel_lyrics::asr::quant;
use jodel_tensors::SafeTensors;

use super::tokens::Tokenizer;
use super::{head_of, shares, DIM, FF, HEADS, KV_HEADS, LAYERS, PER_LAYER, PER_LAYERS, VOCAB};

const MAGIC: &[u8; 8] = b"JDLGEMMA";
/// The packing's version: its layout and its rounding.
const VERSION: u32 = 1;

/// A 4-bit matrix in a buffer: where its planes start (in words) and its
/// shape (rows are outputs, columns inputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Matrix {
    pub(super) at: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
}

impl Matrix {
    /// The words of a matrix of `rows × cols`, both planes, padded to 4.
    fn words(rows: usize, cols: usize) -> usize {
        (rows * cols / 32 * 5).next_multiple_of(4)
    }

    /// Where its scales start.
    pub(super) fn scales(&self) -> usize {
        self.at + self.rows * self.cols / 32 * 4
    }
}

/// One layer's tensors: its matrices in its buffer, its norms (and its
/// scalar) in the small one.
#[derive(Clone, Debug)]
pub(super) struct Layer {
    /// Queries, keys and values one after the other (queries alone in a
    /// layer that shares an earlier one's keys and values).
    pub(super) qkv: Matrix,
    pub(super) o: Matrix,
    /// Gate and up in pairs of rows: gate `j` at `2j`, its up at `2j + 1`.
    pub(super) gate_up: Matrix,
    pub(super) down: Matrix,
    /// The per-layer input's gate and projection.
    pub(super) gate: Matrix,
    pub(super) proj: Matrix,
    /// Its slice of the per-layer embeddings (a row a token).
    pub(super) ple: Matrix,
    pub(super) words: usize,
    /// The buffer it is in ([`LAYERS_A_BUFFER`] layers each), and where in
    /// it (in words: its matrices' places are its own).
    pub(super) group: usize,
    pub(super) base: usize,
    pub(super) input_norm: usize,
    pub(super) post_attn: usize,
    pub(super) pre_ff: usize,
    pub(super) post_ff: usize,
    pub(super) post_ple: usize,
    pub(super) q_norm: usize,
    /// Where it has keys of its own.
    pub(super) k_norm: Option<usize>,
    pub(super) scalar: usize,
}

/// Where every tensor is.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    /// The embeddings (and the head).
    pub(super) embed: Matrix,
    /// The per-layer projection of the embeddings (every layer's 256 rows
    /// one after the other).
    pub(super) per_layer: Matrix,
    pub(super) shared: usize,
    pub(super) layers: Vec<Layer>,
    pub(super) final_norm: usize,
    /// The per-layer projection's norm.
    pub(super) per_layer_norm: usize,
    pub(super) small: usize,
}

impl Layer {
    /// `m` (one of its matrices) where it is in its buffer.
    pub(super) fn placed(&self, m: Matrix) -> Matrix {
        Matrix {
            at: self.base + m.at,
            ..m
        }
    }
}

/// The layers a buffer holds: each buffer above the allocator's blocks of
/// 256 MB (three layers take 282 to 312 MB), so it is allocated on its own
/// rather than two layers to a block with a third of it unused.
pub(super) const LAYERS_A_BUFFER: usize = 3;

/// Places matrices one after the other.
struct Placer(usize);

impl Placer {
    fn matrix(&mut self, rows: usize, cols: usize) -> Matrix {
        let m = Matrix {
            at: self.0,
            rows,
            cols,
        };
        self.0 += Matrix::words(rows, cols);
        m
    }

    fn values(&mut self, n: usize) -> usize {
        self.0 += n;
        self.0 - n
    }
}

impl Layout {
    /// The layout of Gemma 4 E4B's text decoder.
    pub(super) fn new() -> Layout {
        let mut shared = Placer(0);
        let embed = shared.matrix(VOCAB, DIM);
        let per_layer = shared.matrix(PER_LAYERS, DIM);
        let mut small = Placer(0);
        let mut base = 0;
        let layers = (0..LAYERS)
            .map(|l| {
                let d = head_of(l);
                let mut w = Placer(0);
                if l % LAYERS_A_BUFFER == 0 {
                    base = 0;
                }
                let qkv_rows = if shares(l) {
                    HEADS * d
                } else {
                    (HEADS + 2 * KV_HEADS) * d
                };
                let layer = Layer {
                    qkv: w.matrix(qkv_rows, DIM),
                    o: w.matrix(DIM, HEADS * d),
                    gate_up: w.matrix(2 * FF, DIM),
                    down: w.matrix(DIM, FF),
                    gate: w.matrix(PER_LAYER, DIM),
                    proj: w.matrix(DIM, PER_LAYER),
                    ple: w.matrix(VOCAB, PER_LAYER),
                    words: w.0,
                    group: l / LAYERS_A_BUFFER,
                    base,
                    input_norm: small.values(DIM),
                    post_attn: small.values(DIM),
                    pre_ff: small.values(DIM),
                    post_ff: small.values(DIM),
                    post_ple: small.values(DIM),
                    q_norm: small.values(d),
                    k_norm: (!shares(l)).then(|| small.values(d)),
                    scalar: small.values(1),
                };
                base += layer.words;
                layer
            })
            .collect();
        let final_norm = small.values(DIM);
        let per_layer_norm = small.values(PER_LAYER);
        Layout {
            embed,
            per_layer,
            shared: shared.0,
            layers,
            final_norm,
            per_layer_norm,
            small: small.0,
        }
    }

    /// The words of each buffer of layers.
    pub(super) fn groups(&self) -> Vec<usize> {
        let mut words = vec![0; LAYERS.div_ceil(LAYERS_A_BUFFER)];
        for layer in &self.layers {
            words[layer.group] += layer.words;
        }
        words
    }

    /// The bytes of the weights on the GPU: the shared matrices, every
    /// layer's, the small values.
    pub(super) fn bytes(&self) -> u64 {
        let words = self.shared + self.layers.iter().map(|l| l.words).sum::<usize>() + self.small;
        words as u64 * 4
    }
}

/// `values` (`rows × cols`) at 4 bits into `out` at `m` (its planes).
pub(super) fn put(out: &mut [u32], m: Matrix, values: &[f32]) {
    let groups = quant::quantize(values, m.rows, m.cols, false);
    let scales = m.scales();
    for (g, words) in groups.chunks_exact(quant::GROUP_WORDS).enumerate() {
        out[m.at + g * 4..m.at + g * 4 + 4].copy_from_slice(&words[..4]);
        out[scales + g] = words[4];
    }
}

/// The checkpoint's name of a text decoder tensor.
fn name(what: &str) -> String {
    format!("model.language_model.{what}")
}

/// Writes `section` (its length, then its bytes).
fn section(file: &mut impl Write, bytes: &[u8]) -> std::io::Result<()> {
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(bytes)
}

/// Packs the checkpoint in `dir` (Gemma 4 E4B-it: `model.safetensors`,
/// `tokenizer.json`) to the file `out`, whole or not at all; `progress`
/// is told what is packed as it goes. It reads the whole checkpoint
/// (16 GB) and the per-layer embeddings as f32s (11 GB) at once.
pub fn pack(dir: &Path, out: &Path, mut progress: impl FnMut(&str)) -> Result<(), String> {
    let layout = Layout::new();
    let tokens = Tokenizer::from_json(&dir.join("tokenizer.json"))?;
    if tokens.len() != VOCAB {
        return Err(format!("{} pieces, not {VOCAB}", tokens.len()));
    }
    let t = SafeTensors::open(&dir.join("model.safetensors"))?;
    progress("read the checkpoint");
    let get = |what: &str, shape: &[usize]| t.get(&name(what), shape);
    let mut small = vec![0.0f32; layout.small];
    let mut put_values = |at: usize, values: Vec<f32>| {
        small[at..at + values.len()].copy_from_slice(&values);
    };
    for (l, layer) in layout.layers.iter().enumerate() {
        let n = |what: &str| format!("layers.{l}.{what}");
        let d = head_of(l);
        put_values(layer.input_norm, get(&n("input_layernorm.weight"), &[DIM])?);
        put_values(
            layer.post_attn,
            get(&n("post_attention_layernorm.weight"), &[DIM])?,
        );
        put_values(
            layer.pre_ff,
            get(&n("pre_feedforward_layernorm.weight"), &[DIM])?,
        );
        put_values(
            layer.post_ff,
            get(&n("post_feedforward_layernorm.weight"), &[DIM])?,
        );
        put_values(
            layer.post_ple,
            get(&n("post_per_layer_input_norm.weight"), &[DIM])?,
        );
        put_values(layer.q_norm, get(&n("self_attn.q_norm.weight"), &[d])?);
        if let Some(k) = layer.k_norm {
            put_values(k, get(&n("self_attn.k_norm.weight"), &[d])?);
        }
        put_values(layer.scalar, get(&n("layer_scalar"), &[1])?);
    }
    put_values(layout.final_norm, get("norm.weight", &[DIM])?);
    put_values(
        layout.per_layer_norm,
        get("per_layer_projection_norm.weight", &[PER_LAYER])?,
    );
    let mut file =
        durable::AtomicFile::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let io = |e: std::io::Error| format!("{}: {e}", out.display());
    file.write_all(MAGIC).map_err(io)?;
    file.write_all(&VERSION.to_le_bytes()).map_err(io)?;
    section(&mut file, &tokens.to_bytes()).map_err(io)?;
    section(&mut file, &crate::engine::bytes(&small)).map_err(io)?;
    let mut shared = vec![0u32; layout.shared];
    put(
        &mut shared,
        layout.embed,
        &get("embed_tokens.weight", &[VOCAB, DIM])?,
    );
    put(
        &mut shared,
        layout.per_layer,
        &get("per_layer_model_projection.weight", &[PER_LAYERS, DIM])?,
    );
    section(&mut file, &crate::engine::bytes(&shared)).map_err(io)?;
    drop(shared);
    progress("packed the embeddings");
    let ple = get("embed_tokens_per_layer.weight", &[VOCAB, PER_LAYERS])?;
    for (l, layer) in layout.layers.iter().enumerate() {
        let n = |what: &str| name(&format!("layers.{l}.{what}"));
        let d = head_of(l);
        let mut words = vec![0u32; layer.words];
        let mut qkv = t.get(&n("self_attn.q_proj.weight"), &[HEADS * d, DIM])?;
        if !shares(l) {
            qkv.extend(t.get(&n("self_attn.k_proj.weight"), &[KV_HEADS * d, DIM])?);
            qkv.extend(t.get(&n("self_attn.v_proj.weight"), &[KV_HEADS * d, DIM])?);
        }
        put(&mut words, layer.qkv, &qkv);
        put(
            &mut words,
            layer.o,
            &t.get(&n("self_attn.o_proj.weight"), &[DIM, HEADS * d])?,
        );
        let gate = t.get(&n("mlp.gate_proj.weight"), &[FF, DIM])?;
        let up = t.get(&n("mlp.up_proj.weight"), &[FF, DIM])?;
        let pairs: Vec<f32> = gate
            .chunks_exact(DIM)
            .zip(up.chunks_exact(DIM))
            .flat_map(|(g, u)| g.iter().chain(u))
            .copied()
            .collect();
        put(&mut words, layer.gate_up, &pairs);
        put(
            &mut words,
            layer.down,
            &t.get(&n("mlp.down_proj.weight"), &[DIM, FF])?,
        );
        put(
            &mut words,
            layer.gate,
            &t.get(&n("per_layer_input_gate.weight"), &[PER_LAYER, DIM])?,
        );
        put(
            &mut words,
            layer.proj,
            &t.get(&n("per_layer_projection.weight"), &[DIM, PER_LAYER])?,
        );
        let slice: Vec<f32> = ple
            .chunks_exact(PER_LAYERS)
            .flat_map(|row| &row[l * PER_LAYER..(l + 1) * PER_LAYER])
            .copied()
            .collect();
        put(&mut words, layer.ple, &slice);
        section(&mut file, &crate::engine::bytes(&words)).map_err(io)?;
        progress(&format!("packed layer {l}"));
    }
    file.commit().map_err(io)
}

/// A packed file being read: its tokeniser and small values first, then
/// its weights a buffer at a time ([`Reader::next`]), as bytes the GPU
/// takes as they are.
pub(super) struct Reader {
    file: std::io::BufReader<std::fs::File>,
    pub(super) tokens: Tokenizer,
    pub(super) small: Vec<u8>,
}

impl Reader {
    /// Opens the packed file at `path`: its tokeniser and small values
    /// read.
    pub(super) fn open(path: &Path) -> Result<Reader, String> {
        let at = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut file =
            std::io::BufReader::with_capacity(1 << 22, std::fs::File::open(path).map_err(at)?);
        let mut head = [0u8; 12];
        file.read_exact(&mut head).map_err(at)?;
        if &head[..8] != MAGIC {
            return Err(format!("{} is not a packed Gemma", path.display()));
        }
        let version = u32::from_le_bytes(head[8..].try_into().unwrap());
        if version != VERSION {
            return Err(format!(
                "{} is a packed Gemma of version {version}, not {VERSION}",
                path.display()
            ));
        }
        let tokens = Tokenizer::from_bytes(&next(&mut file, None)?)?;
        let small = next(&mut file, Some(Layout::new().small * 4))?;
        Ok(Reader {
            file,
            tokens,
            small,
        })
    }

    /// The next buffer's bytes, `words` u32s long.
    pub(super) fn next(&mut self, words: usize) -> Result<Vec<u8>, String> {
        next(&mut self.file, Some(words * 4))
    }
}

/// The next section's bytes; `Err` where it is not `expect` bytes long
/// (where that is given) or the file ends.
fn next(file: &mut impl Read, expect: Option<usize>) -> Result<Vec<u8>, String> {
    let broken = |e: std::io::Error| format!("a broken packed Gemma: {e}");
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(broken)?;
    let len = u64::from_le_bytes(len) as usize;
    if let Some(expect) = expect.filter(|&e| e != len) {
        return Err(format!(
            "a packed Gemma's section of {len} bytes, not {expect}"
        ));
    }
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes).map_err(broken)?;
    Ok(bytes)
}
