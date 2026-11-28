//! The model as the GGUF holds it ([`Layout::of`]), placed for the GPU,
//! and loaded there ([`load`]).
//!
//! Every matrix stays in the file's blocks of its kind (Q4_K, Q6_K, BF16,
//! F32 in Unsloth's Q4_K_M; `blocks.wgsl` reads them where they lie), its
//! rows as the file has them, each matrix from a multiple of 16 bytes: the
//! queries', keys' and values' one after the other (a layer that shares an
//! earlier one's keys and values takes only its queries; a run of them of
//! one kind is multiplied as one); the feed-forward's gate and up a row
//! each in turn (gate `j` at `2j`, its up at `2j + 1`, so a multiply takes
//! each pair's gated product); and with each layer its slice of the
//! per-layer embeddings (of each token's row of 42 × 256, the layer's 256:
//! a Q5_K block).
//!
//! The weights go in 15 buffers: the token embeddings (the head too: they
//! are tied) and the per-layer projection in one, and the layers three to
//! a buffer ([`LAYERS_A_BUFFER`]). The norms and layer scalars are f32s
//! in a small one. The file's keys and values of the layers that share an
//! earlier one's, and its rotary frequencies (the global layers' quarter
//! turned, which [`super::plan::Sizes::rope`] has), stay in the file.
//!
//! Placing a tensor records where it goes and how it is filled
//! ([`Piece`]); loading is those pieces, in order.

use jodel_tensors::gguf::{Gguf, Header, Kind, Tensor};

use super::kernels::{PLAIN_KINDS, Q8_KINDS};
use super::{
    head_of, shares, Buf, DIM, FF, GLOBAL_HEAD, HEADS, KV_HEADS, LAYERS, PER_LAYER, PER_LAYERS,
    VOCAB,
};

/// A matrix in a buffer: where it starts (in words), its shape (rows are
/// outputs, columns inputs) and its blocks' kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Matrix {
    pub(super) at: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) kind: Kind,
}

impl Matrix {
    /// The bytes of a row.
    pub(super) fn row_bytes(&self) -> usize {
        self.kind
            .bytes(self.cols)
            .expect("rows of whole blocks (the layout's check)")
    }

    /// Its bytes.
    pub(super) fn bytes(&self) -> usize {
        self.rows * self.row_bytes()
    }

    /// The words it takes: whole 16 bytes, so the next starts on 16 (the
    /// kernels read a Q4_K block's words two at a time).
    fn words(&self) -> usize {
        self.bytes().div_ceil(16) * 4
    }

    /// It and `next` as one matrix, where `next` follows it in its buffer
    /// with rows of the same kind and length.
    pub(super) fn and(self, next: Matrix) -> Option<Matrix> {
        (next.kind == self.kind
            && next.cols == self.cols
            && self.bytes().is_multiple_of(16)
            && next.at == self.at + self.words())
        .then_some(Matrix {
            rows: self.rows + next.rows,
            ..self
        })
    }
}

/// One layer's tensors: its matrices in its buffer, its norms (and its
/// scalar) in the small one.
#[derive(Clone, Debug)]
pub(super) struct Layer {
    /// Queries, then keys and values where it has its own.
    pub(super) q: Matrix,
    pub(super) kv: Option<(Matrix, Matrix)>,
    pub(super) o: Matrix,
    /// Gate and up in pairs of rows.
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

/// A place: a buffer, and a word in it.
pub(super) type Place = (Buf, usize);

/// Where a tensor goes and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Fill {
    /// Its bytes as they are.
    Whole(Place),
    /// Its rows each followed by the same row of the tensor `up` (the
    /// gate's and the up's in turn).
    Pairs(Place, String),
    /// Each row cut in as many slices as there are places, slice `i` of
    /// every row one after another at place `i` (each layer's slice of the
    /// per-layer embeddings, with its layer).
    Slices(Vec<Place>),
}

/// A tensor placed: its name, where it goes and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Piece {
    pub(super) tensor: String,
    pub(super) fill: Fill,
}

/// Where every tensor is.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    /// The token embeddings (and the head).
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
    /// What [`load`] fills, in order.
    pub(super) pieces: Vec<Piece>,
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
/// 256 MB (three layers take 300 to 340 MB), so it is allocated on its own
/// rather than two layers to a block with a third of it unused.
pub(super) const LAYERS_A_BUFFER: usize = 3;

/// The file's name of layer `l`'s tensor `what`.
fn layer_name(l: usize, what: &str) -> String {
    format!("blk.{l}.{what}.weight")
}

/// The architecture's numbers the file must have, as ours.
const SHAPE: [(&str, usize); 9] = [
    ("block_count", LAYERS),
    ("embedding_length", DIM),
    ("feed_forward_length", FF),
    ("attention.head_count", HEADS),
    ("attention.head_count_kv", KV_HEADS),
    ("attention.sliding_window", super::WINDOW),
    ("attention.shared_kv_layers", LAYERS - super::OWN_KV),
    ("embedding_length_per_layer_input", PER_LAYER),
    ("attention.key_length", GLOBAL_HEAD),
];

/// The places being given out: the file's tensors checked, the pieces,
/// the small values.
struct Places<'a> {
    file: &'a Header,
    pieces: Vec<Piece>,
    small: usize,
}

/// Matrices placed one after the other in one buffer (from a word of it),
/// each from a multiple of 16 bytes.
struct Placer {
    buf: Buf,
    base: usize,
    at: usize,
}

impl Places<'_> {
    /// Tensor `name`, of `rows × cols`, in one of `kinds`.
    fn tensor(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        kinds: &[Kind],
    ) -> Result<&Tensor, String> {
        let t = self.file.tensor(name)?;
        if t.cols() != cols || t.rows() != rows {
            return Err(format!(
                "{name} is {} × {}, not {rows} × {cols}",
                t.rows(),
                t.cols()
            ));
        }
        if !kinds.contains(&t.kind()) {
            return Err(format!(
                "{name} is {}: not a kind taken here",
                t.kind().name()
            ));
        }
        Ok(t)
    }

    /// A place in `w` for `rows × cols` of `kind`, filled from `tensor` as
    /// `fill` says of the place.
    fn place(
        &mut self,
        w: &mut Placer,
        (rows, cols, kind): (usize, usize, Kind),
        tensor: String,
        fill: impl FnOnce(Place) -> Fill,
    ) -> Matrix {
        let m = Matrix {
            at: w.at,
            rows,
            cols,
            kind,
        };
        let fill = fill((w.buf, w.base + w.at));
        self.pieces.push(Piece { tensor, fill });
        w.at += m.words();
        m
    }

    /// A place in `w` for tensor `name` (`rows × cols`, a kind a matrix
    /// multiply takes), filled as it is.
    fn matrix(
        &mut self,
        w: &mut Placer,
        name: String,
        (rows, cols): (usize, usize),
    ) -> Result<Matrix, String> {
        let kind = self.tensor(&name, rows, cols, PLAIN_KINDS)?.kind();
        Ok(self.place(w, (rows, cols, kind), name, Fill::Whole))
    }

    /// A place in the small buffer for tensor `name` (`len` f32s).
    fn values(&mut self, name: String, len: usize) -> Result<usize, String> {
        self.tensor(&name, 1, len, &[Kind::F32])?;
        let at = self.small;
        self.small += len;
        self.pieces.push(Piece {
            tensor: name,
            fill: Fill::Whole((Buf::Small, at)),
        });
        Ok(at)
    }
}

impl Layout {
    /// Where the tensors of `file` (Gemma 4 E4B's text decoder) go; an
    /// error where it is another model, or a tensor's shape or kind is not
    /// what is taken here.
    pub(super) fn of(file: &Header) -> Result<Layout, String> {
        let arch = file.get("general.architecture")?.as_str();
        if arch != Some("gemma4") {
            return Err(format!("a model of {arch:?}, not gemma4"));
        }
        for (key, want) in SHAPE {
            let got = file.get(&format!("gemma4.{key}"))?.as_u64();
            if got != Some(want as u64) {
                return Err(format!("gemma4.{key} is {got:?}, not {want}"));
            }
        }
        let mut p = Places {
            file,
            pieces: Vec::new(),
            small: 0,
        };
        let mut shared = Placer {
            buf: Buf::Shared,
            base: 0,
            at: 0,
        };
        let embed = p.matrix(&mut shared, "token_embd.weight".into(), (VOCAB, DIM))?;
        let per_layer = p.matrix(
            &mut shared,
            "per_layer_model_proj.weight".into(),
            (PER_LAYERS, DIM),
        )?;
        const PLE: &str = "per_layer_token_embd.weight";
        let ple_kind = p.tensor(PLE, VOCAB, PER_LAYERS, &Kind::ALL)?.kind();
        if ple_kind.bytes(PER_LAYER).is_none() {
            return Err("the per-layer embeddings' blocks span layers".into());
        }
        let mut base = 0;
        let mut layers = Vec::with_capacity(LAYERS);
        let mut slices = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let d = head_of(l);
            let n = |what: &str| layer_name(l, what);
            if l % LAYERS_A_BUFFER == 0 {
                base = 0;
            }
            let mut w = Placer {
                buf: Buf::Layers(l / LAYERS_A_BUFFER),
                base,
                at: 0,
            };
            let q = p.matrix(&mut w, n("attn_q"), (HEADS * d, DIM))?;
            let kv = if shares(l) {
                None
            } else {
                Some((
                    p.matrix(&mut w, n("attn_k"), (KV_HEADS * d, DIM))?,
                    p.matrix(&mut w, n("attn_v"), (KV_HEADS * d, DIM))?,
                ))
            };
            if let Some((k, v)) = kv {
                // The rows for them are written once, in one mode.
                let q8 = |m: Matrix| Q8_KINDS.contains(&m.kind);
                if q8(k) != q8(q) || q8(v) != q8(q) {
                    return Err(format!(
                        "{}'s queries, keys and values are not all quantised or all not",
                        n("attn")
                    ));
                }
            }
            let o = p.matrix(&mut w, n("attn_output"), (DIM, HEADS * d))?;
            let gate = p.tensor(&n("ffn_gate"), FF, DIM, PLAIN_KINDS)?.kind();
            let up = p.tensor(&n("ffn_up"), FF, DIM, PLAIN_KINDS)?.kind();
            if gate != up {
                return Err(format!(
                    "{}'s gate is {} and its up {}: they go as one",
                    n("ffn"),
                    gate.name(),
                    up.name()
                ));
            }
            let gate_up = p.place(&mut w, (2 * FF, DIM, gate), n("ffn_gate"), |to| {
                Fill::Pairs(to, n("ffn_up"))
            });
            let down = p.matrix(&mut w, n("ffn_down"), (DIM, FF))?;
            let pl_gate = p.matrix(&mut w, n("inp_gate"), (PER_LAYER, DIM))?;
            let proj = p.matrix(&mut w, n("proj"), (DIM, PER_LAYER))?;
            let ple = Matrix {
                at: w.at,
                rows: VOCAB,
                cols: PER_LAYER,
                kind: ple_kind,
            };
            slices.push((w.buf, base + w.at));
            w.at += ple.words();
            let layer = Layer {
                q,
                kv,
                o,
                gate_up,
                down,
                gate: pl_gate,
                proj,
                ple,
                words: w.at,
                group: l / LAYERS_A_BUFFER,
                base,
                input_norm: p.values(n("attn_norm"), DIM)?,
                post_attn: p.values(n("post_attention_norm"), DIM)?,
                pre_ff: p.values(n("ffn_norm"), DIM)?,
                post_ff: p.values(n("post_ffw_norm"), DIM)?,
                post_ple: p.values(n("post_norm"), DIM)?,
                q_norm: p.values(n("attn_q_norm"), d)?,
                k_norm: if shares(l) {
                    None
                } else {
                    Some(p.values(n("attn_k_norm"), d)?)
                },
                scalar: p.values(n("layer_output_scale"), 1)?,
            };
            base += layer.words;
            layers.push(layer);
        }
        let final_norm = p.values("output_norm.weight".into(), DIM)?;
        let per_layer_norm = p.values("per_layer_proj_norm.weight".into(), PER_LAYER)?;
        p.pieces.push(Piece {
            tensor: PLE.into(),
            fill: Fill::Slices(slices),
        });
        Ok(Layout {
            embed,
            per_layer,
            shared: shared.at,
            layers,
            final_norm,
            per_layer_norm,
            small: p.small,
            pieces: p.pieces,
        })
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

/// The rows of a tensor cut in slices read at a time (the per-layer
/// embeddings': 30 MB).
const SLICE_ROWS: usize = 4096;

/// `first`'s rows each followed by the same row of `second` (rows of
/// `row` bytes).
pub(super) fn pairs(first: &[u8], second: &[u8], row: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(first.len() + second.len());
    for (a, b) in first.chunks_exact(row).zip(second.chunks_exact(row)) {
        out.extend_from_slice(a);
        out.extend_from_slice(b);
    }
    out
}

/// Slice `i` (of `slice` bytes) of each of `rows` (of `row` bytes), one
/// after another, into `out`.
pub(super) fn slice_of(rows: &[u8], (row, slice, i): (usize, usize, usize), out: &mut Vec<u8>) {
    out.clear();
    for r in rows.chunks_exact(row) {
        out.extend_from_slice(&r[i * slice..(i + 1) * slice]);
    }
}

/// The weights of `file` (as `pieces`, a layout's, place them) given to
/// `put` (a buffer, a word in it, the bytes from there) a piece at a time:
/// a tensor whole as the file holds it (borrowed, nothing copied), a gate's
/// and up's rows in turn, a tensor cut in slices [`SLICE_ROWS`] rows at a
/// time, each slice's part at its place.
pub(super) fn load(
    file: &Gguf,
    pieces: &[Piece],
    mut put: impl FnMut(Buf, usize, &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    fn whole<'a>(file: &'a Gguf, name: &str) -> Result<&'a [u8], String> {
        file.bytes(name, 0..file.header().tensor(name)?.bytes())
    }
    for piece in pieces {
        let name = piece.tensor.as_str();
        match &piece.fill {
            &Fill::Whole((buf, at)) => put(buf, at, whole(file, name)?)?,
            Fill::Pairs((buf, at), up) => {
                let row = file.header().tensor(name)?.row_bytes();
                put(*buf, *at, &pairs(whole(file, name)?, whole(file, up)?, row))?;
            }
            Fill::Slices(to) => {
                let t = file.header().tensor(name)?;
                let (row, rows) = (t.row_bytes(), t.rows());
                let slice = row / to.len();
                let mut part = Vec::with_capacity(SLICE_ROWS * slice);
                for first in (0..rows).step_by(SLICE_ROWS) {
                    let last = (first + SLICE_ROWS).min(rows);
                    let bytes = file.bytes(name, first * row..last * row)?;
                    for (i, &(buf, at)) in to.iter().enumerate() {
                        slice_of(bytes, (row, slice, i), &mut part);
                        // Whole words: SLICE_ROWS slices of a whole block each.
                        put(buf, at + first * slice / 4, &part)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// The layout on GGUF files written here: Gemma's header alone (its data
/// never read, so no GPU and no download), and small ones loaded.
#[cfg(test)]
pub(super) mod tests {
    use std::collections::HashMap;

    use super::*;

    /// A tensor as a file lists it: its name, dimensions, kind.
    type Listed = (String, Vec<usize>, Kind);

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// The header alone of a GGUF of architecture `arch` with the u32s
    /// `meta`, and `tensors` one after another (as [`gguf`] writes it).
    fn header(arch: &str, meta: &[(String, usize)], tensors: &[Listed]) -> Header {
        let (out, data) = listing(arch, meta, tensors);
        Header::from_reader(&out[..], (out.len() + data) as u64).unwrap()
    }

    /// A GGUF of architecture `arch` with the u32s `meta`, and `tensors`
    /// one after another; its data the bytes of their offsets (mod 251).
    fn gguf(arch: &str, meta: &[(String, usize)], tensors: &[Listed]) -> Vec<u8> {
        let (mut out, data) = listing(arch, meta, tensors);
        out.extend((0..data).map(|o| (o % 251) as u8));
        out
    }

    /// The header of [`gguf`]'s file, and the bytes of its data.
    fn listing(arch: &str, meta: &[(String, usize)], tensors: &[Listed]) -> (Vec<u8>, usize) {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((meta.len() as u64 + 1).to_le_bytes());
        string(&mut out, "general.architecture");
        out.extend(8u32.to_le_bytes());
        string(&mut out, arch);
        for (key, value) in meta {
            string(&mut out, key);
            out.extend(4u32.to_le_bytes());
            out.extend(u32::try_from(*value).unwrap().to_le_bytes());
        }
        let mut end = 0;
        for (name, dims, kind) in tensors {
            string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend((d as u64).to_le_bytes());
            }
            out.extend(kind.id().to_le_bytes());
            out.extend((end as u64).to_le_bytes());
            let bytes = kind.bytes(dims.iter().product()).unwrap();
            end += bytes.div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        (out, end)
    }

    /// The architecture's numbers as the file has them.
    fn meta() -> Vec<(String, usize)> {
        SHAPE
            .iter()
            .map(|&(key, v)| (format!("gemma4.{key}"), v))
            .collect()
    }

    /// Gemma 4 E4B's text decoder's tensors, of kinds as Unsloth's
    /// Q4_K_M mixes them.
    fn tensors() -> Vec<Listed> {
        let t = |name: String, dims: &[usize], kind| (name, dims.to_vec(), kind);
        let mut all = vec![
            t("token_embd.weight".into(), &[DIM, VOCAB], Kind::Q6K),
            t(
                "per_layer_model_proj.weight".into(),
                &[DIM, PER_LAYERS],
                Kind::Bf16,
            ),
            t(
                "per_layer_token_embd.weight".into(),
                &[PER_LAYERS, VOCAB],
                Kind::Q5K,
            ),
            t("output_norm.weight".into(), &[DIM], Kind::F32),
            t("per_layer_proj_norm.weight".into(), &[PER_LAYER], Kind::F32),
        ];
        for l in 0..LAYERS {
            let (d, n) = (head_of(l), |what: &str| layer_name(l, what));
            all.push(t(n("attn_q"), &[DIM, HEADS * d], Kind::Q4K));
            if !shares(l) {
                all.push(t(n("attn_k"), &[DIM, KV_HEADS * d], Kind::Q4K));
                all.push(t(n("attn_v"), &[DIM, KV_HEADS * d], Kind::Q4K));
                all.push(t(n("attn_k_norm"), &[d], Kind::F32));
            }
            all.push(t(n("attn_output"), &[HEADS * d, DIM], Kind::Q4K));
            all.push(t(n("ffn_gate"), &[DIM, FF], Kind::Q4K));
            all.push(t(n("ffn_up"), &[DIM, FF], Kind::Q4K));
            all.push(t(n("ffn_down"), &[FF, DIM], Kind::Q6K));
            all.push(t(n("inp_gate"), &[DIM, PER_LAYER], Kind::Bf16));
            all.push(t(n("proj"), &[PER_LAYER, DIM], Kind::Bf16));
            for norm in [
                "attn_norm",
                "post_attention_norm",
                "ffn_norm",
                "post_ffw_norm",
                "post_norm",
            ] {
                all.push(t(n(norm), &[DIM], Kind::F32));
            }
            all.push(t(n("attn_q_norm"), &[d], Kind::F32));
            all.push(t(n("layer_output_scale"), &[1], Kind::F32));
        }
        all
    }

    /// Gemma's layout, from its header alone.
    pub(in crate::llm) fn gemma() -> Layout {
        Layout::of(&header("gemma4", &meta(), &tensors())).unwrap()
    }

    /// Every tensor of the file is in one piece (an up with its gate); the
    /// matrices start on 16 bytes and do not overlap; the gate and up go
    /// in pairs, the per-layer embeddings in slices at each layer's place;
    /// the norms fill the small buffer.
    #[test]
    fn every_tensor_is_placed_once() {
        let listed = tensors();
        let layout = gemma();
        let mut named: Vec<&str> = layout
            .pieces
            .iter()
            .flat_map(|p| {
                let up = match &p.fill {
                    Fill::Pairs(_, up) => Some(up.as_str()),
                    _ => None,
                };
                std::iter::once(p.tensor.as_str()).chain(up)
            })
            .collect();
        named.sort_unstable();
        let mut all: Vec<&str> = listed.iter().map(|t| t.0.as_str()).collect();
        all.sort_unstable();
        assert_eq!(named, all);

        let mut matrices = vec![(Buf::Shared, layout.embed), (Buf::Shared, layout.per_layer)];
        for (l, layer) in layout.layers.iter().enumerate() {
            let buf = Buf::Layers(layer.group);
            let kv = layer.kv.map_or(vec![], |(k, v)| vec![k, v]);
            let own = [layer.q, layer.o, layer.gate_up, layer.down]
                .into_iter()
                .chain(kv)
                .chain([layer.gate, layer.proj, layer.ple]);
            matrices.extend(own.map(|m| (buf, layer.placed(m))));
            let gate_up = Fill::Pairs(
                (buf, layer.placed(layer.gate_up).at),
                layer_name(l, "ffn_up"),
            );
            assert!(layout.pieces.iter().any(|p| p.fill == gate_up), "{l}");
        }
        matrices.sort_by_key(|&(buf, m)| (buf, m.at));
        assert!(matrices.iter().all(|m| m.1.at % 4 == 0));
        for pair in matrices.windows(2) {
            let ((b0, m0), (b1, m1)) = (pair[0], pair[1]);
            assert!(b0 != b1 || m0.at + m0.words() <= m1.at, "{m0:?} {m1:?}");
        }
        let ends = |buf| {
            matrices
                .iter()
                .filter(|m| m.0 == buf)
                .map(|m| m.1.at + m.1.words())
                .max()
        };
        assert_eq!(ends(Buf::Shared), Some(layout.shared));
        for (g, &words) in layout.groups().iter().enumerate() {
            assert_eq!(ends(Buf::Layers(g)), Some(words), "buffer {g}");
        }

        let slices: Vec<Place> = layout
            .layers
            .iter()
            .map(|l| (Buf::Layers(l.group), l.base + l.ple.at))
            .collect();
        assert_eq!(layout.pieces.last().unwrap().fill, Fill::Slices(slices));
        let values: usize = listed
            .iter()
            .filter(|t| t.2 == Kind::F32)
            .map(|t| t.1.iter().product::<usize>())
            .sum();
        assert_eq!(layout.small, values);
    }

    /// Another model, another shape, a kind not taken, a layer's queries,
    /// keys and values not all of one mode, its gate and up not of one
    /// kind, or a tensor missing: each an error saying so.
    #[test]
    fn a_file_not_as_taken_is_refused() {
        let refused = |arch: &str, meta: &[(String, usize)], tensors: &[Listed], says: &str| {
            let e = Layout::of(&header(arch, meta, tensors)).err().unwrap();
            assert!(e.contains(says), "{e}, not {says}");
        };
        let (meta, tensors) = (meta(), tensors());
        let with = |name: &str, change: &dyn Fn(&mut Listed)| {
            let mut t = tensors.clone();
            change(t.iter_mut().find(|t| t.0 == name).unwrap());
            t
        };
        refused("llama", &meta, &tensors, "not gemma4");
        let mut fewer = meta.clone();
        fewer[0].1 = 40;
        refused("gemma4", &fewer, &tensors, "block_count is Some(40)");
        let q = layer_name(3, "attn_q");
        let swapped = with(&q, &|t| t.1.swap(0, 1));
        let says = format!("not {} × {DIM}", HEADS * head_of(3));
        refused("gemma4", &meta, &swapped, &says);
        let down = layer_name(3, "ffn_down");
        let q5k = with(&down, &|t| t.2 = Kind::Q5K);
        refused("gemma4", &meta, &q5k, "Q5_K: not a kind taken here");
        let k = layer_name(0, "attn_k");
        let mixed = with(&k, &|t| t.2 = Kind::Bf16);
        refused("gemma4", &meta, &mixed, "not all quantised or all not");
        let up = layer_name(5, "ffn_up");
        refused(
            "gemma4",
            &meta,
            &with(&up, &|t| t.2 = Kind::Q6K),
            "go as one",
        );
        let mut missing = tensors.clone();
        missing.retain(|t| t.0 != "output_norm.weight");
        refused("gemma4", &meta, &missing, "no tensor output_norm.weight");
    }

    /// Matrices join where one follows the other in rows of one kind and
    /// length, from whole 16 bytes.
    #[test]
    fn matrices_join_only_where_they_follow_alike() {
        let a = Matrix {
            at: 4,
            rows: 2,
            cols: 256,
            kind: Kind::Q4K,
        };
        let b = Matrix { at: 4 + 72, ..a };
        assert_eq!(a.and(b).map(|m| (m.at, m.rows)), Some((4, 4)));
        assert_eq!(a.and(Matrix { at: 80, ..b }), None);
        assert_eq!(a.and(Matrix { cols: 512, ..b }), None);
        assert_eq!(
            a.and(Matrix {
                kind: Kind::Q6K,
                ..b
            }),
            None
        );
        // 210 bytes: the next would not start on 16.
        let odd = Matrix {
            rows: 1,
            kind: Kind::Q6K,
            ..a
        };
        assert_eq!(odd.and(Matrix { at: 4 + 56, ..odd }), None);
    }

    #[test]
    fn pairs_and_slices_take_rows_in_turn() {
        assert_eq!(
            pairs(&[1, 2, 3, 4], &[5, 6, 7, 8], 2),
            [1, 2, 5, 6, 3, 4, 7, 8]
        );
        let mut out = vec![9];
        slice_of(&[0, 1, 2, 3, 4, 5, 6, 7], (4, 2, 1), &mut out);
        assert_eq!(out, [2, 3, 6, 7]);
    }

    /// Each piece where it goes: a tensor whole, the gate's and up's rows
    /// in turn, and each slice of rows past the first read's
    /// ([`SLICE_ROWS`]) at its place after those before.
    #[test]
    fn load_puts_each_piece_where_it_goes() {
        let rows = SLICE_ROWS + 3;
        let listed: Vec<Listed> = vec![
            ("w".into(), vec![5], Kind::F32),
            ("g".into(), vec![8, 3], Kind::F32),
            ("u".into(), vec![8, 3], Kind::F32),
            ("e".into(), vec![512, rows], Kind::Q5K),
        ];
        let pieces = [
            Piece {
                tensor: "w".into(),
                fill: Fill::Whole((Buf::Small, 3)),
            },
            Piece {
                tensor: "g".into(),
                fill: Fill::Pairs((Buf::Layers(0), 10), "u".into()),
            },
            Piece {
                tensor: "e".into(),
                fill: Fill::Slices(vec![(Buf::Layers(1), 0), (Buf::Layers(2), 100)]),
            },
        ];
        let file = Gguf::parse(gguf("test", &[], &listed)).unwrap();
        let mut bufs: HashMap<Buf, Vec<u8>> = HashMap::new();
        load(&file, &pieces, |buf, at, data| {
            let b = bufs.entry(buf).or_default();
            let end = 4 * at + data.len();
            b.resize(b.len().max(end), 0);
            b[4 * at..end].copy_from_slice(data);
            Ok(())
        })
        .unwrap();
        let bytes = |name: &str| {
            let n = file.header().tensor(name).unwrap().bytes();
            file.bytes(name, 0..n).unwrap()
        };
        let (w, g, u, e) = (bytes("w"), bytes("g"), bytes("u"), bytes("e"));
        assert_eq!(bufs[&Buf::Small][12..], *w);
        assert_eq!(bufs[&Buf::Layers(0)][40..], pairs(g, u, 32));
        for (buf, from, i) in [(Buf::Layers(1), 0, 0), (Buf::Layers(2), 400, 1)] {
            let mut want = Vec::new();
            slice_of(e, (352, 176, i), &mut want);
            assert_eq!(bufs[&buf][from..], want, "slice {i}");
        }
    }
}
