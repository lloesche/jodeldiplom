//! A part's weights as they go to the GPU ([`Packing`]): matrices placed
//! in buffers of words (FP8 bytes as the checkpoint holds them, bf16 two a
//! word), with their scales, biases and other f32 tensors beside them; then
//! uploaded ([`Packing::upload`]). The same packing with no checkpoint
//! ([`Source::Shapes`]) places everything without reading anything: what a
//! part takes before it is loaded is what it takes once loaded.

use std::borrow::Cow;

use jodel_tensors::{DType, Tensors};

use super::gpu::{Gpu, Loaded};
use super::plan::{Buf, Format, Mat};

/// The most bytes in one buffer of matrices.
const MOST: u64 = 1 << 30;

/// Where a part's tensors are read from: a checkpoint, or none (their
/// shapes alone: no bytes, and values of zeros).
#[derive(Clone, Copy)]
pub(crate) enum Source<'a> {
    Checkpoint(&'a dyn Tensors),
    Shapes,
}

impl<'a> Source<'a> {
    /// Tensor `name`'s bytes, its type and shape checked (none from shapes
    /// alone).
    pub(crate) fn bytes(
        self,
        name: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Option<&'a [u8]>, String> {
        let Source::Checkpoint(t) = self else {
            return Ok(None);
        };
        let (d, s, bytes) = t.raw(name)?;
        if *d != dtype || s != shape {
            return Err(format!("{name} is {d:?} {s:?}, not {dtype:?} {shape:?}"));
        }
        Ok(Some(bytes))
    }

    /// Tensor `name` (`len` values) as f32s (zeros from shapes alone).
    pub(crate) fn get(self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        match self {
            Source::Checkpoint(t) => t.get(name, &[len]),
            Source::Shapes => Ok(vec![0.0; len]),
        }
    }

    /// A one-value tensor `name` (zero from shapes alone).
    fn scalar(self, name: &str) -> Result<f32, String> {
        match self {
            Source::Checkpoint(t) => Ok(t.get(name, &[])?[0]),
            Source::Shapes => Ok(0.0),
        }
    }
}

/// An FP8 matrix as the checkpoint holds it: `n` rows of `k` bytes (none
/// from shapes alone), and a scale for each 64 rows and 128 values of a
/// row (`n / 64 × ⌈k / 128⌉`).
pub(crate) struct Fp8<'a> {
    pub(crate) bytes: Option<&'a [u8]>,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) scales: Vec<f32>,
}

impl<'a> Fp8<'a> {
    /// `name.weight` (`n × k`) with its one scale `name.weight_scale`
    /// (BFL's checkpoints).
    pub(crate) fn per_tensor(
        src: Source<'a>,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Fp8<'a>, String> {
        let bytes = fp8_bytes(src, &format!("{name}.weight"), (n, k))?;
        let scale = src.scalar(&format!("{name}.weight_scale"))?;
        Ok(Fp8 {
            bytes,
            n,
            k,
            scales: vec![scale; n / 64 * k.div_ceil(128)],
        })
    }

    /// `name.weight` (`n × k`) with its scales per block of 128 × 128
    /// (`name.weight_scale_inv`: Qwen's FP8 checkpoints).
    pub(crate) fn per_block(
        src: Source<'a>,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Fp8<'a>, String> {
        let bytes = fp8_bytes(src, &format!("{name}.weight"), (n, k))?;
        let kb = k.div_ceil(128);
        let blocks = match src {
            Source::Checkpoint(t) => {
                t.get(&format!("{name}.weight_scale_inv"), &[n.div_ceil(128), kb])?
            }
            Source::Shapes => vec![0.0; n.div_ceil(128) * kb],
        };
        let scales = (0..n / 64)
            .flat_map(|g| blocks[g / 2 * kb..(g / 2 + 1) * kb].iter().copied())
            .collect();
        Ok(Fp8 {
            bytes,
            n,
            k,
            scales,
        })
    }

    /// Rows `from..from + n` (multiples of 64).
    pub(crate) fn rows(&self, from: usize, n: usize) -> Fp8<'a> {
        let kb = self.k.div_ceil(128);
        Fp8 {
            bytes: self.bytes.map(|b| &b[from * self.k..(from + n) * self.k]),
            n,
            k: self.k,
            scales: self.scales[from / 64 * kb..(from + n) / 64 * kb].to_vec(),
        }
    }
}

fn fp8_bytes<'a>(
    src: Source<'a>,
    name: &str,
    (n, k): (usize, usize),
) -> Result<Option<&'a [u8]>, String> {
    if !n.is_multiple_of(64) || !k.is_multiple_of(32) {
        return Err(format!(
            "{name}: {n} × {k} is not in rows of 64 and values of 32"
        ));
    }
    src.bytes(name, DType::F8E4M3, &[n, k])
}

/// A buffer's pieces: each one's offset and bytes (none from
/// shapes alone).
type Pieces<'a> = Vec<(u64, Option<Cow<'a, [u8]>>)>;

/// A part's weights before they go up: the matrices' buffers (each its
/// length and its pieces at their offsets) and the f32 tensors.
pub(crate) struct Packing<'a> {
    buffers: Vec<(u64, Pieces<'a>)>,
    pub(crate) small: Vec<f32>,
}

impl<'a> Packing<'a> {
    pub(crate) fn new() -> Packing<'a> {
        Packing {
            buffers: Vec::new(),
            small: Vec::new(),
        }
    }

    /// `len` bytes (`data`, where read) placed in a buffer (16-byte
    /// aligned): its buffer and its first vector of four words.
    fn place(&mut self, len: usize, data: Option<Cow<'a, [u8]>>) -> (Buf, usize) {
        debug_assert!(data.as_ref().is_none_or(|d| d.len() == len));
        let room = (len as u64).next_multiple_of(16);
        if self.buffers.last().is_none_or(|b| b.0 + room > MOST) {
            self.buffers.push((0, Vec::new()));
        }
        let i = self.buffers.len() - 1;
        let b = &mut self.buffers[i];
        let at = b.0;
        b.0 += room;
        b.1.push((at, data));
        (Buf::W(i as u8), (at / 16) as usize)
    }

    /// f32s into `small` (from a multiple of 4): their offset.
    pub(crate) fn vector(&mut self, values: &[f32]) -> usize {
        let at = self.small.len().next_multiple_of(4);
        self.small.resize(at, 0.0);
        self.small.extend_from_slice(values);
        at
    }

    /// A bf16 tensor of `len` values into `small` as f32s.
    pub(crate) fn bf16_vector(
        &mut self,
        src: Source<'a>,
        name: &str,
        len: usize,
    ) -> Result<usize, String> {
        let v = src.get(name, len)?;
        Ok(self.vector(&v))
    }

    /// FP8 bytes (`n` rows of `k`) with their scales, placed.
    fn fp8_placed(
        &mut self,
        bytes: Option<Cow<'a, [u8]>>,
        (n, k): (usize, usize),
        scales: &[f32],
    ) -> Mat {
        let scales = self.vector(scales);
        let (buf, at) = self.place(n * k, bytes);
        Mat {
            buf,
            at,
            n,
            k,
            ldw: k,
            format: Format::Fp8 {
                scales,
                per_row_group: k.div_ceil(128),
            },
            bias: None,
        }
    }

    /// An FP8 matrix as it is.
    pub(crate) fn fp8(&mut self, w: &Fp8<'a>) -> Mat {
        self.fp8_placed(w.bytes.map(Cow::Borrowed), (w.n, w.k), &w.scales)
    }

    /// FP8 matrices of the same width, one under another.
    pub(crate) fn fp8_stack(&mut self, parts: &[Fp8<'a>]) -> Mat {
        let k = parts[0].k;
        assert!(parts.iter().all(|p| p.k == k));
        let bytes = parts
            .iter()
            .map(|p| p.bytes)
            .collect::<Option<Vec<_>>>()
            .map(|b| Cow::Owned(b.concat()));
        let scales: Vec<f32> = parts
            .iter()
            .flat_map(|p| p.scales.iter().copied())
            .collect();
        let n = parts.iter().map(|p| p.n).sum();
        self.fp8_placed(bytes, (n, k), &scales)
    }

    /// Gates and their ups (two FP8 matrices of the same shape) as one, 64
    /// rows of gates then their 64 ups (SwiGLU's pairs: `Tail::SwiGlu`).
    pub(crate) fn fp8_pairs(&mut self, gate: &Fp8<'a>, up: &Fp8<'a>) -> Mat {
        assert!(gate.n == up.n && gate.k == up.k);
        let (n, k) = (gate.n, gate.k);
        let kb = k.div_ceil(128);
        let mut scales = Vec::with_capacity(2 * gate.scales.len());
        for g in 0..n / 64 {
            for part in [gate, up] {
                scales.extend_from_slice(&part.scales[g * kb..(g + 1) * kb]);
            }
        }
        let bytes = gate.bytes.zip(up.bytes).map(|(gate, up)| {
            let mut bytes = Vec::with_capacity(2 * n * k);
            for g in 0..n / 64 {
                for part in [gate, up] {
                    bytes.extend_from_slice(&part[g * 64 * k..(g + 1) * 64 * k]);
                }
            }
            Cow::Owned(bytes)
        });
        self.fp8_placed(bytes, (2 * n, k), &scales)
    }

    /// A bf16 matrix (`n × k`, k a multiple of 32) as it is.
    pub(crate) fn bf16(
        &mut self,
        src: Source<'a>,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Mat, String> {
        assert_eq!(k % 32, 0);
        let bytes = src.bytes(name, DType::Bf16, &[n, k])?;
        let (buf, at) = self.place(n * k * 2, bytes.map(Cow::Borrowed));
        Ok(Mat {
            buf,
            at,
            n,
            k,
            ldw: k,
            format: Format::Bf16,
            bias: None,
        })
    }

    /// A convolution's weight (`[cout][cin][size][size]`, bf16) as rows of
    /// taps, each tap's channels padded to `cin_pad`, the rows padded to a
    /// multiple of 32 (zeros); its bias (`name.bias`) into `small`.
    pub(crate) fn conv(
        &mut self,
        src: Source<'a>,
        name: &str,
        (cin, cout, size): (usize, usize, usize),
        cin_pad: usize,
    ) -> Result<Mat, String> {
        let raw = src.bytes(
            &format!("{name}.weight"),
            DType::Bf16,
            &[cout, cin, size, size],
        )?;
        let k = size * size * cin_pad;
        let ldw = k.next_multiple_of(32);
        let out = raw.map(|raw| {
            let mut out = vec![0u8; cout * ldw * 2];
            for o in 0..cout {
                for i in 0..cin {
                    for tap in 0..size * size {
                        let from = ((o * cin + i) * size * size + tap) * 2;
                        let to = (o * ldw + tap * cin_pad + i) * 2;
                        out[to..to + 2].copy_from_slice(&raw[from..from + 2]);
                    }
                }
            }
            Cow::Owned(out)
        });
        let bias = self.bf16_vector(src, &format!("{name}.bias"), cout)?;
        let (buf, at) = self.place(cout * ldw * 2, out);
        Ok(Mat {
            buf,
            at,
            n: cout,
            k,
            ldw,
            format: Format::Bf16,
            bias: Some(bias),
        })
    }

    /// A linear layer (`name.weight` `[n][k]` bf16, `name.bias`) as a 1 × 1
    /// convolution.
    pub(crate) fn linear(
        &mut self,
        src: Source<'a>,
        name: &str,
        (k, n): (usize, usize),
    ) -> Result<Mat, String> {
        let mut m = self.bf16(src, &format!("{name}.weight"), (n, k))?;
        m.bias = Some(self.bf16_vector(src, &format!("{name}.bias"), n)?);
        Ok(m)
    }

    /// The bytes it takes on the GPU: its buffers of matrices and its f32
    /// tensors ([`Loaded::bytes`] once uploaded).
    pub(crate) fn bytes(&self) -> u64 {
        let whole = |floats: usize| (super::gpu::whole_vectors(floats).max(1) * 4) as u64;
        (self.buffers.iter())
            .map(|b| whole((b.0 / 4) as usize))
            .sum::<u64>()
            + whole(self.small.len())
    }

    /// Everything on the GPU (writes sent a few hundred MB at a time); an
    /// error for a packing of shapes alone.
    pub(crate) fn upload(self, label: &str) -> Result<Loaded, String> {
        let gpu = Gpu::shared()?;
        let program = &gpu.program;
        let (matrices, small) = crate::engine::guarded(&program.device, || {
            let mut upload = program.upload();
            let mut matrices = Vec::new();
            for (i, (len, pieces)) in self.buffers.iter().enumerate() {
                let b = gpu.buffer(&format!("{label} matrices {i}"), *len)?;
                for (at, data) in pieces {
                    let data = data.as_ref().ok_or("weights of shapes alone")?;
                    upload.write(&b, *at, data)?;
                }
                matrices.push(b);
            }
            let small = gpu.buffer(
                &format!("{label} tensors"),
                (self.small.len().max(1) * 4) as u64,
            )?;
            upload.write(&small, 0, &crate::engine::bytes(&self.small))?;
            upload.finish()?;
            Ok((matrices, small))
        })?;
        Ok(Loaded {
            gpu,
            matrices,
            small,
        })
    }
}
