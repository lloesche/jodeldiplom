//! A part's weights as they go to the GPU ([`Packing`]): matrices placed
//! in buffers of words (FP8 bytes as the checkpoint holds them, bf16 two a
//! word), with their scales, biases and other f32 tensors beside them; then
//! uploaded ([`Packing::upload`]).

use std::borrow::Cow;

use jodel_tensors::SafeTensors;

use super::gpu::{Gpu, Loaded};
use super::plan::{Buf, Format, Mat};
use crate::engine::{bytes, guarded};
use crate::Turned;

/// The most bytes in one buffer of matrices.
const MOST: u64 = 1 << 30;
/// Bytes written to the GPU between waits (wgpu stages each write until
/// it is sent).
const BATCH: u64 = 256 << 20;

/// An FP8 matrix as the checkpoint holds it: `n` rows of `k` bytes, and a
/// scale for each 64 rows and 128 values of a row (`n / 64 × ⌈k / 128⌉`).
pub(crate) struct Fp8<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) scales: Vec<f32>,
}

impl<'a> Fp8<'a> {
    /// `name.weight` (`n × k`) with its one scale `name.weight_scale`
    /// (BFL's checkpoints).
    pub(crate) fn per_tensor(
        t: &'a SafeTensors,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Fp8<'a>, String> {
        let bytes = fp8_bytes(t, &format!("{name}.weight"), (n, k))?;
        let scale = t.get(&format!("{name}.weight_scale"), &[])?[0];
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
        t: &'a SafeTensors,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Fp8<'a>, String> {
        let bytes = fp8_bytes(t, &format!("{name}.weight"), (n, k))?;
        let kb = k.div_ceil(128);
        let blocks = t.get(&format!("{name}.weight_scale_inv"), &[n.div_ceil(128), kb])?;
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
            bytes: &self.bytes[from * self.k..(from + n) * self.k],
            n,
            k: self.k,
            scales: self.scales[from / 64 * kb..(from + n) / 64 * kb].to_vec(),
        }
    }
}

fn fp8_bytes<'a>(
    t: &'a SafeTensors,
    name: &str,
    (n, k): (usize, usize),
) -> Result<&'a [u8], String> {
    let (dtype, shape, bytes) = t.raw(name)?;
    if dtype != "F8_E4M3" || shape != [n, k] {
        return Err(format!(
            "{name} is {dtype} {shape:?}, not F8_E4M3 [{n}, {k}]"
        ));
    }
    if n % 64 != 0 || k % 32 != 0 {
        return Err(format!(
            "{name}: {n} × {k} is not in rows of 64 and values of 32"
        ));
    }
    Ok(bytes)
}

/// A bf16 tensor's bytes, its shape checked.
pub(crate) fn bf16_bytes<'a>(
    t: &'a SafeTensors,
    name: &str,
    shape: &[usize],
) -> Result<&'a [u8], String> {
    let (dtype, s, bytes) = t.raw(name)?;
    if dtype != "BF16" || s != shape {
        return Err(format!("{name} is {dtype} {s:?}, not BF16 {shape:?}"));
    }
    Ok(bytes)
}

/// A buffer's pieces: each one's offset and bytes.
type Pieces<'a> = Vec<(u64, Cow<'a, [u8]>)>;

/// A part's weights before they go up: the matrices' buffers (each a list
/// of pieces at their offsets) and the f32 tensors.
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

    /// `data` placed in a buffer (16-byte aligned): its buffer and its
    /// first vector of four words.
    fn place(&mut self, data: Cow<'a, [u8]>) -> (Buf, usize) {
        let len = (data.len() as u64).next_multiple_of(16);
        if self.buffers.last().is_none_or(|b| b.0 + len > MOST) {
            self.buffers.push((0, Vec::new()));
        }
        let i = self.buffers.len() - 1;
        let b = &mut self.buffers[i];
        let at = b.0;
        b.0 += len;
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
        t: &SafeTensors,
        name: &str,
        len: usize,
    ) -> Result<usize, String> {
        let v = t.get(name, &[len])?;
        Ok(self.vector(&v))
    }

    /// FP8 bytes (`n` rows of `k`) with their scales, placed.
    fn fp8_placed(&mut self, bytes: Cow<'a, [u8]>, (n, k): (usize, usize), scales: &[f32]) -> Mat {
        let scales = self.vector(scales);
        let (buf, at) = self.place(bytes);
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
        self.fp8_placed(Cow::Borrowed(w.bytes), (w.n, w.k), &w.scales)
    }

    /// FP8 matrices of the same width, one under another.
    pub(crate) fn fp8_stack(&mut self, parts: &[Fp8<'a>]) -> Mat {
        let k = parts[0].k;
        assert!(parts.iter().all(|p| p.k == k));
        let bytes = parts.iter().map(|p| p.bytes).collect::<Vec<_>>().concat();
        let scales: Vec<f32> = parts
            .iter()
            .flat_map(|p| p.scales.iter().copied())
            .collect();
        let n = parts.iter().map(|p| p.n).sum();
        self.fp8_placed(Cow::Owned(bytes), (n, k), &scales)
    }

    /// Gates and their ups (two FP8 matrices of the same shape) as one, 64
    /// rows of gates then their 64 ups (SwiGLU's pairs: `Tail::SwiGlu`).
    pub(crate) fn fp8_pairs(&mut self, gate: &Fp8<'a>, up: &Fp8<'a>) -> Mat {
        assert!(gate.n == up.n && gate.k == up.k);
        let (n, k) = (gate.n, gate.k);
        let kb = k.div_ceil(128);
        let mut bytes = Vec::with_capacity(2 * n * k);
        let mut scales = Vec::with_capacity(2 * gate.scales.len());
        for g in 0..n / 64 {
            for part in [gate, up] {
                bytes.extend_from_slice(&part.bytes[g * 64 * k..(g + 1) * 64 * k]);
                scales.extend_from_slice(&part.scales[g * kb..(g + 1) * kb]);
            }
        }
        self.fp8_placed(Cow::Owned(bytes), (2 * n, k), &scales)
    }

    /// A bf16 matrix (`n × k`, k a multiple of 32) as it is.
    pub(crate) fn bf16(
        &mut self,
        t: &'a SafeTensors,
        name: &str,
        (n, k): (usize, usize),
    ) -> Result<Mat, String> {
        assert_eq!(k % 32, 0);
        let bytes = bf16_bytes(t, name, &[n, k])?;
        let (buf, at) = self.place(Cow::Borrowed(bytes));
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
        t: &SafeTensors,
        name: &str,
        (cin, cout, size): (usize, usize, usize),
        cin_pad: usize,
    ) -> Result<Mat, String> {
        let raw = bf16_bytes(t, &format!("{name}.weight"), &[cout, cin, size, size])?;
        let k = size * size * cin_pad;
        let ldw = k.next_multiple_of(32);
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
        let bias = self.bf16_vector(t, &format!("{name}.bias"), cout)?;
        let (buf, at) = self.place(Cow::Owned(out));
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
        t: &'a SafeTensors,
        name: &str,
        (k, n): (usize, usize),
    ) -> Result<Mat, String> {
        let mut m = self.bf16(t, &format!("{name}.weight"), (n, k))?;
        m.bias = Some(self.bf16_vector(t, &format!("{name}.bias"), n)?);
        Ok(m)
    }

    /// Everything on the GPU (writes sent a few hundred MB at a time).
    pub(crate) fn upload(self, label: &str) -> Result<Loaded, String> {
        let gpu: std::sync::Arc<Turned<Gpu>> = Gpu::shared()?;
        let program = &gpu.program;
        let (matrices, small) = guarded(&program.device, || {
            let mut matrices = Vec::new();
            let mut sent = 0u64;
            for (i, (len, pieces)) in self.buffers.iter().enumerate() {
                let b = gpu.buffer(&format!("{label} matrices {i}"), *len)?;
                for (at, data) in pieces {
                    let whole = data.len() & !3;
                    program.queue.write_buffer(&b, *at, &data[..whole]);
                    if whole < data.len() {
                        let mut tail = [0u8; 4];
                        tail[..data.len() - whole].copy_from_slice(&data[whole..]);
                        program.queue.write_buffer(&b, *at + whole as u64, &tail);
                    }
                    sent += data.len() as u64;
                    if sent >= BATCH {
                        program.queue.submit([]);
                        program.wait()?;
                        sent = 0;
                    }
                }
                matrices.push(b);
            }
            let small = gpu.buffer(
                &format!("{label} tensors"),
                (self.small.len().max(1) * 4) as u64,
            )?;
            program.queue.write_buffer(&small, 0, &bytes(&self.small));
            program.queue.submit([]);
            program.wait()?;
            Ok((matrices, small))
        })?;
        Ok(Loaded {
            gpu,
            matrices,
            small,
        })
    }
}
