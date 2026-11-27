//! The language model's kernels: through naga's validation and its Metal
//! writer (on any machine), and each on small shapes against a
//! reference on the processor, on both paths and for every kind of
//! matrix (where there is a usable GPU: skipped with a note where there
//! is none). The blocks' reference is llama.cpp's `dequantize_row_*`, as
//! written there.

use super::kernels::{PLAIN_KINDS, Q8_KINDS, TOKENS};
use super::model::Matrix;
use super::*;
use crate::engine::{params, Kernels, Op, PARAM_STRIDE};
use jodel_tensors::gguf::Kind;
use jodel_tensors::half::{bf16_to_f32, f16_to_f32, f32_to_f16};

/// The kernels parse and validate, and Metal's writer takes them (as wgpu
/// would, on a Mac).
#[test]
fn the_kernels_validate_and_write_for_metal() {
    let entries: Vec<&str> = Kernel::ALL.iter().map(|k| k.entry()).collect();
    crate::engine::check_source(
        &source(),
        naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        &entries,
    );
}

/// A small generator of values in [-1, 1).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// The program on the GPU, for single dispatches on buffers of the test's.
struct Rig {
    program: Program<Kernel>,
}

/// What a dispatch binds (src, aux, dst, wq, ints) and what it left in
/// dst and ints.
#[derive(Clone, Default)]
struct Data {
    src: Vec<f32>,
    aux: Vec<f32>,
    dst: Vec<f32>,
    wq: Vec<u32>,
    ints: Vec<u32>,
}

fn rig() -> Option<Rig> {
    match crate::device::open("jodel-gpu gemma test", |device, queue, limits| {
        Ok(Program::new(
            device,
            queue,
            limits,
            "jodel-gpu gemma test",
            &source(),
        ))
    }) {
        Ok((program, _)) => Some(Rig { program }),
        Err(e) => {
            eprintln!("no usable GPU ({e}): skipped");
            None
        }
    }
}

impl Rig {
    /// `kernel` with `numbers` over `groups` on `data`; what it leaves.
    fn run(&self, kernel: Kernel, numbers: &[usize], groups: [u32; 3], data: &Data) -> Data {
        let p = &self.program;
        let make = |words: &[u8]| {
            let b = p.storage("test", words.len().max(4) / 4).unwrap();
            p.queue.write_buffer(&b, 0, words);
            b
        };
        let bufs = [
            make(&bytes(&data.src)),
            make(&bytes(&data.aux)),
            make(&bytes(&data.dst)),
            make(&bytes(&data.wq)),
            make(&bytes(&data.ints)),
        ];
        let op: Op<Kernel, (), 16> = Op {
            kernel,
            buffers: (),
            params: params(numbers),
            groups,
        };
        let recorded = p.record(&[op], |()| bufs.iter().collect()).unwrap();
        let mut pieces = crate::pace::Pieces::foreground();
        p.run(&mut pieces, &recorded, &[], |_| {}, None).unwrap();
        let read = |b: &wgpu::Buffer, n: usize| {
            let staging = p.staging(n);
            let mut encoder = p.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(b, 0, &staging, 0, (n * 4) as u64);
            p.queue.submit([encoder.finish()]);
            crate::device::read_back(&p.device, &staging, n).unwrap()
        };
        assert_eq!(PARAM_STRIDE, 256);
        Data {
            dst: read(&bufs[2], data.dst.len().max(1)),
            ints: read(&bufs[4], data.ints.len().max(1))
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            ..data.clone()
        }
    }
}

/// The half at `b[i..]`.
fn half(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// A Q4_K or Q5_K block's sub-block `j`'s scale and minimum (llama.cpp's
/// `get_scale_min_k4`).
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        (f32::from(q[j] & 63), f32::from(q[j + 4] & 63))
    } else {
        (
            f32::from((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
            f32::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
        )
    }
}

/// The values of one block `b` of `kind`.
fn dequantise(kind: Kind, b: &[u8]) -> Vec<f32> {
    let mut y = vec![0.0f32; kind.block().0];
    match kind {
        Kind::F32 => y[0] = f32::from_le_bytes(b[..4].try_into().unwrap()),
        Kind::Bf16 => y[0] = bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
        Kind::Q4K | Kind::Q5K => {
            let (d, min) = (half(b, 0), half(b, 2));
            let scales = &b[4..16];
            let (qh, qs) = if kind == Kind::Q5K {
                (&b[16..48], &b[48..176])
            } else {
                (&b[..0], &b[16..144])
            };
            let fifth = |l: usize, bit: usize| {
                if qh.is_empty() || qh[l] & (1 << bit) == 0 {
                    0.0
                } else {
                    16.0
                }
            };
            for j in 0..4 {
                let (sc1, m1) = scale_min_k4(2 * j, scales);
                let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
                let (d1, m1, d2, m2) = (d * sc1, min * m1, d * sc2, min * m2);
                for l in 0..32 {
                    let q = qs[32 * j + l];
                    y[64 * j + l] = d1 * (f32::from(q & 0xF) + fifth(l, 2 * j)) - m1;
                    y[64 * j + 32 + l] = d2 * (f32::from(q >> 4) + fifth(l, 2 * j + 1)) - m2;
                }
            }
        }
        Kind::Q6K => {
            let d = half(b, 208);
            for n in 0..2 {
                let (ql, qh) = (&b[64 * n..], &b[128 + 32 * n..]);
                let sc = |i: usize| f32::from(b[192 + 8 * n + i] as i8);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                    let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[128 * n + l] = d * sc(is) * q1 as f32;
                    y[128 * n + l + 32] = d * sc(is + 2) * q2 as f32;
                    y[128 * n + l + 64] = d * sc(is + 4) * q3 as f32;
                    y[128 * n + l + 96] = d * sc(is + 6) * q4 as f32;
                }
            }
        }
    }
    y
}

/// A random block of `kind`: random bytes, its scales small halves (as
/// a quantiser would write them for weights of about a tenth), or random
/// values.
fn block(noise: &mut Noise, kind: Kind) -> Vec<u8> {
    let bytes = kind.block().1;
    let mut b: Vec<u8> = (0..bytes)
        .map(|_| ((noise.next() + 1.0) * 128.0) as u8)
        .collect();
    let put_half = |b: &mut [u8], at: usize, v: f32| {
        b[at..at + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
    };
    match kind {
        Kind::F32 => b.copy_from_slice(&(noise.next() * 0.2).to_le_bytes()),
        Kind::Bf16 => {
            let bits = (noise.next() * 0.2).to_bits() >> 16;
            b.copy_from_slice(&(bits as u16).to_le_bytes());
        }
        Kind::Q4K | Kind::Q5K => {
            put_half(&mut b, 0, 0.0002 + noise.next().abs() * 0.0002);
            put_half(&mut b, 2, 0.0002 + noise.next().abs() * 0.0002);
        }
        Kind::Q6K => put_half(&mut b, 208, 0.00005 + noise.next().abs() * 0.00005),
    }
    b
}

/// A random matrix of `rows × cols` of `kind` at word `at` of a buffer:
/// its words, its place, and the values it holds.
fn matrix(
    noise: &mut Noise,
    kind: Kind,
    (rows, cols): (usize, usize),
    at: usize,
) -> (Vec<u32>, Matrix, Vec<f32>) {
    let m = Matrix {
        at,
        rows,
        cols,
        kind,
    };
    let mut bytes = vec![0u8; at * 4];
    let mut held = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols / kind.block().0 {
        let b = block(noise, kind);
        held.extend(dequantise(kind, &b));
        bytes.extend(b);
    }
    bytes.resize(bytes.len().next_multiple_of(4) + 4, 0);
    let words = bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    (words, m, held)
}

/// Every kind a matrix multiply takes, with its paths (q8 where it has
/// one).
fn kinds_and_paths() -> Vec<(Kind, bool)> {
    PLAIN_KINDS
        .iter()
        .flat_map(|&k| {
            let q8 = Q8_KINDS.contains(&k).then_some((k, true));
            q8.into_iter().chain([(k, false)])
        })
        .collect()
}

/// A Q4_K, Q5_K and Q6_K block of known values: llama.cpp's layout as its
/// reference reads it, on a block written by hand.
#[test]
fn blocks_hold_what_llama_cpp_reads_from_them() {
    // Q4_K: d 1, dmin 0.5, sub-block 0 scale 2 and minimum 3, sub-block 5
    // (scales' high bits in bytes 1 and 5, low ones in byte 9) scale 33
    // and minimum 18; value 0 a 7 (low nibble of byte 0), value 32 + 2 of
    // sub-block 5 (chunk 2, high nibble of its byte 2) a 9.
    let mut b = vec![0u8; 144];
    b[..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    b[2..4].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    b[4] = 2;
    b[8] = 3;
    b[4 + 9] = (33 & 15) | ((18 & 15) << 4);
    b[4 + 1] = (33 >> 4) << 6;
    b[4 + 5] = (18 >> 4) << 6;
    b[16] = 7;
    b[16 + 64 + 2] = 9 << 4;
    let y = dequantise(Kind::Q4K, &b);
    assert_eq!(
        (y[0], y[1], y[5 * 32 + 2]),
        (14.0 - 1.5, -1.5, 33.0 * 9.0 - 9.0)
    );
    // Q5_K: the same, value 0's fifth bit set (bit 0 of qh's byte 0).
    let mut b5 = b[..16].to_vec();
    b5.extend([1u8].iter().chain(&[0u8; 31]));
    b5.extend(&b[16..]);
    let y = dequantise(Kind::Q5K, &b5);
    assert_eq!(y[0], 2.0 * 23.0 - 1.5);
    // Q6_K: d 0.5; value 64 + 3 of the first half (high nibble of ql's
    // byte 3, qh's byte 3 bits 4 and 5) of 6 bits 0b10_0101 = 37, its
    // scale (index 4) -3: 0.5 · -3 · (37 − 32).
    let mut b = vec![0u8; 210];
    b[208..].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    b[3] = 0b0101 << 4;
    b[128 + 3] = 0b10 << 4;
    b[192 + 4] = (-3i8) as u8;
    assert_eq!(dequantise(Kind::Q6K, &b)[67], -7.5);
}

/// Rows as the GPU writes them for a matrix multiply (`put_block`): the
/// words (values, then scales from `qs`) and the values they stand for.
fn written(rows: &[f32], q8: bool, qs: usize) -> (Vec<u32>, Vec<f32>) {
    let mut words = vec![0u32; qs + rows.len() / 32 * 2];
    if !q8 {
        for (w, v) in words.iter_mut().zip(rows) {
            *w = v.to_bits();
        }
        return (words, rows.to_vec());
    }
    let mut held = Vec::with_capacity(rows.len());
    for (b, block) in rows.chunks_exact(32).enumerate() {
        let most = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = most / 127.0;
        let q: Vec<i32> = block
            .iter()
            .map(|&v| {
                if d > 0.0 {
                    (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
                } else {
                    0
                }
            })
            .collect();
        for (w, four) in q.chunks_exact(4).enumerate() {
            words[b * 8 + w] = four
                .iter()
                .rev()
                .fold(0u32, |acc, &n| (acc << 8) | (n as u32 & 255));
        }
        words[qs + 2 * b] = d.to_bits();
        words[qs + 2 * b + 1] = (d * q.iter().sum::<i32>() as f32).to_bits();
        held.extend(q.iter().map(|&n| n as f32 * d));
    }
    (words, held)
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32 + 1e-6).powf(-0.5)
}

/// The largest difference relative to the largest reference value.
fn off(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-12f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

const PATHS: [bool; 2] = [true, false];

#[test]
fn rows_written_for_a_matrix_multiply_are_the_references() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(1);
    let (rows, k) = (3, 96);
    let x = noise.values(rows * k, 3.0);
    let times = noise.values(rows * k, 1.0);
    for q8 in PATHS {
        let qs = rows * k;
        let product: Vec<f32> = x.iter().zip(&times).map(|(a, b)| a * b).collect();
        let (want, _) = written(&product, q8, qs);
        let data = Data {
            src: [x.clone(), times.clone()].concat(),
            dst: vec![0.0],
            ints: vec![0; want.len()],
            ..Data::default()
        };
        let got = rig.run(
            Kernel::Quant,
            &[rows, k, k, 0, qs, usize::from(q8), rows * k, k],
            crate::engine::spread(rows * k / 32),
            &data,
        );
        assert_eq!(got.ints, want, "q8 {q8}");
    }
}

/// Each kind's rows of 768 (three blocks: a Q6_K row's second block, and
/// every odd row, start halfway into a word), on each of its paths; 48
/// sixteens a row over 32 invocations leave a tail.
#[test]
fn a_row_through_a_matrix_is_the_reference_for_every_kind_and_tail() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(2);
    let (n, k, at) = (64, 768, 6);
    let x = noise.values(k, 2.0);
    for (kind, q8) in kinds_and_paths() {
        let (wq, m, held) = matrix(&mut noise, kind, (n, k), at);
        let qs = k;
        let (ints, xs) = written(&x, q8, qs);
        let products: Vec<f32> = (0..n)
            .map(|r| (0..k).map(|c| held[r * k + c] * xs[c]).sum())
            .collect();
        for (flags, want) in [
            (0, products.clone()),
            (4, products.iter().map(|&v| gelu(v)).collect::<Vec<_>>()),
            (512, products.chunks(2).map(|p| gelu(p[0]) * p[1]).collect()),
            (
                128,
                products.iter().map(|v| (v / 30.0).tanh() * 30.0).collect(),
            ),
        ] {
            let best = ints.len();
            let data = Data {
                dst: vec![0.0; n],
                wq: wq.clone(),
                ints: [ints.clone(), vec![0; 2 * n / 16]].concat(),
                ..Data::default()
            };
            let kernel = if q8 {
                Kernel::GemvQ8(kind)
            } else {
                Kernel::GemvF32(kind)
            };
            let got = rig.run(
                kernel,
                &[1, n, k, 0, k, m.at, 0, n, 0, flags, qs, best],
                [(n / 16) as u32, 1, 1],
                &data,
            );
            let e = off(&got.dst[..want.len()], &want);
            assert!(e < 1e-5, "{kind:?}, q8 {q8}, flags {flags}: {e}");
            if flags == 128 {
                for g in 0..n / 16 {
                    let part = &want[g * 16..g * 16 + 16];
                    let i = (0..16).fold(0, |b, i| if part[i] > part[b] { i } else { b });
                    assert_eq!(got.ints[best + 2 * g + 1] as usize, g * 16 + i);
                }
            }
        }
    }
}

#[test]
fn rows_through_a_matrix_are_the_reference_for_every_kind() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(3);
    let (rows, n, k, at) = (70, 128, 768, 2);
    let x = noise.values(rows * k, 2.0);
    for (kind, q8) in kinds_and_paths() {
        let (wq, m, held) = matrix(&mut noise, kind, (n, k), at);
        let qs = rows * k;
        let (ints, xs) = written(&x, q8, qs);
        let products: Vec<f32> = (0..rows * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                (0..k).map(|j| held[c * k + j] * xs[r * k + j]).sum()
            })
            .collect();
        for (flags, width) in [(0, n), (4, n), (512, n / 2)] {
            let want: Vec<f32> = match flags {
                512 => products.chunks(2).map(|p| gelu(p[0]) * p[1]).collect(),
                4 => products.iter().map(|&v| gelu(v)).collect(),
                _ => products.clone(),
            };
            let data = Data {
                dst: vec![0.0; rows * width],
                wq: wq.clone(),
                ints: ints.clone(),
                ..Data::default()
            };
            let kernel = if q8 {
                Kernel::GemmQ8(kind)
            } else {
                Kernel::GemmF32(kind)
            };
            let got = rig.run(
                kernel,
                &[rows, n, k, 0, k, m.at, 0, width, 0, flags, qs],
                [(n / 64) as u32, rows.div_ceil(64) as u32, 1],
                &data,
            );
            let e = off(&got.dst, &want);
            assert!(e < 1e-5, "{kind:?}, q8 {q8}, flags {flags}: {e}");
        }
    }
}

#[test]
fn the_residual_step_is_the_reference() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(4);
    let (rows, k) = (2, 320);
    let x = noise.values(rows * k, 4.0);
    let h = noise.values(rows * k, 9.0);
    let w = noise.values(2 * k + 1, 1.5);
    for q8 in PATHS {
        let mut want_x = x.clone();
        let mut out = Vec::new();
        for r in 0..rows {
            let hr = &h[r * k..(r + 1) * k];
            let inv = rms(hr);
            let xr = &mut want_x[r * k..(r + 1) * k];
            for i in 0..k {
                xr[i] = (xr[i] + hr[i] * inv * w[i]) * w[2 * k];
            }
            let inv = rms(xr);
            out.extend((0..k).map(|i| xr[i] * inv * w[k + i]));
        }
        let qs = rows * k;
        let (want, _) = written(&out, q8, qs);
        let data = Data {
            src: h.clone(),
            aux: w.clone(),
            dst: x.clone(),
            ints: vec![0; want.len()],
            ..Data::default()
        };
        let got = rig.run(
            Kernel::Residual,
            &[k, k, 0, 2 * k, k, 0, qs, usize::from(q8), 7, 0],
            [rows as u32, 1, 1],
            &data,
        );
        assert!(off(&got.dst, &want_x) < 1e-6, "q8 {q8}");
        // The written rows' values or scales may round a hair apart.
        let (got_held, want_held) = (
            held(&got.ints, rows * k, q8, qs),
            held(&want, rows * k, q8, qs),
        );
        assert!(off(&got_held, &want_held) < 1e-5, "q8 {q8}");
    }
}

/// The values `words` (as [`written`] lays them out) stand for.
fn held(words: &[u32], n: usize, q8: bool, qs: usize) -> Vec<f32> {
    if !q8 {
        return words[..n].iter().map(|&w| f32::from_bits(w)).collect();
    }
    (0..n)
        .map(|i| {
            let (b, j) = (i / 32, i % 32);
            let byte = (words[b * 8 + j / 4] >> (8 * (j % 4))) as u8 as i8;
            f32::from(byte) * f32::from_bits(words[qs + 2 * b])
        })
        .collect()
}

#[test]
fn embeddings_and_per_layer_inputs_are_the_references() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(5);
    let (vocab, width, at) = (32, 256, 4);
    let tokens = [7u32, 3, 31];
    let mut ints = vec![0u32; TOKENS];
    ints[0] = 1;
    ints.extend(tokens);
    // From position 1 (the decoder's), two rows, from a table of each kind.
    for kind in [Kind::Q4K, Kind::Q5K, Kind::Q6K, Kind::Bf16, Kind::F32] {
        let (wq, m, table) = matrix(&mut noise, kind, (vocab, width), at);
        let data = Data {
            dst: vec![0.0; 2 * width],
            wq,
            ints: ints.clone(),
            ..Data::default()
        };
        let got = rig.run(
            Kernel::Embed,
            &[
                2,
                0,
                1,
                m.at,
                3.0f32.to_bits() as usize,
                width,
                kind.id() as usize,
            ],
            crate::engine::spread(2 * width),
            &data,
        );
        let want: Vec<f32> = [3usize, 31]
            .iter()
            .flat_map(|&t| table[t * width..(t + 1) * width].iter().map(|v| v * 3.0))
            .collect();
        assert!(off(&got.dst, &want) < 1e-6, "{kind:?}");
    }
    // The per-layer inputs of rows 0 and 1 (projection rows 1 and 2), a
    // layer's slice from 256 of 768, its embeddings Q5_K.
    let (wq, m, table) = matrix(&mut noise, Kind::Q5K, (vocab, width), at);
    let (stride, col, scale, embed_scale) = (768, 256, 0.25f32, 2.0f32);
    let pp = noise.values(3 * stride, 3.0);
    let norm = noise.values(width, 1.0);
    let data = Data {
        src: pp.clone(),
        aux: norm.clone(),
        dst: vec![0.0; 2 * width + 5],
        wq,
        ints,
    };
    let got = rig.run(
        Kernel::PerLayer,
        &[
            2,
            0,
            0,
            stride,
            col,
            0,
            m.at,
            5,
            width,
            1,
            scale.to_bits() as usize,
            embed_scale.to_bits() as usize,
            Kind::Q5K.id() as usize,
        ],
        [2, 1, 1],
        &data,
    );
    let mut want = Vec::new();
    for (r, &t) in [7usize, 3].iter().enumerate() {
        let v: Vec<f32> = (0..width)
            .map(|i| pp[(1 + r) * stride + col + i] * scale)
            .collect();
        let inv = rms(&v);
        want.extend((0..width).map(|i| {
            (v[i] * inv * norm[i] + table[t * width + i] * embed_scale)
                * std::f32::consts::FRAC_1_SQRT_2
        }));
    }
    assert!(off(&got.dst[5..], &want) < 1e-6);
}

#[test]
fn the_next_token_is_the_first_largest() {
    let Some(rig) = rig() else { return };
    let count = 700;
    let mut ints = vec![0u32; TOKENS + 8];
    ints[0] = 2;
    let best = ints.len();
    for g in 0..count {
        let v = if g == 400 || g == 650 {
            99.0f32
        } else {
            (g % 13) as f32
        };
        ints.push(v.to_bits());
        ints.push(g as u32 * 16 + 3);
    }
    let data = Data {
        dst: vec![0.0],
        ints,
        ..Data::default()
    };
    let got = rig.run(Kernel::Argmax, &[count, best, 64], [1, 1, 1], &data);
    assert_eq!(got.ints[0], 3);
    assert_eq!(got.ints[TOKENS + 3], 400 * 16 + 3);
}

/// Queries, keys and values of `rows` rows for heads of `d` (8 query
/// heads on 2 key/value heads), as a layer's matrix writes them.
struct Heads {
    d: usize,
    qkv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl Heads {
    fn new(noise: &mut Noise, rows: usize, d: usize) -> Heads {
        Heads {
            d,
            qkv: noise.values(rows * 12 * d, 2.0),
            q_norm: noise.values(d, 1.5),
            k_norm: noise.values(d, 1.5),
        }
    }

    /// Pair `i`'s cosine and sine at `pos` by a table of `base`.
    fn turn(&self, i: usize, pos: usize, base: f32) -> (f32, f32) {
        let f = 1.0 / base.powf((2 * i) as f32 / self.d as f32);
        let a = f64::from(pos as f32 * f);
        (a.cos() as f32, a.sin() as f32)
    }

    /// Head `h` (of 12: queries, keys, values) of row `r` normed and
    /// turned at `pos` by a table of `base` turning the first `turned`
    /// pairs.
    fn prepared(&self, r: usize, h: usize, pos: usize, (base, turned): (f32, usize)) -> Vec<f32> {
        let d = self.d;
        let x = &self.qkv[(r * 12 + h) * d..(r * 12 + h + 1) * d];
        let inv = rms(x);
        let w: Option<&Vec<f32>> = match h {
            0..=7 => Some(&self.q_norm),
            8 | 9 => Some(&self.k_norm),
            _ => None,
        };
        let v: Vec<f32> = (0..d)
            .map(|i| x[i] * inv * w.map_or(1.0, |w| w[i]))
            .collect();
        if w.is_none() {
            return v;
        }
        let half = d / 2;
        let mut out = v.clone();
        for i in 0..turned {
            let (c, s) = self.turn(i, pos, base);
            out[i] = v[i] * c - v[i + half] * s;
            out[i + half] = v[i + half] * c + v[i] * s;
        }
        out
    }

    /// The rotary table for `positions` of heads of `d`, `base`, its first
    /// `turned` pairs.
    fn table(&self, positions: usize, (base, turned): (f32, usize)) -> Vec<u32> {
        let mut t = Vec::new();
        for pos in 0..positions {
            for i in 0..turned {
                let (c, s) = self.turn(i, pos, base);
                t.extend([c.to_bits(), s.to_bits()]);
            }
        }
        t
    }
}

/// Attention of query `q` at `pos` over keys and values (each a
/// position's) back to `window` (0: all).
fn attention(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    pos: usize,
    window: usize,
) -> Vec<f32> {
    let lo = if window > 0 {
        (pos + 1).saturating_sub(window)
    } else {
        0
    };
    let s: Vec<f32> = (lo..=pos)
        .map(|j| q.iter().zip(&keys[j]).map(|(a, b)| a * b).sum())
        .collect();
    let best = s.iter().fold(f32::MIN, |m, &v| m.max(v));
    let e: Vec<f32> = s.iter().map(|v| (v - best).exp()).collect();
    let total: f32 = e.iter().sum();
    let mut out = vec![0.0; q.len()];
    for (j, w) in (lo..=pos).zip(&e) {
        for (o, v) in out.iter_mut().zip(&values[j]) {
            *o += w / total * v;
        }
    }
    out
}

/// Where the cache keeps its keys, values and queries for heads of `d`
/// with `places` a key's value: from its start, the queries after them.
fn layout(d: usize, places: usize) -> (usize, usize, usize) {
    (0, 2 * d * places, 4 * d * places)
}

/// `rows` rows from position `first` prepared into `cache` (as the
/// kernels keep it: `places` a key's value, a position's place masked by
/// `mask`; the queries of these rows after the keys and values), the
/// rotary table of `turn` (its base and the pairs it turns).
fn prepare_into(
    rig: &Rig,
    heads: &Heads,
    (first, rows): (usize, usize),
    (places, mask): (usize, usize),
    turn: (f32, usize),
    cache: Vec<f32>,
) -> Vec<f32> {
    let d = heads.d;
    let (k_at, v_at, q_at) = layout(d, places);
    let data = Data {
        src: heads.qkv[first * 12 * d..(first + rows) * 12 * d].to_vec(),
        aux: [heads.q_norm.clone(), heads.k_norm.clone()].concat(),
        dst: cache,
        wq: heads.table(first + rows, turn),
        ints: vec![0; TOKENS],
    };
    let numbers = [
        rows,
        first,
        0,
        d,
        12 * d,
        0,
        d,
        q_at,
        k_at,
        v_at,
        places,
        0,
        2,
        8,
        mask,
        turn.1,
    ];
    rig.run(Kernel::Prep, &numbers, [rows as u32, 12, 1], &data)
        .dst
}

/// The mask of a cache with a place for every position (a global layer's).
const ALL_BITS: usize = u32::MAX as usize;

/// A cache for `rows` rows of heads of `d` with `places` a key's value.
fn empty_cache(d: usize, places: usize, rows: usize) -> Vec<f32> {
    vec![0.0; layout(d, places).2 + rows * 8 * d]
}

/// Every pair of heads of `d` turned, at base 10⁴.
fn all_turned(d: usize) -> (f32, usize) {
    (10_000.0, d / 2)
}

#[test]
fn queries_keys_and_values_go_normed_and_turned_to_their_places() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(6);
    let (rows, d, positions) = (3, 128, 8);
    let heads = Heads::new(&mut noise, rows, d);
    // Every pair turned, and (as a global layer's) the first quarter.
    for turn in [all_turned(d), (1e6, d / 8)] {
        let cache = empty_cache(d, positions, rows);
        let got = prepare_into(&rig, &heads, (0, rows), (positions, ALL_BITS), turn, cache);
        let (_, v_at, q_at) = layout(d, positions);
        for r in 0..rows {
            for h in 0..12 {
                let want = heads.prepared(r, h, r, turn);
                let have: Vec<f32> = (0..d)
                    .map(|i| match h {
                        0..=7 => got[q_at + (r * 8 + h) * d + i],
                        8 | 9 => got[((h - 8) * d + i) * positions + r],
                        _ => got[v_at + ((h - 10) * positions + r) * d + i],
                    })
                    .collect();
                assert!(off(&have, &want) < 1e-5, "row {r}, head {h}, {turn:?}");
            }
        }
    }
}

/// The step's attention at `last` (its queries the first row's place in
/// `cache`, as they were written) in `splits` splits (`least` at work, or
/// one every `per` chunks where that is more), joined, against `want` (its
/// eight heads') on both paths.
fn step_is(
    rig: &Rig,
    (d, window): (usize, usize),
    (places, mask): (usize, usize),
    (cache, last): (Vec<f32>, usize),
    (splits, least, per): (usize, usize, usize),
    want: &[f32],
) {
    let (k_at, v_at, q_at) = layout(d, places);
    let mut ints = vec![0u32; TOKENS];
    ints[0] = last as u32;
    let data = Data {
        src: cache,
        dst: vec![0.0; 8 * splits * (d + 2)],
        ints,
        ..Data::default()
    };
    let numbers = [
        d, window, q_at, k_at, v_at, places, splits, mask, least, per,
    ];
    let parts = rig.run(Kernel::Attend, &numbers, [splits as u32, 2, 1], &data);
    for q8 in PATHS {
        let qs = 8 * d;
        // The position first, as the state has it (the rows go over it).
        let mut ints = vec![0; qs + 8 * d / 32 * 2];
        ints[0] = last as u32;
        let data = Data {
            src: parts.dst.clone(),
            dst: vec![0.0],
            ints,
            ..Data::default()
        };
        let got = rig.run(
            Kernel::Join,
            &[d, splits, 0, qs, usize::from(q8), 8, window, least, per],
            [8, 1, 1],
            &data,
        );
        let (want_words, _) = written(want, q8, qs);
        let e = off(
            &held(&got.ints, 8 * d, q8, qs),
            &held(&want_words, 8 * d, q8, qs),
        );
        assert!(
            e < 1e-5,
            "step: window {window}, places {places}, q8 {q8}: {e}"
        );
    }
}

/// A prompt's attention over `rows` rows from `first` (their queries in
/// `cache`), through flash.
fn flash(
    rig: &Rig,
    (d, window): (usize, usize),
    (places, mask): (usize, usize),
    (cache, first, rows): (&[f32], usize, usize),
) -> Vec<f32> {
    let (k_at, v_at, q_at) = layout(d, places);
    let data = Data {
        src: cache.to_vec(),
        dst: vec![0.0; rows * 8 * d],
        ints: vec![0; TOKENS],
        ..Data::default()
    };
    let per_block = 256 / (d / 16);
    let numbers = [rows, first, d, window, q_at, k_at, v_at, places, 8, 4, mask];
    let groups = [rows.div_ceil(per_block) as u32, 8, 1];
    rig.run(Kernel::Flash, &numbers, groups, &data).dst
}

#[test]
fn a_steps_attention_and_a_prompts_are_the_reference() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(7);
    let (rows, d, positions) = (150, 128, 160);
    let heads = Heads::new(&mut noise, rows, d);
    let turn = all_turned(d);
    let whole = (positions, ALL_BITS);
    let cache = prepare_into(
        &rig,
        &heads,
        (0, rows),
        whole,
        turn,
        empty_cache(d, positions, rows),
    );
    let q_at = layout(d, positions).2;
    let kv = |g: usize, kind: usize| -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| heads.prepared(r, 8 + 2 * kind + g, r, turn))
            .collect()
    };
    for window in [0, 40] {
        let want = |r: usize, h: usize| {
            let q = heads.prepared(r, h, r, turn);
            attention(&q, &kv(h / 4, 0), &kv(h / 4, 1), r, window)
        };
        // A prompt's: all rows, from row 0, through flash.
        let got = flash(&rig, (d, window), whole, (&cache, 0, rows));
        for r in [0, 1, 39, 40, 41, 100, rows - 1] {
            for h in [0, 3, 4, 7] {
                let have = &got[(r * 8 + h) * d..(r * 8 + h + 1) * d];
                assert!(
                    off(have, &want(r, h)) < 1e-5,
                    "flash: window {window}, row {r}, head {h}"
                );
            }
        }
        // A step's at the last row (its queries the first row's place).
        let last = rows - 1;
        let mut src = cache.clone();
        src.copy_within(q_at + last * 8 * d..q_at + rows * 8 * d, q_at);
        // Splits idle (the window's one chunk), and one taking two chunks
        // (of 3); then as many as the chunks, or one every two, of more
        // dispatched (the rest idle).
        let want_all: Vec<f32> = (0..8).flat_map(|h| want(last, h)).collect();
        let step = |splits| {
            step_is(
                &rig,
                (d, window),
                whole,
                (src.clone(), last),
                splits,
                &want_all,
            )
        };
        if window > 0 {
            step((3, 3, 1));
        } else {
            step((2, 2, 1));
            step((4, 1, 1));
            step((4, 1, 2));
        }
    }
}

/// A sliding layer's keys and values in a ring of 64 places (its window
/// of 40 and chunks of 24 rows), its rows prepared and attended a chunk at
/// a time as a prompt's are, then a step at the last: each the reference
/// over every position, the ring wrapped many times (the step's window
/// across its end).
#[test]
fn a_ring_holds_what_the_window_reads() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(8);
    let (rows, d, window, chunk) = (281, 128, 40, 24);
    let ring = (64, 63);
    let heads = Heads::new(&mut noise, rows, d);
    let turn = all_turned(d);
    let kv = |g: usize, kind: usize| -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| heads.prepared(r, 8 + 2 * kind + g, r, turn))
            .collect()
    };
    let want = |r: usize, h: usize| {
        let q = heads.prepared(r, h, r, turn);
        attention(&q, &kv(h / 4, 0), &kv(h / 4, 1), r, window)
    };
    let mut cache = empty_cache(d, ring.0, chunk);
    for first in (0..rows).step_by(chunk) {
        let n = chunk.min(rows - first);
        cache = prepare_into(&rig, &heads, (first, n), ring, turn, cache);
        let got = flash(&rig, (d, window), ring, (&cache, first, n));
        for r in first..first + n {
            for h in [0, 5] {
                let have = &got[((r - first) * 8 + h) * d..((r - first) * 8 + h + 1) * d];
                assert!(
                    off(have, &want(r, h)) < 1e-5,
                    "flash in a ring: row {r}, head {h}"
                );
            }
        }
    }
    // The last row as a step would have it: prepared alone at its position.
    let last = rows - 1;
    let cache = prepare_into(&rig, &heads, (last, 1), ring, turn, cache);
    let want_all: Vec<f32> = (0..8).flat_map(|h| want(last, h)).collect();
    step_is(&rig, (d, window), ring, (cache, last), (3, 3, 1), &want_all);
}
