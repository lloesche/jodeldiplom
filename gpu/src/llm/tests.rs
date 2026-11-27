//! The language model's kernels: through naga's validation and its Metal
//! and HLSL writers (on any machine), and each on small shapes against a
//! reference on the processor, on both paths (where there is a usable GPU:
//! skipped with a note where there is none).

use super::pack::{put, Matrix};
use super::plan::TOKENS;
use super::*;
use crate::engine::{params, Op, PARAM_STRIDE};
use jodel_tensors::half::f16_to_f32;

/// The kernels parse and validate, and Metal's and DirectX's writers take
/// them (as wgpu would, on a Mac and through DirectX 12).
#[test]
fn the_kernels_validate_and_write_for_metal_and_directx() {
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
        p.run(
            &mut pieces,
            (&recorded.groups, &recorded.dispatches),
            |_| {},
            None,
            &p.staging(1),
        )
        .unwrap();
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

/// `values` (`rows × cols`) as a 4-bit matrix at word `at` of a buffer,
/// and what it holds (dequantised).
fn matrix(values: &[f32], rows: usize, cols: usize, at: usize) -> (Vec<u32>, Matrix, Vec<f32>) {
    let m = Matrix { at, rows, cols };
    let mut words = vec![0u32; at + rows * cols / 32 * 5 + 4];
    put(&mut words, m, values);
    let mut held = vec![0.0; rows * cols];
    for g in 0..rows * cols / 32 {
        let s = words[m.scales() + g];
        let (scale, offset) = (f16_to_f32(s as u16), f16_to_f32((s >> 16) as u16));
        for j in 0..32 {
            let n = (words[at + g * 4 + j / 8] >> (4 * (j % 8))) & 15;
            held[g * 32 + j] = offset + n as f32 * scale;
        }
    }
    (words, m, held)
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
        for w in 0..4 {
            let o = w * 8;
            let pack = |a: [usize; 4]| {
                a.iter().enumerate().fold(0u32, |acc, (i, &j)| {
                    acc | ((q[o + j] as u32 & 255) << (8 * i))
                })
            };
            words[b * 8 + 2 * w] = pack([0, 2, 4, 6]);
            words[b * 8 + 2 * w + 1] = pack([1, 3, 5, 7]);
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
            &[rows, k, 0, k, 0, qs, usize::from(q8), rows * k, k],
            crate::engine::spread(rows * k / 32),
            &data,
        );
        assert_eq!(got.ints, want, "q8 {q8}");
    }
}

#[test]
fn a_row_through_a_matrix_is_the_reference_for_every_tail() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(2);
    let (n, k, at) = (64, 160, 6);
    let (wq, m, held) = matrix(&noise.values(n * k, 0.2), n, k, at);
    let x = noise.values(k, 2.0);
    for q8 in PATHS {
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
            let kernel = if q8 { Kernel::GemvQ8 } else { Kernel::GemvF32 };
            let got = rig.run(
                kernel,
                &[1, n, k, 0, k, m.at, 0, n, 0, flags, qs, best],
                [(n / 16) as u32, 1, 1],
                &data,
            );
            let e = off(&got.dst[..want.len()], &want);
            assert!(e < 1e-5, "q8 {q8}, flags {flags}: {e}");
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
fn rows_through_a_matrix_are_the_reference() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(3);
    let (rows, n, k, at) = (70, 128, 96, 2);
    let (wq, m, held) = matrix(&noise.values(n * k, 0.2), n, k, at);
    let x = noise.values(rows * k, 2.0);
    for q8 in PATHS {
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
            let kernel = if q8 { Kernel::GemmQ8 } else { Kernel::GemmF32 };
            let got = rig.run(
                kernel,
                &[rows, n, k, 0, k, m.at, 0, width, 0, flags, qs],
                [(n / 64) as u32, rows.div_ceil(64) as u32, 1],
                &data,
            );
            let e = off(&got.dst, &want);
            assert!(e < 1e-5, "q8 {q8}, flags {flags}: {e}");
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
            &[k, 0, 0, k, 0, 2 * k, k, 0, qs, usize::from(q8), 7, 0],
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
            let word = words[b * 8 + (j / 8) * 2 + (j % 2)];
            let byte = (word >> (8 * ((j % 8) / 2))) as u8 as i8;
            f32::from(byte) * f32::from_bits(words[qs + 2 * b])
        })
        .collect()
}

#[test]
fn embeddings_and_per_layer_inputs_are_the_references() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(5);
    let (vocab, width, at) = (32, 64, 4);
    let (wq, m, table) = matrix(&noise.values(vocab * width, 0.5), vocab, width, at);
    let tokens = [7u32, 3, 31];
    let mut ints = vec![0u32; TOKENS];
    ints[0] = 1;
    ints.extend(tokens);
    // From position 1 (the decoder's), two rows.
    let data = Data {
        dst: vec![0.0; 2 * width],
        wq: wq.clone(),
        ints: ints.clone(),
        ..Data::default()
    };
    let got = rig.run(
        Kernel::Embed,
        &[2, 0, 1, 0, m.at, 3.0f32.to_bits() as usize, width, vocab],
        crate::engine::spread(2 * width),
        &data,
    );
    let want: Vec<f32> = [3usize, 31]
        .iter()
        .flat_map(|&t| table[t * width..(t + 1) * width].iter().map(|v| v * 3.0))
        .collect();
    assert!(off(&got.dst, &want) < 1e-7);
    // The per-layer inputs of rows 0 and 1 (projection rows 1 and 2), a
    // layer's slice from 32 of 96.
    let (stride, col, scale, embed_scale) = (96, 32, 0.25f32, 2.0f32);
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
            vocab,
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

    /// Head `h` (of 12: queries, keys, values) of row `r` normed and
    /// turned at `pos` by a table of `base` turning all pairs.
    fn prepared(&self, r: usize, h: usize, pos: usize, base: f32) -> Vec<f32> {
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
        for i in 0..half {
            let f = 1.0 / base.powf((2 * i) as f32 / d as f32);
            let a = f64::from(pos as f32 * f);
            let (c, s) = (a.cos() as f32, a.sin() as f32);
            out[i] = v[i] * c - v[i + half] * s;
            out[i + half] = v[i + half] * c + v[i] * s;
        }
        out
    }

    /// The rotary table for `positions` of heads of `d`, `base`.
    fn table(&self, positions: usize, base: f32) -> Vec<u32> {
        let half = self.d / 2;
        let mut t = Vec::new();
        for pos in 0..positions {
            for i in 0..half {
                let f = 1.0 / base.powf((2 * i) as f32 / self.d as f32);
                let a = f64::from(pos as f32 * f);
                t.push((a.cos() as f32).to_bits());
                t.push((a.sin() as f32).to_bits());
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

/// The cache's layout (as the kernels keep it) of the keys and values of
/// rows prepared at positions 0 on, and the queries after them.
fn prepare(rig: &Rig, heads: &Heads, rows: usize, positions: usize, base: f32) -> Data {
    let d = heads.d;
    let (k_at, v_at, q_at) = (0, 2 * d * positions, 4 * d * positions);
    let data = Data {
        src: heads.qkv.clone(),
        aux: [heads.q_norm.clone(), heads.k_norm.clone()].concat(),
        dst: vec![0.0; q_at + rows * 8 * d],
        wq: heads.table(positions, base),
        ints: vec![0; TOKENS],
    };
    rig.run(
        Kernel::Prep,
        &[
            rows,
            0,
            0,
            d,
            0,
            12 * d,
            0,
            d,
            q_at,
            k_at,
            v_at,
            positions,
            0,
            2,
            8,
        ],
        [rows as u32, 12, 1],
        &data,
    )
}

#[test]
fn queries_keys_and_values_go_normed_and_turned_to_their_places() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(6);
    let (rows, d, positions) = (3, 128, 8);
    let heads = Heads::new(&mut noise, rows, d);
    let got = prepare(&rig, &heads, rows, positions, 10_000.0).dst;
    let (v_at, q_at) = (2 * d * positions, 4 * d * positions);
    for r in 0..rows {
        for h in 0..12 {
            let want = heads.prepared(r, h, r, 10_000.0);
            let have: Vec<f32> = (0..d)
                .map(|i| match h {
                    0..=7 => got[q_at + (r * 8 + h) * d + i],
                    8 | 9 => got[((h - 8) * d + i) * positions + r],
                    _ => got[v_at + ((h - 10) * positions + r) * d + i],
                })
                .collect();
            assert!(off(&have, &want) < 1e-5, "row {r}, head {h}");
        }
    }
}

#[test]
fn a_steps_attention_and_a_prompts_are_the_reference() {
    let Some(rig) = rig() else { return };
    let mut noise = Noise(7);
    let (rows, d, positions) = (150, 128, 160);
    let heads = Heads::new(&mut noise, rows, d);
    let cache = prepare(&rig, &heads, rows, positions, 10_000.0).dst;
    let (k_at, v_at, q_at) = (0, 2 * d * positions, 4 * d * positions);
    let kv = |g: usize, kind: usize| -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| heads.prepared(r, 8 + 2 * kind + g, r, 10_000.0))
            .collect()
    };
    for window in [0, 40] {
        let want = |r: usize, h: usize| {
            let q = heads.prepared(r, h, r, 10_000.0);
            attention(&q, &kv(h / 4, 0), &kv(h / 4, 1), r, window)
        };
        // A prompt's: all rows, from row 0, through flash.
        let data = Data {
            src: cache.clone(),
            dst: vec![0.0; rows * 8 * d],
            ints: vec![0; TOKENS],
            ..Data::default()
        };
        let per_block = 256 / (d / 16);
        let got = rig.run(
            Kernel::Flash,
            &[rows, 0, d, window, q_at, k_at, v_at, positions, 0, 8, 4],
            [rows.div_ceil(per_block) as u32, 8, 1],
            &data,
        );
        for r in [0, 1, 39, 40, 41, 100, rows - 1] {
            for h in [0, 3, 4, 7] {
                let have = &got.dst[(r * 8 + h) * d..(r * 8 + h + 1) * d];
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
        // Splits idle (the window's one chunk), and one taking two chunks (of 3).
        let splits = if window > 0 { 3 } else { 2 };
        let mut ints = vec![0u32; TOKENS];
        ints[0] = last as u32;
        let data = Data {
            src,
            dst: vec![0.0; 8 * splits * (d + 2)],
            ints: ints.clone(),
            ..Data::default()
        };
        let parts = rig.run(
            Kernel::Attend,
            &[d, window, q_at, k_at, v_at, positions, 0, splits],
            [splits as u32, 2, 1],
            &data,
        );
        for q8 in PATHS {
            let qs = 8 * d;
            let data = Data {
                src: parts.dst.clone(),
                dst: vec![0.0],
                ints: vec![0; qs + 8 * d / 32 * 2],
                ..Data::default()
            };
            let got = rig.run(
                Kernel::Join,
                &[d, splits, 0, 0, qs, usize::from(q8), 8],
                [1, 1, 1],
                &data,
            );
            let want_all: Vec<f32> = (0..8).flat_map(|h| want(last, h)).collect();
            let (want_words, _) = written(&want_all, q8, qs);
            let e = off(
                &held(&got.ints, 8 * d, q8, qs),
                &held(&want_words, 8 * d, q8, qs),
            );
            assert!(e < 1e-5, "step: window {window}, q8 {q8}: {e}");
        }
    }
}
