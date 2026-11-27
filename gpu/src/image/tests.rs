//! The picture models' kernels on small shapes against the processor, on
//! f32 multiply-adds and (where the GPU has them) tensor cores. Where there
//! is no usable GPU, each says so and passes.

use jodel_tensors::SafeTensors;

use super::gpu::{Gpu, Made, Tiles};
use super::plan::{At, Buf, Conv, Mat, Norm, Plan, Rope, Tail};
use super::weights::{Fp8, Packing, Source};
use crate::engine::{Input, Output};

/// A slot handed over is caught where its next use reads it (as an input,
/// or adding into it) and passes where that use writes it (no GPU).
#[test]
fn a_slot_handed_over_and_read_is_caught() {
    let check = |build: &dyn Fn(&mut Plan)| {
        let mut p = Plan::new(None);
        for _ in 0..3 {
            p.slot(16);
        }
        p.silu(16, At::new(2, 0), At::new(0, 0));
        build(&mut p);
        p.check()
    };
    let at = |s| At::new(s, 0);
    assert_eq!(check(&|_| ()), None);
    let read = |p: &mut Plan| {
        p.hand_over(0);
        p.silu(16, at(0), at(1));
    };
    assert_eq!(check(&read), Some((0, 1)));
    let added = |p: &mut Plan| {
        p.hand_over(1);
        p.euler(16, at(1), at(2), 0.5);
    };
    assert_eq!(check(&added), Some((1, 1)));
    let written = |p: &mut Plan| {
        p.hand_over(1);
        p.silu(16, at(0), at(1));
        p.euler(16, at(1), at(2), 0.5);
    };
    assert_eq!(check(&written), None);
}

/// The kernels validate, and (without the tensor-core ones) write for
/// Metal; with them, they validate on both tiles, each
/// tensor-core kernel within the workgroup memory its tiles are chosen for
/// (`Tiles::memory`; on 8 × 8 within Apple's 32 KB), and on the 8 × 8 ones
/// (Metal's simdgroup matrices: the multiply, the attention f32) write for
/// Metal in every language version wgpu may take there (2.3, the first
/// with them, to 4.0), their products as simdgroup matrices of f16 into
/// f32.
#[test]
fn the_kernels_validate() {
    use super::Kernel;
    use crate::engine::Kernels;
    let entries = |coop| -> Vec<&str> { Kernel::on(coop).iter().map(|k| k.entry()).collect() };
    crate::engine::check_source(
        &super::source(None),
        naga::valid::Capabilities::empty(),
        &entries(None),
    );
    assert!(
        Tiles::Eight.memory() <= 32 * 1024,
        "8 × 8 within Apple's 32 KB"
    );
    for tiles in [Tiles::Sixteen, Tiles::Eight] {
        let module = naga::front::wgsl::parse_str(&super::source(Some(tiles)))
            .unwrap_or_else(|e| panic!("{tiles:?}: {e}"));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::COOPERATIVE_MATRIX,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{tiles:?}: {e:?}"));
        // The program's kernels are the source's entry points, all of them.
        let mut all = entries(Some(tiles));
        let mut points: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        all.sort_unstable();
        points.sort_unstable();
        assert_eq!(points, all, "{tiles:?}");
        for entry in ["gemm_coop", "attention_coop"] {
            if !all.contains(&entry) {
                continue;
            }
            let bytes = workgroup_bytes(&module, &info, entry);
            assert!(
                bytes <= tiles.memory(),
                "{entry} on {tiles:?}: {bytes} bytes of workgroup memory"
            );
        }
        if tiles == Tiles::Sixteen {
            continue;
        }
        for lang_version in [(2, 3), (3, 2), (4, 0)] {
            let options = naga::back::msl::Options {
                lang_version,
                ..Default::default()
            };
            let (msl, _) = naga::back::msl::write_string(
                &module,
                &info,
                &options,
                &naga::back::msl::PipelineOptions::default(),
            )
            .unwrap_or_else(|e| panic!("Metal {lang_version:?}: {e}"));
            for e in &all {
                assert!(msl.contains(e), "{e} in Metal {lang_version:?}");
            }
            for part in [
                "simdgroup_half8x8",
                "simdgroup_float8x8",
                "simdgroup_multiply_accumulate",
                "simdgroup_store",
            ] {
                assert!(msl.contains(part), "{part} in Metal {lang_version:?}");
            }
            assert!(!msl.contains("16x16"), "no 16 × 16 in Metal");
        }
    }
}

/// The bytes of workgroup memory the entry point `entry` of `module` uses
/// (its workgroup variables', each counted once).
fn workgroup_bytes(module: &naga::Module, info: &naga::valid::ModuleInfo, entry: &str) -> u32 {
    let index = module
        .entry_points
        .iter()
        .position(|e| e.name == entry)
        .unwrap_or_else(|| panic!("{entry} is an entry point"));
    let uses = info.get_entry_point(index);
    module
        .global_variables
        .iter()
        .filter(|(h, g)| g.space == naga::AddressSpace::WorkGroup && !uses[*h].is_empty())
        .map(|(_, g)| module.types[g.ty].inner.size(module.to_ctx()))
        .sum()
}

/// A few weights of each kind packed from `src` (the FP8 matrix's bytes
/// `fp8`).
fn packed<'a>(src: Source<'a>, fp8: Option<&'a [u8]>) -> Packing<'a> {
    let mut pack = Packing::new();
    pack.linear(src, "w", (64, 96)).unwrap();
    pack.conv(src, "c", (4, 8, 3), 4).unwrap();
    pack.fp8(&Fp8 {
        bytes: fp8,
        n: 64,
        k: 64,
        scales: vec![0.5],
    });
    pack.vector(&[1.0, 2.0, 3.0]);
    pack
}

/// A packing of shapes alone places what the checkpoint's does: what a
/// part says it takes before it is loaded is what it takes once loaded.
#[test]
fn shapes_alone_measure_what_is_loaded() {
    let t = bf16_file(&[
        ("w.weight", vec![96, 64], values(1, 96 * 64)),
        ("w.bias", vec![96], values(2, 96)),
        ("c.weight", vec![8, 4, 3, 3], values(3, 288)),
        ("c.bias", vec![8], values(4, 8)),
    ]);
    let w8 = fp8_bytes(5, 64 * 64);
    let measured = packed(Source::Shapes, None).bytes();
    let full = packed(Source::Checkpoint(&t), Some(&w8));
    assert_eq!(full.bytes(), measured);
    if let Err(e) = &crate::chosen().adapter {
        eprintln!("no GPU for the picture kernels: {e}");
        return;
    }
    assert_eq!(full.upload("test").unwrap().bytes(), measured);
    assert!(packed(Source::Shapes, None).upload("test").is_err());
}

/// Values in -1..1 from `seed`.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// One FP8 (e4m3) byte's value.
fn e4m3(b: u8) -> f32 {
    let e = i32::from((b >> 3) & 15);
    let m = f32::from(b & 7);
    let v = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// FP8 bytes (no NaNs) from `seed`.
fn fp8_bytes(seed: u64, n: usize) -> Vec<u8> {
    values(seed, n)
        .iter()
        .map(|v| {
            let b = ((v + 1.0) * 127.9) as u8;
            if b & 0x7f == 0x7f {
                b - 1
            } else {
                b
            }
        })
        .collect()
}

/// A safetensors file of bf16 tensors (name, shape, values).
fn bf16_file(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> SafeTensors {
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for (name, shape, v) in tensors {
        let shape: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let start = data.len();
        data.extend(
            v.iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()),
        );
        entries.push(format!(
            r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{start},{}]}}"#,
            shape.join(","),
            data.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header.as_bytes());
    out.extend(data);
    SafeTensors::parse(out).unwrap()
}

/// bf16's value of `x` (truncated, as `bf16_file` stores it).
fn bf(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0xffff_0000)
}

/// Runs a plan `build` makes on the weights `pack` holds, `inputs` written
/// first, `read` read back; on f32 kernels, or the fast ones. `None`
/// without a usable GPU.
fn run(
    pack: Packing,
    fast: bool,
    build: impl FnOnce(&mut Plan),
    inputs: &[(u8, usize, Vec<f32>)],
    read: (u8, usize, usize),
) -> Option<Vec<f32>> {
    if let Err(e) = &crate::chosen().adapter {
        eprintln!("no GPU for the picture kernels: {e}");
        return None;
    }
    Gpu::shared().expect("the picture kernels set up on the GPU");
    let loaded = pack.upload("test").unwrap();
    let mut p = Plan::new(coop().filter(|_| fast));
    // The tests' slots: 0 to 3.
    for _ in 0..4 {
        p.slot(0);
    }
    for (slot, at, v) in inputs {
        p.need(*slot, at + v.len());
    }
    p.need(read.0, read.1 + read.2);
    build(&mut p);
    let mut made = Made::new(&loaded, &p, read.2).unwrap();
    let inputs: Vec<Input<u8>> = inputs
        .iter()
        .map(|(buf, at, v)| Input {
            buf: *buf,
            at: *at,
            values: v.as_slice(),
        })
        .collect();
    let (buf, first, count) = read;
    let output = Output { buf, first, count };
    Some(made.run(&loaded, &inputs, output).unwrap())
}

/// The tiles of the GPU's tensor cores (without them the fast kernels are
/// the f32 ones, checked once).
fn coop() -> Option<Tiles> {
    Gpu::shared().ok().and_then(|g| g.coop)
}

/// The largest difference of `got` from `want`, against `want`'s largest
/// magnitude.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

/// The tolerances of the f32 kernels and of the f16 inputs.
fn tolerance(fast: bool) -> f32 {
    if fast && coop().is_some() {
        4e-3
    } else {
        1e-5
    }
}

/// C = A · Wᵀ through FP8 (scales per row group and block), bf16 with a
/// bias, and f32 matrices in a slot; written, added, gated and SwiGLU'd;
/// rows and columns past the tiles.
#[test]
fn the_matrix_multiply_matches_the_processor() {
    for fast in [false, true] {
        let (m, k) = (200, 288);
        let a = values(1, m * k);
        // FP8: 192 rows, a scale per 64 rows and 128 of k.
        let n8 = 192;
        let w8 = fp8_bytes(2, n8 * k);
        let kb = k.div_ceil(128);
        let scales: Vec<f32> = (0..n8 / 64 * kb).map(|i| 0.01 * (1.0 + i as f32)).collect();
        let dense8: Vec<f32> = (0..n8 * k)
            .map(|i| e4m3(w8[i]) * scales[(i / k) / 64 * kb + (i % k) / 128])
            .collect();
        let mm = |w: &[f32], n: usize, bias: Option<&[f32]>| -> Vec<f32> {
            (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    let s: f32 = (0..k).map(|j| a[r * k + j] * w[c * k + j]).sum();
                    s + bias.map_or(0.0, |b| b[c])
                })
                .collect()
        };
        let want8 = mm(&dense8, n8, None);
        let prior = values(3, m * n8);
        let gates = values(4, n8);
        for tail in [Tail::Store, Tail::Add, Tail::Gated(At::new(2, 0))] {
            let mut pack = Packing::new();
            let w = pack.fp8(&Fp8 {
                bytes: Some(&w8),
                n: n8,
                k,
                scales: scales.clone(),
            });
            let (width, want): (usize, Vec<f32>) = match tail {
                Tail::Store => (n8, want8.clone()),
                Tail::Add => (n8, want8.iter().zip(&prior).map(|(a, b)| a + b).collect()),
                Tail::Gated(_) => (
                    n8,
                    want8
                        .iter()
                        .zip(&prior)
                        .enumerate()
                        .map(|(i, (a, b))| b + gates[i % n8] * a)
                        .collect(),
                ),
                Tail::SwiGlu => unreachable!(),
            };
            let Some(got) = run(
                pack,
                fast,
                |p| p.gemm((At::new(0, 0), k), m, &w, (At::new(1, 0), width), tail),
                &[
                    (0, 0, a.clone()),
                    (1, 0, prior.clone()),
                    (2, 0, gates.clone()),
                ],
                (1, 0, m * width),
            ) else {
                return;
            };
            let e = worst(&got, &want);
            assert!(e < tolerance(fast), "FP8 {tail:?} fast {fast}: {e}");
        }
        // SwiGLU: gates and ups of 128 rows each, paired.
        let gate = fp8_bytes(5, 128 * k);
        let up = fp8_bytes(6, 128 * k);
        let s1 = vec![0.02; 2 * kb];
        let mut pack = Packing::new();
        let w = pack.fp8_pairs(
            &Fp8 {
                bytes: Some(&gate),
                n: 128,
                k,
                scales: s1.clone(),
            },
            &Fp8 {
                bytes: Some(&up),
                n: 128,
                k,
                scales: s1.clone(),
            },
        );
        let dg: Vec<f32> = gate.iter().map(|&b| e4m3(b) * 0.02).collect();
        let du: Vec<f32> = up.iter().map(|&b| e4m3(b) * 0.02).collect();
        let (g, u) = (mm(&dg, 128, None), mm(&du, 128, None));
        let want: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let Some(got) = run(
            pack,
            fast,
            |p| {
                p.gemm(
                    (At::new(0, 0), k),
                    m,
                    &w,
                    (At::new(1, 0), 128),
                    Tail::SwiGlu,
                )
            },
            &[(0, 0, a.clone())],
            (1, 0, m * 128),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < tolerance(fast), "SwiGLU fast {fast}: {e}");
        // bf16 with a bias: 130 columns.
        let nb = 130;
        let wb = values(7, nb * k);
        let bias = values(8, nb);
        let t = bf16_file(&[
            ("w.weight", vec![nb, k], wb.clone()),
            ("w.bias", vec![nb], bias.clone()),
        ]);
        let mut pack = Packing::new();
        let w = pack.linear(Source::Checkpoint(&t), "w", (k, nb)).unwrap();
        let wbf: Vec<f32> = wb.iter().map(|&x| bf(x)).collect();
        let bbf: Vec<f32> = bias.iter().map(|&x| bf(x)).collect();
        let want = mm(&wbf, nb, Some(&bbf));
        let Some(got) = run(
            pack,
            fast,
            |p| p.gemm((At::new(0, 0), k), m, &w, (At::new(1, 0), nb), Tail::Store),
            &[(0, 0, a.clone())],
            (1, 0, m * nb),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < tolerance(fast), "bf16 fast {fast}: {e}");
        // f32 rows in a slot as the matrix (keys, values).
        let nf = 96;
        let wf = values(9, nf * k);
        let want = mm(&wf, nf, None);
        let w = Mat::of_slot(At::new(2, 0), nf, k, k);
        let Some(got) = run(
            Packing::new(),
            fast,
            |p| p.gemm((At::new(0, 0), k), m, &w, (At::new(1, 0), nf), Tail::Store),
            &[(0, 0, a.clone()), (2, 0, wf)],
            (1, 0, m * nf),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < tolerance(fast), "f32 fast {fast}: {e}");
    }
}

/// The processor's convolution of `x` (`h × w × cin`) by `wt`
/// (`[cout][cin][s][s]`) with `bias`; stride, padding before, the input
/// doubled first where asked; `pre` applied to each input value (by its
/// channel) before.
#[allow(clippy::too_many_arguments)]
fn conv_cpu(
    x: &[f32],
    (h, w, cin): (usize, usize, usize),
    wt: &[f32],
    bias: &[f32],
    (cout, s, stride, pad, up): (usize, usize, usize, usize, bool),
    (oh, ow): (usize, usize),
    pre: &dyn Fn(usize, f32) -> f32,
) -> Vec<f32> {
    let (ih, iw) = if up { (2 * h, 2 * w) } else { (h, w) };
    let mut out = vec![0.0; oh * ow * cout];
    for oy in 0..oh {
        for ox in 0..ow {
            for o in 0..cout {
                let mut sum = bias[o];
                for ky in 0..s {
                    for kx in 0..s {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if iy < 0 || ix < 0 || iy >= ih as isize || ix >= iw as isize {
                            continue;
                        }
                        let (sy, sx) = if up {
                            (iy as usize / 2, ix as usize / 2)
                        } else {
                            (iy as usize, ix as usize)
                        };
                        for c in 0..cin {
                            let v = pre(c, x[(sy * w + sx) * cin + c]);
                            sum += v * wt[((o * cin + c) * s + ky) * s + kx];
                        }
                    }
                }
                out[(oy * ow + ox) * cout + o] = sum;
            }
        }
    }
    out
}

/// Convolutions: 3 × 3 with the group norm and SiLU on the way (its
/// statistics from `gn_stats`), doubled first, the picture's 3 channels;
/// stride 2 with a row and column of zeros after.
#[test]
fn convolutions_match_the_processor() {
    for fast in [false, true] {
        // 3 × 3 over 12 × 10 × 128 → 40, the group norm (32 groups) and
        // SiLU on the input, added to what is there.
        let (h, w, cin, cout) = (12, 10, 128, 40);
        let x: Vec<f32> = values(10, h * w * cin)
            .iter()
            .enumerate()
            .map(|(i, v)| v * 3.0 + (i % cin) as f32 * 0.05)
            .collect();
        let wt = values(11, cout * cin * 9);
        let bias = values(12, cout);
        let gw = values(13, cin);
        let gb = values(14, cin);
        let t = bf16_file(&[
            ("c.weight", vec![cout, cin, 3, 3], wt.clone()),
            ("c.bias", vec![cout], bias.clone()),
        ]);
        let mut pack = Packing::new();
        let mat = pack
            .conv(Source::Checkpoint(&t), "c", (cin, cout, 3), cin)
            .unwrap();
        let gwa = pack.vector(&gw);
        let gba = pack.vector(&gb);
        // The groups' means and spreads on the processor.
        let group = cin / 32;
        let mut stats = Vec::new();
        for g in 0..32 {
            let vals: Vec<f64> = (0..h * w)
                .flat_map(|p| (0..group).map(move |c| (p, g * group + c)))
                .map(|(p, c)| f64::from(x[p * cin + c]))
                .collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            stats.push((mean, 1.0 / (var + 1e-6).sqrt()));
        }
        let pre = |c: usize, v: f32| {
            let (mean, r) = stats[c / group];
            let y = ((f64::from(v) - mean) * r) as f32 * gw[c] + gb[c];
            y / (1.0 + (-y).exp())
        };
        let wbf: Vec<f32> = wt.iter().map(|&v| bf(v)).collect();
        let bbf: Vec<f32> = bias.iter().map(|&v| bf(v)).collect();
        let prior = values(15, h * w * cout);
        let want: Vec<f32> = conv_cpu(
            &x,
            (h, w, cin),
            &wbf,
            &bbf,
            (cout, 3, 1, 1, false),
            (h, w),
            &pre,
        )
        .iter()
        .zip(&prior)
        .map(|(a, b)| a + b)
        .collect();
        let conv = Conv {
            input: At::new(0, 0),
            h,
            w,
            cin,
            size: 3,
            stride: 1,
            pad: 1,
            trail: 1,
            up: false,
        };
        let Some(got) = run(
            pack,
            fast,
            |p| {
                p.need(3, 4096);
                p.gn_stats(
                    (h * w, cin),
                    At::new(0, 0),
                    At::new(3, 0),
                    At::new(2, 0),
                    1e-6,
                );
                let norm = Norm {
                    stats: At::new(2, 0),
                    weight: gwa,
                    bias: gba,
                    group,
                    silu: true,
                };
                p.conv(conv, &mat, At::new(1, 0), Tail::Add, Some(norm));
            },
            &[(0, 0, x.clone()), (1, 0, prior), (2, 0, vec![0.0; 64])],
            (1, 0, h * w * cout),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < 4.0 * tolerance(fast), "normed conv fast {fast}: {e}");
        // Doubled first: 5 × 6 × 3 (padded to 4) → 16, 3 × 3.
        let (h, w, cin, cout) = (5, 6, 3, 16);
        let x3 = values(16, h * w * cin);
        let x4: Vec<f32> = x3.chunks(3).flat_map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let wt = values(17, cout * cin * 9);
        let bias = values(18, cout);
        let t = bf16_file(&[
            ("c.weight", vec![cout, cin, 3, 3], wt.clone()),
            ("c.bias", vec![cout], bias.clone()),
        ]);
        let mut pack = Packing::new();
        let mat = pack
            .conv(Source::Checkpoint(&t), "c", (cin, cout, 3), 4)
            .unwrap();
        let wbf: Vec<f32> = wt.iter().map(|&v| bf(v)).collect();
        let bbf: Vec<f32> = bias.iter().map(|&v| bf(v)).collect();
        let want = conv_cpu(
            &x3,
            (h, w, cin),
            &wbf,
            &bbf,
            (cout, 3, 1, 1, true),
            (2 * h, 2 * w),
            &|_, v| v,
        );
        let conv = Conv {
            input: At::new(0, 0),
            h,
            w,
            cin: 4,
            size: 3,
            stride: 1,
            pad: 1,
            trail: 1,
            up: true,
        };
        assert_eq!(conv.out(), (2 * h, 2 * w));
        let Some(got) = run(
            pack,
            fast,
            |p| p.conv(conv, &mat, At::new(1, 0), Tail::Store, None),
            &[(0, 0, x4)],
            (1, 0, 4 * h * w * cout),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < tolerance(fast), "doubled conv fast {fast}: {e}");
        // Stride 2, zeros after: 9 × 8 × 32 → 24.
        let (h, w, cin, cout) = (9, 8, 32, 24);
        let x = values(19, h * w * cin);
        let wt = values(20, cout * cin * 9);
        let bias = values(21, cout);
        let t = bf16_file(&[
            ("c.weight", vec![cout, cin, 3, 3], wt.clone()),
            ("c.bias", vec![cout], bias.clone()),
        ]);
        let mut pack = Packing::new();
        let mat = pack
            .conv(Source::Checkpoint(&t), "c", (cin, cout, 3), cin)
            .unwrap();
        let wbf: Vec<f32> = wt.iter().map(|&v| bf(v)).collect();
        let bbf: Vec<f32> = bias.iter().map(|&v| bf(v)).collect();
        let conv = Conv {
            input: At::new(0, 0),
            h,
            w,
            cin,
            size: 3,
            stride: 2,
            pad: 0,
            trail: 1,
            up: false,
        };
        let (oh, ow) = conv.out();
        assert_eq!((oh, ow), (4, 4));
        let want = conv_cpu(
            &x,
            (h, w, cin),
            &wbf,
            &bbf,
            (cout, 3, 2, 0, false),
            (oh, ow),
            &|_, v| v,
        );
        let Some(got) = run(
            pack,
            fast,
            |p| p.conv(conv, &mat, At::new(1, 0), Tail::Store, None),
            &[(0, 0, x)],
            (1, 0, oh * ow * cout),
        ) else {
            return;
        };
        let e = worst(&got, &want);
        assert!(e < tolerance(fast), "strided conv fast {fast}: {e}");
    }
}

/// The processor's attention over heads laid out `[head][token][128]`.
fn attention_cpu(
    (q, k, v): (&[f32], &[f32], &[f32]),
    (n, heads, kv): (usize, usize, usize),
    causal: bool,
    valid: usize,
) -> Vec<f32> {
    let scale = 1.0 / 128f32.sqrt();
    let mut out = vec![0.0; n * heads * 128];
    for h in 0..heads {
        let kh = h / (heads / kv);
        for i in 0..n {
            let qi = &q[(h * n + i) * 128..][..128];
            let seen: Vec<usize> = (0..n)
                .filter(|&j| j < valid && (!causal || j <= i))
                .collect();
            let s: Vec<f32> = seen
                .iter()
                .map(|&j| {
                    qi.iter()
                        .zip(&k[(kh * n + j) * 128..][..128])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let top = s.iter().fold(f32::MIN, |m, &x| m.max(x));
            let e: Vec<f32> = s.iter().map(|x| (x - top).exp()).collect();
            let total: f32 = e.iter().sum();
            for d in 0..128 {
                out[i * heads * 128 + h * 128 + d] = seen
                    .iter()
                    .zip(&e)
                    .map(|(&j, w)| w * v[(kh * n + j) * 128 + d])
                    .sum::<f32>()
                    / total;
            }
        }
    }
    out
}

/// Flash attention: all keys over 150 tokens and 2 heads; causal with
/// the padding's keys left out over 4 query heads on 2 key heads; scores
/// spread wide (a later key 30 above the first ones moves the largest).
#[test]
fn attention_matches_the_processor() {
    for fast in [false, true] {
        for (heads, kv, causal, valid) in [(2, 2, false, 150), (4, 2, true, 97)] {
            let n = 150;
            let q: Vec<f32> = values(30, heads * n * 128)
                .iter()
                .map(|v| v * 4.0)
                .collect();
            let mut k: Vec<f32> = values(31, kv * n * 128).iter().map(|v| v * 4.0).collect();
            // A key far above the rest, late.
            for d in 0..128 {
                k[(n - 20) * 128 + d] = q[d] * 3.0;
            }
            let v = values(32, kv * n * 128);
            let want = attention_cpu((&q, &k, &v), (n, heads, kv), causal, valid);
            let mut all = q.clone();
            all.extend(&k);
            all.extend(&v);
            let (ko, vo) = (heads * n * 128, (heads + kv) * n * 128);
            let Some(got) = run(
                Packing::new(),
                fast,
                |p| {
                    p.attention(
                        (n, n, valid),
                        (heads, kv),
                        (At::new(0, 0), At::new(0, ko), At::new(0, vo)),
                        (At::new(1, 0), heads * 128),
                        causal,
                    )
                },
                &[(0, 0, all)],
                (1, 0, n * heads * 128),
            ) else {
                return;
            };
            let e = worst(&got, &want);
            assert!(
                e < 3.0 * tolerance(fast && coop().is_some_and(Tiles::attention)),
                "attention {heads}/{kv} causal {causal} fast {fast}: {e}"
            );
        }
    }
}

/// RMSNorm, LayerNorm with the modulation, the heads' norm and rotary
/// positions (pairs and halves), the sampler's step, SiLU, copies, the
/// latents' patches and back, a softmax and a transpose.
#[test]
fn the_small_kernels_match_the_processor() {
    let (rows, d) = (5, 3072);
    let x: Vec<f32> = values(40, rows * d).iter().map(|v| v * 5.0 + 1.0).collect();
    let wv = values(41, d);
    let mut pack = Packing::new();
    let wat = pack.vector(&wv);
    let rms: Vec<f32> = x
        .chunks(d)
        .flat_map(|r| {
            let s = (r.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6).sqrt();
            r.iter()
                .zip(&wv)
                .map(move |(v, w)| v / s * w)
                .collect::<Vec<_>>()
        })
        .collect();
    let Some(got) = run(
        pack,
        false,
        |p| p.rmsnorm(rows, d, (At::new(0, 0), d), (At::new(1, 0), d), wat, 1e-6),
        &[(0, 0, x.clone())],
        (1, 0, rows * d),
    ) else {
        return;
    };
    assert!(worst(&got, &rms) < 1e-5, "rmsnorm");
    let shift = values(42, d);
    let scale = values(43, d);
    let (sh, sc) = (&shift, &scale);
    let ln: Vec<f32> = x
        .chunks(d)
        .flat_map(|r| {
            let mean = r.iter().sum::<f32>() / d as f32;
            let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let s = (var + 1e-6).sqrt();
            (0..d)
                .map(|c| (r[c] - mean) / s * (1.0 + sc[c]) + sh[c])
                .collect::<Vec<_>>()
        })
        .collect();
    let mut m = shift.clone();
    m.extend(&scale);
    let got = run(
        Packing::new(),
        false,
        |p| {
            p.layernorm_mod(
                rows,
                d,
                (At::new(0, 0), d),
                (At::new(1, 0), d),
                At::new(2, 0),
                At::new(2, d),
                1e-6,
            )
        },
        &[(0, 0, x.clone()), (2, 0, m)],
        (1, 0, rows * d),
    )
    .unwrap();
    assert!(worst(&got, &ln) < 1e-5, "layernorm");
    // The heads: 3 tokens of 2 heads (rows 300 apart from 40), normed,
    // turned by pairs and by halves; and 17 tokens of 3 heads (the heads
    // of two workgroups, the second's partly), normed or not.
    for (tokens, heads, normed) in [(3, 2, true), (17, 3, true), (17, 3, false)] {
        let stride = 44 + heads * 128;
        let src = values(44, tokens * stride);
        let nw = values(45, 128);
        let table: Vec<f32> = (0..(tokens + 2) * 64)
            .flat_map(|i| [(i as f32 * 0.1).cos(), (i as f32 * 0.1).sin()])
            .collect();
        for rope in [Rope::Pairs, Rope::Halves] {
            let mut pack = Packing::new();
            let nat = pack.vector(&nw);
            let tat = pack.vector(&table);
            let mut want = vec![0.0; heads * (tokens + 2) * 128];
            for t in 0..tokens {
                for h in 0..heads {
                    let v = &src[40 + t * stride + h * 128..][..128];
                    let r = (v.iter().map(|x| x * x).sum::<f32>() / 128.0 + 1e-6).sqrt();
                    let n: Vec<f32> = if normed {
                        v.iter().zip(&nw).map(|(x, w)| x / r * w).collect()
                    } else {
                        v.to_vec()
                    };
                    for i in 0..64 {
                        let (a, b) = if rope == Rope::Pairs {
                            (2 * i, 2 * i + 1)
                        } else {
                            (i, i + 64)
                        };
                        let at = ((t + 2) * 64 + i) * 2;
                        let (c, s) = (table[at], table[at + 1]);
                        let out = (h * (tokens + 2) + t + 2) * 128;
                        want[out + a] = n[a] * c - n[b] * s;
                        want[out + b] = n[b] * c + n[a] * s;
                    }
                }
            }
            let got = run(
                pack,
                false,
                |p| {
                    p.qk_prep(
                        (tokens, heads),
                        (At::new(0, 40), stride),
                        (At::new(1, 0), tokens + 2, 2),
                        normed.then_some(nat),
                        Some((rope, Buf::Small, tat)),
                        1e-6,
                    )
                },
                &[
                    (0, 0, src.clone()),
                    (1, 0, vec![0.0; heads * (tokens + 2) * 128]),
                ],
                (1, 0, heads * (tokens + 2) * 128),
            )
            .unwrap();
            let e = worst(&got, &want);
            assert!(
                e < 1e-5,
                "qk_prep {tokens}x{heads} normed {normed} {rope:?}: {e}"
            );
        }
    }
    // The sampler's step, SiLU, a copy.
    let a = values(46, 1000);
    let b = values(47, 1000);
    let got = run(
        Packing::new(),
        false,
        |p| p.euler(1000, At::new(1, 0), At::new(0, 0), -0.25),
        &[(0, 0, a.clone()), (1, 0, b.clone())],
        (1, 0, 1000),
    )
    .unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(a, b)| b - 0.25 * a).collect();
    assert!(worst(&got, &want) < 1e-6, "euler");
    let got = run(
        Packing::new(),
        false,
        |p| p.silu(1000, At::new(0, 0), At::new(1, 0)),
        &[(0, 0, a.clone())],
        (1, 0, 1000),
    )
    .unwrap();
    let want: Vec<f32> = a.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    assert!(worst(&got, &want) < 1e-6, "silu");
    let got = run(
        Packing::new(),
        false,
        |p| p.copy(10, 8, (At::new(0, 4), 12), (At::new(1, 0), 8)),
        &[(0, 0, a.clone())],
        (1, 0, 80),
    )
    .unwrap();
    let want: Vec<f32> = (0..80).map(|i| a[4 + (i / 8) * 12 + i % 8]).collect();
    assert_eq!(got, want, "copy");
    // The latents' patches: 4 × 6 latent pixels of 64 channels → 2 × 3
    // tokens, and the tokens back.
    let lat = values(48, 4 * 6 * 64);
    let mean = values(49, 128);
    let spread: Vec<f32> = values(50, 128).iter().map(|v| v.abs() + 0.5).collect();
    let mut pack = Packing::new();
    let (ma, sa) = (pack.vector(&mean), pack.vector(&spread));
    let mut want = vec![0.0; 6 * 128];
    for y in 0..2 {
        for x in 0..3 {
            for ch in 0..128 {
                let (c, dy, dx) = (ch / 4, ch / 2 % 2, ch % 2);
                let v = lat[((2 * y + dy) * 6 + 2 * x + dx) * 64 + c];
                want[(y * 3 + x) * 128 + ch] = (v - mean[ch]) / spread[ch];
            }
        }
    }
    let got = run(
        pack,
        false,
        |p| p.patchify((2, 3, 64), At::new(0, 0), At::new(1, 0), (ma, sa)),
        &[(0, 0, lat.clone())],
        (1, 0, 6 * 128),
    )
    .unwrap();
    assert!(worst(&got, &want) < 1e-6, "patchify");
    let mut pack = Packing::new();
    let (ma, sa) = (pack.vector(&mean), pack.vector(&spread));
    let back = run(
        pack,
        false,
        |p| p.unpatchify((2, 3), At::new(0, 0), At::new(1, 0), (ma, sa)),
        &[(0, 0, got)],
        (1, 0, 4 * 6 * 32),
    )
    .unwrap();
    let first: Vec<f32> = lat.chunks(64).flat_map(|p| p[..32].to_vec()).collect();
    assert!(worst(&back, &first) < 1e-5, "unpatchify");
    // A softmax over rows of 700, scaled; a transpose of 37 × 21.
    let s = values(51, 3 * 700);
    let got = run(
        Packing::new(),
        false,
        |p| p.softmax(3, 700, (At::new(0, 0), 700), 0.5),
        &[(0, 0, s.clone())],
        (0, 0, 2100),
    )
    .unwrap();
    let want: Vec<f32> = s
        .chunks(700)
        .flat_map(|r| {
            let top = r.iter().fold(f32::MIN, |m, &x| m.max(x * 0.5));
            let e: Vec<f32> = r.iter().map(|x| (x * 0.5 - top).exp()).collect();
            let t: f32 = e.iter().sum();
            e.into_iter().map(move |v| v / t)
        })
        .collect();
    assert!(worst(&got, &want) < 1e-5, "softmax");
    let m = values(52, 37 * 21);
    let got = run(
        Packing::new(),
        false,
        |p| p.transpose((37, 21), (At::new(0, 0), 21), (At::new(1, 0), 40)),
        &[(0, 0, m.clone())],
        (1, 0, 21 * 40),
    )
    .unwrap();
    // Every value out of place, and where in the source each came from
    // (its row and column; none: not from the source), so a fault says
    // its shape.
    let wrong: Vec<String> = (0..37 * 21)
        .filter(|&i| got[(i % 21) * 40 + i / 21] != m[i])
        .map(|i| {
            let (r, c) = (i / 21, i % 21);
            let had = got[c * 40 + r];
            let from = m.iter().position(|&v| v == had).map(|j| (j / 21, j % 21));
            format!("({r},{c}) has {had} from {from:?}")
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "transpose: {} of {} out of place, first {:?}",
        wrong.len(),
        37 * 21,
        &wrong[..wrong.len().min(12)]
    );
}
