//! Depth Anything 3 Small (Apache-2.0) on the GPU: a cover's relative depth.
//!
//! The network as its authors' code runs it on one view: a ViT-S/14 (384
//! wide, 12 layers of 6 heads) whose layers from the fifth on have their
//! queries and keys LayerNormed per head and turned by 2-D rotary positions
//! (the odd ones "global": every patch at one position, which for one view
//! is the only difference), the class token swapped for the camera token
//! there; then the DPT head ("DualDPT", its depth branch only) on layers 6,
//! 8, 10 and 12, each the layer before's tokens beside its own normed. Its
//! first output channel's exponential is the depth (larger is farther).

use std::sync::Mutex;

use jodel_tensors::SafeTensors;

use super::net::Net;
use super::ops::{Img, Plan, Resize, ADD, GELU, RELU, RELU_IN};
use super::prep::{self, Rgb};
use super::{Named, Packer};
use crate::Footprint;

const C: usize = 384;
const HEADS: usize = 6;
const LAYERS: usize = 12;
const FEATURES: usize = 64;
const OUTS: [usize; 4] = [48, 96, 192, 384];
const GRID: usize = 37;

/// A depth map: `width × height` values, rows top to bottom, larger
/// farther (relative: no unit).
#[derive(Clone, Debug, PartialEq)]
pub struct Depth {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

/// The checkpoint's tensors this port uses, read and packed.
#[derive(Clone)]
struct Weights {
    loader: Named,
    /// The class token, the camera token, and the 37 × 37 position table.
    cls: Vec<f32>,
    camera: Vec<f32>,
    pos: Vec<f32>,
}

/// The tensor names' prefixes.
const BACKBONE: &str = "model.backbone.pretrained.";
const HEAD: &str = "model.head.";

impl Weights {
    fn read(t: &SafeTensors) -> Result<Weights, String> {
        let mut w = Weights {
            loader: Named::default(),
            cls: t.get(&format!("{BACKBONE}cls_token"), &[1, 1, C])?,
            camera: t.get(&format!("{BACKBONE}camera_token"), &[1, 2, C])?[..C].to_vec(),
            pos: t.get(&format!("{BACKBONE}pos_embed"), &[1, GRID * GRID + 1, C])?,
        };
        let b = |n: &str| format!("{BACKBONE}{n}");
        let h = |n: &str| format!("{HEAD}{n}");
        w.conv(t, &b("patch_embed.proj"), (3, C, 14), true)?;
        for l in 0..LAYERS {
            let n = |s: &str| b(&format!("blocks.{l}.{s}"));
            for norm in ["norm1", "norm2"] {
                w.vector(t, &n(&format!("{norm}.weight")), C)?;
                w.vector(t, &n(&format!("{norm}.bias")), C)?;
            }
            w.linear(t, &n("attn.qkv"), (C, 3 * C), true, None)?;
            let ls1 = t.get(&n("ls1.gamma"), &[C])?;
            let ls2 = t.get(&n("ls2.gamma"), &[C])?;
            w.linear(t, &n("attn.proj"), (C, C), true, Some(&ls1))?;
            w.linear(t, &n("mlp.fc1"), (C, 4 * C), true, None)?;
            w.linear(t, &n("mlp.fc2"), (4 * C, C), true, Some(&ls2))?;
            if l >= 4 {
                for norm in ["q_norm", "k_norm"] {
                    w.vector(t, &n(&format!("attn.{norm}.weight")), 64)?;
                    w.vector(t, &n(&format!("attn.{norm}.bias")), 64)?;
                }
            }
        }
        w.vector(t, &b("norm.weight"), C)?;
        w.vector(t, &b("norm.bias"), C)?;
        w.vector(t, &h("norm.weight"), 2 * C)?;
        w.vector(t, &h("norm.bias"), 2 * C)?;
        for (s, &oc) in OUTS.iter().enumerate() {
            w.conv(t, &h(&format!("projects.{s}")), (2 * C, oc, 1), true)?;
        }
        w.deconv(t, &h("resize_layers.0"), OUTS[0], 4)?;
        w.deconv(t, &h("resize_layers.1"), OUTS[1], 2)?;
        w.conv(t, &h("resize_layers.3"), (OUTS[3], OUTS[3], 3), true)?;
        for (s, &oc) in OUTS.iter().enumerate() {
            w.conv(
                t,
                &h(&format!("scratch.layer{}_rn", s + 1)),
                (oc, FEATURES, 3),
                false,
            )?;
        }
        for r in 1..=4 {
            let units: &[usize] = if r == 4 { &[2] } else { &[1, 2] };
            for u in units {
                for c in 1..=2 {
                    let name = h(&format!("scratch.refinenet{r}.resConfUnit{u}.conv{c}"));
                    w.conv(t, &name, (FEATURES, FEATURES, 3), true)?;
                }
            }
            w.conv(
                t,
                &h(&format!("scratch.refinenet{r}.out_conv")),
                (FEATURES, FEATURES, 1),
                true,
            )?;
        }
        w.conv(t, &h("scratch.output_conv1"), (FEATURES, 32, 3), true)?;
        w.conv(t, &h("scratch.output_conv2.0"), (32, 32, 3), true)?;
        w.conv(t, &h("scratch.output_conv2.2"), (32, 2, 1), true)?;
        Ok(w)
    }
}

impl std::ops::Deref for Weights {
    type Target = Named;
    fn deref(&self) -> &Named {
        &self.loader
    }
}

impl std::ops::DerefMut for Weights {
    fn deref_mut(&mut self) -> &mut Named {
        &mut self.loader
    }
}

/// PyTorch's bicubic interpolation (A = −0.75, the corners not aligned) of
/// a `GRID × GRID` table of `C` to `rows × cols`, its scale given as
/// (n + 0.1) / 37 per side, as DINOv2 resizes its positions.
fn bicubic_positions(table: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const A: f32 = -0.75;
    let cc1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let taps = |n: usize| -> Vec<(i64, [f32; 4])> {
        // The scale as PyTorch takes it: 1 / scale_factor, in f32.
        let scale = (1.0 / ((n as f64 + 0.1) / GRID as f64)) as f32;
        (0..n)
            .map(|o| {
                let real = scale * (o as f32 + 0.5) - 0.5;
                let i = real.floor();
                let t = real - i;
                (i as i64, [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)])
            })
            .collect()
    };
    let (ty, tx) = (taps(rows), taps(cols));
    let at = |v: i64| v.clamp(0, GRID as i64 - 1) as usize;
    let mut out = vec![0.0f32; rows * cols * C];
    for (y, (iy, wy)) in ty.iter().enumerate() {
        for (x, (ix, wx)) in tx.iter().enumerate() {
            for c in 0..C {
                // Along x for each of four rows, then along y, as PyTorch.
                let mut sum = 0.0f32;
                for (a, wa) in wy.iter().enumerate() {
                    let mut row = 0.0f32;
                    for (b, wb) in wx.iter().enumerate() {
                        let p = at(iy - 1 + a as i64) * GRID + at(ix - 1 + b as i64);
                        row += wb * table[(1 + p) * C + c];
                    }
                    sum += wa * row;
                }
                out[(y * cols + x) * C + c] = sum;
            }
        }
    }
    out
}

/// The head's positions: a UV grid over `rows × cols` for a picture of
/// aspect `aspect` (width / height), as sines and cosines over `dim`
/// channels, times 0.1.
fn uv_positions(rows: usize, cols: usize, dim: usize, aspect: f64) -> Vec<f32> {
    let diag = (aspect * aspect + 1.0).sqrt();
    let (sx, sy) = (aspect / diag, 1.0 / diag);
    let line = |n: usize, span: f64| -> Vec<f64> {
        let (lo, hi) = (
            -span * (n as f64 - 1.0) / n as f64,
            span * (n as f64 - 1.0) / n as f64,
        );
        (0..n)
            .map(|i| {
                if n == 1 {
                    lo
                } else {
                    lo + (hi - lo) * i as f64 / (n as f64 - 1.0)
                }
            })
            .collect()
    };
    let (xs, ys) = (line(cols, sx), line(rows, sy));
    let quarter = dim / 4;
    let omega: Vec<f64> = (0..quarter)
        .map(|i| 1.0 / 100f64.powf(i as f64 / quarter as f64))
        .collect();
    let mut out = Vec::with_capacity(rows * cols * dim);
    for &v in &ys {
        for &u in &xs {
            for pos in [u, v] {
                out.extend(omega.iter().map(|w| ((pos * w).sin() * 0.1) as f32));
                out.extend(omega.iter().map(|w| ((pos * w).cos() * 0.1) as f32));
            }
        }
    }
    out
}

/// The rotary positions' table: for each position up to `most`, 16
/// cosines then 16 sines (frequencies 100^(−j/16)).
fn rope_table(most: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for pos in 0..=most {
        let angles: Vec<f32> = (0..16)
            .map(|j| pos as f32 * (1.0 / 100f32.powf((2 * j) as f32 / 32.0)))
            .collect();
        out.extend(angles.iter().map(|a| a.cos()));
        out.extend(angles.iter().map(|a| a.sin()));
    }
    out
}

/// The plan for a `width × height` input, its size's constants packed
/// into `tail`; the buffer the input goes to and the one the two output
/// channels come from (`height × width × 2`).
fn plan(w: &Weights, tail: &mut Packer, width: usize, height: usize) -> (Plan, usize, usize) {
    let (h0, w0) = (height / 14, width / 14);
    let n = 1 + h0 * w0;
    let consts = {
        let pos = bicubic_positions(&w.pos, h0, w0);
        let mut first: Vec<f32> = w.cls.clone();
        for (c, v) in first.iter_mut().enumerate() {
            *v += w.pos[c];
        }
        (
            tail.push(&first),
            tail.push(&pos),
            tail.push(&w.camera),
            tail.push(&rope_table(h0.max(w0) + 1)),
        )
    };
    let (first, pos, camera, rope) = consts;
    let mut p = Plan::new();
    let input = p.buffer(height * width * 3);
    let x = p.buffer(n * C);
    let xn = p.buffer(n * C);
    let qkv = p.buffer(n * 3 * C);
    let att = p.buffer(n * C);
    let up = p.buffer(n * 4 * C);
    let feats: Vec<usize> = (0..4).map(|_| p.buffer(n * 2 * C)).collect();
    let rows = |buf: usize, c: usize| Img::rows(buf, n, c);

    // The patches, the class token, the positions.
    let patches = Img {
        off: C,
        ..Img::new(x, 1, h0, w0, C)
    };
    p.conv(
        Img::new(input, 1, height, width, 3),
        &w.cw(
            &format!("{BACKBONE}patch_embed.proj"),
            (3, C, 14),
            14,
            0,
            true,
        ),
        patches,
        0,
    );
    p.table(x, 0, C, first, C, false);
    p.table(x, C, (n - 1) * C, pos, (n - 1) * C, true);

    let ln_eps = 1e-6;
    for l in 0..LAYERS {
        let name = |s: &str| format!("{BACKBONE}blocks.{l}.{s}");
        if l == 4 {
            p.table(x, 0, C, camera, C, false);
        }
        p.layernorm(
            rows(x, C),
            rows(xn, C),
            w.at(&name("norm1.weight")),
            w.at(&name("norm1.bias")),
            ln_eps,
        );
        p.linear(
            rows(xn, C),
            w.at(&name("attn.qkv.weight")),
            Some(w.at(&name("attn.qkv.bias"))),
            rows(qkv, 3 * C),
            0,
        );
        if l >= 4 {
            for (which, norm) in ["q_norm", "k_norm"].iter().enumerate() {
                p.layernorm_rows(
                    (qkv, which * C, 3 * C, 64, HEADS),
                    (qkv, which * C, 3 * C, 64),
                    n * HEADS,
                    64,
                    (
                        w.at(&name(&format!("attn.{norm}.weight"))),
                        w.at(&name(&format!("attn.{norm}.bias"))),
                        1e-5,
                    ),
                );
            }
            p.rope(rows(qkv, 3 * C), HEADS, w0, l % 2 == 1, rope);
        }
        p.attend(&super::ops::Attend {
            q: rows(qkv, 3 * C),
            k: Img {
                off: C,
                ..rows(qkv, 3 * C)
            },
            v: Img {
                off: 2 * C,
                ..rows(qkv, 3 * C)
            },
            out: rows(att, C),
            groups: 1,
            nq: n,
            nk: n,
            heads: HEADS,
            d: 64,
            swin: None,
        });
        p.linear(
            rows(att, C),
            w.at(&name("attn.proj.weight")),
            Some(w.at(&name("attn.proj.bias"))),
            rows(x, C),
            ADD,
        );
        p.layernorm(
            rows(x, C),
            rows(xn, C),
            w.at(&name("norm2.weight")),
            w.at(&name("norm2.bias")),
            ln_eps,
        );
        p.linear(
            rows(xn, C),
            w.at(&name("mlp.fc1.weight")),
            Some(w.at(&name("mlp.fc1.bias"))),
            rows(up, 4 * C),
            GELU,
        );
        p.linear(
            rows(up, 4 * C),
            w.at(&name("mlp.fc2.weight")),
            Some(w.at(&name("mlp.fc2.bias"))),
            rows(x, C),
            ADD,
        );
        if l >= 4 {
            let f = feats[(l - 4) / 2];
            let local = Img {
                ld: 2 * C,
                ..rows(f, C)
            };
            if l % 2 == 0 {
                p.copy_img(rows(x, C), local, false);
            } else {
                let normed = Img { off: C, ..local };
                p.layernorm(
                    rows(x, C),
                    normed,
                    w.at(&format!("{BACKBONE}norm.weight")),
                    w.at(&format!("{BACKBONE}norm.bias")),
                    1e-5,
                );
            }
        }
    }

    // The head: each stage's patches normed, projected, placed, resized.
    let hn = |s: &str| format!("{HEAD}{s}");
    let aspect = width as f64 / height as f64;
    let normed = p.buffer((n - 1) * 2 * C);
    let mut levels = Vec::new();
    for (s, &oc) in OUTS.iter().enumerate() {
        let f = Img::rows(feats[s], n, 2 * C).tokens(1, n - 1);
        let nf = Img::rows(normed, n - 1, 2 * C);
        p.layernorm(
            f,
            nf,
            w.at(&hn("norm.weight")),
            w.at(&hn("norm.bias")),
            1e-5,
        );
        let proj = p.buffer(h0 * w0 * oc);
        let pi = Img::new(proj, 1, h0, w0, oc);
        p.conv(
            nf.shaped(1, h0, w0),
            &w.cw(&hn(&format!("projects.{s}")), (2 * C, oc, 1), 1, 0, true),
            pi,
            0,
        );
        let table = tail.push(&uv_positions(h0, w0, oc, aspect));
        p.table(proj, 0, h0 * w0 * oc, table, h0 * w0 * oc, true);
        let resized = match s {
            0 | 1 => {
                let k = if s == 0 { 4 } else { 2 };
                let out = p.buffer(h0 * k * w0 * k * oc);
                let o = Img::new(out, 1, h0 * k, w0 * k, oc);
                let mut cw = w.cw(&hn(&format!("resize_layers.{s}")), (oc, oc, 1), 1, 0, true);
                cw.up = k;
                p.conv(pi, &cw, o, 0);
                o
            }
            2 => pi,
            _ => {
                let (h1, w1) = ((h0 - 1) / 2 + 1, (w0 - 1) / 2 + 1);
                let out = p.buffer(h1 * w1 * oc);
                let o = Img::new(out, 1, h1, w1, oc);
                p.conv(
                    pi,
                    &w.cw(&hn("resize_layers.3"), (oc, oc, 3), 2, 1, true),
                    o,
                    0,
                );
                o
            }
        };
        let rn = p.buffer(resized.h * resized.w * FEATURES);
        let ri = Img::new(rn, 1, resized.h, resized.w, FEATURES);
        p.conv(
            resized,
            &w.cw(
                &hn(&format!("scratch.layer{}_rn", s + 1)),
                (oc, FEATURES, 3),
                1,
                1,
                false,
            ),
            ri,
            0,
        );
        levels.push(ri);
    }

    // The fusion, from the coarsest level up.
    let biggest = levels[0].h * 2 * levels[0].w * 2 * FEATURES;
    let (ya, t, r) = (p.buffer(biggest), p.buffer(biggest), p.buffer(biggest));
    let mut carried: Option<Img> = None;
    for k in (0..4).rev() {
        let unit = |p: &mut Plan, x: Img, into: Img, u: usize| {
            // into += conv2(relu(conv1(relu(x)))).
            let name = |c: usize| {
                hn(&format!(
                    "scratch.refinenet{}.resConfUnit{u}.conv{c}",
                    k + 1
                ))
            };
            let ti = Img::new(t, 1, x.h, x.w, FEATURES);
            p.conv(
                x,
                &w.cw(&name(1), (FEATURES, FEATURES, 3), 1, 1, true),
                ti,
                RELU_IN,
            );
            p.conv(
                ti,
                &w.cw(&name(2), (FEATURES, FEATURES, 3), 1, 1, true),
                into,
                RELU_IN | ADD,
            );
        };
        let lvl = levels[k];
        let y = Img::new(ya, 1, lvl.h, lvl.w, FEATURES);
        match carried {
            None => p.copy_img(lvl, y, false),
            Some(prev) => {
                p.copy_img(prev, y, false);
                p.copy_img(lvl, y, true);
                unit(&mut p, lvl, y, 1);
            }
        }
        // y = RCU2(y): y2 = y + conv2(relu(conv1(relu(y)))), into r.
        let y2 = Img::new(r, 1, lvl.h, lvl.w, FEATURES);
        p.copy_img(y, y2, false);
        unit(&mut p, y, y2, 2);
        let (th, tw) = if k == 0 {
            (lvl.h * 2, lvl.w * 2)
        } else {
            (levels[k - 1].h, levels[k - 1].w)
        };
        let up_img = Img::new(ya, 1, th, tw, FEATURES);
        p.resize(y2, up_img, Resize::Corners, false);
        let out = p.buffer(th * tw * FEATURES);
        let oi = Img::new(out, 1, th, tw, FEATURES);
        p.conv(
            up_img,
            &w.cw(
                &hn(&format!("scratch.refinenet{}.out_conv", k + 1)),
                (FEATURES, FEATURES, 1),
                1,
                0,
                true,
            ),
            oi,
            0,
        );
        carried = Some(oi);
    }
    let fused = carried.expect("four levels");
    let oc1 = Img::new(t, 1, fused.h, fused.w, 32);
    p.conv(
        fused,
        &w.cw(&hn("scratch.output_conv1"), (FEATURES, 32, 3), 1, 1, true),
        oc1,
        0,
    );
    let full = p.buffer(height * width * 32);
    let fi = Img::new(full, 1, height, width, 32);
    p.resize(oc1, fi, Resize::Corners, false);
    let table = tail.push(&uv_positions(height, width, 32, aspect));
    p.table(
        full,
        0,
        height * width * 32,
        table,
        height * width * 32,
        true,
    );
    let mid = p.buffer(height * width * 32);
    let mi = Img::new(mid, 1, height, width, 32);
    p.conv(
        fi,
        &w.cw(&hn("scratch.output_conv2.0"), (32, 32, 3), 1, 1, true),
        mi,
        RELU,
    );
    let out = p.buffer(height * width * 2);
    p.conv(
        mi,
        &w.cw(&hn("scratch.output_conv2.2"), (32, 2, 1), 1, 0, true),
        Img::new(out, 1, height, width, 2),
        0,
    );
    (p, input, out)
}

/// The network on the GPU and the size it is planned for.
struct Planned {
    net: Net,
    /// The size planned, its input's and output's buffers.
    size: Option<(usize, usize, usize, usize)>,
}

/// Depth Anything 3 Small on the GPU.
pub struct GpuDepth {
    weights: Weights,
    on: Mutex<Planned>,
}

/// The size whose constants are the most: a square at the longest side.
const MOST: usize = 504;

impl GpuDepth {
    /// What the model takes on the GPU, from the shapes in its checkpoint
    /// (`model.safetensors`'s bytes): its weights and their constants
    /// while it is set up; while it runs, its largest plan's buffers and
    /// read-back (a square at the longest side) and, on the run that sets
    /// it up, its weights once more while they go up.
    pub fn footprint(checkpoint: Vec<u8>) -> Result<Footprint, String> {
        let weights = Weights::read(&SafeTensors::parse(checkpoint)?)?;
        let base = weights.packer.data.len();
        let mut tail = Packer::at(base);
        let (plan, _, _) = plan(&weights, &mut tail, MOST, MOST);
        let planned: usize = plan.sizes.iter().sum::<usize>() + MOST * MOST * 2;
        Ok(Footprint {
            resident: ((base + tail.data.len()) * 4) as u64,
            working: ((planned + base) * 4) as u64,
        })
    }

    /// The model from its checkpoint (`model.safetensors`'s bytes), its
    /// weights on the GPU (planned when it first runs).
    pub fn new(checkpoint: Vec<u8>) -> Result<GpuDepth, String> {
        let weights = Weights::read(&SafeTensors::parse(checkpoint)?)?;
        let base = weights.packer.data.len();
        let mut tail = Packer::at(base);
        plan(&weights, &mut tail, MOST, MOST);
        let net = Net::new("jodel-gpu depth", &weights.packer.data, tail.data.len())?;
        Ok(GpuDepth {
            weights,
            on: Mutex::new(Planned { net, size: None }),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Planned>, String> {
        self.on
            .lock()
            .map_err(|_| "the depth model failed before".to_owned())
    }

    /// The GPU's name.
    pub fn name(&self) -> String {
        self.lock()
            .map(|p| p.net.gpu_name().to_owned())
            .unwrap_or_default()
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is made instead.
    pub fn is_lost(&self) -> bool {
        self.lock().map_or(true, |p| p.net.lost())
    }

    /// The bytes it keeps on the GPU now.
    pub fn gpu_bytes(&self) -> u64 {
        self.lock().map(|p| p.net.bytes()).unwrap_or(0)
    }

    /// The bytes its weights take on the GPU, and its working buffers while
    /// it runs (the last plan's).
    pub fn gpu_sizes(&self) -> (u64, u64) {
        self.lock().map(|p| p.net.sizes()).unwrap_or((0, 0))
    }

    /// Lets the plan's buffers go until it runs again (the weights stay).
    pub fn release(&self) {
        if let Ok(mut p) = self.lock() {
            p.net.release();
        }
    }

    /// The network planned for `width × height`; its input's and output's
    /// buffers.
    fn planned(
        &self,
        on: &mut Planned,
        width: usize,
        height: usize,
    ) -> Result<(usize, usize), String> {
        if let Some((w, h, input, out)) = on.size {
            if (w, h) == (width, height) && on.net.is_planned() {
                return Ok((input, out));
            }
        }
        on.size = None;
        let mut tail = Packer::at(on.net.tail_origin());
        let (p, input, out) = plan(&self.weights, &mut tail, width, height);
        on.net.plan(&p, &tail.data, width * height * 2)?;
        on.size = Some((width, height, input, out));
        Ok((input, out))
    }

    /// The network's raw output for a normalised input of `width × height`
    /// (both multiples of 14, the longest 504 at most): two channels a
    /// pixel.
    pub fn raw(&self, input: &[f32], width: usize, height: usize) -> Result<Vec<f32>, String> {
        if input.len() != width * height * 3
            || !width.is_multiple_of(14)
            || !height.is_multiple_of(14)
            || width.max(height) > MOST
        {
            return Err(format!(
                "an input of {} values for {width} × {height}",
                input.len()
            ));
        }
        let mut on = self.lock()?;
        let (i, o) = self.planned(&mut on, width, height)?;
        on.net.run(i, input, (o, 0, width * height * 2))
    }

    /// Each kernel's share of a run (see `Net::profile`).
    pub fn profile(
        &self,
        input: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<(String, usize, f64)>, String> {
        let mut on = self.lock()?;
        let (i, _) = self.planned(&mut on, width, height)?;
        on.net.profile(i, input)
    }

    /// The depth of `image`, at the size the network takes it
    /// ([`prep::depth_size`]).
    pub fn depth(&self, image: &Rgb) -> Result<Depth, String> {
        let (input, width, height) = prep::depth_input(image);
        let raw = self.raw(&input, width, height)?;
        Ok(Depth {
            width,
            height,
            values: raw.chunks_exact(2).map(|v| v[0].exp()).collect(),
        })
    }
}
