//! BEN2 Base (MIT) on the GPU: the matte of a cover's subject.
//!
//! As the checkpoint's own code runs it: the picture at 1024 × 1024 is cut
//! into its four quarters and also shrunk to half, five views of 512 that
//! go through the Swin-B backbone (`swin.rs`) together; each level's
//! features through a 3 × 3 convolution, an instance norm and GELU; the
//! coarsest through "MCLM" (the whole view's tokens attending to pooled
//! quarters, each quarter to its part of the whole), then up level by level
//! through "MCRM" blocks (each quarter, weighed by a saliency map of the
//! whole, attending to pooled parts of the whole); the quarters put back
//! together with the whole, a head of three convolutions, a shallow
//! convolution of the picture added at three sizes on the way up to 1024,
//! and a sigmoid. The shallow features are only ever used shrunk by
//! bilinear halving (to 512 and to 256): a 3 × 3 convolution followed by
//! such a halving is the 4 × 4 convolution of its kernel with a box, which
//! is what runs here (never the 1024 × 1024 × 128 features themselves).
//! The last stage at 1024 × 1024 × 128 is made a band of 128 rows at a time
//! (twice: first for its instance norm's statistics, merged from each
//! band's), so its 512 MB are never held at once.

use std::sync::Mutex;

use jodel_tensors::SafeTensors;

use super::net::Net;
use super::ops::{Act, Attend, Band, ConvW, Img, Plan, Resize, ADD, GELU, NORM_CHUNK, UP_IN};
use super::prep::{self, Rgb};
use super::{swin, Named};
use crate::Footprint;

const SIDE: usize = 1024;
const D: usize = 128;
/// The rows of the last stage made at once.
const BAND: usize = 128;

/// A matte: `width × height` values in 0..=1, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Matte {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

/// DETR's sine positions (normalised, 2π, temperature 10000) of a `h × w`
/// grid: per cell 64 for its row then 64 for its column, each pair a sine
/// and a cosine of one frequency.
fn sine_positions(h: usize, w: usize) -> Vec<f32> {
    let dim_t: Vec<f32> = (0..64)
        .map(|k| 10000f32.powf((2 * (k / 2)) as f32 / 64.0))
        .collect();
    let scale = 2.0 * std::f32::consts::PI;
    let mut out = Vec::with_capacity(h * w * D);
    for y in 0..h {
        for x in 0..w {
            let ye = ((y + 1) as f32 - 0.5) / (h as f32 + 1e-6) * scale;
            let xe = ((x + 1) as f32 - 0.5) / (w as f32 + 1e-6) * scale;
            for e in [ye, xe] {
                for (k, t) in dim_t.iter().enumerate() {
                    let v = e / t;
                    out.push(if k % 2 == 0 { v.sin() } else { v.cos() });
                }
            }
        }
    }
    out
}

/// A 3 × 3 kernel (`[out][in][3][3]`) followed by bilinear halving as the
/// 4 × 4 kernel it makes: each tap the mean of the four the halving takes.
fn halved_kernel(w: &[f32], outs: usize, ins: usize) -> Vec<f32> {
    let mut out = vec![0.0; outs * ins * 16];
    for o in 0..outs {
        for i in 0..ins {
            for y in 0..4 {
                for x in 0..4 {
                    let mut sum = 0.0;
                    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        if (a..a + 3).contains(&y) && (b..b + 3).contains(&x) {
                            sum += w[((o * ins + i) * 3 + (y - a)) * 3 + (x - b)];
                        }
                    }
                    out[((o * ins + i) * 4 + y) * 4 + x] = 0.25 * sum;
                }
            }
        }
    }
    out
}

/// The checkpoint's tensors this port uses.
fn read(t: &SafeTensors) -> Result<Named, String> {
    let mut w = Named::default();
    swin::read(&mut w, t)?;
    let shallow = t.get("shallow.0.weight", &[D, 3, 3, 3])?;
    let halved = halved_kernel(&shallow, D, 3);
    w.put_conv("shallow.half.weight", &halved, (3, D, 4));
    w.vector(t, "shallow.0.bias", D)?;
    let ins = [128, 128, 256, 512, 1024];
    for (k, &c) in ins.iter().enumerate() {
        w.conv(t, &format!("output{}.0", k + 1), (c, D, 3), true)?;
    }
    for k in 1..=4 {
        w.conv(t, &format!("conv{k}.0"), (D, D, 3), true)?;
    }
    let attention = |w: &mut Named, name: &str| -> Result<(), String> {
        let inw = t.get(&format!("{name}.in_proj_weight"), &[3 * D, D])?;
        let inb = t.get(&format!("{name}.in_proj_bias"), &[3 * D])?;
        w.put_linear(&format!("{name}.q.weight"), &inw[..D * D], D, D);
        w.put(&format!("{name}.q.bias"), &inb[..D]);
        w.put_linear(&format!("{name}.k.weight"), &inw[D * D..2 * D * D], D, D);
        w.put(&format!("{name}.k.bias"), &inb[D..2 * D]);
        w.put_linear(&format!("{name}.v.weight"), &inw[2 * D * D..], D, D);
        w.put(&format!("{name}.v.bias"), &inb[2 * D..]);
        w.linear(t, &format!("{name}.out_proj"), (D, D), true, None)
    };
    let ffn = |w: &mut Named, name: &str, pairs: &[(usize, usize)]| -> Result<(), String> {
        for &(a, b) in pairs {
            w.linear(t, &format!("{name}.linear{a}"), (D, 2 * D), true, None)?;
            w.linear(t, &format!("{name}.linear{b}"), (2 * D, D), true, None)?;
        }
        for norm in ["norm1", "norm2"] {
            w.vector(t, &format!("{name}.{norm}.weight"), D)?;
            w.vector(t, &format!("{name}.{norm}.bias"), D)?;
        }
        Ok(())
    };
    for a in 0..5 {
        attention(&mut w, &format!("multifieldcrossatt.attention.{a}"))?;
    }
    ffn(&mut w, "multifieldcrossatt", &[(1, 2), (3, 4)])?;
    for k in 1..=4 {
        let name = format!("dec_blk{k}");
        for a in 0..4 {
            attention(&mut w, &format!("{name}.attention.{a}"))?;
        }
        ffn(&mut w, &name, &[(3, 4)])?;
        w.conv(t, &format!("{name}.sal_conv"), (D, 1, 1), true)?;
    }
    w.conv(t, "insmask_head.0", (D, 384, 3), true)?;
    w.conv(t, "insmask_head.3", (384, 384, 3), true)?;
    w.conv(t, "insmask_head.6", (384, D, 3), true)?;
    w.conv(t, "upsample1.0", (D, D, 3), true)?;
    w.conv(t, "upsample2.0", (D, D, 3), true)?;
    w.conv(t, "output.0", (D, 1, 3), true)?;
    for (name, h) in [
        ("mclm.g", 16),
        ("mclm.p16", 16),
        ("mclm.p4", 4),
        ("mclm.p2", 2),
    ] {
        w.put(name, &sine_positions(h, h));
    }
    Ok(w)
}

/// Scratch buffers the attention blocks share.
struct Scratch {
    q: usize,
    kv: usize,
    att: usize,
    up: usize,
    pools: usize,
    side: usize,
}

/// nn.MultiheadAttention (one head of 128): `out` (+)= out_proj(attention
/// of q_in's projections to k_in's and v_in's).
#[allow(clippy::too_many_arguments)]
fn mha(
    p: &mut Plan,
    w: &Named,
    s: &Scratch,
    name: &str,
    q_in: Img,
    k_in: Img,
    v_in: Img,
    out: Img,
    flags: u32,
) {
    let nq = q_in.pixels();
    let nk = k_in.pixels();
    let at = |x: &str| w.at(&format!("{name}.{x}"));
    let q = Img::rows(s.q, nq, D);
    p.linear(q_in, at("q.weight"), Some(at("q.bias")), q, 0);
    let kv = Img::rows(s.kv, nk, 2 * D);
    p.linear(
        k_in,
        at("k.weight"),
        Some(at("k.bias")),
        kv.channels(0, D),
        0,
    );
    p.linear(
        v_in,
        at("v.weight"),
        Some(at("v.bias")),
        kv.channels(D, D),
        0,
    );
    let a = Img::rows(s.att, nq, D);
    p.attend(&Attend {
        q,
        k: kv.channels(0, D),
        v: kv.channels(D, D),
        out: a,
        groups: 1,
        nq,
        nk,
        heads: 1,
        d: D,
        swin: None,
    });
    p.linear(
        a,
        at("out_proj.weight"),
        Some(at("out_proj.bias")),
        out,
        flags,
    );
}

/// x = norm2(norm1(x) + ffn(norm1(x))) in place (the blocks' tail).
fn tail(p: &mut Plan, w: &Named, s: &Scratch, name: &str, x: Img, (a, b): (usize, usize)) {
    let at = |x: &str| w.at(&format!("{name}.{x}"));
    p.layernorm(x, x, at("norm1.weight"), at("norm1.bias"), 1e-5);
    let u = Img::rows(s.up, x.pixels(), 2 * D);
    p.linear(
        x,
        at(&format!("linear{a}.weight")),
        Some(at(&format!("linear{a}.bias"))),
        u,
        GELU,
    );
    p.linear(
        u,
        at(&format!("linear{b}.weight")),
        Some(at(&format!("linear{b}.bias"))),
        x,
        ADD,
    );
    p.layernorm(x, x, at("norm2.weight"), at("norm2.bias"), 1e-5);
}

/// A 3 × 3 convolution, an instance norm and GELU (`make_cbr`), `x` into
/// `out`.
fn cbr(p: &mut Plan, w: &Named, name: &str, x: Img, out: Img, flags: u32) {
    let (cin, cout) = (x.c, out.c);
    p.conv(x, &w.cw(name, (cin, cout, 3), 1, 1, true), out, flags);
    p.instance_norm(out, true);
}

/// MCLM on `e5` (five views of 16 × 16: four quarters, then the whole), in
/// place.
fn mclm(p: &mut Plan, w: &Named, s: &Scratch, e5: Img) {
    let n = "multifieldcrossatt";
    let h = e5.h;
    let loc = e5.views(0, 4);
    let glb = e5.views(4, 1);
    // The quarters put together (32 × 32), pooled to 16, 4 and 2.
    let big = Img::new(s.side, 1, 2 * h, 2 * h, D);
    p.quarters(loc, big, true);
    let pools = s.pools;
    let mut at = 0;
    for r in [1, 4, 8] {
        let side = h / r;
        let pi = Img {
            off: at * D,
            ..Img::new(pools, 1, side, side, D)
        };
        p.pool(big, pi);
        at += side * side;
    }
    let all = Img::rows(pools, at, D);
    // Keys with positions, queries with positions (in their own buffers).
    let keys = p.buffer(at * D);
    p.copy_img(all, Img::rows(keys, at, D), false);
    let mut off = 0;
    for (name, side) in [("mclm.p16", 16), ("mclm.p4", 4), ("mclm.p2", 2)] {
        p.table(
            keys,
            off * D,
            side * side * D,
            w.at(name),
            side * side * D,
            true,
        );
        off += side * side;
    }
    let g = p.buffer(h * h * D);
    let gq = p.buffer(h * h * D);
    let gi = Img::rows(g, h * h, D);
    p.copy_img(glb, gi, false);
    p.copy_img(glb, Img::rows(gq, h * h, D), false);
    p.table(gq, 0, h * h * D, w.at("mclm.g"), h * h * D, true);
    mha(
        p,
        w,
        s,
        &format!("{n}.attention.0"),
        Img::rows(gq, h * h, D),
        Img::rows(keys, at, D),
        all,
        gi,
        ADD,
    );
    tail(p, w, s, n, gi, (1, 2));
    // Each quarter attends to its part of the refreshed whole.
    let parts = p.buffer(h * h * D);
    let pi = Img::new(parts, 4, h / 2, h / 2, D);
    p.quarters(gi.shaped(1, h, h), pi, false);
    for v in 0..4 {
        let lv = loc.views(v, 1);
        let kv = pi.views(v, 1);
        mha(
            p,
            w,
            s,
            &format!("{n}.attention.{}", v + 1),
            lv,
            kv,
            kv,
            lv,
            ADD,
        );
    }
    tail(p, w, s, n, loc, (3, 4));
    p.copy_img(gi, glb, false);
    p.release(&[keys, g, gq, parts]);
}

/// MCRM on `x` (five views of h × h), in place.
fn mcrm(p: &mut Plan, w: &Named, s: &Scratch, name: &str, x: Img) {
    let h = x.h;
    let loc = x.views(0, 4);
    let glb = x.views(4, 1);
    // The whole's quarters, pooled for each quarter's keys.
    let parts = p.buffer(h * h * D);
    let pi = Img::new(parts, 4, h / 2, h / 2, D);
    p.quarters(glb, pi, false);
    // The saliency map of the whole weighs the quarters.
    let map = p.buffer(h * h);
    let mi = Img::new(map, 1, h, h, 1);
    p.conv(
        glb,
        &w.cw(&format!("{name}.sal_conv"), (D, 1, 1), 1, 0, true),
        mi,
        0,
    );
    p.act(mi, Act::Sigmoid);
    p.mulmap(loc, mi);
    let sizes: Vec<usize> = [2, 4, 8].iter().map(|r| h / r).collect();
    let count: usize = sizes.iter().map(|s| s * s).sum();
    let pools = p.buffer(count * D);
    for v in 0..4 {
        let mut off = 0;
        for &side in &sizes {
            p.pool(
                pi.views(v, 1),
                Img {
                    off: off * D,
                    ..Img::new(pools, 1, side, side, D)
                },
            );
            off += side * side;
        }
        let keys = Img::rows(pools, count, D);
        let lv = loc.views(v, 1);
        mha(
            p,
            w,
            s,
            &format!("{name}.attention.{v}"),
            lv,
            keys,
            keys,
            lv,
            ADD,
        );
    }
    tail(p, w, s, name, loc, (3, 4));
    // The whole gains the quarters put together, at its size (nearest).
    let big = Img::new(s.side, 1, 2 * h, 2 * h, D);
    p.quarters(loc, big, true);
    p.resize(big, glb, Resize::Nearest, true);
    p.release(&[parts, map, pools]);
}

/// The plan; the input's buffer (1024 × 1024 × 3) and the matte's.
fn plan(w: &Named) -> (Plan, usize, usize) {
    let mut p = Plan::new();
    let input = p.buffer(SIDE * SIDE * 3);
    let ii = Img::new(input, 1, SIDE, SIDE, 3);
    let half = SIDE / 2;
    let views = p.buffer(5 * half * half * 3);
    let vi = Img::new(views, 5, half, half, 3);
    p.quarters(ii, vi.views(0, 4), false);
    p.resize(ii, vi.views(4, 1), Resize::Bilinear, false);
    let f = swin::plan(&mut p, w, vi);
    p.release(&[views]);

    let scratch = Scratch {
        q: p.buffer(4 * 128 * 128 * D),
        kv: p.buffer(64 * 64 * 2 * D * 2),
        att: p.buffer(128 * 128 * D),
        up: p.buffer(4 * 128 * 128 * 2 * D),
        pools: p.buffer(64 * 64 * D * 2),
        side: p.buffer(256 * 256 * D),
    };
    let es: Vec<Img> = f
        .iter()
        .enumerate()
        .map(|(k, fi)| {
            let e = p.buffer(5 * fi.h * fi.w * D);
            let ei = Img::new(e, 5, fi.h, fi.w, D);
            cbr(&mut p, w, &format!("output{}.0", k + 1), *fi, ei, 0);
            ei
        })
        .collect();
    p.release(&f.map(|fi| fi.buf));
    let (e1, e2, e3, e4, e5) = (es[0], es[1], es[2], es[3], es[4]);
    mclm(&mut p, w, &scratch, e5);
    let tmp = p.buffer(5 * 128 * 128 * D);
    let mut coarse = e5;
    for (k, e) in [(4, e4), (3, e3), (2, e2), (1, e1)] {
        p.resize(coarse, e, Resize::Bilinear, true);
        mcrm(&mut p, w, &scratch, &format!("dec_blk{k}"), e);
        let ti = Img::new(tmp, 5, e.h, e.w, D);
        cbr(&mut p, w, &format!("conv{k}.0"), e, ti, 0);
        p.copy_img(ti, e, false);
        coarse = e;
    }
    p.release(&[
        e2.buf,
        e3.buf,
        e4.buf,
        e5.buf,
        scratch.q,
        scratch.kv,
        scratch.att,
        scratch.up,
        scratch.pools,
    ]);
    // The quarters together (256 × 256) and the whole added.
    let cat = Img::new(scratch.side, 1, 256, 256, D);
    p.quarters(e1.views(0, 4), cat, true);
    p.resize(e1.views(4, 1), cat, Resize::Bilinear, true);
    let m1 = p.buffer(256 * 256 * 384);
    let m2 = p.buffer(256 * 256 * 384);
    // m1 holds the 512 × 512 features later.
    p.grow(m1, 512 * 512 * D);
    let m1i = Img::new(m1, 1, 256, 256, 384);
    let m2i = Img::new(m2, 1, 256, 256, 384);
    cbr(&mut p, w, "insmask_head.0", cat, m1i, 0);
    cbr(&mut p, w, "insmask_head.3", m1i, m2i, 0);
    let fin = Img::new(tmp, 1, 256, 256, D);
    p.conv(
        m2i,
        &w.cw("insmask_head.6", (384, D, 3), 1, 1, true),
        fin,
        0,
    );
    p.release(&[m2, e1.buf]);
    // The shallow features, halved twice (256) and once (512).
    let shallow = |stride: usize, pad: usize| ConvW {
        at: w.at("shallow.half.weight"),
        bias: Some(w.at("shallow.0.bias")),
        cin: 3,
        cout: D,
        k: 4,
        stride,
        pad,
        up: 1,
    };
    p.conv(ii, &shallow(4, 0), fin, ADD);
    let f512 = Img::new(m1, 1, 512, 512, D);
    cbr(&mut p, w, "upsample1.0", fin, f512, UP_IN);
    p.conv(ii, &shallow(2, 1), f512, ADD);
    // The last stage (1024 × 1024 × 128) a band of rows at a time: its
    // instance norm's statistics from each band, then each band made again
    // (with a row above and below, for the output's 3 × 3), normed, GELU,
    // and through the output convolution.
    let up2 = w.cw("upsample2.0", (D, D, 3), 1, 1, true);
    let last = w.cw("output.0", (D, 1, 3), 1, 1, true);
    let band = p.buffer((BAND + 2) * SIDE * D);
    let chunks = SIDE * SIDE / NORM_CHUNK;
    for y0 in (0..SIDE).step_by(BAND) {
        let bi = Img::new(band, 1, BAND, SIDE, D);
        let rows = Band {
            out_row: y0,
            in_row: 0,
            rows_in_all: SIDE,
        };
        p.conv_band(f512, &up2, bi, UP_IN, rows);
        p.norm_stats(bi, y0 * SIDE / NORM_CHUNK, chunks);
    }
    p.norm_merge(1, SIDE * SIDE, D);
    let out = p.buffer(SIDE * SIDE);
    for y0 in (0..SIDE).step_by(BAND) {
        let (lo, hi) = (y0.saturating_sub(1), (y0 + BAND + 1).min(SIDE));
        let bi = Img::new(band, 1, hi - lo, SIDE, D);
        let rows = Band {
            out_row: lo,
            in_row: 0,
            rows_in_all: SIDE,
        };
        p.conv_band(f512, &up2, bi, UP_IN, rows);
        p.norm_apply(bi, true);
        let oi = Img {
            off: y0 * SIDE,
            ..Img::new(out, 1, BAND, SIDE, 1)
        };
        let rows = Band {
            out_row: y0,
            in_row: lo,
            rows_in_all: SIDE,
        };
        p.conv_band(bi, &last, oi, 0, rows);
    }
    p.act(Img::new(out, 1, SIDE, SIDE, 1), Act::Sigmoid);
    (p, input, out)
}

/// BEN2 Base on the GPU.
pub struct GpuMatte {
    plan: Plan,
    input: usize,
    out: usize,
    net: Mutex<Net>,
}

impl GpuMatte {
    /// What the model takes on the GPU, from the shapes in its checkpoint
    /// (`model.safetensors`'s bytes): its weights while it is set up;
    /// while it runs, its plan's buffers and read-back and, on the run that
    /// sets it up, its weights once more while they go up.
    pub fn footprint(checkpoint: Vec<u8>) -> Result<Footprint, String> {
        let w = read(&SafeTensors::parse(checkpoint)?)?;
        let (plan, _, _) = plan(&w);
        let base = w.packer.data.len();
        let planned: usize = plan.sizes.iter().sum::<usize>() + SIDE * SIDE;
        Ok(Footprint {
            resident: (base * 4) as u64,
            working: ((planned + base) * 4) as u64,
        })
    }

    /// The model from its checkpoint (`model.safetensors`'s bytes), its
    /// weights on the GPU (planned when it first runs).
    pub fn new(checkpoint: Vec<u8>) -> Result<GpuMatte, String> {
        let w = read(&SafeTensors::parse(checkpoint)?)?;
        let (plan, input, out) = plan(&w);
        let net = Net::new("jodel-gpu matte", &w.packer.data, 0)?;
        Ok(GpuMatte {
            plan,
            input,
            out,
            net: Mutex::new(net),
        })
    }

    /// The network, planned.
    fn planned(&self) -> Result<std::sync::MutexGuard<'_, Net>, String> {
        let mut net = self
            .net
            .lock()
            .map_err(|_| "the matte model failed before".to_owned())?;
        if !net.is_planned() {
            net.plan(&self.plan, &[], SIDE * SIDE)?;
        }
        Ok(net)
    }

    /// The GPU's name.
    pub fn name(&self) -> String {
        self.net
            .lock()
            .map(|n| n.gpu_name().to_owned())
            .unwrap_or_default()
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is made instead.
    pub fn is_lost(&self) -> bool {
        self.net.lock().map_or(true, |n| n.lost())
    }

    /// The bytes it keeps on the GPU now.
    pub fn gpu_bytes(&self) -> u64 {
        self.net.lock().map(|n| n.bytes()).unwrap_or(0)
    }

    /// The bytes its weights take on the GPU, and its working buffers while
    /// it runs (the last plan's).
    pub fn gpu_sizes(&self) -> (u64, u64) {
        self.net.lock().map(|n| n.sizes()).unwrap_or((0, 0))
    }

    /// Lets the plan's buffers go until it runs again (the weights stay).
    pub fn release(&self) {
        if let Ok(mut n) = self.net.lock() {
            n.release();
        }
    }

    /// The network's sigmoid for a normalised 1024 × 1024 input.
    pub fn raw(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != SIDE * SIDE * 3 {
            return Err(format!(
                "an input of {} values, not 1024 × 1024 × 3",
                input.len()
            ));
        }
        self.planned()?
            .run(self.input, input, (self.out, 0, SIDE * SIDE))
    }

    /// Each kernel's share of a run (see `Net::profile`).
    pub fn profile(&self, input: &[f32]) -> Result<Vec<(String, usize, f64)>, String> {
        self.planned()?.profile(self.input, input)
    }

    /// The matte of `image` at 1024 × 1024, stretched to its range as BEN2's
    /// own code does (its least 0, its most 1).
    pub fn matte(&self, image: &Rgb) -> Result<Matte, String> {
        let mut values = self.raw(&prep::matte_input(image))?;
        let (lo, hi) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let span = (hi - lo).max(1e-6);
        for v in &mut values {
            *v = (*v - lo) / span;
        }
        Ok(Matte {
            width: SIDE,
            height: SIDE,
            values,
        })
    }
}
