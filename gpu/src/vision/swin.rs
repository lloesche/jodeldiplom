//! BEN2's backbone: a Swin-B (128 wide, stages of 2, 2, 18 and 2 blocks with
//! 4, 8, 16 and 32 heads, windows of 12, shifted by 6 every other block,
//! each stage's picture padded to whole windows), as the checkpoint's own
//! copy of it runs: a patch embedding of 4 × 4 with a LayerNorm, then the
//! stages, each one's output normed, the next made by merging 2 × 2 patches.

use jodel_tensors::SafeTensors;

use super::ops::{Attend, Img, Plan, ADD, GELU};
use super::Named;

pub(crate) const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const WINDOW: usize = 12;
const EMBED: usize = 128;

fn dim(stage: usize) -> usize {
    EMBED << stage
}

fn heads(stage: usize) -> usize {
    4 << stage
}

/// The backbone's tensors.
pub(crate) fn read(w: &mut Named, t: &SafeTensors) -> Result<(), String> {
    let b = |n: &str| format!("backbone.{n}");
    w.conv(t, &b("patch_embed.proj"), (3, EMBED, 4), true)?;
    w.vector(t, &b("patch_embed.norm.weight"), EMBED)?;
    w.vector(t, &b("patch_embed.norm.bias"), EMBED)?;
    for (s, &depth) in DEPTHS.iter().enumerate() {
        let c = dim(s);
        for k in 0..depth {
            let n = |x: &str| b(&format!("layers.{s}.blocks.{k}.{x}"));
            for norm in ["norm1", "norm2"] {
                w.vector(t, &n(&format!("{norm}.weight")), c)?;
                w.vector(t, &n(&format!("{norm}.bias")), c)?;
            }
            w.linear(t, &n("attn.qkv"), (c, 3 * c), true, None)?;
            w.linear(t, &n("attn.proj"), (c, c), true, None)?;
            w.linear(t, &n("mlp.fc1"), (c, 4 * c), true, None)?;
            w.linear(t, &n("mlp.fc2"), (4 * c, c), true, None)?;
            let rel = (2 * WINDOW - 1) * (2 * WINDOW - 1);
            let table = t.get(&n("attn.relative_position_bias_table"), &[rel, heads(s)])?;
            w.put(&n("attn.relative_position_bias_table"), &table);
        }
        w.vector(t, &b(&format!("norm{s}.weight")), c)?;
        w.vector(t, &b(&format!("norm{s}.bias")), c)?;
        if s < 3 {
            let m = |x: &str| b(&format!("layers.{s}.downsample.{x}"));
            w.vector(t, &m("norm.weight"), 4 * c)?;
            w.vector(t, &m("norm.bias"), 4 * c)?;
            w.linear(t, &m("reduction"), (4 * c, 2 * c), false, None)?;
        }
    }
    Ok(())
}

/// The backbone over `views` (packed, `v × size × size × 3`): the patch
/// embedding and each stage's output, normed (five images).
pub(crate) fn plan(p: &mut Plan, w: &Named, views: Img) -> [Img; 5] {
    let b = |n: &str| format!("backbone.{n}");
    let v = views.views;
    let side = views.h / 4;
    let f0 = p.buffer(v * side * side * EMBED);
    let f0i = Img::new(f0, v, side, side, EMBED);
    p.conv(
        views,
        &w.cw(&b("patch_embed.proj"), (3, EMBED, 4), 4, 0, true),
        f0i,
        0,
    );
    p.layernorm(
        f0i,
        f0i,
        w.at(&b("patch_embed.norm.weight")),
        w.at(&b("patch_embed.norm.bias")),
        1e-5,
    );
    // The working buffers, at the first stage's sizes (the largest).
    let padded = side.next_multiple_of(WINDOW);
    let tokens = v * padded * padded;
    let x = p.buffer(v * side * side * EMBED);
    let xn = p.buffer(v * side * side * EMBED);
    let win = p.buffer(tokens * EMBED);
    let qkv = p.buffer(tokens * 3 * EMBED);
    let att = p.buffer(tokens * EMBED);
    let up = p.buffer(v * side * side * 4 * EMBED);
    let merged = p.buffer(v * side * side * EMBED);
    p.copy_img(f0i, Img::new(x, v, side, side, EMBED), false);
    let mut outs = vec![f0i];
    let mut h = side;
    for (s, &depth) in DEPTHS.iter().enumerate() {
        let c = dim(s);
        let xi = Img::new(x, v, h, h, c);
        let xni = Img::new(xn, v, h, h, c);
        let hp = h.next_multiple_of(WINDOW);
        let nw = hp / WINDOW;
        let rows = v * hp * hp;
        for k in 0..depth {
            let n = |x: &str| b(&format!("layers.{s}.blocks.{k}.{x}"));
            let shift = if k % 2 == 1 { WINDOW / 2 } else { 0 };
            p.layernorm(
                xi,
                xni,
                w.at(&n("norm1.weight")),
                w.at(&n("norm1.bias")),
                1e-5,
            );
            let wi = Img::rows(win, rows, c);
            p.windows(xni, wi, WINDOW, shift, false);
            let qi = Img::rows(qkv, rows, 3 * c);
            p.linear(
                wi,
                w.at(&n("attn.qkv.weight")),
                Some(w.at(&n("attn.qkv.bias"))),
                qi,
                0,
            );
            let ai = Img::rows(att, rows, c);
            p.attend(&Attend {
                q: qi,
                k: Img { off: c, ..qi },
                v: Img { off: 2 * c, ..qi },
                out: ai,
                groups: v * nw * nw,
                nq: WINDOW * WINDOW,
                nk: WINDOW * WINDOW,
                heads: heads(s),
                d: 32,
                swin: Some((
                    w.at(&n("attn.relative_position_bias_table")),
                    WINDOW,
                    shift,
                    nw,
                    nw,
                )),
            });
            // The projection back into the windows' buffer, then added to x.
            p.linear(
                ai,
                w.at(&n("attn.proj.weight")),
                Some(w.at(&n("attn.proj.bias"))),
                wi,
                0,
            );
            p.windows(wi, xi, WINDOW, shift, true);
            p.layernorm(
                xi,
                xni,
                w.at(&n("norm2.weight")),
                w.at(&n("norm2.bias")),
                1e-5,
            );
            let ui = Img::new(up, v, h, h, 4 * c);
            p.linear(
                xni,
                w.at(&n("mlp.fc1.weight")),
                Some(w.at(&n("mlp.fc1.bias"))),
                ui,
                GELU,
            );
            p.linear(
                ui,
                w.at(&n("mlp.fc2.weight")),
                Some(w.at(&n("mlp.fc2.bias"))),
                xi,
                ADD,
            );
        }
        let out = p.buffer(v * h * h * c);
        let oi = Img::new(out, v, h, h, c);
        p.layernorm(
            xi,
            oi,
            w.at(&b(&format!("norm{s}.weight"))),
            w.at(&b(&format!("norm{s}.bias"))),
            1e-5,
        );
        outs.push(oi);
        if s < 3 {
            // x0..x3 = rows and columns even/odd as (0,0), (1,0), (0,1), (1,1).
            let half = h / 2;
            let mi = Img::new(merged, v, half, half, 4 * c);
            for (q, (dy, dx)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                p.copy(
                    [v, half, half, c],
                    (x, (dy * h + dx) * c, [h * h * c, 2 * h * c, 2 * c]),
                    (merged, q * c, [half * half * 4 * c, half * 4 * c, 4 * c]),
                    false,
                );
            }
            let m = |x: &str| b(&format!("layers.{s}.downsample.{x}"));
            p.layernorm(mi, mi, w.at(&m("norm.weight")), w.at(&m("norm.bias")), 1e-5);
            h = half;
            p.linear(
                mi,
                w.at(&m("reduction.weight")),
                None,
                Img::new(x, v, h, h, 2 * c),
                0,
            );
        }
    }
    p.release(&[x, xn, win, qkv, att, up, merged]);
    [outs[0], outs[1], outs[2], outs[3], outs[4]]
}
