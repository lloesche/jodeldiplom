//! The screen's edges on fire: a pass over the whole finished frame.
//! Tongues of flame lick in from the edges, tallest at the bottom, shorter
//! up the sides, short along the top. They are white-yellow at their
//! roots and orange to red at their tips, rising and flickering, with a
//! warm glow behind them and embers drifting up out of them. The lyric band
//! is kept clear: anything of the fire inside it shows at [`IN_BAND`] at
//! most.
//!
//! [`edge_fire`] is the shader's reference ([`wgsl`]: `fx_edge_fire`),
//! held to it on the GPU ([`REFERENCE`]). It is drawn only in strips along
//! the edges, as far in as the fire can reach at its strength
//! ([`strips`]), and everything past the flames' reach in them returns at
//! once.
//!
//! Its look ([`Spec::params`], `p`): `x` 0 classic orange to 1 a ghostly
//! green, `y` how fast it flickers (1 as made).

use super::helpers::{clamp01, heat, lattice, mix3, smoothstep, speed, vnoise};
use super::{At, Fx, Layer, Live, Out, Reference, Slot, Spec};
use crate::gui::stage_gpu::Instance;

/// The fire: a strip along an edge (`quad`), the lyric band's top and
/// bottom in `shape[0..2]`, `params[1]` its strength (its look the
/// shaders' `looks`).
pub const EDGE_FIRE: f32 = Fx::EdgeFire.kind(0);

/// The screen's edges on fire.
pub const SPEC: Spec = Spec {
    name: "edge fire",
    ident: "edge_fire",
    at: &[Layer::Screen],
    draw,
    kinds: &["edge fire"],
    wgsl: Some(wgsl),
    params: [0.0, 1.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let &At::Screen { area, band, .. } = at else {
        return;
    };
    if live.strength <= 0.0 {
        return;
    }
    out.shapes.extend(
        strips(area, live.strength)
            .into_iter()
            .map(|quad| Instance {
                quad,
                shape: [band.0, band.1, 0.0, 0.0],
                params: [EDGE_FIRE, live.strength, 0.0, 0.0],
                ..Instance::default()
            }),
    );
}

/// The strips of `area` (left, top, width, height) the fire can reach at
/// `strength`, none over another: along the bottom and the top the whole
/// width, up the sides between them.
pub fn strips((x, y, w, h): (f32, f32, f32, f32), strength: f32) -> Vec<[f32; 4]> {
    let reach = h * REACH * (0.3 + 0.7 * strength.min(1.0)) * OUT + 2.0;
    let bottom = reach.min(h);
    let top = (reach * TOP).min(h - bottom);
    let side = (reach * SIDE).min(w / 2.0);
    let (y0, y1) = (y + top, y + h - bottom);
    let mut strips = vec![[x, y1, x + w, y + h]];
    if top > 0.0 {
        strips.push([x, y, x + w, y0]);
    }
    if y1 > y0 {
        strips.push([x, y0, x + side, y1]);
        strips.push([x + w - side, y0, x + w, y1]);
    }
    strips
}

/// How far the flames reach at the bottom at full strength, a share of
/// the frame's height: their tongues' tips, on average (the tallest go
/// half as far again).
const REACH: f32 = 0.14;
/// How far they reach up the sides and along the top, a share of how far
/// at the bottom.
const SIDE: f32 = 0.6;
const TOP: f32 = 0.25;
/// How deep (in reaches) anything of the fire goes: its embers.
const OUT: f32 = 2.4;
/// How much of the fire shows inside the lyric band at most.
pub const IN_BAND: f32 = 0.12;
/// How far round the lyric band the flames shrink back from it, a share
/// of the frame's height.
const FEATHER: f32 = 0.06;
/// The width of a tongue of flame, a share of the frame's height.
const TONGUE: f32 = 0.05;
/// The embers' cells, a share of the frame's height: one in a few has an
/// ember.
const EMBER: f32 = 0.028;

/// The fire at pixel `px` of a frame `size` (pixels) at `time` (seconds),
/// `strength` 0 (none) to 1 (the flames reaching about a sixth of the way
/// in at the bottom), `band` the lyric band's top and bottom (pixels; none
/// where the bottom is not below the top), `p` its look: `x` 0 classic
/// orange to 1 a ghostly green, `y` how fast it flickers (1 as made; 0
/// taken as 1). Premultiplied colour and coverage, in the stage's colours
/// (the frame's bytes; an sRGB frame takes them through its `out_rgb`).
pub fn edge_fire(
    px: (f32, f32),
    size: (f32, f32),
    time: f32,
    strength: f32,
    band: (f32, f32),
    p: [f32; 4],
) -> [f32; 4] {
    if strength <= 0.0 {
        return [0.0; 4];
    }
    let s = strength.min(1.0);
    let h = size.1;
    // The lyric band (its edges feathered wide): the flames shrink back
    // as they come to it, and only a little of anything shows in it.
    let f = h * FEATHER;
    let inside = smoothstep(band.0 - f, band.0, px.1)
        * (1.0 - smoothstep(band.1, band.1 + f, px.1))
        * f32::from(u8::from(band.1 > band.0));
    let reach = h * REACH * (0.3 + 0.7 * s) * (1.0 - 0.6 * inside);
    // How deep in, in reaches, from the nearest edge.
    let u = ((size.1 - px.1) / reach)
        .min(px.0 / (reach * SIDE))
        .min((size.0 - px.0) / (reach * SIDE))
        .min(px.1 / (reach * TOP));
    if u > OUT {
        return [0.0; 4];
    }
    let t = time * speed(p);
    // The tongues: noise stretched upward, rising, swaying on a slower
    // noise; each reaches as far as the noise is high, white hot at its
    // root, red at its tip.
    let l = h * TONGUE;
    let (x, y) = (px.0 / l, px.1 / (l * 1.7));
    let sway = vnoise(x * 0.5 + 3.3, y * 0.5 + t * 1.2) - 0.5;
    let n = 0.65 * vnoise(x + sway * 1.5, y + t * 1.9)
        + 0.35 * vnoise(x * 2.3 + 5.2, y * 2.3 + t * 3.7);
    let tall = smoothstep(0.2, 0.8, n);
    let tip = 0.2 + 1.25 * tall * tall;
    let k = clamp01(1.0 - u / tip);
    let flame_a = smoothstep(0.0, 0.45, k) * 0.95;
    let flame = heat(0.22 + 0.78 * k * k, p[0]);
    // A warm light behind them, close to the edge.
    let glow_a = (-u * 2.2).exp() * 0.22;
    let glow = heat(0.45, p[0]);
    let mut a = flame_a + glow_a * (1.0 - flame_a);
    let mut rgb = [0, 1, 2].map(|i| flame[i] * flame_a + glow[i] * glow_a * (1.0 - flame_a));
    // The embers: one in a few cells of a grid rising (each column at its
    // own speed), swaying, twinkling, fading as they rise.
    let c = h * EMBER;
    let column = (px.0 / c).floor();
    let rise = c * (2.0 + 2.5 * lattice(column, 7.0)) * t;
    let row = ((px.1 + rise) / c).floor();
    let r = lattice(column, row);
    if r > 0.8 {
        let drift = (t * 2.3 + r * 40.0).sin() * 0.15;
        let ex = (column + 0.2 + 0.6 * lattice(column + 31.0, row) + drift) * c;
        let ey = (row + 0.5) * c - rise;
        let d2 = (px.0 - ex).powi(2) + (px.1 - ey).powi(2);
        let big = c * (0.06 + 0.08 * lattice(column, row + 17.0));
        let twinkle = 0.65 + 0.35 * (t * 9.0 + r * 70.0).sin();
        let e = (-d2 / (big * big)).exp() * twinkle * (1.0 - smoothstep(0.9, OUT, u));
        let ember = mix3(heat(0.7, p[0]), [1.0; 3], (-d2 / (big * big * 0.15)).exp());
        rgb = [0, 1, 2].map(|i| ember[i] * e + rgb[i] * (1.0 - e));
        a = e + a * (1.0 - e);
    }
    // Coming in with the strength.
    let keep = (1.0 - (1.0 - IN_BAND) * inside) * smoothstep(0.0, 0.15, s);
    [rgb[0] * keep, rgb[1] * keep, rgb[2] * keep, a * keep]
}

/// [`edge_fire`] in WGSL: `fx_edge_fire` and its helpers (each named
/// `edge_fire_...`), and its strips drawn with it (`edge_fire_shape`: its
/// colours the stage's, as the frame takes them through `out_rgb`).
pub fn wgsl() -> String {
    format!(
        "const EDGE_FIRE_REACH: f32 = {REACH:?};\nconst EDGE_FIRE_SIDE: f32 = {SIDE:?};\nconst EDGE_FIRE_TOP: f32 = {TOP:?};\nconst EDGE_FIRE_OUT: f32 = {OUT:?};\nconst EDGE_FIRE_IN_BAND: f32 = {IN_BAND:?};\nconst EDGE_FIRE_FEATHER: f32 = {FEATHER:?};\nconst EDGE_FIRE_TONGUE: f32 = {TONGUE:?};\nconst EDGE_FIRE_EMBER: f32 = {EMBER:?};\n{SHADER}"
    )
}

const SHADER: &str = r"
fn edge_fire_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let c = fx_edge_fire(p, screen.size, screen.time, o.params.y, o.shape.xy, looks[FX_EDGE_FIRE]);
    return vec4<f32>(out_rgb(c.rgb), c.a);
}

fn fx_edge_fire(px: vec2<f32>, size: vec2<f32>, time: f32, strength: f32, band: vec2<f32>, p: vec4<f32>) -> vec4<f32> {
    if strength <= 0.0 {
        return vec4<f32>(0.0);
    }
    let s = min(strength, 1.0);
    let h = size.y;
    let f = h * EDGE_FIRE_FEATHER;
    let inside = smoothstep(band.x - f, band.x, px.y)
        * (1.0 - smoothstep(band.y, band.y + f, px.y))
        * select(0.0, 1.0, band.y > band.x);
    let reach = h * EDGE_FIRE_REACH * (0.3 + 0.7 * s) * (1.0 - 0.6 * inside);
    let u = min(
        min((size.y - px.y) / reach, px.x / (reach * EDGE_FIRE_SIDE)),
        min((size.x - px.x) / (reach * EDGE_FIRE_SIDE), px.y / (reach * EDGE_FIRE_TOP)),
    );
    if u > EDGE_FIRE_OUT {
        return vec4<f32>(0.0);
    }
    let t = time * fx_speed(p);
    let l = h * EDGE_FIRE_TONGUE;
    let x = px.x / l;
    let y = px.y / (l * 1.7);
    let sway = vnoise(x * 0.5 + 3.3, y * 0.5 + t * 1.2) - 0.5;
    let n = 0.65 * vnoise(x + sway * 1.5, y + t * 1.9)
        + 0.35 * vnoise(x * 2.3 + 5.2, y * 2.3 + t * 3.7);
    let tall = smoothstep(0.2, 0.8, n);
    let tip = 0.2 + 1.25 * tall * tall;
    let k = clamp(1.0 - u / tip, 0.0, 1.0);
    let flame_a = smoothstep(0.0, 0.45, k) * 0.95;
    let flame = fx_heat(0.22 + 0.78 * k * k, p.x);
    let glow_a = exp(-u * 2.2) * 0.22;
    let glow = fx_heat(0.45, p.x);
    var a = flame_a + glow_a * (1.0 - flame_a);
    var rgb = flame * flame_a + glow * glow_a * (1.0 - flame_a);
    let c = h * EDGE_FIRE_EMBER;
    let column = floor(px.x / c);
    let rise = c * (2.0 + 2.5 * lattice(column, 7.0)) * t;
    let row = floor((px.y + rise) / c);
    let r = lattice(column, row);
    if r > 0.8 {
        let drift = sin(t * 2.3 + r * 40.0) * 0.15;
        let ex = (column + 0.2 + 0.6 * lattice(column + 31.0, row) + drift) * c;
        let ey = (row + 0.5) * c - rise;
        let d2 = (px.x - ex) * (px.x - ex) + (px.y - ey) * (px.y - ey);
        let big = c * (0.06 + 0.08 * lattice(column, row + 17.0));
        let twinkle = 0.65 + 0.35 * sin(t * 9.0 + r * 70.0);
        let e = exp(-d2 / (big * big)) * twinkle * (1.0 - smoothstep(0.9, EDGE_FIRE_OUT, u));
        let ember = mix(fx_heat(0.7, p.x), vec3<f32>(1.0), exp(-d2 / (big * big * 0.15)));
        rgb = ember * e + rgb * (1.0 - e);
        a = e + a * (1.0 - e);
    }
    let keep = (1.0 - (1.0 - EDGE_FIRE_IN_BAND) * inside) * smoothstep(0.0, 0.15, s);
    return vec4<f32>(rgb * keep, a * keep);
}
";

/// The fire at pixels near every edge, in and round the lyric band, at
/// strengths, times and looks: the shader's `fx_edge_fire` as
/// [`edge_fire`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        let size = (3024.0f32, 1964.0f32);
        let band = (1330.0, 1650.0);
        (0..3000)
            .flat_map(|k| {
                let f = k as f32;
                let fr = |m: f32| crate::gui::rainbow::fract(f * m);
                // Most within a fifth of an edge.
                let (x, y) = match k % 5 {
                    0 => (fr(0.613) * size.0, size.1 - fr(0.377) * 450.0),
                    1 => (fr(0.613) * 450.0, fr(0.377) * size.1),
                    2 => (size.0 - fr(0.613) * 450.0, fr(0.377) * size.1),
                    3 => (fr(0.613) * size.0, fr(0.377) * 250.0),
                    _ => (fr(0.613) * size.0, fr(0.377) * size.1),
                };
                let strength = [0.0, 0.3, 0.6, 1.0, 1.4][(k / 5) % 5];
                [
                    [x, y, fr(0.291) * 30.0, strength],
                    [size.0, size.1, band.0, band.1],
                    [fr(0.733), fr(0.417) * 2.0, 0.0, 0.0],
                ]
            })
            .collect()
    },
    ins: 3,
    outs: 1,
    body: "
    let a = ins[i * 3u];
    let b = ins[i * 3u + 1u];
    outs[i] = fx_edge_fire(a.xy, b.xy, a.z, a.w, b.zw, ins[i * 3u + 2u]);",
    cpu: |s| {
        let px = (s[0][0], s[0][1]);
        let size = (s[1][0], s[1][1]);
        let band = (s[1][2], s[1][3]);
        vec![edge_fire(px, size, s[0][2], s[0][3], band, s[2])]
    },
    near: (4e-3, 0.0),
    shows: |o| o[0][3] > 0.05,
    least: 600,
};

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (f32, f32) = (3024.0, 1964.0);
    const BAND: (f32, f32) = (1330.0, 1650.0);

    /// Pixels over the frame, on a grid `step` apart.
    fn grid(step: usize) -> impl Iterator<Item = (f32, f32)> {
        (0..SIZE.1 as usize).step_by(step).flat_map(move |y| {
            (0..SIZE.0 as usize)
                .step_by(step)
                .map(move |x| (x as f32, y as f32))
        })
    }

    #[test]
    fn at_no_strength_there_is_none() {
        for px in grid(37) {
            for s in [0.0, -0.5] {
                assert_eq!(edge_fire(px, SIZE, 3.3, s, BAND, [0.0; 4]), [0.0; 4]);
            }
        }
    }

    #[test]
    fn it_is_premultiplied_and_bounded() {
        for (k, px) in grid(13).enumerate() {
            let s = [0.3, 0.6, 1.0, 1.7][k % 4];
            let c = edge_fire(
                px,
                SIZE,
                k as f32 * 0.01,
                s,
                BAND,
                [(k % 2) as f32, 1.0, 0.0, 0.0],
            );
            assert!((0.0..=1.0).contains(&c[3]), "{px:?}: {c:?}");
            for v in &c[..3] {
                assert!(*v >= 0.0 && *v <= c[3] + 1e-5, "{px:?}: {c:?}");
            }
        }
    }

    #[test]
    fn it_burns_at_the_edges_and_leaves_the_middle_alone() {
        let mid = edge_fire((1512.0, 900.0), SIZE, 1.0, 1.0, BAND, [0.0; 4]);
        assert_eq!(mid, [0.0; 4]);
        // Along the bottom, most of its pixels covered at full strength.
        let lit = (0..300)
            .filter(|i| {
                edge_fire(
                    (*i as f32 * 10.0, 1950.0),
                    SIZE,
                    2.0,
                    1.0,
                    (0.0, 0.0),
                    [0.0; 4],
                )[3] > 0.5
            })
            .count();
        assert!(lit > 250, "{lit}");
        // Tallest at the bottom: more of it there than up a side or at the top.
        // Its cover summed in from an edge (`d` in, `along` it).
        let cover = |f: &dyn Fn(f32, f32) -> (f32, f32)| -> f32 {
            (0..1600)
                .map(|i| {
                    let (d, along) = ((i % 40) as f32 * 8.0, (i / 40) as f32 * 23.0);
                    edge_fire(f(d, along), SIZE, 1.5, 1.0, (0.0, 0.0), [0.0; 4])[3]
                })
                .sum()
        };
        let bottom = cover(&|d, along| (1000.0 + along, SIZE.1 - 1.0 - d));
        let side = cover(&|d, along| (d, 500.0 + along));
        let top = cover(&|d, along| (1000.0 + along, d));
        assert!(
            bottom > side && side > top && top > 0.0,
            "{bottom} {side} {top}"
        );
    }

    #[test]
    fn the_lyric_band_is_kept_mostly_clear() {
        let mut most = 0.0f32;
        let mut outside = 0.0f32;
        for x in (0..SIZE.0 as usize).step_by(7) {
            for y in (BAND.0 as usize..BAND.1 as usize).step_by(11) {
                let c = edge_fire((x as f32, y as f32), SIZE, 4.2, 1.0, BAND, [0.0; 4]);
                most = most.max(c[3]);
            }
            let below = edge_fire((x as f32, 1955.0), SIZE, 4.2, 1.0, BAND, [0.0; 4]);
            outside = outside.max(below[3]);
        }
        assert!(most <= IN_BAND + 1e-6, "{most}");
        assert!(outside > 0.8, "{outside}");
    }

    #[test]
    fn it_moves() {
        let changed = grid(29)
            .filter(|&px| {
                let a = edge_fire(px, SIZE, 1.0, 1.0, BAND, [0.0; 4]);
                let b = edge_fire(px, SIZE, 1.25, 1.0, BAND, [0.0; 4]);
                (a[3] - b[3]).abs() > 0.05
            })
            .count();
        assert!(changed > 200, "{changed}");
    }

    #[test]
    fn the_ghostly_look_is_green() {
        let px = (1500.0, 1960.0);
        let classic = edge_fire(px, SIZE, 1.0, 1.0, (0.0, 0.0), [0.0, 1.0, 0.0, 0.0]);
        let ghost = edge_fire(px, SIZE, 1.0, 1.0, (0.0, 0.0), [1.0, 1.0, 0.0, 0.0]);
        assert!(
            classic[0] > classic[1] && ghost[1] > ghost[0],
            "{classic:?} {ghost:?}"
        );
        assert!((classic[3] - ghost[3]).abs() < 1e-6);
    }

    /// The strips the fire is drawn in hold every pixel it lights, at any
    /// strength, and never lie over each other.
    #[test]
    fn the_strips_cover_the_fire_and_never_each_other() {
        for s in [0.1, 0.5, 1.0, 1.6] {
            let strips = strips((0.0, 0.0, SIZE.0, SIZE.1), s);
            let inside = |px: (f32, f32)| {
                (strips.iter())
                    .filter(|q| px.0 >= q[0] && px.0 < q[2] && px.1 >= q[1] && px.1 < q[3])
                    .count()
            };
            for px in grid(17) {
                assert!(inside(px) <= 1, "{s} {px:?}");
                if edge_fire(px, SIZE, 2.7, s, BAND, [0.0; 4])[3] > 0.0 {
                    assert_eq!(inside(px), 1, "{s} {px:?}");
                }
            }
        }
        // A narrow window: the sides meet, never cross.
        let narrow = strips((0.0, 0.0, 300.0, 1900.0), 1.0);
        assert!(narrow.windows(2).all(|w| w[0] != w[1]));
        assert!(narrow[2][2] <= narrow[3][0]);
    }
}
