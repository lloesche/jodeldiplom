//! Fog creeping in at the screen's edges, a screen pass over the finished
//! frame: a cold haze drifting sideways and rising slowly, deepest at the
//! bottom, shallower up the sides and along the top, its inner edge
//! ragged in wisps, over a cold darkening towards the edges (the grade's
//! vignette, dusk purple). Now and then a pair of eyes opens in it along
//! the bottom, glowing, their pupils peering in, blinking, and closes
//! again. It breathes with the bass and
//! swells on a bar's first beat (as sure as the beat grid is).
//!
//! It keeps what the singer reads as the fire does
//! ([`super::edge_fire`]): only a little of it inside the lyric band
//! ([`IN_BAND`]), nothing in the score's corner, the badge or the meter
//! ([`Clear`]), each edge feathered.
//!
//! A theme holds it throughout ([`crate::theme::Theme::always`]); no row
//! lists it. [`fog`] is the shader's reference ([`wgsl`]: `fx_fog`), held
//! to it on the GPU ([`REFERENCE`]). It is drawn only in strips along the
//! edges, as far in as it reaches ([`strips`]), and everything past it in
//! them returns at once.
//!
//! Its look ([`Spec::params`], `p`): `x` how much it breathes and swells
//! with the music (1 as made, 0 not at all), `y` how fast it drifts (1 as
//! made), `z` how much its eyes show (1 as made, 0 none).

use jodel_rhythm::stems::Stem;

use super::helpers::{lattice, sat, smoothstep, speed, vnoise};
use super::{At, Clear, Fx, Layer, Live, Music, Out, Reference, Slot, Spec};
use crate::gui::stage::gpu::Instance;

/// The fog: a strip along an edge (`quad`), the lyric band's top and
/// bottom in `shape[0..2]`, what it keeps clear as [`Clear::on`] puts it,
/// `params[1]` its strength (its look the shaders' `looks`).
pub const FOG: f32 = Fx::Fog.kind(0);

/// Fog at the screen's edges.
pub const SPEC: Spec = Spec {
    name: "fog",
    ident: "fog",
    at: &[Layer::Screen],
    draw,
    kinds: &["fog"],
    wgsl: Some(wgsl),
    params: [1.0, 1.0, 1.0, 0.0],
    reference: Some(REFERENCE),
    // The screen's edges: they keep still as it shakes.
    still: true,
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let &At::Screen {
        area, band, clear, ..
    } = at
    else {
        return;
    };
    if live.strength <= 0.0 {
        return;
    }
    let grow = swell(&live.music, live.look);
    out.shapes
        .extend(strips(area, grow).into_iter().map(|quad| {
            clear.on(Instance {
                quad,
                shape: [band.0, band.1, 0.0, 0.0],
                params: [FOG, live.strength, 0.0, 0.0],
                ..Instance::default()
            })
        }));
}

/// How deep the fog is at the bottom, a share of the frame's height (with
/// no music).
const HIGH: f32 = 0.18;
/// How deep up the sides and along the top, a share of the bottom's.
const SIDE: f32 = 0.65;
const TOP: f32 = 0.35;
/// How much deeper it is with the bass at its loudest, and on a beat and a
/// bar's first beat (their pulses at their height, the grid sure).
const BREATH: f32 = 0.3;
const PULSE: f32 = 0.05;
const SURGE: f32 = 0.15;
/// How much of it covers the frame at most, and how much the cold
/// darkening under it does, at the edge.
const THICK: f32 = 0.5;
const DARK: f32 = 0.62;
/// How much of it shows inside the lyric band at most.
pub const IN_BAND: f32 = 0.3;
/// How far round the lyric band and what is kept clear it thins, a share
/// of the frame's height.
const FEATHER: f32 = 0.06;
/// The haze's colour, and the darkening's (sRGB).
const HAZE: [f32; 3] = [0.7, 0.85, 0.8];
const COLD: [f32; 3] = [0.05, 0.0, 0.09];
/// The eyes: their cells along the bottom (a share of the height), how
/// long a pair stays (seconds), how large an eye is (a share of the
/// height), their glow and their pupils (sRGB); how often a pair blinks
/// (a second).
const EYE_CELL: f32 = 0.11;
const EYE_STAY: f32 = 3.5;
const EYE: f32 = 0.011;
const EYE_GLOW: [f32; 3] = [0.85, 1.0, 0.4];
const PUPIL: [f32; 3] = [0.06, 0.1, 0.0];
const BLINK: f32 = 0.35;
/// How far apart an eye is from the pair's middle, how soft the glow's
/// ridge between them is, and how far an eye's glow reaches (each in eye
/// sizes): it fades smoothly to nothing there, inside the pair's cell and
/// below the fog's inner edge, so nothing cuts it off.
const APART: f32 = 1.4;
const RIDGE: f32 = 0.7;
const HALO: f32 = 3.4;

/// How many times as deep the music `m` makes the fog (1 with no music),
/// `p` its look (`x` how much).
pub fn swell(m: &Music, p: [f32; 4]) -> f32 {
    let bass = m.stems[Stem::Bass as usize];
    1.0 + p[0].max(0.0) * (BREATH * bass + m.kick(PULSE, SURGE))
}

/// The strips of `area` (left, top, width, height) the fog reaches,
/// `grow` times as deep as with no music ([`swell`]), none over another:
/// along the bottom and the top the whole width, up the sides between
/// them.
pub fn strips((x, y, w, h): (f32, f32, f32, f32), grow: f32) -> Vec<[f32; 4]> {
    let reach = h * HIGH * grow + 2.0;
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

/// The fog at pixel `px` of a frame `size` (pixels) at `time` (seconds),
/// `strength` 0 (none) to 1 (as thick as it gets), `band` the lyric band's
/// top and bottom (none where the bottom is not below the top), what it
/// keeps `clear` (each box none where it is not wider than nothing), `p`
/// its look, with the music `m`: premultiplied colour and coverage, in the
/// stage's colours (an sRGB frame takes them through its `out_rgb`), the
/// eyes' glow added.
pub fn fog(
    px: (f32, f32),
    size: (f32, f32),
    time: f32,
    (strength, band): (f32, (f32, f32)),
    clear: &Clear,
    p: [f32; 4],
    m: &Music,
) -> [f32; 4] {
    if strength <= 0.0 {
        return [0.0; 4];
    }
    let s = strength.min(1.0);
    let h = size.1;
    let f = h * FEATHER;
    let across = |a: f32, b: f32, v: f32| smoothstep(a - f, a, v) * (1.0 - smoothstep(b, b + f, v));
    let in_box = |r: [f32; 4]| {
        across(r[0], r[2], px.0) * across(r[1], r[3], px.1) * f32::from(u8::from(r[2] > r[0]))
    };
    let in_band = across(band.0, band.1, px.1) * f32::from(u8::from(band.1 > band.0));
    let kept = clear.boxes().into_iter().map(in_box).fold(0.0, f32::max);
    let deep = h * HIGH * swell(m, p);
    let bottom = size.1 - px.1;
    // How far in, a share of its depth at the nearest edge.
    let u = (bottom / deep)
        .min(px.0 / (deep * SIDE))
        .min((size.0 - px.0) / (deep * SIDE))
        .min(px.1 / (deep * TOP));
    if u >= 1.0 {
        return [0.0; 4];
    }
    let t = time * speed(p);
    let n = 0.6 * vnoise(px.0 / (h * 0.2) + t * 0.12, px.1 / (h * 0.14) - t * 0.04)
        + 0.4
            * vnoise(
                px.0 / (h * 0.08) - t * 0.2 + 17.0,
                px.1 / (h * 0.06) + t * 0.05 + 5.0,
            );
    // Its inner edge ragged in wisps; thicker where the noise is.
    let density = (1.0 - smoothstep(0.1, 0.95, u + (n - 0.5) * 0.7)) * (0.5 + 0.5 * n);
    let haze = density * THICK * s;
    let vignette = 1.0 - smoothstep(0.0, 1.0, u);
    let dark = vignette * vignette * DARK * s;
    let mut rgb = [0, 1, 2].map(|i| HAZE[i] * haze + COLD[i] * dark * (1.0 - haze));
    let mut a = haze + dark * (1.0 - haze);
    // The eyes: a pair now and then in a cell along the bottom, staying a
    // while (fading in and out), blinking.
    let c = h * EYE_CELL;
    let column = (px.0 / c).floor();
    let stay = t / EYE_STAY + lattice(column, 3.0);
    let epoch = stay.floor();
    let r = lattice(column, epoch);
    if r > 0.82 {
        let phase = stay - epoch;
        let er = h * EYE;
        let ex = (column + 0.5) * c;
        let ey = size.1 - deep * (0.3 + 0.35 * lattice(column + 11.0, epoch));
        let blink = ((t * BLINK + r * 13.0).fract() - 0.5).abs();
        let open = smoothstep(0.0, 0.04, blink).max(0.05);
        let fade = smoothstep(0.0, 0.15, phase) * (1.0 - smoothstep(0.8, 1.0, phase));
        let shows = fade * sat(p[2]) * s;
        let dx = (px.0 - ex).abs() - er * APART;
        let dy = px.1 - ey;
        let d = (dx / er).hypot(dy / (er * 1.3 * open)) - 1.0;
        let lit = sat(0.5 - d * er) * shows;
        // The glow: a bell round each eye, nothing from `HALO` out, the
        // two joined smoothly between them.
        let hx = (px.0 - ex).hypot(er * RIDGE) - er * APART.hypot(RIDGE);
        let bell = (1.0 - (hx * hx + dy * dy) / (er * er * HALO * HALO)).max(0.0);
        let halo = bell * bell * 0.35 * shows;
        // A slit of a pupil each, peering in towards the other eye.
        let dp = ((dx + er * 0.3) / (er * 0.3)).hypot(dy / (er * 0.85 * open)) - 1.0;
        let pupil = sat(0.5 - dp * er * 0.3) * lit;
        rgb = [0, 1, 2].map(|i| {
            let eye = EYE_GLOW[i] + (PUPIL[i] - EYE_GLOW[i]) * pupil;
            eye * lit + EYE_GLOW[i] * halo + rgb[i] * (1.0 - lit)
        });
        a = lit + a * (1.0 - lit);
    }
    let keep = (1.0 - (1.0 - IN_BAND) * in_band.max(kept)) * (1.0 - kept);
    [
        rgb[0].min(1.0) * keep,
        rgb[1].min(1.0) * keep,
        rgb[2].min(1.0) * keep,
        a * keep,
    ]
}

/// [`fog`] in WGSL: `fx_fog` and its helpers (each named `fog_...`), and
/// its strips drawn with it (`fog_shape`: its colours the stage's, as the
/// frame takes them through `out_rgb`).
pub fn wgsl() -> String {
    let v = |c: [f32; 3]| format!("vec3<f32>({:?}, {:?}, {:?})", c[0], c[1], c[2]);
    format!(
        "const FOG_HIGH: f32 = {HIGH:?};\nconst FOG_SIDE: f32 = {SIDE:?};\nconst FOG_TOP: f32 = {TOP:?};\nconst FOG_BREATH: f32 = {BREATH:?};\nconst FOG_PULSE: f32 = {PULSE:?};\nconst FOG_SURGE: f32 = {SURGE:?};\nconst FOG_THICK: f32 = {THICK:?};\nconst FOG_DARK: f32 = {DARK:?};\nconst FOG_IN_BAND: f32 = {IN_BAND:?};\nconst FOG_FEATHER: f32 = {FEATHER:?};\nconst FOG_HAZE: vec3<f32> = {};\nconst FOG_COLD: vec3<f32> = {};\nconst FOG_EYE_CELL: f32 = {EYE_CELL:?};\nconst FOG_EYE_STAY: f32 = {EYE_STAY:?};\nconst FOG_EYE: f32 = {EYE:?};\nconst FOG_EYE_GLOW: vec3<f32> = {};\nconst FOG_PUPIL: vec3<f32> = {};\nconst FOG_BLINK: f32 = {BLINK:?};\nconst FOG_APART: f32 = {APART:?};\nconst FOG_RIDGE: f32 = {RIDGE:?};\nconst FOG_HALO: f32 = {HALO:?};\n{SHADER}",
        v(HAZE),
        v(COLD),
        v(EYE_GLOW),
        v(PUPIL)
    )
}

const SHADER: &str = r"
fn fog_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let c = fx_fog(p, screen.size, screen.time, vec3<f32>(o.params.y, o.shape.xy), mat3x4<f32>(o.color2, o.color, o.uv), looks[FX_FOG], music);
    return vec4<f32>(out_rgb(c.rgb), c.a);
}

fn fog_swell(m: Music, p: vec4<f32>) -> f32 {
    let kick = m.tempo.y * (FOG_PULSE * m.beat.z + FOG_SURGE * m.beat.w);
    return 1.0 + max(p.x, 0.0) * (FOG_BREATH * m.stems[STEM_BASS] + kick);
}

fn fog_across(a: f32, b: f32, v: f32, f: f32) -> f32 {
    return smoothstep(a - f, a, v) * (1.0 - smoothstep(b, b + f, v));
}

fn fog_in_box(r: vec4<f32>, px: vec2<f32>, f: f32) -> f32 {
    return fog_across(r.x, r.z, px.x, f) * fog_across(r.y, r.w, px.y, f) * select(0.0, 1.0, r.z > r.x);
}

fn fx_fog(px: vec2<f32>, size: vec2<f32>, time: f32, sb: vec3<f32>, clear: mat3x4<f32>, p: vec4<f32>, m: Music) -> vec4<f32> {
    let strength = sb.x;
    let band = sb.yz;
    if strength <= 0.0 {
        return vec4<f32>(0.0);
    }
    let s = min(strength, 1.0);
    let h = size.y;
    let f = h * FOG_FEATHER;
    let in_band = fog_across(band.x, band.y, px.y, f) * select(0.0, 1.0, band.y > band.x);
    let kept = max(max(fog_in_box(clear[0], px, f), fog_in_box(clear[1], px, f)), fog_in_box(clear[2], px, f));
    let deep = h * FOG_HIGH * fog_swell(m, p);
    let bottom = size.y - px.y;
    let u = min(
        min(bottom / deep, px.x / (deep * FOG_SIDE)),
        min((size.x - px.x) / (deep * FOG_SIDE), px.y / (deep * FOG_TOP)),
    );
    if u >= 1.0 {
        return vec4<f32>(0.0);
    }
    let t = time * fx_speed(p);
    let n = 0.6 * vnoise(px.x / (h * 0.2) + t * 0.12, px.y / (h * 0.14) - t * 0.04)
        + 0.4 * vnoise(px.x / (h * 0.08) - t * 0.2 + 17.0, px.y / (h * 0.06) + t * 0.05 + 5.0);
    let density = (1.0 - smoothstep(0.1, 0.95, u + (n - 0.5) * 0.7)) * (0.5 + 0.5 * n);
    let haze = density * FOG_THICK * s;
    let vignette = 1.0 - smoothstep(0.0, 1.0, u);
    let dark = vignette * vignette * FOG_DARK * s;
    var rgb = FOG_HAZE * haze + FOG_COLD * dark * (1.0 - haze);
    var a = haze + dark * (1.0 - haze);
    let c = h * FOG_EYE_CELL;
    let column = floor(px.x / c);
    let stay = t / FOG_EYE_STAY + lattice(column, 3.0);
    let epoch = floor(stay);
    let r = lattice(column, epoch);
    if r > 0.82 {
        let phase = stay - epoch;
        let er = h * FOG_EYE;
        let ex = (column + 0.5) * c;
        let ey = size.y - deep * (0.3 + 0.35 * lattice(column + 11.0, epoch));
        let blink = abs(fract(t * FOG_BLINK + r * 13.0) - 0.5);
        let open = max(smoothstep(0.0, 0.04, blink), 0.05);
        let fade = smoothstep(0.0, 0.15, phase) * (1.0 - smoothstep(0.8, 1.0, phase));
        let shows = fade * clamp(p.z, 0.0, 1.0) * s;
        let dx = abs(px.x - ex) - er * FOG_APART;
        let dy = px.y - ey;
        let d = length(vec2<f32>(dx / er, dy / (er * 1.3 * open))) - 1.0;
        let lit = clamp(0.5 - d * er, 0.0, 1.0) * shows;
        let hx = length(vec2<f32>(px.x - ex, er * FOG_RIDGE)) - er * length(vec2<f32>(FOG_APART, FOG_RIDGE));
        let bell = max(1.0 - (hx * hx + dy * dy) / (er * er * FOG_HALO * FOG_HALO), 0.0);
        let halo = bell * bell * 0.35 * shows;
        let dp = length(vec2<f32>((dx + er * 0.3) / (er * 0.3), dy / (er * 0.85 * open))) - 1.0;
        let pupil = clamp(0.5 - dp * er * 0.3, 0.0, 1.0) * lit;
        let eye = mix(FOG_EYE_GLOW, FOG_PUPIL, pupil);
        rgb = eye * lit + FOG_EYE_GLOW * halo + rgb * (1.0 - lit);
        a = lit + a * (1.0 - lit);
    }
    let keep = (1.0 - (1.0 - FOG_IN_BAND) * max(in_band, kept)) * (1.0 - kept);
    return vec4<f32>(min(rgb, vec3<f32>(1.0)) * keep, a * keep);
}
";

/// The fog at pixels near every edge, in and round the lyric band and
/// what is kept clear (the score's corner, the badge and the meter: each
/// none in a few), at strengths, times, looks and music (still, swelling
/// on a bar's first beat, the bass loud): the shader's `fx_fog` as
/// [`fog`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        let size = (3024.0f32, 1964.0f32);
        let band = (1330.0, 1650.0);
        let score = [2500.0, 0.0, 3024.0, 160.0];
        let badge = [300.0, 40.0, 1000.0, 150.0];
        let meter = [1100.0, 30.0, 1900.0, 160.0];
        (0..3000)
            .flat_map(|k| {
                let f = k as f32;
                let fr = |m: f32| jodel_looks::math::fract(f * m);
                // Most within a fifth of an edge, many near the bottom
                // (where the eyes are).
                let (x, y) = match k % 5 {
                    0 => (fr(0.613) * size.0, size.1 - fr(0.377) * 450.0),
                    1 => (fr(0.613) * 450.0, fr(0.377) * size.1),
                    2 => (size.0 - fr(0.613) * 450.0, fr(0.377) * size.1),
                    3 => (fr(0.613) * size.0, size.1 - fr(0.377) * 250.0),
                    _ => (fr(0.613) * size.0, fr(0.377) * size.1),
                };
                let strength = [0.0, 0.3, 0.6, 1.0, 1.4][(k / 5) % 5];
                let music = match k % 3 {
                    0 => Music::NEUTRAL,
                    1 => Music::steady(f64::from(fr(0.519)) * 8.0, 120.0, 4, 1.0, 0.0),
                    _ => Music {
                        stems: [fr(0.347), fr(0.123), fr(0.456), fr(0.789)],
                        ..Music::steady(f64::from(fr(0.519)) * 8.0, 96.0, 4, f64::from(fr(0.831)), 0.0)
                    },
                };
                let ins = [
                    [x, y, fr(0.291) * 60.0, strength],
                    [size.0, size.1, band.0, band.1],
                    [fr(0.733) * 1.5, 0.5 + fr(0.417), fr(0.157) * 1.5, 0.0],
                    if k % 7 == 0 { [0.0; 4] } else { score },
                ];
                let badge = if k % 11 == 0 { [0.0; 4] } else { badge };
                let meter = if k % 13 == 0 { [0.0; 4] } else { meter };
                ins.into_iter().chain(music.uniform()).chain([badge, meter])
            })
            .collect()
    },
    ins: 11,
    outs: 1,
    body: "
    let a = ins[i * 11u];
    let b = ins[i * 11u + 1u];
    let m = Music(ins[i * 11u + 4u], ins[i * 11u + 5u], ins[i * 11u + 6u], ins[i * 11u + 7u], ins[i * 11u + 8u]);
    let clear = mat3x4<f32>(ins[i * 11u + 3u], ins[i * 11u + 9u], ins[i * 11u + 10u]);
    outs[i] = fx_fog(a.xy, b.xy, a.z, vec3<f32>(a.w, b.zw), clear, ins[i * 11u + 2u], m);",
    cpu: |s| {
        let px = (s[0][0], s[0][1]);
        let size = (s[1][0], s[1][1]);
        let band = (s[1][2], s[1][3]);
        let m = Music::from_uniform(&[s[4], s[5], s[6], s[7], s[8]]);
        let clear = Clear {
            score: s[3],
            badge: s[9],
            meter: s[10],
        };
        vec![fog(px, size, s[0][2], (s[0][3], band), &clear, s[2], &m)]
    },
    near: (4e-3, 0.0),
    shows: |o| o[0][3] > 0.05,
    least: 500,
};

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (f32, f32) = (3024.0, 1964.0);
    const BAND: (f32, f32) = (1330.0, 1650.0);
    const SCORE: [f32; 4] = [2500.0, 0.0, 3024.0, 160.0];
    const CLEAR: Clear = Clear {
        score: SCORE,
        badge: [300.0, 40.0, 1000.0, 150.0],
        meter: [1100.0, 30.0, 1900.0, 160.0],
    };
    const LOOK: [f32; 4] = SPEC.params;

    fn at(px: (f32, f32), t: f32) -> [f32; 4] {
        fog(px, SIZE, t, (1.0, BAND), &CLEAR, LOOK, &Music::NEUTRAL)
    }

    /// Strength none: nothing anywhere, exactly.
    #[test]
    fn at_no_strength_there_is_none() {
        for k in 0..2000 {
            let f = k as f32;
            let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
            let none = fog(px, SIZE, f, (0.0, BAND), &CLEAR, LOOK, &Music::NEUTRAL);
            assert_eq!(none, [0.0; 4]);
        }
    }

    /// It lies along the edges, thickest at the bottom, and leaves the
    /// middle alone; every value within bounds, premultiplied.
    #[test]
    fn it_lies_at_the_edges_and_leaves_the_middle_alone() {
        let mean = |pts: &mut dyn Iterator<Item = (f32, f32)>| {
            let v: Vec<f32> = pts.map(|p| at(p, 3.0)[3]).collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        let bottom = mean(&mut (0..200).map(|k| (400.0 + k as f32 * 4.0, SIZE.1 - 20.0)));
        let side = mean(&mut (0..200).map(|k| (15.0, 300.0 + k as f32 * 4.0)));
        let middle = mean(&mut (0..200).map(|k| (900.0 + k as f32 * 4.0, 900.0)));
        assert!(bottom > 0.3 && side > 0.2, "{bottom} {side}");
        assert_eq!(middle, 0.0);
        for k in 0..20_000 {
            let f = k as f32;
            let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
            let c = at(px, f * 0.01);
            assert!((0.0..=1.0).contains(&c[3]), "{c:?}");
            assert!(c[..3].iter().all(|&v| (0.0..=1.0).contains(&v)), "{c:?}");
        }
    }

    /// What the singer reads is kept: the score's corner wholly clear,
    /// the lyric band at most a little.
    #[test]
    fn what_the_singer_reads_is_kept_clear() {
        for k in 0..400 {
            let f = k as f32;
            let corner = (
                2600.0 + (f * 0.37).fract() * 400.0,
                (f * 0.61).fract() * 150.0,
            );
            assert_eq!(at(corner, f * 0.1)[3], 0.0, "{corner:?}");
            let band = (
                (f * 0.37).fract() * 300.0,
                1400.0 + (f * 0.61).fract() * 200.0,
            );
            assert!(at(band, f * 0.1)[3] <= IN_BAND + 1e-4, "{band:?}");
        }
    }

    /// The strips hold all of it at its deepest, none over another.
    #[test]
    fn the_strips_hold_it_all() {
        let loud = Music {
            stems: [1.0; 4],
            ..Music::steady(8.0, 120.0, 4, 1.0, 0.0)
        };
        let grow = swell(&loud, LOOK);
        assert!(grow > 1.3, "{grow}");
        let strips = strips((0.0, 0.0, SIZE.0, SIZE.1), grow);
        for k in 0..20_000 {
            let f = k as f32;
            let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
            let c = fog(px, SIZE, f * 0.01, (1.0, BAND), &Clear::NONE, LOOK, &loud);
            let inside = |s: &[f32; 4]| px.0 >= s[0] && px.0 < s[2] && px.1 >= s[1] && px.1 < s[3];
            if c[3] > 0.0 {
                assert!(strips.iter().any(inside), "{px:?}");
            }
        }
    }

    /// Eyes open in it along the bottom now and then: somewhere along it,
    /// over some seconds, a pixel glows brighter than any haze; none with
    /// the eyes turned off.
    #[test]
    fn eyes_open_in_it_now_and_then() {
        let bright = |p: [f32; 4]| {
            let (mut most, mut sum) = (0.0f32, 0.0);
            for t in 0..40 {
                for x in 0..600 {
                    for y in 0..26 {
                        let px = (x as f32 * 5.0 + 2.5, SIZE.1 - 70.0 - y as f32 * 10.0);
                        let c = fog(
                            px,
                            SIZE,
                            t as f32 * 0.5,
                            (1.0, (0.0, 0.0)),
                            &Clear::NONE,
                            p,
                            &Music::NEUTRAL,
                        );
                        most = most.max(c[1]);
                        sum += c[1];
                    }
                }
            }
            (most, sum)
        };
        let (eyes, _) = bright(LOOK);
        let (none, _) = bright([LOOK[0], LOOK[1], 0.0, 0.0]);
        assert!(eyes > 0.8, "{eyes}");
        assert!(none < 0.6, "{none}");
    }

    /// The eyes' glow fades out before anything can cut it off: at their
    /// cell's sides and at the fog's inner edge above them it is nothing
    /// (the fog there as it is with the eyes turned off), in windows of a
    /// few sizes, while in the cells' middles the eyes do show.
    #[test]
    fn the_eyes_glow_fades_out_inside_their_cell() {
        let off = [LOOK[0], LOOK[1], 0.0, 0.0];
        let mut shown = 0;
        for size in [SIZE, (1600.0, 1000.0), (1280.0, 720.0)] {
            let fog_at = |px: (f32, f32), t: f32, p: [f32; 4]| {
                fog(
                    px,
                    size,
                    t,
                    (1.0, (0.0, 0.0)),
                    &Clear::NONE,
                    p,
                    &Music::NEUTRAL,
                )
            };
            let cell = size.1 * EYE_CELL;
            let deep = size.1 * HIGH;
            let columns = (size.0 / cell).ceil() as usize;
            for t in 0..60 {
                let t = t as f32;
                for k in 0..columns {
                    let x0 = k as f32 * cell;
                    let middle = (x0 + cell / 2.0, size.1 - deep / 2.0);
                    shown += usize::from(fog_at(middle, t, LOOK) != fog_at(middle, t, off));
                    for y in 0..60 {
                        let y = size.1 - deep * (y as f32 + 0.5) / 60.0;
                        for x in [x0 + 0.25, x0 + cell - 0.25] {
                            assert_eq!(
                                fog_at((x, y), t, LOOK),
                                fog_at((x, y), t, off),
                                "{x} {y} {t}"
                            );
                        }
                    }
                    for x in 0..40 {
                        let px = (x0 + cell * (x as f32 + 0.5) / 40.0, size.1 - deep + 0.25);
                        assert_eq!(fog_at(px, t, LOOK), fog_at(px, t, off), "{px:?} {t}");
                    }
                }
            }
        }
        assert!(shown > 50, "{shown}");
    }
}
