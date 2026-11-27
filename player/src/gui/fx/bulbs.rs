//! Marquee bulbs round the lane, a screen pass: a row of round light
//! bulbs round the lane ([`super::marquee`]: where they sit), as round a
//! casino's sign, a lit wave chasing round them (a bulb lit, its tail
//! dimming behind it, every few bulbs), with a soft glow, each bulb in a
//! dark socket with a rim so the row reads on any cover. At full strength
//! (the top rung) every other bulb also flashes, the two halves in turn.
//! Their colours are the theme's ([`crate::theme::Theme::bulbs`]: warm
//! white and gold as made), and so is what a bulb is
//! ([`crate::theme::BulbShape`]: a bulb, or a jack-o'-lantern, [`jack`],
//! its stalk on top).
//!
//! Cheap per pixel: a pixel far from the row is done at once, and one
//! near it works out its nearest bulb from where it is (the side nearest
//! it and how far along), and draws that bulb and its two neighbours (no
//! walk over the row). The lane, and the lyric line under it, stay clear:
//! a bulb's glow is gone a little over three radii from its middle.
//!
//! [`bulbs`] is the shader's maths on the processor, held to it on the
//! GPU ([`REFERENCE`]). The colours are the stage's (sRGB, as the frame
//! takes them through `out_rgb`), premultiplied, for drawing over. Its
//! look ([`Spec::params`], `p`): `y` the chase's speed (1 as
//! made), `z` how much it keeps to the beat (1 as made, 0 not at all). It
//! is drawn only in strips along the row ([`strips`]).
//!
//! On the beat ([`lit_at`], the shader reading the music's uniform): the
//! chase steps a bulb a beat, every bulb blinks up hard on each beat and
//! the chase dims between them, the flashing halves turn a beat each, and
//! every bulb flashes on a bar's first beat, all together and brighter
//! than lit (their glow overdriven). As sure as the beat grid is: where it
//! is really unsure, the chase runs by the clock as it does with no music.
//! Its look's `w` is how hard it blinks (1 as made, 0 not at all).
//!
//! Where the lane's flames burn too, they set the bulbs on fire
//! ([`super::flames`], reading the row): the flames burn behind the row,
//! round each bulb and up between them, and the bulbs, drawn over the
//! light, stay crisp on top.

use super::helpers::{mix3, sat, smoothstep};
use super::marquee::{self, bulb_at, Row, DIAMETER, MARGIN, OUT, REACH, SHOWN_FROM, SPACING};
use super::{At, Fx, Layer, Live, Music, Out, Reference, Slot, Spec};
use crate::gui::stage::gpu::Instance;
use crate::theme::{BulbShape, Palette, THEMES};

/// The bulbs: a strip along the row (`quad`), round the lane `shape`
/// (left, top, right, bottom), `params[1]` their strength (their look the
/// shaders' `looks`, their colours and shape the theme's).
pub const BULBS: f32 = Fx::Bulbs.kind(0);

/// Marquee bulbs round the lane.
pub const SPEC: Spec = Spec {
    name: "marquee bulbs",
    ident: "bulbs",
    at: &[Layer::Screen],
    draw,
    kinds: &["bulbs"],
    wgsl: Some(wgsl),
    params: [0.0, 1.0, 1.0, 1.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let &At::Screen { area, lane, .. } = at else {
        return;
    };
    if live.strength <= 0.0 {
        return;
    }
    let size = super::far_corner(area);
    out.shapes
        .extend(strips(lane.rect(), size).into_iter().map(|quad| Instance {
            quad,
            shape: lane.rect(),
            params: [BULBS, live.strength, 0.0, 0.0],
            ..Instance::default()
        }));
}

/// The strips round `lane` in a screen of `size` the bulbs draw in (as far
/// either side of their row as a bulb's glow reaches), none over another:
/// along the top and the bottom, down the sides between them; one box
/// where the row is too low for that.
pub fn strips(lane: [f32; 4], size: (f32, f32)) -> Vec<[f32; 4]> {
    let Some(Row { ring, r, .. }) = marquee::row(lane, size) else {
        return Vec::new();
    };
    let [x0, y0, x1, y1] = ring;
    let reach = r * REACH + 2.0;
    let (top, bottom) = (y0 + reach, y1 - reach);
    if bottom <= top {
        return vec![[x0 - reach, y0 - reach, x1 + reach, y1 + reach]];
    }
    vec![
        [x0 - reach, y0 - reach, x1 + reach, top],
        [x0 - reach, bottom, x1 + reach, y1 + reach],
        [x0 - reach, top, x0 + reach, bottom],
        [x1 - reach, top, x1 + reach, bottom],
    ]
}

/// The chase's steps a second.
const RATE: f32 = 12.0;
/// The chase: a bulb lit every this many.
const PERIOD: i32 = 6;
/// Its tail, in bulbs.
const TAIL: f32 = 1.5;
/// How much a bulb the chase has left is lit.
const BASE: f32 = 0.2;
/// The strength the flashing starts from (all of it at 1).
const FLASH_FROM: f32 = 0.8;
/// The flashing's turns a second.
const FLASH_RATE: f32 = 5.0;
/// How strong the glow is.
const GLOW: f32 = 1.3;
/// How much every bulb blinks up on a beat (its pulse at its height), a
/// share of the way to fully lit, its look's `w` at 1.
const BLINK: f32 = 0.85;
/// How much the chase dims between the beats (the beat's pulse gone).
const DIP: f32 = 0.55;
/// How much brighter than lit every bulb's glow flashes on a bar's first
/// beat.
const OVERDRIVE: f32 = 0.7;

/// A jack-o'-lantern's skin, its candle lit and out, unlit and lit, and
/// its stalk.
const SKIN_DARK: [f32; 3] = [0.36, 0.13, 0.02];
const SKIN: [f32; 3] = [1.0, 0.5, 0.06];
const EMBERS: [f32; 3] = [0.12, 0.04, 0.0];
const CANDLE: [f32; 3] = [1.0, 0.92, 0.5];
const STALK: [f32; 3] = [0.28, 0.42, 0.12];

/// How lit bulb `k` is (0 to 1) at chase step `step`, the chase `dim`
/// as lit as it might be, its flashing `flash` strong, on flashing beat
/// `beat`.
fn lit(k: i32, step: i32, dim: f32, flash: f32, beat: i32) -> f32 {
    let u = ((step - k) % PERIOD + PERIOD) % PERIOD;
    let chase = (BASE + (1.0 - BASE) * (-(u as f32) / TAIL).exp()) * dim;
    let on = if ((k + beat) % 2 + 2) % 2 == 0 {
        1.0
    } else {
        0.0
    };
    chase.max(flash * on)
}

/// How lit bulb `k` is at `time` (its glow overdriven past 1), its
/// flashing `flash` strong, with the music `m` (the beats counted, the
/// bar's and the beat's pulses, how sure the grid is) and its look `p`
/// (`y` the chase's speed, `z` how much it keeps to the beat, `w` how hard
/// it blinks): by the clock (the chase [`RATE`] steps a second, the halves
/// flashing [`FLASH_RATE`] times a second), and on the beat (a step a
/// beat, the chase dimmed between the beats, every bulb blinking up on
/// each beat, the halves turning a beat each, every bulb lit and
/// overdriven on a bar's first), the one going to the other as sure as
/// the grid is.
pub fn lit_at(k: i32, time: f32, flash: f32, p: [f32; 4], m: &Music) -> f32 {
    let free = lit(
        k,
        (time * RATE * p[1]).floor() as i32,
        1.0,
        flash,
        (time * FLASH_RATE).floor() as i32,
    );
    let lock = sat(m.sure * p[2]);
    if lock <= 0.0 {
        return free;
    }
    let beat = m.beats.floor() as i32;
    let hard = p[3].max(0.0);
    let dim = 1.0 - sat(DIP * hard * (1.0 - m.beat));
    let on = lit(k, beat, dim, flash, beat).max(sat(m.downbeat * hard));
    let on = on + (1.0 - on) * sat(BLINK * hard * m.beat);
    let on = on + OVERDRIVE * hard * smoothstep(0.5, 1.0, m.downbeat);
    free + (on - free) * lock
}

/// A jack-o'-lantern of radius `r` at `q` from its middle, its candle
/// `lit` (0 to 1): its skin ribbed and darker to its rim, its face carved
/// (two eyes and a nose, triangles pointing up, and a grin) glowing as its
/// candle burns.
pub fn jack(q: (f32, f32), r: f32, lit: f32) -> [f32; 3] {
    let u = (q.0 / r, q.1 / r);
    let rib = 0.5 + 0.5 * (u.0 * 5.5).cos();
    let shade = (0.72 + 0.28 * rib) * (1.0 - 0.35 * smoothstep(0.3, 1.0, u.0.hypot(u.1)));
    let skin = mix3(SKIN_DARK, SKIN, 0.35 + 0.65 * lit).map(|c| c * shade);
    let tri = |cx: f32, cy: f32, k: f32| {
        let (x, y) = (u.0 - cx, u.1 - cy);
        y.max(0.866 * x.abs() - 0.5 * y) - k
    };
    let grin = ((u.0.hypot(u.1 + 0.55) - 0.85).abs() - 0.12).max(u.0.abs() - 0.6);
    let face = (tri(-0.36, -0.18, 0.1).min(tri(0.36, -0.18, 0.1)))
        .min(tri(0.0, 0.08, 0.055))
        .min(grin);
    let carved = (0.5 - face * r).clamp(0.0, 1.0);
    mix3(skin, mix3(EMBERS, CANDLE, lit), carved)
}

/// How much of a jack-o'-lantern's stalk (radius `r`) covers `q` from its
/// middle: a short stub leaning out of its top.
fn stalk(q: (f32, f32), r: f32) -> f32 {
    use jodel_looks::math::v2;
    let (at, half) = (v2(q.0 / r - 0.06, q.1 / r + 1.18), v2(0.1, 0.17));
    let d = jodel_looks::math::round_box(at, half, 0.05);
    (0.5 - d * r).clamp(0.0, 1.0)
}

/// A bulb of radius `r`, `glowing` (lit, its glow overdriven past 1), at
/// `q` from its middle, in `pal`'s colours, `pumpkin` of the way to a
/// jack-o'-lantern ([`jack`], its stalk on top): what it puts over the
/// frame (premultiplied) and its glow added.
fn bulb(
    q: (f32, f32),
    r: f32,
    (glowing, pumpkin): (f32, f32),
    pal: &Palette,
) -> ([f32; 4], [f32; 3]) {
    let lit = glowing.min(1.0);
    let dd = q.0.hypot(q.1);
    let sock = (r * 1.22 - dd + 0.5).clamp(0.0, 1.0);
    let rim = (1.0 - (dd - r * 1.14).abs() / (r * 0.08)).clamp(0.0, 1.0) * sock;
    let glass = (r - dd + 0.5).clamp(0.0, 1.0);
    let core = (-dd * dd / (r * r * 0.3)).exp();
    let deep = mix3(pal.on, pal.glow, smoothstep(0.5, 1.0, dd / r) * 0.7);
    let on = mix3(deep, [1.0; 3], core * 0.9 * lit);
    let (hx, hy) = (q.0 + 0.35 * r, q.1 + 0.4 * r);
    let spec = (-(hx * hx + hy * hy) / (r * r * 0.03)).exp() * 0.55;
    let col = mix3(pal.off, on, lit).map(|c| c + spec);
    let col = mix3(col, jack(q, r, lit).map(|c| c + spec * 0.6), pumpkin);
    let under = mix3(pal.socket, pal.rim, rim);
    let over = [
        col[0] * glass + under[0] * sock * (1.0 - glass),
        col[1] * glass + under[1] * sock * (1.0 - glass),
        col[2] * glass + under[2] * sock * (1.0 - glass),
        glass + sock * (1.0 - glass),
    ];
    let st = stalk(q, r) * pumpkin;
    let over = [
        STALK[0] * st + over[0] * (1.0 - st),
        STALK[1] * st + over[1] * (1.0 - st),
        STALK[2] * st + over[2] * (1.0 - st),
        st + over[3] * (1.0 - st),
    ];
    let x = (dd - r * 0.5).max(0.0) / (r * 1.3);
    let fade = (1.0 - dd / (r * REACH)).clamp(0.0, 1.0);
    let g = glowing * GLOW * (-x * x).exp() * fade * fade;
    (over, pal.glow.map(|c| c * g))
}

/// The bulbs at pixel `px` of a screen of `size` pixels, `time` seconds
/// on, `strength` strong (0: nothing at all; to 1 fading in; at 1
/// flashing too), round `lane` (left, top, right, bottom, pixels), with
/// `p` (the chase's speed, 1 as made; how much it keeps to the beat; how
/// hard it blinks) and the music `m` ([`lit_at`]), in `pal`'s colours,
/// each a `shape` (a theme's dress): premultiplied, as the frame takes
/// it.
pub fn bulbs(
    px: (f32, f32),
    size: (f32, f32),
    time: f32,
    strength: f32,
    lane: [f32; 4],
    (p, pal, shape): ([f32; 4], &Palette, BulbShape),
    m: &Music,
) -> [f32; 4] {
    if strength <= 0.0 {
        return [0.0; 4];
    }
    let Some(Row {
        ring,
        r,
        gaps: (nw, nh),
    }) = marquee::row(lane, size)
    else {
        return [0.0; 4];
    };
    let (w, h) = (ring[2] - ring[0], ring[3] - ring[1]);
    let c = (px.0.clamp(ring[0], ring[2]), px.1.clamp(ring[1], ring[3]));
    let outside = (px.0 - c.0).hypot(px.1 - c.1);
    let inside = (px.0 - ring[0])
        .min(ring[2] - px.0)
        .min(px.1 - ring[1])
        .min(ring[3] - px.1);
    if outside.max(inside) > r * REACH {
        return [0.0; 4];
    }
    let n = 2 * (nw + nh);
    // The side nearest, and how far along the row that is, in bulbs.
    let sides = [
        (
            (px.0 - c.0).hypot(px.1 - ring[1]),
            (c.0 - ring[0]) / w * nw as f32,
        ),
        (
            (px.0 - ring[2]).hypot(px.1 - c.1),
            nw as f32 + (c.1 - ring[1]) / h * nh as f32,
        ),
        (
            (px.0 - c.0).hypot(px.1 - ring[3]),
            (nw + nh) as f32 + (ring[2] - c.0) / w * nw as f32,
        ),
        (
            (px.0 - ring[0]).hypot(px.1 - c.1),
            (2 * nw + nh) as f32 + (ring[3] - c.1) / h * nh as f32,
        ),
    ];
    let mut best = sides[0];
    for side in &sides[1..] {
        if side.0 < best.0 {
            best = *side;
        }
    }
    let k0 = (best.1 + 0.5).floor() as i32;
    let pumpkin = shape.pumpkin();
    let flash = ((strength - FLASH_FROM) / (1.0 - FLASH_FROM)).clamp(0.0, 1.0);
    let (mut over, mut glow) = ([0.0f32; 4], [0.0f32; 3]);
    for j in -1..=1 {
        let k = ((k0 + j) % n + n) % n;
        let at = bulb_at(k, (nw, nh), ring);
        let (o, g) = bulb(
            (px.0 - at.0, px.1 - at.1),
            r,
            (lit_at(k, time, flash, p, m), pumpkin),
            pal,
        );
        for i in 0..4 {
            over[i] += o[i];
        }
        for i in 0..3 {
            glow[i] += g[i];
        }
    }
    let shown = marquee::shown(strength);
    let bright = 0.55 + 0.45 * strength.min(1.0);
    [
        (over[0] + glow[0] * bright) * shown,
        (over[1] + glow[1] * bright) * shown,
        (over[2] + glow[2] * bright) * shown,
        over[3] * shown,
    ]
}

/// [`bulbs`] in WGSL: `fx_bulbs` and what it needs (its names begun
/// `bulbs_`), and its strips drawn with it (`bulbs_shape`, in the theme's
/// colours and shape: `theme.bulbs`, `theme.bulb`).
pub fn wgsl() -> String {
    format!(
        "
const BULBS_DIAMETER: f32 = {DIAMETER:?};
const BULBS_SPACING: f32 = {SPACING:?};
const BULBS_OUT: f32 = {OUT:?};
const BULBS_MARGIN: f32 = {MARGIN:?};
const BULBS_REACH: f32 = {REACH:?};
const BULBS_SHOWN_FROM: f32 = {SHOWN_FROM:?};
const BULBS_RATE: f32 = {RATE:?};
const BULBS_PERIOD: i32 = {PERIOD};
const BULBS_TAIL: f32 = {TAIL:?};
const BULBS_BASE: f32 = {BASE:?};
const BULBS_FLASH_FROM: f32 = {FLASH_FROM:?};
const BULBS_FLASH_RATE: f32 = {FLASH_RATE:?};
const BULBS_GLOW: f32 = {GLOW:?};
const BULBS_BLINK: f32 = {BLINK:?};
const BULBS_DIP: f32 = {DIP:?};
const BULBS_OVERDRIVE: f32 = {OVERDRIVE:?};
const BULBS_JACK = array<vec3<f32>, 5>({});
{SHADER}",
        [SKIN_DARK, SKIN, EMBERS, CANDLE, STALK]
            .map(|c| format!("vec3<f32>({:?}, {:?}, {:?})", c[0], c[1], c[2]))
            .join(", ")
    )
}

const SHADER: &str = r"
fn bulbs_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let c = fx_bulbs(p, screen.size, screen.time, o.params.y, o.shape, looks[FX_BULBS], theme.bulbs, theme.bulb, music);
    return vec4<f32>(out_rgb(c.rgb), c.a);
}

fn bulbs_at(k: i32, nw: i32, nh: i32, ring: vec4<f32>) -> vec2<f32> {
    let w = ring.z - ring.x;
    let h = ring.w - ring.y;
    let fw = f32(nw);
    let fh = f32(nh);
    if k < nw {
        return vec2<f32>(ring.x + f32(k) * w / fw, ring.y);
    }
    if k < nw + nh {
        return vec2<f32>(ring.z, ring.y + f32(k - nw) * h / fh);
    }
    if k < 2 * nw + nh {
        return vec2<f32>(ring.z - f32(k - nw - nh) * w / fw, ring.w);
    }
    return vec2<f32>(ring.x, ring.w - f32(k - 2 * nw - nh) * h / fh);
}

fn bulbs_lit(k: i32, step: i32, dim: f32, flash: f32, beat: i32) -> f32 {
    let u = ((step - k) % BULBS_PERIOD + BULBS_PERIOD) % BULBS_PERIOD;
    let chase = (BULBS_BASE + (1.0 - BULBS_BASE) * exp(-f32(u) / BULBS_TAIL)) * dim;
    var on = 0.0;
    if ((k + beat) % 2 + 2) % 2 == 0 {
        on = 1.0;
    }
    return max(chase, flash * on);
}

fn bulbs_lit_at(k: i32, time: f32, flash: f32, p: vec4<f32>, m: Music) -> f32 {
    let free = bulbs_lit(k, i32(floor(time * BULBS_RATE * p.y)), 1.0, flash, i32(floor(time * BULBS_FLASH_RATE)));
    let lock = clamp(m.tempo.y * p.z, 0.0, 1.0);
    if lock <= 0.0 {
        return free;
    }
    let beat = i32(floor(m.count.x));
    let hard = max(p.w, 0.0);
    let dim = 1.0 - clamp(BULBS_DIP * hard * (1.0 - m.beat.z), 0.0, 1.0);
    var on = max(bulbs_lit(k, beat, dim, flash, beat), clamp(m.beat.w * hard, 0.0, 1.0));
    on = on + (1.0 - on) * clamp(BULBS_BLINK * hard * m.beat.z, 0.0, 1.0);
    on = on + BULBS_OVERDRIVE * hard * smoothstep(0.5, 1.0, m.beat.w);
    return free + (on - free) * lock;
}

// A bulb: over the frame (premultiplied) in `over`, its glow in `glow`.
struct BulbsLight {
    over: vec4<f32>,
    glow: vec3<f32>,
};

fn bulbs_tri(u: vec2<f32>, c: vec2<f32>, k: f32) -> f32 {
    let v = u - c;
    return max(v.y, 0.866 * abs(v.x) - 0.5 * v.y) - k;
}

fn bulbs_jack(q: vec2<f32>, r: f32, lit: f32) -> vec3<f32> {
    let u = q / r;
    let rib = 0.5 + 0.5 * cos(u.x * 5.5);
    let shade = (0.72 + 0.28 * rib) * (1.0 - 0.35 * smoothstep(0.3, 1.0, length(u)));
    let skin = mix(BULBS_JACK[0], BULBS_JACK[1], 0.35 + 0.65 * lit) * shade;
    let grin = max(abs(length(vec2<f32>(u.x, u.y + 0.55)) - 0.85) - 0.12, abs(u.x) - 0.6);
    let face = min(
        min(min(bulbs_tri(u, vec2<f32>(-0.36, -0.18), 0.1), bulbs_tri(u, vec2<f32>(0.36, -0.18), 0.1)), bulbs_tri(u, vec2<f32>(0.0, 0.08), 0.055)),
        grin,
    );
    let carved = clamp(0.5 - face * r, 0.0, 1.0);
    return mix(skin, mix(BULBS_JACK[2], BULBS_JACK[3], lit), carved);
}

fn bulbs_stalk(q: vec2<f32>, r: f32) -> f32 {
    let d = round_box(q / r - vec2<f32>(0.06, -1.18), vec2<f32>(0.1, 0.17), 0.05);
    return clamp(0.5 - d * r, 0.0, 1.0);
}

fn bulbs_one(q: vec2<f32>, r: f32, glowing: f32, pumpkin: f32, pal: array<vec4<f32>, BULB_COLORS>) -> BulbsLight {
    var o: BulbsLight;
    let lit = min(glowing, 1.0);
    let dd = length(q);
    let sock = clamp(r * 1.22 - dd + 0.5, 0.0, 1.0);
    let rim = clamp(1.0 - abs(dd - r * 1.14) / (r * 0.08), 0.0, 1.0) * sock;
    let glass = clamp(r - dd + 0.5, 0.0, 1.0);
    let core = exp(-dd * dd / (r * r * 0.3));
    let deep = mix(pal[0].rgb, pal[1].rgb, smoothstep(0.5, 1.0, dd / r) * 0.7);
    let on = mix(deep, vec3<f32>(1.0), core * 0.9 * lit);
    let hl = q + vec2<f32>(0.35 * r, 0.4 * r);
    let spec = exp(-dot(hl, hl) / (r * r * 0.03)) * 0.55;
    let plain = mix(pal[2].rgb, on, lit) + vec3<f32>(spec);
    let col = mix(plain, bulbs_jack(q, r, lit) + vec3<f32>(spec * 0.6), pumpkin);
    let under = mix(pal[3].rgb, pal[4].rgb, rim);
    let over = vec4<f32>(col * glass + under * sock * (1.0 - glass), glass + sock * (1.0 - glass));
    let st = bulbs_stalk(q, r) * pumpkin;
    o.over = vec4<f32>(BULBS_JACK[4] * st + over.rgb * (1.0 - st), st + over.a * (1.0 - st));
    let x = max(dd - r * 0.5, 0.0) / (r * 1.3);
    let fade = clamp(1.0 - dd / (r * BULBS_REACH), 0.0, 1.0);
    o.glow = pal[1].rgb * (glowing * BULBS_GLOW * exp(-x * x) * fade * fade);
    return o;
}

fn fx_bulbs(px: vec2<f32>, size: vec2<f32>, time: f32, strength: f32, lane: vec4<f32>, p: vec4<f32>, pal: array<vec4<f32>, BULB_COLORS>, bulb: vec4<f32>, mu: Music) -> vec4<f32> {
    if strength <= 0.0 {
        return vec4<f32>(0.0);
    }
    let r = (lane.w - lane.y) * BULBS_DIAMETER * 0.5;
    let off = r * BULBS_OUT;
    let m = r * BULBS_MARGIN;
    let ring = vec4<f32>(
        max(lane.x - off, m),
        max(lane.y - off, m),
        min(lane.z + off, size.x - m),
        min(lane.w + off, size.y - m),
    );
    let w = ring.z - ring.x;
    let h = ring.w - ring.y;
    if r <= 0.0 || w <= 0.0 || h <= 0.0 {
        return vec4<f32>(0.0);
    }
    let c = clamp(px, ring.xy, ring.zw);
    let outside = length(px - c);
    let inside = min(min(px.x - ring.x, ring.z - px.x), min(px.y - ring.y, ring.w - px.y));
    if max(outside, inside) > r * BULBS_REACH {
        return vec4<f32>(0.0);
    }
    let gap = r * 2.0 * BULBS_SPACING;
    let nw = max(i32(floor(w / gap + 0.5)), 1);
    let nh = max(i32(floor(h / gap + 0.5)), 1);
    let n = 2 * (nw + nh);
    var best = length(vec2<f32>(px.x - c.x, px.y - ring.y));
    var along = (c.x - ring.x) / w * f32(nw);
    let right = length(vec2<f32>(px.x - ring.z, px.y - c.y));
    if right < best {
        best = right;
        along = f32(nw) + (c.y - ring.y) / h * f32(nh);
    }
    let bottom = length(vec2<f32>(px.x - c.x, px.y - ring.w));
    if bottom < best {
        best = bottom;
        along = f32(nw + nh) + (ring.z - c.x) / w * f32(nw);
    }
    let left = length(vec2<f32>(px.x - ring.x, px.y - c.y));
    if left < best {
        along = f32(2 * nw + nh) + (ring.w - c.y) / h * f32(nh);
    }
    let k0 = i32(floor(along + 0.5));
    let flash = clamp((strength - BULBS_FLASH_FROM) / (1.0 - BULBS_FLASH_FROM), 0.0, 1.0);
    var over = vec4<f32>(0.0);
    var glow = vec3<f32>(0.0);
    for (var j = -1; j <= 1; j++) {
        let k = ((k0 + j) % n + n) % n;
        let b = bulbs_one(px - bulbs_at(k, nw, nh, ring), r, bulbs_lit_at(k, time, flash, p, mu), bulb.x, pal);
        over += b.over;
        glow += b.glow;
    }
    let shown = min(strength / BULBS_SHOWN_FROM, 1.0);
    let bright = 0.55 + 0.45 * min(strength, 1.0);
    return vec4<f32>(over.rgb + glow * bright, over.a) * shown;
}
";

/// The bulbs at pixels round and off the row, at strengths and times, in
/// every theme's colours and shape, round lanes of three screens: the
/// shader's `fx_bulbs` as [`bulbs`].
pub const REFERENCE: Reference = Reference {
    samples,
    ins: 15,
    outs: 1,
    body: "
    let j = i * 15u;
    let a = ins[j];
    let b = ins[j + 1u];
    let p = vec4<f32>(0.0, b.w, ins[j + 3u].x, 0.0);
    let m = Music(ins[j + 4u], ins[j + 5u], ins[j + 6u], ins[j + 7u], ins[j + 8u]);
    let pal = array<vec4<f32>, BULB_COLORS>(ins[j + 9u], ins[j + 10u], ins[j + 11u], ins[j + 12u], ins[j + 13u]);
    outs[i] = fx_bulbs(a.xy, a.zw, b.x, b.y, ins[j + 2u], p, pal, ins[j + 14u], m);",
    cpu: |s| {
        let (a, b) = (s[0], s[1]);
        let p = [0.0, b[3], s[3][0], 0.0];
        let m = Music::from_uniform(&[s[4], s[5], s[6], s[7], s[8]]);
        let pal = Palette::of_uniform(&s[9..14]);
        let look = (p, &pal, BulbShape::of_uniform(s[14]));
        vec![bulbs((a[0], a[1]), (a[2], a[3]), b[0], b[1], s[2], look, &m)]
    },
    near: (3e-3, 3e-3),
    shows: |o| o[0][3] > 0.1,
    least: 300,
};

fn samples() -> Vec<[f32; 4]> {
    let lanes = [
        [80.0, 150.0, 1520.0, 600.0],
        [76.0, 284.0, 2948.0, 1130.0],
        [5.0, 5.0, 795.0, 300.0],
    ];
    let sizes = [(1600.0, 1000.0), (3024.0, 1964.0), (800.0, 600.0)];
    (0..3000)
        .flat_map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract();
            let (lane, size) = (lanes[k % 3], sizes[k % 3]);
            let r = (lane[3] - lane[1]) * DIAMETER * 0.5;
            // Most near the row.
            let (x, y) = if k % 4 == 0 {
                (fr(0.613) * size.0, fr(0.377) * size.1)
            } else {
                let along = fr(0.613);
                let across = (fr(0.377) - 0.5) * r * 6.0;
                match k % 4 {
                    1 => (
                        lane[0] + along * (lane[2] - lane[0]),
                        lane[1] - r * OUT + across,
                    ),
                    2 => (
                        lane[0] + along * (lane[2] - lane[0]),
                        lane[3] + r * OUT + across,
                    ),
                    _ => (
                        lane[2] + r * OUT + across,
                        lane[1] + along * (lane[3] - lane[1]),
                    ),
                }
            };
            let strength = [0.0, 0.3, 0.7, 0.9, 1.0][k % 5];
            // The music: none, or a grid somewhere in its beats.
            let music = match k % 3 {
                0 => Music::NEUTRAL,
                _ => Music::steady(
                    f64::from(fr(0.193)) * 300.0 + 0.0123,
                    120.0,
                    4,
                    f64::from(fr(0.811)),
                    0.0,
                ),
            };
            let ins = [
                [x + 0.013, y + 0.029, size.0, size.1],
                [fr(0.291) * 30.0 + 0.0123, strength, 0.0, 0.5 + fr(0.71)],
                lane,
                [[0.0, 1.0, fr(0.643) * 1.4][k % 3], 0.0, 0.0, 0.0],
            ];
            let dress = THEMES[(k / 7) % THEMES.len()].uniform();
            (ins.into_iter().chain(music.uniform()))
                .chain(dress.bulbs)
                .chain([dress.bulb])
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (f32, f32) = (1600.0, 1000.0);
    const LANE: [f32; 4] = [80.0, 150.0, 1520.0, 600.0];
    /// Their look as made (the chase as fast, on the beat, blinking with
    /// it), classic.
    const CLASSIC_P: ([f32; 4], &Palette, BulbShape) = (
        SPEC.params,
        &crate::theme::CLASSIC.bulbs,
        crate::theme::CLASSIC.bulb,
    );
    /// No music.
    const STILL: &Music = &Music::NEUTRAL;

    /// Strength none: nothing anywhere, exactly.
    #[test]
    fn no_strength_draws_nothing() {
        for k in 0..2000 {
            let f = k as f32;
            let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
            assert_eq!(
                bulbs(px, SIZE, f * 0.01, 0.0, LANE, CLASSIC_P, STILL),
                [0.0; 4]
            );
            assert_eq!(
                bulbs(px, SIZE, f * 0.01, -0.5, LANE, CLASSIC_P, STILL),
                [0.0; 4]
            );
        }
    }

    /// Bulbs only round the lane's border: none deep in the lane (its
    /// bubbles and words) nor far off it (the lyrics under it), some just
    /// outside it; every value within bounds.
    #[test]
    fn the_bulbs_ring_the_lane_and_nothing_else() {
        let r = (LANE[3] - LANE[1]) * DIAMETER * 0.5;
        let (mut near, mut lit) = (0, 0);
        for k in 0..20_000 {
            let f = k as f32;
            let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
            let c = bulbs(px, SIZE, 3.3, 1.0, LANE, CLASSIC_P, STILL);
            for v in c {
                assert!((0.0..=2.5).contains(&v) && v.is_finite(), "{px:?}: {c:?}");
            }
            assert!(c[3] <= 1.0 + 1e-5);
            let dx = (LANE[0] - px.0).max(px.0 - LANE[2]);
            let dy = (LANE[1] - px.1).max(px.1 - LANE[3]);
            // How far outside the lane (negative inside).
            let out = if dx > 0.0 && dy > 0.0 {
                dx.hypot(dy)
            } else {
                dx.max(dy)
            };
            if out < -r * (REACH - OUT) || out > r * (OUT * std::f32::consts::SQRT_2 + REACH) {
                assert_eq!(c, [0.0; 4], "{px:?} {out}");
            } else if c.iter().any(|&v| v > 0.0) {
                near += 1;
                lit += usize::from(c[3] > 0.5);
            }
        }
        assert!(near > 500 && lit > 100, "{near} {lit}");
    }

    /// The chase runs round: a bulb's light changes as time goes on, and
    /// the lit bulbs move along the row; the strength fades them in.
    #[test]
    fn the_chase_moves_and_the_strength_fades_in() {
        let Row { ring, r, gaps } = marquee::row(LANE, SIZE).expect("a row");
        let y = LANE[1] - r * OUT;
        let gap = (ring[2] - ring[0]) / gaps.0 as f32;
        let light = |k: f32, t: f32, s: f32| {
            bulbs((ring[0] + k * gap, y), SIZE, t, s, LANE, CLASSIC_P, STILL)[1]
        };
        let row = |t: f32| (0..12).map(|k| light(k as f32, t, 0.6)).collect::<Vec<_>>();
        let (a, b) = (row(1.03), row(1.03 + 1.0 / RATE));
        assert_ne!(a, b);
        // A step on, each bulb as lit as the one before it was.
        for k in 1..12 {
            assert!((b[k] - a[k - 1]).abs() < 1e-4, "{k}: {a:?} {b:?}");
        }
        assert!(
            a.iter().cloned().fold(0.0, f32::max) > 2.0 * a.iter().cloned().fold(9.0, f32::min)
        );
        assert!(light(0.0, 1.03, 0.1) < light(0.0, 1.03, 0.6));
        // At full strength, half the bulbs flash on, the others the next beat.
        let on = |t: f32| (0..12).filter(|&k| light(k as f32, t, 1.0) > 0.99).count();
        assert!(on(1.01) >= 6 && on(1.01 + 1.0 / FLASH_RATE) >= 6);
    }

    /// Nothing of the bulbs is cut off at their strips: flashing on a
    /// bar's first beat (their glow overdriven), their halos are gone (none
    /// at all) at every outer edge of the strips (an edge not shared with
    /// the next strip), so no straight edge of them ever shows.
    #[test]
    fn the_halos_are_gone_at_the_strips_edges() {
        let one = on_beat(8.0, 1.0);
        for lane in [LANE, [5.0, 5.0, 795.0, 300.0], [100.0, 300.0, 900.0, 303.0]] {
            let strips = strips(lane, SIZE);
            let on_another = |p: (f32, f32), n: usize| {
                (strips.iter().enumerate()).any(|(m, q)| {
                    m != n && p.0 >= q[0] && p.0 <= q[2] && p.1 >= q[1] && p.1 <= q[3]
                })
            };
            for (n, &[l, t, r, b]) in strips.iter().enumerate() {
                for k in 0..=400 {
                    let (x, y) = (
                        l + (r - l) * k as f32 / 400.0,
                        t + (b - t) * k as f32 / 400.0,
                    );
                    for p in [(x, t), (x, b), (l, y), (r, y)] {
                        if on_another(p, n) {
                            continue;
                        }
                        for time in [0.0, 1.3] {
                            let c = bulbs(p, SIZE, time, 1.0, lane, CLASSIC_P, &one);
                            assert_eq!(c, [0.0; 4], "{lane:?} {p:?}");
                        }
                    }
                }
            }
        }
    }

    /// The strips the bulbs are drawn in hold every pixel they light, and
    /// never lie over each other; a lane too low for four, one box.
    #[test]
    fn the_strips_cover_the_bulbs_and_never_each_other() {
        for lane in [LANE, [5.0, 5.0, 795.0, 300.0], [100.0, 300.0, 900.0, 303.0]] {
            let strips = strips(lane, SIZE);
            let inside = |px: (f32, f32)| {
                (strips.iter())
                    .filter(|q| px.0 >= q[0] && px.0 < q[2] && px.1 >= q[1] && px.1 < q[3])
                    .count()
            };
            for k in 0..40_000 {
                let f = k as f32;
                let px = ((f * 0.613).fract() * SIZE.0, (f * 0.377).fract() * SIZE.1);
                assert!(inside(px) <= 1, "{lane:?} {px:?}");
                if bulbs(px, SIZE, 2.2, 1.0, lane, CLASSIC_P, STILL) != [0.0; 4] {
                    assert_eq!(inside(px), 1, "{lane:?} {px:?}");
                }
            }
        }
        assert_eq!(
            strips([0.0, 10.0, 100.0, 10.0], SIZE),
            Vec::<[f32; 4]>::new()
        );
    }

    /// The music as the bulbs take it, a steady grid's.
    fn on_beat(beats: f64, sure: f32) -> Music {
        Music::steady(beats, 120.0, 4, f64::from(sure), 0.0)
    }

    /// The row's light, bulb by bulb (a dozen), at `time` with `m`,
    /// flashing `flash`.
    fn row(time: f32, flash: f32, m: &Music) -> Vec<f32> {
        (0..12)
            .map(|k| lit_at(k, time, flash, CLASSIC_P.0, m))
            .collect()
    }

    /// The grid sure: the chase steps a bulb a beat (whatever the clock
    /// says within it), each bulb then as lit as the one before it was the
    /// beat before.
    #[test]
    fn on_the_beat_the_chase_steps_a_bulb_a_beat() {
        let (a, b) = (
            row(3.0, 0.0, &on_beat(9.6, 1.0)),
            row(3.0, 0.0, &on_beat(10.6, 1.0)),
        );
        for k in 1..12 {
            assert!((b[k] - a[k - 1]).abs() < 1e-6, "{k}: {a:?} {b:?}");
        }
        // The clock within the beat moves nothing; another beat does.
        assert_eq!(
            row(3.0, 0.0, &on_beat(9.6, 1.0)),
            row(3.04, 0.0, &on_beat(9.6, 1.0))
        );
        assert_ne!(a, b);
    }

    /// On a bar's first beat every bulb is lit, the flash falling away
    /// after it; each beat lights them all up a little.
    #[test]
    fn every_bulb_flashes_on_the_one_and_glows_up_on_each_beat() {
        let one = row(0.0, 0.0, &on_beat(8.0, 1.0));
        assert!(one.iter().all(|&v| v > 0.99), "{one:?}");
        let after = row(0.0, 0.0, &on_beat(8.5, 1.0));
        assert!(after.iter().any(|&v| v < 0.6), "{after:?}");
        let beat = row(0.0, 0.0, &on_beat(9.0, 1.0));
        let between = row(0.0, 0.0, &on_beat(9.6, 1.0));
        let dimmest = |r: &[f32]| r.iter().copied().fold(9.0, f32::min);
        assert!(
            dimmest(&beat) > dimmest(&between) + 0.1,
            "{beat:?} {between:?}"
        );
        // At full strength, the halves flash in turn a beat each.
        let on = |beats: f64| -> Vec<bool> {
            row(0.0, 1.0, &on_beat(beats, 1.0))
                .iter()
                .map(|&v| v > 0.99)
                .collect()
        };
        let (x, y) = (on(9.6), on(10.6));
        for k in 0..12 {
            assert!(x[k] || y[k], "{k}: {x:?} {y:?}");
        }
        assert_ne!(x, y);
    }

    /// Where the grid is really unsure the bulbs run by the clock as they
    /// do with no music; half sure (the grid's regions agreeing a quarter),
    /// half way between; moderately sure (agreeing half), wholly on the
    /// beat; the look's lock off, by the clock however sure.
    #[test]
    fn unsure_the_bulbs_run_freely() {
        let free = row(1.03, 0.0, STILL);
        assert_eq!(row(1.03, 0.0, &on_beat(9.3, 0.0)), free);
        assert_eq!(row(1.03, 0.0, &on_beat(9.3, 0.1)), free);
        let sure = row(1.03, 0.0, &on_beat(9.3, 1.0));
        assert_eq!(row(1.03, 0.0, &on_beat(9.3, 0.5)), sure);
        let half = row(1.03, 0.0, &on_beat(9.3, 0.25));
        assert_ne!(sure, free);
        for k in 0..12 {
            assert!((half[k] - (free[k] + sure[k]) / 2.0).abs() < 1e-5, "{k}");
        }
        let off = [0.0, 1.0, 0.0, 0.0];
        let m = on_beat(9.3, 1.0);
        assert_eq!(
            lit_at(3, 1.03, 0.0, off, &m),
            lit_at(3, 1.03, 0.0, off, STILL)
        );
        // As the shader has it: drawn round the lane, the music moves it.
        let px = (
            LANE[0] + 300.0,
            LANE[1] - (LANE[3] - LANE[1]) * DIAMETER * 0.5 * OUT,
        );
        let a = bulbs(px, SIZE, 1.03, 0.6, LANE, CLASSIC_P, STILL);
        let b = bulbs(px, SIZE, 1.03, 0.6, LANE, CLASSIC_P, &on_beat(8.0, 1.0));
        assert_ne!(a, b);
    }

    /// Dressed as the theme has them: Spooky's jack-o'-lanterns other than
    /// Classic's bulbs on the same bulb, as much of it covered.
    #[test]
    fn they_are_dressed_as_the_theme_has_them() {
        let Row { ring, .. } = marquee::row(LANE, SIZE).expect("a row");
        let px = (ring[0], ring[1]);
        let s = &crate::theme::SPOOKY;
        let spooky = (SPEC.params, &s.bulbs, s.bulb);
        let a = bulbs(px, SIZE, 1.03, 1.0, LANE, CLASSIC_P, STILL);
        let b = bulbs(px, SIZE, 1.03, 1.0, LANE, spooky, STILL);
        assert_ne!(a, b);
        assert!(a[3] > 0.9 && b[3] > 0.9, "{a:?} {b:?}");
    }
}
