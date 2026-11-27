//! The screen's edges on fire: a pass over the whole finished frame.
//! Tongues of flame lick in from the edges, tallest at the bottom, shorter
//! up the sides (drifting up as they lick in), short along the top: the
//! one fire the lane's flames burn too ([`super::helpers::fire`]), its
//! light over the frame in the theme's fire's colours
//! ([`crate::theme::Theme::fire`]), white-hot at their roots and orange to
//! red at their tips as made, the tips letting the frame show through,
//! rising and flickering, with a warm glow behind them and embers drifting
//! up out of them. The lyric band is kept clear: the flames shrink back as
//! they come to it, and anything of the fire inside it shows at
//! [`KEEP`]'s `in_band` at most. What the singer reads ([`Clear`]: the
//! score's corner, the multiplier's badge, the meter and its verdict) is
//! kept wholly clear: the flames shrink back and fade out round each, and
//! nothing of the fire is inside one.
//!
//! [`edge_fire`] is the shader's reference ([`wgsl`]: `fx_edge_fire`),
//! held to it on the GPU ([`REFERENCE`]). It is drawn only in strips along
//! the edges, as far in as the fire can reach at its strength
//! ([`deepest`]), and everything past the flames' reach in them returns at
//! once.
//!
//! It moves with the music ([`swell`], the shader reading the music's
//! uniform): it surges on each bar's first beat, reaching much further and
//! burning hotter, and pulses on each beat (as sure as the beat grid is:
//! where it is really unsure, not at all), and breathes with the bass,
//! reaching further as it is louder.
//!
//! Its look ([`Spec::params`], `p`): `y` how fast it flickers (1
//! as made), `z` how much it surges and pulses with the beat and `w` how
//! much it breathes with the bass (1 as made, 0 not at all).

use jodel_rhythm::stems::Stem;

use super::helpers::{fire, fire_light, heat, lattice, mix3, smoothstep, speed, TALLEST_FLAME};
use super::screen_pass::{draw_strips, Edges, KeepClear};
use super::{At, Clear, Fx, Layer, Live, Music, Out, Reference, Slot, Spec};
use crate::theme::{Ramp, THEMES};

/// The fire: a strip along an edge (`quad`), the lyric band's top and
/// bottom in `shape[0..2]`, what it keeps clear as [`Clear::on`] puts it,
/// `params[1]` its strength (its look the shaders' `looks`, its colours
/// the theme's `fire`).
pub const EDGE_FIRE: f32 = Fx::EdgeFire.kind(0);

/// The screen's edges on fire.
pub const SPEC: Spec = Spec {
    name: "edge fire",
    ident: "edge_fire",
    at: &[Layer::Screen],
    draw,
    kinds: &["edge fire"],
    wgsl: Some(wgsl),
    params: [0.0, 1.0, 1.0, 1.0],
    reference: Some(REFERENCE),
    // The screen's edges: they keep still as it shakes.
    still: true,
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let (grow, _) = swell(&live.music, live.look);
    draw_strips(EDGE_FIRE, EDGES, (at, live, out), |h| {
        deepest(h, live.strength, grow)
    });
}

/// How far in (pixels) anything of the fire reaches at the bottom of a
/// frame `h` high at `strength`, swollen by the music to `grow` times its
/// reach ([`swell`]): how deep its strips are ([`Edges::strips`]).
pub fn deepest(h: f32, strength: f32, grow: f32) -> f32 {
    h * HIGH * reach(strength.min(1.0)) * grow * OUT
}

/// How high the flames are at the bottom at full strength, a share of the
/// frame's height (the one fire's height, [`fire`]: their tongues' tips
/// about half as high on average, the tallest half as high again).
const HIGH: f32 = 0.2;
/// How high they are up the sides and along the top, a share of how high
/// at the bottom.
const EDGES: Edges = Edges {
    side: 0.6,
    top: 0.25,
};
/// How high they are at strength `s` (0 to 1), a share of how high at
/// full strength: [`REACH_LEAST`] of it at the least, all of it at 1.
fn reach(s: f32) -> f32 {
    REACH_LEAST + REACH_GROWN * s
}
const REACH_LEAST: f32 = 0.3;
const REACH_GROWN: f32 = 0.7;
/// How deep (in heights) anything of the fire goes: past the tallest
/// flames, its embers.
const OUT: f32 = 1.6;
/// How the fire keeps what the singer reads: shrinking back from the lyric
/// band and what is kept clear a share of the frame's height round them,
/// as much as this of it showing in the band at most.
const KEEP: KeepClear = KeepClear {
    feather: 0.06,
    in_band: 0.12,
};
/// How much the flames shrink back inside the band or a box kept clear.
const SHRINK: f32 = 0.6;
/// The embers' cells, a share of the frame's height: one in a few has an
/// ember.
const EMBER: f32 = 0.028;
/// How much higher the flames reach at a bar's first beat (its pulse at
/// its height, the grid sure), and on each beat, a share of their height.
const FLARE: f32 = 0.5;
const PULSE: f32 = 0.2;
/// How much hotter they burn then: their heat raised by as much.
const FLARE_HEAT: f32 = 0.25;
const PULSE_HEAT: f32 = 0.12;
/// How much higher they reach with the bass at its loudest.
const BREATH: f32 = 0.25;

/// How the music `m` swells the fire (its bar's pulse and the beat's, as
/// sure as the grid is; the bass's level), `p` its look (`z` how much it
/// surges with the beat, `w` how much it breathes): how many times as
/// high it reaches (1 with no music), and how much hotter it burns.
pub fn swell(m: &Music, p: [f32; 4]) -> (f32, f32) {
    let bass = m.stems[Stem::Bass as usize];
    (
        1.0 + p[2] * m.kick(PULSE, FLARE) + BREATH * bass * p[3],
        p[2] * m.kick(PULSE_HEAT, FLARE_HEAT),
    )
}

/// The fire at pixel `px` of a frame `size` (pixels) at `time` (seconds),
/// `strength` 0 (none) to 1 (the flames a fifth of the frame high at the
/// bottom), `band` the lyric band's top and bottom (pixels; none
/// where the bottom is not below the top), what it keeps `clear` (each
/// box none where it is not wider than nothing), `p` its look (`y` how
/// fast it flickers: 1 as made, 0 taken as 1; `z` and `w` how much it
/// surges and breathes with the music `m`, [`swell`]) and `fire` its
/// colours. Premultiplied colour and coverage, in the stage's colours (the
/// frame's bytes; an sRGB frame takes them through its `out_rgb`): the
/// flames' light added over the frame, which shows through their tips and
/// less through their roots.
pub fn edge_fire(
    px: (f32, f32),
    size: (f32, f32),
    time: f32,
    (strength, band): (f32, (f32, f32)),
    clear: &Clear,
    (p, fire): ([f32; 4], &Ramp),
    m: &Music,
) -> [f32; 4] {
    if strength <= 0.0 {
        return [0.0; 4];
    }
    let s = strength.min(1.0);
    let h = size.1;
    let (grow, hot) = swell(m, p);
    // The lyric band and what is kept clear (their edges feathered wide):
    // the flames shrink back as they come to them, and only a little of
    // anything shows in the band, nothing in the rest.
    let kept = KEEP.keep(px, h, band, clear);
    let high = h * HIGH * reach(s) * grow * (1.0 - SHRINK * kept.inside);
    let (bottom, top) = (size.1 - px.1, px.1);
    let (left, right) = (px.0, size.0 - px.0);
    // How deep in, in heights, from the nearest edge.
    let u = EDGES.depth(px, size, high);
    if u > OUT {
        return [0.0; 4];
    }
    let t = time * speed(p);
    // The flames from each edge near (the one fire, its tongues as wide as
    // the flames at full strength are high, those up the sides drifting up
    // as they lick in): the hottest of them.
    let (l, Edges { side, top: over }) = (h * HIGH, EDGES);
    let up = |from: f32, d: f32| burn(px.1 / (l * side) + t * 0.7 + from, d, high * side, t);
    let k = burn(px.0 / l, bottom, high, t)
        .max(burn(px.0 / (l * over) + 31.0, top, high * over, t))
        .max(up(57.0, left))
        .max(up(83.0, right))
        * (1.0 + hot);
    let light = fire_light(k, fire);
    let flame_a = smoothstep(0.35, 1.0, k) * 0.85;
    // A warm light behind them, close to the edge, gone by the fire's
    // depth (nothing of it cut off there).
    let glow_a = (-u * 3.0).exp() * 0.2 * (1.0 - smoothstep(0.8, OUT, u));
    let glow = heat(0.45, fire);
    let mut a = flame_a + glow_a * (1.0 - flame_a);
    let mut rgb = [0, 1, 2].map(|i| (light[i] + glow[i] * glow_a * (1.0 - flame_a)).min(1.0));
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
        let e = (-d2 / (big * big)).exp() * twinkle * (1.0 - smoothstep(0.6, OUT, u));
        let ember = mix3(heat(0.7, fire), [1.0; 3], (-d2 / (big * big * 0.15)).exp());
        rgb = [0, 1, 2].map(|i| ember[i] * e + rgb[i] * (1.0 - e));
        a = e + a * (1.0 - e);
    }
    // Coming in with the strength.
    let keep = kept.keep * smoothstep(0.0, 0.15, s);
    [rgb[0] * keep, rgb[1] * keep, rgb[2] * keep, a * keep]
}

/// The one fire's heat at `along` its edge (in heights at full strength)
/// `d` pixels in from it, its flames `high`, at `t`: none past their
/// tallest.
fn burn(along: f32, d: f32, high: f32, t: f32) -> f32 {
    if d >= TALLEST_FLAME * high {
        return 0.0;
    }
    fire(along, d / high, t)
}

/// [`edge_fire`] in WGSL: `fx_edge_fire` and its helpers (each named
/// `edge_fire_...`), and its strips drawn with it (`edge_fire_shape`: its
/// colours the stage's, as the frame takes them through `out_rgb`).
pub fn wgsl() -> String {
    let (side, top, feather, in_band) = (EDGES.side, EDGES.top, KEEP.feather, KEEP.in_band);
    format!(
        "const EDGE_FIRE_HIGH: f32 = {HIGH:?};\nconst EDGE_FIRE_SIDE: f32 = {side:?};\nconst EDGE_FIRE_TOP: f32 = {top:?};\nconst EDGE_FIRE_REACH_LEAST: f32 = {REACH_LEAST:?};\nconst EDGE_FIRE_REACH_GROWN: f32 = {REACH_GROWN:?};\nconst EDGE_FIRE_OUT: f32 = {OUT:?};\nconst EDGE_FIRE_IN_BAND: f32 = {in_band:?};\nconst EDGE_FIRE_FEATHER: f32 = {feather:?};\nconst EDGE_FIRE_SHRINK: f32 = {SHRINK:?};\nconst EDGE_FIRE_EMBER: f32 = {EMBER:?};\nconst EDGE_FIRE_FLARE: f32 = {FLARE:?};\nconst EDGE_FIRE_PULSE: f32 = {PULSE:?};\nconst EDGE_FIRE_FLARE_HEAT: f32 = {FLARE_HEAT:?};\nconst EDGE_FIRE_PULSE_HEAT: f32 = {PULSE_HEAT:?};\nconst EDGE_FIRE_BREATH: f32 = {BREATH:?};\n{SHADER}"
    )
}

const SHADER: &str = r"
fn edge_fire_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let c = fx_edge_fire(p, screen.size, screen.time, vec3<f32>(o.params.y, o.shape.xy), mat3x4<f32>(o.color2, o.color, o.uv), looks[FX_EDGE_FIRE], theme.fire, music);
    return vec4<f32>(out_rgb(c.rgb), c.a);
}

fn edge_fire_reach(s: f32) -> f32 {
    return EDGE_FIRE_REACH_LEAST + EDGE_FIRE_REACH_GROWN * s;
}

fn edge_fire_swell(m: Music, p: vec4<f32>) -> vec2<f32> {
    let reach = m.tempo.y * (EDGE_FIRE_PULSE * m.beat.z + EDGE_FIRE_FLARE * m.beat.w);
    let hot = m.tempo.y * (EDGE_FIRE_PULSE_HEAT * m.beat.z + EDGE_FIRE_FLARE_HEAT * m.beat.w);
    return vec2<f32>(1.0 + p.z * reach + EDGE_FIRE_BREATH * m.stems[STEM_BASS] * p.w, p.z * hot);
}

fn edge_fire_burn(along: f32, d: f32, high: f32, t: f32) -> f32 {
    if d >= FX_TALLEST_FLAME * high {
        return 0.0;
    }
    return fx_fire(along, d / high, t);
}

fn fx_edge_fire(px: vec2<f32>, size: vec2<f32>, time: f32, sb: vec3<f32>, clear: mat3x4<f32>, p: vec4<f32>, fire: array<vec4<f32>, FIRE_STOPS>, m: Music) -> vec4<f32> {
    let strength = sb.x;
    let band = sb.yz;
    if strength <= 0.0 {
        return vec4<f32>(0.0);
    }
    let s = min(strength, 1.0);
    let h = size.y;
    let swell = edge_fire_swell(m, p);
    let kept = fx_keep_clear(px, h, band, clear, EDGE_FIRE_FEATHER, EDGE_FIRE_IN_BAND);
    let high = h * EDGE_FIRE_HIGH * edge_fire_reach(s) * swell.x * (1.0 - EDGE_FIRE_SHRINK * kept.x);
    let bottom = size.y - px.y;
    let right = size.x - px.x;
    let u = fx_edge_depth(px, size, high, EDGE_FIRE_SIDE, EDGE_FIRE_TOP);
    if u > EDGE_FIRE_OUT {
        return vec4<f32>(0.0);
    }
    let t = time * fx_speed(p);
    let l = h * EDGE_FIRE_HIGH;
    let k = max(
        max(
            edge_fire_burn(px.x / l, bottom, high, t),
            edge_fire_burn(px.x / (l * EDGE_FIRE_TOP) + 31.0, px.y, high * EDGE_FIRE_TOP, t),
        ),
        max(
            edge_fire_burn(px.y / (l * EDGE_FIRE_SIDE) + t * 0.7 + 57.0, px.x, high * EDGE_FIRE_SIDE, t),
            edge_fire_burn(px.y / (l * EDGE_FIRE_SIDE) + t * 0.7 + 83.0, right, high * EDGE_FIRE_SIDE, t),
        ),
    ) * (1.0 + swell.y);
    let light = fx_fire_light(k, fire);
    let flame_a = smoothstep(0.35, 1.0, k) * 0.85;
    let glow_a = exp(-u * 3.0) * 0.2 * (1.0 - smoothstep(0.8, EDGE_FIRE_OUT, u));
    let glow = fx_heat(0.45, fire);
    var a = flame_a + glow_a * (1.0 - flame_a);
    var rgb = min(light + glow * glow_a * (1.0 - flame_a), vec3<f32>(1.0));
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
        let e = exp(-d2 / (big * big)) * twinkle * (1.0 - smoothstep(0.6, EDGE_FIRE_OUT, u));
        let ember = mix(fx_heat(0.7, fire), vec3<f32>(1.0), exp(-d2 / (big * big * 0.15)));
        rgb = ember * e + rgb * (1.0 - e);
        a = e + a * (1.0 - e);
    }
    let keep = kept.y * smoothstep(0.0, 0.15, s);
    return vec4<f32>(rgb * keep, a * keep);
}
";

/// The fire at pixels near every edge, in and round the lyric band and
/// what is kept clear (the score's corner, the badge and the meter: each
/// none in a few), at strengths, times, looks, every theme's colours and
/// music (still, flaring on a bar's first beat, the bass loud): the
/// shader's `fx_edge_fire` as [`edge_fire`].
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
                // Most within a fifth of an edge.
                let (x, y) = match k % 5 {
                    0 => (fr(0.613) * size.0, size.1 - fr(0.377) * 450.0),
                    1 => (fr(0.613) * 450.0, fr(0.377) * size.1),
                    2 => (size.0 - fr(0.613) * 450.0, fr(0.377) * size.1),
                    3 => (fr(0.613) * size.0, fr(0.377) * 250.0),
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
                    [x, y, fr(0.291) * 30.0, strength],
                    [size.0, size.1, band.0, band.1],
                    [fr(0.733), fr(0.417) * 2.0, fr(0.157) * 1.5, fr(0.889)],
                    if k % 7 == 0 { [0.0; 4] } else { score },
                ];
                let badge = if k % 11 == 0 { [0.0; 4] } else { badge };
                let meter = if k % 13 == 0 { [0.0; 4] } else { meter };
                let fire = THEMES[k % THEMES.len()].fire.uniform();
                (ins.into_iter().chain(music.uniform()))
                    .chain([badge, meter])
                    .chain(fire)
            })
            .collect()
    },
    ins: 16,
    outs: 1,
    body: "
    let j = i * 16u;
    let a = ins[j];
    let b = ins[j + 1u];
    let m = Music(ins[j + 4u], ins[j + 5u], ins[j + 6u], ins[j + 7u], ins[j + 8u]);
    let clear = mat3x4<f32>(ins[j + 3u], ins[j + 9u], ins[j + 10u]);
    let fire = array<vec4<f32>, FIRE_STOPS>(ins[j + 11u], ins[j + 12u], ins[j + 13u], ins[j + 14u], ins[j + 15u]);
    outs[i] = fx_edge_fire(a.xy, b.xy, a.z, vec3<f32>(a.w, b.zw), clear, ins[j + 2u], fire, m);",
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
        let fire = Ramp::of_uniform(&s[11..]);
        let look = (s[2], &fire);
        vec![edge_fire(px, size, s[0][2], (s[0][3], band), &clear, look, &m)]
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
    const BADGE: [f32; 4] = [300.0, 40.0, 1000.0, 150.0];
    const METER: [f32; 4] = [1100.0, 30.0, 1900.0, 160.0];
    /// What the fire keeps clear in these.
    const CLEAR: Clear = Clear {
        score: SCORE,
        badge: BADGE,
        meter: METER,
    };
    /// Its look as made.
    const LOOK: [f32; 4] = SPEC.params;
    /// The whole frame.
    const AREA: (f32, f32, f32, f32) = (0.0, 0.0, SIZE.0, SIZE.1);

    /// The fire with no music, in `fire`'s colours.
    fn themed(px: (f32, f32), (t, s): (f32, f32), p: [f32; 4], fire: &Ramp) -> [f32; 4] {
        let score = Clear {
            score: SCORE,
            ..Clear::NONE
        };
        edge_fire(px, SIZE, t, (s, BAND), &score, (p, fire), &Music::NEUTRAL)
    }

    /// The fire with no music, classic.
    fn fire(
        px: (f32, f32),
        size: (f32, f32),
        t: f32,
        s: f32,
        band: (f32, f32),
        p: [f32; 4],
    ) -> [f32; 4] {
        let score = Clear {
            score: SCORE,
            ..Clear::NONE
        };
        let look = (p, &crate::theme::CLASSIC.fire);
        edge_fire(px, size, t, (s, band), &score, look, &Music::NEUTRAL)
    }

    /// The fire at full strength, its look as made, to the music `m`.
    fn full(px: (f32, f32), t: f32, m: &Music) -> [f32; 4] {
        let look = (LOOK, &crate::theme::CLASSIC.fire);
        edge_fire(px, SIZE, t, (1.0, BAND), &CLEAR, look, m)
    }

    /// The music with a bar's pulse `downbeat`, the grid `sure`, the bass
    /// at `bass` (and the other stems far quieter).
    fn music(downbeat: f32, sure: f32, bass: f32) -> Music {
        let mut stems = [0.05; 4];
        stems[Stem::Bass as usize] = bass;
        Music {
            downbeat,
            sure,
            stems,
            ..Music::NEUTRAL
        }
    }

    /// How much of the bottom strip `m` lights at full strength.
    fn bottom_cover(m: &Music) -> f32 {
        (0..4000)
            .map(|i| {
                let (x, d) = ((i % 100) as f32 * 30.0, (i / 100) as f32 * 9.0);
                full((x, SIZE.1 - 1.0 - d), 2.2, m)[3]
            })
            .sum()
    }

    /// The flames flare up on a bar's first beat (its pulse high, the grid
    /// sure), reaching further and hotter, as the pulse falls away less;
    /// unsure, not at all; a loud bass swells them a little (the other
    /// stems not); the look's numbers turn either off.
    #[test]
    fn it_flares_on_the_one_and_breathes_with_the_bass() {
        let quiet = music(0.0, 0.0, 0.0);
        let still = bottom_cover(&quiet);
        let flare = bottom_cover(&music(1.0, 1.0, 0.0));
        let later = bottom_cover(&music(0.3, 1.0, 0.0));
        let unsure = bottom_cover(&music(1.0, 0.0, 0.0));
        let bass = bottom_cover(&music(0.0, 0.0, 1.0));
        assert!(flare > later && later > still, "{flare} {later} {still}");
        assert!(flare > still * 1.1, "{flare} {still}");
        assert_eq!(unsure, still);
        assert!(bass > still && bass < flare, "{bass}");
        let mut no_bass = Music {
            stems: [1.0; 4],
            ..quiet
        };
        no_bass.stems[Stem::Bass as usize] = 0.0;
        assert_eq!(swell(&no_bass, LOOK), (1.0, 0.0), "only the bass");
        assert_eq!(swell(&Music::NEUTRAL, LOOK), (1.0, 0.0));
        let (grow, hot) = swell(&music(1.0, 1.0, 1.0), LOOK);
        assert!((grow - (1.0 + FLARE + BREATH)).abs() < 1e-6 && hot == FLARE_HEAT);
        assert_eq!(
            swell(&music(1.0, 1.0, 1.0), [0.0, 1.0, 0.0, 0.0]),
            (1.0, 0.0)
        );
        // Hotter: whiter at a flame's root.
        let root = (1500.0, SIZE.1 - 3.0);
        let a = full(root, 1.0, &quiet);
        let b = full(root, 1.0, &music(1.0, 1.0, 0.0));
        assert!(b[2] / b[3] >= a[2] / a[3], "{a:?} {b:?}");
    }

    /// The strips grow with the swell: they hold the fire flared and
    /// breathing at its fullest.
    #[test]
    fn the_strips_hold_the_fire_swollen() {
        let m = music(1.0, 1.0, 1.0);
        let (grow, _) = swell(&m, LOOK);
        let strips = EDGES.strips(AREA, deepest(SIZE.1, 1.0, grow));
        for px in grid(17) {
            let lit = full(px, 2.7, &m)[3] > 0.0;
            let inside =
                (strips.iter()).any(|q| px.0 >= q[0] && px.0 < q[2] && px.1 >= q[1] && px.1 < q[3]);
            assert!(!lit || inside, "{px:?}");
        }
    }

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
                assert_eq!(fire(px, SIZE, 3.3, s, BAND, [0.0; 4]), [0.0; 4]);
            }
        }
    }

    /// Light over the frame, every number within 0 to 1: the flames' tips
    /// brighter than they cover (the frame shows through them), their
    /// roots covering most of it.
    #[test]
    fn it_is_light_over_the_frame_and_bounded() {
        let (mut tips, mut roots) = (0, 0);
        for (k, px) in grid(13).enumerate() {
            let s = [0.3, 0.6, 1.0, 1.7][k % 4];
            let theme = THEMES[k % THEMES.len()];
            let c = themed(px, (k as f32 * 0.01, s), [0.0, 1.0, 0.0, 0.0], &theme.fire);
            for v in c {
                assert!((0.0..=1.0).contains(&v), "{px:?}: {c:?}");
            }
            tips += usize::from(c[0] > 0.05 && c[3] < 0.5 * c[0]);
            roots += usize::from(c[3] > 0.8);
        }
        assert!(tips > 50 && roots > 50, "{tips} {roots}");
    }

    #[test]
    fn it_burns_at_the_edges_and_leaves_the_middle_alone() {
        let mid = fire((1512.0, 900.0), SIZE, 1.0, 1.0, BAND, [0.0; 4]);
        assert_eq!(mid, [0.0; 4]);
        // Along the bottom, most of its pixels burning bright at full strength.
        let lit = (0..300)
            .filter(|i| {
                fire(
                    (*i as f32 * 10.0, 1950.0),
                    SIZE,
                    2.0,
                    1.0,
                    (0.0, 0.0),
                    [0.0; 4],
                )[0] > 0.4
            })
            .count();
        assert!(lit > 250, "{lit}");
        // Tallest at the bottom: more of it there than up a side or at the top.
        // Its cover summed in from an edge (`d` in, `along` it).
        let cover = |f: &dyn Fn(f32, f32) -> (f32, f32)| -> f32 {
            (0..1600)
                .map(|i| {
                    let (d, along) = ((i % 40) as f32 * 8.0, (i / 40) as f32 * 23.0);
                    fire(f(d, along), SIZE, 1.5, 1.0, (0.0, 0.0), [0.0; 4])[3]
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
                let c = fire((x as f32, y as f32), SIZE, 4.2, 1.0, BAND, [0.0; 4]);
                most = most.max(c[3]);
            }
            let below = fire((x as f32, 1955.0), SIZE, 4.2, 1.0, BAND, [0.0; 4]);
            outside = outside.max(below[3]);
        }
        assert!(most <= KEEP.in_band + 1e-6, "{most}");
        assert!(outside > 0.8, "{outside}");
    }

    /// The score's corner is kept wholly clear, the flames flaring at their
    /// fullest, while the top burns on beside it.
    #[test]
    fn the_score_s_corner_is_kept_clear() {
        let m = music(1.0, 1.0, 1.0);
        let mut most = 0.0f32;
        let mut beside = 0.0f32;
        for x in (SCORE[0] as usize..SCORE[2] as usize).step_by(5) {
            for y in (SCORE[1] as usize..SCORE[3] as usize).step_by(5) {
                let c = full((x as f32, y as f32), 4.2, &m);
                most = most.max(c[3]);
            }
        }
        for x in (2030..2370).step_by(7) {
            let c = full((x as f32, 2.0), 4.2, &m);
            beside = beside.max(c[3]);
        }
        assert_eq!(most, 0.0);
        assert!(beside > 0.8, "{beside}");
    }

    /// The badge and the meter are kept wholly clear, the flames flaring at
    /// their fullest, while the top burns on beside them.
    #[test]
    fn the_badge_and_the_meter_are_kept_clear() {
        let m = music(1.0, 1.0, 1.0);
        for b in [BADGE, METER] {
            for x in (b[0] as usize..=b[2] as usize).step_by(5) {
                for y in (b[1] as usize..=b[3] as usize).step_by(5) {
                    for t in [0.7, 4.2] {
                        let c = full((x as f32, y as f32), t, &m);
                        assert_eq!(c, [0.0; 4], "{x} {y}");
                    }
                }
            }
        }
        let beside = (2030..2370)
            .step_by(7)
            .map(|x| full((x as f32, 2.0), 4.2, &m)[3])
            .fold(0.0, f32::max);
        assert!(beside > 0.8, "{beside}");
    }

    #[test]
    fn it_moves() {
        let changed = grid(29)
            .filter(|&px| {
                let a = fire(px, SIZE, 1.0, 1.0, BAND, [0.0; 4]);
                let b = fire(px, SIZE, 1.25, 1.0, BAND, [0.0; 4]);
                (a[3] - b[3]).abs() > 0.05
            })
            .count();
        assert!(changed > 200, "{changed}");
    }

    /// In the theme's colours: Spooky's fire green, as much of it as
    /// Classic's.
    #[test]
    fn it_burns_in_the_theme_s_colours() {
        use crate::theme::{CLASSIC, SPOOKY};
        let (px, p) = ((1500.0, 1960.0), [0.0, 1.0, 0.0, 0.0]);
        let classic = themed(px, (1.0, 1.0), p, &CLASSIC.fire);
        let ghost = themed(px, (1.0, 1.0), p, &SPOOKY.fire);
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
            let strips = EDGES.strips(AREA, deepest(SIZE.1, s, 1.0));
            let inside = |px: (f32, f32)| {
                (strips.iter())
                    .filter(|q| px.0 >= q[0] && px.0 < q[2] && px.1 >= q[1] && px.1 < q[3])
                    .count()
            };
            for px in grid(17) {
                assert!(inside(px) <= 1, "{s} {px:?}");
                if fire(px, SIZE, 2.7, s, BAND, [0.0; 4])[3] > 0.0 {
                    assert_eq!(inside(px), 1, "{s} {px:?}");
                }
            }
        }
        // A narrow window: the sides meet, never cross.
        let narrow = EDGES.strips((0.0, 0.0, 300.0, 1900.0), deepest(1900.0, 1.0, 1.0));
        assert!(narrow.windows(2).all(|w| w[0] != w[1]));
        assert!(narrow[2][2] <= narrow[3][0]);
    }
}
