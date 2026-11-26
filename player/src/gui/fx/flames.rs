//! Flames burning along the lane's top and bottom, licking out and
//! flickering, white at their roots to red at their tips: the one fire
//! ([`fire`]), light, drawn where it blooms ([`crate::gui::bloom`]), as
//! strong as the rows hold them (Amazing's first).
//!
//! Round the marquee's bulbs ([`super::bulbs`]) they set the bulbs on
//! fire, as the bulbs show: the flames burn from round each bulb instead of
//! the lane's edge, wrapping its outer side and licking up taller between
//! the bulbs, dying away on the lane's side of the row; drawn as light,
//! under the bulbs (a screen pass over it), which stay crisp on top, framed
//! by the fire. Marquee and flames read as one marquee on fire.
//!
//! They never reach what is round the lane: over it, no higher than the
//! bulbs light ([`super::bulbs::lit_top`], the HUD's band over it); under
//! it, not into the lyric band. Each flame fits its room and fades out
//! smoothly over the outer part of it, to nothing before its end; behind
//! its root line it fades out too, and at its span's ends: nothing of it
//! is left at its quad's edges, so no edge of it shows.
//!
//! They surge with the music ([`surge`]): taller and hotter on a bar's
//! first beat, a pulse on each beat, as sure as the beat grid is; its look
//! ([`Spec::params`]): `x` how much (1 as made, 0 not at all), `z` 0
//! classic orange to 1 a ghostly green.
//!
//! [`flames_at`] is the shader's reference ([`wgsl`]), held to it on the
//! GPU ([`REFERENCE`]).

use super::bulbs::{self, Row};
use super::helpers::{fire, fire_light, smoothstep};
use super::{At, Fx, Layer, Live, Music, Out, Reference, Slot, Spec};
use crate::gui::stage_gpu::Instance;

/// The flames along an edge: `shape` the span (from `x`, to `z`), its root
/// line (`y`) and outward (`w`: 1 down, -1 up), `params[1]` their height,
/// `params[2]` their strength, `params[3]` how far they may reach (a line
/// across); `color` the surge (taller, hotter) and the marquee (how much
/// it shows, a bulb's radius), `color2` its row (its first bulb's middle
/// across, the bulbs' gap, the row's line, how many gaps): light.
pub const BLAZE: f32 = Fx::Flames.kind(0);

/// The flames along the lane.
pub const SPEC: Spec = Spec {
    name: "flames",
    ident: "flames",
    at: &[Layer::Lane],
    draw,
    kinds: &["blaze"],
    wgsl: Some(wgsl),
    params: [1.0, 0.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

/// How high the flames are at most, in pills.
const HIGH: f32 = 1.5;
/// How much of their room their height is at most: the tallest tongues
/// reach into the fade at its end ([`FADE`]).
const FIT: f32 = 0.7;
/// Where in their room (out from their root line) the flames begin to
/// fade, and where they are gone: shares of it, gone a little before its
/// end (the quad's edge there).
const FADE: (f32, f32) = (0.35, 0.95);
/// How far behind their root line the flames' quad reaches, in heights;
/// they are gone a little before it.
const BEHIND: f32 = 0.4;
const GONE_BEHIND: f32 = 0.85 * BEHIND;
/// How far round a bulb's middle the flames rise from it, in its radii
/// (just outside its socket).
const HUG: f32 = 1.3;
/// How much taller the flames lick up between two bulbs.
const BETWEEN: f32 = 0.45;
/// How fast round a marquee the fire dies away on the lane's side of the
/// bulbs' row (as many times as it goes out from a bulb).
const INWARD: f32 = 2.0;
/// How hot the flames burn: the one fire's heat scaled, orange more than
/// white once their light blooms.
const HEAT: f32 = 0.7;
/// How much taller and hotter the flames burn on a bar's first beat (its
/// pulse at its height, the grid sure), and on each beat.
const FLARE: f32 = 0.55;
const FLARE_HEAT: f32 = 0.3;
const PULSE: f32 = 0.22;
const PULSE_HEAT: f32 = 0.15;

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let (
        &At::Lane {
            lane,
            pill,
            area,
            band,
            bulbs,
        },
        s,
    ) = (at, live.strength)
    else {
        return;
    };
    if s <= 0.01 {
        return;
    }
    let rect = lane.rect();
    let size = super::far_corner(area);
    let row = bulbs::row(rect, size);
    let marquee = row.map_or(0.0, |_| bulbs::shown(bulbs));
    let Row {
        ring,
        r,
        gaps: (across, _),
    } = row.unwrap_or(Row {
        ring: rect,
        r: 0.0,
        gaps: (1, 1),
    });
    let (grow, hot) = surge(&live.music, live.look);
    let mix = |a: f32, b: f32| a + (b - a) * marquee;
    let (x0, x1) = (mix(rect[0], ring[0]), mix(rect[2], ring[2]));
    let gap = (ring[2] - ring[0]) / across as f32;
    let edges = [
        (rect[1], ring[1], -1.0, bulbs::lit_top(rect, size)),
        (rect[3], ring[3], 1.0, band.0),
    ];
    for (edge, line, outward, limit) in edges {
        let root = mix(edge, line);
        let room = (limit - root) * outward;
        if room <= 1.0 {
            continue;
        }
        let high = (pill * HIGH).min(room * FIT);
        let behind = root - outward * high * BEHIND;
        let (y0, y1) = if outward < 0.0 {
            (limit, behind)
        } else {
            (behind, limit)
        };
        out.shapes.push(Instance {
            quad: [x0, y0, x1, y1],
            shape: [x0, root, x1, outward],
            color: [grow, hot, marquee, r],
            color2: [ring[0], gap, line, across as f32],
            params: [BLAZE, high, s, limit],
            ..Instance::default()
        });
    }
}

/// How the music `m` makes the flames surge, `p` their look (`x` how
/// much): how many times as tall they are (1 with no music) and how much
/// hotter they burn.
pub fn surge(m: &Music, p: [f32; 4]) -> (f32, f32) {
    let k = p[0].max(0.0);
    (
        1.0 + k * m.kick(PULSE, FLARE),
        k * m.kick(PULSE_HEAT, FLARE_HEAT),
    )
}

/// The flames' light at `p` ([`BLAZE`]'s shape, its numbers as its
/// instance has them: `shape`, `params`, `color`, `color2`) at `time`:
/// the one fire ([`fire`], as hot as [`HEAT`]) from their root line or,
/// as the marquee shows, from round its bulbs (dying away on the lane's
/// side of their row) and taller between them, as tall and as hot as they
/// surge, fading out before the line they may reach and at the span's
/// ends; `ghost` of the way to a ghostly green (their look's `z`).
pub fn flames_at(
    p: (f32, f32),
    shape: [f32; 4],
    params: [f32; 4],
    color: [f32; 4],
    color2: [f32; 4],
    (time, ghost): (f32, f32),
) -> [f32; 3] {
    let [x0, root, x1, outward] = shape;
    let [_, high, k, limit] = params;
    let [grow, hot, marquee, r] = color;
    let [first, gap, line, gaps] = color2;
    let h = high.max(1.0);
    let out = (p.1 - root) * outward;
    let room = (limit - root) * outward;
    let mut d = out;
    let mut tall = 1.0;
    if marquee > 0.0 && gap > 0.0 {
        let u = (p.0 - first) / gap;
        let near = (u + 0.5).floor();
        let bulb = first + near.clamp(0.0, gaps) * gap;
        let disc = (p.0 - bulb).hypot(p.1 - line) - r * HUG;
        let wrap = disc + (-d).max(0.0) * INWARD;
        d += (wrap - d) * marquee;
        tall += BETWEEN * marquee * smoothstep(0.15, 0.5, (u - near).abs());
    }
    let along = smoothstep(x0, x0 + h, p.0) * (1.0 - smoothstep(x1 - h, x1, p.0));
    let keep = (1.0 - smoothstep(FADE.0 * room, FADE.1 * room, out))
        * smoothstep(-GONE_BEHIND * h, 0.0, out);
    let heat = fire(p.0 / h, d / (h * grow * tall), time) * (1.0 + hot) * HEAT;
    fire_light(heat, ghost).map(|c| c * k * along * keep)
}

/// [`flames_at`] in WGSL and its shapes (`flames_shape`), for the stage's
/// shader (which gives them the one fire, `fx_fire`).
pub fn wgsl() -> String {
    format!(
        "const FLAMES_FADE: vec2<f32> = vec2<f32>({:?}, {:?});\nconst FLAMES_GONE_BEHIND: f32 = {GONE_BEHIND:?};\nconst FLAMES_HUG: f32 = {HUG:?};\nconst FLAMES_BETWEEN: f32 = {BETWEEN:?};\nconst FLAMES_INWARD: f32 = {INWARD:?};\nconst FLAMES_HEAT: f32 = {HEAT:?};\n{SHADER}",
        FADE.0, FADE.1
    )
}

const SHADER: &str = r"
fn flames_at(p: vec2<f32>, shape: vec4<f32>, params: vec4<f32>, color: vec4<f32>, color2: vec4<f32>, time: f32, ghost: f32) -> vec3<f32> {
    let h = max(params.y, 1.0);
    let out = (p.y - shape.y) * shape.w;
    let room = (params.w - shape.y) * shape.w;
    var d = out;
    var tall = 1.0;
    if color.z > 0.0 && color2.y > 0.0 {
        let u = (p.x - color2.x) / color2.y;
        let near = floor(u + 0.5);
        let bulb = color2.x + clamp(near, 0.0, color2.w) * color2.y;
        let disc = length(vec2<f32>(p.x - bulb, p.y - color2.z)) - color.w * FLAMES_HUG;
        let wrap = disc + max(-d, 0.0) * FLAMES_INWARD;
        d += (wrap - d) * color.z;
        tall += FLAMES_BETWEEN * color.z * smoothstep(0.15, 0.5, abs(u - near));
    }
    let along = smoothstep(shape.x, shape.x + h, p.x) * (1.0 - smoothstep(shape.z - h, shape.z, p.x));
    let keep = (1.0 - smoothstep(FLAMES_FADE.x * room, FLAMES_FADE.y * room, out))
        * smoothstep(-FLAMES_GONE_BEHIND * h, 0.0, out);
    let heat = fx_fire(p.x / h, d / (h * color.x * tall), time) * (1.0 + color.y) * FLAMES_HEAT;
    return fx_fire_light(heat, ghost) * (params.z * along * keep);
}

fn flames_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let l = flames_at(p, o.shape, o.params, o.color, o.color2, screen.time, looks[FX_FLAMES].z);
    return vec4<f32>(out_rgb(l), 0.0);
}
";

/// The flames round lanes along their edges, plain and round a marquee
/// (its bulbs half and wholly shown), still and surging, at points along
/// and off an edge, through their flicker: the shader's `flames_at` as
/// [`flames_at`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        (0..1200)
            .flat_map(|k| {
                let fr = |m: f32| (k as f32 * m).fract();
                let outward = if k % 2 == 0 { -1.0 } else { 1.0 };
                let marquee = [0.0, 0.5, 1.0][k % 3];
                let high = 40.0 + fr(0.43) * 40.0;
                let root = 300.0;
                let limit = root + outward * (high * 1.6 + fr(0.59) * 40.0);
                let p = [
                    100.0 + fr(0.19) * 1000.0,
                    root + outward * (fr(0.31) * 1.8 - 0.4) * high,
                    3.0 + fr(0.37) * 4.0,
                    [0.0, 1.0, 0.5, 0.0][k % 4],
                ];
                let grow = 1.0 + fr(0.71) * FLARE;
                [
                    p,
                    [100.0, root, 1100.0, outward],
                    [BLAZE, high, 0.8, limit],
                    [grow, fr(0.83) * FLARE_HEAT, marquee, 12.0],
                    [96.0, 44.0, root - outward * 15.0, 23.0],
                ]
            })
            .collect()
    },
    ins: 5,
    outs: 1,
    body: "
    let s = ins[i * 5u];
    outs[i] = vec4<f32>(flames_at(s.xy, ins[i * 5u + 1u], ins[i * 5u + 2u], ins[i * 5u + 3u], ins[i * 5u + 4u], s.z, s.w), 0.0);",
    cpu: |s| {
        let [r, g, b] = flames_at((s[0][0], s[0][1]), s[1], s[2], s[3], s[4], (s[0][2], s[0][3]));
        vec![[r, g, b, 0.0]]
    },
    near: (3e-3, 3e-3),
    shows: |o| o[0][0] + o[0][1] + o[0][2] > 0.05,
    least: 150,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A top edge's flames at `p`: root line at 300, 60 high, reaching up
    /// to 160, `marquee` shown round bulbs of radius 12 every 44 from 96 on
    /// a row at 290, surging `grow` and `hot`, at `time`.
    fn top(p: (f32, f32), marquee: f32, (grow, hot): (f32, f32), time: f32) -> [f32; 3] {
        flames_at(
            p,
            [100.0, 300.0, 1100.0, -1.0],
            [BLAZE, 60.0, 1.0, 160.0],
            [grow, hot, marquee, 12.0],
            [96.0, 44.0, 290.0, 23.0],
            (time, 0.0),
        )
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    /// Their light summed over the space over the root line (`over` it).
    fn lit(marquee: f32, surge: (f32, f32), over: std::ops::Range<i32>) -> f32 {
        let mut total = 0.0;
        for t in 0..6 {
            for x in (120..1080).step_by(5) {
                for y in over.clone().step_by(3) {
                    total += sum(top(
                        (x as f32, 300.0 - y as f32),
                        marquee,
                        surge,
                        t as f32 * 0.7,
                    ));
                }
            }
        }
        total
    }

    /// Nothing at or past the line they may reach, nor at the span's ends;
    /// light (never negative, bounded) burning from the root line out.
    #[test]
    fn they_burn_out_from_their_root_and_never_past_their_end() {
        for t in 0..40 {
            let time = t as f32 * 0.13;
            for x in (100..1100).step_by(7) {
                for y in (100..400).step_by(4) {
                    let c = top((x as f32, y as f32), 1.0, (1.0 + FLARE + PULSE, 1.0), time);
                    assert!(c.iter().all(|v| (0.0..=3.0).contains(v)), "{x} {y}: {c:?}");
                    if y <= 160 {
                        assert_eq!(c, [0.0; 3], "past the end: {x} {y}");
                    }
                }
                assert_eq!(top((100.0, 290.0), 0.0, (1.0, 0.0), time), [0.0; 3]);
            }
        }
        assert!(lit(0.0, (1.0, 0.0), 0..60) > 10.0 * lit(0.0, (1.0, 0.0), -40..-10));
    }

    /// Round a marquee they burn from round each bulb (more light over a
    /// bulb's top than with none) and lick up taller between the bulbs.
    #[test]
    fn round_the_bulbs_they_burn_from_each_and_between() {
        let over_bulbs = |marquee: f32| -> f32 {
            (0..23)
                .flat_map(|b| (0..8).map(move |t| (b, t)))
                .map(|(b, t)| {
                    let x = 96.0 + b as f32 * 44.0;
                    sum(top(
                        (x, 290.0 - 12.0 * HUG - 3.0),
                        marquee,
                        (1.0, 0.0),
                        t as f32 * 0.4,
                    ))
                })
                .sum()
        };
        assert!(
            over_bulbs(1.0) > 2.0 * over_bulbs(0.0),
            "{}",
            over_bulbs(1.0)
        );
        // Between two bulbs, high up, more than over a bulb as high.
        let high = |x0: f32| -> f32 {
            (0..23)
                .flat_map(|b| (0..10).map(move |t| (b, t)))
                .map(|(b, t)| {
                    let x = x0 + b as f32 * 44.0;
                    sum(top(
                        (x, 290.0 - 12.0 * HUG - 30.0),
                        1.0,
                        (1.0, 0.0),
                        t as f32 * 0.3,
                    ))
                })
                .sum()
        };
        assert!(high(118.0) > high(96.0), "{} {}", high(118.0), high(96.0));
    }

    /// On a bar's first beat (the grid sure) they burn taller and hotter,
    /// on a beat less, unsure or with their look off not at all.
    #[test]
    fn they_surge_on_the_one_and_pulse_on_each_beat() {
        let one = Music::steady(8.0, 120.0, 4, 1.0, 0.0);
        let beat = Music::steady(9.0, 120.0, 4, 1.0, 0.0);
        let between = Music::steady(9.5, 120.0, 4, 1.0, 0.0);
        let unsure = Music::steady(8.0, 120.0, 4, 0.0, 0.0);
        let look = SPEC.params;
        let (g1, h1) = surge(&one, look);
        let (g2, h2) = surge(&beat, look);
        let (g3, _) = surge(&between, look);
        assert!(g1 > g2 && g2 > g3 && g3 < 1.06, "{g1} {g2} {g3}");
        assert!(g1 > 1.0 + FLARE && h1 > h2 && h2 > 0.1, "{g1} {h1} {h2}");
        assert_eq!(surge(&unsure, look), (1.0, 0.0));
        assert_eq!(surge(&one, [0.0; 4]), (1.0, 0.0));
        assert_eq!(surge(&Music::NEUTRAL, look), (1.0, 0.0));
        let still = lit(1.0, (1.0, 0.0), 0..120);
        assert!(lit(1.0, (g1, h1), 0..120) > 1.5 * still, "{still}");
    }
}
