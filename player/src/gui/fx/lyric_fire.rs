//! The lyrics on fire: a lyric treatment. Flames lick up from the bottom
//! of the line, heating its letters from their feet (orange, yellow, white
//! hot where the flames are tallest) and burning low behind them, between
//! the letters and over the small ones, sparks rising off them. The
//! letters keep their shapes and stay bright, so the line reads as well
//! as ever.
//!
//! [`lyric_fire`] is the shader's reference ([`wgsl`]: `lyric_fire`),
//! held to it on the GPU ([`REFERENCE`]). Its look ([`Spec::params`]):
//! `x` 0 classic orange to 1 a ghostly green, `y` how fast it flickers.

use super::helpers::{clamp01, heat, lattice, letter_up, mix3, smoothstep, speed, vnoise};
use super::{Live, Lyric, Reference, Spec, Text};

/// The line sung on fire.
pub const SPEC: Spec = Spec {
    name: "lyrics on fire",
    ident: "fire",
    wgsl: Some(wgsl),
    lyric: Some(Lyric {
        on: Text::Sung,
        strain: held,
    }),
    params: [0.0, 1.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

/// As strong as it is held, eased on the processor.
pub(super) fn held(live: &Live) -> Option<[f32; 3]> {
    let s = live.strength;
    (s > 0.0).then_some([s, s, 0.0])
}

/// How opaque the flames behind the letters are at most.
const BED: f32 = 0.55;

/// A lyric's colour `col` (straight, 0 to 1) and coverage `a` at pixel
/// `px` of its text run's quad `bx` (left, top, right, bottom, pixels) at
/// `time` (seconds), on fire as strongly as `strength` (0: as it was, 1:
/// flames up to its letters' tops), `p` its look (`x` 0 classic orange to
/// 1 a ghostly green, `y` how fast it flickers: 1 as made, 0 taken as 1):
/// its new colour (straight) and coverage, the flames behind the letters
/// covering some of the quad the letters leave.
pub fn lyric_fire(
    col: [f32; 3],
    a: f32,
    px: (f32, f32),
    bx: [f32; 4],
    time: f32,
    strength: f32,
    p: [f32; 4],
) -> [f32; 4] {
    if strength <= 0.0 {
        return [col[0], col[1], col[2], a];
    }
    let s = strength.min(1.0);
    let high = (bx[3] - bx[1]).max(1.0);
    let z = letter_up(px.1, bx);
    let t = time * speed(p);
    let l = high * 0.085;
    let n = 0.6 * vnoise(px.0 / l, px.1 / (l * 2.6) + t * 2.4)
        + 0.4 * vnoise(px.0 / (l * 0.45) + 3.1, px.1 / (l * 1.3) + t * 4.3);
    // The flames' height here (from the letters' feet, a share of their
    // height) and how hot they are this far up them.
    let tip = s * (0.4 + 1.1 * n);
    let k = clamp01(1.0 - z.max(0.0) / tip.max(1e-3));
    // The letters heated from their feet: never darker than orange.
    let hot = heat(0.55 + 0.45 * k, p[0]);
    let tint = smoothstep(0.0, 0.35, k) * (0.6 + 0.35 * s);
    let letter = mix3(col, hot, tint);
    // The flames behind them: ragged tongues, deep red to orange (darker
    // than the letters), from just under their feet, fading at the quad's
    // ends.
    let along = smoothstep(bx[0], bx[0] + high * 0.4, px.0)
        * (1.0 - smoothstep(bx[2] - high * 0.4, bx[2], px.0));
    let bed_a =
        smoothstep(0.05, 0.4, k) * (0.4 + 0.6 * n) * BED * s * along * smoothstep(0.0, 0.25, z);
    let bed = heat(0.2 + 0.55 * k, p[0]);
    let under_a = a + bed_a * (1.0 - a);
    let under = [0, 1, 2].map(|i| letter[i] * a + bed[i] * bed_a * (1.0 - a));
    // Sparks: one in a few cells of a grid rising (each column at its own
    // speed) off the line, over everything, fading as they rise.
    let c = high * 0.12;
    let column = (px.0 / c).floor();
    let rise = c * (1.5 + 1.5 * lattice(column, 3.0)) * t;
    let row = ((px.1 + rise) / c).floor();
    let r = lattice(column, row);
    let mut spark = 0.0;
    if r > 0.8 {
        let ex = (column + 0.25 + 0.5 * lattice(column + 19.0, row)) * c;
        let ey = (row + 0.5) * c - rise;
        let d2 = (px.0 - ex) * (px.0 - ex) + (px.1 - ey) * (px.1 - ey);
        let big = c * 0.14;
        spark = (-d2 / (big * big)).exp()
            * smoothstep(0.2, 0.6, z)
            * (1.0 - smoothstep(1.0, 1.4, z))
            * s
            * along;
    }
    let glow = heat(0.85, p[0]);
    let out_a = spark + under_a * (1.0 - spark);
    let rgb = [0, 1, 2].map(|i| {
        let premul = glow[i] * spark + under[i] * (1.0 - spark);
        if out_a > 1e-6 {
            premul / out_a.max(1e-6)
        } else {
            letter[i]
        }
    });
    [rgb[0], rgb[1], rgb[2], out_a]
}

/// [`lyric_fire`] in WGSL: `lyric_fire` and its helpers (each named
/// `lyric_fire_...`).
pub fn wgsl() -> String {
    format!("const LYRIC_FIRE_BED: f32 = {BED:?};\n{SHADER}")
}

const SHADER: &str = r"
fn lyric_fire(col: vec3<f32>, a: f32, px: vec2<f32>, box: vec4<f32>, time: f32, strength: f32, p: vec4<f32>) -> vec4<f32> {
    if strength <= 0.0 {
        return vec4<f32>(col, a);
    }
    let s = min(strength, 1.0);
    let high = max(box.w - box.y, 1.0);
    let z = fx_letter_up(px.y, box);
    let t = time * fx_speed(p);
    let l = high * 0.085;
    let n = 0.6 * vnoise(px.x / l, px.y / (l * 2.6) + t * 2.4)
        + 0.4 * vnoise(px.x / (l * 0.45) + 3.1, px.y / (l * 1.3) + t * 4.3);
    let tip = s * (0.4 + 1.1 * n);
    let k = clamp(1.0 - max(z, 0.0) / max(tip, 1e-3), 0.0, 1.0);
    let hot = fx_heat(0.55 + 0.45 * k, p.x);
    let tint = smoothstep(0.0, 0.35, k) * (0.6 + 0.35 * s);
    let letter = mix(col, hot, tint);
    let along = smoothstep(box.x, box.x + high * 0.4, px.x)
        * (1.0 - smoothstep(box.z - high * 0.4, box.z, px.x));
    let bed_a = smoothstep(0.05, 0.4, k) * (0.4 + 0.6 * n) * LYRIC_FIRE_BED * s * along * smoothstep(0.0, 0.25, z);
    let bed = fx_heat(0.2 + 0.55 * k, p.x);
    let under_a = a + bed_a * (1.0 - a);
    let under = letter * a + bed * bed_a * (1.0 - a);
    let c = high * 0.12;
    let column = floor(px.x / c);
    let rise = c * (1.5 + 1.5 * lattice(column, 3.0)) * t;
    let row = floor((px.y + rise) / c);
    let r = lattice(column, row);
    var spark = 0.0;
    if r > 0.8 {
        let ex = (column + 0.25 + 0.5 * lattice(column + 19.0, row)) * c;
        let ey = (row + 0.5) * c - rise;
        let d2 = (px.x - ex) * (px.x - ex) + (px.y - ey) * (px.y - ey);
        let big = c * 0.14;
        spark = exp(-d2 / (big * big)) * smoothstep(0.2, 0.6, z) * (1.0 - smoothstep(1.0, 1.4, z)) * s * along;
    }
    let glow = fx_heat(0.85, p.x);
    let out_a = spark + under_a * (1.0 - spark);
    let premul = glow * spark + under * (1.0 - spark);
    let rgb = select(letter, premul / max(out_a, 1e-6), out_a > 1e-6);
    return vec4<f32>(rgb, out_a);
}
";

/// The quad of the line the treatments' references are drawn over.
pub(super) const QUAD: [f32; 4] = [800.0, 1300.0, 2200.0, 1440.0];

/// Pixels over a line's quad (and just past its ends), its letters'
/// colours and coverage, strengths, times and looks: a treatment's
/// reference's samples (its colour and coverage; where, when, how strong;
/// its look).
pub(super) fn samples() -> Vec<[f32; 4]> {
    let bx = QUAD;
    (0..3000)
        .flat_map(|k| {
            let f = k as f32;
            let fr = |m: f32| crate::gui::rainbow::fract(f * m);
            let x = bx[0] - 10.0 + fr(0.613) * (bx[2] - bx[0] + 20.0);
            let y = bx[1] + fr(0.377) * (bx[3] - bx[1]);
            let strength = [0.0, 0.3, 0.6, 1.0][k % 4];
            [
                [fr(0.11), fr(0.23), fr(0.31), [0.0, 0.4, 1.0][k % 3]],
                [x, y, fr(0.291) * 30.0, strength],
                [fr(0.733), fr(0.417) * 2.0, 0.0, 0.0],
            ]
        })
        .collect()
}

/// A treatment's colour and coverage, premultiplied: a colour where
/// nothing covers (as little as a GPU flushes to none) is no colour.
pub(super) fn premultiplied(c: [f32; 4]) -> [f32; 4] {
    [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
}

/// The line on fire at its quad's pixels: the shader's `lyric_fire` as
/// [`lyric_fire`] (premultiplied).
pub const REFERENCE: Reference = Reference {
    samples,
    ins: 3,
    outs: 1,
    body: "
    let c = ins[i * 3u];
    let a = ins[i * 3u + 1u];
    let quad = vec4<f32>(800.0, 1300.0, 2200.0, 1440.0);
    let o = lyric_fire(c.xyz, c.w, a.xy, quad, a.z, a.w, ins[i * 3u + 2u]);
    outs[i] = vec4<f32>(o.rgb * o.a, o.a);",
    cpu: |s| {
        let (c, a) = (s[0], s[1]);
        let o = lyric_fire(
            [c[0], c[1], c[2]],
            c[3],
            (a[0], a[1]),
            QUAD,
            a[2],
            a[3],
            s[2],
        );
        vec![premultiplied(o)]
    },
    near: (4e-3, 0.0),
    shows: |o| o[0][3] > 0.05,
    least: 1000,
};
#[cfg(test)]
mod tests {
    use super::*;

    const BOX: [f32; 4] = [800.0, 1300.0, 2200.0, 1440.0];

    fn pixels() -> impl Iterator<Item = (f32, f32)> {
        (0..2000).map(|k| {
            let f = k as f32;
            (
                BOX[0] + crate::gui::rainbow::fract(f * 0.613) * (BOX[2] - BOX[0]),
                BOX[1] + crate::gui::rainbow::fract(f * 0.377) * (BOX[3] - BOX[1]),
            )
        })
    }

    #[test]
    fn at_no_strength_it_is_unchanged() {
        for (k, px) in pixels().enumerate() {
            let col = [0.1 + (k % 7) as f32 * 0.1, 0.9, 0.33];
            let a = (k % 11) as f32 / 10.0;
            assert_eq!(
                lyric_fire(col, a, px, BOX, 1.7, 0.0, [0.0; 4]),
                [col[0], col[1], col[2], a]
            );
        }
    }

    #[test]
    fn it_is_bounded_and_keeps_the_letters() {
        let (mut gaps, mut covered) = (0, 0.0);
        for (k, px) in pixels().enumerate() {
            let a = (k % 5) as f32 / 4.0;
            let s = [0.3, 0.6, 1.0][k % 3];
            let c = lyric_fire([1.0, 1.0, 1.0], a, px, BOX, k as f32 * 0.01, s, [0.0; 4]);
            assert!(c.iter().all(|v| (0.0..=1.0 + 1e-5).contains(v)), "{c:?}");
            // A letter covers as much as it did, more where flames are behind.
            assert!(c[3] >= a - 1e-6, "{c:?} {a}");
            if a == 0.0 {
                gaps += 1;
                covered += c[3];
            }
        }
        // The flames behind and the sparks cover the gaps only a little.
        let mean = covered / gaps as f32;
        assert!(mean < 0.2, "{mean}");
    }

    #[test]
    fn the_letters_are_hot_at_their_feet_and_stay_bright() {
        let at = |y: f32| -> [f32; 3] {
            let mut sum = [0.0; 3];
            for i in 0..200 {
                let px = (BOX[0] + 100.0 + i as f32 * 6.0, y);
                let c = lyric_fire([1.0; 3], 1.0, px, BOX, 2.0, 1.0, [0.0; 4]);
                for j in 0..3 {
                    sum[j] += c[j] / 200.0;
                }
            }
            sum
        };
        // A third of the way up the letters, and over their tops.
        let feet = at(BOX[1] + 63.0);
        let top = at(BOX[1] + 14.0);
        // Warmer (less blue) at the feet than at the top.
        assert!(feet[2] < top[2] - 0.2, "{feet:?} {top:?}");
        // Bright: never below orange.
        assert!(feet[0] > 0.95 && feet[1] > 0.45, "{feet:?}");
    }

    #[test]
    fn it_flickers() {
        let changed = pixels()
            .filter(|&px| {
                let a = lyric_fire([1.0; 3], 0.0, px, BOX, 1.0, 1.0, [0.0; 4]);
                let b = lyric_fire([1.0; 3], 0.0, px, BOX, 1.2, 1.0, [0.0; 4]);
                (a[3] - b[3]).abs() > 0.05
            })
            .count();
        assert!(changed > 100, "{changed}");
    }
}
