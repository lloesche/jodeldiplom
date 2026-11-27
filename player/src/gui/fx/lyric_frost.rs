//! The lyrics frosted: a lyric treatment. Frost creeps over the line's
//! letters from their edges (their tops and feet first, then the line's
//! ends) as the strength grows: the letters turn pale and cold, keeping a
//! little of their colour (gold sung, white to come), veined with blue
//! ice crystals, glints twinkling on and round them. The letters keep
//! their shapes and stay light, so the line reads as well as ever.
//!
//! [`lyric_frost`] is the shader's reference ([`wgsl`]: `lyric_frost`),
//! held to it on the GPU ([`REFERENCE`]). Its look ([`Spec::params`]): `y`
//! how fast its glints twinkle.

use super::helpers::{lattice, letter_up, mix3, sat, smoothstep, speed, vnoise};
use super::lyric_fire::{held, premultiplied, samples, QUAD};
use super::{Lyric, Reference, Spec, Text};

/// The line sung frosted.
pub const SPEC: Spec = Spec {
    name: "lyrics frosted",
    ident: "frost",
    wgsl: Some(wgsl),
    lyric: Some(Lyric {
        on: Text::Sung,
        strain: held,
    }),
    params: [0.0, 1.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

/// The letters' cold: how much they pale towards white, and the blue they
/// are lit with.
const PALE: f32 = 0.6;
const COLD: [f32; 3] = [0.88, 0.95, 1.0];
/// The ice crystals' colour, and the glints'.
const VEIN: [f32; 3] = [0.55, 0.78, 1.0];
const GLINT: [f32; 3] = [0.92, 0.97, 1.0];

/// A lyric's colour `col` (straight, 0 to 1) and coverage `a` at pixel
/// `px` of its text run's quad `bx` (left, top, right, bottom, pixels) at
/// `time` (seconds), frosted as strongly as `strength` (0: as it was, 1:
/// all of it frosted), `p` its look (`y` how fast its glints twinkle: 1
/// as made, 0 taken as 1): its new colour (straight) and coverage, the
/// glints covering a little of the quad the letters leave.
pub fn lyric_frost(
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
    // How far in from the letters' edges: 0 at their tops, their feet and
    // the line's ends, 1 half way up them.
    let z = letter_up(px.1, bx);
    let e = (z.min(1.0 - z) * 2.0)
        .min((px.0 - bx[0]).min(bx[2] - px.0) / high * 1.2)
        .max(0.0);
    let t = time * speed(p);
    // The frost's front, ragged, creeping in as the strength grows.
    let m = vnoise(px.0 / (high * 0.25), px.1 / (high * 0.25) + t * 0.15);
    let front = s * 1.05 + (m - 0.5) * 0.5;
    let frost = smoothstep(e - 0.25, e, front);
    // Its crystals: thin ridges of noise stretched along two slants.
    let l = high * 0.09;
    let n1 = vnoise((px.0 + px.1 * 0.55) / l, (px.1 - px.0 * 0.55) / (l * 0.35));
    let n2 = vnoise(
        (px.0 - px.1 * 0.7) / (l * 0.6) + 9.1,
        (px.1 + px.0 * 0.7) / (l * 0.22),
    );
    let r1 = 1.0 - (2.0 * n1 - 1.0).abs();
    let r2 = 1.0 - (2.0 * n2 - 1.0).abs();
    let crystal = smoothstep(0.8, 0.97, r1).max(smoothstep(0.82, 0.98, r2) * 0.8) * frost;
    // The letters paled and lit cold, veined.
    let cold = [0, 1, 2].map(|i| (col[i] + (1.0 - col[i]) * PALE) * COLD[i]);
    let letter = mix3(col, cold, frost * (0.7 + 0.3 * s));
    let letter = mix3(letter, VEIN, crystal * 0.55);
    // The glints: one in some cells of a grid, each twinkling in its own
    // time, a point and a cross of rays, fading at the quad's edges.
    let v = sat((px.1 - bx[1]) / high);
    let ends = smoothstep(bx[0], bx[0] + high * 0.3, px.0)
        * (1.0 - smoothstep(bx[2] - high * 0.3, bx[2], px.0))
        * smoothstep(0.0, 0.1, v)
        * (1.0 - smoothstep(0.9, 1.0, v));
    let cell = high * 0.22;
    let (cx, cy) = ((px.0 / cell).floor(), (px.1 / cell).floor());
    let r = lattice(cx, cy + 53.0);
    let mut glint = 0.0;
    if r > 0.55 {
        let gx = (cx + 0.2 + 0.6 * lattice(cx + 11.0, cy)) * cell;
        let gy = (cy + 0.2 + 0.6 * lattice(cx, cy + 29.0)) * cell;
        let (dx, dy) = (px.0 - gx, px.1 - gy);
        let w = (t * (1.5 + 2.5 * r) + r * 60.0).sin().max(0.0);
        let twinkle = (w * w) * (w * w);
        let (big, thin, long) = (cell * 0.06, cell * 0.02, cell * 0.42);
        let core = (-(dx * dx + dy * dy) / (big * big)).exp();
        let ray_x = (-(dy * dy) / (thin * thin)).exp() * sat(1.0 - dx.abs() / long);
        let ray_y = (-(dx * dx) / (thin * thin)).exp() * sat(1.0 - dy.abs() / long);
        glint = sat(core + ray_x.max(ray_y) * 0.8) * twinkle * frost * ends * s;
    }
    let out_a = glint + a * (1.0 - glint);
    let rgb = [0, 1, 2].map(|i| {
        let premul = GLINT[i] * glint + letter[i] * a * (1.0 - glint);
        if out_a > 1e-6 {
            premul / out_a.max(1e-6)
        } else {
            letter[i]
        }
    });
    [rgb[0], rgb[1], rgb[2], out_a]
}

/// [`lyric_frost`] in WGSL: `lyric_frost` and its helpers (each named
/// `lyric_frost_...`).
pub fn wgsl() -> String {
    let v = |c: [f32; 3]| format!("vec3<f32>({:?}, {:?}, {:?})", c[0], c[1], c[2]);
    format!(
        "const LYRIC_FROST_PALE: f32 = {PALE:?};\nconst LYRIC_FROST_COLD: vec3<f32> = {};\nconst LYRIC_FROST_VEIN: vec3<f32> = {};\nconst LYRIC_FROST_GLINT: vec3<f32> = {};\n{SHADER}",
        v(COLD),
        v(VEIN),
        v(GLINT),
    )
}

const SHADER: &str = r"
fn lyric_frost(col: vec3<f32>, a: f32, px: vec2<f32>, box: vec4<f32>, time: f32, strength: f32, p: vec4<f32>, th: Theme) -> vec4<f32> {
    if strength <= 0.0 {
        return vec4<f32>(col, a);
    }
    let s = min(strength, 1.0);
    let high = max(box.w - box.y, 1.0);
    let z = fx_letter_up(px.y, box);
    let e = max(min(min(z, 1.0 - z) * 2.0, min(px.x - box.x, box.z - px.x) / high * 1.2), 0.0);
    let t = time * fx_speed(p);
    let m = vnoise(px.x / (high * 0.25), px.y / (high * 0.25) + t * 0.15);
    let front = s * 1.05 + (m - 0.5) * 0.5;
    let frost = smoothstep(e - 0.25, e, front);
    let l = high * 0.09;
    let n1 = vnoise((px.x + px.y * 0.55) / l, (px.y - px.x * 0.55) / (l * 0.35));
    let n2 = vnoise((px.x - px.y * 0.7) / (l * 0.6) + 9.1, (px.y + px.x * 0.7) / (l * 0.22));
    let r1 = 1.0 - abs(2.0 * n1 - 1.0);
    let r2 = 1.0 - abs(2.0 * n2 - 1.0);
    let crystal = max(smoothstep(0.8, 0.97, r1), smoothstep(0.82, 0.98, r2) * 0.8) * frost;
    let cold = (col + (vec3<f32>(1.0) - col) * LYRIC_FROST_PALE) * LYRIC_FROST_COLD;
    var letter = mix(col, cold, frost * (0.7 + 0.3 * s));
    letter = mix(letter, LYRIC_FROST_VEIN, crystal * 0.55);
    let v = clamp((px.y - box.y) / high, 0.0, 1.0);
    let ends = smoothstep(box.x, box.x + high * 0.3, px.x)
        * (1.0 - smoothstep(box.z - high * 0.3, box.z, px.x))
        * smoothstep(0.0, 0.1, v)
        * (1.0 - smoothstep(0.9, 1.0, v));
    let cell = high * 0.22;
    let cx = floor(px.x / cell);
    let cy = floor(px.y / cell);
    let r = lattice(cx, cy + 53.0);
    var glint = 0.0;
    if r > 0.55 {
        let gx = (cx + 0.2 + 0.6 * lattice(cx + 11.0, cy)) * cell;
        let gy = (cy + 0.2 + 0.6 * lattice(cx, cy + 29.0)) * cell;
        let dx = px.x - gx;
        let dy = px.y - gy;
        let w = max(sin(t * (1.5 + 2.5 * r) + r * 60.0), 0.0);
        let twinkle = (w * w) * (w * w);
        let big = cell * 0.06;
        let thin = cell * 0.02;
        let long = cell * 0.42;
        let core = exp(-(dx * dx + dy * dy) / (big * big));
        let ray_x = exp(-(dy * dy) / (thin * thin)) * clamp(1.0 - abs(dx) / long, 0.0, 1.0);
        let ray_y = exp(-(dx * dx) / (thin * thin)) * clamp(1.0 - abs(dy) / long, 0.0, 1.0);
        glint = clamp(core + max(ray_x, ray_y) * 0.8, 0.0, 1.0) * twinkle * frost * ends * s;
    }
    let out_a = glint + a * (1.0 - glint);
    let premul = LYRIC_FROST_GLINT * glint + letter * a * (1.0 - glint);
    let rgb = select(letter, premul / max(out_a, 1e-6), out_a > 1e-6);
    return vec4<f32>(rgb, out_a);
}
";

/// The line frosted at its quad's pixels: the shader's `lyric_frost` as
/// [`lyric_frost`] (premultiplied).
pub const REFERENCE: Reference = Reference {
    samples,
    ins: 3,
    outs: 1,
    body: "
    let c = ins[i * 3u];
    let a = ins[i * 3u + 1u];
    let quad = vec4<f32>(800.0, 1300.0, 2200.0, 1440.0);
    let o = lyric_frost(c.xyz, c.w, a.xy, quad, a.z, a.w, ins[i * 3u + 2u], Theme());
    outs[i] = vec4<f32>(o.rgb * o.a, o.a);",
    cpu: |s| {
        let (c, a) = (s[0], s[1]);
        let o = lyric_frost(
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
                BOX[0] + jodel_looks::math::fract(f * 0.613) * (BOX[2] - BOX[0]),
                BOX[1] + jodel_looks::math::fract(f * 0.377) * (BOX[3] - BOX[1]),
            )
        })
    }

    #[test]
    fn at_no_strength_it_is_unchanged() {
        for (k, px) in pixels().enumerate() {
            let col = [0.1 + (k % 7) as f32 * 0.1, 0.9, 0.33];
            let a = (k % 11) as f32 / 10.0;
            assert_eq!(
                lyric_frost(col, a, px, BOX, 1.7, 0.0, [0.0; 4]),
                [col[0], col[1], col[2], a]
            );
        }
    }

    #[test]
    fn it_is_bounded_and_keeps_the_letters() {
        for (k, px) in pixels().enumerate() {
            let a = (k % 5) as f32 / 4.0;
            let s = [0.3, 0.6, 1.0][k % 3];
            let c = lyric_frost([1.0, 0.72, 0.2], a, px, BOX, k as f32 * 0.01, s, [0.0; 4]);
            assert!(c.iter().all(|v| (0.0..=1.0 + 1e-5).contains(v)), "{c:?}");
            assert!(c[3] >= a - 1e-6, "{c:?} {a}");
        }
    }

    #[test]
    fn it_creeps_in_from_the_edges_and_turns_the_letters_cold() {
        // Gold letters: bluer where frosted.
        let blue = |s: f32, y: f32| -> f32 {
            (0..300)
                .map(|i| {
                    let px = (BOX[0] + 100.0 + i as f32 * 4.0, y);
                    lyric_frost([1.0, 0.72, 0.2], 1.0, px, BOX, 2.0, s, [0.0; 4])[2]
                })
                .sum::<f32>()
                / 300.0
        };
        // At the capitals' tops, and half way up the letters.
        let (edge, middle) = (BOX[1] + 28.0, BOX[1] + 52.0);
        // A little: at the edge, hardly in the middle.
        assert!(blue(0.3, edge) > blue(0.3, middle) + 0.15);
        // All of it at full strength, and no longer gold.
        assert!(blue(1.0, middle) > 0.45, "{}", blue(1.0, middle));
    }

    #[test]
    fn its_glints_twinkle() {
        let changed = pixels()
            .filter(|&px| {
                let a = lyric_frost([1.0; 3], 0.0, px, BOX, 1.0, 1.0, [0.0; 4]);
                let b = lyric_frost([1.0; 3], 0.0, px, BOX, 1.4, 1.0, [0.0; 4]);
                (a[3] - b[3]).abs() > 0.01
            })
            .count();
        assert!(changed > 5, "{changed}");
    }
}
