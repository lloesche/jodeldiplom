//! The results screen's background sheet (the result diploma's paper) on
//! the GPU, in the stage's shader: the sheet
//! (cream, its fibres and its aged edges, an ornate border of interlaced
//! waves between ruled lines with a rosette in each corner, in two inks),
//! unrolling from its top, the rest of it a roll lying across under its
//! front, the sheet casting a soft shadow; a red rubber stamp's frame
//! (two rounded rules, the ink worn in places, specks where it did not
//! take over the words inside it); and a signature written in one stroke
//! (the core's: [`crate::core::singing::signature`]), as far as it is
//! written. Text turned round its middle (a stamp's words) is drawn with
//! [`turn_about`].
//!
//! These functions are the shader's reference ([`wgsl`]), held to it on
//! the GPU (`super::shader_check`). Their sizes are in pixels and in `s`,
//! the sheet's unit (as the scene lays it out).

use std::f32::consts::{PI, TAU};

use super::glory::vnoise;
use super::rainbow::round_box;
use super::seal::{clamp01, line, smoothstep};
use crate::core::singing::signature::{pen_point, PEN};

/// The roll's radius at its start, in `s`.
const ROLL: f32 = 3.2;
/// `p` turned by `a` (radians, clockwise on the screen) round `c`.
pub fn turn_about(p: (f32, f32), c: (f32, f32), a: f32) -> (f32, f32) {
    let (s, co) = a.sin_cos();
    let (dx, dy) = (p.0 - c.0, p.1 - c.1);
    (c.0 + co * dx - s * dy, c.1 + s * dx + co * dy)
}

/// The border's two inks at `(x, y)`, `e` inside the sheet's edge, on a
/// sheet `half` its half size, `s` its unit: its rules and rosettes and
/// one family of its waves, and the other family.
fn border(x: f32, y: f32, half: (f32, f32), e: f32, s: f32) -> (f32, f32) {
    let (m, bw) = (2.0 * s, 3.0 * s);
    let v = (e - m) / bw;
    let rules = line(e - m, 0.16 * s)
        + line(e - m - bw, 0.16 * s)
        + 0.6 * line(e - m - bw - 0.8 * s, 0.1 * s);
    // A rosette in each corner, over the band.
    let (cx, cy) = (half.0 - m - bw * 0.5, half.1 - m - bw * 0.5);
    let (qx, qy) = (x.abs() - cx, y.abs() - cy);
    let rr = (qx * qx + qy * qy).sqrt();
    let th = qy.atan2(qx);
    let big = bw * 0.95;
    let mut rosette = line(rr - big, 0.14 * s) + line(rr, 0.3 * s);
    for j in 0..3 {
        let petal = big * (0.55 + 0.33 * (8.0 * th + j as f32 * TAU / 3.0).sin());
        rosette += line(rr - petal, 0.1 * s);
    }
    let corner = 1.0 - smoothstep(big * 1.02, big * 1.12, rr);
    // The band's waves, along its nearer side, interlaced.
    let along = if half.0 - x.abs() < half.1 - y.abs() {
        y
    } else {
        x
    };
    let k = TAU / (3.4 * s);
    let band = smoothstep(0.0, 0.08, v) * (1.0 - smoothstep(0.92, 1.0, v)) * (1.0 - corner);
    let (mut a, mut b) = (0.0, 0.0);
    for i in 0..3 {
        let ph = i as f32 * TAU / 3.0;
        a += line((v - 0.5 - 0.42 * (k * along + ph).sin()) * bw, 0.1 * s);
        b += line((v - 0.5 - 0.42 * (ph - k * along).sin()) * bw, 0.1 * s);
    }
    let ink = rules * (1.0 - corner) + rosette * corner + a * band;
    (clamp01(ink), clamp01(b * band))
}

/// The sheet at `(x, y)` from its middle (down positive), `half` its half
/// size, `s` its unit, unrolled down to `front` (from its middle): how far
/// outside what shows of it (negative inside); inside, its tone (its
/// fibres, its edges aged, darker under the roll; on the roll, the roll's
/// shading), outside, its shadow's darkness; and its border's two inks.
pub fn paper_at(x: f32, y: f32, half: (f32, f32), s: f32, front: f32) -> [f32; 4] {
    let s = s.max(0.5);
    let corner = 0.8 * s;
    let sheet = round_box((x, y), half, corner);
    let left = clamp01((half.1 - front) / (2.0 * half.1));
    let rho = ROLL * s * left.sqrt();
    let shown = sheet.max(y - front);
    let roll = if rho > 0.05 {
        (x.abs() - half.0 - 0.2 * s).max((y - front - rho).abs() - rho)
    } else {
        1e9
    };
    let d = shown.min(roll);
    if d > 0.0 {
        // Its shadow (and the roll's), a little down from it.
        let ds = round_box((x, y - 1.2 * s), half, corner).max(y - 1.2 * s - front - 2.0 * rho);
        return [d, 0.5 * (-ds.max(0.0) / (2.5 * s)).exp(), 0.0, 0.0];
    }
    if roll < shown {
        // The roll: a cylinder lying across, lit from above.
        let v = clamp01((y - front) / (2.0 * rho).max(1e-3));
        let tone = 0.55 + 0.33 * (PI * v).sin() + 0.25 * line(v - 0.3, 0.12) - 0.2 * v * v;
        return [d, tone, 0.0, 0.0];
    }
    let e = -sheet;
    let n = 0.6 * vnoise(x / (4.0 * s), y / (4.0 * s))
        + 0.4 * vnoise(x / (1.3 * s) + 31.0, y / (1.3 * s) - 17.0);
    let fibre = vnoise(x / (7.0 * s) + 5.0, y / (0.7 * s));
    let aged = 1.0 - smoothstep(0.0, 18.0 * s, e);
    let mut tone = 1.0 - 0.06 * n - 0.03 * fibre - 0.16 * aged * aged - 0.05 * aged * n;
    if rho > 0.05 {
        tone *= 1.0 - 0.35 * ((y - front) / (1.5 * rho)).exp();
    }
    let (a, b) = border(x, y, half, e, s);
    [d, tone, a, b]
}

/// A rubber stamp's frame at `(x, y)` from its middle (along it, not
/// turned), `half` its half size, `s` its unit: its ink (two rounded
/// rules, worn in places) and its specks (where the ink did not take,
/// over what is inside it).
pub fn stamp_at(x: f32, y: f32, half: (f32, f32), s: f32) -> [f32; 2] {
    let s = s.max(0.5);
    let d = round_box((x, y), half, 1.2 * s);
    let outer = clamp01(0.5 - ((d + 0.3 * s).abs() - 0.3 * s));
    let inner_half = (half.0 - 1.2 * s, half.1 - 1.2 * s);
    let di = round_box((x, y), inner_half, 0.6 * s);
    let inner = clamp01(0.5 - ((di + 0.12 * s).abs() - 0.12 * s));
    let grain = 0.65 * vnoise(x / (0.8 * s) + 3.0, y / (0.8 * s) - 5.0)
        + 0.35 * vnoise(x / (0.25 * s), y / (0.25 * s));
    let worn = smoothstep(0.22, 0.38, grain);
    let ink = outer.max(inner) * (0.35 + 0.65 * worn);
    let speck = (1.0 - smoothstep(0.12, 0.2, grain)) * clamp01(0.5 - d);
    [ink, speck]
}

/// How far `(x, y)` is from the signature written as far as `drawn` along
/// it (0 to 1), in its own units
/// ([`crate::core::singing::signature::signature_point`]): through its points
/// ([`pen_point`]) up to `drawn` of the way, the last step part of the
/// way; far without it.
pub fn signature_at(x: f32, y: f32, drawn: f32) -> f32 {
    if drawn <= 0.0 {
        return 1e9;
    }
    let step = |d: f32, a: (f32, f32), b: (f32, f32)| {
        let (px, py, bx, by) = (x - a.0, y - a.1, b.0 - a.0, b.1 - a.1);
        let h = clamp01((px * bx + py * by) / (bx * bx + by * by).max(1e-8));
        let (qx, qy) = (px - bx * h, py - by * h);
        d.min((qx * qx + qy * qy).sqrt())
    };
    let n = drawn.min(1.0) * PEN as f32;
    let whole = n.floor() as u32;
    let mut d = f32::MAX;
    for i in 0..whole.min(PEN) {
        d = step(d, pen_point(i), pen_point(i + 1));
    }
    let part = n - whole as f32;
    if whole < PEN && part > 0.0 {
        let (a, b) = (pen_point(whole), pen_point(whole + 1));
        let end = (a.0 + (b.0 - a.0) * part, a.1 + (b.1 - a.1) * part);
        d = step(d, a, end);
    }
    d
}

/// The functions above in WGSL, and the shapes drawn with them: for the
/// stage's shader (which gives them `vnoise`, `line`, `round_box`,
/// `over_plus` and `out_rgb`); the signature's points as a table.
pub fn wgsl() -> String {
    let points: Vec<String> = (0..=PEN)
        .map(|i| {
            let (x, y) = pen_point(i);
            format!("vec2<f32>({x:?}, {y:?})")
        })
        .collect();
    let n = PEN + 1;
    format!(
        "const ROLL: f32 = {ROLL:?};\nconst PEN: u32 = {PEN}u;\nvar<private> PEN_POINTS: array<vec2<f32>, {n}> = array<vec2<f32>, {n}>({});\n{SHADER}",
        points.join(", ")
    )
}

const SHADER: &str = r"
fn turn_about(p: vec2<f32>, c: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let co = cos(a);
    let d = p - c;
    return c + vec2<f32>(co * d.x - s * d.y, s * d.x + co * d.y);
}

fn border(x: f32, y: f32, half: vec2<f32>, e: f32, s: f32) -> vec2<f32> {
    let m = 2.0 * s;
    let bw = 3.0 * s;
    let v = (e - m) / bw;
    let rules = line(e - m, 0.16 * s)
        + line(e - m - bw, 0.16 * s)
        + 0.6 * line(e - m - bw - 0.8 * s, 0.1 * s);
    let cx = half.x - m - bw * 0.5;
    let cy = half.y - m - bw * 0.5;
    let qx = abs(x) - cx;
    let qy = abs(y) - cy;
    let rr = sqrt(qx * qx + qy * qy);
    let th = atan2(qy, qx);
    let big = bw * 0.95;
    var rosette = line(rr - big, 0.14 * s) + line(rr, 0.3 * s);
    for (var j = 0u; j < 3u; j = j + 1u) {
        let petal = big * (0.55 + 0.33 * sin(8.0 * th + f32(j) * 6.2831855 / 3.0));
        rosette = rosette + line(rr - petal, 0.1 * s);
    }
    let corner = 1.0 - smoothstep(big * 1.02, big * 1.12, rr);
    var along = x;
    if half.x - abs(x) < half.y - abs(y) {
        along = y;
    }
    let k = 6.2831855 / (3.4 * s);
    let band = smoothstep(0.0, 0.08, v) * (1.0 - smoothstep(0.92, 1.0, v)) * (1.0 - corner);
    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let ph = f32(i) * 6.2831855 / 3.0;
        a = a + line((v - 0.5 - 0.42 * sin(k * along + ph)) * bw, 0.1 * s);
        b = b + line((v - 0.5 - 0.42 * sin(ph - k * along)) * bw, 0.1 * s);
    }
    let ink = rules * (1.0 - corner) + rosette * corner + a * band;
    return vec2<f32>(clamp(ink, 0.0, 1.0), clamp(b * band, 0.0, 1.0));
}

fn paper_at(x: f32, y: f32, half: vec2<f32>, s0: f32, front: f32) -> vec4<f32> {
    let s = max(s0, 0.5);
    let corner = 0.8 * s;
    let sheet = round_box(vec2<f32>(x, y), half, corner);
    let left = clamp((half.y - front) / (2.0 * half.y), 0.0, 1.0);
    let rho = ROLL * s * sqrt(left);
    let shown = max(sheet, y - front);
    var roll = 1e9;
    if rho > 0.05 {
        roll = max(abs(x) - half.x - 0.2 * s, abs(y - front - rho) - rho);
    }
    let d = min(shown, roll);
    if d > 0.0 {
        let ds = max(round_box(vec2<f32>(x, y - 1.2 * s), half, corner), y - 1.2 * s - front - 2.0 * rho);
        return vec4<f32>(d, 0.5 * exp(-max(ds, 0.0) / (2.5 * s)), 0.0, 0.0);
    }
    if roll < shown {
        let v = clamp((y - front) / max(2.0 * rho, 1e-3), 0.0, 1.0);
        let tone = 0.55 + 0.33 * sin(3.1415927 * v) + 0.25 * line(v - 0.3, 0.12) - 0.2 * v * v;
        return vec4<f32>(d, tone, 0.0, 0.0);
    }
    let e = -sheet;
    let n = 0.6 * vnoise(x / (4.0 * s), y / (4.0 * s))
        + 0.4 * vnoise(x / (1.3 * s) + 31.0, y / (1.3 * s) - 17.0);
    let fibre = vnoise(x / (7.0 * s) + 5.0, y / (0.7 * s));
    let aged = 1.0 - smoothstep(0.0, 18.0 * s, e);
    var tone = 1.0 - 0.06 * n - 0.03 * fibre - 0.16 * aged * aged - 0.05 * aged * n;
    if rho > 0.05 {
        tone = tone * (1.0 - 0.35 * exp((y - front) / (1.5 * rho)));
    }
    let ink = border(x, y, half, e, s);
    return vec4<f32>(d, tone, ink.x, ink.y);
}

fn stamp_at(x: f32, y: f32, half: vec2<f32>, s0: f32) -> vec2<f32> {
    let s = max(s0, 0.5);
    let d = round_box(vec2<f32>(x, y), half, 1.2 * s);
    let outer = clamp(0.5 - (abs(d + 0.3 * s) - 0.3 * s), 0.0, 1.0);
    let di = round_box(vec2<f32>(x, y), half - vec2<f32>(1.2 * s), 0.6 * s);
    let inner = clamp(0.5 - (abs(di + 0.12 * s) - 0.12 * s), 0.0, 1.0);
    let grain = 0.65 * vnoise(x / (0.8 * s) + 3.0, y / (0.8 * s) - 5.0)
        + 0.35 * vnoise(x / (0.25 * s), y / (0.25 * s));
    let worn = smoothstep(0.22, 0.38, grain);
    let ink = max(outer, inner) * (0.35 + 0.65 * worn);
    let speck = (1.0 - smoothstep(0.12, 0.2, grain)) * clamp(0.5 - d, 0.0, 1.0);
    return vec2<f32>(ink, speck);
}

fn signature_step(d: f32, q: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = q - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return min(d, length(pa - ba * h));
}

fn signature_at(x: f32, y: f32, drawn: f32) -> f32 {
    if drawn <= 0.0 {
        return 1e9;
    }
    let q = vec2<f32>(x, y);
    let n = min(drawn, 1.0) * f32(PEN);
    let whole = u32(floor(n));
    var d = 3.4028235e38;
    for (var i = 0u; i < min(whole, PEN); i = i + 1u) {
        d = signature_step(d, q, PEN_POINTS[i], PEN_POINTS[i + 1u]);
    }
    let part = n - f32(whole);
    if whole < PEN && part > 0.0 {
        let a = PEN_POINTS[whole];
        d = signature_step(d, q, a, a + (PEN_POINTS[whole + 1u] - a) * part);
    }
    return d;
}

// The diploma's paper: the sheet (`shape.xy` its middle, `shape.zw` its
// half size, `params.z` its unit, unrolled to `params.y` from its middle;
// in `color`, its border in `color2` and `uv.rgb`), a rubber stamp's
// frame (`shape` as the sheet's, turned `params.y`, `params.z` its unit;
// in `color`, its specks in `color2`) and a signature (`shape.xy` its
// middle, `shape.z` its size, written to `params.y`, `params.z` its pen's
// half width; in `color`).
fn paper_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let q = p - o.shape.xy;
    if k == K_PAPER {
        let g = paper_at(q.x, q.y, o.shape.zw, o.params.z, o.params.y);
        let cover = clamp(0.5 - g.x, 0.0, 1.0);
        var tone = g.y;
        var shadow = 0.0;
        if g.x > 0.0 {
            tone = 0.85;
            shadow = g.y;
        }
        let inked = mix(mix(o.color.rgb * tone, o.color2.rgb, g.z * o.color2.a), o.uv.rgb, g.w * o.uv.a);
        let a = o.color.a;
        return vec4<f32>(out_rgb(inked) * cover * a, (cover + shadow * (1.0 - cover)) * a);
    }
    if k == K_STAMP {
        let l = turn_about(p, o.shape.xy, -o.params.y) - o.shape.xy;
        let g = stamp_at(l.x, l.y, o.shape.zw, o.params.z);
        let ia = g.x * o.color.a;
        let sa = g.y * o.color2.a * (1.0 - ia);
        return vec4<f32>(out_rgb(o.color.rgb) * ia + out_rgb(o.color2.rgb) * sa, ia + sa);
    }
    let size = max(o.shape.z, 1.0);
    let d = signature_at(q.x / size, q.y / size, o.params.y) * size - o.params.z;
    let cover = clamp(0.5 - d, 0.0, 1.0);
    return over_plus(o.color.rgb, cover * o.color.a, vec3<f32>(0.0));
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::singing::signature::signature_point;

    /// The sheet shows down to its front, a roll under it while it
    /// unrolls, gone once it is open; its border inked near its edges, not
    /// in its middle; a shadow round it.
    #[test]
    fn the_paper_unrolls_with_its_border() {
        let half = (500.0, 350.0);
        let s = 7.0;
        let open = |x, y| paper_at(x, y, half, s, half.1);
        assert!(open(0.0, 0.0)[0] < -300.0);
        assert!(open(0.0, 0.0)[2] < 0.01 && open(0.0, 0.0)[3] < 0.01);
        // On the outer rule, near the top edge.
        assert!(open(0.0, -350.0 + 2.0 * s)[2] > 0.9);
        // Outside: a shadow, fading off.
        let near = open(0.0, 352.0);
        let far = open(0.0, 400.0);
        assert!(near[0] > 0.0 && near[1] > far[1] && far[1] < 0.1);
        // Half unrolled: shown above the front, the roll under it, nothing
        // below.
        let half_open = |y| paper_at(0.0, y, half, s, 0.0);
        assert!(half_open(-10.0)[0] < 0.0);
        assert!(half_open(5.0)[0] < 0.0 && half_open(5.0)[2] == 0.0);
        assert!(half_open(100.0)[0] > 50.0);
        // Darker just above the roll than far above it.
        assert!(half_open(-3.0)[1] < half_open(-200.0)[1]);
    }

    /// The stamp's frame is inked round its edge, its middle clear of ink.
    #[test]
    fn the_stamp_is_a_frame() {
        let half = (120.0, 40.0);
        let edge: f32 = (-110..110)
            .step_by(4)
            .map(|x| stamp_at(x as f32, -40.0 + 1.5, half, 7.0)[0])
            .sum::<f32>();
        assert!(edge > 20.0, "{edge}");
        assert!(stamp_at(0.0, 0.0, half, 7.0)[0] < 0.01);
        assert_eq!(stamp_at(0.0, 80.0, half, 7.0), [0.0, 0.0]);
    }

    /// The signature starts at its start and is written on as `drawn`
    /// grows; nothing of it before.
    #[test]
    fn the_signature_is_written_on() {
        let (x0, y0) = signature_point(0.0);
        assert!(signature_at(x0, y0, 0.05) < 1e-4);
        assert!(signature_at(x0, y0, 0.0) > 1e8);
        let (x1, y1) = signature_point(0.9);
        assert!(signature_at(x1, y1, 0.3) > 0.3);
        assert!(signature_at(x1, y1, 1.0) < 1e-3);
        // Within its box.
        for k in 0..=50 {
            let (x, y) = signature_point(k as f32 / 50.0);
            assert!(x.abs() < 1.2 && y > -0.6 && y < 0.45, "{k}: {x} {y}");
        }
    }

    #[test]
    fn turning_about_a_point() {
        let (x, y) = turn_about((2.0, 1.0), (1.0, 1.0), PI / 2.0);
        assert!((x - 1.0).abs() < 1e-6 && (y - 2.0).abs() < 1e-6);
    }
}
