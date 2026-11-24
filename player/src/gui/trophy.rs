//! The result screen's shapes on the GPU, in the stage's shader: a gem
//! (a cut stone seen from the front, its facets lit from the top left, a
//! glint sweeping over it), a medal (a struck disc, its rim bevelled, a
//! star embossed on its face; or a star badge standing for one) with a
//! sheen sweeping across, and a burst of soft rays turning behind them,
//! light. Each is a signed distance and a tone in the shape's own units
//! (its size 1); the processor gives the GPU where each is, its colours
//! and where its glint is, nothing more.
//!
//! These functions are the shader's reference ([`wgsl`]), held to it on
//! the GPU (`super::shader_check`).

/// The gem's outline, as its edges' outward normals and offsets (its size
/// 1, its middle at 0): a flat table on top, the crown sloping out to the
/// girdle, the pavilion down to its point.
const GEM_EDGES: [(f32, f32, f32); 5] = [
    (0.0, -1.0, 0.825),
    (HALF, -HALF, 0.972_271),
    (-HALF, -HALF, 0.972_271),
    (0.768_221_3, 0.640_184_4, 0.528_152),
    (-0.768_221_3, 0.640_184_4, 0.528_152),
];
/// A diagonal's share across (and down): one over the root of two.
const HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// Where the girdle (the widest line, between the crown and the pavilion)
/// is, down from the gem's middle; and the pavilion's height under it.
const GIRDLE: f32 = -0.375;
const PAVILION: f32 = 1.2;

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A line's light `d` off it, `w` wide.
fn line(d: f32, w: f32) -> f32 {
    (-(d / w) * (d / w)).exp()
}

/// The gem at `(x, y)` (its size 1, its middle at 0, down positive): how
/// far outside it (negative inside), and how bright its facet is there
/// (its facets' edges lighter), and how far a glint at `glint` (across it
/// from its top left, -1.5 to 1.5) lights it.
pub fn gem_at(x: f32, y: f32, glint: f32) -> [f32; 3] {
    let d = GEM_EDGES
        .iter()
        .map(|&(nx, ny, c)| nx * x + ny * y - c)
        .fold(f32::NEG_INFINITY, f32::max)
        - 0.03;
    let crown = 1.0 - smoothstep(GIRDLE - 0.02, GIRDLE + 0.02, y);
    // The crown: its table brightest, the left lit, the right in shade.
    let side = smoothstep(-0.32, -0.28, x);
    let right = smoothstep(0.28, 0.32, x);
    let top = mix(mix(0.95, 1.2, side), 0.62, right);
    // The pavilion: four facets down to its point, each lit its own way.
    let w = (1.0 - (y - GIRDLE) / PAVILION).max(1e-3);
    let u = x / w;
    let low = mix(
        mix(0.82, 0.55, smoothstep(-0.42, -0.38, u)),
        mix(0.72, 0.42, smoothstep(0.38, 0.42, u)),
        smoothstep(-0.02, 0.02, u),
    );
    let facet = mix(low, top, crown);
    let edges = line(y - GIRDLE, 0.035) * 0.55
        + crown * (line(x.abs() - 0.3, 0.03) * 0.45)
        + (1.0 - crown) * (line((u.abs() - 0.4) * w, 0.03) + line(u * w, 0.03)) * 0.4;
    let sweep = line((x * 0.6 + y * 0.8) - glint, 0.22) * 0.75;
    [d, facet + edges, sweep]
}

fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// A five-pointed star's signed distance at `(x, y)` (up positive), its
/// points `r` out, its inner corners `rf` of the way.
pub fn star(x: f32, y: f32, r: f32, rf: f32) -> f32 {
    let k1 = (0.809_017, -0.587_785_25);
    let k2 = (-k1.0, k1.1);
    let (mut px, mut py) = (x.abs(), y);
    let d1 = 2.0 * (k1.0 * px + k1.1 * py).max(0.0);
    px -= d1 * k1.0;
    py -= d1 * k1.1;
    let d2 = 2.0 * (k2.0 * px + k2.1 * py).max(0.0);
    px -= d2 * k2.0;
    py -= d2 * k2.1;
    px = px.abs();
    py -= r;
    let (bx, by) = (rf * -k1.1, rf * k1.0 - 1.0);
    let h = ((px * bx + py * by) / (bx * bx + by * by)).clamp(0.0, r);
    let (qx, qy) = (px - bx * h, py - by * h);
    (qx * qx + qy * qy).sqrt() * sign(py * bx - px * by)
}

/// A medal at `(x, y)` (its radius 1, its middle at 0, down positive;
/// `badge`: a star badge instead of a disc): how far outside it (negative
/// inside), its tone (0 its deep colour, 1 its light one) and a sheen at
/// `sheen` (across it from its top left, -1.5 to 1.5).
pub fn medal_at(x: f32, y: f32, badge: bool, sheen: f32) -> [f32; 3] {
    let shine = line(x * HALF + y * HALF - sheen, 0.2) * 0.55;
    if badge {
        let d = star(x, -y, 0.98, 0.5) - 0.07;
        let rim = line(d + 0.07, 0.035) * 0.6;
        return [d, 0.72 - 0.22 * y + rim, shine];
    }
    let r = (x * x + y * y).sqrt();
    let d = r - 1.0;
    // The rim, bevelled: its outer slope lit at the top left, its inner one
    // in shade there; a groove inside it.
    let lit = (-0.6 * x - 0.8 * y) / r.max(1e-4);
    let slope = smoothstep(0.89, 0.93, r) * 2.0 - 1.0;
    let rim = 0.62 + 0.38 * lit * slope;
    let face = 0.7 - 0.22 * y;
    let mut tone = mix(face, rim, smoothstep(0.8, 0.84, r));
    tone *= 1.0 - 0.45 * line(r - 0.82, 0.02);
    // The star struck on its face, its edge pressed in.
    let s = star(x, -y, 0.56, 0.45);
    tone = mix(tone, 0.95 - 0.18 * y, 1.0 - smoothstep(-0.015, 0.015, s));
    tone *= 1.0 - 0.35 * line(s, 0.025);
    [d, tone, shine]
}

/// A burst of rays at `(x, y)` from its middle, `radius` its reach, `n`
/// of them, turned by `turn` (radians of each ray's cycle): its light, 0
/// to 1, falling off outwards.
pub fn rays_at(x: f32, y: f32, radius: f32, n: f32, turn: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let a = if r < 0.5 { 0.0 } else { y.atan2(x) };
    let s = 0.5 + 0.5 * (n * a + turn).cos();
    let fall = (-r / radius.max(1.0) * 2.5).exp();
    (0.15 + 0.85 * s * s * s) * fall
}

/// The functions above in WGSL, and the shapes drawn with them: for the
/// stage's shader (which gives them `glow` and `over_plus`).
pub fn wgsl() -> String {
    let e = |i: usize| {
        let (nx, ny, c) = GEM_EDGES[i];
        format!("    d = max(d, {nx:?} * x + {ny:?} * y - {c:?});\n")
    };
    format!(
        "const GIRDLE: f32 = {GIRDLE:?};\nconst PAVILION: f32 = {PAVILION:?};\nfn gem_edges(x: f32, y: f32) -> f32 {{\n    var d = -1e9;\n{}{}{}{}{}    return d - 0.03;\n}}\n{SHADER}",
        e(0),
        e(1),
        e(2),
        e(3),
        e(4)
    )
}

const SHADER: &str = r"
fn line(d: f32, w: f32) -> f32 {
    return exp(-(d / w) * (d / w));
}

fn gem_at(x: f32, y: f32, glint: f32) -> vec3<f32> {
    let d = gem_edges(x, y);
    let crown = 1.0 - smoothstep(GIRDLE - 0.02, GIRDLE + 0.02, y);
    let side = smoothstep(-0.32, -0.28, x);
    let right = smoothstep(0.28, 0.32, x);
    let top = mix(mix(0.95, 1.2, side), 0.62, right);
    let w = max(1.0 - (y - GIRDLE) / PAVILION, 1e-3);
    let u = x / w;
    let low = mix(
        mix(0.82, 0.55, smoothstep(-0.42, -0.38, u)),
        mix(0.72, 0.42, smoothstep(0.38, 0.42, u)),
        smoothstep(-0.02, 0.02, u),
    );
    let facet = mix(low, top, crown);
    let edges = line(y - GIRDLE, 0.035) * 0.55
        + crown * (line(abs(x) - 0.3, 0.03) * 0.45)
        + (1.0 - crown) * (line((abs(u) - 0.4) * w, 0.03) + line(u * w, 0.03)) * 0.4;
    let sweep = line((x * 0.6 + y * 0.8) - glint, 0.22) * 0.75;
    return vec3<f32>(d, facet + edges, sweep);
}

fn star(x: f32, y: f32, r: f32, rf: f32) -> f32 {
    let k1 = vec2<f32>(0.809017, -0.58778525);
    let k2 = vec2<f32>(-k1.x, k1.y);
    var p = vec2<f32>(abs(x), y);
    p = p - 2.0 * max(dot(k1, p), 0.0) * k1;
    p = p - 2.0 * max(dot(k2, p), 0.0) * k2;
    p.x = abs(p.x);
    p.y = p.y - r;
    let ba = vec2<f32>(rf * -k1.y, rf * k1.x - 1.0);
    let h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

fn medal_at(x: f32, y: f32, badge: bool, sheen: f32) -> vec3<f32> {
    let shine = line(x * 0.7071068 + y * 0.7071068 - sheen, 0.2) * 0.55;
    if badge {
        let d = star(x, -y, 0.98, 0.5) - 0.07;
        let rim = line(d + 0.07, 0.035) * 0.6;
        return vec3<f32>(d, 0.72 - 0.22 * y + rim, shine);
    }
    let r = sqrt(x * x + y * y);
    let d = r - 1.0;
    let lit = (-0.6 * x - 0.8 * y) / max(r, 1e-4);
    let slope = smoothstep(0.89, 0.93, r) * 2.0 - 1.0;
    let rim = 0.62 + 0.38 * lit * slope;
    let face = 0.7 - 0.22 * y;
    var tone = mix(face, rim, smoothstep(0.8, 0.84, r));
    tone = tone * (1.0 - 0.45 * line(r - 0.82, 0.02));
    let s = star(x, -y, 0.56, 0.45);
    tone = mix(tone, 0.95 - 0.18 * y, 1.0 - smoothstep(-0.015, 0.015, s));
    tone = tone * (1.0 - 0.35 * line(s, 0.025));
    return vec3<f32>(d, tone, shine);
}

fn rays_at(x: f32, y: f32, radius: f32, n: f32, turn: f32) -> f32 {
    let r = sqrt(x * x + y * y);
    var a = 0.0;
    if r >= 0.5 {
        a = atan2(y, x);
    }
    let s = 0.5 + 0.5 * cos(n * a + turn);
    let fall = exp(-r / max(radius, 1.0) * 2.5);
    return (0.15 + 0.85 * s * s * s) * fall;
}

// The result's shapes (each glowing round it, not over it): a gem
// (`shape.xy` its middle, `shape.z` its size, in `color`, glowing
// `params.y` in `color2`, its glint at `params.z`), a
// medal (`shape.xy` its middle, `shape.z` its radius, `shape.w` 1 for a
// star badge; from `color2` deep to `color` light, glowing `params.y`, its
// sheen at `params.z`) and rays (round `shape.xy`, `shape.z` their reach,
// `shape.w` of them, turned `params.y`, in `color`: light).
fn trophy(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let size = max(o.shape.z, 0.5);
    let l = (p - o.shape.xy) / size;
    if k == K_RAYS {
        let q = p - o.shape.xy;
        let light = rays_at(q.x, q.y, o.shape.z, o.shape.w, o.params.y);
        return vec4<f32>(out_rgb(o.color.rgb) * light * o.color.a, 0.0);
    }
    if k == K_GEM {
        let g = gem_at(l.x, l.y, o.params.z);
        let d = g.x * size;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let col = min(o.color.rgb * g.y + vec3<f32>(g.z), vec3<f32>(1.0));
        let halo = glow(d, o.params.y) * o.color2.a * (1.0 - cover);
        return over_plus(col, cover * o.color.a, o.color2.rgb * halo);
    }
    let m = medal_at(l.x, l.y, o.shape.w > 0.5, o.params.z);
    let d = m.x * size;
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let col = min(mix(o.color2.rgb, o.color.rgb, m.y) + vec3<f32>(m.z), vec3<f32>(1.0));
    let halo = glow(d, o.params.y) * o.color.a * 0.6 * (1.0 - cover);
    return over_plus(col, cover * o.color.a, o.color.rgb * halo);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    /// The gem's outline goes through its corners; its table is its
    /// brightest facet, its point below its middle.
    #[test]
    fn the_gem_is_cut_as_drawn() {
        let corners = [(-0.55, -0.825), (0.55, -0.825), (1.0, -0.375), (0.0, 0.825)];
        for (x, y) in corners {
            let d = gem_at(x, y, -9.0)[0];
            assert!((d + 0.03).abs() < 2e-3, "({x}, {y}): {d}");
        }
        assert!(gem_at(0.0, 0.0, -9.0)[0] < -0.3);
        assert!(gem_at(0.0, -0.6, -9.0)[1] > gem_at(0.7, -0.5, -9.0)[1]);
        // The glint lights where it is, not elsewhere.
        assert!(gem_at(0.0, 0.0, 0.0)[2] > 0.7 && gem_at(0.0, 0.0, 1.4)[2] < 0.01);
    }

    #[test]
    fn the_medal_and_its_star() {
        assert!((medal_at(1.0, 0.0, false, 9.0)[0]).abs() < 1e-6);
        // The star's points out, its corners in.
        assert!(star(0.0, 0.56, 0.56, 0.45).abs() < 1e-4);
        assert!(star(0.0, 0.0, 0.56, 0.45) < -0.15);
        assert!(star(0.0, -0.4, 0.56, 0.45) > 0.0);
        // The badge is a star: out between its points.
        assert!(medal_at(0.0, -0.9, true, 9.0)[0] < 0.0);
        assert!(medal_at(0.6, -0.6, true, 9.0)[0] > 0.0);
        // Its rim lit at the top left, in shade at the bottom right.
        let at = |a: f32| medal_at(0.95 * a.cos(), 0.95 * a.sin(), false, 9.0)[1];
        assert!(at(-2.2) > at(0.9));
    }

    #[test]
    fn the_rays_turn_and_fall_off() {
        let a = rays_at(100.0, 0.0, 300.0, 12.0, 0.0);
        let b = rays_at(100.0, 0.0, 300.0, 12.0, std::f32::consts::PI);
        assert!(a > 0.4 && b < a * 0.3, "{a} {b}");
        assert!(rays_at(600.0, 0.0, 300.0, 12.0, 0.0) < a * 0.3);
        assert!(rays_at(0.0, 0.0, 300.0, 12.0, 0.0).is_finite());
    }
}
