//! The passes' shaders as the GPU takes them: a pass's own pipeline
//! ([`pass_shader`]: the shared helpers, its code, the lyric band's guard
//! and a triangle over the target), and all of them at once for the shader
//! check ([`check_shader`]: the picture under them made up, the same
//! here and in [`check_src`]).

use crate::math::{v2, v3, V2, V3};
use crate::pass::{Ctx, Pass, Src};
use crate::passes::{self, PASSES};

/// What a pass does at the end, on both sides: its mix over the picture
/// under it, the lyric band given back its share of the picture and held
/// under the ceiling (or what the picture had there), the light kept
/// between none and four.
pub const FINISH: &str = r"
fn lk_band(y: f32) -> f32 {
    return smoothstep(u.band.x - 0.04, u.band.x, y) * (1.0 - smoothstep(u.band.y, u.band.y + 0.04, y));
}
fn lk_finish(c: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let s = lk_src(uv);
    var o = mix(s, c, u.mix.x);
    let w = lk_band(uv.y);
    o = mix(o, s, w * u.band.z);
    let ceiling = max(u.band.w, lk_luma(s));
    let l = lk_luma(o);
    o = o * mix(1.0, min(1.0, ceiling / max(l, 1e-4)), w);
    return clamp(o, vec3<f32>(0.0), vec3<f32>(4.0));
}
";

/// [`FINISH`] on the processor: `c` the pass's colour at `uv`.
pub fn finish(u: &Ctx, s: &dyn Src, c: V3, uv: V2) -> V3 {
    use crate::math::{luma, mix, mix3, smoothstep};
    let src = s.src(uv);
    let mut o = mix3(src, c, u.mix[0]);
    let w = smoothstep(u.band[0] - 0.04, u.band[0], uv.y)
        * (1.0 - smoothstep(u.band[1], u.band[1] + 0.04, uv.y));
    o = mix3(o, src, w * u.band[2]);
    let ceiling = u.band[3].max(luma(src));
    let l = luma(o);
    o = o * mix(1.0, (ceiling / l.max(1e-4)).min(1.0), w);
    o.clamp(0.0, 4.0)
}

/// Every pass family's helpers the others may use.
fn shared() -> String {
    passes::SHARED.concat()
}

/// The reads of the picture under a pass and the frame before, for a
/// target of linear light (sRGB textures: read as linear, the passes'
/// maths in encoded values) or not.
const READS: &str = r"
@group(0) @binding(0) var<uniform> u: LookU;
@group(0) @binding(1) var lk_src_t: texture_2d<f32>;
@group(0) @binding(2) var lk_prev_t: texture_2d<f32>;
@group(0) @binding(3) var lk_soft: sampler;
fn lk_encode(v: vec3<f32>) -> vec3<f32> {
    let x = max(v, vec3<f32>(0.0));
    return select(12.92 * x, 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x > vec3<f32>(0.0031308));
}
fn lk_decode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(x / 12.92, pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x > vec3<f32>(0.04045));
}
fn lk_src(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(lk_src_t, lk_soft, uv, 0.0).xyz;
    return select(c, lk_encode(c), LINEAR > 0.5);
}
fn lk_prev(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(lk_prev_t, lk_soft, uv, 0.0).xyz;
    return select(c, lk_encode(c), LINEAR > 0.5);
}
struct LkOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs(@builtin(vertex_index) i: u32) -> LkOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: LkOut;
    o.pos = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
";

/// `pass`'s pipeline's shader, for a target of linear light (`linear`) or
/// not: its fragment entry `fs`, its vertex entry `vs`; its uniform at
/// binding 0, the picture under it at 1, the frame before at 2, a
/// smooth sampler held to the edge at 3.
pub fn pass_shader(pass: &Pass, linear: bool) -> String {
    format!(
        "const LINEAR: f32 = {};\n{}{}{READS}{}{}{FINISH}
@fragment
fn fs(o: LkOut) -> @location(0) vec4<f32> {{
    let c = lk_finish(look_{}(o.uv), o.uv);
    return vec4<f32>(select(c, lk_decode(c), LINEAR > 0.5), 1.0);
}}
",
        if linear { "1.0" } else { "0.0" },
        crate::math::WGSL,
        crate::pass::CTX_WGSL,
        shared(),
        pass.wgsl,
        pass.name
    )
}

/// The made-up picture the shader check reads: soft waves of colour and
/// a bright disc with a sharp edge (for the passes that look for edges).
pub const CHECK_SRC: &str = r"
fn lk_src(uv: vec2<f32>) -> vec3<f32> {
    let a = 0.5 + 0.5 * sin(vec3<f32>(7.1 * uv.x + 2.3 * uv.y, 5.3 * uv.y - 3.1 * uv.x + 1.0, 9.7 * uv.x * uv.y + 2.0));
    let d = length(uv - vec2<f32>(0.55, 0.45));
    return mix(a, vec3<f32>(0.95, 0.9, 0.8), smoothstep(0.205, 0.2, d));
}
fn lk_prev(uv: vec2<f32>) -> vec3<f32> {
    return 0.5 + 0.5 * cos(vec3<f32>(4.3 * uv.x - 1.7 * uv.y, 3.9 * uv.y + 2.2 * uv.x, 6.1 * uv.x + 0.5));
}
";

/// [`CHECK_SRC`] on the processor.
pub struct CheckSrc;

impl Src for CheckSrc {
    fn src(&self, uv: V2) -> V3 {
        let a = v3(
            7.1 * uv.x + 2.3 * uv.y,
            5.3 * uv.y - 3.1 * uv.x + 1.0,
            9.7 * uv.x * uv.y + 2.0,
        )
        .map(|x| 0.5 + 0.5 * x.sin());
        let d = (uv - v2(0.55, 0.45)).length();
        crate::math::mix3(
            a,
            v3(0.95, 0.9, 0.8),
            crate::math::smoothstep(0.205, 0.2, d),
        )
    }

    fn prev(&self, uv: V2) -> V3 {
        v3(
            4.3 * uv.x - 1.7 * uv.y,
            3.9 * uv.y + 2.2 * uv.x,
            6.1 * uv.x + 0.5,
        )
        .map(|x| 0.5 + 0.5 * x.cos())
    }
}

/// Every pass at once for the shader check, over [`CHECK_SRC`]: `u` a
/// private the check sets from its samples, and `lk_check(p, uv)` pass `p`
/// (its place in [`PASSES`]) finished at `uv`.
pub fn check_shader() -> String {
    let cases: String = PASSES
        .iter()
        .enumerate()
        .map(|(i, p)| format!("        case {i}u: {{ c = look_{}(uv); }}\n", p.name))
        .collect();
    let code: String = PASSES.iter().map(|p| p.wgsl).collect();
    format!(
        "{}{}var<private> u: LookU;\n{CHECK_SRC}{}{code}{FINISH}
fn lk_check(p: u32, uv: vec2<f32>) -> vec3<f32> {{
    var c = vec3<f32>(0.0);
    switch p {{
{cases}        default: {{ c = vec3<f32>(0.0); }}
    }}
    return lk_finish(c, uv);
}}
",
        crate::math::WGSL,
        crate::pass::CTX_WGSL,
        shared(),
    )
}

/// The check's body (over `i`, a sample: its first vec4 the pass's place
/// and the point, the rest its [`Ctx`]): the pass's colour, finished.
pub fn check_body() -> String {
    let n = Ctx::VEC4S;
    let per = n + 1;
    let data: Vec<String> = (0..crate::pass::DATA)
        .map(|j| format!("ins[b + {}u]", 9 + j))
        .collect();
    format!(
        "    let b = i * {per}u + 1u;
    let head = ins[i * {per}u];
    u = LookU(ins[b], ins[b + 1u], ins[b + 2u], ins[b + 3u], ins[b + 4u], ins[b + 5u], ins[b + 6u], ins[b + 7u], ins[b + 8u], array<vec4<f32>, 16>({}));
    let c = lk_check(u32(head.x), head.yz);
    outs[i] = vec4<f32>(c, 0.0);
",
        data.join(", ")
    )
}

/// The shader check's samples: for each pass, `n` points with its knobs
/// anywhere in their ranges, the time, the band and the mix too, each
/// sample's vec4s (its place and point first, then its [`Ctx`]), the same
/// every run.
pub fn check_samples(n: usize) -> Vec<(usize, V2, Ctx)> {
    let mut out = Vec::new();
    for (p, pass) in PASSES.iter().enumerate() {
        for j in 0..n {
            let r = |k: u32| crate::math::hash2(p as i32, j as i32, k);
            let uv = v2(0.02 + 0.96 * r(1), 0.02 + 0.96 * r(2));
            let mut ctx = Ctx {
                size: [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0],
                time: [r(3) * 200.0, (r(4) * 65536.0).floor(), 1.0 / 60.0, 0.0],
                ..Ctx::default()
            };
            for (i, k) in pass.knobs.iter().enumerate() {
                ctx.k[i] = k.clamp(k.lo + (k.hi - k.lo) * r(10 + i as u32));
            }
            if j % 3 == 0 {
                ctx.band = [0.6, 0.8, pass.relief, crate::guard::CEILING];
            }
            ctx.mix[0] = if j % 4 == 1 { 0.5 } else { 1.0 };
            let feed_signals = crate::demo::signals_at(f64::from(ctx.time[0]));
            if let Some(fill) = pass.data {
                ctx.data = fill(&crate::drive::Feed {
                    t: f64::from(ctx.time[0]),
                    signals: &feed_signals,
                    curves: None,
                });
            }
            out.push((p, uv, ctx));
        }
    }
    out
}

/// A check sample as the shader takes it: its place and point, then its
/// numbers.
pub fn sample_vec4s(p: usize, uv: V2, ctx: &Ctx) -> Vec<[f32; 4]> {
    let mut v = vec![[p as f32, uv.x, uv.y, 0.0]];
    v.extend(ctx.as_vec4s());
    v
}

/// A check sample worked out on the processor.
pub fn check_cpu(p: usize, uv: V2, ctx: &Ctx) -> V3 {
    let pass = PASSES[p];
    let c = (pass.cpu)(ctx, &CheckSrc, uv);
    finish(ctx, &CheckSrc, c, uv)
}
