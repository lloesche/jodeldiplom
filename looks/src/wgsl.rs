//! The passes' shaders as the GPU takes them: a pass's own pipeline
//! ([`pass_shader`]: the helpers every pass has ([`helpers`]), its code,
//! the lyric band's guard ([`FINISH`]) and a triangle over the target).

use crate::math::{V2, V3};
use crate::pass::{Ctx, Pass, Src};

/// What a pass does at the end, on both sides: its mix over the picture
/// under it, the lyric band given back its share of the picture and held
/// under the ceiling of light (luminance, linear: or what the picture had
/// there), the light kept
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
    let ceiling = max(u.band.w, lk_lum(s));
    let l = lk_lum(o);
    o = o * mix(1.0, pow(min(1.0, ceiling / max(l, 1e-6)), 1.0 / 2.2), w);
    return clamp(o, vec3<f32>(0.0), vec3<f32>(4.0));
}
";

/// [`FINISH`] on the processor: `c` the pass's colour at `uv`.
pub fn finish(u: &Ctx, s: &dyn Src, c: V3, uv: V2) -> V3 {
    use crate::math::{lum, mix, mix3, smoothstep};
    let src = s.src(uv);
    let mut o = mix3(src, c, u.mix[0]);
    let w = smoothstep(u.band[0] - 0.04, u.band[0], uv.y)
        * (1.0 - smoothstep(u.band[1], u.band[1] + 0.04, uv.y));
    o = mix3(o, src, w * u.band[2]);
    let ceiling = u.band[3].max(lum(src));
    let l = lum(o);
    o = o * mix(1.0, (ceiling / l.max(1e-6)).min(1.0).powf(1.0 / 2.2), w);
    o.clamp(0.0, 4.0)
}

/// What every pass's code reads besides `u` (declared by the shader) and
/// the picture's reads (`lk_src`, `lk_prev`): the maths' helpers, its
/// numbers' struct (`LookU`) and accessors, and the helpers the families
/// share.
pub fn helpers() -> String {
    [
        crate::math::wgsl(),
        crate::pass::ctx_wgsl(),
        crate::passes::common::wgsl(),
    ]
    .concat()
}

/// The reads of the picture under a pass and the frame before, for a
/// target of linear light (sRGB textures: read as linear, the passes'
/// maths in encoded values) or not.
const READS: &str = r"
@group(0) @binding(0) var<uniform> u: LookU;
@group(0) @binding(1) var lk_src_t: texture_2d<f32>;
@group(0) @binding(2) var lk_prev_t: texture_2d<f32>;
@group(0) @binding(3) var lk_soft: sampler;
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
        "const LINEAR: f32 = {};\n{}{READS}{}{FINISH}
@fragment
fn fs(o: LkOut) -> @location(0) vec4<f32> {{
    let c = lk_finish(look_{}(o.uv), o.uv);
    return vec4<f32>(select(c, lk_decode(c), LINEAR > 0.5), 1.0);
}}
",
        if linear { "1.0" } else { "0.0" },
        helpers(),
        pass.wgsl,
        pass.name
    )
}
