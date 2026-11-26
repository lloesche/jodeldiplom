//! The effects in the stage's shader: every effect's number among them
//! (`FX_<IDENT>`: its look in `looks`) and its pipelines' groups
//! (`G_SHAPES_<IDENT>`, `G_LYRIC_<IDENT>`), the WGSL of the effects a
//! pipeline draws (its own; none for the stage's own groups) with the
//! helpers they share (`fx_heat`, ...: [`super::helpers`]), and the two
//! calls the stage's shader makes into them: `fx_shape` (an effect's
//! shape, in its own pipeline) and `treat` (text in a lyric treatment, in
//! its own; plain text as it is). So each pipeline carries only its own
//! effect's code: a GPU compiles and runs only that.

use super::Fx;
use crate::gui::stage::gpu::{wgsl_name as name, Group};

/// The effects' WGSL for the stage's shader, with `effects`' code (each
/// group's pipeline its own effect's; the shaders' checks every one).
pub fn wgsl(effects: &[Fx]) -> String {
    let mut out = format!("const FX_COUNT: u32 = {}u;\n", Fx::COUNT);
    if !effects.is_empty() {
        out += &super::helpers::wgsl();
    }
    let (mut shapes, mut treats) = (String::new(), String::new());
    for fx in Fx::ALL {
        let s = fx.spec();
        let ident = name(s.ident);
        out += &format!("const FX_{ident}: u32 = {}u;\n", fx.index());
        let own = effects.contains(&fx);
        if !s.kinds.is_empty() {
            let g = Group::Shapes(fx).code();
            out += &format!("const G_SHAPES_{ident}: u32 = {g}u;\n");
            if own {
                shapes += &format!(
                    "    if GROUP == G_SHAPES_{ident} {{\n        return {}_shape(o, p, k);\n    }}\n",
                    s.ident
                );
            }
        }
        if s.lyric.is_some() {
            let g = Group::Lyric(fx).code();
            out += &format!("const G_LYRIC_{ident}: u32 = {g}u;\n");
            if own {
                treats += &format!(
                    "    if GROUP == G_LYRIC_{ident} {{\n        return lyric_{}(col, a, p, o.quad, screen.time, k, looks[FX_{ident}]);\n    }}\n",
                    s.ident
                );
            }
        }
        if let (Some(code), true) = (s.wgsl, own) {
            out += &code();
        }
    }
    format!(
        "{out}
// An effect's shape, in its effect's pipeline.
fn fx_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {{
{shapes}    return vec4<f32>(0.0);
}}

// Text's colour `col` and coverage `a` at `p` in its pipeline's lyric
// treatment, its strength eased as `strain.xyz` says; plain, as it is.
fn treat(col: vec3<f32>, a: f32, o: Out, p: vec2<f32>) -> vec4<f32> {{
    let k = shine_at(o.strain.xyz, screen.time).x;
{treats}    return vec4<f32>(col, a);
}}
"
    )
}
