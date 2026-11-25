//! A warm glow round the lane, breathing: light, drawn where it blooms
//! ([`crate::gui::bloom`]), as strong as the rows hold it (Great's first).
//!
//! [`glow_at`] is the shader's reference ([`wgsl`]), held to it on the GPU
//! ([`REFERENCE`]).

use super::{At, Fx, Layer, Live, Out, Reference, Slot, Spec};
use crate::gui::stage_gpu::Instance;

/// The glow round a lane (`shape` its middle and half size) in `color`,
/// `params[1]` its reach, `params[2]` its strength: light.
pub const GLOW: f32 = Fx::Glow.kind(0);
/// Its colour, sRGB.
const WARM: [f32; 4] = [1.0, 0.74, 0.3, 1.0];

/// The glow round the lane.
pub const SPEC: Spec = Spec {
    name: "lane glow",
    ident: "glow",
    at: &[Layer::Lane],
    draw,
    kinds: &["glow"],
    wgsl: Some(wgsl),
    reference: Some(REFERENCE),
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let (&At::Lane { lane, pill }, s) = (at, live.strength) else {
        return;
    };
    if s <= 0.01 {
        return;
    }
    let ((cx, cy), (hx, hy)) = (lane.middle, lane.half);
    let reach = pill * 1.6;
    out.shapes.push(Instance {
        quad: [
            cx - hx - reach,
            cy - hy - reach,
            cx + hx + reach,
            cy + hy + reach,
        ],
        shape: [cx, cy, hx, hy],
        color: WARM,
        params: [GLOW, reach, 0.12 * s, 0.0],
        ..Instance::default()
    });
}

/// How bright the glow round the lane is `d` pixels outside it (negative
/// inside), `wide` its reach, at `time`, `k` its strength: soft, breathing.
pub fn glow_at(d: f32, wide: f32, time: f32, k: f32) -> f32 {
    let t = d.max(0.0) / wide.max(1.0);
    let inside = (d.min(0.0) / (wide * 0.5).max(1.0)).exp();
    (-t * t * 2.5).exp() * inside * (0.78 + 0.22 * (time * 2.6).sin()) * k
}

/// [`glow_at`] in WGSL and its shapes (`glow_shape`), for the stage's
/// shader.
pub fn wgsl() -> String {
    SHADER.to_owned()
}

const SHADER: &str = r"
fn glow_at(d: f32, wide: f32, time: f32, k: f32) -> f32 {
    let t = max(d, 0.0) / max(wide, 1.0);
    let inside = exp(min(d, 0.0) / max(wide * 0.5, 1.0));
    return exp(-t * t * 2.5) * inside * (0.78 + 0.22 * sin(time * 2.6)) * k;
}

fn glow_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let d = round_box(p - o.shape.xy, o.shape.zw, o.shape.w);
    let l = glow_at(d, o.params.y, screen.time, o.params.z);
    return vec4<f32>(out_rgb(o.color.rgb) * l, 0.0);
}
";

/// The glow at distances in and out of the lane, through its breathing:
/// the shader's `glow_at` as [`glow_at`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        (0..360)
            .map(|k| {
                let fr = |m: f32| (k as f32 * m).fract();
                [
                    fr(0.31) * 60.0 - 20.0,
                    10.0 + fr(0.53) * 30.0,
                    3.0 + fr(0.37) * 4.0,
                    fr(0.71),
                ]
            })
            .collect()
    },
    ins: 1,
    outs: 1,
    body: "
    let s = ins[i];
    outs[i] = vec4<f32>(glow_at(s.x, s.y, s.z, s.w), 0.0, 0.0, 0.0);",
    cpu: |s| vec![[glow_at(s[0][0], s[0][1], s[0][2], s[0][3]), 0.0, 0.0, 0.0]],
    near: (3e-3, 3e-3),
    shows: |o| o[0][0] > 0.01,
    least: 100,
};
