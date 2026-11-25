//! Flames blazing off the lane's top and bottom edges, licking up and
//! flickering, white at their root to red at their tips: light, drawn
//! where it blooms ([`crate::gui::bloom`]), as strong as the rows hold
//! them (Amazing's first).
//!
//! [`blaze`] is the shader's reference ([`wgsl`]), held to it on the GPU
//! ([`REFERENCE`]).

use super::{At, Fx, Layer, Live, Out, Reference, Slot, Spec};
use crate::gui::glory::vnoise;
use crate::gui::rainbow::{mix3, smoothstep};
use crate::gui::stage_gpu::Instance;

/// The flames along an edge from `shape[0]` to `shape[2]` at `shape[1]`,
/// outward `shape[3]` (1 down, -1 up), `params[1]` high, `params[2]`
/// strong ([`blaze`]): light.
pub const BLAZE: f32 = Fx::Flames.kind(0);

/// The flames off the lane's edges.
pub const SPEC: Spec = Spec {
    name: "flames",
    ident: "flames",
    at: &[Layer::Lane],
    draw,
    kinds: &["blaze"],
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
    let high = pill * 1.5;
    for (edge, outward) in [(cy - hy, -1.0), (cy + hy, 1.0)] {
        let (y0, y1) = if outward < 0.0 {
            (edge - high * 1.4, edge + high * 0.3)
        } else {
            (edge - high * 0.3, edge + high * 1.4)
        };
        out.shapes.push(Instance {
            quad: [cx - hx, y0, cx + hx, y1],
            shape: [cx - hx, edge, cx + hx, outward],
            color: [1.0; 4],
            params: [BLAZE, high, s, 0.0],
            ..Instance::default()
        });
    }
}

/// The flames `d` pixels off the lane's edge (outwards; negative: into the
/// lane) at `x` along it, `high` their height, at `time`, `k` their
/// strength: their light (bright, for the bloom), licking up and
/// flickering, white at their root to red at their tips.
pub fn blaze(x: f32, d: f32, high: f32, time: f32, k: f32) -> [f32; 3] {
    let h = high.max(1.0);
    let n = 0.65 * vnoise(x / h * 4.0, d / h * 1.2 + time * 2.6)
        + 0.35 * vnoise(x / h * 9.0 + 7.0, d / h * 2.4 + time * 4.1);
    let reach = h * (0.08 + 1.4 * n * n);
    let flame = if d >= 0.0 {
        (1.0 - d / reach).clamp(0.0, 1.0).powf(1.3)
    } else {
        0.5 * (d / (h * 0.06)).exp()
    };
    let heat = flame * (0.45 + 0.85 * n);
    let red = [1.0, 0.18, 0.04];
    let orange = [1.0, 0.55, 0.1];
    let yellow = [1.0, 0.9, 0.45];
    let c = mix3(red, orange, smoothstep(0.15, 0.5, heat));
    let c = mix3(c, yellow, smoothstep(0.5, 0.85, heat));
    let c = mix3(c, [1.0; 3], smoothstep(0.85, 1.1, heat));
    c.map(|v| v * heat * 1.1 * k)
}

/// [`blaze`] in WGSL and its shapes (`flames_shape`), for the stage's
/// shader (which gives them `vnoise`).
pub fn wgsl() -> String {
    SHADER.to_owned()
}

const SHADER: &str = r"
fn blaze(x: f32, d: f32, high: f32, time: f32, k: f32) -> vec3<f32> {
    let h = max(high, 1.0);
    let n = 0.65 * vnoise(x / h * 4.0, d / h * 1.2 + time * 2.6)
        + 0.35 * vnoise(x / h * 9.0 + 7.0, d / h * 2.4 + time * 4.1);
    let reach = h * (0.08 + 1.4 * n * n);
    var flame = 0.5 * exp(d / (h * 0.06));
    if d >= 0.0 {
        flame = pow(clamp(1.0 - d / reach, 0.0, 1.0), 1.3);
    }
    let heat = flame * (0.45 + 0.85 * n);
    var c = mix(vec3<f32>(1.0, 0.18, 0.04), vec3<f32>(1.0, 0.55, 0.1), smoothstep(0.15, 0.5, heat));
    c = mix(c, vec3<f32>(1.0, 0.9, 0.45), smoothstep(0.5, 0.85, heat));
    c = mix(c, vec3<f32>(1.0), smoothstep(0.85, 1.1, heat));
    return c * heat * 1.1 * k;
}

// The flames along an edge: `shape.x` to `shape.z` at `shape.y`, outward
// `shape.w`, `params.y` high, `params.z` strong; light.
fn flames_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let d = (p.y - o.shape.y) * o.shape.w;
    let along = smoothstep(o.shape.x, o.shape.x + o.params.y, p.x)
        * (1.0 - smoothstep(o.shape.z - o.params.y, o.shape.z, p.x));
    let l = blaze(p.x, d, o.params.y, screen.time, o.params.z) * along;
    return vec4<f32>(out_rgb(l), 0.0);
}
";

/// The flames at points along and off an edge, through their flicker:
/// the shader's `blaze` as [`blaze`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        (0..360)
            .map(|k| {
                let fr = |m: f32| (k as f32 * m).fract();
                [
                    200.0 + fr(0.19) * 80.0 - 40.5,
                    fr(0.31) * 60.0 - 20.0,
                    3.0 + fr(0.37) * 4.0,
                    0.0,
                ]
            })
            .collect()
    },
    ins: 1,
    outs: 1,
    body: "
    let s = ins[i];
    outs[i] = vec4<f32>(blaze(s.x * 3.0, s.y, 40.0, s.z, 0.8), 0.0);",
    cpu: |s| {
        let [r, g, b] = blaze(s[0][0] * 3.0, s[0][1], 40.0, s[0][2], 0.8);
        vec![[r, g, b, 0.0]]
    },
    near: (3e-3, 3e-3),
    shows: |o| o[0][0] + o[0][1] + o[0][2] > 0.05,
    least: 50,
};
