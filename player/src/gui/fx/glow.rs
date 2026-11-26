//! A warm glow round the lane, breathing, and pulsing with the beat: light,
//! drawn where it blooms ([`crate::gui::bloom`]), as strong as the rows
//! hold it (Great's first).
//!
//! With the music it pulses ([`pulse`]): it flashes up on each beat and
//! more on a bar's first, dimming between them; as sure as the beat grid
//! is, where it is really unsure it breathes on the clock as it does with
//! no music. Its look (`x`, [`Spec::params`]) is how much it pulses (1 as
//! made, 0 not at all).
//!
//! [`glow_at`] is the shader's reference ([`wgsl`]), held to it on the GPU
//! ([`REFERENCE`]).

use super::helpers::smoothstep;
use super::{At, Fx, Layer, Live, Music, Out, Reference, Slot, Spec};
use crate::gui::stage_gpu::Instance;

/// The glow round a lane (`shape` its middle and half size) in `color`,
/// `params[1]` its reach, `params[2]` its strength, `color2` how sure the
/// beat grid is and the beat's kick ([`pulse`]): light.
pub const GLOW: f32 = Fx::Glow.kind(0);
/// Its colour, sRGB.
const WARM: [f32; 4] = [1.0, 0.74, 0.3, 1.0];
/// How bright it is between the beats, on the beat (the grid sure).
const REST: f32 = 0.55;
/// How much brighter on each beat (its pulse at its height), and on a
/// bar's first beat.
const BEAT: f32 = 0.9;
const ONE: f32 = 1.5;
/// Where out to its reach the glow begins to fade out, and where it is
/// gone (shares of its reach).
const GONE: (f32, f32) = (0.4, 0.95);

/// The glow round the lane.
pub const SPEC: Spec = Spec {
    name: "lane glow",
    ident: "glow",
    at: &[Layer::Lane],
    draw,
    kinds: &["glow"],
    wgsl: Some(wgsl),
    params: [1.0, 0.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    ..Spec::NONE
};

fn draw(at: &At, live: &Live, _: &mut Slot, out: &mut Out) {
    let (
        &At::Lane {
            lane,
            pill,
            area,
            band,
            ..
        },
        s,
    ) = (at, live.strength)
    else {
        return;
    };
    if s <= 0.01 {
        return;
    }
    let ((cx, cy), (hx, hy)) = (lane.middle, lane.half);
    // Never over the HUD's band (no higher than the bulbs light) nor into
    // the lyric band: it reaches no further than either, and is gone
    // before it ([`glow_at`]), its quad holding all of it.
    let hud = super::bulbs::lit_top(lane.rect(), super::far_corner(area));
    let reach = (pill * 1.6)
        .min(cy - hy - hud)
        .min(band.0 - cy - hy)
        .max(1.0);
    out.shapes.push(Instance {
        quad: [
            cx - hx - reach,
            cy - hy - reach,
            cx + hx + reach,
            cy + hy + reach,
        ],
        shape: [cx, cy, hx, hy],
        color: WARM,
        color2: pulse(&live.music, live.look),
        params: [GLOW, reach, 0.12 * s, 0.0],
        ..Instance::default()
    });
}

/// How the music `m` makes the glow pulse, `p` its look (`x` how much):
/// how sure the grid is, and the kick it has on the beat (none where it
/// is unsure), as [`glow_at`] takes them.
pub fn pulse(m: &Music, p: [f32; 4]) -> [f32; 4] {
    [m.sure, p[0].max(0.0) * m.kick(BEAT, ONE), 0.0, 0.0]
}

/// How bright the glow round the lane is `d` pixels outside it (negative
/// inside), `wide` its reach, at `time`, `k` its strength, with the beat's
/// `sure` and `kick` ([`pulse`]): soft, breathing on the clock as the grid
/// is unsure, as sure dim between the beats and flashing up on them; gone
/// a little before its reach ([`GONE`]), so nothing of it is cut off.
pub fn glow_at(d: f32, wide: f32, time: f32, k: f32, (sure, kick): (f32, f32)) -> f32 {
    let t = d.max(0.0) / wide.max(1.0);
    let inside = (d.min(0.0) / (wide * 0.5).max(1.0)).exp();
    let breath = 0.78 + 0.22 * (time * 2.6).sin();
    let bright = breath + (REST - breath) * sure + kick;
    let fade = 1.0 - smoothstep(GONE.0, GONE.1, t);
    (-t * t * 2.5).exp() * fade * inside * bright * k
}

/// [`glow_at`] in WGSL and its shapes (`glow_shape`), for the stage's
/// shader.
pub fn wgsl() -> String {
    format!(
        "const GLOW_REST: f32 = {REST:?};\nconst GLOW_GONE: vec2<f32> = vec2<f32>({:?}, {:?});\n{SHADER}",
        GONE.0, GONE.1
    )
}

const SHADER: &str = r"
fn glow_at(d: f32, wide: f32, time: f32, k: f32, beat: vec2<f32>) -> f32 {
    let t = max(d, 0.0) / max(wide, 1.0);
    let inside = exp(min(d, 0.0) / max(wide * 0.5, 1.0));
    let breath = 0.78 + 0.22 * sin(time * 2.6);
    let bright = breath + (GLOW_REST - breath) * beat.x + beat.y;
    let fade = 1.0 - smoothstep(GLOW_GONE.x, GLOW_GONE.y, t);
    return exp(-t * t * 2.5) * fade * inside * bright * k;
}

fn glow_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let d = round_box(p - o.shape.xy, o.shape.zw, o.shape.w);
    let l = glow_at(d, o.params.y, screen.time, o.params.z, o.color2.xy);
    return vec4<f32>(out_rgb(o.color.rgb) * l, 0.0);
}
";

/// The glow at distances in and out of the lane, through its breathing
/// and its pulses: the shader's `glow_at` as [`glow_at`].
pub const REFERENCE: Reference = Reference {
    samples: || {
        (0..360)
            .flat_map(|k| {
                let fr = |m: f32| (k as f32 * m).fract();
                [
                    [
                        fr(0.31) * 60.0 - 20.0,
                        10.0 + fr(0.53) * 30.0,
                        3.0 + fr(0.37) * 4.0,
                        fr(0.71),
                    ],
                    [fr(0.23), fr(0.47) * (BEAT + ONE), 0.0, 0.0],
                ]
            })
            .collect()
    },
    ins: 2,
    outs: 1,
    body: "
    let s = ins[i * 2u];
    outs[i] = vec4<f32>(glow_at(s.x, s.y, s.z, s.w, ins[i * 2u + 1u].xy), 0.0, 0.0, 0.0);",
    cpu: |s| {
        let [d, wide, time, k] = s[0];
        vec![[glow_at(d, wide, time, k, (s[1][0], s[1][1])), 0.0, 0.0, 0.0]]
    },
    near: (3e-3, 3e-3),
    shows: |o| o[0][0] > 0.01,
    least: 100,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Sure of the grid it flashes up on each beat and more on a bar's
    /// first, far brighter than between the beats; unsure, or its look
    /// off, it breathes as with no music.
    #[test]
    fn it_flashes_on_the_beat() {
        let look = SPEC.params;
        let at = |beats: f64, sure: f64| {
            let [s, kick, _, _] = pulse(&Music::steady(beats, 120.0, 4, sure, 0.0), look);
            glow_at(0.0, 40.0, 1.0, 1.0, (s, kick))
        };
        let (one, beat, between) = (at(8.0, 1.0), at(9.0, 1.0), at(9.5, 1.0));
        assert!(one > beat && beat > 2.0 * between, "{one} {beat} {between}");
        let still = glow_at(0.0, 40.0, 1.0, 1.0, (0.0, 0.0));
        assert_eq!(at(8.0, 0.0), still);
        let [s, kick, _, _] = pulse(&Music::steady(8.0, 120.0, 4, 1.0, 0.0), [0.0; 4]);
        assert_eq!(kick, 0.0);
        assert!((glow_at(0.0, 40.0, 1.0, 1.0, (s, kick)) - REST).abs() < 1e-6);
    }
}
