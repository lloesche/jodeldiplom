//! Every pass's shader held to its maths on the processor: on the GPU,
//! all of them at once in a compute pass over a made-up picture (the same
//! here as WGSL and on the processor), at samples with every knob
//! anywhere in its range, and each pass's own pipeline made there (both
//! skipped, with a note, without a GPU); and every shader parsed and
//! validated by naga, on any machine.

use jodel_looks::math::{v2, v3, V2, V3};
use jodel_looks::pass::{Ctx, Src};
use jodel_looks::passes::{self, PASSES};
use jodel_looks::wgsl;
use jodel_testkit::gpu::{gpu, near, run, Input};

/// The made-up picture the check reads: soft waves of colour and a
/// bright disc with a sharp edge (for the passes that look for edges).
const CHECK_SRC: &str = r"
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
struct CheckSrc;

impl Src for CheckSrc {
    fn src(&self, uv: V2) -> V3 {
        let a = v3(
            7.1 * uv.x + 2.3 * uv.y,
            5.3 * uv.y - 3.1 * uv.x + 1.0,
            9.7 * uv.x * uv.y + 2.0,
        )
        .map(|x| 0.5 + 0.5 * x.sin());
        let d = (uv - v2(0.55, 0.45)).length();
        jodel_looks::math::mix3(
            a,
            v3(0.95, 0.9, 0.8),
            jodel_looks::math::smoothstep(0.205, 0.2, d),
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

/// A check sample as the shader reads it (`LkSample`): its pass's place
/// and the point, then the pass's numbers as its uniform has them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Sample {
    head: [f32; 4],
    ctx: Ctx,
}

impl Input for Sample {
    const WGSL: &'static str = "LkSample";
}

/// Every pass at once over [`CHECK_SRC`]: `u` a private the check sets
/// from its sample, and `lk_check(p, uv)` pass `p` (its place in
/// [`PASSES`]) finished at `uv`.
fn check_shader() -> String {
    let cases: String = PASSES
        .iter()
        .enumerate()
        .map(|(i, p)| format!("        case {i}u: {{ c = look_{}(uv); }}\n", p.name))
        .collect();
    let code: String = PASSES.iter().map(|p| p.wgsl).collect();
    format!(
        "{}var<private> u: LookU;
struct LkSample {{
    head: vec4<f32>,
    u: LookU,
}};
{CHECK_SRC}{code}{}
fn lk_check(p: u32, uv: vec2<f32>) -> vec3<f32> {{
    var c = vec3<f32>(0.0);
    switch p {{
{cases}        default: {{ c = vec3<f32>(0.0); }}
    }}
    return lk_finish(c, uv);
}}
",
        wgsl::helpers(),
        wgsl::FINISH,
    )
}

/// The check's body over sample `i`: its pass's colour, finished.
const CHECK_BODY: &str = "    let s = ins[i];
    u = s.u;
    outs[i] = vec4<f32>(lk_check(u32(s.head.x), s.head.yz), 0.0);
";

/// The check's samples: for each pass, `n` points with its knobs anywhere
/// in their ranges, the time, the band and the mix too, the same every
/// run.
fn check_samples(n: usize) -> Vec<Sample> {
    let mut out = Vec::new();
    for (p, pass) in PASSES.iter().enumerate() {
        for j in 0..n {
            let r = |k: u32| jodel_looks::math::hash2(p as i32, j as i32, k);
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
                ctx.band = [0.6, 0.8, pass.relief, jodel_looks::guard::CEILING];
            }
            ctx.mix[0] = if j % 4 == 1 { 0.5 } else { 1.0 };
            let signals = jodel_looks::demo::signals_at(f64::from(ctx.time[0]));
            if let Some(fill) = pass.data {
                ctx.data = fill(&jodel_looks::Feed {
                    t: f64::from(ctx.time[0]),
                    signals: &signals,
                    curves: None,
                });
            }
            out.push(Sample {
                head: [p as f32, uv.x, uv.y, 0.0],
                ctx,
            });
        }
    }
    out
}

/// A check sample worked out on the processor.
fn check_cpu(s: &Sample) -> V3 {
    let pass = PASSES[s.head[0] as usize];
    let uv = v2(s.head[1], s.head[2]);
    let c = (pass.cpu)(&s.ctx, &CheckSrc, uv);
    wgsl::finish(&s.ctx, &CheckSrc, c, uv)
}

/// The passes whose picture is made of cells: a sample right on a cell's
/// edge may fall in another cell on the GPU.
const MADE_OF_CELLS: &[&str] = &[
    "rain",
    "snow",
    "petals",
    "dust",
    "starfield",
    "confetti",
    "film_damage",
    "bokeh",
    "stained_glass",
    "mosaic",
    "cross_stitch",
    "teletext",
    "cutout",
    "halftone",
    "halftone_cmyk",
    "ben_day",
    "stipple",
    "pixelate",
    "palette",
    "spectrum_ring",
    "bars",
];

/// `code` (named `name`) parsed and validated by naga, as wgpu takes it.
fn valid(name: &str, code: &str) {
    let module = naga::front::wgsl::parse_str(code)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
}

/// Every look pass's shader as its processor's maths
/// ([`jodel_looks::Pass::cpu`]), finished as its pipeline finishes it (its
/// mix, the lyric band's guard), over the check's made-up picture, at
/// samples with every knob anywhere in its range: within a few
/// thousandths, a pass made of cells allowed two samples in its 48 that
/// fall in another cell on the GPU (a sample right on a cell's edge).
#[test]
fn every_look_pass_is_its_reference() {
    for name in MADE_OF_CELLS {
        assert!(passes::named(name).is_some(), "{name} is no pass");
    }
    let Some(gpu) = gpu() else {
        return;
    };
    const EACH: usize = 48;
    let samples = check_samples(EACH);
    let outs = run(
        &gpu,
        &check_shader(),
        CHECK_BODY,
        &samples,
        (samples.len(), 1),
    );
    let mut off = vec![Vec::new(); PASSES.len()];
    for (s, got) in samples.iter().zip(&outs) {
        let want = check_cpu(s);
        let close = [want.x, want.y, want.z]
            .iter()
            .zip(got)
            .all(|(w, g)| near(*g, *w, 4e-3, 2e-3));
        if !close {
            let at = (s.head[1], s.head[2]);
            off[s.head[0] as usize].push(format!("at {at:?} GPU {got:?}, processor {want:?}"));
        }
    }
    for (pass, off) in PASSES.iter().zip(&off) {
        let allowed = if MADE_OF_CELLS.contains(&pass.name) {
            2
        } else {
            0
        };
        assert!(off.len() <= allowed, "{}: {off:#?}", pass.name);
    }
}

/// Every look pass's own pipeline's shader compiles on the GPU, for a
/// target of linear light and not.
#[test]
fn every_look_pass_s_pipeline_compiles() {
    let Some((device, _queue)) = gpu() else {
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    for pass in PASSES {
        for linear in [false, true] {
            let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(pass.name),
                source: wgpu::ShaderSource::Wgsl(wgsl::pass_shader(pass, linear).into()),
            });
        }
    }
    let fault = pollster::block_on(scope.pop());
    assert!(
        fault.is_none(),
        "{}",
        fault.map(|f| f.to_string()).unwrap_or_default()
    );
}

/// Every look pass's own pipeline's shader (for a target of linear light
/// and not) and the check's shader as it runs, parsed and validated, on
/// any machine: without a GPU too, the passes' WGSL is held.
#[test]
fn every_look_shader_is_valid_wgsl() {
    for pass in PASSES {
        for linear in [false, true] {
            valid(pass.name, &wgsl::pass_shader(pass, linear));
        }
    }
    valid(
        "the check",
        &jodel_testkit::gpu::code::<Sample>(&check_shader(), CHECK_BODY),
    );
}
