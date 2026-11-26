//! The shaders held to their processor's references, on the GPU: the
//! particles' (every quad of every kind of burst, from its motion's
//! probes: where, how large, how turned, its colour and how much it shows,
//! [`sprite`]), the stage's liquid (its bulge and wobble, its overfilling,
//! the front's waves and slosh, the shimmer, [`super::liquid`]), a tube's
//! nearest point (how far, where along, which side, [`super::snake`]),
//! every effect's maths ([`super::fx::Spec::reference`]), the glory (the
//! meter, the ring, the marks, [`super::glory`]), the diploma's shapes (a
//! star sticker, a wax seal, the rays, [`super::seal`]; its paper, a
//! rubber stamp, a signature, [`super::paper`]), the backdrop's maths
//! ([`super::backdrop::look`]) and its smooth read of a texture (a cubic
//! B-spline of its texels), each worked out by the shader's own
//! functions in a compute pass over the shader as the window composes it,
//! at sample points, against the processor's; and every pipeline the
//! stage, its light and the bursts make, made. Skipped (with a note)
//! without a GPU.

use super::fx::Fx;
use super::particles::{self, sprite, Spray, MOTIONS, QUADS};
use super::rainbow::Glow;
use super::sparks::TAU;
use super::stage_gpu::{Blend, Group, StagePass};
use super::{glory, liquid, paper, seal};

/// A device on the window's GPU, none without one.
pub(in crate::gui) fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let _turn = jodel_gpu::setting_up();
    let adapter = jodel_gpu::chosen().adapter.clone();
    match adapter.and_then(|a| super::present::open(a, wgpu::Features::empty())) {
        Ok((_, device, queue)) => Some((device, queue)),
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            None
        }
    }
}

/// `body` (WGSL over `i`, the sample, reading `ins` and writing `outs`)
/// after `source`, run on the GPU for `n` samples, `per` vec4s out each.
pub(in crate::gui) fn run(
    gpu: &(wgpu::Device, wgpu::Queue),
    source: &str,
    body: &str,
    ins: &[[f32; 4]],
    (n, per): (usize, usize),
) -> Vec<[f32; 4]> {
    run_reading(gpu, (source, body), ins, (n, per), None)
}

/// As [`run`], `source` and `body` reading a texture (`group(0)`: it at
/// binding 0, a linear sampler at 1) where one is given.
fn run_reading(
    (device, queue): &(wgpu::Device, wgpu::Queue),
    (source, body): (&str, &str),
    ins: &[[f32; 4]],
    (n, per): (usize, usize),
    texture: Option<&wgpu::TextureView>,
) -> Vec<[f32; 4]> {
    let code = format!(
        "{source}
@group(1) @binding(0) var<storage, read> ins: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> outs: array<vec4<f32>>;
@compute @workgroup_size(1)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
{body}
}}
"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jodeldiplom shader check"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jodeldiplom shader check"),
        layout: None,
        module: &module,
        entry_point: Some("check"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes: &[u8] = bytemuck::cast_slice(ins);
    let size = (n * per * 16) as u64;
    let buffer = |size: u64, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom shader check"),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::BufferUsages as U;
    let input = buffer(bytes.len() as u64, U::STORAGE | U::COPY_DST);
    let output = buffer(size, U::STORAGE | U::COPY_SRC);
    let back = buffer(size, U::MAP_READ | U::COPY_DST);
    queue.write_buffer(&input, 0, bytes);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jodeldiplom shader check"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let read = texture.map(|view| {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom shader check"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        if let Some(read) = &read {
            pass.set_bind_group(0, read, &[]);
        }
        pass.set_bind_group(1, &bind, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &back, 0, size);
    queue.submit([encoder.finish()]);
    let slice = back.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU done");
    rx.recv().unwrap().expect("mapped");
    let mapped = slice.get_mapped_range().expect("read back");
    bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec()
}

/// Whether `a` is `b` within `abs` (and a share `rel` of it).
pub(in crate::gui) fn near(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    (a - b).abs() <= abs + rel * b.abs()
}

/// Every pipeline made as the window makes them (the stage's, a group
/// each, over the frame and added into the light; the bursts' likewise)
/// without a fault.
#[test]
fn every_group_compiles() {
    let Some((device, _queue)) = gpu() else {
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    for (format, blend) in [
        (wgpu::TextureFormat::Bgra8Unorm, Blend::Over),
        (super::bloom::LIGHT, Blend::Add),
    ] {
        let _stage = StagePass::with(&device, (format, blend), &Group::all());
        let _bursts = particles::ParticlePass::new(&device, format, blend);
    }
    let _light = StagePass::with(&device, (super::bloom::LIGHT, Blend::Add), &Group::light());
    let fault = pollster::block_on(scope.pop());
    assert!(fault.is_none(), "{fault:?}");
    assert!(Group::all().len() > 5 + 3, "the effects' groups too");
}

/// Every effect's shader as its processor's ([`super::fx::Reference`]),
/// at its samples, in the stage's shader.
#[test]
fn every_effect_s_shader_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let source = super::stage_gpu::every_effect_shader();
    let mut held = 0;
    for fx in Fx::ALL {
        let Some(r) = fx.spec().reference else {
            continue;
        };
        held += 1;
        let ins = (r.samples)();
        let n = ins.len() / r.ins;
        let outs = run(&gpu, &source, r.body, &ins, (n, r.outs));
        let mut shown = 0;
        for k in 0..n {
            let sample = &ins[k * r.ins..(k + 1) * r.ins];
            let want = (r.cpu)(sample);
            let got = &outs[k * r.outs..(k + 1) * r.outs];
            for (g, w) in got.iter().zip(&want) {
                for c in 0..4 {
                    assert!(
                        near(g[c], w[c], r.near.0, r.near.1),
                        "{fx:?} sample {k} ({sample:?}): GPU {got:?}, processor {want:?}"
                    );
                }
            }
            shown += usize::from((r.shows)(&want));
        }
        assert!(shown >= r.least, "{fx:?}: {shown} of {n} show");
    }
    assert!(held >= 7, "{held}");
}

/// The music's uniform as the shaders read it, field by field by the
/// names the effects use (`music.beat.w` the bar's pulse, `music.tempo.y`
/// how sure the grid is, `music.count.x` the beats counted, a stem at its
/// `STEM_` place, ...): each the [`super::fx::Music`] field it says.
#[test]
fn the_music_s_uniform_reads_as_its_fields() {
    use super::fx::Music;
    use jodel_rhythm::stems::Stem;
    let Some(gpu) = gpu() else {
        return;
    };
    let m = Music {
        stems: [0.11, 0.22, 0.33, 0.44],
        level: 0.55,
        progress: 0.66,
        melody: 61.0,
        sung: 62.0,
        singing: 1.0,
        section: 0.77,
        ..Music::steady(13.375, 97.0, 3, 0.88, 0.0)
    };
    let ins = m.uniform().to_vec();
    let source = super::fx::music::wgsl();
    let body = "
    let m = Music(ins[0], ins[1], ins[2], ins[3], ins[4]);
    outs[0] = vec4<f32>(m.beat.x, m.beat.y, m.beat.z, m.beat.w);
    outs[1] = vec4<f32>(m.tempo.x, m.tempo.y, m.tempo.z, m.tempo.w);
    outs[2] = vec4<f32>(m.stems[STEM_BASS], m.stems[STEM_DRUMS], m.stems[STEM_OTHER], m.stems[STEM_VOCALS]);
    outs[3] = vec4<f32>(m.voice.x, m.voice.y, m.voice.z, m.voice.w);
    outs[4] = vec4<f32>(m.count.x, m.count.y, 0.0, 0.0);";
    let outs = run(&gpu, &source, body, &ins, (1, 5));
    let stem = |s: Stem| m.stems[s as usize];
    let want = [
        [m.beat_phase, m.bar_phase, m.beat, m.downbeat],
        [m.tempo, m.sure, m.level, m.progress],
        [Stem::Bass, Stem::Drums, Stem::Other, Stem::Vocals].map(stem),
        [m.melody, m.sung, m.singing, m.section],
        [m.beats, m.bar_beats, 0.0, 0.0],
    ];
    assert_eq!(outs, want);
    assert_eq!(Music::from_uniform(&m.uniform()).uniform(), m.uniform());
}

/// The word slam's taps of the atlas (`slam_cover`) as the processor's
/// ([`super::fx::slam::cover`]), in an atlas of their own: a word's texels
/// and another word's beside them, read in and round the word (between
/// texels, past its edge, inside its letters, its outline's reach).
#[test]
fn the_slam_s_taps_are_their_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let (device, queue) = &gpu;
    const SIDE: u32 = 32;
    let byte = |x: u32, y: u32| -> u8 {
        if x >= 24 {
            200
        } else if (8..12).contains(&x) {
            255
        } else {
            ((x * 7 + y * 13) % 256) as u8
        }
    };
    let bytes: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| [byte(i % SIDE, i / SIDE), 0, 0, 255])
        .collect();
    let size = wgpu::Extent3d {
        width: SIDE,
        height: SIDE,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom shader check"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIDE * 4),
            rows_per_image: Some(SIDE),
        },
        size,
    );
    let view = texture.create_view(&Default::default());
    let rect = [0.0, 0.0, 24.0, 20.0];
    let samples: Vec<[f32; 4]> = (0..2000)
        .map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract();
            [
                fr(0.613) * 30.0,
                fr(0.377) * 24.0,
                0.5 + fr(0.291) * 4.0,
                0.0,
            ]
        })
        .collect();
    let source = format!(
        "
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;
{}",
        super::fx::slam::cover_wgsl()
    );
    let body = format!(
        "
    let a = ins[i];
    outs[i] = vec4<f32>(slam_cover(a.xy, vec4<f32>({:?}, {:?}, {:?}, {:?}), a.z, {SIDE}.0), 0.0, 0.0);",
        rect[0], rect[1], rect[2], rect[3]
    );
    let outs = run_reading(
        &gpu,
        (&source, &body),
        &samples,
        (samples.len(), 1),
        Some(&view),
    );
    let texel = |x: i32, y: i32| {
        let (x, y) = (x.clamp(0, SIDE as i32 - 1), y.clamp(0, SIDE as i32 - 1));
        f32::from(byte(x as u32, y as u32)) / 255.0
    };
    let mut inside = 0;
    for (s, g) in samples.iter().zip(&outs) {
        let want = super::fx::slam::cover(&texel, (s[0], s[1]), rect, s[2]);
        for c in 0..2 {
            // A GPU's linear filter weighs in steps of 1/256.
            assert!(
                near(g[c], want[c], 6e-3, 0.0),
                "{s:?}: GPU {g:?}, processor {want:?}"
            );
        }
        inside += usize::from(want[0] >= 1.0);
    }
    assert!(inside > 20, "{inside}");
}

/// Every quad of every kind's probes ([`particles::Motion::probes`]: a
/// bubble's, small and large, drifting or not; a word scored's, a step
/// up's, the angels', a jackpot's, confetti, the coins') at times through
/// their lives and past them: the shader's [`sprite`] as the processor's.
#[test]
fn the_particles_shader_draws_what_sprite_says() {
    let Some(gpu) = gpu() else {
        return;
    };
    let sprays: Vec<Spray> = MOTIONS
        .iter()
        .flat_map(|m| m.probes.iter().copied())
        .collect();
    let ages = [
        -0.01, 0.0, 0.02, 0.03, 0.05, 0.079, 0.081, 0.1, 0.12, 0.21, 0.25, 0.35, 0.4, 0.44, 0.49,
        0.59, 0.6, 0.8, 0.9, 1.1, 1.2, 1.45, 1.7, 2.0, 2.3, 2.5, 3.1, 3.39, 4.4, 5.9,
    ];
    let mut samples = Vec::new();
    for s in &sprays {
        for &age in &ages {
            for q in 0..QUADS {
                samples.push((*s, q, 10.0 + age));
            }
        }
    }
    let ins: Vec<[f32; 4]> = samples
        .iter()
        .flat_map(|(s, q, t)| {
            let p = s.packed();
            let raw: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&p));
            [
                [raw[0], raw[1], raw[2], raw[3]],
                [raw[4], raw[5], raw[6], raw[7]],
                [raw[8], raw[9], raw[10], raw[11]],
                [*q as f32, *t, 0.0, 0.0],
            ]
        })
        .collect();
    let body = "
    let sp = sprite_of(ins[i * 4u], ins[i * 4u + 1u], ins[i * 4u + 2u], u32(ins[i * 4u + 3u].x), ins[i * 4u + 3u].y);
    outs[i * 3u] = vec4<f32>(sp.at, sp.half);
    outs[i * 3u + 1u] = vec4<f32>(sp.rgb, sp.alpha);
    outs[i * 3u + 2u] = vec4<f32>(f32(sp.look), sp.turn, f32(sp.shows), 0.0);";
    let outs = run(&gpu, &particles::shader(), body, &ins, (samples.len(), 3));
    let mut shown = 0;
    for (k, (s, q, t)) in samples.iter().enumerate() {
        let (a, c, w) = (outs[k * 3], outs[k * 3 + 1], outs[k * 3 + 2]);
        let what = format!("{:?} quad {q} at {t}", s.kind);
        let cpu = sprite(s, *q, *t);
        assert_eq!(cpu.is_some(), w[2] > 0.5, "{what}: shows");
        let Some(p) = cpu else { continue };
        shown += 1;
        let pos = |x: f32, y: f32| near(x, y, 0.05, 2e-4);
        assert!(
            pos(a[0], p.at.0) && pos(a[1], p.at.1),
            "{what}: at {a:?} {p:?}"
        );
        assert!(
            pos(a[2], p.half.0) && pos(a[3], p.half.1),
            "{what}: size {a:?} {p:?}"
        );
        for i in 0..3 {
            assert!(
                near(c[i], p.rgb[i], 2e-3, 0.0),
                "{what}: colour {c:?} {p:?}"
            );
        }
        assert!(near(c[3], p.alpha, 3e-3, 2e-3), "{what}: alpha {c:?} {p:?}");
        assert_eq!(w[0] as u32, p.look.code(), "{what}: look");
        let (dt, ds) = ((w[1] - p.turn).cos(), (w[1] - p.turn).sin());
        assert!(
            dt > 0.0 && ds.abs() < 3e-3,
            "{what}: turn {} {}",
            w[1],
            p.turn
        );
    }
    assert!(shown > 500, "{shown}");
}

/// The liquid's maths, at points across their ranges: the shader's
/// [`liquid::bulge`], [`liquid::swell`], [`liquid::slosh`],
/// [`liquid::front`], [`liquid::shimmer`] and [`liquid::amplitude`] as the
/// processor's.
#[test]
fn the_stage_shader_s_liquid_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut samples = Vec::new();
    for k in 0..240 {
        let f = k as f32;
        let pressure = (f * 0.37).fract();
        let time = f * 0.113;
        let phase = (f * 0.71).fract() * TAU;
        let s = (f * 0.29).fract() * 1.2 - 0.1;
        let since = (f * 0.53).fract() * 0.8 - 0.1;
        let y = (f * 0.43).fract() * 2.4 - 1.2;
        let amp = 1.0 + (f * 0.17).fract() * 6.0;
        let jumped = time - since;
        let (x, y2, half) = (f * 3.7, (f * 1.9).fract() * 40.0, (f * 0.61).fract() * 30.0);
        samples.push([
            [pressure, time, phase, s],
            [since, y, amp, jumped],
            [x, y2, half, 0.0],
        ]);
    }
    let ins: Vec<[f32; 4]> = samples.iter().flatten().copied().collect();
    let body = "
    let a = ins[i * 3u];
    let b = ins[i * 3u + 1u];
    let c = ins[i * 3u + 2u];
    outs[i * 2u] = vec4<f32>(bulge(a.x, a.y, a.z), swell(a.w), slosh(b.x));
    outs[i * 2u + 1u] = vec4<f32>(front(b.y, a.y, b.z, b.w), shimmer(c.x, c.y, a.y), amplitude(c.z), 0.0);";
    let source = super::stage_gpu::shader(super::stage_gpu::Group::Plain);
    let outs = run(&gpu, &source, body, &ins, (samples.len(), 2));
    for (k, [a, b, c]) in samples.iter().enumerate() {
        let (g, h) = (outs[k * 2], outs[k * 2 + 1]);
        let what = format!("sample {k}: {a:?} {b:?} {c:?}");
        let (bx, by) = liquid::bulge(a[0], a[1], a[2]);
        assert!(
            near(g[0], bx, 1e-3, 0.0) && near(g[1], by, 1e-3, 0.0),
            "{what}: bulge {g:?}"
        );
        assert!(
            near(g[2], liquid::swell(a[3]), 1e-4, 0.0),
            "{what}: swell {g:?}"
        );
        assert!(
            near(g[3], liquid::slosh(b[0]), 1e-3, 1e-3),
            "{what}: slosh {g:?}"
        );
        let f = liquid::front(b[1], a[1], b[2], b[3]);
        assert!(near(h[0], f, 5e-3, 2e-3), "{what}: front {h:?} {f}");
        let s = liquid::shimmer(c[0], c[1], a[1]);
        assert!(near(h[1], s, 3e-3, 0.0), "{what}: shimmer {h:?} {s}");
        assert!(
            near(h[2], liquid::amplitude(c[2]), 1e-5, 0.0),
            "{what}: amplitude"
        );
    }
}

/// The light's tone map at brightnesses from none to far past white, into
/// a frame that takes linear light and one that does not: the shader's
/// [`super::bloom::TONE`] as the processor's [`super::bloom::tone`].
#[test]
fn the_bloom_s_tone_map_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let ins: Vec<[f32; 4]> = (0..64)
        .map(|k| {
            let x = k as f32 * 0.15 - 0.5;
            [x, x * 2.0, x * 0.3, f32::from(k % 2 == 0)]
        })
        .collect();
    let body = "
    let a = ins[i];
    outs[i] = vec4<f32>(tone(a.xyz, a.w), 0.0);";
    let outs = run(&gpu, super::bloom::TONE, body, &ins, (ins.len(), 1));
    for (a, g) in ins.iter().zip(&outs) {
        for c in 0..3 {
            let want = super::bloom::tone(a[c], a[3] > 0.5);
            assert!(near(g[c], want, 2e-3, 2e-3), "{a:?}: {g:?}");
        }
    }
}

/// The shine's looks at points across their ranges: the shader's
/// [`glory::meter_fill`], [`glory::fill_color`], [`glory::meter_at`],
/// [`glory::ring_at`], [`glory::mark_pop`] and [`glory::label_pop`] as the
/// processor's.
#[test]
fn the_stage_shader_s_glory_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let mut samples = Vec::new();
    for k in 0..360 {
        let f = k as f32;
        let fr = |m: f32| (f * m).fract();
        let time = 3.0 + fr(0.37) * 4.0;
        samples.push([
            [
                fr(0.61) * 1.2 - 0.1,
                fr(0.29) * 2.0 - 1.0,
                fr(0.43) * 1.1,
                time,
            ],
            [fr(0.17), fr(0.53), time - fr(0.71) * 1.5, fr(0.23) * 0.75],
            // Off the ring's centre and its level line, whose angles (atan2
            // of 0, 0, and ±π, where the fill's edge may be) a GPU may give
            // either way (Apple's did).
            [
                200.0 + fr(0.19) * 80.0 - 40.5,
                300.0 + fr(0.83) * 80.0 - 40.3,
                fr(0.31) * 60.0 - 20.0,
                fr(0.47) * 4.0,
            ],
        ]);
    }
    let ins: Vec<[f32; 4]> = samples.iter().flatten().copied().collect();
    let body = "
    let a = ins[i * 3u];
    let b = ins[i * 3u + 1u];
    let c = ins[i * 3u + 2u];
    let s = vec3<f32>(b.w, 0.25, 3.0);
    outs[i * 5u] = vec4<f32>(meter_fill(b.x, b.y, b.z, a.w), mark_pop(a.w - b.z), 0.0);
    outs[i * 5u + 1u] = vec4<f32>(fill_color(a.z, a.x, a.w), 0.0);
    outs[i * 5u + 2u] = meter_at(a.x, a.y, a.z, a.w);
    outs[i * 5u + 3u] = ring_at(c.xy, vec2<f32>(200.0, 300.0), vec2<f32>(30.0, 6.0), b.x, s, a.w);
    outs[i * 5u + 4u] = vec4<f32>(label_pop(a.w - b.z, 1.8), 0.0, 0.0);";
    let source = super::stage_gpu::shader(super::stage_gpu::Group::Plain);
    let outs = run(&gpu, &source, body, &ins, (samples.len(), 5));
    let close = |g: &[f32], c: &[f32], what: &str| {
        for (x, y) in g.iter().zip(c) {
            assert!(near(*x, *y, 3e-3, 3e-3), "{what}: {g:?} {c:?}");
        }
    };
    let (mut lit, mut shown) = (0, 0);
    for (k, [a, b, c]) in samples.iter().enumerate() {
        let g = &outs[k * 5..k * 5 + 5];
        let what = format!("sample {k}: {a:?} {b:?} {c:?}");
        let shine = Glow {
            level: b[3],
            from: 0.25,
            since: 3.0,
        };
        let pop = glory::mark_pop(a[3] - b[2]);
        close(
            &g[0],
            &[glory::meter_fill(b[0], b[1], b[2], a[3]), pop.0, pop.1, 0.0],
            &format!("{what}: fill, pop"),
        );
        close(
            &g[1][..3],
            &glory::fill_color(a[2], a[0], a[3]),
            &format!("{what}: fill colour"),
        );
        close(
            &g[2],
            &glory::meter_at(a[0], a[1], a[2], a[3]),
            &format!("{what}: meter"),
        );
        let ring = glory::ring_at(
            (c[0], c[1]),
            (200.0, 300.0),
            (30.0, 6.0),
            b[0],
            &shine,
            a[3],
        );
        close(&g[3], &ring, &format!("{what}: ring"));
        lit += usize::from(ring[3] > 0.05);
        let label = glory::label_pop(a[3] - b[2], 1.8);
        close(&g[4][..2], &[label.0, label.1], &format!("{what}: label"));
        shown += usize::from(label.1 > 0.0 && label.1 < 1.0);
    }
    assert!(lit > 50 && shown > 10, "{lit} {shown}");
}

/// The diploma's stuck and pressed shapes at points in and round them:
/// the shader's [`seal::sticker_at`], [`seal::seal_at`], [`seal::star`]
/// and [`seal::rays_at`] as the processor's.
#[test]
fn the_stage_shader_s_seal_and_stars_are_their_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let samples: Vec<[f32; 4]> = (0..600)
        .map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract();
            // Off the middle, where an angle would be undefined and the
            // sticker's facets all meet (a GPU may pick any; Apple's did).
            [
                fr(0.613) * 2.6 - 1.3 + 0.001,
                fr(0.377) * 2.4 - 1.2 + 0.013,
                fr(0.291) * 3.4 - 1.7,
                fr(0.733) * 12.0,
            ]
        })
        .collect();
    let body = "
    let s = ins[i];
    outs[i * 3u] = vec4<f32>(sticker_at(s.x, s.y, s.z), star(s.x, s.y, 0.56, 0.45));
    outs[i * 3u + 1u] = vec4<f32>(seal_at(s.x, s.y, s.z), wax_edge(s.w));
    outs[i * 3u + 2u] = vec4<f32>(rays_at(s.x * 200.0, s.y * 200.0, 150.0, 14.0, s.w), 0.0, 0.0, 0.0);";
    let source = super::stage_gpu::shader(super::stage_gpu::Group::Plain);
    let outs = run(&gpu, &source, body, &samples, (samples.len(), 3));
    let close = |g: &[f32], c: &[f32], what: &str| {
        for (x, y) in g.iter().zip(c) {
            assert!(near(*x, *y, 2e-3, 2e-3), "{what}: {g:?} {c:?}");
        }
    };
    let mut inside = 0;
    for (k, s) in samples.iter().enumerate() {
        let g = &outs[k * 3..k * 3 + 3];
        let what = format!("sample {k}: {s:?}");
        let sticker = seal::sticker_at(s[0], s[1], s[2]);
        let star = seal::star(s[0], s[1], 0.56, 0.45);
        close(
            &g[0],
            &[sticker[0], sticker[1], sticker[2], star],
            &format!("{what}: sticker, star"),
        );
        let wax = seal::seal_at(s[0], s[1], s[2]);
        let edge = seal::wax_edge(s[3]);
        close(
            &g[1],
            &[wax[0], wax[1], wax[2], edge],
            &format!("{what}: seal"),
        );
        let rays = seal::rays_at(s[0] * 200.0, s[1] * 200.0, 150.0, 14.0, s[3]);
        close(&g[2][..1], &[rays], &format!("{what}: rays"));
        inside += usize::from(sticker[0] < 0.0) + usize::from(wax[0] < 0.0);
    }
    assert!(inside > 300, "{inside}");
}

/// The diploma's paper at pixels over, round and off its sheet, unrolling
/// and open: the shader's [`paper::paper_at`] (its tone or shadow and its
/// border's inks), [`paper::stamp_at`], [`paper::signature_at`] (written
/// part of the way and whole) and [`paper::turn_about`] as the
/// processor's.
#[test]
fn the_stage_shader_s_paper_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let (half, s) = ((480.0f32, 320.0f32), 6.5f32);
    let samples: Vec<[f32; 4]> = (0..900)
        .map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract();
            // Most near the sheet's edges, where its border is.
            let edge = k % 3 != 0;
            let (x, y) = if edge {
                let along = fr(0.613) * 2.0 - 1.0;
                let across = fr(0.377) * 9.0 * s - 0.5 * s;
                if k % 2 == 0 {
                    (
                        along * half.0,
                        (half.1 - across) * if k % 4 == 0 { 1.0 } else { -1.0 },
                    )
                } else {
                    (
                        (half.0 - across) * if k % 5 < 2 { 1.0 } else { -1.0 },
                        along * half.1,
                    )
                }
            } else {
                (fr(0.613) * 1100.0 - 550.0, fr(0.377) * 760.0 - 380.0)
            };
            let front = [-half.1 * 0.6, 0.0, half.1 * 0.7, half.1][k % 4];
            [x, y, front, fr(0.291)]
        })
        .collect();
    let body = "
    let s = ins[i];
    let half = vec2<f32>(480.0, 320.0);
    outs[i * 3u] = paper_at(s.x, s.y, half, 6.5, s.z);
    let sig = vec2<f32>(signature_at(s.x / 300.0, s.y / 300.0, s.w), signature_at(s.x / 300.0, s.y / 300.0, 1.0));
    outs[i * 3u + 1u] = vec4<f32>(stamp_at(s.x * 0.4, s.y * 0.4, vec2<f32>(140.0, 45.0), 6.5), sig);
    outs[i * 3u + 2u] = vec4<f32>(turn_about(s.xy, vec2<f32>(30.0, -20.0), s.w * 6.0), 0.0, 0.0);";
    let source = super::stage_gpu::shader(super::stage_gpu::Group::Plain);
    let outs = run(&gpu, &source, body, &samples, (samples.len(), 3));
    let close = |g: &[f32], c: &[f32], what: &str| {
        for (x, y) in g.iter().zip(c) {
            assert!(near(*x, *y, 3e-3, 1e-3), "{what}: {g:?} {c:?}");
        }
    };
    let (mut inked, mut rolled) = (0, 0);
    for (k, p) in samples.iter().enumerate() {
        let g = &outs[k * 3..k * 3 + 3];
        let what = format!("sample {k}: {p:?}");
        let sheet = paper::paper_at(p[0], p[1], half, s, p[2]);
        close(&g[0], &sheet, &format!("{what}: paper"));
        inked += usize::from(sheet[0] < 0.0 && sheet[2] + sheet[3] > 0.5);
        rolled += usize::from(sheet[0] < 0.0 && p[1] > p[2]);
        let stamp = paper::stamp_at(p[0] * 0.4, p[1] * 0.4, (140.0, 45.0), s);
        let part = paper::signature_at(p[0] / 300.0, p[1] / 300.0, p[3]);
        let whole = paper::signature_at(p[0] / 300.0, p[1] / 300.0, 1.0);
        close(
            &g[1],
            &[stamp[0], stamp[1], part, whole],
            &format!("{what}: stamp, signature"),
        );
        // A point turned is off by the sine's rounding times how far out it
        // is (on GitHub's Intel Mac, 0.008 of a pixel at 470 out).
        let (x, y) = paper::turn_about((p[0], p[1]), (30.0, -20.0), p[3] * 6.0);
        let out = (p[0] - 30.0).hypot(p[1] + 20.0);
        for (gv, want) in g[2][..2].iter().zip([x, y]) {
            assert!(
                (gv - want).abs() <= 3e-3 + 1e-4 * out,
                "{what}: turned: {:?} {:?}",
                &g[2][..2],
                [x, y]
            );
        }
    }
    assert!(inked > 50 && rolled > 10, "{inked} {rolled}");
}

/// A tube's nearest points at pixels round and along it, past its ends
/// and out of reach: the shader's `tube_near` as the processor's
/// [`super::snake::nearest`] (how far, where along, which side).
#[test]
fn the_stage_shader_s_tube_is_its_reference() {
    let Some(gpu) = gpu() else {
        return;
    };
    let notes = [
        (100.0, 260.0, 300.0),
        (260.0, 300.0, 220.0),
        (300.0, 330.0, 260.0),
        (330.0, 520.0, 180.0),
    ];
    let snake = super::snake::Snake::through(&notes, (100.0, 520.0), 22.0);
    let points = snake.points.iter().map(|&(x, y)| [x, y, 0.0, 0.0]);
    let n = snake.points.len();
    let samples: Vec<[f32; 4]> = (0..400)
        .map(|k| {
            let f = k as f32;
            let x = 60.0 + (f * 0.61).fract() * 500.0;
            let y = 140.0 + (f * 0.37).fract() * 220.0;
            [x, y, [30.0, 60.0, 8.0][k % 3], 0.0]
        })
        .collect();
    let ins: Vec<[f32; 4]> = points.chain(samples.iter().copied()).collect();
    let source = format!(
        "fn path_point(i: u32) -> vec4<f32> {{\n    return ins[i];\n}}\n{}",
        super::snake::NEAR_WGSL
    );
    let body = format!(
        "
    let s = ins[{n}u + i];
    outs[i] = vec4<f32>(tube_near(s.xy, 0u, {n}u, s.z), 0.0);"
    );
    let outs = run(&gpu, &source, &body, &ins, (samples.len(), 1));
    let mut close_by = 0;
    for (s, g) in samples.iter().zip(&outs) {
        let (d, along, side) = super::snake::nearest(&snake.points, (s[0], s[1]), s[2]);
        let what = format!("{s:?}: {g:?} against {:?}", (d, along, side));
        if d > 1e8 {
            assert!(g[0] > 1e8, "{what}");
            continue;
        }
        close_by += usize::from(d < 40.0);
        assert!(near(g[0], d, 2e-3, 1e-4), "{what}: distance");
        assert!(near(g[1], along, 2e-3, 1e-4), "{what}: along");
        assert!(near(g[2], side, 2e-3, 1e-4), "{what}: side");
    }
    assert!(close_by > 60, "{close_by}");
}

/// The backdrop's maths at points across the window and through time: the
/// shader's [`look::rounded_box`], [`look::card_edge`], [`look::falloff`],
/// [`look::bspline`], [`look::shift`], [`look::vignette`],
/// [`look::sweep`], [`look::drift`], [`look::ray_fan`], [`look::defocus`]
/// and [`look::kicked`] as the
/// processor's.
#[test]
fn the_backdrop_shader_s_maths_is_its_reference() {
    use super::backdrop::look;
    let Some(gpu) = gpu() else {
        return;
    };
    let samples: Vec<[f32; 4]> = (0..500)
        .map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract();
            [fr(0.613), fr(0.377), fr(0.291), fr(0.733) * 60.0]
        })
        .collect();
    let body = "
    let s = ins[i];
    let sh = shift(s.xy, s.z, vec2<f32>(0.01, -0.006), 0.008, vec2<f32>(0.45, 0.4));
    let d = rounded_box((s.x - 0.5) * 400.0, (s.y - 0.5) * 300.0, 120.0, 100.0, 9.0);
    outs[i * 5u] = vec4<f32>(d, card_edge(d, 2000.0), sh);
    let rays = ray_fan(s.x - 0.5, s.y - 0.4, s.w, 0.2);
    let blur = defocus(s.z, s.x);
    outs[i * 5u + 1u] = vec4<f32>(vignette(s.x, s.y), blur, sweep(s.x, s.y, s.w), rays);
    outs[i * 5u + 2u] = vec4<f32>(drift(s.w), falloff(d, 30.0));
    outs[i * 5u + 3u] = bspline(s.z);
    outs[i * 5u + 4u] = vec4<f32>(kicked(s.xy, vec2<f32>(0.45, 0.4), s.z * 0.05), 0.0, 0.0);";
    let source = super::backdrop::gpu::shader(false);
    let outs = run(&gpu, &source, body, &samples, (samples.len(), 5));
    let close = |g: &[f32], c: &[f32], what: &str| {
        for (x, y) in g.iter().zip(c) {
            assert!(near(*x, *y, 3e-3, 2e-3), "{what}: {g:?} {c:?}");
        }
    };
    let (mut edged, mut swept, mut blurred) = (0, 0, 0);
    for (k, s) in samples.iter().enumerate() {
        let g = &outs[k * 5..k * 5 + 5];
        let what = format!("sample {k}: {s:?}");
        let sh = look::shift([s[0], s[1]], s[2], [0.01, -0.006], 0.008, [0.45, 0.4]);
        let d = look::rounded_box(
            (s[0] - 0.5) * 400.0,
            (s[1] - 0.5) * 300.0,
            120.0,
            100.0,
            9.0,
        );
        let edge = look::card_edge(d, 2000.0);
        close(
            &g[0],
            &[d, edge, sh[0], sh[1]],
            &format!("{what}: box, edge, shift"),
        );
        let sweep = look::sweep(s[0], s[1], s[3]);
        close(
            &g[1],
            &[
                look::vignette(s[0], s[1]),
                look::defocus(s[2], s[0]),
                sweep,
                look::ray_fan(s[0] - 0.5, s[1] - 0.4, s[3], 0.2),
            ],
            &format!("{what}: vignette, defocus, sweep, rays"),
        );
        let w = look::drift(s[3]);
        close(
            &g[2],
            &[w[0], w[1], w[2], look::falloff(d, 30.0)],
            &format!("{what}: drift, falloff"),
        );
        close(&g[3], &look::bspline(s[2]), &format!("{what}: bspline"));
        let kicked = look::kicked([s[0], s[1]], [0.45, 0.4], s[2] * 0.05);
        close(&g[4], &[kicked[0], kicked[1]], &format!("{what}: kicked"));
        blurred += usize::from(g[1][1] > look::DEFOCUS_BASE && g[1][1] < 1.0);
        edged += usize::from(edge > 0.0 && edge < 1.0);
        swept += usize::from(sweep > 0.3);
    }
    assert!(
        edged > 20 && swept > 20 && blurred > 20,
        "{edged} {swept} {blurred}"
    );
}

/// The backdrop's smooth read ([`super::backdrop::gpu::SMOOTH`], the
/// ground's stretch) of a 4 × 4 texture at points across it and past its
/// edges: a cubic B-spline of its texels (the edge's held), worked out
/// texel by texel on the processor.
#[test]
fn the_backdrop_s_smooth_read_is_a_b_spline_of_its_texels() {
    let Some(gpu) = gpu() else {
        return;
    };
    let (device, queue) = &gpu;
    // Each texel a fifteenth (exact in a byte).
    let values: Vec<f32> = (0..16).map(|i| ((i * 7) % 16) as f32 / 15.0).collect();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| [(v * 255.0).round() as u8, 0, 0, 255])
        .collect();
    let size = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom shader check"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(4),
        },
        size,
    );
    let view = texture.create_view(&Default::default());
    let samples: Vec<[f32; 4]> = (0..400)
        .map(|k| {
            let f = k as f32;
            let fr = |m: f32| (f * m).fract() * 1.4 - 0.2;
            [fr(0.613), fr(0.377), 0.0, 0.0]
        })
        .collect();
    let source = format!(
        "{}{}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;
",
        super::backdrop::look::wgsl(),
        super::backdrop::gpu::SMOOTH
    );
    let body = "
    outs[i] = smooth_at(src, ins[i].xy, 0u);";
    let outs = run_reading(
        &gpu,
        (&source, body),
        &samples,
        (samples.len(), 1),
        Some(&view),
    );
    // The cubic B-spline's weight `t` texels off.
    let weight = |t: f32| {
        let t = t.abs();
        if t < 1.0 {
            (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0
        } else if t < 2.0 {
            (2.0 - t).powi(3) / 6.0
        } else {
            0.0
        }
    };
    for (s, g) in samples.iter().zip(&outs) {
        let (x, y) = (s[0] * 4.0 - 0.5, s[1] * 4.0 - 0.5);
        let mut want = 0.0;
        for j in -3..7i32 {
            for k in -3..7i32 {
                let texel = values[(k.clamp(0, 3) * 4 + j.clamp(0, 3)) as usize];
                want += weight(x - j as f32) * weight(y - k as f32) * texel;
            }
        }
        assert!(near(g[0], want, 1e-2, 0.0), "{s:?}: {} {want}", g[0]);
    }
}
