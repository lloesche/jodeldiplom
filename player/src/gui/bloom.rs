//! The light that blooms, on the GPU: the stage's shapes and bursts that
//! are light ([`super::stage_gpu::StageFrame::lights`], `light_sprays`:
//! the lane's glow and flames, the angels, the jackpot, [`super::fx`]) are
//! drawn apart, added up in linear light at high range ([`LIGHT`]) a texel
//! a point
//! ([`light_size`]: on a Retina screen a quarter of the pixels to draw,
//! blur and read, as sharp as on any other); that light is blurred down a
//! chain of halved textures and back up, its last step up into a bloom
//! half the frame's size (a dual filter: each step down reads five taps,
//! each up eight, so the glow spreads far for little work), and the light
//! and its bloom are tone mapped ([`tone`]: bright light rolls off to
//! white, never clips) and added over the frame, two reads a pixel.

use super::fx::Music;
use super::stage_gpu::{Blend, Group, StageFrame, StagePass};

/// The light's format: half floats, light brighter than white kept.
pub const LIGHT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The steps down the chain (each half the size of the one before).
const STEPS: usize = 5;
/// Where the bloom is among the textures: after the light and the chain.
const BLOOM: usize = STEPS + 1;
/// How strongly the bloom adds to the light itself.
pub const STRENGTH: f32 = 0.3;

/// The light's texels for a frame of `size` pixels, `density` of them to
/// a point (the window's scale): a texel a point, so as sharp as on a
/// screen of a pixel a point, and never fewer than half the pixels each
/// way (the chain's first step).
pub fn light_size(size: (u32, u32), density: f32) -> (u32, u32) {
    let d = density.clamp(1.0, 2.0);
    let side = |s: u32| ((s as f32 / d).round() as u32).max(s / 2).max(1);
    (side(size.0), side(size.1))
}

/// Light `x` (linear, any brightness) as the frame shows it, 0 to 1: as
/// written to a frame that takes linear light (`linear`), else encoded
/// (as the shader's [`TONE`], held to it in `super::shader_check`).
pub fn tone(x: f32, linear: bool) -> f32 {
    let c = 1.0 - (-x.max(0.0)).exp();
    if linear {
        c
    } else {
        c.powf(1.0 / 2.2)
    }
}

/// [`tone`] in WGSL.
pub const TONE: &str = r"
fn tone(l: vec3<f32>, linear: f32) -> vec3<f32> {
    let c = vec3<f32>(1.0) - exp(-max(l, vec3<f32>(0.0)));
    if linear < 0.5 {
        return pow(c, vec3<f32>(1.0 / 2.2));
    }
    return c;
}
";

const SHADER: &str = r"
struct Pass {
    texel: vec2<f32>,
    strength: f32,
    linear: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;
@group(0) @binding(2) var<uniform> pass_: Pass;
@group(0) @binding(3) var light: texture_2d<f32>;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

fn at(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, soft, uv, 0.0).rgb;
}

// Down a step: the middle and four taps half a texel out on the diagonals.
@fragment
fn down(o: Out) -> @location(0) vec4<f32> {
    let h = pass_.texel * 0.5;
    let c = at(o.uv) * 4.0
        + at(o.uv + vec2<f32>(-h.x, -h.y)) + at(o.uv + vec2<f32>(h.x, -h.y))
        + at(o.uv + vec2<f32>(-h.x, h.y)) + at(o.uv + vec2<f32>(h.x, h.y));
    return vec4<f32>(c / 8.0, 1.0);
}

// Up a step: a tent of eight taps, by the pass's strength, added to what
// is there (the last step's into the bloom alone).
@fragment
fn up(o: Out) -> @location(0) vec4<f32> {
    let uv = o.uv;
    let h = pass_.texel * 0.5;
    var c = at(uv + vec2<f32>(-h.x * 2.0, 0.0)) + at(uv + vec2<f32>(h.x * 2.0, 0.0))
        + at(uv + vec2<f32>(0.0, -h.y * 2.0)) + at(uv + vec2<f32>(0.0, h.y * 2.0));
    c = c + 2.0 * (at(uv + vec2<f32>(-h.x, h.y)) + at(uv + vec2<f32>(h.x, h.y))
        + at(uv + vec2<f32>(-h.x, -h.y)) + at(uv + vec2<f32>(h.x, -h.y)));
    return vec4<f32>(c / 12.0 * pass_.strength, 1.0);
}

// The light and its bloom, tone mapped, added over the frame: two reads.
@fragment
fn composite(o: Out) -> @location(0) vec4<f32> {
    let l = textureSampleLevel(light, soft, o.uv, 0.0).rgb + at(o.uv);
    return vec4<f32>(tone(l, pass_.linear), 0.0);
}
";

/// A pass's numbers as its shader reads them (`Pass`): the texel's size in
/// the texture it reads, how strongly it adds, and whether the frame takes
/// linear light (1) or not (0).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Pass {
    texel: [f32; 2],
    strength: f32,
    linear: f32,
}

/// A texture of the chain and its view.
struct Level {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// The light, its chain, and the passes down, up and onto the frame.
pub struct Bloom {
    layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    linear: bool,
    /// The frame's size and the light's they are for; the light, then the
    /// chain, then the bloom (as large as the chain's first step).
    frame: ((u32, u32), (u32, u32)),
    levels: Vec<Level>,
    /// Each pass's numbers and what it reads: down each step, up each
    /// step, onto the frame.
    passes: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl Bloom {
    /// For frames of `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Bloom {
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom bloom"),
            entries: &[
                texture(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Pass>() as u64),
                    },
                    count: None,
                },
                texture(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom bloom"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom bloom"),
            source: wgpu::ShaderSource::Wgsl(format!("{TONE}{SHADER}").into()),
        });
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = |entry: &str, target: wgpu::TextureFormat, added: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("jodeldiplom bloom"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend: added.then_some(wgpu::BlendState {
                            color: add,
                            alpha: add,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom bloom"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Bloom {
            down: pipeline("down", LIGHT, false),
            up: pipeline("up", LIGHT, true),
            composite: pipeline("composite", format, true),
            layout,
            sampler,
            linear: super::stage_gpu::takes_linear(format),
            frame: ((0, 0), (0, 0)),
            levels: Vec::new(),
            passes: Vec::new(),
        }
    }

    /// The light's texture for a frame of `size` with `density` pixels to
    /// a point ([`light_size`]), made (with its chain) where either
    /// changed.
    pub fn light(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        density: f32,
    ) -> &wgpu::TextureView {
        let size = (size.0.max(1), size.1.max(1));
        let light = light_size(size, density);
        if self.levels.is_empty() || self.frame != (size, light) {
            self.make(device, queue, size, light);
            self.frame = (size, light);
        }
        &self.levels[0].view
    }

    /// The light (`light` texels), the chain from it (its first step half
    /// the frame's `size`, each after half the one before) and the bloom.
    fn make(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        light: (u32, u32),
    ) {
        let level = |s: (u32, u32)| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("jodeldiplom bloom"),
                size: wgpu::Extent3d {
                    width: s.0,
                    height: s.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LIGHT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            Level {
                _texture: texture,
                view,
                size: s,
            }
        };
        let half = |s: (u32, u32)| ((s.0 / 2).max(1), (s.1 / 2).max(1));
        self.levels = vec![level(light)];
        let mut s = size;
        for _ in 0..STEPS {
            s = half(s);
            self.levels.push(level(s));
        }
        self.levels.push(level(half(size)));
        // Down: level k into k + 1, reading k; up: k + 1 into k, reading
        // k + 1, and the last from the first step into the bloom, by its
        // strength; onto the frame: reading the bloom and the light.
        let mut passes = Vec::new();
        let mut pass = |src: usize, strength: f32| {
            let (w, h) = self.levels[src].size;
            let numbers = Pass {
                texel: [1.0 / w as f32, 1.0 / h as f32],
                strength,
                linear: if self.linear { 1.0 } else { 0.0 },
            };
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom bloom"),
                size: std::mem::size_of::<Pass>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&numbers));
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodeldiplom bloom"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.levels[src].view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.levels[0].view),
                    },
                ],
            });
            passes.push((buffer, bind));
        };
        for k in 0..STEPS {
            pass(k, 1.0);
        }
        for k in (1..STEPS).rev() {
            pass(k + 1, 1.0);
        }
        pass(1, STRENGTH);
        pass(BLOOM, 1.0);
        self.passes = passes;
    }

    /// The chain down and back up, from the light drawn this frame.
    pub fn blur(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.levels.len() <= STEPS {
            return;
        }
        let mut run = |target: usize,
                       pipeline: &wgpu::RenderPipeline,
                       bind: &wgpu::BindGroup,
                       clear: bool| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jodeldiplom bloom"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.levels[target].view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        };
        for k in 0..STEPS {
            run(k + 1, &self.down, &self.passes[k].1, true);
        }
        for (i, k) in (1..STEPS).rev().enumerate() {
            run(k, &self.up, &self.passes[STEPS + i].1, false);
        }
        run(BLOOM, &self.up, &self.passes[2 * STEPS - 1].1, true);
    }

    /// The light and its bloom added over the frame being drawn in `pass`.
    pub fn composite(&self, pass: &mut wgpu::RenderPass) {
        let Some((_, bind)) = self.passes.last() else {
            return;
        };
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The light of a frame: its shapes and bursts, drawn apart, and their
/// bloom.
pub struct Light {
    shapes: super::stage_gpu::StagePass,
    bursts: super::particles::ParticlePass,
    bloom: Bloom,
    /// Whether the next frame has any.
    lit: bool,
}

impl Light {
    /// For frames of `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Light {
        Light {
            shapes: StagePass::with(device, (LIGHT, Blend::Add), &Group::light()),
            bursts: super::particles::ParticlePass::new(device, LIGHT, Blend::Add),
            bloom: Bloom::new(device, format),
            lit: false,
        }
    }

    /// The next frame's light (`frame`'s; none: none), in a frame of
    /// `size` pixels, `density` to a point: drawn now (blurred by
    /// [`Light::blur`]), in a submission of its own ahead of the frame's
    /// (iced's shader widget gives the frame only a render pass, which then
    /// adds it over: [`Light::composite`]).
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        frame: Option<&StageFrame>,
        (size, density): ((u32, u32), f32),
    ) {
        self.shapes
            .set((device, queue), frame.map(StageFrame::light_feed), size);
        let (sprays, music, shake, time) = frame
            .map_or((&[][..], Music::NEUTRAL, [0.0; 2], 0.0), |f| {
                (&f.light_sprays[..], f.music, f.shake, f.clock)
            });
        self.bursts
            .set((device, queue), (sprays, &music, shake), time, size);
        self.lit = frame.is_some_and(|f| !f.lights.is_empty() || !f.light_sprays.is_empty());
        if !self.lit {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom light"),
        });
        let view = self.bloom.light(device, queue, size, density);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jodeldiplom light"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.shapes.draw(&mut pass);
            self.bursts.draw(&mut pass);
        }
        queue.submit([encoder.finish()]);
    }

    /// The light drawn ([`Light::set`]) blurred down its chain and back up,
    /// in a submission of its own (nothing where the frame has none).
    pub fn blur(&self, (device, queue): (&wgpu::Device, &wgpu::Queue)) {
        if !self.lit {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom bloom"),
        });
        self.bloom.blur(&mut encoder);
        queue.submit([encoder.finish()]);
    }

    /// The light and its bloom added over the frame in `pass` (nothing
    /// where the frame has none).
    pub fn composite(&self, pass: &mut wgpu::RenderPass) {
        if self.lit {
            self.bloom.composite(pass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_rolls_off_to_white() {
        assert_eq!(tone(0.0, true), 0.0);
        assert!((tone(1.0, true) - 0.632).abs() < 1e-3);
        assert!(tone(8.0, true) > 0.999 && tone(8.0, true) < 1.0);
        assert_eq!(tone(-1.0, true), 0.0);
        assert!((tone(1.0, false) - 0.632f32.powf(1.0 / 2.2)).abs() < 1e-3);
    }

    #[test]
    fn the_light_is_a_texel_a_point() {
        // A Retina MacBook Pro's screen: half each way.
        assert_eq!(light_size((3024, 1964), 2.0), (1512, 982));
        // A screen of a pixel a point: every pixel.
        assert_eq!(light_size((1920, 1080), 1.0), (1920, 1080));
        assert_eq!(light_size((1920, 1080), 1.5), (1280, 720));
        // Denser still: no less than half; nor more than all.
        assert_eq!(light_size((3000, 2000), 3.0), (1500, 1000));
        assert_eq!(light_size((3000, 2000), 0.5), (3000, 2000));
        assert_eq!(light_size((1, 1), 2.0), (1, 1));
    }
}
