//! The backdrop on the GPU ([`super`]): the ground (the blurred cover,
//! small) in a texture of its own, sent over when the cover or the window
//! changes; each cover showing has its pieces in textures (the plate, the
//! subject with its matte, their depth, the rim), their texels made on the
//! covers' thread ([`crate::cover::Texels`]) and sent over once when it
//! first shows, their smaller levels made here (the soft halo and the calm
//! under the text read them), and dropped once it no longer shows. A frame
//! draws the ground stretched over a texture half the window's size, then
//! each cover's shader (`backdrop.wgsl`, its maths in [`super::look`]) over
//! it by its share, in a submission of its own before the window's, and
//! keeps it for the window's pass to read ([`BackdropPass::under`]). A
//! frame like the last one is not drawn again.

use std::sync::Arc;

use super::{BackdropFrame, Ground, Look};
use crate::cover::Layers;

/// The most levels a cover's textures have (a 1024 cover's down to 4).
const LEVELS: u32 = 9;
/// The bytes of a cover's numbers.
const LOOK_BYTES: u64 = std::mem::size_of::<Look>() as u64;

/// The pass that draws a texture stretched over its whole target, filtered:
/// a texture's smaller levels (each the one above, four texels averaged),
/// and the ground scaled up to the window.
const STRETCH: &str = r"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;

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

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, soft, o.uv, 0.0);
}
";

/// The backdrop's shader for a target that takes linear light (`linear`)
/// or not: its maths ([`super::look`]), the light's tone
/// ([`crate::gui::bloom::TONE`]) and `backdrop.wgsl`.
pub fn shader(linear: bool) -> String {
    format!(
        "const LINEAR: f32 = {};\n{}{}{}",
        if linear { "1.0" } else { "0.0" },
        super::look::wgsl(),
        crate::gui::bloom::TONE,
        include_str!("backdrop.wgsl")
    )
}

/// How many levels a texture of `size` has.
fn levels((w, h): (u32, u32)) -> u32 {
    (32 - w.max(h).max(1).leading_zeros()).min(LEVELS)
}

/// The pipelines that stretch a texture over their target, one for each
/// target format.
struct Stretch {
    layout: wgpu::BindGroupLayout,
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl Stretch {
    fn new(device: &wgpu::Device) -> Stretch {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom backdrop stretch"),
            entries: &[texture_entry(0), sampler_entry(1)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom backdrop stretch"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom backdrop stretch"),
            source: wgpu::ShaderSource::Wgsl(STRETCH.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom backdrop stretch"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Stretch {
            layout,
            module,
            pipeline_layout,
            sampler,
            pipelines: Vec::new(),
        }
    }

    /// The pipeline drawing into `format`, made the first time.
    fn pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> usize {
        if let Some(i) = self.pipelines.iter().position(|(f, _)| *f == format) {
            return i;
        }
        let pipeline = full_screen(
            device,
            (&self.pipeline_layout, &self.module),
            format,
            None,
            "jodeldiplom backdrop stretch",
        );
        self.pipelines.push((format, pipeline));
        self.pipelines.len() - 1
    }

    /// What it reads to stretch `view`.
    fn bind(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom backdrop stretch"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// `texture`'s levels below the first, each from the one above.
    fn levels(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        let k = self.pipeline(device, texture.format());
        let view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        for level in 1..texture.mip_level_count() {
            let (src, dst) = (view(level - 1), view(level));
            let bind = self.bind(device, &src);
            let mut pass = begin(encoder, &dst);
            pass.set_pipeline(&self.pipelines[k].1);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// A fragment texture's place in a layout.
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A fragment sampler's place in a layout.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// A pipeline of one triangle over its target (`vs` and `fs` of `module`),
/// into `format`, blending as `blend` says.
fn full_screen(
    device: &wgpu::Device,
    (layout, module): (&wgpu::PipelineLayout, &wgpu::ShaderModule),
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// A render pass into `view`, drawn whole (what was there not kept).
fn begin<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("jodeldiplom backdrop"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Texels for a texture: their format, the bytes a texel, the size, the
/// texels row by row.
struct Upload<'a> {
    format: wgpu::TextureFormat,
    texel: u32,
    size: (u32, u32),
    bytes: &'a [u8],
}

/// A texture of `texels`, with `levels` levels (made by `stretch` in
/// `encoder` below the first).
fn texture(
    (device, queue): (&wgpu::Device, &wgpu::Queue),
    (encoder, stretch): (&mut wgpu::CommandEncoder, &mut Stretch),
    texels: &Upload,
    levels: u32,
) -> wgpu::Texture {
    let (w, h) = (texels.size.0.max(1), texels.size.1.max(1));
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    if levels > 1 {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom backdrop"),
        size,
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texels.format,
        usage,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        texels.bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * texels.texel),
            rows_per_image: Some(h),
        },
        size,
    );
    if levels > 1 {
        stretch.levels(device, encoder, &texture);
    }
    texture
}

/// A cover's pieces as its textures take them: the plate (RGBA), the
/// subject (RGBA, its matte; one clear texel without one), the depth
/// (nearness of the cover and of the plate, a byte each) and the rim; all
/// as the covers' thread made them.
fn pieces(apart: &crate::cover::Apart) -> [Upload<'_>; 4] {
    use wgpu::TextureFormat as F;
    let l = &apart.layers;
    let none: &[u8] = &[0; 4];
    let (subject_size, subject) = l
        .subject
        .as_ref()
        .map_or(((1, 1), none), |s| (s.dimensions(), s.as_raw().as_slice()));
    let texels = |format, texel, size, bytes| Upload {
        format,
        texel,
        size,
        bytes,
    };
    [
        texels(
            F::Rgba8UnormSrgb,
            4,
            l.plate.dimensions(),
            &apart.texels.plate,
        ),
        texels(F::Rgba8UnormSrgb, 4, subject_size, subject),
        texels(
            F::Rg8Unorm,
            2,
            (l.depth.width, l.depth.height),
            &apart.texels.depth,
        ),
        texels(F::R8Unorm, 1, l.outline.dimensions(), l.outline.as_raw()),
    ]
}

/// A cover on the GPU: its layers (whose they are), its textures, its
/// numbers and what its shader reads.
struct Cover {
    layers: Layers,
    _textures: Vec<wgpu::Texture>,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// The ground on the GPU: which it is, its texture and what reads it.
struct GroundTexture {
    of: (u64, (u32, u32)),
    _texture: wgpu::Texture,
    bind: wgpu::BindGroup,
}

/// The texture the backdrop is drawn into, and its view.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// What was drawn last: the ground, each cover (by its key) with its
/// numbers, the target's size.
type Drawn = ((u64, (u32, u32)), Vec<(String, Look)>, (u32, u32));

/// The backdrop's pipeline, the ground and the covers on the GPU and the
/// texture it is drawn into.
pub struct BackdropPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    stretch: Stretch,
    ground: Option<GroundTexture>,
    covers: Vec<Cover>,
    target: Option<Target>,
    /// What the window reads while there is none: one black texel.
    _empty: wgpu::Texture,
    empty_view: wgpu::TextureView,
    /// The target's format (and the ground's).
    format: wgpu::TextureFormat,
    /// Changes whenever what [`BackdropPass::under`] gives does.
    generation: u64,
    drawn: Option<Drawn>,
}

impl BackdropPass {
    /// For a window whose pass reads its textures as linear light
    /// (`linear`: through sRGB views) or as they are.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, linear: bool) -> BackdropPass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom backdrop"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(LOOK_BYTES),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                sampler_entry(5),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom backdrop"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom backdrop"),
            source: wgpu::ShaderSource::Wgsl(shader(linear).into()),
        });
        // The ground's and the target's: what the window reads as it reads
        // the pixmap.
        let format = if linear {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        // Each cover over what is under it by its share.
        let blend = wgpu::BlendState::ALPHA_BLENDING;
        let pipeline = full_screen(
            device,
            (&pipeline_layout, &module),
            format,
            Some(blend),
            "jodeldiplom backdrop",
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom backdrop"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let mut stretch = Stretch::new(device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let black = Upload {
            format,
            texel: 4,
            size: (1, 1),
            bytes: &[0, 0, 0, 255],
        };
        let empty = texture((device, queue), (&mut encoder, &mut stretch), &black, 1);
        queue.submit([encoder.finish()]);
        let empty_view = empty.create_view(&wgpu::TextureViewDescriptor::default());
        BackdropPass {
            pipeline,
            layout,
            sampler,
            stretch,
            ground: None,
            covers: Vec::new(),
            target: None,
            _empty: empty,
            empty_view,
            format,
            generation: 0,
            drawn: None,
        }
    }

    /// What the window's pass reads: the backdrop as last drawn, or one
    /// black texel where there is none.
    pub fn under(&self) -> &wgpu::TextureView {
        self.target.as_ref().map_or(&self.empty_view, |t| &t.view)
    }

    /// Changes whenever [`BackdropPass::under`] does.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many covers are on the GPU.
    pub fn held(&self) -> usize {
        self.covers.len()
    }

    /// The next frame's backdrop (`frame`; none: none, everything dropped)
    /// for a window of `size`: the ground and the covers new to it sent
    /// over, the covers gone dropped, and drawn (unless it is the frame
    /// drawn last).
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        frame: Option<&BackdropFrame>,
        size: (u32, u32),
    ) {
        let Some(frame) = frame else {
            if self.target.is_some() {
                self.covers.clear();
                self.ground = None;
                self.target = None;
                self.drawn = None;
                self.generation += 1;
            }
            return;
        };
        let shows = |c: &Cover| frame.slots.iter().any(|s| s.layers.key == c.layers.key);
        self.covers.retain(shows);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom backdrop"),
        });
        if self.ground.as_ref().is_none_or(|g| g.of != frame.ground.of) {
            self.ground = Some(self.ground_texture((device, queue), &mut encoder, &frame.ground));
        }
        for slot in &frame.slots {
            if !self.covers.iter().any(|c| c.layers.key == slot.layers.key) {
                let cover = self.cover((device, queue), &mut encoder, &slot.layers);
                self.covers.push(cover);
            }
        }
        let half = (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1));
        if self.target.as_ref().is_none_or(|t| t.size != half) {
            self.target = Some(self.target(device, half));
            self.drawn = None;
            self.generation += 1;
        }
        let now: Drawn = (
            frame.ground.of,
            frame
                .slots
                .iter()
                .map(|s| (s.layers.key.clone(), s.look))
                .collect(),
            half,
        );
        if self.drawn.as_ref() != Some(&now) {
            self.draw(queue, &mut encoder, frame);
            self.drawn = Some(now);
        }
        queue.submit([encoder.finish()]);
    }

    /// `frame` drawn into the target: the ground stretched over it, then
    /// its covers back to front, each over what is under it by its share.
    fn draw(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, frame: &BackdropFrame) {
        let (Some(target), Some(ground)) = (&self.target, &self.ground) else {
            return;
        };
        let covers: Vec<&Cover> = frame
            .slots
            .iter()
            .filter_map(|s| {
                let cover = self.covers.iter().find(|c| c.layers.key == s.layers.key)?;
                queue.write_buffer(&cover.uniforms, 0, bytemuck::bytes_of(&s.look));
                Some(cover)
            })
            .collect();
        let stretch = &self.stretch.pipelines[self.stretch_at()].1;
        let mut pass = begin(encoder, &target.view);
        pass.set_pipeline(stretch);
        pass.set_bind_group(0, &ground.bind, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.pipeline);
        for cover in covers {
            pass.set_bind_group(0, &cover.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Where the stretching pipeline into the target's format is (made with
    /// the ground's texture, before any frame is drawn).
    fn stretch_at(&self) -> usize {
        let at = self
            .stretch
            .pipelines
            .iter()
            .position(|(f, _)| *f == self.format);
        at.expect("made with the ground")
    }

    /// `ground` sent over.
    fn ground_texture(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        encoder: &mut wgpu::CommandEncoder,
        ground: &Ground,
    ) -> GroundTexture {
        let bytes: Vec<u8> = ground
            .image
            .pixels()
            .flat_map(|p| [p.0[0], p.0[1], p.0[2], 255])
            .collect();
        let texels = Upload {
            format: self.format,
            texel: 4,
            size: ground.image.dimensions(),
            bytes: &bytes,
        };
        let texture = texture((device, queue), (encoder, &mut self.stretch), &texels, 1);
        self.stretch.pipeline(device, self.format);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        GroundTexture {
            of: ground.of,
            bind: self.stretch.bind(device, &view),
            _texture: texture,
        }
    }

    /// `layers` sent over: its textures (their smaller levels made in
    /// `encoder`), its numbers' buffer and what its shader reads.
    fn cover(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        encoder: &mut wgpu::CommandEncoder,
        layers: &Layers,
    ) -> Cover {
        let textures: Vec<wgpu::Texture> = pieces(layers)
            .iter()
            .map(|t| {
                let levels = levels(t.size);
                texture((device, queue), (encoder, &mut self.stretch), t, levels)
            })
            .collect();
        let views: Vec<wgpu::TextureView> = textures
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom backdrop look"),
            size: LOOK_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        }];
        for (k, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + k as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom backdrop"),
            layout: &self.layout,
            entries: &entries,
        });
        Cover {
            layers: Arc::clone(layers),
            _textures: textures,
            uniforms,
            bind,
        }
    }

    /// A target of `size` for the backdrop, the window's pass reading it.
    fn target(&self, device: &wgpu::Device, size: (u32, u32)) -> Target {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom backdrop"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Target {
            _texture: texture,
            view,
            size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cover_s_textures_have_their_levels_down_to_a_few_texels() {
        assert_eq!(levels((1024, 1024)), 9);
        assert_eq!(levels((300, 200)), 9);
        assert_eq!(levels((1, 1)), 1);
        assert_eq!(levels((40, 3)), 6);
    }
}
