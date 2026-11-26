//! The backdrop on the GPU ([`super`]): the ground (the blurred cover,
//! small) in a texture of its own, sent over when the cover or the window
//! changes; each cover showing has its pieces in textures (the plate, the
//! subject with its matte, their depth, the subject's rim), their texels
//! made on the covers' thread ([`crate::work::cover::Texels`]) and sent over
//! once when it first shows, with the picture they were taken apart from (the
//! song's cover, where it is at hand: the card at rest is the real cover, the
//! plate only where the moving subject uncovers it or the blur round a
//! subject cut out would carry its colours), their smaller levels
//! made here (the blur under the card and round the subject,
//! the subject's shadow and a small card read them), and dropped
//! once it no longer shows. A frame draws the ground stretched smoothly
//! over a texture the window's size, then each cover's shader
//! (`backdrop.wgsl`, its maths in [`super::look`]) over it by its share, in
//! a submission of its own before the window's, and keeps it for the
//! window's pass to read ([`BackdropPass::under`]) pixel for pixel. A frame
//! like the last one is not drawn again.

use std::sync::Arc;

use super::{BackdropFrame, Ground, Look, SlotFrame};
use crate::work::cover::Layers;

/// The most levels a cover's textures have (a 1024 cover's down to 4).
const LEVELS: u32 = 9;
/// The bytes of a cover's numbers.
const LOOK_BYTES: u64 = std::mem::size_of::<Look>() as u64;

/// The pass that draws a texture stretched over its whole target, filtered
/// (after [`super::look`]'s maths and [`SMOOTH`]): a texture's smaller
/// levels (`fs`: each the one above, four texels averaged), and the ground
/// scaled up to the window (`fs_smooth`: smoothly, so its few texels show no
/// edge).
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

@fragment
fn fs_smooth(o: Out) -> @location(0) vec4<f32> {
    return smooth_at(src, o.uv, 0u);
}
";

/// A texture read smoothly (after [`super::look`]'s [`super::look::bspline`],
/// with the sampler `soft`): `t` at `uv`, a cubic B-spline over its level
/// `at` (its last where it has fewer) in four bilinear reads, so a level
/// much enlarged shows no texel's edge.
pub(crate) const SMOOTH: &str = r"
fn smooth_at(t: texture_2d<f32>, uv: vec2<f32>, at: u32) -> vec4<f32> {
    let level = min(at, textureNumLevels(t) - 1u);
    let size = vec2<f32>(textureDimensions(t, level));
    let p = uv * size - vec2<f32>(0.5);
    let i = floor(p);
    let f = p - i;
    let bx = bspline(f.x);
    let by = bspline(f.y);
    let x = (vec2<f32>(i.x + 0.5) + bx.zw) / size.x;
    let y = (vec2<f32>(i.y + 0.5) + by.zw) / size.y;
    let lv = f32(level);
    let top = bx.x * textureSampleLevel(t, soft, vec2<f32>(x.x, y.x), lv)
        + bx.y * textureSampleLevel(t, soft, vec2<f32>(x.y, y.x), lv);
    let bottom = bx.x * textureSampleLevel(t, soft, vec2<f32>(x.x, y.y), lv)
        + bx.y * textureSampleLevel(t, soft, vec2<f32>(x.y, y.y), lv);
    return by.x * top + by.y * bottom;
}
";

/// The backdrop's shader for a target that takes linear light (`linear`)
/// or not: its maths ([`super::look`]), the light's tone
/// ([`crate::gui::bloom::TONE`]), the smooth read ([`SMOOTH`]) and
/// `backdrop.wgsl`.
pub fn shader(linear: bool) -> String {
    format!(
        "const LINEAR: f32 = {};\n{}{}{SMOOTH}{}",
        if linear { "1.0" } else { "0.0" },
        super::look::wgsl(),
        crate::gui::bloom::TONE,
        include_str!("backdrop.wgsl")
    )
}

/// How many levels a texture of `size` has.
pub(super) fn levels((w, h): (u32, u32)) -> u32 {
    (32 - w.max(h).max(1).leading_zeros()).min(LEVELS)
}

/// The pipelines that make a texture's smaller levels, one for each
/// texture format, and what the ground's smooth stretch is made of.
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
            source: wgpu::ShaderSource::Wgsl(
                format!("{}{SMOOTH}{STRETCH}", super::look::wgsl()).into(),
            ),
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

    /// The pipeline making levels of `format`, made the first time.
    fn pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> usize {
        if let Some(i) = self.pipelines.iter().position(|(f, _)| *f == format) {
            return i;
        }
        let pipeline = self.stretching(device, (format, "fs"));
        self.pipelines.push((format, pipeline));
        self.pipelines.len() - 1
    }

    /// A pipeline stretching a texture over a target of `format` by the
    /// fragment entry `fs`.
    fn stretching(
        &self,
        device: &wgpu::Device,
        target: (wgpu::TextureFormat, &str),
    ) -> wgpu::RenderPipeline {
        full_screen(
            device,
            (&self.pipeline_layout, &self.module),
            target,
            None,
            "jodeldiplom backdrop stretch",
        )
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

/// A pipeline of one triangle over its target (`vs` and the fragment entry
/// `fs` of `module`), into `format`, blending as `blend` says.
fn full_screen(
    device: &wgpu::Device,
    (layout, module): (&wgpu::PipelineLayout, &wgpu::ShaderModule),
    (format, fs): (wgpu::TextureFormat, &str),
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
            entry_point: Some(fs),
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
/// (nearness of the cover and of the plate, a byte each), the subject's
/// rim (a byte; one dark texel without a subject); all as the covers'
/// thread made them.
fn pieces(apart: &crate::work::cover::Apart) -> [Upload<'_>; 4] {
    use wgpu::TextureFormat as F;
    let l = &apart.layers;
    let none: &[u8] = &[0; 4];
    let (subject_size, subject) = l
        .subject
        .as_ref()
        .map_or(((1, 1), none), |s| (s.dimensions(), s.as_raw().as_slice()));
    let (rim_size, rim) = match l.subject {
        Some(_) => (l.outline.dimensions(), l.outline.as_raw().as_slice()),
        None => ((1, 1), &none[..1]),
    };
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
        texels(F::R8Unorm, 1, rim_size, rim),
    ]
}

/// A cover on the GPU: its layers (whose they are) and the picture they
/// were taken apart from (its id, none where it is not at hand), its
/// textures, its numbers and what its shader reads.
struct Cover {
    layers: Layers,
    art: Option<u64>,
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

/// What was drawn last: the ground, each cover (by its key and its
/// picture's) with its numbers, the target's size.
type Drawn = ((u64, (u32, u32)), Vec<(Held, Look)>, (u32, u32));

/// Which cover a cover on the GPU is: its layers' key, its picture's id.
type Held = (String, Option<u64>);

/// Which cover `slot` shows.
fn held(slot: &SlotFrame) -> Held {
    (slot.layers.key.clone(), slot.art.as_ref().map(|a| a.id))
}

impl Cover {
    /// Whether it is the cover `slot` shows.
    fn is(&self, slot: &SlotFrame) -> bool {
        self.layers.key == slot.layers.key && self.art == slot.art.as_ref().map(|a| a.id)
    }
}

/// The backdrop's pipeline, the ground and the covers on the GPU and the
/// texture it is drawn into.
pub struct BackdropPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    stretch: Stretch,
    /// The ground stretched smoothly over the target.
    smooth: wgpu::RenderPipeline,
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
                sampler_entry(4),
                texture_entry(5),
                texture_entry(6),
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
            (format, "fs"),
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
        let smooth = stretch.stretching(device, (format, "fs_smooth"));
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
            smooth,
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

    /// Whether there is a backdrop to read ([`BackdropPass::under`]).
    pub fn shows(&self) -> bool {
        self.target.is_some()
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
        self.covers.retain(|c| frame.slots.iter().any(|s| c.is(s)));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom backdrop"),
        });
        if self.ground.as_ref().is_none_or(|g| g.of != frame.ground.of) {
            self.ground = Some(self.ground_texture((device, queue), &mut encoder, &frame.ground));
        }
        for slot in &frame.slots {
            if !self.covers.iter().any(|c| c.is(slot)) {
                let cover = self.cover((device, queue), &mut encoder, slot);
                self.covers.push(cover);
            }
        }
        let size = (size.0.max(1), size.1.max(1));
        if self.target.as_ref().is_none_or(|t| t.size != size) {
            self.target = Some(self.target(device, size));
            self.drawn = None;
            self.generation += 1;
        }
        let now: Drawn = (
            frame.ground.of,
            frame.slots.iter().map(|s| (held(s), s.look)).collect(),
            size,
        );
        if self.drawn.as_ref() != Some(&now) {
            self.draw(queue, &mut encoder, frame);
            self.drawn = Some(now);
        }
        queue.submit([encoder.finish()]);
    }

    /// `frame` drawn into the target: the ground stretched smoothly over
    /// it, then its covers back to front, each over what is under it by its
    /// share.
    fn draw(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, frame: &BackdropFrame) {
        let (Some(target), Some(ground)) = (&self.target, &self.ground) else {
            return;
        };
        let covers: Vec<&Cover> = frame
            .slots
            .iter()
            .filter_map(|s| {
                let cover = self.covers.iter().find(|c| c.is(s))?;
                queue.write_buffer(&cover.uniforms, 0, bytemuck::bytes_of(&s.look));
                Some(cover)
            })
            .collect();
        let mut pass = begin(encoder, &target.view);
        pass.set_pipeline(&self.smooth);
        pass.set_bind_group(0, &ground.bind, &[]);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.pipeline);
        for cover in covers {
            pass.set_bind_group(0, &cover.bind, &[]);
            pass.draw(0..3, 0..1);
        }
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        GroundTexture {
            of: ground.of,
            bind: self.stretch.bind(device, &view),
            _texture: texture,
        }
    }

    /// `slot`'s cover sent over: its layers' textures and the picture's (the
    /// plate's again where it is not at hand; their smaller levels made in
    /// `encoder`), its numbers' buffer and what its shader reads. The
    /// picture is made RGBA here, once a cover.
    fn cover(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        encoder: &mut wgpu::CommandEncoder,
        slot: &SlotFrame,
    ) -> Cover {
        let layers = &slot.layers;
        let rgba: Option<Vec<u8>> = slot.art.as_ref().map(|a| {
            a.image
                .pixels()
                .flat_map(|p| [p.0[0], p.0[1], p.0[2], 255])
                .collect()
        });
        let original = slot
            .art
            .as_ref()
            .zip(rgba.as_deref())
            .map(|(a, bytes)| Upload {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                texel: 4,
                size: a.image.dimensions(),
                bytes,
            });
        let textures: Vec<wgpu::Texture> = pieces(layers)
            .iter()
            .chain(original.as_ref())
            .map(|t| {
                let levels = levels(t.size);
                texture((device, queue), (encoder, &mut self.stretch), t, levels)
            })
            .collect();
        // The plate, the subject, the depth, the rim; the picture.
        let mut views: Vec<wgpu::TextureView> = textures
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        if original.is_none() {
            views.push(textures[0].create_view(&wgpu::TextureViewDescriptor::default()));
        }
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
        for (view, binding) in views.iter().zip([1, 2, 3, 6, 5]) {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom backdrop"),
            layout: &self.layout,
            entries: &entries,
        });
        Cover {
            layers: Arc::clone(layers),
            art: slot.art.as_ref().map(|a| a.id),
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
