//! The backdrop on the GPU ([`super`]): the ground (the blurred cover, or
//! the plain gradient, small; the Shader tab's sharp picture) in a texture
//! of its own, sent over when it or the window changes, and stretched
//! smoothly over a texture the window's size; or the story's scene showing
//! drawn into it instead
//! ([`super::scenes::gpu`]: its pictures' textures sent over ahead of their
//! cuts, their smaller levels made here). Drawn in a submission of its own
//! before the window's, and kept for the window's pass to read
//! ([`BackdropPass::under`]) pixel for pixel. A frame like the last one is
//! not drawn again.

use super::{BackdropFrame, Ground, Source};

/// The most levels a picture's textures have (a 1024 picture's down to 4).
const LEVELS: u32 = 9;

/// The pass that draws a texture stretched over its whole target, filtered
/// (after [`super::look`]'s maths and [`SMOOTH`]): a texture's smaller
/// levels (`fs`: each the one above, four texels averaged), and the ground
/// scaled up to the window (`fs_smooth`: smoothly, so its few texels show no
/// edge).
const STRETCH: &str = r"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;

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

/// How many levels a texture of `size` has.
pub(super) fn levels((w, h): (u32, u32)) -> u32 {
    (32 - w.max(h).max(1).leading_zeros()).min(LEVELS)
}

/// The pipelines that make a texture's smaller levels, one for each
/// texture format, and what the ground's smooth stretch is made of.
pub(super) struct Stretch {
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
                format!(
                    "{}{}{SMOOTH}{STRETCH}",
                    super::look::BSPLINE,
                    super::look::SCREEN
                )
                .into(),
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
pub(super) fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
pub(super) fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// A pipeline of one triangle over its target (`vs` and the fragment entry
/// `fs` of `module`), into `format`, blending as `blend` says.
pub(super) fn full_screen(
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
pub(super) struct Upload<'a> {
    pub(super) format: wgpu::TextureFormat,
    pub(super) texel: u32,
    pub(super) size: (u32, u32),
    pub(super) bytes: &'a [u8],
}

/// A texture of `texels`, with `levels` levels (made by `stretch` in
/// `encoder` below the first).
pub(super) fn texture(
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

/// The ground on the GPU: which it is, its texture and what reads it.
struct GroundTexture {
    of: (Source, (u32, u32)),
    _texture: wgpu::Texture,
    bind: wgpu::BindGroup,
}

/// The texture the backdrop is drawn into, and its view.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

/// What was drawn last (with the target's size: [`BackdropPass::drawn`]).
#[derive(Clone, Debug, PartialEq)]
enum Drawn {
    /// The story's scene.
    Scene(super::scenes::gpu::Drawn),
    /// The ground (which it is).
    Ground((Source, (u32, u32))),
}

/// The backdrop's passes, the ground and the scenes on the GPU and the
/// texture it is drawn into.
pub struct BackdropPass {
    stretch: Stretch,
    /// The ground stretched smoothly over the target.
    smooth: wgpu::RenderPipeline,
    ground: Option<GroundTexture>,
    /// The story's scenes.
    scenes: super::scenes::gpu::ScenePass,
    target: Option<Target>,
    /// What the window reads while there is none: one black texel.
    _empty: wgpu::Texture,
    empty_view: wgpu::TextureView,
    /// The target's format (and the ground's).
    format: wgpu::TextureFormat,
    /// Changes whenever what [`BackdropPass::under`] gives does.
    generation: u64,
    /// What was drawn last, and the target's size then.
    drawn: Option<(Drawn, (u32, u32))>,
}

impl BackdropPass {
    /// For a window whose pass reads its textures as linear light
    /// (`linear`: through sRGB views) or as they are.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, linear: bool) -> BackdropPass {
        // The ground's and the target's: what the window reads as it reads
        // the pixmap.
        let format = if linear {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
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
            stretch,
            smooth,
            ground: None,
            scenes: super::scenes::gpu::ScenePass::new(device, format, linear),
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

    /// The next frame's backdrop (`frame`; none: none, everything dropped)
    /// for a window of `size`: the ground and its scene's textures new to
    /// it sent over, the scenes gone dropped, and drawn (unless it is the
    /// frame drawn last).
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        frame: Option<&BackdropFrame>,
        size: (u32, u32),
    ) {
        let Some(frame) = frame else {
            if self.target.is_some() {
                self.scenes.clear();
                self.ground = None;
                self.target = None;
                self.drawn = None;
                self.generation += 1;
            }
            return;
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom backdrop"),
        });
        let ground = match self.ground.take() {
            Some(g) if g.of == frame.ground.of => g,
            _ => self.ground_texture((device, queue), &mut encoder, &frame.ground),
        };
        let scene = self.scenes.set(
            (device, queue),
            (&mut encoder, &mut self.stretch),
            frame.scene.as_ref(),
        );
        let size = (size.0.max(1), size.1.max(1));
        let target = match self.target.take() {
            Some(t) if t.size == size => t,
            _ => {
                self.drawn = None;
                self.generation += 1;
                self.target(device, size)
            }
        };
        let now = (scene.map_or(Drawn::Ground(ground.of), Drawn::Scene), size);
        if self.drawn.as_ref() != Some(&now) {
            let mut pass = begin(&mut encoder, &target.view);
            match &now.0 {
                Drawn::Scene(scene) => self.scenes.draw(&mut pass, scene),
                Drawn::Ground(_) => {
                    pass.set_pipeline(&self.smooth);
                    pass.set_bind_group(0, &ground.bind, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
            drop(pass);
            self.drawn = Some(now);
        }
        self.ground = Some(ground);
        self.target = Some(target);
        queue.submit([encoder.finish()]);
    }

    /// `ground` sent over.
    fn ground_texture(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        encoder: &mut wgpu::CommandEncoder,
        ground: &Ground,
    ) -> GroundTexture {
        let bytes = crate::work::picture::rgba(&ground.image);
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
    fn a_picture_s_textures_have_their_levels_down_to_a_few_texels() {
        assert_eq!(levels((1024, 1024)), 9);
        assert_eq!(levels((300, 200)), 9);
        assert_eq!(levels((1, 1)), 1);
        assert_eq!(levels((40, 3)), 6);
    }
}
