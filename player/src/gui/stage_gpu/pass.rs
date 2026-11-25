//! The stage's pass ([`StagePass`]): a pipeline a [`Group`], what they
//! read (the window's numbers, the text's atlas, the tubes' lines, the
//! effects' looks and the music), and a frame's shapes in runs of a group.

use super::groups::{runs, Runs};
use super::{shader, Feed, Group, Instance, ATLAS};
use crate::gui::fx::Fx;
use crate::gui::particles::{uniform_entry, MUSIC_BYTES};

/// How many bytes the effects' looks are: a vec4 an effect.
const LOOKS_BYTES: u64 = 16 * Fx::COUNT as u64;
/// How a pass's shapes go onto what is drawn: over it (premultiplied,
/// covering it as much as they cover), or added to it (light).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// Over it.
    Over,
    /// Added.
    Add,
}

impl Blend {
    /// Its blend, premultiplied.
    pub fn state(self) -> wgpu::BlendState {
        let c = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: match self {
                Blend::Over => wgpu::BlendFactor::OneMinusSrcAlpha,
                Blend::Add => wgpu::BlendFactor::One,
            },
            operation: wgpu::BlendOperation::Add,
        };
        wgpu::BlendState { color: c, alpha: c }
    }
}

/// Whether `format` takes linear light from the shaders: an sRGB one
/// (encoded as it is written) or a float one (light as it is).
pub fn takes_linear(format: wgpu::TextureFormat) -> bool {
    format.is_srgb()
        || matches!(
            format,
            wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
        )
}

/// The stage's pipelines (one a [`Group`], by its code: those it was made
/// with, and any other the first frame that draws one) and what they read.
pub struct StagePass {
    pipelines: Vec<Option<wgpu::RenderPipeline>>,
    /// What a pipeline is made with: its layout, the frame's format and
    /// the blend.
    pipeline_layout: wgpu::PipelineLayout,
    target: (wgpu::TextureFormat, Blend),
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    /// The effects' looks ([`crate::gui::fx::Spec::params`]) and the music
    /// this frame.
    looks: wgpu::Buffer,
    music: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// The text's atlas, made with the first text put into it (a pass
    /// never given any has a texel of it), and its view.
    atlas: Option<wgpu::Texture>,
    atlas_view: wgpu::TextureView,
    /// The tubes' middle lines (a point at least).
    paths: wgpu::Buffer,
    instances: Option<wgpu::Buffer>,
    /// The instances in the buffer for the next frame, in runs of a
    /// group: the shapes, those over everything, the screen passes.
    runs: [Runs; 3],
    linear: bool,
}

/// A texture of the atlas's format, `side` square, and its view.
fn atlas_texture(device: &wgpu::Device, side: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom stage atlas"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
    (atlas, view)
}

/// A buffer of tubes' points, `size` bytes.
fn paths_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jodeldiplom stage paths"),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl StagePass {
    /// For frames of `format`, its shapes going on as `blend` says: the
    /// pipelines of the groups the tiers' rows use made now
    /// ([`Group::used`]), any other the first time one is drawn.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, blend: Blend) -> StagePass {
        StagePass::with(device, (format, blend), &Group::used())
    }

    /// As [`StagePass::new`], with the pipelines of `groups` made now (the
    /// light's: [`Group::light`]).
    pub fn with(
        device: &wgpu::Device,
        (format, blend): (wgpu::TextureFormat, Blend),
        groups: &[Group],
    ) -> StagePass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                uniform_entry(4, LOOKS_BYTES),
                uniform_entry(5, MUSIC_BYTES),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipelines = (0..Group::CODES)
            .map(|code| {
                let group = groups.iter().find(|g| g.code() as usize == code)?;
                Some(make(device, &pipeline_layout, (format, blend), *group))
            })
            .collect();
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom stage"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom stage"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (_, atlas_view) = atlas_texture(device, 1);
        let paths = paths_buffer(device, 16);
        let uniform = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let looks = uniform("jodeldiplom stage looks", LOOKS_BYTES);
        let music = uniform("jodeldiplom stage music", MUSIC_BYTES);
        let reads = Reads {
            layout: &layout,
            uniforms: &uniforms,
            looks: &looks,
            music: &music,
            sampler: &sampler,
        };
        let bind = reads.bind(device, &atlas_view, &paths);
        StagePass {
            pipelines,
            pipeline_layout,
            target: (format, blend),
            layout,
            bind,
            uniforms,
            looks,
            music,
            sampler,
            atlas: None,
            atlas_view,
            paths,
            instances: None,
            runs: Default::default(),
            linear: takes_linear(format),
        }
    }

    /// What it draws next (`None`: nothing), for a window of `size`: the
    /// text put into its atlas, its shapes sent over.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        feed: Option<Feed>,
        size: (u32, u32),
    ) {
        self.runs = Default::default();
        let Some(feed) = feed else {
            return;
        };
        let mut rebind = false;
        if self.atlas.is_none() && !feed.uploads.is_empty() {
            let (atlas, view) = atlas_texture(device, ATLAS);
            (self.atlas, self.atlas_view, rebind) = (Some(atlas), view, true);
        }
        let paths: &[u8] = bytemuck::cast_slice(feed.paths);
        if self.paths.size() < paths.len() as u64 {
            self.paths = paths_buffer(device, (paths.len() as u64).next_power_of_two());
            rebind = true;
        }
        if !paths.is_empty() {
            queue.write_buffer(&self.paths, 0, paths);
        }
        if rebind {
            let reads = Reads {
                layout: &self.layout,
                uniforms: &self.uniforms,
                looks: &self.looks,
                music: &self.music,
                sampler: &self.sampler,
            };
            self.bind = reads.bind(device, &self.atlas_view, &self.paths);
        }
        for up in feed.uploads {
            let (x, y, w, h) = up.rect;
            let Some(atlas) = self.atlas.as_ref().filter(|_| w > 0 && h > 0) else {
                continue;
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &up.coverage,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        if feed.instances.is_empty() && feed.top.is_empty() && feed.screen.is_empty() {
            return;
        }
        let parts = [feed.instances, feed.top, feed.screen];
        let bytes: Vec<u8> = (parts.iter())
            .flat_map(|p| bytemuck::cast_slice::<Instance, u8>(p))
            .copied()
            .collect();
        let fits = self
            .instances
            .as_ref()
            .is_some_and(|b| b.size() >= bytes.len() as u64);
        if !fits {
            self.instances = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom stage"),
                size: (bytes.len() as u64).next_power_of_two().max(4096),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(buffer) = &self.instances {
            queue.write_buffer(buffer, 0, &bytes);
        }
        let mut uniforms = [0u8; 32];
        uniforms[0..4].copy_from_slice(&(size.0 as f32).to_le_bytes());
        uniforms[4..8].copy_from_slice(&(size.1 as f32).to_le_bytes());
        uniforms[8..12].copy_from_slice(&(if self.linear { 1f32 } else { 0.0 }).to_le_bytes());
        uniforms[12..16].copy_from_slice(&(ATLAS as f32).to_le_bytes());
        uniforms[16..20].copy_from_slice(&feed.clock.to_le_bytes());
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        queue.write_buffer(&self.music, 0, bytemuck::cast_slice(&feed.music.uniform()));
        queue.write_buffer(&self.looks, 0, bytemuck::cast_slice(&feed.looks.0));
        let mut from = 0;
        for (k, part) in parts.iter().enumerate() {
            self.runs[k] = runs(part, from);
            from += part.len() as u32;
        }
        let groups = self.runs.iter().flatten().map(|(g, _)| *g);
        for group in groups.collect::<Vec<_>>() {
            let slot = &mut self.pipelines[group.code() as usize];
            if slot.is_none() {
                *slot = Some(make(device, &self.pipeline_layout, self.target, group));
            }
        }
    }

    /// Draws its shapes into `pass` (nothing without any).
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        self.draw_runs(pass, &self.runs[0]);
    }

    /// Draws its shapes over everything into `pass`.
    pub fn draw_top(&self, pass: &mut wgpu::RenderPass) {
        self.draw_runs(pass, &self.runs[1]);
    }

    /// Draws its screen passes into `pass` (nothing without any).
    pub fn draw_screen(&self, pass: &mut wgpu::RenderPass) {
        self.draw_runs(pass, &self.runs[2]);
    }

    fn draw_runs(&self, pass: &mut wgpu::RenderPass, runs: &Runs) {
        let Some(buffer) = self.instances.as_ref().filter(|_| !runs.is_empty()) else {
            return;
        };
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        for (group, range) in runs {
            let Some(pipeline) = &self.pipelines[group.code() as usize] else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.draw(0..4, range.clone());
        }
    }
}

/// What the stage's pipelines read, but for the atlas and the tubes'
/// lines (made again as they grow).
struct Reads<'a> {
    layout: &'a wgpu::BindGroupLayout,
    uniforms: &'a wgpu::Buffer,
    looks: &'a wgpu::Buffer,
    music: &'a wgpu::Buffer,
    sampler: &'a wgpu::Sampler,
}

impl Reads<'_> {
    /// The bind group of these with `atlas` and `paths`.
    fn bind(
        &self,
        device: &wgpu::Device,
        atlas: &wgpu::TextureView,
        paths: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom stage"),
            layout: self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: paths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.looks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.music.as_entire_binding(),
                },
            ],
        })
    }
}

/// `group`'s pipeline: the stage's shader with only its group's code
/// ([`shader`]), into frames of `format` as `blend` says.
fn make(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    (format, blend): (wgpu::TextureFormat, Blend),
    group: Group,
) -> wgpu::RenderPipeline {
    let attributes = wgpu::vertex_attr_array![
        0 => Float32x4, 1 => Float32x4, 2 => Float32x4,
        3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4
    ];
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jodeldiplom stage"),
        source: wgpu::ShaderSource::Wgsl(shader(group).into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("jodeldiplom stage"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Instance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &attributes,
            })],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend.state()),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
