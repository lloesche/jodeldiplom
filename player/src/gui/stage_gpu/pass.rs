//! The stage's pass ([`StagePass`]): a pipeline a [`Group`], what they
//! read (the window's numbers, the text's atlas, the tubes' lines, the
//! effects' looks, the music and the diploma's paper baked), and a frame's
//! shapes in runs of a group.

use super::bake::Bake;
use super::groups::{runs, Runs};
use super::{shader, Feed, Group, Instance, Screen, ATLAS, SCREEN_BYTES};
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
    /// The diploma's paper baked, where it draws one ([`Group::Paper`]),
    /// and the bake's generation the bind group reads.
    bake: Option<Bake>,
    bound: u64,
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
                        min_binding_size: wgpu::BufferSize::new(SCREEN_BYTES),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
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
            size: SCREEN_BYTES,
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
        let bake = (groups.contains(&Group::Paper)).then(|| Bake::new(device, &pipeline_layout));
        let reads = Reads {
            layout: &layout,
            looks: &looks,
            music: &music,
            sampler: &sampler,
            atlas: &atlas_view,
            paths: &paths,
        };
        let bind = reads.bind(device, &uniforms, None);
        let bound = bake.as_ref().map_or(0, Bake::generation);
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
            bake,
            bound,
        }
    }

    /// What it draws next (`None`: nothing, the paper's bake let go), for a
    /// window of `size`: its numbers, the text put into its atlas, the
    /// diploma's paper read from its bake where it holds still
    /// ([`super::bake`]), its shapes sent over.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        feed: Option<Feed>,
        size: (u32, u32),
    ) {
        self.runs = Default::default();
        let Some(feed) = feed else {
            if let Some(bake) = &mut self.bake {
                bake.no_stage((device, queue));
            }
            self.rebind(device, false);
            return;
        };
        let mut grown = false;
        if self.atlas.is_none() && !feed.uploads.is_empty() {
            let (atlas, view) = atlas_texture(device, ATLAS);
            (self.atlas, self.atlas_view, grown) = (Some(atlas), view, true);
        }
        let paths: &[u8] = bytemuck::cast_slice(feed.paths);
        if self.paths.size() < paths.len() as u64 {
            self.paths = paths_buffer(device, (paths.len() as u64).next_power_of_two());
            grown = true;
        }
        if !paths.is_empty() {
            queue.write_buffer(&self.paths, 0, paths);
        }
        let screen = Screen {
            size: [size.0 as f32, size.1 as f32],
            srgb: if self.linear { 1.0 } else { 0.0 },
            atlas: ATLAS as f32,
            time: feed.clock,
            shake: feed.shake,
            ..Screen::default()
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&screen));
        queue.write_buffer(&self.music, 0, bytemuck::cast_slice(&feed.music.uniform()));
        queue.write_buffer(&self.looks, 0, bytemuck::cast_slice(&feed.looks.0));
        let reads = Reads {
            layout: &self.layout,
            looks: &self.looks,
            music: &self.music,
            sampler: &self.sampler,
            atlas: &self.atlas_view,
            paths: &self.paths,
        };
        let baked = (self.bake.as_mut())
            .and_then(|bake| bake.frame((device, queue), (feed.instances, &screen, size), &reads));
        self.rebind(device, grown);
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
        // The paper read from its bake in its place.
        let mut shapes = std::borrow::Cow::Borrowed(feed.instances);
        if let Some(baked) = baked {
            shapes.to_mut()[baked.at] = baked.shape;
        }
        let parts = [&shapes[..], feed.top, feed.screen];
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

    /// What the pipelines read made again where the atlas or the tubes'
    /// lines `grown`, or the paper's bake is another.
    fn rebind(&mut self, device: &wgpu::Device, grown: bool) {
        let generation = self.bake.as_ref().map_or(0, Bake::generation);
        if !grown && generation == self.bound {
            return;
        }
        let reads = Reads {
            layout: &self.layout,
            looks: &self.looks,
            music: &self.music,
            sampler: &self.sampler,
            atlas: &self.atlas_view,
            paths: &self.paths,
        };
        let baked = self.bake.as_ref().and_then(Bake::view);
        self.bind = reads.bind(device, &self.uniforms, baked);
        self.bound = generation;
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

/// What the stage's pipelines read but their numbers and the paper's
/// bake: the atlas and the tubes' lines as they are now (made again as
/// they grow).
pub(super) struct Reads<'a> {
    layout: &'a wgpu::BindGroupLayout,
    looks: &'a wgpu::Buffer,
    music: &'a wgpu::Buffer,
    sampler: &'a wgpu::Sampler,
    atlas: &'a wgpu::TextureView,
    paths: &'a wgpu::Buffer,
}

impl Reads<'_> {
    /// The bind group of these with `uniforms` (the frame's numbers) and
    /// `baked` (the paper's bake; none: the atlas stands for it).
    pub(super) fn bind(
        &self,
        device: &wgpu::Device,
        uniforms: &wgpu::Buffer,
        baked: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom stage"),
            layout: self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.paths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.looks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.music.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(baked.unwrap_or(self.atlas)),
                },
            ],
        })
    }
}

/// `group`'s pipeline: the stage's shader with only its group's code
/// ([`shader`]), into frames of `format` as `blend` says.
pub(super) fn make(
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
