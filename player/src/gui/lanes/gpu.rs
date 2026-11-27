//! The lanes on the GPU: one pipeline for every lane ([`Pipeline`]), a
//! primitive a lane ([`Lane`]). Each lane's tiles are in a texture array of
//! its own (tile `t` in layer `t` modulo its slots), kept by the lane's
//! place in the list and let go of when there are fewer lanes; each frame
//! sends over only the tiles newly painted and where the lane stands. The
//! shader reads the lane's two columns either side of where the pixel falls
//! (the view moved by the share of one), draws the playhead over them and
//! rounds the card's corners: the same for every pixel of every lane.

use std::sync::{Arc, Mutex};

use iced::widget::shader::{self, Viewport};
use iced::Rectangle;

use super::tiles::{LaneFrame, TILE};
use super::Shared;
use crate::view::palette;
use jodel_looks::math::srgb_to_linear;

const SHADER: &str = r"
struct Lane {
    // Where it goes in the window: x, y, width, height (pixels).
    rect: vec4<f32>,
    // The playhead's colour.
    head: vec4<f32>,
    // The column at its left edge, the share of the next, the tiles'
    // width, the slots.
    place: vec4<f32>,
    // The playhead's left edge and width (0: none), the corners' radius.
    mark: vec4<f32>,
};

@group(0) @binding(0) var tiles: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> lane: Lane;

// One triangle over the viewport (the lane's bounds).
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Column `col` of the lane's grid, row `y`: from the slot of its tile.
fn column(col: i32, y: i32) -> vec4<f32> {
    let tile = i32(lane.place.z);
    let slots = i32(lane.place.w);
    let t = select(col / tile, (col + 1) / tile - 1, col < 0);
    let slot = ((t % slots) + slots) % slots;
    return textureLoad(tiles, vec2<i32>(col - t * tile, y), slot, 0);
}

@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let p = at.xy - lane.rect.xy;
    let x = floor(p.x);
    let y = i32(floor(p.y));
    let col = i32(lane.place.x) + i32(x);
    var c = mix(column(col, y).rgb, column(col + 1, y).rgb, lane.place.y);
    // The card's corners: how much of the pixel is inside them.
    let half = lane.rect.zw * 0.5;
    let r = min(lane.mark.z, min(half.x, half.y));
    let q = abs(floor(p) + vec2<f32>(0.5) - half) - (half - vec2<f32>(r));
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    // The playhead over all of it, corners and all: never cut.
    let head = floor(lane.mark.x);
    let on = x >= head && x < head + lane.mark.y;
    c = select(c, lane.head.rgb, on);
    let cover = select(clamp(0.5 - d, 0.0, 1.0), 1.0, on);
    return vec4<f32>(c * cover, cover);
}
";

/// The lane's uniforms: four vectors of four floats.
const UNIFORM_BYTES: u64 = 64;

/// The tiles' words (`0x00RRGGBB`) as they lie in memory, B G R x.
const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// A lane's program and primitive: its place in the list (`id`) of
/// `count` lanes, its frame taken from what the lanes share, its corners'
/// radius. It takes no events.
#[derive(Clone, Debug)]
pub(super) struct Lane {
    pub(super) shared: Arc<Mutex<Shared>>,
    pub(super) id: usize,
    pub(super) count: usize,
    pub(super) radius: f32,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("lanes", &self.lanes.len())
            .field("held", &self.held)
            .finish()
    }
}

/// A lane's tiles and uniforms on the GPU.
struct Held {
    tiles: wgpu::Texture,
    /// Its tiles' height and its slots.
    size: (u32, u32),
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// The lanes' pipeline, and each lane's tiles by its place in the list.
pub struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The tiles' format as the shader reads them: sRGB where the frame
    /// is, so the colours come out as they went in.
    view_format: wgpu::TextureFormat,
    lanes: Vec<Option<Held>>,
}

impl shader::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Pipeline {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom lanes"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom lanes"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom lanes"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom lanes"),
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
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Pipeline {
            pipeline,
            layout,
            view_format: if format.is_srgb() {
                wgpu::TextureFormat::Bgra8UnormSrgb
            } else {
                TEXTURE_FORMAT
            },
            lanes: Vec::new(),
        }
    }
}

impl Pipeline {
    /// Lane `id`'s tiles for `frame` (made where it has none of their
    /// size), the new ones sent over, and where it stands.
    fn take(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        id: usize,
        frame: &mut LaneFrame,
        radius: f32,
    ) {
        let size = (frame.rect.3.max(1) as u32, frame.slots.max(1) as u32);
        if self.lanes.len() <= id {
            self.lanes.resize_with(id + 1, || None);
        }
        if self.lanes[id].as_ref().is_none_or(|h| h.size != size) {
            self.lanes[id] = Some(self.held(device, size));
        }
        let Some(held) = &self.lanes[id] else {
            return;
        };
        for (slot, pixels) in std::mem::take(&mut frame.uploads) {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &held.tiles,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: slot as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TILE as u32 * 4),
                    rows_per_image: Some(size.0),
                },
                wgpu::Extent3d {
                    width: TILE as u32,
                    height: size.0,
                    depth_or_array_layers: 1,
                },
            );
        }
        // The playhead's colour as the frame's texels come out: linear
        // where they are read through an sRGB view.
        let srgb = self.view_format.is_srgb();
        let head = palette::ACCENT.map(|v| {
            let c = f32::from(v) / 255.0;
            if srgb {
                srgb_to_linear(c)
            } else {
                c
            }
        });
        let (x, y, w, h) = frame.rect;
        let (head_x, head_w) = frame.playhead.map_or((0.0, 0.0), |(x, w)| (x, w as f32));
        let uniforms: [f32; 16] = [
            x as f32,
            y as f32,
            w as f32,
            h as f32,
            head[0],
            head[1],
            head[2],
            1.0,
            frame.offset as f32,
            frame.frac,
            TILE as f32,
            frame.slots as f32,
            head_x,
            head_w,
            radius,
            0.0,
        ];
        queue.write_buffer(&held.uniforms, 0, bytemuck::cast_slice(&uniforms));
    }

    /// A lane's tiles of `(height, slots)` and its uniforms.
    fn held(&self, device: &wgpu::Device, (height, slots): (u32, u32)) -> Held {
        let tiles = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom lane"),
            size: wgpu::Extent3d {
                width: TILE as u32,
                height,
                depth_or_array_layers: slots,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: if self.view_format == TEXTURE_FORMAT {
                &[]
            } else {
                &[wgpu::TextureFormat::Bgra8UnormSrgb]
            },
        });
        let view = tiles.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom lane"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom lane"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        });
        Held {
            tiles,
            size: (height, slots),
            uniforms,
            bind,
        }
    }

    /// Lane `id` into `pass` (its viewport the lane's bounds), where it
    /// has tiles.
    fn draw(&self, id: usize, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(Some(held)) = self.lanes.get(id) {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &held.bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// The lanes from `count` on let go of.
    fn trim(&mut self, count: usize) {
        self.lanes.truncate(count);
    }
}

impl<M> shader::Program<M> for Lane {
    type State = ();
    type Primitive = Lane;

    fn draw(&self, _state: &(), _cursor: iced::mouse::Cursor, _bounds: Rectangle) -> Lane {
        self.clone()
    }
}

impl shader::Primitive for Lane {
    type Pipeline = Pipeline;

    fn prepare(
        &self,
        pipeline: &mut Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        // The first lane's (or none's) lets go of the lanes gone.
        if self.id == 0 {
            pipeline.trim(self.count);
            shared.held = !pipeline.lanes.is_empty();
        }
        if let Some(frame) = shared.lanes.get_mut(self.id).filter(|_| self.count > 0) {
            pipeline.take((device, queue), self.id, frame, self.radius);
            shared.held = true;
        }
    }

    fn draw(&self, pipeline: &Pipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        pipeline.draw(self.id, pass);
        true
    }
}

/// The lanes drawn off screen as the window draws them, on the window's
/// ground, and read back: for the tests.
pub struct Offscreen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: Pipeline,
    format: wgpu::TextureFormat,
    /// What draws: the adapter's name and backend.
    pub name: String,
}

impl Offscreen {
    /// On the process's GPU, into frames that take the bytes as they are
    /// (`srgb`: that encode them as sRGB); an error where there is no GPU.
    pub fn new(srgb: bool) -> Result<Offscreen, String> {
        let _turn = jodel_gpu::setting_up();
        let adapter = jodel_gpu::chosen().adapter.clone()?;
        let (adapter, device, queue) = crate::gui::present::open(adapter, wgpu::Features::empty())?;
        let info = adapter.get_info();
        let format = if srgb {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            TEXTURE_FORMAT
        };
        let pipeline = <Pipeline as shader::Pipeline>::new(&device, &queue, format);
        Ok(Offscreen {
            device,
            queue,
            pipeline,
            format,
            name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    /// `lanes`' last frame ([`super::Lanes::frame`], on the GPU) in a
    /// window of `width` × `height` pixels, read back as `0x00RRGGBB`
    /// words row by row: the window's ground, each lane over it.
    pub fn show(
        &mut self,
        lanes: &super::Lanes,
        width: u32,
        height: u32,
    ) -> Result<Vec<u32>, String> {
        let count = lanes.places.len();
        let radius = lanes.radius();
        let prims: Vec<Lane> = (0..count.max(1))
            .map(|id| Lane {
                shared: Arc::clone(&lanes.shared),
                id,
                count,
                radius: if lanes.strips.get(id).is_some_and(|s| s.is_wave()) {
                    radius
                } else {
                    0.0
                },
            })
            .collect();
        let viewport = Viewport::with_physical_size(
            iced::Size::new(width, height),
            iced::advanced::renderer::Scale::default(),
        );
        for p in &prims {
            shader::Primitive::prepare(
                p,
                &mut self.pipeline,
                &self.device,
                &self.queue,
                &Rectangle::default(),
                &viewport,
            );
        }
        let pipeline = &self.pipeline;
        let [r, g, b] = palette::BG.map(|v| {
            let c = f64::from(v) / 255.0;
            if self.format.is_srgb() {
                f64::from(srgb_to_linear(c as f32))
            } else {
                c
            }
        });
        crate::gui::present::read_back(
            (&self.device, &self.queue),
            self.format,
            (width, height),
            wgpu::Color { r, g, b, a: 1.0 },
            |pass| {
                for (p, &(x, y, w, h)) in prims.iter().zip(&lanes.places) {
                    pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                    shader::Primitive::draw(p, pipeline, pass);
                }
            },
        )
    }

    /// How many lanes' tiles the GPU holds.
    pub fn held(&self) -> usize {
        self.pipeline.lanes.iter().flatten().count()
    }
}
