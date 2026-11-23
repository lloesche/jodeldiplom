//! The window's frames through the GPU: the pixmap is kept in a texture
//! the size of the window, each frame sends over only the pixels that
//! changed ([`Changes`]), and a triangle over the whole window copies the
//! texture onto it, pixel for pixel. Handing a whole window's buffer to
//! the system each frame (as softbuffer does, and on macOS with a new
//! buffer every time) costs more than drawing it; this costs a few rows.
//!
//! A lane that follows the playhead comes as a layer ([`LayerFrame`]): its
//! tiles kept in a texture array, drawn over the pixmap at the column (and
//! share of one) the frame says, with the playhead over it. A frame that
//! moves only the layer sends over nothing but its place.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use jodel_gpu::Turned;
use winit::window::Window;

use super::canvas::{Canvas, Changes, LayerFrame, PixelRect, MOST_SLOTS};

const SHADER: &str = r"
struct Layer {
    // Where it goes in the window: x, y, width, height.
    rect: vec4<f32>,
    // The playhead's colour.
    head: vec4<f32>,
    // The column at its left edge, the tiles' width, the slots, on (1).
    offset: i32,
    tile: i32,
    slots: i32,
    on: i32,
    // The share of the next column, the playhead's left edge across it,
    // its width, and the row it starts at.
    frac: f32,
    head_x: f32,
    head_w: f32,
    head_top: f32,
    // The tile each slot holds (-1: none), four a vector.
    resident: array<vec4<i32>, 4>,
};

@group(0) @binding(0) var pixels: texture_2d<f32>;
@group(0) @binding(1) var tiles: texture_2d_array<f32>;
@group(0) @binding(2) var<uniform> layer: Layer;

// One triangle that covers the window.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Column `col` of the layer, row `y`: from the slot holding its tile;
// `under` where none does yet.
fn column(col: i32, y: i32, under: vec4<f32>) -> vec4<f32> {
    let t = col / layer.tile;
    let slot = t % layer.slots;
    let held = layer.resident[slot / 4];
    if col < 0 || held[slot % 4] != t {
        return under;
    }
    return textureLoad(tiles, vec2<i32>(col - t * layer.tile, y), slot, 0);
}

// Each window pixel is the texture's pixel there; in the layer's rect, the
// layer's, between two columns by the share of one it is moved, and the
// playhead over it.
@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(at.xy);
    let c = textureLoad(pixels, p, 0);
    if layer.on == 0 {
        return vec4<f32>(c.rgb, 1.0);
    }
    let x = f32(p.x) - layer.rect.x;
    let y = f32(p.y) - layer.rect.y;
    if x < 0.0 || y < 0.0 || x >= layer.rect.z || y >= layer.rect.w {
        return vec4<f32>(c.rgb, 1.0);
    }
    let col = layer.offset + i32(x);
    let a = column(col, i32(y), c);
    let b = column(col + 1, i32(y), c);
    var out = mix(a.rgb, b.rgb, layer.frac);
    let head = floor(layer.head_x);
    if layer.head_w > 0.0 && x >= head && x < head + layer.head_w && y >= layer.head_top {
        out = layer.head.rgb;
    }
    return vec4<f32>(out, 1.0);
}
";

/// The layer's uniforms' size in bytes.
const LAYER_BYTES: u64 = 128;

/// The pixmap's words (`0x00RRGGBB`) as they lie in memory, B G R x.
const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// How a frame went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shown {
    /// It is on its way to the window.
    Yes,
    /// The window took none now (covered, or a frame timed out); the
    /// texture has the changes all the same.
    NotNow,
}

/// The pixmap's copy on the GPU, for one window size.
struct Kept {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The window size it is for.
    window: (u32, u32),
    /// The pixmap size it last took.
    pixmap: (usize, usize),
}

/// The pixmap kept on the GPU and the pass that copies it onto a frame.
struct Painter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The texture's format as the shader reads it: sRGB where the frame
    /// is, so the colours come out as they went in.
    view_format: wgpu::TextureFormat,
    kept: Option<Kept>,
    /// The layer's tiles (none yet: one slot of one pixel), their size
    /// (width, height, slots), and its uniforms.
    tiles: wgpu::Texture,
    tiles_view: wgpu::TextureView,
    tiles_size: (u32, u32, u32),
    uniforms: wgpu::Buffer,
    /// What the pass reads: made again when a texture is.
    bind: Option<wgpu::BindGroup>,
    /// Karaoke mode's stage, drawn over it all, and its bursts over that.
    stage: super::stage_gpu::StagePass,
    particles: super::particles::ParticlePass,
}

/// An adapter and a device opened on it.
type Opened = (wgpu::Adapter, wgpu::Device, wgpu::Queue);

/// A device on `adapter` (not a software renderer), with limits for a
/// texture the size of any window.
pub(super) fn open(adapter: wgpu::Adapter) -> Result<Opened, String> {
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("{}: a software renderer", info.name));
    }
    let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jodeldiplom window"),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| format!("{}: {e}", info.name))?;
    Ok((adapter, device, queue))
}

impl Painter {
    /// For frames of `format`.
    fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat) -> Painter {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom window"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(LAYER_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom window"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom window"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom window"),
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
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let view_format = if format.is_srgb() {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            TEXTURE_FORMAT
        };
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom layer"),
            size: LAYER_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (tiles, tiles_view) = tiles_texture(&device, view_format, (1, 1, 1));
        let stage = super::stage_gpu::StagePass::new(&device, format);
        let particles = super::particles::ParticlePass::new(&device, format);
        Painter {
            device,
            queue,
            pipeline,
            layout,
            view_format,
            kept: None,
            tiles,
            tiles_view,
            tiles_size: (1, 1, 1),
            uniforms,
            bind: None,
            stage,
            particles,
        }
    }

    /// Karaoke mode's stage for the next frame (`None`: none), in a frame
    /// of `size`.
    fn set_stage(&mut self, frame: Option<&super::stage_gpu::StageFrame>, size: (u32, u32)) {
        self.stage.set(&self.device, &self.queue, frame, size);
        self.particles.set(&self.device, &self.queue, frame, size);
    }

    /// The layer for the next frame (`None`: none): its new tiles sent
    /// over, and where it stands.
    fn set_layer(&mut self, frame: Option<LayerFrame>) {
        let mut bytes = [0u8; LAYER_BYTES as usize];
        if let Some(frame) = frame {
            let size = (frame.tile as u32, frame.rect.3 as u32, frame.slots as u32);
            if size != self.tiles_size {
                (self.tiles, self.tiles_view) = tiles_texture(&self.device, self.view_format, size);
                self.tiles_size = size;
                self.bind = None;
            }
            for (slot, pixels) in &frame.uploads {
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.tiles,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: *slot as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(pixels),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size.0 * 4),
                        rows_per_image: Some(size.1),
                    },
                    wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                );
            }
            let mut put = |at: usize, word: [u8; 4]| bytes[at..at + 4].copy_from_slice(&word);
            let (x, y, w, h) = frame.rect;
            for (i, v) in [x, y, w, h].into_iter().enumerate() {
                put(4 * i, (v as f32).to_le_bytes());
            }
            // The playhead's colour as the frame's texels come out: linear
            // where they are read through an sRGB view.
            let srgb = self.view_format.is_srgb();
            for (i, v) in frame.head.into_iter().enumerate() {
                let c = f32::from(v) / 255.0;
                let c = if srgb { srgb_to_linear(c) } else { c };
                put(16 + 4 * i, c.to_le_bytes());
            }
            put(28, 1f32.to_le_bytes());
            put(32, (frame.offset as i32).to_le_bytes());
            put(36, (frame.tile as i32).to_le_bytes());
            put(40, (frame.slots as i32).to_le_bytes());
            put(44, 1i32.to_le_bytes());
            put(48, frame.frac.to_le_bytes());
            let (head_x, head_w) = frame.playhead.map_or((0.0, 0.0), |(x, w)| (x, w as f32));
            put(52, head_x.to_le_bytes());
            put(56, head_w.to_le_bytes());
            put(60, (frame.head_top as f32).to_le_bytes());
            for slot in 0..MOST_SLOTS {
                let tile = frame
                    .resident
                    .get(slot)
                    .copied()
                    .flatten()
                    .map_or(-1, |t| t as i32);
                put(64 + 4 * slot, tile.to_le_bytes());
            }
        }
        self.queue.write_buffer(&self.uniforms, 0, &bytes);
    }

    /// What the pass reads, made where a texture is new.
    fn bind(&mut self) -> Option<&wgpu::BindGroup> {
        let kept = self.kept.as_ref()?;
        if self.bind.is_none() {
            self.bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodeldiplom window"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&kept.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.tiles_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniforms.as_entire_binding(),
                    },
                ],
            }));
        }
        self.bind.as_ref()
    }

    /// The changed pixels into the texture (a new one, filled whole, for a
    /// new window or pixmap size).
    fn upload(&mut self, canvas: &Canvas, changes: &Changes, width: u32, height: u32) {
        let (pixels, pw, ph) = canvas.pixels();
        let same = self
            .kept
            .as_ref()
            .is_some_and(|k| k.window == (width, height) && k.pixmap == (pw, ph));
        if !same {
            if self
                .kept
                .as_ref()
                .is_none_or(|k| k.window != (width, height))
            {
                self.kept = Some(self.texture(width, height));
                self.bind = None;
            }
            // The whole window: the pixmap and the background beyond it.
            let (w, h) = (width as usize, height as usize);
            let mut whole = vec![0u32; w * h];
            canvas.write_xrgb(&mut whole, w, h);
            self.write(&whole, w, (0, 0, w, h));
            if let Some(kept) = &mut self.kept {
                kept.pixmap = (pw, ph);
            }
            return;
        }
        // Within the pixmap and the window.
        let (cw, chh) = (pw.min(width as usize), ph.min(height as usize));
        let all = [(0, 0, cw, chh)];
        let rects: &[PixelRect] = if changes.all { &all } else { &changes.rects };
        for &(x, y, w, h) in rects {
            let (w, h) = (w.min(cw.saturating_sub(x)), h.min(chh.saturating_sub(y)));
            if w > 0 && h > 0 {
                self.write(pixels, pw, (x, y, w, h));
            }
        }
    }

    /// The rect `(x, y, w, h)` of `pixels` (rows of `stride` words) into
    /// the texture at the same place.
    fn write(&self, pixels: &[u32], stride: usize, (x, y, w, h): PixelRect) {
        let Some(kept) = &self.kept else {
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &kept.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: ((y * stride + x) * 4) as u64,
                bytes_per_row: Some((stride * 4) as u32),
                rows_per_image: Some(h as u32),
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    fn texture(&self, width: u32, height: u32) -> Kept {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom pixmap"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        Kept {
            texture,
            view,
            window: (width, height),
            pixmap: (0, 0),
        }
    }

    /// The texture onto `target`, with whatever else `then` adds to the
    /// commands; submitted. False where there is no texture yet.
    fn paint(
        &mut self,
        target: &wgpu::Texture,
        then: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> bool {
        let Some(bind) = self.bind().cloned() else {
            return false;
        };
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jodeldiplom window"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jodeldiplom window"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
            self.stage.draw(&mut pass);
            self.particles.draw(&mut pass);
        }
        then(&mut encoder);
        self.queue.submit([encoder.finish()]);
        true
    }
}

/// The tiles' texture array of `(width, height, slots)`, and its view.
fn tiles_texture(
    device: &wgpu::Device,
    view_format: wgpu::TextureFormat,
    (width, height, slots): (u32, u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom layer"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: slots.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: if view_format == TEXTURE_FORMAT {
            &[]
        } else {
            &[wgpu::TextureFormat::Bgra8UnormSrgb]
        },
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        format: Some(view_format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// An sRGB-encoded value (0 to 1) as linear light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The window's frames through the GPU.
pub struct GpuWindow {
    // Both dropped in the GPU's turn.
    surface: Turned<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    painter: Turned<Painter>,
    /// What draws: the adapter's name and backend.
    pub name: String,
    /// The last frame's work is done on the GPU (set from wgpu's callback).
    done: Arc<AtomicBool>,
    /// A frame was sent.
    pending: bool,
    /// When the last [`GpuWindow::show`] began, the frame before it was not
    /// yet done on the GPU: it missed its turn, the GPU busy with other
    /// work.
    pub behind: bool,
}

impl GpuWindow {
    /// Sets up the GPU for `window`; an error where there is no GPU that
    /// can show into it.
    pub fn new(window: Arc<Window>) -> Result<GpuWindow, String> {
        // In the GPU's turn: the stems and the analyses set theirs up on
        // other threads meanwhile ([`jodel_gpu::setting_up`]).
        let _turn = jodel_gpu::setting_up();
        let size = window.inner_size();
        // The process's one instance and GPU ([`jodel_gpu::chosen`]): the
        // adapters are listed again only where that GPU cannot draw here.
        let chosen = jodel_gpu::chosen();
        let surface = chosen
            .instance
            .create_surface(window.clone())
            .map_err(|e| format!("no surface: {e}"))?;
        let (adapter, device, queue) = window_device(chosen, &surface, monitor_gpu(&window))?;
        let caps = surface.get_capabilities(&adapter);
        // A format that takes the bytes as they are; else an sRGB one,
        // read through an sRGB view so they come out the same.
        let format = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .find(|f| caps.formats.contains(f))
        .or_else(|| caps.formats.first().copied())
        .ok_or("the window takes no format")?;
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: Default::default(),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        crate::trace::note(format_args!("window: on {name}"));
        Ok(GpuWindow {
            surface: Turned::new(surface),
            config,
            painter: Turned::new(Painter::new(device, queue, format)),
            name,
            done: Arc::new(AtomicBool::new(true)),
            pending: false,
            behind: false,
        })
    }

    /// Karaoke mode's stage over the next frame (`None`: none), for a
    /// window of `width` × `height`.
    pub fn set_stage(
        &mut self,
        frame: Option<&super::stage_gpu::StageFrame>,
        width: u32,
        height: u32,
    ) {
        self.painter.set_stage(frame, (width.max(1), height.max(1)));
    }

    /// Shows the canvas in a window of `width` × `height` pixels, sending
    /// over the pixels in `changes` (all of them where the window or the
    /// pixmap changed size), and `layer` over it (with no changes: a frame
    /// that only moves the layer).
    pub fn show(
        &mut self,
        canvas: &Canvas,
        changes: &Changes,
        width: u32,
        height: u32,
        layer: Option<LayerFrame>,
    ) -> Result<Shown, String> {
        let (width, height) = (width.max(1), height.max(1));
        // Whether the last frame is done on the GPU by now (the callbacks
        // of work done run here, without waiting).
        let _ = self.painter.device.poll(wgpu::PollType::Poll);
        self.behind = self.pending && !self.done.load(Ordering::Acquire);
        if (self.config.width, self.config.height) != (width, height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.painter.device, &self.config);
        }
        self.painter.upload(canvas, changes, width, height);
        self.painter.set_layer(layer);
        let device = &self.painter.device;
        let (frame, again) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            // Shown this once, the surface set up again for the next.
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(Shown::NotNow);
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.config);
                return Ok(Shown::NotNow);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("the window's frame was refused".into());
            }
        };
        let shown = if self.painter.paint(&frame.texture, |_| {}) {
            self.done.store(false, Ordering::Release);
            let done = Arc::clone(&self.done);
            self.painter
                .queue
                .on_submitted_work_done(move || done.store(true, Ordering::Release));
            self.pending = true;
            self.painter.queue.present(frame);
            Shown::Yes
        } else {
            Shown::NotNow
        };
        if again {
            self.surface.configure(&self.painter.device, &self.config);
        }
        Ok(shown)
    }
}

/// A device for a window's `surface`, on the first of these that shows
/// into it and opens: the process's GPU ([`jodel_gpu::chosen`]) through
/// DirectX 12 (its sibling), which hands its frames to the compositor
/// most directly; the chosen adapter itself; the adapters listed again
/// for the GPU the window's monitor is on (DirectX 12 first); else the
/// adapter the system offers for the surface. The first three only on the
/// monitor's GPU (`monitor`: its vendor and device ids, [`jodel_gpu::card`]):
/// a frame drawn on another one is copied over to it by the system, and
/// on Windows that shows it a good part of a second late. Where the
/// monitor's GPU is not known, the system's offer.
fn window_device(
    chosen: &jodel_gpu::Chosen,
    surface: &wgpu::Surface<'_>,
    monitor: Option<(u32, u32)>,
) -> Result<Opened, String> {
    let shows = |a: &wgpu::Adapter| {
        let info = a.get_info();
        info.device_type != wgpu::DeviceType::Cpu
            && monitor.is_some_and(|ids| jodel_gpu::card(&info) == ids)
            && a.is_surface_supported(surface)
    };
    // Why the GPU chosen for the process does not draw, for the trace.
    let mut failed = None;
    for kept in [chosen.sibling.as_ref(), chosen.adapter.as_ref().ok()]
        .into_iter()
        .flatten()
    {
        if shows(kept) {
            match open(kept.clone()) {
                Ok(opened) => return Ok(opened),
                Err(e) => failed = Some(e),
            }
        }
    }
    let listed = monitor.and_then(|_| {
        let mut adapters: Vec<wgpu::Adapter> =
            pollster::block_on(chosen.instance.enumerate_adapters(wgpu::Backends::all()))
                .into_iter()
                .filter(|a| shows(a))
                .collect();
        adapters.sort_by_key(|a| a.get_info().backend != wgpu::Backend::Dx12);
        adapters.into_iter().find_map(|a| open(a).ok())
    });
    let opened = match listed {
        Some(opened) => opened,
        None => {
            let adapter = pollster::block_on(chosen.instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: Some(surface),
                    ..Default::default()
                },
            ))
            .map_err(|e| format!("no GPU for the window: {e}"))?;
            open(adapter)?
        }
    };
    if crate::trace::on() {
        let why = match (failed, &chosen.adapter) {
            (Some(e), _) => e,
            (None, Err(e)) => e.clone(),
            (None, Ok(_)) if monitor.is_none() => "the monitor's GPU is not known".into(),
            (None, Ok(a)) => format!(
                "{} cannot show into it, or is not the monitor's",
                a.get_info().name
            ),
        };
        crate::trace::note(format_args!(
            "window: not on the GPU chosen for the process ({why}): on {} instead",
            opened.0.get_info().name
        ));
    }
    Ok(opened)
}

/// The vendor and device ids of the GPU the monitor `window` is on (most
/// of), where the system says.
#[cfg(windows)]
fn monitor_gpu(window: &Window) -> Option<(u32, u32)> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
    use windows::Win32::Graphics::Gdi::{MonitorFromWindow, MONITOR_DEFAULTTONEAREST};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let RawWindowHandle::Win32(handle) = window.window_handle().ok()?.as_raw() else {
        return None;
    };
    let hwnd = HWND(handle.hwnd.get() as *mut std::ffi::c_void);
    // SAFETY: plain calls on a live window and the factory's own objects.
    unsafe {
        let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        for i in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(i) else {
                break;
            };
            for j in 0.. {
                let Ok(output) = adapter.EnumOutputs(j) else {
                    break;
                };
                if output.GetDesc().is_ok_and(|d| d.Monitor == monitor) {
                    let desc = adapter.GetDesc1().ok()?;
                    return Some((desc.VendorId, desc.DeviceId));
                }
            }
        }
    }
    None
}

/// Elsewhere it is not known: the window takes the adapter the system
/// offers for it ([`window_device`]).
#[cfg(not(windows))]
fn monitor_gpu(_: &Window) -> Option<(u32, u32)> {
    None
}

/// The window's frames drawn into a texture off screen and read back:
/// for the tests, the same uploads and pass as the window's.
pub struct Offscreen {
    painter: Turned<Painter>,
    format: wgpu::TextureFormat,
    /// What draws: the adapter's name and backend.
    pub name: String,
}

impl Offscreen {
    /// On the process's GPU ([`jodel_gpu::chosen`]), into frames that take
    /// the bytes as they are (`srgb`: that encode them as sRGB, as some
    /// windows do); an error where there is no GPU.
    pub fn new(srgb: bool) -> Result<Offscreen, String> {
        let _turn = jodel_gpu::setting_up();
        let (adapter, device, queue) = open(jodel_gpu::chosen().adapter.clone()?)?;
        let info = adapter.get_info();
        let format = if srgb {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            TEXTURE_FORMAT
        };
        Ok(Offscreen {
            painter: Turned::new(Painter::new(device, queue, format)),
            format,
            name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    /// Karaoke mode's stage over the next frame, as the window's.
    pub fn set_stage(
        &mut self,
        frame: Option<&super::stage_gpu::StageFrame>,
        width: u32,
        height: u32,
    ) {
        self.painter.set_stage(frame, (width.max(1), height.max(1)));
    }

    /// Draws a frame of `pixels` (`width` × `height`, `0x00RRGGBB` words
    /// row by row) with the stage set over it, and reads it back.
    pub fn show_pixels(
        &mut self,
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Result<Vec<u32>, String> {
        if self
            .painter
            .kept
            .as_ref()
            .is_none_or(|k| k.window != (width, height))
        {
            self.painter.kept = Some(self.painter.texture(width, height));
            self.painter.bind = None;
        }
        let (w, h) = (width as usize, height as usize);
        self.painter.write(pixels, w, (0, 0, w, h));
        self.painter.set_layer(None);
        self.read_back(width, height)
    }

    /// Sends over `changes` and `layer` as the window does, draws a frame
    /// of `width` × `height` and reads it back, as `0x00RRGGBB` words row
    /// by row.
    pub fn show(
        &mut self,
        canvas: &Canvas,
        changes: &Changes,
        width: u32,
        height: u32,
        layer: Option<LayerFrame>,
    ) -> Result<Vec<u32>, String> {
        self.painter.upload(canvas, changes, width, height);
        self.painter.set_layer(layer);
        self.read_back(width, height)
    }

    /// The frame drawn from what was sent over, read back.
    fn read_back(&mut self, width: u32, height: u32) -> Result<Vec<u32>, String> {
        let device = self.painter.device.clone();
        let frame = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Rows padded to the copy's alignment.
        let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom offscreen"),
            size: u64::from(row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let painted = self.painter.paint(&frame, |encoder| {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        });
        if !painted {
            return Err("nothing to draw".into());
        }
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
        let mut words = Vec::with_capacity(width as usize * height as usize);
        for line in mapped.chunks_exact(row as usize) {
            words.extend(
                line[..width as usize * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], 0])),
            );
        }
        Ok(words)
    }
}
