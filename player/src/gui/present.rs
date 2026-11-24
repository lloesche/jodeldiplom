//! The old screens' frames on the GPU ([`Painter`], inside iced's frame
//! through [`super::embed_gpu`]): the pixmap is kept in a texture the size
//! of the window, each frame sends over only the pixels that changed
//! ([`Changes`]), and a triangle over the whole window copies the texture
//! onto it, pixel for pixel; a few rows a frame, not the whole window.
//!
//! A lane that follows the playhead comes as a layer ([`LayerFrame`]): its
//! tiles kept in a texture array, drawn over the pixmap at the column (and
//! share of one) the frame says, with the playhead over it. A frame that
//! moves only the layer sends over nothing but its place. Karaoke mode's
//! backdrop ([`super::backdrop`]) is under the pixmap: its words are
//! premultiplied over it, their top byte how much of it shows through
//! ([`super::raster::SEE_THROUGH`]); its stage and its bursts are drawn
//! over it all.

use jodel_gpu::Turned;

use super::backdrop::BackdropFrame;
use super::canvas::{Canvas, Changes, LayerFrame, MOST_SLOTS};
use super::raster::PixelRect;
use super::stage_gpu::StageFrame;

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
// The backdrop under the pixmap ([`super::backdrop`]).
@group(0) @binding(3) var backdrop: texture_2d<f32>;
@group(0) @binding(4) var soft: sampler;

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

// An sRGB-encoded value as linear light, and back.
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn to_srgb(l: vec3<f32>) -> vec3<f32> {
    let c = max(l, vec3<f32>(0.0));
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), c * 12.92, c <= vec3<f32>(0.0031308));
}

// The pixmap's word at `p`, premultiplied over the backdrop: its own
// colour with what is under added as far as its alpha (the word's top
// byte) lets it through, added as the processor drew it (in the colours
// as encoded).
fn pixel(p: vec2<i32>, at: vec2<f32>) -> vec3<f32> {
    let own = textureLoad(pixels, p, 0);
    if own.a <= 0.0 {
        return own.rgb;
    }
    let uv = at / vec2<f32>(textureDimensions(pixels));
    let under = textureSampleLevel(backdrop, soft, uv, 0.0).rgb;
    if LINEAR_VIEW {
        return to_linear(to_srgb(own.rgb) + own.a * to_srgb(under));
    }
    return own.rgb + own.a * under;
}

// Each window pixel is the pixmap's there over the backdrop; in the
// layer's rect, the layer's, between two columns by the share of one it
// is moved, and the playhead over it.
@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(at.xy);
    let c = vec4<f32>(pixel(p, at.xy), 1.0);
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
pub(super) struct Painter {
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
    /// The light that blooms ([`super::bloom`]).
    light: super::bloom::Light,
    /// Karaoke mode's backdrop under the pixmap, how it is read, and
    /// which of its textures the pass reads ([`BackdropPass::generation`]).
    ///
    /// [`BackdropPass::generation`]: super::backdrop::gpu::BackdropPass::generation
    backdrop: super::backdrop::gpu::BackdropPass,
    soft: wgpu::Sampler,
    bound: u64,
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
    pub(super) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Painter {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom window"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        // The pixmap read as linear light (through an sRGB view) where the
        // frame takes it so.
        let linear_view = format.is_srgb();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom window"),
            source: wgpu::ShaderSource::Wgsl(
                format!("const LINEAR_VIEW: bool = {linear_view};\n{SHADER}").into(),
            ),
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
        let view_format = if linear_view {
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
        let stage =
            super::stage_gpu::StagePass::new(&device, format, super::stage_gpu::Blend::Over);
        let particles =
            super::particles::ParticlePass::new(&device, format, super::stage_gpu::Blend::Over);
        let light = super::bloom::Light::new(&device, format);
        let backdrop = super::backdrop::gpu::BackdropPass::new(&device, &queue, linear_view);
        let soft = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom window"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
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
            light,
            backdrop,
            soft,
            bound: 0,
        }
    }

    /// Karaoke mode's backdrop for the next frame (`None`: none, its
    /// textures dropped), in a frame of `size`: drawn now.
    fn set_backdrop(&mut self, frame: Option<&BackdropFrame>, size: (u32, u32)) {
        self.backdrop.set((&self.device, &self.queue), frame, size);
        if self.backdrop.generation() != self.bound {
            self.bind = None;
        }
    }

    /// Karaoke mode's stage for the next frame (`None`: none), in a frame
    /// of `size`.
    fn set_stage(&mut self, frame: Option<&StageFrame>, size: (u32, u32)) {
        let gpu = (&self.device, &self.queue);
        self.stage.set(gpu, frame.map(StageFrame::feed), size);
        let (sprays, time) = frame.map_or((&[][..], 0.0), |f| (&f.sprays[..], f.clock));
        self.particles.set(gpu, sprays, time, size);
        self.light.set(gpu, frame, size);
    }

    /// The next frame, of `size` pixels: the pixels in `changes` sent over
    /// (all of them where the size or the pixmap's changed), Karaoke mode's
    /// `backdrop` drawn under it, `layer` and `stage` over it.
    pub(super) fn prepare(
        &mut self,
        canvas: &Canvas,
        changes: &Changes,
        size: (u32, u32),
        layer: Option<LayerFrame>,
        stage: Option<&StageFrame>,
        backdrop: Option<&BackdropFrame>,
    ) {
        let size = (size.0.max(1), size.1.max(1));
        self.upload(canvas, changes, size.0, size.1);
        self.set_layer(layer);
        self.set_backdrop(backdrop, size);
        self.set_stage(stage, size);
        self.bind();
    }

    /// The frame prepared, into `pass` (its viewport the window's): the
    /// pixels over the backdrop, the stage and its bursts, the light and its bloom added over
    /// them (drawn and blurred as the frame was prepared), and the stage's
    /// shapes over everything; false where there is none yet.
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) -> bool {
        let Some(bind) = &self.bind else {
            return false;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
        self.stage.draw(pass);
        self.particles.draw(pass);
        self.light.composite(pass);
        self.stage.draw_top(pass);
        true
    }

    /// The device and the queue it draws with.
    pub(super) fn queue(&self) -> &wgpu::Queue {
        &self.queue
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(self.backdrop.under()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.soft),
                    },
                ],
            }));
            self.bound = self.backdrop.generation();
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

    /// The texture onto `target` in a pass of its own, with whatever else
    /// `then` adds to the commands; submitted. False where there is no
    /// texture yet.
    fn paint(
        &mut self,
        target: &wgpu::Texture,
        then: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> bool {
        if self.bind().is_none() {
            return false;
        }
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
            self.draw(&mut pass);
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

    /// Karaoke mode's backdrop under the next frame, as the window's
    /// (none: its textures dropped).
    pub fn set_backdrop(&mut self, frame: Option<&BackdropFrame>, width: u32, height: u32) {
        self.painter
            .set_backdrop(frame, (width.max(1), height.max(1)));
    }

    /// How many covers the backdrop holds on the GPU.
    pub fn backdrop_held(&self) -> usize {
        self.painter.backdrop.held()
    }

    /// Draws a frame of `pixels` (`width` × `height` pixmap words row by
    /// row, premultiplied over the backdrop: [`super::raster::SEE_THROUGH`])
    /// with the backdrop and the stage set, and reads it back.
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
