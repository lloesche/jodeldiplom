//! Karaoke mode's stage on the GPU: the note lane, the singer's arrow, its
//! trail and sparks, the lyrics at the bottom and the song's progress,
//! drawn over the pixmap each frame as instances of a few shapes, each a
//! signed distance in the fragment shader (a pill, a capsule, a star, a
//! disc, an arrow) or a run of text from a glyph atlas. One pipeline,
//! premultiplied: a shape's light that is to add up (a glow, a spark)
//! comes with no coverage, so it adds; the rest goes over.
//!
//! [`StageFrame`] is what the window gets each frame: the instances, and
//! the text the atlas did not hold yet ([`AtlasUpload`]).

/// The atlas's width and height, in pixels.
pub const ATLAS: u32 = 2048;

/// What a shape is ([`Instance::params`]`[0]`).
pub mod kind {
    /// A rounded box: `shape` its centre and half size, `params[3]` its
    /// corner radius (0: round ends), `params[1]` its glow, `params[2]` its
    /// outline. Over in `color`, outlined and glowing in `color2`.
    pub const PILL: f32 = 0.0;
    /// The part of a rounded box (as [`PILL`]) between `uv[0]` and `uv[1]`
    /// across: `color` at its top, `color2` at its bottom, glowing in
    /// `color2` by `params[1]`.
    pub const FILL: f32 = 1.0;
    /// A capsule from (`shape[0]`, `shape[1]`) to (`shape[2]`,
    /// `shape[3]`), `params[1]` thick each way, glowing `params[2]` in
    /// `color2`; its core in `color`.
    pub const SEGMENT: f32 = 2.0;
    /// A four-pointed spark at (`shape[0]`, `shape[1]`), its rays
    /// `shape[2]` long, turned by `shape[3]`: light, added.
    pub const SPARK: f32 = 3.0;
    /// A disc at (`shape[0]`, `shape[1]`), `shape[2]` across each way, in
    /// `color`, glowing `shape[3]` in `color2`.
    pub const DISC: f32 = 4.0;
    /// An arrow pointing right, its tip at (`shape[0]`, `shape[1]`),
    /// `shape[2]` long: in `color`, glowing `params[1]` in `color2`.
    pub const ARROW: f32 = 5.0;
    /// Text from the atlas: `uv` its texels there, drawn over the quad;
    /// left of `params[1]` (pixels across the window) in `color2`, right of
    /// it in `color`.
    pub const TEXT: f32 = 6.0;
}

/// One shape, as the shader takes it: the quad it covers (`x0, y0, x1,
/// y1` in window pixels, its glow included) and what [`kind`] says of the
/// rest. Colours are sRGB, 0 to 1, with their alpha.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Instance {
    /// The quad: left, top, right, bottom.
    pub quad: [f32; 4],
    /// The shape's own numbers.
    pub shape: [f32; 4],
    /// Its colour.
    pub color: [f32; 4],
    /// Its second colour.
    pub color2: [f32; 4],
    /// Its kind, then three numbers.
    pub params: [f32; 4],
    /// Atlas texels, or a fill's span.
    pub uv: [f32; 4],
}

// SAFETY: plain f32s in a `repr(C)` struct, no padding: any bytes are one.
unsafe impl bytemuck::Zeroable for Instance {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Instance {}

/// Text new to the atlas: its place there and its coverage, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasUpload {
    /// Left, top, width, height, in the atlas.
    pub rect: (u32, u32, u32, u32),
    /// The coverage, `width × height`.
    pub coverage: Vec<u8>,
}

/// A frame of the stage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageFrame {
    /// The shapes, back to front.
    pub instances: Vec<Instance>,
    /// Text to put into the atlas first.
    pub uploads: Vec<AtlasUpload>,
}

const SHADER: &str = r"
struct Screen {
    size: vec2<f32>,
    srgb: f32,
    atlas: f32,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var soft: sampler;

struct In {
    @location(0) quad: vec4<f32>,
    @location(1) shape: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) color2: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) p: vec2<f32>,
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) color2: vec4<f32>,
    @location(4) @interpolate(flat) params: vec4<f32>,
    @location(5) @interpolate(flat) uv: vec4<f32>,
    @location(6) @interpolate(flat) quad: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) v: u32, i: In) -> Out {
    let corner = vec2<f32>(f32(v & 1u), f32((v >> 1u) & 1u));
    let p = mix(i.quad.xy, i.quad.zw, corner);
    var o: Out;
    o.pos = vec4<f32>(p.x / screen.size.x * 2.0 - 1.0, 1.0 - p.y / screen.size.y * 2.0, 0.0, 1.0);
    o.p = p;
    o.shape = i.shape;
    o.color = i.color;
    o.color2 = i.color2;
    o.params = i.params;
    o.uv = i.uv;
    o.quad = i.quad;
    return o;
}

// A colour as the frame takes it: linear where the frame is sRGB.
fn out_rgb(c: vec3<f32>) -> vec3<f32> {
    if screen.srgb > 0.5 {
        return pow(c, vec3<f32>(2.2));
    }
    return c;
}

// Over: `c` covering `a`; plus light `l` added.
fn over_plus(c: vec3<f32>, a: f32, l: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(out_rgb(c) * a + out_rgb(l), a);
}

fn round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0.0, 1.0);
    return length(pa - ba * h);
}

// The glow of a shape `d` pixels outside it, over `size`.
fn glow(d: f32, size: f32) -> f32 {
    if size <= 0.0 {
        return 0.0;
    }
    let t = max(d, 0.0) / size;
    return exp(-t * t * 3.0) * (1.0 - smoothstep(0.85, 1.0, t));
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let k = o.params.x;
    let p = o.p;
    if k < 0.5 || (k > 0.5 && k < 1.5) {
        // A pill, or its fill.
        let c = o.shape.xy;
        let half = o.shape.zw;
        var r = o.params.w;
        if r <= 0.0 {
            r = min(half.x, half.y);
        }
        let d = round_box(p - c, half, r);
        let cover = clamp(0.5 - d, 0.0, 1.0);
        if k > 0.5 {
            // The fill, between its span, its top light to its bottom.
            let across = smoothstep(o.uv.x - 0.5, o.uv.x + 0.5, p.x)
                * (1.0 - smoothstep(o.uv.y - 0.5, o.uv.y + 0.5, p.x));
            let down = clamp((p.y - (c.y - half.y)) / max(2.0 * half.y, 1.0), 0.0, 1.0);
            let col = mix(o.color.rgb, o.color2.rgb, down);
            // A shine along its top.
            let shine = (1.0 - smoothstep(0.0, 0.35, down)) * 0.35;
            let gs = max(o.params.y, 1.0);
            let soft = smoothstep(o.uv.x - gs, o.uv.x, p.x)
                * (1.0 - smoothstep(o.uv.y, o.uv.y + gs, p.x));
            let a = cover * across * o.color.a;
            let g = glow(d, o.params.y) * soft * o.color2.a * 0.45;
            return over_plus(min(col + vec3<f32>(shine), vec3<f32>(1.0)), a, o.color2.rgb * g);
        }
        let outline = o.params.z;
        let edge = clamp(0.5 - abs(d + outline * 0.5) + outline * 0.5, 0.0, 1.0) * cover;
        // Lighter at its top, as glass.
        let down = clamp((p.y - (c.y - half.y)) / max(2.0 * half.y, 1.0), 0.0, 1.0);
        let body = o.color.rgb * (1.15 - 0.3 * down);
        let a_body = cover * o.color.a;
        let col = mix(body, o.color2.rgb, edge * o.color2.a / max(a_body + edge * o.color2.a, 1e-4));
        let a = max(a_body, edge * o.color2.a);
        let g = glow(d, o.params.y) * o.color2.a * 0.5;
        return over_plus(col, a, o.color2.rgb * g);
    }
    if k < 2.5 {
        let d = segment(p, o.shape.xy, o.shape.zw) - o.params.y;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let g = glow(d, o.params.z) * o.color2.a;
        return over_plus(o.color.rgb, cover * o.color.a, o.color2.rgb * g);
    }
    if k < 3.5 {
        // A spark: a core and two crossed rays, turned, all light.
        let s = sin(o.shape.w);
        let co = cos(o.shape.w);
        let q0 = p - o.shape.xy;
        let q = abs(vec2<f32>(co * q0.x - s * q0.y, s * q0.x + co * q0.y));
        let size = max(o.shape.z, 0.5);
        let core = exp(-dot(q, q) / (size * size * 0.05));
        let thin = max(size * 0.07, 0.8);
        let ray_x = exp(-q.y * q.y / (thin * thin)) * pow(clamp(1.0 - q.x / size, 0.0, 1.0), 1.6);
        let ray_y = exp(-q.x * q.x / (thin * thin)) * pow(clamp(1.0 - q.y / size, 0.0, 1.0), 1.6);
        let halo = exp(-dot(q, q) / (size * size * 0.18)) * 0.35;
        let light = (core + max(ray_x, ray_y) * 0.9 + halo) * o.color.a;
        return vec4<f32>(out_rgb(o.color.rgb) * light, 0.0);
    }
    if k < 4.5 {
        let d = length(p - o.shape.xy) - o.shape.z;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let g = glow(d, o.shape.w) * o.color2.a;
        return over_plus(o.color.rgb, cover * o.color.a, o.color2.rgb * g);
    }
    if k < 5.5 {
        // An arrow pointing right: a rounded triangle, its tip at shape.xy.
        let len = o.shape.z;
        let q = p - (o.shape.xy - vec2<f32>(len * 0.5, 0.0));
        // The triangle's three edges, as distances (negative inside).
        let h = len * 0.5;
        let w = len * 0.42;
        let n1 = normalize(vec2<f32>(w, h));
        let e1 = dot(vec2<f32>(q.x, abs(q.y)) - vec2<f32>(h, 0.0), n1);
        let e2 = -q.x - h * 0.55;
        let d = max(e1, e2) - len * 0.06;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let inner = clamp(-d / (len * 0.25), 0.0, 1.0);
        let col = mix(o.color.rgb, vec3<f32>(1.0), inner * 0.55);
        let g = glow(d, o.params.y) * o.color2.a;
        return over_plus(col, cover * o.color.a, o.color2.rgb * g);
    }
    // Text: its coverage from the atlas, filled left of the split.
    let t = (p - o.quad.xy) / max(o.quad.zw - o.quad.xy, vec2<f32>(1.0));
    // Kept inside its own texels (their centres): never a column of the
    // text beside it in the atlas.
    let texel = clamp(mix(o.uv.xy, o.uv.zw, t), o.uv.xy + vec2<f32>(0.5), o.uv.zw - vec2<f32>(0.5));
    let uv = texel / screen.atlas;
    let a = textureSampleLevel(atlas, soft, uv, 0.0).r;
    let filled = smoothstep(o.params.y - 0.75, o.params.y + 0.75, p.x);
    let col = mix(o.color2.rgb, o.color.rgb, filled);
    let alpha = mix(o.color2.a, o.color.a, filled) * a;
    return over_plus(col, alpha, vec3<f32>(0.0));
}
";

/// The stage's pipeline and what it reads.
pub struct StagePass {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    atlas: wgpu::Texture,
    instances: Option<wgpu::Buffer>,
    /// The instances in the buffer for the next frame.
    count: u32,
    srgb: bool,
}

impl StagePass {
    /// For frames of `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> StagePass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom stage"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4,
            3 => Float32x4, 4 => Float32x4, 5 => Float32x4
        ];
        let premultiplied = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom stage"),
            layout: Some(&pipeline_layout),
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
                    blend: Some(wgpu::BlendState {
                        color: premultiplied,
                        alpha: premultiplied,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom stage"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom stage atlas"),
            size: wgpu::Extent3d {
                width: ATLAS,
                height: ATLAS,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom stage"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom stage"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        StagePass {
            pipeline,
            bind,
            uniforms,
            atlas,
            instances: None,
            count: 0,
            srgb: format.is_srgb(),
        }
    }

    /// The next frame's stage (`None`: none), for a window of `size`.
    pub fn set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: Option<&StageFrame>,
        size: (u32, u32),
    ) {
        self.count = 0;
        let Some(frame) = frame else {
            return;
        };
        for up in &frame.uploads {
            let (x, y, w, h) = up.rect;
            if w == 0 || h == 0 {
                continue;
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.atlas,
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
        if frame.instances.is_empty() {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&frame.instances);
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
            queue.write_buffer(buffer, 0, bytes);
        }
        let mut uniforms = [0u8; 16];
        uniforms[0..4].copy_from_slice(&(size.0 as f32).to_le_bytes());
        uniforms[4..8].copy_from_slice(&(size.1 as f32).to_le_bytes());
        uniforms[8..12].copy_from_slice(&(if self.srgb { 1f32 } else { 0.0 }).to_le_bytes());
        uniforms[12..16].copy_from_slice(&(ATLAS as f32).to_le_bytes());
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        self.count = frame.instances.len() as u32;
    }

    /// Draws the stage into `pass` (nothing without one).
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        let Some(buffer) = &self.instances else {
            return;
        };
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..4, 0..self.count);
    }
}
