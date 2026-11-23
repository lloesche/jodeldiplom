//! Bursts on the GPU: a bubble popping and its sparkles flying off, a
//! word scored throwing sparks. The processor hands the GPU one small
//! record a burst ([`Spray`], [`Packed`]: where, when on the singing's
//! clock, a seed, a [`Kind`]), and the time each frame; the vertex shader
//! works out every particle from (seed, its index, its age) in closed form
//! (up and out, slowing, shrinking, fading), drawn as instanced quads,
//! [`QUADS`] a burst. Nothing of it is stepped or rasterised on the
//! processor.
//!
//! [`sprite`] is the same maths on the processor: the shader's reference,
//! for the tests; the shader is given its numbers ([`shader`]). Where there
//! is no GPU (the software window, the tests' pixmaps) the bursts are not
//! drawn; the bubbles still show full.

use super::sparks::noise;
use crate::view::bubbles::{BURST, POP};

/// The quads of a burst: its pop, then its particles.
pub const QUADS: u32 = 16;
/// The sparkles a bubble throws.
pub const BUBBLE_SPARKS: u32 = 7;
/// How long a word scored throws sparks for, at most, in seconds.
pub const SCORED_LIFE: f32 = 1.1;

/// What a burst is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A bubble popping (grown and white hot) and throwing sparkles up.
    Bubble,
    /// Its sparkles alone (the stage pops its pill itself).
    Sparkles,
    /// A word scored: sparks out every way.
    Scored,
    /// The multiplier going up: the same, further.
    ScoredUp,
}

impl Kind {
    /// As the shader takes it.
    pub fn code(self) -> f32 {
        match self {
            Kind::Bubble => 0.0,
            Kind::Sparkles => 1.0,
            Kind::Scored => 2.0,
            Kind::ScoredUp => 3.0,
        }
    }

    /// How far a word scored's sparks fly (none: not one).
    fn spread(self) -> f32 {
        match self {
            Kind::ScoredUp => 2.5,
            _ => 1.4,
        }
    }
}

/// A burst: what it bursts from at its start (its middle in window
/// pixels, half its width and height), when it starts (seconds, the
/// singing's clock), how fast what it bursts from goes across (pixels a
/// second: a lane's), its size (the text's or the pill's height), its
/// seed, its colour (sRGB) and its kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spray {
    /// Its middle.
    pub at: (f32, f32),
    /// Half its width and height.
    pub half: (f32, f32),
    /// When it starts.
    pub start: f32,
    /// How fast it goes across.
    pub drift: f32,
    /// Its size.
    pub size: f32,
    /// Its seed (24 bits are taken).
    pub seed: u32,
    /// Its colour.
    pub rgb: [f32; 3],
    /// What it is.
    pub kind: Kind,
}

/// A [`Spray`] as the GPU takes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Packed {
    at: [f32; 4],
    motion: [f32; 4],
    color: [f32; 4],
}

// SAFETY: plain f32s in a `repr(C)` struct, no padding: any bytes are one.
unsafe impl bytemuck::Zeroable for Packed {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Packed {}

impl Spray {
    /// As the GPU takes it.
    pub fn packed(&self) -> Packed {
        Packed {
            at: [self.at.0, self.at.1, self.half.0, self.half.1],
            motion: [self.start, self.drift, self.size, self.seed24() as f32],
            color: [self.rgb[0], self.rgb[1], self.rgb[2], self.kind.code()],
        }
    }

    /// Its seed as the GPU has it (a float holds 24 bits exactly).
    fn seed24(&self) -> u32 {
        self.seed & 0xFF_FFFF
    }

    /// How long it lasts, in seconds.
    pub fn life(&self) -> f32 {
        match self.kind {
            Kind::Scored | Kind::ScoredUp => SCORED_LIFE,
            _ => BURST as f32,
        }
    }

    /// Whether it shows at `time` (none before its start: sought back).
    pub fn alive(&self, time: f64) -> bool {
        let age = time - f64::from(self.start);
        (0.0..f64::from(self.life())).contains(&age)
    }
}

/// How much a bubble has popped `t` through its burst (0 to 1): up at
/// once, then easing back to nothing by [`POP`] of it.
pub fn popped(t: f32) -> f32 {
    let t = (t / POP).clamp(0.0, 1.0);
    if t < 0.2 {
        t / 0.2
    } else {
        (1.0 - (t - 0.2) / 0.8).powi(2)
    }
}

/// What a quad of a burst is at a moment: its middle, half its size, how
/// much it shows (0 to 1), its colour, whether it is the pop (a glowing
/// capsule) or a sparkle, and how a sparkle is turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprite {
    /// Its middle, window pixels.
    pub at: (f32, f32),
    /// Half its width and height (a sparkle: its rays' length both).
    pub half: (f32, f32),
    /// How much it shows.
    pub alpha: f32,
    /// Its colour, sRGB.
    pub rgb: [f32; 3],
    /// The pop, else a sparkle.
    pub pop: bool,
    /// A sparkle's turn, radians.
    pub turn: f32,
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const TAU: f32 = std::f32::consts::TAU;

/// Quad `q` of `s` at `time` (the singing's clock), none where it does not
/// show: the shader's maths ([`shader`]) on the processor.
pub fn sprite(s: &Spray, q: u32, time: f32) -> Option<Sprite> {
    let age = time - s.start;
    if age < 0.0 || age >= s.life() {
        return None;
    }
    let size = s.size;
    let drift = s.drift * age;
    let seed = s.seed24();
    let r = |j: u32| noise(seed.wrapping_mul(31).wrapping_add(q * 7 + j));
    let (x, y, hw, hh) = (s.at.0, s.at.1, s.half.0, s.half.1);
    if q == 0 {
        let pop = popped(age / BURST as f32);
        if s.kind != Kind::Bubble || pop <= 0.0 {
            return None;
        }
        let grow = hh * 1.6 * pop;
        return Some(Sprite {
            at: (x + drift, y),
            half: (hw + grow, hh + grow),
            alpha: 0.9 * pop,
            rgb: mix(s.rgb, WHITE, 0.75 * pop),
            pop: true,
            turn: 0.0,
        });
    }
    if matches!(s.kind, Kind::Bubble | Kind::Sparkles) {
        if q > BUBBLE_SPARKS {
            return None;
        }
        let t = age / BURST as f32;
        let fly = 1.0 - (1.0 - t).powi(3);
        let sx = x - hw + r(0) * 2.0 * hw;
        let (vx, vy) = ((r(1) - 0.5) * 1.6, -(0.35 + 0.65 * r(2)));
        let reach = size * (0.35 + 0.45 * r(3));
        let ray = size * (0.12 + 0.1 * r(4)) * (1.0 - 0.6 * t);
        return Some(Sprite {
            at: (sx + vx * reach * fly + drift, y + vy * reach * fly),
            half: (ray, ray),
            alpha: (1.0 - t).powf(1.3) * (0.7 + 0.3 * (t * 22.0 + r(5) * TAU).sin()),
            rgb: mix(WHITE, s.rgb, 0.4),
            pop: false,
            turn: r(6) * TAU,
        });
    }
    let spread = s.kind.spread();
    let t = age / (0.4 + 0.7 * r(4));
    if t >= 1.0 {
        return None;
    }
    let (vx, vy) = (
        (r(2) - 0.65) * size * 4.0 * spread,
        (r(3) - 0.5) * size * 4.0 * spread,
    );
    let slow = (1.0 - (-2.4 * age).exp()) / 2.4;
    let ray = size * (0.35 + 0.65 * r(5)) * (0.45 + 0.55 * (1.0 - t));
    Some(Sprite {
        at: (
            x + (r(0) - 0.5) * size * 0.6 + vx * slow + drift,
            y + (r(1) - 0.5) * size * 0.6 + vy * slow,
        ),
        half: (ray, ray),
        alpha: (1.0 - t).powf(1.2) * (0.6 + 0.4 * (age * 26.0 + r(6) * TAU).sin()),
        rgb: if r(7) < 0.3 { [1.0, 0.97, 0.88] } else { s.rgb },
        pop: false,
        turn: r(8) * TAU + (r(9) - 0.5) * 6.0 * age,
    })
}

/// The particles' shader: [`SHADER`] given [`sprite`]'s numbers, and
/// [`pop_wgsl`].
pub fn shader() -> String {
    format!(
        "const QUADS: u32 = {QUADS}u;
const BUBBLE_SPARKS: u32 = {BUBBLE_SPARKS}u;
const SCORED_LIFE: f32 = {SCORED_LIFE:?};
const K_BUBBLE: f32 = {:?};
const K_SCORED: f32 = {:?};
const K_SCORED_UP: f32 = {:?};
const SPREAD: f32 = {:?};
const SPREAD_UP: f32 = {:?};
{SHADER}{}",
        Kind::Bubble.code(),
        Kind::Scored.code(),
        Kind::ScoredUp.code(),
        Kind::Scored.spread(),
        Kind::ScoredUp.spread(),
        pop_wgsl()
    )
}
/// [`popped`] and what it needs, in WGSL: for the particles' shader and the
/// stage's (its pills pop by it).
pub fn pop_wgsl() -> String {
    format!(
        "
const BURST: f32 = {BURST:?};
const POP: f32 = {POP:?};
fn popped(t0: f32) -> f32 {{
    let t = clamp(t0 / POP, 0.0, 1.0);
    if t < 0.2 {{
        return t / 0.2;
    }}
    let d = 1.0 - (t - 0.2) / 0.8;
    return d * d;
}}
"
    )
}

/// The particles' shader: [`sprite`] in WGSL, and each sprite drawn (its
/// numbers given it by [`shader`]).
const SHADER: &str = r"
struct Screen {
    size: vec2<f32>,
    srgb: f32,
    time: f32,
};

@group(0) @binding(0) var<uniform> screen: Screen;

struct In {
    @location(0) at: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) p: vec2<f32>,
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) @interpolate(flat) look: vec4<f32>,
    @location(3) @interpolate(flat) what: vec4<f32>,
};

const TAU: f32 = 6.2831853;

fn noise(i: u32) -> f32 {
    var x = (i * 0x9E3779B9u) ^ 0x85EBCA6Bu;
    x = x ^ (x >> 15u);
    x = x * 0x2C1B3C6Du;
    x = x ^ (x >> 12u);
    x = x * 0x297A2D39u;
    x = x ^ (x >> 15u);
    return f32(x >> 8u) / 16777216.0;
}

fn r(seed: u32, q: u32, j: u32) -> f32 {
    return noise(seed * 31u + q * 7u + j);
}

@vertex
fn vs(@builtin(vertex_index) v: u32, i: In) -> Out {
    var o: Out;
    // Nothing (a quad of no size) unless it shows.
    o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    o.p = vec2<f32>(0.0);
    o.shape = vec4<f32>(0.0);
    o.look = vec4<f32>(0.0);
    o.what = vec4<f32>(0.0);
    let q = v / 6u;
    var corners = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let c = corners[v % 6u];
    let corner = vec2<f32>(f32(c & 1u), f32(c >> 1u)) * 2.0 - vec2<f32>(1.0);
    let k = i.color.w;
    let size = i.motion.z;
    let age = screen.time - i.motion.x;
    var life = BURST;
    if k >= K_SCORED {
        life = SCORED_LIFE;
    }
    if age < 0.0 || age >= life {
        return o;
    }
    let drift = i.motion.y * age;
    let seed = u32(i.motion.w);
    var at = vec2<f32>(0.0);
    var half = vec2<f32>(0.0);
    var alpha = 0.0;
    var rgb = i.color.rgb;
    var pop = 0.0;
    var turn = 0.0;
    if q == 0u {
        let pp = popped(age / BURST);
        if k != K_BUBBLE || pp <= 0.0 {
            return o;
        }
        let grow = i.at.w * 1.6 * pp;
        at = vec2<f32>(i.at.x + drift, i.at.y);
        half = i.at.zw + vec2<f32>(grow);
        alpha = 0.9 * pp;
        rgb = mix(rgb, vec3<f32>(1.0), 0.75 * pp);
        pop = 1.0;
    } else if k < K_SCORED {
        if q > BUBBLE_SPARKS {
            return o;
        }
        let t = age / BURST;
        let fly = 1.0 - pow(1.0 - t, 3.0);
        let sx = i.at.x - i.at.z + r(seed, q, 0u) * 2.0 * i.at.z;
        let vx = (r(seed, q, 1u) - 0.5) * 1.6;
        let vy = -(0.35 + 0.65 * r(seed, q, 2u));
        let reach = size * (0.35 + 0.45 * r(seed, q, 3u));
        let ray = size * (0.12 + 0.1 * r(seed, q, 4u)) * (1.0 - 0.6 * t);
        at = vec2<f32>(sx + vx * reach * fly + drift, i.at.y + vy * reach * fly);
        half = vec2<f32>(ray);
        alpha = pow(1.0 - t, 1.3) * (0.7 + 0.3 * sin(t * 22.0 + r(seed, q, 5u) * TAU));
        rgb = mix(vec3<f32>(1.0), rgb, 0.4);
        turn = r(seed, q, 6u) * TAU;
    } else {
        var spread = SPREAD;
        if k >= K_SCORED_UP {
            spread = SPREAD_UP;
        }
        let t = age / (0.4 + 0.7 * r(seed, q, 4u));
        if t >= 1.0 {
            return o;
        }
        let vx = (r(seed, q, 2u) - 0.65) * size * 4.0 * spread;
        let vy = (r(seed, q, 3u) - 0.5) * size * 4.0 * spread;
        let slow = (1.0 - exp(-2.4 * age)) / 2.4;
        let ray = size * (0.35 + 0.65 * r(seed, q, 5u)) * (0.45 + 0.55 * (1.0 - t));
        at = vec2<f32>(
            i.at.x + (r(seed, q, 0u) - 0.5) * size * 0.6 + vx * slow + drift,
            i.at.y + (r(seed, q, 1u) - 0.5) * size * 0.6 + vy * slow,
        );
        half = vec2<f32>(ray);
        alpha = pow(1.0 - t, 1.2) * (0.6 + 0.4 * sin(age * 26.0 + r(seed, q, 6u) * TAU));
        if r(seed, q, 7u) < 0.3 {
            rgb = vec3<f32>(1.0, 0.97, 0.88);
        }
        turn = r(seed, q, 8u) * TAU + (r(seed, q, 9u) - 0.5) * 6.0 * age;
    }
    // The quad: the sprite and its glow.
    var reach = half + vec2<f32>(1.0);
    if pop > 0.5 {
        reach = half + vec2<f32>(half.y * 1.5 + 1.0);
    }
    let p = at + corner * reach;
    o.pos = vec4<f32>(p.x / screen.size.x * 2.0 - 1.0, 1.0 - p.y / screen.size.y * 2.0, 0.0, 1.0);
    o.p = p;
    o.shape = vec4<f32>(at, half);
    o.look = vec4<f32>(rgb, alpha);
    o.what = vec4<f32>(pop, turn, 0.0, 0.0);
    return o;
}

fn out_rgb(c: vec3<f32>) -> vec3<f32> {
    if screen.srgb > 0.5 {
        return pow(c, vec3<f32>(2.2));
    }
    return c;
}

fn round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let half = o.shape.zw;
    if o.what.x > 0.5 {
        // The pop: a capsule, white hot, and its glow; light, added.
        let d = round_box(o.p - o.shape.xy, half, min(half.x, half.y));
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let g = max(d, 0.0) / max(half.y * 1.5, 1.0);
        // Gone by the quad's edge, no rim.
        let glow = exp(-g * g * 3.0) * (1.0 - smoothstep(0.75, 1.0, g));
        let light = (cover * 0.8 + glow * 0.6) * o.look.a;
        return vec4<f32>(out_rgb(o.look.rgb) * light, 0.0);
    }
    // A sparkle: a core and two crossed rays, turned.
    let s = sin(o.what.y);
    let co = cos(o.what.y);
    let q0 = o.p - o.shape.xy;
    let q = abs(vec2<f32>(co * q0.x - s * q0.y, s * q0.x + co * q0.y));
    let size = max(half.x, 0.5);
    let core = exp(-dot(q, q) / (size * size * 0.05));
    let thin = max(size * 0.07, 0.8);
    let ray_x = exp(-q.y * q.y / (thin * thin)) * pow(clamp(1.0 - q.x / size, 0.0, 1.0), 1.6);
    let ray_y = exp(-q.x * q.x / (thin * thin)) * pow(clamp(1.0 - q.y / size, 0.0, 1.0), 1.6);
    let halo = exp(-dot(q, q) / (size * size * 0.18)) * 0.35;
    let light = (core + max(ray_x, ray_y) * 0.9 + halo) * o.look.a;
    return vec4<f32>(out_rgb(o.look.rgb) * light, 0.0);
}
";

/// The particles' pipeline, and the bursts it draws (the stage's and the
/// picture's: [`super::stage_gpu::StageFrame::sprays`]), uploaded only
/// when they change.
pub struct ParticlePass {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    sprays: Option<wgpu::Buffer>,
    /// The bursts in the buffer.
    kept: Vec<Spray>,
    srgb: bool,
}

impl ParticlePass {
    /// For frames of `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> ParticlePass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom particles"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom particles"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let source = shader();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom particles"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4];
        // Light, added: premultiplied, with no coverage.
        let added = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom particles"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Packed>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
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
                    blend: Some(wgpu::BlendState {
                        color: added,
                        alpha: added,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom particles"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom particles"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        ParticlePass {
            pipeline,
            bind,
            uniforms,
            sprays: None,
            kept: Vec::new(),
            srgb: format.is_srgb(),
        }
    }

    /// The next frame's bursts and the singing's time (`frame`; none:
    /// none), in a window of `size`: the time, and the bursts where they
    /// changed.
    pub fn set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: Option<&super::stage_gpu::StageFrame>,
        size: (u32, u32),
    ) {
        let (sprays, time) = frame.map_or((&[][..], 0.0), |f| (&f.sprays[..], f.clock));
        if self.kept.as_slice() != sprays {
            self.kept = sprays.to_vec();
            let packed: Vec<Packed> = sprays.iter().map(Spray::packed).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&packed);
            let fits = self
                .sprays
                .as_ref()
                .is_some_and(|b| b.size() >= bytes.len() as u64);
            if !fits && !bytes.is_empty() {
                self.sprays = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jodeldiplom particles"),
                    size: (bytes.len() as u64).next_power_of_two().max(1024),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            if let (Some(buffer), false) = (&self.sprays, bytes.is_empty()) {
                queue.write_buffer(buffer, 0, bytes);
            }
        }
        let mut uniforms = [0u8; 16];
        uniforms[0..4].copy_from_slice(&(size.0 as f32).to_le_bytes());
        uniforms[4..8].copy_from_slice(&(size.1 as f32).to_le_bytes());
        uniforms[8..12].copy_from_slice(&(if self.srgb { 1f32 } else { 0.0 }).to_le_bytes());
        uniforms[12..16].copy_from_slice(&time.to_le_bytes());
        queue.write_buffer(&self.uniforms, 0, &uniforms);
    }
    /// Draws the bursts into `pass` (nothing without any).
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        let (Some(buffer), false) = (&self.sprays, self.kept.is_empty()) else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..6 * QUADS, 0..self.kept.len() as u32);
    }
}

/// `stage` (none: none) with the picture's bursts `sprays` over it too,
/// at `clock` (the singing's time) where there is no stage.
pub fn with_sprays(
    stage: Option<super::stage_gpu::StageFrame>,
    sprays: &[super::particles::Spray],
    clock: f32,
) -> Option<super::stage_gpu::StageFrame> {
    match stage {
        Some(mut f) => {
            f.sprays.extend_from_slice(sprays);
            Some(f)
        }
        None => (!sprays.is_empty()).then(|| super::stage_gpu::StageFrame {
            sprays: sprays.to_vec(),
            clock,
            ..Default::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bubble(kind: Kind) -> Spray {
        Spray {
            at: (400.0, 300.0),
            half: (30.0, 4.0),
            start: 10.0,
            drift: 0.0,
            size: 60.0,
            seed: 7,
            rgb: [1.0, 0.75, 0.24],
            kind,
        }
    }

    #[test]
    fn a_bubble_pops_at_once_and_settles_back() {
        assert_eq!(popped(0.0), 0.0);
        assert!((popped(POP * 0.2) - 1.0).abs() < 1e-6);
        assert!(popped(POP * 0.6) < 0.5);
        assert_eq!(popped(POP), 0.0);
        assert_eq!(popped(0.9), 0.0);
    }

    /// A bubble's sparkles fly up off it, slowing, and fade; nothing shows
    /// before its start or after its life.
    #[test]
    fn the_sparkles_fly_up_and_fade() {
        let s = bubble(Kind::Bubble);
        let at = |q: u32, t: f32| sprite(&s, q, 10.0 + t);
        assert!(at(1, -0.01).is_none() && at(1, 0.5).is_none());
        for q in 1..=BUBBLE_SPARKS {
            let (a, b, c) = (
                at(q, 0.05).unwrap(),
                at(q, 0.25).unwrap(),
                at(q, 0.45).unwrap(),
            );
            assert!(c.at.1 < b.at.1 && b.at.1 < a.at.1, "up: {q}");
            // Slowing: the second half goes less far than the first.
            assert!((c.at.1 - b.at.1).abs() < (b.at.1 - a.at.1).abs(), "{q}");
            assert!(c.half.0 < a.half.0, "shrinking");
            assert!((a.at.0 - 400.0).abs() < 30.0 + 60.0, "off the bubble");
        }
        let fade = |t: f32| {
            (1..=BUBBLE_SPARKS)
                .map(|q| at(q, t).unwrap().alpha)
                .sum::<f32>()
        };
        assert!(fade(0.45) < 0.25 * fade(0.05));
        assert!(at(BUBBLE_SPARKS + 1, 0.1).is_none());
        // The pop: grown at once, gone by POP of the burst.
        let pop = at(0, 0.02).unwrap();
        assert!(pop.pop && pop.half.0 > 30.0 && pop.half.1 > 4.0);
        assert!(at(0, POP * BURST as f32 + 0.01).is_none());
        assert!(sprite(&bubble(Kind::Sparkles), 0, 10.02).is_none());
    }

    /// Drifting with a lane: each spark where it would be on a still one,
    /// moved by the drift; a burst shows from its start for its life.
    #[test]
    fn a_burst_on_a_lane_goes_by_with_it() {
        let still = bubble(Kind::Scored);
        let moving = Spray {
            drift: -200.0,
            ..still
        };
        let mut seen = 0;
        for q in 1..QUADS {
            let (a, b) = (sprite(&still, q, 10.2), sprite(&moving, q, 10.2));
            if let (Some(a), Some(b)) = (a, b) {
                assert!((b.at.0 - (a.at.0 - 40.0)).abs() < 1e-3 && b.at.1 == a.at.1);
                seen += 1;
            }
        }
        assert!(seen > 5, "{seen}");
        assert!(still.alive(10.5) && !still.alive(10.0 + 1.2) && !still.alive(9.9));
        // Further when the multiplier goes up.
        let up = Spray {
            kind: Kind::ScoredUp,
            ..still
        };
        let reach = |s: &Spray| {
            (1..QUADS)
                .filter_map(|q| sprite(s, q, 10.3))
                .map(|p| (p.at.0 - 400.0).hypot(p.at.1 - 300.0))
                .fold(0.0, f32::max)
        };
        assert!(reach(&up) > reach(&still));
    }
}
