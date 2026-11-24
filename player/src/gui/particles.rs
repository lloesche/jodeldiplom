//! Bursts on the GPU: a bubble bursting (once it has overfilled it tears,
//! its liquid splashing out, [`super::shards`]) and its sparkles flying
//! off, a word scored throwing sparks. The bubbles themselves, their
//! liquid and their overfilling are the stage's ([`super::stage_gpu`]).
//! The processor hands the GPU one small record a burst ([`Spray`],
//! [`Packed`]: where, when on the singing's clock, a seed, a [`Kind`]),
//! and the time each frame; the vertex shader works out every particle
//! from (seed, its index, its age) in closed form (out, falling, slowing,
//! shrinking, fading), drawn as instanced quads, [`QUADS`] a burst.
//! Nothing of it is stepped or rasterised on the processor.
//!
//! [`sprite`] is the same maths on the processor: the shader's reference,
//! held to it on the GPU (`super::shader_check`); the shader is given its
//! numbers ([`shader`]). Where there is no GPU (the software window, the
//! tests' pixmaps) the bursts are not drawn: a bubble burst is simply gone.

use super::shards::{self, DROPS, FIRST_DROP, FIRST_SHARD};
use super::sparks::{mix, rand, TAU, WHITE};
use super::{fanfare, rainbow};
use crate::view::bubbles::BURST;

/// The sparkles a bubble throws.
pub const BUBBLE_SPARKS: u32 = 5;
/// The quads of a burst: a bubble's sheen, its sparkles, the bits of its
/// skin, then the drops of its liquid; a word scored's sparks; a step up's
/// pieces (all of them).
pub const QUADS: u32 = 32;
/// How long a word scored throws sparks for, at most, in seconds.
pub const SCORED_LIFE: f32 = 1.1;
/// How far a word scored's sparks fly.
const SPREAD: f32 = 1.4;

/// What a burst is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A bubble bursting: tearing, its liquid splashing out, a sheen
    /// fading where it was, sparkles.
    Burst,
    /// A word scored: sparks out every way.
    Scored,
    /// The streak a level up: stars, notes and rainbow bits shot out from
    /// the playhead ([`super::rainbow::step_up`]).
    StepUp,
    /// A flight of angels descending, the singing Diva
    /// ([`super::fanfare::angel`]).
    Angels,
    /// A jackpot of coins and stars, a phrase Diva
    /// ([`super::fanfare::coin`]).
    Jackpot,
}

impl Kind {
    /// As the shader takes it.
    pub fn code(self) -> u32 {
        match self {
            Kind::Burst => 0,
            Kind::Scored => 1,
            Kind::StepUp => 2,
            Kind::Angels => 3,
            Kind::Jackpot => 4,
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
            color: [
                self.rgb[0],
                self.rgb[1],
                self.rgb[2],
                self.kind.code() as f32,
            ],
        }
    }

    /// Its seed as the GPU has it (a float holds 24 bits exactly).
    fn seed24(&self) -> u32 {
        self.seed & 0xFF_FFFF
    }

    /// How long it lasts, in seconds.
    pub fn life(&self) -> f32 {
        match self.kind {
            Kind::Scored => SCORED_LIFE,
            Kind::Burst => BURST as f32,
            Kind::StepUp => rainbow::STEP_LIFE,
            Kind::Angels => fanfare::ANGELS_LIFE,
            Kind::Jackpot => fanfare::JACKPOT_LIFE,
        }
    }

    /// Whether it shows at `time` (none before its start: sought back), as
    /// the GPU has the time (a float: a burst made at a moment shows then).
    pub fn alive(&self, time: f64) -> bool {
        let age = time as f32 - self.start;
        (0.0..self.life()).contains(&age)
    }
}

/// How a quad of a burst is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    /// A four-pointed sparkle, light.
    Sparkle,
    /// The wet sheen a bubble leaves: light.
    Sheen,
    /// A bit of its skin: an arc, over.
    Shard,
    /// A drop of its liquid: an ellipse along its way, over.
    Drop,
    /// A note (a step up's): a head, its stem and a flag, over.
    Note,
    /// A rainbow bit (a step up's): a small bar, over.
    Bit,
    /// An angel, its wings beating: light.
    Angel,
    /// A gold coin, spinning: light.
    Coin,
}

impl Look {
    /// As the shader takes it.
    pub fn code(self) -> u32 {
        match self {
            Look::Sparkle => 0,
            Look::Sheen => 1,
            Look::Shard => 2,
            Look::Drop => 3,
            Look::Note => 4,
            Look::Bit => 5,
            Look::Angel => 6,
            Look::Coin => 7,
        }
    }
}

/// What a quad of a burst is at a moment: its middle, half its size (a
/// sparkle: its rays' length both; a bit of skin: half its length and half
/// its thickness; a drop: half its length and its radius), how much it
/// shows (0 to 1), its colour, how it is drawn, and how it is turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sprite {
    /// Its middle, window pixels.
    pub at: (f32, f32),
    /// Half its width and height.
    pub half: (f32, f32),
    /// How much it shows.
    pub alpha: f32,
    /// Its colour, sRGB.
    pub rgb: [f32; 3],
    /// How it is drawn.
    pub look: Look,
    /// Its turn, radians.
    pub turn: f32,
}

/// Quad `q` of `s` at `time` (the singing's clock), none where it does not
/// show: the shader's maths ([`shader`]) on the processor.
pub fn sprite(s: &Spray, q: u32, time: f32) -> Option<Sprite> {
    let age = time - s.start;
    if age < 0.0 || age >= s.life() || q >= QUADS {
        return None;
    }
    match (s.kind, q) {
        (Kind::Burst, 0) => shards::sheen(s, age),
        (Kind::Burst, q) if q < FIRST_SHARD => sparkle(s, q, age),
        (Kind::Burst, q) if q < FIRST_DROP => shards::shard(s, q, age),
        (Kind::Burst, q) if q < FIRST_DROP + DROPS => shards::drop(s, q, age),
        (Kind::Scored, q) if q > 0 && q < SCORED_SPARKS => spark(s, q, age),
        (Kind::StepUp, q) => rainbow::step_up(s, q, age),
        (Kind::Angels, q) => fanfare::angel(s, q, age),
        (Kind::Jackpot, q) => fanfare::coin(s, q, age),
        _ => None,
    }
}

/// The sparks of a word scored (and a quad none).
const SCORED_SPARKS: u32 = 24;

const _: () = assert!(FIRST_DROP + DROPS <= QUADS && SCORED_SPARKS <= QUADS);

/// Sparkle `q` of bubble `s` `age` seconds into its burst: up and out
/// off it, slowing, shrinking, twinkling and fading.
fn sparkle(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let (x, y, hw, size) = (s.at.0, s.at.1, s.half.0, s.size);
    let t = age / BURST as f32;
    let fly = 1.0 - (1.0 - t).powi(3);
    let sx = x - hw + r(0) * 2.0 * hw;
    let (vx, vy) = ((r(1) - 0.5) * 1.6, -(0.35 + 0.65 * r(2)));
    let reach = size * (0.35 + 0.45 * r(3));
    let ray = size * (0.12 + 0.1 * r(4)) * (1.0 - 0.6 * t);
    Some(Sprite {
        at: (sx + vx * reach * fly + s.drift * age, y + vy * reach * fly),
        half: (ray, ray),
        alpha: (1.0 - t).powf(1.3) * (0.7 + 0.3 * (t * 22.0 + r(5) * TAU).sin()),
        rgb: mix(WHITE, s.rgb, 0.4),
        look: Look::Sparkle,
        turn: r(6) * TAU,
    })
}

/// Spark `q` of word scored `s` `age` seconds on: out every way, slowing,
/// shrinking, twinkling and fading, each in its own time.
fn spark(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let (x, y, size, spread) = (s.at.0, s.at.1, s.size, SPREAD);
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
            x + (r(0) - 0.5) * size * 0.6 + vx * slow + s.drift * age,
            y + (r(1) - 0.5) * size * 0.6 + vy * slow,
        ),
        half: (ray, ray),
        alpha: (1.0 - t).powf(1.2) * (0.6 + 0.4 * (age * 26.0 + r(6) * TAU).sin()),
        rgb: if r(7) < 0.3 { [1.0, 0.97, 0.88] } else { s.rgb },
        look: Look::Sparkle,
        turn: r(8) * TAU + (r(9) - 0.5) * 6.0 * age,
    })
}

/// The particles' shader: [`SHADER`] given [`sprite`]'s numbers, a
/// bubble's sheen, bits and drops ([`shards::wgsl`]), and a step up's
/// pieces ([`rainbow::particles_wgsl`]).
pub fn shader() -> String {
    format!(
        "const QUADS: u32 = {QUADS}u;
const BUBBLE_SPARKS: u32 = {BUBBLE_SPARKS}u;
const DROPS: u32 = {DROPS}u;
const SCORED_SPARKS: u32 = {SCORED_SPARKS}u;
const SCORED_LIFE: f32 = {SCORED_LIFE:?};
const K_BURST: u32 = {}u;
const K_SCORED: u32 = {}u;
const K_STEP_UP: u32 = {}u;
const K_ANGELS: u32 = {}u;
const K_JACKPOT: u32 = {}u;
const SPREAD: f32 = {SPREAD:?};
const LOOK_SPARKLE: u32 = {}u;
const LOOK_SHEEN: u32 = {}u;
const LOOK_SHARD: u32 = {}u;
const LOOK_DROP: u32 = {}u;
{SHADER}{}{}{}",
        Kind::Burst.code(),
        Kind::Scored.code(),
        Kind::StepUp.code(),
        Kind::Angels.code(),
        Kind::Jackpot.code(),
        Look::Sparkle.code(),
        Look::Sheen.code(),
        Look::Shard.code(),
        Look::Drop.code(),
        shards::wgsl(),
        rainbow::particles_wgsl(),
        fanfare::wgsl()
    )
}

/// The particles' shader: [`sprite`] in WGSL (`sprite_of`), and each
/// sprite drawn (its numbers given it by [`shader`]).
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

// What a quad shows: as `Sprite`; nothing unless `shows`.
struct Sp {
    at: vec2<f32>,
    half: vec2<f32>,
    rgb: vec3<f32>,
    alpha: f32,
    look: u32,
    turn: f32,
    shows: u32,
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

fn sparkle(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    var o: Sp;
    let t = age / BURST;
    let fly = 1.0 - pow(1.0 - t, 3.0);
    let sx = at.x - at.z + r(seed, q, 0u) * 2.0 * at.z;
    let vx = (r(seed, q, 1u) - 0.5) * 1.6;
    let vy = -(0.35 + 0.65 * r(seed, q, 2u));
    let reach = size * (0.35 + 0.45 * r(seed, q, 3u));
    let ray = size * (0.12 + 0.1 * r(seed, q, 4u)) * (1.0 - 0.6 * t);
    o.at = vec2<f32>(sx + vx * reach * fly + drift, at.y + vy * reach * fly);
    o.half = vec2<f32>(ray);
    o.alpha = pow(1.0 - t, 1.3) * (0.7 + 0.3 * sin(t * 22.0 + r(seed, q, 5u) * TAU));
    o.rgb = mix(vec3<f32>(1.0), rgb, 0.4);
    o.look = LOOK_SPARKLE;
    o.turn = r(seed, q, 6u) * TAU;
    o.shows = 1u;
    return o;
}

fn spark(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32, spread: f32) -> Sp {
    var o: Sp;
    let t = age / (0.4 + 0.7 * r(seed, q, 4u));
    if t >= 1.0 {
        return o;
    }
    let vx = (r(seed, q, 2u) - 0.65) * size * 4.0 * spread;
    let vy = (r(seed, q, 3u) - 0.5) * size * 4.0 * spread;
    let slow = (1.0 - exp(-2.4 * age)) / 2.4;
    let ray = size * (0.35 + 0.65 * r(seed, q, 5u)) * (0.45 + 0.55 * (1.0 - t));
    o.at = vec2<f32>(
        at.x + (r(seed, q, 0u) - 0.5) * size * 0.6 + vx * slow + drift,
        at.y + (r(seed, q, 1u) - 0.5) * size * 0.6 + vy * slow,
    );
    o.half = vec2<f32>(ray);
    o.alpha = pow(1.0 - t, 1.2) * (0.6 + 0.4 * sin(age * 26.0 + r(seed, q, 6u) * TAU));
    o.rgb = rgb;
    if r(seed, q, 7u) < 0.3 {
        o.rgb = vec3<f32>(1.0, 0.97, 0.88);
    }
    o.look = LOOK_SPARKLE;
    o.turn = r(seed, q, 8u) * TAU + (r(seed, q, 9u) - 0.5) * 6.0 * age;
    o.shows = 1u;
    return o;
}

// Quad `q` of a burst (as `In`) at `time`: `sprite`.
fn sprite_of(at: vec4<f32>, motion: vec4<f32>, color: vec4<f32>, q: u32, time: f32) -> Sp {
    var o: Sp;
    let k = u32(color.w);
    let size = motion.z;
    let seed = u32(motion.w);
    let age = time - motion.x;
    var life = SCORED_LIFE;
    if k == K_BURST {
        life = BURST;
    } else if k == K_ANGELS {
        life = ANGELS_LIFE;
    } else if k == K_JACKPOT {
        life = JACKPOT_LIFE;
    } else if k == K_STEP_UP {
        life = STEP_LIFE;
    }
    if age < 0.0 || age >= life || q >= QUADS {
        return o;
    }
    let drift = motion.y * age;
    if k == K_BURST {
        if q == 0u {
            return sheen(at, color.rgb, age, drift);
        }
        if q < FIRST_SHARD {
            return sparkle(at, color.rgb, size, seed, q, age, drift);
        }
        if q < FIRST_DROP {
            return shard(at, color.rgb, size, seed, q, age, drift);
        }
        if q < FIRST_DROP + DROPS {
            return drop(at, color.rgb, size, seed, q, age, drift);
        }
        return o;
    }
    if k == K_ANGELS {
        return angel(at, color.rgb, size, seed, q, age);
    }
    if k == K_JACKPOT {
        return coin(at, color.rgb, size, seed, q, age);
    }
    if k == K_STEP_UP {
        return step_up(at, size, seed, q, age, drift);
    }
    if k == K_SCORED && q > 0u && q < SCORED_SPARKS {
        return spark(at, color.rgb, size, seed, q, age, drift, SPREAD);
    }
    return o;
}

@vertex
fn vs(@builtin(vertex_index) v: u32, i: In) -> Out {
    var o: Out;
    // Nothing (a quad of no size) unless it shows.
    o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let q = v / 6u;
    var corners = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let c = corners[v % 6u];
    let corner = vec2<f32>(f32(c & 1u), f32(c >> 1u)) * 2.0 - vec2<f32>(1.0);
    let sp = sprite_of(i.at, i.motion, i.color, q, screen.time);
    if sp.shows == 0u {
        return o;
    }
    // The quad: the sprite and its glow.
    let half = sp.half;
    var reach = half + vec2<f32>(1.0);
    if sp.look == LOOK_SHEEN {
        reach = half + vec2<f32>(half.y + 2.0);
    } else if sp.look == LOOK_SHARD {
        reach = vec2<f32>(half.x + half.y * 3.0 + 2.0);
    } else if sp.look == LOOK_DROP {
        reach = vec2<f32>(half.x + 2.0);
    } else if sp.look == LOOK_ANGEL {
        reach = half * 1.9 + vec2<f32>(4.0);
    } else if sp.look == LOOK_COIN {
        reach = half * 1.5 + vec2<f32>(2.0);
    } else if sp.look == LOOK_NOTE || sp.look == LOOK_BIT {
        reach = vec2<f32>(length(half) * 1.5 + 2.0);
    }
    let p = sp.at + corner * reach;
    o.pos = vec4<f32>(p.x / screen.size.x * 2.0 - 1.0, 1.0 - p.y / screen.size.y * 2.0, 0.0, 1.0);
    o.p = p;
    o.shape = vec4<f32>(sp.at, half);
    o.look = vec4<f32>(sp.rgb, sp.alpha);
    o.what = vec4<f32>(f32(sp.look), sp.turn, 0.0, 0.0);
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
    let look = u32(o.what.x);
    if look == LOOK_DROP {
        return draw_drop(o.p, o.shape.xy, half, o.what.y, o.look);
    }
    if look == LOOK_SHARD {
        return draw_shard(o.p, o.shape.xy, half, o.what.y, o.look);
    }
    if look == LOOK_SHEEN {
        return draw_sheen(o.p, o.shape.xy, half, o.look);
    }
    if look == LOOK_NOTE {
        return draw_note(o.p, o.shape.xy, half, o.what.y, o.look);
    }
    if look == LOOK_ANGEL {
        return draw_angel(o.p, o.shape.xy, half, o.what.y, o.look);
    }
    if look == LOOK_COIN {
        return draw_coin(o.p, o.shape.xy, half, o.what.y, o.look);
    }
    if look == LOOK_BIT {
        return draw_bit(o.p, o.shape.xy, half, o.what.y, o.look);
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
    linear: bool,
}

impl ParticlePass {
    /// For frames of `format`, the bursts going on as `blend` says.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        blend: super::stage_gpu::Blend,
    ) -> ParticlePass {
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
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4
        ];
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
                    blend: Some(blend.state()),
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
            linear: super::stage_gpu::takes_linear(format),
        }
    }

    /// The next frame's bursts at `time` (the singing's), in a window of
    /// `size`: the time, and the bursts where they changed.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        sprays: &[Spray],
        time: f32,
        size: (u32, u32),
    ) {
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
        uniforms[8..12].copy_from_slice(&(if self.linear { 1f32 } else { 0.0 }).to_le_bytes());
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

    /// A bubble's sparkles fly up off it, slowing, and fade; nothing shows
    /// before its start or after its life.
    #[test]
    fn the_sparkles_fly_up_and_fade() {
        let s = bubble(Kind::Burst);
        let at = |q: u32, t: f32| sprite(&s, q, 10.0 + t);
        assert!(at(1, -0.01).is_none() && at(1, 0.5).is_none());
        for q in 1..=BUBBLE_SPARKS {
            let (a, b, c) = (
                at(q, 0.05).unwrap(),
                at(q, 0.25).unwrap(),
                at(q, 0.45).unwrap(),
            );
            assert_eq!(a.look, Look::Sparkle);
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
        assert!(sprite(&s, QUADS, 10.2).is_none());
    }

    /// Drifting with a lane: each spark where it would be on a still one,
    /// moved by the drift; a burst shows from its start for its life.
    #[test]
    fn a_burst_on_a_lane_goes_by_with_it() {
        for still in [bubble(Kind::Scored), bubble(Kind::Burst)] {
            let moving = Spray {
                drift: -200.0,
                ..still
            };
            let mut seen = 0;
            for q in 0..QUADS {
                let (a, b) = (sprite(&still, q, 10.2), sprite(&moving, q, 10.2));
                if let (Some(a), Some(b)) = (a, b) {
                    assert!((b.at.0 - (a.at.0 - 40.0)).abs() < 1e-3 && b.at.1 == a.at.1);
                    seen += 1;
                }
            }
            assert!(seen > 5, "{seen}");
        }
        let still = bubble(Kind::Scored);
        assert!(still.alive(10.5) && !still.alive(10.0 + 1.2) && !still.alive(9.9));
        let burst = bubble(Kind::Burst);
        assert!(burst.alive(10.45) && !burst.alive(10.5));
        // A step up goes further, with all its quads.
        let up = Spray {
            kind: Kind::StepUp,
            ..still
        };
        let reach = |s: &Spray| {
            (1..QUADS)
                .filter_map(|q| sprite(s, q, 10.3))
                .map(|p| (p.at.0 - 400.0).hypot(p.at.1 - 300.0))
                .fold(0.0, f32::max)
        };
        assert!(reach(&up) > reach(&still));
        assert_eq!(
            (0..QUADS)
                .filter(|&q| sprite(&up, q, 10.05).is_some())
                .count(),
            32
        );
        assert!(up.alive(11.0) && !up.alive(10.01 + rainbow::STEP_LIFE as f64));
    }
}
