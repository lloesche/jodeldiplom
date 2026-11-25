//! Bursts on the GPU: a bubble bursting (once it has overfilled it tears,
//! its liquid splashing out, [`super::shards`]) and its sparkles flying
//! off, a word scored throwing sparks, and the effects' bursts (a step up,
//! the jackpot, the angels, the coins: [`super::fx`]). The bubbles
//! themselves, their liquid and their overfilling are the stage's
//! ([`super::stage_gpu`]). The processor hands the GPU one small record a
//! burst ([`Spray`], [`Packed`]: where, when on the singing's clock, a
//! seed, a [`Kind`]), and the time each frame; the vertex shader works out
//! every particle from (seed, its index, its age) in closed form (out,
//! falling, slowing, shrinking, fading), drawn as instanced quads,
//! [`QUADS`] a burst. Nothing of it is stepped or rasterised on the
//! processor.
//!
//! Each kind moves as its line of [`MOTIONS`] says: how long it lasts, its
//! maths on the processor ([`sprite`], the shader's reference, held to it
//! on the GPU from the line's probes: `super::shader_check`) and its call
//! in the shader ([`shader`], which is given their numbers); each quad is
//! drawn as its look's line of [`LOOKS`] says. Where there is no GPU (the
//! software window, the tests' pixmaps) the bursts are not drawn: a
//! bubble burst is simply gone.

mod motions;

pub use motions::{Drawing, Motion, LOOKS, MOTIONS};

use super::fx::Music;
use super::shards::{self, DROPS, FIRST_DROP, FIRST_SHARD};
use super::sparks::{mix, rand, TAU, WHITE};
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

/// What a burst is: its line in [`MOTIONS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A bubble bursting: tearing, its liquid splashing out, a sheen
    /// fading where it was, sparkles.
    Burst,
    /// A word scored: sparks out every way.
    Scored,
    /// The streak a level up: stars, notes and rainbow bits shot out from
    /// the playhead ([`super::fx::step_up`]).
    StepUp,
    /// A flight of angels descending, the singing Diva
    /// ([`super::fx::angels`]).
    Angels,
    /// A jackpot of coins and stars, a phrase Diva
    /// ([`super::fx::jackpot`]).
    Jackpot,
    /// A shower of confetti, a new best ([`super::confetti::piece`]).
    Confetti,
    /// A fountain of gold coins out of the playhead
    /// ([`super::fx::coins::fountain`]).
    CoinFountain,
    /// Gold coins raining down the screen ([`super::fx::coins::rain`]).
    CoinRain,
    /// A coin or two flicked up out of a word hit
    /// ([`super::fx::coins::flick`]).
    CoinFlick,
}

impl Kind {
    /// How many there are: a line of [`MOTIONS`] each.
    pub const COUNT: usize = MOTIONS.len();

    /// As the shader takes it: its line in [`MOTIONS`].
    pub fn code(self) -> u32 {
        self as u32
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

    /// What falls over the whole screen (a shower of confetti, a flight
    /// of angels) starts in a band this many pixels each way across its
    /// middle (kept in `half.0`).
    pub fn band(&self) -> f32 {
        self.half.0
    }

    /// The screen's height it falls through (kept in `half.1`; the shaders
    /// read it as `at.w`).
    pub fn screen_height(&self) -> f32 {
        self.half.1
    }

    /// How long it lasts, in seconds.
    pub fn life(&self) -> f32 {
        MOTIONS[self.kind.code() as usize].life
    }

    /// Whether it shows at `time` (none before its start: sought back), as
    /// the GPU has the time (a float: a burst made at a moment shows then).
    pub fn alive(&self, time: f64) -> bool {
        let age = time as f32 - self.start;
        (0.0..self.life()).contains(&age)
    }
}

/// How a quad of a burst is drawn: its line in [`LOOKS`].
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
    /// A solid gold coin spinning on its edge, glinting: over.
    Gold,
}

impl Look {
    /// As the shader takes it.
    pub fn code(self) -> u32 {
        self as u32
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
    (MOTIONS[s.kind.code() as usize].quad)(s, q, age)
}

/// The sparks of a word scored (and a quad none).
const SCORED_SPARKS: u32 = 24;

const _: () = assert!(FIRST_DROP + DROPS <= QUADS && SCORED_SPARKS <= QUADS);

/// Quad `q` of bubble `s` `age` seconds into its burst: its sheen, a
/// sparkle, a bit of its skin or a drop of its liquid.
fn burst(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    match q {
        0 => shards::sheen(s, age),
        q if q < FIRST_SHARD => sparkle(s, q, age),
        q if q < FIRST_DROP => shards::shard(s, q, age),
        q if q < FIRST_DROP + DROPS => shards::drop(s, q, age),
        _ => None,
    }
}

/// Quad `q` of word scored `s` `age` seconds on: a spark (the first and
/// those past [`SCORED_SPARKS`] none).
fn scored(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    (q > 0 && q < SCORED_SPARKS)
        .then(|| spark(s, q, age))
        .flatten()
}

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

/// The particles' shader: [`SHADER`] given [`sprite`]'s numbers, each
/// kind's life, call and maths from its line of [`MOTIONS`] (`life_of`,
/// `motion_of`), and each look's reach, drawing and code from its line of
/// [`LOOKS`] (`LOOK_` and its name, `reach_of`, `draw_of`). One pipeline
/// draws every look: a quad's look is the same over its pixels, so the
/// fragment shader's choice of drawing is one branch (PERFORMANCE.md).
pub fn shader() -> String {
    let each = |what: &dyn Fn(&Motion) -> String| -> String {
        (MOTIONS.iter().enumerate())
            .map(|(k, m)| format!("    if k == {k}u {{\n        return {};\n    }}\n", what(m)))
            .collect()
    };
    let lives = each(&|m| format!("{:?}", m.life));
    let calls = each(&|m| m.call.to_owned());
    let motions: String = MOTIONS.iter().filter_map(|m| m.wgsl).map(|w| w()).collect();
    let look = |d: &Drawing| format!("LOOK_{}", format!("{:?}", d.look).to_uppercase());
    let codes: String = (LOOKS.iter())
        .map(|d| format!("const {}: u32 = {}u;\n", look(d), d.look.code()))
        .collect();
    let looks = |what: &dyn Fn(&Drawing) -> String| -> String {
        (LOOKS.iter())
            .map(|d| {
                format!(
                    "    if look == {} {{\n        return {};\n    }}\n",
                    look(d),
                    what(d)
                )
            })
            .collect()
    };
    let reaches = looks(&|d| d.reach.to_owned());
    let draws = looks(&|d| format!("{}(p, c, half, turn, color)", d.draw));
    let drawings: String = LOOKS.iter().map(|d| d.wgsl).collect();
    format!(
        "const QUADS: u32 = {QUADS}u;
const BUBBLE_SPARKS: u32 = {BUBBLE_SPARKS}u;
const DROPS: u32 = {DROPS}u;
const SCORED_SPARKS: u32 = {SCORED_SPARKS}u;
const SPREAD: f32 = {SPREAD:?};
{codes}{}{}{SHADER}
// How long a burst of kind `k` lasts.
fn life_of(k: u32) -> f32 {{
{lives}    return 0.0;
}}

// Quad `q` of a burst of kind `k`, `age` seconds on.
fn motion_of(k: u32, at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {{
{calls}    var o: Sp;
    return o;
}}

// How far a quad of `look` reaches each way from its middle, `half` half
// its size.
fn reach_of(look: u32, half: vec2<f32>) -> vec2<f32> {{
{reaches}    return half;
}}

// A quad of `look` at pixel `p`: its middle `c`, `half` half its size,
// turned by `turn`, `color` its colour and how much it shows.
fn draw_of(look: u32, p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, color: vec4<f32>) -> vec4<f32> {{
{draws}    return vec4<f32>(0.0);
}}
{motions}{drawings}",
        super::fx::music::WGSL,
        super::rainbow::HUE,
    )
}

/// The particles' shader: [`sprite`] in WGSL (`sprite_of`), a bubble's
/// and a word scored's quads, and each sprite drawn (its numbers, its
/// kinds' lives and calls given it by [`shader`]).
const SHADER: &str = r"
struct Screen {
    size: vec2<f32>,
    srgb: f32,
    time: f32,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<uniform> music: Music;

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

// Quad `q` of a bubble bursting: its sheen, a sparkle, a bit of its skin
// or a drop of its liquid.
fn burst(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    if q == 0u {
        return sheen(at, rgb, age, drift);
    }
    if q < FIRST_SHARD {
        return sparkle(at, rgb, size, seed, q, age, drift);
    }
    if q < FIRST_DROP {
        return shard(at, rgb, size, seed, q, age, drift);
    }
    if q < FIRST_DROP + DROPS {
        return drop(at, rgb, size, seed, q, age, drift);
    }
    var o: Sp;
    return o;
}

// Quad `q` of a word scored: a spark.
fn scored(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32, drift: f32) -> Sp {
    if q > 0u && q < SCORED_SPARKS {
        return spark(at, rgb, size, seed, q, age, drift, SPREAD);
    }
    var o: Sp;
    return o;
}

// Quad `q` of a burst (as `In`) at `time`: `sprite`.
fn sprite_of(at: vec4<f32>, motion: vec4<f32>, color: vec4<f32>, q: u32, time: f32) -> Sp {
    var o: Sp;
    let k = u32(color.w);
    let age = time - motion.x;
    if age < 0.0 || age >= life_of(k) || q >= QUADS {
        return o;
    }
    return motion_of(k, at, color.rgb, motion.z, u32(motion.w), q, age, motion.y * age);
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
    let p = sp.at + corner * reach_of(sp.look, half);
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
    return draw_of(u32(o.what.x), o.p, o.shape.xy, o.shape.zw, o.what.y, o.look);
}";

/// How a sparkle is drawn, in WGSL: a core and two crossed rays, turned,
/// light (its look's line, [`LOOKS`]).
pub const DRAW_SPARKLE: &str = r"
fn draw_sparkle(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let s = sin(turn);
    let co = cos(turn);
    let q0 = p - c;
    let q = abs(vec2<f32>(co * q0.x - s * q0.y, s * q0.x + co * q0.y));
    let size = max(half.x, 0.5);
    let core = exp(-dot(q, q) / (size * size * 0.05));
    let thin = max(size * 0.07, 0.8);
    let ray_x = exp(-q.y * q.y / (thin * thin)) * pow(clamp(1.0 - q.x / size, 0.0, 1.0), 1.6);
    let ray_y = exp(-q.x * q.x / (thin * thin)) * pow(clamp(1.0 - q.y / size, 0.0, 1.0), 1.6);
    let halo = exp(-dot(q, q) / (size * size * 0.18)) * 0.35;
    let light = (core + max(ray_x, ray_y) * 0.9 + halo) * look.a;
    return vec4<f32>(out_rgb(look.rgb) * light, 0.0);
}
";

/// The particles' pipeline, and the bursts it draws (the stage's and the
/// picture's: [`super::stage_gpu::StageFrame::sprays`]), uploaded only
/// when they change.
pub struct ParticlePass {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    /// The music this frame ([`super::fx::music::Music::uniform`]).
    music: wgpu::Buffer,
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
            entries: &[uniform_entry(0, 16), uniform_entry(1, MUSIC_BYTES)],
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
        let buffer = |size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom particles"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (uniforms, music) = (buffer(16), buffer(MUSIC_BYTES));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom particles"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: music.as_entire_binding(),
                },
            ],
        });
        ParticlePass {
            pipeline,
            bind,
            uniforms,
            music,
            sprays: None,
            kept: Vec::new(),
            linear: super::stage_gpu::takes_linear(format),
        }
    }

    /// The next frame's bursts at `time` (the singing's) with `music`, in
    /// a window of `size`: the time and the music, and the bursts where
    /// they changed.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        (sprays, music): (&[Spray], &Music),
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
        queue.write_buffer(&self.music, 0, bytemuck::cast_slice(&music.uniform()));
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

/// How many bytes the music's uniform is ([`super::fx::music::Music::uniform`]).
pub const MUSIC_BYTES: u64 = 64;

/// A uniform at `binding`, `size` bytes, both stages reading it.
pub fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::fx::step_up;

    /// Each kind's motion is its line, by its code; each look's drawing
    /// its line, its function in its code.
    #[test]
    fn every_kind_moves_and_every_look_is_drawn_by_its_own_line() {
        for (i, m) in MOTIONS.iter().enumerate() {
            assert_eq!(m.kind.code() as usize, i, "{:?}", m.kind);
            assert!(m.life > 0.0 && !m.probes.is_empty(), "{:?}", m.kind);
            assert!(m.probes.iter().all(|p| p.kind == m.kind), "{:?}", m.kind);
        }
        for (i, d) in LOOKS.iter().enumerate() {
            assert_eq!(d.look.code() as usize, i, "{:?}", d.look);
            assert!(d.wgsl.contains(&format!("fn {}(", d.draw)), "{:?}", d.look);
        }
    }

    /// The particles' fragment shader picks a quad's drawing by its look
    /// alone (flat over the quad), one branch a line of [`LOOKS`]: every
    /// look in the one pipeline, as a burst's quads are small and few
    /// (their step about 0.1 ms a frame at the Diva peak, PERFORMANCE.md),
    /// and nothing else chosen there. A look that costs more is a pipeline
    /// of its own.
    #[test]
    fn the_particles_pick_a_drawing_by_its_look_alone() {
        let why = "the particles' shader picks a drawing by its look alone \
                   (PERFORMANCE.md, the particles' pipeline)";
        let src = shader();
        let body = |name: &str| -> &str {
            let from = src.find(&format!("fn {name}(")).expect(name);
            let end = src[from..].find("\n}\n").expect("its end");
            &src[from..from + end]
        };
        let fs = body("fs");
        assert_eq!(fs.matches("return").count(), 1, "{why}: {fs}");
        assert!(
            fs.contains("return draw_of(u32(o.what.x),") && !fs.contains("if "),
            "{why}"
        );
        assert!(src.contains("@interpolate(flat) what"), "{why}");
        let draw_of = body("draw_of");
        let picks: Vec<&str> = (draw_of.lines().map(str::trim))
            .filter(|l| l.starts_with("if "))
            .collect();
        assert_eq!(picks.len(), LOOKS.len(), "{why}");
        assert!(
            picks.iter().all(|l| l.starts_with("if look == LOOK_")),
            "{why}: {picks:?}"
        );
    }

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
        assert!(up.alive(11.0) && !up.alive(10.01 + step_up::STEP_LIFE as f64));
    }
}
