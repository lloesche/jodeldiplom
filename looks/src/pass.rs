//! What a pass is ([`Pass`]): its knobs ([`Knob`]: name, meaning, range,
//! default, unit, how far the music may move it and how much light it
//! can change), what it costs, what moves it by itself ([`Natural`]), its
//! WGSL and its maths on the processor, the same lines twice. A pass reads
//! the picture under it ([`Src::src`]) and, for the feedback, the frame
//! before ([`Src::prev`]); its numbers come as [`Ctx`] (`u` in WGSL).

use crate::drive::Driver;
use crate::math::{V2, V3};

/// The families the passes come in, for the catalogue and the sheets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// Colour: grades, tones, curves.
    Grade,
    /// Film stocks and their faults.
    Film,
    /// Print: halftones, xerox, woodcut, engraving.
    Print,
    /// Paint and drawing media.
    Paint,
    /// The picture turned into another medium: glass, tiles, thread,
    /// sensors.
    Medium,
    /// Old screens and tapes.
    Retro,
    /// Weather and light in the air.
    Atmosphere,
    /// Symmetry, warps, colour cycling.
    Psychedelic,
    /// The music drawn, and the frame fed back.
    Visualizer,
}

impl Family {
    /// Every family, in order.
    pub const ALL: [Family; 9] = [
        Family::Grade,
        Family::Film,
        Family::Print,
        Family::Paint,
        Family::Medium,
        Family::Retro,
        Family::Atmosphere,
        Family::Psychedelic,
        Family::Visualizer,
    ];

    /// Its name, as the catalogue has it.
    pub fn name(self) -> &'static str {
        match self {
            Family::Grade => "grade",
            Family::Film => "film",
            Family::Print => "print",
            Family::Paint => "paint",
            Family::Medium => "medium",
            Family::Retro => "retro",
            Family::Atmosphere => "atmosphere",
            Family::Psychedelic => "psychedelic",
            Family::Visualizer => "visualizer",
        }
    }
}

/// A knob of a pass: its name (as a look names it), what it does in a
/// line (for the model), its range and default, its unit, how far the
/// music may move it from where it is set (in its own units: a mod's
/// amount of 1 moves it that far), and how much it can change the
/// screen's mean light across its whole range (0 to 1, weighted by the
/// share of the screen it lights: the flash guard's measure, held to the
/// pass by a test).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Knob {
    /// Its name.
    pub name: &'static str,
    /// What it does.
    pub about: &'static str,
    /// The least it can be.
    pub lo: f32,
    /// The most.
    pub hi: f32,
    /// Where it is unless set.
    pub default: f32,
    /// Its unit: `share` (0 to 1 of the effect), `px` (pixels at 1080
    /// lines), `turns` (of the colour wheel or a circle), `steps`, ...
    pub unit: &'static str,
    /// How far the music may move it.
    pub swing: f32,
    /// How much light it moves, across its range.
    pub light: f32,
}

/// A knob `name` that does `about`, from `lo` to `hi`, `default` unless
/// set, in `unit`; the music moves it up to `swing`, it moves `light` of
/// the screen's light.
pub const fn knob(
    name: &'static str,
    about: &'static str,
    (lo, hi, default): (f32, f32, f32),
    unit: &'static str,
    (swing, light): (f32, f32),
) -> Knob {
    Knob {
        name,
        about,
        lo,
        hi,
        default,
        unit,
        swing,
        light,
    }
}

impl Knob {
    /// `v` held to its range (a number that is none at all: its default);
    /// a whole turn's knob (a hue, an angle: 0 to 1 `turns`) goes round
    /// instead, and a count of `steps` is a whole number.
    pub fn clamp(&self, v: f32) -> f32 {
        if !v.is_finite() {
            self.default
        } else if self.wraps() {
            crate::math::fract(v)
        } else if self.unit == "steps" {
            v.round().max(self.lo).min(self.hi)
        } else {
            v.max(self.lo).min(self.hi)
        }
    }

    /// Whether it goes round (a whole turn).
    pub fn wraps(&self) -> bool {
        self.unit == "turns" && self.lo == 0.0 && self.hi == 1.0
    }
}

/// How the music moves a knob unless a look says otherwise: `knob` by
/// `by`, `amount` of its swing (-1 to 1).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Natural {
    /// The knob.
    pub knob: &'static str,
    /// What moves it.
    pub by: Driver,
    /// How far, a share of its swing.
    pub amount: f32,
}

/// `knob` moved by `by`, `amount` of its swing.
pub const fn natural(knob: &'static str, by: Driver, amount: f32) -> Natural {
    Natural { knob, by, amount }
}

/// A pass's numbers for a frame, as its shader reads them (`LookU`, bound
/// as `u`): the target's size (pixels, and their inverses), the song's
/// time (seconds as heard), the look's seed, the frame's step (seconds)
/// and whether the frame before is to be forgotten (1: a seek, a new
/// look); the knobs (eight, in the pass's order); the lyric band (its top
/// and bottom, shares of the height), how much of the pass it takes out
/// there and the band's light's ceiling; the cover's three colours; the
/// layer's mix over what is under it; the pass's music data (a scope's
/// samples, a ring's bins, the sparks' ages).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ctx {
    /// Width, height, 1 / width, 1 / height.
    pub size: [f32; 4],
    /// Time, seed, step, forget.
    pub time: [f32; 4],
    /// The knobs.
    pub k: [f32; 8],
    /// Top, bottom, relief, ceiling.
    pub band: [f32; 4],
    /// The cover's colours (accent, second, ground), each with a 0.
    pub pal: [[f32; 4]; 3],
    /// Mix, then unused.
    pub mix: [f32; 4],
    /// The music data.
    pub data: [[f32; 4]; DATA],
}

/// The vec4s of a pass's music data.
pub const DATA: usize = 16;

/// What works a pass's music data out from the music.
pub type DataOf = fn(&crate::drive::Feed) -> [[f32; 4]; DATA];

impl Default for Ctx {
    fn default() -> Ctx {
        Ctx {
            size: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            time: [0.0; 4],
            k: [0.0; 8],
            band: [0.68, 0.86, 0.0, 1.0],
            pal: [
                [0.9, 0.5, 0.2, 0.0],
                [0.2, 0.4, 0.8, 0.0],
                [0.1, 0.1, 0.12, 0.0],
            ],
            mix: [1.0, 0.0, 0.0, 0.0],
            data: [[0.0; 4]; DATA],
        }
    }
}

impl Ctx {
    /// The vec4s it is, as the shader check sends it.
    pub fn as_vec4s(&self) -> Vec<[f32; 4]> {
        bytemuck::cast_slice::<Ctx, [f32; 4]>(std::slice::from_ref(self)).to_vec()
    }

    /// The vec4s it takes.
    pub const VEC4S: usize = std::mem::size_of::<Ctx>() / 16;

    /// The music data's value `i` (0 to 63).
    pub fn d(&self, i: usize) -> f32 {
        self.data[(i / 4).min(DATA - 1)][i % 4]
    }

    /// The pixels across and down, and 1080 lines' share of a pixel (a
    /// knob's `px` in pixels: times [`Ctx::scale`]).
    pub fn scale(&self) -> f32 {
        self.size[1] / 1080.0
    }

    /// The point `uv` in pixels at 1080 lines.
    pub fn px1080(&self, uv: V2) -> V2 {
        uv * crate::math::v2(self.size[0], self.size[1]) / self.scale()
    }

    /// The cover's colour `i`.
    pub fn pal(&self, i: usize) -> V3 {
        let p = self.pal[i];
        crate::math::v3(p[0], p[1], p[2])
    }
}

/// The WGSL of [`Ctx`].
pub const CTX_WGSL: &str = r"
struct LookU {
    size: vec4<f32>,
    time: vec4<f32>,
    k0: vec4<f32>,
    k1: vec4<f32>,
    band: vec4<f32>,
    pal0: vec4<f32>,
    pal1: vec4<f32>,
    pal2: vec4<f32>,
    mix: vec4<f32>,
    data: array<vec4<f32>, 16>,
};
fn lk_d(i: u32) -> f32 { return u.data[min(i / 4u, 15u)][i % 4u]; }
fn lk_scale() -> f32 { return u.size.y / 1080.0; }
fn lk_px1080(uv: vec2<f32>) -> vec2<f32> { return uv * u.size.xy / lk_scale(); }
fn lk_aspect(uv: vec2<f32>) -> vec2<f32> { return (uv - 0.5) * vec2<f32>(u.size.x / u.size.y, 1.0); }
fn lk_px() -> vec2<f32> { return u.size.zw; }
";

/// What a pass reads: the picture under it and the frame before (each
/// in the encoded values the passes work in, read smoothly, held to the
/// edge).
pub trait Src {
    /// The picture under the pass at `uv`.
    fn src(&self, uv: V2) -> V3;
    /// The frame before at `uv`.
    fn prev(&self, uv: V2) -> V3;
}

/// A pass: its name (as a look names it, and its WGSL's `look_<name>`),
/// its family, what it does (a line for the model), its knobs (eight at
/// most, as `u.k0` and `u.k1` hold them), what it costs (a whole-screen
/// pass reading the picture once is 1), how much of it the lyric band
/// takes out (0 to 1), whether it reads the frame before, whether its
/// picture is made of cells (a sample on a cell's edge may fall either
/// side on the GPU: the shader check lets a few go), how the music moves
/// it by itself, what it goes well with (passes' and presets' names), its
/// WGSL (`fn look_<name>(uv: vec2<f32>) -> vec3<f32>`, reading `u`,
/// `lk_src` and `lk_prev`), its maths on the processor, and its music
/// data (none: zeros).
#[derive(Clone, Copy)]
pub struct Pass {
    /// Its name.
    pub name: &'static str,
    /// Its family.
    pub family: Family,
    /// What it does.
    pub about: &'static str,
    /// Its knobs.
    pub knobs: &'static [Knob],
    /// What it costs.
    pub cost: f32,
    /// How much the lyric band takes out.
    pub relief: f32,
    /// Whether it reads the frame before.
    pub history: bool,
    /// Whether it is made of cells.
    pub cells: bool,
    /// How the music moves it.
    pub natural: &'static [Natural],
    /// What it goes well with.
    pub pairs: &'static [&'static str],
    /// Its WGSL.
    pub wgsl: &'static str,
    /// Its maths.
    pub cpu: fn(&Ctx, &dyn Src, V2) -> V3,
    /// Its music data.
    pub data: Option<DataOf>,
}

impl std::fmt::Debug for Pass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass").field("name", &self.name).finish()
    }
}

impl PartialEq for Pass {
    fn eq(&self, other: &Pass) -> bool {
        self.name == other.name
    }
}

fn as_it_is(_: &Ctx, s: &dyn Src, uv: V2) -> V3 {
    s.src(uv)
}

impl Pass {
    /// What a pass's definition starts from: no knobs, a cost of 1, none
    /// taken out in the lyric band, no history, no cells, still to the
    /// music, the picture as it is.
    pub const BASE: Pass = Pass {
        name: "",
        family: Family::Grade,
        about: "",
        knobs: &[],
        cost: 1.0,
        relief: 0.0,
        history: false,
        cells: false,
        natural: &[],
        pairs: &[],
        wgsl: "",
        cpu: as_it_is,
        data: None,
    };

    /// Its knob named `name`, and its place.
    pub fn knob(&self, name: &str) -> Option<(usize, &'static Knob)> {
        self.knobs.iter().enumerate().find(|(_, k)| k.name == name)
    }

    /// What the music can move by `knob` (one of its knobs, or `mix`):
    /// its slot (the knob's place, or [`crate::MIX`]), how far the music
    /// may move it, and the light a unit of it moves; `None` for a knob
    /// it has not.
    pub fn drivable(&self, knob: &str) -> Option<(usize, f32, f32)> {
        if knob == "mix" {
            return Some((crate::MIX, crate::guard::MIX_SWING, crate::guard::MIX_LIGHT));
        }
        self.knob(knob)
            .map(|(i, k)| (i, k.swing, k.light / (k.hi - k.lo).max(1e-6)))
    }

    /// Its knobs at their defaults.
    pub fn defaults(&self) -> [f32; 8] {
        let mut k = [0.0; 8];
        for (i, knob) in self.knobs.iter().enumerate() {
            k[i] = knob.default;
        }
        k
    }
}
