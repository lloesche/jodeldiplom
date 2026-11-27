//! What a pass is ([`Pass`]): its knobs ([`Knob`]: name, meaning, range,
//! default, unit, how far the music may move it and how much light it
//! can change), what it costs, what moves it by itself ([`natural`]), its
//! WGSL and its maths on the processor, the same lines twice. A pass reads
//! the picture under it ([`Src::src`]) and, for the feedback, the frame
//! before ([`Src::prev`]); its numbers come as [`Ctx`] (`u` in WGSL).

use crate::drive::Driver;
use crate::math::{V2, V3};
use crate::Drive;

/// The families the passes come in, for the catalogue and the sheets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl serde::Serialize for Family {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
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
    /// Its unit.
    pub unit: Unit,
    /// How far the music may move it.
    pub swing: f32,
    /// How much light it moves, across its range.
    pub light: f32,
}

/// What a knob's number is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// A share of the effect, 0 to 1.
    Share,
    /// Pixels at 1080 lines.
    Px,
    /// Turns of the colour wheel or a circle, 0 to 1: it goes round.
    Turns,
    /// A count: a whole number.
    Steps,
    /// One size over another.
    Ratio,
}

impl Unit {
    /// Its name, as the catalogue has it.
    pub fn name(self) -> &'static str {
        match self {
            Unit::Share => "share",
            Unit::Px => "px",
            Unit::Turns => "turns",
            Unit::Steps => "steps",
            Unit::Ratio => "ratio",
        }
    }
}

impl serde::Serialize for Unit {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// A knob `name` that does `about`, from `lo` to `hi`, `default` unless
/// set, in `unit`; the music moves it up to `swing`, it moves `light` of
/// the screen's light.
pub const fn knob(
    name: &'static str,
    about: &'static str,
    (lo, hi, default): (f32, f32, f32),
    unit: Unit,
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
    /// a knob in [`Unit::Turns`] (a hue, an angle) goes round instead, and
    /// a count of [`Unit::Steps`] is a whole number.
    pub fn clamp(&self, v: f32) -> f32 {
        if !v.is_finite() {
            return self.default;
        }
        match self.unit {
            Unit::Turns => crate::math::fract(v),
            Unit::Steps => v.round().max(self.lo).min(self.hi),
            Unit::Share | Unit::Px | Unit::Ratio => v.max(self.lo).min(self.hi),
        }
    }

    /// Whether it goes round: a whole turn, its unit [`Unit::Turns`].
    pub fn wraps(&self) -> bool {
        self.unit == Unit::Turns
    }
}

/// How the music moves a knob unless a look says otherwise: `knob` by
/// `by`, `amount` of its swing (-1 to 1), as a look would write it.
pub const fn natural(knob: &'static str, by: Driver, amount: f32) -> Drive {
    Drive {
        knob: std::borrow::Cow::Borrowed(knob),
        by,
        amount,
    }
}

/// The mix of a layer as a knob, after its pass's own: how much of the
/// pass shows over what is under it, 0 to 1 (its own slot, [`MIX`]).
pub const MIX_KNOB: Knob = knob(
    "mix",
    "how much of the pass shows over what is under it",
    (0.0, 1.0, 1.0),
    Unit::Share,
    (crate::guard::MIX_SWING, crate::guard::MIX_LIGHT),
);

/// The mix's slot among a layer's values: after every knob's
/// ([`Pass::slots`]).
pub const MIX: usize = KNOBS;

/// A layer's values, a slot each: its knobs' ([`KNOBS`], in its pass's
/// order) and its mix ([`MIX`]).
pub type Values = [f32; KNOBS + 1];

/// A pass's numbers for a frame, as its shader reads them (`LookU`, bound
/// as `u`): the target's size (pixels, and their inverses), the song's
/// time (seconds as heard), the look's seed, the frame's step (seconds)
/// and whether the frame before is to be forgotten (1: a seek, a new
/// look); the knobs ([`KNOBS`], in the pass's order); the lyric band (its top
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
    pub k: [f32; KNOBS],
    /// Top, bottom, relief, ceiling.
    pub band: [f32; 4],
    /// The cover's colours (accent, second, ground), each with a 0.
    pub pal: [[f32; 4]; 3],
    /// Mix, then unused.
    pub mix: [f32; 4],
    /// The music data.
    pub data: [[f32; 4]; DATA],
}

/// The cover's colours where there is no cover (and a look is measured
/// with): warm, cool and near black.
pub const PALETTE: [[f32; 3]; 3] = [[0.9, 0.5, 0.2], [0.2, 0.4, 0.8], [0.1, 0.1, 0.12]];

/// The lyric band (its top and bottom, shares of the height) where none
/// is given: about where the lyrics are set.
pub const BAND: (f32, f32) = (0.68, 0.86);

/// The knobs a pass may have (whole vec4s of them on the GPU).
pub const KNOBS: usize = 8;

/// The vec4s of a pass's music data.
pub const DATA: usize = 16;

const _: () = assert!(KNOBS % 4 == 0, "the knobs fill whole vec4s");

/// What works a pass's music data out from the music.
pub type DataOf = fn(&crate::drive::Feed) -> [[f32; 4]; DATA];

impl Default for Ctx {
    fn default() -> Ctx {
        Ctx {
            size: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            time: [0.0; 4],
            k: [0.0; KNOBS],
            band: [BAND.0, BAND.1, 0.0, 1.0],
            pal: PALETTE.map(|c| [c[0], c[1], c[2], 0.0]),
            mix: [1.0, 0.0, 0.0, 0.0],
            data: [[0.0; 4]; DATA],
        }
    }
}

/// Its accessors, each a line here and its twin a line in [`ctx_wgsl`]
/// (its knob `k[i]` is `lk_k(i)` there; the size's two halves are
/// `u.size.xy` and `u.size.zw`).
impl Ctx {
    /// The music data's value `i` (0 to `DATA * 4 - 1`; `lk_d`).
    pub fn d(&self, i: usize) -> f32 {
        self.data[(i / 4).min(DATA - 1)][i % 4]
    }

    /// 1080 lines' share of a pixel (a knob's `px` in pixels: times it;
    /// `lk_scale`).
    pub fn scale(&self) -> f32 {
        self.size[1] / 1080.0
    }

    /// The pixels across and down (`u.size.xy`).
    pub fn size2(&self) -> V2 {
        crate::math::v2(self.size[0], self.size[1])
    }

    /// A pixel's share of the width and of the height (`u.size.zw`).
    pub fn px(&self) -> V2 {
        crate::math::v2(self.size[2], self.size[3])
    }

    /// The point `uv` in pixels at 1080 lines (`lk_px1080`).
    pub fn px1080(&self, uv: V2) -> V2 {
        uv * self.size2() / self.scale()
    }

    /// The point `uv` in the shapes' units: the height 1, the middle 0
    /// (`lk_aspect`).
    pub fn aspect(&self, uv: V2) -> V2 {
        (uv - 0.5) * crate::math::v2(self.size[0] / self.size[1], 1.0)
    }

    /// The cover's colour `i`.
    pub fn pal(&self, i: usize) -> V3 {
        let p = self.pal[i];
        crate::math::v3(p[0], p[1], p[2])
    }
}

/// The WGSL of [`Ctx`] (`LookU`, its knobs and data as whole vec4s) and
/// its accessors' twins.
pub fn ctx_wgsl() -> String {
    format!(
        "
struct LookU {{
    size: vec4<f32>,
    time: vec4<f32>,
    k: array<vec4<f32>, {knobs}>,
    band: vec4<f32>,
    pal0: vec4<f32>,
    pal1: vec4<f32>,
    pal2: vec4<f32>,
    mix: vec4<f32>,
    data: array<vec4<f32>, {DATA}>,
}};
fn lk_k(i: u32) -> f32 {{ return u.k[i / 4u][i % 4u]; }}
fn lk_d(i: u32) -> f32 {{ return u.data[min(i / 4u, {last}u)][i % 4u]; }}
fn lk_scale() -> f32 {{ return u.size.y / 1080.0; }}
fn lk_px1080(uv: vec2<f32>) -> vec2<f32> {{ return uv * u.size.xy / lk_scale(); }}
fn lk_aspect(uv: vec2<f32>) -> vec2<f32> {{ return (uv - 0.5) * vec2<f32>(u.size.x / u.size.y, 1.0); }}
",
        knobs = KNOBS / 4,
        last = DATA - 1,
    )
}

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
/// its family, what it does (a line for the model), its knobs ([`KNOBS`]
/// at most, `k[i]` in its maths and `lk_k(i)` in its WGSL), what it costs
/// (a whole-screen pass reading the picture once is 1), how much of it the
/// lyric band takes out (0 to 1), whether it reads the frame before, how
/// the music moves it by itself, what it goes well with (passes' and presets' names), its
/// WGSL (`fn look_<name>(uv: vec2<f32>) -> vec3<f32>`, reading `u`,
/// `lk_src` and `lk_prev`), its maths on the processor, and its music
/// data (none: zeros).
///
/// As the catalogue has it (serialised), it is what a look may ask of it:
/// its name, family, meaning, cost, knobs, natural drives (`moved_by`)
/// and pairs.
#[derive(Clone, Copy, serde::Serialize)]
pub struct Pass {
    /// Its name.
    pub name: &'static str,
    /// Its family.
    pub family: Family,
    /// What it does.
    pub about: &'static str,
    /// What it costs.
    pub cost: f32,
    /// Its knobs.
    pub knobs: &'static [Knob],
    /// How much the lyric band takes out.
    #[serde(skip)]
    pub relief: f32,
    /// Whether it reads the frame before.
    #[serde(skip)]
    pub history: bool,
    /// How the music moves it.
    #[serde(rename = "moved_by")]
    pub natural: &'static [Drive],
    /// What it goes well with.
    pub pairs: &'static [&'static str],
    /// Its WGSL.
    #[serde(skip)]
    pub wgsl: &'static str,
    /// Its maths.
    #[serde(skip)]
    pub cpu: fn(&Ctx, &dyn Src, V2) -> V3,
    /// Its music data.
    #[serde(skip)]
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
    /// taken out in the lyric band, no history, still to the music, the
    /// picture as it is.
    pub const BASE: Pass = Pass {
        name: "",
        family: Family::Grade,
        about: "",
        knobs: &[],
        cost: 1.0,
        relief: 0.0,
        history: false,
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

    /// Its slots: each knob at its place, then the mix at [`MIX`].
    pub fn slots(&self) -> impl Iterator<Item = (usize, &'static Knob)> {
        (self.knobs.iter().enumerate()).chain([(MIX, &MIX_KNOB)])
    }

    /// The knob in slot `slot` (none: a slot it has not).
    pub fn slot(&self, slot: usize) -> Option<&'static Knob> {
        self.slots().find(|&(i, _)| i == slot).map(|(_, k)| k)
    }

    /// What the music can move by `knob` (one of its knobs, or `mix`):
    /// its slot, how far the music may move it, and the light a unit of it
    /// moves; `None` for a knob it has not.
    pub fn drivable(&self, knob: &str) -> Option<(usize, f32, f32)> {
        (self.slots())
            .find(|(_, k)| k.name == knob)
            .map(|(i, k)| (i, k.swing, k.light / (k.hi - k.lo).max(1e-6)))
    }
}
