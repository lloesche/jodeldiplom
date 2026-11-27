//! Karaoke mode's stage on the GPU: the note lane, the singer's arrow, its
//! trail and sparks, the lyrics, the count-in, the score and the song's
//! progress, drawn over the backdrop each frame as instances of a few
//! shapes, each a signed distance in the fragment shader (a pill, a
//! capsule, a star, a disc, an arrow, a tube and its liquid) or a run of text from a glyph
//! atlas, and the effects' shapes ([`crate::gui::fx`]). Premultiplied: a
//! shape's light that is to add up (a glow, a spark) comes with no
//! coverage, so it adds; the rest goes over. Each group of kinds is drawn
//! by a pipeline of its own ([`Group`]); the diploma's paper, once it holds
//! still, is drawn once and read back after (`bake`). The bubbles' liquid,
//! strain and overfilling are drawn here for the stage's bubbles (tubes,
//! [`super::shapes::snake`]) and the lyrics' set in the middle
//! alike ([`Instance::liquid`],
//! [`super::shapes::liquid`]).
//!
//! [`StageFrame`] is what the window gets each frame: the instances, the
//! screen passes, the music, and the text the atlas did not hold yet
//! ([`AtlasUpload`]).

mod bake;
mod groups;
mod pass;

#[cfg(test)]
pub use groups::every_effect_shader;
pub use groups::{shader, wgsl_name, Group};
pub use pass::{takes_linear, Blend, StagePass};

use super::shapes::liquid::{amplitude, GROW};

/// The atlas's width and height, in pixels.
pub const ATLAS: u32 = 2048;

/// The window's numbers as the stage's and the bursts' shaders read them
/// (`screen`, [`SCREEN_WGSL`]): the frame's size, whether it takes linear
/// light (1) or the colours as encoded (0), the atlas's side (the stage's
/// text), the singing's time, where the frame's corner is in the window's
/// pixels (none but on the diploma's paper's bake, `bake`) and how far
/// the screen is shaken (pixels across and down). Everything the two
/// passes draw shakes with it but the screen passes that keep still
/// ([`crate::gui::fx::Spec::still`]) and the paper's bake (its own numbers, no
/// shake: the paper read from it shakes as it is drawn); the backdrop and
/// the pixmap under the stage, drawn apart, never do.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Screen {
    /// The frame's width and height, pixels.
    pub size: [f32; 2],
    /// Whether it takes linear light.
    pub srgb: f32,
    /// The atlas's side.
    pub atlas: f32,
    /// The singing's time, seconds.
    pub time: f32,
    /// Unused (the origin's alignment).
    pub pad: f32,
    /// Where the frame's corner is.
    pub origin: [f32; 2],
    /// How far the screen is shaken.
    pub shake: [f32; 2],
    /// Unused (the size a multiple of 16 bytes).
    pub tail: [f32; 2],
}

/// [`Screen`] as the shaders declare it.
pub const SCREEN_WGSL: &str = r"
// The window's numbers (`stage_gpu::Screen`): the frame's size, linear
// light (1) or not, the atlas's side, the singing's time, where the
// frame's corner is (pixels; none but on the paper's bake) and how far the
// screen is shaken (pixels; `STILL` pipelines keep still).
struct Screen {
    size: vec2<f32>,
    srgb: f32,
    atlas: f32,
    time: f32,
    pad: f32,
    origin: vec2<f32>,
    shake: vec2<f32>,
    tail: vec2<f32>,
};
";

/// How many bytes [`Screen`] is.
pub const SCREEN_BYTES: u64 = std::mem::size_of::<Screen>() as u64;

/// Makes [`Shape`] from one table: each kind's name (the shaders' `K_`
/// constant, in capitals: [`wgsl_name`]) and the [`Group`] whose pipeline
/// draws it, numbered from 0 in its order.
macro_rules! kinds {
    ($($(#[$doc:meta])* $kind:ident: $name:literal, $group:ident;)*) => {
        /// What a shape is ([`Instance::params`]`[0]`, its [`Shape::code`]):
        /// the stage's own kinds; an effect's are its own
        /// ([`crate::gui::fx::Fx::kind`]), numbered above them.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum Shape {
            $($(#[$doc])* $kind,)*
        }

        impl Shape {
            /// Every kind, by its number.
            pub const ALL: [Shape; Shape::COUNT] = [$(Shape::$kind,)*];
            /// How many there are.
            pub const COUNT: usize = [$($name,)*].len();

            /// Its name.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Shape::$kind => $name,)*
                }
            }

            /// The group whose pipeline draws it (text in a lyric
            /// treatment aside: [`Group::of`]).
            pub const fn group(self) -> Group {
                match self {
                    $(Shape::$kind => Group::$group,)*
                }
            }
        }
    };
}

kinds! {
    /// A rounded box: `shape` its centre and half size, `params[3]` its
    /// corner radius (0: round ends), `params[1]` its glow, `params[2]` its
    /// outline. Over in `color`, outlined and glowing in `color2`.
    Pill: "pill", Plain;
    /// The part of a rounded box (as [`Shape::Pill`], round ends) between
    /// `uv[0]` and `uv[1]` across: `color` at its top, `color2` at its
    /// bottom, glowing in `color2` by `params[1]`.
    Fill: "fill", Plain;
    /// A capsule from (`shape[0]`, `shape[1]`) to (`shape[2]`,
    /// `shape[3]`), `params[1]` thick each way, glowing `params[2]` in
    /// `color2`; its core in `color`.
    Segment: "segment", Plain;
    /// A four-pointed spark at (`shape[0]`, `shape[1]`), its rays
    /// `shape[2]` long, turned by `shape[3]`: light, added.
    Spark: "spark", Plain;
    /// A disc at (`shape[0]`, `shape[1]`), `shape[2]` across each way, in
    /// `color`, glowing `shape[3]` in `color2`.
    Disc: "disc", Plain;
    /// An arrow pointing right, its tip at (`shape[0]`, `shape[1]`),
    /// `shape[2]` long: in `color`, glowing `params[1]` in `color2`.
    Arrow: "arrow", Plain;
    /// Text from the atlas: `uv` its texels there, drawn over the quad;
    /// left of `params[1]` (pixels across the window) in `color2`, right of
    /// it in `color`; turned by `params[2]`; in a lyric treatment where
    /// `params[3]` says (its code, its strength's easing in `strain[0..3]`:
    /// [`crate::gui::fx::Treat`]). Where `shape[3]` is set, it pops round
    /// (`shape[0]`, `shape[1]`) from `shape[2]` and shows `shape[3]`
    /// seconds, as a [`Shape::Badge`] does.
    Text: "text", Plain;
    /// A word's bubble, a tube ([`super::shapes::snake`]): round
    /// its middle line (`params[3]` points from `shape[3]` in the path buffer),
    /// `shape[2]` thick each way, bulging round `shape[0..2]`; straining and
    /// bursting as its [`Instance::strain`] says; in `color`, outlined
    /// `params[2]` and glowing `params[1]` in `color2`.
    Tube: "tube", Bubble;
    /// A picture's bubble's liquid: a [`Shape::Fill`] (its ends `params[2]`
    /// round, 0: round) straining with its bubble, its front living where
    /// [`flags::FRONT`] (its fill jumped at `uv[2]`)
    /// ([`Instance::liquid`]).
    Liquid: "liquid", Bubble;
    /// The shine meter ([`super::shapes::glory::meter_at`]):
    /// `shape` its middle and half size, `params[1..]` its fill from, to and
    /// since.
    Meter: "meter", Glory;
    /// The ring of the time left
    /// ([`super::shapes::glory::ring_at`]): `shape` its middle,
    /// radius and thickness, `params[1]` the share left, `strain[0..3]` the
    /// shine.
    Ring: "ring", Glory;
    /// A phrase's mark on the progress bar: a disc at (`shape[0]`,
    /// `shape[1]`), `shape[2]` across, in `color`, popping from `params[1]`
    /// ([`super::shapes::glory::mark_pop`]).
    Mark: "mark", Glory;
    /// A badge: a round [`Shape::Pill`] (`shape` its middle and half size)
    /// in `color`, ringed `params[3]` wide in `color2`, popping round its
    /// middle from `params[1]` (the singing's time) and showing
    /// `params[2]` seconds ([`super::shapes::glory::label_pop`]).
    Badge: "badge", Plain;
    /// A tube's liquid ([`Shape::Tube`]'s shape) along it from `uv[0]` to
    /// `uv[1]` across, as [`Shape::Liquid`]'s.
    TubeLiquid: "tube liquid", Bubble;
    /// A burst of rays round (`shape[0]`, `shape[1]`), `shape[2]` their
    /// reach, `shape[3]` of them, turned `params[1]`, in `color`: light
    /// ([`super::shapes::seal::rays_at`]).
    Rays: "rays", Seal;
    /// A star sticker at (`shape[0]`, `shape[1]`), `shape[2]` its size,
    /// turned `shape[3]`, in `color`, glowing `params[1]` in `color2`, its
    /// glint at `params[2]`, its shadow `params[3]` dark
    /// ([`super::shapes::seal::sticker_at`]).
    Star: "star", Seal;
    /// A wax seal at (`shape[0]`, `shape[1]`), `shape[2]` its radius: from
    /// `color2` (deep) to `color` (light), glowing `params[1]`, its sheen
    /// at `params[2]`, its shadow `params[3]` dark
    /// ([`super::shapes::seal::seal_at`]).
    Seal: "seal", Seal;
    /// The diploma's sheet: `shape` its middle and half size, `params[2]`
    /// its unit, unrolled down to `params[1]` from its middle; in `color`,
    /// its border in `color2` and `uv`
    /// ([`super::shapes::paper::paper_at`]).
    Paper: "paper", Paper;
    /// A rubber stamp's frame: `shape` its middle and half size, turned
    /// `params[1]`, `params[2]` its unit; in `color`, its specks in
    /// `color2` ([`super::shapes::paper::stamp_at`]).
    Stamp: "stamp", Paper;
    /// A signature at (`shape[0]`, `shape[1]`), `shape[2]` its size,
    /// written to `params[1]` along it, `params[2]` its pen's half width;
    /// in `color` ([`super::shapes::paper::signature_at`]).
    Signature: "signature", Paper;
    /// The diploma's paper read from its bake: the bake's pixels over the
    /// quad, `shape[0..2]` where its corner is (whole pixels). Only the
    /// stage's pass puts it in, for a [`Shape::Paper`] held still (`bake`).
    Baked: "baked", Baked;
}

impl Shape {
    /// Its number, as the shader takes it ([`Instance::params`]`[0]`).
    pub const fn code(self) -> f32 {
        self as u8 as f32
    }

    /// The stage's own kind numbered `code`, if it is one.
    pub fn of(code: f32) -> Option<Shape> {
        let i = code.round();
        (0.0..Shape::COUNT as f32)
            .contains(&i)
            .then(|| Shape::ALL[i as usize])
    }

    /// The name of the kind numbered `code`: the stage's own, or an
    /// effect's ([`crate::gui::fx::Spec::kinds`]); `?` for none.
    pub fn name_of(code: f32) -> &'static str {
        match Shape::of(code) {
            Some(k) => k.name(),
            None => crate::gui::fx::Fx::of_kind(code).map_or("?", |(fx, n)| fx.spec().kinds[n]),
        }
    }
}

/// The flags of a bubble or its liquid ([`Instance::strain`]`[3]`).
pub mod flags {
    /// It is bursting: overfilling from `strain[1]`, gone once it tears.
    pub const POPPING: u32 = 1;
    /// Its liquid's front is a living surface (it is not full).
    pub const FRONT: u32 = 2;
}

/// One shape, as the shader takes it: the quad it covers (`x0, y0, x1,
/// y1` in window pixels, its glow included) and what [`kind`] says of the
/// rest. Colours are sRGB, 0 to 1, with their alpha.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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
    /// A bubble's strain (and its liquid's): its pressure, when it bursts,
    /// its wobble's phase, its [`flags`].
    pub strain: [f32; 4],
}

/// How a bubble strains and bursts: how much it strains (0 to 1), its
/// wobble's phase (radians), and when it bursts (the singing's time), once
/// it has begun to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Strain {
    /// How much it strains.
    pub pressure: f32,
    /// Its wobble's phase.
    pub phase: f32,
    /// When it burst, while it overfills.
    pub popping: Option<f32>,
}

/// A bubble's skin: its body, its edge (outline and glow), how far its
/// glow reaches and how wide its outline is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skin {
    /// Its body.
    pub body: [f32; 4],
    /// Its edge.
    pub edge: [f32; 4],
    /// Its glow's reach, pixels.
    pub glow: f32,
    /// Its outline's width, pixels.
    pub outline: f32,
}

/// A liquid's colours: at its top, at its bottom, and how far it glows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tint {
    /// At its top.
    pub top: [f32; 4],
    /// At its bottom.
    pub bottom: [f32; 4],
    /// Its glow's reach, pixels.
    pub glow: f32,
}

/// A bubble's liquid: from and to across (pixels), whether its front is
/// living (it is not full), when its fill jumped (the singing's time), and
/// its tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pour {
    /// From and to across.
    pub span: (f32, f32),
    /// Whether its front is living.
    pub front: bool,
    /// When its fill jumped.
    pub jumped: f32,
    /// Its tint.
    pub tint: Tint,
}

impl Instance {
    /// A shape of the stage's own `kind` over `quad`, its own numbers
    /// `shape`; the rest none until set ([`Instance::colors`],
    /// [`Instance::params`], [`Instance::uv`]).
    pub fn of(kind: Shape, quad: [f32; 4], shape: [f32; 4]) -> Instance {
        Instance {
            quad,
            shape,
            params: [kind.code(), 0.0, 0.0, 0.0],
            ..Instance::default()
        }
    }

    /// It in `color` and `color2`.
    pub fn colors(self, color: [f32; 4], color2: [f32; 4]) -> Instance {
        Instance {
            color,
            color2,
            ..self
        }
    }

    /// It with its kind's three numbers ([`Shape`]: `params[1..]`).
    pub fn params(self, [a, b, c]: [f32; 3]) -> Instance {
        Instance {
            params: [self.params[0], a, b, c],
            ..self
        }
    }

    /// It with its atlas texels, or its fill's span ([`Shape`]).
    pub fn uv(self, uv: [f32; 4]) -> Instance {
        Instance { uv, ..self }
    }

    /// Whether it is a shape of the stage's own `kind`.
    pub fn is(&self, kind: Shape) -> bool {
        self.params[0] == kind.code()
    }

    /// The liquid of a bubble at `centre`, `half` its size, its ends
    /// `radius` round (0: round ends), poured as `pour` says, straining with
    /// it; none where too little shows.
    pub fn liquid(
        centre: (f32, f32),
        half: (f32, f32),
        radius: f32,
        pour: &Pour,
        strain: &Strain,
    ) -> Option<Instance> {
        if pour.span.1 - pour.span.0 < 0.5 {
            return None;
        }
        let ripple = if pour.front {
            amplitude(half.1) * 3.5 + 1.0
        } else {
            0.0
        };
        let (mx, my) = Self::room(half, pour.tint.glow, ripple);
        Some(Instance {
            quad: [centre.0 - mx, centre.1 - my, centre.0 + mx, centre.1 + my],
            shape: [centre.0, centre.1, half.0, half.1],
            color: pour.tint.top,
            color2: pour.tint.bottom,
            params: [Shape::Liquid.code(), pour.tint.glow, radius, 0.0],
            uv: [pour.span.0, pour.span.1, pour.jumped, 0.0],
            strain: Self::strain_of(strain, pour.front),
        })
    }

    /// How far a bubble `half` its size reaches from its centre, glowing
    /// `glow` and its front's waves `ripple` further across: bulged to
    /// 1 + [`GROW`].
    fn room(half: (f32, f32), glow: f32, ripple: f32) -> (f32, f32) {
        let big = 1.0 + GROW;
        (
            half.0 * big + glow + ripple + 1.0,
            half.1 * big + glow + 1.0,
        )
    }

    /// A bubble's `s` strain as the shader takes it, its liquid's front
    /// living where `front`.
    pub(super) fn strain_of(s: &Strain, front: bool) -> [f32; 4] {
        let mut bits = 0;
        if s.popping.is_some() {
            bits |= flags::POPPING;
        }
        if front {
            bits |= flags::FRONT;
        }
        [s.pressure, s.popping.unwrap_or(0.0), s.phase, bits as f32]
    }

    /// The strain a bubble or its liquid was made with.
    pub fn strained(&self) -> Strain {
        let bits = self.strain[3] as u32;
        Strain {
            pressure: self.strain[0],
            phase: self.strain[2],
            popping: (bits & flags::POPPING != 0).then_some(self.strain[1]),
        }
    }

    /// Whether this is a liquid whose front is living.
    pub fn living(&self) -> bool {
        let liquid = self.is(Shape::Liquid) || self.is(Shape::TubeLiquid);
        liquid && self.strain[3] as u32 & flags::FRONT != 0
    }
}

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
    /// The shapes over everything, the light's bloom too (a tier's name
    /// on its badge), back to front.
    pub top: Vec<Instance>,
    /// Text to put into the atlas first.
    pub uploads: Vec<AtlasUpload>,
    /// The tubes' middle lines, point by point (`x, y`, two unused).
    pub paths: Vec<[f32; 4]>,
    /// The bursts showing ([`crate::gui::particles`]).
    pub sprays: Vec<crate::gui::particles::Spray>,
    /// The shapes that are light to bloom, drawn apart ([`crate::gui::bloom`]).
    pub lights: Vec<Instance>,
    /// The bursts that are light to bloom.
    pub light_sprays: Vec<crate::gui::particles::Spray>,
    /// The screen passes, over the light and under the shapes over
    /// everything ([`crate::gui::fx::Layer::Screen`]).
    pub screen: Vec<Instance>,
    /// The music this frame.
    pub music: crate::gui::fx::Music,
    /// The effects' looks this frame.
    pub looks: crate::gui::fx::Looks,
    /// The theme the effects are dressed in ([`crate::theme::Theme::uniform`]).
    pub theme: &'static crate::theme::Theme,
    /// How far the screen is shaken (pixels across and down): all of it
    /// but the screen passes, which are the screen's own.
    pub shake: [f32; 2],
    /// The singing's time, in seconds: what the bubbles' strain and the
    /// bursts go by.
    pub clock: f32,
}

impl StageFrame {
    /// A frame with nothing in it yet, at `clock` (the singing's time),
    /// dressed in `theme`.
    pub fn bare(clock: f32, theme: &'static crate::theme::Theme) -> StageFrame {
        StageFrame {
            clock,
            theme,
            ..StageFrame::default()
        }
    }
}

/// What a stage pass draws next: its shapes (back to front), the screen
/// passes and those over everything (each drawn apart:
/// [`StagePass::draw_screen`], [`StagePass::draw_top`]), the text to put
/// into the atlas first, the music, the effects' looks, the theme they are
/// dressed in, the screen's shake and the singing's time.
#[derive(Clone, Copy, Debug, Default)]
pub struct Feed<'a> {
    /// The shapes.
    pub instances: &'a [Instance],
    /// The screen passes.
    pub screen: &'a [Instance],
    /// The shapes over everything.
    pub top: &'a [Instance],
    /// The text.
    pub uploads: &'a [AtlasUpload],
    /// The tubes' middle lines.
    pub paths: &'a [[f32; 4]],
    /// The music.
    pub music: crate::gui::fx::Music,
    /// The effects' looks.
    pub looks: crate::gui::fx::Looks,
    /// The theme.
    pub theme: &'static crate::theme::Theme,
    /// The screen's shake.
    pub shake: [f32; 2],
    /// The time.
    pub clock: f32,
}

impl StageFrame {
    /// Its stage, as the stage pass draws it.
    pub fn feed(&self) -> Feed<'_> {
        Feed {
            instances: &self.instances,
            screen: &self.screen,
            top: &self.top,
            uploads: &self.uploads,
            paths: &self.paths,
            music: self.music,
            looks: self.looks,
            theme: self.theme,
            shake: self.shake,
            clock: self.clock,
        }
    }

    /// Its light, as the light's pass draws it (no screen passes: the
    /// light has none), the text put into the atlas too: the light keeps
    /// an atlas of its own, as the stage's, for the effects' big words'
    /// light ([`crate::gui::fx::Letters::glow`]).
    pub fn light_feed(&self) -> Feed<'_> {
        // Nor tubes: their points are not sent with the light.
        let tube = |i: &Instance| i.is(Shape::Tube) || i.is(Shape::TubeLiquid);
        debug_assert!(!self.lights.iter().any(tube), "a tube among the lights");
        Feed {
            instances: &self.lights,
            uploads: &self.uploads,
            music: self.music,
            looks: self.looks,
            theme: self.theme,
            shake: self.shake,
            clock: self.clock,
            ..Feed::default()
        }
    }
}
