//! Karaoke mode's stage on the GPU: the note lane, the singer's arrow, its
//! trail and sparks, the lyrics at the bottom and the song's progress,
//! drawn over the pixmap each frame as instances of a few shapes, each a
//! signed distance in the fragment shader (a pill, a capsule, a star, a
//! disc, an arrow, a tube and its liquid) or a run of text from a glyph
//! atlas, and the effects' shapes ([`super::fx`]). Premultiplied: a
//! shape's light that is to add up (a glow, a spark) comes with no
//! coverage, so it adds; the rest goes over. Each group of kinds is drawn
//! by a pipeline of its own ([`Group`]); the diploma's paper, once it holds
//! still, is drawn once and read back after (`bake`). The bubbles' liquid,
//! strain and overfilling are drawn here for the stage's bubbles (tubes,
//! [`super::snake`]) and the picture's alike ([`Instance::liquid`],
//! [`super::liquid`]).
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

use super::liquid::{amplitude, GROW};

/// The atlas's width and height, in pixels.
pub const ATLAS: u32 = 2048;

/// The window's numbers as the stage's and the bursts' shaders read them
/// (`screen`, [`SCREEN_WGSL`]): the frame's size, whether it takes linear
/// light (1) or the colours as encoded (0), the atlas's side (the stage's
/// text), the singing's time, where the frame's corner is in the window's
/// pixels (none but on the diploma's paper's bake, `bake`) and how far
/// the screen is shaken (pixels across and down). Everything the two
/// passes draw shakes with it but the screen passes that keep still
/// ([`super::fx::Spec::still`]) and the paper's bake (its own numbers, no
/// shake: the paper read from it shakes as it is drawn); the backdrop and
/// the pixmap under the stage, drawn apart, never do.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

// SAFETY: plain f32s in a `repr(C)` struct, no padding (the vec2s'
// alignment is `pad`): any bytes are one.
unsafe impl bytemuck::Zeroable for Screen {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Screen {}

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

/// What a shape is ([`Instance::params`]`[0]`): the stage's own kinds,
/// numbered from 0 as [`kind::NAMES`] names them; an effect's are its
/// own ([`super::fx::Fx::kind`]).
pub mod kind {
    /// A rounded box: `shape` its centre and half size, `params[3]` its
    /// corner radius (0: round ends), `params[1]` its glow, `params[2]` its
    /// outline. Over in `color`, outlined and glowing in `color2`.
    pub const PILL: f32 = 0.0;
    /// The part of a rounded box (as [`PILL`], round ends) between `uv[0]`
    /// and `uv[1]` across: `color` at its top, `color2` at its bottom,
    /// glowing in `color2` by `params[1]`.
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
    /// it in `color`; turned by `params[2]`; in a lyric treatment where
    /// `params[3]` says (its code, its strength's easing in `strain[0..3]`:
    /// [`crate::gui::fx::Treat`]). Where `shape[3]` is set, it pops round
    /// (`shape[0]`, `shape[1]`) from `shape[2]` and shows `shape[3]`
    /// seconds, as a [`BADGE`] does.
    pub const TEXT: f32 = 6.0;
    /// A word's bubble, a tube ([`crate::gui::snake`]): round its middle
    /// line (`params[3]` points from `shape[3]` in the path buffer),
    /// `shape[2]` thick each way, bulging round `shape[0..2]`; straining and
    /// bursting as its [`Instance::strain`] says; in `color`, outlined
    /// `params[2]` and glowing `params[1]` in `color2`.
    ///
    /// [`Instance::strain`]: super::Instance::strain
    pub const TUBE: f32 = 7.0;
    /// A picture's bubble's liquid: a [`FILL`] (its ends `params[2]` round,
    /// 0: round) straining with its bubble, its front living where
    /// [`super::flags::FRONT`] (its fill jumped at `uv[2]`)
    /// ([`super::Instance::liquid`]).
    pub const LIQUID: f32 = 8.0;
    /// The shine meter ([`crate::gui::glory::meter_at`]): `shape` its middle
    /// and half size, `params[1..]` its fill from, to and since.
    pub const METER: f32 = 9.0;
    /// The ring of the time left ([`crate::gui::glory::ring_at`]): `shape`
    /// its middle, radius and thickness, `params[1]` the share left,
    /// `strain[0..3]` the shine.
    pub const RING: f32 = 10.0;
    /// A phrase's mark on the progress bar: a disc at (`shape[0]`,
    /// `shape[1]`), `shape[2]` across, in `color`, popping from `params[1]`
    /// ([`crate::gui::glory::mark_pop`]).
    pub const MARK: f32 = 11.0;
    /// A badge: a round [`PILL`] (`shape` its middle and half size) in
    /// `color`, ringed `params[3]` wide in `color2`, popping round its
    /// middle from `params[1]` (the singing's time) and showing
    /// `params[2]` seconds ([`crate::gui::glory::label_pop`]).
    pub const BADGE: f32 = 12.0;
    /// A tube's liquid ([`TUBE`]'s shape) along it from `uv[0]` to `uv[1]`
    /// across, as [`LIQUID`]'s.
    pub const TUBE_LIQUID: f32 = 13.0;
    /// A burst of rays round (`shape[0]`, `shape[1]`), `shape[2]` their
    /// reach, `shape[3]` of them, turned `params[1]`, in `color`: light
    /// ([`crate::gui::seal::rays_at`]).
    pub const RAYS: f32 = 14.0;
    /// A star sticker at (`shape[0]`, `shape[1]`), `shape[2]` its size,
    /// turned `shape[3]`, in `color`, glowing `params[1]` in `color2`, its
    /// glint at `params[2]`, its shadow `params[3]` dark
    /// ([`crate::gui::seal::sticker_at`]).
    pub const STAR: f32 = 15.0;
    /// A wax seal at (`shape[0]`, `shape[1]`), `shape[2]` its radius: from
    /// `color2` (deep) to `color` (light), glowing `params[1]`, its sheen
    /// at `params[2]`, its shadow `params[3]` dark
    /// ([`crate::gui::seal::seal_at`]).
    pub const SEAL: f32 = 16.0;
    /// The diploma's sheet: `shape` its middle and half size, `params[2]`
    /// its unit, unrolled down to `params[1]` from its middle; in `color`,
    /// its border in `color2` and `uv` ([`crate::gui::paper::paper_at`]).
    pub const PAPER: f32 = 17.0;
    /// A rubber stamp's frame: `shape` its middle and half size, turned
    /// `params[1]`, `params[2]` its unit; in `color`, its specks in
    /// `color2` ([`crate::gui::paper::stamp_at`]).
    pub const STAMP: f32 = 18.0;
    /// A signature at (`shape[0]`, `shape[1]`), `shape[2]` its size,
    /// written to `params[1]` along it, `params[2]` its pen's half width;
    /// in `color` ([`crate::gui::paper::signature_at`]).
    pub const SIGNATURE: f32 = 19.0;
    /// The diploma's paper read from its bake: the bake's pixels over the
    /// quad, `shape[0..2]` where its corner is (whole pixels). Only the
    /// stage's pass puts it in, for a [`PAPER`] held still
    /// (`super::bake`).
    pub const BAKED: f32 = 20.0;

    /// The stage's own kinds' names, by their numbers (the shaders' `K_`
    /// constants, in capitals).
    pub const NAMES: [&str; 21] = [
        "pill",
        "fill",
        "segment",
        "spark",
        "disc",
        "arrow",
        "text",
        "tube",
        "liquid",
        "meter",
        "ring",
        "mark",
        "badge",
        "tube liquid",
        "rays",
        "star",
        "seal",
        "paper",
        "stamp",
        "signature",
        "baked",
    ];

    /// The name of kind `k`: the stage's own, or an effect's
    /// ([`crate::gui::fx::Spec::kinds`]); `?` for none.
    pub fn name(k: f32) -> &'static str {
        let i = k.round();
        if (0.0..NAMES.len() as f32).contains(&i) {
            return NAMES[i as usize];
        }
        crate::gui::fx::Fx::of_kind(k).map_or("?", |(fx, n)| fx.spec().kinds[n])
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
    /// A bubble's strain (and its liquid's): its pressure, when it bursts,
    /// its wobble's phase, its [`flags`].
    pub strain: [f32; 4],
}

// SAFETY: plain f32s in a `repr(C)` struct, no padding: any bytes are one.
unsafe impl bytemuck::Zeroable for Instance {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Instance {}

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
            params: [kind::LIQUID, pour.tint.glow, radius, 0.0],
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
        let liquid = [kind::LIQUID, kind::TUBE_LIQUID].contains(&self.params[0]);
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
    /// The bursts showing ([`super::particles`]).
    pub sprays: Vec<super::particles::Spray>,
    /// The shapes that are light to bloom, drawn apart ([`super::bloom`]).
    pub lights: Vec<Instance>,
    /// The bursts that are light to bloom.
    pub light_sprays: Vec<super::particles::Spray>,
    /// The screen passes, over the light and under the shapes over
    /// everything ([`super::fx::Layer::Screen`]).
    pub screen: Vec<Instance>,
    /// The music this frame.
    pub music: super::fx::Music,
    /// The effects' looks this frame.
    pub looks: super::fx::Looks,
    /// How far the screen is shaken (pixels across and down): all of it
    /// but the screen passes, which are the screen's own.
    pub shake: [f32; 2],
    /// The singing's time, in seconds: what the bubbles' strain and the
    /// bursts go by.
    pub clock: f32,
}

impl StageFrame {
    /// A frame with nothing in it yet, at `clock` (the singing's time).
    pub fn bare(clock: f32) -> StageFrame {
        StageFrame {
            clock,
            ..StageFrame::default()
        }
    }
}

/// `stage` (none: none) with the picture's bubbles' liquid (`bubbles`)
/// and bursts (`sprays`) drawn over it too, at `clock` (the singing's
/// time) where there is no stage.
pub fn with_picture(
    stage: Option<StageFrame>,
    bubbles: &[Instance],
    sprays: &[super::particles::Spray],
    clock: f32,
) -> Option<StageFrame> {
    match stage {
        Some(mut f) => {
            f.instances.extend_from_slice(bubbles);
            f.sprays.extend_from_slice(sprays);
            Some(f)
        }
        None => (!bubbles.is_empty() || !sprays.is_empty()).then(|| StageFrame {
            instances: bubbles.to_vec(),
            sprays: sprays.to_vec(),
            ..StageFrame::bare(clock)
        }),
    }
}

/// What a stage pass draws next: its shapes (back to front), the screen
/// passes and those over everything (each drawn apart:
/// [`StagePass::draw_screen`], [`StagePass::draw_top`]), the text to put
/// into the atlas first, the music, the effects' looks, the screen's shake
/// and the singing's time.
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
    pub music: super::fx::Music,
    /// The effects' looks.
    pub looks: super::fx::Looks,
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
            shake: self.shake,
            clock: self.clock,
        }
    }

    /// Its light, as the light's pass draws it (no screen passes: the
    /// light has none), the text put into the atlas too: the light keeps
    /// an atlas of its own, as the stage's, for the effects' big words'
    /// light ([`super::fx::Letters::glow`]).
    pub fn light_feed(&self) -> Feed<'_> {
        // Nor tubes: their points are not sent with the light.
        let tube = |i: &Instance| [kind::TUBE, kind::TUBE_LIQUID].contains(&i.params[0]);
        debug_assert!(!self.lights.iter().any(tube), "a tube among the lights");
        Feed {
            instances: &self.lights,
            uploads: &self.uploads,
            music: self.music,
            looks: self.looks,
            shake: self.shake,
            clock: self.clock,
            ..Feed::default()
        }
    }
}
