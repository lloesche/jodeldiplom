//! The karaoke game's effects, one module each, as data: a [`Spec`] an
//! effect, registered by one line ([`Fx`]), walked by the stage and the
//! shaders alike. A tier's row lists what it shows ([`Use`]:
//! [`crate::tiers::TierSpec::fx`]: `held`, `on_phrase`, `on_rung`, `at_rung`);
//! the [`Board`] eases each effect's strength from the singing frame by
//! frame (and moves it with the music, [`Mod`]); the stage asks the
//! effects to draw at fixed points of its draw order ([`At`], its
//! [`Layer`]), each drawing what its slot takes ([`Out`]).
//!
//! An effect draws in one or more ways:
//!
//! - **Stage shapes,** instances of its own kinds ([`Fx::kind`]), drawn by
//!   a pipeline of its own ([`crate::gui::stage::gpu::Group::Shapes`]) with
//!   only its WGSL (`<ident>_shape`): the rainbow's bands, the lane's glow, the
//!   flames.
//! - **Screen passes,** shapes over the finished frame (in the window
//!   pass's "fx" step), given the lyric band to keep mostly clear and the
//!   score's corner, the multiplier's badge and the meter to keep clear
//!   ([`Clear`]): the edges on fire, the marquee bulbs, the fog.
//! - **Lyric treatments,** a function of a lyric's colour and coverage at
//!   a pixel of its text (`lyric_<ident>`), each its own pipeline
//!   ([`crate::gui::stage::gpu::Group::Lyric`]), on the shouts (the multiplier,
//!   the verdict) or the line sung ([`Text`]): the rainbow, the lyrics on
//!   fire and frosted.
//! - **Bursts,** records the particles' pass works out
//!   ([`super::particles::MOTIONS`]): a step up's, the jackpot, the
//!   angels, the coins.
//! - **Big words** over everything ([`Layer::Over`]), lettered by the
//!   stage from its atlas ([`Letters`]) and drawn by the effect's own
//!   pipeline, shaking the screen as they will: the word slam.
//!
//! A fired effect can play out over the frames after it ([`Playing`]:
//! the rain's bursts on the beats, the slam waiting for a bar's first
//! beat), what fired it known ([`By`]: a phrase reaching a row, or a
//! row's rung).
//!
//! Each effect's WGSL names its helpers after it, so they all share the
//! stage's shader; a test holds them apart. Its maths is in Rust too, the
//! shader's reference, held to it on the GPU ([`Spec::reference`],
//! `super::shader_check`).

pub mod angels;
pub mod board;
pub mod bulbs;
pub mod coins;
pub mod edge_fire;
pub mod flames;
pub mod fog;
pub mod glow;
mod helpers;
pub mod jackpot;
pub mod lyric_fire;
pub mod lyric_frost;
pub mod marquee;
pub mod music;
pub mod rainbow;
mod screen_pass;
pub mod slam;
pub mod step_up;
pub mod wgsl;

pub use board::{Board, Frame, Live};
pub use music::Music;

use super::particles::Spray;
use super::stage::LaneBox;
use crate::gui::stage::gpu::Instance;

macro_rules! registry {
    ($($fx:ident => $spec:path,)*) => {
        /// An effect of the library, by its [`Spec`].
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Fx {
            $(#[doc = concat!("Its spec: `", stringify!($spec), "`.")] $fx,)*
        }

        impl Fx {
            /// Every effect, in the order they draw within a slot.
            pub const ALL: [Fx; Fx::COUNT] = [$(Fx::$fx,)*];
            /// How many there are.
            pub const COUNT: usize = [$(stringify!($fx),)*].len();

            /// What it is.
            pub const fn spec(self) -> &'static Spec {
                match self {
                    $(Fx::$fx => &$spec,)*
                }
            }
        }
    };
}

// The order within a slot: the lane's glow before its flames keeps the
// light's order; the fog under the fire and the bulbs.
registry! {
    Rainbow => rainbow::SPEC,
    StepUp => step_up::SPEC,
    Jackpot => jackpot::SPEC,
    Glow => glow::SPEC,
    Flames => flames::SPEC,
    Angels => angels::SPEC,
    Fog => fog::SPEC,
    EdgeFire => edge_fire::SPEC,
    Bulbs => bulbs::SPEC,
    LyricFire => lyric_fire::SPEC,
    LyricFrost => lyric_frost::SPEC,
    Fountain => coins::FOUNTAIN,
    Rain => coins::RAIN,
    Flick => coins::FLICK,
    Slam => slam::SPEC,
}

/// The first kind an effect's shapes have ([`Fx::kind`]): the stage's
/// own kinds are below it.
pub const FX_KINDS: u32 = 32;
/// The most kinds of shape an effect has.
pub const KINDS_EACH: u32 = 4;

impl Fx {
    /// Its place in [`Fx::ALL`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Its shapes' kind `n` ([`Spec::kinds`]): one number scheme for every
    /// effect, above the stage's own.
    pub const fn kind(self, n: u32) -> f32 {
        (FX_KINDS + KINDS_EACH * self as u32 + n) as f32
    }

    /// The effect and its kind's place among its kinds, of a shape of kind
    /// `k` (none for the stage's own kinds).
    pub fn of_kind(k: f32) -> Option<(Fx, usize)> {
        let k = (k.round() as u32).checked_sub(FX_KINDS)?;
        let fx = *Fx::ALL.get((k / KINDS_EACH) as usize)?;
        let n = (k % KINDS_EACH) as usize;
        (n < fx.spec().kinds.len()).then_some((fx, n))
    }

    /// Its lyric treatment's code on text ([`Treat`]; 0: none).
    pub const fn treatment_code(self) -> f32 {
        (self as u32 + 1) as f32
    }

    /// The effect whose treatment text of code `code` is in.
    pub fn of_treatment(code: f32) -> Option<Fx> {
        let i = (code.round() as usize).checked_sub(1)?;
        Fx::ALL.get(i).copied().filter(|f| f.spec().lyric.is_some())
    }
}

/// A setting of an effect a [`Mod`] moves: its strength, or one of its
/// look's four numbers ([`Spec::params`], as its module names them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// How strongly it shows.
    Strength,
    /// Its look's number `n` (0 to 3).
    Look(u8),
}

/// A setting moved by the music, as a synth routes a modulation: `param`
/// gains `amount` times `by`'s value ([`Music::get`]) each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mod {
    /// What it moves.
    pub param: Param,
    /// What moves it.
    pub by: jodel_looks::Driver,
    /// How far.
    pub amount: f32,
}

/// An effect as a row lists it: which, how strong (0 to 1: how much of
/// it shows, its light, its size, as the effect has it), and how the
/// music moves it (a fired effect's strength not: it fires as listed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Use {
    /// The effect.
    pub fx: Fx,
    /// How strong.
    pub strength: f32,
    /// How the music moves it.
    pub mods: &'static [Mod],
}

impl Use {
    /// `fx` at `strength`, still to the music.
    pub const fn of(fx: Fx, strength: f32) -> Use {
        Use {
            fx,
            strength,
            mods: &[],
        }
    }
}

/// The effects a tier's row shows: those held while the singing is at it,
/// those a phrase reaching it fires, those reaching its rung of the
/// multiplier fires, and those held while at its rung.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Effects {
    /// Held while at the tier.
    pub held: &'static [Use],
    /// Fired by a phrase reaching it.
    pub on_phrase: &'static [Use],
    /// Fired by reaching its rung.
    pub on_rung: &'static [Use],
    /// Held while at its rung.
    pub at_rung: &'static [Use],
}

impl Effects {
    /// None at all.
    pub const NONE: Effects = Effects {
        held: &[],
        on_phrase: &[],
        on_rung: &[],
        at_rung: &[],
    };

    /// Each of its lists.
    pub fn lists(&self) -> [&'static [Use]; 4] {
        [self.held, self.on_phrase, self.on_rung, self.at_rung]
    }

    /// Whether it lists `fx` anywhere.
    pub fn lists_fx(&self, fx: Fx) -> bool {
        self.lists()
            .iter()
            .flat_map(|l| l.iter())
            .any(|u| u.fx == fx)
    }
}

/// How strong `list` has `fx` (0: not at all).
pub fn strength(list: &[Use], fx: Fx) -> f32 {
    list.iter().find(|u| u.fx == fx).map_or(0.0, |u| u.strength)
}

/// Every effect's look ([`Spec::params`], moved by the music: [`Mod`]),
/// as the shaders read it (`looks`, every frame), the same in every
/// theme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Looks(pub [[f32; 4]; Fx::COUNT]);

impl Looks {
    /// Each as its spec has it.
    pub const SPECS: Looks = {
        let mut looks = [[0.0; 4]; Fx::COUNT];
        let mut i = 0;
        while i < Fx::COUNT {
            looks[i] = Fx::ALL[i].spec().params;
            i += 1;
        }
        Looks(looks)
    };
}

impl Default for Looks {
    fn default() -> Looks {
        Looks::SPECS
    }
}

/// The far corner (right, bottom) of an `area` (left, top, width,
/// height).
pub fn far_corner((x, y, w, h): (f32, f32, f32, f32)) -> (f32, f32) {
    (x + w, y + h)
}

/// Where the stage asks the effects to draw: a point of its draw order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Behind the tubes, over the lane's glass and the playhead: shapes.
    Under,
    /// Round the singer's arrow, while it shows: shapes.
    Arrow,
    /// From the playhead, after the word's burst: bursts over the stage.
    Playhead,
    /// From the meter as a phrase is decided: bursts of light.
    Meter,
    /// Round the lane: shapes of light.
    Lane,
    /// Over the whole screen: bursts of light.
    Sky,
    /// From a phrase's mark on the progress the moment it is decided:
    /// bursts of light.
    Mark,
    /// Over the finished frame: screen passes.
    Screen,
    /// Over everything, the light's bloom too: big words the stage letters
    /// ([`Letters`]), shapes, bursts over the stage, and the screen shaken.
    Over,
}

/// A slot of the stage's draw order with its geometry (pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum At {
    /// The playhead's middle on the lane, and the lane.
    Under {
        /// The playhead's middle.
        head: (f32, f32),
        /// The lane.
        lane: LaneBox,
    },
    /// The arrow's middle and its swirl's radius.
    Arrow {
        /// Its middle.
        at: (f32, f32),
        /// The swirl's radius.
        radius: f32,
    },
    /// The arrow's tip, a pill's height, the lane's pixels a second, where
    /// the word just scored hit ended (if one was), the stage's area.
    Playhead {
        /// The arrow's tip.
        tip: (f32, f32),
        /// A pill's height.
        pill: f32,
        /// The lane's pixels a second.
        pps: f32,
        /// Where the word just scored hit ended.
        word: Option<(f32, f32)>,
        /// The stage's area: left, top, width, height.
        area: (f32, f32, f32, f32),
    },
    /// The meter's right end and its height.
    Meter {
        /// Its right end.
        at: (f32, f32),
        /// Its height.
        size: f32,
    },
    /// The lane, a pill's height, the stage's area and the lyric band (its
    /// top and bottom).
    Lane {
        /// The lane.
        lane: LaneBox,
        /// A pill's height.
        pill: f32,
        /// The area: left, top, width, height.
        area: (f32, f32, f32, f32),
        /// The lyric band's top and bottom.
        band: (f32, f32),
    },
    /// The stage's area: left, top, width, height.
    Sky {
        /// The area.
        area: (f32, f32, f32, f32),
    },
    /// The phrase's mark, its size and the phrase's line.
    Mark {
        /// The mark's middle.
        at: (f32, f32),
        /// Its size.
        size: f32,
        /// The phrase's line.
        line: u32,
    },
    /// The stage's area, the lyric band (its top and bottom), what is kept
    /// clear and the lane.
    Screen {
        /// The area: left, top, width, height.
        area: (f32, f32, f32, f32),
        /// The lyric band's top and bottom.
        band: (f32, f32),
        /// What is kept clear.
        clear: Clear,
        /// The lane.
        lane: LaneBox,
    },
    /// The stage's area, the lyric band (its top and bottom) and the room
    /// big words land in.
    Over {
        /// The area: left, top, width, height.
        area: (f32, f32, f32, f32),
        /// The lyric band's top and bottom.
        band: (f32, f32),
        /// The room: left, top, right, bottom
        /// ([`crate::gui::stage::Stage::over_at`]).
        room: [f32; 4],
    },
}

/// What the effects over the stage keep clear, each a box (left, top,
/// right, bottom; none where it is not wider than nothing): the score's
/// corner ([`crate::gui::stage::score::corner`]), the multiplier's badge
/// and the meter with its verdict, where the stage places them
/// ([`crate::gui::stage::Stage::clear`]). A screen pass's shape carries
/// them as [`Clear::on`] puts them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Clear {
    /// The score's corner.
    pub score: [f32; 4],
    /// The multiplier's badge.
    pub badge: [f32; 4],
    /// The meter and its verdict.
    pub meter: [f32; 4],
}

impl Clear {
    /// Nothing kept clear.
    pub const NONE: Clear = Clear {
        score: [0.0; 4],
        badge: [0.0; 4],
        meter: [0.0; 4],
    };

    /// The boxes.
    pub fn boxes(&self) -> [[f32; 4]; 3] {
        [self.score, self.badge, self.meter]
    }

    /// `shape` carrying them, as a screen pass's shaders read them: the
    /// score's corner in `color2`, the badge in `color`, the meter in `uv`
    /// (each effect's WGSL reading them there).
    pub fn on(&self, shape: Instance) -> Instance {
        Instance {
            color: self.badge,
            color2: self.score,
            uv: self.meter,
            ..shape
        }
    }
}

impl At {
    /// Its slot.
    pub fn layer(&self) -> Layer {
        match self {
            At::Under { .. } => Layer::Under,
            At::Arrow { .. } => Layer::Arrow,
            At::Playhead { .. } => Layer::Playhead,
            At::Meter { .. } => Layer::Meter,
            At::Lane { .. } => Layer::Lane,
            At::Sky { .. } => Layer::Sky,
            At::Mark { .. } => Layer::Mark,
            At::Screen { .. } => Layer::Screen,
            At::Over { .. } => Layer::Over,
        }
    }
}

/// What a slot's effects draw into ([`Board::draw`]): its shapes and its
/// bursts (each slot takes one or the other, but [`Layer::Over`], which
/// takes both), the big words for the stage to letter (only over
/// everything), and how far they shake the screen (pixels: any slot's
/// effects may; the stage adds them all up for the frame).
pub struct Out<'a> {
    /// The shapes.
    pub shapes: &'a mut Vec<Instance>,
    /// The bursts.
    pub sprays: &'a mut Vec<Spray>,
    /// The words.
    pub letters: &'a mut Vec<Letters>,
    /// The shake.
    pub shake: &'a mut [f32; 2],
}

/// Big text an effect asks the stage to letter, in the stage's faces and
/// through its atlas: `text` in `face` at `px` (smaller where it would be
/// wider than `widest`), its box's foot (the middle of its bottom) at
/// `foot`, scaled about it by `scale`, its quad `margin` pixels wider all
/// round at its size (room for an outline), drawn as `shape` says (an
/// instance of the effect's kind with its colours and params; the stage
/// gives it its quad, its box (`shape`: left, top, right, bottom) and its
/// place in the atlas (`uv`)); glowing `glow` strong (a copy of it in the
/// light, which blooms, its `params[3]` the glow; 0: none); and a burst
/// thrown from its foot as wide as the word lettered (`half.0` given it:
/// half the box's width, at its size).
///
/// It is lettered only where the atlas has room for it as it is (never
/// starting the atlas again, which would take the places of text already
/// lettered in the frame): else not that frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Letters {
    /// The text.
    pub text: String,
    /// Its face.
    pub face: crate::gui::big_text::Face,
    /// Its size, pixels.
    pub px: f32,
    /// The widest it may be, pixels.
    pub widest: f32,
    /// Its box's foot.
    pub foot: (f32, f32),
    /// Its scale across and down, about its foot.
    pub scale: (f32, f32),
    /// Its quad's margin, pixels.
    pub margin: f32,
    /// How it is drawn.
    pub shape: Instance,
    /// How strongly it glows.
    pub glow: f32,
    /// The burst from its foot.
    pub burst: Option<Spray>,
}

/// What fired an effect: a phrase reaching a row's tier, or the singing
/// reaching a row's rung (each the row's place, Good's 0). Fired by both
/// in one frame, an effect fires once, by the phrase: the phrase's shout
/// wins and the rung's is let go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum By {
    /// A phrase reaching the row.
    Phrase(usize),
    /// The row's rung reached.
    Rung(usize),
}

/// A firing still playing out over the frames after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playing {
    /// What fired it.
    pub by: By,
    /// When it fired, on the singing's clock (sought back before it, it
    /// is gone).
    pub at: f32,
    /// How strongly it fired.
    pub strength: f32,
    /// Where its next step is, as the effect counts: the rain's next
    /// burst, a beat of the grid ([`Music::beats`]); the slam's landing, on
    /// the singing's clock.
    pub next: f32,
    /// The steps still to come: the rain's bursts; the slam's dust (1
    /// until it has landed).
    pub left: u32,
}

/// What an effect keeps between frames: when it last did what it does
/// every while (0 to begin with; sought back before it, as long ago as
/// can be), and a firing still playing out (sought back before it, none).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Slot {
    /// When, on the singing's clock.
    pub last: f32,
    /// The firing playing out.
    pub playing: Option<Playing>,
}

/// Which text a lyric treatment goes on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Text {
    /// What the stage shouts: the multiplier, a tier's name over the meter.
    Shout,
    /// The line sung.
    Sung,
}

/// A lyric treatment: the text it goes on, and how strong it is, as the
/// shader eases it (level, from, since: [`super::shine::shine_at`];
/// `(s, s, 0)` is `s` throughout), none where it shows nothing.
#[derive(Clone, Copy, Debug)]
pub struct Lyric {
    /// The text.
    pub on: Text,
    /// Its strength.
    pub strain: fn(&Live) -> Option<[f32; 3]>,
}

/// A lyric treatment on text: its code and its strength's easing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Treat {
    /// The treatment's code ([`Fx::treatment_code`]).
    pub code: f32,
    /// Its strength's easing (level, from, since).
    pub strain: [f32; 3],
}

impl Treat {
    /// `text` (an instance of text) in this treatment.
    pub fn apply(&self, text: &mut Instance) {
        let [a, b, c] = self.strain;
        text.params[3] = self.code;
        text.strain = [a, b, c, 0.0];
    }

    /// The treatment of `text`, if it is in one.
    pub fn of(text: &Instance) -> Option<(Fx, [f32; 3])> {
        let fx = Fx::of_treatment(text.params[3])?;
        let [a, b, c, _] = text.strain;
        Some((fx, [a, b, c]))
    }
}

/// The treatment `on` takes in `frame`: of the effects treating that text,
/// the strongest showing (the first registered of equals); none showing,
/// none.
pub fn treatment(on: Text, frame: &Frame) -> Option<Treat> {
    let mut best: Option<(f32, Treat)> = None;
    for fx in Fx::ALL {
        let Some(lyric) = fx.spec().lyric.filter(|l| l.on == on) else {
            continue;
        };
        let live = frame.live(fx);
        let Some(strain) = (lyric.strain)(&live) else {
            continue;
        };
        if best.is_none_or(|(s, _)| live.strength > s) {
            let code = fx.treatment_code();
            best = Some((live.strength, Treat { code, strain }));
        }
    }
    best.map(|(_, t)| t)
}

/// A shader held to its processor's reference on the GPU
/// (`super::shader_check`): samples of `ins` vec4s each, run through
/// `body` (WGSL over `i`, the sample, reading `ins` and writing `outs`
/// vec4s each) in the stage's shader, against `cpu` (a sample's ins to its
/// outs) within `near` (absolute, relative); at least `least` samples
/// whose processor's outs `shows` says show something (so the samples
/// reach where the effect draws).
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    /// The samples, flat.
    pub samples: fn() -> Vec<[f32; 4]>,
    /// The vec4s in a sample.
    pub ins: usize,
    /// The vec4s out of one.
    pub outs: usize,
    /// The WGSL.
    pub body: &'static str,
    /// The processor's.
    pub cpu: fn(&[[f32; 4]]) -> Vec<[f32; 4]>,
    /// How near.
    pub near: (f32, f32),
    /// Whether a sample's outs show something.
    pub shows: fn(&[[f32; 4]]) -> bool,
    /// The least showing something.
    pub least: usize,
}

/// An effect: its name (as frame_cost calls it), its WGSL
/// prefix (`<ident>_shape`, `lyric_<ident>`, `FX_<IDENT>`), the slots it
/// draws in and how ([`Spec::draw`]: with its strength and look this frame,
/// its own keeping, into the slot's shapes or bursts), its shapes' kinds'
/// names, its stage shapes' WGSL (functions with `fn <ident>_shape(o: Out,
/// p: vec2<f32>, k: u32) -> vec4<f32>`, premultiplied, as the stage's
/// shapes are: light comes with no coverage), its lyric treatment (with
/// `fn lyric_<ident>(col, a, px, quad, time, strength, look, theme) ->
/// vec4<f32>` in its WGSL: the text's straight colour and coverage), its
/// look's four numbers (the shaders' `looks[FX_<IDENT>]`: what it moves
/// by and how much, never how a theme dresses it, which the shaders read
/// from `theme`: [`crate::theme::Theme::uniform`]), its shader's reference,
/// whether its shapes keep still as the screen shakes (the edges on fire,
/// which are the screen's own: its pipeline's `STILL`), and whether, fired
/// by a phrase, it shouts the phrase's verdict itself (the word slam: the
/// row that fires it owns the moment, and the meter's verdict for the
/// phrase stays away).
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    /// Its name.
    pub name: &'static str,
    /// Its WGSL prefix.
    pub ident: &'static str,
    /// Its slots.
    pub at: &'static [Layer],
    /// How it draws.
    pub draw: fn(&At, &Live, &mut Slot, &mut Out),
    /// Its kinds' names.
    pub kinds: &'static [&'static str],
    /// Its stage shapes' WGSL (and its treatment's).
    pub wgsl: Option<fn() -> String>,
    /// Its lyric treatment.
    pub lyric: Option<Lyric>,
    /// Its look: `y` the speed every look shares (`helpers::speed`; 0 is
    /// taken as 1), the other numbers as its module names them; a number
    /// an effect does not read is 0 (a mod of it would move nothing: a test
    /// holds the rows and themes to that).
    pub params: [f32; 4],
    /// Its shader's reference.
    pub reference: Option<Reference>,
    /// Whether it keeps still as the screen shakes.
    pub still: bool,
    /// Whether it shouts a phrase's verdict.
    pub shouts: bool,
}

fn draws_nothing(_: &At, _: &Live, _: &mut Slot, _: &mut Out) {}

impl Spec {
    /// An effect of no slots, drawing nothing: the defaults.
    pub const NONE: Spec = Spec {
        name: "",
        ident: "",
        at: &[],
        draw: draws_nothing,
        kinds: &[],
        wgsl: None,
        lyric: None,
        params: [0.0; 4],
        reference: None,
        still: false,
        shouts: false,
    };

    /// Whether it draws screen passes.
    pub fn is_screen(&self) -> bool {
        self.at.contains(&Layer::Screen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::stage::gpu::{Group, Shape};
    use std::collections::HashSet;

    /// Every effect named apart, its WGSL prefix apart, its kinds within
    /// its share of the numbers and each kind's shapes drawn by its own
    /// pipeline; a treatment's code its effect's; a screen pass drawing
    /// shapes.
    #[test]
    fn the_registry_is_whole_and_its_names_unique() {
        let (mut names, mut idents) = (HashSet::new(), HashSet::new());
        for fx in Fx::ALL {
            let s = fx.spec();
            assert!(names.insert(s.name) && idents.insert(s.ident), "{fx:?}");
            let draws = !s.at.is_empty() || s.lyric.is_some();
            assert!(!s.name.is_empty() && draws, "{fx:?}");
            assert!(s.kinds.len() <= KINDS_EACH as usize, "{fx:?}");
            for (n, name) in s.kinds.iter().enumerate() {
                let k = fx.kind(n as u32);
                assert_eq!(Fx::of_kind(k), Some((fx, n)), "{fx:?} {n}");
                assert_eq!(Shape::name_of(k), *name);
                let i = Instance {
                    params: [k, 0.0, 0.0, 0.0],
                    ..Instance::default()
                };
                assert_eq!(Group::of(&i), Group::Shapes(fx), "{fx:?}");
            }
            assert_eq!(s.wgsl.is_some(), !s.kinds.is_empty() || s.lyric.is_some());
            if s.lyric.is_some() {
                assert_eq!(Fx::of_treatment(fx.treatment_code()), Some(fx));
                let text = Instance {
                    params: [Shape::Text.code(), 0.0, 0.0, fx.treatment_code()],
                    ..Instance::default()
                };
                assert_eq!(Group::of(&text), Group::Lyric(fx));
            } else {
                assert_eq!(Fx::of_treatment(fx.treatment_code()), None);
            }
            if s.is_screen() {
                assert!(!s.kinds.is_empty(), "{fx:?}");
            }
        }
        assert_eq!(Fx::of_kind(Shape::Pill.code()), None);
        assert_eq!(Fx::of_kind(FX_KINDS as f32 + 3.0), None, "a kind none has");
        assert!(Shape::COUNT <= FX_KINDS as usize);
    }

    /// No two effects' WGSL define the same function or constant: they
    /// all go into one shader.
    #[test]
    fn no_two_effects_define_the_same_wgsl_name() {
        let mut seen: HashSet<String> = HashSet::new();
        for fx in Fx::ALL {
            let Some(wgsl) = fx.spec().wgsl else {
                continue;
            };
            let code = wgsl();
            for line in code.lines().map(str::trim_start) {
                let name = ["fn ", "const ", "struct "]
                    .iter()
                    .find_map(|p| line.strip_prefix(p))
                    .and_then(|rest| rest.split(['(', ':', ' ', '=', '{']).next());
                if let Some(name) = name.filter(|n| !n.is_empty()) {
                    assert!(seen.insert(name.to_owned()), "{fx:?}: {name} again");
                }
            }
        }
        assert!(seen.len() > 20, "{seen:?}");
    }

    /// No row nor theme moves a number of an effect's look that the effect
    /// does not read (0 in its spec, but the speed every look shares).
    #[test]
    fn every_mod_moves_a_number_its_effect_reads() {
        let rows = crate::tiers::TIERS.iter().flat_map(|t| t.fx.lists());
        let themes = crate::theme::THEMES.iter().map(|t| t.always);
        for u in rows.chain(themes).flatten() {
            for m in u.mods {
                if let Param::Look(n @ (0 | 2 | 3)) = m.param {
                    let read = u.fx.spec().params[usize::from(n)] != 0.0;
                    assert!(read, "{:?}: look {n} is not one it reads", u.fx);
                }
            }
        }
    }

    static TREATED: [Use; 3] = [
        Use::of(Fx::Rainbow, 1.0),
        Use::of(Fx::LyricFire, 0.3),
        Use::of(Fx::LyricFrost, 0.6),
    ];

    /// The line sung takes the strongest of the treatments on it (the
    /// frost over the fire), eased on the processor; the shouts the
    /// rainbow while it shines, as its glow eases; none shining, none.
    #[test]
    fn the_line_sung_takes_the_strongest_treatment() {
        use crate::gui::shine::Glow;
        use crate::gui::streak::Step;
        let row = Effects {
            held: &TREATED,
            ..Effects::NONE
        };
        let rows = [(1.0, &row)];
        let glow = Glow {
            level: 1.0,
            from: 1.0,
            since: 0.0,
        };
        let step = Step {
            tier: Some(crate::tiers::Tier::ALL[0]),
            glow,
            ..Step::default()
        };
        let frame = Board::default().frame_of(rows.into_iter(), &step, &Music::NEUTRAL, 2.0);
        let sung = treatment(Text::Sung, &frame).expect("a treatment");
        assert_eq!(sung.code, Fx::LyricFrost.treatment_code());
        assert_eq!(sung.strain, [0.6, 0.6, 0.0]);
        let shout = treatment(Text::Shout, &frame).expect("the rainbow");
        assert_eq!(shout.code, Fx::Rainbow.treatment_code());
        assert_eq!(shout.strain, glow.triple());
        let none =
            Board::default().frame_of(rows.into_iter(), &Step::default(), &Music::NEUTRAL, 2.0);
        assert_eq!(treatment(Text::Sung, &none), None);
        assert_eq!(treatment(Text::Shout, &none), None);
    }
}
