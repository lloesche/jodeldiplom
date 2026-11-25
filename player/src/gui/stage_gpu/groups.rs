//! Which pipeline draws a shape ([`Group`]): each is the stage's shader
//! with only its group's code (the rest left out where it is compiled), as
//! Apple's GPUs run a shader at the pace its dearest code sets, for every
//! pixel; a frame's shapes are drawn in runs of a group, in their order
//! ([`runs`]). The shader itself is composed here ([`shader`]).

use super::{kind, Instance};
use crate::gui::fx::{self, Fx};

/// A group of shapes drawn by one pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Pills, fills, capsules, sparks, discs, arrows, text and badges.
    Plain,
    /// The words' bubbles and the liquid in them.
    Bubble,
    /// The meter, the ring of the time left, the phrases' marks.
    Glory,
    /// The diploma's rays, stars and seal.
    Seal,
    /// The diploma's paper, stamps and signature.
    Paper,
    /// An effect's shapes ([`fx::Spec::kinds`]).
    Shapes(Fx),
    /// Text in an effect's lyric treatment ([`fx::Spec::lyric`]).
    Lyric(Fx),
}

/// The first code of the effects' shapes' groups, and of their lyric
/// treatments' ([`Group::code`]).
const SHAPES: u32 = 16;
const LYRIC: u32 = 48;
const _: () = assert!(Fx::COUNT as u32 <= LYRIC - SHAPES);

impl Group {
    /// How many codes there are ([`Group::code`] is below it).
    pub const CODES: usize = LYRIC as usize + Fx::COUNT;

    /// Every group, the stage's own then the effects' shapes' and their
    /// treatments'.
    pub fn all() -> Vec<Group> {
        let shapes = Fx::ALL.into_iter().filter(|f| !f.spec().kinds.is_empty());
        let lyrics = Fx::ALL.into_iter().filter(|f| f.spec().lyric.is_some());
        [
            Group::Plain,
            Group::Bubble,
            Group::Glory,
            Group::Seal,
            Group::Paper,
        ]
        .into_iter()
        .chain(shapes.map(Group::Shapes))
        .chain(lyrics.map(Group::Lyric))
        .collect()
    }

    /// The groups the tiers' rows use: the stage's own, and those of
    /// every effect a row lists ([`crate::tiers::TierSpec::fx`]). Their
    /// pipelines are made as the window opens; another effect's the first
    /// time it draws.
    pub fn used() -> Vec<Group> {
        let listed = |g: &Group| match g.effect() {
            Some(fx) => crate::tiers::TIERS.iter().any(|t| t.fx.lists_fx(fx)),
            None => true,
        };
        Group::all().into_iter().filter(listed).collect()
    }

    /// The groups used whose shapes can be light
    /// ([`super::StageFrame::lights`]): the plain shapes (sparks and
    /// discs) and the effects drawing round the lane, where the light is
    /// ([`fx::Layer::Lane`]).
    pub fn light() -> Vec<Group> {
        let lit = |g: &Group| match g.effect() {
            Some(fx) => fx.spec().at.contains(&fx::Layer::Lane),
            None => *g == Group::Plain,
        };
        Group::used().into_iter().filter(lit).collect()
    }

    /// The effect whose code its pipeline carries, if any.
    fn effect(self) -> Option<Fx> {
        match self {
            Group::Shapes(fx) | Group::Lyric(fx) => Some(fx),
            _ => None,
        }
    }

    /// The group that draws `i`: text in a treatment its treatment's, an
    /// effect's shape its effect's, the stage's own by their kind.
    pub fn of(i: &Instance) -> Group {
        let k = i.params[0];
        if k == kind::TEXT {
            return Fx::of_treatment(i.params[3]).map_or(Group::Plain, Group::Lyric);
        }
        if let Some((fx, _)) = Fx::of_kind(k) {
            return Group::Shapes(fx);
        }
        match k {
            kind::TUBE | kind::TUBE_LIQUID | kind::LIQUID => Group::Bubble,
            kind::METER | kind::RING | kind::MARK => Group::Glory,
            kind::RAYS | kind::STAR | kind::SEAL => Group::Seal,
            kind::PAPER | kind::STAMP | kind::SIGNATURE => Group::Paper,
            _ => Group::Plain,
        }
    }

    /// As the shader takes it (`GROUP`, and its `G_` constants): the
    /// stage's own 0 to 4, an effect's shapes [`SHAPES`] on, its
    /// treatment's [`LYRIC`] on.
    pub fn code(self) -> u32 {
        match self {
            Group::Plain => 0,
            Group::Bubble => 1,
            Group::Glory => 2,
            Group::Seal => 3,
            Group::Paper => 4,
            Group::Shapes(fx) => SHAPES + fx.index() as u32,
            Group::Lyric(fx) => LYRIC + fx.index() as u32,
        }
    }
}

/// `instances` in runs of a [`Group`] each, in their order, `from` the
/// first's index in the buffer.
pub(super) fn runs(instances: &[Instance], from: u32) -> Runs {
    let mut runs = Runs::new();
    for (i, s) in (from..).zip(instances) {
        let g = Group::of(s);
        match runs.last_mut() {
            Some((last, range)) if *last == g => range.end = i + 1,
            _ => runs.push((g, i..i + 1)),
        }
    }
    runs
}

/// Runs of a frame's instances, a [`Group`] each ([`runs`]).
pub(super) type Runs = Vec<(Group, std::ops::Range<u32>)>;

/// A name as the shader's constants have it: in capitals, words joined.
pub fn wgsl_name(name: &str) -> String {
    name.to_uppercase().replace(' ', "_")
}

/// The kinds as the shader takes them: the stage's own and every
/// effect's (`K_` and its name).
fn kinds_wgsl() -> String {
    let own = kind::NAMES
        .iter()
        .enumerate()
        .map(|(k, n)| (wgsl_name(n), k as u32));
    let effects = Fx::ALL.into_iter().flat_map(|fx| {
        (fx.spec().kinds.iter().enumerate())
            .map(move |(n, name)| (wgsl_name(name), fx.kind(n as u32) as u32))
    });
    own.chain(effects)
        .map(|(name, k)| format!("const K_{name}: u32 = {k}u;\n"))
        .collect()
}

/// The groups as the shader takes them ([`Group`]): `G_` and its name, and
/// where the effects' shapes' and treatments' begin.
fn groups_wgsl() -> String {
    let own = ["PLAIN", "BUBBLE", "GLORY", "SEAL", "PAPER"]
        .iter()
        .enumerate()
        .map(|(g, name)| format!("const G_{name}: u32 = {g}u;\n"));
    let parts = [("SHAPES", SHAPES), ("LYRIC", LYRIC)]
        .map(|(name, g)| format!("const G_{name}: u32 = {g}u;\n"));
    own.chain(parts).collect()
}

/// The stage's shader drawing `group`: `stage.wgsl`, its kinds
/// ([`kind`]), the liquid's maths ([`crate::gui::liquid::wgsl`]), a tube's
/// nearest point ([`crate::gui::snake::wgsl`]), the shine's clock
/// ([`crate::gui::rainbow::stage_wgsl`]), the glory
/// ([`crate::gui::glory::wgsl`]), the diploma's shapes
/// ([`crate::gui::seal::wgsl`], [`crate::gui::paper::wgsl`]), the effects
/// (only the group's own effect's code: [`fx::wgsl::wgsl`]) and the music
/// ([`fx::music::WGSL`]).
pub fn shader(group: Group) -> String {
    let own: Vec<Fx> = group.effect().into_iter().collect();
    compose(group, &own)
}

/// The stage's shader as the plain shapes' pipeline has it, with every
/// effect's code too: what the shaders' checks run the effects' maths in.
#[cfg(test)]
pub fn every_effect_shader() -> String {
    compose(Group::Plain, &Fx::ALL)
}

/// The stage's shader drawing `group`, with `effects`' code.
fn compose(group: Group, effects: &[Fx]) -> String {
    format!(
        "const GROUP: u32 = {}u;\n{}{}{}{}{}{}{}{}{}{}{}",
        group.code(),
        groups_wgsl(),
        kinds_wgsl(),
        include_str!("../stage.wgsl"),
        crate::gui::liquid::wgsl(),
        crate::gui::snake::wgsl(),
        crate::gui::rainbow::stage_wgsl(),
        crate::gui::glory::wgsl(),
        crate::gui::seal::wgsl(),
        crate::gui::paper::wgsl(),
        fx::wgsl::wgsl(effects),
        fx::music::WGSL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::fx::rainbow::{BANDS, SWIRL};

    /// Every group a code of its own below [`Group::CODES`]; the stage's
    /// own kinds' names in their order.
    #[test]
    fn every_group_has_a_code_of_its_own() {
        let all = Group::all();
        let mut codes: Vec<u32> = all.iter().map(|g| g.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), all.len());
        assert!(codes.iter().all(|&c| (c as usize) < Group::CODES));
        let named = [
            (kind::PILL, "pill"),
            (kind::TEXT, "text"),
            (kind::LIQUID, "liquid"),
            (kind::METER, "meter"),
            (kind::BADGE, "badge"),
            (kind::TUBE_LIQUID, "tube liquid"),
            (kind::SIGNATURE, "signature"),
        ];
        for (k, n) in named {
            assert_eq!(kind::name(k), n);
        }
        assert_eq!(kind::name(99.0), "?");
        let light = Group::light();
        for g in [
            Group::Plain,
            Group::Shapes(Fx::Glow),
            Group::Shapes(Fx::Flames),
        ] {
            assert!(light.contains(&g), "{g:?}");
        }
        assert!(!light.contains(&Group::Shapes(Fx::Rainbow)));
        let used = Group::used();
        assert!(used.contains(&Group::Lyric(Fx::Rainbow)) && used.contains(&Group::Seal));
        assert!(
            !used.contains(&Group::Shapes(Fx::EdgeFire)),
            "no row lists it"
        );
    }

    /// Each pipeline's shader carries only its own effect's code (the
    /// effects' shared helpers only with an effect's), and picks what it
    /// draws by its group's constant alone (`GROUP ==`): no effect's code
    /// in another's pipeline, nor a choice among effects or groups made
    /// pixel by pixel. That is what keeps the stage fast on Apple's GPUs,
    /// which run a shader at the pace of its dearest code for every pixel
    /// (PERFORMANCE.md: the stage pass went from 4.0 / 5.7 / 10.9 ms a
    /// frame to 0.28 / 0.34 / 1.85 ms on an M-series Mac by this).
    #[test]
    fn each_pipeline_carries_only_its_own_effect_and_picks_by_its_group() {
        let why = "each pipeline must carry only its own effect's code and pick by \
                   `GROUP ==` alone (PERFORMANCE.md, the stage's pipelines)";
        for g in Group::all() {
            let src = shader(g);
            for fx in Fx::ALL {
                let s = fx.spec();
                let own = g.effect() == Some(fx);
                if let Some(code) = s.wgsl {
                    let carries = src.contains(&code());
                    assert_eq!(carries, own, "{g:?} and {fx:?}'s code: {why}");
                }
                let number = format!("FX_{}", wgsl_name(s.ident));
                let used = src.match_indices(&number).filter(|&(at, _)| {
                    let line_start = src[..at].rfind('\n').map_or(0, |n| n + 1);
                    !src[line_start..].starts_with("const ")
                });
                for (at, _) in used {
                    let read = src[..at].ends_with("looks[");
                    assert!(read, "{g:?}: {number} only reads its look: {why}");
                }
            }
            let helpers = src.contains("fn fx_heat(");
            assert_eq!(helpers, g.effect().is_some(), "{g:?}'s helpers: {why}");
            for line in src.lines().map(str::trim) {
                if line.contains(" G_") && !line.starts_with("const ") {
                    let by_group =
                        line.starts_with("if GROUP == G_") || line.starts_with("if GROUP >= G_");
                    assert!(by_group, "{g:?}: `{line}`: {why}");
                }
            }
        }
    }

    /// The screen passes draw only strips (round the screen's edges, round
    /// the lane: each no more than half the screen one way, none over
    /// another), never the whole screen: a pass over every pixel is what a
    /// frame cannot afford on a 3024 × 1964 screen (PERFORMANCE.md).
    #[test]
    fn the_screen_passes_draw_only_strips() {
        let area = (0.0, 0.0, 3024.0, 1964.0);
        let lane = crate::gui::stage::Lane::new(area, 10.0);
        let at = crate::gui::stage::screen_at(area, lane.glass(area));
        for fx in Fx::ALL.into_iter().filter(|f| f.spec().is_screen()) {
            let uses: &'static [fx::Use] = Box::leak(Box::new([fx::Use::of(fx, 1.0)]));
            let row = fx::Effects {
                held: uses,
                ..fx::Effects::NONE
            };
            let mut board = fx::Board::default();
            let step = crate::gui::streak::Step {
                glow: crate::gui::rainbow::Glow {
                    level: 1.0,
                    from: 1.0,
                    since: -10.0,
                },
                ..crate::gui::streak::Step::default()
            };
            let frame = board.frame_of([(1.0, &row)].into_iter(), &step, &fx::Music::NEUTRAL, 10.0);
            let mut shapes = Vec::new();
            board.shapes(&at, &frame, &mut shapes);
            let why = "a screen pass draws only strips (PERFORMANCE.md, the stage's pipelines)";
            assert!(!shapes.is_empty(), "{fx:?} draws");
            for (n, a) in shapes.iter().enumerate() {
                let (w, h) = (a.quad[2] - a.quad[0], a.quad[3] - a.quad[1]);
                let strip = w <= area.2 / 2.0 || h <= area.3 / 2.0;
                assert!(strip, "{fx:?}: {:?} {why}", a.quad);
                for b in &shapes[n + 1..] {
                    let apart = a.quad[2] <= b.quad[0]
                        || b.quad[2] <= a.quad[0]
                        || a.quad[3] <= b.quad[1]
                        || b.quad[3] <= a.quad[1];
                    assert!(apart, "{fx:?}: {:?} over {:?}: {why}", a.quad, b.quad);
                }
            }
        }
    }

    /// A frame's shapes in runs by group, text apart by its treatment: the
    /// plain text, the rainbow's, plain again, the rainbow's bands.
    #[test]
    fn runs_split_by_group_and_treatment() {
        let shape = |k: f32, treat: f32| Instance {
            params: [k, 0.0, 0.0, treat],
            ..Instance::default()
        };
        let rainbow = Fx::Rainbow.treatment_code();
        let frame = [
            shape(kind::PILL, 0.0),
            shape(kind::TEXT, 0.0),
            shape(kind::TEXT, rainbow),
            shape(kind::TEXT, rainbow),
            shape(kind::DISC, 0.0),
            shape(BANDS, 0.0),
            shape(SWIRL, 0.0),
        ];
        let got: Vec<(Group, std::ops::Range<u32>)> = runs(&frame, 10);
        assert_eq!(
            got,
            vec![
                (Group::Plain, 10..12),
                (Group::Lyric(Fx::Rainbow), 12..14),
                (Group::Plain, 14..15),
                (Group::Shapes(Fx::Rainbow), 15..17),
            ]
        );
    }
}
