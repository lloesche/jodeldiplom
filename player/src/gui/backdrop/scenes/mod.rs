//! Karaoke's backdrop: a song's story scenes ([`crate::work::scenes`]),
//! the scene of the moment filling the window, cut in on the song's bars
//! one after the other through the whole song ([`cuts`]: the first from
//! the song's start, each on till the next, the last on to the end).
//!
//! Each scene moves by a slow camera push (in, the next out, in turn, as
//! fast as the song's tempo has it), drifting a third of the way towards
//! its subject, and either a gentle warp by its depth (one slow slide
//! across the scene, its heads held flat so each moves as one piece) or,
//! taken apart, its layers in parallax: the farthest (the plate, filled
//! behind the others) one way, the nearest the other, each layer's edge
//! feathered, taking a little of the light behind it and lit on a bar's
//! first beat, the nearest's shadow on what is behind it. The beat kicks
//! the camera a little (more on a bar's first), a cut lands with a punch;
//! the far is hazed into the scene's own palette's largest colour,
//! darkened, its edges lit in its most vivid ([`lights`]); a section
//! begun and the singer's glow lift it ([`Felt::lift`]), and the theme
//! grades it. The song's look draws over it.
//!
//! [`Showing`] is the processor's side: when each scene shows, and each
//! frame's numbers ([`SceneLook`], the camera's maths here, the shader's in
//! [`look`]); [`gpu`] draws them.

pub mod cuts;
pub mod gpu;
pub mod look;

use std::sync::Arc;

use jodel_rhythm::Analysis;

use self::cuts::{cuts, showing, Shown};
use super::music::{Felt, Song};
use super::pulse::Pulse;
use crate::work::picture::Swatch;
use crate::work::scenes::Story;
use jodel_looks::math::srgb_to_linear;

/// How much larger than the window the picture is at the least: room for
/// the warp's and the layers' slide, so no edge ever shows.
pub const MARGIN: f32 = 1.06;
/// How much the camera pushes in (or pulls out) over a scene, at the
/// song's own speed ([`speed`]).
pub const PUSH: f32 = 0.07;
/// How far the camera drifts towards the subject over a scene.
pub const TOWARDS: f32 = 0.35;
/// How much of the beat's push ([`Pulse::grow`]) the camera takes.
pub const KICK: f32 = 0.6;
/// A cut's punch: zoomed in by this, falling to a third over
/// [`PUNCH_FALL`] seconds.
pub const PUNCH: f32 = 0.02;
/// See [`PUNCH`].
pub const PUNCH_FALL: f32 = 0.25;
/// The most the warp moves the nearest (and the farthest the other way),
/// a share of the picture, at either end of a scene.
pub const WARP: f32 = 0.0125;
/// How far the layers slide at either end of a scene (the nearest by all
/// of it), a share of the picture.
pub const SLIDE: f32 = 0.014;
/// How much of a slide is up or down (the rest across): pictures are wide.
pub const UPRIGHT: f32 = 0.5;
/// The turn from one scene's slide to the next's (the golden angle, in
/// radians: every scene its own way), and the first scene's.
pub const TURN: (f32, f32) = (2.399_963, 0.7);
/// How far each layer slides, the farthest to the nearest: the far
/// against the near.
pub const PARALLAX: [f32; 3] = [-0.35, 0.25, 1.0];
/// How much the farthest is hazed, and how dark its haze is (a share of
/// the palette's largest colour).
pub const HAZE: (f32, f32) = (0.22, 0.3);
/// How bright a layer's edge lights on a bar's first beat
/// ([`Pulse::rim`]), and how much white is in its light (the rest the
/// palette's most vivid).
pub const RIM: (f32, f32) = (0.5, 0.4);
/// The picture read about this many texels across: the light behind a
/// layer's edge.
pub const BEHIND_TEXELS: f32 = 40.0;
/// How long before its cut the next scene is sent over to the GPU.
pub const PRELOAD: f64 = 3.0;
/// The tempo the camera moves at its own speed at, and the least and most
/// of that speed.
pub const TEMPO: (f32, f32, f32) = (110.0, 0.7, 1.3);
/// A tempo over this counts each beat twice: it is halved.
pub const DOUBLED: f32 = 140.0;

/// How fast the camera moves for a song at `bpm`: its tempo against
/// [`TEMPO`]'s (halved over [`DOUBLED`]), within its least and most.
pub fn speed(bpm: f32) -> f32 {
    let bpm = if bpm > DOUBLED { bpm / 2.0 } else { bpm };
    (bpm / TEMPO.0).clamp(TEMPO.1, TEMPO.2)
}

/// Smoothly from 0 to 1 as `u` goes so (a half cosine).
pub fn ease(u: f32) -> f32 {
    0.5 - 0.5 * (std::f32::consts::PI * u.clamp(0.0, 1.0)).cos()
}

/// A scene's numbers as its shader reads them (`scenes.wgsl`'s `Scene`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneLook {
    /// The level the picture is read blurred at (the light behind a
    /// layer's edge); `.yzw` unused (a uniform's fields go four floats at
    /// a time).
    pub read: [f32; 4],
    /// The window's extent in the picture at zoom 1 (its shares), the
    /// middle seen.
    pub view: [f32; 4],
    /// The zoom, unused, the slide (the picture's shares: the warp's or
    /// the layers' at its fullest).
    pub cam: [f32; 4],
    /// How far each layer slides ([`PARALLAX`]), the haze ([`HAZE`]).
    pub depth: [f32; 4],
    /// The light on a layer's edge: its colour, how much.
    pub light: [f32; 4],
    /// The haze's colour, the lift ([`Felt::lift`]).
    pub fog: [f32; 4],
    /// The theme's grade ([`crate::theme::Theme::grade`]).
    pub grade: [f32; 4],
    /// How many layers (1: the warp); `.yzw` unused.
    pub layers: [u32; 4],
}

/// The story's scenes in a frame: of which story, the one drawn and its
/// numbers, the next to have ready (none: none yet).
#[derive(Clone, Debug)]
pub struct SceneFrame {
    /// The story.
    pub story: Arc<Story>,
    /// The scene drawn and its numbers.
    pub drawn: (usize, SceneLook),
    /// The next scene, sent over ahead of its cut.
    pub next: Option<usize>,
}

impl SceneFrame {
    /// The picture's size of the scene drawn.
    pub fn picture(&self) -> (u32, u32) {
        self.story.scenes[self.drawn.0].size
    }

    /// The palette of the scene drawn, the largest share first.
    pub fn palette(&self) -> &[Swatch] {
        &self.story.scenes[self.drawn.0].palette
    }
}

/// Where the camera is in a scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// How much larger than the window the picture is.
    pub zoom: f32,
    /// The middle seen (the picture's shares).
    pub centre: [f32; 2],
    /// The slide (the picture's shares).
    pub slide: [f32; 2],
}

/// Scene `index`'s camera, eased `e` of the way through it (0 to 1),
/// `since` seconds after its cut, the beat's push `kick`, at `speed`, its
/// subject at `focus`, the window seeing `extent` of the picture at zoom 1,
/// its slide at its fullest `most` (the warp's or the layers'): pushing in
/// (an even scene) or pulling out, drifting towards the subject, never
/// past the picture's edge, sliding one way across the scene, each scene
/// its own way ([`TURN`]).
pub fn camera(
    (index, e, since): (usize, f32, f32),
    (kick, speed): (f32, f32),
    (focus, extent): ([f32; 2], [f32; 2]),
    most: f32,
) -> Camera {
    let push = PUSH * speed * if index % 2 == 0 { e } else { 1.0 - e };
    let punch = PUNCH * (-since.max(0.0) / PUNCH_FALL).exp();
    let zoom = MARGIN + push + KICK * kick + punch;
    let centre = [0, 1].map(|i| {
        let half = (extent[i] / (2.0 * zoom)).min(0.5);
        (0.5 + (focus[i] - 0.5) * TOWARDS * e).clamp(half, 1.0 - half)
    });
    let angle = index as f32 * TURN.0 + TURN.1;
    let k = (2.0 * e - 1.0) * speed * most;
    Camera {
        zoom,
        centre,
        slide: [angle.cos() * k, UPRIGHT * angle.sin() * k],
    }
}

/// A colour (sRGB bytes) as linear light, as bright as it can be: its
/// largest channel at 1 (a light's colour, not its strength).
fn light_of(rgb: [u8; 3]) -> [f32; 3] {
    let lin = rgb.map(|v| srgb_to_linear(f32::from(v) / 255.0));
    let most = lin[0].max(lin[1]).max(lin[2]).max(1e-3);
    lin.map(|c| c / most)
}

/// A scene's lights, of its `palette`: its most vivid colour (the layers'
/// edges) and its largest (the haze); a warm white for each it does not
/// have.
fn lights(palette: &[Swatch]) -> [[f32; 3]; 2] {
    let vivid = |s: &Swatch| {
        let c = s.rgb.map(f32::from);
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        let sat = if max > 0.0 { (max - min) / max } else { 0.0 };
        sat * (max / 255.0).min(0.7) * (0.25 + s.share).sqrt()
    };
    let warm = [1.0, 0.85, 0.65];
    let most = palette.iter().max_by(|a, b| vivid(a).total_cmp(&vivid(b)));
    let pick = |s: Option<&Swatch>| s.map_or(warm, |s| light_of(s.rgb));
    [pick(most), pick(palette.first())]
}

/// A song's story as the backdrop shows it: its scenes ready, when each
/// shows (cut on the song's bars), each one's lights ([`lights`]) and how
/// fast its camera moves; the analysis it was cut on.
#[derive(Debug)]
pub struct Showing {
    /// The story.
    pub story: Arc<Story>,
    /// When each scene shows.
    pub shown: Vec<Shown>,
    /// Each scene's lights.
    lights: Vec<[[f32; 3]; 2]>,
    /// The camera's speed ([`speed`]).
    pub speed: f32,
    /// The analysis it was cut on (none: not analysed yet).
    analysis: Option<Arc<Analysis>>,
}

impl Showing {
    /// `story` cut on the bars of the `song` heard (none: not analysed yet,
    /// its spans as they are).
    pub fn new(story: Arc<Story>, song: Option<&Song>) -> Showing {
        let spans: Vec<(f64, f64)> = story.scenes.iter().map(|s| s.span).collect();
        let (bars, bpm) = song.map_or((Vec::new(), TEMPO.0), |s| {
            let a = s.analysis();
            let bars: Vec<f64> = (a.downbeats.iter())
                .filter_map(|&i| a.beats.get(i).copied())
                .collect();
            let middle = spans
                .first()
                .zip(spans.last())
                .map_or(0.0, |(f, l)| (f.0 + l.1) / 2.0);
            (bars, a.bpm_at(middle).map_or(TEMPO.0, |b| b as f32))
        });
        Showing {
            shown: cuts(&spans, &bars),
            lights: story.scenes.iter().map(|s| lights(&s.palette)).collect(),
            speed: speed(bpm),
            analysis: song.map(Song::shared_analysis),
            story,
        }
    }

    /// Whether it is `story` cut on `song`'s analysis.
    pub fn is_of(&self, story: &Arc<Story>, song: Option<&Song>) -> bool {
        let analysis = song.map(Song::shared_analysis);
        Arc::ptr_eq(&self.story, story)
            && match (&self.analysis, &analysis) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (a, b) => a.is_none() && b.is_none(),
            }
    }

    /// The scene at `at` seconds into the song (none: the story has no
    /// scene), with the music `felt`, in a `window` of pixels, the theme's
    /// `grade`; the next sent over ahead within [`PRELOAD`] of its cut.
    pub fn frame(
        &self,
        at: f64,
        felt: &Felt,
        window: (u32, u32),
        grade: [f32; 4],
    ) -> Option<SceneFrame> {
        let index = showing(&self.shown, at)?;
        let next = (index + 1 < self.shown.len())
            .then_some(index + 1)
            .filter(|&n| self.shown[n].start - at <= PRELOAD);
        let shown = self.shown[index];
        let scene = &self.story.scenes[index];
        let u = ((at - shown.start) / (shown.end - shown.start).max(1e-3)) as f32;
        let (w, h) = (window.0.max(1) as f32, window.1.max(1) as f32);
        let picture = (scene.size.0 as f32, scene.size.1 as f32);
        let fill = look::fill((w, h), picture);
        let extent = [fill[0], fill[1]];
        let pulse = Pulse::of(felt.beat, felt.tier, felt.flare);
        let layers = scene.moved.count();
        let most = if layers > 1 { SLIDE } else { WARP };
        let cam = camera(
            (index, ease(u), (at - shown.start) as f32),
            (pulse.grow, self.speed),
            (scene.focus, extent),
            most,
        );
        let long = scene.size.0.max(scene.size.1).max(1) as f32;
        let levels = super::gpu::levels(scene.size) as f32;
        let behind = (long / BEHIND_TEXELS).log2().clamp(0.0, levels - 1.0);
        let [vivid, largest] = self.lights[index];
        let rim = vivid.map(|c| (1.0 - RIM.1) * c + RIM.1);
        let fog = largest.map(|c| HAZE.1 * c);
        let look = SceneLook {
            read: [behind, 0.0, 0.0, 0.0],
            view: [extent[0], extent[1], cam.centre[0], cam.centre[1]],
            cam: [cam.zoom, 0.0, cam.slide[0], cam.slide[1]],
            depth: [PARALLAX[0], PARALLAX[1], PARALLAX[2], HAZE.0],
            light: [rim[0], rim[1], rim[2], RIM.0 * pulse.rim],
            fog: [fog[0], fog[1], fog[2], felt.lift()],
            grade,
            layers: [layers, 0, 0, 0],
        };
        Some(SceneFrame {
            story: Arc::clone(&self.story),
            drawn: (index, look),
            next,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_camera_pushes_in_then_out_and_slides_across_a_scene() {
        let at = |index: usize, e: f32| {
            camera((index, e, 10.0), (0.0, 1.0), ([0.8, 0.5], [1.0, 0.9]), WARP)
        };
        let (start, end) = (at(0, 0.0), at(0, 1.0));
        assert!((start.zoom - MARGIN).abs() < 1e-4 && (end.zoom - MARGIN - PUSH).abs() < 1e-4);
        let (start, end) = (at(1, 0.0), at(1, 1.0));
        assert!(start.zoom > end.zoom);
        // Towards the subject, never past the edge.
        assert!(at(0, 1.0).centre[0] > at(0, 0.0).centre[0]);
        for e in [0.0, 0.5, 1.0] {
            let c = at(0, e);
            let half = 1.0 / (2.0 * c.zoom);
            assert!(c.centre[0] >= half - 1e-6 && c.centre[0] <= 1.0 - half + 1e-6);
        }
        // One slide across: from one side through the middle to the other.
        let (a, m, b) = (at(2, 0.0).slide, at(2, 0.5).slide, at(2, 1.0).slide);
        assert!(m[0].abs() < 1e-6 && (a[0] + b[0]).abs() < 1e-6 && a[0] != 0.0);
        assert!(a[0].hypot(a[1] / UPRIGHT) <= WARP + 1e-6);
        // Each scene its own way.
        assert!((at(0, 1.0).slide[0] - at(1, 1.0).slide[0]).abs() > 1e-3);
    }

    #[test]
    fn a_cut_lands_with_a_punch_and_the_beat_kicks() {
        let cam = |since: f32, kick: f32| {
            camera((0, 0.0, since), (kick, 1.0), ([0.5; 2], [1.0; 2]), SLIDE).zoom
        };
        assert!((cam(0.0, 0.0) - MARGIN - PUNCH).abs() < 1e-5);
        assert!(cam(2.0, 0.0) - MARGIN < 1e-4);
        assert!((cam(2.0, 0.01) - cam(2.0, 0.0) - KICK * 0.01).abs() < 1e-6);
    }

    /// A story of three scenes, without bars: a scene shows all through
    /// the song (the first from its start, each on over a gap till the
    /// next, the last on after its span); the next is sent over ahead of
    /// its cut; a scene in layers slides further than one warped; each
    /// scene is lit by its own palette.
    #[test]
    fn a_story_shows_a_scene_all_through_the_song_and_the_next_ahead() {
        use crate::work::scenes::{Cuts, Moved, Ready};
        let scene = |span, cuts: Option<Cuts>, red: u8| Ready {
            span,
            size: (16, 9),
            picture: vec![0; 16 * 9 * 4],
            shape_size: (8, 4),
            shape: vec![0; 8 * 4 * 4],
            moved: cuts.map_or(Moved::Warp, |cuts| Moved::Layers {
                cuts,
                plate: vec![0; 16 * 9 * 4],
            }),
            focus: [0.5, 0.5],
            palette: vec![Swatch {
                rgb: [red, 40, 40],
                share: 1.0,
            }],
        };
        let story = Arc::new(Story::new(
            &["a".into(), "b".into(), "c".into()],
            vec![
                scene((10.0, 20.0), None, 200),
                scene((20.0, 30.0), Some(Cuts::Three(0.3, 0.6)), 20),
                scene((40.0, 50.0), None, 200),
            ],
        ));
        let showing = Showing::new(Arc::clone(&story), None);
        assert!(showing.is_of(&story, None));
        let felt = Felt::default();
        let at = |t: f64| showing.frame(t, &felt, (1600, 900), [0.0; 4]);
        let drawn = |t: f64| at(t).unwrap().drawn.0;
        // The first from the song's start, the second on over the gap, the
        // last on after its span.
        assert_eq!(
            [0.0, 5.0, 19.9, 20.0, 35.0, 39.9, 40.0, 50.0, 300.0].map(drawn),
            [0, 0, 0, 1, 1, 1, 2, 2, 2]
        );
        let first = at(12.0).unwrap();
        let (index, look) = first.drawn;
        assert_eq!((index, first.next), (0, None));
        assert_eq!(first.picture(), (16, 9));
        assert_eq!(first.palette()[0].rgb, [200, 40, 40]);
        assert_eq!(at(17.5).unwrap().next, Some(1));
        assert_eq!(at(38.0).unwrap().next, Some(2));
        assert_eq!(at(45.0).unwrap().next, None);
        let (index, second) = at(25.0).unwrap().drawn;
        assert_eq!(index, 1);
        assert_eq!(second.layers[0], 3);
        // Each its own palette: the first's haze redder than the second's.
        assert!(look.fog[0] > second.fog[0]);
        // At either end of a scene the slide is at its fullest: the
        // layers' further than the warp's.
        let slide = |t: f64| {
            let c = at(t).unwrap().drawn.1.cam;
            c[2].hypot(c[3] / UPRIGHT)
        };
        assert!(slide(39.99) > slide(19.99));
        // No scene: nothing drawn.
        let none = Showing::new(Arc::new(Story::new(&[], Vec::new())), None);
        assert!(none.frame(5.0, &felt, (1600, 900), [0.0; 4]).is_none());
    }

    #[test]
    fn a_scene_is_lit_by_its_palette_s_vivid_colour_and_hazed_in_its_largest() {
        let swatch = |rgb, share| Swatch { rgb, share };
        let palette = [
            swatch([60, 60, 64], 0.6),
            swatch([220, 40, 120], 0.2),
            swatch([30, 120, 200], 0.15),
        ];
        let [vivid, largest] = lights(&palette);
        assert!(vivid[0] == 1.0 && vivid[1] < 0.1, "{vivid:?}");
        assert!(largest.iter().all(|&c| c > 0.8), "{largest:?}");
        assert_eq!(lights(&[]), [[1.0, 0.85, 0.65]; 2]);
    }

    #[test]
    fn the_camera_moves_at_the_song_s_speed() {
        assert_eq!(speed(110.0), 1.0);
        assert_eq!(speed(220.0), 1.0);
        assert_eq!(speed(40.0), TEMPO.1);
        assert!((speed(139.0) - 139.0 / 110.0).abs() < 1e-6);
        assert_eq!(speed(300.0), TEMPO.2);
        assert_eq!(ease(0.0), 0.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-6 && (ease(1.0) - 1.0).abs() < 1e-6);
    }
}
