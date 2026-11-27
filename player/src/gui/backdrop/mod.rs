//! Karaoke mode's backdrop on the GPU, by the game's setting ([`Shows`],
//! [`crate::core::karaoke::Show::story`]): without it (or where no stories
//! are made) the song's cover blurred heavily, dark and still, filling the
//! window ([`ground`]); with it the song's story, its scenes
//! ([`crate::work::scenes`]) each filling the window in turn, cut in on the
//! song's bars through the whole song ([`scenes`]), moving with the music:
//! the camera pushing slowly at the song's tempo and kicked by the beat, a
//! cut landing with a punch, the layers' edges lit on a bar's first beat
//! harder as the singer climbs the tiers, a phrase landing flaring it, a
//! section begun and the singer's glow lifting its light ([`music`],
//! [`pulse`]). The song's look draws over it ([`looks`]). The text needs
//! nothing of it: the lane has its glass, the bubbles their dark bodies
//! and the lyrics their dark outline. The Shader tab shows it as the mode
//! does. The game switched mid-song cuts from one to the other.
//!
//! [`Visualiser`] is the processor's side, frame by frame: the ground (the
//! blurred cover without the game; with it the plain dark gradient the
//! scenes cover once the song's story is there), the story as shown (cut
//! on the song's analysis), the music felt, the scene's numbers for the
//! shader and the look's ([`BackdropFrame`]). [`gpu`] draws it at its share
//! of the window's size. The stage's widget
//! ([`super::stage_view`]) draws it as Karaoke mode's ground, under the
//! stage.

pub mod gpu;
pub mod ground;
pub mod look;
pub mod music;
pub mod pulse;
pub mod scenes;
pub mod song_look;

use std::sync::Arc;

use image::RgbImage;
use jodel_playback::PlayState;

use self::music::{Feel, Hearing, Song};
use crate::app::App;
use crate::work::art::Art;
use crate::work::picture::Swatch;

/// What the backdrop shows: the song's cover blurred alone, filling the
/// window (none: the gradient); or the song's story's scenes (none yet:
/// the plain dark gradient).
#[derive(Clone, Debug)]
pub enum Shows {
    /// The blurred cover.
    Blur(Option<Art>),
    /// The story, as shown.
    Story(Option<Arc<scenes::Showing>>),
}

impl Default for Shows {
    fn default() -> Shows {
        Shows::Blur(None)
    }
}

/// The ground: the blurred cover (the gradient without one) as small as it
/// is made, for a window, and which it is of.
#[derive(Clone, Debug, PartialEq)]
pub struct Ground {
    /// The cover's id (0: none) and the window it is cut for.
    pub of: (u64, (u32, u32)),
    /// The pixels (the GPU scales them up).
    pub image: Arc<RgbImage>,
}

/// What the backdrop is given for a frame.
#[derive(Clone, Debug, Default)]
pub struct Cue {
    /// What it shows.
    pub shows: Shows,
    /// What is heard: the singer's glow (0 none, up to the top tier's 1:
    /// [`crate::tiers::TierSpec::glow`]), a section begun or a phrase
    /// landed since the last frame, the music's clock.
    pub heard: Hearing,
    /// The theme's grade ([`crate::theme::Theme::grade`]; all 0, the
    /// default: none), the scenes graded by it this frame.
    pub grade: [f32; 4],
    /// The song's look and its music this frame (none: no look).
    pub look: Option<song_look::LookCue>,
    /// Where the song is heard, in seconds: which scene shows.
    pub at: f64,
    /// Every driver this frame, playing or not ([`song_look::Looks::hear`]).
    pub signals: jodel_looks::Signals,
}

/// A frame of the backdrop: the ground, the story's scene showing instead
/// (none: the ground), and the song's look over it (none: none).
#[derive(Clone, Debug)]
pub struct BackdropFrame {
    /// The ground.
    pub ground: Ground,
    /// The story's scene showing.
    pub scene: Option<scenes::SceneFrame>,
    /// The look.
    pub look: Option<song_look::LookFrame>,
    /// Every driver this frame (the song's curves read then, the singing's
    /// set live over them: [`song_look::Looks::hear`]), what the look and
    /// the effects move by.
    pub signals: jodel_looks::Signals,
}

impl BackdropFrame {
    /// What it draws in a `window` of pixels, for the share of the window
    /// it is drawn at: the scene's picture filling the window (its size on
    /// the window, pixels) and its texels; none for the ground.
    pub fn picture(&self, window: (u32, u32)) -> Option<((f32, f32), (u32, u32))> {
        let picture = self.scene.as_ref()?.picture();
        Some(((window.0 as f32, window.1 as f32), picture))
    }

    /// The palette of the scene showing (none without one), the largest
    /// share first.
    pub fn palette(&self) -> &[Swatch] {
        self.scene.as_ref().map_or(&[], scenes::SceneFrame::palette)
    }
}

/// The backdrop frame by frame: the ground, the music felt, the song's
/// look, the story as shown.
#[derive(Default)]
pub struct Visualiser {
    ground: Option<Ground>,
    feel: Feel,
    /// Where the song was heard last.
    heard: Option<f64>,
    /// When the last phrase seen ended (none: none yet).
    phrase: Option<f64>,
    /// The look as it is drawn: what it was made safe from, its curves.
    looks: song_look::Looks,
    /// The song's story as shown (while it shows), cut on its analysis.
    showing: Option<Arc<scenes::Showing>>,
}

impl Visualiser {
    /// Whether its frames change by themselves (music or not): what the
    /// music lit still settling under a story's scenes, or the look's
    /// strength being found or eased to. While paused, the frames go on
    /// until it is still.
    pub fn moving(&self) -> bool {
        self.looks.settling() || self.showing.is_some() && !self.feel.settled()
    }

    /// The song's picture `art` made small, as the look's strength is
    /// found over it ([`song_look::Looks::picture`]).
    pub fn picture(&mut self, art: Option<&Art>) -> Arc<jodel_looks::render::Image> {
        self.looks.picture(art)
    }

    /// The frame for `app` at `heard` seconds into the `song` heard (none:
    /// not analysed yet), `dt` seconds after the last, in a `window` of
    /// pixels. None (and nothing held but the look as it was made:
    /// [`song_look::Looks::left`]) outside Karaoke mode and the Shader tab, or
    /// without the GPU (`gpu`).
    pub fn frame(
        &mut self,
        (app, song): (&App, Option<&Song>),
        gpu: bool,
        heard: f64,
        window: (u32, u32),
        dt: f32,
    ) -> Option<BackdropFrame> {
        let state = &app.state;
        if !gpu || !crate::core::karaoke::stage_shown(state) {
            // All let go but the look as it was made (coming back finds
            // its strength again only where it was not found yet).
            let mut looks = std::mem::take(&mut self.looks);
            looks.left();
            *self = Visualiser {
                looks,
                ..Visualiser::default()
            };
            return None;
        }
        let playing = state.playing.state == PlayState::Playing;
        // The look reads the song where it is, playing or not; the effects
        // move by the same drivers.
        let song_heard = self.looks.hear(state, song, heard);
        let signals = song_heard.signals;
        let look = self
            .looks
            .cue(state, (app.art(), &song_heard), (heard, dt), window);
        // The story where it shows, cut on its bars again where it or the
        // analysis is new; else the cover.
        let shows = if state.show().story {
            self.showing = app.story().map(|story| match self.showing.take() {
                Some(s) if s.is_of(&story, song) => s,
                _ => Arc::new(scenes::Showing::new(story, song)),
            });
            Shows::Story(self.showing.clone())
        } else {
            self.showing = None;
            Shows::Blur(app.art().cloned())
        };
        let song = song.filter(|_| playing);
        let section = song.zip(self.heard).is_some_and(|(s, was)| {
            heard > was && heard - was < 0.5 && s.section_between(was, heard)
        });
        let singing = &state.karaoke.singing;
        let game = state.show().scored;
        // A phrase landed on a tier: the last one ended elsewhere than the
        // last seen (none on the first frame: it was there before).
        let last = singing.last_phrase().filter(|_| game);
        let end = last.map(|p| p.end);
        let phrase =
            self.heard.is_some() && end != self.phrase && last.is_some_and(|p| p.tier.is_some());
        self.phrase = end;
        self.heard = Some(heard);
        // The beat as the effects have it, from the same grid; none while
        // the song is not playing.
        let music = song.map_or(crate::gui::fx::Music::NEUTRAL, |song| {
            crate::gui::fx::music::read(&crate::gui::fx::music::Heard {
                song: Some(song),
                singing,
                now: heard,
                clock: state.karaoke.lag.sung_clock(heard),
                signals,
            })
        });
        let cue = Cue {
            shows,
            heard: Hearing {
                glow: singing.glow(game),
                section,
                phrase,
                music,
            },
            grade: state.theme().grade.uniform(),
            look,
            at: heard,
            signals,
        };
        Some(self.step(&cue, window, dt))
    }

    /// `dt` seconds on with `cue`, in a `window` of pixels: the frame. The
    /// ground is the cover blurred where that shows, else the gradient,
    /// under the story's scene where one shows.
    pub fn step(&mut self, cue: &Cue, window: (u32, u32), dt: f32) -> BackdropFrame {
        let dt = dt.clamp(0.0, 0.1);
        let (art, story) = match &cue.shows {
            Shows::Blur(art) => (art.as_ref(), None),
            Shows::Story(story) => (None, story.as_ref()),
        };
        let of = (art.map_or(0, |a| a.id), window);
        let ground = match self.ground.take() {
            Some(g) if g.of == of => g,
            _ => Ground {
                of,
                image: Arc::new(ground::small(art, window)),
            },
        };
        self.ground = Some(ground.clone());
        let felt = self.feel.step(&cue.heard, dt);
        let scene = story.and_then(|s| s.frame(cue.at, &felt, window, cue.grade));
        let palette =
            (scene.as_ref()).map_or(jodel_looks::PALETTE, |s| song_look::palette_of(s.palette()));
        let look = (cue.look.as_ref()).map(|l| self.looks.frame(l, window, (dt, palette)));
        BackdropFrame {
            ground,
            scene,
            look,
            signals: cue.signals,
        }
    }
}

#[cfg(test)]
mod tests;
