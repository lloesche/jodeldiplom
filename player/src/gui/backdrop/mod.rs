//! Karaoke mode's backdrop on the GPU: the song's cover taken apart
//! ([`crate::cover::Apart`]: the plate, the subject cut out, their depth,
//! the rim, the palette) drawn live as a calm visualiser of the music. The
//! whole cover sits in the middle as a card, as large as it looks good
//! ([`look::card`]: never a cover's pixel more than [`look::CAP`] window
//! pixels, read smoothly where it is much enlarged, so no pixel nor block
//! of its coding shows), its edge fading into the same cover blurred
//! heavily, dark, filling the window under it and drifting slowly. The
//! text needs nothing of it: the lane has its glass, the bubbles and the
//! lyrics their own contrast.
//!
//! The life is on the subject. Each stem lights its own piece ([`music`],
//! from the song's analysis, smoothed, as the desk lets it be heard): the
//! drums' hits flash the rim round the cut-out; the vocals light the
//! subject and a glow round it; the bass pops the subject out of the plate
//! (larger, its shadow on the plate deeper); the rest sweeps the palette's
//! light across the plate and lights the blur round the card. A cover
//! without a clear subject has its nearest parts lit so. The big moments
//! are kept for a section starting (the whole lifts, rays from behind the
//! subject) and for the singer's shine: Great glows round the subject,
//! Amazing more with faint rays, Diva brightest, the rays golden.
//!
//! [`Visualiser`] is the processor's side, frame by frame: the ground (the
//! blurred cover, [`super::karaoke::ground`]: what shows while the layers
//! are still being prepared), which covers show over it and how much (one
//! fading in, a song's cover cross-fading into the next's, fading out to
//! the ground), the music felt, and each cover's numbers for the shader
//! ([`BackdropFrame`]). [`gpu`] draws it at the window's size, the
//! textures sent over once a cover and dropped once it no longer shows.
//! The window's pass ([`super::present`]) adds it under the pixmap as far
//! as the pixmap's words let it through ([`super::raster::SEE_THROUGH`]).
//! Without a GPU the pixmap has the blurred cover itself.

pub mod gpu;
pub mod look;
pub mod music;
mod subject;

use std::sync::Arc;

use image::RgbImage;
use jodel_playback::PlayState;

use self::music::{Feel, Felt, Levels, Song};
pub use self::subject::{lights, Subject};
use crate::app::App;
use crate::art::Art;
use crate::core::desk::StripKind;
use crate::cover::Layers;

/// How long a cover takes to fade in or out, in seconds.
pub const FADE: f32 = 1.2;
/// How much larger than the window the blur under the card is drawn:
/// room for its drift.
const FILL_ZOOM: f32 = 1.15;
/// How far the blur drifts, a share of the cover.
const FILL_DRIFT: f32 = 0.04;
/// How far the card's layers drift against each other (parallax), a share
/// of the card.
const DRIFT: f32 = 0.004;
/// The drift's periods across and down, in seconds.
const DRIFT_PERIODS: (f32, f32) = (37.0, 29.0);
/// How much larger the subject grows with the bass, a share.
const BREATH: f32 = 0.022;

/// A cover's numbers as its shader reads them (`backdrop.wgsl`'s `Look`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Look {
    /// The window's width and height in pixels, the drift's time, unused.
    pub size: [f32; 4],
    /// The card: its middle, its half width and height (the window's
    /// shares, [`look::card`]).
    pub card: [f32; 4],
    /// How the card is read ([`look::softness`]): the level a card smaller
    /// than its cover reads, how smoothly and how much softer still; and 1
    /// with a subject cut out (0: its nearest parts stand for it).
    pub read: [f32; 4],
    /// The levels the cover is read blurred at ([`look::blurs`]): the blur
    /// under the card, the glow round the subject, its shadow and the rim's
    /// glow; unused.
    pub levels: [f32; 4],
    /// The layers' drift against each other (the cover's units), the
    /// subject's growing with the bass, the cover's share over what is
    /// under it.
    pub cam: [f32; 4],
    /// The rim's pulse (the drums), the vocals, the rest, the bass.
    pub music: [f32; 4],
    /// A section begun, the singer's glow, the rays, the lift.
    pub moments: [f32; 4],
    /// The window's shares to the blur's cover units: scale, then offset
    /// (drifting; [`look::fill`]).
    pub fill: [f32; 4],
    /// The light behind the subject (the window's shares), its reach (a
    /// share of the height), unused.
    pub light: [f32; 4],
    /// What the subject grows round (the cover's units), unused.
    pub focus: [f32; 4],
    /// The palette's vivid light.
    pub accent: [f32; 4],
    /// Its next.
    pub second: [f32; 4],
    /// Its largest.
    pub ground: [f32; 4],
}

// SAFETY: plain f32s in a `repr(C)` struct, no padding: any bytes are one.
unsafe impl bytemuck::Zeroable for Look {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Look {}

/// The ground under the covers: the blurred cover (the gradient without
/// one) as small as it is made, for a window, and which it is of.
#[derive(Clone, Debug, PartialEq)]
pub struct Ground {
    /// The cover's id (0: none) and the window it is cut for.
    pub of: (u64, (u32, u32)),
    /// The pixels (the GPU scales them up).
    pub image: Arc<RgbImage>,
}

/// A cover showing: its layers, the picture they were taken apart from
/// (the song's cover, once it is seen to be it: [`subject::is_of`]), where
/// its subject is, its palette's lights, how much it shows (0 to 1, eased
/// at the ends) and whether it is wanted (fading in) or not (fading out).
struct Slot {
    layers: Layers,
    art: Option<Art>,
    /// The cover last looked at for it (its id; 0 none).
    tried: u64,
    subject: Subject,
    lights: [[f32; 3]; 3],
    shown: f32,
    wanted: bool,
}

/// What the backdrop is given for a frame.
#[derive(Clone, Debug, Default)]
pub struct Cue {
    /// The song's cover (none: the gradient).
    pub art: Option<Art>,
    /// The cover taken apart, once ready.
    pub layers: Option<Layers>,
    /// The stems' levels heard now.
    pub levels: Levels,
    /// The singer's shine (0 none, 1 Good to 4 Diva).
    pub tier: u32,
    /// A section began since the last frame.
    pub section: bool,
}

/// A cover of a frame, for the shader: its layers, the picture they were
/// taken apart from (none: not at hand) and its numbers.
#[derive(Clone, Debug)]
pub struct SlotFrame {
    /// The cover taken apart.
    pub layers: Layers,
    /// The cover itself.
    pub art: Option<Art>,
    /// Its numbers.
    pub look: Look,
}

/// A frame of the backdrop: the ground, and the covers showing over it,
/// back to front, each over what is under it by its share
/// ([`Look::cam`]).
#[derive(Clone, Debug)]
pub struct BackdropFrame {
    /// The ground.
    pub ground: Ground,
    /// The covers.
    pub slots: Vec<SlotFrame>,
}

/// The backdrop frame by frame: the ground, the covers showing, the music
/// felt, the drift's clock.
#[derive(Default)]
pub struct Visualiser {
    ground: Option<Ground>,
    slots: Vec<Slot>,
    feel: Feel,
    /// The seconds the drift has run.
    clock: f32,
    /// The song heard (its stems' lanes), and where it was heard last.
    song: Option<Song>,
    heard: Option<f64>,
}

impl Visualiser {
    /// How many covers it holds (showing or fading).
    pub fn holds(&self) -> usize {
        self.slots.len()
    }

    /// Whether its frames change by themselves (music or not): a cover
    /// fading, or what the music lit still settling. While paused, the
    /// frames go on until it is still.
    pub fn moving(&self) -> bool {
        let fading = |s: &Slot| s.shown < 1.0 || !s.wanted;
        !self.slots.is_empty() && (self.slots.iter().any(fading) || !self.feel.settled())
    }

    /// The frame for `app` at `heard` seconds into the song, `dt` seconds
    /// after the last, in a `window` of pixels. None (and nothing held)
    /// outside Karaoke mode or without the GPU (`gpu`): the pixmap has its
    /// own ground then.
    pub fn frame(
        &mut self,
        app: &App,
        gpu: bool,
        heard: f64,
        window: (u32, u32),
        dt: f32,
    ) -> Option<BackdropFrame> {
        let state = &app.state;
        if !gpu || !crate::core::karaoke::in_mode(state) {
            *self = Visualiser::default();
            return None;
        }
        self.follow_song(app);
        let playing = state.playing.state == PlayState::Playing;
        let gains = StripKind::PARTS.map(|k| state.desk.heard_gain(k).min(1.0) as f32);
        let song = self.song.as_ref().filter(|_| playing);
        let levels = song.map_or(Levels::default(), |s| s.levels(heard).times(gains));
        let section = song.zip(self.heard).is_some_and(|(s, was)| {
            heard > was && heard - was < 0.5 && s.section_between(was, heard)
        });
        self.heard = Some(heard);
        let cue = Cue {
            art: app.art().cloned(),
            layers: app.cover_layers(),
            levels,
            tier: super::streak::level(&state.karaoke.singing, state.config.karaoke_game),
            section,
        };
        Some(self.step(&cue, window, dt))
    }

    /// The song heard's stems' lanes, read again when its analysis is new.
    fn follow_song(&mut self, app: &App) {
        let state = &app.state;
        let rhythm = crate::core::karaoke::song_index(state)
            .and_then(|i| state.queue.paths.get(i))
            .and_then(|p| state.rhythm.get(p));
        let ready = match rhythm {
            Some(crate::core::Rhythm::Ready(r)) => r.waveform.as_ref().map(|w| (w, &r.analysis)),
            _ => None,
        };
        match ready {
            Some((w, a)) if !self.song.as_ref().is_some_and(|s| s.is_of(w)) => {
                let bars: Vec<f64> = a
                    .downbeats
                    .iter()
                    .filter_map(|&i| a.beats.get(i).copied())
                    .collect();
                self.song = Some(Song::of(Arc::clone(w), &bars));
            }
            Some(_) => {}
            None => self.song = None,
        }
    }

    /// `dt` seconds on with `cue`, in a `window` of pixels: the frame.
    pub fn step(&mut self, cue: &Cue, window: (u32, u32), dt: f32) -> BackdropFrame {
        let dt = dt.clamp(0.0, 0.1);
        let of = (cue.art.as_ref().map_or(0, |a| a.id), window);
        let ground = match self.ground.take() {
            Some(g) if g.of == of => g,
            _ => Ground {
                of,
                image: Arc::new(super::karaoke::ground::small(cue.art.as_ref(), window)),
            },
        };
        self.want(cue.layers.as_ref(), cue.art.as_ref());
        for s in &mut self.slots {
            let to = if s.wanted { 1.0 } else { 0.0 };
            let by = dt / FADE;
            s.shown = if s.shown < to {
                (s.shown + by).min(to)
            } else {
                (s.shown - by).max(to)
            };
        }
        self.slots.retain(|s| s.wanted || s.shown > 0.0);
        let felt = self.feel.step(cue.levels, cue.tier, cue.section, dt);
        self.clock += dt;
        // Each cover's share of the frame (eased); the ground has the rest.
        let shares: Vec<f32> = self
            .slots
            .iter()
            .map(|s| look::smoothstep(0.0, 1.0, s.shown))
            .collect();
        let total: f32 = shares.iter().sum();
        let scale = if total > 1.0 { 1.0 / total } else { 1.0 };
        // Drawn back to front, each over what is under it: its share of
        // all drawn so far (the ground's and its own included).
        let mut so_far = 1.0 - total.min(1.0);
        let slots = self
            .slots
            .iter()
            .zip(&shares)
            .map(|(s, &share)| {
                let share = share * scale;
                so_far += share;
                let alpha = if so_far > 1e-6 { share / so_far } else { 0.0 };
                SlotFrame {
                    layers: Arc::clone(&s.layers),
                    art: s.art.clone(),
                    look: self.look(s, &felt, window, alpha),
                }
            })
            .collect();
        self.ground = Some(ground.clone());
        BackdropFrame { ground, slots }
    }

    /// `layers` wanted now (none: nothing): in front, fading in (back from
    /// fading out where it still shows); the rest fading out. The same
    /// cover is the same however it was read ([`crate::cover::Apart::key`]).
    /// Two at most: a third coming drops the one of the others showing
    /// least. The song's cover `art` is kept with them once it is seen to be
    /// what they were taken apart from (it may come before or after them).
    fn want(&mut self, layers: Option<&Layers>, art: Option<&Art>) {
        let Some(layers) = layers else {
            self.slots.iter_mut().for_each(|s| s.wanted = false);
            return;
        };
        let held = self.slots.iter().position(|s| s.layers.key == layers.key);
        let slot = match held {
            Some(i) => self.slots.remove(i),
            None => Slot {
                layers: Arc::clone(layers),
                art: None,
                tried: 0,
                subject: Subject::of(&layers.layers),
                lights: lights(&layers.layers.palette),
                shown: 0.0,
                wanted: true,
            },
        };
        for s in &mut self.slots {
            s.wanted = false;
        }
        if self.slots.len() >= 2 {
            let least = (0..self.slots.len())
                .min_by(|&a, &b| self.slots[a].shown.total_cmp(&self.slots[b].shown))
                .unwrap_or(0);
            self.slots.remove(least);
        }
        let mut slot = Slot {
            wanted: true,
            ..slot
        };
        // Each cover seen once: is it theirs?
        if let Some(a) = art.filter(|a| slot.art.is_none() && a.id != slot.tried) {
            slot.tried = a.id;
            if subject::is_of(&a.image, &layers.layers) {
                slot.art = Some(a.clone());
            }
        }
        self.slots.push(slot);
    }

    /// `slot`'s numbers for the shader, with the music `felt`, in a
    /// `window` of pixels, drawn over what is under it by `alpha`.
    fn look(&self, slot: &Slot, felt: &Felt, window: (u32, u32), alpha: f32) -> Look {
        let (w, h) = (window.0.max(1) as f32, window.1.max(1) as f32);
        let (cw, ch) = slot.layers.layers.size();
        let cover = (cw as f32, ch as f32);
        let sub = &slot.subject;
        let card = look::card((w, h), cover);
        let ([smooth, softer], level) = look::softness(2.0 * card[3] * h / cover.1.max(1.0));
        let blurs = look::blurs((cw, ch), gpu::levels((cw, ch)));
        let tau = std::f32::consts::TAU;
        let t = self.clock;
        let wave = (
            (t * tau / DRIFT_PERIODS.0).sin(),
            (t * tau / DRIFT_PERIODS.1 + 1.3).sin(),
        );
        let cam = [DRIFT * wave.0, 0.6 * DRIFT * wave.1];
        let mut fill = look::fill((w, h), cover, FILL_ZOOM);
        fill[2] += FILL_DRIFT * fill[0] * wave.1;
        fill[3] += 0.6 * FILL_DRIFT * fill[1] * wave.0;
        // The light behind the subject, in the window's shares.
        let light = (
            card[0] + (2.0 * sub.light.0 - 1.0) * card[2],
            card[1] + (2.0 * sub.light.1 - 1.0) * card[3],
        );
        let lift = 0.35 * felt.section + 0.12 * felt.glow;
        let [accent, second, ground] = slot.lights.map(|c| [c[0], c[1], c[2], 0.0]);
        let cut = if slot.layers.layers.subject.is_some() {
            1.0
        } else {
            0.0
        };
        Look {
            size: [w, h, t, 0.0],
            card,
            read: [level, smooth, softer, cut],
            levels: [blurs[0], blurs[1], blurs[2], 0.0],
            cam: [cam[0], cam[1], BREATH * felt.bass, alpha],
            music: [felt.pulse, felt.vocals, felt.other, felt.bass],
            moments: [felt.section, felt.glow, felt.rays, lift],
            fill,
            light: [light.0, light.1, sub.reach * 2.0 * card[3], 0.0],
            focus: [sub.focus.0, sub.focus.1, 0.0, 0.0],
            accent,
            second,
            ground,
        }
    }
}

#[cfg(test)]
mod tests;
