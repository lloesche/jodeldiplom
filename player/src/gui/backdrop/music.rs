//! What the backdrop and the effects hear: the song as heard ([`Song`]:
//! its analysis, and how loud the mix and each stem is at a moment, read
//! from its waveform's lanes, 10 ms a frame, each against how loud it gets
//! in this song), the sections' starts, where the music lifts
//! ([`sections`]); and for the backdrop the singer's tier, a section begun
//! and a phrase landed, smoothed so nothing flickers, and the beat
//! ([`Feel`]). Read on the main thread, frame by frame, from what the
//! analysis already holds: nothing of it touches the audio thread.

use std::sync::Arc;

use jodel_looks::drive::Range;
use jodel_rhythm::stems::Stem;
use jodel_rhythm::waveform::{Lane, Waveform, FPS};
use jodel_rhythm::Analysis;

use super::pulse::Beat;
use crate::core::Rhythmed;
use crate::gui::fx::Music;
use crate::tiers::held;

/// A value following a target smoothly: rising over `attack` seconds,
/// falling over `release` (each the time to get about two thirds of the
/// way), alike at any frame rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Follower {
    /// Where it is.
    pub value: f32,
    attack: f32,
    release: f32,
}

impl Follower {
    /// At 0, rising over `attack` seconds and falling over `release`.
    pub const fn new(attack: f32, release: f32) -> Follower {
        Follower {
            value: 0.0,
            attack,
            release,
        }
    }

    /// `dt` seconds on towards `target`; where it is then.
    pub fn step(&mut self, target: f32, dt: f32) -> f32 {
        let tau = if target > self.value {
            self.attack
        } else {
            self.release
        };
        let k = 1.0 - (-dt.max(0.0) / tau.max(1e-4)).exp();
        self.value += (target - self.value) * k;
        self.value
    }
}

/// How far back a level is taken from what is heard, in seconds.
const WINDOW: f64 = 0.04;

/// A song as it is heard, for the backdrop and the effects alike: its
/// analysis (its beats and bars), and where there is one its waveform's
/// lanes and how loud each gets, and where its sections start. Made once
/// for an analysis ([`Song::follow`]), shared; its id another for every
/// one made ([`Song::id`]).
#[derive(Clone, Debug)]
pub struct Song {
    id: u64,
    analysis: Arc<Analysis>,
    lanes: Option<Lanes>,
    /// Where the sections start, in seconds.
    pub sections: Vec<f64>,
}

/// A waveform's lanes and how loud each gets: the mix's and the stems'
/// (as [`Waveform::stems`], in [`Stem::ALL`]'s order).
#[derive(Clone, Debug)]
struct Lanes {
    waveform: Arc<Waveform>,
    mix: Range,
    stems: Option<[Range; 4]>,
}

impl Song {
    /// The song `rhythm` is the analysis of, its sections on its bars
    /// (none without its waveform: where the music lifts is not known).
    pub fn of(rhythm: &Rhythmed) -> Song {
        let a = &rhythm.analysis;
        let bars: Vec<f64> = (a.downbeats.iter())
            .filter_map(|&i| a.beats.get(i).copied())
            .collect();
        let lanes = rhythm.waveform.as_ref().map(|w| Lanes {
            waveform: Arc::clone(w),
            mix: Range::of(&w.mix.rms),
            stems: (w.stems.as_ref()).map(|s| s.each_ref().map(|l| Range::of(&l.rms))),
        });
        let sections = (lanes.as_ref()).map_or(Vec::new(), |l| sections(&l.waveform.mix, &bars));
        static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Song {
            id: MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            analysis: Arc::clone(a),
            lanes,
            sections,
        }
    }

    /// `kept` as the song `rhythm` is the analysis of (none: none), made
    /// again only where the analysis or its waveform is new.
    pub fn follow(kept: &mut Option<Arc<Song>>, rhythm: Option<&Rhythmed>) {
        let Some(r) = rhythm else {
            *kept = None;
            return;
        };
        let same = |s: &Song| {
            let waveform = s.lanes.as_ref().map(|l| &l.waveform);
            Arc::ptr_eq(&s.analysis, &r.analysis)
                && match (waveform, &r.waveform) {
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    (a, b) => a.is_none() && b.is_none(),
                }
        };
        if !kept.as_deref().is_some_and(same) {
            *kept = Some(Arc::new(Song::of(r)));
        }
    }

    /// Its id: another for every song made (a new analysis, a grid fixed
    /// by hand, its waveform come), the same for its clones; what is made
    /// from a song is kept by it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Its analysis.
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    /// Its analysis, shared: what is cut on it holds it, to know it again.
    pub fn shared_analysis(&self) -> Arc<Analysis> {
        Arc::clone(&self.analysis)
    }

    /// Its waveform, where there is one.
    pub fn waveform(&self) -> Option<&Waveform> {
        self.lanes.as_ref().map(|l| &*l.waveform)
    }

    /// How loud the mix is at `t` seconds, and each stem (in
    /// [`Stem::ALL`]'s order) where there are stems, each against how loud
    /// it gets in this song (0 to 1, by the drivers' own
    /// [`jodel_looks::drive::Range`]); none without a waveform. Raw: the
    /// loudest of the last [`WINDOW`], for the backdrop's light, which
    /// follows it at its own pace ([`Follower`]) through the desk's gains
    /// (what is muted lights nothing); the look and the effects move by the
    /// drivers' eased levels instead ([`jodel_looks::Driver::Level`]).
    pub fn loudness(&self, t: f64) -> Option<(f32, Option<[f32; 4]>)> {
        let l = self.lanes.as_ref()?;
        let from = t - WINDOW;
        let mix = l.mix.share(l.waveform.mix.rms_between(from, t));
        let stems = match (&l.waveform.stems, &l.stems) {
            (Some(lanes), Some(r)) => {
                Some(Stem::ALL.map(|s| r[s as usize].share(lanes[s as usize].rms_between(from, t))))
            }
            _ => None,
        };
        Some((mix, stems))
    }

    /// Whether a section starts after `from` and by `to` seconds.
    pub fn section_between(&self, from: f64, to: f64) -> bool {
        let i = self.sections.partition_point(|&s| s <= from);
        self.sections.get(i).is_some_and(|&s| s <= to)
    }
}
/// How long either side of a moment the music is compared, in seconds.
const SPAN: f64 = 4.0;
/// How much louder after than before a section starts, in dB.
const LIFT_DB: f32 = 4.0;
/// The least time between two sections, in seconds.
const APART: f64 = 12.0;
/// How far a section's start goes to a bar's, in seconds.
const TO_BAR: f64 = 0.6;

/// Where the music lifts in `mix` (its RMS lane): each moment the
/// [`SPAN`] after is [`LIFT_DB`] louder than the span before (the most
/// within [`APART`] of it), and louder than the song's middle; on the
/// nearest of `downbeats` (seconds) within [`TO_BAR`].
pub fn sections(mix: &Lane, downbeats: &[f64]) -> Vec<f64> {
    let n = mix.rms.len();
    let span = (SPAN * FPS) as usize;
    if n < 2 * span {
        return Vec::new();
    }
    let mut sum = vec![0u64; n + 1];
    for (i, &v) in mix.rms.iter().enumerate() {
        sum[i + 1] = sum[i] + u64::from(v);
    }
    let mean = |a: usize, b: usize| (sum[b] - sum[a]) as f32 / (b - a) as f32;
    let middle = mean(0, n);
    let step = (FPS / 4.0) as usize;
    let lift = LIFT_DB * 255.0 / 60.0;
    let rises: Vec<(usize, f32)> = (span..=n - span)
        .step_by(step)
        .map(|i| (i, mean(i, i + span) - mean(i - span, i)))
        .filter(|&(i, rise)| rise >= lift && mean(i, i + span) > middle)
        .collect();
    let apart = (APART * FPS) as usize;
    let mut found: Vec<f64> = Vec::new();
    for &(i, rise) in &rises {
        let best = rises
            .iter()
            .filter(|&&(j, _)| j.abs_diff(i) < apart)
            .all(|&(j, r)| r < rise || (r == rise && j >= i));
        let t = i as f64 / FPS;
        if best && found.last().is_none_or(|&last| t - last >= APART) {
            found.push(t);
        }
    }
    for t in &mut found {
        let bar = downbeats
            .iter()
            .copied()
            .min_by(|a, b| (a - *t).abs().total_cmp(&(b - *t).abs()));
        if let Some(bar) = bar.filter(|b| (b - *t).abs() <= TO_BAR) {
            *t = bar;
        }
    }
    found
}

/// The music as felt now (each 0 to 1), and the beat.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Felt {
    /// The singer's glow: the singer's tier's (its row's).
    pub glow: f32,
    /// A section just begun: the whole lifting, fading.
    pub section: f32,
    /// How much harder the scene moves with the beat: the singer's tier's
    /// (its row's).
    pub tier: f32,
    /// A phrase just landed: the scene flaring, fading.
    pub flare: f32,
    /// The beat's pushes now (as the grid has them: not smoothed).
    pub beat: Beat,
}

impl Felt {
    /// How much the whole backdrop lifts: a section begun, and the
    /// singer's glow a little.
    pub fn lift(&self) -> f32 {
        0.35 * self.section + 0.12 * self.glow
    }
}

/// How long a section's swell holds before it fades, in seconds.
const SECTION_HOLD: f32 = 0.6;
/// How long a phrase's flare holds before it fades, in seconds.
const FLARE_HOLD: f32 = 0.15;

/// A moment's swell: held a while once it comes, then rising and fading
/// as its follower does.
#[derive(Clone, Debug, PartialEq)]
struct Swell {
    follower: Follower,
    /// How long it holds once it comes, and still holds now, in seconds.
    hold: (f32, f32),
}

impl Swell {
    const fn new(hold: f32, attack: f32, release: f32) -> Swell {
        Swell {
            follower: Follower::new(attack, release),
            hold: (hold, 0.0),
        }
    }

    /// `dt` seconds on, the moment come (`now`) or not: the swell and
    /// where it is going.
    fn step(&mut self, now: bool, dt: f32) -> (f32, f32) {
        if now {
            self.hold.1 = self.hold.0;
        }
        self.hold.1 = (self.hold.1 - dt).max(0.0);
        let aim = if self.hold.1 > 0.0 { 1.0 } else { 0.0 };
        (self.follower.step(aim, dt), aim)
    }
}

/// The moments smoothed: each with its own attack and release.
#[derive(Clone, Debug, PartialEq)]
pub struct Feel {
    glow: Follower,
    tier: Follower,
    section: Swell,
    flare: Swell,
    /// Whether every value is where it is going (nothing moves till the
    /// music does).
    settled: bool,
}

impl Default for Feel {
    fn default() -> Feel {
        Feel {
            glow: Follower::new(0.6, 1.5),
            tier: Follower::new(0.6, 1.5),
            section: Swell::new(SECTION_HOLD, 0.35, 2.2),
            flare: Swell::new(FLARE_HOLD, 0.05, 0.8),
            settled: true,
        }
    }
}

/// The glow lifting the scene and how much harder it moves with the beat
/// at the singer's `glow` (0 none, up to the top tier's 1): each held by
/// the tiers' rows.
fn tier_light(glow: f32) -> [f32; 2] {
    [
        held(glow, |t| t.backdrop_glow),
        held(glow, |t| t.backdrop_beat),
    ]
}

/// What the backdrop hears in a frame: the singer's glow (0 none, up to
/// the top tier's 1), whether a section began and a phrase landed since
/// the last frame, and the music (its beat: [`Beat::of`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hearing {
    /// The singer's glow.
    pub glow: f32,
    /// A section began.
    pub section: bool,
    /// A phrase landed.
    pub phrase: bool,
    /// The music, as its clock has it.
    pub music: Music,
}

impl Feel {
    /// `dt` seconds on with what is `heard`.
    pub fn step(&mut self, heard: &Hearing, dt: f32) -> Felt {
        let (swell, swelling) = self.section.step(heard.section, dt);
        let (flare, flaring) = self.flare.step(heard.phrase, dt);
        let [lit, tier] = tier_light(heard.glow);
        let beat = Beat::of(&heard.music);
        let felt = Felt {
            glow: self.glow.step(lit, dt),
            section: swell,
            tier: self.tier.step(tier, dt),
            flare,
            beat,
        };
        let aims = [
            (felt.glow, lit),
            (felt.tier, tier),
            (swell, swelling),
            (flare, flaring),
        ];
        self.settled = !beat.moving() && aims.iter().all(|(at, aim)| (at - aim).abs() < 0.003);
        felt
    }

    /// Whether every value is where it is going: no frame changes it.
    pub fn settled(&self) -> bool {
        self.settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_follower_rises_and_falls_alike_at_any_frame_rate() {
        let mut ends = Vec::new();
        for fps in [30.0f32, 60.0, 240.0] {
            let mut f = Follower::new(0.1, 0.5);
            for _ in 0..(0.1 * fps) as usize {
                f.step(1.0, 1.0 / fps);
            }
            let risen = f.value;
            for _ in 0..(0.5 * fps) as usize {
                f.step(0.0, 1.0 / fps);
            }
            ends.push((risen, f.value));
        }
        for (risen, fallen) in &ends {
            assert!((risen - 0.632).abs() < 0.02, "{risen}");
            assert!((fallen - 0.632 * 0.368).abs() < 0.02, "{fallen}");
        }
    }

    /// A lane of `secs` seconds whose RMS (and peak) is `level(t)`.
    fn lane(secs: f64, level: impl Fn(f64) -> u8) -> Lane {
        let n = (secs * FPS) as usize;
        let v: Vec<u8> = (0..n).map(|i| level(i as f64 / FPS)).collect();
        Lane {
            peak: v.clone(),
            rms: v,
        }
    }

    #[test]
    fn a_section_starts_where_the_music_lifts_on_its_bar() {
        // Quiet verse, loud chorus at 20 s, quiet again, loud at 50 s.
        let mix = lane(70.0, |t| {
            if (20.0..40.0).contains(&t) || t >= 50.0 {
                200
            } else {
                120
            }
        });
        let found = sections(&mix, &[19.7, 49.9, 30.0]);
        assert_eq!(found, vec![19.7, 49.9]);
        // A song that stays level has none.
        assert!(sections(&lane(70.0, |_| 150), &[]).is_empty());
    }

    #[test]
    fn loudness_is_each_stem_s_own_against_how_loud_it_gets() {
        let quietish = |t: f64| if t.fract() < 0.5 { 90 } else { 210 };
        let mut w = Waveform {
            mix: lane(30.0, quietish),
            stems: None,
            melody: None,
        };
        let heard = |w: &Waveform| Song::of(&Rhythmed::new(Analysis::default(), Some(w.clone())));
        let song = heard(&w);
        let (loud, stems) = song.loudness(10.9).expect("a waveform");
        assert!(loud > 0.9 && stems.is_none(), "{loud}");
        assert!(song.loudness(10.3).expect("a waveform").0 < 0.1);
        // With stems: the vocals' own.
        w.stems = Some([
            lane(30.0, |_| 100),
            lane(30.0, |_| 100),
            lane(30.0, |_| 100),
            lane(30.0, quietish),
        ]);
        let song = heard(&w);
        let vocals = |t: f64| song.loudness(t).and_then(|(_, s)| s).expect("stems")[3];
        assert!(vocals(10.9) > 0.9 && vocals(10.3) < 0.1);
    }

    #[test]
    fn a_section_swells_and_fades_and_diva_glows() {
        let mut feel = Feel::default();
        let dt = 1.0 / 60.0;
        let at = |glow: f32, section: bool| Hearing {
            glow,
            section,
            ..Hearing::default()
        };
        let mut felt = feel.step(&at(0.0, true), dt);
        for _ in 0..40 {
            felt = feel.step(&at(0.0, false), dt);
        }
        assert!(felt.section > 0.6 && felt.lift() > 0.2, "{felt:?}");
        for _ in 0..900 {
            felt = feel.step(&at(0.0, false), dt);
        }
        assert!(felt.section < 0.01 && felt.lift() < 0.01, "{felt:?}");
        for _ in 0..300 {
            felt = feel.step(&at(1.0, false), dt);
        }
        assert!(felt.glow > 0.95 && felt.tier > 0.95, "{felt:?}");
        // Held there long enough, nothing moves.
        for _ in 0..600 {
            feel.step(&at(1.0, false), dt);
        }
        assert!(feel.settled());
        feel.step(&at(0.0, false), dt);
        assert!(!feel.settled());
    }

    /// A phrase landing flares at once and fades within a second or two;
    /// the beat passes through as the grid has it, and moves the frames on.
    #[test]
    fn a_phrase_flares_and_the_beat_moves_the_frames() {
        let mut feel = Feel::default();
        let dt = 1.0 / 60.0;
        let landed = Hearing {
            phrase: true,
            ..Hearing::default()
        };
        let mut felt = feel.step(&landed, dt);
        for _ in 0..8 {
            felt = feel.step(&Hearing::default(), dt);
        }
        assert!(felt.flare > 0.85, "{felt:?}");
        for _ in 0..300 {
            felt = feel.step(&Hearing::default(), dt);
        }
        assert!(felt.flare < 0.01 && feel.settled(), "{felt:?}");
        let music = Music::steady(8.04, 120.0, 4, 0.9, 0.0);
        let felt = feel.step(
            &Hearing {
                music,
                ..Hearing::default()
            },
            dt,
        );
        assert_eq!(felt.beat, Beat::of(&music));
        assert!(felt.beat.downbeat > 0.9 && !feel.settled(), "{felt:?}");
    }
}
