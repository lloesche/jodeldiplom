//! What the backdrop hears: each stem's level at the moment heard (the
//! drums' not: they light nothing), read
//! from the song's analysis (its waveform's lanes, 10 ms a frame: the
//! stems as separated, so the vocals light the subject even with the
//! Karaoke voice turned down for the singer), each against how loud that
//! stem gets in this song ([`Song`]), then as the desk lets it be heard
//! (its fader, mute and solo: [`Levels::times`]); the sections' starts,
//! where the music lifts ([`sections`]); and all of it smoothed, so
//! nothing flickers ([`Feel`]). Read on the main thread, frame by frame,
//! from what the analysis already holds: nothing of it touches the audio
//! thread.

use std::sync::Arc;

use jodel_rhythm::waveform::{Lane, Waveform, FPS};

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

/// The stems' levels at a moment, each 0 (quiet for that stem in this
/// song) to 1 (as loud as it gets).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Levels {
    /// The bass.
    pub bass: f32,
    /// The vocals.
    pub vocals: f32,
    /// Everything else.
    pub other: f32,
}

impl Levels {
    /// Each scaled by its own of `gains` (drums, bass, vocals, other: what
    /// the desk lets each be heard at,
    /// [`crate::core::desk::DeskState::heard_gain`]; a stem muted, or not
    /// soloed, is not seen either; the drums' is not needed).
    pub fn times(self, [_, bass, vocals, other]: [f32; 4]) -> Levels {
        Levels {
            bass: self.bass * bass,
            vocals: self.vocals * vocals,
            other: self.other * other,
        }
    }
}

/// How quiet and how loud a lane gets: its levels' 30th and 97th
/// percentiles (on the lane's byte scale).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Range {
    quiet: f32,
    loud: f32,
}

impl Range {
    fn of(levels: &[u8]) -> Range {
        let mut counts = [0usize; 256];
        for &v in levels {
            counts[usize::from(v)] += 1;
        }
        let at = |q: f64| {
            let wanted = (levels.len() as f64 * q) as usize;
            let mut seen = 0;
            for (v, &n) in counts.iter().enumerate() {
                seen += n;
                if seen > wanted {
                    return v as f32;
                }
            }
            255.0
        };
        let quiet = at(0.3);
        Range {
            quiet,
            loud: at(0.97).max(quiet + 12.0),
        }
    }

    /// `v` (the byte scale, 0 to 1) between quiet and loud, 0 to 1.
    fn share(&self, v: f32) -> f32 {
        ((v * 255.0 - self.quiet) / (self.loud - self.quiet)).clamp(0.0, 1.0)
    }
}

/// How far back a level is taken from what is heard, in seconds.
const WINDOW: f64 = 0.04;

/// A song as the backdrop hears it: its waveform's lanes, how loud each
/// gets, and where its sections start.
#[derive(Clone, Debug)]
pub struct Song {
    waveform: Arc<Waveform>,
    /// The mix's and the stems' ranges (bass, other, vocals: of
    /// [`Waveform::stems`], the drums left out).
    mix: Range,
    stems: Option<[Range; 3]>,
    /// Where the sections start, in seconds.
    pub sections: Vec<f64>,
}

impl Song {
    /// The song of `waveform`, its sections on its bars (`downbeats`, in
    /// seconds; none: where the music lifts).
    pub fn of(waveform: Arc<Waveform>, downbeats: &[f64]) -> Song {
        let mix = Range::of(&waveform.mix.rms);
        let stems = waveform.stems.as_ref().map(|s| {
            [
                Range::of(&s[0].rms),
                Range::of(&s[2].rms),
                Range::of(&s[3].rms),
            ]
        });
        let sections = sections(&waveform.mix, downbeats);
        Song {
            waveform,
            mix,
            stems,
            sections,
        }
    }

    /// Whether it is of `waveform`.
    pub fn is_of(&self, waveform: &Arc<Waveform>) -> bool {
        Arc::ptr_eq(&self.waveform, waveform)
    }

    /// The levels heard at `t` seconds; without stems, the mix's for each
    /// (the vocals not: a mix lights no subject).
    pub fn levels(&self, t: f64) -> Levels {
        let from = t - WINDOW;
        match (&self.waveform.stems, &self.stems) {
            (Some(lanes), Some(r)) => Levels {
                bass: r[0].share(lanes[0].rms_between(from, t)),
                other: r[1].share(lanes[2].rms_between(from, t)),
                vocals: r[2].share(lanes[3].rms_between(from, t)),
            },
            _ => {
                let mix = self.mix.share(self.waveform.mix.rms_between(from, t));
                Levels {
                    bass: mix,
                    vocals: 0.0,
                    other: mix,
                }
            }
        }
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

/// The pieces' light as felt now (each 0 to 1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Felt {
    /// The depth's breathing: the bass.
    pub bass: f32,
    /// The subject's light: the vocals.
    pub vocals: f32,
    /// The plate's colour and sweeping light: everything else.
    pub other: f32,
    /// The subject's glow: the singer's shine (Great and up).
    pub glow: f32,
    /// The rays behind the subject: Amazing and Diva, and the sections.
    pub rays: f32,
    /// A section just begun: the whole lifting, fading.
    pub section: f32,
}

/// How long a section's swell holds before it fades, in seconds.
const SECTION_HOLD: f32 = 0.6;

/// The levels and moments smoothed: each with its own attack and release.
#[derive(Clone, Debug, PartialEq)]
pub struct Feel {
    bass: Follower,
    vocals: Follower,
    other: Follower,
    glow: Follower,
    rays: Follower,
    section: Follower,
    /// How long the section's swell still holds.
    hold: f32,
    /// Whether every value is where it is going (nothing moves till the
    /// music does).
    settled: bool,
}

impl Default for Feel {
    fn default() -> Feel {
        Feel {
            bass: Follower::new(0.25, 0.8),
            vocals: Follower::new(0.15, 0.7),
            other: Follower::new(0.6, 1.6),
            glow: Follower::new(0.6, 1.5),
            rays: Follower::new(0.8, 1.8),
            section: Follower::new(0.35, 2.2),
            hold: 0.0,
            settled: true,
        }
    }
}

/// The glow and the rays a shine tier gives (0 none, 1 Good to 4 Diva).
fn tier_light(tier: u32) -> (f32, f32) {
    match tier {
        0 | 1 => (0.0, 0.0),
        2 => (0.35, 0.0),
        3 => (0.65, 0.25),
        _ => (1.0, 0.7),
    }
}

impl Feel {
    /// `dt` seconds on: the stems at `levels`, the singer's shine at
    /// `tier`, a section begun (`section`).
    pub fn step(&mut self, levels: Levels, tier: u32, section: bool, dt: f32) -> Felt {
        if section {
            self.hold = SECTION_HOLD;
        }
        self.hold = (self.hold - dt).max(0.0);
        let held = if self.hold > 0.0 { 1.0 } else { 0.0 };
        let swell = self.section.step(held, dt);
        let (glow, rays) = tier_light(tier);
        let rays = rays.max(swell * 0.6);
        let felt = Felt {
            bass: self.bass.step(levels.bass, dt),
            vocals: self.vocals.step(levels.vocals, dt),
            other: self.other.step(levels.other, dt),
            glow: self.glow.step(glow, dt),
            rays: self.rays.step(rays, dt),
            section: swell,
        };
        let aims = [
            (felt.bass, levels.bass),
            (felt.vocals, levels.vocals),
            (felt.other, levels.other),
            (felt.glow, glow),
            (felt.rays, rays),
            (swell, held),
        ];
        self.settled = aims.iter().all(|(at, aim)| (at - aim).abs() < 0.003);
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
    fn levels_are_each_stem_s_own_and_the_mix_lights_no_subject() {
        let quietish = |t: f64| if t.fract() < 0.5 { 90 } else { 210 };
        let mut w = Waveform {
            mix: lane(30.0, quietish),
            stems: None,
            melody: None,
        };
        let song = Song::of(Arc::new(w.clone()), &[]);
        let loud = song.levels(10.9);
        assert!(loud.bass > 0.9 && loud.vocals == 0.0, "{loud:?}");
        assert!(song.levels(10.3).bass < 0.1);
        // With stems: the vocals' own.
        w.stems = Some([
            lane(30.0, |_| 100),
            lane(30.0, |_| 100),
            lane(30.0, |_| 100),
            lane(30.0, quietish),
        ]);
        let song = Song::of(Arc::new(w), &[]);
        assert!(song.levels(10.9).vocals > 0.9);
        assert!(song.levels(10.3).vocals < 0.1);
        let muted = song.levels(10.9).times([1.0, 1.0, 0.0, 1.0]);
        assert_eq!(muted.vocals, 0.0);
    }

    #[test]
    fn a_section_swells_and_fades_and_diva_brings_the_rays() {
        let mut feel = Feel::default();
        let dt = 1.0 / 60.0;
        let mut felt = feel.step(Levels::default(), 0, true, dt);
        for _ in 0..40 {
            felt = feel.step(Levels::default(), 0, false, dt);
        }
        assert!(felt.section > 0.6 && felt.rays > 0.1, "{felt:?}");
        for _ in 0..900 {
            felt = feel.step(Levels::default(), 0, false, dt);
        }
        assert!(felt.section < 0.01 && felt.rays < 0.01, "{felt:?}");
        for _ in 0..300 {
            felt = feel.step(Levels::default(), 4, false, dt);
        }
        assert!(felt.glow > 0.95 && felt.rays > 0.65, "{felt:?}");
        // Held there long enough, nothing moves.
        for _ in 0..600 {
            feel.step(Levels::default(), 4, false, dt);
        }
        assert!(feel.settled());
        feel.step(Levels::default(), 0, false, dt);
        assert!(!feel.settled());
    }
}
