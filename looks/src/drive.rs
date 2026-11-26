//! What moves the looks: the music, read at the song's time as heard
//! ([`Driver`], its values a frame [`Signals`]). Most of it is worked out
//! once a song, before it plays, from its analysis ([`Curves`]: the
//! beats, the waveform's lanes of the mix and each stem, the sections,
//! the key, the sung lines), as curves on the song's own frames (100 a
//! second) and lists of moments (a beat, a kick, a line's start), each
//! read at any time: a curve between its frames, a moment's pulse worked
//! out from its time. The same song gives the same curves, and the same
//! time the same values, at any frame rate, across seeks and pauses: the
//! looks follow the time heard, never a counter of their own. Knowing the
//! song ahead, a pulse rises just before its moment and peaks on it
//! ([`LEAD`]), so what is seen lands on what is heard. What only the
//! singing knows (the pitch sung, how well, the game's tier and streak)
//! is set over them live ([`Signals::set`]), on the singing's clock.

use jodel_rhythm::stems::Stem;
use jodel_rhythm::waveform::{Lane, Waveform, FPS};
use jodel_rhythm::Analysis;

/// How long a moment's pulse rises before it, seconds: it peaks on the
/// moment.
pub const LEAD: f64 = 0.025;

macro_rules! drivers {
    ($($d:ident $name:literal $speed:ident $about:literal,)*) => {
        /// What a knob can be moved by.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum Driver {
            $(#[doc = $about] $d,)*
        }

        impl Driver {
            /// Every driver.
            pub const ALL: [Driver; Driver::COUNT] = [$(Driver::$d,)*];
            /// How many there are.
            pub const COUNT: usize = [$($name,)*].len();

            /// Its name, as a look has it.
            pub fn name(self) -> &'static str {
                match self { $(Driver::$d => $name,)* }
            }

            /// What it is and what it is good for, in a line.
            pub fn about(self) -> &'static str {
                match self { $(Driver::$d => $about,)* }
            }

            /// How fast it can move.
            pub fn speed(self) -> Speed {
                match self { $(Driver::$d => Speed::$speed,)* }
            }
        }
    };
}

drivers! {
    Beat "beat" Fast "a pulse on every beat, peaking on it: pumping, ticking",
    Downbeat "downbeat" Fast "a pulse on each bar's first beat: flares, light leaks, accents",
    BeatPhase "beat_phase" Fast "where it is in the beat, 0 to 1 (a ramp): spinning and scrolling on the beat",
    BarPhase "bar_phase" Fast "where it is in the bar, 0 to 1 (a ramp): slower cycles locked to the bar",
    Tempo "tempo" Slow "the tempo, 0.5 at 60 to 1 at 120 to 1.5 at 180 a minute: faster songs move faster",
    Level "level" Fast "how loud the whole music is now: the general energy",
    Drums "drums" Fast "the drums' level: busy textures, shaking",
    Bass "bass" Fast "the bass's level: breathing grain and halation, pushing zooms",
    Vocals "vocals" Fast "the voice's level: glow and bloom that follow the singer",
    Other "other" Fast "the other instruments' level: colour and wash",
    Kick "kick" Fast "a pulse on each kick-like drum hit (a drum hit on beats 1 and 3): thumps, dot sizes",
    Snare "snare" Fast "a pulse on each snare-like drum hit (on beats 2 and 4): glitches, cuts, turns",
    BassNote "bass_note" Fast "a pulse on each bass note's start: wobbles, steps",
    Phrase "phrase" Fast "a pulse as each sung line starts: a sweep or a bloom for the line",
    Low "low" Fast "the low end (bass and kicks): weight, size",
    Mid "mid" Fast "the middle (voice and instruments): colour, body",
    High "high" Fast "the top (the drums' bright hits): sparkle, grain, glints",
    Minor "minor" Slow "1 in a minor key, 0 in a major: cooler or darker grades",
    KeyHue "key_hue" Slow "the key's place on the circle of fifths, 0 to 1: a hue per key",
    Section "section" Fast "a swell as a section starts, fading over seconds: big changes",
    Energy "energy" Slow "how loud the section is against the song, 0 to 1: verses low, choruses high",
    Chorus "chorus" Slow "1 while the loudest sections (the choruses) play, easing in and out: more of everything",
    Part "part" Slow "a new value for each section (steps round 0 to 1), eased: a new hue or place each section",
    Singing "singing" Slow "1 while a line is sung, easing: the look calms under the words",
    Instrumental "instrumental" Slow "1 in a break without singing for a while: more visualizer",
    Quiet "quiet" Slow "1 in the quiet parts, slowly: tighter vignettes, darker",
    Pitch "pitch" Fast "the pitch the singer sings now, C2 to C6 as 0 to 1 (live)",
    Melody "melody" Fast "the song's melody's pitch, as the pitch",
    Accuracy "accuracy" Slow "how well the singer hits the notes lately, 0 to 1 (live)",
    Tier "tier" Slow "the game's tier as a glow, 0 to 1 (live)",
    Streak "streak" Slow "the multiplier's rung, 0 to 1 (live)",
    Progress "progress" Slow "how far the song has gone, 0 to 1: a look that ages with the song",
}

/// How fast a driver moves: fast (pulses, levels, phases: they may flash)
/// or slow (eased over half a second or more).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Speed {
    /// Pulses and levels.
    Fast,
    /// Eased.
    Slow,
}

impl Driver {
    /// The driver named `name`.
    pub fn named(name: &str) -> Option<Driver> {
        Driver::ALL.into_iter().find(|d| d.name() == name)
    }
}

/// Every driver's value at a moment, 0 to 1 each (the tempo to 1.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Signals(pub [f32; Driver::COUNT]);

impl Default for Signals {
    fn default() -> Signals {
        Signals::QUIET
    }
}

impl Signals {
    /// No music.
    pub const QUIET: Signals = Signals([0.0; Driver::COUNT]);

    /// `d`'s value.
    pub fn get(&self, d: Driver) -> f32 {
        self.0[d as usize]
    }

    /// `d` at `v`.
    pub fn set(&mut self, d: Driver, v: f32) {
        self.0[d as usize] = if v.is_finite() { v } else { 0.0 };
    }
}

/// How a level follows: rising over `attack` seconds, falling over
/// `release` (each the time to get two thirds of the way).
#[derive(Clone, Copy, Debug)]
struct Ease {
    attack: f32,
    release: f32,
}

impl Ease {
    /// `v` followed so, frame by frame at the waveform's rate.
    fn over(self, v: &[f32]) -> Vec<f32> {
        let dt = (1.0 / FPS) as f32;
        let (ka, kr) = (
            1.0 - (-dt / self.attack).exp(),
            1.0 - (-dt / self.release).exp(),
        );
        let mut at = 0.0;
        v.iter()
            .map(|&x| {
                at += (x - at) * if x > at { ka } else { kr };
                at
            })
            .collect()
    }
}

/// A lane's RMS as a share between how quiet and how loud it gets in the
/// song (its 30th and 97th percentiles), frame by frame.
fn shares(lane: &[u8]) -> Vec<f32> {
    if lane.is_empty() {
        return Vec::new();
    }
    let mut sorted = lane.to_vec();
    sorted.sort_unstable();
    let at = |q: f64| f32::from(sorted[((sorted.len() - 1) as f64 * q) as usize]);
    let quiet = at(0.3);
    let loud = at(0.97).max(quiet + 12.0);
    lane.iter()
        .map(|&v| ((f32::from(v) - quiet) / (loud - quiet)).clamp(0.0, 1.0))
        .collect()
}

/// Where a lane's level jumps up (onsets), seconds: each frame rising most
/// in its 60 ms, well above the rises round it, 80 ms apart at least.
fn onsets(lane: &[u8]) -> Vec<f64> {
    let n = lane.len();
    if n < 8 {
        return Vec::new();
    }
    let v: Vec<f32> = lane.iter().map(|&b| f32::from(b)).collect();
    let rise: Vec<f32> = (0..n)
        .map(|i| {
            if i < 2 {
                0.0
            } else {
                (v[i] - v[i - 2]).max(0.0)
            }
        })
        .collect();
    let (reach, gap) = (25usize, 8usize);
    let mut found: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if rise[i] < 6.0 || rise[i] < rise[i - 1] || rise[i] < rise[i + 1] {
            continue;
        }
        let (a, b) = (i.saturating_sub(reach), (i + reach).min(n));
        let mean = rise[a..b].iter().sum::<f32>() / (b - a) as f32;
        if rise[i] < 1.5 * mean + 3.0 {
            continue;
        }
        if found.last().is_some_and(|&j| i - j < gap) {
            continue;
        }
        found.push(i);
    }
    found.into_iter().map(|i| (i as f64 - 1.0) / FPS).collect()
}

/// A moment's pulse at `t`: rising over [`LEAD`] before it to 1 on it,
/// falling by `decay` seconds after.
fn pulse_of(moments: &[f64], t: f64, decay: f64) -> f32 {
    let i = moments.partition_point(|&m| m <= t);
    let after = i
        .checked_sub(1)
        .map_or(0.0, |j| (-(t - moments[j]) / decay).exp());
    let before = moments.get(i).map_or(0.0, |&m| {
        let x = 1.0 - (m - t) / LEAD;
        if x > 0.0 {
            x * x
        } else {
            0.0
        }
    });
    after.max(before) as f32
}

/// A curve on the waveform's frames at `t`, between them.
fn curve_at(c: &[f32], t: f64) -> f32 {
    if c.is_empty() {
        return 0.0;
    }
    let x = (t * FPS).max(0.0);
    let i = (x.floor() as usize).min(c.len() - 1);
    let j = (i + 1).min(c.len() - 1);
    let f = (x - i as f64).clamp(0.0, 1.0) as f32;
    c[i] + (c[j] - c[i]) * f
}

/// A song's music as the looks read it, worked out once before it plays:
/// its beats and bars, its curves (each on the waveform's frames) and its
/// moments, its key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curves {
    beats: Vec<f64>,
    bars: Vec<f64>,
    tempo: Vec<f64>,
    level: Vec<f32>,
    /// Each stem's level, in the stems' order.
    stems: [Vec<f32>; 4],
    high: Vec<f32>,
    energy: Vec<f32>,
    chorus: Vec<f32>,
    part: Vec<f32>,
    singing: Vec<f32>,
    instrumental: Vec<f32>,
    quiet: Vec<f32>,
    kicks: Vec<f64>,
    snares: Vec<f64>,
    bass_notes: Vec<f64>,
    phrases: Vec<f64>,
    sections: Vec<f64>,
    minor: f32,
    key_hue: f32,
    length: f64,
    /// Every moment a spark flies from (the kicks, snares and bass notes),
    /// in time.
    hits: Vec<f64>,
}

/// The waveform's frames covering `length` seconds (one at least).
fn frames(length: f64) -> usize {
    ((length * FPS).ceil() as usize).max(1)
}

impl Curves {
    /// The curves of a song: its `analysis` (beats, bars, key), its
    /// `waveform` (the mix's and the stems' lanes; none: the beats alone),
    /// where its `sections` start (seconds) and its sung `lines` (start
    /// and end, seconds, in time).
    pub fn new(
        analysis: &Analysis,
        waveform: Option<&Waveform>,
        sections: &[f64],
        lines: &[(f64, f64)],
    ) -> Curves {
        let length = waveform
            .map_or(analysis.duration, Waveform::duration)
            .max(analysis.duration);
        let n = frames(length);
        let fit = |v: Vec<f32>| -> Vec<f32> {
            let mut v = v;
            v.resize(n, v.last().copied().unwrap_or(0.0));
            v
        };
        let lane = |l: &Lane| fit(shares(&l.rms));
        let env = Ease {
            attack: 0.03,
            release: 0.22,
        };
        let mix = waveform.map_or_else(|| vec![0.0; n], |w| lane(&w.mix));
        let stems: [Vec<f32>; 4] = match waveform.and_then(|w| w.stems.as_ref()) {
            Some(s) => s.each_ref().map(|l| env.over(&lane(l))),
            None => std::array::from_fn(|_| env.over(&mix)),
        };
        let level = env.over(&mix);
        let bars: Vec<f64> = (analysis.downbeats.iter())
            .filter_map(|&i| analysis.beats.get(i).copied())
            .collect();
        // Drum hits on beats 1 and 3 are kicks, on 2 and 4 snares, the
        // rest let go (where there are beats; all kicks without).
        let drum_lane = waveform
            .and_then(|w| w.stems.as_ref())
            .map(|s| &s[Stem::Drums as usize]);
        let drum_hits = onsets(
            &drum_lane
                .or(waveform.map(|w| &w.mix))
                .map_or(Vec::new(), |l| l.rms.clone()),
        );
        let (mut kicks, mut snares) = (Vec::new(), Vec::new());
        for &h in &drum_hits {
            if analysis.beats.is_empty() {
                kicks.push(h);
                continue;
            }
            let i = analysis.beats.partition_point(|&b| b < h);
            let near = [i.checked_sub(1), Some(i)]
                .into_iter()
                .flatten()
                .filter(|&j| j < analysis.beats.len())
                .min_by(|&a, &b| {
                    (analysis.beats[a] - h)
                        .abs()
                        .total_cmp(&(analysis.beats[b] - h).abs())
                });
            let Some(j) = near.filter(|&j| (analysis.beats[j] - h).abs() < 0.07) else {
                continue;
            };
            let first = analysis.downbeats.partition_point(|&d| d <= j);
            let in_bar = first
                .checked_sub(1)
                .map_or(j, |k| j - analysis.downbeats[k]);
            if in_bar % 2 == 0 {
                kicks.push(analysis.beats[j]);
            } else {
                snares.push(analysis.beats[j]);
            }
        }
        kicks.dedup();
        snares.dedup();
        let bass_notes = waveform
            .and_then(|w| w.stems.as_ref())
            .map_or(Vec::new(), |s| onsets(&s[Stem::Bass as usize].rms));
        // The drums' bright hits: how far their peaks stand over their
        // level, eased fast.
        let high = match drum_lane {
            Some(l) => {
                let crest: Vec<f32> = (l.peak.iter().zip(&l.rms))
                    .map(|(&p, &r)| (f32::from(p.saturating_sub(r)) / 40.0).clamp(0.0, 1.0))
                    .collect();
                let gate = lane(l);
                let v: Vec<f32> = fit(crest).iter().zip(&gate).map(|(c, g)| c * g).collect();
                Ease {
                    attack: 0.01,
                    release: 0.12,
                }
                .over(&v)
            }
            None => level.clone(),
        };
        let slow = Ease {
            attack: 0.8,
            release: 0.8,
        };
        // Each section's loudness against the song's: the choruses the
        // loudest.
        let mut starts = vec![0.0];
        starts.extend(sections.iter().copied().filter(|&s| s > 0.0 && s < length));
        let span = |k: usize| {
            let a = ((starts[k] * FPS) as usize).min(n);
            let b = starts
                .get(k + 1)
                .map_or(n, |&e| ((e * FPS) as usize).min(n));
            (a, b.max(a))
        };
        let means: Vec<f32> = (0..starts.len())
            .map(|k| {
                let (a, b) = span(k);
                if b > a {
                    mix[a..b].iter().sum::<f32>() / (b - a) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let (lo, hi) = means
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), &m| (l.min(m), h.max(m)));
        let mut sorted = means.clone();
        sorted.sort_by(f32::total_cmp);
        let loud = sorted[sorted.len() * 2 / 3];
        let mut energy_raw = vec![0.0; n];
        let mut chorus_raw = vec![0.0; n];
        for (k, &m) in means.iter().enumerate() {
            let (a, b) = span(k);
            let e = if hi - lo > 0.05 {
                (m - lo) / (hi - lo)
            } else {
                0.5
            };
            let c = if means.len() > 2 && m >= loud && e > 0.5 {
                1.0
            } else {
                0.0
            };
            energy_raw[a..b].fill(e);
            chorus_raw[a..b].fill(c);
        }
        let energy = slow.over(&energy_raw);
        let chorus = slow.over(&chorus_raw);
        // Each section a new place round the circle, by the golden ratio's
        // steps, eased from the last (the short way round is not looked
        // for: a hue moving on is the point).
        let mut part_raw = vec![0.0; n];
        for k in 0..starts.len() {
            let (a, b) = span(k);
            part_raw[a..b].fill(fract_f(k as f32 * 0.618_034));
        }
        let part = slow.over(&part_raw);
        // The lines sung: 1 within one; a break 2 s from any.
        let mut sung = vec![0.0; n];
        let mut near_line = vec![0.0; n];
        for &(a, b) in lines {
            let at = |t: f64| ((t.max(0.0) * FPS) as usize).min(n);
            sung[at(a)..at(b).max(at(a))].fill(1.0);
            near_line[at(a - 2.0)..at(b + 2.0).max(at(a - 2.0))].fill(1.0);
        }
        let singing = Ease {
            attack: 0.3,
            release: 0.5,
        }
        .over(&sung);
        let instrumental = slow.over(
            &near_line
                .iter()
                .map(|&v| if lines.is_empty() { 0.0 } else { 1.0 - v })
                .collect::<Vec<f32>>(),
        );
        let quiet = Ease {
            attack: 1.0,
            release: 1.0,
        }
        .over(&level.iter().map(|&v| 1.0 - v).collect::<Vec<f32>>());
        let mut phrases: Vec<f64> = lines.iter().map(|&(a, _)| a).collect();
        phrases.sort_by(f64::total_cmp);
        let (minor, key_hue) = analysis.key.map_or((0.0, 0.0), |k| {
            (
                if k.minor() { 1.0 } else { 0.0 },
                f32::from(k.class % 12) / 12.0,
            )
        });
        let mut hits: Vec<f64> = kicks
            .iter()
            .chain(&snares)
            .chain(&bass_notes)
            .copied()
            .collect();
        hits.sort_by(f64::total_cmp);
        Curves {
            beats: analysis.beats.clone(),
            bars,
            tempo: analysis.local_bpm.clone(),
            level,
            stems,
            high,
            energy,
            chorus,
            part,
            singing,
            instrumental,
            quiet,
            kicks,
            snares,
            bass_notes,
            phrases,
            sections: sections.to_vec(),
            minor,
            key_hue,
            length,
            hits,
        }
    }

    /// The song's length, seconds.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Where the beats are, seconds.
    pub fn beats(&self) -> &[f64] {
        &self.beats
    }

    /// Where the bars start, seconds.
    pub fn bars(&self) -> &[f64] {
        &self.bars
    }

    /// The kick-like hits, seconds.
    pub fn kicks(&self) -> &[f64] {
        &self.kicks
    }

    /// The snare-like hits, seconds.
    pub fn snares(&self) -> &[f64] {
        &self.snares
    }

    /// The moments sparks fly from, at or before `t`, the latest first, at
    /// most `n`.
    pub fn hits_before(&self, t: f64, n: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let i = self.hits.partition_point(|&h| h <= t);
        (0..i).rev().take(n).map(|j| (j, self.hits[j]))
    }

    /// Every driver at `t` seconds into the song (the live ones none: the
    /// caller sets them).
    pub fn at(&self, t: f64) -> Signals {
        let mut s = Signals::QUIET;
        let (beat_phase, beat_n) = phase(&self.beats, t);
        let (bar_phase, _) = phase(&self.bars, t);
        s.set(Driver::Beat, pulse_of(&self.beats, t, 0.12));
        s.set(Driver::Downbeat, pulse_of(&self.bars, t, 0.3));
        s.set(Driver::BeatPhase, beat_phase);
        s.set(Driver::BarPhase, bar_phase);
        let bpm = beat_n
            .and_then(|i| self.tempo.get(i).copied())
            .unwrap_or(120.0);
        s.set(Driver::Tempo, (bpm / 120.0).clamp(0.5, 1.5) as f32);
        s.set(Driver::Level, curve_at(&self.level, t));
        let stem = |x: Stem| curve_at(&self.stems[x as usize], t);
        s.set(Driver::Drums, stem(Stem::Drums));
        s.set(Driver::Bass, stem(Stem::Bass));
        s.set(Driver::Vocals, stem(Stem::Vocals));
        s.set(Driver::Other, stem(Stem::Other));
        let kick = pulse_of(&self.kicks, t, 0.12);
        s.set(Driver::Kick, kick);
        s.set(Driver::Snare, pulse_of(&self.snares, t, 0.15));
        s.set(Driver::BassNote, pulse_of(&self.bass_notes, t, 0.2));
        s.set(Driver::Phrase, pulse_of(&self.phrases, t, 0.6));
        s.set(Driver::Low, stem(Stem::Bass).max(0.8 * kick));
        s.set(Driver::Mid, 0.5 * (stem(Stem::Vocals) + stem(Stem::Other)));
        s.set(Driver::High, curve_at(&self.high, t));
        s.set(Driver::Minor, self.minor);
        s.set(Driver::KeyHue, self.key_hue);
        s.set(Driver::Section, pulse_of(&self.sections, t, 1.5));
        s.set(Driver::Energy, curve_at(&self.energy, t));
        s.set(Driver::Chorus, curve_at(&self.chorus, t));
        s.set(Driver::Part, curve_at(&self.part, t));
        s.set(Driver::Singing, curve_at(&self.singing, t));
        s.set(Driver::Instrumental, curve_at(&self.instrumental, t));
        s.set(Driver::Quiet, curve_at(&self.quiet, t));
        if self.length > 0.0 {
            s.set(Driver::Progress, (t / self.length).clamp(0.0, 1.0) as f32);
        }
        s
    }
}

/// Where `t` is between two of `moments` (0 to 1), and the one before
/// (none before the first or from the last on: 0).
fn phase(moments: &[f64], t: f64) -> (f32, Option<usize>) {
    let i = moments.partition_point(|&m| m <= t);
    match (i.checked_sub(1), moments.get(i)) {
        (Some(j), Some(&next)) => {
            let a = moments[j];
            (((t - a) / (next - a).max(1e-6)) as f32, Some(j))
        }
        (Some(j), None) => (0.0, Some(j)),
        _ => (0.0, None),
    }
}

/// What has `x` past its floor.
fn fract_f(x: f32) -> f32 {
    x - x.floor()
}

/// What a pass's music data is worked out from: the song's time, the
/// drivers then and the song's curves (none: no song).
#[derive(Clone, Copy)]
pub struct Feed<'a> {
    /// The time, seconds as heard.
    pub t: f64,
    /// The drivers.
    pub signals: &'a Signals,
    /// The curves.
    pub curves: Option<&'a Curves>,
}

#[cfg(test)]
mod tests;
