//! What moves the looks, and the player's effects with them: the music,
//! read at the song's time as heard ([`Driver`], its values a frame
//! [`Signals`]). This is the one place the music is read from: the beat
//! from the song's grid ([`beat`]: phases, pulses, the tempo, how sure the
//! grid is), a lane's level against how loud it gets ([`Range`]), the
//! melody's pitch ([`melody_at`], [`pitch`]) and a swell after a moment
//! ([`swell`]); the effects read a frame's beat through it, and a song's
//! [`Curves`] are built from it.
//!
//! Most of a song is worked out once, before it plays, from its analysis
//! ([`Curves`]: the beats, the waveform's lanes of the mix and each stem,
//! the sections, the key, the melody, the sung lines), as curves on the
//! song's own frames (100 a second) and lists of moments (a kick, a
//! line's start), each read at any time: a curve between its frames, a
//! moment's pulse worked out from its time. Each driver has its source
//! there ([`Origin::Song`]); the same song gives the same curves, and the
//! same time the same values, at any frame rate, across seeks and pauses:
//! the looks follow the time heard, never a counter of their own. Knowing
//! the song ahead, the pulse of a moment found in it (a kick, a snare, a
//! bass note, a line's start) rises just before it and peaks on it
//! ([`LEAD`]), so what is seen lands on what is heard; the beat's and the
//! bar's pulses fall from the grid's beats, and a section's swell from its
//! start, as the effects have them. What the grid is unsure of pulses not
//! at all; its phases run on where it is unsure (the grid runs on at its
//! tempo there). What only the
//! singing knows ([`Origin::Live`]: the pitch sung, how well, the game's
//! tier and streak) is set over them live ([`Signals::set`]), on the
//! singing's clock.

use jodel_rhythm::grid::{Grid, FREE_TEMPO};
use jodel_rhythm::stems::Stem;
use jodel_rhythm::waveform::{Lane, Melody, Waveform, FPS};
use jodel_rhythm::Analysis;

/// How long the pulse of a moment found in the song rises before it,
/// seconds: it peaks on the moment.
pub const LEAD: f64 = 0.025;
/// How fast a kick's pulse falls away.
const KICK_DECAY: f64 = 0.12;
/// How fast a snare's pulse falls away.
const SNARE_DECAY: f64 = 0.15;
/// How fast a bass note's pulse falls away.
const BASS_NOTE_DECAY: f64 = 0.2;
/// How fast a sung line's pulse falls away.
const PHRASE_DECAY: f64 = 0.6;
/// How much of the kick's pulse the low end takes, where it is louder
/// than the bass.
const LOW_KICK: f32 = 0.8;
/// How far a drum hit may be from a beat and still be on it, seconds.
const ON_BEAT: f64 = 0.07;
/// How far a drum's peak stands over its level for the brightest hit (on
/// the lanes' byte scale).
const CREST: f32 = 40.0;
/// How far from a sung line a break begins, seconds.
const LINE_BREAK: f64 = 2.0;
/// How fast a beat's pulse falls away: to about a third in this many
/// seconds.
pub const BEAT_DECAY: f64 = 0.1;
/// How fast a bar's pulse falls away.
pub const BAR_DECAY: f64 = 0.25;
/// How fast a section's swell falls away.
pub const SECTION_DECAY: f64 = 1.5;
/// How sure a region of the grid is for its beats to show not at all, and
/// wholly: what keeps to the beat does so wholly from a moderate agreement
/// on (the beat is clear from 0.5, [`jodel_rhythm::CLEAR`]) and lets it go
/// only where the grid is really unsure.
pub const SURE: (f32, f32) = (0.15, 0.35);
/// The pitches a pitch spans as a driver, MIDI notes: C2 to C6.
pub const PITCHES: (f32, f32) = (36.0, 84.0);

macro_rules! drivers {
    ($($d:ident $name:literal $speed:ident $origin:ident $about:literal,)*) => {
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

            /// Where its value comes from.
            pub fn origin(self) -> Origin {
                match self { $(Driver::$d => Origin::$origin,)* }
            }
        }
    };
}

drivers! {
    Beat "beat" Fast Song "a pulse on every beat, peaking on it: pumping, ticking",
    Downbeat "downbeat" Fast Song "a pulse on each bar's first beat: flares, light leaks, accents",
    BeatPhase "beat_phase" Fast Song "where it is in the beat, 0 to 1 (a ramp): spinning and scrolling on the beat",
    BarPhase "bar_phase" Fast Song "where it is in the bar, 0 to 1 (a ramp): slower cycles locked to the bar",
    Tempo "tempo" Slow Song "the tempo, 0.5 at 60 to 1 at 120 to 1.5 at 180 a minute: faster songs move faster",
    Level "level" Fast Song "how loud the whole music is now: the general energy",
    Drums "drums" Fast Song "the drums' level: busy textures, shaking",
    Bass "bass" Fast Song "the bass's level: breathing grain and halation, pushing zooms",
    Vocals "vocals" Fast Song "the voice's level: glow and bloom that follow the singer",
    Other "other" Fast Song "the other instruments' level: colour and wash",
    Kick "kick" Fast Song "a pulse on each kick-like drum hit (a drum hit on beats 1 and 3): thumps, dot sizes",
    Snare "snare" Fast Song "a pulse on each snare-like drum hit (on beats 2 and 4): glitches, cuts, turns",
    BassNote "bass_note" Fast Song "a pulse on each bass note's start: wobbles, steps",
    Phrase "phrase" Fast Song "a pulse as each sung line starts: a sweep or a bloom for the line",
    Low "low" Fast Song "the low end (bass and kicks): weight, size",
    Mid "mid" Fast Song "the middle (voice and instruments): colour, body",
    High "high" Fast Song "the top (the drums' bright hits): sparkle, grain, glints",
    Minor "minor" Slow Song "1 in a minor key, 0 in a major: cooler or darker grades",
    KeyHue "key_hue" Slow Song "the key's place on the circle of fifths, 0 to 1: a hue per key",
    Section "section" Fast Song "a swell as a section starts, fading over seconds: big changes",
    Energy "energy" Slow Song "how loud the section is against the song, 0 to 1: verses low, choruses high",
    Chorus "chorus" Slow Song "1 while the loudest sections (the choruses) play, easing in and out: more of everything",
    Part "part" Slow Song "a new value for each section (steps round 0 to 1), eased: a new hue or place each section",
    Singing "singing" Slow Song "1 while a line is sung, easing: the look calms under the words",
    Instrumental "instrumental" Slow Song "1 in a break without singing for a while: more visualizer",
    Quiet "quiet" Slow Song "1 in the quiet parts, slowly: tighter vignettes, darker",
    Pitch "pitch" Fast Live "the pitch the singer sings now, C2 to C6 as 0 to 1 (live)",
    Melody "melody" Fast Song "the song's melody's pitch, as the pitch",
    Accuracy "accuracy" Slow Live "how well the singer hits the notes lately, 0 to 1 (live)",
    Tier "tier" Slow Live "the game's tier as a glow, 0 to 1 (live)",
    Streak "streak" Slow Live "the multiplier's rung, 0 to 1 (live)",
    Progress "progress" Slow Song "how far the song has gone, 0 to 1: a look that ages with the song",
}

/// How fast a driver moves: fast (pulses, levels, phases: they may flash)
/// or slow (eased over half a second or more).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    /// Pulses and levels.
    Fast,
    /// Eased.
    Slow,
}

impl Speed {
    /// Its name, as the catalogue has it.
    pub fn name(self) -> &'static str {
        match self {
            Speed::Fast => "fast",
            Speed::Slow => "slow",
        }
    }
}

impl serde::Serialize for Speed {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl Driver {
    /// The driver of a stem's level.
    pub fn of_stem(stem: Stem) -> Driver {
        match stem {
            Stem::Bass => Driver::Bass,
            Stem::Drums => Driver::Drums,
            Stem::Other => Driver::Other,
            Stem::Vocals => Driver::Vocals,
        }
    }
}

/// Where a driver's value comes from: the song, worked out ahead
/// ([`Curves::at`] sets it), or what only the singing knows, set live
/// over it ([`Signals::set`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The song.
    Song,
    /// The singing, live.
    Live,
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
    /// No music: all none, at the free tempo (1).
    pub const QUIET: Signals = {
        let mut s = [0.0; Driver::COUNT];
        s[Driver::Tempo as usize] = 1.0;
        Signals(s)
    };

    /// `d`'s value.
    pub fn get(&self, d: Driver) -> f32 {
        self.0[d as usize]
    }

    /// `d` at `v`.
    pub fn set(&mut self, d: Driver, v: f32) {
        self.0[d as usize] = if v.is_finite() { v } else { 0.0 };
    }

    /// The drivers `beat` gives ([`Beat::signal`]) as it gives them.
    pub fn set_beat(&mut self, beat: &Beat) {
        for d in Driver::ALL {
            if let Some(v) = beat.signal(d) {
                self.set(d, v);
            }
        }
    }
}

/// The beat about a moment, from the song's grid ([`beat`]): where it is
/// in the beat and the bar (0 to 1), a pulse on each beat and each bar's
/// first (falling away by [`BEAT_DECAY`] and [`BAR_DECAY`]), the tempo
/// (beats a minute), how sure the grid is there (0 to 1, faded between
/// [`SURE`]'s bounds), the beats counted from the analysis's first (the
/// whole ones and the share of the next) and the beats in the bar. Its
/// pulses are as the grid has them: the drivers take them as sure as the
/// grid is ([`Beat::signal`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beat {
    /// Where it is in the beat.
    pub phase: f32,
    /// Where it is in the bar.
    pub bar_phase: f32,
    /// The beat's pulse.
    pub beat: f32,
    /// The bar's pulse.
    pub downbeat: f32,
    /// The tempo, beats a minute.
    pub tempo: f32,
    /// How sure the grid is.
    pub sure: f32,
    /// The beats counted.
    pub beats: f32,
    /// The beats in the bar.
    pub bar_beats: f32,
}

impl Beat {
    /// The value of the driver `d` it gives (none: one it does not): its
    /// pulses as sure as the grid is (none where it is unsure), its phases
    /// as they are (a ramp runs on where the grid is unsure: it runs on at
    /// its tempo there), the tempo 1 at the free tempo (120 a minute), from
    /// 0.5 to 1.5.
    pub fn signal(&self, d: Driver) -> Option<f32> {
        Some(match d {
            Driver::Beat => self.beat * self.sure,
            Driver::Downbeat => self.downbeat * self.sure,
            Driver::BeatPhase => self.phase,
            Driver::BarPhase => self.bar_phase,
            Driver::Tempo => (f64::from(self.tempo) / FREE_TEMPO).clamp(0.5, 1.5) as f32,
            _ => return None,
        })
    }
}

/// The beat of `a` at `now` seconds into the song, and the grid there
/// ([`Analysis::grid_at`], the one the sounds are played on).
pub fn beat(a: &Analysis, now: f64) -> (Beat, Grid) {
    let grid = a.grid_at(now);
    let len = grid.beat_len();
    let bar_len = f64::from(grid.bar()) * len;
    let since_beat = (len - (grid.next_step(now, 1).0 - now)).rem_euclid(len);
    let since_bar = (bar_len - (grid.next_downbeat(now) - now)).rem_euclid(bar_len);
    let i = a.beat_at(now).unwrap_or(0);
    let from = a.beats.get(i).copied().unwrap_or(now - since_beat);
    let confidence = a.region_at(now).map_or(0.0, |r| r.confidence as f32);
    let beat = Beat {
        phase: (since_beat / len) as f32,
        bar_phase: (since_bar / bar_len) as f32,
        beat: fall(since_beat, BEAT_DECAY),
        downbeat: fall(since_bar, BAR_DECAY),
        tempo: grid.tempo() as f32,
        sure: crate::math::smoothstep(SURE.0, SURE.1, confidence),
        beats: (i as f64 + (now - from) / len) as f32,
        bar_beats: grid.bar() as f32,
    };
    (beat, grid)
}

/// A pulse `since` seconds after it, falling away by `decay`.
fn fall(since: f64, decay: f64) -> f32 {
    (-since / decay).exp() as f32
}

/// A swell after the latest of `moments` at or before `t`, falling away by
/// `decay` (none before the first).
pub fn swell(moments: &[f64], t: f64, decay: f64) -> f32 {
    let i = moments.partition_point(|&m| m <= t);
    i.checked_sub(1)
        .map_or(0.0, |j| fall(t - moments[j], decay))
}

/// A pitch (a MIDI note; 0 or less: none) as a share of [`PITCHES`], 0
/// to 1 (none: 0).
pub fn pitch(midi: f32) -> f32 {
    if midi <= 0.0 {
        0.0
    } else {
        ((midi - PITCHES.0) / (PITCHES.1 - PITCHES.0)).clamp(0.0, 1.0)
    }
}

/// The melody's pitch at `t` seconds, a MIDI note (0 where none is sure).
pub fn melody_at(m: &Melody, t: f64) -> f32 {
    let i = (t * FPS).floor();
    if i < 0.0 {
        return 0.0;
    }
    let i = i as usize;
    match (m.cents.get(i), m.sure.get(i)) {
        (Some(&c), Some(&s)) if c > 0 && s >= 128 => f32::from(c) / 100.0,
        _ => 0.0,
    }
}

/// How quiet and how loud a lane gets: its levels' 30th and 97th
/// percentiles (on the lane's byte scale), at least 12 apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    quiet: f32,
    loud: f32,
}

impl Range {
    /// The range of `levels` (a lane's bytes).
    pub fn of(levels: &[u8]) -> Range {
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

    /// `v` (the byte scale as 0 to 1) between quiet and loud, 0 to 1.
    pub fn share(&self, v: f32) -> f32 {
        ((v * 255.0 - self.quiet) / (self.loud - self.quiet)).clamp(0.0, 1.0)
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
    /// Eased alike both ways over `seconds`.
    const fn both(seconds: f32) -> Ease {
        Ease {
            attack: seconds,
            release: seconds,
        }
    }

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

/// A level's envelope: fast up, slower down.
const ENVELOPE: Ease = Ease {
    attack: 0.03,
    release: 0.22,
};
/// The drums' bright hits' envelope: faster still.
const HIGH: Ease = Ease {
    attack: 0.01,
    release: 0.12,
};
/// What follows a line being sung.
const SINGING: Ease = Ease {
    attack: 0.3,
    release: 0.5,
};
/// What follows the quiet parts.
const QUIET: Ease = Ease::both(1.0);
/// What eases over the song's sections.
const SLOW: Ease = Ease::both(0.8);

/// A lane's RMS as a share of its [`Range`], frame by frame.
fn shares(lane: &[u8]) -> Vec<f32> {
    let range = Range::of(lane);
    lane.iter()
        .map(|&v| range.share(f32::from(v) / 255.0))
        .collect()
}

/// How much a lane's level must rise in two frames to be an onset (on the
/// lanes' byte scale).
const ONSET_RISE: f32 = 6.0;
/// How far an onset's rise must stand over the mean rise round it: so many
/// times it, and so much more.
const ONSET_OVER: (f32, f32) = (1.5, 3.0);
/// The frames round an onset its rise is held against, either way.
const ONSET_REACH: usize = 25;
/// The fewest frames between two onsets.
const ONSET_GAP: usize = 8;

/// Where a lane's level jumps up (onsets), seconds: each frame rising most
/// in its 60 ms, well above the rises round it ([`ONSET_RISE`],
/// [`ONSET_OVER`] over [`ONSET_REACH`]), [`ONSET_GAP`] frames apart at
/// least.
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
    let mut found: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if rise[i] < ONSET_RISE || rise[i] < rise[i - 1] || rise[i] < rise[i + 1] {
            continue;
        }
        let (a, b) = (i.saturating_sub(ONSET_REACH), (i + ONSET_REACH).min(n));
        let mean = rise[a..b].iter().sum::<f32>() / (b - a) as f32;
        if rise[i] < ONSET_OVER.0 * mean + ONSET_OVER.1 {
            continue;
        }
        if found.last().is_some_and(|&j| i - j < ONSET_GAP) {
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

/// Where a driver's value comes from in a song's [`Curves`]: the beat
/// ([`Beat::signal`]); a curve on the waveform's frames; a pulse on each of some
/// moments; a swell after each; a number for the whole song; the melody's
/// pitch; how far the song has gone; the louder of two drivers (the
/// second scaled), or their mean; or nothing of the song's (live).
#[derive(Clone, Debug, PartialEq)]
enum Source {
    Beat,
    Curve(Vec<f32>),
    Pulse { at: Vec<f64>, decay: f64 },
    Swell { at: Vec<f64>, decay: f64 },
    Const(f32),
    Melody(Melody),
    Progress,
    Louder(Driver, Driver, f32),
    Mean(Driver, Driver),
    Live,
}

/// A song's music as the looks read it, worked out once before it plays:
/// its analysis (the grid the beat is read from), each driver's source,
/// its length, and every moment a spark flies from (the kicks, snares and
/// bass notes, in time).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curves {
    analysis: Analysis,
    /// Each driver's, in [`Driver::ALL`]'s order (none: no song).
    sources: Vec<Source>,
    length: f64,
    hits: Vec<f64>,
}

/// The waveform's frames covering `length` seconds (one at least).
fn frames(length: f64) -> usize {
    ((length * FPS).ceil() as usize).max(1)
}

/// `v` made `n` frames long (its last held).
fn fit(mut v: Vec<f32>, n: usize) -> Vec<f32> {
    v.resize(n, v.last().copied().unwrap_or(0.0));
    v
}

/// The drum hits of a song (its drums' lane, else its mix's) as kicks
/// and snares: a hit on a beat's 1 or 3 a kick, on its 2 or 4 a snare,
/// the rest let go (where there are beats; all kicks without).
fn drum_hits(analysis: &Analysis, drums: Option<&Lane>) -> (Vec<f64>, Vec<f64>) {
    let hits = onsets(drums.map_or(&[][..], |l| &l.rms));
    let beats = &analysis.beats;
    if beats.is_empty() {
        return (hits, Vec::new());
    }
    let (mut kicks, mut snares) = (Vec::new(), Vec::new());
    for &h in &hits {
        let i = beats.partition_point(|&b| b < h);
        let near = [i.checked_sub(1), Some(i)]
            .into_iter()
            .flatten()
            .filter(|&j| j < beats.len())
            .min_by(|&a, &b| (beats[a] - h).abs().total_cmp(&(beats[b] - h).abs()));
        let Some(j) = near.filter(|&j| (beats[j] - h).abs() < ON_BEAT) else {
            continue;
        };
        let first = analysis.downbeats.partition_point(|&d| d <= j);
        let in_bar = first
            .checked_sub(1)
            .map_or(j, |k| j - analysis.downbeats[k]);
        if in_bar % 2 == 0 {
            kicks.push(beats[j]);
        } else {
            snares.push(beats[j]);
        }
    }
    kicks.dedup();
    snares.dedup();
    (kicks, snares)
}

/// The curves of the song's sections (each starting at one of `sections`,
/// the first at 0) over `mix` (its level, frame by frame): how loud each
/// is against the song's, 1 in the loudest (the choruses), and a new
/// place round the circle for each (by the golden ratio's steps; the
/// short way round is not looked for: a hue moving on is the point), each
/// eased [`SLOW`].
fn section_curves(mix: &[f32], sections: &[f64], length: f64) -> [Vec<f32>; 3] {
    let n = mix.len();
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
    let (mut energy, mut chorus, mut part) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
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
        energy[a..b].fill(e);
        chorus[a..b].fill(c);
        part[a..b].fill(crate::math::fract(k as f32 * 0.618_034));
    }
    [energy, chorus, part].map(|c| SLOW.over(&c))
}

/// The curves of the sung `lines` (start and end, seconds) over `n`
/// frames: 1 within one (eased [`SINGING`]), and 1 in a break
/// [`LINE_BREAK`] from any (eased [`SLOW`]; none in a song without lines).
fn line_curves(lines: &[(f64, f64)], n: usize) -> [Vec<f32>; 2] {
    let mut sung = vec![0.0; n];
    let mut near_line = vec![0.0; n];
    for &(a, b) in lines {
        let at = |t: f64| ((t.max(0.0) * FPS) as usize).min(n);
        sung[at(a)..at(b).max(at(a))].fill(1.0);
        let (from, to) = (at(a - LINE_BREAK), at(b + LINE_BREAK));
        near_line[from..to.max(from)].fill(1.0);
    }
    let away: Vec<f32> = (near_line.iter())
        .map(|&v| if lines.is_empty() { 0.0 } else { 1.0 - v })
        .collect();
    [SINGING.over(&sung), SLOW.over(&away)]
}

impl Curves {
    /// The curves of a song: its `analysis` (beats, bars, key), its
    /// `waveform` (the mix's and the stems' lanes and the melody; none:
    /// the beats alone), where its `sections` start (seconds) and its sung
    /// `lines` (start and end, seconds, in time).
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
        let lane = |l: &Lane| fit(shares(&l.rms), n);
        let stem_lanes = waveform.and_then(|w| w.stems.as_ref());
        let mix = waveform.map_or_else(|| vec![0.0; n], |w| lane(&w.mix));
        let level = ENVELOPE.over(&mix);
        let stems: [Vec<f32>; 4] = match stem_lanes {
            Some(s) => s.each_ref().map(|l| ENVELOPE.over(&lane(l))),
            None => std::array::from_fn(|_| level.clone()),
        };
        let drum_lane = stem_lanes.map(|s| &s[Stem::Drums as usize]);
        let (kicks, snares) = drum_hits(analysis, drum_lane.or(waveform.map(|w| &w.mix)));
        let bass_notes = stem_lanes.map_or(Vec::new(), |s| onsets(&s[Stem::Bass as usize].rms));
        // The drums' bright hits: how far their peaks stand over their
        // level, eased fast.
        let high = match drum_lane {
            Some(l) => {
                let crest: Vec<f32> = (l.peak.iter().zip(&l.rms))
                    .map(|(&p, &r)| (f32::from(p.saturating_sub(r)) / CREST).clamp(0.0, 1.0))
                    .collect();
                let v: Vec<f32> = (fit(crest, n).iter().zip(&lane(l)))
                    .map(|(c, g)| c * g)
                    .collect();
                HIGH.over(&v)
            }
            None => level.clone(),
        };
        let quiet = QUIET.over(&level.iter().map(|&v| 1.0 - v).collect::<Vec<f32>>());
        let [energy, chorus, part] = section_curves(&mix, sections, length);
        let [singing, instrumental] = line_curves(lines, n);
        let mut phrases: Vec<f64> = lines.iter().map(|&(a, _)| a).collect();
        phrases.sort_by(f64::total_cmp);
        let (minor, key_hue) = analysis.key.map_or((0.0, 0.0), |k| {
            (
                if k.minor() { 1.0 } else { 0.0 },
                f32::from(k.class % 12) / 12.0,
            )
        });
        let melody = waveform.and_then(|w| w.melody.clone()).unwrap_or_default();
        let mut hits: Vec<f64> = (kicks.iter().chain(&snares).chain(&bass_notes))
            .copied()
            .collect();
        hits.sort_by(f64::total_cmp);
        let stem = |d: Driver| {
            let s = Stem::ALL.into_iter().find(|&s| Driver::of_stem(s) == d);
            Source::Curve(s.map_or_else(Vec::new, |s| stems[s as usize].clone()))
        };
        let pulse = |at: &[f64], decay: f64| Source::Pulse {
            at: at.to_vec(),
            decay,
        };
        let sources = (Driver::ALL.iter())
            .map(|&d| match d {
                Driver::Beat
                | Driver::Downbeat
                | Driver::BeatPhase
                | Driver::BarPhase
                | Driver::Tempo => Source::Beat,
                Driver::Level => Source::Curve(level.clone()),
                Driver::Drums | Driver::Bass | Driver::Vocals | Driver::Other => stem(d),
                Driver::Kick => pulse(&kicks, KICK_DECAY),
                Driver::Snare => pulse(&snares, SNARE_DECAY),
                Driver::BassNote => pulse(&bass_notes, BASS_NOTE_DECAY),
                Driver::Phrase => pulse(&phrases, PHRASE_DECAY),
                Driver::Low => Source::Louder(Driver::Bass, Driver::Kick, LOW_KICK),
                Driver::Mid => Source::Mean(Driver::Vocals, Driver::Other),
                Driver::High => Source::Curve(high.clone()),
                Driver::Minor => Source::Const(minor),
                Driver::KeyHue => Source::Const(key_hue),
                Driver::Section => Source::Swell {
                    at: sections.to_vec(),
                    decay: SECTION_DECAY,
                },
                Driver::Energy => Source::Curve(energy.clone()),
                Driver::Chorus => Source::Curve(chorus.clone()),
                Driver::Part => Source::Curve(part.clone()),
                Driver::Singing => Source::Curve(singing.clone()),
                Driver::Instrumental => Source::Curve(instrumental.clone()),
                Driver::Quiet => Source::Curve(quiet.clone()),
                Driver::Melody => Source::Melody(melody.clone()),
                Driver::Progress => Source::Progress,
                Driver::Pitch | Driver::Accuracy | Driver::Tier | Driver::Streak => Source::Live,
            })
            .collect();
        Curves {
            analysis: analysis.clone(),
            sources,
            length,
            hits,
        }
    }

    /// The song's length, seconds.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// The beat at `t` seconds into the song, from its grid ([`beat`]).
    pub fn beat(&self, t: f64) -> Beat {
        beat(&self.analysis, t).0
    }

    /// The moments `d` pulses on (none: it pulses on none).
    fn moments(&self, d: Driver) -> &[f64] {
        match self.sources.get(d as usize) {
            Some(Source::Pulse { at, .. }) => at,
            _ => &[],
        }
    }

    /// The kick-like hits, seconds.
    pub fn kicks(&self) -> &[f64] {
        self.moments(Driver::Kick)
    }

    /// The snare-like hits, seconds.
    pub fn snares(&self) -> &[f64] {
        self.moments(Driver::Snare)
    }

    /// The moments sparks fly from, at or before `t`, the latest first, at
    /// most `n`.
    pub fn hits_before(&self, t: f64, n: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let i = self.hits.partition_point(|&h| h <= t);
        (0..i).rev().take(n).map(|j| (j, self.hits[j]))
    }

    /// Every driver at `t` seconds into the song, each from its source
    /// (the live ones none: the caller sets them; no song: quiet).
    pub fn at(&self, t: f64) -> Signals {
        let mut s = Signals::QUIET;
        if self.sources.is_empty() {
            return s;
        }
        let b = self.beat(t);
        for (&d, source) in Driver::ALL.iter().zip(&self.sources) {
            let v = match source {
                Source::Beat => b.signal(d).unwrap_or(0.0),
                Source::Curve(c) => curve_at(c, t),
                Source::Pulse { at, decay } => pulse_of(at, t, *decay),
                Source::Swell { at, decay } => swell(at, t, *decay),
                Source::Const(v) => *v,
                Source::Melody(m) => pitch(melody_at(m, t)),
                Source::Progress if self.length > 0.0 => (t / self.length).clamp(0.0, 1.0) as f32,
                Source::Progress | Source::Louder(..) | Source::Mean(..) | Source::Live => continue,
            };
            s.set(d, v);
        }
        // What is made from the others, from them.
        for (&d, source) in Driver::ALL.iter().zip(&self.sources) {
            match *source {
                Source::Louder(a, b, by) => s.set(d, s.get(a).max(by * s.get(b))),
                Source::Mean(a, b) => s.set(d, 0.5 * (s.get(a) + s.get(b))),
                _ => {}
            }
        }
        s
    }
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
