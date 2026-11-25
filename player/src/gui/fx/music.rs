//! The music as the effects move with it, one clock for them all
//! ([`Music`], worked out once a frame by [`Clock::music`]): where it is
//! in the beat and the bar, a pulse on each beat and each bar's first
//! (falling away after it), the tempo, how loud the music is and each of
//! its stems, the melody's pitch and the singer's, how far the song has
//! gone, whether a line is being sung or it is a gap, and a swell as a
//! section starts ([`read`]). All of it read from what the song heard
//! already holds ([`crate::gui::backdrop::music::Song`], which the
//! backdrop shares): its beats and bars and how sure the analysis is of
//! them, region by region ([`jodel_rhythm::Analysis`]), its waveform's
//! lanes, sections and melody; and the singing. Where the beat grid is
//! unsure (or there is none) the beat's signals fade to none
//! ([`Music::get`]) and what moves with them runs freely.
//!
//! A row's effect moves with a [`Signal`] by its [`super::Mod`]s; the
//! shaders get it all as a small uniform block ([`Music::uniform`],
//! [`WGSL`]) to use in their own maths.

use jodel_rhythm::stems::Stem;
use jodel_rhythm::Analysis;

use crate::core::singing::Singing;
use crate::gui::backdrop::music::Song;
use crate::gui::rainbow::smoothstep;

/// How fast a beat's pulse falls away: to about a third in this many
/// seconds.
const BEAT_DECAY: f64 = 0.1;
/// How fast a bar's pulse falls away.
const BAR_DECAY: f64 = 0.25;
/// How fast a section's swell falls away.
const SECTION_DECAY: f64 = 1.5;
/// How sure a region of the grid is for its beats to show not at all, and
/// wholly.
const SURE: (f32, f32) = (0.35, 0.65);
/// The tempo where there is no grid, beats a minute.
const FREE_TEMPO: f64 = 120.0;
/// Beats in a bar where the grid has no bars there.
const BAR: f64 = 4.0;
/// How long after the singer's last pitch it is still the pitch sung.
const SUNG_HOLD: f64 = 0.1;
/// The pitches a pitch signal spans, MIDI notes: C2 to C6.
const PITCHES: (f32, f32) = (36.0, 84.0);

/// What an effect can move with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// Where it is in the beat, 0 to 1.
    BeatPhase,
    /// Where it is in the bar, 0 to 1.
    BarPhase,
    /// A pulse on each beat, 1 falling to none.
    Beat,
    /// A pulse on each bar's first beat.
    Downbeat,
    /// The tempo, 1 at 120 beats a minute.
    Tempo,
    /// How loud the music is, 0 to 1.
    Level,
    /// How loud a stem is, 0 to 1.
    Stem(Stem),
    /// The melody's pitch, 0 (C2 and below, or none) to 1 (C6).
    Melody,
    /// The singer's pitch, as the melody's.
    Sung,
    /// How far the song has gone, 0 to 1.
    Progress,
    /// 1 while a line is being sung, none in a gap.
    Singing,
    /// A swell as a section starts, falling away.
    Section,
    /// How sure the beat grid is here, 0 to 1.
    Sure,
}

/// The music at a frame: as [`Signal`] says, the phases and pulses as the
/// grid has them (unfaded), the tempo in beats a minute and the pitches as
/// MIDI notes (0: none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Music {
    /// Where it is in the beat.
    pub beat_phase: f32,
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
    /// How loud the music is.
    pub level: f32,
    /// How far the song has gone.
    pub progress: f32,
    /// The stems' levels, in [`Stem::ALL`]'s order.
    pub stems: [f32; 4],
    /// The melody's pitch, a MIDI note (0: none).
    pub melody: f32,
    /// The singer's pitch, a MIDI note (0: none).
    pub sung: f32,
    /// Whether a line is being sung (1) or it is a gap (0).
    pub singing: f32,
    /// A section's swell.
    pub section: f32,
}

impl Default for Music {
    fn default() -> Music {
        Music::NEUTRAL
    }
}

impl Music {
    /// No music: still, quiet, unsure, at the free tempo.
    pub const NEUTRAL: Music = Music {
        beat_phase: 0.0,
        bar_phase: 0.0,
        beat: 0.0,
        downbeat: 0.0,
        tempo: FREE_TEMPO as f32,
        sure: 0.0,
        level: 0.0,
        progress: 0.0,
        stems: [0.0; 4],
        melody: 0.0,
        sung: 0.0,
        singing: 0.0,
        section: 0.0,
    };

    /// `s`'s value, for a [`super::Mod`]: the beat's signals as sure as
    /// the grid is (none where it is unsure), a pitch as a share of
    /// [`PITCHES`].
    pub fn get(&self, s: Signal) -> f32 {
        let pitch = |midi: f32| {
            if midi <= 0.0 {
                0.0
            } else {
                ((midi - PITCHES.0) / (PITCHES.1 - PITCHES.0)).clamp(0.0, 1.0)
            }
        };
        match s {
            Signal::BeatPhase => self.beat_phase * self.sure,
            Signal::BarPhase => self.bar_phase * self.sure,
            Signal::Beat => self.beat * self.sure,
            Signal::Downbeat => self.downbeat * self.sure,
            Signal::Tempo => self.tempo / FREE_TEMPO as f32,
            Signal::Level => self.level,
            Signal::Stem(s) => self.stems[s as usize],
            Signal::Melody => pitch(self.melody),
            Signal::Sung => pitch(self.sung),
            Signal::Progress => self.progress,
            Signal::Singing => self.singing,
            Signal::Section => self.section,
            Signal::Sure => self.sure,
        }
    }

    /// As the shaders take it ([`WGSL`]'s `Music`): the beat (its phase,
    /// the bar's, the beat's pulse, the bar's), the tempo (beats a minute,
    /// how sure, the level, the progress), the stems, the voice (the
    /// melody's pitch, the singer's, singing, the section's swell).
    pub fn uniform(&self) -> [[f32; 4]; 4] {
        [
            [self.beat_phase, self.bar_phase, self.beat, self.downbeat],
            [self.tempo, self.sure, self.level, self.progress],
            self.stems,
            [self.melody, self.sung, self.singing, self.section],
        ]
    }
}

/// [`Music::uniform`] as the shaders declare it (each binds it as
/// `music`).
pub const WGSL: &str = r"
// The music this frame (`fx::music::Music`): the beat (its phase, the
// bar's, the beat's pulse, the bar's), the tempo (beats a minute, how sure
// the grid is, the level, the song's progress), the stems' levels (bass,
// drums, other, vocals), the voice (the melody's and the singer's pitch as
// MIDI notes, 0 none; 1 while a line is sung; a section's swell).
struct Music {
    beat: vec4<f32>,
    tempo: vec4<f32>,
    stems: vec4<f32>,
    voice: vec4<f32>,
};
";

/// Where the beat is at a moment of the song: its phase, the bar's, how
/// long since the beat and the bar began, the tempo, how sure the grid is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatAt {
    /// Where in the beat, 0 to 1.
    pub phase: f64,
    /// Where in the bar, 0 to 1.
    pub bar: f64,
    /// Seconds since the beat.
    pub since_beat: f64,
    /// Seconds since the bar began (none before the first).
    pub since_bar: Option<f64>,
    /// Beats a minute.
    pub tempo: f64,
    /// How sure the grid is, 0 to 1.
    pub sure: f32,
}

/// Where the beat is `t` seconds into the song by `a`'s grid: between two
/// of its beats, as far between them as `t` is ([`Analysis::phase_at`]),
/// at its local tempo, the bar from its last first beat, as sure as its
/// region; before the first beat or from the last on, running freely at
/// the song's tempo from the nearest (and not sure at all).
pub fn beat_at(a: &Analysis, t: f64) -> BeatAt {
    let (Some(beat), Some(phase)) = (a.beat_at(t), a.phase_at(t)) else {
        let tempo = a.summary.map_or(FREE_TEMPO, |s| s.bpm).max(1.0);
        let from = a.beat_at(t).map_or(a.beats.first(), |i| a.beats.get(i));
        let beats = (t - from.copied().unwrap_or(0.0)) * tempo / 60.0;
        let phase = beats.rem_euclid(1.0);
        return BeatAt {
            phase,
            bar: (beats / BAR).rem_euclid(1.0),
            since_beat: phase * 60.0 / tempo,
            since_bar: None,
            tempo,
            sure: 0.0,
        };
    };
    let d = a.downbeats.partition_point(|&j| j <= beat);
    let (bar, since_bar) = match d.checked_sub(1).map(|k| a.downbeats[k]) {
        Some(first) => {
            let len = (a.downbeats.get(d)).map_or(BAR, |&next| (next - first) as f64);
            let into = (beat - first) as f64 + phase;
            ((into / len.max(1.0)).min(1.0), Some(t - a.beats[first]))
        }
        None => (((beat as f64 + phase) / BAR).fract(), None),
    };
    let confidence = a.region_at(t).map_or(0.0, |r| r.confidence as f32);
    BeatAt {
        phase,
        bar,
        since_beat: t - a.beats[beat],
        since_bar,
        tempo: a.bpm_at(t).unwrap_or(FREE_TEMPO),
        sure: smoothstep(SURE.0, SURE.1, confidence),
    }
}

/// What the music is read from at a frame: the song heard (none: not
/// analysed yet), the singing, the song's time (as heard) and the
/// singing's clock, and the song's length (where known).
#[derive(Clone, Copy)]
pub struct Heard<'a> {
    /// The song heard.
    pub song: Option<&'a Song>,
    /// The singing.
    pub singing: &'a Singing,
    /// The song's time, seconds.
    pub now: f64,
    /// The singing's clock.
    pub clock: f64,
    /// The song's length, seconds.
    pub length: Option<f64>,
}

/// The music as `heard` has it.
pub fn read(heard: &Heard) -> Music {
    let mut m = Music::NEUTRAL;
    let singing = heard.singing;
    if let Some(song) = heard.song {
        let beat = beat_at(song.analysis(), heard.now);
        m.beat_phase = beat.phase as f32;
        m.bar_phase = beat.bar as f32;
        m.beat = (-beat.since_beat / BEAT_DECAY).exp() as f32;
        m.downbeat = (beat.since_bar).map_or(0.0, |s| (-s / BAR_DECAY).exp() as f32);
        m.tempo = beat.tempo as f32;
        m.sure = beat.sure;
        if let Some((level, stems)) = song.loudness(heard.now) {
            m.level = level;
            m.stems = stems.unwrap_or([level; 4]);
        }
        let i = song.sections.partition_point(|&s| s <= heard.now);
        if let Some(&s) = i.checked_sub(1).and_then(|i| song.sections.get(i)) {
            m.section = (-(heard.now - s) / SECTION_DECAY).exp() as f32;
        }
        m.melody = (song.waveform())
            .and_then(|w| w.melody.as_ref())
            .map_or(0.0, |mel| melody_at(mel, heard.now));
    }
    m.sung = (singing.heard.iter().rev())
        .find(|h| h.t <= heard.clock)
        .filter(|h| heard.clock - h.t <= SUNG_HOLD)
        .map_or(0.0, |h| (h.cents / 100.0) as f32);
    m.progress = (heard.length)
        .filter(|&l| l > 0.0)
        .map_or(0.0, |l| (heard.now / l).clamp(0.0, 1.0) as f32);
    let sung_now = singing.notes.lines.iter().any(|l| {
        l.notes.first().is_some_and(|n| n.start <= heard.clock)
            && l.notes.last().is_some_and(|n| n.end >= heard.clock)
    });
    m.singing = f32::from(u8::from(sung_now));
    m
}
/// The melody's pitch at `t` seconds, a MIDI note (0 where none is sure).
fn melody_at(m: &jodel_rhythm::waveform::Melody, t: f64) -> f32 {
    let i = (t * 100.0).floor();
    if i < 0.0 {
        return 0.0;
    }
    let i = i as usize;
    match (m.cents.get(i), m.sure.get(i)) {
        (Some(&c), Some(&s)) if c > 0 && s >= 128 => f32::from(c) / 100.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests;
