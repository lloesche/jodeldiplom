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
//! [`wgsl`]) to use in their own maths, the beats counted too.
//!
//! The beat is the song's grid at the moment ([`Analysis::grid_at`]), the
//! one the sounds are played on; what is fired on it lands by the grid's
//! one rule ([`Music::land`]: [`Grid::land`], with the sung lines from
//! then on), so what is seen lands with what is heard.

use jodel_rhythm::grid::{Grid, When};
use jodel_rhythm::stems::Stem;
use jodel_rhythm::{Analysis, Region};

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
const FREE_TEMPO: f64 = jodel_rhythm::grid::FREE_TEMPO;
/// Beats in a bar where the grid has no bars there.
const BAR: u32 = jodel_rhythm::grid::BAR;
/// How long after the singer's last pitch it is still the pitch sung.
const SUNG_HOLD: f64 = 0.1;
/// The pitches a pitch signal spans, MIDI notes: C2 to C6.
const PITCHES: (f32, f32) = (36.0, 84.0);
/// The sung lines a frame's music knows, from the one sung now (or next)
/// on: what a hit landing on a bar's first beat keeps clear of.
pub const LINES: usize = 4;

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
    /// The beats counted from the grid's first (the whole ones and the
    /// share of the next): which beat it is.
    pub beats: f32,
    /// The beats in the bar.
    pub bar_beats: f32,
    /// The song's grid about the moment, on the singing's clock (none: no
    /// music, nothing to wait for).
    pub grid: Option<Grid>,
    /// The sung lines from the one sung now (or next) on, each its start
    /// and end on the singing's clock, earliest first.
    pub lines: [Option<(f64, f64)>; LINES],
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
        beats: 0.0,
        bar_beats: BAR as f32,
        grid: None,
        lines: [None; LINES],
    };

    /// The music of a steady grid `beats` beats in (a beat every `60 /
    /// tempo` seconds from 0, each `bar_beats`-th from the first a bar's
    /// first), its analysis `confidence` sure: its beat as [`read`] has it
    /// from such a song ([`steady_song`]), heard at `clock` on the
    /// singing's clock, nothing else heard. What a preview or a test plays
    /// the effects to.
    pub fn steady(beats: f64, tempo: f64, bar_beats: u32, confidence: f64, clock: f64) -> Music {
        let t = beats * 60.0 / tempo;
        let a = steady_song(beats.max(0.0) as usize + 64, tempo, bar_beats, confidence);
        let mut m = Music::NEUTRAL;
        m.beat_from(&a, t, clock);
        m
    }

    /// Where a hit fired at `at` (the singing's clock) lands, as `when`
    /// says, by the grid's one rule ([`Grid::land`], with the lines sung
    /// from then on): as the sounds fired with it land. With no music, at
    /// once.
    pub fn land(&self, at: f64, when: When) -> f64 {
        let Some(grid) = self.grid else {
            return at;
        };
        let mut sung = [(0.0, 0.0); LINES];
        let mut n = 0;
        for &line in self.lines.iter().flatten() {
            sung[n] = line;
            n += 1;
        }
        grid.land(at, when, &sung[..n])
    }

    /// Whether the grid is sure enough for a hit to wait for it
    /// ([`jodel_rhythm::grid::SURE`]).
    pub fn is_sure(&self) -> bool {
        self.grid.is_some_and(|g| g.is_sure())
    }

    /// Its beat from `a`'s grid `now` seconds into the song, the singing's
    /// clock at `clock` ([`Analysis::grid_at`]): where in the beat and the
    /// bar, the pulses since, the tempo, the beats in a bar and counted
    /// (from the analysis's beat there), how sure (faded in between
    /// [`SURE`]'s bounds), and the grid on the singing's clock.
    fn beat_from(&mut self, a: &Analysis, now: f64, clock: f64) {
        let grid = a.grid_at(now);
        let len = grid.beat_len();
        let bar_len = f64::from(grid.bar()) * len;
        let since_beat = (len - (grid.next_step(now, 1).0 - now)).rem_euclid(len);
        let since_bar = (bar_len - (grid.next_downbeat(now) - now)).rem_euclid(bar_len);
        let i = a.beat_at(now).unwrap_or(0);
        let from = a.beats.get(i).copied().unwrap_or(now - since_beat);
        let confidence = a.region_at(now).map_or(0.0, |r| r.confidence as f32);
        self.beat_phase = (since_beat / len) as f32;
        self.bar_phase = (since_bar / bar_len) as f32;
        self.beat = pulse(since_beat, BEAT_DECAY);
        self.downbeat = pulse(since_bar, BAR_DECAY);
        self.tempo = grid.tempo() as f32;
        self.sure = smoothstep(SURE.0, SURE.1, confidence);
        self.beats = (i as f64 + (now - from) / len) as f32;
        self.bar_beats = grid.bar() as f32;
        self.grid = Some(grid.on(now, clock, 1.0));
    }

    /// How long one beat is, in seconds, at the tempo.
    pub fn beat_length(&self) -> f32 {
        60.0 / self.tempo.max(1.0)
    }

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

    /// As the shaders take it ([`wgsl`]'s `Music`): the beat (its phase,
    /// the bar's, the beat's pulse, the bar's), the tempo (beats a minute,
    /// how sure, the level, the progress), the stems, the voice (the
    /// melody's pitch, the singer's, singing, the section's swell), the
    /// count (the beats, the beats in a bar).
    pub fn uniform(&self) -> Uniform {
        [
            [self.beat_phase, self.bar_phase, self.beat, self.downbeat],
            [self.tempo, self.sure, self.level, self.progress],
            self.stems,
            [self.melody, self.sung, self.singing, self.section],
            [self.beats, self.bar_beats, 0.0, 0.0],
        ]
    }

    /// The music `u` holds ([`Music::uniform`]'s, read back): what an
    /// effect's maths on the processor takes as its shader takes the
    /// uniform.
    pub fn from_uniform(u: &Uniform) -> Music {
        let [[beat_phase, bar_phase, beat, downbeat], [tempo, sure, level, progress], stems, [melody, sung, singing, section], [beats, bar_beats, _, _]] =
            *u;
        Music {
            beat_phase,
            bar_phase,
            beat,
            downbeat,
            tempo,
            sure,
            level,
            progress,
            stems,
            melody,
            sung,
            singing,
            section,
            beats,
            bar_beats,
            ..Music::NEUTRAL
        }
    }
}

/// The music as the shaders take it ([`Music::uniform`]).
pub type Uniform = [[f32; 4]; 5];

/// [`Music::uniform`] as the shaders declare it (each binds it as
/// `music`, and the effects' maths take it whole), with each stem's place
/// among the stems (`STEM_` and its name).
pub fn wgsl() -> String {
    let stems: String = (Stem::ALL.iter().enumerate())
        .map(|(i, s)| format!("const STEM_{}: u32 = {i}u;\n", s.id().to_uppercase()))
        .collect();
    format!("{STRUCT}{stems}")
}

const STRUCT: &str = r"
// The music this frame (`fx::music::Music`): the beat (its phase, the
// bar's, the beat's pulse, the bar's), the tempo (beats a minute, how sure
// the grid is, the level, the song's progress), the stems' levels (each at
// its `STEM_` place), the voice (the melody's and the singer's pitch as
// MIDI notes, 0 none; 1 while a line is sung; a section's swell), the
// count (the beats from the grid's first, the beats in a bar).
struct Music {
    beat: vec4<f32>,
    tempo: vec4<f32>,
    stems: vec4<f32>,
    voice: vec4<f32>,
    count: vec4<f32>,
};
";

/// A pulse `since` seconds after it, falling away by `decay`.
fn pulse(since: f64, decay: f64) -> f32 {
    (-since / decay).exp() as f32
}

/// A song of `n` beats, one every `60 / tempo` seconds from 0, each
/// `bar_beats`-th from the first a bar's first, its analysis `confidence`
/// sure throughout ([`Music::steady`]'s).
pub fn steady_song(n: usize, tempo: f64, bar_beats: u32, confidence: f64) -> Analysis {
    let beat = 60.0 / tempo;
    let beats: Vec<f64> = (0..n).map(|k| k as f64 * beat).collect();
    Analysis {
        downbeats: (0..n).step_by(bar_beats.max(1) as usize).collect(),
        local_bpm: vec![tempo; n],
        regions: vec![Region {
            start: 0.0,
            end: n as f64 * beat,
            confidence,
        }],
        beats,
        ..Analysis::default()
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
        m.beat_from(song.analysis(), heard.now, heard.clock);
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
    let spans = (singing.notes.lines.iter())
        .filter_map(|l| Some((l.notes.first()?.start, l.notes.last()?.end)));
    let sung_now = (spans.clone()).any(|(start, end)| start <= heard.clock && end >= heard.clock);
    m.singing = f32::from(u8::from(sung_now));
    let ahead = spans.filter(|&(_, end)| end > heard.clock);
    for (slot, line) in m.lines.iter_mut().zip(ahead) {
        *slot = Some(line);
    }
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
