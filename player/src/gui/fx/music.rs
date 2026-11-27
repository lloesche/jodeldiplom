//! The music as the effects move with it (audio-reactive), read once a
//! frame by [`read`] ([`Music`]) from the frame's drivers (the song's
//! curves read then, the singing's set live over them: [`Signals`], the
//! same numbers the song's look moves by) and, for what the shaders need
//! that no driver gives, through the one reader of the music,
//! [`jodel_looks::drive`]: the beat as the grid has it (its phases and
//! pulses unfaded, the tempo in beats a minute, how sure the grid is, the
//! beats counted, the grid itself for landing hits), the melody's pitch
//! and the singer's as MIDI notes, and whether a line is being sung now on
//! the singing's clock. How loud the music is and each stem, how far the
//! song has gone and a section's swell are the drivers'. Where the beat
//! grid is unsure (or there is none) the beat's pulses fade to none as
//! drivers and what moves with them runs freely.
//!
//! A row's effect moves with a [`Driver`] by its [`super::Mod`]s
//! ([`Music::get`]); the shaders get the music as a small uniform block
//! ([`Music::uniform`], [`wgsl`]) to use in their own maths, the beats
//! counted too.
//!
//! The beat is the song's grid at the moment ([`Analysis::grid_at`]), the
//! one the sounds are played on; what is fired on it lands by the grid's
//! one rule ([`Music::land`]: [`Grid::land`], with the sung lines from
//! then on), so what is seen lands with what is heard.

use jodel_looks::drive::{self, Driver, Signals};
use jodel_rhythm::grid::{Grid, When};
use jodel_rhythm::stems::Stem;
use jodel_rhythm::{Analysis, Region};

use crate::core::singing::Singing;
use crate::gui::backdrop::music::Song;

/// The tempo where there is no grid, beats a minute.
const FREE_TEMPO: f64 = jodel_rhythm::grid::FREE_TEMPO;
/// Beats in a bar where the grid has no bars there.
const BAR: u32 = jodel_rhythm::grid::BAR;
/// The sung lines a frame's music knows, from the one sung now (or next)
/// on: what a hit landing on a bar's first beat keeps clear of.
pub const LINES: usize = 4;

/// The music at a frame: the phases and pulses as the grid has them
/// (unfaded), the tempo in beats a minute and the pitches as MIDI notes
/// (0: none), for the shaders; and every driver ([`Signals`]) for what
/// moves by one.
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
    /// Every driver.
    pub signals: Signals,
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
        signals: Signals::QUIET,
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
        let beat = m.beat_from(&a, t, clock);
        m.signals.set_beat(&beat);
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
    /// clock at `clock` ([`drive::beat`]): where in the beat and the bar,
    /// the pulses since, the tempo, the beats in a bar and counted, how
    /// sure, and the grid on the singing's clock; the beat read.
    fn beat_from(&mut self, a: &Analysis, now: f64, clock: f64) -> drive::Beat {
        let (b, grid) = drive::beat(a, now);
        self.beat_phase = b.phase;
        self.bar_phase = b.bar_phase;
        self.beat = b.beat;
        self.downbeat = b.downbeat;
        self.tempo = b.tempo;
        self.sure = b.sure;
        self.beats = b.beats;
        self.bar_beats = b.bar_beats;
        self.grid = Some(grid.on(now, clock, 1.0));
        b
    }

    /// A kick for an effect to move by on the beat: `on_beat` of each
    /// beat's pulse and `on_one` of each bar's first's, as sure as the grid
    /// is (none where it is unsure).
    pub fn kick(&self, on_beat: f32, on_one: f32) -> f32 {
        self.sure * (on_beat * self.beat + on_one * self.downbeat)
    }

    /// How long one beat is, in seconds, at the tempo.
    pub fn beat_length(&self) -> f32 {
        60.0 / self.tempo.max(1.0)
    }

    /// `d`'s value this frame, for a [`super::Mod`] (the beat's as sure as
    /// the grid is: none where it is unsure; a pitch as a share of
    /// [`drive::PITCHES`]).
    pub fn get(&self, d: Driver) -> f32 {
        self.signals.get(d)
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
/// singing's clock, the song's length (where known), and every driver then
/// (the song's curves read at the time, the singing's set live over them:
/// [`crate::core::karaoke::live`]).
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
    /// Every driver.
    pub signals: Signals,
}

/// The music as `heard` has it.
pub fn read(heard: &Heard) -> Music {
    let mut m = Music::NEUTRAL;
    let singing = heard.singing;
    let s = heard.signals;
    m.signals = s;
    m.level = s.get(Driver::Level);
    m.stems = Stem::ALL.map(|stem| s.get(Driver::of_stem(stem)));
    m.section = s.get(Driver::Section);
    m.progress = s.get(Driver::Progress);
    if let Some(song) = heard.song {
        m.beat_from(song.analysis(), heard.now, heard.clock);
        m.melody = (song.waveform())
            .and_then(|w| w.melody.as_ref())
            .map_or(0.0, |mel| drive::melody_at(mel, heard.now));
    }
    m.sung = crate::core::karaoke::sung_pitch(singing, heard.clock);
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

#[cfg(test)]
mod tests;
