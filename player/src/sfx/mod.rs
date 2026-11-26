//! The karaoke game's sound effects, played through the desk's effects
//! input on its Effects strip ([`jodel_output::mixer::Effects`]), all of
//! them live by our synth ([`conductor`]): the game's own in the song's
//! tempo and key, each on the frame of the song where its moment is seen
//! ([`moments`]); the result screen's after it ([`reveal`]).
//!
//! The conductor is made for the device's rate (here, off the mixer
//! thread) and its synth handed to the effects input
//! ([`jodel_output::mixer::Effects::live`]), heard there at the level its
//! own bus sets ([`conductor::LEVEL`], following the sung song's loudness
//! as the karaoke plays it: [`conductor::calibration`]); each frame drawn
//! it is told the music clock (the song's beat grid about where it is
//! heard, and the next sung lines, [`conductor::Clock`]) and, as a song
//! comes, its key (the analysis's, shifted as the song is heard). A
//! moment's time is the singing's own (the song as heard less how late
//! the voice comes: [`crate::core::lag`]), so it is seen when the song as
//! heard is that much later: its event is planned on that frame of the
//! playing head's output ([`crate::core::Playing::frame_at`]), and the
//! desk puts it on the synth's frames as it mixes. One with no frame to time it by yet (no
//! position told since a seek or a jump) waits here until there is, or it
//! is too late. A moment for now (the streak lost, a phrase's tier) is
//! planned on the frame of the song the desk mixes next, or with none (the
//! song over: its last phrase's tier is decided then, as the next track
//! comes) on the desk's own frames, by a clock of no grid (at once). The
//! stream going back (a seek, the song sung again), the song's anchor
//! forgotten (a jump takes back frames and gives their numbers to other
//! music), the song paused or left call off what is planned and not heard
//! yet, and let go of what is held.
//!
//! With the Effects strip down (−∞ or muted) nothing is played.
//!
//! A song's result revealed after it ([`reveal`]) sounds its beats at once,
//! each on the frame that reaches it, its paper and its music alike,
//! planned on the desk's own frames (the song is over then and its output
//! closed: the desk's device plays all along, so they sound all the same).
//! The game's live sounds go with its bubbles: with the notes hidden none
//! plays; the diploma shows all the same (the game scored the song), and
//! sounds.

pub mod conductor;
pub mod moments;
pub mod reveal;

use jodel_output::mixer::{Effects, Fader, LiveCommands, Timebase};
use jodel_playback::PlayState;

pub use moments::{Cued, Frame, Moments};

use crate::core::desk::StripKind;
use crate::core::{HeadFrame, State};
use conductor::{Clock, Conductor, Event};

// The queue to the desk's synth takes a frame's worth of the most an
// event sends, several times over (a ripple of pops, a rung and a phrase
// on one frame).
const _: () = assert!(jodel_output::mixer::LIVE_COMMANDS >= 8 * conductor::MOST_COMMANDS);

/// The desk as the effects play on it.
pub struct Desk<'a> {
    /// Its effects input.
    pub effects: &'a Effects,
    /// The engine's heads' input faders, by head id: the frames a moment
    /// is planned on are its head's.
    pub heads: [&'a Fader; 2],
    /// The mixer's clock (its rate 0: the device closed).
    pub timebase: Timebase,
}

/// The most events with a time kept waiting for a position of the song to
/// time them by.
const PENDING: usize = 64;

/// What the conductor was told of the song: which it is, its key as the
/// analysis found it and how far it is shifted, and whether its grid was
/// known (with none: no key and the free clock, so nothing of the song
/// before is played on).
#[derive(Clone, Debug, PartialEq)]
struct Told {
    song: Option<std::path::PathBuf>,
    key: Option<jodel_rhythm::key::Key>,
    shift: i8,
    grid: bool,
}

/// The game's sounds followed frame by frame.
#[derive(Default)]
pub struct Sfx {
    moments: Moments,
    /// The conductor at the device's rate, its synth on the desk.
    conductor: Option<Conductor>,
    /// Whether it planned on a song's frames since it was last called off,
    /// and what it was told of the song.
    on_song: bool,
    told: Option<Told>,
    /// The song's anchors forgotten as last seen
    /// ([`crate::core::Playing::anchors_forgotten`]).
    anchors: u64,
    /// Events with a time (the singing's) and no frame of the song's to
    /// time them by yet (no position told since a seek or a jump).
    pending: Vec<(Event, f64)>,
    /// The result's reveal as last heard.
    reveal: reveal::RevealSounds,
}

impl Sfx {
    /// A frame drawn: the conductor for the device's rate, then the game's
    /// new moments played on `desk`, then the result's reveal's (after what
    /// the moments call off, so it never calls them off).
    pub fn frame(&mut self, state: &State, desk: Desk<'_>) {
        self.conductor_at(&desk);
        self.loudness(state, &desk);
        self.moments(state, &desk);
        self.reveal(state, &desk);
    }

    /// The conductor at the device's rate, its synth sent to the desk, made
    /// again when the device opens at another (none at a rate the synth
    /// takes not: silent there).
    fn conductor_at(&mut self, desk: &Desk<'_>) {
        let rate = desk.timebase.rate;
        let at = self.conductor.as_ref().map(Conductor::rate);
        let takes = (jodel_synth::dsp::Rate::MIN..=jodel_synth::dsp::Rate::MAX).contains(&rate);
        if rate == 0 || at == Some(rate) || !takes {
            return;
        }
        let (conductor, synth) = Conductor::new(rate);
        desk.effects.live(synth);
        self.conductor = Some(conductor);
        self.on_song = false;
        self.told = None;
    }

    /// The sung song's loudness told the conductor, so its synth's level
    /// follows it ([`conductor::calibration`]): as measured, as at the
    /// reference until it is; with no song sung the last one's stays (its
    /// result sounds after it).
    fn loudness(&mut self, state: &State, desk: &Desk<'_>) {
        let Some(song) = state.karaoke.singing.song() else {
            return;
        };
        let lufs = state.karaoke.loudness.get(song).copied().flatten();
        let told = self.conductor.as_ref().map(Conductor::trim);
        if told.is_none_or(|t| t == conductor::calibration::trim(lufs)) {
            return;
        }
        if let Some((c, mut out)) = self.conduct(desk, None) {
            c.set_loudness(&mut out, lufs);
        }
    }

    /// The conductor, and where its commands go: on the frames of `fader`'s
    /// output, or with none the desk's.
    fn conduct<'a>(
        &'a mut self,
        desk: &'a Desk<'_>,
        fader: Option<&'a Fader>,
    ) -> Option<(&'a mut Conductor, LiveCommands<'a>)> {
        let c = self.conductor.as_mut()?;
        Some((c, desk.effects.commands(fader)))
    }

    /// `event` now, on the desk's frames (and their clock, of no grid).
    fn on_desk(&mut self, desk: &Desk<'_>, event: Event) {
        let at = desk.timebase.frame;
        if let Some((c, mut out)) = self.conduct(desk, None) {
            c.set_clock(&mut out, Clock::free(c.rate()));
            c.event(&mut out, at, event);
        }
    }

    /// What is planned on the song's frames called off (they are of
    /// another place now, or the song is left or halted).
    fn call_off(&mut self, desk: &Desk<'_>) {
        if !self.on_song {
            return;
        }
        self.on_song = false;
        self.pending.clear();
        let at = desk.timebase.frame;
        if let Some((c, mut out)) = self.conduct(desk, None) {
            c.cancel(&mut out, at);
        }
    }

    /// The game's new moments played on `desk`.
    fn moments(&mut self, state: &State, desk: &Desk<'_>) {
        let karaoke = &state.karaoke;
        // The song the singing is of, whatever plays now: its last
        // phrase's tier is decided as the next track comes.
        let sung = karaoke.singing.song().is_some() && !karaoke.singing.notes.lines.is_empty();
        // What is planned on the song's frames is called off when the game
        // is left (off, the mode left, nothing to sing, or its bubbles not
        // shown: the notes hidden), the song moved (a seek or a jump: its
        // anchor forgotten, its frames of another place now), halted
        // (paused or stopped, by hand or at its end), or its singing went
        // back (sought back, sung again).
        let config = &state.config;
        let shown = config.karaoke_game && config.karaoke_notes;
        let left = !(sung && crate::core::karaoke::in_mode(state) && shown);
        if left {
            self.moments.reset();
            self.call_off(desk);
            return;
        }
        let moved = state.playing.anchors_forgotten != self.anchors;
        self.anchors = state.playing.anchors_forgotten;
        let halted = state.playing.state != PlayState::Playing;
        let clock = karaoke.lag.sung_clock(state.heard());
        let frame = self.moments.frame(&karaoke.singing, clock);
        if moved || halted || frame.sought {
            self.call_off(desk);
        }
        if state.desk.setting(StripKind::Effects).gain() == 0.0 {
            self.pending.clear();
            return;
        }
        // The song's clock while it plays (not while a result shows: the
        // conductor is on the desk's frames then).
        let playing = state.playing.state == PlayState::Playing && !self.reveal.showing();
        let now = if playing {
            self.song_clock(state, desk)
        } else {
            None
        };
        let waited = std::mem::take(&mut self.pending);
        let waited = waited.into_iter().map(|(event, at)| Cued {
            event,
            at: Some(at),
        });
        for Cued { event, at } in waited.chain(frame.cues) {
            if at.is_some_and(|at| at < clock - moments::STALE) {
                continue;
            }
            // Heard that much later than the singing's time; for now, on
            // the frame the desk mixes next.
            let on = match at {
                Some(at) => state.playing.frame_at(at + karaoke.lag.late),
                None => now,
            };
            match (
                on.and_then(|f| Some((*desk.heads.get(f.head)?, f.frame))),
                at,
            ) {
                (Some((fader, f)), _) => {
                    self.on_song = true;
                    if let Some((c, mut out)) = self.conduct(desk, Some(fader)) {
                        c.event(&mut out, f, event);
                    }
                }
                (None, None) => self.on_desk(desk, event),
                (None, Some(at)) if self.pending.len() < PENDING => self.pending.push((event, at)),
                (None, Some(_)) => {}
            }
        }
    }

    /// The music clock where the song is heard now told the conductor (and
    /// its key as a song comes; with no grid known for it yet, no key and
    /// the free clock, once): the frame of the playing head the desk mixes
    /// next, if there is one to time it by.
    fn song_clock(&mut self, state: &State, desk: &Desk<'_>) -> Option<HeadFrame> {
        let analysis = crate::core::karaoke::rhythm_of(state).map(|r| &*r.analysis);
        let told = Told {
            song: state
                .karaoke
                .singing
                .song()
                .map(std::path::Path::to_path_buf),
            key: analysis.and_then(|a| a.key),
            shift: state.playing.key,
            grid: analysis.is_some(),
        };
        if self.told.as_ref() != Some(&told) {
            if let Some((c, mut out)) = self.conduct(desk, None) {
                c.set_key(told.key.as_ref(), told.shift);
                if !told.grid {
                    c.set_clock(&mut out, Clock::free(c.rate()));
                }
            }
            self.told = Some(told);
        }
        let a = analysis?;
        let t = state.heard();
        let heard = state.playing.frame_at(t)?;
        let speed = state.playing.speed?;
        let fader = *desk.heads.get(heard.head)?;
        let late = state.karaoke.lag.late;
        let mut lines = [None; Clock::LINES];
        let sung = (state.karaoke.singing.notes.lines.iter())
            .filter_map(|l| Some((l.notes.first()?.start, l.notes.last()?.end)))
            .filter(|&(_, end)| end + late > t)
            .filter_map(|(start, end)| {
                let frame = |s: f64| Some(state.playing.frame_at(s + late)?.frame);
                Some((frame(start.max(0.0))?, frame(end)?))
            });
        for (slot, line) in lines.iter_mut().zip(sung) {
            *slot = Some(line);
        }
        let clock = Clock::from_analysis(a, t, heard.frame, speed, lines);
        if let Some((c, mut out)) = self.conduct(desk, Some(fader)) {
            c.set_clock(&mut out, clock);
        }
        // The desk mixes ahead of what is heard by its device's buffer.
        let ahead = desk.timebase.frame.saturating_sub(desk.timebase.heard);
        Some(HeadFrame {
            frame: heard.frame + ahead,
            ..heard
        })
    }

    /// The result's reveal's sounds of this frame played on `desk`, on the
    /// desk's frames (on a clock of their own: every hit as the reveal
    /// draws it).
    fn reveal(&mut self, state: &State, desk: &Desk<'_>) {
        let now = crate::core::karaoke::result_now(state).map(|(s, t)| (&**s, t));
        let heard = self.reveal.frame(now);
        if state.desk.setting(StripKind::Effects).gain() == 0.0 {
            return;
        }
        for d in heard {
            self.on_desk(desk, Event::Diploma(d));
        }
    }
}
