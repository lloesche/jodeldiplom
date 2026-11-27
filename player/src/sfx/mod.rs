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
//! The synth plays every theme's patches ([`conductor::bank`]): the
//! theme in force (the settings') is the patches the game's conductor
//! plays, so another one is in force from the next frame, mid-song too,
//! and what sounds rings on. The Synth tab's sounds and keys are played at
//! once on the desk's frames by a player of the bench's own on the same
//! synth ([`Sfx::bench`], [`bench::Bench`]), in the theme the tab
//! compares: the game's conductor is not touched. The patches as the
//! state's book has them (edited, kept, or built in:
//! [`crate::core::patch_book`]) are what the synth is made with; an edit
//! after is told it as it plays, the patch changed alone (with nothing
//! edited or kept, nothing is). The synth's tempo is kept here, for both
//! ([`conductor::Tempo`]): the bench's sound plays at its tempo, the
//! song's own again on the game's next clock.
//!
//! The two planners share the one synth's tags and voices (a cancel of
//! all, a streak lost, a mono patch's lines): they are kept apart by the
//! screens, not by the synth. The game sounds in Karaoke mode alone
//! ([`crate::core::karaoke::in_mode`]) and the Synth tab is on the main
//! screen, so the two meet only as one hands over to the other (a sound
//! of the tab still ringing out as the mode starts). A use that plays
//! both at once needs a lane of its own in the tags first.
//!
//! A song's result revealed after it ([`reveal`]) sounds its beats at once,
//! each on the frame that reaches it, its paper and its music alike,
//! planned on the desk's own frames (the song is over then and its output
//! closed: the desk's device plays all along, so they sound all the same).
//! The game's live sounds go with its bubbles: with the notes hidden none
//! plays; the diploma shows all the same (the game scored the song), and
//! sounds.
//!
//! A result of a song's preparation found ([`crate::core::prep`]) sounds a
//! soft pluck at once on the desk's frames ([`Sfx::found`]), only while no
//! music plays: with a track playing or a preview on, none; one a frame
//! at most, and only for a result just found (a window that was not drawn
//! for a while plays none of what it missed).

pub mod bench;
pub mod conductor;
pub mod moments;
pub mod reveal;

use jodel_output::mixer::{Effects, Fader, LiveCommands, Timebase};
use jodel_playback::PlayState;
use jodel_synth::Patch;

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

/// How long after a result of a song's preparation is found it may still
/// sound, ms (a frame comes within this while the window is drawn).
const FOUND_FRESH_MS: u64 = 1_000;

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
    /// The Synth tab's player on the same synth.
    bench: Option<bench::Bench>,
    /// What the synth has: every theme's patches (in the bank's order),
    /// the patches' book's revision they are of (none: not all told yet),
    /// its tempo.
    told_patches: Vec<Patch>,
    revision: Option<u64>,
    tempo: conductor::Tempo,
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
    /// The preparation's results found as last seen
    /// ([`crate::core::prep::Prep::found`]); none before the first frame.
    found: Option<u64>,
}

impl Sfx {
    /// A frame drawn: the conductor for the device's rate, then the game's
    /// new moments played on `desk`, then the result's reveal's (after what
    /// the moments call off, so it never calls them off).
    pub fn frame(&mut self, state: &State, desk: Desk<'_>) {
        self.conductor_at(&desk, state);
        self.in_theme(state);
        self.patches(state, &desk);
        self.loudness(state, &desk);
        self.moments(state, &desk);
        self.reveal(state, &desk);
        self.found(state, &desk);
    }

    /// A result of a song's preparation found since the last frame: its
    /// sound on `desk`, at once, while no music plays (no track playing,
    /// no preview), the result just found and the Effects strip up.
    fn found(&mut self, state: &State, desk: &Desk<'_>) {
        let prep = &state.prep;
        let Some(seen) = self.found.replace(prep.found) else {
            return;
        };
        let music =
            state.playing.state == PlayState::Playing || crate::core::preview::active(state);
        let fresh =
            (prep.found_at).is_some_and(|at| state.now_ms.saturating_sub(at) <= FOUND_FRESH_MS);
        if prep.found == seen
            || music
            || !fresh
            || state.desk.setting(StripKind::Effects).gain() == 0.0
        {
            return;
        }
        let step = (prep.songs())
            .filter_map(|(_, s)| s.found.as_ref())
            .max_by_key(|(_, at)| *at)
            .map_or(0, |(found, _)| found.step());
        self.on_desk(desk, Event::Found { step });
    }

    /// The conductor at the device's rate, its synth sent to the desk, made
    /// again when the device opens at another (none at a rate the synth
    /// takes not: silent there), with every theme's patches as the book
    /// has them, and the bench's player on it. Another theme in force is
    /// not a new synth: the conductor plays its patches ([`Sfx::patches`]),
    /// so what sounds rings on and nothing planned is lost.
    fn conductor_at(&mut self, desk: &Desk<'_>, state: &State) {
        let rate = desk.timebase.rate;
        let at = self.conductor.as_ref().map(Conductor::rate);
        let takes = (jodel_synth::dsp::Rate::MIN..=jodel_synth::dsp::Rate::MAX).contains(&rate);
        if rate == 0 || at == Some(rate) || !takes {
            return;
        }
        let book = &state.patches;
        let bank = conductor::bank(|theme, atom| book.patch(theme, atom));
        self.told_patches = (conductor::every())
            .map(|(t, a)| book.patch(t, a))
            .collect();
        self.revision = Some(book.revision());
        desk.effects.live(Conductor::synth(rate, bank));
        self.conductor = Some(Conductor::planner(rate, state.theme()));
        self.bench = Some(bench::Bench::new(rate));
        self.tempo = conductor::Tempo::default();
        self.on_song = false;
        self.told = None;
    }

    /// The game's conductor on the theme in force.
    fn in_theme(&mut self, state: &State) {
        if let Some(c) = self.conductor.as_mut() {
            c.set_theme(state.theme());
        }
    }

    /// Each theme's atoms' patches as the book has them, once it changed,
    /// told the synth where they are not what it has (one not taken, a
    /// full queue, on the next frame).
    fn patches(&mut self, state: &State, desk: &Desk<'_>) {
        if self.conductor.is_none() {
            return;
        }
        let book = &state.patches;
        if self.revision == Some(book.revision()) {
            return;
        }
        let mut out = desk.effects.commands(None);
        let all = tell_patches(&mut self.told_patches, book, |id, p| out.patch(id, p));
        self.revision = all.then_some(book.revision());
    }

    /// What the Synth tab asks played now, on the desk's frames, by the
    /// bench's player: a sound in its key on a steady beat at its tempo
    /// (the synth told it), or a key of its keyboard pressed or let go of;
    /// the patches as the book has them first.
    pub fn bench(&mut self, state: &State, desk: Desk<'_>, ask: crate::core::synth_tab::Bench) {
        use crate::core::synth_tab::Bench;
        self.conductor_at(&desk, state);
        self.in_theme(state);
        self.patches(state, &desk);
        let at = desk.timebase.frame;
        let Some(b) = self.bench.as_mut() else {
            return;
        };
        let mut out = desk.effects.commands(None);
        match ask {
            Bench::Play {
                what,
                key,
                bpm,
                theme,
            } => {
                self.tempo.tell(&mut out, bpm);
                b.play(&mut out, at, what, (theme, key.as_ref(), bpm));
            }
            Bench::Down {
                atom,
                note,
                held,
                theme,
            } => b.down(&mut out, (at, theme), (atom, note), held),
            Bench::Up { held } => b.up(&mut out, at, held),
        }
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
        if let Some(c) = self.conductor.as_mut() {
            let mut out = desk.effects.commands(None);
            self.tempo.time(c, &mut out, Clock::free(c.rate()));
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
            if let Some(c) = self.conductor.as_mut() {
                c.set_key(told.key.as_ref(), told.shift);
                if !told.grid {
                    let mut out = desk.effects.commands(None);
                    self.tempo.time(c, &mut out, Clock::free(c.rate()));
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
        if let Some(c) = self.conductor.as_mut() {
            let mut out = desk.effects.commands(Some(fader));
            self.tempo.time(c, &mut out, clock);
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

/// Each theme's atoms' patches as `book` has them told by `tell` (a
/// synth's patch set: whether it was taken) where they are not what
/// `told` says the synth has (in the bank's order), which they then are;
/// whether every one was taken.
fn tell_patches(
    told: &mut [Patch],
    book: &crate::core::patch_book::PatchBook,
    mut tell: impl FnMut(jodel_synth::PatchId, Patch) -> bool,
) -> bool {
    let mut all = true;
    for (k, (has, (theme, atom))) in told.iter_mut().zip(conductor::every()).enumerate() {
        let patch = book.patch(theme, atom);
        if *has != patch {
            if tell(jodel_synth::PatchId::nth(k), patch) {
                *has = patch;
            } else {
                all = false;
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::patch_book::PatchBook;
    use crate::theme::{CLASSIC, SPOOKY};
    use conductor::Atom;

    /// An edit told the synth is what it then plays, in its theme alone:
    /// the synth made of the book has its patches; the book edited, the
    /// edited patch is told once and is the synth's, the other theme's
    /// untouched; one not taken is told again.
    #[test]
    fn an_edit_is_the_synth_s_patch() {
        let mut book = PatchBook::default();
        let (_, mut synth) = Conductor::new(48_000, &CLASSIC);
        let mut told: Vec<Patch> = (conductor::every())
            .map(|(t, a)| book.patch(t, a))
            .collect();
        let mut sent = 0;
        let mut tell = |told: &mut [Patch], book: &PatchBook| {
            tell_patches(told, book, |id, p| {
                sent += 1;
                synth.set_patch(id, p)
            })
        };
        assert!(tell(&mut told, &book), "nothing to tell");
        book.edit(&SPOOKY, Atom::Pop, |p| p.level += 6.0);
        assert!(
            !tell_patches(&mut told.clone(), &book, |_, _| false),
            "not taken"
        );
        assert!(tell(&mut told, &book));
        assert!(tell(&mut told, &book));
        assert_eq!(sent, 1, "told once");
        let pop = conductor::id(conductor::place(&SPOOKY), Atom::Pop);
        assert_eq!(synth.patch(pop), Some(&book.patch(&SPOOKY, Atom::Pop)));
        assert_ne!(synth.patch(pop), Some(&SPOOKY.patch(Atom::Pop)));
        let classic = conductor::id(conductor::place(&CLASSIC), Atom::Pop);
        assert_eq!(synth.patch(classic), Some(&CLASSIC.patch(Atom::Pop)));
    }
}
