//! The Karaoke tab and Karaoke mode.
//!
//! The tab ([`Side::Karaoke`]) sets karaoke up: the voice stem's level, the
//! microphone (listened to, to hear the singer, never played: what plays
//! never changes for it), and the button into the mode. While the tab or
//! the mode shows, every stem plays with the voice at its level ([`mix`]),
//! so what is heard, metered and recorded on the tab is what the mode
//! plays; leaving the tab puts the stems back as the settings have them
//! ([`follow_stems`]). The mode ([`Screen::Karaoke`], fullscreen in the
//! window) is where karaoke happens:
//!
//! - entering it stops the song and holds it at its start (entering it
//!   with shift, [`Op::KaraokeEnterHere`], leaves the song where it is),
//!   and a song's end holds the next one there: only [`Op::KaraokeStart`]
//!   (the START button, `enter`, `space`) plays a song, with a count-in
//!   of three seconds ([`Countdown`]) that ends as its first word is sung,
//!   drawn from what is heard: the song plays at once and the count comes
//!   in its intro, or, when it is sung from less than three seconds in,
//!   it starts that much later; a song paused in the mode goes on after a
//!   count-in of its own;
//! - `esc` is the only way out: back to the tab (the voice still at its
//!   level), playback where it is (a count-in under way is called off).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_lyrics::{Detail, Lyrics, Source};
use jodel_playback::stems::Stem;
use jodel_playback::{Command, PlayState, StemMix};

use crate::lyrics::Answer;

use crate::core::keymap::Op;
use crate::core::transition::Status;
use crate::core::{Effect, Gpu, Screen, Side, State};

/// The numbers a count-in counts: "3", "2", "1", then the music.
pub const COUNT: usize = 3;
/// How long each number of a count-in shows, in seconds.
pub const COUNT_STEP: f64 = 1.0;
/// How long a song's title and artist show alone as it starts, in seconds,
/// before they fade out.
pub const TITLE_HOLD: f64 = 2.0;
/// How long they take to fade out, in seconds.
pub const TITLE_FADE: f64 = 1.0;
/// How long the mode's controls stay after the pointer last moved, in ms.
pub const CONTROLS_MS: u64 = 3_000;
/// A slider's step for `←`/`→` (and the wheel), and with shift, in percent.
pub const STEP: i32 = 5;
/// The fine step.
pub const FINE: i32 = 1;

/// The tab's sliders: the voice stem's level is the only one (the
/// microphone is never played, so it has none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Slider {
    /// The voice stem's level on the tab and in the mode.
    #[default]
    Voice,
}

impl Slider {
    /// All of them, top to bottom.
    pub const ALL: [Slider; 1] = [Slider::Voice];

    /// Its label.
    pub fn label(self) -> &'static str {
        match self {
            Slider::Voice => "Voice",
        }
    }

    /// Its highest value, in percent.
    pub fn most(self) -> u8 {
        match self {
            Slider::Voice => 100,
        }
    }

    /// Its value in `state`, in percent.
    pub fn value(self, state: &State) -> u8 {
        match self {
            Slider::Voice => state.config.karaoke_voice,
        }
    }

    /// Its value as a share of its range, 0 to 1.
    pub fn share(self, state: &State) -> f64 {
        f64::from(self.value(state)) / f64::from(self.most())
    }
}

/// A count-in: the track, how long each number shows, and the second of
/// the track heard right after "1" (where the singing starts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Countdown {
    /// The track.
    pub index: usize,
    /// How long each number shows, in seconds ([`COUNT_STEP`]).
    pub beat: f64,
    /// Seconds into the track heard right after "1": its first sung word,
    /// or its start.
    pub on: f64,
    /// Seconds from when the song starts to be heard (its lead's start)
    /// before the count may begin: the title's time, or none.
    pub title: f64,
}

impl Countdown {
    /// The count-in for track `index`: a number a second, then the track
    /// from its start.
    pub fn new(index: usize) -> Countdown {
        Countdown::before(index, 0.0)
    }

    /// The count-in for track `index` whose singing starts `on` seconds in:
    /// "1" ends there.
    pub fn before(index: usize, on: f64) -> Countdown {
        Countdown {
            index,
            beat: COUNT_STEP,
            on: on.max(0.0),
            title: 0.0,
        }
    }

    /// [`Countdown::before`], after the song's title has shown and faded
    /// ([`TITLE_HOLD`], [`TITLE_FADE`]): the music waits for it where the
    /// singing starts too soon.
    pub fn after_title(index: usize, on: f64) -> Countdown {
        Countdown {
            title: TITLE_HOLD + TITLE_FADE,
            ..Countdown::before(index, on)
        }
    }

    /// Whether `t` (seconds into the track) is in the silence before it.
    pub fn in_lead(&self, t: f64) -> bool {
        t < 0.0
    }

    /// Seconds of silence before the track: the title's time and the
    /// count's beats before [`Countdown::on`], less what the track itself
    /// has before it.
    pub fn lead(&self) -> f64 {
        (self.title + COUNT as f64 * self.beat - self.on).max(0.0)
    }

    /// The number shown at `t` seconds into the track (negative in the
    /// silence before it) and how far through its beat, 0 to 1; `None`
    /// before the count and once it is over.
    pub fn at(&self, t: f64) -> Option<(usize, f64)> {
        let left = (self.on - t) / self.beat;
        if left <= 0.0 || left > COUNT as f64 {
            return None;
        }
        let n = left.ceil();
        Some((n as usize, n - left))
    }

    /// Whether the count is over at `t`.
    pub fn over(&self, t: f64) -> bool {
        t >= self.on
    }
}

/// The warning shown with lyrics a speech recogniser heard ([`SongLyrics::heard`]).
pub const HEARD_WARNING: &str =
    "No trusted source has these lyrics: machine learning heard them from the voice, so they may be wrong.";

/// What is known of a song's lyrics.
#[derive(Clone, Debug, PartialEq)]
pub enum SongLyrics {
    /// Asked for.
    Pending,
    /// Found: the text as found (its words only), and as shown (with our
    /// word times once the speech model has them).
    Found {
        /// As found.
        lyrics: Arc<Lyrics>,
        /// As shown.
        shown: Arc<Lyrics>,
    },
    /// The song has no words.
    Instrumental,
    /// None were found.
    NotFound,
    /// The look-up failed (it is tried again later).
    Failed(String),
}

impl SongLyrics {
    /// From a look-up's answer.
    pub fn from_answer(answer: Answer) -> SongLyrics {
        match answer {
            Answer::Found { lyrics, timed } => {
                let shown = timed.unwrap_or_else(|| lyrics.clone());
                SongLyrics::Found {
                    lyrics: Arc::new(lyrics),
                    shown: Arc::new(shown),
                }
            }
            Answer::Instrumental => SongLyrics::Instrumental,
            Answer::NotFound => SongLyrics::NotFound,
            Answer::Failed(why) => SongLyrics::Failed(why),
        }
    }

    /// Whether its lyrics were heard from the voice by a speech recogniser,
    /// no trusted source having them ([`Source::Transcribed`]).
    pub fn heard(&self) -> bool {
        matches!(self, SongLyrics::Found { lyrics, .. } if matches!(lyrics.source, Source::Transcribed(_)))
    }

    /// In a few words, for the tab.
    pub fn describe(&self) -> String {
        match self {
            SongLyrics::Pending => "looking…".into(),
            SongLyrics::Found { lyrics, shown } => {
                let from = match lyrics.source {
                    Source::Tag => "from its tags",
                    Source::Sidecar => "from its .lrc file",
                    Source::Lrclib => "from LRCLIB",
                    Source::Transcribed(_) => {
                        "heard from the voice by machine learning (no trusted source)"
                    }
                    Source::Unknown => "found",
                };
                let timed = match (&shown.timing.by, shown.timing.detail) {
                    (Some(by), Detail::Words) => format!("words timed ({})", by.name),
                    _ => "not timed yet".into(),
                };
                // How well the text fit what the recogniser heard.
                let fit = match shown.timing.fit {
                    Some(f) if f.swapped => format!(
                        "; its first text did not fit the song, this one does ({} % heard)",
                        f.percent
                    ),
                    Some(f) if f.fits() => format!("; fits the song ({} % heard)", f.percent),
                    Some(f) => format!("; may not fit the song (only {} % heard)", f.percent),
                    None => String::new(),
                };
                format!("{from}, {timed}{fit}")
            }
            SongLyrics::Instrumental => "instrumental".into(),
            SongLyrics::NotFound => "none found".into(),
            SongLyrics::Failed(why) => format!("not looked up: {why}"),
        }
    }
}

/// The songs whose lyrics are wanted: while the tab or the mode shows,
/// the one playing (or held), then the next two in the queue; while the
/// Mix tab shows, its tracks' (their words go on the voice's lane), unless
/// that is turned off (`mix_lyrics`).
pub fn lyrics_wanted(state: &State) -> Vec<PathBuf> {
    if state.screen == Screen::Main && state.side == Side::Mix {
        if !state.config.mix_lyrics {
            return Vec::new();
        }
        let mut paths: Vec<PathBuf> = Vec::new();
        for block in [
            crate::core::mix::Block::Prev,
            crate::core::mix::Block::Now,
            crate::core::mix::Block::Next,
        ] {
            let path = crate::core::mix::index(state, block).and_then(|i| state.queue.paths.get(i));
            if let Some(path) = path.filter(|p| !paths.contains(p)) {
                paths.push(path.clone());
            }
        }
        return paths;
    }
    if !(on_tab(state) || in_mode(state)) {
        return Vec::new();
    }
    let Some(first) = song_index(state) else {
        return Vec::new();
    };
    state
        .queue
        .paths
        .iter()
        .skip(first)
        .take(3)
        .cloned()
        .collect()
}

/// The queue index of the song the mode is about: the one heard, else the
/// one held for START, else the queue's current one.
pub fn song_index(state: &State) -> Option<usize> {
    state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.karaoke.held)
        .or(state.queue.current)
        .filter(|&i| i < state.queue.paths.len())
}

/// Asks for the lyrics wanted that are not known yet, when what is wanted
/// changed; forgets those no longer wanted.
pub(crate) fn want_lyrics(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = lyrics_wanted(state);
    if wanted == state.karaoke.lyrics_wanted {
        return;
    }
    let k = &mut state.karaoke;
    k.lyrics
        .retain(|p, l| wanted.contains(p) && !matches!(l, SongLyrics::Failed(_)));
    let ask: Vec<PathBuf> = wanted
        .iter()
        .filter(|p| !k.lyrics.contains_key(*p))
        .cloned()
        .collect();
    for path in &ask {
        k.lyrics.insert(path.clone(), SongLyrics::Pending);
    }
    k.lyrics_wanted = wanted;
    if !ask.is_empty() {
        fx.push(Effect::WantLyrics(ask));
    }
}

/// A song's lyrics arrived.
pub(crate) fn lyrics_arrived(state: &mut State, path: PathBuf, answer: Answer) {
    // Words timed by us: the speech model is here (fetched meanwhile).
    if let Answer::Found {
        timed: Some(timed), ..
    } = &answer
    {
        if timed.timing.detail == Detail::Words && timed.timing.by.is_some() {
            state.karaoke.model = ModelState::Ready;
        }
    }
    if state.karaoke.lyrics_wanted.contains(&path) {
        state
            .karaoke
            .lyrics
            .insert(path, SongLyrics::from_answer(answer));
    }
}

/// The lyrics of the song at `index`, when known.
pub fn lyrics_of(state: &State, index: usize) -> Option<&SongLyrics> {
    state.karaoke.lyrics.get(state.queue.paths.get(index)?)
}

/// The speech model that times the words. Setup gets it before the player
/// starts ([`crate::setup`]); a player started without (tests) has none.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ModelState {
    /// Not found, and nowhere to fetch it to: the words are shown, not
    /// timed.
    #[default]
    Missing,
    /// Not here yet: fetched when first needed (unless the lyrics are kept
    /// offline).
    Coming,
    /// There.
    Ready,
}

/// The microphone, as the app opened it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MicShown {
    /// Its name, while it is open.
    pub name: Option<String>,
    /// Why it could not be opened.
    pub error: Option<String>,
    /// The singer's recent peak, 0 to 1 (full scale): the voice the echo
    /// canceller gives (the music's echo out of it), or as captured while
    /// nothing plays.
    pub peak: f32,
    /// Open for a while and nothing but exact zeros came in: muted, or
    /// (macOS) the player is not allowed to use the microphone.
    pub silent: bool,
    /// The echo canceller, while it runs: the echo it removes in dB,
    /// whether it has learnt the echo's way, and how late the echo comes
    /// (seconds).
    pub echo: Option<(f32, bool, Option<f32>)>,
}

/// Whether the microphone is wanted: while the tab or the mode shows.
pub fn mic_wanted(state: &State) -> bool {
    (state.screen == Screen::Main && state.side == Side::Karaoke) || in_mode(state)
}

/// The Karaoke tab's and mode's state.
#[derive(Clone, Debug, Default)]
pub struct KaraokeState {
    /// The speech model.
    pub model: ModelState,
    /// The microphone.
    pub mic: MicShown,
    /// The lyrics known, by file.
    pub lyrics: HashMap<PathBuf, SongLyrics>,
    /// The files whose lyrics were last wanted, in order.
    pub lyrics_wanted: Vec<PathBuf>,
    /// The slider the arrow keys move on the tab.
    pub focus: Slider,
    /// The slider being dragged with the mouse.
    pub dragging: Option<Slider>,
    /// The track START plays, when known (the one the engine holds).
    pub held: Option<usize>,
    /// The count-in under way.
    pub countdown: Option<Countdown>,
    /// A paused song goes on at this time ([`State::now_ms`]), after the
    /// count-in that shows until then.
    pub resume_at: Option<u64>,
    /// The mode's controls show until then ([`State::now_ms`]).
    pub controls_until: u64,
    /// The voice level the engine was last sent with the karaoke mix
    /// ([`stems_due`] then): `None` when it was sent the settings' stems.
    pub stems_sent: Option<u8>,
    /// Where the song playing has been heard from without a jump (the mode
    /// entered where it was, a seek, another song, a pause ended): a
    /// count-in shows only if it fits whole after it. The track, seconds.
    pub counts_from: Option<(usize, f64)>,
    /// Where the song was heard at the last tick, and when
    /// ([`State::now_ms`]): what a jump is told by.
    pub last_heard: Option<(usize, f64, u64)>,
    /// The singing of the song in the mode, scored.
    pub singing: super::singing::Singing,
    /// How the last song was sung, shown until the next is started.
    pub result: Option<super::singing::SongResult>,
    /// The best scores sung, by song and level.
    pub bests: crate::scores::Bests,
    /// Whether the best scores changed since they were last saved.
    pub bests_changed: bool,
}

/// Whether the Karaoke tab is what keys act on.
pub fn on_tab(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Karaoke && state.palette.is_none()
}

/// Whether Karaoke mode is on.
pub fn in_mode(state: &State) -> bool {
    state.screen == Screen::Karaoke
}

/// Whether the Karaoke tab or the mode shows (a palette over the tab
/// included): the voice then plays at its level.
pub fn shown(state: &State) -> bool {
    (state.screen == Screen::Main && state.side == Side::Karaoke) || in_mode(state)
}

/// The stems the tab and the mode play: all of them, the voice at its
/// level.
pub fn mix(state: &State) -> StemMix {
    StemMix::ALL.with_level(Stem::Vocals, state.config.karaoke_voice)
}

/// The voice level the engine should have the karaoke mix at now: while
/// the tab or the mode shows and the stems can be separated (a GPU);
/// `None` for the settings' stems.
pub fn stems_due(state: &State) -> Option<u8> {
    (shown(state) && matches!(state.gpu, Gpu::Present(_))).then_some(state.config.karaoke_voice)
}

/// After any action: the engine sent the stems in force when they are no
/// longer what it has (the tab shown or left, by any way; the slider
/// moved; the GPU found).
pub(crate) fn follow_stems(state: &mut State, fx: &mut Vec<Effect>) {
    if stems_due(state) != state.karaoke.stems_sent {
        super::update::send_settings_now(state, fx);
    }
}

/// Whether nothing plays in the mode, so START shows.
pub fn waiting(state: &State) -> bool {
    let k = &state.karaoke;
    let counting = k.countdown.is_some() || k.resume_at.is_some();
    // A song held at its start (stopping, or stopped) waits too.
    in_mode(state) && !counting && (state.playing.state != PlayState::Playing || k.held.is_some())
}

/// Whether the mode's controls show (over the picture, which never moves
/// for them): while the pointer moved lately near the bottom
/// ([`in_reach`]), while a slider or the seek bar is held, after a key
/// moved the slider, and while waiting for START.
pub fn controls_shown(state: &State) -> bool {
    waiting(state)
        || state.karaoke.dragging.is_some()
        || state.seek_drag.is_some()
        || state.now_ms < state.karaoke.controls_until
}

/// Whether the seek bar does nothing now: in the mode, while a song is held
/// for START or counting in (its silence before it, or a paused song's
/// count before it goes on).
pub fn seek_refused(state: &State) -> bool {
    in_mode(state)
        && (state.karaoke.held.is_some() || state.karaoke.resume_at.is_some() || in_lead(state))
}

/// The share of the screen, from the bottom, where the pointer brings the
/// mode's controls up.
pub const REACH: f64 = 0.25;

/// Whether `row` is near enough the bottom to bring the controls up: the
/// last [`REACH`] of the rows, and never fewer than the controls take.
pub fn in_reach(state: &State, row: u16) -> bool {
    let rows = state.size.1;
    let reach = ((f64::from(rows) * REACH).ceil() as u16).max(6).min(rows);
    row >= rows - reach
}

/// The count-in before a paused song goes on at `now_ms`: the number shown
/// and how far through its second, if one is under way.
pub fn resume_count(state: &State, now_ms: u64) -> Option<(usize, f64)> {
    let at = state.karaoke.resume_at?;
    let left = at.saturating_sub(now_ms) as f64 / 1000.0 / COUNT_STEP;
    if left <= 0.0 || left > COUNT as f64 {
        return None;
    }
    let n = left.ceil();
    Some((n as usize, n - left))
}

/// The count-in at the time heard now, if one is under way.
pub fn count_now(state: &State) -> Option<(usize, f64)> {
    if state.karaoke.resume_at.is_some() {
        return resume_count(state, state.now_ms);
    }
    let countdown = state.karaoke.countdown?;
    let heard = state.playing.track.as_ref()?;
    (heard.index == countdown.index)
        .then(|| countdown.at(state.heard()))
        .flatten()
}

/// Carries out `op` on the Karaoke tab or in the mode; `false` when it is
/// not theirs (or neither is shown).
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if in_mode(state) && state.palette.is_none() {
        return apply_mode(state, op, fx);
    }
    if !on_tab(state) {
        return false;
    }
    match op {
        Op::KaraokeEnter | Op::Open | Op::Accept => enter(state, fx, false),
        Op::KaraokeEnterHere => enter(state, fx, true),
        Op::KaraokeMicNext => fx.push(Effect::NextMic),
        Op::KaraokeRecordEcho => fx.push(Effect::RecordEcho),
        Op::KaraokeGame | Op::KaraokeLevel | Op::KaraokeNotes | Op::KaraokeKeep => {
            game_setting(state, op, fx)
        }
        Op::Up | Op::Down => state.karaoke.focus = Slider::Voice,
        Op::SliderDown => slide_by(state, state.karaoke.focus, -STEP, fx),
        Op::SliderUp => slide_by(state, state.karaoke.focus, STEP, fx),
        Op::SliderFineDown => slide_by(state, state.karaoke.focus, -FINE, fx),
        Op::SliderFineUp => slide_by(state, state.karaoke.focus, FINE, fx),
        // Esc does nothing here: only tab and the tab buttons leave.
        Op::Back => {}
        _ => return false,
    }
    true
}

/// The game's settings: scoring on or off (`g`), the next level (`d`),
/// the notes shown or not (`p`); saved, and said.
fn game_setting(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let config = &mut state.config;
    let said = match op {
        Op::KaraokeGame => {
            config.karaoke_game = !config.karaoke_game;
            if config.karaoke_game {
                format!(
                    "Game on: the singing is scored ({})",
                    config.karaoke_level.name()
                )
            } else {
                "Game off: just singing, nothing scored".to_owned()
            }
        }
        Op::KaraokeLevel => {
            config.karaoke_level = config.karaoke_level.next();
            level_said(config.karaoke_level)
        }
        Op::KaraokeKeep => {
            config.karaoke_keep = !config.karaoke_keep;
            if config.karaoke_keep {
                "Your singing of each song is kept (recordings, beside the settings)".to_owned()
            } else {
                "Your singing is not kept".to_owned()
            }
        }
        _ => {
            config.karaoke_notes = !config.karaoke_notes;
            if config.karaoke_notes {
                "The song's notes show under the line".to_owned()
            } else {
                "The song's notes hidden".to_owned()
            }
        }
    };
    fx.push(Effect::SaveConfig);
    super::update::say_info(state, said);
}

/// What is said of `level` when it is picked.
fn level_said(level: super::singing::Level) -> String {
    format!("Level {}: {}", level.name(), level.what())
}

/// `level` picked on the tab (a click): kept in the settings.
pub(crate) fn set_level(state: &mut State, level: super::singing::Level, fx: &mut Vec<Effect>) {
    if state.config.karaoke_level == level {
        return;
    }
    state.config.karaoke_level = level;
    fx.push(Effect::SaveConfig);
    super::update::say_info(state, level_said(level));
}

/// [`apply`] in the mode: START, play and pause, the sliders, the next or
/// previous song (held at its start), and `esc`; quitting and the device
/// volume as anywhere; nothing else (no tabs, screens or seeking).
fn apply_mode(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    match op {
        Op::Back => leave(state, fx),
        Op::KaraokeStart | Op::Accept | Op::Open => start(state, fx),
        // Space pauses a song playing; anything else is START (never a
        // song going on without its count-in).
        Op::PlayPause => {
            let k = &state.karaoke;
            // The silence before a song is no song to pause (its intro,
            // with the count-in to come, is).
            let leading = k.countdown.is_some() && in_lead(state);
            let counting = leading || k.resume_at.is_some();
            if state.playing.state == PlayState::Playing && k.held.is_none() && !counting {
                fx.push(Effect::Engine(Command::Pause));
            } else if k.resume_at.is_some() {
                // Space again during a resume's count-in: it stays paused.
                state.karaoke.resume_at = None;
            } else {
                start(state, fx);
            }
        }
        Op::Next | Op::Previous => {
            let current = state
                .playing
                .track
                .as_ref()
                .map(|t| t.index)
                .or(state.karaoke.held)
                .or(state.queue.current)
                .unwrap_or(0);
            let index = if op == Op::Next {
                current + 1
            } else {
                current.saturating_sub(1)
            };
            if index < state.queue.paths.len() {
                state.karaoke.countdown = None;
                state.karaoke.resume_at = None;
                state.karaoke.held = Some(index);
                state.queue.cursor = index;
                fx.push(Effect::Engine(Command::Stop));
            }
        }
        Op::Up | Op::Down => state.karaoke.focus = Slider::Voice,
        Op::SliderDown => slide_by(state, state.karaoke.focus, -STEP, fx),
        Op::SliderUp => slide_by(state, state.karaoke.focus, STEP, fx),
        Op::SliderFineDown => slide_by(state, state.karaoke.focus, -FINE, fx),
        Op::SliderFineUp => slide_by(state, state.karaoke.focus, FINE, fx),
        Op::KaraokeMicNext => fx.push(Effect::NextMic),
        Op::KaraokeRecordEcho => fx.push(Effect::RecordEcho),
        Op::Quit | Op::VolumeUp | Op::VolumeDown => return false,
        _ => {}
    }
    true
}

/// Why track `index` cannot be sung yet, if it cannot: its lyrics still
/// looked for, or their words still being timed (a big folder queued is
/// got ready a track at a time), or known to be missing (none found, an
/// instrumental, the look-up failed). Words that cannot be timed (no speech
/// model, or the timing failed) are sung as they are.
pub fn not_singable(state: &State, index: usize) -> Option<&'static str> {
    let path = state.queue.paths.get(index)?;
    let settled = state
        .ready
        .get(path)
        .is_some_and(|jobs| jobs.contains(&super::Job::Lyrics));
    match lyrics_of(state, index) {
        None | Some(SongLyrics::Pending) => Some("still looking for its lyrics"),
        Some(SongLyrics::NotFound) => Some("no lyrics were found for it"),
        Some(SongLyrics::Instrumental) => Some("it is an instrumental"),
        Some(SongLyrics::Failed(_)) => Some("its lyrics could not be looked up"),
        Some(SongLyrics::Found { shown, .. }) => {
            let timed = shown.timing.detail == jodel_lyrics::Detail::Words;
            (!timed && !settled && state.karaoke.model != ModelState::Missing)
                .then_some("its words are still being timed")
        }
    }
}
/// Whether track `index` (if any) cannot be sung yet ([`not_singable`]),
/// said in the status line.
fn refused(state: &mut State, index: Option<usize>) -> bool {
    let Some((index, why)) = index.and_then(|i| not_singable(state, i).map(|why| (i, why))) else {
        return false;
    };
    let title = super::shown_title(state, &state.queue.paths[index]);
    let next = if in_mode(state) {
        "; N for the next song"
    } else {
        ""
    };
    super::update::say_info(state, format!("No karaoke for {title}: {why}{next}"));
    true
}

/// Into the mode: fullscreen (the window), every stem with the voice at
/// its level, songs held at their ends. The song playing (or paused)
/// stops and is held at its start for START, with its count-in; `here`:
/// it goes on where it is instead (nothing stopped, no count-in). A
/// transition armed or under way is called off. Not for a song without its
/// lyrics ready ([`not_singable`]): it says why instead.
pub(crate) fn enter(state: &mut State, fx: &mut Vec<Effect>, here: bool) {
    if refused(state, song_index(state)) {
        return;
    }
    state.screen = Screen::Karaoke;
    state.karaoke.controls_until = state.now_ms + CONTROLS_MS;
    state.karaoke.countdown = None;
    state.karaoke.resume_at = None;
    if state.mix.draft.status == Status::Armed {
        fx.push(Effect::Engine(Command::CancelTransition));
        state.mix.draft.status = Status::Draft;
        state.mix.draft.sent = None;
    }
    let song = state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.queue.current)
        .or((!state.queue.paths.is_empty()).then_some(state.queue.cursor))
        .filter(|&i| i < state.queue.paths.len());
    let stopped = state.playing.state == PlayState::Stopped;
    // Entered where the song is: a count-in shows only whole from here (a
    // first word under three seconds away gets none).
    if let (true, Some(index)) = (here, state.playing.track.as_ref().map(|t| t.index)) {
        state.karaoke.counts_from = Some((index, state.heard()));
        state.karaoke.last_heard = Some((index, state.heard(), state.now_ms));
    }
    if !here && !stopped {
        fx.push(Effect::Engine(Command::Stop));
    }
    // Where the song goes on, nothing is held (unless nothing plays).
    state.karaoke.held = if here && !stopped { None } else { song };
    fx.push(Effect::Engine(Command::HoldAtEnd(true)));
    match state.gpu {
        Gpu::Present(_) => {}
        Gpu::Looking => {
            super::update::say_info(state, "Still looking for a GPU: the voice stays in for now")
        }
        Gpu::Absent => super::update::say_info(state, "No GPU for the stems: the voice stays in"),
    }
}

/// Out of the mode, back to the tab (the voice still at its level), songs
/// running into each other again. A song still in the silence before it
/// is called off (it stops); anything else plays on.
pub(crate) fn leave(state: &mut State, fx: &mut Vec<Effect>) {
    state.screen = Screen::Main;
    state.side = Side::Karaoke;
    state.karaoke.resume_at = None;
    if let Some(countdown) = state.karaoke.countdown.take() {
        let heard = state.playing.track.as_ref().map(|t| t.index);
        if heard == Some(countdown.index) && countdown.in_lead(state.heard()) {
            state.karaoke.held = Some(countdown.index);
            fx.push(Effect::Engine(Command::Stop));
        }
    }
    state.karaoke.dragging = None;
    fx.push(Effect::Engine(Command::HoldAtEnd(false)));
}

/// START: the song held (or the queue's current one) plays from its start
/// after its count-in; a paused song goes on after the same count-in.
fn start(state: &mut State, fx: &mut Vec<Effect>) {
    if state.karaoke.countdown.is_some() || state.karaoke.resume_at.is_some() {
        return;
    }
    state.karaoke.result = None;
    // A song held (being stopped as the mode began) counts as stopped.
    let held = state.karaoke.held.is_some();
    match state.playing.state {
        PlayState::Playing if !held => {}
        PlayState::Paused if !held => {
            let paused = state.playing.track.as_ref().map(|t| t.index);
            if refused(state, paused) {
                return;
            }
            state.karaoke.resume_at =
                Some(state.now_ms + (COUNT as f64 * COUNT_STEP * 1000.0) as u64);
        }
        _ => {
            let index = state
                .karaoke
                .held
                .or(state.queue.current)
                .or((!state.queue.paths.is_empty()).then_some(state.queue.cursor))
                .filter(|&i| i < state.queue.paths.len());
            let Some(index) = index else {
                super::update::say_info(state, "Nothing to sing: queue some songs first");
                return;
            };
            if refused(state, Some(index)) {
                return;
            }
            let countdown = countdown_for(state, index);
            state.karaoke.countdown = Some(countdown);
            state.karaoke.held = None;
            fx.push(Effect::Engine(Command::PlayIn {
                index,
                lead: countdown.lead(),
            }));
        }
    }
}

/// The count-in for track `index`: after its title has shown and faded,
/// ending as its first word is sung, when its words are timed; else before
/// its start (the title stays, with what is known of the lyrics).
pub fn countdown_for(state: &State, index: usize) -> Countdown {
    match first_sung(state, index) {
        Some(on) => Countdown::after_title(index, on),
        None => Countdown::new(index),
    }
}

/// How much of track `index`'s title and artist shows at `t` (seconds into
/// it, as drawn), 0 to 1: all of it until it plays (held for START, or
/// waiting for its lead), then [`TITLE_HOLD`] from when it is first heard
/// (its lead's start, after START), fading out over [`TITLE_FADE`]; once,
/// whatever comes after (the count-in over, its record gone). None for a
/// song heard from further in (the mode entered with shift, a seek, a pause
/// ended).
pub fn title_shown(state: &State, index: usize, t: f64) -> f64 {
    let playing = state.playing.state == PlayState::Playing
        && state
            .playing
            .track
            .as_ref()
            .is_some_and(|tr| tr.index == index);
    let from = match state.karaoke.countdown {
        Some(c) if c.index == index && !playing => return 1.0,
        Some(c) if c.index == index => -c.lead(),
        _ if !playing && t < TITLE_HOLD => return 1.0,
        // Heard without a jump from there: its lead's start after START
        // (also once the count-in is over), else where it was joined.
        _ => counts_from(state, index),
    };
    if from > 0.05 {
        return 0.0;
    }
    (1.0 - (t - from - TITLE_HOLD) / TITLE_FADE).clamp(0.0, 1.0)
}

/// When the first word of track `index` is sung (seconds in), once its
/// words are timed.
pub fn first_sung(state: &State, index: usize) -> Option<f64> {
    let Some(SongLyrics::Found { shown, .. }) = lyrics_of(state, index) else {
        return None;
    };
    shown
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .filter_map(|w| w.start)
        .reduce(f64::min)
}

/// Whether the song heard is in the silence before it (a count-in's lead).
fn in_lead(state: &State) -> bool {
    match (state.karaoke.countdown, &state.playing.track) {
        (Some(c), Some(t)) => t.index == c.index && c.in_lead(state.heard()),
        (Some(_), None) => true,
        _ => false,
    }
}

/// The engine holds track `index` at its start (the song before ended).
pub(crate) fn held(state: &mut State, index: usize) {
    state.karaoke.held = Some(index);
    state.karaoke.countdown = None;
    sung_to_the_end(state);
}

/// The song in the mode sung to its end, the game on: how it went (its
/// points, words hit, medal and best before), kept to show, and its best
/// score at the level raised where it was beaten.
fn sung_to_the_end(state: &mut State) {
    let singing = &state.karaoke.singing;
    let Some(path) = singing.song().map(Path::to_path_buf) else {
        return;
    };
    if !state.config.karaoke_game || singing.scored_words == 0 {
        return;
    }
    let (points, hit, half, words, streak, level) = (
        singing.score,
        singing.hit_words,
        singing.half_words,
        singing.scored_words,
        singing.best_streak,
        singing.level,
    );
    let key = (path.display().to_string(), level);
    let best = state.karaoke.bests.get(&key).map(|b| b.points);
    let title = state
        .details
        .get(&path)
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| super::shown_title(state, &path));
    // Only a song sung from its start (most of its words heard) wins a
    // medal or a best.
    let whole = u64::from(words) * 10 >= singing.words() as u64 * 9;
    let result = super::singing::SongResult {
        title,
        points,
        hit,
        half,
        words,
        streak,
        level,
        medal: whole
            .then(|| super::singing::Medal::of(hit, half, words))
            .flatten(),
        best,
        whole,
    };
    if result.whole && result.new_best() {
        state
            .karaoke
            .bests
            .insert(key, crate::scores::Best { points, hit, words });
        state.karaoke.bests_changed = true;
    }
    state.karaoke.result = Some(result);
    state.karaoke.singing.restart();
}

/// Notes where the song is heard from when it jumped (another song, a
/// seek, playing again after a pause): the count-ins ([`counts_from`])
/// only show whole after that.
fn note_heard(state: &mut State) {
    let Some(index) = state.playing.track.as_ref().map(|t| t.index) else {
        state.karaoke.last_heard = None;
        return;
    };
    let at = state.heard();
    let playing = state.playing.state == PlayState::Playing;
    let jumped = match state.karaoke.last_heard {
        Some((i, was, then)) if i == index => {
            let went = (state.now_ms.saturating_sub(then)) as f64 / 1000.0;
            // Playing: about as far as the clock went. Not playing: nothing
            // is heard, so only where it plays from again counts.
            playing && (at - was - went).abs() > 1.0
        }
        _ => true,
    };
    if jumped || !playing {
        state.karaoke.counts_from = Some((index, at));
    }
    state.karaoke.last_heard = Some((index, at, state.now_ms));
}

/// Where the song playing (`index`) has been heard from without a jump: a
/// count-in shows only if it fits whole after it. Before anything is known,
/// from its start.
pub fn counts_from(state: &State, index: usize) -> f64 {
    match state.karaoke.counts_from {
        Some((i, at)) if i == index => at,
        _ => 0.0,
    }
}

/// The clock moved: a count-in heard to its end is over; a paused song's
/// goes on once its count-in is.
pub(crate) fn tick(state: &mut State, fx: &mut Vec<Effect>) {
    note_heard(state);
    if let Some(at) = state.karaoke.resume_at {
        if !in_mode(state) || state.playing.state != PlayState::Paused {
            state.karaoke.resume_at = None;
        } else if state.now_ms >= at {
            state.karaoke.resume_at = None;
            fx.push(Effect::Engine(Command::Resume));
        }
    }
    if let Some(countdown) = state.karaoke.countdown {
        let heard = state.playing.track.as_ref().map(|t| t.index);
        let over = match heard {
            Some(index) => index != countdown.index || countdown.over(state.playing.position),
            None => state.playing.state == PlayState::Stopped && state.karaoke.held.is_some(),
        };
        if over {
            state.karaoke.countdown = None;
        }
    }
}

/// The pointer moved over the mode to `row`: near the bottom its controls
/// show for a while; anywhere above, they go at once.
pub(crate) fn pointer_moved(state: &mut State, row: u16) {
    if in_mode(state) {
        state.karaoke.controls_until = if in_reach(state, row) {
            state.now_ms + CONTROLS_MS
        } else {
            0
        };
    }
}

/// Moves `slider` by `delta` percent.
fn slide_by(state: &mut State, slider: Slider, delta: i32, fx: &mut Vec<Effect>) {
    let value = (i32::from(slider.value(state)) + delta).clamp(0, i32::from(slider.most()));
    set(state, slider, value as u8, fx);
}

/// Moves `slider` by `by` percent (the wheel over it); the arrow keys move
/// it next.
pub(crate) fn step(state: &mut State, slider: Slider, by: i32, fx: &mut Vec<Effect>) {
    state.karaoke.focus = slider;
    slide_by(state, slider, by, fx);
}

/// Puts `slider` at a share of its range (a click or drag on it), at the
/// nearest whole percent.
pub(crate) fn slide_to(state: &mut State, slider: Slider, share: f64, fx: &mut Vec<Effect>) {
    state.karaoke.focus = slider;
    let value = (share.clamp(0.0, 1.0) * f64::from(slider.most())).round() as u8;
    set(state, slider, value, fx);
}

/// `slider` at `value` percent: kept in the settings, and the voice heard
/// at it at once on the tab and in the mode ([`follow_stems`], after the
/// action).
fn set(state: &mut State, slider: Slider, value: u8, fx: &mut Vec<Effect>) {
    if slider.value(state) == value {
        return;
    }
    match slider {
        Slider::Voice => state.config.karaoke_voice = value,
    }
    if in_mode(state) {
        state.karaoke.controls_until = state.now_ms + CONTROLS_MS;
    }
    if !fx.contains(&Effect::SaveConfig) {
        fx.push(Effect::SaveConfig);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_in_is_a_number_a_second_then_the_song_from_its_start() {
        let c = Countdown::new(0);
        assert_eq!((c.beat, c.on), (1.0, 0.0));
        assert!((c.lead() - 3.0).abs() < 1e-12);
        // "3" as the silence starts, "2" a second later, "1" the last.
        assert_eq!(c.at(-3.0), Some((3, 0.0)));
        assert_eq!(c.at(-1.5).map(|(n, _)| n), Some(2));
        let (n, through) = c.at(-0.25).unwrap();
        assert_eq!(n, 1);
        assert!((through - 0.75).abs() < 1e-9);
        assert_eq!(c.at(0.0), None);
        assert!(c.over(0.0) && !c.over(-0.01));
    }

    /// After START the title shows and fades, then the count runs, then
    /// the singing: the music waits for all that where the singing starts
    /// sooner, and plays at once (counted in over its intro) where not.
    #[test]
    fn the_title_fades_before_the_count_and_the_music_waits_for_it() {
        let title = TITLE_HOLD + TITLE_FADE;
        // Sung from the start: the title and the count in silence.
        let soon = Countdown::after_title(0, 0.0);
        assert!((soon.lead() - (title + 3.0)).abs() < 1e-12);
        // The title (from the lead's start) is gone as "3" comes.
        let starts = -soon.lead();
        assert_eq!(soon.at(starts + title - 0.01), None);
        assert_eq!(soon.at(starts + title).map(|(n, _)| n), Some(3));
        // Sung 2.5 s in: the music waits for what it lacks.
        let near = Countdown::after_title(0, 2.5);
        assert!((near.lead() - (title + 0.5)).abs() < 1e-12);
        // Sung 20 s in: no wait; the count over the intro.
        let far = Countdown::after_title(0, 20.0);
        assert_eq!(far.lead(), 0.0);
        assert_eq!(far.at(17.0).map(|(n, _)| n), Some(3));
    }
}
