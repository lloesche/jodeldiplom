//! The one table of key bindings.
//!
//! The window translates its key events into [`Key`] and asks [`lookup`]
//! (or [`action`]) what it means in the current state. The help screen
//! ([`help_sections`]) and the command palette ([`palette_entries`]) are
//! generated from the same [`KEYMAP`] table, so they always match.

use crate::core::{Screen, Side, State, TYPE_AHEAD_MS};

use jodel_playback::Stem;

/// A key, as the window reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// A character, already shifted (`A`, `?`).
    Char(char),
    /// Enter / return.
    Enter,
    /// Escape.
    Esc,
    /// Backspace.
    Backspace,
    /// Tab.
    Tab,
    /// Arrow up.
    Up,
    /// Arrow down.
    Down,
    /// Arrow left.
    Left,
    /// Arrow right.
    Right,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Home.
    Home,
    /// End.
    End,
    /// Delete.
    Delete,
}

/// A key with its modifiers (shift is in the character).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    /// The key.
    pub code: KeyCode,
    /// Control held.
    pub ctrl: bool,
    /// Alt (option) held.
    pub alt: bool,
    /// Shift held, for keys other than characters (a character already
    /// says it: `Q`, `?`).
    pub shift: bool,
}

impl Key {
    /// A key without modifiers.
    pub const fn plain(code: KeyCode) -> Key {
        Key {
            code,
            ctrl: false,
            alt: false,
            shift: false,
        }
    }

    /// A key other than a character, with shift.
    pub const fn shift(code: KeyCode) -> Key {
        Key {
            code,
            ctrl: false,
            alt: false,
            shift: true,
        }
    }

    /// A character key without modifiers.
    pub const fn ch(c: char) -> Key {
        Key::plain(KeyCode::Char(c))
    }

    /// A character key with alt.
    pub const fn alt(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: false,
            alt: true,
            shift: false,
        }
    }

    /// A character key with shift, as the window reports a digit held with
    /// it (the key without the modifier, for any keyboard layout).
    pub const fn shifted(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: false,
            alt: false,
            shift: true,
        }
    }

    /// A character key with control.
    pub const fn ctrl(c: char) -> Key {
        Key {
            code: KeyCode::Char(c),
            ctrl: true,
            alt: false,
            shift: false,
        }
    }

    /// How the help screen writes the key.
    pub fn label(self) -> String {
        let base = match self.code {
            KeyCode::Char(' ') => "space".to_owned(),
            KeyCode::Char(c) => c.to_string(),
            KeyCode::Enter => "enter".into(),
            KeyCode::Esc => "esc".into(),
            KeyCode::Backspace => "backspace".into(),
            KeyCode::Tab => "tab".into(),
            KeyCode::Up => "↑".into(),
            KeyCode::Down => "↓".into(),
            KeyCode::Left => "←".into(),
            KeyCode::Right => "→".into(),
            KeyCode::PageUp => "pgup".into(),
            KeyCode::PageDown => "pgdn".into(),
            KeyCode::Home => "home".into(),
            KeyCode::End => "end".into(),
            KeyCode::Delete => "del".into(),
        };
        match (self.ctrl, self.alt, self.shift) {
            (true, _, _) => format!("ctrl+{base}"),
            (false, true, _) => format!("alt+{base}"),
            (false, false, true) => format!("shift+{base}"),
            _ => base,
        }
    }
}

/// A named operation: what keys, buttons and palette entries do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// Play or pause (plays the selected file when nothing is playing).
    PlayPause,
    /// Seek back [`crate::core::SEEK_STEP`] seconds.
    SeekBack,
    /// Seek forward [`crate::core::SEEK_STEP`] seconds.
    SeekForward,
    /// The next track.
    Next,
    /// The previous track.
    Previous,
    /// Stop playback.
    Stop,
    /// Switch the list pane: browser, queue, transition, karaoke, mixer.
    ToggleSide,
    /// Open or close Now playing.
    NowPlaying,
    /// Open the settings.
    Settings,
    /// Open the help.
    Help,
    /// Open the command palette.
    Palette,
    /// Filter the current folder.
    Filter,
    /// Quit.
    Quit,
    /// Close what is open, clear the filter, or leave a screen.
    Back,
    /// Move up.
    Up,
    /// Move down.
    Down,
    /// Move up a page.
    PageUp,
    /// Move down a page.
    PageDown,
    /// The first row.
    Top,
    /// The last row.
    Bottom,
    /// Open a folder, play a file, change a setting.
    Open,
    /// The folder above.
    Parent,
    /// Queue the selected file, or the selected folder recursively.
    Enqueue,
    /// Queue the selected file and play it (the queue kept).
    QueuePlay,
    /// Play the selected file with its folder in place of the queue.
    OpenReplace,
    /// Queue: take the selected track out.
    QueueRemove,
    /// Queue: move the selected track up one.
    QueueMoveUp,
    /// Queue: move the selected track down one.
    QueueMoveDown,
    /// Save the queue as a playlist.
    QueueSave,
    /// Show the saved queues in the browser.
    QueueOpenSaved,
    /// Search below the current folder for the filter text.
    SearchBelow,
    /// Accept the filter or run the palette's selection.
    Accept,
    /// Delete the last typed character.
    Erase,
    /// Paste the clipboard's text into the filter or the palette (on the
    /// list, into a new filter).
    Paste,
    /// The previous value of a setting.
    ValueDown,
    /// The next value of a setting.
    ValueUp,
    /// Show the playing track in the browser, selected.
    GoTrack,
    /// Show the playing track's folder (its album) in the browser.
    GoAlbum,
    /// Show the folder above the album (its artist) with the album
    /// selected.
    GoArtist,
    /// Show or hide the tempo and the beat pulse.
    ToggleBpm,
    /// The drums on or off.
    ToggleDrums,
    /// The bass on or off.
    ToggleBass,
    /// The vocals on or off.
    ToggleVocals,
    /// Everything else on or off.
    ToggleOther,
    /// Every stem on: the file as it is.
    AllStems,
    /// The output device's volume a step up.
    VolumeUp,
    /// The output device's volume a step down.
    VolumeDown,
    /// Transition: the cursor a bar back.
    TransitionBarBack,
    /// Transition: the cursor a bar on.
    TransitionBarForward,
    /// Transition: the cursor a beat back.
    TransitionBeatBack,
    /// Transition: the cursor a beat on.
    TransitionBeatForward,
    /// Transition: select a bar more, back.
    TransitionSelectBack,
    /// Transition: select a bar more, on.
    TransitionSelectForward,
    /// Transition: fewer bars shown.
    TransitionZoomIn,
    /// Transition: more bars shown.
    TransitionZoomOut,
    /// Transition: the track selected to the widest zoom that leaves out
    /// no word of its lyrics.
    TransitionFitLyrics,
    /// Transition: follow the playhead again.
    TransitionFollow,
    /// Transition: the lyrics on the vocals lane, on or off.
    TransitionLyrics,
    /// Grid: every bar starts a beat earlier.
    GridEarlier,
    /// Grid: every bar starts a beat later.
    GridLater,
    /// Grid: every beat a little earlier.
    GridNudgeBack,
    /// Grid: every beat a little later.
    GridNudgeForward,
    /// Grid: half the tempo.
    GridHalf,
    /// Grid: double the tempo.
    GridDouble,
    /// Grid: the beat under the cursor starts a bar (bars of four).
    GridOne,
    /// Grid: back to the beats the analysis found.
    GridReset,
    /// Transition: arm it at the two cues, or call it off.
    TransitionGo,
    /// Transition: the next preset.
    TransitionPreset,
    /// The next track key: cut, or mix in by the next preset.
    SkipMix,
    /// Transition: varispeed or time-stretch.
    TransitionTempo,
    /// Transition: how the beats pair (by the tempi, 1:1, 2:1, 1:2).
    TransitionPairing,
    /// Transition: edit the fades (or stop editing).
    TransitionEdit,
    /// Editing: the bar before.
    EditBack,
    /// Editing: the bar after.
    EditForward,
    /// Editing: a step louder at this bar.
    EditRaise,
    /// Editing: a step quieter at this bar.
    EditLower,
    /// Editing: off at this bar.
    EditOff,
    /// Editing: full at this bar.
    EditFull,
    /// Editing: this bar cuts (or fades) to its gain.
    EditCut,
    /// Transition: the playing track jumps to the start of the bar this many
    /// bars from the one heard (0: its own start), at once.
    JumpBar(i8),
    /// Transition: the same when the bar heard ends.
    JumpBarAtEnd(i8),
    /// Karaoke: into Karaoke mode (fullscreen).
    KaraokeEnter,
    /// Karaoke: into Karaoke mode where the song is: no rewind, no
    /// count-in (shift+enter, shift+click; for trying things out).
    KaraokeEnterHere,
    /// Karaoke mode: START, the song held after a count-in.
    KaraokeStart,
    /// Karaoke mode: the demo sings while the key is held
    /// ([`crate::core::singing::Demo`]).
    KaraokeDemo,
    /// Karaoke: the slider a step down.
    SliderDown,
    /// Karaoke: the slider a step up.
    SliderUp,
    /// Karaoke: the slider a percent down.
    SliderFineDown,
    /// Karaoke: the slider a percent up.
    SliderFineUp,
    /// The next microphone.
    MicNext,
    /// Karaoke: record 30 s of what the echo canceller works with.
    KaraokeRecordEcho,
    /// Karaoke: the game (scoring) on or off.
    KaraokeGame,
    /// Karaoke: the song's notes shown or hidden.
    KaraokeNotes,
    /// Karaoke: each song's singing kept, or not.
    KaraokeKeep,
    /// Karaoke: sync, measuring how late the voice comes.
    KaraokeSync,
    /// Syncing: keep the delay measured.
    SyncKeep,
    /// Syncing: again.
    SyncAgain,
    /// Mixer: the strip to the left.
    DeskPrevStrip,
    /// Mixer: the strip to the right.
    DeskNextStrip,
    /// Mixer: the strip selected muted, or not.
    DeskMute,
    /// Mixer: the strip selected soloed, or not.
    DeskSolo,
    /// The microphone's echo canceller on or off.
    DeskEcho,
    /// The song a semitone lower, its tempo kept ([`crate::core::key`]).
    KeyDown,
    /// The song a semitone higher.
    KeyUp,
    /// The song in its own key again.
    KeyReset,
}

impl Op {
    /// The operation's name in the palette.
    pub fn name(self) -> &'static str {
        match self {
            Op::PlayPause => "Play / pause",
            Op::SeekBack => "Seek back",
            Op::SeekForward => "Seek forward",
            Op::Next => "Next track",
            Op::Previous => "Previous track",
            Op::Stop => "Stop",
            Op::ToggleSide => "Browser / queue / transition / karaoke / mixer",
            Op::NowPlaying => "Now playing",
            Op::Settings => "Settings",
            Op::Help => "Help",
            Op::Palette => "Command palette",
            Op::Filter => "Filter this folder",
            Op::Quit => "Quit",
            Op::Back => "Back",
            Op::Up => "Up",
            Op::Down => "Down",
            Op::PageUp => "Page up",
            Op::PageDown => "Page down",
            Op::Top => "First row",
            Op::Bottom => "Last row",
            Op::Open => "Open / play",
            Op::Parent => "Parent folder",
            Op::Enqueue => "Queue file or folder",
            Op::QueuePlay => "Queue and play",
            Op::OpenReplace => "Play with its folder, in place of the queue",
            Op::QueueRemove => "Queue: remove the track",
            Op::QueueMoveUp => "Queue: move the track up",
            Op::QueueMoveDown => "Queue: move the track down",
            Op::QueueSave => "Queue: save as a playlist",
            Op::QueueOpenSaved => "Queue: the saved queues",
            Op::SearchBelow => "Search subfolders",
            Op::Accept => "Accept",
            Op::Erase => "Delete a character",
            Op::Paste => "Paste",
            Op::ValueDown => "Previous value",
            Op::ValueUp => "Next value",
            Op::GoTrack => "Go to playing track",
            Op::GoAlbum => "Go to album",
            Op::GoArtist => "Go to artist",
            Op::ToggleBpm => "Tempo and beat pulse on / off",
            Op::ToggleDrums => "Stems: drums on / off",
            Op::ToggleBass => "Stems: bass on / off",
            Op::ToggleVocals => "Stems: vocals on / off",
            Op::ToggleOther => "Stems: other on / off",
            Op::AllStems => "Stems: all on (the file as it is)",
            Op::VolumeUp => "Volume: up",
            Op::VolumeDown => "Volume: down",
            Op::TransitionBarBack => "Transition: a bar back",
            Op::TransitionBarForward => "Transition: a bar on",
            Op::TransitionBeatBack => "Transition: a beat back",
            Op::TransitionBeatForward => "Transition: a beat on",
            Op::TransitionSelectBack => "Transition: select back",
            Op::TransitionSelectForward => "Transition: select on",
            Op::TransitionZoomIn => "Transition: zoom in",
            Op::TransitionZoomOut => "Transition: zoom out",
            Op::TransitionFitLyrics => "Transition: zoom the track selected to fit its lyrics",
            Op::TransitionFollow => "Transition: follow the playhead",
            Op::TransitionLyrics => "Transition: lyrics on the vocals lane on/off",
            Op::GridEarlier => "Grid: bars a beat earlier",
            Op::GridLater => "Grid: bars a beat later",
            Op::GridNudgeBack => "Grid: nudge 5 ms earlier",
            Op::GridNudgeForward => "Grid: nudge 5 ms later",
            Op::GridHalf => "Grid: half tempo",
            Op::GridDouble => "Grid: double tempo",
            Op::GridOne => "Grid: a bar starts at the cursor",
            Op::GridReset => "Grid: back to the analysis",
            Op::TransitionGo => "Transition: arm at the cues / call off",
            Op::TransitionPreset => "Transition: next preset",
            Op::SkipMix => "Skipping: cut / mix in",
            Op::TransitionTempo => "Transition: varispeed / time-stretch",
            Op::TransitionPairing => "Transition: beats paired by the tempi / 1:1 / 2:1 / 1:2",
            Op::TransitionEdit => "Transition: edit the fades",
            Op::EditBack => "Fades: bar before",
            Op::EditForward => "Fades: bar after",
            Op::EditRaise => "Fades: louder",
            Op::EditLower => "Fades: quieter",
            Op::EditOff => "Fades: off here",
            Op::EditFull => "Fades: full here",
            Op::EditCut => "Fades: cut / fade here",
            Op::JumpBar(_) => "Transition: jump to a bar",
            Op::JumpBarAtEnd(_) => "Transition: jump to a bar at the bar's end",
            Op::KaraokeEnter => "Karaoke mode",
            Op::KaraokeEnterHere => "Karaoke mode where the song is",
            Op::KaraokeStart => "Karaoke: start the song",
            Op::KaraokeDemo => "Karaoke: the demo sings (held)",
            Op::SliderDown => "Slider down",
            Op::SliderUp => "Slider up",
            Op::SliderFineDown => "Slider a percent down",
            Op::SliderFineUp => "Slider a percent up",
            Op::MicNext => "Microphone: the next one",
            Op::KaraokeRecordEcho => "Karaoke: record the echo canceller (30 s)",
            Op::KaraokeGame => "Karaoke: the game (scoring) on or off",
            Op::KaraokeNotes => "Karaoke: show or hide the song's notes",
            Op::KaraokeKeep => "Karaoke: keep your singing of each song, or not",
            Op::KaraokeSync => "Karaoke: sync (measure how late your voice comes)",
            Op::SyncKeep => "Sync: keep the delay measured",
            Op::SyncAgain => "Sync: again",
            Op::DeskPrevStrip => "Mixer: the strip to the left",
            Op::DeskNextStrip => "Mixer: the strip to the right",
            Op::DeskMute => "Mixer: mute the strip selected",
            Op::DeskSolo => "Mixer: solo the strip selected",
            Op::DeskEcho => "Microphone: the echo canceller on or off",
            Op::KeyDown => "Key: a semitone down",
            Op::KeyUp => "Key: a semitone up",
            Op::KeyReset => "Key: the song's own",
        }
    }

    /// The stem it switches, for the four stem operations.
    pub fn stem(self) -> Option<Stem> {
        match self {
            Op::ToggleDrums => Some(Stem::Drums),
            Op::ToggleBass => Some(Stem::Bass),
            Op::ToggleVocals => Some(Stem::Vocals),
            Op::ToggleOther => Some(Stem::Other),
            _ => None,
        }
    }

    /// The operation that switches `stem`.
    pub fn for_stem(stem: Stem) -> Op {
        match stem {
            Stem::Drums => Op::ToggleDrums,
            Stem::Bass => Op::ToggleBass,
            Stem::Vocals => Op::ToggleVocals,
            Stem::Other => Op::ToggleOther,
        }
    }

    /// Whether the palette lists it (commands, not movements or typing).
    pub fn in_palette(self) -> bool {
        !matches!(
            self,
            Op::Up
                | Op::Down
                | Op::PageUp
                | Op::PageDown
                | Op::Top
                | Op::Bottom
                | Op::Accept
                | Op::Erase
                | Op::Paste
                | Op::ValueDown
                | Op::ValueUp
                | Op::Back
                | Op::Palette
                | Op::Open
                | Op::TransitionBarBack
                | Op::TransitionBarForward
                | Op::TransitionBeatBack
                | Op::TransitionBeatForward
                | Op::TransitionSelectBack
                | Op::TransitionSelectForward
                | Op::EditBack
                | Op::EditForward
                | Op::EditRaise
                | Op::EditLower
                | Op::EditOff
                | Op::EditFull
                | Op::EditCut
                | Op::JumpBar(_)
                | Op::JumpBarAtEnd(_)
                | Op::SliderDown
                | Op::SliderUp
                | Op::SliderFineDown
                | Op::SliderFineUp
                | Op::DeskPrevStrip
                | Op::DeskNextStrip
                | Op::SyncKeep
                | Op::SyncAgain
                | Op::KaraokeDemo
        )
    }
}

/// Where a binding applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Context {
    /// Everywhere, unless a more specific context binds the key, and not
    /// while typing (except keys with control).
    Global,
    /// The browser and queue list.
    List,
    /// The queue (before [`Context::List`]).
    Queue,
    /// Typing a filter after `/`.
    Filter,
    /// Typing in the command palette.
    Palette,
    /// The Now playing screen.
    NowPlaying,
    /// The settings screen.
    Settings,
    /// The device list in the settings.
    DeviceList,
    /// The folder list the data folder is chosen in.
    FolderList,
    /// The help screen.
    Help,
    /// Typing a search of the help after `/`.
    HelpSearch,
    /// The Transition tab.
    Transition,
    /// Editing a transition's fades on the Transition tab.
    Fades,
    /// The Karaoke tab.
    Karaoke,
    /// Karaoke mode.
    KaraokeMode,
    /// Syncing, on the Karaoke tab.
    Sync,
    /// The Mixer tab.
    Desk,
}

impl Context {
    /// The help section title.
    pub fn title(self) -> &'static str {
        match self {
            Context::Global => "Anywhere",
            Context::List => "Browser and queue",
            Context::Queue => "Queue",
            Context::Filter => "Filter (after /)",
            Context::Palette => "Command palette (after :)",
            Context::NowPlaying => "Now playing",
            Context::Settings => "Settings",
            Context::DeviceList => "Device list",
            Context::FolderList => "Data folder list",
            Context::Help => "Help",
            Context::HelpSearch => "Searching the help (after /)",
            Context::Transition => "Transition tab",
            Context::Fades => "Transition tab, editing the fades (after e)",
            Context::Karaoke => "Karaoke tab",
            Context::KaraokeMode => "Karaoke mode (fullscreen)",
            Context::Sync => "Syncing (s on the Karaoke tab)",
            Context::Desk => "Mixer tab",
        }
    }

    /// Keys type text here: unbound characters are typed.
    pub fn types(self) -> bool {
        matches!(
            self,
            Context::Filter | Context::Palette | Context::HelpSearch
        )
    }
}

/// One line of the keymap.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    /// Where it applies.
    pub context: Context,
    /// The keys (any of them).
    pub keys: &'static [Key],
    /// What they do.
    pub op: Op,
    /// The help text.
    pub help: &'static str,
}

const fn b(context: Context, keys: &'static [Key], op: Op, help: &'static str) -> Binding {
    Binding {
        context,
        keys,
        op,
        help,
    }
}

use KeyCode as K;

const UP: Key = Key::plain(K::Up);
const DOWN: Key = Key::plain(K::Down);
const LEFT: Key = Key::plain(K::Left);
const RIGHT: Key = Key::plain(K::Right);
const ENTER: Key = Key::plain(K::Enter);
const ESC: Key = Key::plain(K::Esc);
const SHIFT_ENTER: Key = Key::shift(K::Enter);
const ALT_ENTER: Key = Key {
    code: K::Enter,
    ctrl: false,
    alt: true,
    shift: false,
};
const BACKSPACE: Key = Key::plain(K::Backspace);
const ALT_LEFT: Key = Key {
    code: K::Left,
    ctrl: false,
    alt: true,
    shift: false,
};
const ALT_RIGHT: Key = Key {
    code: K::Right,
    ctrl: false,
    alt: true,
    shift: false,
};
const ALT_UP: Key = Key {
    code: K::Up,
    ctrl: false,
    alt: true,
    shift: false,
};
const ALT_DOWN: Key = Key {
    code: K::Down,
    ctrl: false,
    alt: true,
    shift: false,
};

/// Every key binding, in help order.
pub const KEYMAP: &[Binding] = &[
    // Anywhere.
    b(
        Context::Global,
        &[Key::ch(' ')],
        Op::PlayPause,
        "play / pause",
    ),
    b(Context::Global, &[LEFT], Op::SeekBack, "seek back 5 s"),
    b(
        Context::Global,
        &[RIGHT],
        Op::SeekForward,
        "seek forward 5 s",
    ),
    b(
        Context::Global,
        &[Key::ch('N'), Key::shift(KeyCode::Right)],
        Op::Next,
        "next track",
    ),
    b(
        Context::Global,
        &[Key::ch('P'), Key::shift(KeyCode::Left)],
        Op::Previous,
        "previous track",
    ),
    b(
        Context::Global,
        &[Key::ch('X')],
        Op::SkipMix,
        "what N does: cut, or mix in (by the Transition tab's preset, or the one set in the settings)",
    ),
    b(Context::Global, &[Key::ch('S')], Op::Stop, "stop"),
    b(
        Context::Global,
        &[Key::plain(K::Tab)],
        Op::ToggleSide,
        "browser / queue / transition / karaoke / mixer",
    ),
    b(
        Context::Global,
        &[Key::ch('F')],
        Op::NowPlaying,
        "now playing, full screen",
    ),
    b(
        Context::Global,
        &[Key::ch('T')],
        Op::GoTrack,
        "go to the playing track",
    ),
    b(
        Context::Global,
        &[Key::ch('B')],
        Op::GoAlbum,
        "go to its album folder",
    ),
    b(
        Context::Global,
        &[Key::ch('R')],
        Op::GoArtist,
        "go to its artist folder",
    ),
    b(
        Context::Global,
        &[Key::ch('M')],
        Op::ToggleBpm,
        "tempo and beat pulse on / off",
    ),
    b(
        Context::Global,
        &[Key::alt('1')],
        Op::ToggleDrums,
        "stems: drums on / off (any off: the stems are heard, not the file)",
    ),
    b(
        Context::Global,
        &[Key::alt('2')],
        Op::ToggleBass,
        "stems: bass on / off",
    ),
    b(
        Context::Global,
        &[Key::alt('3')],
        Op::ToggleVocals,
        "stems: vocals on / off",
    ),
    b(
        Context::Global,
        &[Key::alt('4')],
        Op::ToggleOther,
        "stems: other on / off",
    ),
    b(
        Context::Global,
        &[Key::alt('0')],
        Op::AllStems,
        "stems: all on, the file as it is",
    ),
    b(
        Context::Global,
        &[ALT_UP],
        Op::VolumeUp,
        "device volume a step up (macOS: the system's keys do not reach it while played)",
    ),
    b(
        Context::Global,
        &[ALT_DOWN],
        Op::VolumeDown,
        "device volume a step down",
    ),
    b(Context::Global, &[Key::ch(',')], Op::Settings, "settings"),
    b(
        Context::Global,
        &[Key::ch('/')],
        Op::Filter,
        "filter this folder",
    ),
    b(
        Context::Global,
        &[Key::ch(':')],
        Op::Palette,
        "command palette",
    ),
    b(Context::Global, &[Key::ch('?')], Op::Help, "this help"),
    b(
        Context::Global,
        &[Key::ch('Q'), Key::ctrl('c')],
        Op::Quit,
        "quit",
    ),
    b(
        Context::Global,
        &[Key::ctrl('s')],
        Op::QueueSave,
        "save the queue as a playlist (among the saved queues)",
    ),
    b(
        Context::Global,
        &[Key::ctrl('l')],
        Op::QueueOpenSaved,
        "the saved queues, in the browser (enter queues one, alt+enter plays it instead)",
    ),
    // The list.
    b(Context::List, &[UP], Op::Up, "up"),
    b(Context::List, &[DOWN], Op::Down, "down"),
    b(
        Context::List,
        &[Key::plain(K::PageUp)],
        Op::PageUp,
        "page up",
    ),
    b(
        Context::List,
        &[Key::plain(K::PageDown)],
        Op::PageDown,
        "page down",
    ),
    b(Context::List, &[Key::plain(K::Home)], Op::Top, "first row"),
    b(
        Context::List,
        &[Key::plain(K::End), Key::ch('G')],
        Op::Bottom,
        "last row",
    ),
    b(
        Context::List,
        &[ENTER],
        Op::Open,
        "open folder; queue file or playlist (queue: play the track)",
    ),
    b(
        Context::List,
        &[SHIFT_ENTER],
        Op::QueuePlay,
        "queue file or playlist and play it, the queue kept (a folder: queue all of it)",
    ),
    b(
        Context::List,
        &[ALT_ENTER],
        Op::OpenReplace,
        "play file with its whole folder, or folder or playlist, in place of the queue",
    ),
    b(
        Context::Queue,
        &[Key::plain(K::Delete), BACKSPACE],
        Op::QueueRemove,
        "remove the track (the one playing: the next plays)",
    ),
    b(
        Context::Queue,
        &[Key::ch('K'), Key::shift(K::Up)],
        Op::QueueMoveUp,
        "move the track up (or drag it with the mouse)",
    ),
    b(
        Context::Queue,
        &[Key::ch('J'), Key::shift(K::Down)],
        Op::QueueMoveDown,
        "move the track down",
    ),
    b(Context::List, &[BACKSPACE], Op::Parent, "parent folder"),
    b(
        Context::List,
        &[ESC],
        Op::Back,
        "clear the filter or search, else parent folder",
    ),
    b(
        Context::List,
        &[Key::ctrl('v')],
        Op::Paste,
        "filter by the clipboard's text (a path goes there with enter)",
    ),
    // Typing a filter.
    b(
        Context::Filter,
        &[ENTER],
        Op::Accept,
        "go to the path typed, if it exists; else search all subfolders for it",
    ),
    b(
        Context::Filter,
        &[Key::plain(K::Tab)],
        Op::ToggleSide,
        "browser / queue (the filter stays)",
    ),
    b(Context::Filter, &[UP], Op::Up, "up"),
    b(Context::Filter, &[DOWN], Op::Down, "down"),
    b(
        Context::Filter,
        &[BACKSPACE],
        Op::Erase,
        "delete a character",
    ),
    b(Context::Filter, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(Context::Filter, &[ESC], Op::Back, "clear the filter"),
    // The palette.
    b(
        Context::Palette,
        &[ENTER],
        Op::Accept,
        "run the selected command",
    ),
    b(Context::Palette, &[UP], Op::Up, "up"),
    b(Context::Palette, &[DOWN], Op::Down, "down"),
    b(
        Context::Palette,
        &[BACKSPACE],
        Op::Erase,
        "delete a character",
    ),
    b(Context::Palette, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(Context::Palette, &[ESC], Op::Back, "close"),
    // The Transition tab.
    b(Context::Transition, &[UP], Op::Up, "the playing track"),
    b(Context::Transition, &[DOWN], Op::Down, "the next track"),
    b(Context::Transition, &[LEFT], Op::TransitionBarBack, "cursor a bar back"),
    b(Context::Transition, &[RIGHT], Op::TransitionBarForward, "cursor a bar on"),
    b(
        Context::Transition,
        &[ALT_LEFT],
        Op::TransitionBeatBack,
        "cursor a beat back",
    ),
    b(
        Context::Transition,
        &[ALT_RIGHT],
        Op::TransitionBeatForward,
        "cursor a beat on",
    ),
    b(
        Context::Transition,
        &[Key::shift(K::Left)],
        Op::TransitionSelectBack,
        "select bars back",
    ),
    b(
        Context::Transition,
        &[Key::shift(K::Right)],
        Op::TransitionSelectForward,
        "select bars on",
    ),
    b(
        Context::Transition,
        &[Key::plain(K::PageUp)],
        Op::PageUp,
        "eight bars back",
    ),
    b(
        Context::Transition,
        &[Key::plain(K::PageDown)],
        Op::PageDown,
        "eight bars on",
    ),
    b(
        Context::Transition,
        &[Key::ch('+'), Key::ch('=')],
        Op::TransitionZoomIn,
        "zoom in (fewer bars)",
    ),
    b(
        Context::Transition,
        &[Key::ch('-')],
        Op::TransitionZoomOut,
        "zoom out (more bars, then whole tracks)",
    ),
    b(
        Context::Transition,
        &[Key::ch('z')],
        Op::TransitionFitLyrics,
        "zoom the track selected to fit its lyrics",
    ),
    b(
        Context::Transition,
        &[Key::ch('.')],
        Op::TransitionFollow,
        "follow the playhead again",
    ),
    b(
        Context::Transition,
        &[Key::ch('L')],
        Op::TransitionLyrics,
        "the lyrics on the vocals lane on/off",
    ),
    // Bar jumps (help: two lines, [`help_sections`]): 5 the bar heard, 6 the
    // next, 4 the one before, when the bar heard ends; with shift at once.
    b(Context::Transition, &[Key::ch('1')], Op::JumpBarAtEnd(-4), ""),
    b(Context::Transition, &[Key::ch('2')], Op::JumpBarAtEnd(-3), ""),
    b(Context::Transition, &[Key::ch('3')], Op::JumpBarAtEnd(-2), ""),
    b(Context::Transition, &[Key::ch('4')], Op::JumpBarAtEnd(-1), ""),
    b(Context::Transition, &[Key::ch('5')], Op::JumpBarAtEnd(0), ""),
    b(Context::Transition, &[Key::ch('6')], Op::JumpBarAtEnd(1), ""),
    b(Context::Transition, &[Key::ch('7')], Op::JumpBarAtEnd(2), ""),
    b(Context::Transition, &[Key::ch('8')], Op::JumpBarAtEnd(3), ""),
    b(Context::Transition, &[Key::ch('9')], Op::JumpBarAtEnd(4), ""),
    b(
        Context::Transition,
        &[Key::shifted('1')],
        Op::JumpBar(-4),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('2')],
        Op::JumpBar(-3),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('3')],
        Op::JumpBar(-2),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('4')],
        Op::JumpBar(-1),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('5')],
        Op::JumpBar(0),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('6')],
        Op::JumpBar(1),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('7')],
        Op::JumpBar(2),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('8')],
        Op::JumpBar(3),
        "",
    ),
    b(
        Context::Transition,
        &[Key::shifted('9')],
        Op::JumpBar(4),
        "",
    ),
    b(
        Context::Transition,
        &[ENTER],
        Op::Accept,
        "transition cue on this track: the selected bars, else 8 from the cursor",
    ),
    b(
        Context::Transition,
        &[Key::ch('[')],
        Op::GridEarlier,
        "grid: bars start a beat earlier",
    ),
    b(
        Context::Transition,
        &[Key::ch(']')],
        Op::GridLater,
        "grid: bars start a beat later",
    ),
    b(
        Context::Transition,
        &[Key::ch('{')],
        Op::GridNudgeBack,
        "grid: every beat 5 ms earlier",
    ),
    b(
        Context::Transition,
        &[Key::ch('}')],
        Op::GridNudgeForward,
        "grid: every beat 5 ms later",
    ),
    b(
        Context::Transition,
        &[Key::ch('<')],
        Op::GridHalf,
        "grid: half tempo",
    ),
    b(
        Context::Transition,
        &[Key::ch('>')],
        Op::GridDouble,
        "grid: double tempo",
    ),
    b(
        Context::Transition,
        &[Key::ch('D')],
        Op::GridOne,
        "grid: a bar starts on the cursor's beat (bars of four)",
    ),
    b(
        Context::Transition,
        &[Key::ch('U')],
        Op::GridReset,
        "grid: undo the fixes, back to the analysis",
    ),
    b(
        Context::Transition,
        &[Key::ch('g')],
        Op::TransitionGo,
        "transition: arm it at the two cues, or call it off",
    ),
    b(
        Context::Transition,
        &[Key::ch('p')],
        Op::TransitionPreset,
        "transition: next preset (fade, drums first, bass swap, vocals last)",
    ),
    b(
        Context::Transition,
        &[Key::ch('m')],
        Op::TransitionTempo,
        "transition: varispeed / time-stretch",
    ),
    b(
        Context::Transition,
        &[Key::ch('r')],
        Op::TransitionPairing,
        "transition: beats paired by the tempi, 1:1, 2:1 (the next at half tempo), 1:2 (at double)",
    ),
    b(
        Context::Transition,
        &[Key::ch('e')],
        Op::TransitionEdit,
        "transition: edit the fades",
    ),
    b(
        Context::Transition,
        &[ESC],
        Op::Back,
        "clear the selection, else call off a transition armed, else hide it",
    ),
    // Editing the fades.
    b(Context::Fades, &[UP], Op::Up, "lane above"),
    b(Context::Fades, &[DOWN], Op::Down, "lane below"),
    b(Context::Fades, &[LEFT], Op::EditBack, "bar before"),
    b(Context::Fades, &[RIGHT], Op::EditForward, "bar after"),
    b(
        Context::Fades,
        &[Key::ch('+'), Key::ch('=')],
        Op::EditRaise,
        "a quarter louder at this bar",
    ),
    b(
        Context::Fades,
        &[Key::ch('-')],
        Op::EditLower,
        "a quarter quieter at this bar",
    ),
    b(
        Context::Fades,
        &[Key::ch('0')],
        Op::EditOff,
        "off at this bar",
    ),
    b(
        Context::Fades,
        &[Key::ch('1')],
        Op::EditFull,
        "full at this bar",
    ),
    b(
        Context::Fades,
        &[Key::ch('c')],
        Op::EditCut,
        "this bar cuts to its gain, or fades to it",
    ),
    b(
        Context::Fades,
        &[ESC, Key::ch('e')],
        Op::Back,
        "stop editing",
    ),
    // The Karaoke tab.
    b(
        Context::Karaoke,
        &[ENTER],
        Op::KaraokeEnter,
        "enter Karaoke mode (fullscreen; esc there comes back)",
    ),
    b(
        Context::Karaoke,
        &[SHIFT_ENTER],
        Op::KaraokeEnterHere,
        "enter it where the song is: no rewind, no count-in (also shift+click)",
    ),
    b(
        Context::Karaoke,
        &[Key::ch('i')],
        Op::MicNext,
        "the next microphone (input device)",
    ),
    b(
        Context::Karaoke,
        &[Key::ch('r')],
        Op::KaraokeRecordEcho,
        "record 30 s of the microphone, the music and the voice cleaned (to tune the echo canceller)",
    ),
    b(
        Context::Karaoke,
        &[Key::ch('g')],
        Op::KaraokeGame,
        "the game on or off: the singing scored, or just sung",
    ),
    b(
        Context::Karaoke,
        &[Key::ch('p')],
        Op::KaraokeNotes,
        "show or hide the song's notes under the line",
    ),
    b(Context::Karaoke, &[UP], Op::Up, "the row above (voice, delay, key)"),
    b(Context::Karaoke, &[DOWN], Op::Down, "the row below"),
    b(
        Context::Karaoke,
        &[Key::ch('s')],
        Op::KaraokeSync,
        "sync: clicks to answer with “ta” (or a clap), to measure how late your voice comes (a microphone open)",
    ),
    b(
        Context::Karaoke,
        &[LEFT],
        Op::SliderDown,
        "the row's value a step down: the voice 1 dB (at −45 dB or below: off), the delay 10 ms (its nudge), the key a semitone (↑ ↓ pick the row)",
    ),
    b(
        Context::Karaoke,
        &[RIGHT],
        Op::SliderUp,
        "the row's value a step up",
    ),
    b(
        Context::Karaoke,
        &[Key::shift(K::Left)],
        Op::SliderFineDown,
        "the voice 1 % of its travel down",
    ),
    b(
        Context::Karaoke,
        &[Key::shift(K::Right)],
        Op::SliderFineUp,
        "the voice 1 % of its travel up",
    ),
    b(Context::Karaoke, &[Key::ch('-')], Op::KeyDown, "the key a semitone down (the tempo kept; not the drums)"),
    b(Context::Karaoke, &[Key::ch('+'), Key::ch('=')], Op::KeyUp, "the key a semitone up"),
    b(Context::Karaoke, &[Key::ch('0')], Op::KeyReset, "the song's own key (the Delay row at hand: not nudged)"),
    // Syncing.
    b(
        Context::Sync,
        &[ENTER],
        Op::SyncKeep,
        "keep the delay measured, for this output and microphone",
    ),
    b(Context::Sync, &[Key::ch('r')], Op::SyncAgain, "again"),
    b(Context::Sync, &[ESC], Op::Back, "cancel"),
    // Karaoke mode.
    b(
        Context::KaraokeMode,
        &[ENTER],
        Op::KaraokeStart,
        "START: the song after a count-in in its tempo (space too)",
    ),
    b(
        Context::KaraokeMode,
        &[ESC],
        Op::Back,
        "leave Karaoke mode, back to the tab (the stems as before)",
    ),
    b(
        Context::KaraokeMode,
        &[Key::ch('i')],
        Op::MicNext,
        "the next microphone",
    ),
    b(
        Context::KaraokeMode,
        &[Key::ch('r')],
        Op::KaraokeRecordEcho,
        "record 30 s for tuning the echo canceller",
    ),
    b(
        Context::KaraokeMode,
        &[Key::ch('p')],
        Op::KaraokeDemo,
        "hold: the demo sings the song perfectly, AUTO showing (its score no best, its singing not kept)",
    ),
    b(
        Context::KaraokeMode,
        &[LEFT],
        Op::SliderDown,
        "the voice 1 dB down (at −45 dB or below: off)",
    ),
    b(
        Context::KaraokeMode,
        &[RIGHT],
        Op::SliderUp,
        "the voice 1 dB up",
    ),
    b(
        Context::KaraokeMode,
        &[Key::shift(K::Left)],
        Op::SliderFineDown,
        "the voice 1 % of its travel down",
    ),
    b(
        Context::KaraokeMode,
        &[Key::shift(K::Right)],
        Op::SliderFineUp,
        "the voice 1 % of its travel up",
    ),
    // The Mixer tab.
    b(
        Context::Desk,
        &[LEFT],
        Op::DeskPrevStrip,
        "the strip to the left (player, preview, microphone, master)",
    ),
    b(
        Context::Desk,
        &[RIGHT],
        Op::DeskNextStrip,
        "the strip to the right",
    ),
    b(
        Context::Desk,
        &[UP],
        Op::Up,
        "the fader 1 dB up (the master: the player's volume, up to 0 dB)",
    ),
    b(
        Context::Desk,
        &[DOWN],
        Op::Down,
        "the fader 1 dB down (at −45 dB or below: off)",
    ),
    b(
        Context::Desk,
        &[Key::plain(K::PageUp)],
        Op::PageUp,
        "the fader 6 dB up",
    ),
    b(
        Context::Desk,
        &[Key::plain(K::PageDown)],
        Op::PageDown,
        "the fader 6 dB down",
    ),
    b(
        Context::Desk,
        &[Key::plain(K::Home)],
        Op::Top,
        "the fader to 0 dB: nothing added",
    ),
    b(
        Context::Desk,
        &[Key::plain(K::End)],
        Op::Bottom,
        "the fader off (−∞)",
    ),
    b(
        Context::Desk,
        &[Key::ch('m')],
        Op::DeskMute,
        "mute the strip, or not",
    ),
    b(
        Context::Desk,
        &[Key::ch('s')],
        Op::DeskSolo,
        "solo the strip, or not (while one is soloed, the others are silent)",
    ),
    b(
        Context::Desk,
        &[Key::ch('e')],
        Op::DeskEcho,
        "the microphone's echo canceller on or off",
    ),
    b(
        Context::Desk,
        &[Key::ch('i')],
        Op::MicNext,
        "the next microphone (input device)",
    ),
    b(Context::Desk, &[Key::ch('-')], Op::KeyDown, "the song a semitone down (on the PLAYER strip)"),
    b(Context::Desk, &[Key::ch('+'), Key::ch('=')], Op::KeyUp, "the key a semitone up"),
    b(Context::Desk, &[Key::ch('0')], Op::KeyReset, "the song's own key"),
    // Now playing.
    b(Context::NowPlaying, &[ESC], Op::Back, "back"),
    // Settings.
    b(Context::Settings, &[UP], Op::Up, "up"),
    b(Context::Settings, &[DOWN], Op::Down, "down"),
    b(Context::Settings, &[LEFT], Op::ValueDown, "previous value"),
    b(Context::Settings, &[RIGHT], Op::ValueUp, "next value"),
    b(
        Context::Settings,
        &[ENTER],
        Op::Open,
        "change / open the device list",
    ),
    b(Context::Settings, &[ESC, Key::ch(',')], Op::Back, "back"),
    b(Context::DeviceList, &[UP], Op::Up, "up"),
    b(Context::DeviceList, &[DOWN], Op::Down, "down"),
    b(
        Context::DeviceList,
        &[ENTER],
        Op::Open,
        "play on this device",
    ),
    b(Context::DeviceList, &[ESC], Op::Back, "close the list"),
    b(Context::FolderList, &[UP], Op::Up, "up"),
    b(Context::FolderList, &[DOWN], Op::Down, "down"),
    b(
        Context::FolderList,
        &[ENTER, RIGHT],
        Op::Open,
        "open the folder, or move the data folder into the one shown",
    ),
    b(Context::FolderList, &[BACKSPACE, LEFT], Op::Parent, "up a folder"),
    b(Context::FolderList, &[ESC], Op::Back, "close the list"),
    // Help.
    b(Context::Help, &[UP], Op::Up, "scroll up"),
    b(Context::Help, &[DOWN], Op::Down, "scroll down"),
    b(
        Context::Help,
        &[Key::plain(K::PageUp)],
        Op::PageUp,
        "page up",
    ),
    b(
        Context::Help,
        &[Key::plain(K::PageDown)],
        Op::PageDown,
        "page down",
    ),
    b(
        Context::Help,
        &[Key::ch('/')],
        Op::Filter,
        "search this help (keys and text)",
    ),
    b(
        Context::Help,
        &[ESC, Key::ch('?')],
        Op::Back,
        "clear the search, else back",
    ),
    // Typing a search of the help.
    b(
        Context::HelpSearch,
        &[ENTER],
        Op::Accept,
        "keep the search, stop typing",
    ),
    b(Context::HelpSearch, &[UP], Op::Up, "scroll up"),
    b(Context::HelpSearch, &[DOWN], Op::Down, "scroll down"),
    b(
        Context::HelpSearch,
        &[Key::plain(K::PageUp)],
        Op::PageUp,
        "page up",
    ),
    b(
        Context::HelpSearch,
        &[Key::plain(K::PageDown)],
        Op::PageDown,
        "page down",
    ),
    b(
        Context::HelpSearch,
        &[BACKSPACE],
        Op::Erase,
        "delete a character",
    ),
    b(Context::HelpSearch, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(Context::HelpSearch, &[ESC], Op::Back, "clear the search"),
];

/// The contexts in force, most specific first.
pub fn contexts(state: &State) -> Vec<Context> {
    if state.palette.is_some() {
        return vec![Context::Palette];
    }
    match state.screen {
        Screen::Main => {
            if state.browser.filter.as_ref().is_some_and(|f| f.editing) {
                vec![Context::Filter]
            } else if state.side == Side::Transition && state.transition.draft.edit.is_some() {
                vec![Context::Fades, Context::Global]
            } else if state.side == Side::Transition {
                vec![Context::Transition, Context::Global]
            } else if state.side == Side::Karaoke && state.karaoke.sync.is_some() {
                vec![Context::Sync, Context::Global]
            } else if state.side == Side::Karaoke {
                vec![Context::Karaoke, Context::Global]
            } else if state.side == Side::Mixer {
                vec![Context::Desk, Context::Global]
            } else if state.side == Side::Queue {
                vec![Context::Queue, Context::List, Context::Global]
            } else {
                vec![Context::List, Context::Global]
            }
        }
        Screen::NowPlaying => vec![Context::NowPlaying, Context::Global],
        Screen::Settings if state.settings.folders.is_some() => {
            vec![Context::FolderList, Context::Global]
        }
        Screen::Settings if state.settings.picking => vec![Context::DeviceList, Context::Global],
        Screen::Settings => vec![Context::Settings, Context::Global],
        Screen::Help if state.help_search.as_ref().is_some_and(|s| s.editing) => {
            vec![Context::HelpSearch]
        }
        Screen::Help => vec![Context::Help, Context::Global],
        Screen::Karaoke => vec![Context::KaraokeMode, Context::Global],
    }
}

/// What a key means now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meaning {
    /// An operation.
    Op(Op),
    /// A character to type into the filter or the palette.
    Type(char),
}

fn find(context: Context, key: Key) -> Option<Op> {
    KEYMAP
        .iter()
        .find(|binding| binding.context == context && binding.keys.contains(&key))
        .map(|binding| binding.op)
}

/// Whether an unmodified `c` jumps to a name in the list (as typing does
/// in a file explorer) rather than running its command: on the main
/// screen, in the browser and in the queue, lowercase letters, digits and
/// the other name characters do; capitals, `,`, `/`, `:` and `?` keep
/// their commands. Space is part of the name while one is being typed
/// (within [`TYPE_AHEAD_MS`] of the last letter), and play/pause otherwise.
fn types_ahead(state: &State, c: char) -> bool {
    state.palette.is_none()
        && state.screen == Screen::Main
        && !state.side.wide()
        && !state.browser.filter.as_ref().is_some_and(|f| f.editing)
        && !c.is_uppercase()
        && !c.is_control()
        && (c != ' ' || typing_a_name(state))
        && !matches!(c, ',' | '/' | ':' | '?')
}

/// Whether a name is being typed: something was, less than
/// [`TYPE_AHEAD_MS`] ago.
fn typing_a_name(state: &State) -> bool {
    !state.browser.typed.is_empty()
        && state.now_ms.saturating_sub(state.browser.typed_at) <= TYPE_AHEAD_MS
}

/// What a key does in `state`, as an action.
pub fn action(state: &State, key: Key) -> Option<crate::core::Action> {
    use crate::core::Action;
    match lookup(state, key)? {
        Meaning::Op(op) => Some(Action::Op(op)),
        Meaning::Type(c) => Some(Action::Type(c)),
    }
}

/// What `key` does in `state`, if anything.
pub fn lookup(state: &State, key: Key) -> Option<Meaning> {
    // Ctrl held for a preview: its moves and `enter` are the keys alone
    // (no other binding has ctrl with them).
    let moves = matches!(
        key.code,
        KeyCode::Up
            | KeyCode::Down
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::Enter
    );
    let key = if key.ctrl && moves && crate::core::preview::held(state) {
        Key { ctrl: false, ..key }
    } else {
        key
    };
    if let KeyCode::Char(c) = key.code {
        if !key.ctrl && !key.alt && types_ahead(state, c) {
            return Some(Meaning::Type(c));
        }
    }
    let contexts = contexts(state);
    let typing = contexts.first().is_some_and(|c| c.types());
    for &context in &contexts {
        if let Some(op) = find(context, key) {
            return Some(Meaning::Op(op));
        }
    }
    if typing {
        if key.ctrl {
            if let Some(op) = find(Context::Global, key) {
                return Some(Meaning::Op(op));
            }
        }
        if let KeyCode::Char(c) = key.code {
            if !key.ctrl && !key.alt {
                return Some(Meaning::Type(c));
            }
        }
    }
    None
}

/// The keys bound to `op` in `context`, written for people (`←/h`).
pub fn keys_label(binding: &Binding) -> String {
    binding
        .keys
        .iter()
        .map(|k| k.label())
        .collect::<Vec<_>>()
        .join(" / ")
}

/// One section of the help screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HelpSection {
    /// The context title.
    pub title: &'static str,
    /// (keys, what they do).
    pub lines: Vec<(String, &'static str)>,
}

/// The help screen, from [`KEYMAP`], in table order.
pub fn help_sections() -> Vec<HelpSection> {
    let mut sections: Vec<(Context, HelpSection)> = Vec::new();
    for binding in KEYMAP {
        if binding.help.is_empty() {
            // The bar jumps: two lines for eighteen keys.
            if binding.op == Op::JumpBar(0) || binding.op == Op::JumpBarAtEnd(0) {
                let line = if binding.op == Op::JumpBarAtEnd(0) {
                    (
                        "1 … 9".to_owned(),
                        "when the bar heard ends, jump to a bar's start: 5 the bar heard (it plays again), 6 the next, 4 the one before …",
                    )
                } else {
                    (
                        "shift+1 … 9".to_owned(),
                        "the same at once (5: the bar heard starts again)",
                    )
                };
                if let Some((_, section)) = sections.iter_mut().find(|(c, _)| *c == binding.context)
                {
                    section.lines.push(line);
                }
            }
            continue;
        }
        let line = (keys_label(binding), binding.help);
        match sections.iter_mut().find(|(c, _)| *c == binding.context) {
            Some((_, section)) => section.lines.push(line),
            None => sections.push((
                binding.context,
                HelpSection {
                    title: binding.context.title(),
                    lines: vec![line],
                },
            )),
        }
    }
    let mut out: Vec<HelpSection> = sections.into_iter().map(|(_, s)| s).collect();
    out.push(HelpSection {
        title: "Typing in the lists",
        lines: vec![
            (
                "a-z 0-9 …".into(),
                "jump to the first name (as shown) starting with what you type",
            ),
            (
                "the same letter".into(),
                "again: the next name starting with it",
            ),
            (
                "capitals".into(),
                "the commands (Q quit, N next, S stop …); lowercase letters only type",
            ),
        ],
    });
    out.push(HelpSection {
        title: "Preview (hold ctrl in the browser)",
        lines: vec![
            (
                "hold ctrl".into(),
                "the song playing fades out and waits; the file selected plays from a third in",
            ),
            (
                "ctrl+↑ / ↓ / pgup / pgdn / home / end".into(),
                "the file selected next plays from its third (a folder: silence)",
            ),
            (
                "ctrl+click / pointer / wheel".into(),
                "the same with the mouse: the row under the pointer is selected",
            ),
            (
                "release ctrl".into(),
                "back to the song that was playing, where it was (paused stays paused)",
            ),
            (
                "ctrl+enter".into(),
                "queue the file (as enter) and go back; nothing previewed is queued",
            ),
        ],
    });
    out.push(HelpSection {
        title: "Mouse",
        lines: vec![
            (
                "click".into(),
                "select a row; Browser / Queue switches the list",
            ),
            (
                "double click".into(),
                "as enter: open folder; queue file or playlist (queue: play the track)",
            ),
            ("right click".into(), "queue file or folder"),
            ("wheel".into(), "scroll the list"),
            ("click / drag".into(), "the progress bar seeks"),
            (
                "click".into(),
                "drums / bass / vocals / other switch a stem on or off (with a GPU)",
            ),
            (
                "click".into(),
                "the cover opens Now playing; the device name its setting",
            ),
        ],
    });
    out
}

/// The help screen as a search after `/` leaves it: only the lines whose
/// keys or text contain `query` (case ignored, as the browser's filter),
/// each under its section's title. Sections left empty go; an empty query
/// keeps everything.
pub fn help_matching(query: &str) -> Vec<HelpSection> {
    let needle = query.to_lowercase();
    help_sections()
        .into_iter()
        .filter_map(|mut section| {
            section.lines.retain(|(keys, text)| {
                keys.to_lowercase().contains(&needle) || text.to_lowercase().contains(&needle)
            });
            (!section.lines.is_empty()).then_some(section)
        })
        .collect()
}

/// One palette entry: an operation and its key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteEntry {
    /// The operation.
    pub op: Op,
    /// Its name.
    pub name: &'static str,
    /// Its key, written for people.
    pub keys: String,
}

/// Every command the palette offers, from [`KEYMAP`] (each operation
/// once, with its first binding's keys).
pub fn palette_entries() -> Vec<PaletteEntry> {
    let mut out: Vec<PaletteEntry> = Vec::new();
    for binding in KEYMAP {
        if !binding.op.in_palette() || out.iter().any(|e| e.op == binding.op) {
            continue;
        }
        out.push(PaletteEntry {
            op: binding.op,
            name: binding.op.name(),
            keys: keys_label(binding),
        });
    }
    out
}

/// The palette entries matching `text` (every word of it, case ignored).
pub fn palette_matches(text: &str) -> Vec<PaletteEntry> {
    let words: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
    palette_entries()
        .into_iter()
        .filter(|e| {
            let name = e.name.to_lowercase();
            words.iter().all(|w| name.contains(w.as_str()))
        })
        .collect()
}
