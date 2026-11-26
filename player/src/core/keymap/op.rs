//! The operations ([`Op`]): what keys, buttons and palette entries do,
//! and how the palette names them.

use jodel_playback::Stem;

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
