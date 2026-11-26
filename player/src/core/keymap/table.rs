//! The table ([`KEYMAP`]): every key binding, a row each, in help order.
//! A row is where it applies, its keys (any of them), what they do, and
//! the help's words for it; a row without words is not in the help (the
//! bar jumps: the first of each nine says it for all of them,
//! [`Binding::said`]).

use super::Context::*;
use super::{Binding, Context, Key, KeyCode as K, Op};

const fn b(context: Context, keys: &'static [Key], op: Op, help: &'static str) -> Binding {
    Binding {
        context,
        keys,
        op,
        help,
        help_keys: None,
    }
}

/// A key other than a character, with alt.
const fn alt(code: K) -> Key {
    Key {
        code,
        ctrl: false,
        alt: true,
        shift: false,
    }
}

const UP: Key = Key::plain(K::Up);
const DOWN: Key = Key::plain(K::Down);
const LEFT: Key = Key::plain(K::Left);
const RIGHT: Key = Key::plain(K::Right);
const PGUP: Key = Key::plain(K::PageUp);
const PGDN: Key = Key::plain(K::PageDown);
const HOME: Key = Key::plain(K::Home);
const END: Key = Key::plain(K::End);
const ENTER: Key = Key::plain(K::Enter);
const ESC: Key = Key::plain(K::Esc);
const TAB: Key = Key::plain(K::Tab);
const BACKSPACE: Key = Key::plain(K::Backspace);
const DELETE: Key = Key::plain(K::Delete);
const SHIFT_UP: Key = Key::shift(K::Up);
const SHIFT_DOWN: Key = Key::shift(K::Down);
const SHIFT_LEFT: Key = Key::shift(K::Left);
const SHIFT_RIGHT: Key = Key::shift(K::Right);
const SHIFT_ENTER: Key = Key::shift(K::Enter);
const ALT_UP: Key = alt(K::Up);
const ALT_DOWN: Key = alt(K::Down);
const ALT_LEFT: Key = alt(K::Left);
const ALT_RIGHT: Key = alt(K::Right);
const ALT_ENTER: Key = alt(K::Enter);

const fn ch(c: char) -> Key {
    Key::ch(c)
}

/// Every key binding, in help order.
#[rustfmt::skip]
pub const KEYMAP: &[Binding] = &[
    // Anywhere.
    b(Global, &[ch(' ')], Op::PlayPause, "play / pause"),
    b(Global, &[LEFT], Op::SeekBack, "seek back 5 s"),
    b(Global, &[RIGHT], Op::SeekForward, "seek forward 5 s"),
    b(Global, &[ch('N'), SHIFT_RIGHT], Op::Next, "next track"),
    b(Global, &[ch('P'), SHIFT_LEFT], Op::Previous, "previous track"),
    b(Global, &[ch('X')], Op::SkipMix, "what N does: cut, or mix in (by the Transition tab's preset, or the one set in the settings)"),
    b(Global, &[ch('S')], Op::Stop, "stop"),
    b(Global, &[TAB], Op::ToggleSide, "browser / queue / transition / karaoke / mixer"),
    b(Global, &[ch('F')], Op::NowPlaying, "now playing, full screen"),
    b(Global, &[ch('T')], Op::GoTrack, "go to the playing track"),
    b(Global, &[ch('B')], Op::GoAlbum, "go to its album folder"),
    b(Global, &[ch('R')], Op::GoArtist, "go to its artist folder"),
    b(Global, &[ch('M')], Op::ToggleBpm, "tempo and beat pulse on / off"),
    b(Global, &[Key::alt('1')], Op::ToggleDrums, "stems: drums on / off (any off: the stems are heard, not the file)"),
    b(Global, &[Key::alt('2')], Op::ToggleBass, "stems: bass on / off"),
    b(Global, &[Key::alt('3')], Op::ToggleVocals, "stems: vocals on / off"),
    b(Global, &[Key::alt('4')], Op::ToggleOther, "stems: other on / off"),
    b(Global, &[Key::alt('0')], Op::AllStems, "stems: all on, the file as it is"),
    b(Global, &[ALT_UP], Op::VolumeUp, "device volume a step up (macOS: the system's keys do not reach it while played)"),
    b(Global, &[ALT_DOWN], Op::VolumeDown, "device volume a step down"),
    b(Global, &[ch(',')], Op::Settings, "settings"),
    b(Global, &[ch('/')], Op::Filter, "filter this folder"),
    b(Global, &[ch(':')], Op::Palette, "command palette"),
    b(Global, &[ch('?')], Op::Help, "this help"),
    b(Global, &[ch('Q'), Key::ctrl('c')], Op::Quit, "quit"),
    b(Global, &[Key::ctrl('s')], Op::QueueSave, "save the queue as a playlist (among the saved queues)"),
    b(Global, &[Key::ctrl('l')], Op::QueueOpenSaved, "the saved queues, in the browser (enter queues one, alt+enter plays it instead)"),
    // The list (the browser and the queue).
    b(List, &[UP], Op::Up, "up"),
    b(List, &[DOWN], Op::Down, "down"),
    b(List, &[PGUP], Op::PageUp, "page up"),
    b(List, &[PGDN], Op::PageDown, "page down"),
    b(List, &[HOME], Op::Top, "first row"),
    b(List, &[END, ch('G')], Op::Bottom, "last row"),
    b(List, &[ENTER], Op::Open, "open folder; queue file or playlist (queue: play the track)"),
    b(List, &[SHIFT_ENTER], Op::QueuePlay, "queue file or playlist and play it, the queue kept (a folder: queue all of it)"),
    b(List, &[ALT_ENTER], Op::OpenReplace, "play file with its whole folder, or folder or playlist, in place of the queue"),
    b(Queue, &[DELETE, BACKSPACE], Op::QueueRemove, "remove the track (the one playing: the next plays)"),
    b(Queue, &[ch('K'), SHIFT_UP], Op::QueueMoveUp, "move the track up (or drag it with the mouse)"),
    b(Queue, &[ch('J'), SHIFT_DOWN], Op::QueueMoveDown, "move the track down"),
    b(List, &[BACKSPACE], Op::Parent, "parent folder"),
    b(List, &[ESC], Op::Back, "clear the filter or search, else parent folder"),
    b(List, &[Key::ctrl('v')], Op::Paste, "filter by the clipboard's text (a path goes there with enter)"),
    // Typing a filter.
    b(Filter, &[ENTER], Op::Accept, "go to the path typed, if it exists; else search all subfolders for it"),
    b(Filter, &[TAB], Op::ToggleSide, "browser / queue (the filter stays)"),
    b(Filter, &[UP], Op::Up, "up"),
    b(Filter, &[DOWN], Op::Down, "down"),
    b(Filter, &[BACKSPACE], Op::Erase, "delete a character"),
    b(Filter, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(Filter, &[ESC], Op::Back, "clear the filter"),
    // The palette.
    b(Palette, &[ENTER], Op::Accept, "run the selected command"),
    b(Palette, &[UP], Op::Up, "up"),
    b(Palette, &[DOWN], Op::Down, "down"),
    b(Palette, &[BACKSPACE], Op::Erase, "delete a character"),
    b(Palette, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(Palette, &[ESC], Op::Back, "close"),
    // The Transition tab.
    b(Transition, &[UP], Op::Up, "the playing track"),
    b(Transition, &[DOWN], Op::Down, "the next track"),
    b(Transition, &[LEFT], Op::TransitionBarBack, "cursor a bar back"),
    b(Transition, &[RIGHT], Op::TransitionBarForward, "cursor a bar on"),
    b(Transition, &[ALT_LEFT], Op::TransitionBeatBack, "cursor a beat back"),
    b(Transition, &[ALT_RIGHT], Op::TransitionBeatForward, "cursor a beat on"),
    b(Transition, &[SHIFT_LEFT], Op::TransitionSelectBack, "select bars back"),
    b(Transition, &[SHIFT_RIGHT], Op::TransitionSelectForward, "select bars on"),
    b(Transition, &[PGUP], Op::PageUp, "eight bars back"),
    b(Transition, &[PGDN], Op::PageDown, "eight bars on"),
    b(Transition, &[ch('+'), ch('=')], Op::TransitionZoomIn, "zoom in (fewer bars)"),
    b(Transition, &[ch('-')], Op::TransitionZoomOut, "zoom out (more bars, then whole tracks)"),
    b(Transition, &[ch('z')], Op::TransitionFitLyrics, "zoom the track selected to fit its lyrics"),
    b(Transition, &[ch('.')], Op::TransitionFollow, "follow the playhead again"),
    b(Transition, &[ch('L')], Op::TransitionLyrics, "the lyrics on the vocals lane on/off"),
    // The bar jumps: 5 the bar heard, 6 the next, 4 the one before, when
    // the bar heard ends; with shift at once.
    b(Transition, &[ch('1')], Op::JumpBarAtEnd(-4), "when the bar heard ends, jump to a bar's start: 5 the bar heard (it plays again), 6 the next, 4 the one before …").said("1 … 9"),
    b(Transition, &[ch('2')], Op::JumpBarAtEnd(-3), ""),
    b(Transition, &[ch('3')], Op::JumpBarAtEnd(-2), ""),
    b(Transition, &[ch('4')], Op::JumpBarAtEnd(-1), ""),
    b(Transition, &[ch('5')], Op::JumpBarAtEnd(0), ""),
    b(Transition, &[ch('6')], Op::JumpBarAtEnd(1), ""),
    b(Transition, &[ch('7')], Op::JumpBarAtEnd(2), ""),
    b(Transition, &[ch('8')], Op::JumpBarAtEnd(3), ""),
    b(Transition, &[ch('9')], Op::JumpBarAtEnd(4), ""),
    b(Transition, &[Key::shifted('1')], Op::JumpBar(-4), "the same at once (5: the bar heard starts again)").said("shift+1 … 9"),
    b(Transition, &[Key::shifted('2')], Op::JumpBar(-3), ""),
    b(Transition, &[Key::shifted('3')], Op::JumpBar(-2), ""),
    b(Transition, &[Key::shifted('4')], Op::JumpBar(-1), ""),
    b(Transition, &[Key::shifted('5')], Op::JumpBar(0), ""),
    b(Transition, &[Key::shifted('6')], Op::JumpBar(1), ""),
    b(Transition, &[Key::shifted('7')], Op::JumpBar(2), ""),
    b(Transition, &[Key::shifted('8')], Op::JumpBar(3), ""),
    b(Transition, &[Key::shifted('9')], Op::JumpBar(4), ""),
    b(Transition, &[ENTER], Op::Accept, "transition cue on this track: the selected bars, else 8 from the cursor"),
    b(Transition, &[ch('[')], Op::GridEarlier, "grid: bars start a beat earlier"),
    b(Transition, &[ch(']')], Op::GridLater, "grid: bars start a beat later"),
    b(Transition, &[ch('{')], Op::GridNudgeBack, "grid: every beat 5 ms earlier"),
    b(Transition, &[ch('}')], Op::GridNudgeForward, "grid: every beat 5 ms later"),
    b(Transition, &[ch('<')], Op::GridHalf, "grid: half tempo"),
    b(Transition, &[ch('>')], Op::GridDouble, "grid: double tempo"),
    b(Transition, &[ch('D')], Op::GridOne, "grid: a bar starts on the cursor's beat (bars of four)"),
    b(Transition, &[ch('U')], Op::GridReset, "grid: undo the fixes, back to the analysis"),
    b(Transition, &[ch('g')], Op::TransitionGo, "transition: arm it at the two cues, or call it off"),
    b(Transition, &[ch('p')], Op::TransitionPreset, "transition: next preset (fade, drums first, bass swap, vocals last)"),
    b(Transition, &[ch('m')], Op::TransitionTempo, "transition: varispeed / time-stretch"),
    b(Transition, &[ch('r')], Op::TransitionPairing, "transition: beats paired by the tempi, 1:1, 2:1 (the next at half tempo), 1:2 (at double)"),
    b(Transition, &[ch('e')], Op::TransitionEdit, "transition: edit the fades"),
    b(Transition, &[ESC], Op::Back, "clear the selection, else call off a transition armed, else hide it"),
    // Editing the fades.
    b(Fades, &[UP], Op::Up, "lane above"),
    b(Fades, &[DOWN], Op::Down, "lane below"),
    b(Fades, &[LEFT], Op::EditBack, "bar before"),
    b(Fades, &[RIGHT], Op::EditForward, "bar after"),
    b(Fades, &[ch('+'), ch('=')], Op::EditRaise, "a quarter louder at this bar"),
    b(Fades, &[ch('-')], Op::EditLower, "a quarter quieter at this bar"),
    b(Fades, &[ch('0')], Op::EditOff, "off at this bar"),
    b(Fades, &[ch('1')], Op::EditFull, "full at this bar"),
    b(Fades, &[ch('c')], Op::EditCut, "this bar cuts to its gain, or fades to it"),
    b(Fades, &[ESC, ch('e')], Op::Back, "stop editing"),
    // The Karaoke tab.
    b(Karaoke, &[ENTER], Op::KaraokeEnter, "enter Karaoke mode (fullscreen; esc there comes back)"),
    b(Karaoke, &[SHIFT_ENTER], Op::KaraokeEnterHere, "enter it where the song is: no rewind, no count-in (also shift+click)"),
    b(Karaoke, &[ch('i')], Op::MicNext, "the next microphone (input device)"),
    b(Karaoke, &[ch('r')], Op::KaraokeRecordEcho, "record 30 s of the microphone, the music and the voice cleaned (to tune the echo canceller)"),
    b(Karaoke, &[ch('g')], Op::KaraokeGame, "the game on or off: the singing scored, or just sung"),
    b(Karaoke, &[ch('p')], Op::KaraokeNotes, "show or hide the song's notes under the line"),
    b(Karaoke, &[UP], Op::Up, "the row above (voice, delay, key)"),
    b(Karaoke, &[DOWN], Op::Down, "the row below"),
    b(Karaoke, &[ch('s')], Op::KaraokeSync, "sync: clicks to answer with “ta” (or a clap), to measure how late your voice comes (a microphone open)"),
    b(Karaoke, &[LEFT], Op::SliderDown, "the row's value a step down: the voice 1 dB (at −45 dB or below: off), the delay 10 ms (its nudge), the key a semitone (↑ ↓ pick the row)"),
    b(Karaoke, &[RIGHT], Op::SliderUp, "the row's value a step up"),
    b(Karaoke, &[SHIFT_LEFT], Op::SliderFineDown, "the voice 1 % of its travel down"),
    b(Karaoke, &[SHIFT_RIGHT], Op::SliderFineUp, "the voice 1 % of its travel up"),
    b(Karaoke, &[ch('-')], Op::KeyDown, "the key a semitone down (the tempo kept; not the drums)"),
    b(Karaoke, &[ch('+'), ch('=')], Op::KeyUp, "the key a semitone up"),
    b(Karaoke, &[ch('0')], Op::KeyReset, "the song's own key (the Delay row at hand: not nudged)"),
    // Syncing.
    b(Sync, &[ENTER], Op::SyncKeep, "keep the delay measured, for this output and microphone"),
    b(Sync, &[ch('r')], Op::SyncAgain, "again"),
    b(Sync, &[ESC], Op::Back, "cancel"),
    // Karaoke mode.
    b(KaraokeMode, &[ENTER], Op::KaraokeStart, "START: the song after a count-in, a second a number (space too)"),
    b(KaraokeMode, &[ESC], Op::Back, "leave Karaoke mode, back to the tab (the stems as before)"),
    b(KaraokeMode, &[ch('i')], Op::MicNext, "the next microphone"),
    b(KaraokeMode, &[ch('r')], Op::KaraokeRecordEcho, "record 30 s for tuning the echo canceller"),
    b(KaraokeMode, &[ch('p')], Op::KaraokeDemo, "hold: the demo sings the song perfectly, AUTO showing (its score no best, its singing not kept)"),
    b(KaraokeMode, &[LEFT], Op::SliderDown, "the voice 1 dB down (at −45 dB or below: off)"),
    b(KaraokeMode, &[RIGHT], Op::SliderUp, "the voice 1 dB up"),
    b(KaraokeMode, &[SHIFT_LEFT], Op::SliderFineDown, "the voice 1 % of its travel down"),
    b(KaraokeMode, &[SHIFT_RIGHT], Op::SliderFineUp, "the voice 1 % of its travel up"),
    // The Mixer tab.
    b(Desk, &[LEFT], Op::DeskPrevStrip, "the strip to the left (player, preview, microphone, master)"),
    b(Desk, &[RIGHT], Op::DeskNextStrip, "the strip to the right"),
    b(Desk, &[UP], Op::Up, "the fader 1 dB up (the master: the player's volume, up to 0 dB)"),
    b(Desk, &[DOWN], Op::Down, "the fader 1 dB down (at −45 dB or below: off)"),
    b(Desk, &[PGUP], Op::PageUp, "the fader 6 dB up"),
    b(Desk, &[PGDN], Op::PageDown, "the fader 6 dB down"),
    b(Desk, &[HOME], Op::Top, "the fader to 0 dB: nothing added"),
    b(Desk, &[END], Op::Bottom, "the fader off (−∞)"),
    b(Desk, &[ch('m')], Op::DeskMute, "mute the strip, or not"),
    b(Desk, &[ch('s')], Op::DeskSolo, "solo the strip, or not (while one is soloed, the others are silent)"),
    b(Desk, &[ch('e')], Op::DeskEcho, "the microphone's echo canceller on or off"),
    b(Desk, &[ch('i')], Op::MicNext, "the next microphone (input device)"),
    b(Desk, &[ch('-')], Op::KeyDown, "the song a semitone down (on the PLAYER strip)"),
    b(Desk, &[ch('+'), ch('=')], Op::KeyUp, "the key a semitone up"),
    b(Desk, &[ch('0')], Op::KeyReset, "the song's own key"),
    // Now playing.
    b(NowPlaying, &[ESC], Op::Back, "back"),
    // The settings, their device list and their data folder list.
    b(Settings, &[UP], Op::Up, "up"),
    b(Settings, &[DOWN], Op::Down, "down"),
    b(Settings, &[LEFT], Op::ValueDown, "previous value"),
    b(Settings, &[RIGHT], Op::ValueUp, "next value"),
    b(Settings, &[ENTER], Op::Open, "change / open the device list"),
    b(Settings, &[ESC, ch(',')], Op::Back, "back"),
    b(DeviceList, &[UP], Op::Up, "up"),
    b(DeviceList, &[DOWN], Op::Down, "down"),
    b(DeviceList, &[ENTER], Op::Open, "play on this device"),
    b(DeviceList, &[ESC], Op::Back, "close the list"),
    b(FolderList, &[UP], Op::Up, "up"),
    b(FolderList, &[DOWN], Op::Down, "down"),
    b(FolderList, &[ENTER, RIGHT], Op::Open, "open the folder, or move the data folder into the one shown"),
    b(FolderList, &[BACKSPACE, LEFT], Op::Parent, "up a folder"),
    b(FolderList, &[ESC], Op::Back, "close the list"),
    // The help.
    b(Help, &[UP], Op::Up, "scroll up"),
    b(Help, &[DOWN], Op::Down, "scroll down"),
    b(Help, &[PGUP], Op::PageUp, "page up"),
    b(Help, &[PGDN], Op::PageDown, "page down"),
    b(Help, &[ch('/')], Op::Filter, "search this help (keys and text)"),
    b(Help, &[ESC, ch('?')], Op::Back, "clear the search, else back"),
    // Typing a search of the help.
    b(HelpSearch, &[ENTER], Op::Accept, "keep the search, stop typing"),
    b(HelpSearch, &[UP], Op::Up, "scroll up"),
    b(HelpSearch, &[DOWN], Op::Down, "scroll down"),
    b(HelpSearch, &[PGUP], Op::PageUp, "page up"),
    b(HelpSearch, &[PGDN], Op::PageDown, "page down"),
    b(HelpSearch, &[BACKSPACE], Op::Erase, "delete a character"),
    b(HelpSearch, &[Key::ctrl('v')], Op::Paste, "paste"),
    b(HelpSearch, &[ESC], Op::Back, "clear the search"),
];
