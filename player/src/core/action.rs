//! What goes into [`crate::core::update`] and what comes out.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_playback::{Command, Event};

use crate::browser::{Details, Entry};
use crate::core::keymap::Op;
use crate::core::{DeviceChoice, SettingsRow, Side};

/// Work the workers do on a file in the background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Job {
    /// Its beats, bars and stems analysed.
    Rhythm,
    /// Its lyrics looked up, or their words timed.
    Lyrics,
}

/// What a file's analysis found, as the queue's icons show it.
#[derive(Clone, Debug, PartialEq)]
pub struct RhythmFacts {
    /// Its tempo, when its beat is clear.
    pub bpm: Option<f64>,
    /// Its key on the Camelot wheel (`8A`), when found.
    pub key: Option<String>,
    /// Whether its stems were separated (and kept).
    pub stems: bool,
}

impl RhythmFacts {
    /// The facts of `analysis`.
    pub fn of(analysis: &jodel_rhythm::Analysis) -> RhythmFacts {
        RhythmFacts {
            bpm: analysis.summary.map(|s| s.bpm).filter(|b| *b > 0.0),
            key: analysis.key.as_ref().map(|k| k.camelot()),
            stems: analysis.stems,
        }
    }
}

/// How a file's lyrics were settled, as the queue's icons show it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LyricsFacts {
    /// Found, and their words timed by us.
    Timed,
    /// Found, their words not timed (no speech model, or it failed).
    Untimed,
    /// None found anywhere: heard from the voice by a speech recogniser
    /// (and timed), so they may be wrong.
    Heard,
    /// The track has no words.
    Instrumental,
    /// None found.
    NotFound,
    /// They could not be looked up (asked again later).
    Failed,
}

impl LyricsFacts {
    /// How `answer` settles them.
    pub fn of(answer: &crate::lyrics::Answer) -> LyricsFacts {
        use crate::lyrics::Answer;
        match answer {
            Answer::Found { lyrics, .. }
                if matches!(lyrics.source, jodel_lyrics::Source::Transcribed(_)) =>
            {
                LyricsFacts::Heard
            }
            Answer::Found { timed: Some(_), .. } => LyricsFacts::Timed,
            Answer::Found { .. } => LyricsFacts::Untimed,
            Answer::Instrumental => LyricsFacts::Instrumental,
            Answer::NotFound => LyricsFacts::NotFound,
            Answer::Failed(_) => LyricsFacts::Failed,
        }
    }
}

/// A part of a file's analysis heard while it goes on, in the order they
/// come ([`Action::Heard`]): the key once the file is decoded, the stems
/// once separated (the tempo comes with the whole analysis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heard {
    /// The key was looked for: whether one was found.
    Key(bool),
    /// The stems are separated and kept.
    Stems,
}

/// What a finished job found ([`Action::Ready`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Facts {
    /// The analysis's.
    Rhythm(RhythmFacts),
    /// The lyrics'.
    Lyrics(LyricsFacts),
}

/// Every input to the UI core.
#[derive(Clone, Debug)]
pub enum Action {
    /// A named operation (from a key, a button or the palette).
    Op(Op),
    /// A character typed into the filter or the palette.
    Type(char),
    /// Text pasted: from the terminal, or from the clipboard after
    /// [`Effect::ReadClipboard`]. It goes into the palette or the filter,
    /// and opens the filter on the main screen.
    Paste(String),

    /// The Ctrl key, as the frontend saw it: held in the browser, it
    /// previews the file selected ([`crate::core::preview`]).
    Ctrl(crate::core::preview::Ctrl),
    /// Select a list row (mouse).
    SelectRow(usize),
    /// Do on a list row what `enter` does (a double click).
    OpenRow(usize),
    /// A click on a list row: selects it, and a second click soon after
    /// on the same row does what `enter` does on it.
    ClickRow(usize),
    /// Queue a list row (right click).
    EnqueueRow(usize),
    /// The queue was saved as a playlist there, or why not.
    QueueSaved(Result<PathBuf, String>),
    /// A playlist's tracks, or why they could not be read.
    PlaylistRead {
        /// The playlist.
        path: PathBuf,
        /// Its tracks.
        tracks: Result<Vec<PathBuf>, String>,
        /// Where they go.
        how: Placing,
    },
    /// The queue row being dragged is over this row: it moves there.
    QueueDrag(usize),
    /// The drag of a queue row ended.
    QueueDragEnd,
    /// Scroll the list by rows (mouse wheel).
    Scroll(i32),
    /// Show this side of the list pane.
    ShowSide(Side),
    /// Seek to a fraction of the track (a click on the progress bar, or the
    /// end of a drag).
    SeekTo(f64),
    /// The progress bar is being dragged to this fraction.
    SeekDrag(f64),
    /// Open the settings at this row (the device name in the signal path
    /// opens the device list).
    OpenSettings(SettingsRow),
    /// Select a settings row and step its value (`-1`, `0`: select only, or
    /// for the device row open the list, `1`).
    SettingsClick(usize, i32),
    /// Choose a device from the open list.
    ChooseDevice(usize),
    /// A row of the data folder's list clicked ([`crate::core::FolderPick::rows`]).
    FolderRow(usize),
    /// A folder's subfolders listed for the data folder's list (`None`:
    /// the drives).
    FoldersListed(Option<PathBuf>, Vec<PathBuf>),
    /// Run the palette's match at this index.
    PaletteRun(usize),
    /// Scroll the help screen.
    HelpScroll(i32),
    /// A point of a Transition block's lanes, `time` seconds into its track,
    /// was pressed (`extend`: dragged to, selecting bars). A second press on
    /// the same beat soon after is a double click: the playing track plays
    /// from there, the next track's cue goes there.
    TransitionPoint {
        /// The block.
        block: super::lanes::Block,
        /// Seconds into its track.
        time: f64,
        /// Dragged to rather than pressed.
        extend: bool,
    },
    /// A Transition block's overview was clicked: the playing track seeks
    /// there, the next one's view goes there.
    TransitionJump {
        /// The block.
        block: super::lanes::Block,
        /// Seconds into its track.
        time: f64,
    },
    /// The wheel over the Transition tab: zoom in (negative) or out; with
    /// `scroll` (over the whole track), the block's view moves by that
    /// many tenths instead.
    TransitionWheel {
        /// The block under the pointer.
        block: super::lanes::Block,
        /// Steps, negative up.
        steps: i32,
        /// Shift held: scroll rather than zoom.
        scroll: bool,
        /// Seconds into the block's track under the pointer (the lanes): a
        /// zoom keeps it where it is.
        at: Option<f64>,
    },
    /// The secondary button went down on a Transition block: a drag of its view
    /// starts.
    TransitionPanStart {
        /// The block.
        block: super::lanes::Block,
        /// The pointer's column.
        column: u16,
        /// The width of the block's waveforms in columns.
        width: u16,
        /// Over the whole track (else the lanes).
        overview: bool,
    },
    /// The pointer moved to this column while dragging a Transition block's
    /// view.
    TransitionPan(u16),
    /// The drag of a Transition block's view ended.
    TransitionPanEnd,
    /// A Karaoke slider was pressed or dragged to a share of its range.
    KaraokeSlide {
        /// The slider.
        slider: super::karaoke::Slider,
        /// Where, 0 to 1.
        share: f64,
    },
    /// The drag of a Karaoke slider ended.
    KaraokeSlideEnd,
    /// A level picked on the Karaoke tab.
    KaraokeLevel(crate::core::singing::Level),
    /// The wheel turned over a Karaoke slider (no drag starts).
    KaraokeStep {
        /// The slider.
        slider: super::karaoke::Slider,
        /// By how much, in its own unit (percent, ms).
        by: i32,
    },
    /// What the app tells of the sync under way ([`crate::core::sync`]).
    Sync(crate::core::sync::SyncEvent),
    /// The mouse on the Mix tab's desk.
    Desk(crate::core::desk::DeskAction),
    /// The pointer moved to this row (Karaoke mode shows its controls for a
    /// while when it is near the bottom).
    PointerMoved {
        /// The pointer's row.
        row: u16,
    },

    /// The window or terminal size in cells.
    Resize(u16, u16),
    /// The clock: milliseconds since the start.
    Tick(u64),

    /// A folder was listed (or failed to be).
    Listed {
        /// The folder.
        dir: PathBuf,
        /// Its rows, or the error.
        result: Result<Vec<Entry>, String>,
    },
    /// A file's details were read.
    Details(PathBuf, Details),
    /// These files were not read after all (no longer on screen).
    DetailsDropped(Vec<PathBuf>),
    /// Search results.
    SearchHits {
        /// The search.
        id: u64,
        /// What it found since the last batch.
        hits: Vec<Entry>,
    },
    /// A search finished.
    SearchDone {
        /// The search.
        id: u64,
    },
    /// A folder walk for queueing finished.
    Walked {
        /// What the walk was for.
        purpose: WalkPurpose,
        /// The playable files found, in order.
        paths: Vec<PathBuf>,
    },
    /// What was typed after `/` is not an existing folder, file or drive:
    /// it is searched for below the folder instead.
    JumpMissed {
        /// What was typed.
        text: String,
    },
    /// What was typed after `/` is an existing folder or file: show this
    /// folder, with this file selected.
    JumpTo {
        /// The folder.
        dir: PathBuf,
        /// The file to select in it.
        select: Option<PathBuf>,
    },
    /// The output devices were listed.
    Devices(Result<Vec<DeviceChoice>, String>),
    /// The rates the output device takes exclusively, lowest first
    /// (`None`: not known), probed when the engine starts on it. The
    /// engine starts with [`crate::core::State::engine_settings`] after
    /// this, so this sends nothing to it.
    DeviceRates(Option<Vec<u32>>),
    /// The playback engine reported something.
    Engine(Event),
    /// A file's beats were analysed by a recipe (or could not be).
    Rhythm(
        PathBuf,
        super::RhythmRecipe,
        Result<Arc<super::Rhythmed>, String>,
    ),
    /// The GPU for stems, by name, or that there is none.
    Gpu(Option<String>),
    /// What was found of a file's lyrics.
    Lyrics(PathBuf, crate::lyrics::Answer),
    /// A worker started (`on`) or ended work on a file: its analysis, or
    /// its lyrics looked up or timed.
    Working {
        /// The file.
        path: PathBuf,
        /// Which work.
        job: Job,
        /// Started, else ended.
        on: bool,
    },
    /// What a job makes of a file is in the cache (or nothing more can be
    /// made of it): nothing is left to do for it.
    Ready {
        /// The file.
        path: PathBuf,
        /// Which work.
        job: Job,
        /// What it found, for the queue's icons (none when it failed).
        facts: Option<Facts>,
    },
    /// A part of a file's analysis is done, the rest still going on (the
    /// queue's icons light up one after the other).
    Heard {
        /// The file.
        path: PathBuf,
        /// Which part.
        part: Heard,
    },
    /// The output device's volume control after [`Effect::DeviceVolume`]:
    /// as it is now, `None` when the player cannot set it (the system's
    /// keys can), or why it could not be read or set.
    DeviceVolume {
        /// The control, or the failure.
        result: Result<Option<super::DeviceVolume>, String>,
        /// It was changed by a key (say so), not only read.
        changed: bool,
    },
    /// A failure outside the engine (a config file that could not be
    /// written, a device that could not be opened).
    Failed(String),
    /// Something to tell from outside the core (a recording done).
    Info(String),
    /// The playback engine of this start (the app's count of engine
    /// starts) failed and is gone; the app starts another, with the queue.
    EngineGone(u64),
}

/// What a folder walk is for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalkPurpose {
    /// Put the files in the queue.
    Enqueue {
        /// The folder's name, for the message.
        name: String,
        /// Where they go.
        how: Placing,
    },
}

/// Where tracks chosen in the browser go (`enter`, `shift+enter`,
/// `alt+enter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placing {
    /// After the queue.
    Append,
    /// After the queue, and the first of them plays now (the queue kept).
    AppendPlay,
    /// In place of the queue, playing.
    Replace,
}

/// Work for the app to carry out after an update.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// Send a command to the playback engine.
    Engine(Command),
    /// Tell the preview's players ([`crate::core::preview`]).
    Preview(crate::core::preview::Preview),
    /// List a folder (answered with [`Action::Listed`]).
    ListDir(PathBuf),
    /// List a folder's subfolders for the data folder's list, or the
    /// drives (`None`) (answered with [`Action::FoldersListed`]).
    ListFolders(Option<PathBuf>),
    /// Move the data folder into this folder (the settings save it once the
    /// move is written down).
    MoveData(PathBuf),
    /// Go on with a move of the data folder that stopped (a full disk).
    RetryMove,
    /// Read the details of exactly these files now (answered with
    /// [`Action::Details`] and [`Action::DetailsDropped`]).
    WantDetails(Vec<PathBuf>),
    /// Start a recursive search (answered with [`Action::SearchHits`] and
    /// [`Action::SearchDone`]); a running one is cancelled.
    Search {
        /// Matches results to the search.
        id: u64,
        /// Where to search.
        root: PathBuf,
        /// The text to search for.
        query: String,
    },
    /// Cancel the running search.
    CancelSearch,
    /// Check whether what was typed after `/` names an existing folder,
    /// file or drive, relative to `from` unless absolute (answered with
    /// [`Action::JumpTo`] if it does, else [`Action::JumpMissed`]).
    Jump {
        /// What was typed.
        text: String,
        /// The folder it was typed in.
        from: PathBuf,
    },
    /// Collect every playable file below a folder (answered with
    /// [`Action::Walked`]).
    Walk {
        /// The folder.
        root: PathBuf,
        /// What for.
        purpose: WalkPurpose,
    },
    /// Analyse the beats of exactly these files, in this order (answered
    /// with [`Action::Rhythm`]); what was asked for before and not started
    /// is dropped.
    WantRhythm(Vec<PathBuf>, super::RhythmRecipe),
    /// Look up the lyrics of exactly these files, in this order (answered
    /// with [`Action::Lyrics`]); what was asked for before and not started
    /// is dropped.
    WantLyrics(Vec<PathBuf>),
    /// Get these files ready ahead, in this order, after what is wanted:
    /// their analysis by the recipe (`None`: none) and their lyrics, into
    /// the caches; what was asked for before and not started is dropped.
    Prepare(Vec<PathBuf>, Option<super::RhythmRecipe>),
    /// Karaoke: use the next microphone (input device), and keep it in the
    /// settings.
    NextMic,
    /// Karaoke: record what the echo canceller works with, for tuning it.
    RecordEcho,
    /// Karaoke: play the sync's clicks and listen, or a press, or stop
    /// ([`crate::core::sync`]).
    Sync(crate::core::sync::SyncCommand),
    /// Set a strip of the mixer as the Mix tab has it.
    Desk(crate::core::desk::DeskChange),
    /// List the output devices (answered with [`Action::Devices`]).
    ListDevices,
    /// Play on this device from now on (`None`: the system default),
    /// carrying the queue and the position over.
    SwitchDevice(Option<String>),
    /// Read or step the output device's own volume control (answered with
    /// [`Action::DeviceVolume`]).
    DeviceVolume(super::VolumeChange),
    /// Read the clipboard's text (answered with [`Action::Paste`], or a
    /// message when it holds none).
    ReadClipboard,
    /// Keep a beat grid fixed by hand for a file (in the grid store).
    SaveGrid(PathBuf, super::grid::Grid),
    /// Forget a file's grid fixed by hand.
    ResetGrid(PathBuf),
    /// Write the settings to the config file.
    SaveConfig,
    /// Save these tracks as a playlist among the saved queues (answered
    /// with [`Action::QueueSaved`]).
    SaveQueue(Vec<PathBuf>),
    /// Read a playlist's tracks (answered with [`Action::PlaylistRead`]).
    ReadPlaylist {
        /// The playlist.
        path: PathBuf,
        /// Where its tracks go.
        how: Placing,
    },
    /// Leave.
    Quit,
}
