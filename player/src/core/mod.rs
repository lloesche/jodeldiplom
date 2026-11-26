//! The UI core: everything the window shows and does, and nothing that
//! draws.
//!
//! - [`State`]: what the player shows and knows.
//! - [`Action`]: every input, from keys (through [`keymap`]) and the
//!   pointer, and from the engine and the background workers.
//! - [`update`]: `(state, action) -> effects`, pure. The [`Effect`]s are
//!   carried out by [`crate::app::App`].
//! - [`Fit`]: what the window's size does to the keys and the pointer, as
//!   the window reports it; the core keeps no geometry of its own.
//! - [`keymap`]: the one table of key bindings, from which the help screen
//!   and the command palette are generated.
//! - [`text`]: the strings the screens show (the signal path, the list
//!   toggle, the status light).
//!
//! The window translates its input into [`Action`]s and draws [`State`]
//! with the [`crate::view`] widgets.

pub mod action;
pub mod bubbles;
pub mod desk;
mod fit;
pub mod grid;
pub mod karaoke;
pub mod key;
pub mod keymap;
pub mod lag;
pub mod lanes;
pub mod lyric_edit;
pub mod mic;
pub mod preview;
pub mod shine;
pub mod singing;
pub mod sliders;
pub mod stems;
pub mod sync;
pub mod synth_tab;
pub mod text;
pub mod transition;
mod update;
pub mod voice_curve;
pub mod voice_time;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use jodel_output::StreamInfo;
use jodel_playback::{PlayState, Route, Settings, TrackInfo};

pub use action::{
    Action, Effect, Facts, Heard, Job, LyricsFacts, Placing, RhythmFacts, WalkPurpose,
};
pub use fit::Fit;
pub use update::{
    browser_rows, device_window, folder_window, help_line_count, help_shown, job_done, list_cursor,
    list_height, list_len, palette_window, readiness, row_entry, shown_title, update,
    visible_missing, Readiness, Row,
};

use crate::browser::{Details, Entry};
use crate::config::Config;

/// The beat analysis of a file, as far as it has come.
#[derive(Clone, Debug)]
pub enum Rhythm {
    /// Asked for, not done yet.
    Pending,
    /// Done.
    Ready(Arc<Rhythmed>),
    /// The file could not be analysed.
    Failed,
}

/// What the analysis of a file gave: its beats (with the grid fixed by
/// hand, where it was) and its waveform.
#[derive(Clone, Debug)]
pub struct Rhythmed {
    /// The beats as they count: the analysis, with a grid fixed by hand
    /// in place of its own.
    pub analysis: Arc<jodel_rhythm::Analysis>,
    /// The analysis as it found the beats.
    pub found: Arc<jodel_rhythm::Analysis>,
    /// The mix's waveform and the stems', when they were made.
    pub waveform: Option<Arc<jodel_rhythm::waveform::Waveform>>,
}

impl Rhythmed {
    /// The analysis as found, nothing fixed.
    pub fn new(
        analysis: jodel_rhythm::Analysis,
        waveform: Option<jodel_rhythm::waveform::Waveform>,
    ) -> Rhythmed {
        let analysis = Arc::new(analysis);
        Rhythmed {
            found: Arc::clone(&analysis),
            analysis,
            waveform: waveform.map(Arc::new),
        }
    }

    /// The analysis with `grid` fixed by hand in place of its own.
    pub fn with_grid(mut self, grid: &grid::Grid) -> Rhythmed {
        self.analysis = Arc::new(grid.apply(&self.found));
        self
    }

    /// Whether the grid was fixed by hand.
    pub fn fixed(&self) -> bool {
        !Arc::ptr_eq(&self.analysis, &self.found)
    }
}

/// What a file's work found so far, as the queue's icons show it: each
/// `None` until that part is done, then whether the file has it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QueueFacts {
    /// Its key.
    pub key: Option<bool>,
    /// Its stems, separated and kept.
    pub stems: Option<bool>,
    /// Its tempo (a clear beat).
    pub tempo: Option<bool>,
    /// Its lyrics, once settled.
    pub lyrics: Option<LyricsFacts>,
    /// Its cover, taken apart.
    pub cover: Option<bool>,
}

impl QueueFacts {
    /// Takes in the whole analysis's facts.
    pub fn analysed(&mut self, f: &RhythmFacts) {
        self.key = Some(f.key.is_some());
        self.stems = Some(f.stems);
        self.tempo = Some(f.bpm.is_some());
    }
}

/// Below this agreement among the beat trackers, no BPM is shown.
pub const MIN_CONFIDENCE: f64 = 0.5;
/// Tracks after the playing one whose beats are analysed ahead.
pub const RHYTHM_AHEAD: usize = 2;

/// How far `←`/`→` seek, in seconds.
pub const SEEK_STEP: f64 = 5.0;
/// How long a message stays in the status line, in milliseconds.
pub const MESSAGE_MS: u64 = 6_000;
/// Two clicks on the same row within this many milliseconds do what
/// `enter` does on it.
pub const DOUBLE_CLICK_MS: u64 = 450;
/// Settings changed with keys or clicks reach the engine this many
/// milliseconds after the last change ([`State::settings_due`]), so
/// stepping through values does not reopen the output at each step.
pub const SETTINGS_DELAY_MS: u64 = 400;

/// Which screen fills the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    /// The list pane, the right column and the bottom bar.
    Main,
    /// The cover as large as the window holds it, the track's title,
    /// artist, album, format and tempo over its bottom.
    NowPlaying,
    /// The output device, the headroom and the player's other settings.
    Settings,
    /// Every key binding, from the keymap.
    Help,
    /// Karaoke mode ([`karaoke`]): the lyrics over the cover, fullscreen in
    /// the window.
    Karaoke,
    /// The lyrics editor ([`lyric_edit`]): a song's words over its voice's
    /// waveform, the bottom bar under it.
    Lyrics,
}

/// What the list pane shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The file browser.
    Browser,
    /// The play queue.
    Queue,
    /// The track playing and the next one, their waveforms and bars
    /// ([`transition::tab`]); it takes the whole width.
    Transition,
    /// Karaoke set up: the voice's and the microphone's levels, and the
    /// way into Karaoke mode ([`karaoke`]); the whole width too.
    Karaoke,
    /// The mixer as a desk: the strips with their faders and meters, the
    /// microphone's strip and the master ([`desk`]); the whole width too.
    Mixer,
    /// The game's sounds as a workbench: each played, its patch shown and
    /// edited, a keyboard ([`synth_tab`]); the whole width too.
    Synth,
}

impl Side {
    /// The tabs, in the order `tab` goes through them.
    pub const ALL: [Side; 6] = [
        Side::Browser,
        Side::Queue,
        Side::Transition,
        Side::Karaoke,
        Side::Mixer,
        Side::Synth,
    ];

    /// Whether it takes the whole width (no right column).
    pub fn wide(self) -> bool {
        matches!(
            self,
            Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth
        )
    }
}

/// The file browser's state.
#[derive(Clone, Debug, Default)]
pub struct BrowserState {
    /// The folder shown.
    pub dir: PathBuf,
    /// Its parent row, folders and playable files, sorted folders first.
    pub entries: Vec<Entry>,
    /// The folder is being listed.
    pub loading: bool,
    /// Why the folder could not be listed.
    pub error: Option<String>,
    /// The selected row.
    pub cursor: usize,
    /// The first row shown.
    pub scroll: usize,
    /// The filter typed after `/`.
    pub filter: Option<Filter>,
    /// The recursive filename search below [`BrowserState::dir`].
    pub search: Option<Search>,
    /// Ids for searches, so late results of a cancelled one are ignored.
    pub next_search: u64,
    /// The child last selected in each folder, to reselect it on the way
    /// back up.
    pub last_child: HashMap<PathBuf, PathBuf>,
    /// What was typed to jump to a name (lowercase), and when the last key
    /// came ([`State::now_ms`]); a pause of [`TYPE_AHEAD_MS`] starts anew.
    pub typed: String,
    /// When the last type-ahead key came.
    pub typed_at: u64,
    /// Where to go when the folder shown at the start (the one the session
    /// remembered) cannot be listed.
    pub fallback: Option<PathBuf>,
}

/// How long a pause in typing starts a new type-ahead jump, in ms.
pub const TYPE_AHEAD_MS: u64 = 1000;

/// The filter of the current folder (or the search of the help).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    /// What was typed.
    pub text: String,
    /// Keys go to the filter line (until `enter` or `esc`).
    pub editing: bool,
}

/// A recursive filename search.
#[derive(Clone, Debug, Default)]
pub struct Search {
    /// Matches results to the search that produced them.
    pub id: u64,
    /// The text searched for.
    pub query: String,
    /// Files and folders whose name contains the query, as found.
    pub hits: Vec<Entry>,
    /// The walk has finished.
    pub done: bool,
}

/// The play queue, as the engine reports it.
#[derive(Clone, Debug, Default)]
pub struct QueueState {
    /// The files, in play order.
    pub paths: Vec<PathBuf>,
    /// The track being heard.
    pub current: Option<usize>,
    /// The selected row.
    pub cursor: usize,
    /// The first row shown.
    pub scroll: usize,
    /// Changes sent to the engine (a remove or a move) whose queue has not
    /// come back yet: the queue here has them already, and the engine's
    /// queues from before them are not taken.
    pub pending: usize,
    /// A row is being dragged to another place with the mouse.
    pub dragging: bool,
}

/// The track being heard, from the engine's `TrackStarted`.
#[derive(Clone, Debug)]
pub struct NowTrack {
    /// Its index in the queue.
    pub index: usize,
    /// The file: format, tags, cover.
    pub info: Arc<TrackInfo>,
    /// The route the rate policy chose.
    pub route: Route,
    /// What the output was opened at.
    pub output: StreamInfo,
    /// The output's name.
    pub device: String,
}

/// What is playing.
#[derive(Clone, Debug)]
pub struct Playing {
    /// The track being heard.
    pub track: Option<NowTrack>,
    /// Playing, paused or stopped.
    pub state: PlayState,
    /// Seconds into the track, as heard.
    pub position: f64,
    /// The last position the engine told: where a head's output frames are
    /// in the song. None since a track started, a seek or a jump, until the
    /// next.
    pub anchor: Option<Anchor>,
    /// Output frames a second of the track took between the last two
    /// positions told (a transition's next track goes at another speed for
    /// a while); `None` before two.
    pub speed: Option<f64>,
    /// How many times the anchor was forgotten ([`Playing::forget_anchor`]):
    /// a frame worked out before one is not a frame of the same song's
    /// place after it.
    pub anchors_forgotten: u64,
    /// Samples the resampler clipped, as heard, since the track (or the
    /// last seek) started.
    pub clipped: u64,
    /// The key the song plays in, in semitones from its own, as asked
    /// (shown at once; the engine's word on it, [`super::key`]).
    pub key: i8,
}

impl Default for Playing {
    fn default() -> Self {
        Playing {
            track: None,
            state: PlayState::Stopped,
            position: 0.0,
            anchor: None,
            speed: None,
            anchors_forgotten: 0,
            clipped: 0,
            key: 0,
        }
    }
}

/// A frame of the player's output: stream frame `frame` of head `head`
/// ([`jodel_playback::TrackStart::head`]). A transition's next track plays
/// on the player's other head, and from its hand-over the output frames
/// told are that head's, counted from its first: a frame is a frame of the
/// head it was told with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadFrame {
    /// The head.
    pub head: usize,
    /// Its output's stream frame.
    pub frame: u64,
}

/// A position the engine told ([`jodel_playback::Position`]): track `index`
/// heard at `at`, `seconds` into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    /// The track's index in the queue.
    pub index: usize,
    /// The output frame heard, and its head.
    pub at: HeadFrame,
    /// Seconds into the track there.
    pub seconds: f64,
}

impl Playing {
    /// A position the engine told, of the track heard: where it is, and
    /// the anchor there.
    pub fn told(&mut self, position: &jodel_playback::Position) {
        if self
            .track
            .as_ref()
            .is_some_and(|t| t.index == position.index)
        {
            self.position = position.seconds;
            let (head, frame) = (position.head, position.output_frame);
            self.anchor_at(Anchor {
                index: position.index,
                at: HeadFrame { head, frame },
                seconds: position.seconds,
            });
        }
    }

    /// A position told: the output's speed through the track since the
    /// last (of the same track on the same head: none across a hand-over),
    /// and the anchor there.
    pub fn anchor_at(&mut self, at: Anchor) {
        if let Some(was) = self.anchor {
            let dt = at.seconds - was.seconds;
            let same = was.index == at.index && was.at.head == at.at.head;
            if same && dt > 0.01 && at.at.frame > was.at.frame {
                self.speed = Some((at.at.frame - was.at.frame) as f64 / dt);
            }
        }
        self.anchor = Some(at);
    }

    /// The positions told before say nothing of what follows (another
    /// track, a seek, a jump).
    pub fn forget_anchor(&mut self) {
        self.anchor = None;
        self.speed = None;
        self.anchors_forgotten += 1;
    }

    /// The frame of the playing track's output where `seconds` into it is
    /// heard: from the last position told, at the output's speed through
    /// the track. `None` before two positions of it were told since it
    /// started (or a seek, or a jump), and for a time before the output's
    /// first frame.
    pub fn frame_at(&self, seconds: f64) -> Option<HeadFrame> {
        let track = self.track.as_ref()?;
        let anchor = self.anchor.filter(|a| a.index == track.index)?;
        let frame = anchor.at.frame as f64 + (seconds - anchor.seconds) * self.speed?;
        (frame >= 0.0).then(|| HeadFrame {
            head: anchor.at.head,
            frame: frame.round() as u64,
        })
    }

    /// The track's length in seconds, when known.
    pub fn duration(&self) -> Option<f64> {
        let track = self.track.as_ref()?;
        track.info.duration().map(|d| d.as_secs_f64())
    }
}

/// An output device the settings offer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceChoice {
    /// The id to store; `None` is "the system default".
    pub id: Option<String>,
    /// The name.
    pub name: String,
    /// What it takes (its mix format), in a few words.
    pub summary: String,
}

/// The settings rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsRow {
    /// The output device.
    Device,
    /// The gain below full scale on resampled routes.
    Headroom,
    /// Tempo, beat pulse and beat analysis on or off.
    Tempo,
    /// The beat methods vote, or one decides.
    Voting,
    /// A beat method: a checkbox while voting, else a radio button.
    Method(jodel_rhythm::Method),
    /// The cues listen to stems separated on the GPU, or to the mix.
    Stems,
    /// A stem turned on or off is heard from the next bar, or at once.
    StemSwitch,
    /// What the next-track key does: cut, or mix in.
    SkipMix,
    /// The preset it mixes in by: the Transition tab's, or one of its own.
    SkipPreset,
    /// Where the player's data is kept (chosen in a folder list; moved there).
    DataFolder,
}

impl SettingsRow {
    /// Every row, top to bottom.
    pub const ALL: [SettingsRow; 11] = [
        SettingsRow::Device,
        SettingsRow::Headroom,
        SettingsRow::Tempo,
        SettingsRow::Voting,
        SettingsRow::Method(jodel_rhythm::Method::Flux),
        SettingsRow::Method(jodel_rhythm::Method::Network),
        SettingsRow::Stems,
        SettingsRow::StemSwitch,
        SettingsRow::SkipMix,
        SettingsRow::SkipPreset,
        SettingsRow::DataFolder,
    ];

    /// The row's label.
    pub fn label(self) -> &'static str {
        match self {
            SettingsRow::Device => "Output device",
            SettingsRow::Headroom => "Headroom",
            SettingsRow::Tempo => "Tempo and beats",
            SettingsRow::Voting => "Beat voting",
            SettingsRow::Method(jodel_rhythm::Method::Flux) => "Spectral flux",
            SettingsRow::Method(jodel_rhythm::Method::Network) => "Beat This! network",
            SettingsRow::Stems => "Stems",
            SettingsRow::StemSwitch => "Stem switching",
            SettingsRow::SkipMix => "Next track (N)",
            SettingsRow::SkipPreset => "Its transition",
            SettingsRow::DataFolder => "Data folder",
        }
    }

    /// Whether it belongs to the row above it (shown set in under it).
    pub fn under(self) -> bool {
        matches!(self, SettingsRow::Method(_) | SettingsRow::SkipPreset)
    }
}

/// What the beat analysis runs: its methods, and whether its cues listen
/// to stems separated on the GPU.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RhythmRecipe {
    /// The beat methods.
    pub methods: Vec<jodel_rhythm::Method>,
    /// Stems for the cues.
    pub stems: bool,
}

/// The GPU for stems.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Gpu {
    /// Still being looked for.
    #[default]
    Looking,
    /// There is none (or none that works).
    Absent,
    /// This one.
    Present(String),
}

/// The settings screen's state.
#[derive(Clone, Debug, Default)]
pub struct SettingsState {
    /// The selected row ([`SettingsRow::ALL`] index).
    pub row: usize,
    /// The device list is open.
    pub picking: bool,
    /// The selected device in the list.
    pub device_cursor: usize,
    /// The devices, once listed.
    pub devices: Option<Vec<DeviceChoice>>,
    /// The folder list for the data folder, while it is open.
    pub folders: Option<FolderPick>,
}

/// The folder list the data folder is chosen in: a folder's subfolders,
/// or (above a Windows drive) the drives. The data folder is made inside
/// the folder chosen ([`crate::dirs::DATA`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FolderPick {
    /// The folder shown; `None`: the drives.
    pub dir: Option<PathBuf>,
    /// Its subfolders (or the drives), once listed.
    pub folders: Option<Vec<PathBuf>>,
    /// The row selected ([`FolderPick::rows`]).
    pub cursor: usize,
}

/// A row of the folder list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FolderRow {
    /// Make the data folder in this folder.
    Choose(PathBuf),
    /// Up a folder (or to the drives).
    Up,
    /// A subfolder (or a drive): opened.
    Folder(PathBuf),
}

impl FolderPick {
    /// Its rows: the choice of the folder shown, up, then its folders.
    pub fn rows(&self) -> Vec<FolderRow> {
        let mut rows = Vec::new();
        if let Some(dir) = &self.dir {
            rows.push(FolderRow::Choose(dir.clone()));
            rows.push(FolderRow::Up);
        }
        rows.extend(
            self.folders
                .iter()
                .flatten()
                .cloned()
                .map(FolderRow::Folder),
        );
        rows
    }
}

/// A move of the data folder, as the settings show it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DataMove {
    /// Where it goes.
    pub to: PathBuf,
    /// The share moved (0 to 1).
    pub share: f64,
    /// Bytes moved, and in all.
    pub done: u64,
    /// See [`DataMove::done`].
    pub total: u64,
    /// Why it stopped (a full disk): `enter` on the row tries again.
    pub failure: Option<String>,
    /// It is done.
    pub finished: bool,
}

/// The command palette.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    /// What was typed.
    pub text: String,
    /// The selected command among the matches.
    pub cursor: usize,
}

/// How a message is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Something happened.
    Info,
    /// Something failed.
    Error,
}

/// A line in the status bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// The text.
    pub text: String,
    /// Info or error.
    pub level: Level,
    /// When it goes away ([`State::now_ms`] clock).
    pub until_ms: u64,
}

/// Where the last click landed, for double clicks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Click {
    /// The list side and row.
    pub side: Side,
    /// The row.
    pub row: usize,
    /// When ([`State::now_ms`]).
    pub at_ms: u64,
}

/// Everything the UI shows and knows.
#[derive(Clone, Debug)]
pub struct State {
    /// What the window's size does to the keys and the pointer, as last
    /// reported ([`Action::Fit`]).
    pub fit: Fit,
    /// Milliseconds since the start, from [`Action::Tick`].
    pub now_ms: u64,
    /// The screen shown.
    pub screen: Screen,
    /// What the list pane shows.
    pub side: Side,
    /// The file browser.
    pub browser: BrowserState,
    /// The queue.
    pub queue: QueueState,
    /// What each file is, read in the background, for this session.
    pub details: HashMap<PathBuf, Details>,
    /// Where queues are saved as playlists (`ctrl+s`, `ctrl+l`); `None`:
    /// nowhere.
    pub queues_dir: Option<PathBuf>,
    /// Files whose details were asked for and have not arrived.
    pub pending: HashSet<PathBuf>,
    /// What is playing.
    pub playing: Playing,
    /// The settings screen.
    pub settings: SettingsState,
    /// The settings in force (and in the config file).
    pub config: Config,
    /// Where settings changed in the UI are saved, if anywhere.
    pub config_file: Option<PathBuf>,
    /// The data folder in use ([`crate::dirs::Data`]), if there is one.
    pub data_dir: Option<PathBuf>,
    /// A move of the data folder, while one is under way (and once it has
    /// ended this run).
    pub data_move: Option<DataMove>,
    /// The help screen's first line.
    pub help_scroll: usize,
    /// The search of the help typed after `/` there.
    pub help_search: Option<Filter>,
    /// The open command palette.
    pub palette: Option<Palette>,
    /// The status line.
    pub message: Option<Message>,
    /// The seek bar is being dragged: the fraction under the pointer.
    pub seek_drag: Option<f64>,
    /// The last click on a list row.
    pub last_click: Option<Click>,
    /// A file to play as soon as its folder is listed (the `path` argument).
    pub autoplay: Option<PathBuf>,
    /// This run plays into the null output whatever the device setting.
    pub null_output: bool,
    /// When changed settings are to be sent to the engine ([`State::now_ms`]
    /// clock; [`SETTINGS_DELAY_MS`] after the last change).
    pub settings_due: Option<u64>,
    /// Quit was asked for.
    pub quit: bool,
    /// The beat analyses of the playing track and the next ones.
    pub rhythm: HashMap<PathBuf, Rhythm>,
    /// The files last asked to be analysed, in order.
    pub rhythm_wanted: Vec<PathBuf>,
    /// The recipe they were asked for with.
    pub rhythm_recipe: RhythmRecipe,
    /// The files last asked to be got ready ahead, and the recipe.
    pub prepared: (Vec<PathBuf>, Option<RhythmRecipe>),
    /// The files a worker is busy with, and with what.
    pub working: HashMap<PathBuf, Vec<Job>>,
    /// The work done for each file, its results in the caches
    /// ([`Action::Ready`]); the analysis by the recipe in `prepared`.
    pub ready: HashMap<PathBuf, Vec<Job>>,
    /// What each file's finished work found, for the queue's icons.
    pub facts: HashMap<PathBuf, QueueFacts>,
    /// The work each file of the queue is got ready by (a worker for
    /// each): the analysis always, the lyrics and the cover where the app
    /// has their workers. A file is ready when all of it is done.
    pub preparing: Vec<Job>,
    /// The GPU for stems.
    pub gpu: Gpu,
    /// What the engine and the mixer were last given of the stems
    /// ([`stems::follow`]).
    pub stems_sent: stems::StemsSent,
    /// The Transition tab.
    pub transition: transition::tab::TransitionTab,
    /// The Karaoke tab and mode.
    pub karaoke: karaoke::KaraokeState,
    /// The Mixer tab: the desk's settings and what it shows.
    pub desk: desk::DeskState,
    /// The Synth tab: the sound selected, the patches edited and kept.
    pub synth: synth_tab::SynthTab,
    /// The microphone as last read (once a drawn frame), while one is
    /// open.
    pub mic: Option<mic::MicReading>,
    /// Seconds into the playing track as heard when the last frame was
    /// drawn (a frontend's smoothed position), so the Transition tab's keys and
    /// clicks act on what was seen; `None`: the position reported.
    pub drawn_heard: Option<f64>,
    /// The clock ([`State::now_ms`]) when the last frame was drawn: what
    /// moves by itself between ticks (a block sliding) is drawn by it.
    pub drawn_ms: Option<u64>,
    /// The preview while Ctrl is held in the browser ([`preview`]).
    pub preview: preview::PreviewState,
}

impl State {
    /// When the soonest thing is due ([`State::now_ms`] clock): changed
    /// settings to send, a paused song's count-in, a preview to start
    /// (Ctrl held long enough), Karaoke mode's controls to fade out, its
    /// pointer to hide.
    pub fn due(&self) -> Option<u64> {
        [
            self.settings_due,
            self.karaoke.resume_at,
            self.preview.due,
            karaoke::controls::due(self),
            karaoke::pointer::due(self),
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// A state with the browser in `dir` and the given settings, before the
    /// window has reported its size ([`State::fit`]).
    pub fn new(dir: PathBuf, config: Config) -> State {
        let desk = desk::DeskState::new(&config);
        let transition =
            transition::tab::TransitionTab::with(config.transition_preset, config.transition_tempo);
        State {
            fit: Fit::default(),
            now_ms: 0,
            screen: Screen::Main,
            side: Side::Browser,
            browser: BrowserState {
                dir,
                loading: true,
                ..BrowserState::default()
            },
            queue: QueueState::default(),
            details: HashMap::new(),
            queues_dir: None,
            pending: HashSet::new(),
            playing: Playing::default(),
            settings: SettingsState::default(),
            config,
            config_file: None,
            data_dir: None,
            data_move: None,
            help_scroll: 0,
            help_search: None,
            palette: None,
            message: None,
            seek_drag: None,
            last_click: None,
            autoplay: None,
            null_output: false,
            settings_due: None,
            quit: false,
            rhythm: HashMap::new(),
            rhythm_wanted: Vec::new(),
            prepared: (Vec::new(), None),
            working: HashMap::new(),
            ready: HashMap::new(),
            facts: HashMap::new(),
            preparing: vec![Job::Rhythm],
            rhythm_recipe: RhythmRecipe::default(),
            gpu: Gpu::Looking,
            stems_sent: stems::StemsSent::default(),
            transition,
            karaoke: karaoke::KaraokeState::default(),
            desk,
            synth: synth_tab::SynthTab::default(),
            mic: None,
            drawn_heard: None,
            drawn_ms: None,
            preview: preview::PreviewState::default(),
        }
    }

    /// Seconds into the playing track as last drawn ([`State::drawn_heard`]),
    /// else as reported.
    pub fn heard(&self) -> f64 {
        self.drawn_heard.unwrap_or(self.playing.position)
    }

    /// The settings the engine plays with.
    pub fn engine_settings(&self) -> Settings {
        self.config.settings()
    }

    /// The selected settings row.
    pub fn settings_row(&self) -> SettingsRow {
        SettingsRow::ALL[self.settings.row.min(SettingsRow::ALL.len() - 1)]
    }

    /// The device the settings name, for people.
    pub fn device_label(&self) -> String {
        if self.null_output {
            return "null output (--null)".into();
        }
        match &self.config.device {
            None => "system default".into(),
            Some(id) => self
                .settings
                .devices
                .as_ref()
                .and_then(|list| list.iter().find(|d| d.id.as_deref() == Some(id)))
                .map_or_else(|| id.clone(), |d| d.name.clone()),
        }
    }
}
