//! The running player, shared by both frontends: the [`State`], the
//! playback engine, the background workers, the spectrum and the cover.
//!
//! The engine plays on the Lead input of a mixer that owns the device
//! ([`jodel_output::mixer`]): alone at gain 1 it reaches the device bit for
//! bit. The preview's players ([`crate::preview`]) are guests on it, and
//! the spectrum is the mixer's: what the device plays.
//!
//! A frontend creates an [`App`], feeds it [`Action`]s from its input
//! ([`App::dispatch`]), calls [`App::pump`] to take in what the engine and
//! the workers sent, and draws [`App::state`] with [`crate::view`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use image::RgbImage;
use jodel_output::mixer::{Fader, MicPort, MicView, Mixer, MixerOptions, Role, Strip};
use jodel_output::SimClock;
use jodel_playback::{
    any_capabilities, Command, Cover, CoverOrigin, DeviceTarget, Event, NullTarget, OutputTarget,
    PlayState, Player, PlayerOptions, SpectrumTap,
};

use crate::art::{self, Art};
use crate::browser::{self, DetailsWorker, Notify};
use crate::config::Config;
use crate::core::{self, Action, DeviceChoice, Effect, State};
use crate::rhythm::RhythmWorker;
use crate::session::Session;
use crate::spectrum::{Analyzer, SpectrumFrame};

/// Seconds of the playing track the engine keeps behind what is heard and
/// decodes ahead of it, so that a bar jump within is heard at once.
const JUMP_WINDOW: f64 = 24.0;

/// Lists the output devices with a summary of what each takes.
pub type DeviceLister = Arc<dyn Fn() -> Result<Vec<DeviceChoice>, String> + Send + Sync>;

/// Reads or steps the volume control of a device (`None`: the system
/// default; else an id or a name, the id of the device playing once there
/// is one, see [`volume_device`]): the control as it is then, or `None`
/// when the player cannot set it.
pub type VolumeControl = Arc<
    dyn Fn(Option<&str>, core::VolumeChange) -> Result<Option<core::DeviceVolume>, String>
        + Send
        + Sync,
>;

/// Opens a microphone: the input with this id, or the system's default.
pub type MicOpener =
    Arc<dyn Fn(Option<&str>) -> Result<jodel_output::input::Input, String> + Send + Sync>;

/// Lists the microphones (input devices) as (id, name).
pub type MicLister = Arc<dyn Fn() -> Result<Vec<(String, String)>, String> + Send + Sync>;

/// The real microphone with this id, or the system's default input.
pub fn open_mic(id: Option<&str>) -> Result<jodel_output::input::Input, String> {
    let id = match id {
        Some(id) => id.to_owned(),
        None => {
            jodel_output::input::default_input()
                .map_err(|e| e.to_string())?
                .id
        }
    };
    jodel_output::input::open_input(&id).map_err(|e| e.to_string())
}

/// The real input devices.
pub fn list_mics() -> Result<Vec<(String, String)>, String> {
    jodel_output::input::input_devices()
        .map(|list| list.into_iter().map(|d| (d.id, d.name)).collect())
        .map_err(|e| e.to_string())
}

/// Where the engine plays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    /// A device: `None` the system default, else an id or a name (as
    /// [`jodel_output::find_device`] takes).
    Device(Option<String>),
    /// Nowhere: a null output that accepts every format, on this clock.
    Null(SimClock),
}

/// How an [`App`] starts.
pub struct AppOptions {
    /// The folder the browser starts in.
    pub dir: PathBuf,
    /// A file to play at once (in `dir`).
    pub autoplay: Option<PathBuf>,
    /// The settings for this run.
    pub config: Config,
    /// The settings as the file has them (without the command line's
    /// overrides, which are not saved).
    pub saved: Config,
    /// Where settings changed in the UI are saved.
    pub config_file: Option<PathBuf>,
    /// Where the engine plays.
    pub output: Output,
    /// The size in cells.
    pub size: (u16, u16),
    /// Called after a background result arrives (wakes a waiting loop).
    pub notify: Notify,
    /// Lists the devices for the settings.
    pub devices: DeviceLister,
    /// The device's own volume control.
    pub volume: VolumeControl,
    /// Messages to show at the start (config warnings).
    pub notes: Vec<String>,
    /// Where beat analyses are kept ([`crate::rhythm::cache_dir`]); `None`: not
    /// kept.
    pub rhythm_cache: Option<PathBuf>,
    /// Where beat grids fixed by hand are kept ([`crate::grids::dir`]);
    /// `None`: not kept.
    pub grids: Option<PathBuf>,
    /// Where the session is kept ([`crate::session::path`]): the queue and
    /// the browser's place come back from it and are saved to it; `None`:
    /// not kept.
    pub session: Option<PathBuf>,
    /// The browser starts where the session left it (no folder was given).
    pub restore_browser: bool,
    /// Where queues are saved as playlists ([`crate::session::queues_dir`]).
    pub queues: Option<PathBuf>,
    /// How lyrics are looked up ([`crate::lyrics::Options`]); `None`: in
    /// the files' tags and `.lrc` files only, nothing cached or fetched.
    pub lyrics: Option<crate::lyrics::Options>,
    /// The speech model for the word timing, where it is or is fetched to
    /// ([`crate::lyrics::Models`]); `None`: nowhere, the words shown untimed.
    pub model: Option<PathBuf>,
    /// How the microphone is opened (the real ones: [`open_mic`]).
    pub mic: MicOpener,
    /// How the microphones are listed (the real ones: [`list_mics`]).
    pub mics: MicLister,
    /// The data folder ([`crate::dirs::Data`]); `None`: none (tests).
    pub data: Option<crate::dirs::Data>,
    /// A move of the data folder under way ([`crate::mover`]), shown on the
    /// settings page.
    pub mover: Option<Arc<crate::mover::Mover>>,
}

impl AppOptions {
    /// Options for `dir` with the defaults: the system default device, the
    /// real device list, no config file.
    pub fn new(dir: PathBuf) -> AppOptions {
        AppOptions {
            dir,
            autoplay: None,
            config: Config::default(),
            saved: Config::default(),
            config_file: None,
            output: Output::Device(None),
            size: (120, 40),
            notify: Arc::new(|| {}),
            devices: Arc::new(list_devices),
            volume: Arc::new(device_volume),
            notes: Vec::new(),
            rhythm_cache: None,
            grids: None,
            session: None,
            restore_browser: false,
            queues: None,
            lyrics: None,
            model: None,
            mic: Arc::new(open_mic),
            mics: Arc::new(list_mics),
            data: None,
            mover: None,
        }
    }
}

/// The device whose own volume `alt+↑` / `alt+↓` read and move: the one
/// the engine plays on (`playing_on`, its id) once it has found it, else
/// the one `output` names (`None` inside: the system default). `None`: no
/// device (the null output).
///
/// The device playing, not the system default looked up anew: on macOS
/// the default can name another device or none while the player holds its
/// device in hog mode.
pub fn volume_device(output: &Output, playing_on: Option<&str>) -> Option<Option<String>> {
    match output {
        Output::Null(_) => None,
        Output::Device(query) => Some(playing_on.map(str::to_owned).or_else(|| query.clone())),
    }
}

/// The real device's volume control ([`jodel_output::volume`]), read or
/// moved a step ([`core::volume_step`]).
pub fn device_volume(
    query: Option<&str>,
    change: core::VolumeChange,
) -> Result<Option<core::DeviceVolume>, String> {
    let device = match query {
        Some(query) => jodel_output::find_device(query),
        None => jodel_output::default_device(),
    }
    .map_err(|e| e.to_string())?;
    let Some(now) = jodel_output::volume(&device.id).map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    let volume = match change {
        core::VolumeChange::Read => now,
        core::VolumeChange::Step(steps) => {
            jodel_output::set_volume(&device.id, core::volume_step(now.level, steps))
                .map_err(|e| e.to_string())?
        }
    };
    Ok(Some(core::DeviceVolume {
        level: volume.level,
        db: volume.db,
        muted: volume.muted,
    }))
}

/// The subfolders of `dir`, by name (none hidden or the system's), for the
/// data folder's list; `None`: the drives (on Windows; elsewhere `/`).
pub fn list_folders(dir: Option<&Path>) -> Vec<PathBuf> {
    let Some(dir) = dir else {
        if cfg!(windows) {
            return (b'A'..=b'Z')
                .map(|letter| PathBuf::from(format!("{}:\\", letter as char)))
                .filter(|drive| drive.is_dir())
                .collect();
        }
        return vec![PathBuf::from("/")];
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut folders: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                !n.starts_with('.') && !n.starts_with('$') && n != "System Volume Information"
            })
        })
        .collect();
    folders.sort_by_key(|p| p.to_string_lossy().to_lowercase());
    folders
}

/// The real devices, each with its exclusive capabilities in a few words.
pub fn list_devices() -> Result<Vec<DeviceChoice>, String> {
    let devices = jodel_output::devices().map_err(|e| e.to_string())?;
    Ok(devices
        .into_iter()
        .map(|device| {
            let summary = match jodel_output::capabilities(&device.id) {
                Ok(caps) => capability_summary(&caps),
                Err(error) => format!("cannot probe: {error}"),
            };
            let name = if device.is_default {
                format!("{} (default)", device.name)
            } else {
                device.name.clone()
            };
            DeviceChoice {
                id: Some(device.id),
                name,
                summary,
            }
        })
        .collect())
}

/// The rates a device takes exclusively in stereo (else in any channel
/// count), lowest first.
pub fn device_rates(caps: &jodel_output::Capabilities) -> Vec<u32> {
    let mut rates = caps.rates(2);
    if rates.is_empty() {
        rates = caps
            .channel_counts()
            .into_iter()
            .flat_map(|channels| caps.rates(channels))
            .collect();
    }
    rates.sort_unstable();
    rates.dedup();
    rates
}

/// `44.1 48 88.2 96 kHz · S24_3/S16 · 2 ch`: the exclusive rates, formats
/// and channel counts.
pub fn capability_summary(caps: &jodel_output::Capabilities) -> String {
    if caps.is_empty() {
        return "no exclusive formats".into();
    }
    let channels = caps.channel_counts();
    let main = if channels.contains(&2) {
        2
    } else {
        channels[0]
    };
    let rates: Vec<String> = caps
        .rates(main)
        .into_iter()
        .map(core::text::khz_short)
        .collect();
    let mut formats: Vec<jodel_output::SampleFormat> = caps
        .iter()
        .filter(|f| f.channels == main)
        .map(|f| f.sample)
        .collect();
    formats.sort();
    formats.dedup();
    formats.reverse();
    let formats: Vec<&str> = formats.iter().map(|f| f.name()).collect();
    let channels: Vec<String> = channels.iter().map(u16::to_string).collect();
    format!(
        "{} kHz · {} · {} ch",
        rates.join(" "),
        formats.join("/"),
        channels.join("/")
    )
}

/// The engine and what it plays on. The fields drop in order: the
/// players before the mixer they play on.
struct Engine {
    previews: Option<crate::preview::Previews>,
    player: Player,
    /// The engine's fader on the mixer.
    lead: Fader,
    /// The preview's strip on the desk (the engine's input holds its own).
    preview_strip: Strip,
    /// The microphone's port as last told.
    mic: MicView,
    /// The mixer's monitor: what the device plays.
    tap: SpectrumTap,
    mixer: Mixer,
}

/// The running player.
pub struct App {
    /// What the UI shows.
    pub state: State,
    engine: Option<Engine>,
    generation: Arc<AtomicU64>,
    output: Output,
    /// The id of the device the engine plays on, once it has found it.
    playing_on: Option<String>,
    tx: Sender<Action>,
    rx: Receiver<Action>,
    notify: Notify,
    devices: DeviceLister,
    volume: VolumeControl,
    saved: Config,
    baseline: Config,
    details: DetailsWorker,
    rhythm: RhythmWorker,
    lyrics: crate::lyrics::LyricsWorker,
    /// The microphone, open while Karaoke shows; how it opens, how they
    /// are listed, and why it last did not open (not tried again until
    /// Karaoke shows anew).
    mic: Option<jodel_output::input::Input>,
    open_mic: MicOpener,
    list_mics: MicLister,
    mic_failed: bool,
    /// The microphone being opened (its ask), and the count of asks; the
    /// channel the openings come back on.
    mic_opening: Option<u64>,
    mic_ask: u64,
    mic_tx: Sender<(u64, Result<jodel_output::input::Input, String>)>,
    mic_rx: Receiver<(u64, Result<jodel_output::input::Input, String>)>,
    /// Keeps the beat grids fixed by hand, where they are kept.
    grids: Option<crate::grids::GridStore>,
    /// When the last position event arrived, for [`App::heard`].
    position_at: Instant,
    /// What the engine was last told about keeping the stems
    /// ([`Command::KeepStems`]): while the Transition tab is shown.
    kept_stems: Option<bool>,
    /// The track and position the last frame drew, and when
    /// ([`App::heard_for_frame`]).
    last_drawn: Option<(Option<usize>, f64, Instant)>,
    /// When the microphone's line last went into the trace.
    mic_noted: Option<Instant>,
    /// The singer's voice the cleaner gave since the last frame.
    sung: Vec<jodel_output::VoiceBlock>,
    /// The singer's pitch, tracked at the voice's rate.
    yin: Option<(u32, jodel_rhythm::yin::Yin)>,
    /// The singing of the song in Karaoke mode, kept ([`crate::keep`]).
    kept: Option<crate::keep::Kept>,
    /// A recording of the echo canceller under way.
    echo_recording: Option<jodel_output::EchoRecording>,
    /// Where the tracks' stems are kept, for the engine to play them from.
    stem_store: Option<Arc<dyn jodel_playback::StemStore>>,
    search: Option<Arc<AtomicBool>>,
    analyzer: Analyzer,
    analyzed: Option<Instant>,
    /// The spectrum changed in the last [`App::spectrum`] (frozen while
    /// paused, it does not).
    spectrum_moved: bool,
    /// The spectrum [`App::spectrum`] gave last.
    last_spectrum: Option<SpectrumFrame>,
    art: Option<Art>,
    art_key: Option<String>,
    art_tx: Sender<(u64, Option<RgbImage>)>,
    art_rx: Receiver<(u64, Option<RgbImage>)>,
    next_art: u64,
    started: Instant,
    /// Where the session is kept, what was last saved there, and since
    /// when it has been out of date.
    session: Option<PathBuf>,
    session_saved: Option<Session>,
    session_stale: Option<Instant>,
    /// A move of the data folder ([`crate::mover`]), where it goes, and
    /// where it stood when last shown.
    mover: Option<Arc<crate::mover::Mover>>,
    move_to: Option<PathBuf>,
    move_seen: Option<crate::mover::Status>,
}

/// How long the session waits after a change before it is saved (a run of
/// moves in the queue is one write).
const SESSION_SAVE_AFTER: Duration = Duration::from_secs(2);

/// How far the position reported may be from the frames' own clock and
/// still be eased towards (further: a seek, a jump; it is taken at once).
const HELD_BACK: f64 = 0.25;

/// The share of the way the frames' clock goes to the position reported
/// each frame.
const CATCH_UP: f64 = 0.1;

/// The spectrum is analysed at most this often (30 a second), however fast
/// frames come.
const SPECTRUM_EVERY: Duration = Duration::from_millis(30);

/// How long a microphone giving only exact zeros is open before the tab
/// says so.
const MIC_SILENT_AFTER: Duration = Duration::from_millis(1_500);

/// How often the microphone's line goes into the trace while it is open.
const MIC_NOTE_EVERY: Duration = Duration::from_secs(1);

/// The microphone and its echo canceller, for the trace: the voice's peak
/// as captured and as cleaned (the music's echo out of it), what the
/// canceller does, and how the capture keeps up with the output (a skip
/// moves the echo's delay).
fn note_mic(mic: &jodel_output::Input, cleaned: Option<f32>, echo: &jodel_output::aec::EchoStatus) {
    let dbfs = |v: f32| 20.0 * v.max(1e-6).log10();
    let status = mic.status();
    let cleaned = cleaned.map_or_else(
        || "no output tapped".to_owned(),
        |v| format!("{:.0} dBFS cleaned", dbfs(v)),
    );
    let echo_line = if !mic.echo_cancel() {
        "off".to_owned()
    } else if echo.learnt {
        format!(
            "{:.0} dB out ({:.0} by the filter), the echo {} late",
            echo.removed_db,
            echo.filter_db,
            echo.delay
                .map_or_else(|| "?".to_owned(), |d| format!("{:.0} ms", d * 1000.0))
        )
    } else {
        "learning".to_owned()
    };
    crate::trace::note(format_args!(
        "microphone: {:.0} dBFS captured, {cleaned}; echo canceller {echo_line}; \
         {:.1} ms buffered (target {} frames at {} Hz), clocks {:+.0} ppm apart, \
         {} underruns, {} resyncs, {} overruns",
        dbfs(mic.peak()),
        status.buffered().as_secs_f64() * 1000.0,
        status.target,
        status.rate,
        status.adjust * 1e6,
        status.underruns,
        status.resyncs,
        status.overruns,
    ));
}

fn cover_key(cover: &Cover) -> String {
    match &cover.origin {
        CoverOrigin::Folder(path) => format!("file:{}", path.display()),
        CoverOrigin::Embedded => {
            let data = &cover.data;
            let sample: u64 = data
                .iter()
                .step_by((data.len() / 64).max(1))
                .fold(0u64, |h, &b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
            format!("embedded:{}:{sample}", data.len())
        }
    }
}

/// How late the music reaches the singer and the microphone when the echo
/// canceller has not found it (headphones, or the canceller off): the
/// output's buffer and a little more, in seconds.
const SUNG_LATE: f64 = 0.05;

impl App {
    /// Starts the engine and the workers, and lists the start folder.
    pub fn new(options: AppOptions) -> App {
        let (tx, rx) = mpsc::channel();
        let (art_tx, art_rx) = mpsc::channel();
        let (mic_tx, mic_rx) = mpsc::channel();
        let details = DetailsWorker::new(16, tx.clone(), options.notify.clone());
        let rhythm = RhythmWorker::new(
            tx.clone(),
            options.notify.clone(),
            options.rhythm_cache.clone(),
            options.grids.clone(),
        );
        let lyrics = {
            let (tx, notify) = (tx.clone(), options.notify.clone());
            let mut lyrics_options = options.lyrics.clone().unwrap_or_else(|| {
                let mut offline = crate::lyrics::Options::new(false);
                offline.cache = None;
                offline
            });
            lyrics_options.online &= options.config.karaoke_online;
            let (working_tx, working_notify) = (tx.clone(), notify.clone());
            lyrics_options.working = Some(Arc::new(move |path: &Path, on: bool| {
                let action = Action::Working {
                    path: path.to_owned(),
                    job: core::Job::Lyrics,
                    on,
                };
                if working_tx.send(action).is_ok() {
                    working_notify();
                }
            }));
            let (done_tx, done_notify) = (tx.clone(), notify.clone());
            lyrics_options.done = Some(Arc::new(move |path: &Path, facts| {
                let action = Action::Ready {
                    path: path.to_owned(),
                    job: core::Job::Lyrics,
                    facts: Some(core::Facts::Lyrics(facts)),
                };
                if done_tx.send(action).is_ok() {
                    done_notify();
                }
            }));
            crate::lyrics::LyricsWorker::new(lyrics_options, move |path, answer| {
                if tx.send(Action::Lyrics(path, answer)).is_ok() {
                    notify();
                }
            })
        };
        let grids = options
            .grids
            .clone()
            .map(|dir| crate::grids::GridStore::new(dir, tx.clone(), options.notify.clone()));
        // The session: the browser where it was (when no folder was
        // given), the queue as it was (when nothing plays at once).
        let restored = options.session.as_deref().and_then(Session::load);
        // Not checked here (a network drive waking up would hold the window
        // up): a folder that cannot be listed gives way to the start one.
        let remembered = restored
            .as_ref()
            .filter(|_| options.restore_browser)
            .and_then(|s| s.browser.clone())
            .filter(|d| *d != options.dir);
        let dir = remembered.clone().unwrap_or_else(|| options.dir.clone());
        let mut state = State::new(dir.clone(), options.config.clone(), options.size);
        state.browser.fallback = remembered.map(|_| options.dir.clone());
        if let Some(selected) = restored.as_ref().and_then(|s| s.selected.clone()) {
            state.browser.last_child.insert(dir.clone(), selected);
        }
        if let Some(queues) = &options.queues {
            let _ = std::fs::create_dir_all(queues);
        }
        state.queues_dir = options.queues.clone();
        // There is always a lyrics worker (one without the web or a cache
        // where the options give none): the queue waits for it too.
        state.prepares_lyrics = true;
        state.karaoke.model = match &options.model {
            Some(path) if path.is_file() => core::karaoke::ModelState::Ready,
            Some(_) => core::karaoke::ModelState::Coming,
            None => core::karaoke::ModelState::Missing,
        };
        state.config_file = options.config_file.clone();
        state.null_output = matches!(options.output, Output::Null(_));
        state.autoplay = options.autoplay.clone();
        let mut app = App {
            state,
            engine: None,
            generation: Arc::new(AtomicU64::new(0)),
            output: options.output.clone(),
            playing_on: None,
            tx,
            rx,
            notify: options.notify.clone(),
            devices: options.devices.clone(),
            volume: options.volume.clone(),
            saved: options.saved.clone(),
            baseline: options.config.clone(),
            details,
            rhythm,
            lyrics,
            mic: None,
            open_mic: options.mic.clone(),
            list_mics: options.mics.clone(),
            mic_failed: false,
            mic_opening: None,
            mic_ask: 0,
            mic_tx,
            mic_rx,
            grids,
            position_at: Instant::now(),
            kept_stems: None,
            last_drawn: None,
            mic_noted: None,
            sung: Vec::new(),
            yin: None,
            kept: None,
            echo_recording: None,
            stem_store: options
                .rhythm_cache
                .as_deref()
                .map(|dir| crate::stems_cache::Store::shared(crate::stems_cache::beside(dir))),
            search: None,
            analyzer: Analyzer::new(),
            analyzed: None,
            spectrum_moved: true,
            last_spectrum: None,
            art: None,
            art_key: None,
            art_tx,
            art_rx,
            next_art: 0,
            started: Instant::now(),
            session: options.session.clone(),
            session_saved: restored.clone(),
            session_stale: None,
            mover: options.mover.clone(),
            move_to: options.data.as_ref().map(|d| d.0.clone()),
            move_seen: None,
        };
        app.state.data_dir = options.data.as_ref().map(|d| d.0.clone());
        // The best scores sung, from the data folder.
        if let Some(data) = &options.data {
            app.state.karaoke.bests = crate::scores::load(&data.scores());
        }
        let output = app.output.clone();
        let started = Instant::now();
        app.start_engine(&output);
        crate::trace::took("app: engine started", started, true);
        if let Some(session) =
            restored.filter(|s| !s.queue.is_empty() && options.autoplay.is_none())
        {
            app.state.queue.cursor = session.current.unwrap_or(0);
            app.engine_command(Command::Load(session.queue));
        }
        for note in options.notes {
            app.dispatch(Action::Failed(note));
        }
        app.run(Effect::ListDir(dir));
        app
    }

    /// The data folder goes into `parent`: the move written down (from now
    /// on what is written goes there), the settings saved, the files moved
    /// in the background ([`crate::mover`]).
    fn move_data(&mut self, parent: PathBuf) {
        if self.mover.as_ref().is_some_and(|m| !m.status().finished) {
            return self.dispatch(Action::Failed(
                "The data folder is moving already: choose another once it is there".into(),
            ));
        }
        let Some(from) = self.state.data_dir.clone() else {
            return;
        };
        let to = crate::dirs::Data::in_parent(&parent).0;
        let settings = self
            .state
            .config_file
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .or_else(crate::dirs::settings_dir);
        let Some(settings) = settings else {
            return self.dispatch(Action::Failed(
                "No settings folder to note the move in".into(),
            ));
        };
        if let Err(error) = crate::mover::begin(&settings, &[(from.clone(), to.clone())]) {
            return self.dispatch(Action::Failed(format!(
                "Could not note the move in {}: {error}",
                settings.display()
            )));
        }
        // The system's place is kept as none, so it follows the system.
        let default = crate::dirs::default_data_parent();
        self.state.config.data_folder = (default.as_ref() != Some(&parent)).then_some(parent);
        self.run(Effect::SaveConfig);
        self.state.data_dir = Some(to.clone());
        self.move_to = Some(to.clone());
        self.move_seen = None;
        self.state.data_move = Some(core::DataMove {
            to: to.clone(),
            ..Default::default()
        });
        self.mover = crate::mover::Mover::start(settings, self.notify.clone()).map(Arc::new);
        if self.mover.is_none() {
            // Nothing there to move: it is where it goes already.
            self.state.data_move = Some(core::DataMove {
                to: to.clone(),
                share: 1.0,
                finished: true,
                ..Default::default()
            });
        }
        self.dispatch(Action::Info(format!(
            "The data folder is now {}",
            to.display()
        )));
    }

    fn start_engine(&mut self, output: &Output) {
        self.engine = None;
        self.playing_on = None;
        let mut target: Box<dyn OutputTarget> = match output {
            Output::Null(clock) => Box::new(NullTarget::new(any_capabilities(), *clock)),
            Output::Device(query) => {
                match DeviceTarget::find(query.as_deref().unwrap_or("default")) {
                    Ok(target) => {
                        self.playing_on = Some(target.device().id.clone());
                        Box::new(target)
                    }
                    Err(error) => {
                        self.dispatch(Action::Failed(format!(
                            "No output device ({error}); choose one in the settings (,)"
                        )));
                        return;
                    }
                }
            }
        };
        // What the device takes, asked here once (the engine reuses it), so
        // the settings offer only the rate caps it can satisfy and the
        // engine starts with the cap in force before anything plays.
        let capabilities = target.capabilities().ok();
        self.dispatch(Action::DeviceRates(capabilities.as_ref().map(device_rates)));
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let mixer = Mixer::open(
            target,
            MixerOptions {
                capabilities: capabilities.clone(),
                ..MixerOptions::default()
            },
        );
        // The fine bass block at 384 kHz, plus the device buffer that runs
        // ahead of what is heard.
        let tap = SpectrumTap::new(1 << 19);
        mixer.set_monitor(Some(Box::new(tap.clone())));
        let player_strip = mixer.strip("Player");
        let preview_strip = mixer.strip("Preview");
        let (lead_input, lead) = mixer.input(&player_strip, Role::Lead, 1.0);
        let options = PlayerOptions {
            settings: self.state.engine_settings(),
            position_interval: Duration::from_millis(100),
            capabilities,
            separator: Some(gpu_separator),
            stem_store: self.stem_store.clone(),
            ..PlayerOptions::default()
        };
        let (player, events) = Player::new(Box::new(lead_input), options);
        // The bars a jump can reach are kept ([`Op::JumpBar`]): four either
        // side of the one heard down to 48 BPM, and that one.
        let _ = player.jump_window(JUMP_WINDOW, JUMP_WINDOW);
        self.kept_stems = None;
        // Whatever the old engine had armed or mixing is gone with it.
        core::transition::draft::engine_restarted(&mut self.state);
        let tx = self.tx.clone();
        let notify = self.notify.clone();
        let current = self.generation.clone();
        thread::spawn(move || {
            for event in events {
                if current.load(Ordering::SeqCst) != generation {
                    return;
                }
                if tx.send(Action::Engine(event)).is_err() {
                    return;
                }
                notify();
            }
            // The engine's thread ended while it was still the one in use:
            // it failed.
            if current.load(Ordering::SeqCst) == generation
                && tx.send(Action::EngineGone(generation)).is_ok()
            {
                notify();
            }
        });
        self.engine = Some(Engine {
            previews: None,
            player,
            lead,
            preview_strip,
            mic: MicView::default(),
            tap,
            mixer,
        });
        // A new mixer takes the microphone over.
        self.set_mic();
    }

    /// Applies an action and carries out its effects.
    pub fn dispatch(&mut self, action: Action) {
        let gone = match action {
            // An engine already replaced.
            Action::EngineGone(generation)
                if generation != self.generation.load(Ordering::SeqCst) =>
            {
                return;
            }
            Action::EngineGone(_) => true,
            _ => false,
        };
        if let Action::Engine(Event::TrackStarted(start)) = &action {
            self.cover_for(start.info.cover.as_ref());
        }
        if let Action::Engine(
            Event::Position(_)
            | Event::TrackStarted(_)
            | Event::Seeked { .. }
            | Event::Jumped { .. }
            | Event::State(_),
        ) = &action
        {
            self.position_at = Instant::now();
        }
        let started = Instant::now();
        let what =
            crate::trace::on().then(|| format!("{action:?}").chars().take(120).collect::<String>());
        for effect in core::update(&mut self.state, action) {
            self.run(effect);
        }
        if let Some(what) = what {
            crate::trace::took(format_args!("update {what}"), started, false);
        }
        self.note_session();
        self.mic_turn();
        // A best score beaten: kept in the data folder at once.
        if std::mem::take(&mut self.state.karaoke.bests_changed) {
            if let Some(dir) = &self.state.data_dir {
                let file = crate::dirs::Data(dir.clone()).scores();
                if let Err(e) = crate::scores::save(&file, &self.state.karaoke.bests) {
                    self.dispatch(Action::Failed(format!(
                        "The best scores were not saved: {e}"
                    )));
                }
            }
        }
        // On the Transition tab the playing track's stems are kept while its
        // file plays, so a stem switched off has those from before it for a
        // jump back (a separation runs meanwhile; elsewhere it does not).
        let transition = self.state.side == core::Side::Transition;
        if self.kept_stems != Some(transition) {
            if let Some(engine) = &self.engine {
                let _ = engine.player.keep_stems(transition);
                self.kept_stems = Some(transition);
            }
        }
        if gone {
            self.restart_engine();
        }
    }

    /// Opens the microphone while Karaoke shows (in the background: an
    /// audio device can take a while to answer) and closes it after. It is
    /// only listened to: nothing of it is played.
    fn mic_turn(&mut self) {
        let want = core::karaoke::mic_wanted(&self.state);
        if want && self.mic.is_none() && self.mic_opening.is_none() && !self.mic_failed {
            self.mic_ask += 1;
            let ask = self.mic_ask;
            self.mic_opening = Some(ask);
            let id = self.state.config.karaoke_mic_device.clone();
            let (open, tx, notify) = (
                self.open_mic.clone(),
                self.mic_tx.clone(),
                self.notify.clone(),
            );
            thread::spawn(move || {
                let result = open(id.as_deref());
                if tx.send((ask, result)).is_ok() {
                    notify();
                }
            });
        } else if !want {
            self.mic_failed = false;
            // One being opened is dropped when it comes.
            self.mic_opening = None;
            if self.mic.take().is_some() {
                self.state.karaoke.mic = core::karaoke::MicShown::default();
                self.set_mic();
            }
        }
        if let Some(mic) = &self.mic {
            mic.set_echo_cancel(self.state.config.karaoke_echo);
        }
    }

    /// The singing kept saved, once its song is over: another song in
    /// the mode, the mode left, or keeping turned off (`now`: whatever).
    fn save_kept(&mut self, now: bool) {
        let Some(kept) = &self.kept else {
            return;
        };
        let state = &self.state;
        let still = crate::core::karaoke::in_mode(state)
            && state.config.karaoke_keep
            && crate::core::karaoke::song_index(state)
                .and_then(|i| state.queue.paths.get(i))
                .is_some_and(|p| *p == kept.song);
        if still && !now {
            return;
        }
        let Some(kept) = self.kept.take() else {
            return;
        };
        if kept.is_empty() {
            return;
        }
        let dir = match self.state.config_file.as_deref().and_then(Path::parent) {
            Some(dir) => dir.join("recordings"),
            None => std::env::temp_dir(),
        };
        self.dispatch(match kept.save(&dir) {
            Ok(folder) => Action::Info(format!("Your singing kept: {}", folder.display())),
            Err(why) => Action::Failed(format!("Your singing could not be kept: {why}")),
        });
    }

    /// Records 30 s of what the echo canceller works with, beside the
    /// settings file (`recordings`), for tuning it to a room.
    fn record_echo(&mut self) {
        let Some(mic) = &self.mic else {
            return self.dispatch(Action::Failed("No microphone open".into()));
        };
        let dir = match self.state.config_file.as_deref().and_then(Path::parent) {
            Some(dir) => dir.join("recordings"),
            None => std::env::temp_dir(),
        };
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let path = dir.join(format!("echo-{stamp}.wav"));
        let started = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|()| mic.record_echo(&path, 30.0));
        match started {
            Ok(recording) => {
                self.echo_recording = Some(recording);
                self.dispatch(Action::Info(format!(
                    "Recording 30 s for the echo canceller (sing over the music): {}",
                    path.display()
                )));
            }
            Err(why) => self.dispatch(Action::Failed(format!("Could not record: {why}"))),
        }
    }

    /// A microphone opened (or not) in the background: taken if it is the
    /// one still wanted.
    fn mic_opened(&mut self, ask: u64, result: Result<jodel_output::input::Input, String>) {
        if self.mic_opening != Some(ask) {
            return;
        }
        self.mic_opening = None;
        match result {
            Ok(input) => {
                self.state.karaoke.mic = core::karaoke::MicShown {
                    name: Some(input.info().name.clone()),
                    error: None,
                    peak: 0.0,
                    silent: false,
                    echo: None,
                };
                input.set_echo_cancel(self.state.config.karaoke_echo);
                self.mic = Some(input);
                self.set_mic();
            }
            Err(why) => {
                self.mic_failed = true;
                self.state.karaoke.mic = core::karaoke::MicShown {
                    name: None,
                    error: Some(why),
                    peak: 0.0,
                    silent: false,
                    echo: None,
                };
            }
        }
    }
    /// Gives the mixer the microphone (or none): its port on the desk.
    fn set_mic(&self) {
        if let Some(engine) = &self.engine {
            engine.mixer.set_mic(self.mic.clone());
        }
    }

    /// `i`: the next microphone, kept in the settings.
    fn next_mic(&mut self) {
        let list = match (self.list_mics)() {
            Ok(list) if !list.is_empty() => list,
            Ok(_) => return self.dispatch(Action::Failed("No microphone found".into())),
            Err(why) => {
                return self.dispatch(Action::Failed(format!("Listing microphones failed: {why}")))
            }
        };
        let current = self
            .mic
            .as_ref()
            .map(|m| m.info().id.clone())
            .or_else(|| self.state.config.karaoke_mic_device.clone());
        let at = current
            .and_then(|id| list.iter().position(|(i, _)| *i == id))
            .map_or(0, |i| (i + 1) % list.len());
        let (id, name) = list[at].clone();
        self.state.config.karaoke_mic_device = Some(id);
        self.mic = None;
        self.mic_failed = false;
        self.mic_opening = None;
        self.state.karaoke.mic = core::karaoke::MicShown::default();
        self.set_mic();
        self.run(Effect::SaveConfig);
        self.mic_turn();
        self.state.message = Some(core::Message {
            text: format!("Microphone: {name}"),
            level: core::Level::Info,
            until_ms: self.state.now_ms + core::MESSAGE_MS,
        });
    }

    /// Starts the engine again on the same output after it failed, with
    /// the queue loaded and nothing playing.
    fn restart_engine(&mut self) {
        let queue = self.state.queue.paths.clone();
        self.engine = None;
        self.state.playing = core::Playing::default();
        let output = self.output.clone();
        self.start_engine(&output);
        if let (Some(engine), false) = (&self.engine, queue.is_empty()) {
            let _ = engine.player.send(Command::Load(queue));
        }
    }

    /// Takes in everything the engine and the workers sent. Returns whether
    /// anything arrived.
    pub fn pump(&mut self) -> bool {
        if self
            .session_stale
            .is_some_and(|since| since.elapsed() >= SESSION_SAVE_AFTER)
        {
            self.save_session();
        }
        let mut any = false;
        while let Ok(action) = self.rx.try_recv() {
            any = true;
            self.dispatch(action);
        }
        while let Ok((ask, result)) = self.mic_rx.try_recv() {
            any = true;
            self.mic_opened(ask, result);
        }
        any |= self.pump_previews();
        // Bit-perfect only while the mixer passes the song through; a
        // device that refused the microphone's tap is told, each time.
        if let Some(engine) = &mut self.engine {
            let mixed = !engine.mixer.passthrough();
            if self.state.playing.mixed != mixed {
                self.state.playing.mixed = mixed;
                any = true;
            }
            let mic = engine.mixer.mic();
            if mic != engine.mic {
                engine.mic = mic.clone();
                if let MicPort::CannotHear(why) = mic.port {
                    any = true;
                    self.dispatch(Action::Info(format!(
                        "The microphone cannot hear this output (its echo stays in): {why}"
                    )));
                }
            }
        }
        // A move of the data folder, for the settings' row.
        if let Some(mover) = &self.mover {
            let status = mover.status();
            if self.move_seen.as_ref() != Some(&status) {
                any = true;
                self.state.data_move = Some(core::DataMove {
                    to: self.move_to.clone().unwrap_or_default(),
                    share: status.share(),
                    done: status.done,
                    total: status.total,
                    failure: status.failure.clone(),
                    finished: status.finished,
                });
                self.move_seen = Some(status);
            }
        }
        while let Ok((id, image)) = self.art_rx.try_recv() {
            if id == self.next_art {
                any = true;
                self.art = image.map(|image| Art {
                    id,
                    image: Arc::new(image),
                });
            }
        }
        let now = self.started.elapsed().as_millis() as u64;
        // A tick every 250 ms, and one as soon as something is due (which
        // then shows).
        let due = self.state.due().is_some_and(|due| due <= now);
        if now / 250 != self.state.now_ms / 250 || due {
            self.dispatch(Action::Tick(now));
            any |= due;
        }
        any
    }

    /// Waits up to `timeout` for something from the engine or the workers,
    /// then takes it in (for headless use and tests).
    pub fn pump_wait(&mut self, timeout: Duration) -> bool {
        match self.rx.recv_timeout(timeout) {
            Ok(action) => {
                self.dispatch(action);
                self.pump();
                true
            }
            Err(_) => self.pump(),
        }
    }

    /// The singer's pitch from the voice the cleaner gave this frame
    /// (`self.sung`), each frame of it at the time in the song it was sung
    /// (the output frame playing as it was caught, less `delay`: how late
    /// the music reaches the microphone, as the echo canceller found it,
    /// else [`SUNG_LATE`]), held against the song's notes in Karaoke mode
    /// ([`crate::core::singing`]). `delay` none: no microphone; the notes
    /// made all the same, nothing heard.
    fn sing(&mut self, delay: Option<Option<f64>>) {
        let state = &mut self.state;
        let song = crate::core::karaoke::song_index(state)
            .filter(|_| crate::core::karaoke::in_mode(state));
        let (Some(index), Some(track)) = (song, state.playing.track.as_ref()) else {
            return;
        };
        if track.index != index {
            return;
        }
        let out_rate = f64::from(track.output.format.rate.max(1));
        let Some((_, frame0, seconds0)) = state.playing.anchor.filter(|a| a.0 == index) else {
            return;
        };
        let path = state.queue.paths[index].clone();
        let lyrics = match crate::core::karaoke::lyrics_of(state, index) {
            Some(crate::core::karaoke::SongLyrics::Found { shown, .. }) => Arc::clone(shown),
            _ => return,
        };
        let waveform = match state.rhythm.get(&path) {
            Some(crate::core::Rhythm::Ready(r)) => r.waveform.clone(),
            _ => None,
        };
        let Some(waveform) = waveform.filter(|w| w.melody.is_some()) else {
            return;
        };
        let level = state.config.karaoke_level;
        let singing = &mut state.karaoke.singing;
        if let Some(melody) = &waveform.melody {
            singing.prepare(
                &path,
                &lyrics,
                melody,
                Arc::as_ptr(&waveform) as usize,
                level,
            );
        }
        // Without a microphone the notes are still made (the lane shows
        // them), nothing heard.
        let Some(delay) = delay else {
            return;
        };
        let late = delay.unwrap_or(SUNG_LATE);
        // A frame's pitch is that of the window before it: its middle.
        let centre = jodel_rhythm::yin::WINDOW / 2.0;
        // Kept as it is heard, where that is on.
        if state.config.karaoke_keep {
            if self.kept.as_ref().is_none_or(|k| k.song != path) {
                self.kept = Some(crate::keep::Kept::new(
                    path.clone(),
                    Arc::clone(&lyrics),
                    Arc::clone(&waveform),
                    level.name(),
                ));
            }
            if let Some(kept) = self.kept.as_mut() {
                for block in &self.sung {
                    let at = seconds0 + (block.stream_frame as f64 - frame0 as f64) / out_rate;
                    kept.push(&block.samples, block.rate, at, block.continues, late);
                }
            }
        }
        for block in &self.sung {
            let fresh = match &self.yin {
                Some((rate, _)) => *rate != block.rate || !block.continues,
                None => true,
            };
            if fresh {
                self.yin = Some((block.rate, jodel_rhythm::yin::Yin::new(block.rate)));
            }
            let Some((_, yin)) = self.yin.as_mut() else {
                continue;
            };
            let factor = out_rate / f64::from(block.rate.max(1));
            for (i, &x) in block.samples.iter().enumerate() {
                yin.push(&[x], |cents| {
                    let frame = block.stream_frame as f64 + i as f64 * factor;
                    let t = seconds0 + (frame - frame0 as f64) / out_rate - late - centre;
                    singing.hear(t, cents);
                });
            }
        }
    }

    /// [`App::heard`] for a frame about to be drawn, kept in the state so
    /// that a click on the Transition tab lands where it was seen.
    pub fn heard_for_frame(&mut self) -> f64 {
        let reported = self.heard();
        let mut heard = reported;
        let now = Instant::now();
        // The position comes every 100 ms, a little early or late: drawn as it
        // is, what scrolls by (the Transition lanes, the Karaoke tab's words)
        // steps unevenly. While the track plays on, the frames keep a clock of
        // their own, running with time from the last frame's position and
        // going a share of the way to the one reported, never back; a seek or
        // a jump (further off) is taken at once.
        let track = self.state.playing.track.as_ref().map(|t| t.index);
        if self.state.playing.state == PlayState::Playing {
            if let Some((was, last, at)) = self.last_drawn {
                let ahead = last + now.duration_since(at).as_secs_f64();
                let off = reported - ahead;
                if was == track && off.abs() < HELD_BACK {
                    heard = (ahead + CATCH_UP * off).max(last);
                }
            }
        }
        self.last_drawn = Some((track, heard, now));
        // How late the music reaches the microphone, once the voice is in.
        let mut sing = None;
        if let Some(mic) = &self.mic {
            // The singer as the cleaner gives them (the music's echo out)
            // where an output is tapped; else as captured.
            let cleaned = mic.voice_peak();
            self.state.karaoke.mic.peak = cleaned.unwrap_or_else(|| mic.peak());
            self.state.karaoke.mic.silent = mic.only_silence(MIC_SILENT_AFTER);
            let status = mic.echo_status();
            self.state.karaoke.mic.echo = (mic.echo_cancel() && cleaned.is_some()).then_some((
                status.removed_db,
                status.learnt,
                status.delay,
            ));
            // The voice cleaned, block by block with where in the song it
            // was sung: for scoring the singing once that comes. Taken so
            // the cleaner's queue keeps the newest.
            self.sung.clear();
            mic.take_voice(&mut self.sung);
            sing = Some(status.delay.map(f64::from));
            if crate::trace::on() && self.mic_noted.is_none_or(|t| t.elapsed() >= MIC_NOTE_EVERY) {
                self.mic_noted = Some(Instant::now());
                note_mic(mic, cleaned, &status);
            }
        }
        self.sing(sing);
        self.save_kept(false);
        // A recording of the echo canceller done: where it is.
        if let Some(done) = self.echo_recording.as_mut().and_then(|r| r.done()) {
            self.echo_recording = None;
            self.dispatch(match done {
                Ok(path) => Action::Info(format!("Recorded: {}", path.display())),
                Err(why) => Action::Failed(format!("The recording failed: {why}")),
            });
        }
        self.state.drawn_heard = Some(heard);
        self.state.drawn_ms = Some(self.now_ms());
        heard
    }

    /// Seconds into the playing track as heard now: the last position
    /// reported (every 100 ms) and the time since, while playing.
    pub fn heard(&self) -> f64 {
        let playing = &self.state.playing;
        let since = if playing.state == PlayState::Playing {
            self.position_at.elapsed().as_secs_f64().min(0.25)
        } else {
            0.0
        };
        playing.position + since
    }

    /// The milliseconds since the start (the core's clock).
    pub fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// The cover of the track heard, once decoded.
    pub fn art(&self) -> Option<&Art> {
        self.art.as_ref()
    }

    /// Sets the cover directly (tests and screenshots).
    pub fn set_art(&mut self, image: RgbImage) {
        self.next_art += 1;
        self.art = Some(Art {
            id: self.next_art,
            image: Arc::new(image),
        });
    }

    /// Whether the picture changes by itself (something is playing, the
    /// spectrum still falling, the microphone's meter shown), so the
    /// frontend should keep redrawing. A spectrum standing still (frozen
    /// while paused) is no reason: the window then stops drawing until
    /// something changes.
    pub fn animating(&self) -> bool {
        self.loading()
            // A preview's players come and go as their fades end.
            || self.engine.as_ref().is_some_and(|e| e.previews.is_some())
            || self.state.playing.state == PlayState::Playing
            // The Karaoke tab's meter moves with the voice, music or not.
            || (self.mic.is_some() && core::karaoke::on_tab(&self.state))
            || self.state.seek_drag.is_some()
            || self.state.transition.slide.is_some()
            || self.state.karaoke.resume_at.is_some()
            || (self.state.playing.state != PlayState::Paused
                && self.spectrum_moved
                && !self.analyzer.is_quiet())
    }

    /// Whether something shown is still loading, with a spinner turning:
    /// the folder listed, the microphone opening, lyrics looked up, a
    /// track in the queue shown being got ready.
    fn loading(&self) -> bool {
        let state = &self.state;
        let karaoke = core::karaoke::mic_wanted(state);
        state.browser.loading
            || (state.screen == core::Screen::Main
                && state.side == core::Side::Queue
                && !state.working.is_empty())
            || (karaoke && self.mic_opening.is_some())
            || (karaoke
                && state
                    .karaoke
                    .lyrics
                    .values()
                    .any(|l| matches!(l, core::karaoke::SongLyrics::Pending)))
    }

    /// Whether what moves needs 60 frames a second to look smooth: what
    /// follows the pointer (a seek drag, a view being dragged), a block
    /// sliding, and words to be read as they go by (Karaoke mode; the
    /// Karaoke tab's voice lane is [`App::layer_moves`]). The spectrum, the
    /// clock and the Transition tab's lanes going by as a track plays (a few
    /// pixels a frame at 30) do with 30.
    pub fn smooth(&self) -> bool {
        let state = &self.state;
        state.seek_drag.is_some()
            || state.transition.slide.is_some()
            || state.transition.pan.is_some()
            || state.karaoke.dragging.is_some()
            // Karaoke mode: the lyrics fill and the count-in's numbers
            // shrink as they are sung.
            || (core::karaoke::in_mode(state) && state.playing.state == PlayState::Playing)
            // A paused song's count-in.
            || state.karaoke.resume_at.is_some()
    }

    /// Whether a lane that follows the playhead goes by (a song playing:
    /// the Karaoke tab's voice lane, the Transition tab's playing track): it
    /// needs a frame every refresh too, but where the window scrolls it by
    /// itself ([`crate::gui::canvas::Canvas::layer_frame`]) only the lane
    /// moves in most of them.
    pub fn layer_moves(&self) -> bool {
        let state = &self.state;
        let playing = state.playing.state == PlayState::Playing;
        // The Transition tab's playing track, its view following the playhead.
        let transition = state.screen == core::Screen::Main
            && state.side == core::Side::Transition
            && state.palette.is_none()
            && core::transition::tab::block(state, core::lanes::Block::Now)
                .start
                .is_none()
            && state.transition.slide.is_none();
        playing && (core::karaoke::on_tab(state) || transition)
    }

    /// The spectrum of what is heard now: computed only where the screen
    /// shows one, at most [`SPECTRUM_EVERY`] apart (the last otherwise).
    pub fn spectrum(&mut self) -> SpectrumFrame {
        let now = Instant::now();
        let shown = core::layout::compute(&self.state)
            .bottom()
            .is_some_and(|b| b.spectrum.is_some());
        let recent = self
            .analyzed
            .is_some_and(|then| now - then < SPECTRUM_EVERY);
        if !shown || recent {
            if !shown {
                // Shown again later: it starts afresh, not from long ago.
                self.analyzed = None;
                self.spectrum_moved = false;
            }
            if let Some(last) = &self.last_spectrum {
                return last.clone();
            }
        }
        let seconds = self
            .analyzed
            .map_or(0.0, |then| (now - then).as_secs_f32().min(0.5));
        self.analyzed = Some(now);
        match &self.engine {
            // A song, or a preview (with no song too): the mixer's tap has
            // what the device plays.
            Some(engine)
                if engine.previews.is_some() || self.state.playing.state != PlayState::Stopped =>
            {
                self.analyzer.update(&engine.tap, seconds)
            }
            // Nothing is heard: back to the empty spectrum of the start,
            // without the last track's rate on the axis.
            _ => self.analyzer.reset(),
        }
        let frame = self.analyzer.frame();
        self.spectrum_moved = self.last_spectrum.as_ref() != Some(&frame);
        self.last_spectrum = Some(frame.clone());
        frame
    }

    /// The engine's spectrum tap, if an engine runs.
    pub fn tap(&self) -> Option<SpectrumTap> {
        self.engine.as_ref().map(|e| e.tap.clone())
    }

    fn cover_for(&mut self, cover: Option<&Cover>) {
        let Some(cover) = cover else {
            self.art = None;
            self.art_key = None;
            self.next_art += 1;
            return;
        };
        let key = cover_key(cover);
        if self.art_key.as_ref() == Some(&key) {
            return;
        }
        self.art_key = Some(key);
        self.next_art += 1;
        let id = self.next_art;
        let data = cover.data.clone();
        let tx = self.art_tx.clone();
        let notify = self.notify.clone();
        thread::spawn(move || {
            let image = art::decode(&data);
            if tx.send((id, image)).is_ok() {
                notify();
            }
        });
    }

    /// Carries out one effect.
    pub fn run(&mut self, effect: Effect) {
        match effect {
            Effect::Engine(command) => self.engine_command(command),
            Effect::Preview(what) => self.preview(what),
            Effect::ListDir(dir) => {
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                thread::spawn(move || {
                    let result =
                        browser::list_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()));
                    if tx.send(Action::Listed { dir, result }).is_ok() {
                        notify();
                    }
                });
            }
            Effect::Jump { text, from } => {
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                thread::spawn(move || {
                    let action = match browser::resolve_jump(&text, &from) {
                        Some((dir, select)) => Action::JumpTo { dir, select },
                        None => Action::JumpMissed { text },
                    };
                    if tx.send(action).is_ok() {
                        notify();
                    }
                });
            }
            Effect::WantDetails(paths) => self.details.want(paths),
            Effect::WantRhythm(paths, recipe) => self.rhythm.want(paths, recipe),
            Effect::WantLyrics(paths) => self.lyrics.want(paths),
            Effect::Prepare(paths, recipe) => {
                self.lyrics.prepare(paths.clone());
                self.rhythm.prepare(paths, recipe);
            }
            Effect::NextMic => self.next_mic(),
            Effect::RecordEcho => self.record_echo(),
            Effect::Search { id, root, query } => {
                if let Some(cancel) = self.search.take() {
                    cancel.store(true, Ordering::Relaxed);
                }
                let cancel = Arc::new(AtomicBool::new(false));
                self.search = Some(cancel.clone());
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                thread::spawn(move || {
                    browser::search(&root, &query, &cancel, |hits| {
                        if tx.send(Action::SearchHits { id, hits }).is_ok() {
                            notify();
                        }
                    });
                    if tx.send(Action::SearchDone { id }).is_ok() {
                        notify();
                    }
                });
            }
            Effect::CancelSearch => {
                if let Some(cancel) = self.search.take() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Effect::Walk { root, purpose } => {
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                thread::spawn(move || {
                    let paths = browser::walk_playable(&root);
                    if tx.send(Action::Walked { purpose, paths }).is_ok() {
                        notify();
                    }
                });
            }
            Effect::ListDevices => {
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                let devices = self.devices.clone();
                thread::spawn(move || {
                    if tx.send(Action::Devices(devices())).is_ok() {
                        notify();
                    }
                });
            }
            Effect::ListFolders(dir) => {
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                thread::spawn(move || {
                    let folders = list_folders(dir.as_deref());
                    if tx.send(Action::FoldersListed(dir, folders)).is_ok() {
                        notify();
                    }
                });
            }
            Effect::MoveData(parent) => self.move_data(parent),
            Effect::RetryMove => {
                if let Some(mover) = &self.mover {
                    mover.retry();
                }
            }
            Effect::SwitchDevice(id) => self.switch_device(id),
            Effect::DeviceVolume(change) => {
                let changed = change != core::VolumeChange::Read;
                let Some(query) = volume_device(&self.output, self.playing_on.as_deref()) else {
                    self.dispatch(Action::DeviceVolume {
                        result: Ok(None),
                        changed,
                    });
                    return;
                };
                let tx = self.tx.clone();
                let notify = self.notify.clone();
                let volume = self.volume.clone();
                thread::spawn(move || {
                    let result = volume(query.as_deref(), change);
                    if tx.send(Action::DeviceVolume { result, changed }).is_ok() {
                        notify();
                    }
                });
            }
            Effect::SaveGrid(path, grid) => {
                if let Some(store) = &self.grids {
                    store.save(path, grid);
                }
            }
            Effect::ResetGrid(path) => {
                if let Some(store) = &self.grids {
                    store.remove(path);
                }
            }
            Effect::ReadClipboard => {
                let text = arboard::Clipboard::new().and_then(|mut c| c.get_text());
                match text {
                    Ok(text) => self.dispatch(Action::Paste(text)),
                    Err(e) => self.dispatch(Action::Failed(format!("Nothing to paste: {e}"))),
                }
            }
            Effect::SaveQueue(tracks) => {
                let result = match &self.state.queues_dir {
                    Some(dir) => {
                        let path = crate::session::queue_name(dir, &tracks);
                        crate::session::write_playlist(&path, &tracks, &[])
                            .map(|()| path)
                            .map_err(|e| e.to_string())
                    }
                    None => Err("no folder to save queues in".into()),
                };
                self.dispatch(Action::QueueSaved(result));
            }
            Effect::ReadPlaylist { path, how } => {
                let tracks = crate::session::read_playlist(&path).map_err(|e| e.to_string());
                self.dispatch(Action::PlaylistRead { path, tracks, how });
            }
            Effect::SaveConfig => {
                // Only what was changed in the player is saved; the command
                // line's overrides stay out of the file.
                let now = self.state.config.clone();
                let base = &self.baseline;
                let mut file = self.saved.clone();
                // Every setting, each taken over where it changed. The
                // pattern names every field (it fails to build without one),
                // so a new one cannot be left out of the file unnoticed.
                macro_rules! carry {
                    ($($field:ident),* $(,)?) => {{
                        let Config { $($field: _),* } = &now;
                        $(if now.$field != base.$field {
                            file.$field = now.$field.clone();
                        })*
                    }};
                }
                carry!(
                    device,
                    max_rate,
                    resample,
                    headroom,
                    bpm,
                    beat_voting,
                    beat_methods,
                    beat_method,
                    stems,
                    skip_mix,
                    skip_preset,
                    transition_preset,
                    transition_tempo,
                    transition_lyrics,
                    stems_on_bar,
                    karaoke_voice,
                    karaoke_mic_device,
                    karaoke_online,
                    karaoke_lead,
                    karaoke_echo,
                    karaoke_game,
                    karaoke_level,
                    karaoke_notes,
                    karaoke_keep,
                    data_folder,
                );
                self.saved = file.clone();
                self.baseline = now;
                if let Some(path) = self.state.config_file.clone() {
                    if let Err(error) = file.save(&path) {
                        self.dispatch(Action::Failed(format!(
                            "Could not save {}: {error}",
                            path.display()
                        )));
                    }
                }
            }
            Effect::Quit => {
                if let Some(engine) = &self.engine {
                    let _ = engine.player.stop();
                }
            }
        }
    }

    /// Tells the preview's players, made for the first file of a preview.
    fn preview(&mut self, what: core::preview::Preview) {
        use core::preview::Preview;
        let Some(engine) = &mut self.engine else {
            return;
        };
        match (what, &mut engine.previews) {
            (Preview::Play { path }, previews) => previews
                .get_or_insert_with(|| {
                    crate::preview::Previews::new(
                        engine.lead.clone(),
                        engine.preview_strip.clone(),
                        self.state.engine_settings(),
                    )
                })
                .play(&engine.mixer, path),
            (Preview::Silent, Some(p)) => p.silent(),
            (Preview::End, Some(p)) => p.end(),
            (_, None) => {}
        }
    }

    /// What the preview's players said; once they are done, they go (on a
    /// thread of their own: a player stopping can take a while).
    fn pump_previews(&mut self) -> bool {
        let Some(engine) = &mut self.engine else {
            return false;
        };
        let Some(previews) = &mut engine.previews else {
            return false;
        };
        let said = previews.pump(&engine.mixer);
        if previews.done() {
            engine.previews = None;
        }
        let any = !said.is_empty();
        for text in said {
            self.dispatch(Action::Failed(text));
        }
        any
    }

    fn engine_command(&mut self, command: Command) {
        match &self.engine {
            Some(engine) => {
                if engine.player.send(command.clone()).is_err() {
                    // Its thread is gone: a new engine takes the command.
                    let generation = self.generation.load(Ordering::SeqCst);
                    self.dispatch(Action::EngineGone(generation));
                    if let Some(engine) = &self.engine {
                        if let Err(error) = engine.player.send(command) {
                            self.dispatch(Action::Failed(format!("The engine stopped: {error}")));
                        }
                    }
                }
            }
            None => self.dispatch(Action::Failed(
                "No output device; choose one in the settings (,)".into(),
            )),
        }
    }

    /// Moves playback to another device: the queue and the position
    /// carry over.
    fn switch_device(&mut self, id: Option<String>) {
        if matches!(self.output, Output::Null(_)) {
            return;
        }
        let queue = self.state.queue.paths.clone();
        let current = self.state.queue.current;
        let position = self.state.playing.position;
        let was = self.state.playing.state;
        self.output = Output::Device(id);
        let output = self.output.clone();
        // The old engine stops (and its late events are ignored) first.
        if let Some(engine) = self.engine.take() {
            let _ = engine.player.stop();
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.state.playing = core::Playing::default();
        self.start_engine(&output);
        if queue.is_empty() || self.engine.is_none() {
            return;
        }
        self.engine_command(Command::Load(queue));
        if let (Some(index), PlayState::Playing | PlayState::Paused) = (current, was) {
            self.engine_command(Command::Play(index));
            if position > 0.5 {
                self.engine_command(Command::Seek(position));
            }
            if was == PlayState::Paused {
                self.engine_command(Command::Pause);
            }
        }
    }

    /// The session as the state has it now: the queue, the track heard (or
    /// the queue's row selected), and the browser's folder and row.
    fn session_now(&self) -> Session {
        let s = &self.state;
        Session {
            queue: s.queue.paths.clone(),
            current: self.session_current(),
            browser: Some(s.browser.dir.clone()),
            selected: self.session_selected().cloned(),
        }
    }

    /// The session's track: the one heard, else the queue's row selected.
    fn session_current(&self) -> Option<usize> {
        let q = &self.state.queue;
        q.current.or((!q.paths.is_empty()).then_some(q.cursor))
    }

    /// The browser row selected, for the session.
    fn session_selected(&self) -> Option<&PathBuf> {
        let s = &self.state;
        core::browser_rows(s)
            .get(s.browser.cursor)
            .and_then(|&r| core::row_entry(s, r))
            .map(|e| &e.path)
    }

    /// Notes when the session went out of date (it is saved a little
    /// after); compared in place, nothing copied.
    fn note_session(&mut self) {
        if self.session.is_none() || self.session_stale.is_some() {
            return;
        }
        let s = &self.state;
        let same = self.session_saved.as_ref().is_some_and(|saved| {
            saved.queue == s.queue.paths
                && saved.browser.as_deref() == Some(s.browser.dir.as_path())
                && saved.current == self.session_current()
                && saved.selected.as_ref() == self.session_selected()
        });
        if !same {
            self.session_stale = Some(Instant::now());
        }
    }

    /// Saves the session where it is kept.
    fn save_session(&mut self) {
        self.session_stale = None;
        let Some(path) = self.session.clone() else {
            return;
        };
        let session = self.session_now();
        if self.session_saved.as_ref() == Some(&session) {
            return;
        }
        // Saved or not, it is not tried again until it changes.
        let _ = session.save(&path);
        self.session_saved = Some(session);
    }

    /// Saves the session, and stops playback and the workers.
    pub fn shutdown(&mut self) {
        self.save_session();
        if let Some(cancel) = self.search.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.engine = None;
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // The singing of a song cut short by quitting, kept all the same.
        self.save_kept(true);
        self.shutdown();
    }
}

/// The separator stem playback uses: the GPU's, set up on first use.
fn gpu_separator() -> Option<&'static dyn jodel_playback::stems::Separator> {
    jodel_gpu::GpuSeparator::shared().map(|g| g as &dyn jodel_playback::stems::Separator)
}
