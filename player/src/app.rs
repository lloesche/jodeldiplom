//! The running player behind the window: the [`State`], the
//! playback engine, the background workers, the spectrum and the cover.
//!
//! The engine plays on the Lead input of a mixer that owns the device
//! ([`crate::desk_wiring`]): its main alone at gain 1 reaches the device
//! bit for bit, its stems on strips of their own. The preview's players
//! ([`crate::preview`]) are guests on it, and the spectrum is the mixer's:
//! what the device plays.
//!
//! The window creates an [`App`], feeds it [`Action`]s from its input
//! ([`App::dispatch`]), calls [`App::pump`] to take in what the engine and
//! the workers sent, and draws [`App::state`] with [`crate::view`].
//!
//! The app is a composition of subsystems, each owning its own fields:
//! the microphone and the singing ([`mic::MicSession`]), the cover heard
//! ([`crate::work::art::ArtWorker`]), the spectrum ([`SpectrumGate`]), the
//! session kept ([`SessionKeeper`]) and a data folder's move
//! ([`MoveView`]). The workers send their results through an [`Outbox`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use image::RgbImage;
use jodel_output::mixer::DeskView;
use jodel_playback::{Command, Event, PlayState, PlayerOptions, SeparatorSource, SpectrumTap};

use crate::browser::DetailsWorker;
pub use crate::clock::AppClock;
use crate::config::Config;
use crate::core::{self, Action, DeviceChoice, Effect, State};
use crate::desk_wiring::Engine;
pub use crate::desk_wiring::{MemoryOutput, Output};
use crate::spectrum::SpectrumGate;
use crate::store::mover::MoveView;
use crate::store::session::{Session, SessionKeeper};
use crate::work::art::{Art, ArtWorker};
use crate::work::outbox::{Notify, Outbox};
use crate::work::rhythm::RhythmWorker;

mod effects;
mod frame;
pub mod mic;
#[cfg(test)]
mod tests;

use mic::{MicLister, MicOpener, MicSession};

/// Seconds of the playing track the engine keeps behind what is heard and
/// decodes ahead of it, so that a bar jump within is heard at once.
const JUMP_WINDOW: f64 = 24.0;

/// Lists the output devices with a summary of what each takes.
pub type DeviceLister = Arc<dyn Fn() -> Result<Vec<DeviceChoice>, String> + Send + Sync>;

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
    /// Called after a background result arrives (wakes a waiting loop).
    pub notify: Notify,
    /// Lists the devices for the settings.
    pub devices: DeviceLister,
    /// Messages to show at the start (config warnings).
    pub notes: Vec<String>,
    /// Where beat analyses are kept ([`crate::store::dirs::Data::analyses`]);
    /// `None`: not kept.
    pub rhythm_cache: Option<PathBuf>,
    /// Where beat grids fixed by hand are kept
    /// ([`crate::store::dirs::Folders::grids`]); `None`: not kept.
    pub grids: Option<PathBuf>,
    /// Where the session is kept ([`crate::store::dirs::Folders::session`]):
    /// the queue and the browser's place come back from it and are saved to it;
    /// `None`: not kept.
    pub session: Option<PathBuf>,
    /// The browser starts where the session left it (no folder was given).
    pub restore_browser: bool,
    /// Where queues are saved as playlists
    /// ([`crate::store::dirs::Folders::queues`]).
    pub queues: Option<PathBuf>,
    /// How lyrics are looked up ([`crate::work::lyrics::Options`]); `None`: in
    /// the files' tags and `.lrc` files only, nothing cached or fetched.
    pub lyrics: Option<crate::work::lyrics::Options>,
    /// The speech model for the word timing, where it is or is fetched to
    /// ([`crate::work::lyrics::models::Models`]); `None`: nowhere, the words
    /// shown untimed.
    pub model: Option<PathBuf>,
    /// How the microphone is opened (the real ones: [`mic::open_mic`]).
    pub mic: MicOpener,
    /// How the microphones are listed (the real ones: [`mic::list_mics`]).
    pub mics: MicLister,
    /// The data folder ([`crate::store::dirs::Data`]); `None`: none (tests).
    pub data: Option<crate::store::dirs::Data>,
    /// A move of the data folder under way ([`crate::store::mover`]), shown on
    /// the settings page.
    pub mover: Option<Arc<crate::store::mover::Mover>>,
    /// Where the engine separates stems (the real one: the GPU's).
    pub separator: SeparatorSource,
    /// The covers' models, as setup found them ([`crate::setup::Found`]);
    /// `None`: covers are not taken apart.
    pub covers: Option<crate::work::cover::CoverModels>,
    /// Karaoke story mode's models, as setup found them
    /// ([`crate::setup::Found`]); `None`: no stories are made (the covers
    /// show, as without).
    pub story: Option<crate::work::story_models::StoryModels>,
    /// What the app times by ([`AppClock`]): the wall clock but in tests
    /// that step it.
    pub clock: AppClock,
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
            notify: Arc::new(|| {}),
            devices: Arc::new(crate::desk_wiring::list_devices),
            notes: Vec::new(),
            rhythm_cache: None,
            grids: None,
            session: None,
            restore_browser: false,
            queues: None,
            lyrics: None,
            model: None,
            mic: Arc::new(mic::open_mic),
            mics: Arc::new(mic::list_mics),
            data: None,
            mover: None,
            separator: gpu_separator,
            covers: None,
            story: None,
            clock: AppClock::default(),
        }
    }
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
    /// Where the workers send to, and what they sent.
    outbox: Outbox,
    rx: Receiver<Action>,
    devices: DeviceLister,
    saved: Config,
    baseline: Config,
    details: DetailsWorker,
    rhythm: RhythmWorker,
    lyrics: crate::work::lyrics::LyricsWorker,
    /// The microphone, open while Karaoke shows, and the singing heard
    /// through it.
    mic: MicSession,
    /// Keeps the beat grids fixed by hand, where they are kept.
    grids: Option<crate::store::grids::GridStore>,
    /// Keeps the Synth tab's patches, in the data folder.
    patches: Option<crate::store::patches::PatchStore>,
    /// Keeps the Shader tab's looks, in the data folder.
    looks: Option<crate::store::looks::LookStore>,
    /// When the last position event arrived, for [`App::heard`].
    position_at: Instant,
    /// Where the engine separates stems.
    separator: SeparatorSource,
    /// The track and position the last frame drew, and when
    /// ([`App::heard_for_frame`]).
    last_drawn: Option<(Option<usize>, f64, Instant)>,
    /// The sync's clicks playing ([`crate::sync`]).
    clicks: Option<crate::sync::Clicks>,
    /// Where the tracks' stems are kept, for the engine to play them from.
    stem_store: Option<Arc<dyn jodel_playback::StemStore>>,
    search: Option<Arc<AtomicBool>>,
    spectrum: SpectrumGate,
    /// The cover of the track heard.
    art: ArtWorker,
    /// The covers' step of the preparation ([`crate::work::cover`]), holding
    /// the song playing's layers: none without a data folder or the
    /// covers' models.
    covers: Option<crate::work::cover::CoverWorker>,
    /// The stories of the songs Karaoke mode plays, planned and drawn
    /// ([`crate::work::story::worker`]): none without the story's models,
    /// or without the covers' worker that shows them.
    stories: Option<crate::work::story::worker::StoryWorker>,
    /// When the app started, on its clock ([`AppClock`]): every time it
    /// keeps but the trace's is on that clock.
    started: Instant,
    clock: AppClock,
    session: SessionKeeper,
    /// A move of the data folder, for the settings' row.
    moving: MoveView,
    /// A song being sung, as the GPU's stages are told
    /// ([`crate::work::stages::sing`]).
    singing: Option<crate::work::stages::Singing<'static>>,
}

/// The steps of the player's start ([`App::start`]), in order, as the
/// setup screen tells them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Starting {
    /// The workers started, the session and what is kept read.
    Session,
    /// The output device opened, the queue loaded on it.
    Device,
}

impl Starting {
    /// Every step, in order.
    pub const ALL: [Starting; 2] = [Starting::Session, Starting::Device];

    /// What it does, as the setup screen says it.
    pub fn title(self) -> &'static str {
        match self {
            Starting::Session => "Loading the session",
            Starting::Device => "Opening the sound device",
        }
    }

    /// Why, as the setup screen says it.
    pub fn why(self) -> &'static str {
        match self {
            Starting::Session => "The queue and the browser come back as you left them.",
            Starting::Device => {
                "It is opened once, shared with the system's other sounds, and stays open \
                 until the player quits."
            }
        }
    }
}

impl App {
    /// Starts the engine and the workers, and lists the start folder.
    pub fn new(options: AppOptions) -> App {
        App::start(options, &mut |_| {})
    }

    /// [`App::new`], telling `begins` as each step of it after the first
    /// ([`Starting`]) begins.
    pub fn start(options: AppOptions, begins: &mut dyn FnMut(Starting)) -> App {
        let began = Instant::now();
        let (tx, rx) = mpsc::channel();
        let outbox = Outbox::new(tx, options.notify.clone());
        let details = DetailsWorker::new(16, outbox.clone());
        let rhythm = RhythmWorker::new(
            outbox.clone(),
            options.rhythm_cache.clone(),
            options.grids.clone(),
        );
        let lyrics = {
            let mut lyrics_options = options
                .lyrics
                .clone()
                .unwrap_or_else(|| crate::work::lyrics::Options::new(false));
            lyrics_options.online &= options.config.karaoke_online;
            let working = outbox.clone();
            lyrics_options.working = Some(Arc::new(move |path: &Path, on: bool| {
                working.send(Action::Working {
                    path: path.to_owned(),
                    job: core::Job::Lyrics,
                    on,
                });
            }));
            let done = outbox.clone();
            lyrics_options.done = Some(Arc::new(move |path: &Path, facts| {
                done.send(Action::Ready {
                    path: path.to_owned(),
                    job: core::Job::Lyrics,
                    facts: Some(core::Facts::Lyrics(facts)),
                });
            }));
            let told = outbox.clone();
            lyrics_options.told = Some(Arc::new(move |path: &Path, words| {
                told.send(Action::Heard {
                    path: path.to_owned(),
                    part: core::Heard::Lyrics(words),
                });
            }));
            let answers = outbox.clone();
            crate::work::lyrics::LyricsWorker::new(lyrics_options, move |path, answer| {
                answers.send(Action::Lyrics(path, answer));
            })
        };
        let grids = (options.grids.clone())
            .map(|dir| crate::store::grids::GridStore::new(dir, outbox.clone()));
        let patches = (options.data.as_ref())
            .map(|d| crate::store::patches::PatchStore::new(d.patches(), outbox.clone()));
        let looks = (options.data.as_ref())
            .map(|d| crate::store::looks::LookStore::new(d.looks(), outbox.clone()));
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
        let mut state = State::new(dir.clone(), options.config.clone());
        state.browser.fallback = remembered.map(|_| options.dir.clone());
        if let Some(selected) = restored.as_ref().and_then(|s| s.selected.clone()) {
            state.browser.last_child.insert(dir.clone(), selected);
        }
        state.queues_dir = options.queues.clone();
        // There is always a lyrics worker (one without the web or a cache
        // where the options give none): the queue waits for it too.
        state.preparing.push(core::Job::Lyrics);
        // Covers are taken apart where there is a data folder to keep them,
        // and the songs' story scenes are there.
        let scenes = (options.data.as_ref())
            .map(|d| crate::work::scenes::SceneStore::new(d.scenes(), d.scene_parts()));
        let covers = (options.data.as_ref().zip(options.covers.clone()))
            .zip(scenes.clone())
            .map(|((data, models), scenes)| {
                crate::work::cover::CoverWorker::start(
                    data.covers(),
                    models,
                    scenes,
                    outbox.clone(),
                )
            });
        if covers.is_some() {
            state.preparing.push(core::Job::Cover);
        }
        // Stories are made where their scenes are shown: by the covers'
        // worker, told of each one drawn.
        let stories = (options.data.as_ref().zip(options.story.clone()))
            .zip(scenes.zip(covers.as_ref()))
            .map(|((data, files), (scenes, covers))| {
                use crate::work::story::worker::{GpuModels, Places, StoryWorker};
                let places = Places {
                    stories: data.stories(),
                    scenes,
                };
                let models = Arc::new(GpuModels::new(&files));
                StoryWorker::start(models, places, outbox.clone(), Arc::new(covers.on_drawn()))
            });
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
            mic: MicSession::new(
                options.mic.clone(),
                options.mics.clone(),
                options.rhythm_cache.as_deref(),
                options.notify.clone(),
            ),
            art: ArtWorker::new(options.notify.clone()),
            outbox,
            rx,
            devices: options.devices.clone(),
            saved: options.saved.clone(),
            baseline: options.config.clone(),
            details,
            rhythm,
            lyrics,
            grids,
            patches,
            looks,
            position_at: options.clock.now(),
            separator: options.separator,
            last_drawn: None,
            clicks: None,
            stem_store: options
                .rhythm_cache
                .as_deref()
                .map(|dir| crate::store::stems::Store::shared(crate::store::stems::beside(dir))),
            search: None,
            spectrum: SpectrumGate::new(),
            covers,
            stories,
            started: options.clock.now(),
            clock: options.clock.clone(),
            session: SessionKeeper::new(options.session.clone(), restored.clone()),
            moving: MoveView::new(
                options.mover.clone(),
                options.data.as_ref().map(|d| d.0.clone()),
            ),
            singing: None,
        };
        app.state.data_dir = options.data.as_ref().map(|d| d.0.clone());
        // The delays kept, from beside the settings.
        app.state.karaoke.lag.kept =
            crate::store::delays::load_beside(options.config_file.as_deref());
        // The best scores sung, from the data folder (while it moves, from
        // where the file is: [`crate::store::scores::load`]).
        if let Some(data) = &options.data {
            app.state.karaoke.bests = crate::store::scores::load(&data.scores());
            // The Synth tab's patches kept, over the built-in ones.
            app.state.patches = crate::store::patches::load_all(&data.patches());
            // The Shader tab's looks kept.
            app.state.shader.kept = crate::store::looks::load_all(&data.looks());
        }
        crate::trace::took("app: the session read", began, true);
        begins(Starting::Device);
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

    fn start_engine(&mut self, output: &Output) {
        // A sync's clicks played on the mixer going: it ends, said so.
        if self.clicks.take().is_some() {
            let why = "the output changed while the clicks played".to_owned();
            self.dispatch(Action::Sync(core::sync::SyncEvent::Failed(why)));
        }
        self.engine = None;
        self.playing_on = None;
        let mut target = match output.target() {
            Ok((target, device)) => {
                self.playing_on = device;
                target
            }
            Err(error) => {
                self.dispatch(Action::Failed(format!(
                    "No output device ({error}); choose one in the settings (,)"
                )));
                return;
            }
        };
        // What the device takes (its mix format), asked here once.
        let capabilities = target.capabilities().ok();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let options = PlayerOptions {
            settings: self.state.config.settings(),
            position_interval: Duration::from_millis(100),
            separator: Some(self.separator),
            stem_store: self.stem_store.clone(),
            ..PlayerOptions::default()
        };
        let (engine, events) = Engine::start(target, capabilities, options);
        // The bars a jump can reach are kept ([`Op::JumpBar`]): four either
        // side of the one heard down to 48 BPM, and that one.
        let _ = engine.player.jump_window(JUMP_WINDOW, JUMP_WINDOW);
        // Whatever the old engine had armed or mixing is gone with it.
        core::transition::draft::engine_restarted(&mut self.state);
        let (outbox, current) = (self.outbox.clone(), self.generation.clone());
        thread::spawn(move || {
            for event in events {
                if current.load(Ordering::SeqCst) != generation
                    || !outbox.send(Action::Engine(event))
                {
                    return;
                }
            }
            // The engine's thread ended while it was still the one in use:
            // it failed.
            if current.load(Ordering::SeqCst) == generation {
                outbox.send(Action::EngineGone(generation));
            }
        });
        // A new mixer's strips start at 0 dB: the Mixer tab's settings are
        // made on them again, and the engine told of the stems.
        engine.set_up(&self.state);
        self.engine = Some(engine);
        let mut fx = Vec::new();
        core::stems::engine_started(&mut self.state, &mut fx);
        for effect in fx {
            self.run(effect);
        }
        // A new mixer takes the microphone over.
        self.mic.give(self.engine.as_ref().map(|e| &e.mixer));
    }

    /// Applies an action and carries out its effects: what the app takes
    /// from it itself first ([`App::take_in`]), then the core's update and
    /// the effects it asks for, then what follows from the state it left
    /// ([`App::follow`]).
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
        self.take_in(&action);
        let started = Instant::now();
        let what =
            crate::trace::on().then(|| format!("{action:?}").chars().take(120).collect::<String>());
        for effect in core::update(&mut self.state, action) {
            self.run(effect);
        }
        if let Some(what) = what {
            crate::trace::took(format_args!("update {what}"), started, false);
        }
        self.follow();
        if gone {
            self.restart_engine();
        }
    }

    /// What the app takes from an action before the core does: a track
    /// started brings its cover, the engine's news of the position when it
    /// was heard of ([`App::heard`]); and the pair of devices in use.
    fn take_in(&mut self, action: &Action) {
        if let Action::Engine(Event::TrackStarted(start)) = action {
            self.art.want(start.info.cover.as_ref());
            if let Some(covers) = &self.covers {
                covers.want(&start.info);
            }
        }
        if let Action::Engine(
            Event::Position(_)
            | Event::TrackStarted(_)
            | Event::Seeked { .. }
            | Event::Jumped { .. }
            | Event::State(_),
        ) = action
        {
            self.position_at = self.clock.now();
        }
        self.note_pair();
    }

    /// What follows from the state an update left: the session noted as
    /// out of date, the microphone opened or closed, the delays and the
    /// best scores kept where the core changed them.
    fn follow(&mut self) {
        self.session.note(&self.state, self.clock.now());
        self.mic_turn();
        // Singing takes the GPU first.
        let singing = core::karaoke::singing(&self.state);
        if singing != self.singing.is_some() {
            self.singing = singing.then(crate::work::stages::sing);
        }
        // The delays changed: kept beside the settings at once.
        if std::mem::take(&mut self.state.karaoke.lag.changed) {
            let lag = &self.state.karaoke.lag;
            if let Some(why) =
                crate::store::delays::save_beside(self.state.config_file.as_deref(), &lag.kept)
            {
                self.dispatch(Action::Failed(why));
            }
        }
        // A best score beaten: kept in the data folder at once.
        if std::mem::take(&mut self.state.karaoke.bests_changed) {
            if let Some(dir) = &self.state.data_dir {
                let file = crate::store::dirs::Data(dir.clone()).scores();
                if let Err(e) = crate::store::scores::save(&file, &self.state.karaoke.bests) {
                    self.dispatch(Action::Failed(format!(
                        "The best scores were not saved: {e}"
                    )));
                }
            }
        }
    }

    /// Opens the microphone while Karaoke shows and closes it after
    /// ([`MicSession::turn`]).
    fn mic_turn(&mut self) {
        let mixer = self.engine.as_ref().map(|e| &e.mixer);
        self.mic.turn(&mut self.state, mixer);
    }

    /// The mixer's desk as it stands (none without an engine), for tests
    /// to see what the strips were set to. Looking changes nothing; the
    /// player itself looks once a frame ([`App::heard_for_frame`]).
    pub fn desk(&self) -> Option<DeskView> {
        self.engine.as_ref().map(|e| e.mixer.desk())
    }

    /// What the desk's effects input took and dropped (none without an
    /// engine), for tests to see the game's sounds and the Synth tab's
    /// reach it.
    pub fn effects(&self) -> Option<jodel_output::mixer::EffectsView> {
        self.engine.as_ref().map(|e| e.effects())
    }

    /// The output device and the microphone in use, for the delays kept
    /// for them ([`crate::core::lag`]).
    fn note_pair(&mut self) {
        let output = match &self.output {
            Output::Null(_) => "null".to_owned(),
            Output::Memory(_) => "memory".to_owned(),
            Output::Device(query) => self
                .playing_on
                .clone()
                .or_else(|| query.clone())
                .unwrap_or_else(|| "default".to_owned()),
        };
        let mic = self.mic.input().map_or("", |m| m.info().id.as_str());
        core::lag::set_pair(&mut self.state, &output, mic);
    }

    /// The sync's clicks' clock, while they play (tests: what the output is
    /// at).
    pub fn sync_clock(&self) -> Option<jodel_output::Clock> {
        self.clicks.as_ref().and_then(crate::sync::Clicks::clock)
    }

    /// Starts the engine again on the same output after it failed, with
    /// the queue loaded and nothing playing.
    fn restart_engine(&mut self) {
        let queue = self.state.queue.paths.clone();
        self.engine = None;
        let first = core::key::for_new_engine(&mut self.state, queue);
        let output = self.output.clone();
        self.start_engine(&output);
        if let Some(engine) = &self.engine {
            first.into_iter().for_each(|c| drop(engine.player.send(c)));
        }
    }

    /// Takes in everything the engine and the workers sent. Returns whether
    /// anything arrived.
    pub fn pump(&mut self) -> bool {
        self.session.save_when_due(&self.state, &self.clock);
        let mut any = false;
        while let Ok(action) = self.rx.try_recv() {
            any = true;
            self.dispatch(action);
        }
        let mixer = self.engine.as_ref().map(|e| &e.mixer);
        any |= self.mic.pump(&mut self.state, mixer);
        any |= self.pump_previews();
        // A device that refused the microphone's tap is told, each time.
        if let Some(why) = self.engine.as_mut().and_then(Engine::mic_refused) {
            any = true;
            self.dispatch(Action::Info(format!(
                "The microphone cannot hear this output (its echo stays in): {why}"
            )));
        }
        any |= self.moving.pump(&mut self.state.data_move);
        any |= self.art.pump();
        any |= self.follow_story();
        let now = self.now_ms();
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

    /// The milliseconds since the start (the core's clock).
    pub fn now_ms(&self) -> u64 {
        self.clock.since(self.started).as_millis() as u64
    }

    /// The cover of the track heard, once decoded.
    pub fn art(&self) -> Option<&Art> {
        self.art.art()
    }

    /// The cover of the track heard taken apart (subject, plate, depth,
    /// rim, palette), once ready; `None` where it has no cover or this
    /// machine cannot take covers apart.
    pub fn cover_layers(&self) -> Option<crate::work::cover::Layers> {
        self.covers
            .as_ref()
            .and_then(crate::work::cover::CoverWorker::current)
    }

    /// The track heard's story scenes, once ready
    /// ([`crate::work::cover::CoverWorker::story`]); `None` where it has
    /// none or this machine cannot take covers apart.
    pub fn story(&self) -> Option<Arc<crate::work::scenes::Story>> {
        self.covers
            .as_ref()
            .and_then(crate::work::cover::CoverWorker::story)
    }

    /// The song's own look follows its story: the look of the story held
    /// for the track heard, or none (the theme's). Whether it changed.
    fn follow_story(&mut self) -> bool {
        let look = self.story().map(|s| Arc::clone(&s.look));
        let held = &mut self.state.karaoke.look;
        let changed = look.as_ref().map(Arc::as_ptr) != held.as_ref().map(Arc::as_ptr);
        if changed {
            *held = look;
        }
        changed
    }

    /// Sets the cover directly, or none (tests and screenshots).
    pub fn set_art(&mut self, image: Option<RgbImage>) {
        self.art.set(image);
    }

    /// The engine's spectrum tap, if an engine runs.
    pub fn tap(&self) -> Option<SpectrumTap> {
        self.engine.as_ref().map(|e| e.tap.clone())
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

    /// Plays on `output` from now on (another device, or the same at
    /// another rate: its mixer opens it once, at the rate set), carrying
    /// the queue and the position over.
    fn restart_on(&mut self, output: Output) {
        let queue = self.state.queue.paths.clone();
        let current = self.state.queue.current;
        let position = self.state.playing.position;
        let was = self.state.playing.state;
        self.output = output.clone();
        // The old engine stops (and its late events are ignored) first.
        if let Some(engine) = self.engine.take() {
            let _ = engine.player.stop();
        }
        self.generation.fetch_add(1, Ordering::SeqCst);
        let first = core::key::for_new_engine(&mut self.state, queue);
        self.start_engine(&output);
        if first.is_empty() || self.engine.is_none() {
            return;
        }
        first.into_iter().for_each(|c| self.engine_command(c));
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

    /// Saves the session, and stops playback and the workers.
    pub fn shutdown(&mut self) {
        self.session.save(&self.state);
        self.cancel_search();
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.engine = None;
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // The singing of a song cut short by quitting, kept all the same.
        if let Some(said) = self.mic.save_kept(&self.state, true) {
            self.dispatch(said);
        }
        self.shutdown();
    }
}

/// The separator stem playback uses: the GPU's, set up on first use.
fn gpu_separator() -> Option<&'static dyn jodel_playback::stems::Separator> {
    jodel_gpu::GpuSeparator::shared().map(|g| g as &dyn jodel_playback::stems::Separator)
}
