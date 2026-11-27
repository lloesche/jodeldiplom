//! The command line: `jodeldiplom [options] [path]`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use jodel_output::SimClock;
use jodel_playback::Headroom;

use crate::app::{App, AppOptions, Output, Starting};
use crate::config::{self, Config};
use crate::setup::{Need, Places, Setup, Then, Work};

/// The usage text (`{bin}` is replaced by the binary's name).
pub const USAGE: &str = "\
Usage: {bin} [options] [path]

A music player for PCM files (FLAC, WAV, AIFF, CAF, ALAC) that plays
through the system's mixer, beside every other application: it opens the
output device once, at the rate the system runs it at (its sound settings
choose it), plays files at that rate as they are and resamples the
others. AAC-LC (in M4A/MP4), MP3 (and MP2, MP1) and Ogg Vorbis play too,
gapless, decoded to 24 bits.

  path                 a folder to start the file browser in, or a file to
                       play at once (the browser opens on its folder);
                       default: the current folder
  --device ID|NAME     the output device: an id or a unique part of a name
                       from --list-devices; default: the system default
  --headroom DB        routes that change the samples (resampled, floating-point
                       files) play this many dB below full scale
                       (0, 0.5, 1, 2, 3; default 0), so inter-sample overs
                       do not clip; files played as they are are never touched
  --config FILE        the settings file; default: {config}
  --list-devices       list the output devices and what each takes (its
                       rate and format, as the system mixes it), then exit
  --diag               print what the player finds on this system (build,
                       settings, devices and their volume, the GPU and
                       whether stems separate on it) for a bug report, then
                       exit; plays nothing
  --clean-cache        forget the beat analyses, the stems kept and the
                       lyrics (each made again when wanted), then exit;
                       the models stay
  --clean-diag         forget the record of the last analyses --diag shows,
                       then exit
  --help               this text

Options override the settings file for this run. Settings changed in the
player are saved to the settings file. Press ? in the player for the keys.";

/// The parsed command line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args {
    /// The folder or file.
    pub path: Option<PathBuf>,
    /// `--device`.
    pub device: Option<String>,
    /// `--headroom`.
    pub headroom: Option<Headroom>,
    /// `--config`.
    pub config: Option<PathBuf>,
    /// `--list-devices`.
    pub list_devices: bool,
    /// `--diag`.
    pub diag: bool,
    /// `--clean-cache`.
    pub clean_cache: bool,
    /// `--clean-diag`.
    pub clean_diag: bool,
    /// `--help`.
    pub help: bool,
    /// `--null` (not in the usage): play into a null output, no device.
    pub null: bool,
    /// `--exit-after SECONDS` (not in the usage): quit after a while.
    pub exit_after: Option<f64>,
    /// `--save-frame FILE` (not in the usage): the window writes its last
    /// frame to this PNG when it closes.
    pub save_frame: Option<PathBuf>,
}

/// Parses the arguments after the program name.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut out = Args::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => out.help = true,
            "--device" => out.device = Some(value("--device")?),
            "--headroom" => {
                let text = value("--headroom")?;
                out.headroom = Some(Headroom::parse(&text).ok_or_else(|| {
                    format!("--headroom takes dB (0, 0.5, 1, 2, 3), not {text:?}")
                })?);
            }
            "--config" => out.config = Some(value("--config")?.into()),
            "--list-devices" => out.list_devices = true,
            "--diag" => out.diag = true,
            "--clean-cache" => out.clean_cache = true,
            "--clean-diag" => out.clean_diag = true,
            "--save-frame" => out.save_frame = Some(value("--save-frame")?.into()),
            "--null" => out.null = true,
            "--exit-after" => {
                let text = value("--exit-after")?;
                out.exit_after = Some(
                    text.parse()
                        .map_err(|_| format!("--exit-after takes seconds, not {text:?}"))?,
                );
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path => {
                if out.path.is_some() {
                    return Err("give at most one folder or file".into());
                }
                out.path = Some(path.into());
            }
        }
    }
    Ok(out)
}

/// The usage text for a binary.
pub fn usage(bin: &str) -> String {
    let config = config::default_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "none".into());
    USAGE.replace("{bin}", bin).replace("{config}", &config)
}

/// The settings file as read, and the settings for this run.
pub struct Settings {
    /// The file, if any.
    pub file: Option<PathBuf>,
    /// The settings in the file.
    pub saved: Config,
    /// The settings with the command line's overrides.
    pub effective: Config,
    /// Warnings about the file.
    pub warnings: Vec<String>,
}

/// Reads the settings file and applies the command line's overrides. A
/// file with settings the player does not know (any longer) says so once:
/// it is written again without them.
pub fn settings(args: &Args) -> Result<Settings, String> {
    let file = args.config.clone().or_else(config::default_path);
    let (saved, mut warnings) = match &file {
        Some(path) => Config::load(path)?,
        None => (Config::default(), Vec::new()),
    };
    if let (Some(path), false) = (&file, warnings.is_empty()) {
        match saved.save(path) {
            Ok(()) => warnings.push(format!("{}: written again without them", path.display())),
            Err(e) => warnings.push(format!("{}: could not write it again: {e}", path.display())),
        }
    }
    let mut effective = saved.clone();
    if let Some(device) = &args.device {
        effective.device = (device != "default").then(|| device.clone());
    }
    if let Some(headroom) = args.headroom {
        effective.headroom = headroom;
    }
    Ok(Settings {
        file,
        saved,
        effective,
        warnings,
    })
}

/// Where the browser starts and what plays at once, from the `path`
/// argument.
pub fn start(path: Option<&Path>) -> Result<(PathBuf, Option<PathBuf>), String> {
    let path = match path {
        Some(p) => std::path::absolute(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => {
            let cwd = std::env::current_dir().map_err(|e| format!("the current folder: {e}"))?;
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(PathBuf::from);
            default_folder(
                cwd,
                std::env::current_exe().ok().as_deref(),
                home.as_deref(),
            )
        }
    };
    if path.is_dir() {
        Ok((path, None))
    } else if path.is_file() {
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("{}: no folder", path.display()))?;
        Ok((dir, Some(path)))
    } else {
        Err(format!("{}: no such file or folder", path.display()))
    }
}

/// Where the player keeps things by its settings ([`settings`]): the
/// settings file's folder, and the data folder those settings choose.
pub fn folders(settings: &Settings) -> Result<crate::store::dirs::Folders, String> {
    settings
        .file
        .as_deref()
        .and_then(|file| {
            crate::store::dirs::Folders::new(file, settings.effective.data_folder.as_deref())
        })
        .ok_or_else(|| "no folder for the player's settings or data (no home folder?)".into())
}

/// [`folders`] of the usual settings, for a program started without the
/// player's command line (the examples).
pub fn usual_folders() -> Result<crate::store::dirs::Folders, String> {
    folders(&settings(&Args::default())?)
}

/// The app options for a command line (the window fills in the wake-up).
pub fn app_options(args: &Args) -> Result<AppOptions, String> {
    let settings = settings(args)?;
    let (dir, autoplay) = start(args.path.as_deref())?;
    let mut options = AppOptions::new(dir);
    // Where this player keeps things, decided here once: one with a
    // settings file of its own (--config) has its own of everything.
    let folders = folders(&settings)?;
    // A move of the data folder written down and not finished goes on in
    // the background ([`start_mover`]): until it has, what is not moved yet
    // is read where it is.
    crate::store::mover::resume(&folders.settings);
    // What a crash left half written (never under a file's name, so never
    // read) goes, in the background, once a day old.
    let mut swept = folders.data.swept();
    swept.push(folders.settings.clone());
    std::thread::spawn(move || {
        for dir in swept {
            durable::sweep(&dir, durable::STALE);
        }
    });
    let data = folders.data.clone();
    options.rhythm_cache = Some(data.analyses());
    options.grids = Some(folders.grids());
    options.session = Some(folders.session());
    options.restore_browser = args.path.is_none();
    options.queues = Some(folders.queues());
    // Lyrics: cached in the data folder, edited by hand beside the cache,
    // LRCLIB asked unless turned off. The speech models come once setup
    // has found them ([`set_models`]).
    let mut lyrics = crate::work::lyrics::Options::new(true);
    lyrics.cache = Some(data.lyrics());
    lyrics.edits = Some(data.lyric_edits());
    options.lyrics = Some(lyrics);
    options.data = Some(data);
    options.autoplay = autoplay;
    options.output = if args.null {
        Output::Null(SimClock::Realtime)
    } else {
        Output::Device(settings.effective.device.clone())
    };
    options.config = settings.effective;
    options.saved = settings.saved;
    options.config_file = settings.file;
    options.notes = settings.warnings;
    options.devices = Arc::new(crate::desk_wiring::list_devices);
    Ok(options)
}

/// Starts the moves the journal holds (a data folder chosen anew), telling
/// `options.notify` as they go: before setup, which reads what has not
/// moved yet where it is.
pub fn start_mover(options: &mut AppOptions) {
    // The journal beside the settings file in use.
    if let Some(dir) = options.config_file.as_deref().and_then(Path::parent) {
        options.mover =
            crate::store::mover::Mover::start(dir.to_path_buf(), options.notify.clone())
                .map(Arc::new);
    }
}

/// Starts setup for the player with `options`, on a thread of its own (it
/// returns at once): what `manifest` names (the player's is
/// [`crate::setup::manifest`], which chooses the graphics card) found, or
/// got through `work`, then the player itself started ([`App::start`]),
/// each step told on the setup screen; the player, once started, put in
/// `player`.
pub fn start_setup(
    mut options: AppOptions,
    manifest: impl FnOnce() -> Vec<Need> + Send + 'static,
    work: Arc<dyn Work>,
    player: Arc<Mutex<Option<App>>>,
) -> Setup {
    let data = options
        .data
        .clone()
        .unwrap_or_else(|| crate::store::dirs::Data::in_parent(Path::new(".")));
    let places = Places::player(&data);
    let notify = options.notify.clone();
    let then = Then {
        steps: Starting::ALL
            .iter()
            .map(|s| (s.title().to_owned(), s.why().to_owned()))
            .collect(),
        run: Box::new(move |paths, next| {
            set_models(&mut options, &paths);
            let app = App::start(options, &mut |_| next());
            *player.lock().unwrap_or_else(PoisonError::into_inner) = Some(app);
        }),
    };
    Setup::start(manifest, places, work, notify, Some(then))
}

/// Gives the app the models setup found (each known by its path in the
/// store: [`crate::setup::Found`]): the speech models to the lyrics (the voice
/// from the stems kept in the data folder), the covers' models to the
/// covers' step, the story's to the stories' thread.
pub fn set_models(options: &mut AppOptions, paths: &[(String, PathBuf)]) {
    let Some(found) = crate::setup::Found::of(paths) else {
        return;
    };
    options.covers = found.covers.clone();
    options.story = found.story.clone();
    let (speech, asr) = (found.speech, found.asr);
    let models = crate::work::lyrics::models::Models::new(speech.clone(), asr);
    options.model = Some(speech);
    let stems = options.data.as_ref().map(crate::store::dirs::Data::stems);
    if let Some(lyrics) = &mut options.lyrics {
        lyrics.timer = Some(
            Arc::new(crate::work::lyrics::timer::SpeechTimer::new(models, stems))
                as Arc<dyn crate::work::lyrics::WordTimer>,
        );
    }
}

/// The `--list-devices` text.
pub fn device_list() -> Result<String, String> {
    let devices = crate::desk_wiring::list_devices()?;
    if devices.is_empty() {
        return Ok(format!("No output devices ({}).\n", jodel_output::BACKEND));
    }
    let mut out = format!("Output devices ({}):\n", jodel_output::BACKEND);
    for device in devices {
        out.push_str(&format!(
            "  {}\n    id: {}\n    {}\n",
            device.name,
            device.id.unwrap_or_default(),
            device.summary
        ));
    }
    Ok(out)
}

/// The folder to start in without a path: the current one, unless the
/// player runs from a macOS app bundle (`exe` in `….app/Contents/MacOS`),
/// which Finder starts in `/`: then `~/Music`, else the home folder.
pub fn default_folder(cwd: PathBuf, exe: Option<&Path>, home: Option<&Path>) -> PathBuf {
    let in_bundle = exe
        .and_then(Path::parent)
        .is_some_and(|dir| dir.ends_with("Contents/MacOS"));
    match home {
        Some(home) if in_bundle => {
            let music = home.join("Music");
            if music.is_dir() {
                music
            } else {
                home.to_path_buf()
            }
        }
        _ => cwd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        parse(list.iter().map(|s| s.to_string()))
    }

    /// A player with a settings file of its own (`--config`) keeps all of
    /// its own: the session, the queues and the grids beside that file, the
    /// caches in the data folder it names; none of the usual player's.
    #[test]
    fn a_config_player_keeps_its_own_folders() {
        let dir = std::env::temp_dir().join(format!("jodel-own-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (settings, music) = (dir.join("settings"), dir.join("music"));
        durable::create_dir_all(&settings).unwrap();
        durable::create_dir_all(&music).unwrap();
        let file = settings.join("mine.toml");
        Config {
            data_folder: Some(dir.join("parent")),
            ..Config::default()
        }
        .save(&file)
        .unwrap();
        let path = |p: &Path| p.to_string_lossy().into_owned();
        let a = args(&["--config", &path(&file), &path(&music)]).unwrap();
        let options = app_options(&a).unwrap();
        let data = dir.join("parent").join(crate::store::dirs::DATA);
        assert_eq!(options.session, Some(settings.join("session.m3u8")));
        assert_eq!(options.queues, Some(settings.join("queues")));
        assert_eq!(options.grids, Some(settings.join("grids")));
        assert_eq!(options.data, Some(crate::store::dirs::Data(data.clone())));
        assert_eq!(options.rhythm_cache, Some(data.join("analyses")));
        assert_eq!(
            options.lyrics.as_ref().and_then(|l| l.cache.clone()),
            Some(data.join("lyrics"))
        );
        // Everything under its own folder, nothing in the usual settings.
        let usual = crate::store::dirs::settings_dir();
        for p in [&options.session, &options.queues, &options.grids] {
            let p = p.as_deref().unwrap();
            assert!(p.starts_with(&dir), "{}", p.display());
            assert!(usual.as_deref().is_none_or(|u| !p.starts_with(u)));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// What setup gives (every needed model's path, in the manifest's
    /// order) sets the speech model, the timer, the covers' and the
    /// story's models, with or without the GPU's models.
    #[test]
    fn the_models_setup_found_reach_the_app() {
        use crate::setup::{manifest_for, Found};
        let dir = std::path::Path::new("/models");
        for gpu in [true, false] {
            let paths: crate::setup::Paths = manifest_for(gpu)
                .iter()
                .map(|n| (n.file.clone(), n.at(dir)))
                .collect();
            let mut options = AppOptions::new(PathBuf::from("."));
            options.lyrics = Some(crate::work::lyrics::Options::new(false));
            set_models(&mut options, &paths);
            assert_eq!(
                options.model,
                Some(dir.join(crate::work::lyrics::models::MODEL_FILE))
            );
            assert!(options.lyrics.as_ref().unwrap().timer.is_some());
            assert_eq!(options.covers.is_some(), gpu);
            assert_eq!(options.story.is_some(), gpu);
            let found = Found::of(&paths).unwrap();
            assert_eq!(found.asr, dir.join(crate::work::lyrics::models::ASR_FILE));
        }
    }

    #[test]
    fn options_and_the_path() {
        let a = args(&[
            "--device",
            "USB",
            "--headroom",
            "0.5",
            "--config",
            "c.toml",
            "music",
        ])
        .unwrap();
        assert_eq!(a.device.as_deref(), Some("USB"));
        assert_eq!(a.headroom, Some(Headroom(5)));
        assert!(args(&["--headroom", "lots"])
            .unwrap_err()
            .contains("--headroom"));
        assert!(args(&["--no-resample"])
            .unwrap_err()
            .contains("--no-resample"));
        assert_eq!(a.config, Some("c.toml".into()));
        assert_eq!(a.path, Some("music".into()));
        assert!(args(&["--list-devices"]).unwrap().list_devices);
        assert!(args(&["--diag"]).unwrap().diag);
        assert!(args(&["--fetch-model", "out/models"]).is_err());

        assert!(args(&["--null", "--exit-after", "2.5"]).unwrap().null);
    }

    #[test]
    fn bad_arguments_say_why() {
        assert!(args(&["--device"]).unwrap_err().contains("needs a value"));
        assert!(args(&["a", "b"]).is_err());
        assert!(args(&["--frobnicate"])
            .unwrap_err()
            .contains("--frobnicate"));
        assert!(usage("jodeldiplom").starts_with("Usage: jodeldiplom [options] [path]"));
    }

    #[test]
    fn overrides_apply_on_top_of_the_file() {
        let dir = std::env::temp_dir().join(format!("jodeldiplom-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("c.toml");
        durable::write_atomic(&file, b"headroom = 1\n").unwrap();
        let a = args(&["--config", file.to_str().unwrap()]).unwrap();
        let s = settings(&a).unwrap();
        assert_eq!(s.effective.headroom, Headroom(10));
        // --headroom for this run only.
        let a = args(&["--config", file.to_str().unwrap(), "--headroom", "2"]).unwrap();
        let s = settings(&a).unwrap();
        assert_eq!(s.saved.headroom, Headroom(10));
        assert_eq!(s.effective.headroom, Headroom(20));
        // A file starts playing with the browser on its folder.
        let track = dir.join("t.flac");
        durable::write_atomic(&track, b"").unwrap();
        let (start_dir, autoplay) = start(Some(&track)).unwrap();
        assert_eq!(start_dir, dir);
        assert_eq!(autoplay, Some(track));
        assert!(start(Some(&dir.join("missing"))).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A setting the player does not know is said once: the file is written
    /// again without it, the rest kept, and the next start is clean.
    #[test]
    fn settings_not_known_are_said_once() {
        let dir = std::env::temp_dir().join(format!("jodeldiplom-unknown-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("c.toml");
        durable::write_atomic(&file, b"resample = false\nheadroom = 1\n").unwrap();
        let a = args(&["--config", file.to_str().unwrap()]).unwrap();
        let s = settings(&a).unwrap();
        assert_eq!(s.warnings.len(), 2, "{:?}", s.warnings);
        assert!(s.warnings[0].contains("unknown setting \"resample\""));
        assert!(s.warnings[1].contains("written again without them"));
        assert_eq!(s.saved.headroom, Headroom(10));
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(!text.contains("resample") && text.contains("headroom = 1"));
        let again = settings(&a).unwrap();
        assert!(again.warnings.is_empty(), "{:?}", again.warnings);
        assert_eq!(again.saved, s.saved);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_app_bundle_starts_in_the_music_folder() {
        let home = std::env::temp_dir().join(format!("jodeldiplom-home-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        let cwd = PathBuf::from("/somewhere");
        let bundled = Path::new("/Applications/Jodeldiplom.app/Contents/MacOS/jodeldiplom");
        let bare = Path::new("/usr/local/bin/jodeldiplom");
        // From a shell: the current folder.
        assert_eq!(default_folder(cwd.clone(), Some(bare), Some(&home)), cwd);
        // From the bundle: home without a Music folder, then Music.
        assert_eq!(
            default_folder(cwd.clone(), Some(bundled), Some(&home)),
            home
        );
        std::fs::create_dir_all(home.join("Music")).unwrap();
        assert_eq!(
            default_folder(cwd.clone(), Some(bundled), Some(&home)),
            home.join("Music")
        );
        assert_eq!(default_folder(cwd.clone(), Some(bundled), None), cwd);
        std::fs::remove_dir_all(&home).unwrap();
    }
}
