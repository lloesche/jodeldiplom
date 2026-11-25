//! The command line: `jodeldiplom [options] [path]`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_output::SimClock;
use jodel_playback::Headroom;

use crate::app::{self, AppOptions, Output};
use crate::config::{self, Config};

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

/// The app options for a command line (the window fills in the size and
/// the wake-up).
pub fn app_options(args: &Args) -> Result<AppOptions, String> {
    let settings = settings(args)?;
    let (dir, autoplay) = start(args.path.as_deref())?;
    let mut options = AppOptions::new(dir);
    // The data folder: the analyses, stems, lyrics and models. A move of it
    // written down and not finished goes on in the background
    // ([`start_mover`]): until it has, what is not moved yet is read where
    // it is.
    let data = crate::dirs::Data::chosen(settings.effective.data_folder.as_deref())
        .ok_or("no folder for the player's data (no home folder?)")?;
    // Only with the usual settings: another settings file (--config) is a
    // player of its own.
    if args.config.is_none() {
        if let Some(dir) = crate::dirs::settings_dir() {
            crate::mover::resume(&dir);
        }
    }
    // What a crash left half written (never under a file's name, so never
    // read) goes, in the background, once a day old.
    let swept = data.clone();
    std::thread::spawn(move || {
        let mut dirs = vec![
            swept.analyses(),
            swept.stems(),
            swept.lyrics(),
            swept.models(),
            swept.sounds(),
        ];
        dirs.extend(crate::cover::CoverStore::dirs_in(&swept.covers()));
        dirs.extend(crate::dirs::settings_dir());
        for dir in dirs {
            durable::sweep(&dir, durable::STALE);
        }
    });
    options.rhythm_cache = Some(data.analyses());
    options.grids = crate::grids::dir();
    options.session = crate::session::path();
    options.restore_browser = args.path.is_none();
    options.queues = crate::session::queues_dir();
    // Lyrics: cached in the data folder, LRCLIB asked unless turned off. The
    // speech models come once setup has found them ([`set_models`]).
    let mut lyrics = crate::lyrics::Options::new(true);
    lyrics.cache = Some(data.lyrics());
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
    options.devices = Arc::new(app::list_devices);
    Ok(options)
}

/// Starts the moves the journal holds (a data folder chosen anew), telling `options.notify` as they go: before
/// setup, which reads what has not moved yet where it is.
pub fn start_mover(options: &mut AppOptions) {
    // The journal beside the settings file in use.
    let dir = options
        .config_file
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .or_else(crate::dirs::settings_dir);
    if let Some(dir) = dir {
        options.mover = crate::mover::Mover::start(dir, options.notify.clone()).map(Arc::new);
    }
}

/// The checked models' paths when every one is there (the player starts at
/// once); else what setup is to get, and where.
pub fn models_ready(
    options: &AppOptions,
) -> Result<Vec<PathBuf>, (Vec<crate::setup::Need>, crate::setup::Places)> {
    let manifest = crate::setup::manifest();
    let data = options
        .data
        .clone()
        .or_else(crate::dirs::Data::current)
        .unwrap_or_else(|| crate::dirs::Data::in_parent(Path::new(".")));
    let places = crate::setup::Places::player(&data);
    crate::setup::ready(&manifest, &places).ok_or((manifest, places))
}

/// Gives the app the models setup found (each known by its file:
/// [`crate::setup::Found`]): the speech models to the lyrics, the covers'
/// models to the covers' step.
pub fn set_models(options: &mut AppOptions, paths: &[PathBuf]) {
    let Some(found) = crate::setup::Found::of(paths) else {
        return;
    };
    options.covers = found.covers.clone();
    let (speech, asr) = (found.speech, found.asr);
    let models = crate::lyrics::models::Models::new(speech.clone(), asr);
    options.model = Some(speech);
    if let Some(lyrics) = &mut options.lyrics {
        lyrics.timer = Some(Arc::new(crate::lyrics::timer::SpeechTimer::new(models))
            as Arc<dyn crate::lyrics::WordTimer>);
    }
}

/// The `--list-devices` text.
pub fn device_list() -> Result<String, String> {
    let devices = app::list_devices()?;
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

    /// What setup's `ready` gives (every needed file's path, in the
    /// manifest's order) sets the speech model, the timer and the covers'
    /// models, with or without the covers' files.
    #[test]
    fn the_models_setup_found_reach_the_app() {
        use crate::setup::{manifest_for, Found};
        let dir = std::path::Path::new("/models");
        for gpu in [true, false] {
            let paths: Vec<PathBuf> = manifest_for(gpu)
                .iter()
                .map(|n| dir.join(&n.file))
                .collect();
            let mut options = AppOptions::new(PathBuf::from("."));
            options.lyrics = Some(crate::lyrics::Options::new(false));
            set_models(&mut options, &paths);
            assert_eq!(
                options.model,
                Some(dir.join(crate::lyrics::models::MODEL_FILE))
            );
            assert!(options.lyrics.as_ref().unwrap().timer.is_some());
            assert_eq!(options.covers.is_some(), gpu);
            let found = Found::of(&paths).unwrap();
            assert_eq!(found.asr, dir.join(crate::lyrics::models::ASR_FILE));
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
