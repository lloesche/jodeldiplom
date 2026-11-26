//! `jodeldiplom`: a file-browser music player for PCM and lossy files, in a
//! window: a UI core, and an iced window around it that draws every screen
//! from the core's presentation models.
//!
//! # Layers
//!
//! - [`core`]: the state, the actions, `update`, the keymap, and what the
//!   window's size does to the keys ([`core::Fit`]). Pure: no IO, no
//!   drawing.
//! - [`view`]: the presentation models, what each screen says.
//! - [`app`]: the running player: the engine ([`jodel_playback::Player`]) on
//!   a mixer that owns the device ([`desk_wiring`]), the background
//!   workers, the spectrum ([`spectrum`]) and the cover; [`preview`]: the
//!   players of a preview, on the mixer too; [`sync`]: the sync's clicks
//!   played and its answers heard; [`sfx`]: the karaoke game's sound
//!   effects, made in code, played through the mixer's effects input on
//!   the frames of their moments; [`clock`]: what the app times by (the
//!   wall clock, or one a test steps).
//! - [`browser`]: folder listing, lazy file details, recursive search.
//! - [`work`]: the background workers (the beats and tempo, the lyrics,
//!   the covers taken apart, the cover art, the songs' loudness, the
//!   original singer's phrases, the files fetched for the setup), what
//!   they are built of, and the way their results come back to the app.
//! - [`store`]: what is kept on disk between runs, and the schema of each
//!   kind of file: the two folders and a move between them, the session,
//!   the user's own records (the grids, the delays, the scores, the takes,
//!   the patches) and the stems kept.
//! - [`config`], [`cli`], [`setup`], [`diag`]: the settings file, the
//!   command line, the self-check (and the setup when it fails), and
//!   `--diag`.
//! - [`tiers`], [`theme`]: the game's tiers, and the themes over them.
//! - [`single`], [`icon`], [`calendar`], [`trace`]: one player at a time,
//!   its icon, the day, a timing trace.
//! - [`gui`]: the iced window; every screen in iced's widgets, the lanes
//!   and Karaoke mode's stage on the GPU through iced's shader widget.

#![warn(missing_docs)]

pub mod app;
pub mod browser;
pub mod calendar;
pub mod cli;
pub mod clock;
pub mod config;
pub mod core;
pub mod desk_wiring;
pub mod diag;
pub mod gui;
pub mod icon;
pub mod preview;
pub mod setup;
pub mod sfx;
pub mod single;
pub mod spectrum;
pub mod store;
pub mod sync;
pub mod theme;
pub mod tiers;
pub mod trace;
pub mod view;
pub mod work;

use std::process::ExitCode;

/// The binary's name, in its messages and its usage.
pub const BIN: &str = "jodeldiplom";

/// The binary's `main`: parses the command line and runs the job it
/// names (`--help`, `--diag`, `--clean-cache`, `--clean-diag`,
/// `--list-devices`), else the window.
pub fn main() -> ExitCode {
    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{BIN}: {error}\n\n{}", cli::usage(BIN));
            return ExitCode::from(2);
        }
    };
    if args.help {
        println!("{}", cli::usage(BIN));
        let (main, emoji) = gui::fonts::built_in_fonts();
        println!(
            "\nFont: {}{}",
            if main {
                "Sarasa Fixed J (built in)"
            } else {
                "Cascadia Mono (Sarasa Fixed J not built in)"
            },
            if emoji {
                ", emoji: Noto Emoji (built in)"
            } else {
                ", emoji: system fonts only"
            }
        );
        return ExitCode::SUCCESS;
    }
    if args.diag {
        return match diag::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{BIN}: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.clean_cache || args.clean_diag {
        let folders = match cli::settings(&args).and_then(|s| cli::folders(&s)) {
            Ok(folders) => folders,
            Err(error) => {
                eprintln!("{BIN}: {error}");
                return ExitCode::FAILURE;
            }
        };
        if args.clean_cache {
            for line in store::dirs::forget_cache(&folders) {
                println!("{line}");
            }
        }
        if args.clean_diag {
            for line in store::dirs::forget_diag(&folders) {
                println!("{line}");
            }
        }
        return ExitCode::SUCCESS;
    }
    if args.list_devices {
        return match cli::device_list() {
            Ok(text) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{BIN}: {error}");
                ExitCode::FAILURE
            }
        };
    }
    // One player at a time: held until the player ends.
    let _only = match store::dirs::settings_dir().map(|dir| single::take(&dir)) {
        Some(single::Start::Running) => {
            single::say_running(BIN);
            return ExitCode::FAILURE;
        }
        Some(single::Start::Alone(only)) => only,
        None => None,
    };
    match gui::run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{BIN}: {error}");
            ExitCode::FAILURE
        }
    }
}
