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
//!   a mixer that owns the device, the background workers, the spectrum
//!   and the cover; [`preview`]: the players of a preview, on the mixer
//!   too; [`sfx`]: the karaoke game's sound effects, made in code, played
//!   through the mixer's effects input on the frames of their moments;
//!   [`clock`]: what the app times by (the wall clock, or one a test
//!   steps).
//! - [`browser`]: folder listing, lazy file details, recursive search.
//! - [`work`], [`outbox`]: what the background workers are built of, and
//!   the way their results come back to the app.
//! - [`config`], [`cli`]: the settings file and the command line.
//! - [`cache`]: what is kept on disk between runs, and its schemas;
//!   [`loudness`]: each song's loudness as the karaoke plays it, measured
//!   once and kept, for the game's sounds to follow.
//! - [`spectrum`], [`art`]: the spectrum of what is heard, cover images.
//! - [`gui`]: the iced window; every screen in iced's widgets, the lanes
//!   and Karaoke mode's stage on the GPU through iced's shader widget.

#![warn(missing_docs)]

pub mod app;
pub mod art;
pub mod browser;
pub mod cache;
pub mod calendar;
pub mod cli;
pub mod clock;
pub mod config;
pub mod core;
pub mod cover;
pub mod delays;
pub mod desk_wiring;
pub mod diag;
pub mod dirs;
pub mod download;
pub mod grids;
pub mod gui;
pub mod icon;
pub mod keep;
pub mod loudness;
pub mod lyrics;
pub mod mover;
pub mod outbox;
pub mod patches;
pub mod preview;
pub mod rhythm;
pub mod scores;
pub mod session;
pub mod setup;
pub mod sfx;
pub mod shine_cache;
pub mod single;
pub mod spectrum;
pub mod stems_cache;
pub mod sync;
pub mod theme;
pub mod tiers;
#[cfg(test)]
mod told;
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
            for line in dirs::forget_cache(&folders) {
                println!("{line}");
            }
        }
        if args.clean_diag {
            for line in dirs::forget_diag(&folders) {
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
    let _only = match dirs::settings_dir().map(|dir| single::take(&dir)) {
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
