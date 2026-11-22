//! `jodeldiplom`: a file-browser music player for PCM and lossy files, with a
//! terminal frontend (`jodeldiplom`) and a window frontend
//! (`jodeldiplom-gui`) that share one UI core and the same widgets.
//!
//! # Layers
//!
//! - [`core`]: the state, the actions, `update`, the layout, the keymap
//!   and the mouse regions. Pure: no IO, no drawing.
//! - [`view`]: the ratatui widgets that draw the state, for both
//!   frontends; covers and spectra go through [`view::Pictures`].
//! - [`app`]: the running player: the engine ([`jodel_playback::Player`]) on
//!   a mixer that owns the device, the background workers, the spectrum
//!   and the cover; [`preview`]: the players of a preview, on the mixer
//!   too.
//! - [`browser`]: folder listing, lazy file details, recursive search.
//! - [`config`], [`cli`]: the settings file and the command line.
//! - [`cache`]: what is kept on disk between runs, and its schemas.
//! - [`spectrum`], [`art`]: the spectrum of what is heard, cover images.
//! - [`tui`]: crossterm + ratatui on the terminal.
//! - [`gui`]: winit + softbuffer; the same widgets rasterised by a small
//!   ratatui pixel backend with an embedded font, covers and spectra
//!   painted in pixels.

#![warn(missing_docs)]

pub mod app;
pub mod art;
pub mod browser;
pub mod cache;
pub mod cli;
pub mod config;
pub mod core;
pub mod delays;
pub mod desk_wiring;
pub mod diag;
pub mod dirs;
pub mod grids;
pub mod gui;
pub mod icon;
pub mod keep;
pub mod lyrics;
pub mod mover;
pub mod preview;
pub mod rhythm;
pub mod scores;
pub mod session;
pub mod setup;
pub mod single;
pub mod spectrum;
pub mod stems_cache;
pub mod sync;
pub mod trace;
pub mod tui;
pub mod view;

use std::process::ExitCode;

/// Which frontend a binary runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frontend {
    /// The terminal.
    Terminal,
    /// The window.
    Window,
}

/// The binaries' `main`: parses the command line and runs `frontend`.
pub fn main(frontend: Frontend) -> ExitCode {
    let bin = match frontend {
        Frontend::Terminal => "jodeldiplom",
        Frontend::Window => "jodeldiplom-gui",
    };
    let args = match cli::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{bin}: {error}\n\n{}", cli::usage(bin));
            return ExitCode::from(2);
        }
    };
    if args.help {
        println!("{}", cli::usage(bin));
        if frontend == Frontend::Window {
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
        }
        return ExitCode::SUCCESS;
    }
    if args.diag {
        return match diag::run(&args, frontend) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{bin}: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.clean_cache || args.clean_diag {
        let data = cli::settings(&args)
            .ok()
            .and_then(|s| dirs::Data::chosen(s.effective.data_folder.as_deref()));
        let Some(data) = data else {
            eprintln!("{bin}: no data folder (no home folder?)");
            return ExitCode::FAILURE;
        };
        if args.clean_cache {
            for line in dirs::forget_cache(&data) {
                println!("{line}");
            }
        }
        if args.clean_diag {
            println!("{}", dirs::forget_diag(&data));
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
                eprintln!("{bin}: {error}");
                ExitCode::FAILURE
            }
        };
    }
    // One player at a time: held until the player ends.
    let _only = match dirs::settings_dir().map(|dir| single::take(&dir)) {
        Some(single::Start::Running) => {
            single::say_running(bin, frontend == Frontend::Window);
            return ExitCode::FAILURE;
        }
        Some(single::Start::Alone(only)) => only,
        None => None,
    };
    let result = match frontend {
        Frontend::Terminal => tui::run(&args),
        Frontend::Window => gui::run(&args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{bin}: {error}");
            ExitCode::FAILURE
        }
    }
}
