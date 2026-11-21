//! The window player. `jodeldiplom-gui --help` for the options, which are
//! the same as the terminal player's.

// No console window for a release build on Windows.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

// The fonts build.rs found: `EMBEDDED_MAIN_FONT` and `EMBEDDED_EMOJI_FONT`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

fn main() -> std::process::ExitCode {
    attach_console();
    jodeldiplom::gui::fonts::use_fonts(
        EMBEDDED_MAIN_FONT,
        EMBEDDED_EMOJI_FONT,
        EMBEDDED_LYRICS_FONT,
    );
    jodeldiplom::main(jodeldiplom::Frontend::Window)
}

/// Started from a console, print `--help`, `--list-devices` and errors
/// there (a release build has no console of its own).
#[cfg(windows)]
fn attach_console() {
    use windows::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    // Fails when there is no parent console or one is attached already;
    // either way there is nothing more to do.
    // SAFETY: AttachConsole takes a process id and has no other
    // preconditions.
    let _ = unsafe { AttachConsole(ATTACH_PARENT_PROCESS) };
}

#[cfg(not(windows))]
fn attach_console() {}
