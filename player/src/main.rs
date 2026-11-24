//! The player. `jodeldiplom --help` for the options.

// No console window for a release build on Windows: started from a
// console, it attaches to that one ([`attach_console`]).
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

// The fonts build.rs found: `EMBEDDED_FONTS`.
include!(concat!(env!("OUT_DIR"), "/fonts.rs"));

fn main() -> std::process::ExitCode {
    attach_console();
    jodeldiplom::gui::fonts::use_fonts(EMBEDDED_FONTS);
    jodeldiplom::main()
}

/// Started from a console, print `--help`, `--diag`, `--list-devices`,
/// the cleaning jobs' lines and errors there (a release build has no
/// console of its own). Output redirected to a file or a pipe goes there
/// as it is.
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
