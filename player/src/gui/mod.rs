//! The window: an iced application ([`window`]) around the player. Every
//! screen draws in iced's own widgets, in the native look ([`look`],
//! [`widgets`]): the main screen laid out from the window's measures
//! ([`metrics`]), the Mixer tab ([`mixer`]), the lanes ([`lanes`]) and the
//! covers in their places; Karaoke mode's stage a widget of its own
//! ([`stage_view`]), its controls iced's widgets over it.

pub mod backdrop;
pub mod big_text;
pub mod bloom;
pub mod confetti;
#[cfg(target_os = "macos")]
mod cover;
pub mod fonts;
pub mod fx;
pub mod glory;
pub mod keys;
pub mod lanes;
pub mod liquid;
pub mod look;
pub mod metrics;
pub mod mixer;
mod pacer;
mod paint;
pub mod paper;
pub mod particles;
pub mod pixels;
pub mod pointer;
pub mod pop;
pub mod present;
pub mod rainbow;
mod screens;
pub mod seal;
#[cfg(test)]
mod shader_check;
pub mod shards;
pub mod snake;
pub mod sparks;
pub mod stage;
pub mod stage_gpu;
pub mod stage_view;
mod stamps;
pub mod streak;
pub(crate) mod style;
mod wake;
pub mod widgets;
pub mod window;

use std::sync::{Arc, Mutex};

use crate::cli::Args;

pub use keys::key;
pub use screens::{lane_spots, transition};

/// The Dock icon: macOS has no window icons, and a bare executable (not
/// in the app bundle, whose `Info.plist` names the icon) would show a
/// generic one. Set on the main thread, once the application exists.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::{AllocAnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let png = crate::icon::png(crate::icon::Style::MacOs, 512);
    let data = NSData::with_bytes(&png);
    let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    // SAFETY: called on the main thread (the marker proves it) with an
    // image AppKit made from our PNG; AppKit keeps its own reference.
    unsafe { app.setApplicationIconImage(Some(&image)) };
}

/// The player's icon at `size` pixels, for the window (the taskbar's comes
/// from the executable's resources on Windows).
fn window_icon(size: u32) -> Option<iced::window::Icon> {
    iced::window::icon::from_rgba(
        crate::icon::rgba(crate::icon::Style::Windows, size),
        size,
        size,
    )
    .ok()
}

/// What iced runs with: the fonts its widgets draw with (the player's
/// own: Nunito for the native look's text, [`look::font`]), and the one
/// they draw in unless they say.
pub fn iced_settings() -> iced::Settings {
    let fonts = [
        Some(fonts::main_font()),
        fonts::emoji_font(),
        fonts::lyrics_font(),
        fonts::words_font(),
    ];
    iced::Settings {
        fonts: fonts.into_iter().flatten().map(Into::into).collect(),
        font: style::ui(),
        ..iced::Settings::default()
    }
}

/// Runs the player in a window until it is closed. `JODELDIPLOM_SOFTWARE`
/// set: iced's CPU renderer, as where there is no GPU.
pub fn run(args: &Args) -> Result<(), String> {
    let mut options = crate::cli::app_options(args)?;
    let (wake, wakes) = wake::Wake::new();
    let notify = wake.clone();
    options.notify = Arc::new(move || notify.notify());
    let error = Arc::new(Mutex::new(None));
    let start = Mutex::new(Some(window::Start {
        options,
        wake,
        wakes,
        exit_after: args.exit_after,
        save_frame: args.save_frame.clone(),
        error: error.clone(),
    }));
    let boot = move || {
        let start = start.lock().ok().and_then(|mut s| s.take());
        window::Gui::new(start.expect("the window boots once"))
    };
    let settings = iced::window::Settings {
        size: iced::Size::new(1280.0, 800.0),
        min_size: Some(iced::Size::new(480.0, 300.0)),
        icon: window_icon(32),
        exit_on_close_request: false,
        ..Default::default()
    };
    let mut application = iced::application(boot, window::Gui::update, window::Gui::view)
        .title(window::Gui::title)
        .subscription(window::Gui::subscription)
        .theme(|_: &window::Gui| iced::Theme::Dark)
        .settings(iced_settings())
        .window(settings);
    if std::env::var_os("JODELDIPLOM_SOFTWARE").is_some() {
        application = application.backend(iced::Backend::Software);
    }
    crate::trace::note("window: starting");
    application
        .run()
        .map_err(|e| format!("the window failed: {e}"))?;
    let error = error.lock().ok().and_then(|mut e| e.take());
    error.map_or(Ok(()), Err)
}
