//! macOS: Karaoke mode covers the screen as a frameless window, not as a
//! fullscreen Space.
//!
//! In a fullscreen Space macOS showed the window's frames at 30 a second
//! on a 60 Hz display, however idle the GPU (2 ms a frame), and the words
//! in the bubbles smeared to the eye; the same window covering the screen
//! without a Space ran at 60. So, as fullscreen was before Spaces: the
//! title bar goes, the window takes the screen's whole frame, the menu bar
//! and the Dock hide, and it cannot be moved or resized; leaving, all of it
//! comes back as it was.

use std::cell::Cell;

use objc2::MainThreadMarker;
use objc2_app_kit::{NSApplication, NSApplicationPresentationOptions, NSView, NSWindowStyleMask};
use objc2_foundation::NSRect;

/// The window as it was before it covered the screen.
#[derive(Clone, Copy)]
struct Before {
    style: NSWindowStyleMask,
    frame: NSRect,
    options: NSApplicationPresentationOptions,
}

thread_local! {
    /// While the window covers the screen: how it was before.
    static BEFORE: Cell<Option<Before>> = const { Cell::new(None) };
}

/// Makes `window` cover its screen (`on`) or puts it back as it was. On
/// the main thread, as iced runs window tasks.
pub fn cover(window: &dyn iced::window::Window, on: bool) {
    use iced::window::raw_window_handle::RawWindowHandle;
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return;
    };
    // SAFETY: an AppKit window handle's view is the window's NSView, alive
    // while the window is (this runs inside a task on it).
    let view: &NSView = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
    let Some(ns) = view.window() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    if on {
        if BEFORE.get().is_some() {
            return;
        }
        let Some(screen) = ns.screen() else {
            return;
        };
        BEFORE.set(Some(Before {
            style: ns.styleMask(),
            frame: ns.frame(),
            options: app.presentationOptions(),
        }));
        app.setPresentationOptions(
            NSApplicationPresentationOptions::HideDock
                | NSApplicationPresentationOptions::HideMenuBar,
        );
        let fixed = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Resizable
            | NSWindowStyleMask::Miniaturizable;
        ns.setStyleMask(ns.styleMask() & !fixed);
        ns.setFrame_display(screen.frame(), true);
        ns.setMovable(false);
    } else if let Some(before) = BEFORE.take() {
        app.setPresentationOptions(before.options);
        ns.setStyleMask(before.style);
        ns.setFrame_display(before.frame, true);
        ns.setMovable(true);
    }
}
