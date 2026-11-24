//! macOS: the window's Metal layer keeps three drawables, not two.
//!
//! iced configures its surface for the least latency, which on Metal is a
//! layer of two drawables: one on screen, one being drawn. A frame not
//! through by the refresh then waits a whole refresh more, and the window
//! ran at about 34 frames a second on a 60 Hz display, unevenly (words in
//! the bubbles smeared to the eye). Not waiting for the display at all ran
//! it at 60, but tore. With three drawables it waits for the display and
//! keeps up. iced sets two again whenever it configures the surface (at a
//! resize, inside the redraw, before the widgets see it), so the canvas's
//! widget puts three back on each redraw ([`keep_three`]).

use std::sync::atomic::{AtomicBool, Ordering};

use objc2::msg_send;
use objc2::runtime::{AnyClass, AnyObject};

/// The drawables the layer keeps.
const DRAWABLES: usize = 3;

/// The window's view (its `NSView`), as an address, to find its Metal
/// layer under; none where the window is not an AppKit one.
pub fn view_of(window: &dyn iced::window::Window) -> Option<usize> {
    use iced::window::raw_window_handle::RawWindowHandle;
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::AppKit(handle) => Some(handle.ns_view.as_ptr() as usize),
        _ => None,
    }
}

/// Gives the Metal layer under `view` (the view's own layer or one of its
/// sublayers, where wgpu put it) [`DRAWABLES`] drawables when it has fewer.
/// On the main thread, inside a redraw of the window `view` is of.
pub fn keep_three(view: usize) {
    let Some(metal) = AnyClass::get(c"CAMetalLayer") else {
        return;
    };
    // SAFETY: `view` is the window's NSView (from its window handle), alive
    // while the window is, and this runs on the main thread inside that
    // window's redraw. Each message is sent to an object of a class that
    // answers it: `layer` to an NSView, `isKindOfClass:` and `sublayers`
    // to CALayers, `count`/`objectAtIndex:` to the NSArray `sublayers`
    // returns, the drawable count only to a CAMetalLayer.
    unsafe {
        let top: *mut AnyObject = msg_send![view as *mut AnyObject, layer];
        let first = !SEEN.swap(true, Ordering::Relaxed);
        let Some(layer) = find(top, metal, 2) else {
            if first {
                let name = if top.is_null() {
                    "none".to_owned()
                } else {
                    (*top).class().name().to_string_lossy().into_owned()
                };
                crate::trace::note(format_args!(
                    "metal layer: none under the window's view (its layer: {name})"
                ));
            }
            return;
        };
        let count: usize = msg_send![layer, maximumDrawableCount];
        let opaque: bool = msg_send![layer, isOpaque];
        let sync: bool = msg_send![layer, displaySyncEnabled];
        if first {
            crate::trace::note(format_args!(
                "metal layer: found ({}), {count} drawables, opaque {opaque}, display sync {sync}",
                if layer == top {
                    "the view's own"
                } else {
                    "a sublayer"
                }
            ));
        }
        if count < DRAWABLES {
            let () = msg_send![layer, setMaximumDrawableCount: DRAWABLES];
            let now: usize = msg_send![layer, maximumDrawableCount];
            crate::trace::note(format_args!("metal layer: {count} drawables, now {now}"));
        }
        // An experiment (JODELDIPLOM_LAYER=translucent): a layer not opaque
        // is composited, never shown directly by the display.
        if opaque && translucent() {
            let () = msg_send![layer, setOpaque: false];
            crate::trace::note("metal layer: made translucent");
        }
    }
}

/// Whether the layer is to be made translucent (`JODELDIPLOM_LAYER`).
fn translucent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("JODELDIPLOM_LAYER").is_ok_and(|v| v.eq_ignore_ascii_case("translucent"))
    })
}

/// Whether the trace was told what the layer is like (once).
static SEEN: AtomicBool = AtomicBool::new(false);

/// `layer` where it is a `class`, else the first of its sublayers (down to
/// `depth` levels) that is.
///
/// # Safety
/// `layer` is null or a CALayer, and this runs on the main thread.
unsafe fn find(layer: *mut AnyObject, class: &AnyClass, depth: u32) -> Option<*mut AnyObject> {
    if layer.is_null() {
        return None;
    }
    // SAFETY: as the caller promises; `sublayers` is nil or an NSArray of
    // CALayers.
    unsafe {
        let is: bool = msg_send![layer, isKindOfClass: class];
        if is {
            return Some(layer);
        }
        if depth == 0 {
            return None;
        }
        let subs: *mut AnyObject = msg_send![layer, sublayers];
        if subs.is_null() {
            return None;
        }
        let n: usize = msg_send![subs, count];
        (0..n).find_map(|i| {
            let sub: *mut AnyObject = msg_send![subs, objectAtIndex: i];
            find(sub, class, depth - 1)
        })
    }
}
