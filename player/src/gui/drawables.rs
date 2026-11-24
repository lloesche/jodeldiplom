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
        let layer: *mut AnyObject = msg_send![view as *mut AnyObject, layer];
        let Some(layer) = find(layer, metal, 2) else {
            return;
        };
        let count: usize = msg_send![layer, maximumDrawableCount];
        if count < DRAWABLES {
            let () = msg_send![layer, setMaximumDrawableCount: DRAWABLES];
        }
    }
}

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
