//! Dropping what the playback thread is done with (a track's file and
//! buffers, a switch's or a jump's second reading) on one long-lived
//! thread of its own, so closing a file on a network share never holds
//! up the writing.

use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;

/// Drops `value` on the `jodel-drop` thread, started with the first one.
///
/// When that thread is not there, `value` is dropped here, on the caller's
/// thread: when it could not be started, and once a `Drop` that panicked
/// on it has ended it (every drop after that happens on the playback
/// thread).
pub(super) fn drop_later<T: Send + 'static>(value: T) {
    static DROPS: OnceLock<Option<Sender<Box<dyn Send>>>> = OnceLock::new();
    let drops = DROPS.get_or_init(|| {
        let (sender, values) = mpsc::channel::<Box<dyn Send>>();
        std::thread::Builder::new()
            .name("jodel-drop".into())
            .spawn(move || values.into_iter().for_each(drop))
            .ok()
            .map(|_| sender)
    });
    if let Some(drops) = drops {
        // A send to a thread that has ended gives the value back, dropped
        // here.
        let _ = drops.send(Box::new(value));
    }
}
