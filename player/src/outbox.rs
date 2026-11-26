//! The way back from the background to the app: the [`Action`]s a worker
//! sends, each with the window woken to take it in ([`Notify`]).

use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;

use crate::core::Action;

/// Something to call after a result was sent, so a sleeping UI loop wakes
/// up (the window frontend posts an event to its loop).
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// Where a worker's results go: the app's actions, each sent with the
/// window woken to take it in.
#[derive(Clone)]
pub struct Outbox {
    tx: Sender<Action>,
    notify: Notify,
}

impl Outbox {
    /// Sends to `tx`, waking with `notify`.
    pub fn new(tx: Sender<Action>, notify: Notify) -> Outbox {
        Outbox { tx, notify }
    }

    /// Sends `action` and wakes the app; false when the app is gone.
    pub fn send(&self, action: Action) -> bool {
        let sent = self.tx.send(action).is_ok();
        if sent {
            (self.notify)();
        }
        sent
    }

    /// Works `action` out on a thread of its own, then sends it.
    pub fn spawn(&self, action: impl FnOnce() -> Action + Send + 'static) {
        let outbox = self.clone();
        thread::spawn(move || {
            outbox.send(action());
        });
    }

    /// The wake-up alone, for a worker whose results the app takes itself.
    pub fn notify(&self) -> Notify {
        Arc::clone(&self.notify)
    }
}
