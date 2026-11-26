//! The player's threads waking the window (an event-loop wake-up): the
//! engine, the workers and
//! setup call [`Wake::notify`]; the window takes one message for each
//! burst of them ([`super::window::Message::Wake`]), not one a call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use iced::futures::channel::mpsc;
use iced::Task;

use super::window::Message;

/// The window's wake-up, for the player's threads.
#[derive(Clone)]
pub struct Wake {
    /// A wake-up was sent and not taken yet.
    pending: Arc<AtomicBool>,
    sender: mpsc::UnboundedSender<()>,
}

/// The window's end of it: the wake-ups as its messages.
pub struct Wakes(mpsc::UnboundedReceiver<()>);

impl Wake {
    /// A wake-up and the window's end of it.
    pub fn new() -> (Wake, Wakes) {
        let (sender, receiver) = mpsc::unbounded();
        let wake = Wake {
            pending: Arc::new(AtomicBool::new(false)),
            sender,
        };
        (wake, Wakes(receiver))
    }

    /// The window has news (from any thread).
    pub fn notify(&self) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            let _ = self.sender.unbounded_send(());
        }
    }

    /// The window took the news: the next call wakes it again.
    pub fn taken(&self) {
        self.pending.store(false, Ordering::Release);
    }
}

impl Wakes {
    /// The wake-ups as the window's messages, for as long as it runs.
    pub fn task(self) -> Task<Message> {
        Task::run(self.0, |()| Message::Wake)
    }
}
