//! A device thread that owns the OS stream and takes commands from the
//! handle, for the backends whose API is driven from a thread of ours
//! (WASAPI, ALSA).

use crate::error::Error;
use crate::output::Driver;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Command {
    Start,
    Pause,
    Flush,
    Close,
}

/// One command and where its result goes.
pub(crate) struct Request {
    pub(crate) command: Command,
    reply: SyncSender<Result<(), Error>>,
}

impl Request {
    pub(crate) fn reply(self, result: Result<(), Error>) {
        let _ = self.reply.send(result);
    }
}

/// How long the handle waits for the device thread to carry out a command.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(5);

/// The handle side: sends commands and waits for their result.
pub(crate) struct ThreadDriver {
    tx: Sender<Request>,
    wake: Box<dyn Fn() + Send>,
    thread: Option<JoinHandle<()>>,
}

impl ThreadDriver {
    /// Spawns `body` on a thread named `name`; it gets the command
    /// receiver. `wake` is called after each command is sent (to signal an
    /// OS event the thread waits on).
    pub(crate) fn spawn(
        name: &str,
        wake: Box<dyn Fn() + Send>,
        body: impl FnOnce(Receiver<Request>) + Send + 'static,
    ) -> Result<ThreadDriver, Error> {
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || body(rx))
            .map_err(Error::from)?;
        Ok(ThreadDriver {
            tx,
            wake,
            thread: Some(thread),
        })
    }

    fn request(&mut self, command: Command) -> Result<(), Error> {
        let (reply, result) = mpsc::sync_channel(1);
        self.tx
            .send(Request { command, reply })
            .map_err(|_| Error::Closed)?;
        (self.wake)();
        match result.recv_timeout(COMMAND_TIMEOUT) {
            Ok(r) => r,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(Error::Timeout("device thread command")),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::Closed),
        }
    }
}

impl Driver for ThreadDriver {
    fn start(&mut self) -> Result<(), Error> {
        self.request(Command::Start)
    }

    fn pause(&mut self) -> Result<(), Error> {
        self.request(Command::Pause)
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.request(Command::Flush)
    }

    fn close(&mut self) -> Result<(), Error> {
        let result = self.request(Command::Close);
        if let Some(t) = self.thread.take() {
            if result.is_ok() {
                let _ = t.join();
            }
        }
        result
    }
}

impl Drop for ThreadDriver {
    fn drop(&mut self) {
        if self.thread.is_some() {
            let _ = self.close();
        }
    }
}
