//! A device thread that owns the OS stream, for the backends whose API is
//! driven from a thread of ours (WASAPI, ALSA): opened there
//! ([`spawn_opened`]), and for an output taking commands from the handle
//! ([`ThreadDriver`]).

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

/// How long an opener waits for its device thread to open the device.
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);

/// Spawns a device thread named `name`, which opens the device there with
/// `open` (as some APIs want: the stream lives on the thread that made
/// it) and then runs it with `run`. Waits up to [`OPEN_TIMEOUT`] for the
/// open, and returns what `open` gave the opener and the thread; or its
/// error (the thread joined), or [`Error::Timeout`] for `opening`. A
/// thread that opens the device after its opener stopped waiting lets it
/// go, and ends.
pub(crate) fn spawn_opened<I, T>(
    name: &str,
    opening: &'static str,
    open: impl FnOnce() -> Result<(I, T), Error> + Send + 'static,
    run: impl FnOnce(T) + Send + 'static,
) -> Result<(I, JoinHandle<()>), Error>
where
    I: Send + 'static,
{
    let (ready, opened) = mpsc::sync_channel(1);
    let thread = std::thread::Builder::new()
        .name(name.into())
        .spawn(move || match open() {
            Ok((info, device)) => {
                if ready.send(Ok(info)).is_ok() {
                    run(device);
                }
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        })
        .map_err(Error::from)?;
    match opened.recv_timeout(OPEN_TIMEOUT) {
        Ok(Ok(info)) => Ok((info, thread)),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => Err(Error::Timeout(opening)),
    }
}

/// The handle side: sends commands and waits for their result.
pub(crate) struct ThreadDriver {
    tx: Sender<Request>,
    wake: Box<dyn Fn() + Send>,
    thread: Option<JoinHandle<()>>,
}

impl ThreadDriver {
    /// Opens an output device on a thread named `name` ([`spawn_opened`])
    /// and drives it there: `run` gets the device and the command
    /// receiver. `wake` is called after each command is sent (to signal an
    /// OS event the thread waits on).
    pub(crate) fn open<I, T>(
        name: &str,
        wake: Box<dyn Fn() + Send>,
        open: impl FnOnce() -> Result<(I, T), Error> + Send + 'static,
        run: impl FnOnce(T, Receiver<Request>) + Send + 'static,
    ) -> Result<(I, ThreadDriver), Error>
    where
        I: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let (info, thread) = spawn_opened(name, "opening the device", open, move |device| {
            run(device, rx)
        })?;
        let driver = ThreadDriver {
            tx,
            wake,
            thread: Some(thread),
        };
        Ok((info, driver))
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
