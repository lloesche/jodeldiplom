//! A mixer input's streams: full [`Output`]s whose ring the mixer drains,
//! and the map that keeps the microphone's tap on the device in the Lead
//! stream's frames.
//!
//! An output is made on the thread that opens it ([`open`]): its ring,
//! its handle and what the mixer thread keeps of it, with room for every
//! chunk, so the mixer thread takes it in, mixes it and lets it go
//! without allocating.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender, SyncSender};
use std::sync::Arc;

use super::follow::Follow;
use super::{MixShared, Msg};
use crate::error::Error;
use crate::format::StreamFormat;
use crate::lanes::lane_format;
use crate::output::{Driver, OpenRequest, Output, StreamInfo};
use crate::pack::unpack_sample;
use crate::shared::Shared;
use crate::tap::Tap;

/// What a stream's handle shares with the mixer thread.
pub(crate) struct Link {
    pub(crate) id: u64,
    pub(crate) shared: Arc<Shared>,
    /// The stream is started (not paused, flushed or closed): the mixer
    /// takes frames from it.
    pub(crate) running: AtomicBool,
    /// The stream is paused (not just held back for a flush or a prefill).
    pub(crate) paused: AtomicBool,
}

/// An output open on an input, as the mixer thread drains it.
pub(crate) struct Feed {
    pub(crate) link: Arc<Link>,
    /// The lanes of each frame (1: a plain stream; see [`crate::lanes`]).
    pub(crate) lanes: u16,
    /// The output it follows, while it does ([`super::Fader::follow`]).
    pub(crate) follow: Option<Follow>,
    /// This chunk: taken from (it was running), the frames it gave, the
    /// chunk's frame they start at, and its stream frame there.
    pub(crate) taken: bool,
    pub(crate) got: u64,
    pub(crate) at: usize,
    pub(crate) frame: u64,
    bytes: Vec<u8>,
    pub(crate) samples: Vec<i32>,
    /// Its chunks on the device not all played yet: the device frame its
    /// frames start at, the stream frames dropped just before it (played
    /// once the device is there), and how many of its frames follow. Room
    /// for what the device's ring holds is made at the start ([`open`]);
    /// past it the two oldest become one ([`Feed::note_sent`]).
    sent: VecDeque<(u64, u64, u64)>,
    /// The restarts (by a flush) of the stream that what was taken last
    /// came after.
    restarts: u64,
}

/// What a stream gave in a chunk: from the chunk's frame `at`, its stream
/// frame `frame` and `got` more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Given {
    pub(super) at: usize,
    pub(super) frame: u64,
    pub(super) got: u64,
}

/// The frames of a chunk a stream fills: from the chunk's frame `from` to
/// `to`; for a follower, with the stream frame due at `from`, its frames
/// before that dropped (late), or its frames from a later one (early).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) due: Option<u64>,
}

impl Span {
    /// Frames `0..n`, whatever comes next.
    pub(crate) fn whole(n: usize) -> Span {
        Span {
            from: 0,
            to: n,
            due: None,
        }
    }

    /// None of the chunk.
    pub(crate) const NONE: Span = Span {
        from: 0,
        to: 0,
        due: None,
    };
}

impl Feed {
    /// The stream of `link`, of `lanes` lanes, with room for chunks of up
    /// to `chunk` frames, `sent` of them on the device not played yet.
    pub(crate) fn new(link: Arc<Link>, lanes: u16, chunk: usize, sent: usize) -> Feed {
        let format = link.shared.format;
        Feed {
            link,
            lanes,
            follow: None,
            taken: false,
            got: 0,
            at: 0,
            frame: 0,
            bytes: Vec::with_capacity(chunk * format.frame_bytes()),
            samples: Vec::with_capacity(chunk * usize::from(format.channels)),
            sent: VecDeque::with_capacity(sent),
            restarts: 0,
        }
    }

    /// Takes its frames of `span` in the chunk of `n` frames that goes to
    /// the device from device frame `first` into `samples`, in its ring's
    /// format (every lane of a lane stream; silence for the rest of the
    /// chunk and what it lacks, or all of it when it is not running or a
    /// flush keeps the mixer out). Returns whether it was rendered.
    pub(crate) fn take(&mut self, n: usize, first: u64, span: Span) -> bool {
        let format = self.link.shared.format;
        self.taken = self.link.running.load(Ordering::SeqCst);
        self.got = 0;
        if !self.taken {
            self.at = span.from;
            return false;
        }
        let fb = format.frame_bytes();
        self.bytes.resize(n * fb, 0);
        let (from, to, dropped) = self.lock_to(span);
        self.bytes[..from * fb].fill(0);
        self.bytes[to * fb..].fill(0);
        let rendered = if from < to {
            self.link.shared.take(&mut self.bytes[from * fb..to * fb])
        } else {
            None
        };
        self.at = from;
        if let Some((frame, got, restarts)) = rendered {
            self.restarted(restarts);
            self.frame = frame;
            self.got = got;
        }
        if dropped > 0 || self.got > 0 {
            self.note_sent((first + from as u64, dropped, self.got));
        }
        self.samples.resize(n * usize::from(format.channels), 0);
        let sb = format.sample.bytes();
        for (s, b) in self.samples.iter_mut().zip(self.bytes.chunks_exact(sb)) {
            *s = unpack_sample(b, format.sample);
        }
        rendered.is_some()
    }

    /// Notes a chunk sent to the device. With no room left (the device
    /// holds more than its ring), the two oldest become one: frames that
    /// follow on, as they do from chunk to chunk, are counted exactly
    /// (but for the frames dropped before the second, played as soon as
    /// the device reaches the first); a gap between them counts the oldest
    /// as played.
    fn note_sent(&mut self, chunk: (u64, u64, u64)) {
        if self.sent.len() == self.sent.capacity() {
            if let Some((d, dropped, n)) = self.sent.pop_front() {
                let follows = self.sent.front_mut().filter(|next| next.0 == d + n);
                if let Some(next) = follows {
                    *next = (d, dropped + next.1, n + next.2);
                }
            }
        }
        self.sent.push_back(chunk);
    }

    /// What it gave in the chunk taken last (`None`: nothing).
    pub(super) fn given(&self) -> Option<Given> {
        (self.got > 0).then_some(Given {
            at: self.at,
            frame: self.frame,
            got: self.got,
        })
    }

    /// The frames of `span` it fills, its frame `span.due` due at
    /// `span.from`: late, it drops its frames before that one (as many as
    /// are queued) and gives none while still late; early, its frames
    /// start where they are due. Returns those frames, and how many it
    /// dropped.
    fn lock_to(&mut self, span: Span) -> (usize, usize, u64) {
        let Some(due) = span.due else {
            return (span.from, span.to, 0);
        };
        let shared = &self.link.shared;
        let mut head = shared.head();
        let mut dropped = 0;
        if head < due {
            if let Some((n, restarts)) = shared.skip(due - head) {
                self.restarted(restarts);
                dropped = n;
                head += n;
            }
        }
        if head < due {
            // Still late: what it has is not yet on its frames.
            return (span.to, span.to, dropped);
        }
        let from = (span.from as u64 + (head - due)).min(span.to as u64);
        (from as usize, span.to, dropped)
    }

    /// The output's stream is at restart `restarts`: what was sent before
    /// a restart is not its to count.
    fn restarted(&mut self, restarts: u64) {
        if restarts != self.restarts {
            self.sent.clear();
            self.restarts = restarts;
        }
    }

    /// The device has played up to device frame `played`: the output is
    /// told how far its own stream is played, all it took but what is
    /// still on the device.
    pub(crate) fn played(&mut self, played: u64) {
        while self.sent.front().is_some_and(|&(d, _, n)| d + n <= played) {
            self.sent.pop_front();
        }
        let unplayed: u64 = self
            .sent
            .iter()
            .map(|&(d, dropped, n)| {
                let before = if played < d { dropped } else { 0 };
                before + n - played.saturating_sub(d).min(n)
            })
            .sum();
        let shared = &self.link.shared;
        shared.set_played(shared.head().saturating_sub(unplayed));
    }
}

/// Asks the mixer thread and waits for its answer (`Closed` when it is
/// gone). The answer's way is made here, room for it and all, so the
/// mixer thread's answer allocates nothing.
pub(crate) fn ask<T>(tx: &Sender<Msg>, msg: impl FnOnce(SyncSender<T>) -> Msg) -> Result<T, Error> {
    let (reply, answer) = mpsc::sync_channel(1);
    tx.send(msg(reply)).map_err(|_| Error::Closed)?;
    answer.recv().map_err(|_| Error::Closed)
}

/// The mixer's answer to an output made for it ([`Msg::Open`]).
pub(crate) enum Opened {
    /// It plays: on a device of this backend, in real time or not.
    Yes {
        backend: &'static str,
        realtime: bool,
    },
    /// Not in that format: the desk is in this one now.
    Desk(StreamFormat),
}

/// Opens an output for `request` on input `slot` (of `lanes` lanes) of the
/// mixer of `tx` and `shared`. Its ring, its handle and the mixer's side of
/// it are made here, on the opener's thread, at the desk's format as the
/// handles see it (asked of the mixer when they see none, and made again
/// when the desk has changed since); the mixer thread takes it in, the
/// output on the input before it closing only then.
pub(crate) fn open(
    tx: &Sender<Msg>,
    shared: &MixShared,
    slot: u64,
    lanes: u16,
    request: &OpenRequest,
) -> Result<Box<dyn Output>, Error> {
    request.validate()?;
    if request.lanes != 1 && request.lanes != lanes {
        return Err(Error::InvalidArgument(format!(
            "an input of {lanes} lanes opens 1 or {lanes}, not {}",
            request.lanes
        )));
    }
    let seen = *shared.format();
    let mut format = match seen {
        Some(format) => format,
        None => {
            let caps = ask(tx, |done| Msg::Capabilities { slot, done })??;
            let desk = caps.iter().next().copied();
            desk.ok_or(Error::NoDevice)?
        }
    };
    loop {
        // Every output plays at the desk's format (its player resamples to
        // it).
        if !request.takes(format) {
            return Err(Error::FormatNotAccepted {
                rate: request.rate,
                channels: request.channels,
            });
        }
        let chunk = format.duration_to_frames(shared.period).max(1);
        let ring = lane_format(format, request.lanes);
        let ring = Arc::new(Shared::mixed(ring, request.ring_frames(&format, chunk)));
        let link = Arc::new(Link {
            id: shared.ids.fetch_add(1, Ordering::Relaxed),
            shared: ring.clone(),
            running: AtomicBool::new(false),
            paused: AtomicBool::new(false),
        });
        // What the device's ring holds of it, and the device's own buffer.
        let sent = shared.buffer.as_secs_f64() / shared.period.as_secs_f64().max(1e-3);
        let sent = sent.ceil() as usize + 2;
        let feed = Feed::new(link.clone(), request.lanes, chunk as usize, sent);
        match ask(tx, |done| Msg::Open { slot, feed, done })?? {
            Opened::Yes { backend, realtime } => {
                let info = StreamInfo {
                    format,
                    backend,
                    device_buffer_frames: chunk,
                    ring_frames: ring.ring.capacity(),
                    realtime,
                };
                let driver = InputDriver {
                    link,
                    tx: tx.clone(),
                };
                let stream = crate::Stream::new(ring, Box::new(driver), info, request);
                return Ok(Box::new(stream));
            }
            Opened::Desk(desk) => format = desk,
        }
    }
}

/// The "device" of a stream: the mixer.
pub(crate) struct InputDriver {
    pub(crate) link: Arc<Link>,
    pub(crate) tx: Sender<Msg>,
}

impl Driver for InputDriver {
    fn start(&mut self) -> Result<(), Error> {
        self.link.paused.store(false, Ordering::SeqCst);
        if !self.link.running.swap(true, Ordering::SeqCst) {
            // A mixer with nothing running waits for word.
            let _ = self.tx.send(Msg::Wake);
        }
        Ok(())
    }

    fn pause(&mut self) -> Result<(), Error> {
        self.link.running.store(false, Ordering::SeqCst);
        self.link.paused.store(true, Ordering::SeqCst);
        // Its gain settles (nothing of it is heard while paused).
        let _ = self.tx.send(Msg::Wake);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.link.running.store(false, Ordering::SeqCst);
        self.link.shared.discard_all();
        // What the device holds of it goes too, unless another input is
        // heard there.
        let id = self.link.id;
        ask(&self.tx, |done| Msg::Flush { feed: id, done })?
    }

    fn close(&mut self) -> Result<(), Error> {
        self.link.running.store(false, Ordering::SeqCst);
        // A mixer already gone has closed it.
        let _ = self.tx.send(Msg::Closed(self.link.id));
        Ok(())
    }

    /// The device's tap is the microphone's port
    /// ([`super::Mixer::set_mic`]).
    fn runs_taps(&self) -> bool {
        false
    }
}

/// How many chunks a [`FrameMap`] remembers (seconds of them).
const MAP_ENTRIES: usize = 1024;

/// Where the Lead stream's frames went on the device, chunk by chunk: the
/// device frame a chunk starts at, the Lead's stream frame there, and how
/// many of the Lead's frames follow (the rest of the chunk had none). The
/// mixer thread adds entries before it writes the chunk; the device thread
/// reads them ([`MappedTap`]).
pub(crate) struct FrameMap {
    entries: Box<[[AtomicU64; 3]]>,
    /// Entries ever added.
    head: AtomicU64,
}

impl FrameMap {
    pub(crate) fn new() -> FrameMap {
        FrameMap {
            entries: (0..MAP_ENTRIES)
                .map(|_| [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)])
                .collect(),
            head: AtomicU64::new(0),
        }
    }

    /// The mixer: device frame `device` is the Lead's stream frame `lead`,
    /// and `frames` of the Lead's follow.
    pub(crate) fn push(&self, device: u64, lead: u64, frames: u64) {
        let head = self.head.load(Ordering::Relaxed);
        let e = &self.entries[head as usize % MAP_ENTRIES];
        e[0].store(device, Ordering::Relaxed);
        e[1].store(lead, Ordering::Relaxed);
        e[2].store(frames, Ordering::Relaxed);
        self.head.store(head + 1, Ordering::Release);
    }

    /// The entry added next.
    pub(crate) fn head(&self) -> u64 {
        self.head.load(Ordering::Acquire)
    }

    fn get(&self, k: u64) -> (u64, u64, u64) {
        let e = &self.entries[k as usize % MAP_ENTRIES];
        (
            e[0].load(Ordering::Relaxed),
            e[1].load(Ordering::Relaxed),
            e[2].load(Ordering::Relaxed),
        )
    }
}

/// The microphone's tap on the device: it sees what the device plays, with
/// the stream frames told in the Lead's frames (as the tap would have on
/// the Lead's own output).
pub(crate) struct MappedTap {
    inner: Box<dyn Tap>,
    map: Arc<FrameMap>,
    /// The next entry to look at, and the last one at or before the
    /// device frame being played.
    next: u64,
    at: (u64, u64, u64),
}

impl MappedTap {
    /// `inner` told the Lead's frames of `map` from its entry `next` on
    /// (what came before is another Lead's, or none).
    pub(crate) fn new(inner: Box<dyn Tap>, map: Arc<FrameMap>, next: u64) -> MappedTap {
        MappedTap {
            inner,
            map,
            next,
            at: (0, 0, 0),
        }
    }
}

impl Tap for MappedTap {
    fn begin(
        &mut self,
        frames: usize,
        stream_frame: u64,
        stream_frames: usize,
        format: StreamFormat,
    ) {
        let head = self.map.head.load(Ordering::Acquire);
        while self.next < head {
            let entry = self.map.get(self.next);
            if entry.0 > stream_frame {
                break;
            }
            self.at = entry;
            self.next += 1;
        }
        let (device, lead, n) = self.at;
        let into = stream_frame.saturating_sub(device).min(n);
        let first = lead + into;
        // The Lead's frames from there, on through the entries that follow
        // without a gap.
        let mut count = n - into;
        let (mut end_device, mut end_lead) = (device + n, lead + n);
        let want = stream_frames.min(frames) as u64;
        let mut k = self.next;
        while count < want && k < head {
            let (d, l, m) = self.map.get(k);
            if d != end_device || l != end_lead {
                break;
            }
            count += m;
            end_device += m;
            end_lead += m;
            k += 1;
        }
        self.inner
            .begin(frames, first, count.min(want) as usize, format);
    }

    fn see(&mut self, samples: &[i32], format: StreamFormat) {
        self.inner.see(samples, format);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SampleFormat;

    #[derive(Default)]
    struct Begins(Arc<std::sync::Mutex<Vec<(u64, usize)>>>);

    impl Tap for Begins {
        fn begin(&mut self, _: usize, stream_frame: u64, stream_frames: usize, _: StreamFormat) {
            self.0.lock().unwrap().push((stream_frame, stream_frames));
        }
        fn see(&mut self, _: &[i32], _: StreamFormat) {}
    }

    #[test]
    fn device_frames_map_to_the_leads() {
        let format = StreamFormat::new(48_000, 2, SampleFormat::S32);
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let map = Arc::new(FrameMap::new());
        // Another Lead's frames before: not this tap's.
        map.push(0, 5_000, 300);
        let mut tap = MappedTap::new(Box::new(Begins(seen.clone())), map.clone(), map.head());
        // The Lead starts at device frame 300 with its frame 0; three full
        // chunks, a starved one (40 of 100), and on after a gap.
        map.push(300, 0, 100);
        map.push(400, 100, 100);
        map.push(500, 200, 100);
        map.push(600, 300, 40);
        map.push(800, 340, 100);
        for (frame, n) in [
            (0, 0),
            (250, 0),
            (350, 150),
            (550, 150),
            (620, 60),
            (700, 0),
            (810, 90),
        ] {
            tap.begin(150, frame, n, format);
        }
        assert_eq!(
            *seen.lock().unwrap(),
            [
                (0, 0),
                (0, 0),
                (50, 150),
                (250, 90),
                (320, 20),
                (340, 0),
                (350, 90)
            ]
        );
    }
}
