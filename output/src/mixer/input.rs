//! A mixer input's streams: full [`Output`]s whose ring the mixer drains,
//! and the map that keeps the microphone's tap on the device in the Lead
//! stream's frames.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;

use super::Msg;
use crate::error::Error;
use crate::format::StreamFormat;
use crate::output::{forward_output, Core, Driver};
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
pub(crate) struct Stream {
    pub(crate) link: Arc<Link>,
    /// The lanes of each frame (1: a plain stream; see [`crate::lanes`]).
    pub(crate) lanes: u16,
    /// This chunk: taken from (it was running), and the frames it gave.
    pub(crate) taken: bool,
    pub(crate) got: u64,
    bytes: Vec<u8>,
    pub(crate) samples: Vec<i32>,
    /// Its chunks on the device not all played yet: the device frame each
    /// starts at and how many of its frames it has.
    sent: VecDeque<(u64, u64)>,
    /// The timeline epoch (restarts by flush) of what was taken last.
    epoch: u64,
}

impl Stream {
    pub(crate) fn new(link: Arc<Link>, lanes: u16) -> Stream {
        Stream {
            link,
            lanes,
            taken: false,
            got: 0,
            bytes: Vec::new(),
            samples: Vec::new(),
            sent: VecDeque::new(),
            epoch: 0,
        }
    }

    /// Takes the chunk of `n` frames that goes to the device from device
    /// frame `first` into `samples`, in its ring's format (every lane of a
    /// lane stream; silence for what it lacks, or all of it when it is not
    /// running or a flush keeps the mixer out). Returns, when it was
    /// rendered, its stream frame there and the frames it gave.
    pub(crate) fn take(&mut self, n: usize, first: u64) -> Option<(u64, u64)> {
        let format = self.link.shared.format;
        self.taken = self.link.running.load(Ordering::SeqCst);
        self.got = 0;
        if !self.taken {
            return None;
        }
        self.bytes.resize(n * format.frame_bytes(), 0);
        let rendered = self.link.shared.render_input(&mut self.bytes);
        if let Some((_, got, epoch)) = rendered {
            if epoch != self.epoch {
                self.sent.clear();
                self.epoch = epoch;
            }
            if got > 0 {
                self.sent.push_back((first, got));
            }
            self.got = got;
        }
        self.samples.resize(n * usize::from(format.channels), 0);
        let sb = format.sample.bytes();
        for (s, b) in self.samples.iter_mut().zip(self.bytes.chunks_exact(sb)) {
            *s = unpack_sample(b, format.sample);
        }
        rendered.map(|(lead, got, _)| (lead, got))
    }

    /// The device has played up to device frame `played`: the output is
    /// told what of it is still downstream.
    pub(crate) fn played(&mut self, played: u64) {
        while self.sent.front().is_some_and(|&(d, n)| d + n <= played) {
            self.sent.pop_front();
        }
        let unplayed: u64 = self
            .sent
            .iter()
            .map(|&(d, n)| n - played.saturating_sub(d).min(n))
            .sum();
        self.link.shared.set_downstream(self.epoch, unplayed);
    }
}

/// Asks the mixer thread and waits for its answer (`Closed` when it is
/// gone).
pub(crate) fn ask<T>(tx: &Sender<Msg>, msg: impl FnOnce(Sender<T>) -> Msg) -> Result<T, Error> {
    let (reply, answer) = mpsc::channel();
    tx.send(msg(reply)).map_err(|_| Error::Closed)?;
    answer.recv().map_err(|_| Error::Closed)
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
        ask(&self.tx, |done| Msg::Flush { stream: id, done })?
    }

    fn close(&mut self) -> Result<(), Error> {
        self.link.running.store(false, Ordering::SeqCst);
        // A mixer already gone has closed it.
        let _ = self.tx.send(Msg::Closed(self.link.id));
        Ok(())
    }
}

/// A stream of a mixer input, as the engine writing it sees it: an
/// [`Output`] in the device's format. It runs no taps: the device's tap is
/// the microphone's port ([`super::Mixer::set_mic`]).
pub(crate) struct InputOutput {
    pub(crate) core: Core,
}

impl crate::output::Output for InputOutput {
    forward_output!(core);
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
    pub(crate) fn new(inner: Box<dyn Tap>, map: Arc<FrameMap>) -> MappedTap {
        let next = map.head.load(Ordering::Acquire);
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
        let mut tap = MappedTap::new(Box::new(Begins(seen.clone())), map.clone());
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
