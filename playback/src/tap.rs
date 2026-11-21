//! The spectrum tap: the most recent frames the device plays, tagged with
//! their device frame index, for a UI to analyse what is heard now.
//!
//! Set as a mixer's monitor ([`jodel_output::mixer::Mixer::set_monitor`]),
//! it keeps every frame the mixer gives the device (every player on it,
//! and their fades) in a ring of [`SpectrumTap::capacity`] stereo frames.
//! Each frame is tagged by its index in the device stream (the same count
//! as the device's [`jodel_output::Output::position`]), and each opening of
//! the device is a new *generation*. A reader asks for the frames that end
//! at a given index, usually the one being heard
//! ([`SpectrumTap::read_heard`]), and computes its own spectrum.
//!
//! The ring is lock-free for the writer and for readers: one writer stores
//! frames and then publishes the new end; readers copy and then check that
//! the writer did not overwrite what they copied (a sequence check). A read
//! that raced the writer reports [`TapMiss::Overwritten`] and can simply be
//! retried. Only the output clock, which changes when an output opens, sits
//! behind a mutex that the audio path never takes while writing.

use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use jodel_output::mixer::Monitor;
use jodel_output::{Clock, StreamInfo};

/// One stereo frame: `[left, right]`, 24-bit values in `i32` words. Mono
/// output is tapped on both sides; outputs with more channels tap the first
/// two.
pub type TapFrame = [i32; 2];

/// What a successful read returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapWindow {
    /// The output stream the frames belong to.
    pub generation: u64,
    /// How many frames at the front are silence because they lie before
    /// the start of the stream.
    pub silence: usize,
    /// The output frame index of the first frame after that silence.
    pub start: u64,
    /// The output rate.
    pub rate: u32,
}

/// Why a read returned nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapMiss {
    /// No output is open, or the asked-for generation is not the current one.
    NoStream,
    /// The frames asked for have not been written yet.
    NotYet,
    /// The frames asked for are older than the ring holds, or the writer
    /// overwrote them during the read. Retry with a later index.
    Overwritten,
}

/// The output clock as the tap sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapCursor {
    /// The output stream.
    pub generation: u64,
    /// The output frame being heard.
    pub frame: u64,
    /// The output rate.
    pub rate: u32,
}

struct Shared {
    frames: Box<[AtomicU64]>,
    /// The end of the chunk the writer is storing (published before the
    /// samples), or `u64::MAX` while a new stream starts.
    begun: AtomicU64,
    /// The end of the frames written (published after the samples).
    end: AtomicU64,
    /// Frames before this index are not in the ring (a gap in the indices).
    valid_from: AtomicU64,
    generation: AtomicU64,
    rate: AtomicU32,
    clock: Mutex<Option<(u64, u32, Clock)>>,
}

/// The read side of the tap: cheap to clone and to share between threads.
#[derive(Clone)]
pub struct SpectrumTap {
    shared: Arc<Shared>,
}

fn pack(frame: TapFrame) -> u64 {
    (u64::from(frame[0] as u32) << 32) | u64::from(frame[1] as u32)
}

fn unpack(word: u64) -> TapFrame {
    [(word >> 32) as u32 as i32, word as u32 as i32]
}

impl SpectrumTap {
    /// A tap that keeps the last `capacity` frames.
    pub fn new(capacity: usize) -> SpectrumTap {
        let capacity = capacity.max(1);
        SpectrumTap {
            shared: Arc::new(Shared {
                frames: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
                begun: AtomicU64::new(0),
                end: AtomicU64::new(0),
                valid_from: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                rate: AtomicU32::new(0),
                clock: Mutex::new(None),
            }),
        }
    }

    /// How many frames the ring holds.
    pub fn capacity(&self) -> usize {
        self.shared.frames.len()
    }

    /// The current output stream (0: none has been opened yet).
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    /// The output frame index after the last frame written.
    pub fn written(&self) -> u64 {
        self.shared.end.load(Ordering::Acquire)
    }

    /// The frame being heard now, from the open output's clock.
    pub fn heard(&self) -> Option<TapCursor> {
        let guard = self.shared.clock.lock().unwrap_or_else(|e| e.into_inner());
        let (generation, rate, clock) = guard.as_ref()?;
        Some(TapCursor {
            generation: *generation,
            frame: clock.position(),
            rate: *rate,
        })
    }

    /// Copies the `out.len()` frames that end just before output frame
    /// `end` of stream `generation` into `out`. Frames before the start of
    /// the stream read as silence.
    pub fn read(
        &self,
        generation: u64,
        end: u64,
        out: &mut [TapFrame],
    ) -> Result<TapWindow, TapMiss> {
        let s = &self.shared;
        let cap = s.frames.len() as u64;
        let n = out.len() as u64;
        if generation == 0 || s.generation.load(Ordering::Acquire) != generation {
            return Err(TapMiss::NoStream);
        }
        let written = s.end.load(Ordering::Acquire);
        if end > written {
            return Err(TapMiss::NotYet);
        }
        let start = end.saturating_sub(n);
        let lead = (n - (end - start)) as usize;
        if n > cap || start < s.valid_from.load(Ordering::Acquire) || start + cap < written {
            return Err(TapMiss::Overwritten);
        }
        out[..lead].fill([0, 0]);
        for (slot, index) in out[lead..].iter_mut().zip(start..end) {
            *slot = unpack(s.frames[(index % cap) as usize].load(Ordering::Relaxed));
        }
        fence(Ordering::Acquire);
        let begun = s.begun.load(Ordering::Relaxed);
        if s.generation.load(Ordering::Relaxed) != generation
            || begun == u64::MAX
            || start + cap < begun
        {
            return Err(TapMiss::Overwritten);
        }
        Ok(TapWindow {
            generation,
            silence: lead,
            start,
            rate: s.rate.load(Ordering::Relaxed),
        })
    }

    /// Copies the `out.len()` frames that end at the frame being heard now.
    pub fn read_heard(&self, out: &mut [TapFrame]) -> Result<TapWindow, TapMiss> {
        let cursor = self.heard().ok_or(TapMiss::NoStream)?;
        // The clock may run a little ahead of what the tap has published.
        let end = cursor.frame.min(self.written());
        self.read(cursor.generation, end, out)
    }

    /// The writer: starts a new output stream.
    fn begin(&self, rate: u32, clock: Clock) {
        let s = &self.shared;
        let mut guard = s.clock.lock().unwrap_or_else(|e| e.into_inner());
        s.begun.store(u64::MAX, Ordering::Relaxed);
        fence(Ordering::Release);
        let generation = s.generation.load(Ordering::Relaxed) + 1;
        s.end.store(0, Ordering::Relaxed);
        s.valid_from.store(0, Ordering::Relaxed);
        s.rate.store(rate, Ordering::Relaxed);
        s.generation.store(generation, Ordering::Release);
        s.begun.store(0, Ordering::Release);
        *guard = Some((generation, rate, clock));
    }

    /// The writer: the output closed.
    fn end_stream(&self) {
        let mut guard = self.shared.clock.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    /// The writer: `samples` (interleaved, `channels` per frame) were
    /// handed to the output starting at output frame `start`.
    fn push(&self, samples: &[i32], channels: usize, start: u64) {
        let s = &self.shared;
        let cap = s.frames.len();
        if channels == 0 || samples.is_empty() {
            return;
        }
        let frames = samples.len() / channels;
        let end = start + frames as u64;
        if start != s.end.load(Ordering::Relaxed) {
            s.valid_from.store(start, Ordering::Relaxed);
        }
        // Only the last `cap` frames can be kept; store in chunks so readers
        // of older frames are invalidated only as far as needed.
        let skip = frames.saturating_sub(cap);
        let mut index = start + skip as u64;
        for chunk in samples[skip * channels..].chunks(channels * (cap / 4).max(1)) {
            let chunk_end = index + (chunk.len() / channels) as u64;
            s.begun.store(chunk_end, Ordering::Relaxed);
            fence(Ordering::Release);
            for frame in chunk.chunks_exact(channels) {
                let pair = if channels == 1 {
                    [frame[0], frame[0]]
                } else {
                    [frame[0], frame[1]]
                };
                s.frames[(index % cap as u64) as usize].store(pack(pair), Ordering::Relaxed);
                index += 1;
            }
            s.end.store(chunk_end, Ordering::Release);
        }
        debug_assert_eq!(index, end);
    }
}

/// A mixer's monitor: the tap holds what the device plays, the mix.
impl Monitor for SpectrumTap {
    fn opened(&mut self, info: &StreamInfo, clock: Clock) {
        self.begin(info.format.rate, clock);
    }

    fn mixed(&mut self, samples: &[i32], channels: usize, first_frame: u64) {
        self.push(samples, channels, first_frame);
    }

    fn closed(&mut self) {
        self.end_stream();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_output::{FileOutput, Output, SampleFormat, SimClock, StreamFormat};

    fn clock() -> Clock {
        let (out, _) = FileOutput::to_memory(
            StreamFormat::new(48_000, 2, SampleFormat::S32),
            SimClock::Instant,
        );
        out.clock()
    }

    fn ramp(start: i32, frames: usize) -> Vec<i32> {
        (0..frames as i32)
            .flat_map(|i| [start + i, -(start + i)])
            .collect()
    }

    #[test]
    fn reads_the_window_ending_at_an_index() {
        let tap = SpectrumTap::new(16);
        let mut out = [[0; 2]; 4];
        assert_eq!(tap.read(1, 4, &mut out), Err(TapMiss::NoStream));
        tap.begin(48_000, clock());
        let g = tap.generation();
        tap.push(&ramp(0, 10), 2, 0);
        assert_eq!(tap.read(g, 11, &mut out), Err(TapMiss::NotYet));
        let w = tap.read(g, 10, &mut out).unwrap();
        assert_eq!(
            w,
            TapWindow {
                generation: g,
                silence: 0,
                start: 6,
                rate: 48_000
            }
        );
        assert_eq!(out, [[6, -6], [7, -7], [8, -8], [9, -9]]);
        // Before the stream start: silence.
        let w = tap.read(g, 2, &mut out).unwrap();
        assert_eq!((w.silence, w.start), (2, 0));
        assert_eq!(out, [[0, 0], [0, 0], [0, 0], [1, -1]]);
        // Wrap around, and frames older than the ring.
        tap.push(&ramp(10, 20), 2, 10);
        let w = tap.read(g, 30, &mut out).unwrap();
        assert_eq!(w.start, 26);
        assert_eq!(out[3], [29, -29]);
        assert_eq!(tap.read(g, 12, &mut out), Err(TapMiss::Overwritten));
        let mut big = vec![[0; 2]; 17];
        assert_eq!(tap.read(g, 30, &mut big), Err(TapMiss::Overwritten));
    }

    #[test]
    fn mono_and_wide_outputs_tap_two_channels() {
        let tap = SpectrumTap::new(8);
        tap.begin(44_100, clock());
        let g = tap.generation();
        tap.push(&[1, 2, 3], 1, 0);
        let mut out = [[0; 2]; 3];
        tap.read(g, 3, &mut out).unwrap();
        assert_eq!(out, [[1, 1], [2, 2], [3, 3]]);
        tap.push(&[4, 5, 6, 7, 8, 9], 3, 3);
        let mut out = [[0; 2]; 2];
        tap.read(g, 5, &mut out).unwrap();
        assert_eq!(out, [[4, 5], [7, 8]]);
    }

    #[test]
    fn a_new_stream_or_a_gap_invalidates_old_frames() {
        let tap = SpectrumTap::new(8);
        tap.begin(44_100, clock());
        let g = tap.generation();
        tap.push(&ramp(0, 4), 2, 0);
        tap.push(&ramp(100, 2), 2, 6);
        let mut out = [[0; 2]; 2];
        assert_eq!(tap.read(g, 8, &mut out).unwrap().start, 6);
        assert_eq!(tap.read(g, 6, &mut out), Err(TapMiss::Overwritten));
        tap.begin(96_000, clock());
        assert_eq!(tap.read(g, 8, &mut out), Err(TapMiss::NoStream));
        assert_eq!(tap.generation(), g + 1);
        let cursor = tap.heard().unwrap();
        assert_eq!((cursor.generation, cursor.rate), (g + 1, 96_000));
        tap.end_stream();
        assert_eq!(tap.heard(), None);
    }

    #[test]
    fn concurrent_reads_never_return_torn_frames() {
        let tap = SpectrumTap::new(256);
        tap.begin(48_000, clock());
        let g = tap.generation();
        let reader = tap.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_reader = stop.clone();
        let reads = Arc::new(AtomicU32::new(0));
        let reads_seen = reads.clone();
        let handle = std::thread::spawn(move || {
            let mut out = [[0; 2]; 64];
            let mut good = 0u32;
            while !stop_reader.load(Ordering::Relaxed) {
                let end = reader.written();
                if end < 64 {
                    continue;
                }
                if let Ok(w) = reader.read(g, end, &mut out) {
                    assert_eq!(w.silence, 0);
                    for (i, frame) in out.iter().enumerate() {
                        assert_eq!(frame[0] as u64, w.start + i as u64);
                        assert_eq!(frame[1], -frame[0]);
                    }
                    good += 1;
                    reads.store(good, Ordering::Relaxed);
                }
            }
            good
        });
        // On a busy machine the reader may not have been scheduled yet:
        // after the first 400,000 frames, go on (yielding) until it has
        // had a good read.
        let mut at = 0u64;
        while at < 400_000 || (reads_seen.load(Ordering::Relaxed) == 0 && at < 100_000_000) {
            tap.push(&ramp(at as i32, 100), 2, at);
            at += 100;
            if at > 400_000 {
                std::thread::yield_now();
            }
        }
        stop.store(true, Ordering::Relaxed);
        assert!(handle.join().unwrap() > 0);
    }
}
