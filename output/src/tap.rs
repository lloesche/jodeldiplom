//! Taps: code the device thread runs on every buffer it hands the device,
//! after the stream's frames are copied out of the ring, that sees what
//! plays and cannot change it.
//!
//! A tap gets the buffer unpacked to 24-bit values in a scratch of its own;
//! the buffer itself is never packed again, so the device gets exactly the
//! bytes written, tap or not.
//!
//! The device thread never locks for it. [`Output::set_tap`] hands the new
//! tap over through an atomic slot; the device thread takes it at the start
//! of its next buffer and hands the old one back in the same slot, so it is
//! dropped on the caller's thread, never on the device's.
//!
//! [`Output::set_tap`]: crate::Output::set_tap

use crate::format::StreamFormat;
use crate::pack::unpack_sample;
use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Code run on the device thread on every buffer it hands the device, that
/// sees what plays: a microphone's echo canceller taking its reference
/// ([`crate::input::MicTap`]), or a test.
///
/// It runs where the device's deadline is, so it must not block, lock,
/// allocate or do I/O.
pub trait Tap: Send {
    /// Called once per buffer the device thread makes, before its
    /// [`Tap::see`] calls: its length in frames, the stream frames copied
    /// out before it (the stream's frame its first frame is, as
    /// [`crate::Output::position`] counts them), and how many of its frames
    /// are the stream's (the rest is silence padding it).
    fn begin(
        &mut self,
        frames: usize,
        stream_frame: u64,
        stream_frames: usize,
        format: StreamFormat,
    ) {
        let _ = (frames, stream_frame, stream_frames, format);
    }

    /// Sees the buffer: interleaved frames of `format.channels` signed
    /// 24-bit values in `i32` words (see [`crate::pack`]), what the device
    /// is about to play, padding included. A device buffer comes in one or
    /// more calls of at most [`TAP_CHUNK`] frames, in order.
    fn see(&mut self, samples: &[i32], format: StreamFormat);
}

/// The most frames a [`Tap`] gets in one call.
pub const TAP_CHUNK: usize = 1024;

/// A tap with the scratch the device thread unpacks into.
pub(crate) struct Installed {
    tap: Box<dyn Tap>,
    scratch: Vec<i32>,
}

impl Installed {
    fn new(tap: Box<dyn Tap>, channels: u16) -> Installed {
        Installed {
            tap,
            scratch: vec![0; TAP_CHUNK * usize::from(channels).max(1)],
        }
    }

    /// Unpacks `out` into the scratch, chunk by chunk, for the tap to see;
    /// `out` is only read.
    fn run(&mut self, out: &[u8], stream_frame: u64, stream_frames: usize, format: StreamFormat) {
        let sb = format.sample.bytes();
        let ch = usize::from(format.channels);
        self.tap.begin(
            out.len() / format.frame_bytes(),
            stream_frame,
            stream_frames,
            format,
        );
        for chunk in out.chunks(TAP_CHUNK * format.frame_bytes()) {
            let n = chunk.len() / sb / ch * ch;
            let samples = &mut self.scratch[..n];
            for (s, b) in samples.iter_mut().zip(chunk.chunks_exact(sb)) {
                *s = unpack_sample(b, format.sample);
            }
            self.tap.see(samples, format);
        }
    }
}

type Node = Option<Installed>;

/// Where the handle hands a tap to the device thread.
pub(crate) struct TapSlot {
    /// The tap in use. Only the consumer touches it, inside its render (or
    /// the handle, while the consumer is suspended).
    current: UnsafeCell<Node>,
    /// A boxed node the handle posted and the consumer has not taken yet.
    pending: AtomicPtr<Node>,
    /// The same node, back from the consumer with the tap it replaced.
    retired: AtomicPtr<Node>,
}

// SAFETY: `current` is only reached by one side at a time (the consumer
// inside its render, the handle only while the consumer is suspended);
// `pending` and `retired` move whole boxes between the sides atomically.
unsafe impl Sync for TapSlot {}

impl TapSlot {
    pub(crate) fn new() -> TapSlot {
        TapSlot {
            current: UnsafeCell::new(None),
            pending: AtomicPtr::new(ptr::null_mut()),
            retired: AtomicPtr::new(ptr::null_mut()),
        }
    }

    // ---- consumer side ----

    /// Takes a posted tap and shows the current one `out`.
    ///
    /// # Safety
    ///
    /// Only the consumer calls it, inside its render (see [`Self::current`]).
    pub(crate) unsafe fn run(
        &self,
        out: &[u8],
        stream_frame: u64,
        stream_frames: usize,
        format: StreamFormat,
    ) {
        if !self.pending.load(Ordering::Relaxed).is_null() {
            let node = self.pending.swap(ptr::null_mut(), Ordering::AcqRel);
            if !node.is_null() {
                // The old tap goes back in the node the new one came in.
                ptr::swap(self.current.get(), node);
                self.retired.store(node, Ordering::Release);
            }
        }
        if let Some(installed) = (*self.current.get()).as_mut() {
            installed.run(out, stream_frame, stream_frames, format);
        }
    }

    // ---- handle side ----

    /// Posts `tap` for the consumer to take; see [`Self::collect`].
    pub(crate) fn post(&self, tap: Option<Box<dyn Tap>>, channels: u16) {
        let node = Box::into_raw(Box::new(tap.map(|t| Installed::new(t, channels))));
        let old = self.pending.swap(node, Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: a node the consumer never took is still ours.
            drop(unsafe { Box::from_raw(old) });
        }
    }

    /// Whether the consumer has taken the posted node; drops what it gave
    /// back.
    pub(crate) fn collect(&self) -> bool {
        let node = self.retired.swap(ptr::null_mut(), Ordering::AcqRel);
        if node.is_null() {
            return false;
        }
        // SAFETY: the consumer stored it and no longer touches it.
        drop(unsafe { Box::from_raw(node) });
        true
    }

    /// Takes the posted node back if the consumer has not taken it.
    pub(crate) fn unpost(&self) -> Option<Box<Node>> {
        let node = self.pending.swap(ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: a node the consumer never took is still ours.
        (!node.is_null()).then(|| unsafe { Box::from_raw(node) })
    }

    /// Installs `node` directly and returns the tap it replaced.
    ///
    /// # Safety
    ///
    /// The consumer must be kept out of its render (suspended) meanwhile.
    pub(crate) unsafe fn install(&self, mut node: Box<Node>) -> Box<Node> {
        ptr::swap(self.current.get(), &mut *node);
        node
    }
}

impl Drop for TapSlot {
    fn drop(&mut self) {
        for p in [&self.pending, &self.retired] {
            let node = p.swap(ptr::null_mut(), Ordering::AcqRel);
            if !node.is_null() {
                // SAFETY: nothing else holds the slot any more.
                drop(unsafe { Box::from_raw(node) });
            }
        }
    }
}
