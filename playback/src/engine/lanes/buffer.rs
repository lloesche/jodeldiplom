//! The stem parts made and not yet dropped ([`Buffer`]): [`crate::stems::STEMS`]
//! × the track's channels a frame, from an output frame on, one stretch
//! with no gaps.

use std::collections::VecDeque;

use crate::engine::track::fade_weight;

/// Where parts made again (a new start of them) meet the ones there: over
/// `[from, until)` they replace what is there, the first `fade` frames
/// crossfaded from it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Blend {
    from: u64,
    fade: usize,
    until: u64,
}

#[derive(Debug, Default)]
pub(super) struct Buffer {
    parts: VecDeque<i32>,
    at: u64,
    width: usize,
    blend: Option<Blend>,
}

impl Buffer {
    /// An empty buffer of `width` samples a frame.
    pub(super) fn new(width: usize) -> Buffer {
        Buffer {
            width,
            ..Buffer::default()
        }
    }

    pub(super) fn width(&self) -> usize {
        self.width
    }

    /// Whether nothing is there.
    pub(super) fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// The output frame after the last one there.
    pub(super) fn end(&self) -> u64 {
        self.at + (self.parts.len() / self.width.max(1)) as u64
    }

    /// The end of the parts there from output frame `from` on (`from`
    /// itself: none).
    pub(super) fn there(&self, from: u64) -> u64 {
        if self.at <= from && self.end() > from {
            self.end()
        } else {
            from
        }
    }

    /// Nothing there; what comes next starts at output frame `at`.
    pub(super) fn clear(&mut self, at: u64) {
        self.parts.clear();
        self.at = at;
        self.blend = None;
    }

    /// Drops what is before output frame `before`.
    pub(super) fn trim(&mut self, before: u64) {
        let drop = before.saturating_sub(self.at).min(self.end() - self.at);
        if drop > 0 {
            self.parts.drain(..drop as usize * self.width);
            self.at += drop;
        }
    }

    /// Keeps what is there before output frame `keep` (no further than it
    /// goes); parts made from `from` on replace it up to there, their first
    /// `fade` frames crossfaded from it.
    pub(super) fn restart(&mut self, from: u64, keep: u64, fade: usize) {
        if keep <= self.at || self.parts.is_empty() {
            self.clear(from);
            return;
        }
        if keep < self.end() {
            self.parts.truncate((keep - self.at) as usize * self.width);
        }
        let until = self.end();
        self.blend = (until > from).then_some(Blend { from, fade, until });
    }

    /// Takes parts made for output frames from `frame` on (frames before
    /// the output's first are dropped): what is before the end is dropped,
    /// or put in place of what is there where a new start of them meets
    /// it; the rest goes on the end. Past a gap, what is there goes.
    pub(super) fn push(&mut self, frame: i64, samples: &[i32]) {
        let w = self.width;
        let mut n = (samples.len() / w) as i64;
        let mut frame = frame;
        let mut samples = samples;
        if frame < 0 {
            let skip = (-frame).min(n);
            samples = &samples[skip as usize * w..];
            n -= skip;
            frame = 0;
        }
        if n == 0 {
            return;
        }
        let (frame, n) = (frame as u64, n as u64);
        if frame > self.end() {
            self.clear(frame);
        }
        let end = self.end();
        let over = (end - frame).min(n);
        if let Some(b) = self.blend {
            for i in 0..over {
                let o = frame + i;
                // (What was dropped behind meanwhile is gone.)
                if o < b.from.max(self.at) || o >= b.until {
                    continue;
                }
                let k = (o - b.from) as usize;
                let k = if k < b.fade {
                    fade_weight(k, b.fade)
                } else {
                    1.0
                };
                let at = (o - self.at) as usize * w;
                let new = &samples[i as usize * w..(i as usize + 1) * w];
                for (j, &new) in new.iter().enumerate() {
                    let old = &mut self.parts[at + j];
                    *old = (f64::from(*old) * (1.0 - k) + f64::from(new) * k).round() as i32;
                }
            }
            if frame + n >= b.until {
                self.blend = None;
            }
        }
        self.parts
            .extend(samples[over as usize * w..].iter().copied());
    }

    /// Appends frame `o`'s parts (there) to `out`.
    pub(super) fn copy(&self, o: u64, out: &mut Vec<i32>) {
        let at = (o - self.at) as usize * self.width;
        out.extend(self.parts.range(at..at + self.width));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(b: &Buffer) -> Vec<i32> {
        b.parts.iter().copied().collect()
    }

    #[test]
    fn a_buffer_keeps_one_run_of_frames() {
        let mut b = Buffer::new(1);
        b.push(10, &[1, 2, 3]);
        assert_eq!((b.at, b.end()), (10, 13));
        // Before the end: dropped.
        b.push(11, &[9, 9, 4]);
        assert_eq!(frames(&b), [1, 2, 3, 4]);
        assert_eq!((b.there(12), b.there(9), b.there(14)), (14, 9, 14));
        // Past a gap: what was there goes.
        b.push(20, &[5]);
        assert_eq!((b.at, frames(&b)), (20, vec![5]));
        // Before the output's first frame: dropped.
        let mut b = Buffer::new(1);
        b.push(-2, &[7, 7, 1, 2]);
        assert_eq!((b.at, frames(&b)), (0, vec![1, 2]));
        b.trim(1);
        assert_eq!((b.at, frames(&b)), (1, vec![2]));
    }

    #[test]
    fn a_new_start_crossfades_and_takes_the_place_of_what_was_kept() {
        let mut b = Buffer::new(1);
        b.push(0, &[1_000; 10]);
        // From 2, a fade of 4, what is there kept to 8.
        b.restart(2, 8, 4);
        assert_eq!(b.end(), 8);
        b.push(0, &[0; 12]);
        let got = frames(&b);
        assert_eq!(&got[..2], [1_000, 1_000]);
        assert!(got[2] > 900 && got[5] < 100, "{got:?}");
        assert!(got[2..6].windows(2).all(|w| w[1] <= w[0]));
        assert_eq!(&got[6..], [0; 6]);
        assert_eq!(b.blend, None);
        // Kept to before anything there: from there on.
        let mut b = Buffer::new(1);
        b.push(0, &[1; 4]);
        b.restart(4, 4, 0);
        b.push(3, &[2, 3]);
        assert_eq!(frames(&b), [1, 1, 1, 1, 3]);
    }
}
