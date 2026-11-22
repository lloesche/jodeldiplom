//! An input's gain on the mixer's clock: a level, changed by ramps that
//! start when the mixer says ([`Start`], resolved to a frame as it mixes).

use std::f64::consts::FRAC_PI_2;
use std::time::Duration;

/// How a [`Ramp`] goes from one gain to the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Curve {
    /// Equal power: a rise follows a quarter sine, a fall a quarter cosine,
    /// so two inputs crossfading with opposite ramps keep their summed
    /// power.
    EqualPower,
    /// A straight line.
    Linear,
}

/// When a [`Ramp`] starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Start {
    /// The next frame mixed.
    Now,
    /// `time` into the clock of device epoch `epoch` (see
    /// [`crate::mixer::Timebase`]); the next frame mixed once that is past,
    /// and never if the device has opened again since.
    At {
        /// The epoch.
        epoch: u64,
        /// The time from the epoch's first frame.
        time: Duration,
    },
    /// The first frame supplied by an output opened on the input after the
    /// ramp is set: a fade in from the first audio of what is loaded next,
    /// whatever an output open before still supplies.
    FirstAudio,
}

/// A change of an input's gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    /// When it starts.
    pub start: Start,
    /// The gain it ends at (1: as written).
    pub to: f64,
    /// How long it takes (rounded to frames at the device's rate); zero:
    /// at once.
    pub over: Duration,
    /// Its shape.
    pub curve: Curve,
}

impl Ramp {
    /// The gain `to`, at once.
    pub fn set(to: f64) -> Ramp {
        Ramp {
            start: Start::Now,
            to,
            over: Duration::ZERO,
            curve: Curve::Linear,
        }
    }

    /// An equal-power fade to `to` over `over` from `start`.
    pub fn fade(to: f64, over: Duration, start: Start) -> Ramp {
        Ramp {
            start,
            to,
            over,
            curve: Curve::EqualPower,
        }
    }
}

/// When a waiting ramp starts, as the mixer keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum When {
    /// The next frame mixed.
    Now,
    /// A frame of the device's current epoch.
    Frame(u64),
    /// The first frame of an output with an id above `after` (outputs'
    /// ids grow as they are opened): one opened on the input since, or for
    /// a strip's change, on an input of its subtree.
    FirstAudio { after: u64 },
    /// The frame input `slot`'s waiting ramp starts at (a crossfade).
    With(u64),
}

impl When {
    /// The frame it comes at in the chunk of frames `first..end` (a frame
    /// gone by: `first`), if it comes there; `audio` is the newest output
    /// (as `FirstAudio` counts them) that supplied audio in the chunk. A
    /// `With` is left to the ramp it follows.
    pub(crate) fn due(self, first: u64, end: u64, audio: Option<u64>) -> Option<u64> {
        match self {
            When::Now => Some(first),
            When::Frame(f) if f < end => Some(f.max(first)),
            When::FirstAudio { after } if audio.is_some_and(|o| o > after) => Some(first),
            _ => None,
        }
    }
}

/// Where the waiting ramps of inputs start in the chunk of frames
/// `first..end`, into `starts` as (input, frame): their own frame, the
/// first audio of an output opened since, or with the ramp they follow.
/// `waiting` gives each input with a ramp waiting, when it starts, and the
/// id of its output if that supplied audio in the chunk.
pub(crate) fn resolve_starts(
    first: u64,
    end: u64,
    waiting: impl Iterator<Item = (u64, When, Option<u64>)> + Clone,
    starts: &mut Vec<(u64, u64)>,
) {
    starts.clear();
    for (id, when, audio) in waiting.clone() {
        if let Some(at) = when.due(first, end, audio) {
            starts.push((id, at));
        }
    }
    let led = starts.len();
    for (id, when, _) in waiting {
        if let When::With(lead) = when {
            if let Some(at) = start_of(&starts[..led], lead) {
                starts.push((id, at));
            }
        }
    }
}

/// The frame input `id`'s waiting ramp starts at, among `starts`.
pub(crate) fn start_of(starts: &[(u64, u64)], id: u64) -> Option<u64> {
    starts.iter().find(|(i, _)| *i == id).map(|&(_, at)| at)
}

/// A ramp waiting for its start.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Waiting {
    pub(crate) when: When,
    pub(crate) to: f64,
    pub(crate) over: Duration,
    pub(crate) curve: Curve,
}

/// A ramp under way.
#[derive(Clone, Copy, Debug)]
struct Running {
    start: u64,
    from: f64,
    to: f64,
    frames: u64,
    curve: Curve,
}

impl Running {
    /// The gain `k` frames in (`k < frames`).
    fn at(&self, k: u64) -> f64 {
        let x = k as f64 / self.frames as f64;
        let (from, to) = (self.from, self.to);
        match self.curve {
            Curve::Linear => from + (to - from) * x,
            Curve::EqualPower if to >= from => from + (to - from) * (x * FRAC_PI_2).sin(),
            Curve::EqualPower => to + (from - to) * (x * FRAC_PI_2).cos(),
        }
    }
}

/// An input's gain as the mixer runs it.
#[derive(Clone, Debug)]
pub(crate) struct Gain {
    /// The gain while no ramp runs.
    level: f64,
    running: Option<Running>,
    next: Option<Waiting>,
    /// The gain of the last frame filled.
    last: f64,
}

impl Gain {
    pub(crate) fn new(level: f64) -> Gain {
        Gain {
            level,
            running: None,
            next: None,
            last: level,
        }
    }

    /// `ramp` waits for its start, in place of one waiting already; one
    /// running goes on until then (the new one starts from where it is).
    /// One due now, of no length or to the gain it is at, is the gain at
    /// once.
    pub(crate) fn set(&mut self, ramp: Waiting) {
        let at = self.running.is_none() && self.level == ramp.to;
        if ramp.when == When::Now && (ramp.over.is_zero() || at) {
            self.next = None;
            self.running = None;
            self.level = ramp.to;
            self.last = ramp.to;
        } else {
            self.next = Some(ramp);
        }
    }

    /// When the waiting ramp starts.
    pub(crate) fn next(&self) -> Option<When> {
        self.next.map(|r| r.when)
    }

    /// Drops the waiting ramp.
    pub(crate) fn drop_next(&mut self) {
        self.next = None;
    }

    /// The gain of the last frame filled.
    pub(crate) fn last(&self) -> f64 {
        self.last
    }

    /// Whether the gain is changing or a ramp waits.
    pub(crate) fn moving(&self) -> bool {
        self.running.is_some() || self.next.is_some()
    }

    /// The gains of frames `first..first + out.len()` into `out` (at
    /// `rate`); the waiting ramp starts at `start` (a frame among them)
    /// when given. Returns the gain instead when it is the same for all of
    /// them (`out` is not written then).
    pub(crate) fn fill(
        &mut self,
        first: u64,
        start: Option<u64>,
        rate: u32,
        out: &mut [f64],
    ) -> Option<f64> {
        let end = first + out.len() as u64;
        if start.is_none() && self.running.is_none_or(|r| r.start + r.frames <= first) {
            self.finish();
            self.last = self.level;
            return Some(self.level);
        }
        for (frame, g) in (first..end).zip(out.iter_mut()) {
            if start == Some(frame) {
                self.start(frame, rate);
            }
            *g = self.gain_at(frame);
        }
        self.last = out.last().copied().unwrap_or(self.level);
        None
    }

    /// The device opened again, its frames from 0: a ramp under way ends
    /// at once (its frames were the last epoch's); one waiting stays.
    pub(crate) fn new_epoch(&mut self) {
        self.finish();
        self.last = self.level;
    }

    /// Nothing of the input is mixed: a ramp under way, or due now, ends
    /// at once (one waiting for a frame, audio or another input waits on).
    pub(crate) fn settle(&mut self) {
        if let Some(r) = self.next.filter(|r| r.when == When::Now) {
            self.next = None;
            self.running = None;
            self.level = r.to;
        }
        self.finish();
        self.last = self.level;
    }

    /// The waiting ramp starts at `frame`, from the gain there.
    fn start(&mut self, frame: u64, rate: u32) {
        let Some(ramp) = self.next.take() else {
            return;
        };
        let from = self.gain_at(frame);
        let frames = (ramp.over.as_secs_f64() * f64::from(rate)).round() as u64;
        if frames == 0 || from == ramp.to {
            self.running = None;
            self.level = ramp.to;
            return;
        }
        self.running = Some(Running {
            start: frame,
            from,
            to: ramp.to,
            frames,
            curve: ramp.curve,
        });
    }

    /// The gain at `frame` (at or after the running ramp's start), ending
    /// the ramp there if it is over.
    fn gain_at(&mut self, frame: u64) -> f64 {
        if let Some(r) = self.running {
            let k = frame - r.start;
            if k < r.frames {
                return r.at(k);
            }
            self.finish();
        }
        self.level
    }

    fn finish(&mut self) {
        if let Some(r) = self.running.take() {
            self.level = r.to;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn frames(n: u64) -> Duration {
        Duration::from_secs_f64(n as f64 / f64::from(RATE))
    }

    fn ramp(when: When, to: f64, n: u64, curve: Curve) -> Waiting {
        Waiting {
            when,
            to,
            over: frames(n),
            curve,
        }
    }

    /// Gains of `n` frames from `first`, constant ones written out.
    fn run(g: &mut Gain, first: u64, n: usize, start: Option<u64>) -> Vec<f64> {
        let mut out = vec![f64::NAN; n];
        if let Some(level) = g.fill(first, start, RATE, &mut out) {
            out.fill(level);
        }
        out
    }

    #[test]
    fn a_ramp_lands_on_its_frames() {
        let mut g = Gain::new(1.0);
        g.set(ramp(When::Frame(110), 0.0, 20, Curve::Linear));
        assert_eq!(run(&mut g, 0, 100, None), vec![1.0; 100]);
        let got = run(&mut g, 100, 40, Some(110));
        assert_eq!(got[..10], [1.0; 10]);
        for k in 0..20 {
            assert!((got[10 + k] - (1.0 - k as f64 / 20.0)).abs() < 1e-12, "{k}");
        }
        assert_eq!(got[30..], [0.0; 10]);
        assert!(!g.moving());
        assert_eq!(g.last(), 0.0);
    }

    #[test]
    fn equal_power_ramps_keep_the_summed_power() {
        let (mut up, mut down) = (Gain::new(0.0), Gain::new(1.0));
        up.set(ramp(
            When::FirstAudio { after: 0 },
            1.0,
            64,
            Curve::EqualPower,
        ));
        down.set(ramp(When::With(1), 0.0, 64, Curve::EqualPower));
        assert_eq!(run(&mut up, 0, 5, None), vec![0.0; 5]);
        let a = run(&mut up, 5, 80, Some(5));
        let b = run(&mut down, 5, 80, Some(5));
        assert_eq!((a[0], b[0]), (0.0, 1.0));
        for (x, y) in a.iter().zip(&b) {
            assert!((x * x + y * y - 1.0).abs() < 1e-12);
        }
        assert_eq!((a[64], b[64]), (1.0, 0.0));
    }

    #[test]
    fn a_new_ramp_starts_from_where_the_last_one_is() {
        let mut g = Gain::new(0.0);
        g.set(ramp(When::Now, 1.0, 100, Curve::Linear));
        run(&mut g, 0, 50, Some(0));
        g.set(ramp(When::Now, 0.0, 10, Curve::Linear));
        let got = run(&mut g, 50, 20, Some(50));
        assert!((got[0] - 0.5).abs() < 1e-12);
        assert!((got[5] - 0.25).abs() < 1e-12);
        assert_eq!(got[10..], [0.0; 10]);
        // At once.
        g.set(ramp(When::Now, 0.25, 0, Curve::Linear));
        assert_eq!(g.last(), 0.25);
        assert!(!g.moving());
    }

    #[test]
    fn a_new_epoch_ends_a_ramp_under_way_and_keeps_one_waiting() {
        let mut g = Gain::new(1.0);
        g.set(ramp(When::Now, 0.0, 960, Curve::Linear));
        run(&mut g, 50_000, 480, Some(50_000));
        g.set(ramp(When::FirstAudio { after: 1 }, 1.0, 10, Curve::Linear));
        g.new_epoch();
        assert_eq!(g.last(), 0.0);
        // Frames from 0 again: no frame of the last epoch is asked for.
        assert_eq!(run(&mut g, 0, 4, None), vec![0.0; 4]);
        assert_eq!(g.next(), Some(When::FirstAudio { after: 1 }));
    }

    #[test]
    fn waiting_ramps_start_on_their_frames() {
        let waiting = [
            (1, When::Now, None),
            (2, When::Frame(130), None),
            (3, When::Frame(900), None),
            (4, When::FirstAudio { after: 1 }, Some(2)),
            (5, When::FirstAudio { after: 2 }, Some(2)),
            (6, When::With(2), None),
            (7, When::With(3), None),
        ];
        let mut starts = Vec::new();
        resolve_starts(100, 200, waiting.into_iter(), &mut starts);
        assert_eq!(starts, [(1, 100), (2, 130), (4, 100), (6, 130)]);
        assert_eq!(start_of(&starts, 6), Some(130));
        assert_eq!(start_of(&starts, 3), None);
    }

    #[test]
    fn with_nothing_mixed_a_ramp_settles_and_one_for_audio_waits() {
        let mut g = Gain::new(1.0);
        g.set(ramp(When::Now, 0.0, 9_600, Curve::EqualPower));
        g.settle();
        assert_eq!((g.last(), g.moving()), (0.0, false));
        g.set(ramp(When::Now, 1.0, 9_600, Curve::EqualPower));
        run(&mut g, 0, 10, Some(0));
        g.settle();
        assert_eq!((g.last(), g.moving()), (1.0, false));
        g.set(ramp(When::FirstAudio { after: 1 }, 0.0, 10, Curve::Linear));
        g.settle();
        assert!(g.moving());
        assert_eq!(g.next(), Some(When::FirstAudio { after: 1 }));
    }
}
