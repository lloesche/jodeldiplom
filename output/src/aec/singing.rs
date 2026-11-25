//! Telling singing from the echo the filter leaves (the canceller's
//! double-talk detector): all the canceller knows of whether someone sings.
//!
//! A quiet singer leaves the filter's leftover hardly louder than the echo
//! it usually leaves: on a MacBook's speakers the singer was 8 dB under the
//! echo, and the leftover's level alone (a block leaving ten times the
//! usual) never told the singing apart, so the filters learnt on the voice.
//! What tells it apart is the pitch: once the filter takes most of the echo
//! out, a voice stands in the leftover as one clear pitch (in that
//! recording on 55 % of the sung moments, against none elsewhere), and the
//! echo the filter predicts has none there (the music's own notes, in the
//! leftover too, are as clear in the estimate). So, once the canceller has
//! learnt (before, its estimate is next to nothing and has no pitch to
//! compare), a look every [`LOOK_EVERY`] blocks low-passes the leftover and
//! the estimate and takes them down to about 8 to 12 kHz, finds the lag
//! where the leftover is most like itself (McLeod's normalised square
//! difference, over pitches from [`LOWEST`] to [`HIGHEST`]), and hears a
//! voice where that is [`CLEAR`] and the estimate at that lag is under
//! [`MUSIC`]; a few looks of the last ones must ([`NEED`] of [`SPAN`]: a
//! single one is often the music's), and the leftover must stand over its
//! own floor (a hum or a fan in the room is pitched too, but steady).
//! Besides, a leftover far louder than the filter usually leaves
//! ([`LOUDER`]) is something else in the microphone (someone loud, a
//! knock, a note held as loud as the music's own). Either is held [`HOLD`] (the
//! voice's unpitched sounds, its breaths, between; the levels the meter
//! reads still carry its tail).
//!
//! Neither looks at the music's way to the microphone: a room that changed
//! (the laptop moved, the volume turned up) leaves the music's echo,
//! pitched where the estimate is, so the filters learn it again, unless it
//! is loud enough for [`LOUDER`]; then the canceller's trial filter does
//! (and its shadow, at a tenth of its pace).

use super::{BLOCK, QUIET};

/// The voice's lowest and highest pitch looked for, in Hz.
const LOWEST: f32 = 70.0;
const HIGHEST: f32 = 800.0;
/// The low-pass before the pitch is looked for, in Hz (the voice's
/// fundamental and first harmonics).
const LOW_PASS: f32 = 1500.0;
/// The pitch is looked for at the canceller's rate halved until it is
/// under twice this (Hz).
const PITCH_RATE: u32 = 8000;
/// The window a look takes, in samples at the pitch's rate (about 23 ms).
const WINDOW: usize = 256;
/// Blocks between looks (about 6 ms).
const LOOK_EVERY: u32 = 2;
/// How alike the leftover must be to itself a period on to be pitched, and
/// how alike the estimate may be at that lag for the pitch still not to be
/// the music's: on a recording of a MacBook's echo and a singer 8 dB under
/// it, the echo alone (31 s of it) passed 0.1 looks a second at 0.85 and
/// 0.5 (the singing 31 of its 172), 0.9 at 0.8 and 0.4 (the singing 44);
/// the [`NEED`] of [`SPAN`] takes most of those few out.
const CLEAR: f32 = 0.8;
const MUSIC: f32 = 0.4;
/// Looks of the last [`SPAN`] that must hear a voice.
const NEED: u32 = 3;
const SPAN: u32 = 8;
/// Seconds singing is held after either cue was last heard.
const HOLD: f32 = 0.25;
/// How far over the leftover's floor (its quietest in the last
/// [`FLOOR_SPAN`] seconds, in [`FLOOR_PARTS`] parts) a voice must be
/// (+6 dB; at +3 a hum 20 dB under a laptop's echo, what the filter left
/// of the echo coming and going over it, passed for singing on a tenth of
/// the blocks). A floor that only rose slowly from its quietest stuck near
/// nothing after a silent start, and a hum passed as singing throughout.
/// A loud note held longer than the span is still held by [`LOUDER`].
const OVER_FLOOR: f32 = 4.0;
const FLOOR_SPAN: f32 = 3.0;
const FLOOR_PARTS: usize = 12;
/// A block leaving this many times the echo the filter usually leaves
/// (+8.5 dB) is something else: the singer above the music's echo, as a
/// MacBook's leaves it, at 10 times (as before) the pitch was found on 19 %
/// of the frames of a voice 10 dB under that echo, at 7 on 22 %; at 5
/// times, a room that changed was learnt too slowly. The leftover is
/// smoothed first, by [`LOUDER_FOLLOW`] a block (about 15 ms): held as
/// singing is, single blocks of the echo alone over it froze the filters
/// on a twentieth of a laptop's echo.
pub(super) const LOUDER: f32 = 7.0;
const LOUDER_FOLLOW: f32 = 0.2;
/// Sums of squares under this are no signal (a silent window).
const NOTHING: f64 = 1e-20;

/// A second-order Butterworth low-pass, a sample at a time.
#[derive(Clone, Copy)]
struct LowPass {
    b: [f32; 3],
    a: [f32; 2],
    x: [f32; 2],
    y: [f32; 2],
}

impl LowPass {
    fn new(rate: u32, at: f32) -> LowPass {
        let w = 2.0 * std::f32::consts::PI * at / rate.max(1) as f32;
        let alpha = w.sin() / std::f32::consts::SQRT_2;
        let c = w.cos();
        let a0 = 1.0 + alpha;
        let b = (1.0 - c) / (2.0 * a0);
        LowPass {
            b: [b, 2.0 * b, b],
            a: [-2.0 * c / a0, (1.0 - alpha) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }

    fn next(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x = [x, self.x[0]];
        self.y = [y, self.y[0]];
        y
    }

    /// Forgets what it heard.
    fn clear(&mut self) {
        self.x = [0.0; 2];
        self.y = [0.0; 2];
    }
}

/// A signal low-passed and taken down to the pitch's rate: its newest
/// samples, oldest first, and their running sums of squares.
struct Decimated {
    filter: LowPass,
    every: usize,
    samples: Vec<f32>,
    /// Per sample, the sum of the squares of those before it (one more
    /// than the samples), in f64: after a loud stretch, the differences of
    /// a quiet one's sums in f32 were rounding alone.
    squares: Vec<f64>,
    /// Samples pushed since it was cleared (at most its length).
    filled: usize,
}

impl Decimated {
    fn new(rate: u32, every: usize, len: usize) -> Decimated {
        debug_assert!(BLOCK % every == 0);
        Decimated {
            filter: LowPass::new(rate, LOW_PASS),
            every,
            samples: vec![0.0; len],
            squares: vec![0.0; len + 1],
            filled: 0,
        }
    }

    fn clear(&mut self) {
        self.filter.clear();
        self.samples.fill(0.0);
        self.filled = 0;
    }

    /// It holds a whole window and its longest lag.
    fn full(&self) -> bool {
        self.filled == self.samples.len()
    }

    fn push(&mut self, block: &[f32; BLOCK]) {
        let n = BLOCK / self.every;
        let len = self.samples.len();
        self.samples.copy_within(n.., 0);
        for (i, v) in block.iter().enumerate() {
            let y = self.filter.next(*v);
            if i % self.every == self.every - 1 {
                self.samples[len - n + i / self.every] = y;
            }
        }
        self.filled = (self.filled + n).min(len);
        // Nothing but numbers stays (a device's glitch must not stick).
        if !self.samples[len - n..].iter().all(|v| v.is_finite()) {
            self.clear();
        }
    }

    /// The newest window's power.
    fn power(&self) -> f32 {
        let s = &self.samples[self.samples.len() - WINDOW..];
        s.iter().map(|v| v * v).sum::<f32>() / WINDOW as f32
    }

    /// The running sums of squares, for [`Decimated::alike`].
    fn sum_squares(&mut self) {
        for i in 0..self.samples.len() {
            let v = f64::from(self.samples[i]);
            self.squares[i + 1] = self.squares[i] + v * v;
        }
    }

    /// How alike the newest window is to the one `lag` samples before it
    /// (1: the same, 0: nothing in common, −1: turned over; McLeod's
    /// normalised square difference), its squares summed since the last
    /// push.
    fn alike(&self, lag: usize) -> f32 {
        let end = self.samples.len();
        let now = &self.samples[end - WINDOW..];
        let before = &self.samples[end - WINDOW - lag..end - lag];
        let r: f32 = now.iter().zip(before).map(|(a, b)| a * b).sum();
        let q = &self.squares;
        let m = (q[end] - q[end - WINDOW]) + (q[end - lag] - q[end - WINDOW - lag]);
        if m > NOTHING {
            (2.0 * f64::from(r) / m) as f32
        } else {
            0.0
        }
    }
}

/// The double-talk detector ([`crate::aec::singing`]).
pub(super) struct Singing {
    leftover: Decimated,
    estimate: Decimated,
    /// The lags looked at, in samples at the pitch's rate: the highest
    /// pitch's and the lowest's.
    shortest: usize,
    longest: usize,
    /// The leftover's likeness at each lag (scratch).
    alike: Vec<f32>,
    since_look: u32,
    /// The leftover's power, smoothed ([`LOUDER_FOLLOW`]).
    level: f32,
    /// The last looks, a bit each, the newest lowest: a voice heard.
    looks: u32,
    /// Blocks since either cue was last heard, and the blocks it is held.
    since: u32,
    hold: u32,
    /// The leftover's quietest power in each part of the last
    /// [`FLOOR_SPAN`] (a ring), the part now, and its looks so far of the
    /// looks a part takes.
    quietest: [f32; FLOOR_PARTS],
    part: usize,
    part_looks: u32,
    looks_a_part: u32,
}

impl Singing {
    /// A detector for a canceller at `rate`.
    pub(super) fn new(rate: u32) -> Singing {
        let mut every = 1usize;
        while every < BLOCK && rate / (2 * every as u32) >= PITCH_RATE {
            every *= 2;
        }
        let at = rate as f32 / every as f32;
        let shortest = (at / HIGHEST).floor().max(2.0) as usize;
        let longest = (at / LOWEST).ceil() as usize;
        let len = WINDOW + longest;
        let looks_a_second = rate as f32 / (BLOCK as f32 * LOOK_EVERY as f32);
        Singing {
            leftover: Decimated::new(rate, every, len),
            estimate: Decimated::new(rate, every, len),
            shortest,
            longest,
            alike: vec![0.0; longest + 1],
            since_look: 0,
            level: 0.0,
            looks: 0,
            since: u32::MAX,
            hold: (HOLD * rate as f32 / BLOCK as f32) as u32,
            quietest: [f32::INFINITY; FLOOR_PARTS],
            part: 0,
            part_looks: 0,
            looks_a_part: (looks_a_second * FLOOR_SPAN / FLOOR_PARTS as f32).max(1.0) as u32,
        }
    }

    /// Forgets what it heard (a new output).
    pub(super) fn clear(&mut self) {
        self.leftover.clear();
        self.estimate.clear();
        self.since_look = 0;
        self.level = 0.0;
        self.looks = 0;
        self.since = u32::MAX;
        self.quietest = [f32::INFINITY; FLOOR_PARTS];
        self.part_looks = 0;
    }

    /// Someone sings (or sang within [`HOLD`]).
    pub(super) fn singing(&self) -> bool {
        self.since <= self.hold
    }

    /// Nobody has sung for a while: what the canceller leaves is the echo's.
    pub(super) fn settled(&self) -> bool {
        !self.singing()
    }

    /// One block: `leftover` what the filter in use left of the
    /// microphone (its power `pe`), `estimate` the echo it predicted;
    /// `usual` what of the echo the filter usually leaves, once it has
    /// learnt.
    pub(super) fn hear(
        &mut self,
        leftover: &[f32; BLOCK],
        estimate: &[f32; BLOCK],
        pe: f32,
        usual: Option<f32>,
    ) {
        self.leftover.push(leftover);
        self.estimate.push(estimate);
        self.since = self.since.saturating_add(1);
        self.since_look += 1;
        if self.since_look >= LOOK_EVERY {
            self.since_look = 0;
            let voiced = usual.is_some() && self.voiced();
            self.looks = (self.looks << 1) | u32::from(voiced);
            if (self.looks & ((1 << SPAN) - 1)).count_ones() >= NEED {
                self.since = 0;
            }
        }
        self.level += (pe - self.level) * LOUDER_FOLLOW;
        if usual.is_some_and(|u| self.level > LOUDER * u + QUIET) {
            self.since = 0;
        }
    }

    /// A look: the leftover over its floor holds a clear pitch, and the
    /// estimate not that one.
    fn voiced(&mut self) -> bool {
        if !self.leftover.full() {
            return false;
        }
        let power = self.leftover.power();
        self.part_looks += 1;
        if self.part_looks > self.looks_a_part {
            self.part_looks = 1;
            self.part = (self.part + 1) % FLOOR_PARTS;
            self.quietest[self.part] = f32::INFINITY;
        }
        self.quietest[self.part] = self.quietest[self.part].min(power);
        let floor = self.quietest.iter().copied().fold(f32::INFINITY, f32::min);
        if power <= OVER_FLOOR * floor {
            return false;
        }
        // The leftover's likeness at each lag, from the first where it is
        // unlike itself (half a period on; a sound without a pitch, a bass
        // above all, may never be within the lags) to the longest.
        self.leftover.sum_squares();
        let mut from = None;
        for lag in 1..=self.longest {
            let n = self.leftover.alike(lag);
            self.alike[lag] = n;
            if from.is_none() && n < 0.0 {
                from = Some(lag);
            }
        }
        let Some(from) = from else {
            return false;
        };
        let (lag, clear) = (from.max(self.shortest)..=self.longest)
            .map(|l| (l, self.alike[l]))
            .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
        if clear < CLEAR {
            return false;
        }
        self.estimate.sum_squares();
        let music = (lag - 1..=(lag + 1).min(self.longest))
            .map(|l| self.estimate.alike(l))
            .fold(f32::MIN, f32::max);
        music < MUSIC
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const RATE: u32 = 48_000;

    /// A repeatable noise, −0.5 to 0.5.
    struct Noise(u32);

    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        }
    }

    /// A harmonic tone at `f` Hz, `at` seconds in, with a vibrato of
    /// `vibrato` (a share of `f`, at 5.5 Hz).
    fn note(f: f32, vibrato: f32, at: f32) -> f32 {
        let w = 2.0 * PI * 5.5;
        let phase = 2.0 * PI * f * (at + vibrato * (1.0 - (w * at).cos()) / w);
        (1..6).map(|h| (h as f32 * phase).sin() / h as f32).sum()
    }

    /// Music as the estimate predicts it: coloured noise and chords that
    /// change every half second (none near the voice's 220 Hz).
    fn music(noise: &mut Noise, lp: &mut f32, t: f32) -> f32 {
        *lp += (noise.next() - *lp) * 0.2;
        let root = [330.0, 392.0, 494.0, 587.0][(t * 2.0) as usize % 4];
        0.4 * *lp + 0.05 * note(root, 0.0, t) + 0.03 * note(root * 1.5, 0.0, t)
    }

    /// Feeds 6 s of `leftover` and `estimate` (as functions of the time in
    /// seconds) to a learnt detector; the share of the blocks after the
    /// first second it said someone sings.
    fn sung(mut leftover: impl FnMut(f32) -> f32, mut estimate: impl FnMut(f32) -> f32) -> f32 {
        let mut d = Singing::new(RATE);
        let blocks = 6 * RATE as usize / BLOCK;
        let mut heard = 0;
        for b in 0..blocks {
            let t = |i: usize| (b * BLOCK + i) as f32 / RATE as f32;
            let e: [f32; BLOCK] = std::array::from_fn(|i| leftover(t(i)));
            let y: [f32; BLOCK] = std::array::from_fn(|i| estimate(t(i)));
            let pe = e.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
            // The leftover never far over the usual: only the pitch tells.
            d.hear(&e, &y, pe, Some(1.0));
            if b * BLOCK >= RATE as usize {
                heard += usize::from(d.singing());
            }
        }
        heard as f32 / (blocks - RATE as usize / BLOCK) as f32
    }

    /// Syllables: 0.35 s sung, 0.15 s of breath.
    fn syllable(t: f32) -> f32 {
        f32::from(u8::from((t * 2.0).fract() < 0.7))
    }

    /// A voice in the leftover (its syllables over a little noise), the
    /// music in the estimate: heard nearly throughout.
    #[test]
    fn a_voice_in_the_leftover_is_singing() {
        let (mut n1, mut n2, mut lp) = (Noise(1), Noise(2), 0.0);
        let share = sung(
            |t| 0.05 * syllable(t) * note(220.0, 0.02, t) + 0.002 * n1.next(),
            |t| music(&mut n2, &mut lp, t),
        );
        assert!(share > 0.8, "heard on {:.0} % of the blocks", 100.0 * share);
    }

    /// The music's note in the leftover (what the filter left of it) and in
    /// the estimate: never singing.
    #[test]
    fn the_musics_own_note_is_not_singing() {
        let (mut n1, mut n2, mut lp) = (Noise(3), Noise(4), 0.0);
        let share = sung(
            |t| 0.05 * syllable(t) * note(220.0, 0.02, t) + 0.002 * n1.next(),
            |t| music(&mut n2, &mut lp, t) + 0.3 * syllable(t) * note(220.0, 0.02, t),
        );
        assert!(
            share < 0.02,
            "heard on {:.0} % of the blocks",
            100.0 * share
        );
    }

    /// A steady hum (100 Hz and its harmonics) in the leftover, nothing of
    /// it in the estimate: never singing (it is the leftover's floor).
    #[test]
    fn a_steady_hum_is_not_singing() {
        let (mut n1, mut n2, mut lp) = (Noise(5), Noise(6), 0.0);
        let share = sung(
            |t| 0.05 * note(100.0, 0.0, t) + 0.002 * n1.next(),
            |t| music(&mut n2, &mut lp, t),
        );
        assert!(
            share < 0.02,
            "heard on {:.0} % of the blocks",
            100.0 * share
        );
    }

    /// Noise in the leftover, in bursts as loud as the voice's syllables:
    /// never singing (no pitch).
    #[test]
    fn noise_is_not_singing() {
        let (mut n1, mut n2, mut lp) = (Noise(7), Noise(8), 0.0);
        let share = sung(
            |t| 0.05 * syllable(t) * n1.next() + 0.002 * n1.next(),
            |t| music(&mut n2, &mut lp, t),
        );
        assert!(
            share < 0.02,
            "heard on {:.0} % of the blocks",
            100.0 * share
        );
    }
}
