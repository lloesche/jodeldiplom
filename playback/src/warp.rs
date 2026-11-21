//! Beat lock: one track's time laid onto another's, and audio read along it.
//!
//! [`TimeMap`] maps the playing track's time to the incoming track's, from
//! the beats the two are to share: beat `i` of one at beat `i` of the
//! other. The beat times are first smoothed (each is moved onto a straight
//! line fitted to the beats around it, so the jitter of an analysis does
//! not become a wobble in speed, while a live drummer's drift is kept),
//! then joined by a monotone cubic, so the speed (the map's slope) is
//! continuous. Before the first pair the map runs straight at the first
//! speed; after the last it glides to speed 1 over a set time and stays
//! there, so the incoming track ends at its own tempo.
//!
//! [`VarResampler`] reads interleaved frames at positions that change
//! continuously (from a [`TimeMap`] or anything else): each output frame
//! is a windowed-sinc interpolation of the input around its position. The
//! kernel widens and its cutoff falls when the input is read faster than
//! it was recorded, so nothing above the new band folds back.

use std::collections::VecDeque;

use crate::resample::{bessel_i0, sinc};

/// Beats on each side whose straight line a beat is moved onto.
const SMOOTH_BEATS: usize = 4;
/// Zero crossings of the kernel on each side at full band.
const HALF_WIDTH: usize = 32;
/// The kernel's cutoff as a share of the Nyquist frequency at full band.
const CUTOFF: f64 = 0.92;
/// The Kaiser window's beta (about 90 dB of stopband).
const BETA: f64 = 9.0;
/// Table entries per unit of the kernel's argument.
const TABLE_STEPS: usize = 512;

/// A map from one track's time to another's, through paired beats.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeMap {
    /// The knots: the playing track's time, the incoming track's.
    knots: Vec<(f64, f64)>,
    /// The slope at each knot.
    slopes: Vec<f64>,
    /// Seconds after the last knot over which the speed glides to 1.
    settle: f64,
}

/// The median of `values` (reordered); 1 when empty.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

impl TimeMap {
    /// The map through `pairs` (the playing track's beat, the incoming
    /// track's, both ascending), gliding to speed 1 over `settle` seconds
    /// after the last pair. With fewer than two pairs it runs at speed 1
    /// through the one pair (or through the start).
    pub fn new(pairs: &[(f64, f64)], settle: f64) -> TimeMap {
        let mut pairs: Vec<(f64, f64)> = pairs.to_vec();
        pairs.dedup_by(|b, a| b.0 <= a.0 || b.1 <= a.1);
        if pairs.len() < 2 {
            let knot = pairs.first().copied().unwrap_or((0.0, 0.0));
            return TimeMap {
                knots: vec![knot],
                slopes: vec![1.0],
                settle: 0.0,
            };
        }
        let n = pairs.len();
        // Each beat onto the line through its neighbours (least squares).
        let knots: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(SMOOTH_BEATS);
                let hi = (i + SMOOTH_BEATS + 1).min(n);
                let span = &pairs[lo..hi];
                let m = span.len() as f64;
                let (ma, mb) = span
                    .iter()
                    .fold((0.0, 0.0), |(x, y), &(a, b)| (x + a / m, y + b / m));
                let (mut sab, mut saa) = (0.0, 0.0);
                for &(a, b) in span {
                    sab += (a - ma) * (b - mb);
                    saa += (a - ma) * (a - ma);
                }
                let slope = if saa > 0.0 { sab / saa } else { 1.0 };
                (pairs[i].0, mb + slope * (pairs[i].0 - ma))
            })
            .collect();
        // The knots must still rise: fall back on the beats where not.
        let knots = if knots.windows(2).all(|w| w[1].1 > w[0].1) {
            knots
        } else {
            pairs
        };
        let secants: Vec<f64> = knots
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0))
            .collect();
        // Fritsch-Carlson slopes: harmonic means of the secants, so the
        // cubic between two knots stays monotone.
        let mut slopes = vec![0.0; n];
        slopes[0] = secants[0];
        slopes[n - 1] = secants[n - 2];
        for i in 1..n - 1 {
            let (s0, s1) = (secants[i - 1], secants[i]);
            slopes[i] = if s0 * s1 <= 0.0 {
                0.0
            } else {
                2.0 * s0 * s1 / (s0 + s1)
            };
        }
        TimeMap {
            knots,
            slopes,
            settle: settle.max(0.0),
        }
    }

    /// The speed (incoming seconds per playing second) over the map's
    /// beats: the median of their secants.
    pub fn typical_speed(&self) -> f64 {
        let mut s: Vec<f64> = self
            .knots
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0))
            .collect();
        median(&mut s)
    }

    /// The first knot's times.
    pub fn first(&self) -> (f64, f64) {
        self.knots[0]
    }

    /// Where the glide to speed 1 ends, in the playing track's time.
    pub fn settled_at(&self) -> f64 {
        self.knots[self.knots.len() - 1].0 + self.settle
    }

    /// The incoming track's time at the playing track's time `a`.
    pub fn at(&self, a: f64) -> f64 {
        let n = self.knots.len();
        let (a0, b0) = self.knots[0];
        if a <= a0 {
            return b0 + self.slopes[0] * (a - a0);
        }
        let (an, bn) = self.knots[n - 1];
        if a >= an {
            let s = self.slopes[n - 1];
            let t = a - an;
            if self.settle <= 0.0 {
                return bn + t;
            }
            // The speed glides linearly from `s` to 1 over `settle`.
            let g = t.min(self.settle);
            let during = s * g + (1.0 - s) * g * g / (2.0 * self.settle);
            return bn + during + (t - g);
        }
        let i = self.knots.partition_point(|k| k.0 <= a) - 1;
        let ((x0, y0), (x1, y1)) = (self.knots[i], self.knots[i + 1]);
        let h = x1 - x0;
        let u = (a - x0) / h;
        let (u2, u3) = (u * u, u * u * u);
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * y0 + h10 * h * self.slopes[i] + h01 * y1 + h11 * h * self.slopes[i + 1]
    }

    /// The playing track's time at which the incoming track is at `b` (the
    /// map is monotone, so this is unique): Newton steps from where speed
    /// 1 would put it, then halving where those do not settle.
    pub fn inverse(&self, b: f64) -> f64 {
        let (a0, b0) = self.knots[0];
        let mut a = a0 + (b - b0);
        for _ in 0..8 {
            let err = self.at(a) - b;
            if err.abs() < 1e-9 {
                return a;
            }
            a -= err / self.speed(a).max(1e-3);
        }
        // Halving between bounds that bracket it.
        let (mut lo, mut hi) = (a - 10.0, a + 10.0);
        while self.at(lo) > b {
            lo -= 10.0;
        }
        while self.at(hi) < b {
            hi += 10.0;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.at(mid) < b {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The speed at `a`: the map's slope there.
    pub fn speed(&self, a: f64) -> f64 {
        let n = self.knots.len();
        let (a0, _) = self.knots[0];
        if a <= a0 {
            return self.slopes[0];
        }
        let (an, _) = self.knots[n - 1];
        if a >= an {
            if self.settle <= 0.0 {
                return 1.0;
            }
            let g = ((a - an) / self.settle).min(1.0);
            return self.slopes[n - 1] + (1.0 - self.slopes[n - 1]) * g;
        }
        let i = self.knots.partition_point(|k| k.0 <= a) - 1;
        let ((x0, y0), (x1, y1)) = (self.knots[i], self.knots[i + 1]);
        let h = x1 - x0;
        let u = (a - x0) / h;
        let u2 = u * u;
        let d00 = 6.0 * u2 - 6.0 * u;
        let d10 = 3.0 * u2 - 4.0 * u + 1.0;
        let d01 = -6.0 * u2 + 6.0 * u;
        let d11 = 3.0 * u2 - 2.0 * u;
        (d00 * y0 + d01 * y1) / h + d10 * self.slopes[i] + d11 * self.slopes[i + 1]
    }
}

impl TimeMap {
    /// The map through `knots` with these `slopes` exactly (no smoothing),
    /// gliding to speed 1 over `settle` seconds after the last knot; at
    /// least one knot.
    pub fn from_knots(knots: Vec<(f64, f64)>, slopes: Vec<f64>, settle: f64) -> TimeMap {
        debug_assert!(!knots.is_empty() && knots.len() == slopes.len());
        TimeMap {
            knots,
            slopes,
            settle: settle.max(0.0),
        }
    }

    /// This map (from the playing track's time) read from the output's
    /// time while the playing track goes along `glide`: each knot moved to
    /// the output time its playing-track time is heard at, its slope
    /// times the playing track's speed there. The map as it is where
    /// `glide` changes nothing.
    pub fn through_glide(&self, glide: &Glide) -> TimeMap {
        if glide.is_identity() {
            return self.clone();
        }
        let (knots, slopes) = self
            .knots
            .iter()
            .zip(&self.slopes)
            .map(|(&(a, b), &s)| {
                let u = glide.inverse(a);
                ((u, b), s * glide.speed(u))
            })
            .unzip();
        TimeMap::from_knots(knots, slopes, self.settle)
    }
}

/// The playing track's time along the output's while a transition's tempo
/// glides: up to `start` the two are the same; from there the playing
/// track's speed goes from 1 to `end` along a smoothstep (so its
/// acceleration has no jumps either) over `length` seconds of output, in
/// which it covers its window, and stays at `end` after.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glide {
    start: f64,
    length: f64,
    end: f64,
}

/// A speed this near 1 at the glide's end is 1: no glide.
const GLIDE_NONE: f64 = 0.002;
/// The most a glide changes the playing track's speed by (either way).
pub const GLIDE_MOST: f64 = 1.3;

impl Glide {
    /// From the playing track's time `start`, its `span` seconds covered
    /// at a speed going from 1 to `end`, kept within 30 % of 1 (more is
    /// not a tempo change but another track: the next one settles the
    /// rest after the window).
    pub fn new(start: f64, span: f64, end: f64) -> Glide {
        let end = if end.is_finite() {
            end.clamp(1.0 / GLIDE_MOST, GLIDE_MOST)
        } else {
            1.0
        };
        let end = if (end - 1.0).abs() < GLIDE_NONE {
            1.0
        } else {
            end
        };
        Glide {
            start,
            length: 2.0 * span.max(0.0) / (1.0 + end),
            end,
        }
    }

    /// No glide: the playing track at speed 1 throughout.
    pub fn is_identity(&self) -> bool {
        self.end == 1.0
    }

    /// Where it starts.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Its length in the output's seconds.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// The playing track's speed at its end.
    pub fn end_speed(&self) -> f64 {
        self.end
    }

    /// How far through it the output's time `u` is (0 to 1).
    fn share(&self, u: f64) -> f64 {
        if self.length <= 0.0 {
            return if u > self.start { 1.0 } else { 0.0 };
        }
        ((u - self.start) / self.length).clamp(0.0, 1.0)
    }

    /// The playing track's speed at the output's time `u`.
    pub fn speed(&self, u: f64) -> f64 {
        let x = self.share(u);
        1.0 + (self.end - 1.0) * x * x * (3.0 - 2.0 * x)
    }

    /// The playing track's time at the output's time `u`.
    pub fn at(&self, u: f64) -> f64 {
        if u <= self.start || self.is_identity() {
            return u;
        }
        let x = self.share(u);
        // The smoothstep's integral: x + (end − 1)(x³ − x⁴/2).
        let covered = self.length * (x + (self.end - 1.0) * (x * x * x - x * x * x * x / 2.0));
        let after = (u - self.start - self.length).max(0.0) * self.end;
        self.start + covered + after
    }

    /// The output's time at which the playing track is at `a` (the glide
    /// rises, so this is unique).
    pub fn inverse(&self, a: f64) -> f64 {
        if a <= self.start || self.is_identity() {
            return a;
        }
        let mut u = self.start + (a - self.start) / (0.5 * (1.0 + self.end));
        for _ in 0..30 {
            let err = self.at(u) - a;
            if err.abs() < 1e-12 {
                break;
            }
            u -= err / self.speed(u);
        }
        u
    }

    /// The glide as a map from the output's time to the playing track's,
    /// with knots every quarter second from a second before it to two
    /// after it, and one an hour on: at its end speed after it (straight,
    /// so the one far knot is exact), for a track left reading along it.
    pub fn map(&self) -> TimeMap {
        let from = self.start - 1.0;
        let to = self.start + self.length + 2.0;
        let n = ((to - from) / 0.25).ceil() as usize + 1;
        let (mut knots, mut slopes): (Vec<(f64, f64)>, Vec<f64>) = (0..n)
            .map(|i| {
                let u = from + 0.25 * i as f64;
                ((u, self.at(u)), self.speed(u))
            })
            .unzip();
        let far = to + 3600.0;
        knots.push((far, self.at(far)));
        slopes.push(self.end);
        TimeMap::from_knots(knots, slopes, 0.0)
    }
}

/// The interpolation kernel at full band, tabulated over `0..=HALF_WIDTH`.
fn kernel_table() -> Vec<f64> {
    let n = HALF_WIDTH * TABLE_STEPS;
    let norm = bessel_i0(BETA);
    (0..=n + 1)
        .map(|i| {
            let x = i as f64 / TABLE_STEPS as f64;
            let r = x / HALF_WIDTH as f64;
            if r >= 1.0 {
                return 0.0;
            }
            let window = bessel_i0(BETA * (1.0 - r * r).sqrt()) / norm;
            CUTOFF * sinc(CUTOFF * x) * window
        })
        .collect()
}

/// Reads interleaved frames at continuously changing positions.
///
/// Push input with [`VarResampler::push`]; [`VarResampler::pull`] makes
/// output frames, each at the input position the caller gives for it
/// (frames since the first pushed, fractional). Positions must not go
/// back by more than the kernel reaches; input well behind the last
/// position is dropped.
#[derive(Clone, Debug)]
pub struct VarResampler {
    channels: usize,
    table: Vec<f64>,
    /// Input from frame `base` on, interleaved.
    input: VecDeque<f64>,
    base: u64,
    /// Output frames made.
    made: u64,
    /// The input has ended (silence after it).
    ended: bool,
}

impl VarResampler {
    /// A resampler of `channels` channels.
    pub fn new(channels: usize) -> VarResampler {
        VarResampler {
            channels: channels.max(1),
            table: kernel_table(),
            input: VecDeque::new(),
            base: 0,
            made: 0,
            ended: false,
        }
    }

    /// Output frames made so far.
    pub fn made(&self) -> u64 {
        self.made
    }

    /// Input frames taken so far.
    pub fn taken(&self) -> u64 {
        self.base + (self.input.len() / self.channels) as u64
    }

    /// Takes interleaved input frames.
    pub fn push(&mut self, frames: &[f64]) {
        self.input.extend(frames.iter().copied());
    }

    /// The input has ended: what follows it is silence.
    pub fn end(&mut self) {
        self.ended = true;
    }

    /// Input frames, from the first, a position needs taken before it can
    /// be read at `speed`.
    pub fn needs(position: f64, speed: f64) -> u64 {
        let reach = HALF_WIDTH as f64 * speed.max(1.0) + 2.0;
        (position + reach).ceil().max(0.0) as u64
    }

    /// The kernel at `x` input frames from the position, scaled for
    /// `cutoff` (1 at full band, lower when reading faster).
    fn kernel(&self, x: f64, cutoff: f64) -> f64 {
        let t = (x * cutoff).abs() * TABLE_STEPS as f64;
        let i = t as usize;
        if i + 1 >= self.table.len() {
            return 0.0;
        }
        let f = t - i as f64;
        cutoff * (self.table[i] + (self.table[i + 1] - self.table[i]) * f)
    }

    /// Makes up to `frames` output frames onto `out`, frame `k` (counted
    /// over the whole stream) read at input position `position(k)`; stops
    /// early where the input does not reach yet. Returns the frames made.
    pub fn pull(
        &mut self,
        out: &mut Vec<f64>,
        frames: usize,
        mut position: impl FnMut(u64) -> f64,
    ) -> usize {
        let ch = self.channels;
        let mut made = 0;
        while made < frames {
            let k = self.made;
            let p = position(k);
            let speed = (position(k + 1) - p).abs().max(1e-6);
            let cutoff = (1.0 / speed).min(1.0);
            let reach = HALF_WIDTH as f64 / cutoff;
            let last = (p + reach).floor() as i64;
            if !self.ended && last >= self.taken() as i64 {
                break;
            }
            let first = (p - reach).ceil() as i64;
            let start = out.len();
            out.resize(start + ch, 0.0);
            let available = self.taken() as i64;
            for j in first.max(self.base as i64)..=last.min(available - 1) {
                let w = self.kernel(p - j as f64, cutoff);
                if w == 0.0 {
                    continue;
                }
                let at = (j as u64 - self.base) as usize * ch;
                for c in 0..ch {
                    out[start + c] += w * self.input[at + c];
                }
            }
            self.made += 1;
            made += 1;
            // Drop input the kernel cannot reach again (positions only
            // move on, give or take its width).
            let keep_from = ((p - 2.0 * reach).floor().max(0.0) as u64).max(self.base);
            let drop = ((keep_from - self.base) as usize).min(self.input.len() / ch);
            if drop > 4096 {
                self.input.drain(..drop * ch);
                self.base += drop as u64;
            }
        }
        made
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::Resampler;

    const RATE: f64 = 44_100.0;

    fn tone(hz: f64, seconds: f64) -> Vec<f64> {
        (0..(seconds * RATE) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / RATE).sin())
            .collect()
    }

    /// The frequency of a mono signal from its upward zero crossings.
    fn frequency(x: &[f64]) -> f64 {
        let ups: Vec<usize> = (1..x.len())
            .filter(|&i| x[i - 1] < 0.0 && x[i] >= 0.0)
            .collect();
        (ups.len() - 1) as f64 * RATE / (ups[ups.len() - 1] - ups[0]) as f64
    }

    fn read(input: &[f64], frames: usize, position: impl Fn(u64) -> f64) -> Vec<f64> {
        let mut r = VarResampler::new(1);
        r.push(input);
        r.end();
        let mut out = Vec::new();
        r.pull(&mut out, frames, position);
        out
    }

    #[test]
    fn at_speed_one_the_input_comes_back() {
        let x = tone(1000.0, 1.0);
        let y = read(&x, 40_000, |k| k as f64);
        let err = (2000..38_000)
            .map(|i| (y[i] - x[i]).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-3, "{err}");
    }

    #[test]
    fn a_steady_speed_matches_the_rational_resampler() {
        // Read 5 % faster: the same as resampling 44.1 kHz to 42 kHz.
        let x = tone(1000.0, 1.0);
        let y = read(&x, 40_000, |k| k as f64 * 1.05);
        let mut r = Resampler::new(44_100, 42_000, 1);
        let mut z = Vec::new();
        r.process_f64(&x, &mut z);
        let err = (2000..38_000)
            .map(|i| (y[i] - z[i]).abs())
            .fold(0.0, f64::max);
        assert!(err < 2e-3, "{err}");
        assert!((frequency(&y[2000..38_000]) - 1050.0).abs() < 1.0);
    }

    #[test]
    fn a_changing_speed_has_no_clicks() {
        // The speed ramps from 1 to 1.08 over a second.
        let x = tone(1000.0, 2.0);
        let pos = |k: u64| {
            let t = k as f64 / RATE;
            RATE * (t + 0.04 * t * t)
        };
        let y = read(&x, 44_000, pos);
        // The second difference of a sine at up to 1.08 kHz stays below
        // (2π f / rate)² everywhere.
        let bound = (2.0 * std::f64::consts::PI * 1100.0 / RATE).powi(2) * 1.1;
        let worst = (2000..43_000)
            .map(|i| (y[i + 1] - 2.0 * y[i] + y[i - 1]).abs())
            .fold(0.0, f64::max);
        assert!(worst < bound, "{worst} {bound}");
    }

    #[test]
    fn reading_faster_does_not_fold_the_top_back() {
        // 20.5 kHz read 20 % faster would be 24.6 kHz, above the band: it
        // must go, not come back as 19.5 kHz.
        let x = tone(20_500.0, 1.0);
        let y = read(&x, 30_000, |k| k as f64 * 1.2);
        let rms = (y[2000..28_000].iter().map(|v| v * v).sum::<f64>() / 26_000.0).sqrt();
        assert!(rms < 1e-3, "{rms}");
    }

    #[test]
    fn the_map_runs_through_steady_beats_and_settles() {
        // The playing track at 120 BPM, the incoming at 125 from 10 s.
        let pairs: Vec<(f64, f64)> = (0..32)
            .map(|i| (100.0 + 0.5 * f64::from(i), 10.0 + 0.48 * f64::from(i)))
            .collect();
        let map = TimeMap::new(&pairs, 8.0);
        for &(a, b) in &pairs {
            assert!((map.at(a) - b).abs() < 1e-4, "{a}: {} {b}", map.at(a));
        }
        assert!((map.speed(105.0) - 0.96).abs() < 1e-9);
        assert!((map.typical_speed() - 0.96).abs() < 1e-9);
        // Then it glides to speed 1 and stays.
        let end = pairs[31].0;
        assert!((map.speed(end + 4.0) - 0.98).abs() < 1e-9);
        assert_eq!(map.speed(end + 9.0), 1.0);
        assert!((map.at(end + 10.0) - map.at(end + 9.0) - 1.0).abs() < 1e-9);
        // Continuous where the glide starts and ends.
        for t in [end, end + 8.0] {
            assert!((map.at(t + 1e-6) - map.at(t - 1e-6)).abs() < 1e-5);
        }
    }

    #[test]
    fn a_glide_covers_its_span_from_speed_one_to_its_end() {
        // 16 s of the playing track from 100 s, ending 5 % faster.
        let g = Glide::new(100.0, 16.0, 1.05);
        assert!((g.length() - 32.0 / 2.05).abs() < 1e-12);
        assert_eq!(g.at(90.0), 90.0);
        assert_eq!(g.speed(99.0), 1.0);
        assert!((g.speed(100.0 + g.length()) - 1.05).abs() < 1e-12);
        assert!((g.at(100.0 + g.length()) - 116.0).abs() < 1e-9);
        // After it, at its end speed.
        let end = 100.0 + g.length();
        assert!((g.at(end + 2.0) - 116.0 - 2.1).abs() < 1e-9);
        // The speed is the slope of the time, and the inverse undoes it.
        for k in 0..40 {
            let u = 99.0 + 0.5 * f64::from(k);
            let slope = (g.at(u + 1e-5) - g.at(u - 1e-5)) / 2e-5;
            assert!((slope - g.speed(u)).abs() < 1e-6, "{u}");
            assert!((g.inverse(g.at(u)) - u).abs() < 1e-9, "{u}");
        }
        // Its map agrees with it, also long after it.
        let m = g.map();
        assert!((m.at(end + 100.0) - g.at(end + 100.0)).abs() < 1e-6);
        assert!((m.speed(end + 100.0) - 1.05).abs() < 1e-9);
        for k in 0..60 {
            let u = 99.5 + 0.3 * f64::from(k);
            assert!((m.at(u) - g.at(u)).abs() < 1e-6, "{u}");
        }
        // Near 1 is none.
        assert!(Glide::new(0.0, 8.0, 1.001).is_identity());
    }

    #[test]
    fn through_a_glide_the_incoming_track_ends_at_its_own_tempo() {
        // The playing track at 120 BPM, the incoming at 126: locked, the
        // incoming plays at 0.952 of its speed; the glide takes the playing
        // one to 1.05, so the incoming ends at speed 1.
        let pairs: Vec<(f64, f64)> = (0..33)
            .map(|i| {
                (
                    100.0 + 0.5 * f64::from(i),
                    10.0 + (0.5 / 1.05) * f64::from(i),
                )
            })
            .collect();
        let map = TimeMap::new(&pairs, 1.0);
        let g = Glide::new(100.0, 16.0, 1.0 / map.typical_speed());
        let through = map.through_glide(&g);
        for &(a, b) in &pairs {
            // Each incoming beat where its playing beat is heard.
            assert!((through.at(g.inverse(a)) - b).abs() < 1e-4, "{a}");
        }
        let end = g.start() + g.length();
        assert!(
            (through.speed(end) - 1.0).abs() < 1e-6,
            "{}",
            through.speed(end)
        );
        assert!((through.speed(99.0) - 1.0 / 1.05).abs() < 1e-6);
        // No glide: the map as it is.
        assert_eq!(map.through_glide(&Glide::new(100.0, 16.0, 1.0)), map);
    }

    #[test]
    fn jitter_in_the_beats_does_not_wobble_the_speed() {
        // The incoming beats jitter by up to ±10 ms around a steady grid.
        let jitter = [0.008, -0.01, 0.004, -0.006, 0.01, -0.002, 0.006, -0.008];
        let pairs: Vec<(f64, f64)> = (0..40)
            .map(|i| {
                let j = jitter[i % jitter.len()];
                (0.5 * i as f64, 0.48 * i as f64 + j)
            })
            .collect();
        let map = TimeMap::new(&pairs, 0.0);
        let speeds: Vec<f64> = (40..300).map(|k| map.speed(k as f64 * 0.05)).collect();
        let (lo, hi) = speeds
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &s| (l.min(s), h.max(s)));
        assert!(hi - lo < 0.02, "{lo}..{hi}");
        // The beats stay within the jitter of where they are.
        for &(a, b) in &pairs[5..35] {
            assert!((map.at(a) - b).abs() < 0.012);
        }
    }
}
