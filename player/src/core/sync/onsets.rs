//! The starts of sounds in the voice heard (a "ta", a clap), for the
//! sync: [`Onsets`].

use std::collections::VecDeque;

/// A hop of the level the starts of sounds are found on, in seconds.
const HOP: f64 = 0.001;
/// The hops the level is the mean of.
const SMOOTH: usize = 3;
/// How far above the room's level a sound must rise to start, in dB.
const RISE_DB: f64 = 15.0;
/// How loud it must be at least, in dBFS.
const LOUD_DB: f64 = -50.0;
/// How far back towards the room's level it must fall before another can
/// start, in dB above it.
const REARM_DB: f64 = 9.0;
/// The least time between two starts, in hops.
const REFRACTORY: u32 = 60;
/// How long after its start a sound's strength (its peak) is taken, in
/// hops.
const PEAK_HOPS: u32 = 40;
/// How fast the room's level follows a louder level, a share of the way
/// a hop, in dB (it follows a quieter one at once).
const FLOOR_RISE: f64 = 0.001;

/// The starts of sounds in the voice heard (a "ta", a clap): where its
/// level, a mean over [`SMOOTH`] hops of [`HOP`], rises [`RISE_DB`] over
/// the room's and above [`LOUD_DB`], each at the stream frame of the first
/// hop over that, with its strength (its peak level in dB over the
/// [`PEAK_HOPS`] after).
#[derive(Clone, Debug)]
pub struct Onsets {
    /// The hop being summed: its first stream frame, the sum of its
    /// samples squared, how many.
    hop: Option<(f64, f64, usize)>,
    /// The last hops: their first frames and levels in dB.
    recent: VecDeque<(f64, f64)>,
    /// The room's level, in dB.
    floor: Option<f64>,
    /// Whether a start may come (the level fell back since the last).
    armed: bool,
    /// Hops left before one may.
    refractory: u32,
    /// A start being measured: its frame, its peak so far, hops left.
    rising: Option<(u64, f64, u32)>,
    /// The starts found: their stream frames and strengths.
    pub found: Vec<(u64, f32)>,
}

impl Default for Onsets {
    fn default() -> Self {
        Onsets {
            hop: None,
            recent: VecDeque::new(),
            floor: None,
            armed: true,
            refractory: 0,
            rising: None,
            found: Vec::new(),
        }
    }
}

/// `x` (a mean square) in dB.
fn db(x: f64) -> f64 {
    10.0 * x.max(1e-12).log10()
}

impl Onsets {
    /// A block of the voice: `samples` at `rate`, its first sample at
    /// stream frame `frame`, `per_sample` stream frames a sample;
    /// `continues`: right after the block before.
    pub fn push(
        &mut self,
        frame: u64,
        per_sample: f64,
        rate: u32,
        continues: bool,
        samples: &[f32],
    ) {
        if !continues {
            self.hop = None;
        }
        let hop_len = ((f64::from(rate) * HOP).round() as usize).max(1);
        for (i, &x) in samples.iter().enumerate() {
            let at = frame as f64 + i as f64 * per_sample;
            let (start, sum, n) = self.hop.get_or_insert((at, 0.0, 0));
            *sum += f64::from(x) * f64::from(x);
            *n += 1;
            if *n >= hop_len {
                let (start, level) = (*start, db(*sum / *n as f64));
                self.hop = None;
                self.hear(start, level);
            }
        }
    }

    /// A hop starting at stream frame `start` at `level` dB.
    fn hear(&mut self, start: f64, level: f64) {
        self.recent.push_back((start, level));
        if self.recent.len() > SMOOTH {
            self.recent.pop_front();
        }
        let mean = self
            .recent
            .iter()
            .map(|r| 10f64.powf(r.1 / 10.0))
            .sum::<f64>()
            / self.recent.len() as f64;
        let now = db(mean);
        let floor = *self.floor.get_or_insert(now);
        if let Some((at, peak, left)) = &mut self.rising {
            *peak = peak.max(now);
            *left -= 1;
            if *left == 0 {
                self.found.push((*at, *peak as f32));
                self.rising = None;
            }
        }
        self.refractory = self.refractory.saturating_sub(1);
        if !self.armed && now < floor + REARM_DB {
            self.armed = true;
        }
        let threshold = (floor + RISE_DB).max(LOUD_DB);
        if self.armed && self.refractory == 0 && self.rising.is_none() && now > threshold {
            // At the first of the last hops that is over it alone.
            let at = self
                .recent
                .iter()
                .find(|r| r.1 > threshold)
                .map_or(start, |r| r.0);
            self.rising = Some((at.round() as u64, now, PEAK_HOPS));
            self.armed = false;
            self.refractory = REFRACTORY;
        }
        self.floor = Some(if now < floor {
            now
        } else {
            floor + (now - floor) * FLOOR_RISE
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A "ta" at each of `starts` (seconds) in a quiet room, at 48 kHz:
    /// noise at −70 dBFS, then a 150 ms burst at −15 dBFS.
    fn voice(starts: &[f64], seconds: f64) -> Vec<f32> {
        let n = (seconds * 48_000.0) as usize;
        let mut seed = 1u32;
        let mut noise = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f64::from(seed) / f64::from(u32::MAX) - 0.5) as f32
        };
        (0..n)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                let loud = starts.iter().any(|&s| t >= s && t < s + 0.15);
                let a = if loud { 0.25 } else { 0.0006 };
                a * noise() * 2.0
            })
            .collect()
    }

    #[test]
    fn the_start_of_each_ta_is_found_to_the_millisecond() {
        let starts = [1.15, 1.81, 2.49, 3.13];
        let samples = voice(&starts, 4.0);
        let mut onsets = Onsets::default();
        // In blocks of 128, the stream at twice the voice's rate (96 kHz).
        for (k, block) in samples.chunks(128).enumerate() {
            onsets.push((k * 256) as u64, 2.0, 48_000, true, block);
        }
        // One more block to let the last one's strength be taken.
        onsets.push((samples.len() * 2) as u64, 2.0, 48_000, true, &[0.0; 4800]);
        let found: Vec<f64> = onsets.found.iter().map(|f| f.0 as f64 / 96_000.0).collect();
        assert_eq!(found.len(), starts.len(), "{found:?}");
        for (f, s) in found.iter().zip(starts) {
            assert!((f - s).abs() < 0.002, "{f} for {s}");
        }
    }
}
