//! The meters' ballistics, as a desk's peak meters move: up at once to a
//! peak, down at [`FALL_DB_PER_S`], a peak-hold tick that stays
//! [`HOLD_MS`] before it falls too, and a clip lamp. They move by the
//! frame times the app gives ([`super::feed`]), so they fall as fast
//! however often frames come.

/// The bottom of a meter's range, in dBFS (0 dBFS is its top).
pub const FLOOR_DB: f32 = -60.0;

/// How fast a meter falls, in dB a second.
pub const FALL_DB_PER_S: f32 = 20.0;

/// How long the peak-hold tick stays before it falls, in milliseconds.
pub const HOLD_MS: u64 = 1_500;

/// How long the clip lamp stays lit after a peak at full scale.
pub const CLIP_HOLD_MS: u64 = 1_500;

/// A peak this near full scale lights the clip lamp: within a thousandth
/// of a dB (a 16-bit sample at full scale is).
pub const CLIP_AT: f32 = 0.9999;

/// Where the meter's colours change, in dBFS: green below the first,
/// yellow to the second, red above.
pub const ZONES: (f32, f32) = (-18.0, -6.0);

/// One meter's needle (a bar of light): its level, its peak-hold tick and
/// its clip lamp, in dBFS (−∞: nothing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meter {
    /// The level shown.
    pub db: f32,
    /// The peak-hold tick.
    pub peak_db: f32,
    /// When the tick was last raised.
    peak_at: u64,
    /// When a peak last reached full scale.
    clip_at: Option<u64>,
    /// When it last moved.
    at: Option<u64>,
}

impl Default for Meter {
    fn default() -> Self {
        Meter {
            db: f32::NEG_INFINITY,
            peak_db: f32::NEG_INFINITY,
            peak_at: 0,
            clip_at: None,
            at: None,
        }
    }
}

impl Meter {
    /// Takes the peak since the last look (`None`: nothing came) at
    /// `now_ms`: the bar rises to it at once, else falls.
    pub fn take(&mut self, peak: Option<f32>, now_ms: u64) {
        let dt = self
            .at
            .map_or(0.0, |at| now_ms.saturating_sub(at) as f32 / 1000.0);
        self.at = Some(now_ms);
        let fall = FALL_DB_PER_S * dt;
        let new = peak.map_or(f32::NEG_INFINITY, |p| {
            super::taper::db_of_gain(f64::from(p)) as f32
        });
        let fallen = self.db - fall;
        self.db = if new > fallen { new } else { fallen };
        if self.db < FLOOR_DB - 1.0 {
            self.db = f32::NEG_INFINITY;
        }
        if new >= self.peak_db {
            self.peak_db = new;
            self.peak_at = now_ms;
        } else if now_ms.saturating_sub(self.peak_at) > HOLD_MS {
            self.peak_db = (self.peak_db - fall).max(self.db);
        }
        if peak.is_some_and(|p| p >= CLIP_AT) {
            self.clip_at = Some(now_ms);
        }
    }

    /// Whether the clip lamp is lit at `now_ms`.
    pub fn clipped(&self, now_ms: u64) -> bool {
        self.clip_at
            .is_some_and(|at| now_ms.saturating_sub(at) < CLIP_HOLD_MS)
    }

    /// The share of the meter's height lit by `db` (0: the floor or below,
    /// 1: full scale or above).
    pub fn share(db: f32) -> f32 {
        if db == f32::NEG_INFINITY {
            return 0.0;
        }
        ((db - FLOOR_DB) / -FLOOR_DB).clamp(0.0, 1.0)
    }
}

/// A pair of meters (left and right; or one channel twice for mono).
pub type Pair = [Meter; 2];

/// `pair` takes `peaks` (one per channel; none: nothing came): the first
/// two channels, a single one on both.
pub fn take_pair(pair: &mut Pair, peaks: &[f32], now_ms: u64) {
    let left = peaks.first().copied();
    let right = peaks.get(1).copied().or(left);
    pair[0].take(left, now_ms);
    pair[1].take(right, now_ms);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_meter_rises_at_once_and_falls_at_its_rate() {
        let mut m = Meter::default();
        m.take(Some(1.0), 0);
        assert_eq!(m.db, 0.0);
        assert!(m.clipped(0));
        // Half a second later, nothing: 10 dB down.
        m.take(None, 500);
        assert!((m.db + 10.0).abs() < 1e-4, "{}", m.db);
        // The tick holds for 1.5 s, then falls (never below the bar).
        assert_eq!(m.peak_db, 0.0);
        m.take(None, 1_500);
        assert_eq!(m.peak_db, 0.0);
        m.take(None, 1_600);
        assert!(m.peak_db < 0.0 && m.peak_db >= m.db, "{m:?}");
        assert!(m.clipped(1_400) && !m.clipped(1_600));
        // A louder peak takes the bar up again at once.
        m.take(Some(0.5), 1_700);
        assert!((m.db + 6.0206).abs() < 1e-3, "{}", m.db);
        // Long after, it is out.
        m.take(None, 10_000);
        assert_eq!(m.db, f32::NEG_INFINITY);
    }

    #[test]
    fn the_range_is_sixty_db_and_mono_goes_on_both() {
        assert_eq!(Meter::share(0.0), 1.0);
        assert_eq!(Meter::share(-30.0), 0.5);
        assert_eq!(Meter::share(-90.0), 0.0);
        assert_eq!(Meter::share(f32::NEG_INFINITY), 0.0);
        let mut pair = Pair::default();
        take_pair(&mut pair, &[0.25], 0);
        assert_eq!(pair[0].db, pair[1].db);
    }
}
