//! A whole track's loudness as ITU-R BS.1770-4 measures it: its
//! integrated loudness, in LUFS.
//!
//! Each channel is K-weighted (two biquads: a high shelf of about +4 dB
//! above 1.5 kHz, the head's effect, then a high pass at about 38 Hz, the
//! RLB curve), its square summed over blocks of 400 ms that start every
//! 100 ms (three quarters overlapped), the two channels' added (each
//! weighed 1). A block's loudness is `-0.691 + 10 log10` of that mean
//! square. Blocks under -70 LUFS are gated out (the absolute gate), then
//! those more than 10 LU under the loudness of the rest (the relative
//! gate); the mean square of those left is the track's loudness. A
//! stereo sine at 1 kHz peaking at -20 dBFS is -20 LUFS.
//!
//! The filters are those of the standard's table at 48 kHz, made for any
//! rate from their analogue prototype (its corner, gain and Q) by the
//! bilinear transform: at 48 kHz they are the table's to six places. A
//! mono track is measured as it is heard, on both sides.

use crate::biquad::Biquad;

/// The gate under which a block is not counted at all, LUFS.
pub const ABSOLUTE_GATE: f64 = -70.0;
/// How far under the loudness of the blocks over [`ABSOLUTE_GATE`] a block
/// is not counted either, LU.
pub const RELATIVE_GATE: f64 = 10.0;
/// The steps a block takes (400 ms in steps of 100 ms).
const STEPS: usize = 4;

/// The K-weighting at `rate` Hz: the shelf, then the high pass.
fn k_weighting(rate: f64) -> [Biquad<f64>; 2] {
    // The shelf: its corner (Hz), gain (dB) and Q, fitted to the table.
    let (f0, gain, q) = (
        1_681.974_450_955_533,
        3.999_843_853_973_347,
        0.707_175_236_955_419_6,
    );
    let k = (std::f64::consts::PI * f0 / rate).tan();
    let vh = 10f64.powf(gain / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad::new(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    );
    // The high pass.
    let (f0, q) = (38.135_470_876_024_44, 0.500_327_037_323_877_3);
    let k = (std::f64::consts::PI * f0 / rate).tan();
    let a0 = 1.0 + k / q + k * k;
    let high_pass = Biquad::new(
        [1.0, -2.0, 1.0],
        [2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
    );
    [shelf, high_pass]
}

/// Loudness from a mean square (the channels' summed).
fn lufs(mean_square: f64) -> f64 {
    -0.691 + 10.0 * mean_square.log10()
}

/// A loudness meter over a whole track, fed stereo at one rate.
#[derive(Clone, Debug)]
pub struct Meter {
    filters: [[Biquad<f64>; 2]; 2],
    /// Frames a step takes, and those of the step under way so far.
    step: usize,
    filled: usize,
    /// The K-weighted squares of the step under way, the channels' summed.
    sum: f64,
    /// Each step's summed squares, in order.
    steps: Vec<f64>,
}

impl Meter {
    /// A meter for stereo at `rate` Hz.
    pub fn new(rate: u32) -> Meter {
        let rate = rate.max(1);
        let weighting = k_weighting(f64::from(rate));
        Meter {
            filters: [weighting, weighting],
            step: ((f64::from(rate) / 10.0).round() as usize).max(1),
            filled: 0,
            sum: 0.0,
            steps: Vec::new(),
        }
    }

    /// The next frames, the two sides (the shorter's length of each).
    pub fn push(&mut self, left: &[f32], right: &[f32]) {
        for (&l, &r) in left.iter().zip(right) {
            let [fl, fr] = &mut self.filters;
            let l = fl.iter_mut().fold(f64::from(l), |x, f| f.run(x));
            let r = fr.iter_mut().fold(f64::from(r), |x, f| f.run(x));
            self.sum += l * l + r * r;
            self.filled += 1;
            if self.filled == self.step {
                self.steps.push(self.sum);
                self.sum = 0.0;
                self.filled = 0;
            }
        }
    }

    /// The integrated loudness of all pushed so far, LUFS (a last part
    /// step not counted); `None` when no block is over the gates (silence,
    /// or less than 400 ms).
    pub fn integrated(&self) -> Option<f64> {
        let frames = (STEPS * self.step) as f64;
        let blocks: Vec<f64> = self
            .steps
            .windows(STEPS)
            .map(|w| w.iter().sum::<f64>() / frames)
            .filter(|&z| z > 0.0 && lufs(z) > ABSOLUTE_GATE)
            .collect();
        let mean = |zs: &mut dyn Iterator<Item = f64>| {
            let (sum, n) = zs.fold((0.0, 0usize), |(s, n), z| (s + z, n + 1));
            (n > 0).then(|| sum / n as f64)
        };
        let relative = lufs(mean(&mut blocks.iter().copied())?) - RELATIVE_GATE;
        let gated = mean(&mut blocks.iter().copied().filter(|&z| lufs(z) > relative))?;
        Some(lufs(gated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `seconds` of a stereo sine of `hz` at `dbfs` (its peak), into `meter`.
    fn sine(meter: &mut Meter, rate: u32, hz: f64, dbfs: f64, seconds: f64) {
        let amplitude = 10f64.powf(dbfs / 20.0);
        let n = (seconds * f64::from(rate)) as usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                (amplitude * (std::f64::consts::TAU * hz * t).sin()) as f32
            })
            .collect();
        meter.push(&x, &x);
    }

    #[test]
    fn the_filters_are_the_tables_at_48_khz() {
        let [shelf, high_pass] = k_weighting(48_000.0);
        let table_shelf = (
            [
                1.535_124_859_586_97,
                -2.691_696_189_406_38,
                1.198_392_810_852_85,
            ],
            [-1.690_659_293_182_41, 0.732_480_774_215_85],
        );
        let table_high_pass = (
            [1.0, -2.0, 1.0],
            [-1.990_047_454_833_98, 0.990_072_250_366_21],
        );
        for (f, (b, a)) in [(shelf, table_shelf), (high_pass, table_high_pass)] {
            let (fb, fa) = f.coefficients();
            for (x, y) in fb.iter().zip(b).chain(fa.iter().zip(a)) {
                assert!((x - y).abs() < 1e-6, "{x} against the table's {y}");
            }
        }
    }

    /// A stereo sine at 1 kHz peaking at -20 dBFS is -20 LUFS (the
    /// standard's -0.691 is the K-weighting's gain there), at the rates a
    /// track comes in.
    #[test]
    fn a_sine_at_minus_20_dbfs_is_minus_20_lufs() {
        for rate in [44_100, 48_000, 96_000] {
            let mut meter = Meter::new(rate);
            sine(&mut meter, rate, 1_000.0, -20.0, 20.0);
            let l = meter.integrated().unwrap();
            assert!((l + 20.0).abs() < 0.05, "{l} LUFS at {rate} Hz");
        }
    }

    /// The gates: quiet stretches around a loud one count not at all when
    /// more than 10 LU under it (EBU Tech 3341's third case), and silence
    /// is no loudness.
    #[test]
    fn quiet_stretches_are_gated_out() {
        let rate = 48_000;
        let mut meter = Meter::new(rate);
        sine(&mut meter, rate, 1_000.0, -36.0, 10.0);
        sine(&mut meter, rate, 1_000.0, -23.0, 60.0);
        sine(&mut meter, rate, 1_000.0, -36.0, 10.0);
        let l = meter.integrated().unwrap();
        assert!((l + 23.0).abs() < 0.1, "{l} LUFS");
        // Under 10 LU down they count (its fifth: -26, -20, -26 dBFS).
        let mut meter = Meter::new(rate);
        sine(&mut meter, rate, 1_000.0, -26.0, 20.0);
        sine(&mut meter, rate, 1_000.0, -20.0, 20.1);
        sine(&mut meter, rate, 1_000.0, -26.0, 20.0);
        let l = meter.integrated().unwrap();
        assert!((l + 23.0).abs() < 0.1, "{l} LUFS");
        let mut silent = Meter::new(rate);
        silent.push(&[0.0; 48_000], &[0.0; 48_000]);
        assert_eq!(silent.integrated(), None);
        assert_eq!(Meter::new(rate).integrated(), None);
    }
}
