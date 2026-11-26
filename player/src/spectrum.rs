//! The spectrum of what is heard, from the engine's [`SpectrumTap`].
//!
//! Each update reads the most recent output frames that end at the frame
//! the DAC plays now, applies a Hann window to their mid signal
//! ((L + R) / 2) and takes two power spectra: one of about 22 Hz bins for
//! the whole range, and one eight times longer (about 2.7 Hz bins) for the
//! bass below [`FINE_UP_TO_HZ`], where a log axis spreads few bins over
//! many columns. The block lengths scale with the output rate, so every
//! rate has about the same resolution in hertz. A column shows the mean
//! power of the bins it spans (the level density there, not its loudest
//! bin); a column narrower than a bin reads the level between the two
//! nearest bins. Levels rise at once and fall at [`FALL_DB_PER_SECOND`],
//! as a level meter does.

use std::sync::Arc;

use jodel_dsp::fft::{Complex, Fft};
use jodel_dsp::window::hann;
use jodel_dsp::FULL_SCALE;
use jodel_playback::{SpectrumTap, TapFrame, TapMiss};

/// Hertz per analysis bin, about the same for every rate.
const BIN_HZ: f64 = 22.0;
/// The fine spectrum's block is this many times longer than the main one.
const FINE_FACTOR: usize = 8;
/// Columns below this frequency read the fine spectrum.
pub const FINE_UP_TO_HZ: f64 = 500.0;
/// The tap frames the analyser needs at `rate`: the fine block.
pub fn tap_frames_for(rate: u32) -> usize {
    Analyzer::block_for(rate) * FINE_FACTOR
}
/// The lowest frequency shown.
pub const LOW_HZ: f64 = 20.0;
/// The level range shown, in dB relative to a full-scale sine.
pub const RANGE_DB: (f32, f32) = (-160.0, 0.0);
/// How fast a level falls once the sound is gone.
pub const FALL_DB_PER_SECOND: f32 = 90.0;

/// A spectrum to draw.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpectrumFrame {
    /// The output rate it was taken at (0: nothing yet).
    pub rate: u32,
    /// The level of each FFT bin in dB, from 0 Hz to the Nyquist frequency.
    pub db: Arc<[f32]>,
    /// The fine spectrum's bins in dB, from 0 Hz up to
    /// [`FINE_UP_TO_HZ`] (and a little beyond), each `fine_hz` wide.
    pub fine: Arc<[f32]>,
    /// The width of a fine bin in hertz (0: no fine spectrum).
    pub fine_hz: f64,
}

impl SpectrumFrame {
    /// The Nyquist frequency.
    pub fn nyquist(&self) -> f64 {
        f64::from(self.rate) / 2.0
    }

    /// Whether there is anything to draw.
    pub fn is_empty(&self) -> bool {
        self.rate == 0 || self.db.len() < 2
    }

    /// The frequency at position `x` of `n` on the log axis from
    /// [`LOW_HZ`] to the Nyquist frequency.
    pub fn hz_at(&self, x: f64, n: usize) -> f64 {
        let (lo, hi) = (LOW_HZ.log10(), self.nyquist().max(LOW_HZ * 2.0).log10());
        10f64.powf(lo + (hi - lo) * x / n.max(1) as f64)
    }

    /// Where `hz` falls on an axis of `n` positions (may be outside).
    pub fn x_of(&self, hz: f64, n: usize) -> f64 {
        let (lo, hi) = (LOW_HZ.log10(), self.nyquist().max(LOW_HZ * 2.0).log10());
        (hz.max(1.0).log10() - lo) / (hi - lo) * n as f64
    }

    /// `n` levels on the log frequency axis, one per equal step from
    /// [`LOW_HZ`] to the Nyquist frequency: the mean power of the bins
    /// whose centres fall in the step, or, for a step narrower than a bin,
    /// the level between the two nearest bins (so the bass reads as a
    /// curve, not as steps). Steps below [`FINE_UP_TO_HZ`] use the fine
    /// spectrum.
    pub fn levels(&self, n: usize) -> Vec<f32> {
        if self.is_empty() || n == 0 {
            return vec![RANGE_DB.0; n];
        }
        let main_hz = self.nyquist() / (self.db.len() - 1) as f64;
        (0..n)
            .map(|i| {
                let from = self.hz_at(i as f64, n);
                let to = self.hz_at(i as f64 + 1.0, n);
                let fine = self.fine_hz > 0.0
                    && to <= FINE_UP_TO_HZ
                    && to / self.fine_hz < (self.fine.len() - 1) as f64;
                if fine {
                    band_level(&self.fine, self.fine_hz, from, to)
                } else {
                    band_level(&self.db, main_hz, from, to)
                }
            })
            .collect()
    }
}

/// The level of the band `from`..`to` Hz in `bins` (dB, `bin_hz` apart,
/// bin 0 at 0 Hz): the mean power of the bins centred in the band, else
/// the level interpolated (in dB) at the band's centre.
fn band_level(bins: &[f32], bin_hz: f64, from: f64, to: f64) -> f32 {
    let last = bins.len() - 1;
    let first = ((from / bin_hz).ceil() as usize).max(1);
    let end = ((to / bin_hz).ceil() as usize).min(last + 1);
    if first < end {
        let power: f64 = bins[first..end]
            .iter()
            .map(|&db| 10f64.powf(f64::from(db) / 10.0))
            .sum::<f64>()
            / (end - first) as f64;
        return ((10.0 * power.max(1e-30).log10()) as f32).max(RANGE_DB.0);
    }
    let at = ((from * to).sqrt() / bin_hz).clamp(1.0, last as f64);
    let below = at.floor() as usize;
    let above = (below + 1).min(last);
    let t = (at - below as f64) as f32;
    bins[below] * (1.0 - t) + bins[above] * t
}

/// The analyser: keeps the last spectrum and lets it fall.
pub struct Analyzer {
    rate: u32,
    size: usize,
    window: Vec<f64>,
    window_sum: f64,
    fft: Option<Fft<f64>>,
    fine_window: Vec<f64>,
    fine_window_sum: f64,
    fine_fft: Option<Fft<f64>>,
    frames: Vec<TapFrame>,
    db: Vec<f32>,
    fine: Vec<f32>,
}

impl Default for Analyzer {
    fn default() -> Self {
        Analyzer::new()
    }
}

impl Analyzer {
    /// An analyser with nothing seen yet.
    pub fn new() -> Analyzer {
        Analyzer {
            rate: 0,
            size: 0,
            window: Vec::new(),
            window_sum: 1.0,
            fft: None,
            fine_window: Vec::new(),
            fine_window_sum: 1.0,
            fine_fft: None,
            frames: Vec::new(),
            db: Vec::new(),
            fine: Vec::new(),
        }
    }

    /// The block length for a rate: a power of two near `rate / 22`.
    pub fn block_for(rate: u32) -> usize {
        (f64::from(rate.max(1_000)) / BIN_HZ).log2().round().exp2() as usize
    }

    fn set_rate(&mut self, rate: u32) {
        if rate == self.rate {
            return;
        }
        let size = Analyzer::block_for(rate);
        self.rate = rate;
        self.size = size;
        self.window = hann(size);
        self.window_sum = self.window.iter().sum();
        self.fft = Some(Fft::new(size));
        let fine_size = size * FINE_FACTOR;
        self.fine_window = hann(fine_size);
        self.fine_window_sum = self.fine_window.iter().sum();
        self.fine_fft = Some(Fft::new(fine_size));
        self.frames = vec![[0, 0]; fine_size];
        self.db = vec![RANGE_DB.0; size / 2 + 1];
        // The fine bins up to a little past where the main ones take over.
        let fine_hz = f64::from(rate) / fine_size as f64;
        let fine_bins = ((FINE_UP_TO_HZ * 1.5) / fine_hz).ceil() as usize + 2;
        self.fine = vec![RANGE_DB.0; fine_bins.min(fine_size / 2 + 1)];
    }

    /// Reads the frames heard now from `tap` and analyses them; `seconds`
    /// is the time since the last update (for the fall). With nothing
    /// playing the levels just fall.
    pub fn update(&mut self, tap: &SpectrumTap, seconds: f32) {
        let Some(cursor) = tap.heard() else {
            self.fall(seconds);
            return;
        };
        self.set_rate(cursor.rate);
        let mut frames = std::mem::take(&mut self.frames);
        let mut result = tap.read_heard(&mut frames);
        if result == Err(TapMiss::Overwritten) {
            result = tap.read_heard(&mut frames);
        }
        match result {
            Ok(_) => self.analyse(cursor.rate, &frames, seconds),
            Err(_) => self.fall(seconds),
        }
        self.frames = frames;
    }

    /// Forgets the rate and the levels: nothing seen yet, as at the start
    /// (so the axis is no longer labelled with the last track's rate).
    pub fn reset(&mut self) {
        if self.rate != 0 {
            *self = Analyzer::new();
        }
    }

    fn fall(&mut self, seconds: f32) {
        let drop = FALL_DB_PER_SECOND * seconds.max(0.0);
        for level in self.db.iter_mut().chain(self.fine.iter_mut()) {
            *level = (*level - drop).max(RANGE_DB.0);
        }
    }

    /// Analyses `frames` (the newest last) at `rate`: the main spectrum
    /// from the newest main block, the fine one from the newest fine block
    /// (missing frames count as silence).
    pub fn analyse(&mut self, rate: u32, frames: &[TapFrame], seconds: f32) {
        self.set_rate(rate);
        let drop = FALL_DB_PER_SECOND * seconds.max(0.0);
        if let Some(fft) = &self.fft {
            spectrum_into(
                frames,
                &self.window,
                self.window_sum,
                fft,
                &mut self.db,
                drop,
            );
        }
        if let Some(fft) = &self.fine_fft {
            spectrum_into(
                frames,
                &self.fine_window,
                self.fine_window_sum,
                fft,
                &mut self.fine,
                drop,
            );
        }
    }

    /// Whether every level is at the bottom of the range (nothing to
    /// animate).
    pub fn is_quiet(&self) -> bool {
        self.db.iter().all(|&d| d <= RANGE_DB.0)
    }

    /// The spectrum to draw now.
    pub fn frame(&self) -> SpectrumFrame {
        SpectrumFrame {
            rate: self.rate,
            db: self.db.clone().into(),
            fine: self.fine.clone().into(),
            fine_hz: if self.fine.is_empty() {
                0.0
            } else {
                f64::from(self.rate) / (self.size * FINE_FACTOR) as f64
            },
        }
    }
}

/// The power spectrum of the newest `window.len()` of `frames` into
/// `levels` (as many bins as it holds), each rising at once and falling by
/// at most `drop` dB.
fn spectrum_into(
    frames: &[TapFrame],
    window: &[f64],
    window_sum: f64,
    fft: &Fft<f64>,
    levels: &mut [f32],
    drop: f32,
) {
    // The transform's buffer is kept between calls (up to 4 MB at the
    // highest rates).
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<Complex<f64>>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|scratch| {
        let bins = &mut *scratch.borrow_mut();
        let size = window.len();
        bins.clear();
        bins.resize(size, Complex::default());
        let start = frames.len().saturating_sub(size);
        let offset = size - (frames.len() - start);
        for (i, &[l, r]) in frames[start..].iter().enumerate() {
            let x = (f64::from(l) + f64::from(r)) / (2.0 * FULL_SCALE);
            bins[offset + i].re = x * window[offset + i];
        }
        fft.forward(bins);
        let scale = 2.0 / window_sum;
        for (bin, level) in levels.iter_mut().enumerate() {
            let z = bins[bin];
            let power = (z.re * z.re + z.im * z.im) * scale * scale;
            let db = (10.0 * power.max(1e-30).log10()) as f32;
            let db = db.clamp(RANGE_DB.0, RANGE_DB.1 + 6.0);
            *level = db.max(*level - drop);
        }
    });
}

/// A test signal: a sine at `hz` and `amplitude` (of full scale).
pub fn sine(rate: u32, hz: f64, amplitude: f64, frames: usize) -> Vec<TapFrame> {
    (0..frames)
        .map(|i| {
            let x = amplitude
                * FULL_SCALE
                * (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin();
            [x as i32, x as i32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_sine_peaks_at_its_frequency() {
        let mut analyzer = Analyzer::new();
        let rate = 96_000;
        analyzer.analyse(rate, &sine(rate, 10_000.0, 0.5, tap_frames_for(rate)), 0.0);
        let frame = analyzer.frame();
        let levels = frame.levels(200);
        let peak = levels
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let hz = frame.hz_at(peak.0 as f64 + 0.5, 200);
        assert!((8_500.0..11_500.0).contains(&hz), "{hz}");
        // A column at 10 kHz spans several bins, so the tone's -6 dB is
        // shared with its quieter neighbours.
        assert!((-20.0..=-5.0).contains(peak.1), "{}", peak.1);
        // Far from the tone the level is low.
        assert!(levels[10] < -90.0, "{}", levels[10]);
    }

    #[test]
    fn the_bass_reads_from_the_fine_spectrum() {
        let rate = 192_000;
        let mut analyzer = Analyzer::new();
        analyzer.analyse(rate, &sine(rate, 60.0, 0.5, tap_frames_for(rate)), 0.0);
        let frame = analyzer.frame();
        assert!(
            frame.fine_hz > 2.0 && frame.fine_hz < 4.0,
            "{}",
            frame.fine_hz
        );
        let n = 400;
        let levels = frame.levels(n);
        let at = |hz: f64| levels[frame.x_of(hz, n) as usize];
        // The tone reads near its level (-6 dB) at 60 Hz, and 30 and
        // 120 Hz are far below it: the main spectrum's 22 Hz bins would
        // smear it over that whole range.
        assert!((-9.0..=-5.0).contains(&at(60.0)), "{}", at(60.0));
        assert!(at(30.0) < -60.0, "{}", at(30.0));
        assert!(at(120.0) < -60.0, "{}", at(120.0));
        // Neighbouring bass columns change smoothly: no stairs.
        let jumps = (frame.x_of(25.0, n) as usize..frame.x_of(45.0, n) as usize)
            .map(|i| (levels[i + 1] - levels[i]).abs())
            .fold(0.0, f32::max);
        assert!(jumps < 12.0, "{jumps}");
    }

    #[test]
    fn noise_reads_flat_not_rising() {
        // White noise: equal power per hertz. Mean power per column keeps
        // it level across the axis; the loudest bin per column would make
        // the wide high columns read several dB louder.
        let rate = 48_000;
        let mut seed = 1u32;
        let frames: Vec<TapFrame> = (0..tap_frames_for(rate))
            .map(|_| {
                let x = (jodel_dsp::xorshift32(&mut seed) as i32) >> 10;
                [x, x]
            })
            .collect();
        let mut analyzer = Analyzer::new();
        analyzer.analyse(rate, &frames, 0.0);
        let frame = analyzer.frame();
        let n = 200;
        let levels = frame.levels(n);
        let at = |hz: f64| levels[frame.x_of(hz, n) as usize];
        let (mid, high) = (at(2_000.0), at(15_000.0));
        assert!((high - mid).abs() < 4.0, "{mid} {high}");
    }

    #[test]
    fn levels_fall_when_the_sound_stops() {
        let mut analyzer = Analyzer::new();
        analyzer.analyse(48_000, &sine(48_000, 1_000.0, 0.9, 4_096), 0.0);
        let before = analyzer.frame().levels(50);
        analyzer.analyse(48_000, &vec![[0, 0]; 4_096], 0.5);
        let after = analyzer.frame().levels(50);
        let loudest = |v: &[f32]| v.iter().copied().fold(f32::MIN, f32::max);
        let fell = loudest(&before) - loudest(&after);
        assert!((40.0..=50.0).contains(&fell), "{fell}");
    }

    #[test]
    fn a_reset_forgets_the_rate() {
        let mut analyzer = Analyzer::new();
        analyzer.analyse(352_800, &sine(352_800, 1_000.0, 0.9, 16_384), 0.0);
        assert_eq!(analyzer.frame().rate, 352_800);
        analyzer.reset();
        assert!(analyzer.frame().is_empty() && analyzer.is_quiet());
        assert_eq!(analyzer.frame(), SpectrumFrame::default());
        analyzer.analyse(48_000, &sine(48_000, 1_000.0, 0.9, 4_096), 0.0);
        assert_eq!(analyzer.frame().rate, 48_000);
    }

    #[test]
    fn every_rate_has_about_the_same_resolution() {
        for rate in [44_100, 48_000, 88_200, 192_000, 352_800, 384_000] {
            let size = Analyzer::block_for(rate);
            let hz = f64::from(rate) / size as f64;
            assert!((15.0..=30.0).contains(&hz), "{rate}");
        }
        assert_eq!(SpectrumFrame::default().levels(3), vec![RANGE_DB.0; 3]);
        // The tap holds the fine block at every rate.
        assert!(tap_frames_for(384_000) <= 1 << 18);
    }
}
