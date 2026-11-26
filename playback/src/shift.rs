//! The key shift's pass ([`Shift`]): channels read along a time map with
//! every frequency times a ratio, their length and their time the map's.
//!
//! The channels are stretched by the ratio (one [`Stretcher`] a channel,
//! all at the same speed: the map's over the ratio when the map is followed
//! by stretching, else one over the ratio), then read through a
//! [`VarResampler`] at the stretched frame whose input time is the one the
//! map has for each output frame ([`Stretcher::output_at`]): read that much
//! faster than it was made, it is as long as the map has it and every
//! frequency is the ratio times what the map makes it. So a channel shifted
//! and one read along the map alone stay on the same instants.

use crate::stretch::Stretcher;
use crate::warp::VarResampler;

/// The most a key goes either way, in semitones.
pub const KEY_RANGE: i8 = 6;

/// The pitch ratio of `semitones`: 2^(n/12).
pub fn ratio_of(semitones: i8) -> f64 {
    2f64.powf(f64::from(semitones) / 12.0)
}

/// Channels read along a time map a key apart: see the module.
pub(crate) struct Shift {
    ratio: f64,
    stretchers: Vec<Stretcher>,
    reader: VarResampler,
    /// What each stretcher made that the reader has not taken yet.
    made: Vec<Vec<f64>>,
    /// Scratch: one channel's input, and the stretched frames interleaved.
    channel: Vec<f64>,
    interleaved: Vec<f64>,
    /// Input frames pushed.
    pushed: u64,
    /// The input has ended and every stretcher has made all it will.
    drained: bool,
}

impl Shift {
    /// `channels` channels at `rate` Hz shifted by `semitones`.
    pub(crate) fn new(semitones: i8, channels: usize, rate: u32) -> Shift {
        let ratio = ratio_of(semitones);
        let stretchers = (0..channels)
            .map(|_| {
                let mut s = Stretcher::new(rate);
                s.set_speed(1.0 / ratio);
                s
            })
            .collect();
        Shift {
            ratio,
            stretchers,
            reader: VarResampler::new(channels),
            made: vec![Vec::new(); channels],
            channel: Vec::new(),
            interleaved: Vec::new(),
            pushed: 0,
            drained: false,
        }
    }

    /// Output frames made so far.
    pub(crate) fn made(&self) -> u64 {
        self.reader.made()
    }

    /// Takes `input` (interleaved; `ended`: the last of it), the map at
    /// `speed` there (1 for a map read at a changing speed), and appends to
    /// `out` the frames it can make, frame `k` at input frame `position(k)`
    /// (from the first pushed): to where the input reaches, all of them
    /// once it has ended (up to the first frame the map puts past it).
    pub(crate) fn read(
        &mut self,
        input: &[f64],
        ended: bool,
        speed: f64,
        position: impl Fn(u64) -> f64,
        out: &mut Vec<f64>,
    ) {
        let ch = self.stretchers.len().max(1);
        for (c, stretcher) in self.stretchers.iter_mut().enumerate() {
            self.channel.clear();
            self.channel
                .extend(input.iter().skip(c).step_by(ch).copied());
            stretcher.push(&self.channel);
            if ended {
                stretcher.finish();
            }
            stretcher.set_speed(speed / self.ratio);
        }
        self.pushed += (input.len() / ch) as u64;
        // A frame's length of stretched frames at a time, then what the
        // reader can make of them: its positions are looked up while the
        // stretchers still hold the frame centres around them.
        let piece = self.stretchers[0].frame_len();
        loop {
            let mut pulled = 0;
            for (stretcher, made) in self.stretchers.iter_mut().zip(self.made.iter_mut()) {
                pulled = pulled.max(stretcher.pull(made, piece));
            }
            let n = self.made.iter().map(Vec::len).min().unwrap_or(0);
            self.interleaved.clear();
            self.interleaved.reserve(n * ch);
            for i in 0..n {
                self.interleaved.extend(self.made.iter().map(|m| m[i]));
            }
            for made in &mut self.made {
                made.drain(..n);
            }
            self.reader.push(&self.interleaved);
            if ended && !self.drained && self.stretchers.iter().all(Stretcher::is_done) {
                self.drained = true;
                self.reader.end();
            }
            let limit = if self.drained {
                frames_before(self.reader.made(), self.pushed, &position)
            } else {
                usize::MAX
            };
            let Shift {
                stretchers, reader, ..
            } = self;
            let first = &stretchers[0];
            reader.pull(out, limit, |k| first.output_at(position(k)));
            if pulled == 0 {
                break;
            }
        }
    }
}

/// How many frames from frame `made` on the map puts before input frame
/// `taken` (`position(k)` the input frame of frame `k`).
pub(crate) fn frames_before(made: u64, taken: u64, position: impl Fn(u64) -> f64) -> usize {
    let taken = taken as f64;
    let mut n = 0usize;
    while position(made + n as u64) < taken && n < 1 << 20 {
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::stems::RATE;
    use crate::warping::{Reading, Warping};

    /// The frequency of the strongest bin of `x` (a Hann window, the peak
    /// placed between its bins), at `rate`.
    pub(crate) fn peak_hz(x: &[f64], rate: u32) -> f64 {
        let n = x.len();
        let w = |i: usize| jodel_dsp::window::hann_at(i, n);
        let bin = |k: f64| {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, &v) in x.iter().enumerate() {
                let a = 2.0 * std::f64::consts::PI * k * i as f64 / n as f64;
                re += v * w(i) * a.cos();
                im -= v * w(i) * a.sin();
            }
            re.hypot(im)
        };
        let hz = |k: f64| k * f64::from(rate) / n as f64;
        // Coarse, then fine around the best.
        let best = (1..n / 2)
            .filter(|&k| hz(k as f64) > 50.0 && hz(k as f64) < 2_000.0)
            .max_by(|&a, &b| bin(a as f64).total_cmp(&bin(b as f64)))
            .unwrap_or(1) as f64;
        let mut k = best - 1.0;
        let mut top = (best, bin(best));
        while k <= best + 1.0 {
            let m = bin(k);
            if m > top.1 {
                top = (k, m);
            }
            k += 0.01;
        }
        hz(top.0)
    }

    fn sine(hz: f64, frames: usize) -> Vec<f64> {
        (0..frames)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / f64::from(RATE)).sin())
            .collect()
    }

    /// Shifted along the identity map in pieces: as long as it went in.
    fn shifted(x: &[f64], semitones: i8) -> Vec<f64> {
        let mut shift = Shift::new(semitones, 1, RATE);
        let mut out = Vec::new();
        let chunks: Vec<&[f64]> = x.chunks(4096).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            shift.read(chunk, last, 1.0, |k| k as f64, &mut out);
        }
        out
    }

    #[test]
    fn a_sine_two_semitones_up_is_two_semitones_higher_and_as_long() {
        let x = sine(440.0, RATE as usize * 2);
        let y = shifted(&x, 2);
        assert_eq!(y.len(), x.len(), "as long");
        let mid = &y[RATE as usize / 2..RATE as usize / 2 + 16_384];
        let hz = peak_hz(mid, RATE);
        assert!((hz - 493.88).abs() < 1.0, "{hz} Hz");
        let down = shifted(&x, -6);
        let hz = peak_hz(&down[RATE as usize / 2..RATE as usize / 2 + 16_384], RATE);
        assert!((hz - 311.13).abs() < 1.0, "{hz} Hz");
    }

    /// A sine shifted holds its level from 10 ms after its first frame (the
    /// stream of a key starts 0.15 s before it is heard): no ramp while the
    /// stretchers fill, each 5 ms within 5 % of the sine's.
    #[test]
    fn a_sine_shifted_holds_its_level_from_its_start() {
        for semitones in [-6, 2, 6] {
            let y = shifted(&sine(440.0, RATE as usize), semitones);
            let rms = 1.0 / 2f64.sqrt();
            let first = RATE as usize / 100;
            for (k, piece) in y[first..RATE as usize / 2].chunks_exact(220).enumerate() {
                let got = (piece.iter().map(|v| v * v).sum::<f64>() / 220.0).sqrt() / rms;
                assert!(
                    (got - 1.0).abs() < 0.05,
                    "{semitones:+}: {got:.3} of the level at {} ms",
                    10 + k * 5
                );
            }
        }
    }

    /// A click shifted stays on its instant (to a few milliseconds).
    #[test]
    fn a_click_shifted_stays_where_it_was() {
        let mut x = vec![0.0; RATE as usize];
        let at = RATE as usize / 2;
        for (i, v) in x[at..at + 200].iter_mut().enumerate() {
            *v = (2.0 * std::f64::consts::PI * 1_000.0 * i as f64 / f64::from(RATE)).sin();
        }
        for semitones in [-6, 3, 6] {
            let y = shifted(&x, semitones);
            let loudest = (0..y.len())
                .max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs()))
                .unwrap();
            let off = (loudest as f64 - (at + 100) as f64) / f64::from(RATE);
            assert!(off.abs() < 0.01, "{semitones}: {off} s off");
        }
    }

    /// What the key shift costs a head at the separator's rate: six
    /// channels shifted (bass, vocals, other) and two read along the map
    /// alone (the drums), in the stem run's pieces of 4096 frames. A
    /// measurement, not a check: run it in a release build with
    /// `--ignored`; `PERFORMANCE.md` has the numbers.
    #[test]
    #[ignore = "a measurement: run it in a release build"]
    fn six_channels_shifted_and_two_read_along() {
        if cfg!(debug_assertions) {
            println!("speed is measured in release builds only");
            return;
        }
        let seconds = 8.0;
        let frames = (f64::from(RATE) * seconds) as usize;
        for semitones in [-6i8, 2, 6] {
            let channels = 8;
            let input: Vec<f64> = (0..frames * channels)
                .map(|k| (k as f64 * 0.001).sin() * 4e6)
                .collect();
            let map = std::sync::Arc::new(crate::warp::TimeMap::new(&[], 0.0));
            let mut warping = Warping::new(Reading {
                map,
                a0: 0.0,
                b_origin: 0.0,
                tempo: crate::transition::TempoMode::Varispeed,
                channels,
                key: semitones,
                kept: vec![0, 1],
            });
            let (mut made, mut piece) = (0usize, Vec::new());
            let start = Instant::now();
            for chunk in input.chunks(4096 * channels) {
                piece.clear();
                piece.extend_from_slice(chunk);
                warping.read(&mut piece);
                made += piece.len() / channels;
            }
            warping.end();
            piece.clear();
            warping.read(&mut piece);
            made += piece.len() / channels;
            let took = start.elapsed().as_secs_f64();
            let realtime = made as f64 / f64::from(RATE) / took;
            println!(
                "key {semitones:+}: 6 ch shifted + 2 read along at 44100: {realtime:>6.1}x real time ({:.1} % of a core)",
                100.0 / realtime,
            );
            assert_eq!(made, frames, "{semitones}: as long");
        }
    }
}
