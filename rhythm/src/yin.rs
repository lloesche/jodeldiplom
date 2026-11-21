//! The pitch of one voice as it is sung (a singer at a microphone), frame
//! by frame as the samples come in: YIN (de Cheveigné and Kawahara, 2002).
//! [`crate::pitch`] hears a whole song at once and needs the frames around
//! each; this needs only the last [`WINDOW`] seconds, a frame every
//! [`HOP`] seconds, so it keeps up live on the processor.
//!
//! The voice comes in at any rate and is brought to [`RATE`] (a box filter
//! over each output sample's input span: enough below 4 kHz, where a
//! voice's pitch is). For each frame over [`WINDOW`]: the difference of the
//! signal with itself `tau` samples later, normalised by its running mean
//! (YIN's cumulative mean normalised difference), the first dip under
//! [`THRESHOLD`] from the shortest period up (else none: not a clear
//! pitch), refined between its neighbours by a parabola. A frame quieter
//! than [`QUIET_DB`] has none either (breath, the room).

/// The rate the voice is heard at.
pub const RATE: u32 = 16_000;
/// The seconds a frame hears.
pub const WINDOW: f64 = 0.04;
/// The seconds from frame to frame (as [`crate::waveform::FPS`]).
pub const HOP: f64 = 0.01;
/// The lowest pitch heard, in Hz (a bass's low notes).
pub const LOWEST: f64 = 70.0;
/// The highest, in Hz (a soprano's high ones).
pub const HIGHEST: f64 = 1000.0;
/// How clear a period must be (YIN's threshold: lower is stricter).
pub const THRESHOLD: f32 = 0.15;
/// A frame quieter than this (dB of full scale, RMS) has no pitch.
pub const QUIET_DB: f32 = -50.0;

/// A live pitch tracker: [`Yin::push`] the voice as it comes, a pitch (or
/// none) a frame.
pub struct Yin {
    /// Input samples to an output one, and how far into the next one.
    step: f64,
    at: f64,
    sum: f32,
    count: u32,
    /// The voice at [`RATE`], the last [`WINDOW`] of it and what came
    /// since the last frame.
    voice: Vec<f32>,
    /// Samples since the last frame.
    since: usize,
    window: usize,
    hop: usize,
    diff: Vec<f32>,
}

impl Yin {
    /// A tracker for a voice at `rate` samples a second.
    pub fn new(rate: u32) -> Yin {
        let window = (WINDOW * f64::from(RATE)).round() as usize;
        Yin {
            step: f64::from(rate.max(1)) / f64::from(RATE),
            at: 0.0,
            sum: 0.0,
            count: 0,
            voice: Vec::with_capacity(2 * window),
            since: 0,
            window,
            hop: (HOP * f64::from(RATE)).round() as usize,
            diff: Vec::new(),
        }
    }

    /// Takes `samples` of the voice; calls `frame` with each frame's pitch
    /// (hundredths of a MIDI note, 6900 the A at 440 Hz) or none, in order.
    pub fn push(&mut self, samples: &[f32], mut frame: impl FnMut(Option<f32>)) {
        for &x in samples {
            self.sum += x;
            self.count += 1;
            self.at += 1.0;
            if self.at < self.step {
                continue;
            }
            self.at -= self.step;
            let y = self.sum / self.count as f32;
            self.sum = 0.0;
            self.count = 0;
            self.voice.push(y);
            if self.voice.len() > self.window {
                self.voice.drain(..self.voice.len() - self.window);
            }
            self.since += 1;
            if self.since >= self.hop && self.voice.len() == self.window {
                self.since = 0;
                frame(self.pitch());
            }
        }
    }

    /// The pitch of the last [`WINDOW`], if clear.
    fn pitch(&mut self) -> Option<f32> {
        let x = &self.voice;
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        if 20.0 * rms.max(1e-9).log10() < QUIET_DB {
            return None;
        }
        let rate = f64::from(RATE);
        let shortest = (rate / HIGHEST).floor() as usize;
        let longest = ((rate / LOWEST).ceil() as usize).min(x.len() / 2);
        let span = x.len() - longest;
        self.diff.clear();
        self.diff.resize(longest + 1, 0.0);
        for (tau, slot) in self.diff.iter_mut().enumerate().skip(1) {
            *slot = x[..span]
                .iter()
                .zip(&x[tau..tau + span])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
        }
        // The cumulative mean normalised difference.
        let mut running = 0.0f32;
        self.diff[0] = 1.0;
        for (tau, slot) in self.diff.iter_mut().enumerate().skip(1) {
            running += *slot;
            *slot = if running > 0.0 {
                *slot * tau as f32 / running
            } else {
                1.0
            };
        }
        let d = &self.diff;
        let mut tau = shortest.max(2);
        while tau < longest {
            if d[tau] < THRESHOLD {
                // To the bottom of the dip.
                while tau + 1 < longest && d[tau + 1] < d[tau] {
                    tau += 1;
                }
                let (a, b, c) = (d[tau - 1], d[tau], d[tau + 1]);
                let bend = a - 2.0 * b + c;
                let shift = if bend.abs() > 1e-9 {
                    0.5 * (a - c) / bend
                } else {
                    0.0
                };
                let period = tau as f64 + f64::from(shift.clamp(-1.0, 1.0));
                let hz = rate / period;
                return Some((100.0 * (69.0 + 12.0 * (hz / 440.0).log2())) as f32);
            }
            tau += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f64, seconds: f64, level: f64) -> Vec<f32> {
        (0..(seconds * f64::from(rate)) as usize)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                // A voice-like tone: its fundamental and two overtones.
                (level
                    * (0.6 * (std::f64::consts::TAU * hz * t).sin()
                        + 0.3 * (std::f64::consts::TAU * 2.0 * hz * t).sin()
                        + 0.1 * (std::f64::consts::TAU * 3.0 * hz * t).sin()))
                    as f32
            })
            .collect()
    }

    fn frames(rate: u32, samples: &[f32]) -> Vec<Option<f32>> {
        let mut yin = Yin::new(rate);
        let mut out = Vec::new();
        // In pieces, as a device gives them.
        for piece in samples.chunks(441) {
            yin.push(piece, |p| out.push(p));
        }
        out
    }

    #[test]
    fn a_sung_note_is_heard_at_its_pitch_at_any_rate() {
        for rate in [16_000, 44_100, 48_000] {
            for (hz, cents) in [(110.0, 4500.0), (220.0, 5700.0), (523.25, 7200.0)] {
                let out = frames(rate, &tone(rate, hz, 0.5, 0.3));
                // A frame each 10 ms once the first window is full.
                assert!((44..=47).contains(&out.len()), "{rate} {hz}: {}", out.len());
                for p in &out {
                    let p = p.expect("a pitch");
                    assert!((p - cents).abs() < 10.0, "{rate} {hz}: {p}");
                }
            }
        }
    }

    #[test]
    fn quiet_or_noise_has_no_pitch() {
        let quiet = frames(48_000, &tone(48_000, 220.0, 0.3, 0.001));
        assert!(quiet.iter().all(Option::is_none));
        // Noise: no clear period.
        let mut seed = 1u32;
        let noise: Vec<f32> = (0..14_400)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let heard = frames(48_000, &noise);
        let pitched = heard.iter().filter(|p| p.is_some()).count();
        assert!(pitched * 10 < heard.len(), "{pitched} of {}", heard.len());
    }
}
