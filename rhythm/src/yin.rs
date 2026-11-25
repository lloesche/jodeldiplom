//! The pitch of one voice as it is sung (a singer at a microphone), frame
//! by frame as the samples come in: YIN (de Cheveigné and Kawahara, 2002).
//! [`crate::pitch`] hears a whole song at once and needs the frames around
//! each; this needs only the last [`WINDOW`] seconds, a frame every
//! [`HOP`] seconds, so it keeps up live on the processor.
//!
//! The voice comes in at any rate and is brought to [`RATE`] (a box filter
//! over each output sample's input span), then to the band its pitch is
//! clearest in, [`BAND_LOW`] to [`BAND_HIGH`]. For each frame
//! over [`WINDOW`]: the difference of the signal with itself `tau` samples
//! later, normalised by its running mean (YIN's cumulative mean normalised
//! difference), the first dip under [`THRESHOLD`] from the shortest period
//! up (else none: not a clear pitch), refined between its neighbours by a
//! parabola. A frame quieter than [`QUIET_DB`] in the band has none either
//! (breath, the room).
//!
//! The band matters at a laptop's own microphone: what the echo canceller
//! leaves of the music's cymbals and hi-hats, and a fan's hiss, lie above
//! a kilohertz, as loud as the voice; a fan's rumble and the bass below
//! the voice. In the difference they weigh as much as the voice does, so
//! a voice sung clearly under them has no dip under the threshold. A
//! voice's period shows as well in its harmonics in the band; a bass's
//! too, its lowest notes under [`BAND_LOW`]: its harmonics repeat at its
//! own period.

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
/// The band the voice is heard in, Hz: a fourth-order high-pass at the
/// low edge, a second-order low-pass at the high one (both Butterworth).
/// On kept singing at a laptop, the low edge anywhere from 110 to 130 Hz
/// hears as much; from 150 Hz a low voice's notes jump an octave more
/// often, and without it a bass left by the echo canceller is a pitch.
pub const BAND_LOW: f64 = 120.0;
/// See [`BAND_LOW`].
pub const BAND_HIGH: f64 = 1000.0;
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
    band: Band,
}

/// The filter to the voice's band, at [`RATE`].
struct Band([Biquad; 3]);

impl Band {
    fn new() -> Band {
        // A fourth-order Butterworth's two sections: quality 1/(2·cos(π/8))
        // and 1/(2·cos(3π/8)).
        Band([
            Biquad::new(BAND_LOW, 0.541_196_1, true),
            Biquad::new(BAND_LOW, 1.306_563, true),
            Biquad::new(BAND_HIGH, std::f64::consts::FRAC_1_SQRT_2, false),
        ])
    }

    fn run(&mut self, x: f32) -> f32 {
        self.0.iter_mut().fold(x, |y, section| section.run(y))
    }
}

/// One second-order section (transposed direct form II).
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

impl Biquad {
    /// A high-pass (`high`) or low-pass section at `hz` with quality `q`,
    /// at [`RATE`].
    fn new(hz: f64, q: f64, high: bool) -> Biquad {
        let w = std::f64::consts::TAU * hz / f64::from(RATE);
        let (sin, cos) = w.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        let (b0, b1) = if high {
            ((1.0 + cos) / 2.0, -(1.0 + cos))
        } else {
            ((1.0 - cos) / 2.0, 1.0 - cos)
        };
        Biquad {
            b: [(b0 / a0) as f32, (b1 / a0) as f32, (b0 / a0) as f32],
            a: [(-2.0 * cos / a0) as f32, ((1.0 - alpha) / a0) as f32],
            z: [0.0; 2],
        }
    }

    fn run(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
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
            band: Band::new(),
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
            let y = self.band.run(self.sum / self.count as f32);
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
                // The first window holds the band's filter settling on the
                // note's sudden start: a little off.
                for (k, p) in out.iter().enumerate() {
                    let p = p.expect("a pitch");
                    let off = if k == 0 { 20.0 } else { 10.0 };
                    assert!((p - cents).abs() < off, "{rate} {hz} frame {k}: {p}");
                }
            }
        }
    }

    /// White noise: `n` samples in -0.5..0.5 from `seed`.
    fn noise(mut seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// `x` brought to `rms`.
    fn at_rms(mut x: Vec<f32>, rms: f64) -> Vec<f32> {
        let now = (x.iter().map(|v| f64::from(v * v)).sum::<f64>() / x.len() as f64).sqrt();
        for v in &mut x {
            *v = (f64::from(*v) * rms / now) as f32;
        }
        x
    }

    /// A sung vowel at `hz`: its overtones up to 4 kHz, the `k`th at 1/k.
    fn vowel(rate: u32, hz: f64, seconds: f64, rms: f64) -> Vec<f32> {
        let overtones = (4000.0 / hz) as usize;
        let x = (0..(seconds * f64::from(rate)) as usize)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                (1..=overtones)
                    .map(|k| (std::f64::consts::TAU * k as f64 * hz * t).sin() / k as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        at_rms(x, rms)
    }

    /// `x` through two one-pole low-passes at `hz`.
    fn smoothed(rate: u32, x: &[f32], hz: f64) -> Vec<f32> {
        let k = (-std::f64::consts::TAU * hz / f64::from(rate)).exp() as f32;
        let (mut a, mut b) = (0.0f32, 0.0f32);
        x.iter()
            .map(|&v| {
                a = k * a + (1.0 - k) * v;
                b = k * b + (1.0 - k) * a;
                b
            })
            .collect()
    }

    /// Hiss from 1.5 to 8 kHz (the cymbals an echo canceller leaves, a
    /// fan's hiss): white noise less its part under 1.5 kHz, smoothed
    /// above 8.
    fn hiss(rate: u32, seconds: f64, rms: f64, seed: u32) -> Vec<f32> {
        let x = noise(seed, (seconds * f64::from(rate)) as usize);
        let low = smoothed(rate, &x, 1500.0);
        let high: Vec<f32> = x.iter().zip(&low).map(|(v, l)| v - l).collect();
        at_rms(smoothed(rate, &high, 8000.0), rms)
    }

    /// Rumble under 80 Hz (a fan's, or the bass left in the room).
    fn rumble(rate: u32, seconds: f64, rms: f64, seed: u32) -> Vec<f32> {
        let x = noise(seed, (seconds * f64::from(rate)) as usize);
        at_rms(smoothed(rate, &x, 80.0), rms)
    }

    /// `parts` played together.
    fn mixed(parts: &[&[f32]]) -> Vec<f32> {
        let n = parts.iter().map(|p| p.len()).min().unwrap_or(0);
        (0..n).map(|i| parts.iter().map(|p| p[i]).sum()).collect()
    }

    fn cents(hz: f64) -> f32 {
        (100.0 * (69.0 + 12.0 * (hz / 440.0).log2())) as f32
    }

    /// The share of `heard` at `cents`, within `off`.
    fn share_at(heard: &[Option<f32>], cents: f32, off: f32) -> f64 {
        let on = heard
            .iter()
            .filter(|p| p.is_some_and(|p| (p - cents).abs() < off))
            .count();
        on as f64 / heard.len() as f64
    }

    #[test]
    fn quiet_or_noise_has_no_pitch() {
        let quiet = frames(48_000, &tone(48_000, 220.0, 0.3, 0.001));
        assert!(quiet.iter().all(Option::is_none));
        // Noise: no clear period.
        let heard = frames(48_000, &noise(1, 14_400));
        let pitched = heard.iter().filter(|p| p.is_some()).count();
        assert!(pitched * 10 < heard.len(), "{pitched} of {}", heard.len());
    }

    /// A voice sung clearly under hiss as loud as it (what a laptop's echo
    /// canceller leaves of the music's cymbals): the hiss lies above the
    /// voice's band, so the voice is heard.
    #[test]
    fn a_voice_under_hiss_as_loud_is_heard() {
        let voice = vowel(48_000, 140.0, 1.0, 0.03);
        let hiss = hiss(48_000, 1.0, 0.03, 7);
        let heard = frames(48_000, &mixed(&[&voice, &hiss]));
        let share = share_at(&heard, cents(140.0), 25.0);
        assert!(share > 0.9, "{share}");
    }

    /// A laptop's fan, its rumble and its hiss, louder than the room: a
    /// voice over it is heard, the fan alone is no pitch.
    #[test]
    fn a_voice_over_a_fan_is_heard_and_the_fan_alone_is_no_pitch() {
        let fan = mixed(&[
            &hiss(48_000, 2.0, 0.02, 3),
            &rumble(48_000, 2.0, 0.02, 5),
            &at_rms(noise(9, 96_000), 0.005),
        ]);
        let alone = frames(48_000, &fan);
        let pitched = alone.iter().filter(|p| p.is_some()).count();
        assert_eq!(pitched, 0, "{pitched} of {}", alone.len());
        let voice = vowel(48_000, 140.0, 2.0, 0.03);
        let heard = frames(48_000, &mixed(&[&voice, &fan]));
        let share = share_at(&heard, cents(140.0), 25.0);
        assert!(share > 0.8, "{share}");
    }

    /// A bass's low note, its fundamental under the band: heard at its own
    /// pitch, not an octave up; a soprano's high one, its overtones over
    /// the band, too.
    #[test]
    fn the_lowest_and_highest_notes_are_heard_at_their_pitch() {
        for hz in [75.0, 950.0] {
            let heard = frames(44_100, &vowel(44_100, hz, 0.5, 0.05));
            let share = share_at(&heard[1..], cents(hz), 15.0);
            assert!(share == 1.0, "{hz}: {share}");
        }
    }
}
