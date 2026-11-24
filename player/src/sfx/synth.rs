//! The few pieces the game's sounds are made of, at any rate: a stereo
//! buffer, tones (partials on a pitch that may glide and wobble, under an
//! envelope, their brightness a low-pass worked out per partial, so a tone
//! is band-limited at every rate), bursts of noise through a band-pass,
//! a small room to put them in, and the peak a sound is set to. Every
//! sound is computed, never sampled: the same at every rate, and each one
//! the same every time (the noise is seeded).

use std::f64::consts::{PI, TAU};

/// A stereo sound being made, at `rate`.
pub struct Buf {
    /// Its rate.
    pub rate: f64,
    /// The left side.
    pub l: Vec<f64>,
    /// The right side.
    pub r: Vec<f64>,
}

impl Buf {
    /// `seconds` of silence at `rate`.
    pub fn new(rate: u32, seconds: f64) -> Buf {
        let n = (seconds * f64::from(rate)).ceil() as usize;
        Buf {
            rate: f64::from(rate),
            l: vec![0.0; n],
            r: vec![0.0; n],
        }
    }

    /// Its length in seconds.
    pub fn seconds(&self) -> f64 {
        self.l.len() as f64 / self.rate
    }

    /// Adds `x(t)` (seconds from `start`, for `length` seconds) at `pan`
    /// (-1 left to 1 right; at 0 both sides get it whole).
    pub fn add(&mut self, start: f64, length: f64, pan: f64, mut x: impl FnMut(f64) -> f64) {
        let (gl, gr) = balance(pan);
        let from = (start * self.rate).round().max(0.0) as usize;
        let n = (length * self.rate).ceil() as usize;
        let to = (from + n).min(self.l.len());
        for i in from..to {
            let v = x((i - from) as f64 / self.rate);
            self.l[i] += gl * v;
            self.r[i] += gr * v;
        }
    }

    /// Adds `tone` from `start` at `pan`.
    pub fn tone(&mut self, start: f64, pan: f64, tone: &Tone) {
        let rate = self.rate;
        let mut voice = tone.voice(rate);
        self.add(start, tone.length, pan, |t| voice.next(t));
    }

    /// Adds `noise` from `start` at `pan`.
    pub fn noise(&mut self, start: f64, pan: f64, noise: &Noise) {
        let rate = self.rate;
        let mut bp = Biquad::band(noise.from, noise.q, rate);
        let mut seed = noise.seed;
        let (from, to) = (noise.from, noise.to);
        let glide = noise.length.max(1e-6);
        let mut k = 0usize;
        self.add(start, noise.length, pan, |t| {
            // The band moves now and then, not every sample.
            if k.is_multiple_of(32) && from != to {
                let f = from * (to / from).powf((t / glide).min(1.0));
                bp = bp.retuned(f, noise.q, rate);
            }
            k += 1;
            noise.envelope.at(t) * bp.run(white(&mut seed)) * noise.amp
        });
    }

    /// A small room around what is in it: `wet` of a stereo reverb with a
    /// tail of about `seconds`, the dry sound kept whole.
    pub fn room(&mut self, wet: f64, seconds: f64) {
        let rate = self.rate;
        let mut left = Room::new(rate, seconds, 0);
        let mut right = Room::new(rate, seconds, 23);
        for i in 0..self.l.len() {
            let mono = 0.5 * (self.l[i] + self.r[i]);
            self.l[i] += wet * left.run(mono);
            self.r[i] += wet * right.run(mono);
        }
    }

    /// Scaled to peak at `db` dBFS (nothing to scale: silence).
    pub fn peak_at(&mut self, db: f64) {
        let peak = self
            .l
            .iter()
            .chain(&self.r)
            .fold(0.0f64, |m, v| m.max(v.abs()));
        if peak > 0.0 {
            let g = 10f64.powf(db / 20.0) / peak;
            for v in self.l.iter_mut().chain(self.r.iter_mut()) {
                *v *= g;
            }
        }
    }

    /// Fades the last `seconds` out (a raised cosine), so it ends on 0.
    pub fn fade_out(&mut self, seconds: f64) {
        let n = ((seconds * self.rate) as usize).min(self.l.len());
        let len = self.l.len();
        for k in 0..n {
            let g = 0.5 - 0.5 * (PI * k as f64 / n as f64).cos();
            self.l[len - 1 - k] *= g;
            self.r[len - 1 - k] *= g;
        }
    }

    /// Interleaved stereo, as the desk takes it.
    pub fn interleaved(&self) -> Vec<f32> {
        self.l
            .iter()
            .zip(&self.r)
            .flat_map(|(&l, &r)| [l as f32, r as f32])
            .collect()
    }
}

/// The gains of each side at `pan` (-1 to 1): the near side whole, the far
/// one less, so a sound in the middle is as loud as it was made.
pub fn balance(pan: f64) -> (f64, f64) {
    let p = pan.clamp(-1.0, 1.0);
    ((1.0 - p).min(1.0), (1.0 + p).min(1.0))
}

/// One partial of a tone: its frequency as a ratio of the tone's, its
/// amplitude, and how fast it dies away (the time constant of its own
/// decay, seconds; infinite: only the tone's).
#[derive(Clone, Copy, Debug)]
pub struct Partial {
    /// Its frequency, times the tone's.
    pub ratio: f64,
    /// Its amplitude.
    pub amp: f64,
    /// Its own decay's time constant, seconds.
    pub decay: f64,
}

impl Partial {
    /// A partial at `ratio`, `amp`, dying away over `decay`.
    pub const fn new(ratio: f64, amp: f64, decay: f64) -> Partial {
        Partial { ratio, amp, decay }
    }
}

/// How loud a sound is along its time: a raised-cosine attack, a hold,
/// then an exponential decay (time constant `decay`), let go of with a
/// fade to 0 over its last quarter (`length` seconds from its start).
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    /// Its attack, seconds.
    pub attack: f64,
    /// How long it holds after it.
    pub hold: f64,
    /// Its decay's time constant.
    pub decay: f64,
    /// Its whole length.
    pub length: f64,
}

impl Envelope {
    /// The gain at `t` seconds, 0 to 1.
    pub fn at(&self, t: f64) -> f64 {
        if t < 0.0 || t >= self.length {
            return 0.0;
        }
        let rise = if t < self.attack {
            0.5 - 0.5 * (PI * t / self.attack).cos()
        } else {
            1.0
        };
        let fall = if t > self.attack + self.hold {
            (-(t - self.attack - self.hold) / self.decay).exp()
        } else {
            1.0
        };
        // The last quarter (50 ms at most) fades to 0: a sound cut while
        // it still rings is let go of, not chopped.
        let tail = (0.05f64).min(self.length / 4.0);
        let end = ((self.length - t) / tail).min(1.0);
        rise * fall * (0.5 - 0.5 * (PI * end).cos())
    }
}

/// A tone: partials on `freq`, gliding to `glide.0` times it with time
/// constant `glide.1` (1: none), with a vibrato (`vibrato.0` Hz,
/// `vibrato.1` cents, coming in over its first `vibrato.2` seconds), under
/// `envelope`, its partials above `bright` Hz (a moving low-pass: from
/// `bright.0` towards `bright.1`, time constant `bright.2`) turned down.
#[derive(Clone, Debug)]
pub struct Tone {
    /// Its frequency, Hz.
    pub freq: f64,
    /// Where it glides to (times it), and how fast.
    pub glide: (f64, f64),
    /// Its vibrato: rate, depth, how long it takes to come in.
    pub vibrato: (f64, f64, f64),
    /// How loud along its time.
    pub envelope: Envelope,
    /// Its brightness: from, to, how fast.
    pub bright: (f64, f64, f64),
    /// Its partials.
    pub partials: Vec<Partial>,
    /// Its amplitude.
    pub amp: f64,
    /// Its length, seconds.
    pub length: f64,
}

impl Tone {
    /// A tone of `partials` on `freq` under `envelope`, nothing moving.
    pub fn new(freq: f64, envelope: Envelope, partials: &[Partial]) -> Tone {
        Tone {
            freq,
            glide: (1.0, 1.0),
            vibrato: (0.0, 0.0, 0.0),
            envelope,
            bright: (f64::INFINITY, f64::INFINITY, 1.0),
            partials: partials.to_vec(),
            amp: 1.0,
            length: envelope.length,
        }
    }

    /// The same, at `amp`.
    pub fn amp(mut self, amp: f64) -> Tone {
        self.amp = amp;
        self
    }

    /// The same, gliding to `to` times its frequency with time constant
    /// `tau` seconds.
    pub fn glide(mut self, to: f64, tau: f64) -> Tone {
        self.glide = (to, tau);
        self
    }

    /// The same, wobbling at `hz` by `cents`, in over `after` seconds.
    pub fn vibrato(mut self, hz: f64, cents: f64, after: f64) -> Tone {
        self.vibrato = (hz, cents, after);
        self
    }

    /// The same, its brightness moving from `from` Hz to `to` Hz with time
    /// constant `tau`.
    pub fn bright(mut self, from: f64, to: f64, tau: f64) -> Tone {
        self.bright = (from, to, tau);
        self
    }

    fn voice(&self, rate: f64) -> Voice<'_> {
        Voice {
            tone: self,
            rate,
            phases: vec![0.0; self.partials.len()],
            vib: 0.0,
        }
    }
}

/// A tone sounding: each partial's phase, the vibrato's.
struct Voice<'a> {
    tone: &'a Tone,
    rate: f64,
    phases: Vec<f64>,
    vib: f64,
}

impl Voice<'_> {
    /// The sample at `t` seconds (called once a sample, in order).
    fn next(&mut self, t: f64) -> f64 {
        let tone = self.tone;
        let (to, tau) = tone.glide;
        let mut f = tone.freq * (1.0 + (to - 1.0) * (1.0 - (-t / tau.max(1e-6)).exp()));
        let (hz, cents, after) = tone.vibrato;
        if cents > 0.0 {
            self.vib += TAU * hz / self.rate;
            let depth = cents * (t / after.max(1e-6)).min(1.0);
            f *= 2f64.powf(depth * self.vib.sin() / 1200.0);
        }
        let (b0, b1, btau) = tone.bright;
        let cutoff = if b0.is_finite() {
            b1 + (b0 - b1) * (-t / btau.max(1e-6)).exp()
        } else {
            f64::INFINITY
        };
        let nyquist = 0.45 * self.rate;
        let mut sum = 0.0;
        for (p, phase) in tone.partials.iter().zip(&mut self.phases) {
            let pf = f * p.ratio;
            *phase += TAU * pf / self.rate;
            if *phase > TAU {
                *phase -= TAU;
            }
            if pf >= nyquist {
                continue;
            }
            let lowpass = if cutoff.is_finite() {
                1.0 / (1.0 + (pf / cutoff).powi(4))
            } else {
                1.0
            };
            let own = if p.decay.is_finite() {
                (-t / p.decay).exp()
            } else {
                1.0
            };
            sum += p.amp * own * lowpass * phase.sin();
        }
        sum * tone.envelope.at(t) * tone.amp
    }
}

/// A burst of noise through a band-pass gliding from `from` Hz to `to`
/// (`q` its sharpness), under `envelope`, at `amp`; `seed` its own noise.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    /// The band's frequency at the start, Hz.
    pub from: f64,
    /// At the end.
    pub to: f64,
    /// Its sharpness.
    pub q: f64,
    /// How loud along its time.
    pub envelope: Envelope,
    /// Its amplitude.
    pub amp: f64,
    /// Its length, seconds.
    pub length: f64,
    /// Its noise's seed.
    pub seed: u64,
}

/// A white noise sample, -1 to 1, from `seed` (xorshift).
pub fn white(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// A number from 0 to 1 from `seed` (to scatter a sound's pieces, the same
/// every time).
pub fn scatter(seed: &mut u64) -> f64 {
    0.5 + 0.5 * white(seed)
}

/// A band-pass (constant peak gain), transposed direct form II.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn band(f: f64, q: f64, rate: f64) -> Biquad {
        Biquad {
            b0: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
        .retuned(f, q, rate)
    }

    /// The same filter, its state kept, at `f` Hz.
    fn retuned(self, f: f64, q: f64, rate: f64) -> Biquad {
        let w = TAU * f.min(0.45 * rate) / rate;
        let alpha = w.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Biquad {
            b0: alpha / a0,
            b2: -alpha / a0,
            a1: -2.0 * w.cos() / a0,
            a2: (1.0 - alpha) / a0,
            ..self
        }
    }

    fn run(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = -self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// A small Schroeder room: four damped combs side by side, two all-passes
/// after them; `spread` samples on each delay set one side apart from the
/// other.
struct Room {
    combs: Vec<(Vec<f64>, usize, f64, f64)>,
    passes: Vec<(Vec<f64>, usize)>,
}

impl Room {
    fn new(rate: f64, seconds: f64, spread: usize) -> Room {
        let scale = rate / 44_100.0;
        let len = |n: usize| ((n + spread) as f64 * scale) as usize + 1;
        // Feedback for a tail of about `seconds` (60 dB down).
        let combs = [1_116, 1_188, 1_277, 1_356]
            .map(|n| {
                let d = len(n);
                let g = 10f64.powf(-3.0 * d as f64 / rate / seconds.max(0.05));
                (vec![0.0; d], 0, g, 0.0)
            })
            .to_vec();
        let passes = [556, 441].map(|n| (vec![0.0; len(n)], 0)).to_vec();
        Room { combs, passes }
    }

    fn run(&mut self, x: f64) -> f64 {
        let mut y = 0.0;
        for (line, at, g, low) in &mut self.combs {
            let out = line[*at];
            // A little damping: the tail darkens as it dies.
            *low = 0.7 * out + 0.3 * *low;
            line[*at] = x + *g * *low;
            *at = (*at + 1) % line.len();
            y += out;
        }
        y *= 0.25;
        for (line, at) in &mut self.passes {
            let out = line[*at];
            let v = y + 0.5 * out;
            line[*at] = v;
            *at = (*at + 1) % line.len();
            y = out - 0.5 * v;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_envelope_rises_holds_falls_and_ends_on_zero() {
        let e = Envelope {
            attack: 0.01,
            hold: 0.02,
            decay: 0.05,
            length: 0.2,
        };
        assert_eq!(e.at(0.0), 0.0);
        assert!((e.at(0.02) - 1.0).abs() < 1e-9);
        assert!(e.at(0.1) < e.at(0.05) && e.at(0.1) > 0.0);
        assert_eq!(e.at(0.2), 0.0);
        assert!(e.at(0.1999) < 1e-3);
    }

    #[test]
    fn a_tone_is_band_limited_at_every_rate() {
        // A partial over the Nyquist frequency is left out, not folded.
        for rate in [44_100, 384_000] {
            let mut b = Buf::new(rate, 0.05);
            let e = Envelope {
                attack: 0.001,
                hold: 1.0,
                decay: 1.0,
                length: 0.05,
            };
            b.tone(
                0.0,
                0.0,
                &Tone::new(1_000.0, e, &[Partial::new(30.0, 1.0, 1.0)]),
            );
            let silent = b.l.iter().all(|v| v.abs() < 1e-12);
            assert_eq!(silent, rate == 44_100, "{rate}");
        }
    }

    #[test]
    fn a_sound_is_set_to_its_peak_and_the_same_every_time() {
        let make = || {
            let mut b = Buf::new(48_000, 0.1);
            let e = Envelope {
                attack: 0.0005,
                hold: 0.0,
                decay: 0.01,
                length: 0.05,
            };
            b.noise(
                0.0,
                0.3,
                &Noise {
                    from: 3_000.0,
                    to: 5_000.0,
                    q: 1.0,
                    envelope: e,
                    amp: 1.0,
                    length: 0.05,
                    seed: 7,
                },
            );
            b.room(0.3, 0.3);
            b.peak_at(-20.0);
            b.interleaved()
        };
        let a = make();
        assert_eq!(a, make());
        let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            (20.0 * f64::from(peak).log10() + 20.0).abs() < 0.01,
            "{peak}"
        );
    }
}
