//! Generated music for tests and demonstrations: drum kits, clicks,
//! chords and bass at given beat times, at [`SAMPLE_RATE`](crate::SAMPLE_RATE).

use std::f64::consts::PI;

const SR: f64 = crate::SAMPLE_RATE as f64;

/// A small deterministic noise source (xorshift64).
pub struct Noise(u64);

impl Noise {
    /// A source started from `seed`.
    pub fn new(seed: u64) -> Noise {
        Noise(seed.max(1))
    }

    /// The next value in `-1..1`.
    pub fn sample(&mut self) -> f64 {
        let x = jodel_dsp::xorshift64(&mut self.0);
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Beat times from `start` until `end` seconds at the tempo `bpm(t)`.
pub fn beat_times(start: f64, end: f64, bpm: impl Fn(f64) -> f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut t = start;
    while t < end {
        out.push(t);
        t += 60.0 / bpm(t);
    }
    out
}

/// A sound to place at a time.
#[derive(Clone, Copy, Debug)]
pub enum Hit {
    /// A kick drum: a falling sine from 150 to 50 Hz.
    Kick,
    /// A snare: a noise burst over a 190 Hz body.
    Snare,
    /// A closed hi-hat: a short burst of bright noise.
    Hat,
    /// A metronome click at 1 kHz.
    Click,
    /// A plucked note at a frequency, with a few partials.
    Pluck(f64),
}

/// A signal of `seconds` built from hits and held chords.
pub struct Track {
    /// The samples.
    pub samples: Vec<f32>,
    noise: Noise,
}

impl Track {
    /// Silence of `seconds`.
    pub fn new(seconds: f64) -> Track {
        Track {
            samples: vec![0.0; (seconds * SR) as usize],
            noise: Noise::new(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn add(&mut self, at: f64, len: f64, mut f: impl FnMut(f64, &mut Noise) -> f64) {
        let start = (at * SR).round() as usize;
        let count = (len * SR) as usize;
        for i in 0..count {
            let Some(s) = self.samples.get_mut(start + i) else {
                break;
            };
            *s += f(i as f64 / SR, &mut self.noise) as f32;
        }
    }

    /// Places `hit` at `at` seconds with `gain`.
    pub fn hit(&mut self, at: f64, hit: Hit, gain: f64) {
        match hit {
            Hit::Kick => {
                let mut phase = 0.0;
                self.add(at, 0.4, |t, _| {
                    let f = 50.0 + 100.0 * (-t / 0.04).exp();
                    phase += 2.0 * PI * f / SR;
                    gain * phase.sin() * (-t / 0.12).exp()
                });
            }
            Hit::Snare => self.add(at, 0.25, |t, n| {
                gain * (0.7 * n.sample() * (-t / 0.06).exp()
                    + 0.3 * (2.0 * PI * 190.0 * t).sin() * (-t / 0.1).exp())
            }),
            Hit::Hat => {
                let mut last = 0.0;
                self.add(at, 0.08, |t, n| {
                    let x = n.sample();
                    let y = x - last;
                    last = x;
                    gain * 0.5 * y * (-t / 0.015).exp()
                });
            }
            Hit::Click => self.add(at, 0.05, |t, _| {
                gain * (2.0 * PI * 1000.0 * t).sin() * (-t / 0.01).exp()
            }),
            Hit::Pluck(f) => self.add(at, 1.5, |t, _| {
                let partials: f64 = (1..=4)
                    .map(|k| (2.0 * PI * f * k as f64 * t).sin() / k as f64)
                    .sum();
                gain * partials * (-t / 0.4).exp() * (t / 0.003).min(1.0)
            }),
        }
    }

    /// Holds a chord of `freqs` from `start` to `end` seconds, with short
    /// fades.
    pub fn chord(&mut self, start: f64, end: f64, freqs: &[f64], gain: f64) {
        let len = end - start;
        let freqs = freqs.to_vec();
        self.add(start, len, |t, _| {
            let fade = (t / 0.01).min(1.0).min((len - t) / 0.01).max(0.0);
            gain * fade * freqs.iter().map(|f| (2.0 * PI * f * t).sin()).sum::<f64>()
                / freqs.len() as f64
        });
    }

    /// Adds white noise at `gain`.
    pub fn noise(&mut self, gain: f64) {
        for s in &mut self.samples {
            *s += (gain * self.noise.sample()) as f32;
        }
    }
}

/// Chords that change every bar (a I–vi–IV–V loop in C).
pub const PROGRESSION: [[f64; 3]; 4] = [
    [261.6, 329.6, 392.0],
    [220.0, 261.6, 329.6],
    [174.6, 220.0, 261.6],
    [196.0, 246.9, 293.7],
];

/// A drum kit on `beats` in bars of `meter` beats, with chords changing
/// each bar: kick on the first beat of the bar (and the third in 4/4),
/// snare on the others' backbeats, hats on the eighths (`swing` moves the
/// off-beat hat from halfway to that fraction of the beat).
pub fn band(seconds: f64, beats: &[f64], meter: usize, swing: f64) -> Track {
    let mut track = Track::new(seconds);
    for (i, &b) in beats.iter().enumerate() {
        let position = i % meter;
        let next = beats.get(i + 1).copied().unwrap_or(b + 0.5);
        match (meter, position) {
            (_, 0) => track.hit(b, Hit::Kick, 0.8),
            (4, 2) => track.hit(b, Hit::Kick, 0.6),
            (4, _) => track.hit(b, Hit::Snare, 0.5),
            _ => track.hit(b, Hit::Pluck(PROGRESSION[(i / meter) % 4][0]), 0.25),
        }
        track.hit(b, Hit::Hat, 0.3);
        track.hit(b + swing * (next - b), Hit::Hat, 0.2);
        if position == 0 {
            let end = beats.get(i + meter).copied().unwrap_or(seconds);
            track.chord(b, end, &PROGRESSION[(i / meter) % 4], 0.12);
        }
    }
    track
}

/// The point `fraction` of the way from beat `i` to the next (a beat
/// length past the last).
fn within(beats: &[f64], i: usize, fraction: f64) -> f64 {
    let b = beats[i];
    let next = beats.get(i + 1).copied().unwrap_or_else(|| {
        if i > 0 {
            2.0 * b - beats[i - 1]
        } else {
            b + 0.5
        }
    });
    b + fraction * (next - b)
}

/// A slow backbeat on `beats` in bars of four, as in hip-hop and rock
/// ballads: kick on 1 and 3, snare on 2 and 4, loud hi-hats on every
/// eighth note, a bass note on each kick and a chord a bar. The hats
/// make the eighth notes the steadiest pulse.
pub fn backbeat(seconds: f64, beats: &[f64]) -> Track {
    let mut track = Track::new(seconds);
    for (i, &b) in beats.iter().enumerate() {
        let bar = (i / 4) % 4;
        if i % 2 == 0 {
            track.hit(b, Hit::Kick, 0.8);
            track.hit(b, Hit::Pluck(PROGRESSION[bar][0] / 2.0), 0.3);
        } else {
            track.hit(b, Hit::Snare, 0.6);
        }
        track.hit(b, Hit::Hat, 0.4);
        track.hit(within(beats, i, 0.5), Hit::Hat, 0.4);
        if i % 4 == 0 {
            let end = beats.get(i + 4).copied().unwrap_or(seconds);
            track.chord(b, end, &PROGRESSION[bar], 0.1);
        }
    }
    track
}

/// Music in 6/8 on `beats` (the dotted quarter notes, two to a bar): a
/// hi-hat on each of the three eighth notes of a beat, the kick and a
/// bass note on the first beat of a bar, the snare on the second, and
/// a chord a bar. The eighth notes are the steadiest pulse.
pub fn six_eight(seconds: f64, beats: &[f64]) -> Track {
    let mut track = Track::new(seconds);
    for (i, &b) in beats.iter().enumerate() {
        let bar = (i / 2) % 4;
        if i % 2 == 0 {
            track.hit(b, Hit::Kick, 0.8);
            track.hit(b, Hit::Pluck(PROGRESSION[bar][0] / 2.0), 0.3);
            let end = beats.get(i + 2).copied().unwrap_or(seconds);
            track.chord(b, end, &PROGRESSION[bar], 0.1);
        } else {
            track.hit(b, Hit::Snare, 0.6);
        }
        for k in 0..3 {
            track.hit(within(beats, i, f64::from(k) / 3.0), Hit::Hat, 0.4);
        }
    }
    track
}

/// Arpeggios without drums on `beats` (the quarter notes), as in a
/// Baroque prelude: four even sixteenth notes of the chord to a beat, a
/// bass note on each beat, and the chord changing each bar of four.
pub fn arpeggios(seconds: f64, beats: &[f64]) -> Track {
    let mut track = Track::new(seconds);
    for (i, &b) in beats.iter().enumerate() {
        let chord = PROGRESSION[(i / 4) % 4];
        track.hit(b, Hit::Pluck(chord[0] / 2.0), 0.35);
        for k in 0..4u8 {
            let note = chord[(usize::from(k) + i) % 3] * 2.0;
            track.hit(within(beats, i, f64::from(k) / 4.0), Hit::Pluck(note), 0.2);
        }
    }
    track
}
