//! FORMANT: a sung or whispered vowel, for "aww", "ooh" choirs and
//! whispers.
//!
//! The source is the time derivative of a Rosenberg glottal pulse (his
//! trigonometric pulse: a raised-cosine opening, a quarter-cosine closing,
//! then the glottis closed). Its abrupt end at the closure is band-limited
//! with a polyBLEP, then a one-pole low-pass tilts its spectrum (softer
//! voices lose their highs). Aspiration noise is added, gated by the
//! glottal flow as Klatt does: loud while the glottis is open, quieter
//! while it is closed; with all breath the voicing is gone and the noise
//! flows freely, a whisper. Four formant resonators follow in cascade, each
//! Klatt's y = A x + B y1 + C y2 with C = -e^(-2 pi BW T),
//! B = 2 e^(-pi BW T) cos(2 pi F T) and A = 1 - B - C, so each has unity
//! gain at DC and the cascade too.
//!
//! What the host's [`VoiceMod`] does here: `phase_inc` sets the pitch (a
//! slow random drift rides on it; the vibrato is the patch's LFO, delayed
//! and fading in); `shape` moves the vowel ([`SHAPE`]); `bright` moves the
//! spectral tilt's corner ([`BRIGHT`]). The velocity is the host's level.
//!
//! Sources: Klatt (1980, "Software for a cascade/parallel formant
//! synthesizer", JASA 67) for the cascade, its resonators and the gated
//! aspiration; Hillenbrand, Getty, Clark and Wheeler (1995, "Acoustic
//! characteristics of American English vowels", JASA 97) for the vowels'
//! formants; Rosenberg (1971, "Effect of glottal pulse shape on the quality
//! of natural vowels", JASA 49) for the pulse; polyBLEP as in
//! [`crate::dsp`].

use crate::dsp::{self, Rate};
use crate::{Engine, NoteOn, VoiceMod, BLOCK};
use std::f32::consts::{PI, TAU};

/// F1, F2, F3 in Hz of the vowels i (heed), e (hayed), a (hod), o (hoed),
/// u (who'd), for adult men and women: the averages of Hillenbrand et al.
/// (1995), table V.
const MEN: [[f32; 3]; 5] = [
    [342.0, 2322.0, 3000.0],
    [476.0, 2089.0, 2691.0],
    [768.0, 1333.0, 2522.0],
    [497.0, 910.0, 2459.0],
    [378.0, 997.0, 2343.0],
];
const WOMEN: [[f32; 3]; 5] = [
    [437.0, 2761.0, 3372.0],
    [536.0, 2530.0, 3047.0],
    [936.0, 1551.0, 2815.0],
    [555.0, 1035.0, 2828.0],
    [459.0, 1105.0, 2735.0],
];

/// F4 in Hz for men and women: our choice (the table has no F4), above
/// every F3 there.
const F4: [f32; 2] = [3500.0, 4200.0];

/// The formants' bandwidths in Hz, F1 to F4: our choice, of the order of
/// those Klatt (1980) lists for vowels.
const BANDWIDTH: [f32; 4] = [80.0, 90.0, 120.0, 200.0];

/// The slow random pitch drift's size, cents (its spread about a tenth).
const DRIFT: f32 = 30.0;

/// The aspiration noise's level against the voicing's.
const NOISE: f32 = 1.8;

/// The output gain: a full voice peaks at about 1 before the voice level.
const OUTPUT: f32 = 0.24;

/// The vowel parameter of each vowel.
pub const I: f32 = 0.0;
/// "e".
pub const E: f32 = 0.25;
/// "a".
pub const A: f32 = 0.5;
/// "o".
pub const O: f32 = 0.75;
/// "u".
pub const U: f32 = 1.0;

/// FORMANT's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The vowel, 0 to 1 along [`I`], [`E`], [`A`], [`O`], [`U`].
    pub vowel: f32,
    /// Aspiration against voicing, 0 (a clear voice) to 1 (a whisper).
    pub breath: f32,
    /// The vocal tract, 0 (a man's formants) to 1 (a woman's).
    pub size: f32,
    /// The glottal pulse, 0 (pressed and bright: open half a period, a
    /// 5 kHz tilt) to 1 (soft and dark: open 80 %, 0.9 kHz).
    pub soft: f32,
    /// How wide the formants are, times [`BANDWIDTH`] (1: as spoken; wider
    /// for a voice sung high, whose few harmonics then come through every
    /// vowel alike instead of one ringing where it meets a formant).
    pub width: f32,
}

impl Default for Params {
    fn default() -> Params {
        SUNG
    }
}

/// [`Params::default`]: a man's clear "a".
const SUNG: Params = Params {
    vowel: A,
    breath: 0.0,
    size: 0.0,
    soft: 0.3,
    width: 1.0,
};

/// A woman's soft, breathy "o": glided from "a" (and given a vibrato) by
/// the patch, an "aww".
pub const AWW: Params = Params {
    vowel: O,
    breath: 0.25,
    size: 1.0,
    soft: 0.6,
    width: 1.0,
};

/// A choir's "ooh".
pub const OOH: Params = Params {
    vowel: 0.95,
    breath: 0.2,
    size: 0.6,
    soft: 0.7,
    width: 1.0,
};

/// A low, breathy whisper.
pub const WHISPER: Params = Params {
    vowel: 0.6,
    breath: 0.92,
    size: 0.0,
    soft: 0.8,
    width: 1.0,
};

/// What the host's shape moves: the vowel, by up to half its range.
pub const SHAPE: &str = "vowel";

/// What the host's brightness opens: the spectral tilt's corner, an octave
/// an octave.
pub const BRIGHT: &str = "tilt";

/// The presets, by name.
pub const PRESETS: [(&str, Params); 3] = [("AWW", AWW), ("OOH", OOH), ("WHISPER", WHISPER)];

/// One voice: the pulse, the pitch's wobble, the resonators.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voice {
    phase: u32,
    /// The slow random pitch drift, cents.
    drift: f32,
    rng: u32,
    tilt: f32,
    y1: [f32; 4],
    y2: [f32; 4],
}

/// FORMANT at one rate.
#[derive(Clone, Copy, Debug)]
pub struct Formant {
    rate: Rate,
}

impl Formant {
    /// Klatt's resonator coefficients (A, B, C) for `f` and `bw` in Hz.
    fn resonator(&self, f: f32, bw: f32) -> [f32; 3] {
        let t = 1.0 / self.rate.hz();
        let f = f.min(0.45 * self.rate.hz());
        let c = -(-TAU * bw * t).exp();
        let b = 2.0 * (-PI * bw * t).exp() * (TAU * f * t).cos();
        [1.0 - b - c, b, c]
    }
}

/// The formant frequencies at `vowel` and tract `size`.
fn formants(vowel: f32, size: f32) -> [f32; 4] {
    let x = vowel.clamp(0.0, 1.0) * 4.0;
    let i = (x as usize).min(3);
    let t = x - i as f32;
    let size = size.clamp(0.0, 1.0);
    let at = |table: &[[f32; 3]; 5], k: usize| table[i][k] + (table[i + 1][k] - table[i][k]) * t;
    let mix = |k: usize| at(&MEN, k) + (at(&WOMEN, k) - at(&MEN, k)) * size;
    [mix(0), mix(1), mix(2), F4[0] + (F4[1] - F4[0]) * size]
}

impl Engine for Formant {
    type Params = Params;
    type Voice = Voice;

    fn new(rate: Rate) -> Formant {
        Formant { rate }
    }

    /// A new drift from the note's seed. A retrigger keeps the pulse's
    /// phase and the resonators: a legato re-attack.
    fn note_on(&self, _p: &Params, v: &mut Voice, n: &NoteOn) {
        v.rng = dsp::seed(n.seed);
    }

    fn render(&self, p: &Params, v: &mut Voice, m: &VoiceMod, out: &mut [f32]) {
        let n = out.len().min(BLOCK);

        // The pitch: a slow random drift on it.
        v.drift += 0.02 * (dsp::white(&mut v.rng) * DRIFT - v.drift);
        let inc = (m.phase_inc as f32 * (v.drift / 1_200.0).exp2()) as u32;

        // The pulse: opening for 70 % of the open phase, closing for 30 %,
        // scaled so the closing slope peaks at -1.
        let soft = p.soft.clamp(0.0, 1.0);
        let open = 0.5 + 0.3 * soft;
        let rise = 0.7 * open;
        let fall = 0.3 * open;
        let (open_k, close_k) = (PI / rise, PI / (2.0 * fall));
        let scale = 1.0 / close_k;
        let closure = (open * 4_294_967_296.0) as u32;
        let tilt_hz = 5_000.0 * (-2.5 * soft + m.bright).exp2();
        let tilt = self.rate.pole(tilt_hz.min(0.45 * self.rate.hz()));

        let breath = p.breath.clamp(0.0, 1.0);
        let voicing = 1.0 - breath;
        let noise = breath * NOISE;
        // How much the glottis gates the noise: none in a whisper.
        let gate = 0.8 * voicing;

        let f = formants(p.vowel + 0.5 * m.shape, p.size);
        let width = p.width.clamp(0.25, 8.0);
        let k: [[f32; 3]; 4] = std::array::from_fn(|i| self.resonator(f[i], BANDWIDTH[i] * width));

        for o in out[..n].iter_mut() {
            let phase = v.phase as f32 * (1.0 / 4_294_967_296.0);
            // The closure steps the derivative from -1 back to 0.
            let step = 0.5 * dsp::blep(v.phase.wrapping_sub(closure), inc);
            v.phase = v.phase.wrapping_add(inc);
            let (pulse, flow) = if phase < rise {
                let (s, c) = (open_k * phase).sin_cos();
                (0.5 * open_k * s * scale, 0.5 * (1.0 - c))
            } else if phase < open {
                let (s, c) = (close_k * (phase - rise)).sin_cos();
                (-s, c)
            } else {
                (0.0, 0.0)
            };
            v.tilt += tilt * ((pulse + step) * voicing - v.tilt);
            let hiss = (dsp::white(&mut v.rng) + dsp::white(&mut v.rng)) * 0.5;
            let mut x = v.tilt + hiss * noise * (1.0 - gate + gate * flow);
            for (j, [a, b, c]) in k.iter().enumerate() {
                let y = a * x + b * v.y1[j] + c * v.y2[j];
                v.y2[j] = v.y1[j];
                v.y1[j] = y;
                x = y;
            }
            *o += x * OUTPUT;
        }
    }
}
