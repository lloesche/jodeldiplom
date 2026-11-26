//! The key a song is in: our own inference of MusicalKeyCNN's KeyNet
//! (Alexander Sommer, after Korzeniowski and Widmer, "Genre-Agnostic Key
//! Classification With Convolutional Neural Networks", ISMIR 2018; code and
//! weights under the MIT licence, see `models/LICENSE-musical-key-cnn`).
//!
//! The song, at 44.1 kHz mono, becomes a constant-Q spectrum as the
//! authors' `librosa.cqt` makes it: 105 bins, 24 an octave from 65 Hz, a
//! frame every 8820 samples (0.2 s), each bin's magnitude through
//! `log1p`. With these settings librosa's octave-by-octave downsampling
//! scales out exactly, so each bin is computed directly at the full rate:
//! √L times the Hann-windowed (and window-normalised) DFT at the bin's
//! frequency over the bin's L samples, L = Q·44100/f. The whole song (but
//! its last two frames, as the authors' prediction drops them) goes
//! through the network at once: nine convolutions with batch norm (folded)
//! and ELU, three 2×2 max pools, the average over all of it; the largest
//! of the 24 scores is the key.

use std::f64::consts::PI;
use std::sync::OnceLock;

use jodel_dsp::window::hann;
use jodel_tensors::kernels::{par_chunks, threads};

use crate::nn::weights::Weights;

/// The weights, built in (`tools/convert_keynet.py`).
static WEIGHTS: &[u8] = include_bytes!("../models/keynet.bin");

/// The rate the spectrum is made at.
pub const RATE: u32 = 44_100;
/// Samples between frames.
pub const HOP: usize = 8_820;
/// The spectrum's bins.
pub const BINS: usize = 105;
/// Bins an octave.
const PER_OCTAVE: f64 = 24.0;
/// The lowest bin's frequency.
const FMIN: f64 = 65.0;

/// A key, as the network numbers them: 0 to 11 minor, 12 to 23 major, each
/// in the Camelot wheel's order (0 G♯ minor, 1A; 12 B major, 1B; each next
/// a fifth up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    /// The class, 0 to 23.
    pub class: u8,
    /// How sure the network is: the class's share of the 24 (softmax);
    /// 1 for a key from a tag.
    pub confidence: f32,
    /// Whether it was read from the file's tags rather than heard.
    pub tagged: bool,
}

/// The pitch classes' names, C first; the spelling usual for a key's
/// tonic (major, minor).
const MAJOR_NAMES: [&str; 12] = [
    "C", "D♭", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B",
];
const MINOR_NAMES: [&str; 12] = [
    "C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "G♯", "A", "B♭", "B",
];

impl Key {
    /// The key on `tonic` (0 C … 11 B), minor or major.
    pub fn on(tonic: u8, minor: bool) -> Key {
        // A fifth's inverse round the twelve is a fifth (7 · 7 = 49).
        let (first, base) = if minor { (8, 0) } else { (11, 12) };
        let step = (u32::from(tonic % 12) + 12 - first) * 7 % 12;
        Key {
            class: base + step as u8,
            confidence: 1.0,
            tagged: false,
        }
    }

    /// The key a tag names (`INITIALKEY`, `TKEY`): Camelot ("8A"), Open Key
    /// ("1d", "1m"), or a note with its mode ("Am", "E♭", "F# minor",
    /// "Bbmaj"); `None` for anything else ("o", the off-key mark, too).
    pub fn from_tag(text: &str) -> Option<Key> {
        let t = text.trim();
        let digits = t.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 {
            let n: u8 = t[..digits].parse().ok().filter(|n| (1..=12).contains(n))?;
            let place = match t[digits..].trim() {
                "A" | "a" => (n - 1, true),
                "B" | "b" => (n - 1, false),
                // Open Key: 1d is C major, 8B on the Camelot wheel.
                "m" | "M" => ((n + 6) % 12, true),
                "d" | "D" => ((n + 6) % 12, false),
                _ => return None,
            };
            return Some(Key {
                class: place.0 + if place.1 { 0 } else { 12 },
                confidence: 1.0,
                tagged: true,
            });
        }
        let mut chars = t.chars();
        let letter = chars.next()?.to_ascii_uppercase();
        let natural = match letter {
            'C' => 0,
            'D' => 2,
            'E' => 4,
            'F' => 5,
            'G' => 7,
            'A' => 9,
            'B' => 11,
            _ => return None,
        };
        let rest = chars.as_str();
        let (shift, rest) = match rest.chars().next() {
            Some('#' | '♯') => (1, &rest[rest.chars().next()?.len_utf8()..]),
            Some('b' | '♭') => (11, &rest[rest.chars().next()?.len_utf8()..]),
            _ => (0, rest),
        };
        let minor = match rest.trim().to_ascii_lowercase().as_str() {
            "" | "maj" | "major" | "dur" => false,
            "m" | "min" | "minor" | "moll" => true,
            _ => return None,
        };
        Some(Key {
            tagged: true,
            ..Key::on((natural + shift) % 12, minor)
        })
    }

    /// Whether it is a minor key.
    pub fn minor(&self) -> bool {
        self.class < 12
    }

    /// The tonic's pitch class (0 C, 1 C♯ … 11 B).
    pub fn tonic(&self) -> u8 {
        let step = self.class % 12;
        // A step round the wheel is a fifth (7 semitones); 1A is G♯
        // minor, 1B B major.
        let first = if self.minor() { 8 } else { 11 };
        (first + 7 * step) % 12
    }

    /// Its name: "A minor", "E♭ major".
    pub fn name(&self) -> String {
        let t = usize::from(self.tonic());
        if self.minor() {
            format!("{} minor", MINOR_NAMES[t])
        } else {
            format!("{} major", MAJOR_NAMES[t])
        }
    }

    /// Its place on the Camelot wheel: "8A" (A minor), "8B" (C major).
    pub fn camelot(&self) -> String {
        let mode = if self.minor() { 'A' } else { 'B' };
        format!("{}{mode}", self.class % 12 + 1)
    }

    /// The pitch classes of its scale (major, or natural minor), from the
    /// tonic.
    pub fn scale(&self) -> [u8; 7] {
        let steps: [u8; 7] = if self.minor() {
            [0, 2, 3, 5, 7, 8, 10]
        } else {
            [0, 2, 4, 5, 7, 9, 11]
        };
        steps.map(|s| (self.tonic() + s) % 12)
    }
}

/// The frames the spectrum of `len` samples has: librosa's for each octave
/// rate (the full one, half, a quarter), the fewest.
fn frames(len: usize) -> usize {
    let half = len.div_ceil(2);
    let quarter = half.div_ceil(2);
    (1 + len / HOP)
        .min(1 + half / (HOP / 2))
        .min(1 + quarter / (HOP / 4))
}

/// The constant-Q spectrum of `samples` (44.1 kHz mono): `log1p` of each
/// bin's magnitude, bin by bin, `[BINS][frames]`; and the frames.
pub fn spectrum(samples: &[f32]) -> (Vec<f32>, usize) {
    let n = frames(samples.len());
    let mut out = vec![0.0f32; BINS * n];
    // librosa's filter lengths: Q = 1/α, α = (r² − 1)/(r² + 1),
    // r = 2^(1/24).
    let r2 = 2f64.powf(2.0 / PER_OCTAVE);
    let q = (r2 + 1.0) / (r2 - 1.0);
    par_chunks(&mut out, n.max(1), threads(), |k, row| {
        let f = FMIN * 2f64.powf(k as f64 / PER_OCTAVE);
        let length = q * f64::from(RATE) / f;
        // The taps: np.arange(-L//2, L//2) (floats), a periodic Hann
        // window over them, normalised to sum to one.
        let first = (-length / 2.0).floor() as i64;
        let count = ((length / 2.0).floor() as i64 - first) as usize;
        let w = hann(count);
        let sum: f64 = w.iter().sum();
        let omega = 2.0 * PI * f / f64::from(RATE);
        let (re, im): (Vec<f32>, Vec<f32>) = w
            .iter()
            .enumerate()
            .map(|(i, &wi)| {
                let phase = omega * (first + i as i64) as f64;
                let g = wi / sum;
                ((g * phase.cos()) as f32, (-g * phase.sin()) as f32)
            })
            .unzip();
        let gain = length.sqrt() as f32;
        for (t, value) in row.iter_mut().enumerate() {
            let centre = (t * HOP) as i64;
            let start = centre + first;
            // The taps that fall within the song (zeros beyond it).
            let from = (-start).max(0) as usize;
            let to = count.min((samples.len() as i64 - start).max(0) as usize);
            let (mut a, mut b) = (0.0f32, 0.0f32);
            if from < to {
                let x = &samples[(start + from as i64) as usize..(start + to as i64) as usize];
                for ((&xv, &c), &s) in x.iter().zip(&re[from..to]).zip(&im[from..to]) {
                    a += xv * c;
                    b += xv * s;
                }
            }
            *value = (gain * (a * a + b * b).sqrt()).ln_1p();
        }
    });
    (out, n)
}

/// A convolution with its batch norm folded in, then ELU.
struct Block {
    weight: Vec<f32>,
    scale: Vec<f32>,
    shift: Vec<f32>,
    out: usize,
    inp: usize,
    k: usize,
}

impl Block {
    fn load(
        w: &mut Weights,
        name: &str,
        out: usize,
        inp: usize,
        k: usize,
    ) -> Result<Block, String> {
        Ok(Block {
            weight: w.take(&format!("{name}.conv.weight"), &[out, inp, k, k])?,
            scale: w.take(&format!("{name}.bn.scale"), &[out])?,
            shift: w.take(&format!("{name}.bn.shift"), &[out])?,
            out,
            inp,
            k,
        })
    }

    /// `x` (`[inp][f][t]`) through it, `[out][f][t]`: zero padding keeps
    /// the size ("same").
    fn run(&self, x: &[f32], f: usize, t: usize) -> Vec<f32> {
        let plane = f * t;
        let pad = self.k / 2;
        let mut y = vec![0.0f32; self.out * plane];
        par_chunks(&mut y, plane, threads(), |o, out| {
            for i in 0..self.inp {
                let src = &x[i * plane..(i + 1) * plane];
                for dy in 0..self.k {
                    for dx in 0..self.k {
                        let wv = self.weight[((o * self.inp + i) * self.k + dy) * self.k + dx];
                        // Output (r, c) takes input (r + dy − pad, c + dx − pad).
                        let r0 = pad.saturating_sub(dy);
                        let r1 = (f + pad).saturating_sub(dy).min(f);
                        let c0 = pad.saturating_sub(dx);
                        let c1 = (t + pad).saturating_sub(dx).min(t);
                        if c0 >= c1 {
                            continue;
                        }
                        for r in r0..r1 {
                            let sr = r + dy - pad;
                            let dst = &mut out[r * t + c0..r * t + c1];
                            let s = &src[sr * t + c0 + dx - pad..sr * t + c1 + dx - pad];
                            for (d, &v) in dst.iter_mut().zip(s) {
                                *d += wv * v;
                            }
                        }
                    }
                }
            }
            let (scale, shift) = (self.scale[o], self.shift[o]);
            for v in out.iter_mut() {
                let z = *v * scale + shift;
                *v = if z > 0.0 { z } else { z.exp_m1() };
            }
        });
        y
    }
}

/// 2×2 max pooling of `[c][f][t]` (an odd last row or column dropped).
fn pool(x: &[f32], c: usize, f: usize, t: usize) -> (Vec<f32>, usize, usize) {
    let (f2, t2) = (f / 2, t / 2);
    let mut y = vec![0.0f32; c * f2 * t2];
    for ch in 0..c {
        let src = &x[ch * f * t..(ch + 1) * f * t];
        let dst = &mut y[ch * f2 * t2..(ch + 1) * f2 * t2];
        for r in 0..f2 {
            for col in 0..t2 {
                let at = |rr: usize, cc: usize| src[(2 * r + rr) * t + 2 * col + cc];
                dst[r * t2 + col] = at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1));
            }
        }
    }
    (y, f2, t2)
}

/// The network.
pub struct KeyNet {
    blocks: Vec<Block>,
}

impl KeyNet {
    /// The network with `weights`.
    pub fn load(mut w: Weights) -> Result<KeyNet, String> {
        let shapes = [
            (20, 1, 5),
            (20, 20, 3),
            (40, 20, 3),
            (40, 40, 3),
            (80, 40, 3),
            (80, 80, 3),
            (160, 80, 3),
            (160, 160, 3),
            (24, 160, 1),
        ];
        let blocks = shapes
            .iter()
            .enumerate()
            .map(|(i, &(o, inp, k))| Block::load(&mut w, &format!("conv{}", i + 1), o, inp, k))
            .collect::<Result<Vec<_>, _>>()?;
        if !w.left().is_empty() {
            return Err(format!("weights left over: {:?}", w.left()));
        }
        Ok(KeyNet { blocks })
    }

    /// The network with the built-in weights, made once.
    pub fn built_in() -> &'static KeyNet {
        static NET: OnceLock<KeyNet> = OnceLock::new();
        NET.get_or_init(|| {
            Weights::parse(WEIGHTS)
                .and_then(KeyNet::load)
                .expect("the built-in key weights load")
        })
    }

    /// The 24 scores for a spectrum `x` (`[BINS][t]`), and each block's
    /// output on the way (conv1, conv2, pool1, conv3, conv4, pool2, conv5,
    /// conv6, pool3, conv7, conv8, conv9), for checking against the
    /// original.
    pub fn run(&self, x: &[f32], t: usize) -> ([f32; 24], Vec<Vec<f32>>) {
        let mut steps = Vec::new();
        let (mut h, mut f, mut t) = (x.to_vec(), BINS, t);
        let mut c = 1;
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.run(&h, f, t);
            c = block.out;
            steps.push(h.clone());
            if matches!(i, 1 | 3 | 5) {
                (h, f, t) = pool(&h, c, f, t);
                steps.push(h.clone());
            }
        }
        let plane = (f * t).max(1);
        let mut scores = [0.0f32; 24];
        for (o, s) in scores.iter_mut().enumerate().take(c) {
            *s = h[o * plane..(o + 1) * plane].iter().sum::<f32>() / plane as f32;
        }
        (scores, steps)
    }
}

/// The key of a song, `samples` its 44.1 kHz mono samples; `None` where it
/// is too short to tell (under about 2 s).
pub fn key_of(samples: &[f32]) -> Option<Key> {
    let (spec, n) = spectrum(samples);
    // The authors' prediction drops the last two frames.
    let t = n.checked_sub(2).filter(|&t| t >= 8)?;
    let mut x = vec![0.0f32; BINS * t];
    for b in 0..BINS {
        x[b * t..(b + 1) * t].copy_from_slice(&spec[b * n..b * n + t]);
    }
    let (scores, _) = KeyNet::built_in().run(&x, t);
    Some(key_from(&scores))
}

/// The key the scores choose, and the softmax share of its score.
pub fn key_from(scores: &[f32; 24]) -> Key {
    let (class, &best) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap_or((0, &0.0));
    let total: f32 = scores.iter().map(|&s| (s - best).exp()).sum();
    Key {
        class: class as u8,
        confidence: 1.0 / total.max(1.0),
        tagged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classes name the keys as the Camelot wheel does.
    #[test]
    fn the_classes_are_the_wheel() {
        let key = |class| Key {
            class,
            confidence: 1.0,
            tagged: false,
        };
        assert_eq!(
            (key(7).name(), key(7).camelot()),
            ("A minor".into(), "8A".into())
        );
        assert_eq!(
            (key(19).name(), key(19).camelot()),
            ("C major".into(), "8B".into())
        );
        assert_eq!(
            (key(0).name(), key(0).camelot()),
            ("G♯ minor".into(), "1A".into())
        );
        assert_eq!(
            (key(12).name(), key(12).camelot()),
            ("B major".into(), "1B".into())
        );
        assert_eq!(
            (key(23).name(), key(23).camelot()),
            ("E major".into(), "12B".into())
        );
        assert_eq!(key(19).scale(), [0, 2, 4, 5, 7, 9, 11]);
        assert_eq!(key(7).scale(), [9, 11, 0, 2, 4, 5, 7]);
        for class in 0..24 {
            let k = key(class);
            assert_eq!(Key::on(k.tonic(), k.minor()), k);
        }
    }

    /// Tags in each spelling name the same keys.
    #[test]
    fn tags_name_keys() {
        let named = |t: &str| Key::from_tag(t).map(|k| k.name());
        for (tag, name) in [
            ("8A", "A minor"),
            ("8B", "C major"),
            ("12B", "E major"),
            ("1d", "C major"),
            ("1m", "A minor"),
            ("12d", "F major"),
            ("Am", "A minor"),
            ("A minor", "A minor"),
            ("Ebm", "E♭ minor"),
            ("E♭", "E♭ major"),
            ("F#", "F♯ major"),
            ("bm", "B minor"),
            ("Bbmaj", "B♭ major"),
            (" c# Min ", "C♯ minor"),
        ] {
            assert_eq!(named(tag).as_deref(), Some(name), "{tag}");
        }
        for tag in ["o", "", "13A", "0B", "H", "Am7", "8C"] {
            assert_eq!(Key::from_tag(tag), None, "{tag}");
        }
        assert!(Key::from_tag("8A").unwrap().tagged);
    }

    /// The built-in weights load whole.
    #[test]
    fn the_weights_load() {
        let _ = KeyNet::built_in();
    }

    /// A C major chord held (C, E, G and their octaves, through the
    /// spectrum): the bins at its notes stand out of the others.
    #[test]
    fn a_chord_lights_its_bins() {
        let n = RATE as usize * 4;
        let notes = [261.63, 329.63, 392.0];
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                notes.iter().map(|f| (2.0 * PI * f * t).sin()).sum::<f64>() as f32 * 0.2
            })
            .collect();
        let (spec, frames) = spectrum(&x);
        let mid = frames / 2;
        let bin = |f: f64| ((f / FMIN).log2() * PER_OCTAVE).round() as usize;
        let at = |b: usize| spec[b * frames + mid];
        let between = at(bin(290.0));
        for f in notes {
            assert!(
                at(bin(f)) > 2.0 * between,
                "{f} Hz: {} against {between}",
                at(bin(f))
            );
        }
    }
}
