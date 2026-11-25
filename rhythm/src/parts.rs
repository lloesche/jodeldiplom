//! The music split into parts: drums and tonal sound.
//!
//! [`Parts`] is what the cues listen to. The split here is the fast one:
//! a harmonic/percussive separation of the mel spectrogram. Sustained
//! sound is smooth along time, so a median over 0.2 s keeps it; a drum
//! hit is smooth across frequency, so a median over 11 neighbouring
//! bands keeps that. Each band of each frame then goes to the two parts
//! by soft masks: each part's median squared over the sum of both
//! squared.
//!
//! A stem separator fills the same struct: the drum stem as `drums`,
//! bass and the other instruments as `tonal`. The cues do not change.
//! It also gives the stems' energy, so the drum stem can track beats
//! where the drums play and the cues can follow where the drums and the
//! bass are ([`crate::presence`]).

use crate::frontend::{compress, expand, Mel};
use crate::presence::Levels;
use crate::stats::upper_median;
use crate::waveform::Lane;

/// Frames on each side of the median along time (0.2 s at 100 frames a
/// second).
const TIME_RADIUS: usize = 10;
/// Bands on each side of the median across frequency.
const BAND_RADIUS: usize = 5;

/// The music in parts, each a mel spectrogram like the whole.
#[derive(Clone, Debug, Default)]
pub struct Parts {
    /// Drums and other hits.
    pub drums: Mel,
    /// Sustained, pitched sound: bass, chords, melody.
    pub tonal: Mel,
    /// The stems' energy frame by frame, from a stem separator.
    pub levels: Option<Levels>,
    /// Each stem's waveform (bass, drums, other, vocals), from a stem
    /// separator, for display.
    pub lanes: Option<[Lane; 4]>,
}

impl Parts {
    /// The fast split of `mel` into drums and tonal sound.
    pub fn split(mel: &Mel) -> Parts {
        let (frames, bands) = (mel.frames(), mel.bands);
        let x: Vec<f32> = mel.data.iter().map(|&v| expand(v)).collect();
        let at = |t: usize, b: usize| x[t * bands + b];
        let mut drums = vec![0.0; x.len()];
        let mut tonal = vec![0.0; x.len()];
        let mut along = Vec::with_capacity(2 * TIME_RADIUS + 1);
        let mut across = Vec::with_capacity(2 * BAND_RADIUS + 1);
        for t in 0..frames {
            let (t0, t1) = (
                t.saturating_sub(TIME_RADIUS),
                (t + TIME_RADIUS + 1).min(frames),
            );
            for b in 0..bands {
                along.clear();
                along.extend((t0..t1).map(|u| at(u, b)));
                let h = upper_median(&mut along, f32::total_cmp).unwrap_or(0.0);
                across.clear();
                let (b0, b1) = (
                    b.saturating_sub(BAND_RADIUS),
                    (b + BAND_RADIUS + 1).min(bands),
                );
                across.extend((b0..b1).map(|c| at(t, c)));
                let p = upper_median(&mut across, f32::total_cmp).unwrap_or(0.0);
                let (h2, p2) = (h * h, p * p);
                let total = h2 + p2;
                let v = at(t, b);
                let i = t * bands + b;
                if total > 0.0 {
                    drums[i] = compress(v * p2 / total);
                    tonal[i] = compress(v * h2 / total);
                }
            }
        }
        let part = |data| Mel {
            fps: mel.fps,
            bands,
            centers: mel.centers.clone(),
            data,
        };
        Parts {
            drums: part(drums),
            tonal: part(tonal),
            levels: None,
            lanes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{Frontend, FrontendConfig};
    use crate::synth::{Hit, Track};

    /// The mean magnitude in `mel` over `frames` and the bands between
    /// `low_hz` and `high_hz`.
    fn energy(mel: &Mel, frames: std::ops::Range<usize>, low_hz: f64, high_hz: f64) -> f64 {
        let bands = mel.bands_between(low_hz, high_hz);
        let mut sum = 0.0;
        let mut count = 0.0;
        for t in frames {
            for b in bands.clone() {
                sum += f64::from(expand(mel.frame(t)[b]));
                count += 1.0;
            }
        }
        sum / count
    }

    #[test]
    fn a_held_chord_is_tonal_and_hits_are_drums() {
        let mut track = Track::new(6.0);
        track.chord(0.0, 6.0, &[220.0, 277.2, 329.6], 0.2);
        for i in 0..12 {
            track.hit(0.25 + 0.5 * f64::from(i), Hit::Snare, 0.5);
        }
        let mut frontend = Frontend::new(FrontendConfig::FLUX);
        frontend.push(&track.samples);
        let parts = Parts::split(&frontend.finish());
        // Between the hits, the chord's band is tonal.
        let gap = 140..160; // 1.40–1.60 s, between the hits at 1.25 and 1.75
        let chord_tonal = energy(&parts.tonal, gap.clone(), 210.0, 230.0);
        let chord_drums = energy(&parts.drums, gap, 210.0, 230.0);
        // Low down the bands are only a few FFT bins wide, so the partials
        // leak across them and the split is less clean there.
        assert!(
            chord_tonal > 2.0 * chord_drums,
            "{chord_tonal} {chord_drums}"
        );
        // At a hit, high up where the chord has nothing, it is drums.
        let hit = 175..178;
        let hit_drums = energy(&parts.drums, hit.clone(), 3000.0, 6000.0);
        let hit_tonal = energy(&parts.tonal, hit, 3000.0, 6000.0);
        assert!(hit_drums > 3.0 * hit_tonal, "{hit_drums} {hit_tonal}");
    }
}
