//! What a listener hears in the beat, as curves (onset detection
//! functions).
//!
//! From the [`Parts`] of the music, frame by frame and each in `0..=1`:
//! the kick drum, the snare (and claps), the hi-hats, bass notes, and
//! chord changes. The meter reads them at the beats it considers: the
//! snare marks every second beat, kick and bass land on beats, and
//! chords change on beats and bars rather than between them. From stems,
//! the drum cues count only as far as the drums are present, and the
//! bass cue as far as the bass is ([`crate::presence::present`]): what
//! leaks into an absent stem, scaled up like any onset, is no cue.

use crate::activation::{normalised, Curve, Flux};
use crate::frontend::Mel;
use crate::parts::Parts;
use crate::presence::{self, Levels};

/// The span on each side of a frame whose spectra are compared for a
/// chord change, in seconds.
const CHANGE_SPAN: f64 = 0.15;

/// The cues of a track.
#[derive(Clone, Debug, Default)]
pub struct Cues {
    /// Kick-drum onsets.
    pub kick: Curve,
    /// Snare and clap onsets.
    pub snare: Curve,
    /// Hi-hat and cymbal onsets.
    pub hats: Curve,
    /// Bass-note onsets.
    pub bass: Curve,
    /// How much the tonal sound changes here: chords and bass lines.
    pub change: Curve,
    /// How present the drums are ([`crate::presence::present`]), from
    /// stems; `None`: everywhere. The kick, snare and hats are scaled by
    /// it.
    pub drums_present: Option<Curve>,
    /// How present the bass is, likewise; the bass cue is scaled by it.
    pub bass_present: Option<Curve>,
}

impl Cues {
    /// The cues of `parts`.
    pub fn new(parts: &Parts) -> Cues {
        let present = |energy: fn(&Levels) -> &Vec<f32>| {
            parts
                .levels
                .as_ref()
                .map(|l| presence::present(energy(l), &l.total, l.fps))
        };
        let (drums_present, bass_present) = (present(|l| &l.drums), present(|l| &l.bass));
        let by = |curve: Curve, present: &Option<Curve>| match present {
            Some(p) => presence::scaled(&curve, p),
            None => curve,
        };
        Cues {
            kick: by(Flux::KICK.curve(&parts.drums), &drums_present),
            snare: by(Flux::SNARE.curve(&parts.drums), &drums_present),
            hats: by(Flux::HATS.curve(&parts.drums), &drums_present),
            bass: by(Flux::BASS.curve(&parts.tonal), &bass_present),
            change: change(&parts.tonal),
            drums_present,
            bass_present,
        }
    }
}

/// The tonal change at each frame: the mean absolute difference between
/// the mean spectrum over the 0.15 s before the frame and that
/// after it, less its slowly moving part, scaled by the largest value
/// within three seconds (as the flux is).
pub fn change(mel: &Mel) -> Curve {
    let (frames, bands) = (mel.frames(), mel.bands);
    let fps = mel.fps;
    if frames == 0 {
        return Curve {
            values: Vec::new(),
            fps,
        };
    }
    let span = ((CHANGE_SPAN * fps).round() as usize).max(1);
    // Each band's sum over the frames before `t` (`t - span..t`) and from
    // it on (`t..t + span`), carried along as `t` moves.
    let add = |sums: &mut [f64], t: usize, sign: f64| {
        for (s, &v) in sums.iter_mut().zip(mel.frame(t)) {
            *s += sign * f64::from(v);
        }
    };
    let (mut before, mut after) = (vec![0.0f64; bands], vec![0.0f64; bands]);
    for t in 0..span.min(frames) {
        add(&mut after, t, 1.0);
    }
    let mut raw = vec![0.0f32; frames];
    for (t, r) in raw.iter_mut().enumerate() {
        if t > 0 {
            add(&mut before, t - 1, 1.0);
            add(&mut after, t - 1, -1.0);
            if t > span {
                add(&mut before, t - 1 - span, -1.0);
            }
            if t + span <= frames {
                add(&mut after, t + span - 1, 1.0);
            }
        }
        let (lo, hi) = (t.saturating_sub(span), (t + span).min(frames));
        if lo == t || hi == t {
            continue;
        }
        let (n_before, n_after) = ((t - lo) as f64, (hi - t) as f64);
        let diff: f64 = before
            .iter()
            .zip(&after)
            .map(|(b, a)| (a / n_after - b / n_before).abs())
            .sum();
        *r = (diff / bands as f64) as f32;
    }
    normalised(&raw, fps, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{Frontend, FrontendConfig};
    use crate::synth::{Hit, Track, PROGRESSION};

    fn cues_of(track: &Track) -> Cues {
        let mut frontend = Frontend::new(FrontendConfig::FLUX);
        frontend.push(&track.samples);
        Cues::new(&Parts::split(&frontend.finish()))
    }

    /// The mean of `curve` near each of `times`.
    fn at(curve: &Curve, times: impl Iterator<Item = f64>) -> f64 {
        let v: Vec<f64> = times.map(|t| f64::from(curve.max_near(t, 0.03))).collect();
        v.iter().sum::<f64>() / v.len() as f64
    }

    #[test]
    fn each_drum_lights_its_own_cue() {
        let mut track = Track::new(12.0);
        // Kick on the even half seconds, snare on the odd ones.
        for i in 0..24 {
            let t = 0.25 + 0.5 * f64::from(i);
            let hit = if i % 2 == 0 { Hit::Kick } else { Hit::Snare };
            track.hit(t, hit, 0.6);
        }
        let cues = cues_of(&track);
        let kicks = || (0..12).map(|i| 0.25 + f64::from(i));
        let snares = || (0..12).map(|i| 0.75 + f64::from(i));
        assert!(at(&cues.kick, kicks()) > 2.0 * at(&cues.kick, snares()));
        assert!(at(&cues.snare, snares()) > 2.0 * at(&cues.snare, kicks()));
    }

    #[test]
    fn chord_changes_light_the_change_cue() {
        let mut track = Track::new(16.0);
        // A chord every two seconds, nothing else.
        for (i, chord) in PROGRESSION.iter().cycle().take(8).enumerate() {
            let start = 2.0 * i as f64;
            track.chord(start, start + 2.0, chord, 0.15);
        }
        let cues = cues_of(&track);
        let changes = (1..8).map(|i| 2.0 * f64::from(i));
        let between = (1..8).map(|i| 2.0 * f64::from(i) - 1.0);
        let (on, off) = (at(&cues.change, changes), at(&cues.change, between));
        assert!(on > 0.5 && on > 4.0 * off, "{on} {off}");
    }
}
