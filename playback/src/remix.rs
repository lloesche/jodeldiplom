//! A file's channels made stereo, for a device (a desk) that takes stereo
//! only ([`Remix`]): mono doubled onto both sides, exactly; more channels
//! folded down to two, each channel onto the side it stands on (in the
//! WAVE and FLAC order: front left and right, centre, LFE, back, side),
//! the centre and the surrounds at −3 dB (the ITU-R BS.775 coefficients
//! for 5.1), the LFE left out, each side then scaled so it cannot go over
//! full scale.

/// A remix from `from` channels to stereo.
#[derive(Clone, Debug, PartialEq)]
pub struct Remix {
    /// The file's channels.
    pub from: u16,
    /// Each channel's share on the left and on the right.
    gains: Vec<(f64, f64)>,
}

/// −3 dB.
const HALF_POWER: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Where channel `k` of `n` stands, as its share of each side: front
/// left, front right, centre, LFE, back left, back right, side left, side
/// right (in WAVE's order for each count), else, past those, nowhere.
fn place(n: u16, k: u16) -> (f64, f64) {
    const L: (f64, f64) = (1.0, 0.0);
    const R: (f64, f64) = (0.0, 1.0);
    const C: (f64, f64) = (HALF_POWER, HALF_POWER);
    const SL: (f64, f64) = (HALF_POWER, 0.0);
    const SR: (f64, f64) = (0.0, HALF_POWER);
    const NONE: (f64, f64) = (0.0, 0.0);
    let order: &[(f64, f64)] = match n {
        // L R C
        3 => &[L, R, C],
        // FL FR BL BR (quad)
        4 => &[L, R, SL, SR],
        // FL FR C BL BR
        5 => &[L, R, C, SL, SR],
        // FL FR C LFE BL BR (5.1)
        6 => &[L, R, C, NONE, SL, SR],
        // FL FR C LFE BC SL SR (6.1): the back centre on both, at −6 dB.
        7 => &[L, R, C, NONE, (0.5, 0.5), SL, SR],
        // FL FR C LFE BL BR SL SR (7.1)
        _ => &[L, R, C, NONE, SL, SR, SL, SR],
    };
    order.get(usize::from(k)).copied().unwrap_or(NONE)
}

impl Remix {
    /// The remix of `from` channels to stereo (`None`: stereo already).
    pub fn to_stereo(from: u16) -> Option<Remix> {
        let gains = match from {
            2 => return None,
            0 | 1 => vec![(1.0, 1.0)],
            n => {
                let raw: Vec<(f64, f64)> = (0..n).map(|k| place(n, k)).collect();
                let left: f64 = raw.iter().map(|g| g.0).sum();
                let right: f64 = raw.iter().map(|g| g.1).sum();
                let most = left.max(right).max(1.0);
                raw.into_iter().map(|(l, r)| (l / most, r / most)).collect()
            }
        };
        Some(Remix { from, gains })
    }

    /// Whether every sample reaches a side as it is (mono doubled).
    pub fn exact(&self) -> bool {
        self.from <= 1
    }

    /// Appends `input` (whole frames of [`Remix::from`] channels, 24-bit
    /// integers) to `out` as stereo, each folded sample rounded.
    pub fn apply(&self, input: &[i32], out: &mut Vec<i32>) {
        let n = usize::from(self.from.max(1));
        for frame in input.chunks_exact(n) {
            if self.exact() {
                out.extend([frame[0], frame[0]]);
                continue;
            }
            let (l, r) = self.fold(frame.iter().map(|&s| f64::from(s)));
            out.extend([l.round() as i32, r.round() as i32]);
        }
    }

    /// Appends `input` (whole frames of [`Remix::from`] channels, in
    /// 24-bit units) to `out` as stereo.
    pub fn apply_wide(&self, input: &[f64], out: &mut Vec<f64>) {
        let n = usize::from(self.from.max(1));
        for frame in input.chunks_exact(n) {
            let (l, r) = self.fold(frame.iter().copied());
            out.extend([l, r]);
        }
    }

    fn fold(&self, frame: impl Iterator<Item = f64>) -> (f64, f64) {
        frame
            .zip(&self.gains)
            .fold((0.0, 0.0), |(l, r), (s, &(gl, gr))| {
                (l + s * gl, r + s * gr)
            })
    }

    /// A word for it: `mono → stereo`, `5.1 → stereo`.
    pub fn describe(&self) -> String {
        let from = match self.from {
            0 | 1 => "mono".to_owned(),
            6 => "5.1".to_owned(),
            7 => "6.1".to_owned(),
            8 => "7.1".to_owned(),
            n => format!("{n} ch"),
        };
        format!("{from} → stereo")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mono doubled exactly; 5.1 folded down with the ITU coefficients
    /// (the centre and the surrounds at −3 dB, the LFE left out), scaled so
    /// a full-scale frame on every channel stays within full scale.
    #[test]
    fn mono_doubles_and_5_1_folds_down() {
        assert_eq!(Remix::to_stereo(2), None);
        let mono = Remix::to_stereo(1).unwrap();
        assert!(mono.exact());
        let mut out = Vec::new();
        mono.apply(&[5, -7], &mut out);
        assert_eq!(out, [5, 5, -7, -7]);
        let surround = Remix::to_stereo(6).unwrap();
        assert!(!surround.exact());
        let scale = 1.0 + 2.0 * HALF_POWER;
        // Front left alone: on the left only; the LFE nowhere.
        let mut out = Vec::new();
        surround.apply_wide(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &mut out);
        assert!((out[0] - 1.0 / scale).abs() < 1e-12 && out[1] == 0.0);
        out.clear();
        surround.apply_wide(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &mut out);
        assert_eq!(out, [0.0, 0.0]);
        // The centre on both at −3 dB, the back right on the right.
        out.clear();
        surround.apply_wide(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &mut out);
        let c = HALF_POWER / scale;
        assert!((out[0] - c).abs() < 1e-12 && (out[1] - 2.0 * c).abs() < 1e-12);
        // Everything at full scale: within it.
        let full = 8_388_607;
        let mut out = Vec::new();
        surround.apply(&[full; 6], &mut out);
        assert!(out.iter().all(|&s| s <= full), "{out:?}");
        for n in 3..=8 {
            let r = Remix::to_stereo(n).unwrap();
            let mut out = Vec::new();
            r.apply(&vec![full; usize::from(n)], &mut out);
            assert!(out.iter().all(|&s| (0..=full).contains(&s)), "{n}: {out:?}");
            assert!(out[0] > full / 2, "{n}: {out:?}");
        }
    }
}
