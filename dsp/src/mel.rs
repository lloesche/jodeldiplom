//! The Slaney mel scale (Auditory Toolbox; librosa's default): linear to
//! 1 kHz, 15 mel there, logarithmic above, 27 mel to a factor of 6.4.

/// Mel a step of `ln 6.4 / 27` in `ln(hz)` makes above 1 kHz.
fn step() -> f64 {
    6.4f64.ln() / 27.0
}

/// Hertz to mel.
pub fn hz_to_mel(hz: f64) -> f64 {
    if hz >= 1000.0 {
        15.0 + (hz / 1000.0).ln() / step()
    } else {
        hz * 3.0 / 200.0
    }
}

/// Mel to hertz.
pub fn mel_to_hz(mel: f64) -> f64 {
    if mel >= 15.0 {
        1000.0 * (step() * (mel - 15.0)).exp()
    } else {
        mel * 200.0 / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_turns_at_1_khz_and_inverts() {
        assert_eq!(hz_to_mel(1000.0), 15.0);
        assert!((hz_to_mel(6400.0) - 42.0).abs() < 1e-12);
        for hz in [0.0, 200.0, 999.0, 1000.0, 4321.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9, "{hz}");
        }
    }
}
