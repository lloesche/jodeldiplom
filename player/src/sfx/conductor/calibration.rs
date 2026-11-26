//! The game's sounds calibrated to the song's loudness: a loud song
//! takes them a little higher, a quiet one a little lower, so they sit
//! the same over each.
//!
//! The song's loudness is what the karaoke plays of it, integrated over
//! the whole song (ITU-R BS.1770, LUFS: [`crate::work::loudness`]). The bus's
//! level is [`super::LEVEL`] at [`REFERENCE`], and moves by [`SLOPE`] dB
//! for each LU the song is louder or quieter, at most [`MOST`] dB either
//! way; a song not measured yet (or silent) is taken as at the reference.
//! The Effects strip's fader is the user's own on top: at 0 dB, as
//! calibrated.
//!
//! Fitted by ear: six songs set to sound right over them on the Effects
//! strip (at the old bus level, 8 dB), from -22.2 to -9.4 LUFS as played
//! (their instrumentals), the strip from -7 to -3 dB. [`REFERENCE`] is
//! their mean; the level there (their settings' mean, -4.67 dB) is folded
//! into [`super::LEVEL`]. [`SLOPE`] is their least-squares slope (0.05 its
//! standard error, R² 0.87; against the full mixes' loudness 0.28, R²
//! 0.84): the sounds follow a little over a quarter of the song's change.
//! One line fits them all, the quietest (a dynamic master 8 LU under the
//! reference) within 0.1 dB of its setting; the one song off it, by 0.9
//! dB, is a loud one set higher.

/// The loudness the bus's level is set for, LUFS as played.
pub const REFERENCE: f64 = -13.81;

/// How far the bus's level follows the song's loudness, dB a LU.
pub const SLOPE: f64 = 0.27;

/// The most the level moves either way, dB: a song 15 LU from the
/// reference (-29 or 1 LUFS, a quiet classical track or one past any a
/// master gets to) is as far as it goes.
pub const MOST: f64 = 4.0;

/// The bus's level for a song of loudness `lufs` (none: not measured, or
/// silent), dB from [`super::LEVEL`].
pub fn trim(lufs: Option<f64>) -> f32 {
    let Some(l) = lufs.filter(|l| l.is_finite()) else {
        return 0.0;
    };
    (SLOPE * (l - REFERENCE)).clamp(-MOST, MOST) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_level_follows_the_loudness_by_the_slope_and_no_further_than_most() {
        assert_eq!(trim(Some(REFERENCE)), 0.0);
        assert_eq!(trim(None), 0.0);
        assert_eq!(trim(Some(f64::NEG_INFINITY)), 0.0);
        assert!((trim(Some(REFERENCE + 4.0)) - 1.08).abs() < 1e-5);
        assert!((trim(Some(REFERENCE - 2.0)) + 0.54).abs() < 1e-5);
        assert_eq!(trim(Some(REFERENCE + 40.0)), MOST as f32);
        assert_eq!(trim(Some(REFERENCE - 40.0)), -MOST as f32);
    }

    /// The six songs set by ear: their loudness as played and the Effects
    /// strip's setting (at the old bus level, 8 dB): a louder song is
    /// never put under a quieter one, and each lands within 1.0 dB of its
    /// setting (all but one within 0.6).
    #[test]
    fn the_songs_set_by_ear_land_near_their_settings_in_order() {
        let set = [
            (-11.06, -3.0),
            (-14.96, -5.0),
            (-14.95, -5.0),
            (-9.35, -4.0),
            (-10.39, -4.0),
            (-22.16, -7.0),
        ];
        let old = super::super::LEVEL - 8.0;
        let mut off = Vec::new();
        for (lufs, strip) in set {
            let at = f64::from(old + trim(Some(lufs)));
            off.push((at - strip).abs());
            assert!(
                (at - strip).abs() <= 1.0,
                "{lufs} LUFS: {at:.2} dB, set {strip}"
            );
        }
        assert_eq!(off.iter().filter(|&&d| d > 0.6).count(), 1, "{off:?}");
        let mut by_loudness = set.map(|(lufs, _)| lufs);
        by_loudness.sort_by(f64::total_cmp);
        let levels = by_loudness.map(|l| trim(Some(l)));
        assert!(levels.windows(2).all(|w| w[0] <= w[1]), "{levels:?}");
        assert!(levels[0] < 0.0 && levels[5] > 0.0);
    }
}
