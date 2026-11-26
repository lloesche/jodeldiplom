//! The guards a look is held to, whatever it asks.
//!
//! - **The budget:** a stack costs at most [`BUDGET`] (a whole-screen pass
//!   reading the picture once costs 1) in at most [`MOST_LAYERS`] passes;
//!   over it, its dearest optional pass goes (the later of equals),
//!   until it is within ([`crate::Look::check`]).
//! - **Flashes:** what the music moves can change the screen's mean light
//!   only so far: each knob says how much light its whole range moves
//!   ([`crate::Knob::light`], held to the pass by a test), and the fast
//!   drivers (pulses, levels, phases: [`crate::drive::Speed::Fast`])
//!   together may move at most [`FAST_LIGHT`] of it, under the tenth of
//!   the screen's light a flash is counted from (WCAG 2.3.1), however
//!   often they beat; the slow ones (eased over half a second and more,
//!   which cannot flash three times a second) [`SLOW_LIGHT`]. A stack
//!   asking more has its drives scaled down to it ([`cap_light`]). On the
//!   frames themselves, [`flashes`] counts what WCAG counts, and the tests
//!   hold every preset to at most three in any second over whole songs.
//! - **The lyric band:** in the band the lyrics are set in, each pass
//!   takes out its share of itself ([`crate::Pass::relief`]: the busy
//!   ones half), and the light is held under [`CEILING`] (or what the
//!   picture had there, if brighter): white words keep at least 3 to 1
//!   against what is under them, before their dark outline
//!   ([`band_contrast`]).

use crate::math::V3;
use crate::CheckedLayer;

/// The most a stack may cost.
pub const BUDGET: f32 = 8.0;
/// The most passes in a stack.
pub const MOST_LAYERS: usize = 6;
/// The most light the fast drivers may move together.
pub const FAST_LIGHT: f32 = 0.08;
/// The most light the slow drivers may move together.
pub const SLOW_LIGHT: f32 = 0.35;
/// The light (encoded, 0 to 1) the lyric band is held under.
pub const CEILING: f32 = 0.55;
/// How far the music may move a layer's mix.
pub const MIX_SWING: f32 = 1.0;
/// The light a layer's whole mix moves, at most (a pass's own light is
/// at most this).
pub const MIX_LIGHT: f32 = 0.6;

/// The drives of `layers` scaled down, each speed's together, to the
/// light it may move.
pub fn cap_light(layers: &mut [CheckedLayer]) {
    use crate::drive::Speed;
    for (speed, most) in [(Speed::Fast, FAST_LIGHT), (Speed::Slow, SLOW_LIGHT)] {
        let moved: f32 = layers
            .iter()
            .flat_map(|l| &l.drives)
            .filter(|d| d.by.speed() == speed)
            .map(|d| d.light * d.amount.abs())
            .sum();
        if moved > most {
            let by = most / moved;
            for d in layers.iter_mut().flat_map(|l| l.drives.iter_mut()) {
                if d.by.speed() == speed {
                    d.amount *= by;
                }
            }
        }
    }
}

/// The light a frame gives off, 0 to 1: its pixels' mean relative
/// luminance (linear light from the encoded values).
pub fn light(pixels: &[V3]) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let lin = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    pixels
        .iter()
        .map(|p| 0.2126 * lin(p.x) + 0.7152 * lin(p.y) + 0.0722 * lin(p.z))
        .sum::<f32>()
        / pixels.len() as f32
}

/// The most flashes in any second of `lights` (a frame's light each,
/// `fps` frames a second), as WCAG counts them: a flash is a pair of
/// opposing changes of at least a tenth of the light, the darker side
/// under 0.8.
pub fn flashes(lights: &[f32], fps: f32) -> usize {
    // The turns: where the light stops rising or falling by a tenth.
    let mut turns: Vec<(usize, f32)> = Vec::new();
    let (mut lo, mut hi) = (0usize, 0usize);
    let mut rising: Option<bool> = None;
    for (i, &l) in lights.iter().enumerate() {
        if l < lights[lo] {
            lo = i;
        }
        if l > lights[hi] {
            hi = i;
        }
        match rising {
            None => {
                if l - lights[lo] >= 0.1 && lights[lo].min(l) < 0.8 {
                    turns.push((lo, lights[lo]));
                    rising = Some(true);
                    hi = i;
                } else if lights[hi] - l >= 0.1 && lights[hi].min(l) < 0.8 {
                    turns.push((hi, lights[hi]));
                    rising = Some(false);
                    lo = i;
                }
            }
            Some(true) => {
                if lights[hi] - l >= 0.1 && l.min(lights[hi]) < 0.8 {
                    turns.push((hi, lights[hi]));
                    rising = Some(false);
                    lo = i;
                }
            }
            Some(false) => {
                if l - lights[lo] >= 0.1 && l.min(lights[lo]) < 0.8 {
                    turns.push((lo, lights[lo]));
                    rising = Some(true);
                    hi = i;
                }
            }
        }
    }
    // Each two changes a flash; the most whose changes start within a
    // second of each other.
    let second = fps.max(1.0) as usize;
    let starts: Vec<usize> = turns.iter().map(|&(i, _)| i).collect();
    let mut most = 0;
    for (a, &from) in starts.iter().enumerate() {
        let within = starts[a..]
            .iter()
            .take_while(|&&i| i < from + second)
            .count();
        most = most.max(within / 2);
    }
    most
}

/// How readable white words are over the lyric band of `pixels` (`w`
/// wide, a frame's rows from the top): the contrast of white against the
/// band's brightest tenth (WCAG's ratio, 1 to 21), the band's top and
/// bottom shares of the height.
pub fn band_contrast(pixels: &[V3], w: usize, band: (f32, f32)) -> f32 {
    let h = pixels.len() / w.max(1);
    let (a, b) = (
        ((band.0 * h as f32) as usize).min(h),
        ((band.1 * h as f32).ceil() as usize).min(h),
    );
    let mut ls: Vec<f32> = pixels[a * w..b * w]
        .iter()
        .map(|&p| light(std::slice::from_ref(&p)))
        .collect();
    if ls.is_empty() {
        return 21.0;
    }
    ls.sort_by(f32::total_cmp);
    let bright = ls[(ls.len() - 1) * 9 / 10];
    1.05 / (bright + 0.05)
}

/// The encoded light under which white keeps 3 to 1: [`CEILING`]'s
/// reason, in linear light 0.3 (a little more than the 0.3 that 3 to 1
/// allows is lost to the band's soft edges).
pub fn ceiling_contrast() -> f32 {
    let c = crate::math::s3(CEILING);
    1.05 / (light(&[c]) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_steady_light_or_a_slow_one_does_not_flash() {
        assert_eq!(flashes(&[0.3; 120], 30.0), 0);
        let slow: Vec<f32> = (0..300)
            .map(|i| 0.3 + 0.3 * (i as f32 / 30.0 * std::f32::consts::TAU * 0.5).sin())
            .collect();
        assert!(flashes(&slow, 30.0) <= 1, "{}", flashes(&slow, 30.0));
    }

    #[test]
    fn a_strobe_flashes_as_often_as_it_strobes() {
        // Five times a second, from dark to light and back.
        let strobe: Vec<f32> = (0..120)
            .map(|i| if (i / 3) % 2 == 0 { 0.1 } else { 0.6 })
            .collect();
        let n = flashes(&strobe, 30.0);
        assert!((4..=6).contains(&n), "{n}");
        // Small: no flash at all.
        let small: Vec<f32> = strobe.iter().map(|l| 0.3 + 0.1 * (l - 0.35)).collect();
        assert_eq!(flashes(&small, 30.0), 0);
    }

    #[test]
    fn the_ceiling_keeps_white_words_readable() {
        assert!(ceiling_contrast() >= 3.0, "{}", ceiling_contrast());
    }
}
