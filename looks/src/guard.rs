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
//! - **Washing out and blacking out:** over a whole frame at most
//!   [`CLIPPED`] of the pixels may be clipped (nearly white or nearly
//!   black) beyond what the picture itself has, and its mean light stays
//!   within [`MEAN`] (or within a tenth of the picture's own); a look whose frames
//!   over the song's picture would not is drawn less strongly, the same
//!   strength for the whole song, found once ([`strength`]).
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
/// The light (linear luminance, 0 to 1) the lyric band is held under:
/// white keeps 3.5 to 1 over it ([`ceiling_contrast`]).
pub const CEILING: f32 = 0.24;
/// How far the music may move a layer's mix.
pub const MIX_SWING: f32 = 1.0;
/// The light a layer's whole mix moves, at most (a pass's own light is
/// at most this).
pub const MIX_LIGHT: f32 = 0.6;

/// The most of a frame's pixels clipped (beyond the picture's own).
pub const CLIPPED: f32 = 0.15;
/// The frame's mean light (encoded luma) is kept between these.
pub const MEAN: (f32, f32) = (0.08, 0.75);
/// How light a pixel is before it counts as white, and dark as black.
const CLIP: (f32, f32) = (0.02, 0.95);
/// The strengths a look is tried at, strongest first.
const STRENGTHS: [f32; 7] = [1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.0];
/// The moments of a song a look's strength is found from.
const MOMENTS: usize = 32;
/// How far inside the limits the moments a strength is found from must
/// be (the frames between them are let come that much nearer).
const MARGIN: f32 = 0.03;

/// A frame's exposure: the share of its pixels clipped, its mean light
/// (encoded luma).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure {
    /// The share clipped.
    pub clipped: f32,
    /// The mean light.
    pub mean: f32,
}

/// The exposure of `pixels`.
pub fn exposure(pixels: &[V3]) -> Exposure {
    let n = pixels.len().max(1) as f32;
    let (mut clipped, mut sum) = (0usize, 0.0);
    for p in pixels {
        let l = crate::math::luma(*p);
        sum += l;
        clipped += usize::from(!(CLIP.0..=CLIP.1).contains(&l));
    }
    Exposure {
        clipped: clipped as f32 / n,
        mean: sum / n,
    }
}

impl Exposure {
    /// Whether a frame of this exposure is within the limits over a
    /// picture of exposure `picture`.
    pub fn within(&self, picture: &Exposure) -> bool {
        self.within_by(picture, 0.0)
    }

    /// As [`Exposure::within`], `margin` inside the limits.
    fn within_by(&self, picture: &Exposure, margin: f32) -> bool {
        self.clipped <= CLIPPED.max(picture.clipped + 0.05) - margin
            && self.mean >= MEAN.0.min(picture.mean - 0.1) + margin
            && self.mean <= MEAN.1.max(picture.mean + 0.1) - margin
    }
}

/// The strength `look` is drawn at over `picture` (encoded colours, small
/// is enough): the strongest of [`STRENGTHS`] at which its frames at
/// [`MOMENTS`] moments through the song (its `curves`; none: still music)
/// are all within the limits. The same look, song and picture, the same
/// strength.
pub fn strength(
    look: &crate::Checked,
    picture: &crate::render::Image,
    curves: Option<&crate::Curves>,
) -> f32 {
    let base = exposure(&picture.px);
    let length = curves.map_or(0.0, crate::Curves::length);
    STRENGTHS
        .into_iter()
        .find(|&s| {
            let mut tried = look.clone();
            tried.strength = s;
            (0..MOMENTS).all(|i| {
                let t = (i as f64 + 0.5) / MOMENTS as f64 * length;
                let signals = curves.map_or(crate::Signals::QUIET, |c| c.at(t));
                let moment = crate::Moment {
                    size: (picture.w as u32, picture.h as u32),
                    t,
                    seed: 1,
                    dt: 1.0 / 30.0,
                    forget: true,
                    band: None,
                    palette: crate::MEASURING_PALETTE,
                };
                let out =
                    crate::render::draw(&tried.frame(&moment, &signals, curves), picture, None);
                exposure(&out.px).within_by(&base, MARGIN)
            })
        })
        .unwrap_or(0.0)
}

/// What a look did over a whole song: the most flashes in any second, the
/// worst exposure (the most clipped, the darkest and the lightest mean),
/// the lyric band's least contrast, and whether any number was none.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverSong {
    /// The most flashes in a second.
    pub flashes: usize,
    /// The most clipped.
    pub clipped: f32,
    /// The darkest and the lightest mean.
    pub mean: (f32, f32),
    /// The band's least contrast.
    pub contrast: f32,
    /// Whether every number was one.
    pub finite: bool,
}

/// `look` drawn over `picture` through the whole song (its `curves`),
/// frame after frame at `fps`, the frame before fed back as in the window,
/// at the strength [`strength`] finds: what it did. The lyric band at
/// `band`.
pub fn over_song(
    look: &crate::Checked,
    picture: &crate::render::Image,
    curves: &crate::Curves,
    (fps, band): (f32, (f32, f32)),
) -> OverSong {
    let mut look = look.clone();
    look.strength = strength(&look, picture, Some(curves));
    let mut prev: Option<crate::render::Image> = None;
    let mut lights = Vec::new();
    let mut out = OverSong {
        flashes: 0,
        clipped: 0.0,
        mean: (1.0, 0.0),
        contrast: 21.0,
        finite: true,
    };
    let frames = (curves.length() * f64::from(fps)) as usize;
    for f in 0..frames {
        let t = f as f64 / f64::from(fps);
        let moment = crate::Moment {
            size: (picture.w as u32, picture.h as u32),
            t,
            seed: crate::seed(&look.id, "over a song"),
            dt: 1.0 / fps,
            forget: f == 0,
            band: Some(band),
            palette: crate::MEASURING_PALETTE,
        };
        let frame = crate::render::draw(
            &look.frame(&moment, &curves.at(t), Some(curves)),
            picture,
            prev.as_ref(),
        );
        out.finite &= frame
            .px
            .iter()
            .all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
        lights.push(light(&frame.px));
        let e = exposure(&frame.px);
        out.clipped = out.clipped.max(e.clipped);
        out.mean = (out.mean.0.min(e.mean), out.mean.1.max(e.mean));
        out.contrast = out.contrast.min(band_contrast(
            &frame.px,
            frame.w,
            (band.0 + 0.02, band.1 - 0.02),
        ));
        prev = Some(frame);
    }
    out.flashes = flashes(&lights, fps);
    out
}

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
    // The rows whose middles are in the band.
    let (a, b) = (
        ((band.0 * h as f32 - 0.5).ceil().max(0.0) as usize).min(h),
        ((band.1 * h as f32 - 0.5).floor() as usize + 1).min(h),
    );
    let b = b.max(a);
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

/// The contrast white keeps over the band held under [`CEILING`]
/// (WCAG's ratio).
pub fn ceiling_contrast() -> f32 {
    1.05 / (CEILING + 0.05)
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
