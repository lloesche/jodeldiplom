//! What the backdrop takes from a cover once, when it first shows: where
//! its subject is ([`Subject`]), the palette's lights ([`lights`]) and
//! whether the song's cover is the picture it was taken apart from
//! ([`is_of`]).

use image::RgbImage;

use crate::cover::{CoverLayers, Swatch};

/// Where a cover's subject is, in the cover's units (0 to 1 across and
/// down): where the cut is centred (its upper part, where a face is), the
/// light behind it and that light's reach.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    /// What the cut is centred on.
    pub focus: (f32, f32),
    /// The light behind it.
    pub light: (f32, f32),
    /// The light's reach, a share of the cover's height.
    pub reach: f32,
}

impl Subject {
    /// Where `layers`' subject is (the middle of the cover without one).
    pub fn of(layers: &CoverLayers) -> Subject {
        let none = Subject {
            focus: (0.5, 0.45),
            light: (0.5, 0.4),
            reach: 0.35,
        };
        let Some(s) = &layers.subject else {
            return none;
        };
        let (w, h) = s.dimensions();
        let step = (w.max(h) / 256).max(1);
        let (mut sum, mut sx, mut top, mut bottom) = (0.0f64, 0.0f64, h, 0);
        for y in (0..h).step_by(step as usize) {
            for x in (0..w).step_by(step as usize) {
                let a = f64::from(s.get_pixel(x, y).0[3]) / 255.0;
                sum += a;
                sx += a * f64::from(x);
                if a > 0.5 {
                    top = top.min(y);
                    bottom = bottom.max(y);
                }
            }
        }
        if sum <= 0.0 || bottom < top {
            return none;
        }
        let (fw, fh) = (w as f32, h as f32);
        let cx = (sx / sum) as f32 / fw;
        let (t, b) = (top as f32 / fh, bottom as f32 / fh);
        Subject {
            focus: (cx, t + 0.4 * (b - t)),
            light: (cx, t + 0.35 * (b - t)),
            reach: (0.35 * (b - t)).max(0.1),
        }
    }
}

/// Whether `art` is the picture `layers` were taken apart from (or the same
/// picture coded again, at another size): its shape theirs, and on a grid
/// over it, where nothing was cut out, its pixels the plate's within what
/// coding changes (the plate is the picture there but for where a hole
/// was filled: the middle difference is taken).
pub fn is_of(art: &RgbImage, layers: &CoverLayers) -> bool {
    const GRID: u32 = 16;
    let ((w, h), (aw, ah)) = (layers.size(), art.dimensions());
    let shape = |a: u32, b: u32| u64::from(a) * u64::from(b);
    if w == 0 || h == 0 || shape(aw, h).abs_diff(shape(ah, w)) * 100 > shape(ah, w) {
        return false;
    }
    let at = |size: u32, k: u32| ((2 * k + 1) * size / (2 * GRID)).min(size.saturating_sub(1));
    let mut diffs: Vec<u32> = Vec::new();
    for gy in 0..GRID {
        for gx in 0..GRID {
            let (x, y) = (at(w, gx), at(h, gy));
            if layers
                .subject
                .as_ref()
                .is_some_and(|s| s.get_pixel(x, y).0[3] > 0)
            {
                continue;
            }
            let (p, q) = (
                layers.plate.get_pixel(x, y).0,
                art.get_pixel(at(aw, gx), at(ah, gy)).0,
            );
            diffs.push((0..3).map(|c| u32::from(p[c].abs_diff(q[c]))).sum());
        }
    }
    if diffs.len() < 16 {
        return false;
    }
    diffs.sort_unstable();
    diffs[diffs.len() / 2] <= 3 * 12
}

/// A colour (sRGB bytes) as linear light, as bright as it can be: its
/// largest channel at 1 (a light's colour, not its strength).
fn light_of(rgb: [u8; 3]) -> [f32; 3] {
    let lin = rgb.map(|v| {
        let c = f32::from(v) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let most = lin[0].max(lin[1]).max(lin[2]).max(1e-3);
    lin.map(|c| c / most)
}

/// The palette's lights: its most vivid colour (the rays and the plate's
/// drift), the next most vivid (the sweep and the drift) and its largest
/// (the drift); a warm white for each it does not have.
pub fn lights(palette: &[Swatch]) -> [[f32; 3]; 3] {
    let vivid = |s: &Swatch| {
        let c = s.rgb.map(f32::from);
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        let sat = if max > 0.0 { (max - min) / max } else { 0.0 };
        sat * (max / 255.0).min(0.7) * (0.25 + s.share).sqrt()
    };
    let mut by: Vec<&Swatch> = palette.iter().collect();
    by.sort_by(|a, b| vivid(b).total_cmp(&vivid(a)));
    let warm = [1.0, 0.85, 0.65];
    let pick = |s: Option<&&Swatch>| s.map_or(warm, |s| light_of(s.rgb));
    [
        pick(by.first()),
        pick(by.get(1)),
        pick(palette.first().as_ref()),
    ]
}
