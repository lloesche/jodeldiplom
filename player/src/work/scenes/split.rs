//! A scene's picture taken apart for the backdrop ([`derive`]): its depth
//! (from the GPU's model, [`super::SceneNets`]), its heads held flat
//! ([`heads`], [`flatten`]: from a face and the depth round it, so a
//! profile's nose and the hair come with it and a head moves as one piece,
//! never bending), how it is split ([`layering`]: two or three layers of
//! depth where it splits cleanly, a warp where it does not) and each
//! layer's share of a pixel ([`alphas`], the nearest's edges the subject's
//! matte's where it agrees), the plate behind the layers filled, its
//! palette, with the picture tools of [`crate::work::picture`].

use std::time::Instant;

use image::{GrayAlphaImage, LumaA, RgbImage};
use jodel_gpu::vision::prep::Rgb;

use super::faces::{Face, NEAR_AS_FACE};
use super::{SceneNets, Took};
use crate::work::picture::mask::{
    blur, face_rect, fill_rect, grow, joined, median_in, shrink, Size,
};
use crate::work::picture::{box_blur, nearness, palette, plate, resize, NearMap, Planes, Swatch};

/// A head on a picture: its face's box and the nearness the whole head is
/// held at (the face's median), so it moves as one piece.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Head {
    /// The face's box: its left, top, width and height (shares of the
    /// picture).
    pub face: [f32; 4],
    /// The nearness it is held at, 0 to 1.
    pub near: f32,
}

/// Where a picture is cut into layers, by nearness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cuts {
    /// Once: two layers.
    Two(f32),
    /// Twice (the farther cut first): three layers.
    Three(f32, f32),
}

impl Cuts {
    /// How many layers it makes.
    pub fn layers(&self) -> u32 {
        match self {
            Cuts::Two(_) => 2,
            Cuts::Three(..) => 3,
        }
    }

    /// The cut in front of the nearest layer.
    pub fn nearest(&self) -> f32 {
        match *self {
            Cuts::Two(c) | Cuts::Three(_, c) => c,
        }
    }

    /// The cut in front of the middle layer, where there is one.
    pub fn middle(&self) -> Option<f32> {
        match *self {
            Cuts::Two(_) => None,
            Cuts::Three(c, _) => Some(c),
        }
    }
}

/// How a picture is moved.
#[derive(Clone, Debug, PartialEq)]
pub enum Split {
    /// Whole, by a gentle warp of its depth.
    Warp,
    /// In layers, in parallax.
    Layers {
        /// Where it is cut.
        cuts: Cuts,
        /// Each layer's share of a pixel, at half the picture's size: the
        /// middle layer's (luma; 0 with two layers) and the nearest's
        /// (alpha).
        alphas: GrayAlphaImage,
        /// The picture with every layer in front of the farthest taken out
        /// and the hole filled.
        plate: RgbImage,
    },
}

/// What is derived from a scene's picture for the backdrop.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    /// The picture's file's hash, what it was derived from.
    pub picture: String,
    /// Its nearness as the backdrop moves it (0 farthest, 1 nearest), at
    /// the depth model's size: smoothed (no step to tear), its heads held
    /// flat.
    pub near: NearMap,
    /// Its heads.
    pub heads: Vec<Head>,
    /// How it is moved.
    pub split: Split,
    /// Its main colours, the largest share first (the cover's way).
    pub palette: Vec<Swatch>,
}

/// The window round a face a head is looked for in: wider each side, and
/// higher and lower (shares of the face's width and height).
pub const HEAD_WINDOW: (f32, f32, f32) = (1.2, 1.2, 1.0);
/// The face's box grown to what is always of its head: wider each side,
/// the hair above, the neck below.
pub const HEAD_BOX: (f32, f32, f32) = (0.6, 0.6, 0.9);
/// How far a head's region is grown (pixels at a 1024 picture's size).
pub const HEAD_GROW: f32 = 6.0;
/// How softly a head's edge goes (pixels at a 1024 picture's size), or
/// this share of its face's width where that is more.
pub const HEAD_FEATHER: (f32, f32) = (4.6, 0.2);
/// How much the nearness is smoothed for the warp (pixels at a 1024
/// picture's size).
pub const SMOOTH: f32 = 4.0;

/// The heads of `faces` (boxes in the picture's shares) that count (as
/// the cover counts them: [`Face::counts`]) on a picture of nearness
/// `near`: each held at the median nearness inside its face (a face too
/// small for a pixel of the depth has none).
pub fn heads(near: &Planes, faces: &[Face]) -> Vec<Head> {
    let size = Size::of(near.width, near.height);
    faces
        .iter()
        .filter(|f| f.counts())
        .filter_map(|f| {
            let rect = face_rect(f.bounds(), (0.0, 0.0, 0.0), size);
            Some(Head {
                face: f.bounds(),
                near: median_in(&near.data, near.width, rect)?,
            })
        })
        .collect()
}

/// How much each pixel of a picture of nearness `near` is of `head` (0 to
/// 1): what is about as near as its face ([`NEAR_AS_FACE`]) and joined to
/// its middle, in the [`HEAD_WINDOW`] round the face (a profile's nose and
/// the hair stick out of a face's box), with the face's [`HEAD_BOX`];
/// grown by [`HEAD_GROW`], its edge feathered ([`HEAD_FEATHER`]).
pub fn head_mask(near: &Planes, head: &Head) -> Vec<f32> {
    let (w, h) = (near.width, near.height);
    let size = Size::of(w, h);
    let [fx, fy, fw, fh] = head.face;
    let (x0, y0, x1, y1) = face_rect(head.face, HEAD_WINDOW, size);
    let like: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (x0..x1).contains(&x)
                && (y0..y1).contains(&y)
                && (near.data[i] - head.near).abs() < NEAR_AS_FACE
        })
        .collect();
    let middle = (
        (((fx + fw / 2.0) * w as f32) as usize).min(w - 1),
        (((fy + fh / 2.0) * h as f32) as usize).min(h - 1),
    );
    let mut region =
        joined(&like, w, h, middle.1 * w + middle.0).unwrap_or_else(|| vec![false; w * h]);
    fill_rect(&mut region, w, face_rect(head.face, HEAD_BOX, size));
    let grown = grow(&region, size, size.px(HEAD_GROW));
    // Two box blurs of radius r: about a Gaussian of r / 1.22.
    let sigma = (HEAD_FEATHER.0 * size.scale).max(HEAD_FEATHER.1 * fw * w as f32);
    let r = ((1.22 * sigma).round() as usize).max(1);
    let once = Planes {
        width: w,
        height: h,
        channels: 1,
        data: blur(&grown, size, r),
    };
    box_blur(&once, r).data
}

/// `near` with each of `heads` held flat at its nearness ([`head_mask`]):
/// a whole head moves as one piece.
pub fn flatten(near: &Planes, heads: &[Head]) -> Planes {
    let mut out = near.clone();
    for head in heads {
        let m = head_mask(near, head);
        for (v, m) in out.data.iter_mut().zip(m) {
            *v = *v * (1.0 - m) + head.near * m;
        }
    }
    out
}

/// The least share of the variance a cut of the depth must explain (the
/// variance between the two sides, Otsu's), to be a clean one; evenly
/// spread nearness explains 0.75. With [`VALLEY`], fitted on the 32
/// FLUX.2 klein pictures of Fast Car, Thunderclouds and Family Portrait
/// (`--example story_scenes`): six of them split, each a subject well in
/// front; none of the floors running away and rooms close round.
pub const CLEAN: f32 = 0.82;
/// The most the depth's histogram round a clean cut may hold, a share of
/// the lesser of its peaks either side: a real valley between them.
pub const VALLEY: f32 = 0.3;
/// The least share of the picture a layer is.
pub const LEAST_LAYER: f32 = 0.05;
/// The least share of the picture the farthest of three layers is.
pub const LEAST_FARTHEST: f32 = 0.15;
/// The most share of the picture the nearest layer is.
pub const MOST_NEAR: f32 = 0.6;
/// How near a head's nearness may be to a cut: closer, the cut goes
/// behind it, by [`HEAD_CLEARANCE`], so the head is in the nearer layer.
pub const FEATHER: f32 = 0.03;
/// How far behind a head a cut moved behind it goes.
pub const HEAD_CLEARANCE: f32 = 0.035;
/// The least nearness a cut is at: one farther back leaves nothing behind
/// it to be a layer.
pub const LEAST_CUT: f32 = 0.03;
/// The least nearness between two cuts (closer, they are one).
pub const LEAST_GAP: f32 = 0.09;
/// The bins of the depth's histogram.
const BINS: usize = 128;
/// The fewest nearnesses a cut is looked for in.
const FEWEST: f64 = 16.0;

/// A cut of a set of nearnesses (Otsu's): where, the share of the
/// variance it explains, how deep the valley round it is ([`VALLEY`]), and
/// the shares of the whole picture below and above it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cut {
    at: f32,
    clean: f32,
    valley: f32,
    below: f32,
    above: f32,
}

impl Cut {
    /// Whether it splits cleanly ([`CLEAN`], [`VALLEY`]), a layer either
    /// side ([`LEAST_LAYER`]).
    fn splits(&self) -> bool {
        self.clean >= CLEAN
            && self.valley <= VALLEY
            && self.below >= LEAST_LAYER
            && self.above >= LEAST_LAYER
    }
}

/// Otsu's cut of the nearnesses in `values` (0 to 1) that are in `range`
/// (in the middle of the empty bins it could as well be in); none with too
/// few of them.
fn otsu(values: &[f32], range: (f32, f32)) -> Option<Cut> {
    let (lo, hi) = range;
    let span = (hi - lo).max(1e-6);
    let mut hist = [0.0f64; BINS];
    let mut n = 0.0f64;
    for &v in values.iter().filter(|&&v| v >= lo && v < hi) {
        let b = (((v - lo) / span) * BINS as f32) as usize;
        hist[b.min(BINS - 1)] += 1.0;
        n += 1.0;
    }
    if n < FEWEST {
        return None;
    }
    let centre = |b: usize| (b as f64 + 0.5) / BINS as f64;
    let mean: f64 = (0..BINS).map(|b| hist[b] * centre(b)).sum::<f64>() / n;
    let total: f64 = (0..BINS)
        .map(|b| hist[b] * (centre(b) - mean).powi(2))
        .sum::<f64>()
        / n;
    let (mut w0, mut s0) = (0.0f64, 0.0f64);
    let between: Vec<f64> = (0..BINS - 1)
        .map(|b| {
            w0 += hist[b];
            s0 += hist[b] * centre(b);
            let w1 = n - w0;
            if w0 < 1.0 || w1 < 1.0 {
                return 0.0;
            }
            let (m0, m1) = (s0 / w0, (mean * n - s0) / w1);
            w0 * w1 * (m0 - m1).powi(2) / (n * n)
        })
        .collect();
    let best = between.iter().copied().fold(0.0, f64::max);
    let tied = |b: &usize| between[*b] >= best * (1.0 - 1e-9);
    let first = (0..BINS - 1).find(tied).unwrap_or(0);
    let last = (first..BINS - 1)
        .take_while(|b| tied(b) || hist[*b] == 0.0)
        .filter(tied)
        .last()
        .unwrap_or(first);
    let at = (first + last) / 2;
    // The histogram smoothed over five bins: the valley at the cut against
    // the peaks either side.
    let smooth: Vec<f64> = (0..BINS)
        .map(|b| {
            let (a, z) = (b.saturating_sub(2), (b + 3).min(BINS));
            hist[a..z].iter().sum::<f64>() / (z - a) as f64
        })
        .collect();
    let peak = |r: std::ops::Range<usize>| smooth[r].iter().copied().fold(0.0, f64::max);
    let lesser = peak(0..at + 1).min(peak(at + 1..BINS)).max(1e-9);
    let below: f64 = hist[..=at].iter().sum();
    let whole = values.len() as f32;
    Some(Cut {
        at: lo + span * (at + 1) as f32 / BINS as f32,
        clean: (best / total.max(1e-12)) as f32,
        valley: ((smooth[at] + smooth[(at + 1).min(BINS - 1)]) / 2.0 / lesser) as f32,
        below: below as f32 / whole,
        above: (n - below) as f32 / whole,
    })
}

/// How a picture of nearness `near` (its heads held flat) is cut into
/// layers, or none (the warp): at Otsu's cut where it splits cleanly
/// ([`Cut::splits`]); in three where the side behind it (the farthest no
/// less than [`LEAST_FARTHEST`]) or the side in front (the nearest no more
/// than [`MOST_NEAR`]) splits cleanly again, the cleaner of the two; in two
/// where the nearest is no more than [`MOST_NEAR`]. Then no cut runs
/// through a head ([`clear_heads`]).
pub fn layering(near: &Planes, heads: &[Head]) -> Option<Cuts> {
    let values = &near.data;
    let first = otsu(values, (0.0, 1.0001)).filter(Cut::splits)?;
    let nearest_fits = |c: &Cut| c.above <= MOST_NEAR;
    let behind = otsu(values, (0.0, first.at))
        .filter(|c| c.splits() && c.below >= LEAST_FARTHEST && nearest_fits(&first))
        .map(|c| (c.clean, Cuts::Three(c.at, first.at)));
    let front = otsu(values, (first.at, 1.0001))
        .filter(|c| c.splits() && nearest_fits(c))
        .map(|c| (c.clean, Cuts::Three(first.at, c.at)));
    let three = [behind, front]
        .into_iter()
        .flatten()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, cuts)| cuts);
    let two = nearest_fits(&first).then_some(Cuts::Two(first.at));
    clear_heads(three.or(two)?, heads)
}

/// `cuts` with none through a head: each cut within [`FEATHER`] of a
/// head's nearness goes behind it by [`HEAD_CLEARANCE`], again until it is
/// clear of every head (so it only ever goes back, whatever the heads'
/// order). Two cuts brought within [`LEAST_GAP`] are one (the nearer);
/// a cut gone back past [`LEAST_CUT`] makes no layer.
fn clear_heads(cuts: Cuts, heads: &[Head]) -> Option<Cuts> {
    let clear = |mut cut: f32| {
        while let Some(h) = heads.iter().find(|h| (h.near - cut).abs() < FEATHER) {
            cut = h.near - HEAD_CLEARANCE;
        }
        cut
    };
    let made = |c: f32| c > LEAST_CUT;
    match cuts {
        Cuts::Two(c) => Some(clear(c)).filter(|&c| made(c)).map(Cuts::Two),
        Cuts::Three(a, b) => {
            let (b, a) = (clear(b), clear(a));
            let (a, b) = (a.min(b), a.max(b));
            if !made(b) {
                None
            } else if b - a < LEAST_GAP || !made(a) {
                Some(Cuts::Two(b))
            } else {
                Some(Cuts::Three(a, b))
            }
        }
    }
}

/// How far a layer's specks are opened away and its small holes closed,
/// and how softly its edge goes (pixels at a 1024 picture's size).
pub const OPEN: f32 = 4.0;
/// See [`OPEN`].
pub const CLOSE: f32 = 10.0;
/// See [`OPEN`].
pub const EDGE: f32 = 3.0;
/// How much the subject's matte and the nearest layer must agree (their
/// overlap over their union) for the nearest to take the matte's edges.
pub const MATTE_AGREES: f32 = 0.5;
/// How far past the depth's nearest layer the matte's may reach (pixels at
/// a 1024 picture's size).
pub const MATTE_REACH: f32 = 20.0;

/// Each layer's share of a pixel of a picture of nearness `near` (its
/// heads held flat) cut at `cuts`, at `size`: the middle layer's (luma; 0
/// with two layers) and the nearest's (alpha). Each is what is nearer than
/// its cut, its specks opened away ([`OPEN`]) and its small holes closed
/// ([`CLOSE`]); where the subject's `matte` (at any size) agrees with the
/// nearest ([`MATTE_AGREES`]), the nearest is the matte's (its edges
/// sharper than the depth's), within [`MATTE_REACH`] of the depth's; the
/// edges feathered ([`EDGE`]).
pub fn alphas(
    near: &Planes,
    cuts: &Cuts,
    (w, h): (usize, usize),
    matte: Option<&Planes>,
) -> GrayAlphaImage {
    let size = Size::of(w, h);
    let up = resize(near, w, h);
    let nearer = |cut: f32| -> Vec<bool> {
        let m: Vec<bool> = up.data.iter().map(|&d| d > cut).collect();
        let (open, close) = (size.px(OPEN), size.px(CLOSE));
        let opened = grow(&shrink(&m, size, open), size, open);
        shrink(&grow(&opened, size, close), size, close)
    };
    let depth = nearer(cuts.nearest());
    let mut nearest = depth.clone();
    if let Some(m) = matte {
        let m: Vec<bool> = resize(m, w, h).data.iter().map(|&a| a > 0.5).collect();
        let both = m.iter().zip(&depth).filter(|(a, b)| **a && **b).count();
        let either = m.iter().zip(&depth).filter(|(a, b)| **a || **b).count();
        if either > 0 && both as f32 >= MATTE_AGREES * either as f32 {
            let reach = grow(&depth, size, size.px(MATTE_REACH));
            nearest = m.iter().zip(&reach).map(|(&a, &r)| a && r).collect();
        }
    }
    let middle: Vec<bool> = cuts.middle().map_or_else(
        || vec![false; w * h],
        |c| {
            nearer(c)
                .iter()
                .zip(&nearest)
                .map(|(&a, &n)| a && !n)
                .collect()
        },
    );
    let edge = size.px(EDGE);
    let (middle, nearest) = (blur(&middle, size, edge), blur(&nearest, size, edge));
    GrayAlphaImage::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        LumaA([byte(middle[i]), byte(nearest[i])])
    })
}

/// A share as a byte.
pub(super) fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// What the layers in front of the farthest (their `alphas`) take of each
/// pixel of a picture of `size`: their shares added, enlarged to it.
fn taken(alphas: &GrayAlphaImage, (w, h): (u32, u32)) -> Vec<f32> {
    let any = Planes {
        width: alphas.width() as usize,
        height: alphas.height() as usize,
        channels: 1,
        data: (alphas.pixels())
            .map(|p| ((f32::from(p.0[0]) + f32::from(p.0[1])) / 255.0).min(1.0))
            .collect(),
    };
    resize(&any, w as usize, h as usize).data
}

/// The size of a picture of `size`'s layers and shape: half of it.
pub fn half((w, h): (u32, u32)) -> (usize, usize) {
    ((w / 2).max(1) as usize, (h / 2).max(1) as usize)
}

/// The parts of a scene's `picture` (its file's `hash`) derived by `nets`,
/// and how long each step took.
pub fn derive(
    picture: &RgbImage,
    hash: String,
    nets: &dyn SceneNets,
) -> Result<(Derived, Took), String> {
    let (w, h) = picture.dimensions();
    let rgb = Rgb::new(w as usize, h as usize, picture.as_raw().clone());
    let mut took = Took::default();
    let started = Instant::now();
    let near = nearness(&nets.depth(&rgb)?);
    took.depth = started.elapsed();
    let started = Instant::now();
    let heads = heads(&near, &nets.faces(picture));
    let flat = flatten(&near, &heads);
    let cuts = layering(&flat, &heads);
    let size = Size::of(near.width, near.height);
    let moved = flatten(&box_blur(&near, size.px(SMOOTH)), &heads);
    took.faces = started.elapsed();
    let split = match cuts {
        None => Split::Warp,
        Some(cuts) => {
            let started = Instant::now();
            let matte = Planes::from(nets.matte(&rgb)?);
            took.matte = started.elapsed();
            let started = Instant::now();
            let alphas = alphas(&flat, &cuts, half((w, h)), Some(&matte));
            let plate = plate(picture, &taken(&alphas, (w, h))).0;
            took.split = started.elapsed();
            Split::Layers {
                cuts,
                alphas,
                plate,
            }
        }
    };
    let derived = Derived {
        picture: hash,
        near: NearMap::from(&moved),
        heads,
        split,
        palette: palette(picture),
    };
    Ok((derived, took))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Planes {
        Planes {
            width: w,
            height: h,
            channels: 1,
            data: (0..w * h).map(|i| f(i % w, i / w)).collect(),
        }
    }

    fn face(x: f32, y: f32, w: f32, h: f32) -> Face {
        Face {
            x,
            y,
            w,
            h,
            score: 0.9,
        }
    }

    /// The middle layer's and the nearest's share at (`x`, `y`).
    fn at(a: &GrayAlphaImage, x: u32, y: u32) -> (u8, u8) {
        let p = a.get_pixel(x, y).0;
        (p[0], p[1])
    }

    /// A profile looking left: its face's box on the cheek, its nose
    /// sticking out of the box to the left, nearer than the face, its hair
    /// above and behind, a little farther; a wall far behind. Held flat,
    /// the nose, the face and the hair are all at the face's nearness, the
    /// wall as it was.
    #[test]
    fn a_whole_head_is_held_flat_its_nose_and_hair_with_it() {
        let (w, h) = (120, 80);
        let head = |x: usize, y: usize| {
            let (dx, dy) = (x as f32 - 60.0, y as f32 - 40.0);
            dx * dx / 400.0 + dy * dy / 300.0 < 1.0
        };
        let nose = |x: usize, y: usize| (36..42).contains(&x) && (38..44).contains(&y);
        let near = field(w, h, |x, y| {
            if nose(x, y) {
                0.86
            } else if head(x, y) {
                // The face nearer at the front, the hair farther at the
                // back: a slope across the head.
                0.82 - 0.004 * (x as f32 - 45.0).max(0.0)
            } else {
                0.1
            }
        });
        // The face's box: the front of the head, not the nose.
        let found = heads(&near, &[face(0.36, 0.38, 0.1, 0.16)]);
        assert_eq!(found.len(), 1);
        let held = found[0].near;
        assert!((held - 0.8).abs() < 0.03, "{held}");
        let flat = flatten(&near, &found);
        let at = |x: usize, y: usize| flat.data[y * w + x];
        for (x, y) in [(38, 40), (50, 40), (60, 28), (64, 40), (60, 50)] {
            assert!((at(x, y) - held).abs() < 0.01, "({x}, {y}): {}", at(x, y));
        }
        // Far from the head the wall is as it was.
        for (x, y) in [(2, 2), (117, 77), (115, 5)] {
            assert_eq!(at(x, y), 0.1, "({x}, {y})");
        }
        // Without a face, nothing is held.
        assert_eq!(flatten(&near, &[]), near);
    }

    /// The head's region is what is joined to the face at its depth: a
    /// second figure as near but apart from it is not held with it.
    #[test]
    fn a_head_takes_only_what_is_joined_to_it() {
        let (w, h) = (200, 100);
        let near = field(w, h, |x, y| {
            let disc = |cx: f32, cy: f32| {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                dx * dx + dy * dy < 225.0
            };
            if disc(50.0, 50.0) {
                0.7
            } else if disc(150.0, 50.0) {
                0.74
            } else {
                0.2
            }
        });
        let found = heads(&near, &[face(0.22, 0.42, 0.06, 0.16)]);
        let m = head_mask(&near, &found[0]);
        assert!(m[50 * w + 50] > 0.99 && m[50 * w + 44] > 0.99);
        assert_eq!(m[50 * w + 150], 0.0);
    }

    #[test]
    fn a_subject_well_in_front_splits_into_two_layers() {
        let (w, h) = (128, 72);
        let near = field(w, h, |x, y| {
            let (dx, dy) = (x as f32 - 64.0, y as f32 - 40.0);
            let noise = ((x * 7 + y * 13) % 10) as f32 * 0.004;
            if dx * dx / 600.0 + dy * dy / 500.0 < 1.0 {
                0.85 + noise
            } else {
                0.15 + noise + 0.1 * y as f32 / h as f32
            }
        });
        let Some(Cuts::Two(cut)) = layering(&near, &[]) else {
            panic!("two layers");
        };
        assert!(cut > 0.3 && cut < 0.8, "{cut}");
        let a = alphas(&near, &Cuts::Two(cut), (w, h), None);
        assert!(a.pixels().all(|p| p.0[0] == 0));
        assert!(at(&a, 64, 40).1 == 255 && at(&a, 5, 5).1 == 0);
    }

    #[test]
    fn three_plateaus_of_depth_are_three_layers() {
        let (w, h) = (120, 60);
        let near = field(w, h, |x, y| {
            let noise = ((x * 5 + y * 11) % 7) as f32 * 0.005;
            if (50..80).contains(&x) && y > 20 {
                0.9 + noise
            } else if y > 35 {
                0.5 + noise
            } else {
                0.05 + noise
            }
        });
        let Some(cuts @ Cuts::Three(far, near_cut)) = layering(&near, &[]) else {
            panic!("three layers");
        };
        assert!(far > 0.1 && far < 0.5 && near_cut > 0.55 && near_cut < 0.9);
        let a = alphas(&near, &cuts, (w, h), None);
        // The ground in the middle layer, the figure in the nearest, the
        // sky in neither.
        assert_eq!(at(&a, 10, 50), (255, 0));
        assert_eq!(at(&a, 65, 45), (0, 255));
        assert_eq!(at(&a, 10, 5), (0, 0));
    }

    #[test]
    fn depth_that_does_not_split_cleanly_is_warped() {
        // A floor running away: evenly spread.
        let ramp = field(100, 60, |_, y| y as f32 / 59.0);
        assert_eq!(layering(&ramp, &[]), None);
        // Flat.
        let flat = field(100, 60, |_, _| 0.4);
        assert_eq!(layering(&flat, &[]), None);
        // A speck in front: too small to be a layer.
        let speck = field(100, 60, |x, y| if x < 5 && y < 5 { 0.9 } else { 0.2 });
        assert_eq!(layering(&speck, &[]), None);
    }

    /// A head held at a nearness right on the cut would be half in each
    /// layer: the cut goes behind it, and behind a second head it then
    /// meets too, whatever their order.
    #[test]
    fn no_cut_runs_through_a_head() {
        let (w, h) = (128, 72);
        let near = field(w, h, |x, _| if x > 80 { 0.8 } else { 0.2 });
        let Some(Cuts::Two(cut)) = layering(&near, &[]) else {
            panic!("two layers");
        };
        let head = |at: f32| Head {
            face: [0.3, 0.3, 0.1, 0.1],
            near: at,
        };
        let (first, second) = (head(cut), head(cut - HEAD_CLEARANCE));
        for heads in [[first, second], [second, first]] {
            let Some(Cuts::Two(moved)) = layering(&near, &heads) else {
                panic!("two layers");
            };
            assert!(moved + FEATHER <= second.near, "{moved} {heads:?}");
        }
        // Three cuts brought together are one; a cut gone to the back is
        // none.
        let Some(Cuts::Two(one)) = clear_heads(Cuts::Three(0.4, 0.45), &[head(0.45)]) else {
            panic!("one cut");
        };
        assert!((one - (0.45 - HEAD_CLEARANCE)).abs() < 1e-6, "{one}");
        assert_eq!(clear_heads(Cuts::Two(0.05), &[head(0.06)]), None);
        // A face too weak or too small to count holds no head.
        let weak = Face {
            score: 0.6,
            ..face(0.3, 0.3, 0.2, 0.2)
        };
        let tiny = face(0.3, 0.3, 0.01, 0.01);
        assert!(heads(&near, &[weak, tiny]).is_empty());
    }

    /// Where the subject's matte agrees with the nearest layer, the nearest
    /// is the matte's (sharper), only where the depth reaches.
    #[test]
    fn the_nearest_layer_takes_the_subject_s_matte_where_it_agrees() {
        let (w, h) = (100, 100);
        let inside = |x: usize, y: usize, r: f32| {
            let (dx, dy) = (x as f32 - 50.0, y as f32 - 50.0);
            dx * dx + dy * dy < r * r
        };
        let near = field(
            25,
            25,
            |x, y| if inside(x * 4, y * 4, 24.0) { 0.9 } else { 0.1 },
        );
        // The matte a little smaller than the depth's blob, and a stray
        // blob of it far off, where the depth is far.
        let matte = field(100, 100, |x, y| {
            if inside(x, y, 20.0) || (x < 8 && y < 8) {
                1.0
            } else {
                0.0
            }
        });
        let cuts = Cuts::Two(0.5);
        let plain = alphas(&near, &cuts, (w, h), None);
        let cut = alphas(&near, &cuts, (w, h), Some(&matte));
        assert_eq!(at(&cut, 50, 50).1, 255);
        // Between the matte's edge and the depth's: the depth's alone.
        assert!(at(&plain, 72, 50).1 > 128 && at(&cut, 72, 50).1 < 25);
        assert!(at(&cut, 3, 3).1 < 3);
    }
}
