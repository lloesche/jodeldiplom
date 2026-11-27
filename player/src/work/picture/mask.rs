//! Masks on a picture, as the taking apart of a story's scene works them:
//! a picture's size and its lengths at a 1024's ([`Size`]), a face's box
//! grown ([`face_rect`]), the median nearness in a box ([`median_in`]),
//! the parts of a mask and the part joined to a point ([`parts`],
//! [`joined`]), masks grown, shrunk and blurred.

use super::fill::{box_blur, dilate, Planes};

/// A box on a picture: left, top, right, bottom (the last two past it).
pub type Rect = (usize, usize, usize, usize);

/// A `w × h` picture's masks and their sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    /// Its width.
    pub w: usize,
    /// Its height.
    pub h: usize,
    /// A 1024 picture's pixels to this one's (lengths at 1024 are scaled).
    pub scale: f32,
}

impl Size {
    /// A `w × h` picture's.
    pub fn of(w: usize, h: usize) -> Size {
        Size {
            w,
            h,
            scale: w.max(h) as f32 / 1024.0,
        }
    }

    /// A length of `at_1024` pixels at a 1024 picture's size, at this one's
    /// (one at least).
    pub fn px(&self, at_1024: f32) -> usize {
        ((at_1024 * self.scale).round() as usize).max(1)
    }
}

/// A face's box (`[x, y, w, h]`, the picture's shares) on a picture of
/// `size`, made `wide` wider each side (a share of its width), `up` higher
/// and `down` lower (shares of its height).
pub fn face_rect([x, y, fw, fh]: [f32; 4], (wide, up, down): (f32, f32, f32), size: Size) -> Rect {
    let (w, h) = (size.w as f32, size.h as f32);
    let px = |v: f32, most: f32| v.clamp(0.0, most) as usize;
    (
        px((x - wide * fw) * w, w),
        px((y - up * fh) * h, h),
        px((x + fw + wide * fw) * w, w),
        px((y + fh + down * fh) * h, h),
    )
}

/// The median of the `w`-wide `near` in `rect`; none in an empty one.
pub fn median_in(near: &[f32], w: usize, (x0, y0, x1, y1): Rect) -> Option<f32> {
    let mut inside: Vec<f32> = (y0..y1)
        .flat_map(|y| near[y * w + x0..y * w + x1].iter().copied())
        .collect();
    inside.sort_by(f32::total_cmp);
    inside.get(inside.len() / 2).copied()
}

/// Sets `rect` of the `w`-wide `mask`.
pub fn fill_rect(mask: &mut [bool], w: usize, (x0, y0, x1, y1): Rect) {
    for y in y0..y1 {
        mask[y * w + x0..y * w + x1].fill(true);
    }
}

/// `mask` grown by `r` pixels each way.
pub fn grow(mask: &[bool], size: Size, r: usize) -> Vec<bool> {
    let as_f: Vec<f32> = mask.iter().map(|&m| f32::from(u8::from(m))).collect();
    dilate(&as_f, size.w, size.h, r)
        .into_iter()
        .map(|v| v > 0.5)
        .collect()
}

/// `mask` shrunk by `r` pixels each way (its outside grown).
pub fn shrink(mask: &[bool], size: Size, r: usize) -> Vec<bool> {
    let outside: Vec<bool> = mask.iter().map(|&m| !m).collect();
    grow(&outside, size, r).iter().map(|&o| !o).collect()
}

/// `mask` blurred over `r` pixels: 0 to 1.
pub fn blur(mask: &[bool], size: Size, r: usize) -> Vec<f32> {
    let p = Planes {
        width: size.w,
        height: size.h,
        channels: 1,
        data: mask.iter().map(|&m| f32::from(u8::from(m))).collect(),
    };
    box_blur(&p, r).data
}

/// The 4-connected parts of `on` (`w × h`): each pixel's part (0 off, 1
/// and up), and how many there are.
pub fn parts(on: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
    let mut labels = vec![0u32; on.len()];
    let mut count = 0u32;
    let mut stack = Vec::new();
    for start in 0..on.len() {
        if !on[start] || labels[start] != 0 {
            continue;
        }
        count += 1;
        labels[start] = count;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if on[j] && labels[j] == 0 {
                    labels[j] = count;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
    }
    (labels, count as usize)
}

/// The part of `on` (`w × h`) joined to pixel `seed` (none where `seed` is
/// off).
pub fn joined(on: &[bool], w: usize, h: usize, seed: usize) -> Option<Vec<bool>> {
    let (labels, _) = parts(on, w, h);
    let part = labels[seed];
    (part != 0).then(|| labels.iter().map(|&l| l == part).collect())
}
