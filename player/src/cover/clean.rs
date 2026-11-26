//! The subject's matte made fit to show, or found not to be: lettering
//! taken out of it, a face the matte could not cut grown from the depth,
//! and a cut that is not sure of itself refused (the cover then stays
//! whole, blurred as without a subject).
//!
//! The cut-out model often takes some of a cover's lettering with the
//! singer, so a title or a logo would stand sharp and pulse with them. The
//! text detector's own pixels ([`TextMap`]), grown a little and feathered,
//! are taken out of the matte; never a box, so the subject never loses a
//! straight-edged piece. Nothing is cut near a face the matte holds
//! ([`Face`]): its box made half as wide again, from above the hair down to
//! the shoulders. A part of the matte that is mostly lettering goes whole,
//! as do crumbs left beside the rest.
//!
//! Whether there is a subject at all is decided first. Without a face, by
//! the lettering's boxes (grown as DB's post-processing grows them): a cut
//! that would lose more than a tenth of the matte to them, or a matte much
//! of it covered by them (a logo), is not trusted. With a face, by what the
//! cut loses. What is left is a subject where it covers some of the cover
//! but not nearly all of it and is mostly sure of itself; a face lets it
//! cover more and be a little less sure.
//!
//! Where the matte is not sure of itself but a large, clear face is there
//! (a close-up the cut-out model breaks into patches), the subject is
//! grown from the face through the depth instead: the near region joined
//! to it, down to the cover's bottom.

use super::faces::Face;
use super::fill::{box_blur, dilate, Planes};
use super::text::TextMap;

/// The least score of a face that counts.
const FACE_SCORE: f32 = 0.75;
/// The least share of the cover a face that counts covers.
const FACE_AREA: f32 = 0.002;
/// How much of a face's box the matte must cover for the face to be in
/// the subject.
const FACE_HELD: f32 = 0.5;
/// The text map's threshold for a pixel of text, and the least mean of a
/// region that is lettering (DB's own).
const TEXT_PIXEL: f32 = 0.3;
const TEXT_REGION: f32 = 0.6;
/// How far a region's box grows: this times its area over its perimeter
/// (DB's unclip ratio).
const UNCLIP: f32 = 1.5;
/// A box the subject covers this much of is text on the subject.
const ON_SUBJECT: f32 = 0.85;
/// Without a face, the most of the subject such text may cover.
const PRINTED_MOST: f32 = 0.15;
/// A part of the matte with more of its area in lettering than this goes.
const PART_TEXT: f32 = 0.3;
/// Parts smaller than this share of the largest go.
const CRUMB: f32 = 0.02;
/// The matte's share that may be cut away (without a face, by the
/// lettering's boxes; with one, by its pixels).
const CUT_MOST: f32 = 0.1;
const FACE_CUT_MOST: f32 = 0.25;
/// The subject's share of the cover: at least, at most (without a face,
/// with one); and the share of the cover the matte may be unsure about
/// (between 0.15 and 0.85).
const SMALLEST: f32 = 0.01;
const LARGEST: f32 = 0.7;
const FACE_LARGEST: f32 = 0.92;
const UNSURE: f32 = 0.12;
const FACE_UNSURE: f32 = 0.16;
/// A face a subject is grown from: this sure and this large a share of
/// the cover, or (a face filling the cover) the second pair.
const SEED_FACE: (f32, f32) = (0.85, 0.03);
const SEED_HUGE: (f32, f32) = (0.6, 0.15);
/// How much less near than the face (its depth's nearness, 0 to 1) the
/// grown subject may be.
const SEED_DEPTH: f32 = 0.12;
/// The grown subject's share of the cover: at least, at most.
const SEED_SMALLEST: f32 = 0.05;

/// A box on the cover: left, top, right, bottom (the last two past it).
type Rect = (usize, usize, usize, usize);

/// A `w × h` picture's masks and their sizes.
#[derive(Clone, Copy)]
struct Size {
    w: usize,
    h: usize,
    /// A 1024 cover's pixels to this one's (lengths at 1024 are scaled).
    scale: f32,
}

impl Size {
    fn px(&self, at_1024: f32) -> usize {
        ((at_1024 * self.scale).round() as usize).max(1)
    }
}

/// `alpha` (a `w × h` matte, 0 to 1) cleaned of lettering, or a subject
/// grown from a face through `near` (the depth's nearness, `w × h`, 0 to
/// 1), or why there is no subject.
pub fn subject(
    alpha: &[f32],
    near: &[f32],
    w: usize,
    h: usize,
    text: &TextMap,
    faces: &[Face],
) -> Result<Vec<f32>, String> {
    let size = Size {
        w,
        h,
        scale: w.max(h) as f32 / 1024.0,
    };
    let (boxes, core) = lettering(text, w, h);
    let cut = cleaned(alpha, size, &boxes, &core, faces);
    match cut {
        Err(Refused::Lettering(why)) => Err(why),
        Ok(keep) => Ok(keep),
        Err(Refused::Unsure(why)) => grown(near, size, &core, faces).ok_or(why),
    }
}

/// Why the matte makes no subject: its lettering (final), or its being
/// unsure of itself (a face may still make one).
enum Refused {
    Lettering(String),
    Unsure(String),
}

/// The matte cleaned of lettering, or why not.
fn cleaned(
    alpha: &[f32],
    size: Size,
    boxes: &[Rect],
    core: &[bool],
    faces: &[Face],
) -> Result<Vec<f32>, Refused> {
    let (w, h) = (size.w, size.h);
    let n = w * h;
    let solid: Vec<bool> = alpha.iter().map(|&a| a > 0.5).collect();
    // Near the faces the matte holds, nothing is cut.
    let mut zone = vec![false; n];
    let mut held = false;
    for f in faces
        .iter()
        .filter(|f| f.score >= FACE_SCORE && f.w * f.h >= FACE_AREA)
    {
        let (x0, y0, x1, y1) = rect(f, 0.0, 0.0, 0.0, size);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let sum: f32 = (y0..y1)
            .map(|y| alpha[y * w + x0..y * w + x1].iter().sum::<f32>())
            .sum();
        if sum / ((x1 - x0) * (y1 - y0)) as f32 >= FACE_HELD {
            held = true;
            fill(&mut zone, w, rect(f, 0.25, 0.35, 1.0, size));
        }
    }
    let total: f32 = alpha.iter().sum();
    if !held {
        decide(alpha, &solid, size, boxes, core)?;
    }
    // The lettering's pixels, grown a little; and the parts of the matte
    // that are mostly lettering, whole.
    let mut cut: Vec<bool> = grow(core, size, size.px(4.0));
    let (labels, count) = parts(&solid, w, h);
    let mut part_size = vec![0usize; count + 1];
    let mut text_in = vec![0usize; count + 1];
    let mut safe = vec![false; count + 1];
    for i in 0..n {
        let l = labels[i] as usize;
        part_size[l] += 1;
        text_in[l] += usize::from(core[i]);
        safe[l] |= zone[i];
    }
    let drop: Vec<bool> = labels
        .iter()
        .map(|&l| {
            let l = l as usize;
            l > 0 && !safe[l] && text_in[l] as f32 > PART_TEXT * part_size[l] as f32
        })
        .collect();
    for (c, d) in cut.iter_mut().zip(grow(&drop, size, size.px(4.0))) {
        *c |= d;
    }
    cut.iter_mut().zip(&zone).for_each(|(c, &z)| *c &= !z);
    // Feathered out, never into the faces' zones.
    let soft = blur(&cut, size, size.px(3.0));
    let mut keep: Vec<f32> = (0..n)
        .map(|i| {
            if zone[i] {
                alpha[i]
            } else {
                alpha[i] * (1.0 - soft[i])
            }
        })
        .collect();
    no_crumbs(&mut keep, size, &zone);
    let lost = 1.0 - keep.iter().sum::<f32>() / total.max(1e-6);
    if held && lost > FACE_CUT_MOST {
        return Err(Refused::Lettering(format!(
            "{:.0} % of it cut away",
            lost * 100.0
        )));
    }
    clear(&keep, held).map_err(Refused::Unsure)?;
    Ok(keep)
}

/// `keep` without its crumbs: parts (where it is over a half) smaller than
/// [`CRUMB`] of the largest, but in `zone`.
fn no_crumbs(keep: &mut [f32], size: Size, zone: &[bool]) {
    let firm: Vec<bool> = keep.iter().map(|&a| a > 0.5).collect();
    let (labels, count) = parts(&firm, size.w, size.h);
    if count < 2 {
        return;
    }
    let mut part_size = vec![0usize; count + 1];
    labels.iter().for_each(|&l| part_size[l as usize] += 1);
    let largest = part_size[1..].iter().copied().max().unwrap_or(0);
    let crumbs: Vec<bool> = labels
        .iter()
        .map(|&l| l > 0 && (part_size[l as usize] as f32) < CRUMB * largest as f32)
        .collect();
    for ((k, c), z) in keep
        .iter_mut()
        .zip(grow(&crumbs, size, size.px(2.0)))
        .zip(zone)
    {
        if c && !z {
            *k = 0.0;
        }
    }
}

/// Without a face: whether the lettering's boxes leave the matte a
/// subject (they are only measured with, never cut).
fn decide(
    alpha: &[f32],
    solid: &[bool],
    size: Size,
    boxes: &[Rect],
    core: &[bool],
) -> Result<(), Refused> {
    let (w, h) = (size.w, size.h);
    let n = w * h;
    let mut off = vec![false; n];
    let mut on = vec![false; n];
    for &(x0, y0, x1, y1) in boxes {
        let covered = (y0..y1)
            .map(|y| solid[y * w + x0..y * w + x1].iter().filter(|&&s| s).count())
            .sum::<usize>();
        let into = if (covered as f32) < ON_SUBJECT * ((x1 - x0) * (y1 - y0)) as f32 {
            &mut off
        } else {
            &mut on
        };
        fill(into, w, (x0, y0, x1, y1));
    }
    let whole = solid.iter().filter(|&&s| s).count().max(1) as f32;
    let printed = (0..n).filter(|&i| on[i] && solid[i]).count() as f32;
    if printed > PRINTED_MOST * whole {
        return Err(Refused::Lettering("much of it lettering".into()));
    }
    // What the boxes off the subject, and the parts mostly lettering,
    // would take.
    let (labels, count) = parts(solid, w, h);
    let mut part_size = vec![0usize; count + 1];
    let mut text_in = vec![0usize; count + 1];
    for i in 0..n {
        let l = labels[i] as usize;
        part_size[l] += 1;
        text_in[l] += usize::from(core[i] && off[i]);
    }
    let drop: Vec<bool> = labels
        .iter()
        .map(|&l| {
            let l = l as usize;
            l > 0 && text_in[l] as f32 > PART_TEXT * part_size[l] as f32
        })
        .collect();
    let dropped = grow(&drop, size, size.px(4.0));
    let mut left: Vec<f32> = (0..n)
        .map(|i| if off[i] || dropped[i] { 0.0 } else { alpha[i] })
        .collect();
    no_crumbs(&mut left, size, &vec![false; n]);
    let total: f32 = alpha.iter().sum();
    let taken = total - left.iter().sum::<f32>();
    if taken > CUT_MOST * total.max(1e-6) {
        return Err(Refused::Lettering(format!(
            "{:.0} % of it lettering",
            taken / total.max(1e-6) * 100.0
        )));
    }
    Ok(())
}

/// A subject grown from the large, clear faces through the depth's
/// nearness `near`: the region joined to each face no less near than it
/// (by [`SEED_DEPTH`]), from a little above it down to the bottom, a face
/// wide each side; its holes filled, its spurs opened away, its edge
/// feathered, the lettering away from the faces taken out.
fn grown(near: &[f32], size: Size, core: &[bool], faces: &[Face]) -> Option<Vec<f32>> {
    let (w, h) = (size.w, size.h);
    let n = w * h;
    let strong: Vec<&Face> = faces
        .iter()
        .filter(|f| {
            let area = f.w * f.h;
            (f.score >= SEED_FACE.0 && area >= SEED_FACE.1)
                || (f.score >= SEED_HUGE.0 && area >= SEED_HUGE.1)
        })
        .collect();
    let mut region = vec![false; n];
    let mut zone = vec![false; n];
    for f in &strong {
        let (x0, y0, x1, y1) = rect(f, 0.0, 0.0, 0.0, size);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let (fw, fh) = (x1 - x0, y1 - y0);
        let mut inner: Vec<f32> = (y0 + fh / 5..y1 - fh / 5)
            .flat_map(|y| {
                near[y * w + x0 + fw / 5..y * w + x1 - fw / 5]
                    .iter()
                    .copied()
            })
            .collect();
        if inner.is_empty() {
            continue;
        }
        inner.sort_by(f32::total_cmp);
        let level = inner[inner.len() / 2] - SEED_DEPTH;
        let (zx0, zx1) = (x0.saturating_sub(fw), (x1 + fw).min(w));
        let zy0 = y0.saturating_sub(fh * 3 / 5);
        let candidate: Vec<bool> = (0..n)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                x >= zx0 && x < zx1 && y >= zy0 && near[i] >= level
            })
            .collect();
        let (labels, _) = parts(&candidate, w, h);
        let centre = labels[((y0 + y1) / 2) * w + (x0 + x1) / 2];
        if centre == 0 {
            continue;
        }
        labels
            .iter()
            .zip(region.iter_mut())
            .for_each(|(&l, r)| *r |= l == centre);
        fill(&mut zone, w, rect(f, 0.25, 0.35, 1.0, size));
    }
    if !region.iter().any(|&r| r) {
        return None;
    }
    let region = open(&filled(&region, w, h), size, 16.0);
    let text: Vec<bool> = grow(core, size, size.px(4.0))
        .iter()
        .zip(&zone)
        .map(|(&t, &z)| t && !z)
        .collect();
    let (body, away) = (
        blur(&region, size, size.px(5.0)),
        blur(&text, size, size.px(3.0)),
    );
    let alpha: Vec<f32> = body
        .iter()
        .zip(&away)
        .map(|(b, t)| (b * (1.0 - t)).clamp(0.0, 1.0))
        .collect();
    let cover = alpha.iter().sum::<f32>() / n as f32;
    (SEED_SMALLEST..=FACE_LARGEST)
        .contains(&cover)
        .then_some(alpha)
}

/// Whether `alpha` makes a clear subject: some of the cover but not nearly
/// all of it, and mostly sure of itself (more of it with a face).
fn clear(alpha: &[f32], face: bool) -> Result<(), String> {
    let n = alpha.len() as f32;
    let cover = alpha.iter().sum::<f32>() / n;
    let unsure = alpha.iter().filter(|&&a| a > 0.15 && a < 0.85).count() as f32 / n;
    let (largest, most_unsure) = if face {
        (FACE_LARGEST, FACE_UNSURE)
    } else {
        (LARGEST, UNSURE)
    };
    if cover < SMALLEST {
        return Err("too small".into());
    }
    if cover > largest {
        return Err("nearly all of the cover".into());
    }
    if unsure >= most_unsure {
        return Err("unsure of itself".into());
    }
    Ok(())
}

/// Face `f`'s box on the cover, made `wide` wider each side (a share of
/// its width), `up` higher and `down` lower (shares of its height).
fn rect(f: &Face, wide: f32, up: f32, down: f32, size: Size) -> Rect {
    let (w, h) = (size.w as f32, size.h as f32);
    let px = |v: f32, most: f32| v.clamp(0.0, most) as usize;
    (
        px((f.x - wide * f.w) * w, w),
        px((f.y - up * f.h) * h, h),
        px((f.x + f.w + wide * f.w) * w, w),
        px((f.y + f.h + down * f.h) * h, h),
    )
}

/// Sets `rect` of the `w`-wide `mask`.
fn fill(mask: &mut [bool], w: usize, (x0, y0, x1, y1): Rect) {
    for y in y0..y1 {
        mask[y * w + x0..y * w + x1].fill(true);
    }
}

/// `mask` grown by `r` pixels each way.
fn grow(mask: &[bool], size: Size, r: usize) -> Vec<bool> {
    let as_f: Vec<f32> = mask.iter().map(|&m| f32::from(u8::from(m))).collect();
    dilate(&as_f, size.w, size.h, r)
        .into_iter()
        .map(|v| v > 0.5)
        .collect()
}

/// `mask` blurred over `r` pixels: 0 to 1.
fn blur(mask: &[bool], size: Size, r: usize) -> Vec<f32> {
    let p = Planes {
        width: size.w,
        height: size.h,
        channels: 1,
        data: mask.iter().map(|&m| f32::from(u8::from(m))).collect(),
    };
    box_blur(&p, r).data
}

/// `mask` with its holes (what the outside cannot reach) filled.
fn filled(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let outside: Vec<bool> = mask.iter().map(|&m| !m).collect();
    let (labels, count) = parts(&outside, w, h);
    let mut border = vec![false; count + 1];
    for x in 0..w {
        border[labels[x] as usize] = true;
        border[labels[(h - 1) * w + x] as usize] = true;
    }
    for y in 0..h {
        border[labels[y * w] as usize] = true;
        border[labels[y * w + w - 1] as usize] = true;
    }
    labels
        .iter()
        .map(|&l| l == 0 || !border[l as usize])
        .collect()
}

/// `mask` opened by a disc of radius `r` (at a 1024 cover's size): what a
/// disc rolled inside it reaches. Worked at a quarter of the size.
fn open(mask: &[bool], size: Size, r: f32) -> Vec<bool> {
    let (w, h) = (size.w, size.h);
    let (sw, sh) = (w.div_ceil(4), h.div_ceil(4));
    let small: Vec<bool> = (0..sw * sh)
        .map(|i| mask[((i / sw) * 4).min(h - 1) * w + ((i % sw) * 4).min(w - 1)])
        .collect();
    let rr = (r * size.scale / 4.0).round().max(1.0) as isize;
    let disc: Vec<(isize, isize)> = (-rr..=rr)
        .flat_map(|dy| (-rr..=rr).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| dx * dx + dy * dy <= rr * rr)
        .collect();
    let pass = |m: &[bool], all: bool| -> Vec<bool> {
        (0..sw * sh)
            .map(|i| {
                let (x, y) = ((i % sw) as isize, (i / sw) as isize);
                let at = |&(dx, dy): &(isize, isize)| {
                    let (px, py) = (x + dx, y + dy);
                    px >= 0
                        && py >= 0
                        && px < sw as isize
                        && py < sh as isize
                        && m[py as usize * sw + px as usize]
                };
                if all {
                    disc.iter().all(at)
                } else {
                    disc.iter().any(at)
                }
            })
            .collect()
    };
    let opened = pass(&pass(&small, true), false);
    (0..w * h)
        .map(|i| opened[((i / w) / 4) * sw + (i % w) / 4])
        .collect()
}

/// The lettering on a `w × h` cover: each region's box (grown as DB grows
/// them), and whether each pixel is in a region.
fn lettering(text: &TextMap, w: usize, h: usize) -> (Vec<Rect>, Vec<bool>) {
    let side = text.side;
    let on: Vec<bool> = text.prob.iter().map(|&p| p > TEXT_PIXEL).collect();
    let (labels, count) = parts(&on, side, side);
    let mut sum = vec![0.0f32; count + 1];
    let mut size = vec![0usize; count + 1];
    let mut bounds = vec![(usize::MAX, usize::MAX, 0usize, 0usize); count + 1];
    for (i, &l) in labels.iter().enumerate() {
        let l = l as usize;
        if l == 0 {
            continue;
        }
        sum[l] += text.prob[i];
        size[l] += 1;
        let (x, y) = (i % side, i / side);
        let b = &mut bounds[l];
        *b = (b.0.min(x), b.1.min(y), b.2.max(x + 1), b.3.max(y + 1));
    }
    let lettering: Vec<bool> = (0..=count)
        .map(|l| l > 0 && sum[l] >= TEXT_REGION * size[l] as f32)
        .collect();
    let (sx, sy) = (w as f32 / side as f32, h as f32 / side as f32);
    let boxes = (1..=count)
        .filter(|&l| lettering[l])
        .filter_map(|l| {
            let (x0, y0, x1, y1) = bounds[l];
            let (bw, bh) = ((x1 - x0) as f32, (y1 - y0) as f32);
            if bw.min(bh) < 3.0 {
                return None;
            }
            let d = (UNCLIP * bw * bh / (2.0 * (bw + bh))).round();
            let px = |v: f32, s: f32, most: usize| ((v * s).max(0.0) as usize).min(most);
            Some((
                px(x0 as f32 - d, sx, w),
                px(y0 as f32 - d, sy, h),
                px(x1 as f32 + d, sx, w),
                px(y1 as f32 + d, sy, h),
            ))
        })
        .collect();
    let core = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let m = (y * side / h) * side + x * side / w;
            lettering[labels[m] as usize]
        })
        .collect();
    (boxes, core)
}

/// The 4-connected parts of `on` (`w × h`): each pixel's part (0 off, 1
/// and up), and how many there are.
fn parts(on: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 100;

    /// A disc of the matte in the middle of an `S × S` cover.
    fn disc() -> Vec<f32> {
        (0..S * S)
            .map(|i| {
                let (x, y) = ((i % S) as f32 - 50.0, (i / S) as f32 - 55.0);
                if (x * x + y * y).sqrt() < 25.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// A text map of side `S` with lettering (probability 0.9) in `rects`.
    fn text(rects: &[Rect]) -> TextMap {
        let mut prob = vec![0.0; S * S];
        for &(x0, y0, x1, y1) in rects {
            for y in y0..y1 {
                prob[y * S + x0..y * S + x1].fill(0.9);
            }
        }
        TextMap { side: S, prob }
    }

    /// A flat depth: nothing nearer than anything.
    fn flat() -> Vec<f32> {
        vec![0.5; S * S]
    }

    fn face(x: f32, y: f32, w: f32, h: f32, score: f32) -> Face {
        Face { x, y, w, h, score }
    }

    #[test]
    fn a_title_cut_with_the_subject_is_taken_out_of_it() {
        // The subject, and a title above it that the matte took too.
        let mut alpha = disc();
        for y in 12..16 {
            alpha[y * S + 30..y * S + 70].fill(1.0);
        }
        let kept =
            subject(&alpha, &flat(), S, S, &text(&[(31, 12, 69, 16)]), &[]).expect("a subject");
        assert!(kept[14 * S + 50] < 0.1 && kept[14 * S + 31] < 0.1);
        assert_eq!(kept[55 * S + 50], 1.0);
        // Without the text detector's word, the title stays.
        let kept = subject(&alpha, &flat(), S, S, &TextMap::none(), &[]).unwrap();
        assert_eq!(kept[14 * S + 50], 1.0);
    }

    #[test]
    fn a_cut_that_is_mostly_lettering_makes_no_subject() {
        // Only lettering: a wide band the text detector sees.
        let mut alpha = vec![0.0; S * S];
        for y in 40..60 {
            alpha[y * S + 10..y * S + 90].fill(1.0);
        }
        let words = text(&[(12, 42, 88, 58)]);
        assert!(subject(&alpha, &flat(), S, S, &words, &[]).is_err());
        // A subject that would lose more than a tenth of itself to
        // lettering, without a face: not trusted.
        let mut alpha = disc();
        for y in 10..30 {
            alpha[y * S + 10..y * S + 90].fill(1.0);
        }
        let words = text(&[(12, 12, 88, 28)]);
        assert!(subject(&alpha, &flat(), S, S, &words, &[]).is_err());
        // A logo: the matte a disc with its lettering across much of it.
        let words = text(&[(32, 45, 68, 52), (32, 56, 68, 63)]);
        assert!(subject(&disc(), &flat(), S, S, &words, &[]).is_err());
    }

    #[test]
    fn near_a_face_nothing_is_cut_and_no_straight_edge_is_bitten_off() {
        let alpha = disc();
        let f = face(0.4, 0.38, 0.2, 0.2, 0.9);
        // Lettering across the face's zone and beyond it.
        let words = text(&[(20, 45, 80, 50)]);
        let kept = subject(&alpha, &flat(), S, S, &words, &[f]).expect("a subject");
        // In the zone (the face half as wide again each side), kept.
        assert_eq!(kept[47 * S + 50], 1.0);
        assert_eq!(kept[47 * S + 36], 1.0);
        // Beyond it, the lettering's own pixels go; a pixel of the disc
        // away from the lettering stays.
        assert!(kept[47 * S + 29] < 0.5);
        assert_eq!(kept[79 * S + 50], 1.0);
    }

    #[test]
    fn a_face_filling_the_cover_is_a_subject_and_a_wall_without_one_is_not() {
        // The matte over 80 % of the cover.
        let alpha: Vec<f32> = (0..S * S)
            .map(|i| if i % S < 80 { 1.0 } else { 0.0 })
            .collect();
        assert!(subject(&alpha, &flat(), S, S, &TextMap::none(), &[]).is_err());
        let f = face(0.2, 0.2, 0.4, 0.5, 0.9);
        assert!(subject(&alpha, &flat(), S, S, &TextMap::none(), &[f]).is_ok());
        // A small face the matte does not hold does not count (a large one
        // would grow a subject of its own).
        let off = face(0.85, 0.2, 0.1, 0.2, 0.9);
        assert!(subject(&alpha, &flat(), S, S, &TextMap::none(), &[off]).is_err());
    }

    #[test]
    fn a_close_up_the_matte_breaks_up_is_grown_from_its_face_and_depth() {
        // A patchwork matte (half sure everywhere), and a head nearer than
        // the wall behind it: a disc down to the bottom.
        let alpha = vec![0.5; S * S];
        let near: Vec<f32> = (0..S * S)
            .map(|i| {
                let (x, y) = ((i % S) as f32 - 50.0, (i / S) as f32 - 45.0);
                if (x * x + y * y).sqrt() < 22.0 || ((x.abs() < 30.0) && y > 0.0) {
                    0.9
                } else {
                    0.2
                }
            })
            .collect();
        let f = face(0.38, 0.33, 0.24, 0.24, 0.9);
        let kept = subject(&alpha, &near, S, S, &TextMap::none(), &[f]).expect("grown");
        assert!(kept[45 * S + 50] > 0.9 && kept[90 * S + 50] > 0.9);
        assert!(kept[10 * S + 5] < 0.1 && kept[90 * S + 5] < 0.1);
        // Without a face sure and large enough, nothing.
        let small = face(0.45, 0.4, 0.1, 0.1, 0.9);
        assert!(subject(&alpha, &near, S, S, &TextMap::none(), &[small]).is_err());
    }

    #[test]
    fn crumbs_go_holes_fill_and_parts_join_by_sides() {
        let mut alpha = disc();
        alpha[5 * S + 5] = 1.0;
        let kept = subject(&alpha, &flat(), S, S, &TextMap::none(), &[]).unwrap();
        assert_eq!(kept[5 * S + 5], 0.0);
        let ring: Vec<bool> = (0..25)
            .map(|i| i != 12 && i % 5 > 0 && i % 5 < 4 && i / 5 > 0 && i / 5 < 4)
            .collect();
        assert!(filled(&ring, 5, 5)[12]);
        let (labels, count) = parts(&[true, false, true, true], 2, 2);
        assert_eq!((count, labels), (1, vec![1, 0, 1, 1]));
    }
}
