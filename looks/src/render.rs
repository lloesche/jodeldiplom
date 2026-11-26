//! A look drawn on the processor: each pass's maths over a picture held
//! as encoded colours, read smoothly and held to its edge as the GPU's
//! sampler reads a texture. What the tests measure the guards on (the
//! light of whole songs' frames, the lyric band), small and fast; the
//! GPU draws what is seen.

use crate::math::{v2, v3, V2, V3};
use crate::pass::{Ctx, Pass, Src};

/// A picture: its size and its pixels (encoded colours, rows from the
/// top).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Its width.
    pub w: usize,
    /// Its height.
    pub h: usize,
    /// Its pixels.
    pub px: Vec<V3>,
}

impl Image {
    /// A picture of `w` by `h` from RGB bytes.
    pub fn from_rgb(w: usize, h: usize, bytes: &[u8]) -> Image {
        let px = bytes
            .chunks_exact(3)
            .map(|p| v3(f32::from(p[0]), f32::from(p[1]), f32::from(p[2])) / 255.0)
            .collect();
        Image { w, h, px }
    }

    /// One of `w` by `h`, all `c`.
    pub fn flat(w: usize, h: usize, c: V3) -> Image {
        Image {
            w,
            h,
            px: vec![c; w * h],
        }
    }

    /// It made `w` by `h`, read smoothly.
    pub fn resized(&self, w: usize, h: usize) -> Image {
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                px.push(self.at(v2((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32)));
            }
        }
        Image { w, h, px }
    }

    /// Its colour at `uv`, between its pixels, held to its edge.
    pub fn at(&self, uv: V2) -> V3 {
        let x = (uv.x * self.w as f32 - 0.5).clamp(0.0, (self.w - 1) as f32);
        let y = (uv.y * self.h as f32 - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |x: usize, y: usize| self.px[y * self.w + x];
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bottom = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// Its RGB bytes.
    pub fn to_rgb(&self) -> Vec<u8> {
        self.px
            .iter()
            .flat_map(|c| [c.x, c.y, c.z].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect()
    }
}

/// What a pass reads on the processor: the picture under it and the frame
/// before.
struct Reads<'a> {
    src: &'a Image,
    prev: &'a Image,
}

impl Src for Reads<'_> {
    fn src(&self, uv: V2) -> V3 {
        self.src.at(uv)
    }

    fn prev(&self, uv: V2) -> V3 {
        self.prev.at(uv)
    }
}

/// `passes` (each with its numbers, for a target the size of `src`) drawn
/// in turn over `src`, the frame before `prev` (none: the picture itself,
/// as after a seek).
pub fn draw(passes: &[(&'static Pass, Ctx)], src: &Image, prev: Option<&Image>) -> Image {
    let mut now = src.clone();
    let prev = prev.unwrap_or(src);
    for (pass, ctx) in passes {
        let reads = Reads { src: &now, prev };
        let mut px = Vec::with_capacity(now.w * now.h);
        for y in 0..now.h {
            for x in 0..now.w {
                let uv = v2(
                    (x as f32 + 0.5) / now.w as f32,
                    (y as f32 + 0.5) / now.h as f32,
                );
                let c = (pass.cpu)(ctx, &reads, uv);
                px.push(crate::wgsl::finish(ctx, &reads, c, uv));
            }
        }
        now = Image {
            w: now.w,
            h: now.h,
            px,
        };
    }
    now
}
