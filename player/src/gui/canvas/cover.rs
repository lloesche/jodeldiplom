//! The cover painted into its cells, fitted and kept scaled for its rect,
//! and the caption over it: the cover darkened under the text, the text
//! drawn again on it.

use std::sync::Arc;

use image::RgbImage;
use ratatui::layout::Rect;

use super::Canvas;
use crate::art::{self, Art};
use crate::gui::pixels::pack;

/// A cover scaled for one rect: the art id, the rect's pixel size, the
/// scaled image and its offset in the rect.
pub(super) type FittedCover = (u64, u32, u32, Arc<(RgbImage, u32, u32)>);

impl Canvas {
    /// The cover fitted into `rect`, centred; with `below_line` from half
    /// a cell down (under the border line through the first row) and at
    /// the left.
    pub(super) fn paint_cover(&mut self, rect: Rect, art: &Art, below_line: bool) {
        let (x0, y0, w, h) = self.pixel_rect(rect);
        let inset = if below_line { self.cell().1 / 2 } else { 0 };
        let (y0, h) = (y0 + inset, h.saturating_sub(inset));
        if w == 0 || h == 0 {
            return;
        }
        let fitted = match &self.cover {
            Some((id, cw, chh, fitted)) if *id == art.id && *cw == w as u32 && *chh == h as u32 => {
                fitted.clone()
            }
            _ => {
                let fitted = Arc::new(art::fit(&art.image, w as u32, h as u32));
                self.cover = Some((art.id, w as u32, h as u32, fitted.clone()));
                fitted
            }
        };
        let (image, ox, oy) = &*fitted;
        let (ox, oy) = if below_line { (&0, &0) } else { (ox, oy) };
        let backend = self.terminal.backend_mut();
        let (stride, height) = (backend.width, backend.height);
        for (y, row) in image.rows().enumerate() {
            let py = y0 + *oy as usize + y;
            if py >= height {
                break;
            }
            for (x, pixel) in row.enumerate() {
                let px = x0 + *ox as usize + x;
                if px >= stride {
                    break;
                }
                backend.pixmap[py * stride + px] = pack(pixel.0);
            }
        }
    }

    /// Text over the cover: the cover darkened towards its bottom, from
    /// four rows above the text, then the text's glyphs drawn again over
    /// it (without their cells' background).
    pub(super) fn paint_caption(&mut self, cover: Rect, caption: Rect) {
        let rect = self.pixel_rect(cover);
        let (_, y0, w, h) = rect;
        let (_, cy, _, _) = self.pixel_rect(caption);
        let (_, ch) = self.cell();
        let backend = self.terminal.backend_mut();
        let mut target = backend.target();
        let mut px = target.sub(rect);
        let from = cy.saturating_sub(4 * ch).max(y0) - y0;
        let span = (h - from.min(h)).max(1) as f32;
        for y in from..h {
            let t = (y - from) as f32 / span;
            // Smooth from clear to 80% dark.
            let alpha = 0.8 * t * t * (3.0 - 2.0 * t);
            for x in 0..w {
                px.blend(x, y, [0, 0, 0], alpha);
            }
        }
        backend.overlay(caption);
    }
}
