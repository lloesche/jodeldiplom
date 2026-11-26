//! The window's old drawing surface without the window: the shared
//! ratatui widgets rasterised into an RGB pixmap ([`PixelBackend`]). No
//! screen draws into it any more; its cells' size is the lanes' letters'
//! ([`Canvas::metrics`]). Used by the window and, headless, by the tests.

use image::RgbImage;
use ratatui::layout::Rect;
use ratatui::Terminal;

use super::lanes::Metrics;
use super::pixels::{pack, unpack, PixelRect};
use super::raster::PixelBackend;
use crate::view::layout::Grid;
use crate::view::{self, palette, Scene};

/// The font size at a scale factor of 1, in pixels.
pub const BASE_FONT_PX: f64 = 15.0;

/// The font size (em) in pixels for a window scale factor (HiDPI).
pub fn font_px(scale: f64) -> f32 {
    (BASE_FONT_PX * scale).round().clamp(8.0, 96.0) as f32
}

/// The window's pixels.
pub struct Canvas {
    terminal: Terminal<PixelBackend>,
    font_px: f32,
    /// Whether the last [`Canvas::draw`] changed a pixel: a window that
    /// shows the pixmap already needs no new frame where it did not.
    pub changed: bool,
    /// The pixels changed since [`Canvas::take_changes`].
    changes: Changes,
}

/// The pixels of the pixmap that changed, for a window that keeps a copy
/// of it (on the GPU) and sends over only those.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Changes {
    /// All of them.
    pub all: bool,
    /// Otherwise these rects (they may overlap).
    pub rects: Vec<PixelRect>,
}

impl Changes {
    /// More rects than this become the one around them all.
    const MOST: usize = 48;

    /// Whether no pixel changed.
    pub fn is_empty(&self) -> bool {
        !self.all && self.rects.is_empty()
    }

    fn add(&mut self, rect: PixelRect) {
        if self.all || rect.2 == 0 || rect.3 == 0 {
            return;
        }
        self.rects.push(rect);
        if self.rects.len() > Self::MOST {
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
            for &(x, y, w, h) in &self.rects {
                (x0, y0) = (x0.min(x), y0.min(y));
                (x1, y1) = (x1.max(x + w), y1.max(y + h));
            }
            self.rects = vec![(x0, y0, x1 - x0, y1 - y0)];
        }
    }

    /// `other`'s pixels too.
    pub fn merge(&mut self, other: Changes) {
        if other.all {
            self.all = true;
            self.rects.clear();
        }
        for rect in other.rects {
            self.add(rect);
        }
    }
}

impl Canvas {
    /// A canvas of `cols` × `rows` cells at `font_px`.
    pub fn new(cols: u16, rows: u16, font_px: f32) -> Canvas {
        let mut backend = PixelBackend::new(cols, rows, super::fonts::main_font(), font_px);
        backend.default_bg = palette::BG;
        backend.default_fg = palette::TEXT;
        // The pixel backend cannot fail (its error type is `Infallible`).
        let Ok(terminal) = Terminal::new(backend);
        Canvas {
            terminal,
            font_px,
            changed: true,
            changes: Changes {
                all: true,
                rects: Vec::new(),
            },
        }
    }

    /// The cell size in pixels.
    pub fn cell(&self) -> (usize, usize) {
        self.terminal.backend().cell()
    }

    /// The grid of cells.
    pub fn grid(&self) -> Grid {
        let area = self.terminal.backend().buffer().area;
        Grid::new(area.width, area.height)
    }

    /// Changes the font size (a new scale factor).
    pub fn set_font_px(&mut self, px: f32) {
        if px != self.font_px {
            self.font_px = px;
            self.terminal.backend_mut().set_em(px);
            let Ok(()) = self.terminal.clear();
            self.changes.all = true;
        }
    }

    /// Fits the grid to a window of `width` × `height` pixels; returns the
    /// grid.
    pub fn fit(&mut self, width: u32, height: u32) -> Grid {
        let (cw, ch) = self.cell();
        let cols = (width as usize / cw).clamp(1, 1000) as u16;
        let rows = (height as usize / ch).clamp(1, 1000) as u16;
        let grid = Grid::new(cols, rows);
        if grid != self.grid() {
            self.terminal.backend_mut().resize(cols, rows);
            let Ok(()) = self.terminal.resize(grid.area());
            self.changes.all = true;
        }
        grid
    }

    /// The pixels changed since the last call (all of them at first).
    pub fn take_changes(&mut self) -> Changes {
        std::mem::take(&mut self.changes)
    }

    /// The cells the widgets redrew, as rects of pixels: per row of cells
    /// from the first to the last redrawn (and a cell beyond, for a glyph
    /// reaching into it).
    fn cells_changed(&self, frame: &mut Changes) {
        let backend = self.terminal.backend();
        let damage = &backend.damage;
        if damage.all {
            frame.all = true;
            return;
        }
        let Grid { cols, rows, .. } = self.grid();
        let mut spans: Vec<Option<(u16, u16)>> = vec![None; usize::from(rows)];
        for &(x, y) in &damage.cells {
            if let Some(span) = spans.get_mut(usize::from(y)) {
                *span = Some(span.map_or((x, x), |(a, b)| (a.min(x), b.max(x))));
            }
        }
        for (y, span) in spans.into_iter().enumerate() {
            if let Some((a, b)) = span {
                let width = (b + 2).min(cols) - a;
                frame.add(self.pixel_rect(Rect::new(a, y as u16, width, 1)));
            }
        }
    }

    /// Draws a frame of the widgets ([`view::draw`]): only the cells that
    /// changed are rasterised.
    pub fn draw(&mut self, scene: &Scene) {
        self.terminal.backend_mut().reset_damage();
        let Ok(_) = self.terminal.draw(|frame| view::draw(frame, scene));
        let mut frame = Changes::default();
        self.cells_changed(&mut frame);
        self.changed = !frame.is_empty();
        self.changes.merge(frame);
    }

    fn pixel_rect(&self, rect: Rect) -> PixelRect {
        let (cw, ch) = self.cell();
        (
            usize::from(rect.x) * cw,
            usize::from(rect.y) * ch,
            usize::from(rect.width) * cw,
            usize::from(rect.height) * ch,
        )
    }

    /// The sizes the lanes are painted at: the cells' and the font's
    /// ([`super::lanes`]: the words' letters take a cell's width each).
    pub fn metrics(&self) -> Metrics {
        Metrics {
            cell: self.cell(),
            font_px: self.font_px,
        }
    }

    /// The pixmap: `0x00RRGGBB` words, width, height.
    pub fn pixels(&self) -> (&[u32], usize, usize) {
        let b = self.terminal.backend();
        (&b.pixmap, b.width, b.height)
    }

    /// The pixmap into a window buffer of `width` × `height` pixels (the
    /// same `0x00RRGGBB` words, row by row), the background beyond it.
    pub fn write_xrgb(&self, out: &mut [u32], width: usize, height: usize) {
        let (pixels, pw, ph) = self.pixels();
        let bg = pack(palette::BG);
        let shared = pw.min(width);
        for y in 0..height.min(out.len() / width.max(1)) {
            let row = &mut out[y * width..(y + 1) * width];
            if y >= ph {
                row.fill(bg);
                continue;
            }
            row[..shared].copy_from_slice(&pixels[y * pw..y * pw + shared]);
            row[shared..].fill(bg);
        }
    }

    /// The pixmap as an image.
    pub fn image(&self) -> RgbImage {
        let (data, w, h) = self.pixels();
        let bytes: Vec<u8> = data.iter().flat_map(|&word| unpack(word)).collect();
        RgbImage::from_raw(w as u32, h as u32, bytes).expect("pixmap size")
    }

    /// The text of the cells (for tests).
    pub fn text(&self) -> String {
        view::buffer_text(self.terminal.backend().buffer())
    }
}
