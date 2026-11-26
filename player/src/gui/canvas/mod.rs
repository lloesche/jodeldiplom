//! The window's drawing surface without the window: the shared ratatui
//! widgets rasterised into an RGB pixmap ([`PixelBackend`]), and the
//! pictures painted into it at their cell rects ([`pictures`]): Karaoke
//! mode's picture ([`karaoke`]). The covers and the lanes (the Transition
//! tab's waveforms and bar numbers, the Karaoke tab's voice lane) are the
//! window's own widgets over the cells left for them ([`Canvas::covers`],
//! [`Canvas::take_lanes`]). Used by the window and, headless, by the tests
//! and the screenshots.

mod karaoke;
mod pictures;

use image::RgbImage;
use ratatui::layout::Rect;
use ratatui::Terminal;

use super::big_text::Faces;
use super::lanes::{LaneSpot, LaneSpots, Metrics, Part, Rows};
use super::pixels::{pack, unpack, PixelRect};
use super::raster::PixelBackend;
use super::widgets::cover::{Align, Spot};
use crate::art::Art;
use crate::view::karaoke::KaraokePicture;
use crate::view::layout::Grid;
use crate::view::{self, palette, Scene};
use pictures::{Paint, PixelPictures};

/// The font size at a scale factor of 1, in pixels.
pub const BASE_FONT_PX: f64 = 15.0;

/// The font size (em) in pixels for a window scale factor (HiDPI).
pub fn font_px(scale: f64) -> f32 {
    (BASE_FONT_PX * scale).round().clamp(8.0, 96.0) as f32
}

/// How long the last [`Canvas::draw`] took: the widgets with their cells
/// into pixels, then the pictures painted over them.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawTimes {
    /// The widgets drawn and the changed cells rasterised.
    pub widgets: std::time::Duration,
    /// The pictures painted, the covers and the lanes placed.
    pub pictures: std::time::Duration,
}

/// The window's pixels.
pub struct Canvas {
    terminal: Terminal<PixelBackend>,
    font_px: f32,
    painted: Vec<Rect>,
    /// The covers as last drawn, for the window to draw over the cells
    /// left for them.
    covers: Vec<Spot>,
    /// The lanes as last drawn, for the window to draw over the cells
    /// left for them.
    lanes: Vec<LaneSpot>,
    /// Karaoke mode's backdrop, the text drawn large (by face), the picture
    /// now and its cells.
    backdrop: Option<super::karaoke::Backdrop>,
    faces: Faces,
    karaoke: Option<(Rect, KaraokePicture)>,
    /// What the picture in the pixmap drew over the backdrop.
    karaoke_layer: super::karaoke::Layer,
    /// Karaoke mode's stage, drawn by the GPU over the pixmap
    /// ([`Canvas::stage_frame`]: START's pill, the result, the song's
    /// lane), and the picture it draws from (all its lines) with where,
    /// while the mode shows on the GPU.
    stage: super::stage::Stage,
    staged: Option<(KaraokePicture, PixelRect)>,
    /// How long the last frame took.
    pub last_draw: DrawTimes,
    /// Whether the last [`Canvas::draw`] changed a pixel: a window that
    /// shows the pixmap already needs no new frame where it did not.
    pub changed: bool,
    /// The pixels changed since [`Canvas::take_changes`].
    changes: Changes,
    /// The window shows the pixmap on the GPU ([`Canvas::set_gpu`]):
    /// Karaoke mode's stage, its bubbles' liquid and its ground are its,
    /// rather than in the pixmap.
    gpu: bool,
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
            painted: Vec::new(),
            covers: Vec::new(),
            lanes: Vec::new(),
            backdrop: None,
            faces: Faces::default(),
            karaoke: None,
            karaoke_layer: super::karaoke::Layer::default(),
            stage: super::stage::Stage::default(),
            staged: None,
            gpu: false,
            last_draw: DrawTimes::default(),
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

    /// Whether the window shows the pixmap on the GPU: Karaoke mode's
    /// stage and ground are drawn there; otherwise all of it is painted
    /// into the pixmap each frame.
    pub fn set_gpu(&mut self, on: bool) {
        if on != self.gpu {
            self.gpu = on;
            self.changes.all = true;
        }
    }

    /// Whether the pixmap's pictures move by themselves, so frames must
    /// keep coming while nothing else moves: Karaoke mode's score rolling
    /// up.
    pub fn moving(&self) -> bool {
        self.karaoke_moving()
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

    /// Draws a frame: the widgets, then the pictures, and where the covers
    /// and the lanes go ([`Canvas::covers`], [`Canvas::take_lanes`]). The
    /// text over a cover is made again only when it moved or a cell of it
    /// was redrawn.
    pub fn draw(&mut self, scene: &Scene, art: Option<&Art>) {
        let started = std::time::Instant::now();
        self.terminal.backend_mut().reset_damage();
        self.staged = None;
        let mut pictures = PixelPictures {
            has_art: art.is_some(),
            paints: Vec::new(),
        };
        let Ok(_) = self
            .terminal
            .draw(|frame| view::draw(frame, scene, &mut pictures));
        // Only the cells that changed are redrawn; where a picture was and
        // is no more, the cells did not change, so repaint them all.
        let rects: Vec<Rect> = pictures.paints.iter().map(Paint::rect).collect();
        if rects != self.painted {
            self.terminal.backend_mut().redraw();
            self.painted = rects;
        }
        let widgets = started.elapsed();
        let mut frame = Changes::default();
        self.cells_changed(&mut frame);
        self.covers.clear();
        self.lanes.clear();
        let karaoke = self.karaoke.take();
        for paint in &pictures.paints {
            match paint {
                Paint::Cover(rect, below_line) => {
                    self.covers.push(self.cover_spot(*rect, *below_line))
                }
                Paint::Wave(plot) => {
                    let rect = self.pixel_rect(plot.area);
                    let ch = self.cell().1;
                    let above = usize::from(plot.rows.0) * ch;
                    let rows = Rows {
                        above,
                        full: (usize::from(plot.rows.1) * ch).max(rect.3 + above),
                    };
                    let part = Part::Wave(plot.clone(), rows);
                    self.lanes.push(LaneSpot { rect, part });
                }
                Paint::Ruler(plot) => {
                    let rect = self.pixel_rect(plot.area);
                    let part = Part::Ruler(plot.clone());
                    self.lanes.push(LaneSpot { rect, part });
                }
                Paint::Karaoke(area, picture) => {
                    let wanted = super::stage::wanted(scene.state);
                    let last = karaoke.as_ref();
                    let now = (*area, &**picture);
                    for rect in self.karaoke_picture(now, last, wanted, art, started) {
                        frame.add(rect);
                    }
                }
            }
        }
        self.changed = !frame.is_empty();
        self.changes.merge(frame);
        self.last_draw = DrawTimes {
            widgets,
            pictures: started.elapsed() - widgets,
        };
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

    /// Where the covers go as last drawn, in the pixmap's pixels: the
    /// window draws them over the cells left blank for them.
    pub fn covers(&self) -> &[Spot] {
        &self.covers
    }

    /// The cover in `rect`: with `below_line` from half a cell down
    /// (under the border line through its first row) and at the left,
    /// otherwise in the middle.
    fn cover_spot(&self, rect: Rect, below_line: bool) -> Spot {
        let (x, y, w, h) = self.pixel_rect(rect);
        let inset = if below_line { self.cell().1 / 2 } else { 0 };
        Spot {
            place: (x, y + inset, w, h.saturating_sub(inset)),
            align: if below_line {
                Align::TopLeft
            } else {
                Align::Center
            },
        }
    }

    /// The lanes as last drawn, in the pixmap's pixels, for the window
    /// to draw over the cells left blank for them; taken (each frame
    /// draws them again).
    pub fn take_lanes(&mut self) -> LaneSpots {
        LaneSpots {
            metrics: Metrics {
                cell: self.cell(),
                font_px: self.font_px,
            },
            spots: std::mem::take(&mut self.lanes),
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
