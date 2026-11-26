//! The window's drawing surface without the window: the shared ratatui
//! widgets rasterised into an RGB pixmap ([`PixelBackend`]), and the
//! pictures painted into it at their cell rects ([`pictures`]): the cover
//! ([`cover`]), the spectrum ([`spectrum`]), the Transition lanes
//! ([`lane`]) and their bar numbers ([`ruler`]), and Karaoke mode's
//! picture ([`karaoke`]). Lanes that follow the playhead go into a layer
//! of tiles the window scrolls by itself ([`strip`]). Used by the window
//! and, headless, by the tests and the screenshots.

mod cover;
mod karaoke;
mod lane;
mod pictures;
mod ruler;
mod spectrum;
mod strip;

pub use strip::{LayerFrame, MOST_SLOTS};

use image::RgbImage;
use ratatui::layout::Rect;
use ratatui::Terminal;

use super::big_text::Faces;
use super::pixels::{pack, unpack, PixelRect, Target};
use super::raster::{CellFont, PixelBackend};
use crate::art::Art;
use crate::view::karaoke::KaraokePicture;
use crate::view::{self, palette, Plot, Scene};
use cover::FittedCover;
use lane::{KeptWave, Melodies};
use pictures::{Paint, PixelPictures};
use ruler::KeptRuler;
use strip::{Part, Strip};

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
    /// The covers, spectrum and Transition lanes painted.
    pub pictures: std::time::Duration,
}

/// The window's pixels.
pub struct Canvas {
    terminal: Terminal<PixelBackend>,
    font_px: f32,
    cover: Option<FittedCover>,
    painted: Vec<Rect>,
    /// The covers in the pixmap now: rect, art id, whether below a line.
    covers: Vec<(Rect, u64, bool)>,
    /// The Transition lanes in the pixmap now.
    waves: Vec<KeptWave>,
    /// The Transition blocks' bar numbers in the pixmap now.
    rulers: Vec<KeptRuler>,
    /// The spectrum in the pixmap now.
    spectrum: Option<Plot>,
    /// Karaoke mode's backdrop, the text drawn large (by face), the picture now.
    backdrop: Option<super::karaoke::Backdrop>,
    faces: Faces,
    karaoke: Option<KaraokePicture>,
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
    /// The window shows the pixmap on the GPU ([`Canvas::set_gpu`]): lanes
    /// that follow the playhead go into a layer it scrolls itself, and
    /// Karaoke mode's stage, its bubbles' liquid and its ground are its,
    /// rather than in the pixmap.
    gpu: bool,
    /// That layer, while such a lane shows.
    strip: Option<Strip>,
    /// The melodies the lanes last drew.
    melodies: Melodies,
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

/// What the lanes and the bar numbers are painted with besides their
/// target: the cells' font (the numbers, and words a cell high), the faces
/// drawn large (larger words), the melodies the lanes last drew, and the
/// font size the window's scale gives.
struct Pen<'a> {
    font: &'a mut CellFont,
    faces: &'a mut Faces,
    melodies: &'a mut Melodies,
    font_px: f32,
}

impl Pen<'_> {
    /// The pixels per point of the window's scale factor.
    fn scale(&self) -> f64 {
        f64::from(self.font_px) / BASE_FONT_PX
    }

    /// The cell size in pixels.
    fn cell(&self) -> (usize, usize) {
        self.font.cell()
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
            cover: None,
            painted: Vec::new(),
            covers: Vec::new(),
            waves: Vec::new(),
            rulers: Vec::new(),
            spectrum: None,
            backdrop: None,
            faces: Faces::default(),
            karaoke: None,
            karaoke_layer: super::karaoke::Layer::default(),
            stage: super::stage::Stage::default(),
            staged: None,
            gpu: false,
            strip: None,
            melodies: Melodies::default(),
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

    /// The grid in cells.
    pub fn grid(&self) -> (u16, u16) {
        let area = self.terminal.backend().buffer().area;
        (area.width, area.height)
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
    pub fn fit(&mut self, width: u32, height: u32) -> (u16, u16) {
        let (cw, ch) = self.cell();
        let cols = (width as usize / cw).clamp(1, 1000) as u16;
        let rows = (height as usize / ch).clamp(1, 1000) as u16;
        if (cols, rows) != self.grid() {
            self.terminal.backend_mut().resize(cols, rows);
            let Ok(()) = self.terminal.resize(Rect::new(0, 0, cols, rows));
            self.changes.all = true;
        }
        (cols, rows)
    }

    /// Whether the window shows the pixmap on the GPU: lanes that follow the
    /// playhead then go into a layer it scrolls ([`Canvas::layer_frame`]),
    /// Karaoke mode's stage and ground are drawn there; otherwise all of it
    /// is painted into the pixmap each frame.
    pub fn set_gpu(&mut self, on: bool) {
        if on != self.gpu {
            self.gpu = on;
            self.strip = None;
            self.changes.all = true;
        }
    }

    /// Whether a layer shows now.
    pub fn has_layer(&self) -> bool {
        self.strip.is_some()
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
        let (cols, rows) = self.grid();
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

    /// Draws a frame of widgets alone (the setup screen, before the player
    /// has started): `draw`'s result back.
    pub fn draw_widgets<R>(&mut self, draw: impl FnOnce(&mut ratatui::Frame) -> R) -> R {
        self.terminal.backend_mut().reset_damage();
        let mut out = None;
        let Ok(_) = self.terminal.draw(|frame| out = Some(draw(frame)));
        let mut frame = Changes::default();
        self.cells_changed(&mut frame);
        self.changed = !frame.is_empty();
        self.changes.merge(frame);
        out.expect("the frame was drawn")
    }

    /// Draws a frame: the widgets, then the cover and the spectrum. A
    /// cover (and the caption over it) is painted again only when it is
    /// new or moved, or a cell under it was redrawn: it stays in the
    /// pixmap otherwise.
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
        let mut covers = Vec::new();
        let mut fresh: Vec<Rect> = Vec::new();
        let mut waves = std::mem::take(&mut self.waves);
        let mut rulers = std::mem::take(&mut self.rulers);
        let spectrum = self.spectrum.take();
        let karaoke = self.karaoke.take();
        let mut following: Vec<(Part, PixelRect)> = Vec::new();
        for paint in &pictures.paints {
            match paint {
                Paint::Cover(rect, below_line) => {
                    if let Some(art) = art {
                        let key = (*rect, art.id, *below_line);
                        let kept = self.covers.contains(&key)
                            && !self.terminal.backend().damage.touches(*rect);
                        if !kept {
                            self.paint_cover(*rect, art, *below_line);
                            fresh.push(*rect);
                            frame.add(self.pixel_rect(*rect));
                        }
                        covers.push(key);
                    }
                }
                Paint::Spectrum(plot) => {
                    let kept = spectrum.as_ref() == Some(plot)
                        && !self.terminal.backend().damage.touches(plot.area);
                    if !kept {
                        let rect = self.pixel_rect(plot.area);
                        let scale = self.scale();
                        spectrum::paint(
                            &mut self.terminal.backend_mut().target(),
                            rect,
                            scale,
                            plot,
                        );
                        frame.add(rect);
                    }
                    self.spectrum = Some(plot.clone());
                }
                Paint::Caption(cover, caption) => {
                    // Over a cover painted afresh only (its pixels are
                    // among the changed already): on one kept, the caption
                    // is still there.
                    if fresh.contains(cover) {
                        self.paint_caption(*cover, *caption);
                    }
                }
                Paint::Wave(plot) if self.gpu && plot.follow.is_some() => {
                    let r = self.pixel_rect(plot.area);
                    following.push((Part::Wave(plot.clone(), 0, 0), r));
                }
                Paint::Ruler(plot) if self.gpu && plot.follow.is_some() => {
                    let r = self.pixel_rect(plot.area);
                    following.push((Part::Ruler(plot.clone(), 0, 0), r));
                }
                Paint::Wave(plot) => {
                    let kept = waves
                        .iter()
                        .position(|k| k.area() == plot.area)
                        .map(|i| waves.swap_remove(i));
                    let (painted, kept) = self.paint_wave(plot, kept);
                    if let Some(rect) = painted {
                        frame.add(rect);
                    }
                    self.waves.extend(kept);
                }
                Paint::Karaoke(picture) => {
                    let wanted = super::stage::wanted(scene.state);
                    let last = karaoke.as_ref();
                    for rect in self.karaoke_picture(picture, last, wanted, art, started) {
                        frame.add(rect);
                    }
                }
                Paint::Ruler(plot) => {
                    let kept = rulers
                        .iter()
                        .position(|k| k.area == plot.area)
                        .map(|i| rulers.swap_remove(i));
                    let (painted, kept) = self.paint_ruler(plot, kept);
                    if painted {
                        frame.add(self.pixel_rect(plot.area));
                    }
                    self.rulers.extend(kept);
                }
            }
        }
        self.covers = covers;
        following.sort_by_key(|(_, r)| r.1);
        self.take_strip(following);
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

    /// The pixels per point of the window's scale factor.
    fn scale(&self) -> f64 {
        f64::from(self.font_px) / BASE_FONT_PX
    }

    /// What the lanes and bar numbers are painted with, for a target of
    /// their own (a tile).
    fn pen(&mut self) -> Pen<'_> {
        Pen {
            font: self.terminal.backend_mut().font(),
            faces: &mut self.faces,
            melodies: &mut self.melodies,
            font_px: self.font_px,
        }
    }

    /// The pixmap to paint the lanes and bar numbers into, and what they
    /// are painted with.
    fn pixmap_and_pen(&mut self) -> (Target<'_>, Pen<'_>) {
        let (target, font) = self.terminal.backend_mut().split();
        let pen = Pen {
            font,
            faces: &mut self.faces,
            melodies: &mut self.melodies,
            font_px: self.font_px,
        };
        (target, pen)
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
