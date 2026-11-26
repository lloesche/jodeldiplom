//! The lanes: the Transition tab's waveforms and bar numbers, the Karaoke
//! tab's voice lane with its notes and words, drawn by the window where the
//! Transition tab leaves room for them ([`crate::gui::transition`]) and
//! where the main screen has the voice's lane ([`crate::gui::metrics`]),
//! as their places in the window say ([`LaneSpots`]); their widget sits in
//! that part of the screen's layout ([`Lanes::view`]).
//!
//! Each lane is painted in tiles once ([`tiles`]: [`paint`] for a
//! waveform, [`words`] for the words along the voice's, [`ruler`] for the
//! bar numbers) and shown from them by [`LaneView`]: on the GPU, one
//! pipeline and a primitive a lane, each lane's tiles in a texture array
//! of its own kept by its place in the list and let go of when the lanes
//! go ([`gpu`]), the lane scrolled by the share of a column and the
//! playhead drawn over it each frame; with iced's CPU renderer (no GPU), an
//! image of each lane made from its tiles each frame. Which renderer draws
//! is what the widget last saw ([`Lanes::gpu`]).
//!
//! The waveforms' lanes are cards (raised), their corners rounded; the
//! bar numbers stand on the Transition tab's panel. Neither takes the
//! pointer: what is under them does (the Transition tab's places for it).

mod gpu;
mod keys;
mod paint;
mod ruler;
mod tiles;
mod words;

pub use gpu::Offscreen;
pub use paint::Rows;
pub use tiles::{LaneFrame, Part, TILE};

use std::sync::{Arc, Mutex, MutexGuard};

use iced::advanced::image::{self as image, Renderer as _};
use iced::advanced::layout::{self, Layout};
use iced::advanced::mouse;
use iced::advanced::renderer::{self, Renderer as _};
use iced::advanced::widget::{self, Tree, Widget};
use iced::widget::shader::Shader;
use iced::{Element, Length, Rectangle, Size, Theme, Widget as _};

use super::big_text::{Face, Faces};
use super::canvas::BASE_FONT_PX;
use super::look;
use super::pixels::PixelRect;
use crate::view::palette;
use paint::Melodies;
use tiles::Strip;

/// A waveform lane's ground: a card's ([`look::RAISED`]), on the panel
/// the Transition tab is on.
const GROUND: [u8; 3] = [0x1f, 0x1f, 0x27];
/// The bar numbers' ground: that panel's ([`look::SURFACE`]).
const RULER: [u8; 3] = [0x17, 0x17, 0x1d];
/// The window's ground.
const BASE: [u8; 3] = palette::BG;
/// The bar numbers' face: Nunito, semi-bold, as the interface's figures.
const NUMBERS: Face = Face::Words(600);

/// The sizes the lanes are painted at: a cell's (the words' letters take a
/// cell's width each, as the layout counts them), and the font's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    /// A cell's width and height, in pixels.
    pub cell: (usize, usize),
    /// The cells' font size, in pixels.
    pub font_px: f32,
}

impl Metrics {
    /// The pixels per point of the window's scale factor.
    fn scale(&self) -> f64 {
        f64::from(self.font_px) / BASE_FONT_PX
    }

    /// A line's width (a bar's, a mark's), in pixels.
    fn line(&self) -> usize {
        self.scale().round().max(1.0) as usize
    }
}

/// A lane: where it goes in the window's pixels, and what it shows.
#[derive(Clone, Debug)]
pub struct LaneSpot {
    /// Its place, in pixels.
    pub rect: PixelRect,
    /// What it shows.
    pub part: Part,
}

/// The lanes of a frame, top to bottom, at their sizes.
#[derive(Clone, Debug)]
pub struct LaneSpots {
    /// The sizes they are painted at.
    pub metrics: Metrics,
    /// The lanes.
    pub spots: Vec<LaneSpot>,
}

/// What the lanes are painted with: the faces of the words and the bar
/// numbers, the melodies the lanes last drew, the sizes.
pub(super) struct Pen {
    faces: Faces,
    melodies: Melodies,
    metrics: Metrics,
}

/// What a lane's primitive takes on the GPU: its last frame, the tiles
/// still to send over among it.
pub(super) struct Shared {
    lanes: Vec<LaneFrame>,
    /// The GPU holds lanes' tiles (to let go of once none shows).
    held: bool,
    /// Whether iced draws on the GPU, as the widget last saw (`None`
    /// before it was drawn).
    gpu: Option<bool>,
}

/// The lanes that show, their tiles, and how they reach the screen.
pub struct Lanes {
    pen: Pen,
    strips: Vec<Strip>,
    /// The frames for the GPU, where it shows them.
    shared: Arc<Mutex<Shared>>,
    /// Each lane's image where the processor shows them, and where it
    /// stood when it was made (made again only when it moved or a tile in
    /// it was painted).
    images: Vec<(Stand, image::Handle)>,
    /// Each lane's place in the window, in pixels, where it stands on its
    /// grid (the column at its left edge, the share of the next, the
    /// seconds a column), and its playhead, as last framed.
    places: Vec<PixelRect>,
    stands: Vec<(i64, f32, f64)>,
    heads: Vec<Option<(f32, usize)>>,
    /// Whether the GPU shows them; the window's pixels to a point.
    gpu: bool,
    scale: f32,
}

impl std::fmt::Debug for Lanes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lanes")
            .field("places", &self.places)
            .field("gpu", &self.gpu)
            .finish_non_exhaustive()
    }
}

impl Default for Lanes {
    fn default() -> Lanes {
        Lanes {
            pen: Pen {
                faces: Faces::default(),
                melodies: Melodies::default(),
                metrics: Metrics {
                    cell: (1, 1),
                    font_px: BASE_FONT_PX as f32,
                },
            },
            strips: Vec::new(),
            shared: Arc::new(Mutex::new(Shared {
                lanes: Vec::new(),
                held: false,
                gpu: None,
            })),
            images: Vec::new(),
            places: Vec::new(),
            stands: Vec::new(),
            heads: Vec::new(),
            gpu: true,
            scale: 1.0,
        }
    }
}

impl Lanes {
    fn shared(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether iced draws the lanes on the GPU, as their widget last saw
    /// (until it was drawn: as if it did).
    pub fn gpu(&self) -> bool {
        self.shared().gpu.unwrap_or(true)
    }

    /// The lanes a frame shows (`spots`), in a window of `scale` pixels a
    /// point, shown on the GPU (`gpu`) or by the processor: each kept
    /// where only its marks changed, made afresh where it did.
    pub fn show(&mut self, spots: LaneSpots, scale: f32, gpu: bool) {
        if spots.metrics != self.pen.metrics || gpu != self.gpu {
            self.strips.clear();
            self.pen.metrics = spots.metrics;
            self.gpu = gpu;
        }
        self.scale = scale;
        let mut kept = std::mem::take(&mut self.strips).into_iter();
        for spot in spots.spots {
            let strip = match kept.next() {
                Some(mut s) if s.same(&spot.part, spot.rect) => {
                    s.update(spot.part, &self.pen);
                    s
                }
                _ => Strip::new(spot.part, spot.rect, &self.pen),
            };
            self.strips.push(strip);
        }
    }

    /// A frame of the lanes with `heard` seconds into the track: the tiles
    /// each needs painted, and where it stands, for the GPU or as the
    /// processor's image of it.
    pub fn frame(&mut self, heard: f64) {
        let keep = !self.gpu;
        let frames: Vec<LaneFrame> = self
            .strips
            .iter_mut()
            .map(|s| s.frame(&mut self.pen, heard, keep))
            .collect();
        self.places = frames.iter().map(|f| f.rect).collect();
        self.stands = frames
            .iter()
            .zip(&self.strips)
            .map(|(f, s)| (f.offset, f.frac, s.px()))
            .collect();
        self.heads = frames.iter().map(|f| f.playhead).collect();
        if keep {
            let radius = self.radius();
            let mut before = std::mem::take(&mut self.images).into_iter();
            for (s, f) in self.strips.iter().zip(&frames) {
                let stand = (f.rect, f.offset);
                let kept = before
                    .next()
                    .filter(|(was, _)| *was == stand && f.uploads.is_empty() && !f.fresh);
                let image = kept.unwrap_or_else(|| {
                    let r = if s.is_wave() { radius } else { 0.0 };
                    let (w, h) = (f.rect.2 as u32, f.rect.3 as u32);
                    (stand, image::Handle::from_rgba(w, h, composite(s, f, r)))
                });
                self.images.push(image);
            }
            return;
        }
        self.images.clear();
        let mut shared = self.shared();
        let pending = std::mem::take(&mut shared.lanes);
        shared.lanes = frames;
        // What the GPU has not taken yet goes with this frame (a later
        // tile in a slot in place of an earlier one), where the lane's
        // tiles are the same.
        for (now, old) in shared.lanes.iter_mut().zip(pending) {
            if now.fresh || (now.rect.3, now.slots) != (old.rect.3, old.slots) {
                continue;
            }
            let mut uploads: Vec<_> = old
                .uploads
                .into_iter()
                .filter(|(slot, _)| now.uploads.iter().all(|(s, _)| s != slot))
                .collect();
            uploads.append(&mut now.uploads);
            now.uploads = uploads;
        }
    }

    /// Whether a lane that follows the playhead shows.
    pub fn following(&self) -> bool {
        self.strips.iter().any(Strip::follows)
    }

    /// Where each lane is in the window, in pixels, as last framed.
    pub fn places(&self) -> &[PixelRect] {
        &self.places
    }

    /// Where each lane stood on its grid as last framed: the column at its
    /// left edge, the share of the next, the seconds a column (a lane
    /// following the playhead stands on a whole column `share × seconds`
    /// earlier).
    pub fn stands(&self) -> &[(i64, f32, f64)] {
        &self.stands
    }

    /// A card's corners' radius, in pixels.
    fn radius(&self) -> f32 {
        look::radius::PANEL * self.scale
    }

    /// The lanes' last frame as the processor shows them (without a GPU),
    /// each over the window's ground in a window of `width` × `height`
    /// pixels: `0x00RRGGBB` words row by row, for the tests and the
    /// examples (the GPU's frame, [`Offscreen::show`], is the same).
    pub fn picture(&self, width: usize, height: usize) -> Vec<u32> {
        let mut out = vec![crate::gui::pixels::pack(BASE); width * height];
        let accent = crate::gui::pixels::pack(palette::ACCENT);
        for ((&(x0, y0, w, h), (_, handle)), head) in
            self.places.iter().zip(&self.images).zip(&self.heads)
        {
            let image::Handle::Rgba { pixels, .. } = handle else {
                continue;
            };
            for (i, px) in pixels.as_chunks::<4>().0.iter().enumerate() {
                let (x, y) = (x0 + i % w, y0 + i / w);
                if x >= width || y >= height {
                    continue;
                }
                let a = f32::from(px[3]) / 255.0;
                let at = y * width + x;
                out[at] = crate::gui::pixels::over(out[at], [px[0], px[1], px[2]], a);
            }
            // The playhead over it, as the window draws it.
            if let Some((hx, hw)) = *head {
                let from = x0 + hx.floor() as usize;
                for y in y0..(y0 + h).min(height) {
                    for x in from..(from + hw).min(x0 + w).min(width) {
                        out[y * width + x] = accent;
                    }
                }
            }
        }
        out
    }

    /// The lanes' widget, filling the part of the screen's layout whose
    /// top left is at `origin` in the window (points): each lane where
    /// its place is in it.
    pub fn view<'a, M: 'a>(&'a self, origin: iced::Point) -> Element<'a, M> {
        let s = self.scale;
        let points = |&(x, y, w, h): &PixelRect| {
            Rectangle::new(
                iced::Point::new(x as f32 / s - origin.x, y as f32 / s - origin.y),
                Size::new(w as f32 / s, h as f32 / s),
            )
        };
        let mut places: Vec<Rectangle> = self.places.iter().map(points).collect();
        let count = places.len();
        let lane = |id: usize, radius: f32| gpu::Lane {
            shared: Arc::clone(&self.shared),
            id,
            count,
            radius,
        };
        let mut lanes: Vec<gpu::Lane> = self
            .strips
            .iter()
            .enumerate()
            .take(count)
            .map(|(id, strip)| lane(id, if strip.is_wave() { self.radius() } else { 0.0 }))
            .collect();
        // No lane, but the GPU still holds some: a lane of none lets go of
        // them.
        if count == 0 && self.gpu && self.shared().held {
            lanes.push(lane(0, 0.0));
            places.push(Rectangle::default());
        }
        let shaders = lanes
            .into_iter()
            .zip(&places)
            .map(|(lane, place)| Shader::new(lane).width(place.width).height(place.height))
            .collect();
        LaneView {
            lanes: self,
            shaders,
            places,
        }
        .boxed()
    }
}

/// Where a lane's image stood: its place and its column.
type Stand = (PixelRect, i64);

/// A lane's frame as the processor shows it: its tiles from its column
/// (whole columns: the share of the next is the GPU's), its corners
/// rounded to `radius` pixels (clear beyond); the playhead is drawn over it
/// apart. Its RGBA bytes, straight.
fn composite(strip: &Strip, frame: &LaneFrame, radius: f32) -> Vec<u8> {
    let (_, _, w, h) = frame.rect;
    let mut out = vec![0u8; w * h * 4];
    let ground = crate::gui::pixels::pack(BASE);
    // The runs of columns from one tile: where in the lane, the tile, and
    // its first column in it.
    let mut runs = Vec::new();
    let mut x = 0;
    while x < w {
        let col = frame.offset + x as i64;
        let from = col.rem_euclid(TILE as i64) as usize;
        let n = (TILE - from).min(w - x);
        runs.push((x, n, strip.kept(col.div_euclid(TILE as i64)), from));
        x += n;
    }
    let rgba = |word: u32| [(word >> 16) as u8, (word >> 8) as u8, word as u8, 255];
    for (y, row) in out.chunks_exact_mut(w * 4).enumerate() {
        let row = row.as_chunks_mut::<4>().0;
        for &(x, n, tile, from) in &runs {
            let to = &mut row[x..x + n];
            match tile {
                Some(t) => {
                    let line = &t[y * TILE + from..y * TILE + from + n];
                    for (px, &word) in to.iter_mut().zip(line) {
                        *px = rgba(word);
                    }
                }
                None => to.fill(rgba(ground)),
            }
        }
    }
    // The corners: only their squares are not covered whole.
    let r = radius.ceil() as usize;
    if r > 0 {
        for y in (0..r.min(h)).chain(h.saturating_sub(r).max(r.min(h))..h) {
            for x in (0..r.min(w)).chain(w.saturating_sub(r).max(r.min(w))..w) {
                let cover = corner_cover(x, y, (w, h), radius);
                out[(y * w + x) * 4 + 3] = (cover * 255.0).round() as u8;
            }
        }
    }
    out
}

/// How much of pixel (`x`, `y`) of a card `(w, h)` with corners of
/// `radius` it covers, as the GPU's shader has it.
fn corner_cover(x: usize, y: usize, (w, h): (usize, usize), radius: f32) -> f32 {
    let (hw, hh) = (w as f32 / 2.0, h as f32 / 2.0);
    let r = radius.min(hw).min(hh);
    let qx = (x as f32 + 0.5 - hw).abs() - (hw - r);
    let qy = (y as f32 + 0.5 - hh).abs() - (hh - r);
    let d = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r;
    (0.5 - d).clamp(0.0, 1.0)
}

/// The lanes in their part of the screen: on the GPU a shader a lane (one
/// pipeline, a primitive each), else each lane's image. It takes no
/// events.
struct LaneView<'a, M> {
    lanes: &'a Lanes,
    /// Each lane's shader and where it goes in the widget, in points.
    shaders: Vec<Shader<M, gpu::Lane>>,
    places: Vec<Rectangle>,
}

impl<M> widget::Meta for LaneView<'_, M> {}

impl<M> Widget<M, Theme, iced::Renderer> for LaneView<'_, M> {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children::<M, Theme, iced::Renderer>(&mut self.shaders);
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &iced::Renderer, limits: &layout::Limits) {
        for ((shader, place), child) in self
            .shaders
            .iter_mut()
            .zip(&self.places)
            .zip(&mut tree.children)
        {
            let limits = layout::Limits::new(Size::ZERO, place.size());
            Widget::<M, Theme, iced::Renderer>::layout(shader, child, renderer, &limits);
            child.translation = iced::Vector::new(place.x, place.y);
        }
        tree.size = limits.max;
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let gpu = matches!(renderer, iced::Renderer::Primary(_));
        self.lanes.shared().gpu = Some(gpu);
        if gpu {
            for (shader, (layout, tree)) in self.shaders.iter().zip(layout.iter(&tree.children)) {
                shader.draw(tree, renderer, theme, style, layout, cursor, viewport);
            }
            return;
        }
        let origin = iced::Vector::new(layout.bounds().x, layout.bounds().y);
        let lanes = self.lanes;
        for (place, (_, handle)) in self.places.iter().zip(&lanes.images) {
            let bounds = *place + origin;
            renderer.draw_image(
                image::Image::new(handle.clone()).filter_method(image::FilterMethod::Nearest),
                bounds,
                bounds,
            );
        }
        // The playheads over the images (a layer of their own: iced draws
        // its quads under its images).
        renderer.with_layer(*viewport, |renderer| {
            for (place, head) in self.places.iter().zip(&lanes.heads) {
                let Some((x, width)) = *head else {
                    continue;
                };
                let s = lanes.scale;
                let bounds = Rectangle::new(
                    iced::Point::new(place.x + x.floor() / s, place.y) + origin,
                    Size::new(width as f32 / s, place.height),
                );
                renderer.fill_quad(
                    renderer::Quad {
                        bounds,
                        ..renderer::Quad::default()
                    },
                    look::ACCENT,
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes' grounds are the look's: the cards raised, the bar
    /// numbers on the panel's, over the window's.
    #[test]
    fn the_lanes_stand_on_the_looks_grounds() {
        assert_eq!(GROUND, look::bytes(look::RAISED));
        assert_eq!(RULER, look::bytes(look::SURFACE));
        assert_eq!(BASE, look::bytes(look::BG));
    }

    /// A card's corners are clear beyond their radius and covered inside;
    /// its edges between them covered whole.
    #[test]
    fn a_cards_corners_are_rounded() {
        let size = (100, 40);
        assert_eq!(corner_cover(0, 0, size, 10.0), 0.0);
        assert_eq!(corner_cover(99, 39, size, 10.0), 0.0);
        assert_eq!(corner_cover(50, 0, size, 10.0), 1.0);
        assert_eq!(corner_cover(0, 20, size, 10.0), 1.0);
        assert_eq!(corner_cover(5, 5, size, 10.0), 1.0);
        // No radius: every pixel.
        assert_eq!(corner_cover(0, 0, size, 0.0), 1.0);
    }
}
