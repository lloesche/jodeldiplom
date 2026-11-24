//! Karaoke mode in pixels: the cover blurred over the whole screen under a
//! dark layer, the lyrics in large type over it, filling as they are sung,
//! and the count-in's numbers shrinking and fading on their beats.

use std::collections::HashMap;
use std::sync::Arc;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::imageops::{self, FilterType};
use image::RgbImage;

use super::fonts;
use super::notes::SungOp;
use super::raster::{pack, unpack};
use crate::art::Art;
use crate::view::colors;
use crate::view::karaoke::{line_color, KaraokePicture, LineKind, LyricLine, HIT};

/// How dark the layer over the blurred cover is (0 clear, 1 black).
const SHADE: f32 = 0.55;
/// The long side of the cover before it is blurred, in pixels: the blur
/// is done small and scaled up, so it is cheap at any screen size.
const SMALL: u32 = 48;

/// The screen's background: the cover blurred and shaded, at the screen's
/// size, `0x00RRGGBB` words.
pub struct Backdrop {
    /// The cover it is made of (0: none).
    pub art: u64,
    /// Its size.
    pub size: (usize, usize),
    /// The pixels.
    pub pixels: Vec<u32>,
}

impl Backdrop {
    /// The backdrop for `art` (none: a plain dark gradient) at `width` ×
    /// `height`.
    pub fn new(art: Option<&Art>, width: usize, height: usize) -> Backdrop {
        let (w, h) = (width.max(1) as u32, height.max(1) as u32);
        let pixels = match art {
            Some(art) => blurred(&art.image, w, h),
            None => gradient(w, h),
        };
        Backdrop {
            art: art.map_or(0, |a| a.id),
            size: (width, height),
            pixels,
        }
    }
}

/// `image` cropped to the screen's shape, shrunk, blurred three times,
/// scaled up to `w` × `h` and shaded.
fn blurred(image: &RgbImage, w: u32, h: u32) -> Vec<u32> {
    // The middle of the cover in the screen's shape (it fills the screen).
    let (iw, ih) = (image.width().max(1), image.height().max(1));
    let screen = f64::from(w) / f64::from(h);
    let (cw, ch) = if f64::from(iw) / f64::from(ih) > screen {
        (((f64::from(ih) * screen).round() as u32).clamp(1, iw), ih)
    } else {
        (iw, ((f64::from(iw) / screen).round() as u32).clamp(1, ih))
    };
    let crop = imageops::crop_imm(image, (iw - cw) / 2, (ih - ch) / 2, cw, ch).to_image();
    let (sw, sh) = if w >= h {
        (SMALL, (SMALL * h / w).max(1))
    } else {
        ((SMALL * w / h).max(1), SMALL)
    };
    let mut small = imageops::resize(&crop, sw, sh, FilterType::Triangle);
    for _ in 0..3 {
        box_blur(&mut small, 2);
    }
    let big = imageops::resize(&small, w, h, FilterType::Triangle);
    big.pixels().map(|p| pack(shade(p.0, SHADE))).collect()
}

/// A plain backdrop without a cover: the window's background, a little
/// lighter towards the middle.
fn gradient(w: u32, h: u32) -> Vec<u32> {
    let base = rgb(colors::BG);
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let t = 1.0 - (2.0 * f64::from(y) / f64::from(h.max(1)) - 1.0).abs();
        let lift = (18.0 * t) as u8;
        let c = base.map(|v| v.saturating_add(lift));
        out.extend(std::iter::repeat_n(pack(c), w as usize));
    }
    out
}

/// A box blur of `radius` pixels, across and then down, edges held.
fn box_blur(image: &mut RgbImage, radius: i64) {
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let n = (2 * radius + 1) as u32;
    let get = |img: &RgbImage, x: i64, y: i64| {
        img.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32)
            .0
    };
    let source = image.clone();
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u32; 3];
            for d in -radius..=radius {
                let p = get(&source, x + d, y);
                (0..3).for_each(|i| sum[i] += u32::from(p[i]));
            }
            image.put_pixel(x as u32, y as u32, image::Rgb(sum.map(|s| (s / n) as u8)));
        }
    }
    let source = image.clone();
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0u32; 3];
            for d in -radius..=radius {
                let p = get(&source, x, y + d);
                (0..3).for_each(|i| sum[i] += u32::from(p[i]));
            }
            image.put_pixel(x as u32, y as u32, image::Rgb(sum.map(|s| (s / n) as u8)));
        }
    }
}

fn shade(c: [u8; 3], dark: f32) -> [u8; 3] {
    c.map(|v| (f32::from(v) * (1.0 - dark)).round() as u8)
}

pub(super) fn rgb(color: ratatui::style::Color) -> [u8; 3] {
    match color {
        ratatui::style::Color::Rgb(r, g, b) => [r, g, b],
        _ => [200, 200, 200],
    }
}

pub(super) fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (f32::from(a[i]) * (1.0 - t) + f32::from(b[i]) * t).round() as u8)
}

/// The pixels round text made for the atlas ([`BigText::coverage`]).
pub const COVER_PAD: usize = 2;

/// One glyph at one size: its coverage and where it sits from the pen.
struct Glyph {
    left: i32,
    top: i32,
    width: usize,
    height: usize,
    coverage: Vec<u8>,
    advance: f32,
}

/// Text at any size, glyphs kept per character and size: in the main font,
/// or (made with [`BigText::lyrics`]) in the lyrics font where there is one
/// ([`fonts::lyrics_font`]), the main font for what it lacks.
#[derive(Default)]
pub struct BigText {
    glyphs: HashMap<(char, u32), Option<Arc<Glyph>>>,
    lyrics: bool,
}

/// Where to draw into: the pixmap, its width and height.
pub struct Target<'a> {
    /// `0x00RRGGBB` words, row by row.
    pub pixels: &'a mut [u32],
    /// Its width.
    pub width: usize,
    /// Its height.
    pub height: usize,
}

impl BigText {
    /// Text in the lyrics font.
    pub fn lyrics() -> BigText {
        BigText {
            glyphs: HashMap::new(),
            lyrics: true,
        }
    }

    fn main_font() -> Option<FontRef<'static>> {
        FontRef::try_from_slice(fonts::main_font()).ok()
    }

    fn lyrics_font() -> Option<FontRef<'static>> {
        fonts::lyrics_font().and_then(|data| FontRef::try_from_slice(data).ok())
    }

    fn glyph(&mut self, c: char, px: f32) -> Option<Arc<Glyph>> {
        let key = (c, px.round() as u32);
        if let Some(g) = self.glyphs.get(&key) {
            return g.clone();
        }
        let lyrics = self
            .lyrics
            .then(Self::lyrics_font)
            .flatten()
            .filter(|f| f.glyph_id(c).0 != 0);
        let font = lyrics
            .or_else(|| Self::main_font().filter(|f| f.glyph_id(c).0 != 0))
            .or_else(|| fonts::font_for(c))
            .or_else(Self::main_font);
        let glyph = font.map(|font| {
            let scale = PxScale::from(px);
            let scaled = font.as_scaled(scale);
            let id = font.glyph_id(c);
            let advance = scaled.h_advance(id);
            let ascent = scaled.ascent();
            let g = id.with_scale_and_position(scale, ab_glyph::point(0.0, ascent));
            match font.outline_glyph(g) {
                Some(outline) => {
                    let bounds = outline.px_bounds();
                    let (width, height) = (bounds.width() as usize, bounds.height() as usize);
                    let mut coverage = vec![0u8; width * height];
                    outline.draw(|x, y, v| {
                        let (x, y) = (x as usize, y as usize);
                        if x < width && y < height {
                            coverage[y * width + x] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    });
                    Arc::new(Glyph {
                        left: bounds.min.x as i32,
                        top: bounds.min.y as i32,
                        width,
                        height,
                        coverage,
                        advance,
                    })
                }
                // A space: nothing to draw, only its advance.
                None => Arc::new(Glyph {
                    left: 0,
                    top: 0,
                    width: 0,
                    height: 0,
                    coverage: Vec::new(),
                    advance,
                }),
            }
        });
        self.glyphs.insert(key, glyph.clone());
        // Sizes change with the screen: keep the cache from growing.
        if self.glyphs.len() > 4096 {
            self.glyphs.clear();
        }
        glyph
    }

    /// The width of `text` at `px`.
    pub fn measure(&mut self, text: &str, px: f32) -> f32 {
        self.measure_from(text, px, px)
    }

    /// `text` at `px` as coverage (0 to 255), row by row, for the GPU's
    /// atlas: [`COVER_PAD`] pixels round it, its line's top [`COVER_PAD`]
    /// down. Returns its width, height and coverage.
    pub fn coverage(&mut self, text: &str, px: f32) -> (usize, usize, Vec<u8>) {
        let w = self.measure(text, px).ceil().max(0.0) as usize + 2 * COVER_PAD;
        let h = (px * 1.3).ceil().max(1.0) as usize + 2 * COVER_PAD;
        let mut out = vec![0u8; w * h];
        let mut pen = COVER_PAD as f32;
        for c in text.chars() {
            let Some(g) = self.glyph(c, px) else {
                continue;
            };
            for gy in 0..g.height {
                let ty = COVER_PAD as i32 + g.top + gy as i32;
                if ty < 0 || ty >= h as i32 {
                    continue;
                }
                for gx in 0..g.width {
                    let tx = pen.floor() as i32 + g.left + gx as i32;
                    if tx < 0 || tx >= w as i32 {
                        continue;
                    }
                    let at = ty as usize * w + tx as usize;
                    out[at] = out[at].max(g.coverage[gy * g.width + gx]);
                }
            }
            pen += g.advance;
        }
        (w, h, out)
    }

    /// The width of `text` at `px`, its glyphs made at `base`.
    fn measure_from(&mut self, text: &str, base: f32, px: f32) -> f32 {
        let scale = px / base.max(1.0);
        let sum: f32 = text
            .chars()
            .map(|c| self.glyph(c, base).map_or(0.0, |g| g.advance))
            .sum();
        sum * scale
    }

    /// Draws `text` at `px` with its top-left at `x`, `y`: left of `fill_x`
    /// in `sung`, right of it in `rest`, at `alpha`. The glyphs are made at
    /// `base` and scaled to `px`, so a size that changes from frame to
    /// frame (the count-in shrinking, a line moving into its place) makes
    /// no new glyphs. Only the pixels in `clip` are drawn. Returns the
    /// pixels it covered.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        target: &mut Target,
        text: &str,
        base: f32,
        px: f32,
        (x, y): (f32, f32),
        fill_x: f32,
        colors: ([u8; 3], [u8; 3]),
        alpha: f32,
        clip: Rect,
    ) -> Option<Rect> {
        let (cx0, cy0) = (clip.0, clip.1);
        let cx1 = (clip.0 + clip.2).min(target.width);
        let cy1 = (clip.1 + clip.3).min(target.height);
        let scale = px / base.max(1.0);
        let exact = (scale - 1.0).abs() < 1e-3;
        let mut pen = x;
        let mut covered: Option<Rect> = None;
        let mut put = |target: &mut Target, tx: usize, ty: usize, a: f32| {
            let color = if (tx as f32) < fill_x {
                colors.0
            } else {
                colors.1
            };
            let i = ty * target.width + tx;
            target.pixels[i] = pack(mix(unpack(target.pixels[i]), color, alpha * a));
            covered = Some(union(covered, (tx, ty, 1, 1)));
        };
        for c in text.chars() {
            let Some(g) = self.glyph(c, base) else {
                continue;
            };
            if exact {
                for gy in 0..g.height {
                    let ty = y.floor() as i32 + g.top + gy as i32;
                    if ty < cy0 as i32 || ty >= cy1 as i32 {
                        continue;
                    }
                    for gx in 0..g.width {
                        let a = g.coverage[gy * g.width + gx];
                        let tx = pen.floor() as i32 + g.left + gx as i32;
                        if a == 0 || tx < cx0 as i32 || tx >= cx1 as i32 {
                            continue;
                        }
                        put(target, tx as usize, ty as usize, f32::from(a) / 255.0);
                    }
                }
            } else {
                // Each pixel of the scaled glyph read from the glyph made
                // at `base`, between its four nearest coverage values.
                let (left, top) = (pen + g.left as f32 * scale, y + g.top as f32 * scale);
                let (w, h) = (g.width as f32 * scale, g.height as f32 * scale);
                let x0 = (left.floor().max(0.0) as usize).max(cx0);
                let y0 = (top.floor().max(0.0) as usize).max(cy0);
                let x1 = ((left + w).ceil().max(0.0) as usize).min(cx1);
                let y1 = ((top + h).ceil().max(0.0) as usize).min(cy1);
                for ty in y0..y1 {
                    let v = (ty as f32 + 0.5 - top) / scale - 0.5;
                    for tx in x0..x1 {
                        let u = (tx as f32 + 0.5 - left) / scale - 0.5;
                        let a = g.sample(u, v);
                        if a > 0.5 {
                            put(target, tx, ty, a / 255.0);
                        }
                    }
                }
            }
            pen += g.advance * scale;
        }
        covered
    }
}

impl Glyph {
    /// The coverage at (`u`, `v`) in the glyph's pixels, 0 to 255, between
    /// the four nearest; nothing outside.
    fn sample(&self, u: f32, v: f32) -> f32 {
        let (fx, fy) = (u.floor(), v.floor());
        let (tx, ty) = (u - fx, v - fy);
        let (x, y) = (fx as i64, fy as i64);
        let at = |x: i64, y: i64| -> f32 {
            if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
                0.0
            } else {
                f32::from(self.coverage[y as usize * self.width + x as usize])
            }
        };
        let top = at(x, y) * (1.0 - tx) + at(x + 1, y) * tx;
        let bottom = at(x, y + 1) * (1.0 - tx) + at(x + 1, y + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// A rect of pixels: x, y, width, height.
pub type Rect = (usize, usize, usize, usize);

/// The rect around `a` (if any) and `b`.
pub fn union(a: Option<Rect>, b: Rect) -> Rect {
    match a {
        None => b,
        Some((x, y, w, h)) => {
            let (x0, y0) = (x.min(b.0), y.min(b.1));
            let (x1, y1) = ((x + w).max(b.0 + b.2), (y + h).max(b.1 + b.3));
            (x0, y0, x1 - x0, y1 - y0)
        }
    }
}

/// The size glyphs of `px` are made at: their own up to 200 px (lines of
/// lyrics, a few sizes each), else the next of steps a quarter apart
/// (the count-in's numbers, shrinking every frame), scaled down from it.
fn base_size(px: f32) -> f32 {
    if px <= 200.0 {
        return px.round().max(1.0);
    }
    let step = 1.25f32;
    step.powf((px.ln() / step.ln()).ceil())
}

/// Where a line goes: its size and the top of its text.
#[derive(Clone, Copy)]
struct Slot {
    px: f32,
    y: f32,
}

/// How far down a line's text is from the next one's: its size and more.
fn gap(px: f32) -> f32 {
    px * 1.45
}

/// The room under the line being sung for its notes, as a share of its
/// size; where the notes go in it (their highest's top and lowest's
/// bottom, from the text's top), and a note's thickness.
const NOTES_ROOM: f32 = 1.45;
pub(super) const NOTES_TOP: f32 = 1.3;
pub(super) const NOTES_BOTTOM: f32 = 2.5;
pub(super) const NOTE_THICK: f32 = 0.09;

/// Whether `line` gets room for notes under it: a lyric being sung.
fn noted(line: &LyricLine) -> bool {
    line.kind == LineKind::Current && line.id.is_some()
}

/// The lines' sizes (by what they are, smaller where too wide) and places,
/// stacked and centred in `room` rows of pixels from `top`.
fn slots(
    text: &mut BigText,
    lines: &[LyricLine],
    width: usize,
    top: usize,
    room: usize,
    big: f32,
) -> Vec<Slot> {
    let widest = width as f32 * 0.92;
    let sizes: Vec<f32> = lines
        .iter()
        .map(|l| {
            let px = match l.kind {
                LineKind::Current => big,
                LineKind::Next | LineKind::Previous => big * 0.62,
                LineKind::Info => big * 0.42,
                LineKind::Warning => big * 0.36,
                LineKind::Medal(_) => big * 0.8,
            };
            let w = text.measure(&l.text, px);
            if w > widest && w > 0.0 {
                (px * widest / w).max(10.0).round()
            } else {
                px.round()
            }
        })
        .collect();
    let below = |l: &LyricLine, px: f32| if noted(l) { px * NOTES_ROOM } else { 0.0 };
    let total: f32 = lines
        .iter()
        .zip(&sizes)
        .map(|(l, &px)| gap(px) + below(l, px))
        .sum();
    let mut y = top as f32 + (room as f32 - total).max(0.0) / 2.0;
    lines
        .iter()
        .zip(sizes)
        .map(|(l, px)| {
            let slot = Slot {
                px,
                y: y + (gap(px) - px) / 2.0,
            };
            y += gap(px) + below(l, px);
            slot
        })
        .collect()
}

/// How far into `line` (at `px`, glyphs made at `base`) character `at` is,
/// in pixels (a fraction through one, through its width).
fn x_of_char(text: &mut BigText, line: &str, base: f32, px: f32, at: f32) -> f32 {
    let at = at.max(0.0);
    let whole = at.floor() as usize;
    let before: String = line.chars().take(whole).collect();
    let x = text.measure_from(&before, base, px);
    match line.chars().nth(whole) {
        Some(c) => x + (at - whole as f32) * text.measure_from(&c.to_string(), base, px),
        None => x,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// One piece of text the picture draws, and where.
#[derive(Clone, Debug, PartialEq)]
struct TextOp {
    text: String,
    base: f32,
    px: f32,
    at: (f32, f32),
    fill_x: f32,
    colors: ([u8; 3], [u8; 3]),
    alpha: f32,
    /// The notes under it and the singing on them, where it has notes.
    sung: Option<SungOp>,
}

impl TextOp {
    /// The pixels it may cover (a little more than its glyphs), within
    /// `bounds`.
    fn rect(&self, width: f32, bounds: Rect) -> Rect {
        let pad = self.px * 0.25 + 2.0;
        let x0 = (self.at.0 - pad).max(bounds.0 as f32);
        let y0 = (self.at.1 - pad).max(bounds.1 as f32);
        let x1 = (self.at.0 + width + pad).min((bounds.0 + bounds.2) as f32);
        let y1 = (self.at.1
            + self.px
                * if self.sung.is_none() {
                    1.5
                } else {
                    NOTES_BOTTOM
                }
            + pad)
            .min((bounds.1 + bounds.3) as f32);
        if x1 <= x0 || y1 <= y0 {
            return (0, 0, 0, 0);
        }
        let (x0, y0) = (x0.floor() as usize, y0.floor() as usize);
        (x0, y0, x1.ceil() as usize - x0, y1.ceil() as usize - y0)
    }
}

/// What Karaoke mode's picture drew last: each piece of text and the
/// pixels it may cover. A frame redraws only the pieces that changed (a
/// line filling as it is sung), with the backdrop put back under them.
/// And what of its bubbles the GPU draws over it ([`Layer::over`]: their
/// liquid, with a GPU, and their bursts), as last painted.
#[derive(Default)]
pub struct Layer {
    drawn: Vec<(TextOp, Rect)>,
    /// What the GPU draws over the picture; whether there is one to (set
    /// it before painting: the bars' fill is painted only without).
    pub over: super::notes::Over,
}

/// Where `a` and `b` meet, if they do.
fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let (x0, y0) = (a.0.max(b.0), a.1.max(b.1));
    let (x1, y1) = ((a.0 + a.2).min(b.0 + b.2), (a.1 + a.3).min(b.1 + b.3));
    (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
}

fn overlaps(a: Rect, b: Rect) -> bool {
    a.2 > 0
        && a.3 > 0
        && b.2 > 0
        && b.3 > 0
        && a.0 < b.0 + b.2
        && b.0 < a.0 + a.2
        && a.1 < b.1 + b.3
        && b.1 < a.1 + a.3
}

/// Paints `picture` (its area in pixels: `area`) into `target` over the
/// backdrop; the controls' bar (pixels) darker still. `full`: the whole
/// area afresh; otherwise only the text that changed since `layer`'s last
/// frame, the backdrop put back under it. Returns the pixels changed.
#[allow(clippy::too_many_arguments)]
pub fn paint(
    target: &mut Target,
    text: &mut BigText,
    layer: &mut Layer,
    backdrop: &Backdrop,
    picture: &KaraokePicture,
    area: Rect,
    bar: Option<Rect>,
    full: bool,
) -> Vec<Rect> {
    let (ax, ay, aw, ah) = area;
    // The lyrics, centred on the whole screen (the controls never move
    // them), and drawn only above the controls' bar while it shows over
    // them.
    let room = bar.map_or(ah, |(_, by, _, _)| by.saturating_sub(ay));
    let lyrics = (ax, ay, aw, room);
    let (ops, over) = text_ops(text, picture, (area, ah), layer.over.gpu);
    layer.over = over;
    let now: Vec<(TextOp, Rect)> = ops
        .into_iter()
        .map(|op| {
            let w = text.measure_from(&op.text, op.base, op.px);
            let rect = op.rect(w, lyrics);
            (op, rect)
        })
        .collect();
    let put_back = |target: &mut Target, (rx, ry, rw, rh): Rect| {
        // Within the area and the backdrop.
        let x0 = rx.max(ax);
        let x1 = (rx + rw)
            .min(ax + aw.min(backdrop.size.0))
            .min(target.width);
        let y1 = (ry + rh)
            .min(ay + ah.min(backdrop.size.1))
            .min(target.height);
        if x1 <= x0 {
            return;
        }
        for ty in ry.max(ay)..y1 {
            let from = (ty - ay) * backdrop.size.0 + (x0 - ax);
            let at = ty * target.width + x0;
            target.pixels[at..at + (x1 - x0)]
                .copy_from_slice(&backdrop.pixels[from..from + (x1 - x0)]);
        }
    };
    let draw_in = |target: &mut Target, text: &mut BigText, ops: &[(TextOp, Rect)], clip: Rect| {
        for (op, rect) in ops {
            // Only within what the op may cover, so a frame drawn where it
            // changed is the frame drawn afresh.
            if let Some(clip) = intersect(*rect, clip) {
                text.draw(
                    target, &op.text, op.base, op.px, op.at, op.fill_x, op.colors, op.alpha, clip,
                );
                if let Some(sung) = &op.sung {
                    super::notes::draw(target, sung, op.at, op.px, op.fill_x, op.alpha, clip);
                }
            }
        }
    };
    let changed = if full {
        put_back(target, area);
        if let Some((bx, by, bw, bh)) = bar {
            for y in by..(by + bh).min(target.height) {
                for x in bx..(bx + bw).min(target.width) {
                    let i = y * target.width + x;
                    target.pixels[i] = pack(shade(unpack(target.pixels[i]), 0.6));
                }
            }
        }
        draw_in(target, text, &now, lyrics);
        vec![area]
    } else {
        // The pieces that differ from last frame's at the same place in
        // the list (or are new, or gone): where they were and are.
        let n = now.len().max(layer.drawn.len());
        let mut dirty: Vec<Rect> = Vec::new();
        for k in 0..n {
            let (old, new) = (layer.drawn.get(k), now.get(k));
            if old.map(|o| &o.0) == new.map(|n| &n.0) {
                continue;
            }
            for (_, rect) in old.into_iter().chain(new) {
                if rect.2 > 0 && rect.3 > 0 {
                    dirty.push(*rect);
                }
            }
        }
        // Overlapping ones as one, so nothing is drawn twice.
        let mut merged: Vec<Rect> = Vec::new();
        for rect in dirty {
            let mut rect = rect;
            while let Some(i) = merged.iter().position(|&m| overlaps(m, rect)) {
                rect = union(Some(merged.swap_remove(i)), rect);
            }
            merged.push(rect);
        }
        for &rect in &merged {
            put_back(target, rect);
            draw_in(target, text, &now, rect);
        }
        merged
    };
    layer.drawn = now;
    changed
}

/// The pieces of text of `picture`: its lines (moving into their places
/// while they scroll) and the count-in's number; what the GPU draws over
/// them (with one, `gpu`).
fn text_ops(
    text: &mut BigText,
    picture: &KaraokePicture,
    (area, room): (Rect, usize),
    gpu: bool,
) -> (Vec<TextOp>, super::notes::Over) {
    let (ax, ay, aw, ah) = area;
    let big = (ah as f32 / 11.0).clamp(20.0, 160.0).round();
    let accent = rgb(colors::ACCENT);
    let (mut ops, mut over) = (Vec::new(), super::notes::Over::new(gpu));
    let mut line_op =
        |text: &mut BigText, line: &LyricLine, base: f32, slot: Slot, rest: [u8; 3], alpha: f32| {
            let alpha = alpha * line.alpha.clamp(0.0, 1.0) as f32;
            if alpha <= 0.004 {
                return;
            }
            let w = text.measure_from(&line.text, base, slot.px);
            let x = ax as f32 + (aw as f32 - w) / 2.0;
            // The fill to the character sung, measured (the font's letters
            // are not all as wide).
            let chars = line.text.chars().count() as f32;
            let fill_x = x + x_of_char(
                text,
                &line.text,
                base,
                slot.px,
                chars * line.fill.clamp(0.0, 1.0) as f32,
            );
            let sung_color = match line.kind {
                LineKind::Current => accent,
                _ => rest,
            };
            let x_of = |text: &mut BigText, at| x_of_char(text, &line.text, base, slot.px, at);
            let at = (x, slot.y);
            over.add(line, at, slot.px, |c| x_of(text, c));
            let sung = SungOp::of(line, slot.px, gpu, |c| x_of(text, c));
            ops.push(TextOp {
                text: line.text.clone(),
                base,
                px: slot.px,
                at: (x, slot.y),
                fill_x,
                colors: (sung_color, rest),
                alpha,
                sung,
            });
        };
    let faded = |kind: LineKind| match kind {
        LineKind::Previous => 0.5,
        _ => 1.0,
    };
    let now = slots(text, &picture.lines, aw, ay, room, big);
    let moving = picture.moved < 1.0 && !picture.before.is_empty();
    let then = if moving {
        slots(text, &picture.before, aw, ay, room, big)
    } else {
        Vec::new()
    };
    let m = picture.moved.clamp(0.0, 1.0) as f32;
    for (k, line) in picture.lines.iter().enumerate() {
        let slot = now[k];
        let rest = rgb(line_color(line.kind));
        let was = moving
            .then(|| {
                picture
                    .before
                    .iter()
                    .position(|b| b.id.is_some() && b.id == line.id)
            })
            .flatten();
        match (moving, was) {
            (false, _) => line_op(text, line, slot.px, slot, rest, faded(line.kind)),
            // From where it was to where it goes, its size and colour too;
            // its glyphs made at the larger size, scaled down.
            (true, Some(j)) => {
                let old = then[j];
                let at = Slot {
                    px: lerp(old.px, slot.px, m),
                    y: lerp(old.y, slot.y, m),
                };
                let kind = picture.before[j].kind;
                let color = mix(rgb(line_color(kind)), rest, m);
                let alpha = lerp(faded(kind), faded(line.kind), m);
                line_op(text, line, old.px.max(slot.px), at, color, alpha);
            }
            // New: rising into its place as it fades in.
            (true, None) => {
                let at = Slot {
                    y: slot.y + (1.0 - m) * gap(slot.px) * 0.5,
                    ..slot
                };
                line_op(text, line, slot.px, at, rest, faded(line.kind) * m);
            }
        }
    }
    // Lines that went: rising away as they fade out.
    if moving {
        for (j, gone) in picture.before.iter().enumerate() {
            let stays = gone.id.is_some() && picture.lines.iter().any(|l| l.id == gone.id);
            if !stays {
                let old = then[j];
                let at = Slot {
                    y: old.y - m * gap(old.px) * 0.5,
                    ..old
                };
                let rest = rgb(line_color(gone.kind));
                line_op(text, gone, old.px, at, rest, faded(gone.kind) * (1.0 - m));
            }
        }
    }
    // The score, top right: the points, the level under them.
    if let Some(score) = picture.score {
        let px = (big * 0.42).round().max(12.0);
        let points = crate::view::karaoke::points_text(score.points);
        let w = text.measure_from(&points, px, px);
        let margin = px * 0.8;
        ops.push(TextOp {
            text: points,
            base: px,
            px,
            at: (ax as f32 + aw as f32 - margin - w, ay as f32 + margin * 0.6),
            fill_x: f32::MAX,
            colors: (rgb(HIT), rgb(HIT)),
            alpha: 1.0,
            sung: None,
        });
        let small = (px * 0.45).round().max(10.0);
        // Under it the tier the last phrase reached, and the multiplier.
        let tier = score.tier.map(|t| t.name().to_owned());
        let times = (score.multiplier() > 1).then(|| format!("×{}", score.multiplier()));
        let under: Vec<String> = tier.into_iter().chain(times).collect();
        if !under.is_empty() {
            let times = under.join(" · ");
            let lw = text.measure_from(&times, small, small);
            ops.push(TextOp {
                text: times,
                base: small,
                px: small,
                at: (
                    ax as f32 + aw as f32 - margin - lw,
                    ay as f32 + margin * 0.6 + px * 1.25,
                ),
                fill_x: f32::MAX,
                colors: (rgb(colors::DIM), rgb(colors::DIM)),
                alpha: 1.0,
                sung: None,
            });
        }
    }
    // The count-in: its number at the top, above the lines and apart from
    // them (it never moves with them), shrinking and fading over its beat;
    // the one before the singing comes back after a break half as large.
    let count = picture
        .count
        .map(|c| (c, 0.26))
        .or(picture.cue.map(|c| ((c.number, c.through), 0.13)));
    if let Some(((n, through), size)) = count {
        // Smaller at once (ease-out), fading late, so it reads through
        // most of its beat.
        let t = through.clamp(0.0, 1.0) as f32;
        let shrink = 1.0 - (1.0 - t).powi(3);
        let most = room as f32 * size;
        let px = most * (1.0 - 0.4 * shrink);
        let base = base_size(px);
        let digit = n.to_string();
        let w = text.measure_from(&digit, base, px);
        ops.push(TextOp {
            text: digit,
            base,
            px,
            at: (
                ax as f32 + (aw as f32 - w) / 2.0,
                ay as f32 + room as f32 * 0.03 + (most - px) / 2.0,
            ),
            fill_x: f32::MAX,
            colors: (accent, accent),
            alpha: 1.0 - t * t * t,
            sung: None,
        });
    }
    (ops, over)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backdrop_fills_the_screen_shaded() {
        let cover = crate::art::sample_cover(64);
        let art = Art {
            id: 7,
            image: Arc::new(cover),
        };
        let b = Backdrop::new(Some(&art), 320, 180);
        assert_eq!((b.art, b.size, b.pixels.len()), (7, (320, 180), 320 * 180));
        // Shaded: nothing near white is left.
        assert!(b.pixels.iter().all(|&p| unpack(p).iter().all(|&v| v < 160)));
        // Blurred: neighbours differ little.
        let step = b
            .pixels
            .chunks(320)
            .flat_map(|row| row.windows(2))
            .map(|w| {
                let (a, c) = (unpack(w[0]), unpack(w[1]));
                (0..3).map(|i| a[i].abs_diff(c[i])).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        assert!(step < 40, "{step}");
        let plain = Backdrop::new(None, 10, 10);
        assert_eq!(plain.art, 0);
    }

    fn lyric(text: &str, fill: f64, kind: LineKind, id: usize) -> LyricLine {
        LyricLine {
            text: text.into(),
            fill,
            kind,
            id: Some(id),
            alpha: 1.0,
            notes: Vec::new(),
            hits: Vec::new(),
            liquid: Vec::new(),
            trail: Vec::new(),
            voice: None,
        }
    }

    fn picture(lines: Vec<LyricLine>, before: Vec<LyricLine>, moved: f64) -> KaraokePicture {
        KaraokePicture {
            area: ratatui::layout::Rect::new(0, 0, 80, 24),
            lines,
            count: None,
            cue: None,
            controls: None,
            start: None,
            before,
            moved,
            score: None,
        }
    }

    /// The count before the singing comes back sits at the top, the same
    /// whatever lines show or how they move: never part of them.
    #[test]
    fn the_break_count_stays_put_while_the_lines_move() {
        use crate::view::karaoke::Cue;
        let mut text = BigText::default();
        let cue = Some(Cue {
            number: 3,
            through: 0.2,
            line: 1,
        });
        let digit_at = |text: &mut BigText, p: KaraokePicture| {
            let ops = text_ops(text, &p, ((0, 0, 640, 360), 360), false).0;
            let digit: Vec<&TextOp> = ops.iter().filter(|o| o.text == "3").collect();
            assert_eq!(digit.len(), 1);
            let at = digit[0].at;
            assert!(at.1 < 360.0 * 0.2, "at the top: {at:?}");
            at
        };
        let pausing = vec![
            lyric("sung before", 1.0, LineKind::Previous, 0),
            LyricLine {
                text: "♪".into(),
                fill: 0.0,
                kind: LineKind::Info,
                id: None,
                alpha: 1.0,
                notes: Vec::new(),
                hits: Vec::new(),
                liquid: Vec::new(),
                trail: Vec::new(),
                voice: None,
            },
            lyric("coming back", 0.0, LineKind::Next, 1),
        ];
        let coming = vec![
            lyric("sung before", 1.0, LineKind::Previous, 0),
            lyric("coming back", 0.0, LineKind::Current, 1),
            lyric("after it", 0.0, LineKind::Next, 2),
        ];
        let a = digit_at(
            &mut text,
            KaraokePicture {
                cue,
                ..picture(pausing.clone(), Vec::new(), 1.0)
            },
        );
        let b = digit_at(
            &mut text,
            KaraokePicture {
                cue,
                ..picture(coming.clone(), pausing, 0.5)
            },
        );
        let c = digit_at(
            &mut text,
            KaraokePicture {
                cue,
                ..picture(coming, Vec::new(), 1.0)
            },
        );
        assert_eq!((a, b), (c, c));
    }

    /// A frame drawn over the last one, only where its text changed, is the
    /// frame drawn afresh, pixel for pixel: through a line filling, the
    /// lines scrolling up, and the count-in shrinking.
    #[test]
    fn a_frame_drawn_where_it_changed_is_the_frame_drawn_afresh() {
        let (w, h) = (640, 360);
        let art = Art {
            id: 3,
            image: Arc::new(crate::art::sample_cover(64)),
        };
        let backdrop = Backdrop::new(Some(&art), w, h);
        let area = (0, 0, w, h);
        let a = || lyric("The first line we sing", 1.0, LineKind::Previous, 0);
        let b = |fill| lyric("and the second one", fill, LineKind::Current, 1);
        let b_next = || lyric("and the second one", 0.0, LineKind::Next, 1);
        let c = || lyric("a third to come", 0.0, LineKind::Next, 2);
        let a_now = || lyric("The first line we sing", 0.4, LineKind::Current, 0);
        let mut frames = vec![
            picture(vec![a_now(), b_next(), c()], Vec::new(), 1.0),
            picture(
                vec![
                    lyric("The first line we sing", 0.7, LineKind::Current, 0),
                    b_next(),
                    c(),
                ],
                Vec::new(),
                1.0,
            ),
        ];
        for m in [0.1, 0.45, 0.8] {
            frames.push(picture(
                vec![a(), b(0.0), c()],
                vec![a_now(), b_next(), c()],
                m,
            ));
        }
        frames.push(picture(vec![a(), b(0.2), c()], Vec::new(), 1.0));
        frames.push(picture(vec![a(), b(0.5), c()], Vec::new(), 1.0));
        for through in [0.0, 0.3, 0.9] {
            let mut p = picture(vec![a(), b(0.5), c()], Vec::new(), 1.0);
            p.count = Some((2, through));
            frames.push(p);
        }
        frames.push(picture(vec![a(), b(0.5), c()], Vec::new(), 1.0));
        let mut text = BigText::default();
        let mut layer = Layer::default();
        let mut kept = vec![0u32; w * h];
        for (k, frame) in frames.iter().enumerate() {
            let mut target = Target {
                pixels: &mut kept,
                width: w,
                height: h,
            };
            let changed = paint(
                &mut target,
                &mut text,
                &mut layer,
                &backdrop,
                frame,
                area,
                None,
                k == 0,
            );
            if k > 0 {
                // Only part of the screen is redrawn and sent.
                let sent: usize = changed.iter().map(|r| r.2 * r.3).sum();
                assert!(sent < w * h * 3 / 4, "frame {k}: {sent} pixels");
            }
            let mut fresh = vec![0u32; w * h];
            let mut target = Target {
                pixels: &mut fresh,
                width: w,
                height: h,
            };
            paint(
                &mut target,
                &mut BigText::default(),
                &mut Layer::default(),
                &backdrop,
                frame,
                area,
                None,
                true,
            );
            let differ = kept.iter().zip(&fresh).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "frame {k}");
        }
    }

    /// While a line fills, only that line is redrawn.
    #[test]
    fn a_line_filling_redraws_only_itself() {
        let (w, h) = (640, 360);
        let backdrop = Backdrop::new(None, w, h);
        let area = (0, 0, w, h);
        let lines = |fill| {
            vec![
                lyric("before it", 1.0, LineKind::Previous, 0),
                lyric("the line sung now", fill, LineKind::Current, 1),
                lyric("after it", 0.0, LineKind::Next, 2),
            ]
        };
        let mut text = BigText::default();
        let mut layer = Layer::default();
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        paint(
            &mut target,
            &mut text,
            &mut layer,
            &backdrop,
            &picture(lines(0.2), Vec::new(), 1.0),
            area,
            None,
            true,
        );
        let changed = paint(
            &mut target,
            &mut text,
            &mut layer,
            &backdrop,
            &picture(lines(0.3), Vec::new(), 1.0),
            area,
            None,
            false,
        );
        assert_eq!(changed.len(), 1, "{changed:?}");
        let (_, _, rw, rh) = changed[0];
        assert!(rh < h / 4 && rw <= w, "{changed:?}");
        // Nothing changed: nothing redrawn.
        let again = paint(
            &mut target,
            &mut text,
            &mut layer,
            &backdrop,
            &picture(lines(0.3), Vec::new(), 1.0),
            area,
            None,
            false,
        );
        assert!(again.is_empty(), "{again:?}");
    }

    #[test]
    fn text_fills_to_its_split() {
        let mut text = BigText::default();
        let (w, h) = (400, 120);
        let mut pixels = vec![0u32; w * h];
        let mut target = Target {
            pixels: &mut pixels,
            width: w,
            height: h,
        };
        let px = 60.0;
        let width = text.measure("HHHH", px);
        assert!(width > 60.0);
        let split = 20.0 + width / 2.0;
        let (x, _, cw, _) = text
            .draw(
                &mut target,
                "HHHH",
                px,
                px,
                (20.0, 10.0),
                split,
                ([255, 0, 0], [0, 0, 255]),
                1.0,
                (0, 0, w, h),
            )
            .unwrap();
        assert!(x >= 20 && cw > 0);
        let reds = pixels.iter().filter(|&&p| unpack(p)[0] > 128).count();
        let blues = pixels.iter().filter(|&&p| unpack(p)[2] > 128).count();
        // About half each.
        assert!(reds > 0 && blues > 0);
        let share = reds as f64 / (reds + blues) as f64;
        assert!((0.35..0.65).contains(&share), "{share}");
    }
}
