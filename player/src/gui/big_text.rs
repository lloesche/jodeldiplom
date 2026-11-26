//! Text at any size in pixels ([`BigText`]): glyphs rasterised once per
//! character and size, drawn into the pixmap or made into coverage for the
//! GPU's atlas, in one of the window's faces ([`Face`]).
//!
//! Karaoke mode has faces of its own: Nunito, heavy, for the words
//! ([`Face::Words`]), Fredoka for its shouts and numbers ([`Face::Shout`],
//! its figures all as wide, so numbers that change do not jitter), and
//! M PLUS Rounded 1c for the Japanese (and the Chinese characters it has)
//! either lacks; the main font for the rest.

use std::collections::HashMap;
use std::sync::Arc;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, VariableFont};

use super::fonts;
use super::raster::{over, union, PixelRect, Target};

/// The pixels round text made for the atlas ([`BigText::coverage`]).
pub const COVER_PAD: usize = 2;
/// How tall a line of text is in the atlas, in its sizes (its coverage's
/// rows, [`BigText::coverage`], less its pad).
pub const LINE: f32 = 1.3;

/// A face text is drawn in, and the fonts it falls back on for what it
/// lacks (the main font, then the system's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Face {
    /// The main font (Sarasa Fixed J, else Cascadia Mono).
    #[default]
    Main,
    /// The lyrics font of the tabs (Atkinson Hyperlegible Next).
    Lyrics,
    /// Karaoke mode's words: Nunito at this weight (800 to 900), M PLUS
    /// Rounded 1c for Japanese.
    Words(u16),
    /// Karaoke mode's shouts and numbers: Fredoka at this weight (600 to
    /// 700), its figures tabular; M PLUS Rounded 1c for Japanese.
    Shout(u16),
}

impl Face {
    /// Karaoke mode's words in their bubbles: Nunito Black.
    pub const BUBBLE: Face = Face::Words(900);
    /// Karaoke mode's lines (the one sung, the next): Nunito ExtraBold.
    pub const LINE: Face = Face::Words(800);
    /// Karaoke mode's shouts (a tier, the multiplier): Fredoka Bold.
    pub const SHOUT: Face = Face::Shout(700);
    /// Karaoke mode's numbers (the score, the streak, the time left):
    /// Fredoka SemiBold.
    pub const NUMBER: Face = Face::Shout(600);

    /// Its own font, at its weight, where the binary has it.
    fn font(self) -> Option<FontRef<'static>> {
        let (data, weight) = match self {
            Face::Main => return FontRef::try_from_slice(fonts::main_font()).ok(),
            Face::Lyrics => (fonts::lyrics_font()?, None),
            Face::Words(w) => (fonts::words_font()?, Some(w)),
            Face::Shout(w) => (fonts::shout_font()?, Some(w)),
        };
        let mut font = FontRef::try_from_slice(data).ok()?;
        if let Some(w) = weight {
            font.set_variation(b"wght", f32::from(w));
        }
        Some(font)
    }

    /// The font it falls back on first for what its own lacks (Japanese).
    fn second(self) -> Option<FontRef<'static>> {
        match self {
            Face::Words(_) | Face::Shout(_) => FontRef::try_from_slice(fonts::rounded_font()?).ok(),
            Face::Main | Face::Lyrics => None,
        }
    }

    /// Whether its figures are all as wide.
    fn tabular(self) -> bool {
        matches!(self, Face::Shout(_))
    }

    /// The room added between its letters, a share of the size: the
    /// bubbles' black weight sets its letters close, small and moving.
    fn tracking(self) -> f32 {
        if self == Face::BUBBLE {
            0.06
        } else {
            0.0
        }
    }
}

/// Text in several faces, each made as it is first asked for.
#[derive(Default)]
pub struct Faces(HashMap<Face, BigText>);

impl Faces {
    /// Its text in `face`.
    pub fn get(&mut self, face: Face) -> &mut BigText {
        self.0.entry(face).or_insert_with(|| BigText::of(face))
    }
}

/// One glyph at one size: its coverage and where it sits from the pen.
struct Glyph {
    left: i32,
    top: i32,
    width: usize,
    height: usize,
    coverage: Vec<u8>,
    advance: f32,
}

/// Text at any size, glyphs kept per character and size, in its face
/// ([`Face`]).
#[derive(Default)]
pub struct BigText {
    glyphs: HashMap<(char, u32), Option<Arc<Glyph>>>,
    face: Face,
    /// Its own font and the one it falls back on first, read once.
    fonts: Option<(Option<FontRef<'static>>, Option<FontRef<'static>>)>,
}

impl BigText {
    /// Text in `face`.
    pub fn of(face: Face) -> BigText {
        BigText {
            face,
            ..BigText::default()
        }
    }

    /// Its face.
    pub fn face(&self) -> Face {
        self.face
    }

    fn main_font() -> Option<FontRef<'static>> {
        FontRef::try_from_slice(fonts::main_font()).ok()
    }

    /// The font of its face for `c`: its own, the one after it, the main
    /// font, the system's.
    fn font_for(&mut self, c: char) -> Option<FontRef<'static>> {
        let face = self.face;
        let (own, second) = self
            .fonts
            .get_or_insert_with(|| (face.font(), face.second()))
            .clone();
        let has = |f: &FontRef<'static>| f.glyph_id(c).0 != 0;
        own.filter(has)
            .or_else(|| second.filter(has))
            .or_else(|| Self::main_font().filter(has))
            .or_else(|| fonts::font_for(c))
            .or_else(Self::main_font)
    }

    /// How wide its figures all are at `scale` in `font` (their widest),
    /// for a tabular face.
    fn figure(font: &FontRef<'static>, scale: PxScale) -> f32 {
        let scaled = font.as_scaled(scale);
        ('0'..='9')
            .map(|d| scaled.h_advance(font.glyph_id(d)))
            .fold(0.0, f32::max)
    }

    fn glyph(&mut self, c: char, px: f32) -> Option<Arc<Glyph>> {
        let key = (c, px.round() as u32);
        if let Some(g) = self.glyphs.get(&key) {
            return g.clone();
        }
        let tabular = self.face.tabular() && c.is_ascii_digit();
        let font = self.font_for(c);
        let glyph = font.map(|font| {
            let scale = PxScale::from(px);
            let scaled = font.as_scaled(scale);
            let id = font.glyph_id(c);
            let own = scaled.h_advance(id);
            // A tabular figure: as wide as the widest, itself in the
            // middle.
            let (advance, shift) = match tabular {
                true => {
                    let wide = Self::figure(&font, scale);
                    (wide, ((wide - own) / 2.0).round() as i32)
                }
                false => (own, 0),
            };
            // The face's tracking, half on either side of the letter.
            let track = self.face.tracking() * px;
            let (advance, shift) = (advance + track, shift + (track / 2.0).round() as i32);
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
                        left: bounds.min.x as i32 + shift,
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

    /// The rows the ink of `chars` at `px` covers, from its glyphs as
    /// made: its top and bottom, down from the line's top (no ink: the
    /// line's, 0 to `px`).
    pub fn ink_band(&mut self, chars: &str, px: f32) -> (f32, f32) {
        let inked: Vec<Arc<Glyph>> = chars
            .chars()
            .filter_map(|c| self.glyph(c, px))
            .filter(|g| g.height > 0)
            .collect();
        let top = inked.iter().map(|g| g.top).min();
        let bottom = inked.iter().map(|g| g.top + g.height as i32).max();
        match (top, bottom) {
            (Some(top), Some(bottom)) => (top as f32, bottom as f32),
            _ => (0.0, px),
        }
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
        let h = (px * LINE).ceil().max(1.0) as usize + 2 * COVER_PAD;
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
    pub fn measure_from(&mut self, text: &str, base: f32, px: f32) -> f32 {
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
        clip: PixelRect,
    ) -> Option<PixelRect> {
        let (cx0, cy0) = (clip.0, clip.1);
        let cx1 = (clip.0 + clip.2).min(target.width);
        let cy1 = (clip.1 + clip.3).min(target.height);
        let scale = px / base.max(1.0);
        let exact = (scale - 1.0).abs() < 1e-3;
        let mut pen = x;
        let mut covered: Option<PixelRect> = None;
        let mut put = |target: &mut Target, tx: usize, ty: usize, a: f32| {
            let color = if (tx as f32) < fill_x {
                colors.0
            } else {
                colors.1
            };
            let i = ty * target.width + tx;
            target.pixels[i] = over(target.pixels[i], color, alpha * a);
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
