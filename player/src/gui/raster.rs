//! A ratatui backend that draws cells into a pixmap (`0x00RRGGBB` words,
//! the window buffer's own layout, [`super::pixels`]) with a TrueType
//! font, for the window.
//!
//! Glyphs come from the font through ab_glyph, with the baseline at the
//! font's ascent so every row lines up. Box-drawing and block characters
//! are drawn as shapes instead, so borders join exactly between cells and
//! bars fill them completely at any size. Bold is the regular face drawn
//! twice, a pixel apart (the embedded font has one face).
//! Characters the embedded font has no glyph for (Japanese, Chinese and
//! Korean text, symbols) come from a system font ([`super::fonts`]),
//! centred in their one or two cells.
//!
//! Like a terminal, it redraws only the cells ratatui reports as changed,
//! and keeps which ones it redrew ([`PixelBackend::damage`]) so pictures
//! over unchanged cells need not be painted again.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use ratatui::backend::{Backend, ClearType, WindowSize};
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Position, Rect, Size};
use ratatui::style::{Color, Modifier};

use super::fonts;
use super::pixels::{pack, Target};

/// One rasterised glyph: coverage relative to the cell's top-left corner.
struct Glyph {
    left: i32,
    top: i32,
    width: usize,
    height: usize,
    coverage: Vec<u8>,
}

/// The cells redrawn since [`PixelBackend::reset_damage`].
#[derive(Clone, Debug, Default)]
pub struct Damage {
    /// Every cell was redrawn.
    pub all: bool,
    /// The cells redrawn, when not all.
    pub cells: Vec<(u16, u16)>,
}

impl Damage {
    /// Whether any cell in `area` was redrawn.
    pub fn touches(&self, area: Rect) -> bool {
        self.all
            || self
                .cells
                .iter()
                .any(|&(x, y)| area.contains(Position::new(x, y)))
    }
}

/// The cells' font at the cells' size: the cell it makes, and its glyphs,
/// each rasterised once.
pub struct CellFont {
    font: FontRef<'static>,
    em: f32,
    scale: PxScale,
    ascent: f32,
    cell_w: usize,
    cell_h: usize,
    glyphs: HashMap<(char, bool), Option<Arc<Glyph>>>,
}

/// The pixels and the cell grid.
pub struct PixelBackend {
    buffer: Buffer,
    font: CellFont,
    cursor: Position,
    /// `0x00RRGGBB` words ([`pack`]), row by row.
    pub pixmap: Vec<u32>,
    /// The cells redrawn since the last [`PixelBackend::reset_damage`].
    pub damage: Damage,
    /// The pixmap's width in pixels.
    pub width: usize,
    /// The pixmap's height in pixels.
    pub height: usize,
    /// The colour of `Color::Reset` text.
    pub default_fg: [u8; 3],
    /// The colour of a `Color::Reset` background.
    pub default_bg: [u8; 3],
}

/// The coverage of `glyph` in `font`, widened by `extra` pixels (bold).
fn rasterize<F: Font>(font: &F, glyph: ab_glyph::Glyph, extra: usize) -> Option<Arc<Glyph>> {
    font.outline_glyph(glyph).map(|outline| {
        let bounds = outline.px_bounds();
        let width = bounds.width() as usize + extra;
        let height = bounds.height() as usize;
        let mut coverage = vec![0u8; width * height];
        outline.draw(|x, y, c| {
            let v = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
            let (x, y) = (x as usize, y as usize);
            for dx in 0..=extra {
                if x + dx < width && y < height {
                    let slot = &mut coverage[y * width + x + dx];
                    *slot = (*slot).max(v);
                }
            }
        });
        Arc::new(Glyph {
            left: bounds.min.x as i32,
            top: bounds.min.y as i32,
            width,
            height,
            coverage,
        })
    })
}

fn named(color: Color) -> Option<[u8; 3]> {
    Some(match color {
        Color::Black => [0, 0, 0],
        Color::Red => [205, 49, 49],
        Color::Green => [13, 188, 121],
        Color::Yellow => [229, 229, 16],
        Color::Blue => [36, 114, 200],
        Color::Magenta => [188, 63, 188],
        Color::Cyan => [17, 168, 205],
        Color::Gray => [204, 204, 204],
        Color::DarkGray => [118, 118, 118],
        Color::LightRed => [241, 76, 76],
        Color::LightGreen => [35, 209, 139],
        Color::LightYellow => [245, 245, 67],
        Color::LightBlue => [59, 142, 234],
        Color::LightMagenta => [214, 112, 214],
        Color::LightCyan => [41, 184, 219],
        Color::White => [242, 242, 242],
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Indexed(i) => {
            // The 6 × 6 × 6 cube and the grey ramp; the first 16 as above.
            match i {
                16..=231 => {
                    let i = i - 16;
                    let level = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
                    [level(i / 36), level(i / 6 % 6), level(i % 6)]
                }
                232..=255 => {
                    let v = 8 + (i - 232) * 10;
                    [v, v, v]
                }
                _ => [128, 128, 128],
            }
        }
        Color::Reset => return None,
    })
}

fn blend(fg: [u8; 3], bg: [u8; 3], alpha: u8) -> [u8; 3] {
    let a = u32::from(alpha);
    [0, 1, 2].map(|i| ((u32::from(fg[i]) * a + u32::from(bg[i]) * (255 - a) + 127) / 255) as u8)
}

/// Lines out of a box-drawing character's centre: (left, right, up, down),
/// 0 none, 1 light, 2 heavy; and whether the corner is rounded.
fn box_lines(c: char) -> Option<([u8; 4], bool)> {
    Some(match c {
        '─' => ([1, 1, 0, 0], false),
        '━' => ([2, 2, 0, 0], false),
        '│' => ([0, 0, 1, 1], false),
        '┃' => ([0, 0, 2, 2], false),
        '┌' => ([0, 1, 0, 1], false),
        '┐' => ([1, 0, 0, 1], false),
        '└' => ([0, 1, 1, 0], false),
        '┘' => ([1, 0, 1, 0], false),
        '├' => ([0, 1, 1, 1], false),
        '┤' => ([1, 0, 1, 1], false),
        '┬' => ([1, 1, 0, 1], false),
        '┴' => ([1, 1, 1, 0], false),
        '┼' => ([1, 1, 1, 1], false),
        '╭' => ([0, 1, 0, 1], true),
        '╮' => ([1, 0, 0, 1], true),
        '╰' => ([0, 1, 1, 0], true),
        '╯' => ([1, 0, 1, 0], true),
        _ => return None,
    })
}

/// A block element as a filled fraction of the cell: (x0, y0, x1, y1) in
/// eighths, and its opacity.
fn block(c: char) -> Option<((u8, u8, u8, u8), u8)> {
    let lower = |n: u8| ((0, 8 - n, 8, 8), 255);
    Some(match c {
        '█' => ((0, 0, 8, 8), 255),
        '▀' => ((0, 0, 8, 4), 255),
        '▁' => lower(1),
        '▂' => lower(2),
        '▃' => lower(3),
        '▄' => lower(4),
        '▅' => lower(5),
        '▆' => lower(6),
        '▇' => lower(7),
        '▌' => ((0, 0, 4, 8), 255),
        '▐' => ((4, 0, 8, 8), 255),
        '▏' => ((0, 0, 1, 8), 255),
        '▕' => ((7, 0, 8, 8), 255),
        '░' => ((0, 0, 8, 8), 64),
        '▒' => ((0, 0, 8, 8), 128),
        '▓' => ((0, 0, 8, 8), 192),
        _ => return None,
    })
}

impl CellFont {
    /// `font`, sized by [`CellFont::set_em`].
    fn new(font: &'static [u8]) -> CellFont {
        CellFont {
            font: FontRef::try_from_slice(font).expect("the embedded font parses"),
            em: 0.0,
            scale: PxScale::from(1.0),
            ascent: 0.0,
            cell_w: 1,
            cell_h: 1,
            glyphs: HashMap::new(),
        }
    }

    /// The font at an em of `em_px` pixels (its glyphs made afresh);
    /// whether that changed its size.
    fn set_em(&mut self, em_px: f32) -> bool {
        let em_px = em_px.max(4.0);
        if (em_px - self.em).abs() < f32::EPSILON {
            return false;
        }
        self.em = em_px;
        let units = self.font.units_per_em().unwrap_or(1000.0);
        let height = self.font.ascent_unscaled() - self.font.descent_unscaled();
        self.scale = PxScale::from(em_px * height / units);
        let scaled = self.font.as_scaled(self.scale);
        self.ascent = scaled.ascent().round();
        self.cell_h = (scaled.ascent() - scaled.descent() + scaled.line_gap())
            .ceil()
            .max(1.0) as usize;
        self.cell_w = scaled.h_advance(self.font.glyph_id('M')).round().max(1.0) as usize;
        self.glyphs.clear();
        true
    }

    /// The cell size in pixels.
    pub fn cell(&self) -> (usize, usize) {
        (self.cell_w, self.cell_h)
    }

    fn glyph(&mut self, c: char, bold: bool) -> Option<Arc<Glyph>> {
        if let Some(found) = self.glyphs.get(&(c, bold)) {
            return found.clone();
        }
        let extra = if bold { self.bold_shift() } else { 0 };
        let id = self.font.glyph_id(c);
        let glyph = if id.0 != 0 || c == ' ' {
            let at = ab_glyph::point(0.0, self.ascent);
            rasterize(
                &self.font,
                id.with_scale_and_position(self.scale, at),
                extra,
            )
        } else if let Some(font) = fonts::font_for(c) {
            // Another font's glyph at the same em, on the same baseline,
            // centred in its cells (one or two) and narrowed to fit them.
            let cells = unicode_width::UnicodeWidthChar::width(c)
                .unwrap_or(1)
                .clamp(1, 2);
            let room = (self.cell_w * cells) as f32;
            let units = font.units_per_em().unwrap_or(1000.0);
            let height = font.ascent_unscaled() - font.descent_unscaled();
            let mut scale = PxScale::from(self.em * height / units);
            let id = font.glyph_id(c);
            let mut advance = font.as_scaled(scale).h_advance(id);
            if advance > room {
                scale = PxScale::from(scale.y * room / advance);
                advance = room;
            }
            let at = ab_glyph::point(((room - advance) / 2.0).max(0.0), self.ascent);
            rasterize(&font, id.with_scale_and_position(scale, at), extra)
        } else {
            None
        };
        self.glyphs.insert((c, bold), glyph.clone());
        glyph
    }

    fn bold_shift(&self) -> usize {
        (self.em / 24.0).round().max(1.0) as usize
    }

    fn thickness(&self, weight: u8) -> usize {
        let light = (self.em / 14.0).round().max(1.0) as usize;
        if weight >= 2 {
            light * 2 + 1
        } else {
            light
        }
    }
}

impl PixelBackend {
    /// A grid of `cols` × `rows` cells with `font` at an em size of
    /// `em_px` pixels.
    pub fn new(cols: u16, rows: u16, font: &'static [u8], em_px: f32) -> PixelBackend {
        let mut backend = PixelBackend {
            buffer: Buffer::empty(Rect::new(0, 0, cols.max(1), rows.max(1))),
            font: CellFont::new(font),
            cursor: Position::new(0, 0),
            pixmap: Vec::new(),
            damage: Damage::default(),
            width: 0,
            height: 0,
            default_fg: [220, 220, 225],
            default_bg: [17, 17, 21],
        };
        backend.set_em(em_px);
        backend
    }

    /// The cell size in pixels.
    pub fn cell(&self) -> (usize, usize) {
        self.font.cell()
    }

    /// The cells as last drawn.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The pixmap as a target to paint into.
    pub fn target(&mut self) -> Target<'_> {
        Target {
            pixels: &mut self.pixmap,
            width: self.width,
            height: self.height,
        }
    }

    /// Changes the font size (an em of `em_px` pixels) and redraws.
    pub fn set_em(&mut self, em_px: f32) {
        if self.font.set_em(em_px) {
            self.alloc();
        }
    }

    /// Changes the grid and redraws.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.buffer
            .resize(Rect::new(0, 0, cols.max(1), rows.max(1)));
        self.alloc();
    }

    fn alloc(&mut self) {
        let area = self.buffer.area;
        let (cell_w, cell_h) = self.cell();
        self.width = usize::from(area.width) * cell_w;
        self.height = usize::from(area.height) * cell_h;
        self.pixmap = vec![0; self.width * self.height];
        self.redraw();
    }

    /// Draws every cell again.
    pub fn redraw(&mut self) {
        let area = self.buffer.area;
        for y in 0..area.height {
            for x in 0..area.width {
                self.draw_cell(x, y);
            }
        }
        self.damage.all = true;
    }

    /// Starts counting the cells redrawn afresh.
    pub fn reset_damage(&mut self) {
        self.damage.all = false;
        self.damage.cells.clear();
    }

    fn put_over(&mut self, x: usize, y: usize, color: [u8; 3], alpha: u8) {
        self.target().put_over(x, y, color, alpha);
    }

    fn fill(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]) {
        self.target().fill(x0, y0, x1, y1, color);
    }

    fn colors(&self, cell: &Cell) -> ([u8; 3], [u8; 3]) {
        let mut fg = named(cell.fg).unwrap_or(self.default_fg);
        let mut bg = named(cell.bg).unwrap_or(self.default_bg);
        let m = cell.modifier;
        if m.contains(Modifier::REVERSED) {
            std::mem::swap(&mut fg, &mut bg);
        }
        if m.contains(Modifier::DIM) {
            fg = blend(fg, bg, 140);
        }
        if m.contains(Modifier::HIDDEN) {
            fg = bg;
        }
        (fg, bg)
    }

    fn draw_box(&mut self, ox: usize, oy: usize, lines: [u8; 4], rounded: bool, fg: [u8; 3]) {
        let (w, h) = self.cell();
        let (cx, cy) = (w / 2, h / 2);
        let [left, right, up, down] = lines;
        let span = |center: usize, weight: u8, this: &Self| {
            let t = this.font.thickness(weight);
            let start = center.saturating_sub(t / 2);
            (start, start + t)
        };
        if rounded {
            // A quarter circle joining the two lines, then the straight
            // rest of each line to the cell's edges.
            let t = self.font.thickness(1) as f32;
            let r = (w.min(h) as f32 / 2.0).floor().max(1.0);
            let (dir_x, dir_y): (f32, f32) = (
                if right > 0 { 1.0 } else { -1.0 },
                if down > 0 { 1.0 } else { -1.0 },
            );
            let (mx, my) = (cx as f32 + 0.5 * (t % 2.0), cy as f32 + 0.5 * (t % 2.0));
            let (acx, acy) = (mx + dir_x * r, my + dir_y * r);
            for y in 0..h {
                for x in 0..w {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let inside_x = (px - acx) * dir_x <= 0.0;
                    let inside_y = (py - acy) * dir_y <= 0.0;
                    let alpha = if inside_x && inside_y {
                        let d = ((px - acx).powi(2) + (py - acy).powi(2)).sqrt();
                        (t / 2.0 + 0.5 - (d - r).abs()).clamp(0.0, 1.0)
                    } else if !inside_x && (py - my).abs() < t / 2.0 + 0.01 {
                        // The horizontal line beyond the arc.
                        1.0
                    } else if !inside_y && (px - mx).abs() < t / 2.0 + 0.01 {
                        1.0
                    } else {
                        0.0
                    };
                    if alpha > 0.0 {
                        self.put_over(ox + x, oy + y, fg, (alpha * 255.0) as u8);
                    }
                }
            }
            return;
        }
        let horizontal = left.max(right);
        let vertical = up.max(down);
        let (vy0, vy1) = span(cy, horizontal.max(1), self);
        let (hx0, hx1) = span(cx, vertical.max(1), self);
        if left > 0 {
            let (y0, y1) = span(cy, left, self);
            self.fill(ox, oy + y0, ox + hx1, oy + y1, fg);
        }
        if right > 0 {
            let (y0, y1) = span(cy, right, self);
            self.fill(ox + hx0, oy + y0, ox + w, oy + y1, fg);
        }
        if up > 0 {
            let (x0, x1) = span(cx, up, self);
            self.fill(ox + x0, oy, ox + x1, oy + vy1, fg);
        }
        if down > 0 {
            let (x0, x1) = span(cx, down, self);
            self.fill(ox + x0, oy + vy0, ox + x1, oy + h, fg);
        }
    }

    /// Draws the text of the cells in `area` again over the pixels there,
    /// without their background (text over a picture).
    pub fn overlay(&mut self, area: Rect) {
        let area = area.intersection(self.buffer.area);
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                self.paint_cell(x, y, false);
            }
        }
    }

    fn draw_cell(&mut self, x: u16, y: u16) {
        self.paint_cell(x, y, true);
    }

    fn paint_cell(&mut self, x: u16, y: u16, background: bool) {
        let cell = self.buffer[(x, y)].clone();
        let symbol = cell.symbol();
        let (fg, bg) = self.colors(&cell);
        let (w, h) = self.cell();
        let (ox, oy) = (usize::from(x) * w, usize::from(y) * h);
        // The second half of a wide character, drawn with the first (ratatui
        // leaves it empty or a space).
        let covered =
            x > 0 && unicode_width::UnicodeWidthStr::width(self.buffer[(x - 1, y)].symbol()) >= 2;
        if symbol.is_empty() || covered {
            return;
        }
        let wide = unicode_width::UnicodeWidthStr::width(symbol).clamp(1, 2);
        if background {
            self.fill(ox, oy, ox + w * wide, oy + h, bg);
        }
        let c = symbol.chars().next().unwrap_or(' ');
        if let Some((lines, rounded)) = box_lines(c) {
            self.draw_box(ox, oy, lines, rounded, fg);
        } else if let Some(((x0, y0, x1, y1), alpha)) = block(c) {
            let px = |e: u8| usize::from(e) * w / 8;
            let py = |e: u8| usize::from(e) * h / 8;
            let color = blend(fg, bg, alpha);
            self.fill(ox + px(x0), oy + py(y0), ox + px(x1), oy + py(y1), color);
        } else if c != ' ' {
            let bold = cell.modifier.contains(Modifier::BOLD);
            match self.font.glyph(c, bold) {
                Some(glyph) => {
                    for gy in 0..glyph.height {
                        let py = glyph.top + gy as i32;
                        if py < 0 || py as usize >= h {
                            continue;
                        }
                        for gx in 0..glyph.width {
                            let px = glyph.left + gx as i32;
                            if px < 0 || px as usize >= w * wide {
                                continue;
                            }
                            let a = glyph.coverage[gy * glyph.width + gx];
                            if a > 0 {
                                let (tx, ty) = (ox + px as usize, oy + py as usize);
                                self.put_over(tx, ty, fg, a);
                            }
                        }
                    }
                }
                None => {
                    // No glyph in the font: a small hollow box.
                    let (x0, x1) = (ox + w / 5, ox + w - w / 5);
                    let (y0, y1) = (oy + h / 4, oy + h - h / 4);
                    self.fill(x0, y0, x1, y0 + 1, fg);
                    self.fill(x0, y1 - 1, x1, y1, fg);
                    self.fill(x0, y0, x0 + 1, y1, fg);
                    self.fill(x1 - 1, y0, x1, y1, fg);
                }
            }
        }
        let t = self.font.thickness(1);
        if cell.modifier.contains(Modifier::UNDERLINED) {
            let uy = (oy + self.font.ascent as usize + t + 1).min(oy + h - t);
            self.fill(ox, uy, ox + w * wide, uy + t, fg);
        }
        if cell.modifier.contains(Modifier::CROSSED_OUT) {
            let sy = oy + h / 2;
            self.fill(ox, sy, ox + w * wide, sy + t, fg);
        }
    }
}

impl Backend for PixelBackend {
    /// Drawing into memory cannot fail.
    type Error = Infallible;

    fn draw<'a, I>(&mut self, content: I) -> Result<(), Infallible>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut changed = Vec::new();
        for (x, y, cell) in content {
            if x < self.buffer.area.width && y < self.buffer.area.height {
                self.buffer[(x, y)] = cell.clone();
                changed.push((x, y));
            }
        }
        for &(x, y) in &changed {
            self.draw_cell(x, y);
        }
        self.damage.cells.extend(changed);
        Ok(())
    }

    fn hide_cursor(&mut self) -> Result<(), Infallible> {
        Ok(())
    }

    fn show_cursor(&mut self) -> Result<(), Infallible> {
        Ok(())
    }

    fn get_cursor_position(&mut self) -> Result<Position, Infallible> {
        Ok(self.cursor)
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> Result<(), Infallible> {
        self.cursor = position.into();
        Ok(())
    }

    fn clear(&mut self) -> Result<(), Infallible> {
        self.buffer.reset();
        self.pixmap.fill(pack(self.default_bg));
        self.damage.all = true;
        Ok(())
    }

    fn clear_region(&mut self, clear_type: ClearType) -> Result<(), Infallible> {
        let area = self.buffer.area;
        let x = self.cursor.x.min(area.width - 1);
        let y = self.cursor.y.min(area.height - 1);
        let at = |x: u16, y: u16| usize::from(y) * usize::from(area.width) + usize::from(x);
        let cells = match clear_type {
            ClearType::All => return self.clear(),
            ClearType::AfterCursor => at(x, y)..self.buffer.content.len(),
            ClearType::BeforeCursor => 0..at(x, y) + 1,
            ClearType::CurrentLine => at(0, y)..at(area.width - 1, y) + 1,
            ClearType::UntilNewLine => at(x, y)..at(area.width - 1, y) + 1,
        };
        for i in cells {
            self.buffer.content[i].reset();
            let (cx, cy) = (i % usize::from(area.width), i / usize::from(area.width));
            self.draw_cell(cx as u16, cy as u16);
            self.damage.cells.push((cx as u16, cy as u16));
        }
        Ok(())
    }

    fn size(&self) -> Result<Size, Infallible> {
        Ok(self.buffer.area.as_size())
    }

    fn window_size(&mut self) -> Result<WindowSize, Infallible> {
        Ok(WindowSize {
            columns_rows: self.buffer.area.as_size(),
            pixels: Size::new(self.width as u16, self.height as u16),
        })
    }

    fn flush(&mut self) -> Result<(), Infallible> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::fonts::FONT;
    use crate::gui::pixels::unpack;
    use ratatui::style::Style;

    fn backend() -> PixelBackend {
        PixelBackend::new(10, 3, FONT, 15.0)
    }

    #[test]
    fn cells_have_a_terminal_like_shape() {
        let b = backend();
        let (w, h) = b.cell();
        assert!((8..=10).contains(&w) && (16..=20).contains(&h), "{w}x{h}");
        assert_eq!((b.width, b.height), (10 * w, 3 * h));
        let big = PixelBackend::new(10, 3, FONT, 30.0);
        assert!(big.cell().1 >= 2 * h - 1);
    }

    #[test]
    fn clearing_to_the_end_of_a_line_resets_and_repaints_those_cells() {
        let mut b = backend();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 10, 3));
        for y in 0..3 {
            buffer.set_string(0, y, "██████████", Style::new());
        }
        b.draw(
            buffer
                .content()
                .iter()
                .enumerate()
                .map(|(i, c)| ((i % 10) as u16, (i / 10) as u16, c)),
        )
        .unwrap();
        b.set_cursor_position((4, 1)).unwrap();
        b.clear_region(ClearType::UntilNewLine).unwrap();
        // Whether the centre of a cell shows ink.
        fn lit(b: &PixelBackend, col: usize, row: usize) -> bool {
            let (w, h) = b.cell();
            unpack(b.pixmap[(row * h + h / 2) * b.width + col * w + w / 2]) != b.default_bg
        }
        assert!((0..4).all(|col| lit(&b, col, 1)));
        assert!((4..10).all(|col| !lit(&b, col, 1) && b.buffer()[(col as u16, 1)].symbol() == " "));
        assert!((0..10).all(|col| lit(&b, col, 0) && lit(&b, col, 2)));
        b.clear_region(ClearType::CurrentLine).unwrap();
        assert!((0..10).all(|col| !lit(&b, col, 1)));
        assert!(lit(&b, 0, 0) && lit(&b, 9, 2));
    }

    #[test]
    fn vertical_lines_join_between_rows_and_text_has_ink() {
        let mut b = backend();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 10, 3));
        buffer[(0, 0)].set_symbol("│");
        buffer[(0, 1)].set_symbol("│");
        buffer.set_string(2, 1, "Hg", Style::new().fg(Color::Rgb(255, 255, 255)));
        let cells: Vec<(u16, u16, Cell)> = buffer
            .content()
            .iter()
            .enumerate()
            .map(|(i, c)| ((i % 10) as u16, (i / 10) as u16, c.clone()))
            .collect();
        b.draw(cells.iter().map(|(x, y, c)| (*x, *y, c))).unwrap();
        let (w, h) = b.cell();
        // The line is lit on every pixel row of both cells.
        for y in 0..2 * h {
            let lit = (0..w).any(|x| unpack(b.pixmap[y * b.width + x]) != b.default_bg);
            assert!(lit, "gap at row {y}");
        }
        // The text has ink inside its cells.
        let ink = (2 * w..4 * w)
            .flat_map(|x| (h..2 * h).map(move |y| (x, y)))
            .filter(|&(x, y)| unpack(b.pixmap[y * b.width + x]) != b.default_bg)
            .count();
        assert!(ink > w * h / 4, "{ink}");
    }

    /// A character the embedded font lacks (here katakana) is drawn from a
    /// system font when one has it: ink across its two cells, not the
    /// hollow box. Skipped where no system font has it.
    #[test]
    fn missing_glyphs_come_from_a_system_font() {
        if super::fonts::font_for('ア').is_none() {
            eprintln!("no system font with katakana; skipped");
            return;
        }
        let mut b = backend();
        let mut buffer = Buffer::empty(Rect::new(0, 0, 10, 3));
        buffer.set_string(0, 1, "アス", Style::new());
        b.draw(
            buffer
                .content()
                .iter()
                .enumerate()
                .map(|(i, c)| ((i % 10) as u16, (i / 10) as u16, c)),
        )
        .unwrap();
        let glyph = b.font.glyph('ア', false).expect("a glyph");
        let (w, _) = b.cell();
        assert!(glyph.width > w, "spans two cells: {} > {w}", glyph.width);
        assert!(glyph.left >= 0 && glyph.left as usize + glyph.width <= 2 * w + 1);
    }
}
