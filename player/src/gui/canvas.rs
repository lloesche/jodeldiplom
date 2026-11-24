//! The window's drawing surface without the window: the shared ratatui
//! widgets rasterised into an RGB pixmap ([`PixelBackend`]), and the cover
//! and spectrum painted into it at their cell rects. Used by the window
//! and, headless, by the tests and the screenshots.

use std::sync::Arc;

use crate::gui::raster::{pack, unpack, PixelBackend};
use image::RgbImage;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::{Frame, Terminal};

use crate::art::{self, Art};
use crate::core::layout::Lane;
use crate::view::karaoke::KaraokePicture;
use crate::view::lanes::WavePlot;
use crate::view::plot::{self, Plot};
use crate::view::transition::RulerPlot;
use crate::view::{self, colors, Pictures, Scene};
use jodel_rhythm::waveform::Waveform;

/// The embedded font: Cascadia Mono, SIL Open Font License 1.1
/// (`assets/CascadiaMono-OFL.txt`); the main font when the player ships
/// without Sarasa Fixed ([`super::fonts`]).
pub static FONT: &[u8] = include_bytes!("../../assets/CascadiaMono-Regular.ttf");

/// The font size at a scale factor of 1, in pixels.
pub const BASE_FONT_PX: f64 = 15.0;

/// The font size (em) in pixels for a window scale factor (HiDPI).
pub fn font_px(scale: f64) -> f32 {
    (BASE_FONT_PX * scale).round().clamp(8.0, 96.0) as f32
}

#[derive(Clone)]
enum Paint {
    /// A cover, and whether it starts under a line through its first row.
    Cover(Rect, bool),
    Spectrum(Plot),
    /// Text over a cover: the cover's rect and the text's.
    Caption(Rect, Rect),
    /// A lane of the Transition tab.
    Wave(Box<WavePlot>),
    /// A Transition block's bar numbers.
    Ruler(Box<RulerPlot>),
    /// Karaoke mode's picture.
    Karaoke(Box<KaraokePicture>),
}

impl Paint {
    fn rect(&self) -> Rect {
        match self {
            Paint::Cover(r, _) | Paint::Caption(_, r) => *r,
            Paint::Spectrum(plot) => plot.area,
            Paint::Wave(plot) => plot.area,
            Paint::Ruler(plot) => plot.area,
            Paint::Karaoke(picture) => picture.area,
        }
    }
}

/// Collects the pictures while the widgets draw; their cells stay empty.
struct PixelPictures {
    has_art: bool,
    paints: Vec<Paint>,
}

impl Pictures for PixelPictures {
    fn cover(&mut self, frame: &mut Frame, area: Rect, below_line: bool) {
        if self.has_art {
            view::blank(frame, area);
            self.paints.push(Paint::Cover(area, below_line));
        } else {
            view::placeholder(frame, area);
        }
    }

    fn spectrum(&mut self, frame: &mut Frame, plot: &Plot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Spectrum(plot.clone()));
    }

    fn wave(&mut self, frame: &mut Frame, plot: &WavePlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Wave(Box::new(plot.clone())));
    }

    fn ruler(&mut self, frame: &mut Frame, plot: &RulerPlot) {
        view::blank(frame, plot.area);
        self.paints.push(Paint::Ruler(Box::new(plot.clone())));
    }

    fn karaoke(&mut self, frame: &mut Frame, picture: &KaraokePicture) {
        view::blank(frame, picture.area);
        self.paints.push(Paint::Karaoke(Box::new(picture.clone())));
    }

    fn text_over_cover(&self) -> bool {
        self.has_art
    }

    fn caption(&mut self, cover: Rect, caption: Rect) {
        self.paints.push(Paint::Caption(cover, caption));
    }
}

fn rgb(color: Color) -> [u8; 3] {
    match color {
        Color::Rgb(r, g, b) => [r, g, b],
        _ => [128, 128, 128],
    }
}

fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (f32::from(a[i]) * (1.0 - t) + f32::from(b[i]) * t).round() as u8)
}

/// A cover scaled for one rect: the art id, the rect's pixel size, the
/// scaled image and its offset in the rect.
type FittedCover = (u64, u32, u32, Arc<(RgbImage, u32, u32)>);

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
    /// Karaoke mode's backdrop, its text, and the picture in the pixmap now.
    backdrop: Option<super::karaoke::Backdrop>,
    big_text: super::karaoke::BigText,
    karaoke: Option<KaraokePicture>,
    /// What the picture in the pixmap drew over the backdrop.
    karaoke_layer: super::karaoke::Layer,
    /// Karaoke mode's stage, drawn by the GPU over the pixmap
    /// ([`Canvas::stage_frame`]), and the picture it draws from (all its
    /// lines) with where, while it shows.
    stage: super::stage::Stage,
    staged: Option<(KaraokePicture, PixelRect)>,
    /// How long the last frame took.
    pub last_draw: DrawTimes,
    /// Whether the last [`Canvas::draw`] changed a pixel: a window that
    /// shows the pixmap already needs no new frame where it did not.
    pub changed: bool,
    /// The pixels changed since [`Canvas::take_changes`].
    changes: Changes,
    /// Lanes that follow the playhead go into a layer the window scrolls
    /// itself ([`Canvas::set_layers`]), rather than into the pixmap.
    layers: bool,
    /// That layer, while such a lane shows.
    strip: Option<Strip>,
    /// The range of the last melody drawn, by its waveform.
    melody_range: Option<(usize, Option<(u8, u8)>)>,
    /// The last melody drawn with what is not sung taken out, by its
    /// waveform and the lyrics whose words it was kept to.
    melody_clean: Option<(usize, usize, Arc<jodel_rhythm::waveform::Melody>)>,
}

/// The width of a layer's tiles, in pixels.
pub const TILE: usize = 512;

/// The most tile slots a layer has (the window's shader holds this many).
pub const MOST_SLOTS: usize = 16;

/// What follows the playhead together ([`WavePlot::follow`],
/// [`RulerPlot::follow`]): the Karaoke tab's voice lane, or the playing
/// track's block on the Transition tab (its bar numbers and lanes, stacked), on
/// one grid (column `n` is `n × px` seconds in). Painted in tiles of
/// [`TILE`] columns, each once, for the window to scroll through by itself:
/// a tile is painted again only when what is in it changes (the word heard,
/// a mark: the cursor, a selection, the cue, the fades), all of them when
/// the parts do (another track, zoom, size).
struct Strip {
    /// The parts as last drawn (the lanes' playhead set per tile), each at
    /// its rows of pixels from the top.
    parts: Vec<Part>,
    /// Its pixels in the window, and the grid's seconds a column.
    rect: PixelRect,
    px: f64,
    /// The playhead's share of the way in, and the span shown.
    lead: f64,
    span: f64,
    /// Its tile slots: the tile each holds, if any.
    slots: Vec<Option<i64>>,
    /// Each word's times and the columns it takes (its outline in), in
    /// the order sung.
    words: Vec<(f64, f64, i64, i64)>,
    /// The word heard when the tiles were painted.
    heard: Option<usize>,
    /// Tiles to paint again (a mark in them changed).
    stale: Vec<i64>,
}

/// A part of a [`Strip`] and its rows of pixels: from, how many.
#[derive(Clone)]
enum Part {
    Wave(Box<WavePlot>, usize, usize),
    Ruler(Box<RulerPlot>, usize, usize),
}

impl Part {
    /// Whether `self` and `other` paint the same tiles but for their marks
    /// (another track, zoom, size, colours, bars: all of them afresh).
    fn same_ground(&self, other: &Part) -> bool {
        match (self, other) {
            (Part::Wave(a, ay, ah), Part::Wave(b, by, bh)) => {
                (ay, ah) == (by, bh)
                    && Arc::ptr_eq(&a.waveform, &b.waveform)
                    && match (&a.lyrics, &b.lyrics) {
                        (Some(x), Some(y)) => Arc::ptr_eq(x, y),
                        (None, None) => true,
                        _ => false,
                    }
                    && (a.lane, a.color, a.rows, a.text_scale, a.lines, a.follow)
                        == (b.lane, b.color, b.rows, b.text_scale, b.lines, b.follow)
                    && (a.deep, a.key) == (b.deep, b.key)
                    && a.downbeats == b.downbeats
                    && a.unclear == b.unclear
            }
            (Part::Ruler(a, ay, ah), Part::Ruler(b, by, bh)) => {
                (ay, ah) == (by, bh) && a.bars == b.bars && a.follow == b.follow
            }
            _ => false,
        }
    }

    /// The times (from, to) where `self`'s marks differ from `other`'s.
    fn marks_changed(&self, other: &Part) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut points = |x: Option<f64>, y: Option<f64>| {
            if x != y {
                out.extend(x.into_iter().chain(y).map(|t| (t, t)));
            }
        };
        match (self, other) {
            (Part::Wave(a, ..), Part::Wave(b, ..)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
                points(a.edit, b.edit);
                for (x, y) in [
                    (a.selection, b.selection),
                    (a.shown, b.shown),
                    (a.transition, b.transition),
                ] {
                    if x != y {
                        out.extend(x.into_iter().chain(y));
                    }
                }
                if a.gain != b.gain {
                    out.extend(a.transition.into_iter().chain(b.transition));
                    for g in [&a.gain, &b.gain] {
                        if let (Some(first), Some(last)) = (g.first(), g.last()) {
                            out.push((first.0, last.0));
                        }
                    }
                }
            }
            (Part::Ruler(a, ..), Part::Ruler(b, ..)) => {
                points(a.cursor, b.cursor);
                points(a.cue, b.cue);
            }
            _ => {}
        }
        out
    }
}
/// A layer's frame for the window ([`Canvas::layer_frame`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerFrame {
    /// Where it goes in the window, in pixels.
    pub rect: PixelRect,
    /// Its tiles' width (their height is the rect's).
    pub tile: usize,
    /// How many tile slots there are.
    pub slots: usize,
    /// The grid's seconds a column.
    pub px: f64,
    /// The column at its left edge, whole.
    pub offset: i64,
    /// And the share of the next one.
    pub frac: f32,
    /// The playhead's left edge across it, in pixels (fractional), and its
    /// width, where it shows.
    pub playhead: Option<(f32, usize)>,
    /// The row of pixels it starts at (under the bar numbers).
    pub head_top: usize,
    /// Its colour.
    pub head: [u8; 3],
    /// The tile in each slot.
    pub resident: Vec<Option<i64>>,
    /// Tiles painted since the last frame, to send over: slot and pixels
    /// (`tile` × `rect.3`, rows of `0x00RRGGBB`).
    pub uploads: Vec<(usize, Vec<u32>)>,
}

/// A rect of pixels: x, y, width, height.
pub type PixelRect = (usize, usize, usize, usize);

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
        backend.default_bg = rgb(colors::BG);
        backend.default_fg = rgb(colors::TEXT);
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
            big_text: super::karaoke::BigText::lyrics(),
            karaoke: None,
            karaoke_layer: super::karaoke::Layer::default(),
            stage: super::stage::Stage::new(),
            staged: None,
            layers: false,
            strip: None,
            melody_range: None,
            melody_clean: None,
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

    /// Whether lanes that follow the playhead go into a layer the window
    /// scrolls ([`Canvas::layer_frame`]): only where the window can (the
    /// GPU's); otherwise they are painted into the pixmap each frame.
    pub fn set_layers(&mut self, on: bool) {
        if on != self.layers {
            self.layers = on;
            self.strip = None;
            self.changes.all = true;
        }
    }

    /// Karaoke mode's stage for the frame just drawn, at `heard` seconds
    /// into the song, `dt` seconds after the last; none where it does not
    /// show.
    pub fn stage_frame(
        &mut self,
        state: &crate::core::State,
        heard: f64,
        dt: f32,
    ) -> Option<super::stage_gpu::StageFrame> {
        let (picture, rect) = self.staged.as_ref()?;
        let (x, y, w, h) = *rect;
        Some(self.stage.frame(
            &super::stage::Song::of(state),
            picture,
            (x as f32, y as f32, w as f32, h as f32),
            heard,
            dt,
        ))
    }

    /// Whether a layer shows now.
    pub fn has_layer(&self) -> bool {
        self.strip.is_some()
    }

    /// The layer's frame with `heard` seconds into the track: where it
    /// stands, and the tiles it needs painted (in view, and the next one
    /// coming in). `None` where none shows.
    pub fn layer_frame(&mut self, heard: f64) -> Option<LayerFrame> {
        let strip = self.strip.as_ref()?;
        let (rect, px) = (strip.rect, strip.px);
        let w = rect.2;
        let start = (heard - strip.lead * strip.span).max(0.0);
        let left = start / px;
        let offset = left.floor() as i64;
        let frac = (left - offset as f64) as f32;
        // The word heard: its tiles again, and those of the one before; and
        // those a mark changed in.
        let heard_word = strip
            .words
            .iter()
            .position(|&(a, b, _, _)| heard >= a && heard < b);
        let slots_n = strip.slots.len();
        let mut stale = strip.stale.clone();
        if heard_word != strip.heard {
            for word in [strip.heard, heard_word].into_iter().flatten() {
                let (_, _, c0, c1) = strip.words[word];
                stale.extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
            }
        }
        let first = offset.div_euclid(TILE as i64);
        let last = (offset + w as i64 + 1).div_euclid(TILE as i64) + 1;
        let head_top = strip
            .parts
            .iter()
            .find_map(|p| match p {
                Part::Wave(_, y, _) => Some(*y),
                Part::Ruler(..) => None,
            })
            .unwrap_or(0);
        let mut uploads = Vec::new();
        let parts: Vec<Part> = strip
            .parts
            .iter()
            .map(|p| match p {
                Part::Wave(plot, y, h) => {
                    let mut plot = plot.clone();
                    plot.playhead = Some(heard);
                    Part::Wave(plot, *y, *h)
                }
                other => other.clone(),
            })
            .collect();
        for tile in first..=last {
            let slot = tile.rem_euclid(slots_n as i64) as usize;
            let held = self
                .strip
                .as_ref()
                .is_some_and(|s| s.slots[slot] == Some(tile));
            if held && !stale.contains(&tile) {
                continue;
            }
            let pixels = self.paint_tile(&parts, px, tile, rect.3);
            if let Some(s) = self.strip.as_mut() {
                s.slots[slot] = Some(tile);
            }
            uploads.push((slot, pixels));
        }
        let strip = self.strip.as_mut()?;
        strip.heard = heard_word;
        // Stale tiles out of view now: painted when they come in.
        strip.stale.retain(|t| *t < first || *t > last);
        for slot in &mut strip.slots {
            if slot.is_some_and(|t| strip.stale.contains(&t)) {
                *slot = None;
            }
        }
        // The playhead as the lanes draw it: two lines wide, from the
        // column it falls in.
        let line = self.scale().round().max(1.0) as usize;
        let head_x = (heard / px - left) as f32;
        let strip = self.strip.as_ref()?;
        Some(LayerFrame {
            rect,
            tile: TILE,
            slots: slots_n,
            px,
            offset,
            frac,
            playhead: (head_x >= 0.0 && (head_x as usize) < w).then_some((head_x, 2 * line)),
            head_top,
            head: rgb(colors::ACCENT),
            resident: strip.slots.clone(),
            uploads,
        })
    }

    /// Tile `tile` of the layer's `parts` on the grid of `px` seconds a
    /// column, `h` pixels high: painted as they would be (the playhead
    /// apart), into pixels of its own.
    fn paint_tile(&mut self, parts: &[Part], px: f64, tile: i64, h: usize) -> Vec<u32> {
        let cols = Columns {
            px,
            n0: tile * TILE as i64,
        };
        // The pixmap lent to the painters for the tile's pixels.
        let bg = pack(rgb(colors::BG));
        let backend = self.terminal.backend_mut();
        let saved = (
            std::mem::replace(&mut backend.pixmap, vec![bg; TILE * h]),
            backend.width,
            backend.height,
        );
        backend.width = TILE;
        backend.height = h;
        for part in parts {
            match part {
                Part::Wave(plot, y, ph) => {
                    self.paint_lane(plot, cols, (0, *y, TILE, *ph), None, false, false);
                }
                Part::Ruler(plot, y, ph) => {
                    self.paint_ruler_at(plot, cols, (0, *y, TILE, *ph));
                }
            }
        }
        let backend = self.terminal.backend_mut();
        let tile_pixels = std::mem::replace(&mut backend.pixmap, saved.0);
        backend.width = saved.1;
        backend.height = saved.2;
        tile_pixels
    }

    /// What follows the playhead this frame (`found`: each part and its
    /// pixels in the window, top to bottom) as the layer: kept where only
    /// its place moved, its tiles under a changed mark painted again, made
    /// afresh where the parts changed.
    fn take_strip(&mut self, found: Vec<(Part, PixelRect)>) {
        let Some(&(_, first)) = found.first() else {
            self.strip = None;
            return;
        };
        let top = found.iter().map(|(_, r)| r.1).min().unwrap_or(first.1);
        let bottom = found
            .iter()
            .map(|(_, r)| r.1 + r.3)
            .max()
            .unwrap_or(first.1);
        let rect = (first.0, top, first.2, bottom - top);
        let (start, end, lead) = match &found[0].0 {
            Part::Wave(p, ..) => (p.start, p.end, p.follow.unwrap_or(0.0)),
            Part::Ruler(p, ..) => (p.start, p.end, p.follow.unwrap_or(0.0)),
        };
        let cols = Columns::new(start, end, rect.2);
        let parts: Vec<Part> = found
            .into_iter()
            .map(|(part, r)| match part {
                Part::Wave(p, ..) => Part::Wave(p, r.1 - top, r.3),
                Part::Ruler(p, ..) => Part::Ruler(p, r.1 - top, r.3),
            })
            .collect();
        let same = self.strip.as_ref().is_some_and(|s| {
            s.rect == rect
                && s.px == cols.px
                && s.parts.len() == parts.len()
                && s.parts.iter().zip(&parts).all(|(a, b)| a.same_ground(b))
        });
        if same {
            // The tiles under the marks that moved, a little either side
            // (a mark's width, a bar number pushed along by one).
            let pad = (6 * self.cell().0) as i64;
            if let Some(s) = self.strip.as_mut() {
                for (a, b) in s.parts.iter().zip(&parts) {
                    for (t0, t1) in a.marks_changed(b) {
                        let c0 = (t0 / s.px).floor() as i64 - pad;
                        let c1 = (t1 / s.px).floor() as i64 + pad;
                        s.stale
                            .extend(c0.div_euclid(TILE as i64)..=c1.div_euclid(TILE as i64));
                    }
                }
                s.stale.sort_unstable();
                s.stale.dedup();
                s.parts = parts;
                s.span = end - start;
            }
            return;
        }
        // Each word's columns, as the lane with words lays them out.
        let mut words = Vec::new();
        for part in &parts {
            let Part::Wave(plot, ..) = part else {
                continue;
            };
            let Some(lyrics) = plot.lyrics.as_deref() else {
                continue;
            };
            let text = usize::from(plot.text_scale.max(1));
            let cw = (self.cell().0 * text) as f64;
            let margin = (self.scale().round().max(1.0) as usize * text) as i64 + 1;
            let timed = crate::core::lanes::timed_words(lyrics);
            let x_of = |t: f64| (t / cols.px).floor();
            crate::core::lanes::lay_out_words(
                &timed,
                &x_of,
                1.0 / cols.px,
                cw,
                plot.lines,
                &mut |k, at, wide, _| {
                    let w = timed[k];
                    if let (Some(a), Some(b)) = (w.start, w.end) {
                        let c0 = at as i64 - margin;
                        let c1 = at as i64 + (wide as f64 * cw) as i64 + margin;
                        words.push((a, b, c0, c1));
                    }
                    true
                },
            );
            break;
        }
        let slots = (rect.2.div_ceil(TILE) + 3).min(MOST_SLOTS);
        self.strip = Some(Strip {
            parts,
            rect,
            px: cols.px,
            lead,
            span: end - start,
            slots: vec![None; slots],
            words,
            heard: None,
            stale: Vec::new(),
        });
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
                        self.paint_spectrum(plot);
                        frame.add(self.pixel_rect(plot.area));
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
                Paint::Wave(plot) if self.layers && plot.follow.is_some() => {
                    let r = self.pixel_rect(plot.area);
                    following.push((Part::Wave(plot.clone(), 0, 0), r));
                }
                Paint::Ruler(plot) if self.layers && plot.follow.is_some() => {
                    let r = self.pixel_rect(plot.area);
                    following.push((Part::Ruler(plot.clone(), 0, 0), r));
                }
                Paint::Wave(plot) => {
                    let kept = waves
                        .iter()
                        .position(|k| k.shape.area == plot.area)
                        .map(|i| waves.swap_remove(i));
                    let (painted, kept) = self.paint_wave(plot, kept);
                    if let Some(rect) = painted {
                        frame.add(rect);
                    }
                    self.waves.extend(kept);
                }
                Paint::Karaoke(full_picture) => {
                    // Where the GPU draws the stage, its lyrics are not in
                    // the pixmap.
                    let stripped;
                    let picture: &KaraokePicture = if self.layers
                        && super::stage::wanted(scene.state)
                    {
                        self.staged =
                            Some(((**full_picture).clone(), self.pixel_rect(full_picture.area)));
                        stripped = super::stage::without_lyrics(full_picture);
                        &stripped
                    } else {
                        full_picture
                    };
                    let damaged = self.terminal.backend().damage.touches(picture.area);
                    let kept = karaoke.as_ref() == Some(picture) && !damaged;
                    if !kept {
                        // Afresh where the cells under it were redrawn or
                        // its frame (the area, the controls) changed; else
                        // only where its text was and is.
                        let full = damaged
                            || karaoke.as_ref().is_none_or(|k| {
                                (k.area, k.controls, k.start)
                                    != (picture.area, picture.controls, picture.start)
                            });
                        for rect in self.paint_karaoke(picture, art, full) {
                            frame.add(rect);
                        }
                    }
                    self.karaoke = Some(picture.clone());
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

    fn pixel_rect(&self, rect: Rect) -> (usize, usize, usize, usize) {
        let (cw, ch) = self.cell();
        (
            usize::from(rect.x) * cw,
            usize::from(rect.y) * ch,
            usize::from(rect.width) * cw,
            usize::from(rect.height) * ch,
        )
    }

    /// The cover fitted into `rect`, centred; with `below_line` from half
    /// a cell down (under the border line through the first row) and at
    /// the left.
    fn paint_cover(&mut self, rect: Rect, art: &Art, below_line: bool) {
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

    /// What the GPU draws over Karaoke mode's picture as last painted (none
    /// without it): its bubbles' liquid and bursts.
    pub fn picture_over(&self) -> Option<&super::notes::Over> {
        self.karaoke.as_ref().map(|_| &self.karaoke_layer.over)
    }

    /// Karaoke mode's picture: the backdrop (made again for another cover
    /// or size), the lyrics and the count-in, then the controls' text over
    /// it. `full`: all of it; else only the text, where it was and is (the
    /// backdrop put back under the last frame's). Returns the pixels it
    /// changed.
    fn paint_karaoke(
        &mut self,
        picture: &KaraokePicture,
        art: Option<&Art>,
        full: bool,
    ) -> Vec<PixelRect> {
        let area = self.pixel_rect(picture.area);
        let bar = picture.controls.map(|r| self.pixel_rect(r));
        let id = art.map_or(0, |a| a.id);
        let fits = self
            .backdrop
            .as_ref()
            .is_some_and(|b| b.art == id && b.size == (area.2, area.3));
        if !fits {
            self.backdrop = Some(super::karaoke::Backdrop::new(art, area.2, area.3));
        }
        let full = full || !fits;
        self.karaoke_layer.over.gpu = self.layers;
        let Some(backdrop) = &self.backdrop else {
            return Vec::new();
        };
        let backend = self.terminal.backend_mut();
        let (width, height) = (backend.width, backend.height);
        let mut target = super::karaoke::Target {
            pixels: &mut backend.pixmap,
            width,
            height,
        };
        let changed = super::karaoke::paint(
            &mut target,
            &mut self.big_text,
            &mut self.karaoke_layer,
            backdrop,
            picture,
            area,
            bar,
            full,
        );
        if full {
            if let Some(controls) = picture.controls {
                backend.overlay(controls);
            }
        }
        changed
    }

    /// Text over the cover: the cover darkened towards its bottom, from
    /// four rows above the text, then the text's glyphs drawn again over
    /// it (without their cells' background).
    fn paint_caption(&mut self, cover: Rect, caption: Rect) {
        let (x0, y0, w, h) = self.pixel_rect(cover);
        let (_, cy, _, _) = self.pixel_rect(caption);
        let (_, ch) = self.cell();
        let backend = self.terminal.backend_mut();
        let mut px = Pixels {
            data: &mut backend.pixmap,
            stride: backend.width,
            x0,
            y0,
            w,
            h,
        };
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

    /// The range of `waveform`'s melody ([`Melody::range`]), kept from the
    /// last time it was asked for the same waveform.
    ///
    /// [`Melody::range`]: jodel_rhythm::waveform::Melody::range
    fn melody_range(&mut self, waveform: &Arc<Waveform>) -> Option<(u8, u8)> {
        let id = Arc::as_ptr(waveform) as usize;
        match self.melody_range {
            Some((kept, range)) if kept == id => range,
            _ => {
                let range = waveform.melody.as_ref().and_then(|m| m.range());
                self.melody_range = Some((id, range));
                range
            }
        }
    }

    /// `waveform`'s melody with what is not sung taken out
    /// ([`Melody::cleaned`]): kept to the timed words of `lyrics` (all of
    /// it without), crumbs dropped; kept from the last time it was asked
    /// for the same waveform and lyrics.
    ///
    /// [`Melody::cleaned`]: jodel_rhythm::waveform::Melody::cleaned
    fn melody_clean(
        &mut self,
        waveform: &Arc<Waveform>,
        lyrics: Option<&Arc<jodel_lyrics::Lyrics>>,
    ) -> Option<Arc<jodel_rhythm::waveform::Melody>> {
        let melody = waveform.melody.as_ref()?;
        let id = Arc::as_ptr(waveform) as usize;
        let words = lyrics.map_or(0, |l| Arc::as_ptr(l) as usize);
        if let Some((w, l, m)) = &self.melody_clean {
            if (*w, *l) == (id, words) {
                return Some(Arc::clone(m));
            }
        }
        let spans: Option<Vec<(f64, f64)>> = lyrics.map(|l| {
            l.lines
                .iter()
                .flat_map(|line| &line.words)
                .filter_map(|w| w.start.zip(w.end))
                .collect()
        });
        let spans = spans.filter(|s| !s.is_empty());
        let clean = Arc::new(melody.cleaned(spans.as_deref()));
        self.melody_clean = Some((id, words, Arc::clone(&clean)));
        Some(clean)
    }

    /// The pixels per point of the window's scale factor.
    fn scale(&self) -> f64 {
        f64::from(self.font_px) / BASE_FONT_PX
    }

    /// A Transition lane in pixels: the marks behind (the selection, the span
    /// the lanes show, the cursor and the cue), the RMS filled and the peak
    /// outline over it, mirrored about the middle, the bars as lines, the
    /// playhead over all; on the voice's lane the words across its middle, a
    /// held sound a line. The lane's pixels stay from the last frame: a view
    /// that scrolled is shifted by whole columns, and only the columns new to
    /// it or whose marks changed are painted, with the words over them; all of
    /// them where the lane is new, moved, or a cell under it was redrawn. The
    /// words stand on the columns' grid, so a word kept moves with the pixels;
    /// one that changed (the word heard now, one coming in) is painted afresh
    /// with the columns under it. Returns the pixels that changed, if any.
    fn paint_wave(
        &mut self,
        plot: &WavePlot,
        kept: Option<KeptWave>,
    ) -> (Option<PixelRect>, Option<KeptWave>) {
        let rect = self.pixel_rect(plot.area);
        let cols = Columns::new(plot.start, plot.end, rect.2);
        let damaged = self.terminal.backend().damage.touches(plot.area);
        self.paint_lane(plot, cols, rect, kept, damaged, true)
    }

    /// [`Canvas::paint_wave`] into the pixels `rect` of the pixmap, on the
    /// grid `cols`; `damaged`: the cells under it were redrawn; `head`: the
    /// playhead drawn (not where something else draws it over).
    fn paint_lane(
        &mut self,
        plot: &WavePlot,
        cols: Columns,
        (x0, y0, w, h): PixelRect,
        kept: Option<KeptWave>,
        damaged: bool,
        head: bool,
    ) -> (Option<PixelRect>, Option<KeptWave>) {
        if w == 0 || h == 0 {
            return (None, None);
        }
        let waveform = Arc::clone(&plot.waveform);
        let Some(levels) = (match plot.lane.stem() {
            None => Some(&waveform.mix),
            Some(i) => waveform.stems.as_ref().map(|s| &s[i]),
        }) else {
            return (None, None);
        };
        let scale = self.scale();
        let line = scale.round().max(1.0) as usize;
        // The whole lane's height, of which the rect shows the pixels from
        // `above` on (while its block slides).
        let ch = self.cell().1;
        let above = usize::from(plot.rows.0) * ch;
        let full = (usize::from(plot.rows.1) * ch).max(h + above);
        let keys = wave_keys(plot, cols, w, h, line, above, full, head);
        let shape = WaveShape {
            area: plot.area,
            pixels: (x0, y0, w, h),
            lane: plot.lane,
            color: plot.color,
            line,
            above,
            full,
            px: cols.px,
            text: usize::from(plot.text_scale.max(1)),
            rows: plot.lines.rows.max(1),
            key: plot.key,
        };
        // The voice's notes, on the lane that is read: each on its row of
        // the song's range (a semitone more each side), the scale's rows
        // lined across.
        let clean = if plot.deep {
            self.melody_clean(&waveform, plot.lyrics.as_ref())
        } else {
            None
        };
        let notes = match &clean {
            Some(m) => self
                .melody_range(&waveform)
                .map(|(lo, hi)| NoteRows::new(m, lo, hi, full, line, plot.key)),
            None => None,
        };
        // The words: on the grid's columns (absolute: `n0` on), a
        // character `text` cells wide and as many high (one, but on the
        // Karaoke tab), centred on the lane, where that shows.
        let text = shape.text;
        let cw = self.cell().0 * text;
        let tch = ch * text;
        let step = scale.round().max(1.0) as usize;
        // The first line centred on the lane, those under it after it (all
        // of them moved up as far as they need to fit).
        let lines = shape.rows;
        let middle = (full / 2)
            .saturating_sub(tch / 2)
            .min(full.saturating_sub(tch * lines));
        let row = (middle >= above && middle - above + tch <= h).then(|| middle - above);
        let labels: Vec<WaveLabel> = match (plot.lyrics.as_deref(), row) {
            (Some(lyrics), Some(_)) => {
                let words = crate::core::lanes::timed_words(lyrics);
                let x_of = |t: f64| (t / cols.px).floor();
                // A word just past either edge too: its outline reaches in.
                let reach = (step * text) as f64 + 1.0;
                let view = (cols.n0 as f64 - reach, (cols.n0 + w as i64) as f64 + reach);
                view::lanes::lane_labels(
                    &words,
                    &x_of,
                    1.0 / cols.px,
                    cw as f64,
                    plot.lines,
                    plot.playhead,
                    view,
                )
                .into_iter()
                .filter(|l| middle + (l.row + 1) * tch <= full)
                .map(|l| WaveLabel {
                    col: l.at as i64,
                    text: l.text,
                    heard: l.heard,
                    row: l.row,
                })
                .collect()
            }
            _ => Vec::new(),
        };
        // The columns a label's pixels may touch (its outline and a glyph
        // past its cell included), on this frame's grid.
        let margin = (step * text) as i64 + 1;
        let span = |l: &WaveLabel| -> (i64, i64) {
            let a = l.col - cols.n0 - margin;
            let b = l.col - cols.n0 + (l.text.chars().count() * cw) as i64 + margin;
            (a.clamp(0, w as i64), b.clamp(0, w as i64))
        };
        let backend = self.terminal.backend_mut();
        let mut px = Pixels {
            data: &mut backend.pixmap,
            stride: backend.width,
            x0,
            y0,
            w,
            h,
        };
        // The columns to paint, after the kept pixels are shifted (which
        // changes them all).
        let mut shifted = true;
        let mut mask: Vec<bool> = match &kept {
            Some(old)
                if !damaged
                    && old.shape == shape
                    && Arc::ptr_eq(&old.waveform, &plot.waveform)
                    && (cols.n0 - old.n0).unsigned_abs() < w as u64 =>
            {
                let k = cols.n0 - old.n0;
                if k != 0 {
                    px.shift(k);
                } else {
                    shifted = false;
                }
                let mut mask: Vec<bool> = (0..w)
                    .map(|x| {
                        let from = x as i64 + k;
                        from < 0 || from >= w as i64 || old.keys[from as usize] != keys[x]
                    })
                    .collect();
                // A word gone or changed: its old pixels go; a word new or
                // changed: painted. A word kept stays where the shift put
                // it.
                let mut mark = |(a, b): (i64, i64)| {
                    for x in a..b {
                        mask[x as usize] = true;
                    }
                };
                for l in old.labels.iter().filter(|l| !labels.contains(l)) {
                    mark(span(l));
                }
                for l in labels.iter().filter(|l| !old.labels.contains(l)) {
                    mark(span(l));
                }
                mask
            }
            _ => vec![true; w],
        };
        // A word with a column under it painted is painted whole (its
        // glyphs blended once over clean pixels), its columns with it.
        loop {
            let mut grew = false;
            for l in &labels {
                let (a, b) = span(l);
                let range = a as usize..b.max(a) as usize;
                let any = mask[range.clone()].iter().any(|&d| d);
                if any && !mask[range.clone()].iter().all(|&d| d) {
                    mask[range].iter_mut().for_each(|d| *d = true);
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        let dirty: Vec<usize> = (0..w).filter(|&x| mask[x]).collect();
        let changed = match (shifted, dirty.first(), dirty.last()) {
            (true, _, _) => Some((x0, y0, w, h)),
            (false, Some(&a), Some(&b)) => Some((x0 + a, y0, b + 1 - a, h)),
            _ => None,
        };
        let bg = rgb(colors::BG);
        let mid = full as f64 / 2.0;
        let curve = rgb(view::lanes::CURVE);
        for &x in &dirty {
            let key = keys[x];
            let (t0, t1) = (cols.time(x as f64), cols.time(x as f64 + 1.0));
            let back = match key.back {
                Back::Selected => rgb(view::lanes::SELECT_BG),
                Back::Shown => rgb(view::lanes::SHOWN_BG),
                Back::Outside => rgb(view::lanes::OUTSIDE_BG),
                Back::Transition => rgb(view::lanes::TRANSITION_BG),
                Back::None => bg,
            };
            let color = if key.unclear {
                view::lanes::dimmed(plot.color)
            } else {
                plot.color
            };
            let color = view::lanes::faded(color, f32::from_bits(key.gain));
            let color = if key.back == Back::Outside {
                view::lanes::faded(color, view::lanes::OUTSIDE_GAIN)
            } else {
                color
            };
            // The RMS filled, the peak beyond it outlined; under words to be
            // read, the RMS deep and the peak bright.
            let (body, edge) = if plot.deep {
                (rgb(color), deep(rgb(color)))
            } else {
                (rgb(view::lanes::dimmed(color)), rgb(color))
            };
            let peak = f64::from(levels.peak_between(t0, t1)) * mid;
            let rms = f64::from(levels.rms_between(t0, t1)) * mid;
            let plain = (back, body, edge);
            let (back, body, edge) = (pack(back), pack(body), pack(edge));
            // The column in one pass: RMS filled, the peak outline, else
            // the background; the scale's lines across all of it.
            px.column(x, |y| {
                let d = ((y + above) as f64 + 0.5 - mid).abs();
                let (packed, c) = if d <= rms {
                    (edge, plain.2)
                } else if d <= peak {
                    (body, plain.1)
                } else {
                    (back, plain.0)
                };
                match notes.as_ref().map_or(0.0, |n| n.line_at(y + above)) {
                    0.0 => packed,
                    share => pack(mix(c, rgb(colors::TEXT), share)),
                }
            });
            // The note sung in the column's middle, a bar on its row with a
            // dark edge, over the voice.
            if let Some(n) = notes.as_ref() {
                let frame = ((t0 + t1) / 2.0 * jodel_rhythm::waveform::FPS).floor();
                if let Some((top, bottom)) = (frame >= 0.0).then(|| n.bar(frame as usize)).flatten()
                {
                    let bar = rgb(view::lanes::NOTE);
                    for yy in top.saturating_sub(line)..bottom + line {
                        let Some(y) = yy.checked_sub(above) else {
                            continue;
                        };
                        if yy < top || yy >= bottom {
                            px.blend(x, y, rgb(colors::BG), 0.7);
                        } else {
                            px.blend(x, y, bar, 0.92);
                        }
                    }
                }
            }
            // The lines over it, in their order: the bars, the cue, the
            // cursor, the bar being edited, the playhead.
            let mut over = |c: [u8; 3], share: f32| {
                for y in 0..h {
                    px.blend(x, y, c, share);
                }
            };
            for _ in 0..key.downbeats {
                over(rgb(colors::TEXT), 0.22);
            }
            for (bit, c, share) in [
                (MARK_CUE, view::lanes::CUE, 0.9),
                (MARK_CURSOR, colors::ACCENT, 0.55),
                (MARK_EDIT, view::lanes::CURVE, 0.8),
                (MARK_PLAYHEAD, colors::ACCENT, 1.0),
            ] {
                if key.marks & bit != 0 {
                    over(rgb(c), share);
                }
            }
            // The transition's fade: a line from the bottom (off) to the
            // top (full), joined to the column before.
            if let Some((lo, hi)) = key.curve {
                for yy in lo..=hi {
                    for dy in 0..line {
                        px.blend(x, (yy + dy).min(h - 1), curve, 0.9);
                    }
                }
            }
        }
        // The words over the columns just painted (each is painted whole
        // or not at all).
        if let Some(row) = row {
            let big = &mut self.big_text;
            let text_px = self.font_px * text as f32;
            let backend = self.terminal.backend_mut();
            let y = y0 + row;
            let clip = (x0, x0 + w);
            let edge = rgb(colors::BG);
            // The outline as wide as a stroke at the size drawn.
            let outline = step * text.div_ceil(2);
            let s = outline as f64;
            // The edges above and below only where they stay in the lane.
            let below = row + tch * lines + outline <= h;
            let up = row >= outline;
            // A held sound's line: as thick as a stroke, through the
            // middle of the cell.
            let thick = step.max(2) * text.div_ceil(2);
            let line_y = y + tch / 2 - thick / 2;
            // Letters in the cells' own glyphs, a cell each (one cell high);
            // larger, a run in the lyrics font.
            let put = |backend: &mut crate::gui::raster::PixelBackend,
                       x: f64,
                       y: usize,
                       run: &str,
                       fg: [u8; 3]| backend.text_at(x, y, run, fg, clip);
            let target_size = (backend.width, backend.height);
            let clip_rect = (x0, y0, w, h);
            let draw_run = |backend: &mut crate::gui::raster::PixelBackend,
                            big: &mut super::karaoke::BigText,
                            (x, y): (f64, usize),
                            run: &str,
                            size: f32,
                            fg: [u8; 3]| {
                let mut target = super::karaoke::Target {
                    pixels: &mut backend.pixmap,
                    width: target_size.0,
                    height: target_size.1,
                };
                big.draw(
                    &mut target,
                    run,
                    size,
                    size,
                    (x as f32, y as f32),
                    f32::MAX,
                    (fg, fg),
                    1.0,
                    clip_rect,
                );
            };
            for l in &labels {
                let (a, b) = span(l);
                if a >= b || !mask[a as usize] {
                    continue;
                }
                let fg = rgb(if l.heard {
                    colors::ACCENT
                } else {
                    colors::TEXT
                });
                let x = (x0 as i64 + l.col - cols.n0) as f64;
                // Its line's top, and its held sound's line.
                let y = y + l.row * tch;
                let line_y = line_y + l.row * tch;
                if y + tch > y0 + h {
                    continue;
                }
                let below = below || l.row + 1 < lines;
                let up = up || l.row > 0;
                // The letters, and the runs of the line between them.
                let chars: Vec<char> = l.text.chars().collect();
                let mut runs: Vec<(usize, usize, bool)> = Vec::new();
                for (i, &c) in chars.iter().enumerate() {
                    let held = c == view::lanes::HOLD;
                    match runs.last_mut() {
                        Some(r) if r.2 == held => r.1 = i + 1,
                        _ => runs.push((i, i + 1, held)),
                    }
                }
                // Each run where it goes: letters from their first cell, a
                // held sound's line between. In the cells' own glyphs, one
                // a cell; larger, in the lyrics font with its own spacing
                // (narrowed where a run would not fit its cells), the line
                // from just after the letters before it.
                let mut pieces: Vec<Piece> = Vec::new();
                let mut after: Option<f64> = None;
                for (n, &(i, j, held)) in runs.iter().enumerate() {
                    let (slot0, slot1) = (x + (i * cw) as f64, x + (j * cw) as f64);
                    if held {
                        let from = match (text, after) {
                            (1, _) | (_, None) => slot0,
                            (_, Some(end)) => end + s,
                        };
                        let to = if text > 1 && n + 1 < runs.len() {
                            slot1 - s
                        } else {
                            slot1
                        };
                        pieces.push((from, to.max(from), None));
                        continue;
                    }
                    let run: String = chars[i..j].iter().collect();
                    if text == 1 {
                        pieces.push((slot0, slot1, Some((run, 0.0))));
                        continue;
                    }
                    let natural = f64::from(big.measure(&run, text_px));
                    // Up to the next letters (into a held sound's line
                    // between, which then starts later), else to the end of
                    // the word's cells.
                    let next = runs[n + 1..]
                        .iter()
                        .find(|r| !r.2)
                        .map_or(x + (chars.len() * cw) as f64, |r| x + (r.0 * cw) as f64 - s);
                    let room = (next - slot0).max(slot1 - slot0);
                    let size = if natural > room {
                        (text_px * (room / natural) as f32).floor()
                    } else {
                        text_px
                    };
                    let width = natural.min(room);
                    after = Some(slot0 + width);
                    pieces.push((slot0, slot0 + width, Some((run, size))));
                }
                let px = |v: f64| (v.round().max(0.0) as usize).clamp(x0, x0 + w);
                // A dark edge first, so the words read over a loud voice.
                for (from, to, letters) in &pieces {
                    let Some((run, size)) = letters else {
                        let top = line_y.saturating_sub(outline).max(y0);
                        let bottom = (line_y + thick + outline).min(y0 + h);
                        let left = px(*from).saturating_sub(outline).max(x0);
                        backend.fill(left, top, (px(*to) + outline).min(x0 + w), bottom, edge);
                        continue;
                    };
                    for (dx, dy) in [(-s, 0.0), (s, 0.0), (0.0, -s), (0.0, s)] {
                        if (dy > 0.0 && !below) || (dy < 0.0 && !up) {
                            continue;
                        }
                        let row = (y as f64 + dy).max(0.0) as usize;
                        if text == 1 {
                            put(backend, from + dx, row, run, edge);
                        } else {
                            draw_run(backend, big, (from + dx, row), run, *size, edge);
                        }
                    }
                }
                for (from, to, letters) in &pieces {
                    match letters {
                        None => backend.fill(px(*from), line_y, px(*to), line_y + thick, fg),
                        Some((run, _)) if text == 1 => put(backend, *from, y, run, fg),
                        Some((run, size)) => draw_run(backend, big, (*from, y), run, *size, fg),
                    }
                }
            }
        }
        let kept = KeptWave {
            shape,
            waveform: Arc::clone(&plot.waveform),
            n0: cols.n0,
            keys,
            labels,
        };
        (changed, Some(kept))
    }

    /// A Transition block's bar numbers in pixels: each starting at its bar's
    /// line in the lanes below (to the pixel, so they move with the
    /// waveforms), a tick where there is no room for it, the cue and the
    /// cursor centred on theirs. On the lanes' grid of columns
    /// ([`Columns`]); kept where nothing in it changed. Returns whether a
    /// pixel changed.
    fn paint_ruler(
        &mut self,
        plot: &RulerPlot,
        kept: Option<KeptRuler>,
    ) -> (bool, Option<KeptRuler>) {
        let (x0, y0, w, h) = self.pixel_rect(plot.area);
        if w == 0 || h == 0 {
            return (false, None);
        }
        let cols = Columns::new(plot.start, plot.end, w);
        let now = KeptRuler {
            area: plot.area,
            cols,
            bars: plot.bars.clone(),
            cursor: plot.cursor,
            cue: plot.cue,
        };
        if kept.as_ref() == Some(&now) && !self.terminal.backend().damage.touches(plot.area) {
            return (false, Some(now));
        }
        self.paint_ruler_at(plot, cols, (x0, y0, w, h));
        (true, Some(now))
    }

    /// [`Canvas::paint_ruler`] into the pixels `(x0, y0, w, h)` of the
    /// pixmap, on the grid `cols`. The numbers are laid out from the first
    /// bar given (for a ruler following the playhead, the track's first),
    /// so they stand where they are as the view moves.
    fn paint_ruler_at(&mut self, plot: &RulerPlot, cols: Columns, (x0, y0, w, h): PixelRect) {
        let (cw, _) = self.cell();
        let line = self.scale().round().max(1.0) as usize;
        let backend = self.terminal.backend_mut();
        let mut px = Pixels {
            data: &mut backend.pixmap,
            stride: backend.width,
            x0,
            y0,
            w,
            h,
        };
        px.fill(rgb(colors::BG));
        let column = |t: f64| cols.of(t) as f64;
        let dim = rgb(colors::DIM);
        let mut ticks = Vec::new();
        let mut labels = Vec::new();
        let mut free = f64::NEG_INFINITY;
        let marks: Vec<f64> = [plot.cue, plot.cursor]
            .into_iter()
            .flatten()
            .map(column)
            .collect();
        for &(t, bar) in &plot.bars {
            let mut x = column(t);
            if x >= w as f64 {
                break;
            }
            // A mark on the bar goes before its number.
            if marks.iter().any(|&m| (m - x).abs() < cw as f64) {
                x += cw as f64 / 2.0 + line as f64;
            }
            let label = bar.to_string();
            if x < free {
                if x >= 0.0 {
                    ticks.push(x as usize);
                }
                continue;
            }
            free = x + ((label.len() + 1) * cw) as f64;
            // Left of the view, but reaching into it: drawn, cut.
            if x + (label.len() * cw) as f64 > 0.0 {
                labels.push((x, label));
            }
        }
        for x in ticks {
            for dx in 0..line {
                for y in 0..h / 2 {
                    px.set(x + dx, y, dim);
                }
            }
        }
        let clip = (x0, x0 + w);
        for (x, label) in labels {
            backend.text_at(x0 as f64 + x, y0, &label, dim, clip);
        }
        for (time, symbol, color) in [
            (plot.cue, "◆", view::lanes::CUE),
            (plot.cursor, "▼", colors::ACCENT),
        ] {
            if let Some(x) = time.map(column).filter(|&x| x >= 0.0 && x < w as f64) {
                let left = x0 as f64 + x - (cw / 2) as f64;
                backend.text_at(left, y0, symbol, rgb(color), clip);
            }
        }
    }

    /// The spectrum in pixels: the grid as single dots every few pixels,
    /// the curve as round dots at a fixed spacing along it, both at the
    /// window's scale.
    fn paint_spectrum(&mut self, plot: &Plot) {
        let (x0, y0, w, h) = self.pixel_rect(plot.area);
        let spectrum = &plot.spectrum;
        if w == 0 || h == 0 || spectrum.is_empty() {
            return;
        }
        let scale = self.scale();
        let grid_dot = scale.round().max(1.0) as usize;
        let grid_step = (4.0 * scale).round().max(3.0) as usize;
        let radius = scale.max(1.0);
        let spacing = 4.0 * scale.max(1.0);
        let bg = rgb(colors::BG);
        let backend = self.terminal.backend_mut();
        let mut px = Pixels {
            data: &mut backend.pixmap,
            stride: backend.width,
            x0,
            y0,
            w,
            h,
        };
        px.fill(bg);
        // The grid.
        let grid = rgb(colors::GRID);
        for &db in &plot.db_lines {
            let y = plot::unit_of(db, h);
            for x in (0..w).step_by(grid_step) {
                px.square(x, y, grid_dot, grid);
            }
        }
        let lines = plot.hz_lines.iter().map(|&hz| (hz, grid));
        let edges = plot.edges.iter().map(|&(hz, c)| (hz, rgb(plot::faint(c))));
        for (hz, color) in lines.chain(edges) {
            let x = spectrum.x_of(hz, w).floor();
            if (0.0..w as f64).contains(&x) {
                for y in (0..h).step_by(grid_step) {
                    px.square(x as usize, y, grid_dot, color);
                }
            }
        }
        // The curve: one point per dot width, dots every `spacing` along it.
        let n = (w as f64 / (2.0 * radius)).round().max(2.0) as usize;
        let points: Vec<(f64, f64)> = spectrum
            .levels(n)
            .iter()
            .enumerate()
            .map(|(i, &db)| {
                let x = (i as f64 + 0.5) * w as f64 / n as f64;
                (x, plot::depth(db) * h as f64)
            })
            .collect();
        let floor = h as f64 - 0.5;
        let mut dot = |x: f64, y: f64| {
            if y < floor {
                let hz = spectrum.hz_at(x, w);
                let color = rgb(view::zone_color(&plot.zones, hz));
                px.disc(x, y.clamp(radius, h as f64 - radius), radius, color);
            }
        };
        let mut next = 0.0;
        let mut walked = 0.0;
        for pair in points.windows(2) {
            let ((ax, ay), (bx, by)) = (pair[0], pair[1]);
            let length = (bx - ax).hypot(by - ay);
            while next <= walked + length {
                let t = (next - walked) / length;
                dot(ax + (bx - ax) * t, ay + (by - ay) * t);
                next += spacing;
            }
            walked += length;
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
        let bg = pack(rgb(colors::BG));
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

/// A word's run on a lane: from, to (pixels), and its letters and their
/// size (none: a held sound's line).
type Piece = (f64, f64, Option<(String, f32)>);

/// Where the voice's notes go on a lane `full` pixels high: a row a
/// semitone from a semitone over the range `hi` down to one under `lo`,
/// the lane's top and bottom sixteenth left free; the scale's rows of
/// `key` lined.
struct NoteRows<'a> {
    melody: &'a jodel_rhythm::waveform::Melody,
    hi: u8,
    top: f64,
    row: f64,
    /// Each pixel row's line, as the share of the text's colour over the
    /// background (0 for none).
    lines: Vec<f32>,
    /// A bar's thickness.
    thick: usize,
}

impl<'a> NoteRows<'a> {
    fn new(
        melody: &'a jodel_rhythm::waveform::Melody,
        lo: u8,
        hi: u8,
        full: usize,
        line: usize,
        key: Option<jodel_rhythm::key::Key>,
    ) -> NoteRows<'a> {
        let (lo, hi) = (lo.saturating_sub(1), hi.saturating_add(1).max(lo + 2));
        let top = full as f64 / 16.0;
        let row = (full as f64 - 2.0 * top) / f64::from(hi - lo + 1);
        let thick = ((row * 0.6).round() as usize).clamp(2 * line, 5 * line);
        let mut rows = NoteRows {
            melody,
            hi,
            top,
            row,
            lines: vec![0.0; full],
            thick,
        };
        if let Some(key) = key {
            let scale = key.scale();
            for n in lo..=hi {
                let class = n % 12;
                if !scale.contains(&class) {
                    continue;
                }
                let share = if class == key.tonic() { 0.2 } else { 0.08 };
                let y = rows.middle(n).round() as usize;
                for v in rows
                    .lines
                    .iter_mut()
                    .skip(y.saturating_sub(line / 2))
                    .take(line)
                {
                    *v = share;
                }
            }
        }
        rows
    }

    /// Note `n`'s row's middle, in pixels from the lane's top.
    fn middle(&self, n: u8) -> f64 {
        self.top + (f64::from(self.hi) - f64::from(n) + 0.5) * self.row
    }

    /// Pixel row `y`'s line.
    fn line_at(&self, y: usize) -> f32 {
        self.lines.get(y).copied().unwrap_or(0.0)
    }

    /// The bar of the note sung at `frame`, as pixel rows from the lane's
    /// top (from, to).
    fn bar(&self, frame: usize) -> Option<(usize, usize)> {
        let n = self.melody.sung(frame)?;
        let y = (self.middle(n) - self.thick as f64 / 2.0).round().max(0.0) as usize;
        Some((y, y + self.thick))
    }
}

/// `c` at an eighth of its light (in linear light): a deep shade of it,
/// near-white text on it reading at about 9 : 1.
fn deep(c: [u8; 3]) -> [u8; 3] {
    let lin = |v: u8| {
        let v = f32::from(v) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let enc = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    c.map(|v| enc(lin(v) * 0.12))
}

/// The pixels of one rect of the pixmap, clipped to it.
struct Pixels<'a> {
    data: &'a mut [u32],
    stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

impl Pixels<'_> {
    fn index(&self, x: usize, y: usize) -> Option<usize> {
        let i = (self.y0 + y) * self.stride + self.x0 + x;
        (x < self.w && y < self.h && i < self.data.len()).then_some(i)
    }

    fn set(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if let Some(i) = self.index(x, y) {
            self.data[i] = pack(c);
        }
    }

    fn blend(&mut self, x: usize, y: usize, c: [u8; 3], t: f32) {
        if let Some(i) = self.index(x, y) {
            self.data[i] = pack(mix(unpack(self.data[i]), c, t));
        }
    }

    /// The rows of the rect inside the pixmap, as (first index, width).
    fn rows(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let w = self.w.min(self.stride.saturating_sub(self.x0));
        let (stride, x0, y0) = (self.stride, self.x0, self.y0);
        let h = self
            .h
            .min((self.data.len() / stride.max(1)).saturating_sub(y0));
        (0..h).map(move |y| ((y0 + y) * stride + x0, w))
    }

    /// The whole rect in one colour.
    fn fill(&mut self, c: [u8; 3]) {
        let word = pack(c);
        for (at, w) in self.rows() {
            self.data[at..at + w].fill(word);
        }
    }

    /// The rect's pixels moved `k` columns left (right where negative); the
    /// columns that leaves behind keep what they had.
    fn shift(&mut self, k: i64) {
        let n = k.unsigned_abs() as usize;
        for (at, w) in self.rows() {
            if n >= w {
                continue;
            }
            let row = &mut self.data[at..at + w];
            if k > 0 {
                row.copy_within(n.., 0);
            } else {
                row.copy_within(..w - n, n);
            }
        }
    }

    /// Column `x`, each pixel the colour `color_at` gives for its row.
    fn column(&mut self, x: usize, mut color_at: impl FnMut(usize) -> u32) {
        if x >= self.w || self.x0 + x >= self.stride {
            return;
        }
        for (y, (at, _)) in self.rows().enumerate() {
            self.data[at + x] = color_at(y);
        }
    }

    fn square(&mut self, x: usize, y: usize, side: usize, c: [u8; 3]) {
        for dy in 0..side {
            for dx in 0..side {
                self.set(x + dx, y + dy, c);
            }
        }
    }

    /// A round dot centred at (`cx`, `cy`), its edge smoothed.
    fn disc(&mut self, cx: f64, cy: f64, radius: f64, c: [u8; 3]) {
        let reach = radius + 1.0;
        let (x_from, x_to) = ((cx - reach).floor().max(0.0), (cx + reach).ceil());
        let (y_from, y_to) = ((cy - reach).floor().max(0.0), (cy + reach).ceil());
        for y in y_from as usize..y_to as usize {
            for x in x_from as usize..x_to as usize {
                let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
                let cover = (radius + 0.5 - d).clamp(0.0, 1.0) as f32;
                if cover > 0.0 {
                    self.blend(x, y, c, cover);
                }
            }
        }
    }
}

/// How a picture's pixel columns map to time: column `x` is `n0 + x` on a
/// grid of `px` seconds from the start of the track. The grid is the same
/// from frame to frame while the span shown and the width are, so a view
/// that scrolls shows the same columns, moved by whole pixels (its start
/// is within a pixel of the plot's).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Columns {
    px: f64,
    n0: i64,
}

impl Columns {
    fn new(start: f64, end: f64, w: usize) -> Columns {
        // Rounded, so a span that differs in its last bits from the last
        // frame's gives the same grid.
        let px = ((end - start) / w.max(1) as f64 * 1e12).round().max(1.0) / 1e12;
        Columns {
            px,
            n0: (start / px).round() as i64,
        }
    }

    /// The seconds at the left edge of column `x`.
    fn time(&self, x: f64) -> f64 {
        (self.n0 as f64 + x) * self.px
    }

    /// The column `time` falls in (outside `0..w` where it is not shown).
    fn of(&self, time: f64) -> i64 {
        (time / self.px).floor() as i64 - self.n0
    }
}

/// A Transition lane column's background.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Back {
    #[default]
    None,
    Selected,
    Shown,
    /// The overview outside the span shown: shaded.
    Outside,
    Transition,
}

const MARK_CUE: u8 = 1;
const MARK_CURSOR: u8 = 2;
const MARK_EDIT: u8 = 4;
const MARK_PLAYHEAD: u8 = 8;

/// Everything a Transition lane's pixel column shows besides the waveform at
/// its place on the grid: two columns with the same key at the same place are
/// the same pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ColumnKey {
    back: Back,
    unclear: bool,
    /// The gain the waveform is faded by (its bits).
    gain: u32,
    /// The bar lines through it.
    downbeats: u8,
    /// The other lines through it (`MARK_*`).
    marks: u8,
    /// The fade's line in it: from row to row.
    curve: Option<(usize, usize)>,
}

/// What a lane's kept pixels were painted for, besides the columns.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WaveShape {
    area: Rect,
    pixels: (usize, usize, usize, usize),
    lane: Lane,
    color: Color,
    line: usize,
    above: usize,
    full: usize,
    px: f64,
    /// The words' size, in cells, and their lines.
    text: usize,
    rows: usize,
    /// The key the notes' lines stand on.
    key: Option<jodel_rhythm::key::Key>,
}

/// A Transition lane as the pixmap holds it.
struct KeptWave {
    shape: WaveShape,
    waveform: Arc<Waveform>,
    n0: i64,
    keys: Vec<ColumnKey>,
    /// The words drawn over it.
    labels: Vec<WaveLabel>,
}

/// A word drawn on a Transition lane: its first column on the grid (absolute,
/// [`Columns::n0`] on), its characters, whether in the accent colour.
#[derive(Clone, Debug, PartialEq)]
struct WaveLabel {
    col: i64,
    text: String,
    heard: bool,
    /// Its line ([`crate::core::lanes::WordLines`]).
    row: usize,
}

/// A Transition block's bar numbers as the pixmap holds them.
#[derive(Clone, Debug, PartialEq)]
struct KeptRuler {
    area: Rect,
    cols: Columns,
    bars: Vec<(f64, usize)>,
    cursor: Option<f64>,
    cue: Option<f64>,
}

/// The key of each of a lane's `w` pixel columns.
#[allow(clippy::too_many_arguments)]
fn wave_keys(
    plot: &WavePlot,
    cols: Columns,
    w: usize,
    h: usize,
    line: usize,
    above: usize,
    full: usize,
    head: bool,
) -> Vec<ColumnKey> {
    let mut keys = vec![ColumnKey::default(); w];
    // The stretches where the beat is unclear are in order: walked along.
    let mut unclear = plot.unclear.iter().peekable();
    for (x, key) in keys.iter_mut().enumerate() {
        let (t0, t1) = (cols.time(x as f64), cols.time(x as f64 + 1.0));
        let over = |r: Option<(f64, f64)>| r.is_some_and(|(a, b)| t1 > a && t0 < b);
        key.back = if over(plot.selection) {
            Back::Selected
        } else if over(plot.shown) {
            Back::Shown
        } else if over(plot.transition) {
            Back::Transition
        } else if plot.shown.is_some() {
            Back::Outside
        } else {
            Back::None
        };
        let middle = 0.5 * (t0 + t1);
        while unclear.next_if(|&&(_, b)| b <= middle).is_some() {}
        key.unclear = unclear
            .peek()
            .is_some_and(|&&(a, b)| middle >= a && middle < b);
        key.gain = plot.gain_at(middle).to_bits();
    }
    // A line `width` pixels wide from the column `t` falls in, where that
    // column is shown.
    let span = |t: f64, width: usize| {
        let c = cols.of(t);
        if c < 0 || c >= w as i64 {
            return 0..0;
        }
        c as usize..(c as usize + width).min(w)
    };
    for &d in &plot.downbeats {
        for x in span(d, line) {
            keys[x].downbeats = keys[x].downbeats.saturating_add(1);
        }
    }
    for (t, width, bit) in [
        (plot.cue, 2 * line, MARK_CUE),
        (plot.cursor, line, MARK_CURSOR),
        (plot.edit, 2 * line, MARK_EDIT),
        (plot.playhead.filter(|_| head), 2 * line, MARK_PLAYHEAD),
    ] {
        if let Some(t) = t {
            for x in span(t, width) {
                keys[x].marks |= bit;
            }
        }
    }
    if !plot.gain.is_empty() {
        // The fade's row at column `x` (-1: the one before the first, so
        // the line joins it as it would were it shown: a tile, a view
        // scrolled, draw it the same).
        let row_at = |x: f64| {
            let t = cols.time(x + 0.5);
            if !plot.transition.is_some_and(|(a, b)| t >= a && t <= b) {
                return None;
            }
            let g = f64::from(plot.gain_at(t).clamp(0.0, 1.0));
            (((1.0 - g) * (full - 1) as f64).round() as usize)
                .checked_sub(above)
                .filter(|&y| y < h)
        };
        let mut last: Option<usize> = row_at(-1.0);
        for (x, key) in keys.iter_mut().enumerate() {
            let Some(y) = row_at(x as f64) else {
                last = None;
                continue;
            };
            key.curve = Some(last.map_or((y, y), |p| (p.min(y), p.max(y))));
            last = Some(y);
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::lanes::WordLines;
    use jodel_rhythm::waveform;

    /// The overview shades the track outside the span the lanes show; with
    /// the whole track shown, nothing.
    #[test]
    fn the_overview_shades_the_track_outside_the_span_shown() {
        // 100 s over 100 columns, the lanes showing 40 to 60 s.
        let frames = (100.0 * waveform::FPS) as usize;
        let lane = waveform::Lane {
            peak: vec![200; frames],
            rms: vec![150; frames],
        };
        let plot = WavePlot {
            area: Rect::new(0, 0, 100, 1),
            waveform: Arc::new(Waveform {
                mix: lane,
                stems: None,
                melody: None,
            }),
            lane: Lane::Song,
            start: 0.0,
            end: 100.0,
            color: colors::DIM,
            downbeats: Vec::new(),
            unclear: Vec::new(),
            selection: None,
            shown: Some((40.0, 60.0)),
            playhead: None,
            cursor: None,
            cue: None,
            transition: None,
            gain: Vec::new(),
            edit: None,
            rows: (0, 1),
            lyrics: None,
            text_scale: 1,
            lines: WordLines::ONE,
            follow: None,
            deep: false,
            key: None,
        };
        let keys = |plot: &WavePlot| {
            let cols = Columns::new(plot.start, plot.end, 100);
            wave_keys(plot, cols, 100, 4, 1, 0, 4, false)
        };
        let shaded = keys(&plot);
        assert_eq!(shaded[50].back, Back::Shown);
        for x in [5, 30, 70, 95] {
            assert_eq!(shaded[x].back, Back::Outside, "column {x}");
        }
        // The whole track shown: nothing shaded.
        let whole = WavePlot {
            shown: None,
            ..plot
        };
        assert_ne!(keys(&whole)[5].back, Back::Outside);
    }
}
