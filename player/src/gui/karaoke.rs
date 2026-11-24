//! Karaoke mode in pixels: the cover blurred over the whole screen under a
//! dark layer, the lyrics in large type over it, filling as they are sung,
//! and the count-in's numbers shrinking and fading on their beats. The
//! ground under them is [`ground`]'s.

pub mod ground;

use super::big_text::{BigText, Face, Faces};
use super::notes::SungOp;
use super::raster::{dim, mix, union, PixelRect, Target};
use crate::view::colors;
use crate::view::karaoke::{line_color, KaraokePicture, LineKind, LyricLine, HIT};

pub use ground::Backdrop;

pub(super) fn rgb(color: ratatui::style::Color) -> [u8; 3] {
    match color {
        ratatui::style::Color::Rgb(r, g, b) => [r, g, b],
        _ => [200, 200, 200],
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
    let sizes: Vec<(f32, bool)> = lines
        .iter()
        .map(|l| {
            let px = size_of(l.kind, big);
            let w = text.measure(&l.text, px);
            let px = if w > widest && w > 0.0 {
                (px * widest / w).max(10.0).round()
            } else {
                px.round()
            };
            (px, noted(l))
        })
        .collect();
    stack(&sizes, top, room)
}

/// A line's size by what it is, `big` the line sung's.
fn size_of(kind: LineKind, big: f32) -> f32 {
    match kind {
        LineKind::Current => big,
        LineKind::Next | LineKind::Previous => big * 0.62,
        LineKind::Info => big * 0.42,
        LineKind::Warning => big * 0.36,
    }
}

/// The lines of `sizes` (each its size, and whether it has room for notes
/// under it) stacked and centred in `room` rows of pixels from `top`.
fn stack(sizes: &[(f32, bool)], top: usize, room: usize) -> Vec<Slot> {
    let below = |(px, noted): (f32, bool)| if noted { px * NOTES_ROOM } else { 0.0 };
    let total: f32 = sizes.iter().map(|&s| gap(s.0) + below(s)).sum();
    let mut y = top as f32 + (room as f32 - total).max(0.0) / 2.0;
    sizes
        .iter()
        .map(|&s| {
            let slot = Slot {
                px: s.0,
                y: y + (gap(s.0) - s.0) / 2.0,
            };
            y += gap(s.0) + below(s);
            slot
        })
        .collect()
}

/// The line sung's size in a picture `height` pixels high.
fn big_of(height: usize) -> f32 {
    (height as f32 / 11.0).clamp(20.0, 160.0).round()
}

/// Where the lyrics in large type are on a screen `height` pixels high
/// (the line before, the line sung with room for its notes, the two
/// coming), stacked as [`slots`] stacks them: their top and bottom, shares
/// of the height.
pub fn text_band(height: usize) -> [f32; 2] {
    let big = big_of(height);
    let kinds = [
        LineKind::Previous,
        LineKind::Current,
        LineKind::Next,
        LineKind::Next,
    ];
    let sizes: Vec<(f32, bool)> = kinds
        .iter()
        .map(|&k| (size_of(k, big).round(), k == LineKind::Current))
        .collect();
    let slots = stack(&sizes, 0, height);
    let h = height.max(1) as f32;
    let (first, last) = (slots[0], slots[slots.len() - 1]);
    [first.y / h, (last.y + last.px * 1.3) / h]
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
    face: Face,
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
    fn rect(&self, width: f32, bounds: PixelRect) -> PixelRect {
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
    drawn: Vec<(TextOp, PixelRect)>,
    /// What the GPU draws over the picture; whether there is one to (set
    /// it before painting: the bars' fill is painted only without).
    pub over: super::notes::Over,
}

/// Where `a` and `b` meet, if they do.
fn intersect(a: PixelRect, b: PixelRect) -> Option<PixelRect> {
    let (x0, y0) = (a.0.max(b.0), a.1.max(b.1));
    let (x1, y1) = ((a.0 + a.2).min(b.0 + b.2), (a.1 + a.3).min(b.1 + b.3));
    (x1 > x0 && y1 > y0).then(|| (x0, y0, x1 - x0, y1 - y0))
}

fn overlaps(a: PixelRect, b: PixelRect) -> bool {
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
/// backdrop, its text in Karaoke mode's faces (`faces`: [`Face::LINE`],
/// the score and the count-in [`Face::NUMBER`] and [`Face::SHOUT`]); the
/// controls' bar (pixels) darker still. `full`: the whole area afresh;
/// otherwise only the text that changed since `layer`'s last frame, the
/// backdrop put back under it. Returns the pixels changed.
#[allow(clippy::too_many_arguments)]
pub fn paint(
    target: &mut Target,
    faces: &mut Faces,
    layer: &mut Layer,
    backdrop: &Backdrop,
    picture: &KaraokePicture,
    area: PixelRect,
    bar: Option<PixelRect>,
    full: bool,
) -> Vec<PixelRect> {
    let (ax, ay, aw, ah) = area;
    // The lyrics, centred on the whole screen (the controls never move
    // them), and drawn only above the controls' bar while it shows over
    // them.
    let room = bar.map_or(ah, |(_, by, _, _)| by.saturating_sub(ay));
    let lyrics = (ax, ay, aw, room);
    let (ops, over) = text_ops(faces, picture, (area, ah), layer.over.gpu);
    layer.over = over;
    let now: Vec<(TextOp, PixelRect)> = ops
        .into_iter()
        .map(|op| {
            let w = faces.get(op.face).measure_from(&op.text, op.base, op.px);
            let rect = op.rect(w, lyrics);
            (op, rect)
        })
        .collect();
    let put_back = |target: &mut Target, (rx, ry, rw, rh): PixelRect| {
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
    let draw_in =
        |target: &mut Target, faces: &mut Faces, ops: &[(TextOp, PixelRect)], clip: PixelRect| {
            for (op, rect) in ops {
                // Only within what the op may cover, so a frame drawn where it
                // changed is the frame drawn afresh.
                if let Some(clip) = intersect(*rect, clip) {
                    faces.get(op.face).draw(
                        target, &op.text, op.base, op.px, op.at, op.fill_x, op.colors, op.alpha,
                        clip,
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
                    target.pixels[i] = dim(target.pixels[i], 0.4);
                }
            }
        }
        draw_in(target, faces, &now, lyrics);
        vec![area]
    } else {
        // The pieces that differ from last frame's at the same place in
        // the list (or are new, or gone): where they were and are.
        let n = now.len().max(layer.drawn.len());
        let mut dirty: Vec<PixelRect> = Vec::new();
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
        let mut merged: Vec<PixelRect> = Vec::new();
        for rect in dirty {
            let mut rect = rect;
            while let Some(i) = merged.iter().position(|&m| overlaps(m, rect)) {
                rect = union(Some(merged.swap_remove(i)), rect);
            }
            merged.push(rect);
        }
        for &rect in &merged {
            put_back(target, rect);
            draw_in(target, faces, &now, rect);
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
    faces: &mut Faces,
    picture: &KaraokePicture,
    (area, room): (PixelRect, usize),
    gpu: bool,
) -> (Vec<TextOp>, super::notes::Over) {
    let (ax, ay, aw, ah) = area;
    let big = big_of(ah);
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
                face: text.face(),
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
    let now = slots(faces.get(Face::LINE), &picture.lines, aw, ay, room, big);
    let moving = picture.moved < 1.0 && !picture.before.is_empty();
    let then = if moving {
        slots(faces.get(Face::LINE), &picture.before, aw, ay, room, big)
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
            (false, _) => line_op(
                faces.get(Face::LINE),
                line,
                slot.px,
                slot,
                rest,
                faded(line.kind),
            ),
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
                line_op(
                    faces.get(Face::LINE),
                    line,
                    old.px.max(slot.px),
                    at,
                    color,
                    alpha,
                );
            }
            // New: rising into its place as it fades in.
            (true, None) => {
                let at = Slot {
                    y: slot.y + (1.0 - m) * gap(slot.px) * 0.5,
                    ..slot
                };
                line_op(
                    faces.get(Face::LINE),
                    line,
                    slot.px,
                    at,
                    rest,
                    faded(line.kind) * m,
                );
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
                line_op(
                    faces.get(Face::LINE),
                    gone,
                    old.px,
                    at,
                    rest,
                    faded(gone.kind) * (1.0 - m),
                );
            }
        }
    }
    // The score, top right: the points, the level under them.
    if let Some(score) = picture.score {
        let px = (big * 0.42).round().max(12.0);
        let points = crate::view::karaoke::points_text(score.points);
        let w = faces.get(Face::NUMBER).measure_from(&points, px, px);
        let margin = px * 0.8;
        ops.push(TextOp {
            text: points,
            face: Face::NUMBER,
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
            let lw = faces.get(Face::SHOUT).measure_from(&times, small, small);
            ops.push(TextOp {
                text: times,
                face: Face::SHOUT,
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
    // The count-in: its number in the lane behind the playhead, where the
    // singing starts and no bubble is yet (clear of the meter and badges
    // above the lane and of the lines below it; it never moves with
    // them), shrinking and fading over its beat; the one before the
    // singing comes back after a break half as large.
    let count = picture
        .count
        .map(|c| (c, 0.26))
        .or(picture.cue.map(|c| ((c.number, c.through), 0.13)));
    if let Some(((n, through), size)) = count {
        let lane = super::stage::Lane::new((ax as f32, ay as f32, aw as f32, ah as f32), 0.0);
        let behind = lane.head - ax as f32;
        // Smaller at once (ease-out), fading late, so it reads through
        // most of its beat.
        let t = through.clamp(0.0, 1.0) as f32;
        let shrink = 1.0 - (1.0 - t).powi(3);
        let most = (room as f32 * size).min(behind * 0.9);
        let px = most * (1.0 - 0.4 * shrink);
        let base = base_size(px);
        let digit = n.to_string();
        let w = faces.get(Face::SHOUT).measure_from(&digit, base, px);
        ops.push(TextOp {
            text: digit,
            face: Face::SHOUT,
            base,
            px,
            at: (
                ax as f32 + (behind - w) / 2.0,
                (lane.band.0 + lane.band.1 - px) / 2.0,
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
    use super::super::raster::{pack, unpack};
    use super::*;
    use crate::art::Art;
    use std::sync::Arc;

    /// Over the GPU's ground, a frame's words are premultiplied: each
    /// word's own colour with as much of what is under added as its top
    /// byte says comes out as the frame painted over that ground itself
    /// (a line faded, the line sung, the count-in, the controls' bar).
    #[test]
    fn a_frame_over_the_see_through_ground_is_the_frame_over_the_ground() {
        use super::super::raster::SEE_THROUGH;
        let (w, h) = (320, 180);
        let ground = [40u8, 70, 110];
        let solid = Backdrop {
            art: Some(1),
            size: (w, h),
            pixels: vec![pack(ground); w * h],
        };
        let lines = vec![
            LyricLine {
                alpha: 0.4,
                ..lyric("before", 1.0, LineKind::Previous, 0)
            },
            lyric("sung now", 0.4, LineKind::Current, 1),
            lyric("next", 0.0, LineKind::Next, 2),
        ];
        let p = KaraokePicture {
            count: Some((2, 0.5)),
            controls: Some(ratatui::layout::Rect::new(0, 21, 80, 3)),
            ..picture(lines, Vec::new(), 1.0)
        };
        let bar = Some((0, 158, w, 22));
        let paint_over = |backdrop: &Backdrop| {
            let mut pixels = vec![0u32; w * h];
            let mut target = Target {
                pixels: &mut pixels,
                width: w,
                height: h,
            };
            let mut faces = Faces::default();
            let mut layer = Layer::default();
            let area = (0, 0, w, h);
            paint(
                &mut target,
                &mut faces,
                &mut layer,
                backdrop,
                &p,
                area,
                bar,
                true,
            );
            pixels
        };
        let opaque = paint_over(&solid);
        let through = paint_over(&Backdrop::through(w, h));
        let mut mixed = 0;
        for (&o, &t) in opaque.iter().zip(&through) {
            let keep = (t >> 24) as f32 / 255.0;
            let own = unpack(t);
            for i in 0..3 {
                let shown = f32::from(own[i]) + keep * f32::from(ground[i]);
                let painted = f32::from(unpack(o)[i]);
                assert!((shown - painted).abs() <= 1.5, "{o:08x} {t:08x}");
            }
            mixed += usize::from(t != SEE_THROUGH && t >> 24 != 0);
        }
        assert!(mixed > 100, "{mixed}");
    }

    /// The lyrics in large type as a picture lays them out lie in the band
    /// the backdrop keeps calm under them.
    #[test]
    fn the_lyrics_lie_in_their_text_band() {
        let mut faces = Faces::default();
        let lines = [
            lyric("the line before it", 1.0, LineKind::Previous, 0),
            lyric("the line sung now", 0.3, LineKind::Current, 1),
            lyric("the next one", 0.0, LineKind::Next, 2),
            lyric("and the one after", 0.0, LineKind::Next, 3),
        ];
        for h in [360usize, 900, 1600] {
            let band = text_band(h);
            let at = slots(faces.get(Face::LINE), &lines, 1600, 0, h, big_of(h));
            let hf = h as f32;
            assert!(
                band[0] < band[1] && band[0] > 0.0 && band[1] < 1.0,
                "{band:?}"
            );
            for s in at {
                assert!(s.y >= band[0] * hf - 1.0, "{h}: {} {band:?}", s.y);
                assert!(s.y + s.px <= band[1] * hf + 1.0, "{h}: {} {band:?}", s.y);
            }
        }
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
            result: None,
        }
    }

    /// The count before the singing comes back sits at the top, the same
    /// whatever lines show or how they move: never part of them.
    #[test]
    fn the_break_count_stays_put_while_the_lines_move() {
        use crate::view::karaoke::Cue;
        let mut text = Faces::default();
        let cue = Some(Cue {
            number: 3,
            through: 0.2,
            line: 1,
        });
        let digit_at = |text: &mut Faces, p: KaraokePicture| {
            let ops = text_ops(text, &p, ((0, 0, 640, 360), 360), false).0;
            let digit: Vec<&TextOp> = ops.iter().filter(|o| o.text == "3").collect();
            assert_eq!(digit.len(), 1);
            let at = digit[0].at;
            // In the lane, behind the playhead.
            let lane = crate::gui::stage::Lane::new((0.0, 0.0, 640.0, 360.0), 0.0);
            assert!(at.0 < lane.head, "behind the playhead: {at:?}");
            assert!(
                at.1 > lane.band.0 && at.1 < lane.band.1,
                "in the lane: {at:?}"
            );
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
        let mut text = Faces::default();
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
                &mut Faces::default(),
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
        let mut text = Faces::default();
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
