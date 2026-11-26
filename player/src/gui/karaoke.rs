//! Karaoke mode in pixels, for a window without a GPU (with one, the
//! stage draws it all: [`super::stage`]): the cover blurred over the whole
//! screen under a dark layer, the lyrics in large type over it, filling as
//! they are sung, the notes under the line sung ([`super::notes`]), and
//! the count-in's numbers shrinking and fading on their beats; the score
//! top right, rolling up as an odometer does. Where each goes is the
//! stage's ([`lyrics::set`], [`count::placed`], [`score::placed`]). The
//! ground under them is [`ground`]'s.

pub mod ground;

use super::big_text::{Face, Faces};
use super::notes::SungOp;
use super::pixels::{dim, union, PixelRect, Target};
use super::stage::{count, lyrics, score};
use crate::view::karaoke::KaraokePicture;
use crate::view::palette;

pub use ground::Backdrop;

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
    /// The pixels it is kept to, where it is (a figure of the score in its
    /// slot).
    clip: Option<PixelRect>,
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
                    lyrics::NOTES_BOTTOM
                }
            + pad)
            .min((bounds.1 + bounds.3) as f32);
        if x1 <= x0 || y1 <= y0 {
            return (0, 0, 0, 0);
        }
        let (x0, y0) = (x0.floor() as usize, y0.floor() as usize);
        let rect = (x0, y0, x1.ceil() as usize - x0, y1.ceil() as usize - y0);
        match self.clip {
            Some(clip) => intersect(rect, clip).unwrap_or((0, 0, 0, 0)),
            None => rect,
        }
    }
}

/// What Karaoke mode's picture drew last: each piece of text and the
/// pixels it may cover. A frame redraws only the pieces that changed (a
/// line filling as it is sung), with the backdrop put back under them.
/// And the score's counter ([`Layer::odometer`]), stepped as each frame is
/// painted by the time since the last ([`Layer::at`]).
#[derive(Default)]
pub struct Layer {
    drawn: Vec<(TextOp, PixelRect)>,
    /// The score as it rolls up.
    pub odometer: score::Odometer,
    /// When the frame painted next is, and the seconds since the one
    /// before (none set: none).
    at: Option<std::time::Instant>,
    dt: f32,
}

impl Layer {
    /// The frame painted next is at `now`: the score's counter steps by the
    /// time since the last.
    pub fn at(&mut self, now: std::time::Instant) {
        self.dt = self.at.map_or(0.0, |then| {
            now.saturating_duration_since(then).as_secs_f32()
        });
        self.at = Some(now);
    }
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
    let points = picture.score.map(|s| s.points);
    layer.odometer.step(points, std::mem::take(&mut layer.dt));
    let ops = text_ops(faces, picture, area, layer);
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

/// The pieces of text of `picture` in `area` (pixels): its lines (moving
/// into their places while they scroll), the count-in's number and the
/// score (as `layer`'s counter shows it), each where the stage sets it.
fn text_ops(
    faces: &mut Faces,
    picture: &KaraokePicture,
    (ax, ay, aw, ah): PixelRect,
    layer: &Layer,
) -> Vec<TextOp> {
    let area = (ax as f32, ay as f32, aw as f32, ah as f32);
    let mut ops = Vec::new();
    let text = faces.get(Face::LINE);
    for s in lyrics::set(text, picture, |_| true, area) {
        let sung = SungOp::of(s.line, s.px, |c| s.x_of(text, c));
        ops.push(TextOp {
            text: s.line.text.clone(),
            face: Face::LINE,
            base: s.base,
            px: s.px,
            at: s.at,
            fill_x: s.fill_x,
            colors: s.colors,
            alpha: s.alpha,
            sung,
            clip: None,
        });
    }
    let accent = palette::ACCENT;
    let shout = faces.get(Face::SHOUT);
    let base = |px: f32, _: f32| base_size(px);
    if let Some(c) = count::placed(shout, picture, area, area.3, base) {
        ops.push(TextOp {
            text: c.text,
            face: Face::SHOUT,
            base: c.base,
            px: c.px,
            at: c.at,
            fill_x: f32::MAX,
            colors: (accent, accent),
            alpha: c.alpha,
            sung: None,
            clip: None,
        });
    }
    // The score, top right: the level under it, and the points rolling
    // up, last (however many pieces they are, those before keep their
    // places in the list: they are not drawn again for it).
    if let Some(points) = picture.score {
        let placed = score::placed(faces, points, &layer.odometer.reading(), area);
        if let Some((label, px, at)) = placed.rung {
            ops.push(TextOp {
                text: label,
                face: Face::SHOUT,
                base: px,
                px,
                at,
                fill_x: f32::MAX,
                colors: (palette::DIM, palette::DIM),
                alpha: 1.0,
                sung: None,
                clip: None,
            });
        }
        let color = placed.color;
        ops.extend(placed.figures.into_iter().map(|f| TextOp {
            text: f.text,
            face: Face::NUMBER,
            base: f.base,
            px: f.px,
            at: f.at,
            fill_x: f32::MAX,
            colors: (color, color),
            alpha: f.alpha,
            sung: None,
            clip: Some(f.slot),
        }));
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::super::big_text::BigText;
    use super::super::pixels::{pack, unpack};
    use super::*;
    use crate::art::Art;
    use crate::view::karaoke::{LineKind, LyricLine};
    use std::sync::Arc;

    /// Over the GPU's ground, a frame's words are premultiplied: each
    /// word's own colour with as much of what is under added as its top
    /// byte says comes out as the frame painted over that ground itself
    /// (a line faded, the line sung, the count-in, the controls' bar).
    #[test]
    fn a_frame_over_the_see_through_ground_is_the_frame_over_the_ground() {
        use super::super::pixels::SEE_THROUGH;
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
            bar: 3.0 / 24.0,
            controls: Some(1.0),
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
            bar: 5.0 / 24.0,
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

    /// The score and its rung (a long score, the top rung's name) lie in
    /// the score's corner, which the effects over the frame keep clear.
    #[test]
    fn the_score_lies_in_its_corner() {
        let mut faces = Faces::default();
        let (w, h) = (1600usize, 900usize);
        let mut layer = Layer::default();
        for _ in 0..200 {
            layer.odometer.step(Some(987_654_321), 0.05);
        }
        let p = KaraokePicture {
            score: Some(crate::view::karaoke::Score {
                points: 987_654_321,
                streak: 60,
            }),
            ..picture(Vec::new(), Vec::new(), 1.0)
        };
        let ops = text_ops(&mut faces, &p, (0, 0, w, h), &layer);
        let corner = score::corner((0.0, 0.0, w as f32, h as f32));
        assert!(ops.len() > 9, "{}", ops.len());
        for op in &ops {
            let width = faces.get(op.face).measure_from(&op.text, op.base, op.px);
            let (x, y) = op.at;
            assert!(x >= corner[0] && x + width <= corner[2], "{}: {x}", op.text);
            assert!(y >= corner[1] && y + op.px <= corner[3], "{}: {y}", op.text);
        }
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
