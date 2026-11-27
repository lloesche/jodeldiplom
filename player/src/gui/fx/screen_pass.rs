//! A screen pass along the screen's edges: the scaffolding every one
//! shares (the fire at the edges, the fog). How deep it reaches in from
//! each edge ([`Edges`]: its strips, [`Edges::strips`], and how deep a
//! pixel is, [`Edges::depth`]), how it keeps what the singer reads
//! ([`KeepClear`]), and its strips drawn ([`draw_strips`]); on the
//! processor here, and in WGSL ([`WGSL`]: `fx_edge_depth`,
//! `fx_keep_clear`) once in a shader that carries any effect's code.

use super::helpers::smoothstep;
use super::{At, Clear, Live, Out};
use crate::gui::stage::gpu::Instance;

/// How deep a screen pass along the screen's edges reaches in from each:
/// at the bottom as deep as it is, up the sides `side` of that, along the
/// top `top` of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Edges {
    /// Up the sides, a share of the bottom's.
    pub side: f32,
    /// Along the top, a share of the bottom's.
    pub top: f32,
}

impl Edges {
    /// The strips of `area` (left, top, width, height) a pass reaching
    /// `reach` pixels in at the bottom draws in (two more, for its edge),
    /// none over another: along the bottom and the top the whole width, up
    /// the sides between them.
    pub(super) fn strips(self, (x, y, w, h): (f32, f32, f32, f32), reach: f32) -> Vec<[f32; 4]> {
        let reach = reach + 2.0;
        let bottom = reach.min(h);
        let top = (reach * self.top).min(h - bottom);
        let side = (reach * self.side).min(w / 2.0);
        let (y0, y1) = (y + top, y + h - bottom);
        let mut strips = vec![[x, y1, x + w, y + h]];
        if top > 0.0 {
            strips.push([x, y, x + w, y0]);
        }
        if y1 > y0 {
            strips.push([x, y0, x + side, y1]);
            strips.push([x + w - side, y0, x + w, y1]);
        }
        strips
    }

    /// How deep pixel `px` of a frame `size` (pixels) is in from the
    /// nearest edge, a share of how deep the pass reaches there (`deep`
    /// pixels at the bottom).
    pub(super) fn depth(self, px: (f32, f32), size: (f32, f32), deep: f32) -> f32 {
        ((size.1 - px.1) / deep)
            .min(px.0 / (deep * self.side))
            .min((size.0 - px.0) / (deep * self.side))
            .min(px.1 / (deep * self.top))
    }
}

/// How a screen pass keeps what the singer reads: how far round the lyric
/// band and what is kept clear ([`Clear`]) it thins (`feather`, a share of
/// the frame's height), and how much of it shows inside the band at most
/// (`in_band`); nothing of it inside what is kept clear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct KeepClear {
    /// How far round it thins.
    pub feather: f32,
    /// How much shows inside the band.
    pub in_band: f32,
}

/// What is kept at a pixel ([`KeepClear::keep`]): how far inside the
/// lyric band or a box kept clear it is (0 to 1, the most of them,
/// feathered), and how much of the pass shows there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Kept {
    /// How far inside.
    pub inside: f32,
    /// How much shows.
    pub keep: f32,
}

impl KeepClear {
    /// What is kept at pixel `px` of a frame `high` pixels high, the lyric
    /// band's top and bottom `band` (none where the bottom is not below the
    /// top), what is kept `clear` (each box none where it is not wider than
    /// nothing).
    pub(super) fn keep(self, px: (f32, f32), high: f32, band: (f32, f32), clear: &Clear) -> Kept {
        let f = high * self.feather;
        let across =
            |a: f32, b: f32, v: f32| smoothstep(a - f, a, v) * (1.0 - smoothstep(b, b + f, v));
        let in_box = |r: [f32; 4]| {
            across(r[0], r[2], px.0) * across(r[1], r[3], px.1) * f32::from(u8::from(r[2] > r[0]))
        };
        let in_band = across(band.0, band.1, px.1) * f32::from(u8::from(band.1 > band.0));
        let kept = clear.boxes().into_iter().map(in_box).fold(0.0, f32::max);
        let inside = in_band.max(kept);
        Kept {
            inside,
            keep: (1.0 - (1.0 - self.in_band) * inside) * (1.0 - kept),
        }
    }
}

/// A screen pass along the screen's edges drawn at `at` (its slot,
/// [`At::Screen`]) as `live` is: its shapes of `kind` at its strength, one
/// a strip of the area reaching `reach(height)` pixels in at the bottom
/// ([`Edges::strips`]), the lyric band's top and bottom in `shape[0..2]`,
/// what it keeps clear as [`Clear::on`] puts it; nothing at no strength.
pub(super) fn draw_strips(
    kind: f32,
    edges: Edges,
    (at, live, out): (&At, &Live, &mut Out),
    reach: impl FnOnce(f32) -> f32,
) {
    let &At::Screen {
        area, band, clear, ..
    } = at
    else {
        return;
    };
    if live.strength <= 0.0 {
        return;
    }
    let strips = edges.strips(area, reach(area.3));
    out.shapes.extend(strips.into_iter().map(|quad| {
        clear.on(Instance {
            quad,
            shape: [band.0, band.1, 0.0, 0.0],
            params: [kind, live.strength, 0.0, 0.0],
            ..Instance::default()
        })
    }));
}

/// [`Edges::depth`] and [`KeepClear::keep`] in WGSL.
pub const WGSL: &str = r"
// How deep `px` is in from the frame's nearest edge, a share of how deep
// a pass reaches there: `deep` at the bottom, `side` of it up the sides,
// `top` of it along the top.
fn fx_edge_depth(px: vec2<f32>, size: vec2<f32>, deep: f32, side: f32, top: f32) -> f32 {
    return min(min((size.y - px.y) / deep, px.x / (deep * side)), min((size.x - px.x) / (deep * side), px.y / (deep * top)));
}

fn fx_across(a: f32, b: f32, v: f32, f: f32) -> f32 {
    return smoothstep(a - f, a, v) * (1.0 - smoothstep(b, b + f, v));
}

fn fx_in_box(r: vec4<f32>, px: vec2<f32>, f: f32) -> f32 {
    return fx_across(r.x, r.z, px.x, f) * fx_across(r.y, r.w, px.y, f) * select(0.0, 1.0, r.z > r.x);
}

// What a screen pass keeps at `px` of a frame `high` pixels high, the
// lyric band `band` and the boxes `clear` kept clear, thinning `feather`
// of the height round them, `in_band` of it showing in the band at most:
// how far inside them it is, and how much of the pass shows.
fn fx_keep_clear(px: vec2<f32>, high: f32, band: vec2<f32>, clear: mat3x4<f32>, feather: f32, in_band: f32) -> vec2<f32> {
    let f = high * feather;
    let banded = fx_across(band.x, band.y, px.y, f) * select(0.0, 1.0, band.y > band.x);
    let kept = max(max(fx_in_box(clear[0], px, f), fx_in_box(clear[1], px, f)), fx_in_box(clear[2], px, f));
    let inside = max(banded, kept);
    return vec2<f32>(inside, (1.0 - (1.0 - in_band) * inside) * (1.0 - kept));
}
";
