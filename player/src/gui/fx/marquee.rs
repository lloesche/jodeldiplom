//! The marquee round the lane: where its bulbs sit ([`row`]: a row of
//! them just outside the lane's border, a whole number of gaps to a side,
//! kept inside the screen), how far up their light reaches ([`lit_top`],
//! which what is over the lane keeps above: [`crate::gui::stage::Lane::hud`])
//! and how much they show at a strength ([`shown`]). The bulbs draw it
//! ([`super::bulbs`]), the lane's flames set it on fire
//! ([`super::flames`]), the lane's glow keeps under it ([`super::glow`]).

use super::helpers::sat;

/// A bulb's size: its width, a share of the lane's height.
pub(super) const DIAMETER: f32 = 0.075;
/// Bulbs this many of their widths apart (round the corners, a whole
/// number to a side).
pub(super) const SPACING: f32 = 1.8;
/// A bulb's middle this many of its radii outside the lane.
pub(super) const OUT: f32 = 1.2;
/// A bulb's middle at least this many radii inside the screen.
pub(super) const MARGIN: f32 = 1.3;
/// How far a bulb draws (its glow), in its radii.
pub(super) const REACH: f32 = 3.2;
/// The strength from which the bulbs show wholly (fading in under it).
pub(super) const SHOWN_FROM: f32 = 0.35;

/// The bulbs' row round a lane ([`row`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// The corners of the bulbs' middles: left, top, right, bottom.
    pub ring: [f32; 4],
    /// A bulb's radius.
    pub r: f32,
    /// The gaps between bulbs across the row and down it.
    pub gaps: (i32, i32),
}

/// The bulbs' row round `lane` (left, top, right, bottom) in a screen of
/// `size`, whether they show or not; none where the lane is too low for
/// any or the row has no room.
pub fn row(lane: [f32; 4], size: (f32, f32)) -> Option<Row> {
    let r = (lane[3] - lane[1]) * DIAMETER * 0.5;
    if r <= 0.0 {
        return None;
    }
    let ring = ring(lane, size, r);
    let (w, h) = (ring[2] - ring[0], ring[3] - ring[1]);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let gap = r * 2.0 * SPACING;
    let gaps = (
        ((w / gap + 0.5).floor() as i32).max(1),
        ((h / gap + 0.5).floor() as i32).max(1),
    );
    Some(Row { ring, r, gaps })
}

/// The highest any bulb round `lane` (left, top, right, bottom) in a
/// screen of `size` lights: its top row's middle less its glow's reach
/// (the lane's top where it has no bulbs).
pub fn lit_top(lane: [f32; 4], size: (f32, f32)) -> f32 {
    row(lane, size).map_or(lane[1], |row| row.ring[1] - row.r * REACH)
}

/// How much the bulbs show at `strength` (wholly from a little in).
pub fn shown(strength: f32) -> f32 {
    sat(strength / SHOWN_FROM)
}

/// The row's corners (left, top, right, bottom) round `lane` in a screen
/// of `size`, for bulbs of radius `r`.
fn ring(lane: [f32; 4], size: (f32, f32), r: f32) -> [f32; 4] {
    let (off, m) = (r * OUT, r * MARGIN);
    [
        (lane[0] - off).max(m),
        (lane[1] - off).max(m),
        (lane[2] + off).min(size.0 - m),
        (lane[3] + off).min(size.1 - m),
    ]
}

/// Bulb `k`'s middle, of `nw` gaps across the row and `nh` down it,
/// clockwise from its top left corner.
pub(super) fn bulb_at(k: i32, (nw, nh): (i32, i32), ring: [f32; 4]) -> (f32, f32) {
    let (w, h) = (ring[2] - ring[0], ring[3] - ring[1]);
    let (fw, fh) = (nw as f32, nh as f32);
    if k < nw {
        (ring[0] + k as f32 * w / fw, ring[1])
    } else if k < nw + nh {
        (ring[2], ring[1] + (k - nw) as f32 * h / fh)
    } else if k < 2 * nw + nh {
        (ring[2] - (k - nw - nh) as f32 * w / fw, ring[3])
    } else {
        (ring[0], ring[3] - (k - 2 * nw - nh) as f32 * h / fh)
    }
}
