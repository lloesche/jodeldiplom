//! A picture's depth as nearness: from a depth model's ([`nearness`]),
//! kept in 16 bits ([`NearMap`]).

use jodel_gpu::vision::{Depth, Matte};

use super::Planes;

/// A depth map: nearness in 0..=65535 (larger nearer: the parallax a pixel
/// moves by), `width × height`, rows top to bottom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NearMap {
    /// Its size.
    pub width: u32,
    /// Its size.
    pub height: u32,
    /// The nearness of each pixel.
    pub near: Vec<u16>,
}

impl From<&Planes> for NearMap {
    /// One channel of nearness in 0..1, kept in 16 bits.
    fn from(p: &Planes) -> NearMap {
        NearMap {
            width: p.width as u32,
            height: p.height as u32,
            near: p
                .data
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
                .collect(),
        }
    }
}

impl From<&NearMap> for Planes {
    /// The nearness in 0..1, one channel.
    fn from(n: &NearMap) -> Planes {
        Planes {
            width: n.width as usize,
            height: n.height as usize,
            channels: 1,
            data: n.near.iter().map(|&v| f32::from(v) / 65535.0).collect(),
        }
    }
}

impl From<Matte> for Planes {
    /// The matte's values, one channel.
    fn from(m: Matte) -> Planes {
        Planes {
            width: m.width,
            height: m.height,
            channels: 1,
            data: m.values,
        }
    }
}

/// Nearness from depth (its reciprocal, the 1st to 99th percentile spread
/// over 0..=1).
pub fn nearness(depth: &Depth) -> Planes {
    let disparity: Vec<f32> = depth.values.iter().map(|&d| 1.0 / d.max(1e-6)).collect();
    let mut sorted = disparity.clone();
    sorted.sort_by(f32::total_cmp);
    let pick = |q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    let (lo, hi) = (pick(0.01), pick(0.99));
    let span = (hi - lo).max(1e-9);
    Planes {
        width: depth.width,
        height: depth.height,
        channels: 1,
        data: disparity
            .iter()
            .map(|&d| ((d - lo) / span).clamp(0.0, 1.0))
            .collect(),
    }
}
