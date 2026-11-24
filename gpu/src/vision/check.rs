//! What the networks are checked on and with: a generated scene (a figure
//! standing on a checkered floor under a sky) and the measures an output is
//! held to its reference by.

use super::prep::Rgb;

/// A generated square scene of `side` pixels: a sky, a checkered floor
/// going off to the horizon, and a figure (a head on a body) in front.
pub fn scene(side: usize) -> Rgb {
    let mut data = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            let (u, v) = (
                (x as f32 + 0.5) / side as f32,
                (y as f32 + 0.5) / side as f32,
            );
            let horizon = 0.55;
            let mut rgb = if v < horizon {
                let t = v / horizon;
                [60.0 + 150.0 * t, 110.0 + 110.0 * t, 200.0 + 40.0 * t]
            } else {
                // The floor: depth grows as 1 / (v - horizon).
                let d = 0.08 / (v - horizon + 0.01);
                let fx = (u - 0.5) * d * 6.0;
                let fz = d * 6.0;
                let check = ((fx.floor() + fz.floor()) as i64).rem_euclid(2) == 0;
                let fade = (1.0 / (1.0 + d)).clamp(0.2, 1.0);
                let base = if check { 200.0 } else { 70.0 };
                [
                    base * fade + 40.0,
                    base * fade + 30.0,
                    base * fade * 0.8 + 20.0,
                ]
            };
            // The figure: a body (an ellipse) and a head (a circle), lit
            // from the upper left.
            let body = ((u - 0.5) / 0.16).powi(2) + ((v - 0.68) / 0.24).powi(2);
            let head = ((u - 0.5) / 0.085).powi(2) + ((v - 0.36) / 0.085).powi(2);
            for (r, centre, colour) in [
                (body, (0.5, 0.68), [190.0, 50.0, 60.0]),
                (head, (0.5, 0.36), [230.0, 180.0, 140.0]),
            ] {
                if r < 1.0 {
                    let nz = (1.0 - r).sqrt();
                    let light = 0.35
                        + 0.65 * (nz * 0.8 + (centre.0 - u + centre.1 - v) * 1.5).clamp(0.0, 1.0);
                    rgb = colour.map(|c: f32| c * light);
                }
            }
            data.extend(rgb.map(|c| c.round().clamp(0.0, 255.0) as u8));
        }
    }
    Rgb::new(side, side, data)
}

/// How an output compares with its reference.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Pearson's correlation.
    pub correlation: f64,
    /// The largest and the mean absolute difference.
    pub max_abs: f64,
    pub mean_abs: f64,
    /// The same, over the reference's range (its 1st to 99th percentile).
    pub max_rel: f64,
    pub mean_rel: f64,
}

/// `ours` against `reference` (the same length).
pub fn stats(ours: &[f32], reference: &[f32]) -> Stats {
    assert_eq!(ours.len(), reference.len());
    let n = ours.len() as f64;
    let mean = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (ma, mb) = (mean(ours), mean(reference));
    let (mut sab, mut saa, mut sbb, mut max, mut sum) = (0.0, 0.0, 0.0, 0.0f64, 0.0);
    for (&a, &b) in ours.iter().zip(reference) {
        let (a, b) = (f64::from(a), f64::from(b));
        sab += (a - ma) * (b - mb);
        saa += (a - ma) * (a - ma);
        sbb += (b - mb) * (b - mb);
        max = max.max((a - b).abs());
        sum += (a - b).abs();
    }
    let mut sorted: Vec<f32> = reference.to_vec();
    sorted.sort_by(f32::total_cmp);
    let pick = |q: f64| f64::from(sorted[((sorted.len() - 1) as f64 * q) as usize]);
    let range = (pick(0.99) - pick(0.01)).max(1e-12);
    Stats {
        correlation: sab / (saa * sbb).sqrt().max(1e-30),
        max_abs: max,
        mean_abs: sum / n,
        max_rel: max / range,
        mean_rel: sum / n / range,
    }
}

/// `values` (`h × w`) averaged over blocks of `k × k` (the sides multiples
/// of `k`): a reference small enough to keep beside the tests.
pub fn pooled(values: &[f32], h: usize, w: usize, k: usize) -> Vec<f32> {
    let (ph, pw) = (h / k, w / k);
    let mut out = vec![0.0f32; ph * pw];
    for y in 0..ph * k {
        for x in 0..pw * k {
            out[(y / k) * pw + x / k] += values[y * w + x];
        }
    }
    let n = (k * k) as f32;
    out.iter_mut().for_each(|v| *v /= n);
    out
}
