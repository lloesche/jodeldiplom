//! Comparing beat lists.

/// The F-measure of `estimated` against `reference` (both ascending, in
/// seconds): a beat matches when it lies within `tolerance` of a
/// reference beat not matched yet. 1 when both are empty, 0 when only
/// one is.
pub fn f_measure(estimated: &[f64], reference: &[f64], tolerance: f64) -> f64 {
    if estimated.is_empty() && reference.is_empty() {
        return 1.0;
    }
    if estimated.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let matched = matches(estimated, reference, tolerance) as f64;
    let precision = matched / estimated.len() as f64;
    let recall = matched / reference.len() as f64;
    if matched == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// How many beats of `a` match one of `b` each, in order.
fn matches(a: &[f64], b: &[f64], tolerance: f64) -> usize {
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let d = a[i] - b[j];
        if d.abs() <= tolerance {
            count += 1;
            i += 1;
            j += 1;
        } else if d < 0.0 {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

/// Every other beat, starting with beat `phase` (0 or 1): the same grid at
/// half the tempo.
pub fn halved(beats: &[f64], phase: usize) -> Vec<f64> {
    beats.iter().skip(phase).step_by(2).copied().collect()
}

/// The beats with a beat between each two: the grid at double the tempo.
pub fn doubled(beats: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(beats.len() * 2);
    for pair in beats.windows(2) {
        out.push(pair[0]);
        out.push(0.5 * (pair[0] + pair[1]));
    }
    out.extend(beats.last());
    out
}

/// The F-measure two unrelated grids reach by chance: the share of the
/// denser grid's beat interval that the tolerance window covers.
fn chance(a: &[f64], b: &[f64], tolerance: f64) -> f64 {
    let interval = |beats: &[f64]| {
        let mut gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        if gaps.is_empty() {
            return f64::INFINITY;
        }
        gaps.sort_by(f64::total_cmp);
        gaps[gaps.len() / 2]
    };
    (2.0 * tolerance / interval(a).min(interval(b))).min(0.95)
}

/// How much `a` and `b` agree on where the beats are, beyond chance, at
/// the tempo level where they agree best (as it is, halved in either
/// phase, or doubled): the F-measure `f` with the chance level `c` taken
/// out, `(f − c) / (1 − c)`, at least 0. Taken both ways round, so it is
/// symmetric. Without the correction two fast grids would agree on
/// noise, since a tolerance window covers much of a short beat.
pub fn agreement(a: &[f64], b: &[f64], tolerance: f64) -> f64 {
    let corrected = |a: &[f64], b: &[f64]| {
        let c = chance(a, b, tolerance);
        ((f_measure(a, b, tolerance) - c) / (1.0 - c)).max(0.0)
    };
    let one_way = |a: &[f64], b: &[f64]| {
        [
            corrected(a, b),
            corrected(a, &halved(b, 0)),
            corrected(a, &halved(b, 1)),
            corrected(a, &doubled(b)),
        ]
        .into_iter()
        .fold(0.0, f64::max)
    };
    one_way(a, b).max(one_way(b, a))
}

/// The beats within `start..=end`.
pub fn between(beats: &[f64], start: f64, end: f64) -> &[f64] {
    let lo = beats.partition_point(|&b| b < start);
    let hi = beats.partition_point(|&b| b <= end);
    &beats[lo..hi.max(lo)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_measure_and_tempo_levels() {
        let grid: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        assert_eq!(f_measure(&grid, &grid, 0.07), 1.0);
        let late: Vec<f64> = grid.iter().map(|b| b + 0.05).collect();
        assert_eq!(f_measure(&late, &grid, 0.07), 1.0);
        let off: Vec<f64> = grid.iter().map(|b| b + 0.25).collect();
        assert_eq!(f_measure(&off, &grid, 0.07), 0.0);
        let half = halved(&grid, 1);
        assert!((f_measure(&half, &grid, 0.07) - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(agreement(&half, &grid, 0.07), 1.0);
        assert_eq!(agreement(&grid, &half, 0.07), 1.0);
        assert_eq!(agreement(&doubled(&grid), &grid, 0.07), 1.0);
        assert_eq!(between(&grid, 1.0, 2.0), &[1.0, 1.5, 2.0]);
        assert_eq!(f_measure(&[], &[], 0.07), 1.0);
    }
}
