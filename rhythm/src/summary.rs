//! Tempo from beats: per beat, and for the whole track.

use crate::stats::median;

/// A track's tempo in one look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    /// The tempo: from a straight-line fit through all beats when the
    /// tempo is steady, else the median of the local tempi.
    pub bpm: f64,
    /// The 10th percentile of the local tempi.
    pub low: f64,
    /// The 90th percentile.
    pub high: f64,
    /// Whether the local tempi stay within 3 % of each other (between
    /// the percentiles).
    pub steady: bool,
}

/// Beats on each side whose intervals give a beat's local tempo.
const SPAN: usize = 4;

/// Each beat's local tempo in beats per minute: from the median of the
/// intervals within four beats of it. Empty for fewer than two beats.
pub fn local_bpm(beats: &[f64]) -> Vec<f64> {
    if beats.len() < 2 {
        return Vec::new();
    }
    let intervals: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    (0..beats.len())
        .map(|i| {
            let lo = i.saturating_sub(SPAN);
            let hi = (i + SPAN).min(intervals.len());
            let mut near: Vec<f64> = intervals[lo..hi].to_vec();
            60.0 / median(&mut near).expect("intervals near every beat")
        })
        .collect()
}

/// The summary of the `trusted` ones of `beats` (with their
/// [`local_bpm`]); `None` below eight trusted beats. A steady tempo comes
/// from straight-line fits through each run of at least eight trusted
/// beats in a row whose local tempo is near the median (so a stretch at
/// another tempo level does not bend the fit), weighted by their lengths.
pub fn summarize(beats: &[f64], local: &[f64], trusted: &[bool]) -> Option<Summary> {
    if local.len() != beats.len() || trusted.len() != beats.len() {
        return None;
    }
    let mut sorted: Vec<f64> = local
        .iter()
        .zip(trusted)
        .filter(|(_, &t)| t)
        .map(|(&l, _)| l)
        .collect();
    if sorted.len() < 8 {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    let (low, mid, high) = (at(0.1), at(0.5), at(0.9));
    let steady = (high - low) / mid < 0.03;
    let fits = |i: usize| trusted[i] && (local[i] / mid - 1.0).abs() < 0.03;
    let (mut weighted, mut total) = (0.0, 0.0);
    let mut start = 0;
    while start < beats.len() {
        if !fits(start) {
            start += 1;
            continue;
        }
        let end = (start..beats.len())
            .find(|&i| !fits(i))
            .unwrap_or(beats.len());
        let run = &beats[start..end];
        if run.len() >= 8 {
            weighted += slope(run) * run.len() as f64;
            total += run.len() as f64;
        }
        start = end;
    }
    let bpm = if steady && total > 0.0 {
        60.0 / (weighted / total)
    } else {
        mid
    };
    Some(Summary {
        bpm,
        low,
        high,
        steady,
    })
}

/// The least-squares slope of beat time against beat number.
fn slope(beats: &[f64]) -> f64 {
    let n = beats.len() as f64;
    let mean_i = (n - 1.0) / 2.0;
    let mean_t = beats.iter().sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &t) in beats.iter().enumerate() {
        let di = i as f64 - mean_i;
        num += di * (t - mean_t);
        den += di * di;
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_and_varying_tempi() {
        // 127.9 BPM with ±5 ms of jitter.
        let beats: Vec<f64> = (0..200)
            .map(|i| i as f64 * 60.0 / 127.9 + if i % 3 == 0 { 0.005 } else { -0.003 })
            .collect();
        let all = vec![true; 200];
        let s = summarize(&beats, &local_bpm(&beats), &all).unwrap();
        assert!(s.steady);
        assert!((s.bpm - 127.9).abs() < 0.05, "{}", s.bpm);
        // Untrusted beats at another tempo do not count.
        let mut mixed = beats.clone();
        for (i, b) in mixed.iter_mut().enumerate().take(50) {
            *b = i as f64 * 0.25;
        }
        let shift = mixed[50] - mixed[49] - 60.0 / 127.9;
        for b in mixed.iter_mut().skip(50) {
            *b -= shift;
        }
        let trusted: Vec<bool> = (0..200).map(|i| i >= 55).collect();
        let s = summarize(&mixed, &local_bpm(&mixed), &trusted).unwrap();
        assert!(s.steady && (s.bpm - 127.9).abs() < 0.05, "{s:?}");

        let mut t = 0.0;
        let mut ramp = Vec::new();
        for i in 0..200 {
            ramp.push(t);
            t += 60.0 / (90.0 + 0.2 * i as f64);
        }
        let local = local_bpm(&ramp);
        let s = summarize(&ramp, &local, &all).unwrap();
        assert!(!s.steady);
        assert!(s.low > 90.0 && s.low < 96.0 && s.high > 124.0 && s.high < 130.0);
        assert!((local[100] - 110.0).abs() < 0.5, "{}", local[100]);
        assert!(summarize(&ramp[..7], &local[..7], &all[..7]).is_none());
    }
}
