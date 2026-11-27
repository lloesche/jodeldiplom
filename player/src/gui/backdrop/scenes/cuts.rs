//! When each of a story's scenes shows: cut in on the song's bars
//! ([`cuts`]), one after the other through the whole song, and which shows
//! at a moment ([`showing`]).

/// When a scene shows, in the song's seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    /// Its cut in (the song's start for the first).
    pub start: f64,
    /// Its end: the next scene's cut; the last's, its span's end (it stays
    /// on after it, the camera at rest).
    pub end: f64,
}

/// The most a cut comes before its scene's span starts: a bar at most.
pub const EARLY: f64 = 4.0;
/// A bar this little after a span's start is still taken for it.
pub const LATE: f64 = 0.15;
/// The least a scene shows.
pub const LEAST_SCENE: f64 = 2.0;

/// When each of the scenes of `spans` shows, cut on the song's `bars`
/// (the downbeats' times, in order): the first from the song's start; each
/// other in on the bar at or just before its span's start (no more than
/// [`EARLY`] before), at least [`LEAST_SCENE`] after the one before (then
/// on the next bar after that); each on till the next's cut, the last till
/// the bar at or just after its span's end (and on after it). Without a
/// bar near, a span's own times.
pub fn cuts(spans: &[(f64, f64)], bars: &[f64]) -> Vec<Shown> {
    let before = |t: f64| {
        let i = bars.partition_point(|&b| b <= t + LATE);
        i.checked_sub(1)
            .map(|i| bars[i])
            .filter(|&b| b >= t - EARLY)
    };
    let after = |t: f64| {
        let i = bars.partition_point(|&b| b < t - LATE);
        bars.get(i).copied().filter(|&b| b <= t + EARLY)
    };
    let mut ins: Vec<f64> = Vec::with_capacity(spans.len());
    for &(start, end) in spans {
        let at = match ins.last() {
            None => 0.0,
            Some(&last) => {
                let at = before(start).unwrap_or(start);
                let least = last + LEAST_SCENE;
                if at < least {
                    after(least + LATE).filter(|&b| b < end).unwrap_or(least)
                } else {
                    at
                }
            }
        };
        ins.push(at);
    }
    let last = spans
        .last()
        .map_or(0.0, |&(_, end)| after(end).unwrap_or(end));
    (0..ins.len())
        .map(|i| {
            let start = ins[i];
            let end = ins.get(i + 1).copied().unwrap_or(last);
            Shown {
                start,
                end: end.max(start),
            }
        })
        .collect()
}

/// The scene showing at `t` of those `shown`: the last cut in by then (the
/// first before any is); none of none.
pub fn showing(shown: &[Shown], t: f64) -> Option<usize> {
    (!shown.is_empty()).then(|| shown.partition_point(|s| s.start <= t).saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_cut_in_on_the_bar_before_their_span_one_after_the_other() {
        // A bar every two seconds, 0.3 s past.
        let bars: Vec<f64> = (0..30).map(|k| 0.3 + 2.0 * k as f64).collect();
        let spans = [(5.0, 9.0), (9.0, 15.2), (20.0, 25.0)];
        let shown = cuts(&spans, &bars);
        assert_eq!(
            shown,
            vec![
                // The first from the song's start.
                Shown {
                    start: 0.0,
                    end: 8.3
                },
                // On till the next's cut, over the gap.
                Shown {
                    start: 8.3,
                    end: 18.3
                },
                // The last out on the bar after its end.
                Shown {
                    start: 18.3,
                    end: 26.3
                },
            ]
        );
        assert_eq!(showing(&shown, 0.0), Some(0));
        assert_eq!(showing(&shown, 8.2), Some(0));
        assert_eq!(showing(&shown, 8.3), Some(1));
        assert_eq!(showing(&shown, 17.0), Some(1));
        assert_eq!(showing(&shown, 26.0), Some(2));
        assert_eq!(showing(&shown, 400.0), Some(2));
        assert_eq!(showing(&[], 4.0), None);
    }

    #[test]
    fn a_scene_shows_a_while_and_without_bars_its_span() {
        let bars: Vec<f64> = (0..30).map(|k| 0.3 + 2.0 * k as f64).collect();
        // The third span starts right after the second: it would cut on
        // the same bar, so it comes on the next.
        let shown = cuts(&[(1.0, 4.0), (5.0, 6.0), (6.1, 12.0)], &bars);
        assert_eq!(shown[1].start, 4.3);
        assert_eq!(shown[2].start, 6.3);
        assert_eq!(shown[1].end, 6.3);
        assert!(shown[2].start - shown[1].start >= LEAST_SCENE);
        // A bar a little after the span's start is its.
        assert_eq!(cuts(&[(0.0, 1.0), (4.2, 8.0)], &bars)[1].start, 4.3);
        // No bars: the spans themselves, the second held off the first.
        let plain = cuts(&[(1.0, 2.5), (2.5, 7.0)], &[]);
        assert_eq!(
            plain,
            vec![
                Shown {
                    start: 0.0,
                    end: 2.5
                },
                Shown {
                    start: 2.5,
                    end: 7.0
                }
            ]
        );
    }
}
