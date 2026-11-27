//! When each of a story's scenes shows: cut in on the song's bars
//! ([`cuts`]), and which shows at a moment ([`showing`]).

/// When a scene shows, in the song's seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    /// Its cut in.
    pub start: f64,
    /// Its end: the next scene's cut, or the cover's back.
    pub end: f64,
}

/// The most a cut comes before its scene's span starts: a bar at most.
pub const EARLY: f64 = 4.0;
/// A bar this little after a span's start is still taken for it.
pub const LATE: f64 = 0.15;
/// The least a scene shows.
pub const LEAST_SCENE: f64 = 2.0;
/// A scene ending this little before the next starts runs on to it.
pub const JOINED: f64 = 1.0;

/// When each of the scenes of `spans` shows, cut on the song's `bars`
/// (the downbeats' times, in order): each in on the bar at or just before
/// its span's start (no more than [`EARLY`] before), at least
/// [`LEAST_SCENE`] after the one before (then on the next bar after
/// that); each running on to the next where they are [`JOINED`], else out
/// on the bar at or just after its span's end, the cover showing in the
/// gap. Without a bar near, a span's own times.
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
    // Each scene's cut in, and its own end (none: it runs on to the next).
    let mut ins: Vec<(f64, Option<f64>)> = Vec::with_capacity(spans.len());
    for (i, &(start, end)) in spans.iter().enumerate() {
        let mut at = before(start).unwrap_or(start);
        if let Some(&(last, _)) = ins.last() {
            let least = last + LEAST_SCENE;
            if at < least {
                at = after(least + LATE).filter(|&b| b < end).unwrap_or(least);
            }
        }
        let joined = spans.get(i + 1).is_some_and(|n| n.0 - end <= JOINED);
        let own = (!joined).then(|| after(end).unwrap_or(end));
        ins.push((at, own));
    }
    (0..ins.len())
        .map(|i| {
            let (start, own) = ins[i];
            let next = ins.get(i + 1).map(|n| n.0);
            let end = match (own, next) {
                (Some(e), Some(n)) => e.min(n),
                (Some(e), None) => e,
                (None, Some(n)) => n,
                (None, None) => start,
            };
            Shown {
                start,
                end: end.max(start),
            }
        })
        .collect()
}

/// The scene showing at `t` of those `shown` (the last whose span holds
/// it), none in a gap.
pub fn showing(shown: &[Shown], t: f64) -> Option<usize> {
    let i = shown.partition_point(|s| s.start <= t);
    (0..i).rev().find(|&k| t < shown[k].end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_cut_in_on_the_bar_before_their_span() {
        // A bar every two seconds, 0.3 s past.
        let bars: Vec<f64> = (0..30).map(|k| 0.3 + 2.0 * k as f64).collect();
        let spans = [(5.0, 9.0), (9.0, 15.2), (20.0, 25.0)];
        let shown = cuts(&spans, &bars);
        assert_eq!(
            shown,
            vec![
                Shown {
                    start: 4.3,
                    end: 8.3
                },
                // Not joined to the next: out on the bar after its end.
                Shown {
                    start: 8.3,
                    end: 16.3
                },
                Shown {
                    start: 18.3,
                    end: 26.3
                },
            ]
        );
        for s in &shown {
            assert!(bars.contains(&s.start) && bars.contains(&s.end), "{s:?}");
        }
        assert_eq!(showing(&shown, 4.0), None);
        assert_eq!(showing(&shown, 4.3), Some(0));
        assert_eq!(showing(&shown, 8.3), Some(1));
        assert_eq!(showing(&shown, 17.0), None);
        assert_eq!(showing(&shown, 26.0), Some(2));
        assert_eq!(showing(&shown, 26.3), None);
    }

    #[test]
    fn a_scene_shows_a_while_and_without_bars_its_span() {
        let bars: Vec<f64> = (0..30).map(|k| 0.3 + 2.0 * k as f64).collect();
        // The second span starts right after the first: it would cut on
        // the same bar, so it comes on the next.
        let shown = cuts(&[(5.0, 6.0), (6.1, 12.0)], &bars);
        assert_eq!(shown[0].start, 4.3);
        assert_eq!(shown[1].start, 6.3);
        assert_eq!(shown[0].end, 6.3);
        assert!(shown[1].start - shown[0].start >= LEAST_SCENE);
        // A bar a little after the span's start is its.
        assert_eq!(cuts(&[(4.2, 8.0)], &bars)[0].start, 4.3);
        // No bars: the spans themselves, the second held off the first.
        let plain = cuts(&[(1.0, 2.5), (2.5, 7.0)], &[]);
        assert_eq!(
            plain,
            vec![
                Shown {
                    start: 1.0,
                    end: 3.0
                },
                Shown {
                    start: 3.0,
                    end: 7.0
                }
            ]
        );
    }
}
