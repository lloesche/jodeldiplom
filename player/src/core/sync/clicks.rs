//! The sync's clicks and what the answers to them measure: when each
//! click plays ([`click_frames`]), each click's answer ([`answers`]) and
//! the delay they agree on ([`outcome`]).

/// The clicks played.
pub const CLICKS: usize = 8;
/// Their tempo.
pub const BPM: f64 = 90.0;
/// The silence before the first, in seconds.
pub const LEAD_IN: f64 = 1.0;
/// The time after the last to answer it, in seconds.
pub const TAIL: f64 = 1.0;
/// The earliest an answer may come, from its click, in seconds.
pub const EARLIEST: f64 = -0.2;
/// The latest.
pub const LATEST: f64 = 0.45;
/// How close to the median of the answers one must be to count, in
/// seconds.
pub const AGREE: f64 = 0.15;
/// How many answers must agree.
pub const ENOUGH: usize = 5;

/// When click `k` starts, in seconds from the start of the clicks' output.
pub fn click_time(k: usize) -> f64 {
    LEAD_IN + k as f64 * 60.0 / BPM
}

/// The stream frames the clicks start at, at `rate`.
pub fn click_frames(rate: u32) -> Vec<u64> {
    (0..CLICKS)
        .map(|k| (click_time(k) * f64::from(rate)).round() as u64)
        .collect()
}

/// How long the clicks' output plays, in seconds.
pub fn length() -> f64 {
    click_time(CLICKS - 1) + TAIL
}

/// How a sync ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The delay, in seconds.
    Measured(f64),
    /// Too few answers agreed: how many did.
    TooFew(usize),
    /// The clicks could not be played, or stopped: why.
    Failed(String),
}

/// Each click's answer: the strongest of `responses` (stream frames at
/// `rate`, with how strong each is) from [`EARLIEST`] to [`LATEST`] of it
/// (the first of the strongest), in seconds after it; none where there is
/// none.
pub fn answers(clicks: &[u64], rate: u32, responses: &[(u64, f32)]) -> Vec<Option<f64>> {
    let rate = f64::from(rate.max(1));
    clicks
        .iter()
        .map(|&c| {
            let mut best: Option<(f64, f32)> = None;
            for &(frame, strength) in responses {
                let off = (frame as f64 - c as f64) / rate;
                if (EARLIEST..=LATEST).contains(&off) && best.is_none_or(|b| strength > b.1) {
                    best = Some((off, strength));
                }
            }
            best.map(|b| b.0)
        })
        .collect()
}

/// The median of `values` (sorted here; the upper of the middle two).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// What `answers` measure: the median of those within [`AGREE`] of the
/// median of them all, once [`ENOUGH`] of them are; else how many were.
pub fn outcome(answers: &[Option<f64>]) -> Outcome {
    let mut got: Vec<f64> = answers.iter().flatten().copied().collect();
    if got.len() < ENOUGH {
        return Outcome::TooFew(got.len());
    }
    let middle = median(&mut got);
    let mut agree: Vec<f64> = got
        .into_iter()
        .filter(|a| (a - middle).abs() <= AGREE)
        .collect();
    if agree.len() < ENOUGH {
        return Outcome::TooFew(agree.len());
    }
    Outcome::Measured(median(&mut agree))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn at(seconds: f64) -> u64 {
        (seconds * f64::from(RATE)).round() as u64
    }

    #[test]
    fn eight_clicks_at_90_bpm_after_a_second() {
        let clicks = click_frames(RATE);
        assert_eq!(clicks.len(), 8);
        assert_eq!(clicks[0], 48_000);
        assert_eq!(clicks[1] - clicks[0], 32_000);
        assert!((length() - (1.0 + 7.0 * 2.0 / 3.0 + TAIL)).abs() < 1e-9);
    }

    #[test]
    fn the_answers_give_their_median_the_odd_one_out_left_out() {
        let clicks = click_frames(RATE);
        // "ta" 150 ms after each click, give or take 10 ms; the fourth
        // 400 ms late (a cough), the seventh unheard; and a weak sound
        // early in the third's window (the click's own echo).
        let offs = [0.15, 0.16, 0.14, 0.4, 0.15, 0.155, f64::NAN, 0.145];
        let mut responses: Vec<(u64, f32)> = clicks
            .iter()
            .zip(offs)
            .filter(|(_, o)| !o.is_nan())
            .map(|(&c, o)| (c + at(o), -20.0))
            .collect();
        responses.push((clicks[2] + at(0.02), -45.0));
        let got = answers(&clicks, RATE, &responses);
        assert_eq!(got.iter().flatten().count(), 7);
        assert!((got[2].unwrap() - 0.14).abs() < 1e-4, "the stronger answer");
        assert_eq!(got[6], None);
        let Outcome::Measured(d) = outcome(&got) else {
            panic!("{:?}", outcome(&got));
        };
        // The median of the six that agree.
        assert!((d - 0.15).abs() < 1e-3, "{d}");
    }

    #[test]
    fn too_few_answers_that_agree_ask_for_another_go() {
        let some = [
            Some(0.1),
            Some(0.12),
            None,
            Some(0.11),
            None,
            None,
            None,
            Some(0.1),
        ];
        assert_eq!(outcome(&some), Outcome::TooFew(4));
        // Eight answers all over the place: only three agree.
        let scattered = [0.0, 0.05, 0.1, 0.4, -0.18, 0.3, 0.44, -0.1].map(Some);
        assert!(matches!(outcome(&scattered), Outcome::TooFew(n) if n < ENOUGH));
    }
}
