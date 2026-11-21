//! Our inference of the network against the authors' PyTorch code, whose
//! output for the same generated track is in `fixtures/net_logits.txt`.

use jodel_rhythm::net::{NetFrontend, Network};
use jodel_rhythm::synth::{band, beat_times};

fn fixture() -> (Vec<f32>, Vec<f32>) {
    let text = include_str!("fixtures/net_logits.txt");
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let mut parts = l.split_whitespace().map(|v| v.parse::<f32>().unwrap());
            (parts.next().unwrap(), parts.next().unwrap())
        })
        .unzip()
}

/// Frames that are the largest within ±3 frames and above 0 (probability
/// above one half): the authors' simplest beat picking.
fn peaks(logits: &[f32]) -> Vec<usize> {
    (0..logits.len())
        .filter(|&i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 4).min(logits.len());
            logits[i] > 0.0 && logits[lo..hi].iter().all(|&v| v <= logits[i])
        })
        .collect()
}

#[test]
fn logits_match_the_reference() {
    let truth = beat_times(0.5, 39.5, |_| 120.0);
    let samples = band(40.0, &truth, 4, 0.5).samples;
    let mut frontend = NetFrontend::new();
    frontend.push(&samples);
    let logits = Network::shared().predict(&frontend.finish());
    let (beat, downbeat) = fixture();
    assert_eq!(logits.beat.len(), beat.len());
    for (name, ours, theirs) in [
        ("beat", &logits.beat, &beat),
        ("downbeat", &logits.downbeat, &downbeat),
    ] {
        let (at, worst) = ours
            .iter()
            .zip(theirs.iter())
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(worst < 0.02, "{name}: off by {worst} at frame {at}");
        assert_eq!(peaks(ours), peaks(theirs), "{name} peaks");
    }
    // And the network hears the beats and the bars.
    let near = |p: &[usize], t: f64| p.iter().any(|&f| (f as f64 / 50.0 - t).abs() <= 0.04);
    let (beat_peaks, bar_peaks) = (peaks(&beat), peaks(&downbeat));
    assert!(truth.iter().all(|&t| near(&beat_peaks, t)));
    let bars = truth
        .iter()
        .step_by(4)
        .filter(|&&t| near(&bar_peaks, t))
        .count();
    assert!(bars >= truth.len() / 4 - 1, "{bars} bars found");
}
