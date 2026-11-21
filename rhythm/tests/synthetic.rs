//! The whole analysis on generated music with known beats, with both
//! methods voting and with each alone.

use jodel_rhythm::frontend::{Frontend, FrontendConfig};
use jodel_rhythm::metrics::f_measure;
use jodel_rhythm::parts::Parts;
use jodel_rhythm::presence::Levels;
use jodel_rhythm::synth::{arpeggios, backbeat, band, beat_times, six_eight, Hit, Track};
use jodel_rhythm::{analyze_with, Analysis, Analyzer, Evidence, Method, SAMPLE_RATE};

/// Both methods, then each alone.
const SETS: [&[Method]; 3] = [
    &[Method::Flux, Method::Network],
    &[Method::Flux],
    &[Method::Network],
];

fn report(a: &Analysis) -> String {
    let mut s = format!(
        "methods {:?} chosen {} confidence {:.2} summary {:?} beats {}\n",
        a.methods,
        a.chosen,
        a.confidence,
        a.summary,
        a.beats.len()
    );
    for c in &a.candidates {
        s += &format!(
            "  {:16} {:7.2} BPM  agreement {:.2}\n",
            c.name, c.bpm, c.agreement
        );
    }
    s
}

/// The analyses of `samples` with each set of methods.
fn analyses(samples: &[f32]) -> Vec<Analysis> {
    SETS.iter().map(|m| analyze_with(samples, m)).collect()
}

/// The beats from `skip` seconds on (the trackers settle in first).
fn after(beats: &[f64], skip: f64) -> Vec<f64> {
    beats.iter().copied().filter(|&b| b >= skip).collect()
}

fn check_beats(a: &Analysis, truth: &[f64], min_f: f64) {
    let f = f_measure(&after(&a.beats, 2.0), &after(truth, 2.0), 0.07);
    assert!(f >= min_f, "F {f:.3}\n{}", report(a));
}

fn check_steady(a: &Analysis, bpm: f64) {
    let s = a.summary.expect("a summary");
    assert!(s.steady, "{}", report(a));
    assert!(
        (s.bpm - bpm).abs() / bpm < 0.01,
        "BPM {:.2}\n{}",
        s.bpm,
        report(a)
    );
}

#[test]
fn steady_drums_at_three_tempi() {
    for bpm in [90.0, 128.0, 174.0] {
        let seconds = 40.0;
        let truth = beat_times(0.5, seconds - 0.5, |_| bpm);
        for a in analyses(&band(seconds, &truth, 4, 0.5).samples) {
            check_beats(&a, &truth, 0.95);
            check_steady(&a, bpm);
            assert!(a.confidence > 0.8, "{}", report(&a));
        }
    }
}

#[test]
fn beats_land_within_ten_milliseconds() {
    let truth = beat_times(0.5, 39.5, |_| 120.0);
    for a in analyses(&band(40.0, &truth, 4, 0.5).samples) {
        let mut errors: Vec<f64> = after(&a.beats, 2.0)
            .iter()
            .map(|b| {
                truth
                    .iter()
                    .map(|t| b - t)
                    .min_by(|x, y| x.abs().total_cmp(&y.abs()))
                    .unwrap()
            })
            .collect();
        errors.sort_by(f64::total_cmp);
        let median = errors[errors.len() / 2];
        assert!(
            median.abs() < 0.01,
            "median offset {median:.4} s\n{}",
            report(&a)
        );
    }
}

#[test]
fn a_ramp_is_followed() {
    let seconds = 70.0;
    let bpm = |t: f64| 100.0 + 20.0 * ((t - 5.0) / 60.0).clamp(0.0, 1.0);
    let truth = beat_times(0.5, seconds - 0.5, bpm);
    for a in analyses(&band(seconds, &truth, 4, 0.5).samples) {
        check_beats(&a, &truth, 0.95);
        let s = a.summary.unwrap();
        assert!(!s.steady, "{}", report(&a));
        for (i, &b) in a.beats.iter().enumerate().filter(|(_, &b)| b > 4.0) {
            let want = bpm(b);
            assert!(
                (a.local_bpm[i] - want).abs() / want < 0.02,
                "at {b:.1} s: {:.1} BPM, want {want:.1}\n{}",
                a.local_bpm[i],
                report(&a)
            );
        }
    }
}

#[test]
fn a_step_is_followed() {
    let seconds = 60.0;
    let bpm = |t: f64| if t < 30.0 { 90.0 } else { 140.0 };
    let truth = beat_times(0.5, seconds - 0.5, bpm);
    for a in analyses(&band(seconds, &truth, 4, 0.5).samples) {
        check_beats(&a, &truth, 0.93);
        for (i, &b) in a.beats.iter().enumerate() {
            // One bar plus the local window either side of the change.
            if b < 2.0 || (b > 27.0 && b < 33.5) {
                continue;
            }
            let want = bpm(b);
            assert!(
                (a.local_bpm[i] - want).abs() / want < 0.02,
                "at {b:.1} s: {:.1} BPM, want {want}\n{}",
                a.local_bpm[i],
                report(&a)
            );
        }
    }
}

#[test]
fn swing_keeps_the_quarter_notes() {
    let truth = beat_times(0.5, 39.5, |_| 110.0);
    for a in analyses(&band(40.0, &truth, 4, 2.0 / 3.0).samples) {
        check_beats(&a, &truth, 0.95);
        check_steady(&a, 110.0);
    }
}

fn downbeat_hits(a: &Analysis, truth: &[f64], meter: usize) -> f64 {
    let truth_down: Vec<f64> = truth.iter().copied().step_by(meter).collect();
    let found: Vec<f64> = a.downbeats.iter().map(|&i| a.beats[i]).collect();
    f_measure(&after(&found, 4.0), &after(&truth_down, 4.0), 0.07)
}

#[test]
fn bars_of_four_and_of_three() {
    let truth = beat_times(0.5, 47.5, |_| 120.0);
    for a in analyses(&band(48.0, &truth, 4, 0.5).samples) {
        let f = downbeat_hits(&a, &truth, 4);
        assert!(f > 0.9, "4/4 downbeats F {f:.2}\n{}", report(&a));
    }

    let truth = beat_times(0.5, 47.5, |_| 140.0);
    for a in analyses(&band(48.0, &truth, 3, 0.5).samples) {
        check_beats(&a, &truth, 0.95);
        let f = downbeat_hits(&a, &truth, 3);
        assert!(f > 0.9, "3/4 downbeats F {f:.2}\n{}", report(&a));
    }
}

#[test]
fn a_slow_piece_stays_slow() {
    let seconds = 60.0;
    let truth = beat_times(1.0, seconds - 1.0, |_| 50.0);
    let mut track = Track::new(seconds);
    for (i, &b) in truth.iter().enumerate() {
        let root = [110.0, 98.0, 87.3, 82.4][(i / 4) % 4];
        track.hit(b, Hit::Pluck(root), 0.5);
        track.hit(b, Hit::Pluck(root * 2.0), 0.3);
    }
    for a in analyses(&track.samples) {
        check_beats(&a, &truth, 0.9);
        check_steady(&a, 50.0);
    }
}

#[test]
fn drums_in_noise() {
    let truth = beat_times(0.5, 39.5, |_| 128.0);
    let mut track = band(40.0, &truth, 4, 0.5);
    track.noise(0.08);
    for a in analyses(&track.samples) {
        check_beats(&a, &truth, 0.93);
        check_steady(&a, 128.0);
    }
}

#[test]
fn clicks_alone() {
    let truth = beat_times(0.5, 29.5, |_| 100.0);
    let mut track = Track::new(30.0);
    for &b in &truth {
        track.hit(b, Hit::Click, 0.5);
    }
    for a in analyses(&track.samples) {
        check_beats(&a, &truth, 0.95);
        check_steady(&a, 100.0);
    }
}

#[test]
fn noise_is_not_sure_and_silence_has_no_beat() {
    let mut track = Track::new(30.0);
    track.noise(0.3);
    for a in analyses(&track.samples) {
        assert!(a.confidence < 0.5, "{}", report(&a));
    }

    for a in analyses(&Track::new(10.0).samples) {
        assert!(a.beats.is_empty() && a.summary.is_none(), "{}", report(&a));
    }
}

#[test]
fn methods_have_names() {
    for m in Method::ALL {
        assert_eq!(Method::parse(m.id()), Some(m));
    }
    assert_eq!(Method::parse("tea leaves"), None);
}

// The beat a listener counts, where the steadiest pulse is faster.

#[test]
fn a_slow_backbeat_is_not_counted_in_eighths() {
    let truth = beat_times(0.5, 47.5, |_| 84.0);
    for a in analyses(&backbeat(48.0, &truth).samples) {
        check_beats(&a, &truth, 0.9);
        check_steady(&a, 84.0);
    }
}

#[test]
fn six_eight_is_counted_in_dotted_quarters() {
    let truth = beat_times(0.5, 59.5, |_| 56.0);
    for a in analyses(&six_eight(60.0, &truth).samples) {
        check_beats(&a, &truth, 0.9);
        check_steady(&a, 56.0);
    }
}

#[test]
fn arpeggios_are_counted_in_quarter_notes() {
    let truth = beat_times(0.5, 59.5, |_| 60.0);
    for a in analyses(&arpeggios(60.0, &truth).samples) {
        check_beats(&a, &truth, 0.9);
        check_steady(&a, 60.0);
    }
}

#[test]
fn a_fast_backbeat_stays_fast() {
    let truth = beat_times(0.5, 39.5, |_| 172.0);
    for a in analyses(&band(40.0, &truth, 4, 0.5).samples) {
        check_beats(&a, &truth, 0.9);
        check_steady(&a, 172.0);
    }
}

/// Parts as a stem separator would give them, from the stems as
/// generated: `drums`, `bass` and the rest (`tonal`; the bass joins it).
fn stem_parts(drums: &[f32], bass: &[f32], tonal: &[f32]) -> Parts {
    let mel = |samples: &[f32]| {
        let mut frontend = Frontend::new(FrontendConfig::FLUX);
        frontend.push(samples);
        frontend.finish()
    };
    let hop = SAMPLE_RATE as usize / 100;
    let energy = |samples: &[f32]| -> Vec<f32> {
        samples
            .chunks(hop)
            .map(|c| c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32)
            .collect()
    };
    let all: Vec<f32> = (0..drums.len())
        .map(|i| drums[i] + bass[i] + tonal[i])
        .collect();
    let mut both = bass.to_vec();
    for (b, t) in both.iter_mut().zip(tonal) {
        *b += t;
    }
    Parts {
        drums: mel(drums),
        tonal: mel(&both),
        levels: Some(Levels {
            fps: 100.0,
            drums: energy(drums),
            bass: energy(bass),
            total: energy(&all),
        }),
        lanes: None,
    }
}

/// The evidence of music made of `drums`, `bass` and `tonal`, with those
/// as its stems (flux only, for speed).
fn with_stems(drums: &Track, bass: &Track, tonal: &Track) -> Evidence {
    let mix: Vec<f32> = (0..drums.samples.len())
        .map(|i| drums.samples[i] + bass.samples[i] + tonal.samples[i])
        .collect();
    let parts = stem_parts(&drums.samples, &bass.samples, &tonal.samples);
    let mut analyzer = Analyzer::with_methods(&[Method::Flux]);
    analyzer.push(&mix);
    analyzer.listen_with(|| Some(parts))
}

/// Sustained chords over the whole of `seconds`, with a soft attack on
/// each beat of `beats` (a pad that pulses gently).
fn pad(seconds: f64, beats: &[f64]) -> Track {
    let mut t = Track::new(seconds);
    t.chord(0.0, seconds, &[220.0, 277.2, 329.6], 0.15);
    for &b in beats {
        t.hit(b, Hit::Pluck(440.0), 0.05);
    }
    t
}

#[test]
fn the_drum_stem_votes_only_where_the_drums_play() {
    let truth = beat_times(0.5, 39.5, |_| 120.0);
    // Drums from 16 s on; no bass.
    let mut drums = Track::new(40.0);
    for (i, &b) in truth.iter().enumerate().filter(|(_, &b)| b >= 16.0) {
        drums.hit(b, if i % 2 == 0 { Hit::Kick } else { Hit::Snare }, 0.5);
    }
    let evidence = with_stems(&drums, &Track::new(40.0), &pad(40.0, &truth));
    let stem = |name: &str| {
        evidence
            .candidates
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no {name}"))
    };
    let kicks = stem("drums/hmm");
    assert!(kicks.is_partial());
    assert!(kicks.beats.len() > 20, "{:?}", kicks.beats);
    // Its say rises as the drums come in (over the share's few seconds).
    assert!(kicks.beats[0] >= 12.0, "{:?}", kicks.beats);
    assert!(kicks.say_between(0.0, 10.0) < 0.05);
    assert!(kicks.say_between(20.0, 38.0) > 0.8);
    let f = f_measure(&kicks.beats, &after(&truth, kicks.beats[0] - 0.1), 0.07);
    assert!(f > 0.9, "F {f:.3}");
    // The drum stem is the only one that votes.
    assert_eq!(
        evidence
            .candidates
            .iter()
            .filter(|c| c.is_partial())
            .count(),
        1
    );
    let a = evidence.conclude();
    check_beats(&a, &truth, 0.9);
}

#[test]
fn without_drums_the_stems_change_nothing() {
    let truth = beat_times(0.5, 29.5, |_| 100.0);
    let silence = Track::new(30.0);
    let evidence = with_stems(&silence, &silence, &pad(30.0, &truth));
    let mut without = evidence.clone();
    without.candidates.retain(|c| !c.is_partial());
    assert!(without.candidates.len() < evidence.candidates.len());
    let (a, b) = (evidence.conclude(), without.conclude());
    assert_eq!(a.beats, b.beats);
    assert_eq!(a.chosen, b.chosen);
    assert_eq!(a.confidence, b.confidence);
}
