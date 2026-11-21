//! The beat network on the GPU against the authors' PyTorch output for a
//! generated track (`rhythm/tests/fixtures/net_logits.txt`, as the CPU
//! network is tested) and against the CPU network; skipped (with a note)
//! where there is no usable GPU.

use std::time::Instant;

use jodel_gpu::GpuNetwork;
use jodel_rhythm::net::{NetFrontend, NetRunner, Network};
use jodel_rhythm::synth::{band, beat_times};

fn reference() -> (Vec<f32>, Vec<f32>) {
    let text = include_str!("../../rhythm/tests/fixtures/net_logits.txt");
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let mut parts = l.split_whitespace().map(|v| v.parse::<f32>().unwrap());
            (parts.next().unwrap(), parts.next().unwrap())
        })
        .unzip()
}

/// Frames that are the largest within ±3 frames and above 0.
fn peaks(logits: &[f32]) -> Vec<usize> {
    (0..logits.len())
        .filter(|&i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 4).min(logits.len());
            logits[i] > 0.0 && logits[lo..hi].iter().all(|&v| v <= logits[i])
        })
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .enumerate()
        .max_by(|p, q| p.1.total_cmp(&q.1))
        .unwrap()
}

/// The network on the GPU; `None` (skip) only without a usable GPU: one
/// that is there must set it up.
fn gpu() -> Option<&'static GpuNetwork> {
    let found = GpuNetwork::shared();
    if found.is_none() {
        if let (Some(name), Err(e)) = (jodel_gpu::gpu_name(), GpuNetwork::new()) {
            panic!("{name} is there, but the network did not set up on it: {e}");
        }
        eprintln!("no usable GPU: skipped");
    }
    found
}

#[test]
fn the_gpu_network_gives_the_reference_logits() {
    let Some(gpu) = gpu() else {
        return;
    };
    // 40 s at 120 BPM: two pieces of 30 s.
    let truth = beat_times(0.5, 39.5, |_| 120.0);
    let samples = band(40.0, &truth, 4, 0.5).samples;
    let mut frontend = NetFrontend::new();
    frontend.push(&samples);
    let mel = frontend.finish();
    let started = Instant::now();
    let cpu = Network::shared().predict(&mel);
    let cpu_time = started.elapsed();
    let started = Instant::now();
    let ours = Network::shared().predict_with(&mel, Some(gpu));
    let gpu_time = started.elapsed();
    let started = Instant::now();
    let again = Network::shared().predict_with(&mel, Some(gpu));
    eprintln!(
        "{}: the network on {} frames in {:.3} s (then {:.3} s), on the processor {:.3} s",
        gpu.name(),
        mel.frames,
        gpu_time.as_secs_f64(),
        started.elapsed().as_secs_f64(),
        cpu_time.as_secs_f64()
    );
    assert_eq!(again.beat, ours.beat, "the same twice");
    let (beat, downbeat) = reference();
    for (name, got, cpu, theirs) in [
        ("beat", &ours.beat, &cpu.beat, &beat),
        ("downbeat", &ours.downbeat, &cpu.downbeat, &downbeat),
    ] {
        assert_eq!(got.len(), theirs.len());
        let (at, off) = worst(got, theirs);
        let (at_cpu, off_cpu) = worst(got, cpu);
        eprintln!("{name}: off the reference by {off} at {at}, off the processor's by {off_cpu} at {at_cpu}");
        assert!(
            off < 0.02,
            "{name}: off the reference by {off} at frame {at}"
        );
        assert!(
            off_cpu < 0.02,
            "{name}: off the processor's by {off_cpu} at frame {at_cpu}"
        );
        assert_eq!(peaks(got), peaks(theirs), "{name} peaks");
    }
}

#[test]
fn a_piece_that_is_wrong_is_refused() {
    let Some(gpu) = gpu() else {
        return;
    };
    assert!(gpu.forward(&[0.0; 10], 3).is_err());
    assert!(gpu.forward(&[], 0).is_err());
}
