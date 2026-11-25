//! The GPU separator against the CPU one, on the same chunk; skipped
//! (with a note) where there is no usable GPU.

use std::f64::consts::PI;
use std::time::Instant;

use jodel_gpu::GpuSeparator;
use jodel_rhythm::stems::{self, CpuSeparator, Separator, RATE};

/// `seconds` of stereo: bass, a tone each side, kicks every half second.
fn clip(seconds: f64) -> (Vec<f32>, Vec<f32>) {
    let n = (seconds * f64::from(RATE)) as usize;
    let sr = f64::from(RATE);
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr;
        let k = t % 0.5;
        let kick = 0.6 * (2.0 * PI * 60.0 * k).sin() * (-k / 0.08).exp();
        let bass = 0.3 * (2.0 * PI * 110.0 * t).sin();
        left.push((bass + kick + 0.2 * (2.0 * PI * 440.0 * t).sin()) as f32);
        right.push((bass + kick + 0.2 * (2.0 * PI * 554.37 * t).sin()) as f32);
    }
    (left, right)
}

#[test]
fn the_gpu_gives_the_cpu_masks() {
    let Some(gpu) = GpuSeparator::shared() else {
        eprintln!("no usable GPU: skipped");
        return;
    };
    let (left, right) = clip(3.0);
    let (frames, packed) = stems::packed_stft(&left, &right);
    let cpu = CpuSeparator::shared().masks(frames, &packed).unwrap();
    let started = Instant::now();
    let got = gpu.masks(frames, &packed).unwrap();
    eprintln!(
        "{}: {frames} frames in {:.3} s",
        gpu.name(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(got.len(), cpu.len());
    let mut diffs: Vec<f32> = got.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).collect();
    diffs.sort_by(f32::total_cmp);
    let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
    let p999 = diffs[diffs.len() * 999 / 1000];
    eprintln!(
        "mean {mean:e}, 99.9th percentile {p999:e}, max {:e}",
        diffs[diffs.len() - 1]
    );
    assert!(mean < 1e-4, "mean difference {mean}");
    assert!(p999 < 5e-3, "99.9th percentile {p999}");
}

/// A thread marked as doing background work gets the background
/// separator, any other the shared one; and the background one separates
/// as the shared one does.
#[test]
fn each_thread_gets_its_kind_of_separator() {
    let Some(shared) = GpuSeparator::shared() else {
        eprintln!("no usable GPU: skipped");
        return;
    };
    let address = |s: Option<&'static GpuSeparator>| s.map(|s| s as *const GpuSeparator as usize);
    assert_eq!(
        address(GpuSeparator::for_this_thread()),
        address(Some(shared))
    );
    let on_background = std::thread::spawn(move || {
        jodel_gpu::pace::background();
        address(GpuSeparator::for_this_thread())
    })
    .join()
    .unwrap();
    let background = GpuSeparator::background();
    assert_eq!(on_background, address(background));
    let (left, right) = clip(1.0);
    let (frames, packed) = stems::packed_stft(&left, &right);
    assert_eq!(
        background.unwrap().masks(frames, &packed).unwrap(),
        shared.masks(frames, &packed).unwrap()
    );
}

#[test]
fn a_full_chunk_is_quick() {
    let Some(gpu) = GpuSeparator::shared() else {
        eprintln!("no usable GPU: skipped");
        return;
    };
    let (left, right) = clip(6.0);
    let (frames, packed) = stems::packed_stft(&left, &right);
    gpu.masks(frames, &packed).unwrap();
    let started = Instant::now();
    for _ in 0..3 {
        gpu.masks(frames, &packed).unwrap();
    }
    let each = started.elapsed().as_secs_f64() / 3.0;
    eprintln!("{}: a 6 s chunk in {each:.3} s", gpu.name());
}
