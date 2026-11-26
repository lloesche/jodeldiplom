//! Every network here on one song, timed: the stem separator over the
//! whole track, the beat network over its spectrogram, the speech
//! recogniser over the separated voice (its transcript and its glimpses,
//! as the player hears them), the speech model over the voice, and the
//! cover's depth and cut-out (on a generated picture: they run at fixed
//! sizes). For each, the wall time of a run (after a first one that sets
//! its plans up) with the GPU's submissions and waits, then a second run
//! profiled ([`jodel_gpu::profile`]): the GPU time of each kernel.
//!
//! ```text
//! cargo run --release -p jodel-gpu --example gpu_profile -- SONG MODELS [NETWORK...]
//! ```
//!
//! `MODELS` is the player's models folder (the packed recogniser, the
//! speech model, the depth and cut-out checkpoints); `NETWORK`s pick some
//! of `stems`, `beats`, `asr`, `speech`, `vision` (all by default).

use std::path::Path;
use std::time::{Duration, Instant};

use jodel_gpu::{profile, GpuAsr, GpuDepth, GpuMatte, GpuNetwork, GpuSeparator, GpuSpeech};
use jodel_lyrics::aligner::Acoustic;
use jodel_playback::{Resampler, Source, Stem, StereoFeed};
use jodel_rhythm::stems::{Splitter, StemBlock};

/// The song as the separator takes it: stereo at its rate.
fn stereo(path: &Path) -> (Vec<f32>, Vec<f32>) {
    let mut source = Source::open(path).unwrap();
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    let mut feed = StereoFeed::new(rate);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    let mut put = |l: Vec<f32>, r: Vec<f32>| {
        left.extend(l);
        right.extend(r);
    };
    let mut block = Vec::new();
    loop {
        block.clear();
        if source.read_unquantized(&mut block).unwrap() == 0 {
            break;
        }
        feed.push(&block, channels, &mut put);
    }
    feed.finish(&mut put);
    (left, right)
}

/// `x` from the separator's rate to `to`.
fn resampled(x: &[f64], to: u32) -> Vec<f32> {
    let mut r = Resampler::new(jodel_playback::stems::RATE, to, 1);
    let mut out = Vec::new();
    r.process_f64(x, &mut out);
    r.finish_f64(&mut out);
    out.into_iter().map(|v| v as f32).collect()
}

/// The separated voice, mono at the separator's rate.
fn separate(gpu: &'static GpuSeparator, left: &[f32], right: &[f32]) -> Vec<f64> {
    let mut voice: Vec<f64> = vec![0.0; left.len()];
    let mut take = |b: &StemBlock| {
        let [l, r] = b.stems[Stem::Vocals];
        for (i, (a, c)) in l.iter().zip(r).enumerate() {
            if let Some(v) = voice.get_mut(b.start + i) {
                *v = (f64::from(*a) + f64::from(*c)) / 2.0;
            }
        }
    };
    let mut splitter = Splitter::overlapped(gpu);
    for (l, r) in left.chunks(1 << 16).zip(right.chunks(1 << 16)) {
        splitter.push(l, r, &mut take).unwrap();
    }
    splitter.finish(&mut take).unwrap();
    voice
}

/// A generated cover: soft gradients and a few discs.
fn cover() -> jodel_gpu::vision::prep::Rgb {
    let side = 1200;
    let mut data = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            let (fx, fy) = (x as f32 / side as f32, y as f32 / side as f32);
            let disc = [(0.3, 0.4, 0.15), (0.7, 0.6, 0.2), (0.5, 0.2, 0.08)]
                .iter()
                .any(|&(cx, cy, r): &(f32, f32, f32)| (fx - cx).hypot(fy - cy) < r);
            let (r, g, b) = if disc {
                (230.0, 200.0 * fy, 40.0)
            } else {
                (60.0 + 120.0 * fx, 80.0 * fy, 160.0 - 100.0 * fx)
            };
            data.extend([r as u8, g as u8, b as u8]);
        }
    }
    jodel_gpu::vision::prep::Rgb::new(side, side, data)
}

/// Timed runs, the quickest kept (the GPU may be shared with other work).
const TIMED: usize = 3;

/// `run` once to set up, [`TIMED`] times timed, twice profiled by
/// submission and twice by kernel: the quickest timed run's wall time and
/// what it cost (`cost`: what `since` makes of a `snap` before and after
/// it), the GPU's busy time (its submissions' spans), then the kernels'
/// profile of least GPU time.
fn measure<S, T>(
    name: &str,
    snap: impl Fn() -> S,
    since: impl Fn(&S, &S) -> String,
    mut run: impl FnMut() -> T,
) -> T {
    let mut out = run();
    let mut best = (Duration::MAX, String::new());
    for _ in 0..TIMED {
        let before = snap();
        let started = Instant::now();
        out = run();
        let wall = started.elapsed();
        if wall < best.0 {
            best = (wall, since(&snap(), &before));
        }
    }
    let mut least = |detail| {
        (0..2)
            .map(|_| {
                profile::start(detail);
                run();
                profile::take()
            })
            .min_by_key(profile::Profile::total)
            .unwrap_or_default()
    };
    let busy = least(profile::Detail::Submissions).all();
    let found = least(profile::Detail::Kernels);
    println!(
        "== {name}: wall {:.1} ms, GPU busy {:.1} ms\n   {}\n   {found}",
        ms(best.0),
        ms(busy),
        best.1
    );
    println!("   the longest by workgroups:");
    for s in found.shapes.iter().take(12) {
        println!(
            "  {:>9.2} ms  {:>7} × {:>8.1} µs  {} {:?}",
            ms(s.typical),
            s.calls,
            ms(s.typical) * 1e3 / s.calls.max(1) as f64,
            s.kernel,
            s.groups
        );
    }
    out
}

fn tally_since(after: &jodel_gpu::pace::Tally, before: &jodel_gpu::pace::Tally) -> String {
    after.since(before).to_string()
}

fn none(_: &(), _: &()) -> String {
    "(no tally)".into()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (song, models) = (Path::new(&args[0]), Path::new(&args[1]));
    let chosen: Vec<&str> = args[2..].iter().map(String::as_str).collect();
    let wanted = |n: &str| chosen.is_empty() || chosen.contains(&n);
    let (left, right) = stereo(song);
    println!(
        "{}: {:.1} s on {}",
        song.display(),
        left.len() as f64 / f64::from(jodel_playback::stems::RATE),
        jodel_gpu::gpu_name().unwrap_or_default()
    );
    let separator = GpuSeparator::shared().expect("a GPU");
    let voice = if wanted("stems") {
        measure(
            "stems",
            || separator.tally(),
            tally_since,
            || separate(separator, &left, &right),
        )
    } else {
        separate(separator, &left, &right)
    };
    let voice_16k = resampled(&voice, 16_000);
    if wanted("beats") {
        let gpu = GpuNetwork::shared().expect("the beat network");
        let mono: Vec<f64> = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (f64::from(*a) + f64::from(*b)) / 2.0)
            .collect();
        let mut front = jodel_rhythm::net::NetFrontend::new();
        front.push(&resampled(&mono, jodel_rhythm::SAMPLE_RATE));
        let mel = front.finish();
        measure(
            "beats",
            || gpu.tally(),
            tally_since,
            || jodel_rhythm::net::Network::shared().predict_with(&mel, Some(gpu)),
        );
    }
    if wanted("asr") {
        let packed =
            jodel_lyrics::asr::packed::PackedModel::open(&models.join("qwen3-asr-0.6b-q4.jdlasr"))
                .unwrap();
        let asr = GpuAsr::new(&packed).unwrap();
        drop(packed);
        let hearing = || asr.hearing().unwrap();
        let since = |after: &jodel_gpu::Hearing, before: &jodel_gpu::Hearing| {
            let h = after.since(before);
            format!(
                "{h}\n   prompts: {}\n   steps: {}",
                h.prompt_gpu, h.step_gpu
            )
        };
        let heard = measure("asr transcript", hearing, since, || {
            asr.transcribe(&voice_16k).unwrap()
        });
        println!("   [{}] {}", heard.language, heard.text);
        let glimpses = measure("asr glimpses", hearing, since, || {
            asr.glimpses(&voice_16k).unwrap()
        });
        let texts: Vec<&str> = glimpses.iter().map(|g| g.text.as_str()).collect();
        println!("   {}", texts.join(" | "));
    }
    if wanted("speech") {
        let cpu =
            jodel_lyrics::speech::Wav2Vec2::open(&models.join("wav2vec2-base-960h.safetensors"))
                .unwrap();
        let gpu = GpuSpeech::new(&cpu).unwrap();
        measure("speech", || (), none, || gpu.emissions(&voice_16k));
    }
    if wanted("vision") {
        let image = cover();
        let read = |name: &str| std::fs::read(models.join(name)).unwrap();
        let depth = GpuDepth::new(read("da3-small.safetensors")).unwrap();
        measure("depth", || (), none, || depth.depth(&image).unwrap());
        depth.release();
        let matte = GpuMatte::new(read("ben2-base.safetensors")).unwrap();
        measure("matte", || (), none, || matte.matte(&image).unwrap());
    }
}
