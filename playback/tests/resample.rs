//! The resampler against its specification, over a matrix of rate pairs:
//! up and down, within and across the 44.1 and 48 kHz families, integer
//! and non-integer ratios. Measured on the designed taps (a dense FFT of
//! the prototype) and end to end (tones through the resampler, before
//! requantization, analysed with a long Kaiser-windowed FFT).
//!
//! The numbers behind the assertions are printed; see them with
//! `cargo test --release -p jodel-playback --test resample -- --nocapture`.

use std::f64::consts::PI;
use std::time::Instant;

use jodel_playback::resample::{output_frames, FilterDesign, Resampler, STOPBAND_DB};

/// The rate pairs every test runs over.
const MATRIX: [(u32, u32); 15] = [
    (44_100, 48_000),
    (48_000, 44_100),
    (88_200, 96_000),
    (96_000, 88_200),
    (176_400, 192_000),
    (44_100, 192_000),
    (192_000, 96_000),
    (384_000, 192_000),
    (352_800, 192_000),
    (96_000, 44_100),
    (384_000, 48_000),
    (44_100, 32_000),
    (48_000, 22_050),
    (44_100, 88_200),
    (48_000, 192_000),
];

/// The passband ripple allowed, in dB.
const RIPPLE_DB: f64 = 0.001;

/// In-place radix-2 FFT (forward).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    assert!(n.is_power_of_two() && im.len() == n);
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let twiddles: Vec<(f64, f64)> = (0..n / 2)
        .map(|k| {
            let (s, c) = (-2.0 * PI * k as f64 / n as f64).sin_cos();
            (c, s)
        })
        .collect();
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = twiddles[k * step];
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len *= 2;
    }
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn bessel_i0(x: f64) -> f64 {
    let (mut term, mut sum, mut k) = (1.0, 1.0, 1.0);
    while term > sum * 1e-18 {
        term *= x * x / 4.0 / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    let i0 = bessel_i0(beta);
    (0..n)
        .map(|k| {
            let r = 2.0 * k as f64 / (n - 1) as f64 - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0
        })
        .collect()
}

struct Response {
    ripple_db: f64,
    stopband_db: f64,
}

/// The prototype's response on a dense grid (at least 8 points per
/// sidelobe), normalized to its DC gain.
fn measure_design(d: &FilterDesign) -> Response {
    let h = d.prototype();
    let (up, _) = d.ratio();
    let size = (8 * h.len()).next_power_of_two().max(1 << 16);
    let mut re = vec![0.0; size];
    let mut im = vec![0.0; size];
    re[..h.len()].copy_from_slice(&h);
    fft(&mut re, &mut im);
    let proto_rate = f64::from(up) * f64::from(d.from_rate());
    let bin = proto_rate / size as f64;
    let gain = f64::from(up);
    let mut ripple: f64 = 0.0;
    let mut stop: f64 = f64::NEG_INFINITY;
    for k in 0..=size / 2 {
        let f = k as f64 * bin;
        let mag = re[k].hypot(im[k]) / gain;
        if f <= d.passband_hz() {
            ripple = ripple.max(db(mag).abs());
        } else if f >= d.stopband_hz() {
            stop = stop.max(db(mag));
        }
    }
    Response {
        ripple_db: ripple,
        stopband_db: -stop,
    }
}

#[test]
fn designs_meet_the_passband_and_stopband_spec() {
    println!(
        "{:>9} {:>9} {:>9} {:>6} {:>9} {:>9} {:>12} {:>11} {:>10}",
        "from", "to", "ratio", "taps", "pass Hz", "stop Hz", "ripple dB", "stopband", "table KiB"
    );
    let mut largest = 0;
    for (from, to) in MATRIX {
        let d = FilterDesign::new(from, to);
        let r = measure_design(&d);
        let (up, down) = d.ratio();
        largest = largest.max(d.table_bytes());
        println!(
            "{from:>9} {to:>9} {:>9} {:>6} {:>9.1} {:>9.1} {:>12.2e} {:>8.1} dB {:>10}",
            format!("{up}/{down}"),
            d.taps_per_phase(),
            d.passband_hz(),
            d.stopband_hz(),
            r.ripple_db,
            r.stopband_db,
            d.table_bytes() / 1024
        );
        let low = f64::from(from.min(to));
        assert_eq!(d.passband_hz(), (0.45 * low).min(20_000.0));
        assert_eq!(d.stopband_hz(), 0.5 * low);
        assert!(
            r.ripple_db <= RIPPLE_DB,
            "{from}->{to}: ripple {}",
            r.ripple_db
        );
        assert!(
            r.stopband_db >= STOPBAND_DB,
            "{from}->{to}: stopband {} dB",
            r.stopband_db
        );
    }
    println!("largest coefficient table: {} KiB", largest / 1024);
    assert!(largest <= 4 << 20, "coefficient table of {largest} bytes");
}

/// A tone through the resampler (no requantization): its level at the
/// output, and the highest other component anywhere in the output band,
/// both relative to the input amplitude.
struct ToneResult {
    level_db: f64,
    spur_db: f64,
}

const ANALYSIS: usize = 1 << 16;

fn tone_through(from: u32, to: u32, freq: f64, expect_tone: bool) -> ToneResult {
    let design = FilterDesign::shared(from, to);
    let (up, down) = design.ratio();
    let half = design.half_width() as u64;
    // Skip the start, where the filter still reads the silence before the
    // stream, and leave the end, which needs the silence after it.
    let skip = ((half + 2) * u64::from(up)).div_ceil(u64::from(down)) as usize + 16;
    let wanted = skip + ANALYSIS;
    let input_frames = (wanted as u64 * u64::from(down)).div_ceil(u64::from(up)) + 2 * half + 16;
    let omega = 2.0 * PI * freq / f64::from(from);
    let input: Vec<f64> = (0..input_frames)
        .map(|n| (omega * n as f64).sin())
        .collect();
    let mut r = Resampler::with_design(design, 1);
    let mut out = Vec::new();
    for chunk in input.chunks(4096) {
        r.process_f64(chunk, &mut out);
    }
    assert!(out.len() >= wanted, "{} < {wanted}", out.len());
    let y = &out[skip..skip + ANALYSIS];
    let w = kaiser(ANALYSIS, 30.0);
    let gain: f64 = w.iter().sum();
    // The tone's level: the windowed DTFT at its frequency.
    let level = if expect_tone {
        let omega_out = 2.0 * PI * freq / f64::from(to);
        let (mut sr, mut si) = (0.0, 0.0);
        for (k, (&v, &wk)) in y.iter().zip(&w).enumerate() {
            let (s, c) = (omega_out * (k + skip) as f64).sin_cos();
            sr += v * wk * c;
            si -= v * wk * s;
        }
        2.0 * sr.hypot(si) / gain
    } else {
        0.0
    };
    let mut re: Vec<f64> = y.iter().zip(&w).map(|(v, wk)| v * wk).collect();
    let mut im = vec![0.0; ANALYSIS];
    fft(&mut re, &mut im);
    let bin = f64::from(to) / ANALYSIS as f64;
    let tone_bin = freq / bin;
    let mut spur: f64 = 0.0;
    for k in 0..=ANALYSIS / 2 {
        if expect_tone && (k as f64 - tone_bin).abs() < 24.0 {
            continue;
        }
        spur = spur.max(2.0 * re[k].hypot(im[k]) / gain);
    }
    ToneResult {
        level_db: db(level),
        spur_db: db(spur),
    }
}

#[test]
fn passband_tones_keep_their_level_and_images_stay_down() {
    for (from, to) in MATRIX {
        let d = FilterDesign::shared(from, to);
        for freq in [997.0, 0.6 * d.passband_hz(), 0.98 * d.passband_hz()] {
            let r = tone_through(from, to, freq, true);
            println!(
                "{from:>6} -> {to:>6}  tone {freq:>8.1} Hz: level {:+.2e} dB, worst spur {:.1} dB",
                r.level_db, r.spur_db
            );
            assert!(
                r.level_db.abs() <= RIPPLE_DB,
                "{from}->{to} {freq}: {}",
                r.level_db
            );
            assert!(
                r.spur_db <= -STOPBAND_DB,
                "{from}->{to} {freq}: spur {} dB",
                r.spur_db
            );
        }
    }
}

#[test]
fn stopband_tones_do_not_alias_into_the_output() {
    for (from, to) in MATRIX.into_iter().filter(|(from, to)| from > to) {
        let d = FilterDesign::shared(from, to);
        let nyquist_in = 0.5 * f64::from(from);
        let stop = d.stopband_hz();
        for freq in [
            stop * 1.002,
            stop + 0.3 * (nyquist_in - stop),
            nyquist_in * 0.97,
        ] {
            let r = tone_through(from, to, freq, false);
            println!(
                "{from:>6} -> {to:>6}  stopband tone {freq:>9.1} Hz: worst output {:.1} dB",
                r.spur_db
            );
            assert!(
                r.spur_db <= -STOPBAND_DB,
                "{from}->{to} {freq}: {} dB",
                r.spur_db
            );
        }
    }
}

#[test]
fn impulse_response_is_symmetric_and_not_delayed() {
    for (from, to) in MATRIX {
        let design = FilterDesign::shared(from, to);
        let (up, down) = design.ratio();
        let half = design.half_width();
        // An impulse at an input frame that falls on an output frame.
        let n0 = (half + 10).div_ceil(down as usize) * down as usize;
        let j0 = n0 / down as usize * up as usize;
        let frames = n0 * 2 + 1;
        let mut input = vec![0.0; frames];
        input[n0] = 1.0;
        let mut r = Resampler::with_design(design.clone(), 1);
        let mut out = Vec::new();
        r.process_f64(&input, &mut out);
        r.finish_f64(&mut out);
        assert_eq!(out.len() as u64, output_frames(from, to, frames as u64));
        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, j0, "{from}->{to}");
        let reach = j0.min(out.len() - 1 - j0);
        for k in 1..=reach {
            let (a, b) = (out[j0 - k], out[j0 + k]);
            assert!((a - b).abs() <= 1e-15, "{from}->{to}: k {k}: {a} vs {b}");
        }
        // The peak is the prototype's centre tap (gain `up` over the
        // zero-stuffed input, so `up` times the cutoff over the rate).
        let centre = design.prototype()[half * up as usize];
        assert_eq!(out[j0], centre);
    }
}

#[test]
fn lengths_are_exact_and_chunking_does_not_matter() {
    for (from, to) in MATRIX {
        let design = FilterDesign::shared(from, to);
        for frames in [0usize, 1, 2, 3, 7, 160, 1001, 9_999] {
            let input: Vec<i32> = (0..frames * 2)
                .map(|k| ((k as i64 * 7_919 % 16_000_000) - 8_000_000) as i32 / 4)
                .collect();
            let mut whole = Vec::new();
            let mut r = Resampler::with_design(design.clone(), 2);
            r.process(&input, &mut whole);
            r.finish(&mut whole);
            assert_eq!(
                whole.len() as u64,
                2 * output_frames(from, to, frames as u64),
                "{from}->{to} {frames}"
            );
            assert_eq!(r.output_frames(), output_frames(from, to, frames as u64));
            for chunk in [1, 3, 1024] {
                let mut parts = Vec::new();
                r.reset();
                for c in input.chunks(chunk * 2) {
                    r.process(c, &mut parts);
                }
                r.finish(&mut parts);
                assert!(parts == whole, "{from}->{to} {frames} in chunks of {chunk}");
            }
        }
    }
}

#[test]
fn channels_are_independent() {
    let (from, to) = (44_100, 48_000);
    let design = FilterDesign::shared(from, to);
    let frames = 5_000;
    let channel = |c: usize| -> Vec<f64> {
        (0..frames)
            .map(|n| ((n * (c + 3)) as f64 * 0.01).sin() * (c as f64 + 1.0) / 9.0)
            .collect()
    };
    for channels in [1usize, 2, 6, 8] {
        let interleaved: Vec<f64> = (0..frames)
            .flat_map(|n| (0..channels).map(move |c| (n, c)))
            .map(|(n, c)| channel(c)[n])
            .collect();
        let mut r = Resampler::with_design(design.clone(), channels);
        let mut out = Vec::new();
        r.process_f64(&interleaved, &mut out);
        r.finish_f64(&mut out);
        for c in 0..channels {
            let mut mono = Resampler::with_design(design.clone(), 1);
            let mut alone = Vec::new();
            mono.process_f64(&channel(c), &mut alone);
            mono.finish_f64(&mut alone);
            let mine: Vec<f64> = out.iter().skip(c).step_by(channels).copied().collect();
            assert_eq!(mine, alone, "{channels} channels, channel {c}");
        }
    }
}

#[test]
fn requantization_is_dithered_24_bit() {
    // Silence stays near silence: TPDF dither of ±1 LSB rounds to -1, 0 or 1.
    let mut r = Resampler::new(44_100, 48_000, 2);
    let mut out = Vec::new();
    r.process(&vec![0; 2 * 44_100], &mut out);
    r.finish(&mut out);
    assert_eq!(out.len(), 2 * 48_000);
    assert!(out.iter().all(|s| (-1..=1).contains(s)));
    let nonzero = out.iter().filter(|&&s| s != 0).count();
    assert!(
        nonzero > out.len() / 10 && nonzero < out.len() / 2,
        "{nonzero}"
    );
    let mean: f64 = out.iter().map(|&s| f64::from(s)).sum::<f64>() / out.len() as f64;
    assert!(mean.abs() < 0.01, "{mean}");
    assert_eq!(r.clipped(), 0);
    // A quantized tone stays within a dither step of the unquantized one.
    let tone: Vec<i32> = (0..20_000)
        .map(|n| ((n as f64 * 0.05).sin() * 4_000_000.0).round() as i32)
        .collect();
    let mut q = Vec::new();
    let mut r = Resampler::new(48_000, 44_100, 1);
    r.process(&tone, &mut q);
    r.finish(&mut q);
    let mut f = Vec::new();
    let mut r = Resampler::new(48_000, 44_100, 1);
    let tone_f: Vec<f64> = tone.iter().map(|&s| f64::from(s)).collect();
    r.process_f64(&tone_f, &mut f);
    r.finish_f64(&mut f);
    assert_eq!(q.len(), f.len());
    for (a, b) in q.iter().zip(&f) {
        assert!((f64::from(*a) - b).abs() <= 1.5, "{a} vs {b}");
    }
}

#[test]
fn full_scale_overs_saturate_and_are_counted() {
    // A full-scale square wave rings past full scale on resampling.
    let square: Vec<i32> = (0..48_000)
        .map(|n| {
            if (n / 24) % 2 == 0 {
                8_388_607
            } else {
                -8_388_608
            }
        })
        .collect();
    let mut r = Resampler::new(48_000, 44_100, 1);
    let mut out = Vec::new();
    r.process(&square, &mut out);
    r.finish(&mut out);
    assert!(r.clipped() > 100, "{}", r.clipped());
    assert!(out.contains(&8_388_607));
    assert!(out.contains(&-8_388_608));
    r.reset();
    assert_eq!(r.clipped(), 0);
}

/// Speed per pair, the integer path. In release builds the worst pair must
/// run at least 20 times faster than real time in stereo and 5 times at 8
/// channels (generous bounds for a loaded machine; the printed margins are
/// larger).
#[test]
fn runs_faster_than_real_time() {
    if cfg!(debug_assertions) {
        println!("speed is measured in release builds only");
        return;
    }
    for (channels, bound) in [(2usize, 20.0), (8, 5.0)] {
        let mut worst = f64::INFINITY;
        for (from, to) in MATRIX {
            let mut r = Resampler::new(from, to, channels);
            let seconds = 2.0;
            let frames = (f64::from(from) * seconds) as usize;
            let input: Vec<i32> = (0..frames * channels)
                .map(|k| ((k as f64 * 0.001).sin() * 4e6) as i32)
                .collect();
            let mut out = Vec::with_capacity((frames * channels) * 9);
            let start = Instant::now();
            for chunk in input.chunks(4096 * channels) {
                r.process(chunk, &mut out);
            }
            r.finish(&mut out);
            let took = start.elapsed().as_secs_f64();
            let macs = (out.len() * r.design().taps_per_phase()) as f64;
            let realtime = seconds / took;
            worst = worst.min(realtime);
            println!(
                "{channels} ch {from:>6} -> {to:>6}: {:>5} taps, {:>7.1} M MAC/s needed, {:>6.0} M MAC/s done, {realtime:>6.1}x real time",
                r.design().taps_per_phase(),
                macs / seconds / 1e6,
                macs / took / 1e6,
            );
        }
        println!("{channels} channels, worst: {worst:.1}x real time");
        assert!(
            worst >= bound,
            "{channels} channels: only {worst:.1}x real time"
        );
    }
}

/// The stem lanes' resampling, as the engine runs it beside main: four
/// stereo stems (8 channels of `f64`) from the separator's 44.1 kHz to the
/// output's rate, in the engine's pieces of 4096 frames. Release builds
/// only; the numbers are printed (`PERFORMANCE.md` has them), and each
/// rate must keep at least twice ahead of real time (a generous bound for
/// a loaded machine).
#[test]
fn lane_resampling_keeps_ahead_of_real_time() {
    if cfg!(debug_assertions) {
        println!("speed is measured in release builds only");
        return;
    }
    let (from, channels, seconds) = (44_100u32, 8usize, 4.0);
    let frames = (f64::from(from) * seconds) as usize;
    let input: Vec<f64> = (0..frames * channels)
        .map(|k| (k as f64 * 0.001).sin() * 4e6)
        .collect();
    for to in [48_000u32, 96_000, 192_000, 384_000] {
        let mut r = Resampler::new(from, to, channels);
        let mut out = Vec::with_capacity(frames * channels * (to / from + 1) as usize);
        let start = Instant::now();
        for piece in input.chunks(4096 * channels) {
            r.process_float(piece, &mut out);
        }
        r.finish(&mut out);
        let took = start.elapsed().as_secs_f64();
        let made = (out.len() / channels) as f64;
        let realtime = seconds / took;
        println!(
            "8 ch 44100 -> {to:>6}: {:>4} taps a phase, {:>6.2} M output frames/s, {realtime:>6.1}x real time ({:.1} % of a core)",
            r.design().taps_per_phase(),
            made / took / 1e6,
            100.0 / realtime,
        );
        assert!(realtime >= 2.0, "{to} Hz: only {realtime:.1}x real time");
    }
}
