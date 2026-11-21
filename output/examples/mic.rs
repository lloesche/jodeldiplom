//! Plays digital silence with a microphone tapped beside it, to try the
//! microphone path: the voice is heard, cleaned and metered, never played.
//!
//! `cargo run -p jodel-output --example mic -- <output> <input> [rate] [seconds] [--mono first|mean|<n>] [--every MS]`
//!
//! With no arguments it lists the output and input devices. `<output>` and
//! `<input>` are `default`, an id, or part of a name. The output opens
//! exclusively in stereo at `rate` (48000 by default) for `seconds` (20);
//! the microphone opens in shared mode at its own rate and is tapped beside
//! it. Every `MS` (500) it prints the microphone ring's fill against its
//! target, the correction to the ratio of the two clocks, the voice's peak
//! as captured and as cleaned, and the counters.

use jodel_output::input::{find_input, input_devices, open_input_with, Mono, INPUT_BACKEND};
use jodel_output::{devices, find_device, open, OpenRequest, Output, BACKEND};
use std::time::{Duration, Instant};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn list() -> Result<(), Box<dyn std::error::Error>> {
    println!("outputs ({BACKEND}):");
    for d in devices()? {
        println!(
            "  {} {}  [{}]",
            if d.is_default { "*" } else { " " },
            d.name,
            d.id
        );
    }
    println!("inputs ({INPUT_BACKEND}):");
    for d in input_devices()? {
        println!(
            "  {} {}  [{}]",
            if d.is_default { "*" } else { " " },
            d.name,
            d.id
        );
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut mono = Mono::Auto;
    let mut every = Duration::from_millis(500);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--every" => {
                every = Duration::from_millis(it.next().ok_or("--every needs a value")?.parse()?)
            }
            "--mono" => {
                mono = match it.next().ok_or("--mono needs a value")?.as_str() {
                    "first" => Mono::First,
                    "mean" => Mono::Mean,
                    "auto" => Mono::Auto,
                    n => Mono::Channel(n.parse()?),
                }
            }
            _ => positional.push(a.as_str()),
        }
    }
    let (output, input) = match positional[..] {
        [] => return list(),
        [o, i, ..] => (o, i),
        _ => return Err("usage: mic <output> <input> [rate] [seconds] [--mono M]".into()),
    };
    let rate: u32 = positional.get(2).map_or(Ok(48_000), |r| r.parse())?;
    let seconds: f64 = positional.get(3).map_or(Ok(20.0), |s| s.parse())?;

    let output = find_device(output)?;
    let input = find_input(input)?;
    let mut out = open(&output.id, &OpenRequest::new(rate, 2))?;
    let info = out.info().clone();
    println!(
        "output: {} — {} via {}, device buffer {} frames ({:.1} ms), device latency {:?}",
        output.name,
        info.format,
        info.backend,
        info.device_buffer_frames,
        info.device_buffer_frames as f64 * 1000.0 / f64::from(rate),
        info.device_latency
    );
    let t_open = Instant::now();
    let mic = open_input_with(&input.id, mono)?;
    let m = mic.info().clone();
    println!(
        "input:  {} — {} Hz, {} ch ({:?}), period {} frames ({:.1} ms) via {}, opened in {:?}",
        m.name,
        m.rate,
        m.channels,
        m.mono,
        m.period_frames,
        f64::from(m.period_frames) * 1000.0 / f64::from(m.rate),
        m.backend,
        t_open.elapsed()
    );
    out.set_tap(Some(Box::new(mic.tap())))?;
    let mut voice = Vec::new();

    let chunk = (rate / 100) as usize;
    let silence = vec![0i32; 2 * chunk];
    let total = (seconds * f64::from(rate)) as u64;
    let start = Instant::now();
    let mut last = Instant::now();
    while out.written() < total {
        out.write(&silence, Some(Duration::from_secs(2)))?;
        if let Some(e) = mic.failure() {
            return Err(format!("capture failed: {e}").into());
        }
        if last.elapsed() >= every {
            last = Instant::now();
            let s = mic.status();
            let ms = |frames: f64| frames * 1000.0 / f64::from(s.rate.max(1));
            let peak = mic.peak();
            let cleaned = mic.voice_peak().unwrap_or(0.0);
            mic.take_voice(&mut voice);
            voice.clear();
            let dbfs = |p: f32| {
                if p > 0.0 {
                    20.0 * p.log10()
                } else {
                    f32::NEG_INFINITY
                }
            };
            println!(
                "  t={:>5.1}s fill {:>5.2} ms (target {:>5.2}, packet {} frames) adjust {:>+8.1} ppm peak {:>6.1} dBFS cleaned {:>6.1} dBFS  underruns {} overruns {} resyncs {}  output underruns {}",
                start.elapsed().as_secs_f64(),
                ms(s.fill),
                ms(f64::from(s.target)),
                s.packet_frames,
                s.adjust * 1e6,
                dbfs(peak),
                dbfs(cleaned),
                s.underruns,
                s.overruns,
                s.resyncs,
                out.stats().underruns
            );
        }
    }
    out.drain(Some(Duration::from_secs(2)))?;
    out.set_tap(None)?;
    let s = mic.status();
    println!(
        "done: adjust {:+.1} ppm, underruns {} overruns {} resyncs {}; output {:?}",
        s.adjust * 1e6,
        s.underruns,
        s.overruns,
        s.resyncs,
        out.stats()
    );
    out.close()?;
    Ok(())
}
