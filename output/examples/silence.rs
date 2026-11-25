//! Plays digital silence (all-zero samples) on a device, shared.
//!
//! `cargo run -p jodel-output --example silence -- <device> <seconds> [--controls]`
//!
//! `<device>` is `default`, a device id, or part of a device name (see the
//! `devices` example). The stream is in the device's mix format: stereo,
//! at the rate the system runs the device at. The example prints the
//! opened format, then the device position and latency as it plays, then
//! the counters. With `--controls` it also pauses for 200 ms halfway,
//! resumes, and flushes once near the end, checking the position each
//! time.

use jodel_output::{find_device, mix_format, open, OpenRequest, Output};
use std::time::{Duration, Instant};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let controls = args.iter().any(|a| a == "--controls");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [device, seconds] = positional[..] else {
        return Err("usage: silence <device> <seconds> [--controls]".into());
    };
    let seconds: f64 = seconds.parse()?;
    let device = find_device(device)?;
    println!("device: {} ({})", device.name, device.id);

    let t_open = Instant::now();
    let format = mix_format(&device.id)?;
    let rate = format.rate;
    let mut out = open(&device.id, &OpenRequest::new(rate, format.channels))?;
    let info = out.info().clone();
    println!(
        "opened in {:?}: {} via {}, device buffer {} frames, ring {} frames",
        t_open.elapsed(),
        info.format,
        info.backend,
        info.device_buffer_frames,
        info.ring_frames,
    );

    let total = (seconds * f64::from(rate)) as u64;
    let chunk = (rate / 100).max(1) as u64; // 10 ms
    let silence = vec![0i32; 2 * chunk as usize];
    let start = Instant::now();
    let mut last_print = Instant::now();
    let mut paused_once = false;
    let mut flushed_once = false;
    while out.written() < total {
        let n = chunk.min(total - out.written()) as usize;
        let queued = out.write(&silence[..2 * n], Some(Duration::from_secs(2)))?;
        if queued < n {
            return Err(format!("write queued {queued} of {n} frames within 2 s").into());
        }
        if last_print.elapsed() >= Duration::from_millis(250) {
            last_print = Instant::now();
            println!(
                "  t={:>6.3}s written {:>7} position {:>7} latency {:?}",
                start.elapsed().as_secs_f64(),
                out.written(),
                out.position()?,
                out.latency()
            );
        }
        if controls && !paused_once && out.position()? >= total / 2 {
            paused_once = true;
            out.pause()?;
            let p0 = out.position()?;
            std::thread::sleep(Duration::from_millis(200));
            let p1 = out.position()?;
            println!("  paused 200 ms: position {p0} -> {p1} (should hold)");
            out.resume()?;
        }
        if controls && !flushed_once && out.written() >= total * 9 / 10 {
            flushed_once = true;
            let before = (out.written(), out.position()?);
            out.flush()?;
            println!(
                "  flush: written {} position {} -> position {} (should equal written)",
                before.0,
                before.1,
                out.position()?
            );
        }
    }
    let drained = out.drain(Some(Duration::from_secs_f64(seconds + 2.0)))?;
    println!(
        "drained: {drained} after {:.3}s; written {} position {}",
        start.elapsed().as_secs_f64(),
        out.written(),
        out.position()?
    );
    println!("stats: {:?}", out.stats());
    let t_close = Instant::now();
    out.close()?;
    println!("closed in {:?}", t_close.elapsed());
    Ok(())
}
