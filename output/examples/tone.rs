//! Plays a 440 Hz tone on a device, shared, and prints how its clock runs.
//!
//! `cargo run --release -p jodel-output --example tone -- [device] [seconds]`
//!
//! `device` is `default` (the default), an id, or part of a name. The tone
//! is −20 dBFS, stereo, at the rate the system runs the device at, in its
//! mix format, for `seconds` (3). Other applications play beside it. Every
//! 250 ms the example prints the device position against the time played
//! and the latency (written but not yet heard); at the end the stream's
//! counters, and how far the position ran from the wall clock.

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
    let query = args.first().map_or("default", String::as_str);
    let seconds: f64 = args.get(1).map_or(Ok(3.0), |s| s.parse())?;
    let device = find_device(query)?;
    let format = mix_format(&device.id)?;
    println!("device: {} ({}), takes {format}", device.name, device.id);
    let rate = format.rate;
    let mut out = open(&device.id, &OpenRequest::new(rate, format.channels))?;
    let info = out.info().clone();
    println!(
        "opened {} via {}: engine period {} frames ({:.1} ms), ring {} frames",
        info.format,
        info.backend,
        info.device_buffer_frames,
        info.device_buffer_frames as f64 * 1000.0 / f64::from(rate),
        info.ring_frames
    );
    let amplitude = 0.1 * 8_388_607.0;
    let total = (seconds * f64::from(rate)) as u64;
    let chunk = u64::from(rate / 100);
    let mut buf = vec![0i32; 2 * chunk as usize];
    let mut frame = 0u64;
    let mut tone = |out: &mut dyn Output, frame: &mut u64| -> Result<(), jodel_output::Error> {
        let n = chunk.min(total - *frame);
        for i in 0..n {
            let t = (*frame + i) as f64 / f64::from(rate);
            let v = (amplitude * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as i32;
            buf[2 * i as usize] = v;
            buf[2 * i as usize + 1] = v;
        }
        out.write(&buf[..2 * n as usize], Some(Duration::from_secs(2)))?;
        *frame += n;
        Ok(())
    };
    // 200 ms written, the prefill (100 ms) starting the device on the way;
    // 80 ms later, how far behind the time since the start its clock is:
    // what the system holds between the stream and the speakers.
    let start = loop {
        tone(&mut out, &mut frame)?;
        if frame >= u64::from(rate) / 10 {
            break Instant::now();
        }
    };
    while frame < u64::from(rate) / 5 {
        tone(&mut out, &mut frame)?;
    }
    std::thread::sleep(Duration::from_millis(80).saturating_sub(start.elapsed()));
    let (now, position) = (Instant::now(), out.position()?);
    println!(
        "started: {:.1} ms after the start the clock says {:.1} ms played: {:.1} ms behind",
        (now - start).as_secs_f64() * 1000.0,
        position as f64 * 1000.0 / f64::from(rate),
        (now - start).as_secs_f64() * 1000.0 - position as f64 * 1000.0 / f64::from(rate)
    );
    let first_heard = start;
    let mut last_print = Instant::now();
    while frame < total {
        tone(&mut out, &mut frame)?;
        let position = out.position()?;
        if last_print.elapsed() >= Duration::from_millis(250) {
            last_print = Instant::now();
            println!(
                "  t={:>6.3}s position {:>7} ({:>6.3}s) latency {:>6.1} ms",
                start.elapsed().as_secs_f64(),
                position,
                position as f64 / f64::from(rate),
                out.latency().as_secs_f64() * 1000.0
            );
        }
    }
    out.drain(Some(Duration::from_secs_f64(seconds + 2.0)))?;
    let done = Instant::now();
    let played = out.position()? as f64 / f64::from(rate);
    let wall = (done - first_heard).as_secs_f64();
    println!(
        "position ran {played:.3} s in {wall:.3} s of wall clock from its first frame ({:+.1} ms)",
        (played - wall) * 1000.0
    );
    println!("stats: {:?}", out.stats());
    out.close()?;
    Ok(())
}
