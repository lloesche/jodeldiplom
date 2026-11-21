//! Takes the drums of a track out on one bar and back on another, as the
//! player's stems do on bars, and says how far from those bars the changes
//! were heard: the engine plays into a memory output in real time, nothing
//! goes to a device.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example stems_at -- FILE [--from SECONDS] [--keep]
//! ```
//!
//! It plays from `--from` (default 60 s), the output at the file's rate. The drums go on the first bar at least
//! 2.5 s on (the track moves onto its stems first), and come back on the
//! first bar at least a second after that bar was heard. With `--keep` it
//! plays as the Mix tab does (the stems kept as the file plays, and a jump
//! window), and the drums go 4 s in, on the first bar a second on. They come back on the
//! first bar at least a second after that bar was heard. Where the drums
//! went and came back is found in what was played less the file itself.

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use jodel_gpu::GpuSeparator;
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::source::Source;
use jodel_playback::stems::{Separator, Stem, StemMix};
use jodel_playback::{Event, MemoryTarget, Player, PlayerOptions, Settings};
use jodel_rhythm::{Analysis, Method};
use jodeldiplom::core::RhythmRecipe;
use jodeldiplom::rhythm::analyze_file;

fn gpu() -> Option<&'static dyn Separator> {
    GpuSeparator::shared().map(|g| g as &dyn Separator)
}

fn main() -> ExitCode {
    let mut file = None;
    let mut from = 60.0;
    let mut keep = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--from" => from = args.next().and_then(|s| s.parse().ok()).unwrap_or(60.0),
            "--keep" => keep = true,
            _ => file = Some(arg),
        }
    }
    let Some(file) = file else {
        eprintln!("usage: stems_at FILE [--from SECONDS] [--keep]");
        return ExitCode::FAILURE;
    };
    match run(Path::new(&file), from, keep) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// The first bar of `a` at or after `t`: its time and number.
fn bar_after(a: &Analysis, t: f64) -> Option<(f64, usize)> {
    a.downbeats
        .iter()
        .enumerate()
        .map(|(k, &d)| (a.beats[d], k + 1))
        .find(|&(b, _)| b >= t)
}

fn run(path: &Path, from: f64, keep: bool) -> Result<(), String> {
    if GpuSeparator::shared().is_none() {
        return Err("stems need a GPU".into());
    }
    let recipe = RhythmRecipe {
        methods: Method::ALL.to_vec(),
        stems: true,
    };
    let a = analyze_file(path, &recipe).map_err(|e| e.to_string())?;
    let info = Source::open(path)
        .map_err(|e| e.to_string())?
        .info()
        .clone();
    let rate = info.rate;
    println!(
        "{:.1} BPM, {} bars, {} Hz",
        a.summary.map_or(0.0, |s| s.bpm),
        a.downbeats.len(),
        rate
    );
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(rate, 2, SampleFormat::S24In32Low));
    let target = MemoryTarget::new(caps, SimClock::Realtime);
    let captures = target.captures();
    let options = PlayerOptions {
        settings: Settings::default(),
        separator: Some(gpu as fn() -> _),
        position_interval: Duration::from_millis(20),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::new(Box::new(target), options);
    if keep {
        player.jump_window(24.0, 24.0).map_err(|e| e.to_string())?;
        player.keep_stems(true).map_err(|e| e.to_string())?;
    }
    // When the drums go: at once after the seek, or (stems kept) 4 s on,
    // and how far ahead of what is heard.
    let (wait, lead) = if keep { (4.0, 1.0) } else { (0.0, 2.5) };
    let mut seeked = None;
    player
        .load(vec![path.to_path_buf()])
        .map_err(|e| e.to_string())?;
    player.play(0).map_err(|e| e.to_string())?;
    player.seek(from).map_err(|e| e.to_string())?;
    let next = || {
        events
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| "the engine went quiet".to_owned())
    };
    let mut plan: Vec<(StemMix, f64, usize)> = Vec::new();
    let drums_off = StemMix::ALL.toggle(Stem::Drums);
    // Where the file is in the output (from the positions heard), and the
    // last second heard.
    let (offset, heard) = loop {
        match next()? {
            Event::Seeked { seconds, .. } => seeked = Some(seconds),
            Event::Position(p)
                if plan.is_empty() && seeked.is_some_and(|s| p.seconds >= s + wait) =>
            {
                let seconds = p.seconds;
                let (at, bar) = bar_after(&a, seconds + lead).ok_or("no bar left")?;
                player
                    .set_stems_at(drums_off, at)
                    .map_err(|e| e.to_string())?;
                println!("drums off at bar {bar} ({at:.3} s), asked at {seconds:.2} s");
                plan.push((drums_off, at, bar));
            }
            Event::Position(p) if !plan.is_empty() => {
                let heard = p.seconds;
                let (_, first, _) = plan[0];
                if plan.len() == 1 && heard >= first + 0.1 {
                    let (at, bar) = bar_after(&a, heard + 1.0).ok_or("no bar left")?;
                    player
                        .set_stems_at(StemMix::ALL, at)
                        .map_err(|e| e.to_string())?;
                    println!("drums on at bar {bar} ({at:.3} s), asked at {heard:.2} s");
                    plan.push((StemMix::ALL, at, bar));
                }
                if plan.len() == 2 && heard >= plan[1].1 + 3.0 {
                    break (p.output_frame as i64 - p.frame as i64, heard);
                }
            }
            Event::TrackStarted(s) => println!(
                "on {:?} from output frame {} (its frame {})",
                s.route.stage, s.output_frame, s.from_frame
            ),
            Event::Error { error, .. } => return Err(error.to_string()),
            Event::StemsStopped { reason, .. } => return Err(reason),
            Event::QueueEnded => return Err("the track ended".into()),
            _ => {}
        }
    };
    drop(player);
    // What was played and the file, frame by frame, over the stretch.
    let list = captures.list();
    let bytes = list.last().ok_or("nothing was played")?.sink.bytes();
    let out: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| (i32::from_le_bytes(*b) << 8) >> 8)
        .collect();
    let first = ((plan[0].1 - 2.0) * f64::from(rate)) as u64;
    let last = ((heard - 0.2) * f64::from(rate)) as u64;
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    source.seek(first).map_err(|e| e.to_string())?;
    let mut file = Vec::new();
    let channels = usize::from(info.channels);
    while ((file.len() / channels) as u64) < last - first {
        if source.read(&mut file).map_err(|e| e.to_string())? == 0 {
            break;
        }
    }
    // The level of what was played less the file, every 5 ms.
    let step = (f64::from(rate) * 0.005) as u64;
    let level = |f0: u64| -> f64 {
        let mut sum = 0.0;
        for f in f0..f0 + step {
            let o = (f as i64 + offset) as usize;
            let i = (f - first) as usize;
            if 2 * o + 1 >= out.len() || channels * i + 1 >= file.len() {
                return 0.0;
            }
            for c in 0..2 {
                let d =
                    f64::from(out[2 * o + c]) - f64::from(file[channels * i + c.min(channels - 1)]);
                sum += d * d;
            }
        }
        (sum / (2 * step) as f64).sqrt()
    };
    let levels: Vec<(f64, f64)> = (first..last - step)
        .step_by(step as usize)
        .map(|f| (f as f64 / f64::from(rate), level(f)))
        .collect();
    let median = |xs: &mut Vec<f64>| {
        xs.sort_by(f64::total_cmp);
        xs.get(xs.len() / 2).copied().unwrap_or(0.0)
    };
    let (off_at, on_at) = (plan[0].1, plan[1].1);
    let mut quiet: Vec<f64> = levels
        .iter()
        .filter(|(t, _)| *t < off_at - 0.2 || *t > on_at + 0.5)
        .map(|l| l.1)
        .collect();
    let mut loud: Vec<f64> = levels
        .iter()
        .filter(|(t, _)| *t > off_at + 0.2 && *t < on_at - 0.2)
        .map(|l| l.1)
        .collect();
    let (quiet, loud) = (median(&mut quiet), median(&mut loud));
    let half = 0.5 * (quiet + loud);
    println!("what was played less the file: {quiet:.0} steps with the drums, {loud:.0} without");
    let went = levels
        .iter()
        .find(|(t, l)| *t > off_at - 1.0 && *l > half)
        .map(|l| l.0);
    // Back after the last stretch without them (the drums may rest in the
    // music before the bar).
    let back = levels
        .iter()
        .rev()
        .find(|(t, l)| *t > off_at && *t < on_at + 1.0 && *l > half)
        .map(|l| l.0 + step as f64 / f64::from(rate));
    for (what, at, seen) in [("went", off_at, went), ("came back", on_at, back)] {
        match seen {
            Some(t) => println!(
                "the drums {what} at {t:.3} s: {:+.0} ms from the bar",
                1000.0 * (t - at)
            ),
            None => println!("the drums never {what}"),
        }
    }
    Ok(())
}
