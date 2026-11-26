//! Takes the drums of a stereo track out on one bar and back on another, as
//! the player's stem keys do: the drums' strip of a desk muted on the bar's
//! frame of the player's output, and unmuted on a later one. It says how far
//! from those bars the changes were heard: the engine plays through a mixer
//! (the player's strip a group, a strip for each stem inside it) into a
//! memory output in real time, nothing goes to a device.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example stems_at -- FILE [--from SECONDS]
//! ```
//!
//! It plays from `--from` (default 60 s), the output at the file's rate, the
//! stems made on the GPU beside it. Once the mixer has them, a second on,
//! the drums go off on the first bar at least 0.1 s on, and come back on
//! the first bar at least a second after that bar was heard. Where the drums went and
//! came back is found in what was played less the file itself (which the
//! mixer plays bit for bit while no strip is touched: that is where the file
//! is found in it).

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use jodel_gpu::GpuSeparator;
use jodel_output::lanes::Parts;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::source::Source;
use jodel_playback::stems::{Separator, Stem};
use jodel_playback::{Event, MakeStems, MemoryTarget, Player, PlayerOptions, Position, StemDesk};
use jodel_rhythm::{Analysis, Method};
use jodeldiplom::core::RhythmRecipe;
use jodeldiplom::work::rhythm::analyze_file;

fn gpu() -> Option<&'static dyn Separator> {
    GpuSeparator::shared().map(|g| g as &dyn Separator)
}

fn main() -> ExitCode {
    let mut file = None;
    let mut from = 60.0;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--from" => from = args.next().and_then(|s| s.parse().ok()).unwrap_or(60.0),
            _ => file = Some(arg),
        }
    }
    let Some(file) = file else {
        eprintln!("usage: stems_at FILE [--from SECONDS]");
        return ExitCode::FAILURE;
    };
    match run(Path::new(&file), from) {
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

/// The output frame of `at` seconds into the track, from where position `p`
/// was heard (the output at the track's rate).
fn frame_of(p: &Position, at: f64, rate: u32) -> u64 {
    (p.output_frame as f64 + (at - p.seconds) * f64::from(rate))
        .round()
        .max(0.0) as u64
}

fn run(path: &Path, from: f64) -> Result<(), String> {
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
    if info.channels != 2 {
        return Err("a stereo file, please".into());
    }
    println!(
        "{:.1} BPM, {} bars, {} Hz",
        a.summary.map_or(0.0, |s| s.bpm),
        a.downbeats.len(),
        rate
    );
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(rate, 2, SampleFormat::S24In32Low));
    let device = MemoryTarget::new(caps, SimClock::Realtime);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    let (lead, drums) = (head.faders.input.clone(), desk.stem(Stem::Drums));
    let options = PlayerOptions {
        separator: Some(gpu as fn() -> _),
        position_interval: Duration::from_millis(20),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::new(head.target, options);
    player
        .make_stems(MakeStems::Separate)
        .map_err(|e| e.to_string())?;
    // When the drums go (a second after the mixer has the stems), and how
    // far ahead of what is heard: as the player's stem keys, a mixer's
    // change.
    let (wait, soon) = (1.0, 0.1);
    let mut seeked = None;
    // When the mixer first had the stems after the seek, in seconds of the
    // track.
    let mut present: Option<f64> = None;
    let stems_there = || {
        mixer.desk().strips.iter().any(|s| {
            s.inputs
                .iter()
                .any(|i| i.parts.as_ref().is_some_and(|p| p.state == Parts::Present))
        })
    };
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
    // The bars the drums go and come back on (seconds, number), and the
    // last second heard.
    let mut plan: Vec<(f64, usize)> = Vec::new();
    let heard = loop {
        match next()? {
            Event::Seeked { seconds, .. } => seeked = Some(seconds),
            Event::Position(p) if plan.is_empty() && seeked.is_some() && present.is_none() => {
                if stems_there() {
                    present = Some(p.seconds);
                }
            }
            Event::Position(p)
                if plan.is_empty() && present.is_some_and(|s| p.seconds >= s + wait) =>
            {
                let seconds = p.seconds;
                let (at, bar) = bar_after(&a, seconds + soon).ok_or("no bar left")?;
                drums.set_mute_at(true, lead.at_stream(frame_of(&p, at, rate)));
                println!("drums off at bar {bar} ({at:.3} s), asked at {seconds:.2} s");
                plan.push((at, bar));
            }
            Event::Position(p) if !plan.is_empty() => {
                let heard = p.seconds;
                if plan.len() == 1 && heard >= plan[0].0 + 0.1 {
                    let (at, bar) = bar_after(&a, heard + 1.0).ok_or("no bar left")?;
                    drums.set_mute_at(false, lead.at_stream(frame_of(&p, at, rate)));
                    println!("drums on at bar {bar} ({at:.3} s), asked at {heard:.2} s");
                    plan.push((at, bar));
                }
                if plan.len() == 2 && heard >= plan[1].0 + 3.0 {
                    break heard;
                }
            }
            Event::TrackStarted(s) => println!(
                "on {} from output frame {} (its frame {})",
                s.route.describe(),
                s.output_frame,
                s.from_frame
            ),
            Event::Error { error, .. } => return Err(error.to_string()),
            Event::PartsStopped { reason, .. } => return Err(reason),
            Event::QueueEnded => return Err("the track ended".into()),
            _ => {}
        }
    };
    drop(player);
    drop(mixer);
    // What was played and the file, frame by frame, over the stretch.
    let list = captures.list();
    let bytes = list.last().ok_or("nothing was played")?.sink.bytes();
    let out: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| (i32::from_le_bytes(*b) << 8) >> 8)
        .collect();
    let first = ((plan[0].0 - 2.0) * f64::from(rate)) as u64;
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
    // Where the file is in what was played: its frames from `first` on,
    // played as they are (nothing was touched yet).
    const PROBE: usize = 2 * 256;
    let probe = file.get(..PROBE).ok_or("the file ends too soon")?;
    let found = out
        .windows(PROBE)
        .step_by(2)
        .position(|w| w == probe)
        .ok_or("the file is not found in what was played")?;
    let offset = found as i64 - first as i64;
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
    let (off_at, on_at) = (plan[0].0, plan[1].0);
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
