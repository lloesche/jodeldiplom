//! Plays files through the engine and prints the route and every event.
//!
//! ```text
//! cargo run --release -p jodel-playback --example play -- FILE... [options]
//!   --device X       the output device: default, an id or part of a name
//!   --null           play into a null output on a real-time clock (no device)
//!   --seconds N      stop after N seconds
//! ```
//!
//! The player plays on an input of a mixer, which opens the device once,
//! shared, at the system's mix format (with `--null`, a stand-in for a
//! device at 48 kHz); a file at another rate is resampled to it.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::SimClock;
use jodel_playback::{
    any_capabilities, DeviceTarget, Event, NullTarget, OutputTarget, Player, PlayerOptions,
};

struct Arguments {
    files: Vec<PathBuf>,
    device: String,
    null: bool,
    seconds: Option<f64>,
}

fn parse() -> Result<Arguments, String> {
    let mut arguments = Arguments {
        files: Vec::new(),
        device: "default".into(),
        null: false,
        seconds: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--device" => arguments.device = value("--device")?,
            "--null" => arguments.null = true,
            "--seconds" => {
                arguments.seconds = Some(
                    value("--seconds")?
                        .parse()
                        .map_err(|_| "--seconds takes a number")?,
                )
            }
            "-h" | "--help" => return Err(String::new()),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            file => arguments.files.push(file.into()),
        }
    }
    if arguments.files.is_empty() {
        return Err("give at least one file".into());
    }
    Ok(arguments)
}

fn main() -> ExitCode {
    let arguments = match parse() {
        Ok(arguments) => arguments,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("play: {message}");
            }
            eprintln!("usage: play FILE... [--device X] [--null] [--seconds N]");
            return ExitCode::from(2);
        }
    };
    let target: Box<dyn OutputTarget> = if arguments.null {
        Box::new(NullTarget::new(
            any_capabilities(48_000),
            SimClock::Realtime,
        ))
    } else {
        match DeviceTarget::find(&arguments.device) {
            Ok(target) => Box::new(target),
            Err(e) => {
                eprintln!("play: {e}");
                return ExitCode::FAILURE;
            }
        }
    };
    println!("output: {}", target.name());
    let mixer = Mixer::open(target, MixerOptions::default());
    let (input, _fader) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let options = PlayerOptions {
        position_interval: Duration::from_millis(500),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::new(Box::new(input), options);
    if player.load(arguments.files).is_err() || player.play(0).is_err() {
        eprintln!("play: the engine stopped");
        return ExitCode::FAILURE;
    }
    let deadline = arguments
        .seconds
        .map(|s| Instant::now() + Duration::from_secs_f64(s.max(0.0)));
    let mut failed = false;
    loop {
        let wait = deadline.map_or(Duration::from_millis(200), |d| {
            d.saturating_duration_since(Instant::now())
                .min(Duration::from_millis(200))
        });
        if deadline.is_some_and(|d| Instant::now() >= d) {
            println!("stopping after {} s", arguments.seconds.unwrap_or(0.0));
            break;
        }
        let Ok(event) = events.recv_timeout(wait) else {
            continue;
        };
        match event {
            Event::QueueChanged { queue, .. } => println!("queue: {} tracks", queue.len()),
            Event::State(state) => println!("state: {state:?}"),
            Event::TrackStarted(start) => {
                let info = &start.info;
                let title = info.tags.title.as_deref().unwrap_or("(no title)");
                let artist = info.tags.artist.as_deref().unwrap_or("(no artist)");
                println!("track {}: {title} / {artist}", start.index + 1);
                println!("  file: {}", info.path.display());
                println!(
                    "  {} Hz, {} ch, {} bit, {:.2} s, cover: {}",
                    info.rate,
                    info.channels,
                    info.bits,
                    info.duration().map_or(0.0, |d| d.as_secs_f64()),
                    info.cover.as_ref().map_or("none".into(), |c| format!(
                        "{} bytes {}",
                        c.data.len(),
                        c.mime
                    )),
                );
                let route = &start.route;
                println!("  route: {}", route.describe());
                println!(
                    "  output: {} on {} ({})",
                    start.output.format, start.device, start.output.backend,
                );
            }
            Event::Position(p) => println!("  {:7.2} s  (frame {})", p.seconds, p.frame),
            Event::Seeked { seconds, .. } => println!("  seeked to {seconds:.2} s"),
            Event::Clipped { samples, .. } => println!("  resampler clipped {samples} samples"),
            Event::TrackEnded { index, finished } => println!(
                "track {} {}",
                index + 1,
                if finished { "ended" } else { "left" }
            ),
            Event::Error {
                index,
                error,
                skipped,
            } => {
                let which = index.map_or(String::new(), |i| format!("track {}: ", i + 1));
                println!(
                    "error: {which}{error}{}",
                    if skipped { " (skipped)" } else { "" }
                );
                failed |= !skipped;
            }
            Event::Notice { message, .. } => println!("  note: {message}"),
            Event::PartsStopped { reason, .. } => println!("  stem parts stopped: {reason}"),
            Event::Transition(state) => println!("  transition: {state:?}"),
            Event::Jumped { seconds, .. } => println!("  jumped to {seconds:.2} s"),
            Event::Key { semitones } => println!("  key: {semitones:+}"),
            Event::Held { index } => {
                println!("holding before track {index}");
                break;
            }
            Event::QueueEnded => {
                println!("queue ended");
                break;
            }
        }
    }
    drop(player);
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
