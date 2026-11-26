//! Bar jumps on a real track, as the Transition tab's digits make them, in real
//! time through a mixer (the player's strip a group, a strip for each stem
//! inside it) into a memory output (nothing goes to a device): to the bar
//! heard, the one before, three ahead, one at the bar's end, a fast run of
//! them (a stutter), and the same with the drums' strip muted (the mixer
//! playing the stems made beside the file, which follow each jump; the first
//! jump right after the drums went off). It says how long after each command
//! the jump was heard and the longest run of digital silence in what was
//! played: a gap (an underrun is padded with silence) would show there.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example jumps -- FILE [--from SECONDS]
//! ```
//!
//! It plays at the file's rate.

use std::path::Path;
use std::process::ExitCode;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use jodel_gpu::GpuSeparator;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::source::Source;
use jodel_playback::stems::{Separator, Stem};
use jodel_playback::{Event, JumpAt, MakeStems, MemoryTarget, Player, PlayerOptions, StemDesk};
use jodel_rhythm::{Analysis, Method};
use jodeldiplom::core::transition::jump::bar_jump;
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
        eprintln!("usage: jumps FILE [--from SECONDS]");
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

/// The player and what it reports, with the last second heard.
struct Session {
    player: Player,
    events: Receiver<Event>,
    heard: f64,
}

impl Session {
    /// Takes the events for `wait`.
    fn pump(&mut self, wait: Duration) -> Vec<Event> {
        let until = Instant::now() + wait;
        let mut seen = Vec::new();
        while let Some(left) = until.checked_duration_since(Instant::now()) {
            match self.events.recv_timeout(left) {
                Ok(event) => {
                    if let Event::Position(p) = &event {
                        self.heard = p.seconds;
                    }
                    seen.push(event);
                }
                Err(_) => break,
            }
        }
        seen
    }

    /// A jump to the bar `offset` bars from the one heard; how long until
    /// it was heard, and where to.
    fn jump(&mut self, a: &Analysis, offset: isize, at_end: bool) -> Result<String, String> {
        let (to, end) = bar_jump(a, self.heard, offset).ok_or("no bar there")?;
        let at = if at_end { JumpAt::At(end) } else { JumpAt::Now };
        let asked = Instant::now();
        let heard = self.heard;
        self.player.jump(0, to, at).map_err(|e| e.to_string())?;
        let until = asked + Duration::from_secs(10);
        while Instant::now() < until {
            for event in self.pump(Duration::from_millis(2)) {
                if let Event::Jumped { seconds, .. } = event {
                    let ms = asked.elapsed().as_secs_f64() * 1e3;
                    let wait = if at_end {
                        format!(
                            " (the bar ended {:.0} ms after it was asked)",
                            (end - heard) * 1e3
                        )
                    } else {
                        String::new()
                    };
                    return Ok(format!(
                        "{offset:+} bars{}: heard {ms:.0} ms after the command{wait}, at {seconds:.3} s (bar start {to:.3} s)",
                        if at_end { " at the bar's end" } else { "" }
                    ));
                }
            }
        }
        Err(format!("the jump {offset:+} was never heard"))
    }
}

/// The longest run of samples that are all zero (both channels), in ms.
fn longest_silence(samples: &[i32], rate: u32) -> f64 {
    let (mut run, mut most) = (0usize, 0usize);
    for [left, right] in samples.as_chunks::<2>().0 {
        if *left == 0 && *right == 0 {
            run += 1;
            most = most.max(run);
        } else {
            run = 0;
        }
    }
    most as f64 * 1e3 / f64::from(rate)
}

fn run(path: &Path, from: f64) -> Result<(), String> {
    let recipe = RhythmRecipe {
        methods: Method::ALL.to_vec(),
        stems: GpuSeparator::shared().is_some(),
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
    let device = MemoryTarget::new(caps, SimClock::Realtime);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    let options = PlayerOptions {
        separator: Some(gpu as fn() -> _),
        position_interval: Duration::from_millis(5),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::new(head.target, options);
    player.jump_window(24.0, 24.0).map_err(|e| e.to_string())?;
    // The stems made beside the file throughout, as the Transition tab has
    // them, so a strip muted is heard at once.
    if GpuSeparator::shared().is_some() {
        player
            .make_stems(MakeStems::Separate)
            .map_err(|e| e.to_string())?;
    }
    player
        .load(vec![path.to_path_buf()])
        .map_err(|e| e.to_string())?;
    player.play(0).map_err(|e| e.to_string())?;
    player.seek(from).map_err(|e| e.to_string())?;
    let mut s = Session {
        player,
        events,
        heard: 0.0,
    };
    // The window fills.
    s.pump(Duration::from_secs(6));
    for (offset, at_end, then) in [
        (0, false, 1500),
        (-1, false, 1500),
        (3, false, 1500),
        (0, true, 2500),
        (0, false, 120),
        (0, false, 120),
        (0, false, 120),
        (0, false, 1500),
    ] {
        println!("  {}", s.jump(&a, offset, at_end)?);
        s.pump(Duration::from_millis(then));
    }
    if GpuSeparator::shared().is_some() {
        println!("drums off (their strip muted, the stems played on the desk):");
        desk.stem(Stem::Drums).set_mute(true);
        // The drums go off, then the bar heard again, which started before
        // they did.
        s.pump(Duration::from_millis(600));
        for (offset, then) in [
            (0, 1500),
            (-1, 1500),
            (2, 1500),
            (0, 120),
            (0, 120),
            (0, 1500),
        ] {
            println!("  {}", s.jump(&a, offset, false)?);
            s.pump(Duration::from_millis(then));
        }
    }
    s.player.stop().map_err(|e| e.to_string())?;
    s.pump(Duration::from_millis(300));
    drop(s);
    drop(mixer);
    for capture in captures.list() {
        let bytes = capture.sink.bytes();
        let samples: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| (i32::from_le_bytes(b) << 8) >> 8)
            .collect();
        // The stream's own start (before the seek took) is left out.
        let skip = (rate as usize * 2).min(samples.len() / 2) & !1;
        println!(
            "stream at {} Hz: {:.1} s played, longest digital silence after its first second {:.2} ms",
            capture.format.rate,
            samples.len() as f64 / 2.0 / f64::from(capture.format.rate),
            longest_silence(&samples[skip..], capture.format.rate)
        );
    }
    Ok(())
}
