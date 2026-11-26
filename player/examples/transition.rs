//! Renders a transition between two tracks to a 24-bit WAV file, to listen
//! to: the engine plays them on a desk with two heads, mixed into a memory
//! output in real time; nothing goes to a device.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example transition -- FROM INTO OUT.wav [--preset fade|drums|bass|vocals] [--stretch] [--bars N] [--rate HZ]
//! ```
//!
//! Both tracks are analysed (with stems on the GPU when there is one). The
//! window is `--bars` bars (default 8) of FROM ending about half a minute
//! before its end; INTO comes in at its first bar. The file holds from
//! twelve seconds before the window to ten after the next track is back at
//! its own tempo. At the end it says how far the beats of what was
//! rendered fall from FROM's beats in the window (from an analysis of the
//! file).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use jodel_gpu::GpuSeparator;
use jodel_output::mixer::{Mixer, MixerOptions, Start};
use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::stems::Separator;
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Event, MakeStems, MemoryTarget, Player, PlayerOptions, Settings, StemDesk, TempoMode,
    TransitionState,
};
use jodel_rhythm::{Analysis, Method};
use jodeldiplom::core::transition::{self, Preset};
use jodeldiplom::core::RhythmRecipe;
use jodeldiplom::work::rhythm::analyze_file;

/// Seconds of FROM before the window that the file starts at.
const BEFORE: f64 = 12.0;
/// Seconds after the next track is back at its tempo that the file ends.
const AFTER: f64 = 10.0;

fn gpu() -> Option<&'static dyn Separator> {
    GpuSeparator::shared().map(|g| g as &dyn Separator)
}

fn main() -> ExitCode {
    let mut files = Vec::new();
    let (mut preset, mut tempo, mut bars, mut rate) =
        (Preset::Fade, TempoMode::Varispeed, 8, 48_000);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--preset" => {
                preset = match args.next().as_deref() {
                    Some("fade") => Preset::Fade,
                    Some("drums") => Preset::DrumsFirst,
                    Some("bass") => Preset::BassSwap,
                    Some("vocals") => Preset::VocalsLast,
                    other => {
                        eprintln!("--preset takes fade, drums, bass or vocals: {other:?}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--stretch" => tempo = TempoMode::Stretch,
            "--bars" => bars = args.next().and_then(|b| b.parse().ok()).unwrap_or(8),
            "--rate" => rate = args.next().and_then(|r| r.parse().ok()).unwrap_or(48_000),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    let [from, into, out] = &files[..] else {
        eprintln!("usage: transition FROM INTO OUT.wav [--preset fade|drums|bass|vocals] [--stretch] [--bars N] [--rate HZ]");
        return ExitCode::FAILURE;
    };
    match render(from, into, out, preset, tempo, bars, rate) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn analysed(path: &Path) -> Result<Analysis, String> {
    let recipe = RhythmRecipe {
        methods: Method::ALL.to_vec(),
        stems: GpuSeparator::shared().is_some(),
    };
    analyze_file(path, &recipe).map_err(|e| format!("{}: {e}", path.display()))
}

#[allow(clippy::too_many_arguments)]
fn render(
    from: &Path,
    into: &Path,
    out: &Path,
    preset: Preset,
    tempo: TempoMode,
    bars: usize,
    rate: u32,
) -> Result<(), String> {
    let started = Instant::now();
    let (a, b) = (analysed(from)?, analysed(into)?);
    let bpm = |x: &Analysis| x.summary.map_or(0.0, |s| s.bpm);
    println!(
        "{:.1} BPM into {:.1} BPM ({:+.1} %), analysed in {:.1} s",
        bpm(&a),
        bpm(&b),
        100.0 * (bpm(&b) / bpm(&a) - 1.0),
        started.elapsed().as_secs_f64()
    );
    // The window: the bar about 30 s plus its length before FROM's end.
    let first = a
        .downbeats
        .iter()
        .rev()
        .copied()
        .find(|&d| a.beats[d] + 30.0 + bars as f64 * 2.5 < a.duration)
        .ok_or("the track is too short")?;
    // The beats paired as the Transition tab pairs them by the tempi (whole
    // bars of the slower track).
    let entry = b.downbeats.first().copied().unwrap_or(0);
    let span = transition::TEMPO_SPAN;
    let per = transition::pairing_for(
        transition::beat_tempo_near(&a, a.beats[first], span),
        transition::beat_tempo_near(&b, b.beats.get(entry).copied().unwrap_or(0.0), span),
    );
    let bars = bars.div_ceil(per.bars_per_step()) * per.bars_per_step();
    let window = transition::window(&a, first, bars).ok_or("no bars")?;
    let plan = transition::plan(0, &a, &b, window, entry, preset.lanes(window.2), tempo, per);
    println!(
        "beats paired {}, the tempo gliding across the window",
        per.label()
    );
    println!(
        "{} over {} bars from {:.2} s of the first, into {:.2} s of the next, {:?}",
        preset.name(),
        window.2,
        plan.start,
        plan.entry,
        tempo
    );
    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(rate, 2, SampleFormat::S24In32Low));
    let device = MemoryTarget::new(caps, SimClock::Realtime);
    let captures = device.captures();
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (_desk, lead, guest) = StemDesk::two_heads(&mixer, None, "Player", 1.0);
    let first = lead.faders.input.clone();
    let separator = GpuSeparator::shared().is_some().then_some(gpu as fn() -> _);
    let options = PlayerOptions {
        settings: Settings::default(),
        separator,
        position_interval: Duration::from_millis(50),
        ..PlayerOptions::default()
    };
    let (player, events) = Player::on_desk(lead, guest, options);
    // Each stem of either track faded by its own lane (without a separator
    // each track fades whole).
    if separator.is_some() {
        player
            .make_stems(MakeStems::Separate)
            .map_err(|e| e.to_string())?;
    }
    player
        .load(vec![from.to_path_buf(), into.to_path_buf()])
        .map_err(|e| e.to_string())?;
    player.play(0).map_err(|e| e.to_string())?;
    let next = |what: &str| -> Result<Event, String> {
        events
            .recv_timeout(Duration::from_secs(60))
            .map_err(|_| format!("no event while waiting for {what}"))
    };
    loop {
        if let Event::TrackStarted(_) = next("the start")? {
            break;
        }
    }
    player
        .seek(plan.start - BEFORE)
        .map_err(|e| e.to_string())?;
    loop {
        match next("the seek")? {
            Event::Seeked { .. } => break,
            Event::Error { error, .. } => return Err(error.to_string()),
            _ => {}
        }
    }
    player.transition(plan.clone()).map_err(|e| e.to_string())?;
    let (mut w0, mut finished_at) = (None, None);
    loop {
        match next("the transition")? {
            Event::Transition(TransitionState::Started { output_frame, .. }) => {
                // Where that frame of the playing track is on the device.
                let at = match first.at_stream(output_frame) {
                    Start::At { time, .. } => (time.as_secs_f64() * f64::from(rate)).round(),
                    _ => output_frame as f64,
                };
                println!("the window started at output frame {output_frame} (device frame {at})");
                w0 = Some(at as u64);
            }
            Event::Transition(TransitionState::Finished) => {
                println!("the next track is back at its own tempo");
                finished_at = Some(Instant::now());
            }
            Event::Transition(TransitionState::Cancelled { reason }) => {
                return Err(format!("the transition was called off: {reason}"));
            }
            Event::TrackStarted(s) => println!(
                "track {} started at output frame {} (its frame {}, {})",
                s.index,
                s.output_frame,
                s.from_frame,
                s.route.describe()
            ),
            Event::Error { error, .. } => return Err(error.to_string()),
            Event::Notice { message, .. } => println!("notice: {message}"),
            Event::QueueEnded => break,
            _ => {}
        }
        if finished_at.is_some_and(|t| t.elapsed().as_secs_f64() >= AFTER) {
            break;
        }
    }
    drop(player);
    drop(mixer);
    let w0 = w0.ok_or("the window never started")?;
    // The file starts BEFORE seconds ahead of the window: the track plays
    // straight from the seek up to there (the device's frames, as the desk
    // put the window's first frame on them).
    let cut = w0.saturating_sub((BEFORE * f64::from(rate)).round() as u64);
    // The stream from the seek on, as a WAV file.
    let list = captures.list();
    let stream = list.last().ok_or("nothing was played")?;
    let bytes = stream.sink.bytes();
    let samples: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .skip(2 * cut as usize)
        .map(|b| (i32::from_le_bytes(*b) << 8) >> 8)
        .collect();
    write_wav(out, rate, &samples)?;
    println!(
        "wrote {} ({:.1} s), rendered in {:.0} s",
        out.display(),
        samples.len() as f64 / 2.0 / f64::from(rate),
        started.elapsed().as_secs_f64()
    );
    // How far the rendered beats fall from FROM's in the window, and
    // before it (where only FROM plays: how close the analysis gets).
    let rendered = analysed(out)?;
    let window_out = (w0 - cut) as f64 / f64::from(rate);
    let before: Vec<f64> = a
        .beats
        .iter()
        .copied()
        .filter(|&t| t >= plan.start - BEFORE + 1.0 && t < plan.start)
        .collect();
    // In the window FROM is heard along the glide: each of its beats at
    // the output's time the glide puts it at (in FROM's time before the
    // window, where the two are the same).
    let glide = plan.glide_of(&plan.time_map());
    let within: Vec<f64> = plan.beats.iter().map(|b| glide.inverse(b.0)).collect();
    // After the window only INTO plays, along the map (in the output's
    // time) until it is back at its own tempo, then straight on.
    let map: TimeMap = TimeMap::new(&plan.beats, plan.settle).through_glide(&glide);
    let settled_b = map.at(map.settled_at());
    let end = glide.start() + glide.length();
    println!(
        "the first track's speed glides to {:.3} over {:.2} s; the next one's is {:.4} at the hand-over",
        glide.end_speed(),
        glide.length(),
        map.speed(end)
    );
    let after: Vec<f64> = b
        .beats
        .iter()
        .map(|&tb| {
            if tb <= settled_b {
                map.inverse(tb)
            } else {
                map.settled_at() + (tb - settled_b)
            }
        })
        .filter(|&t| t > end + 0.5 && t < end + plan.settle + AFTER)
        .collect();
    for (what, beats) in [
        ("before the window", before),
        ("in the window", within),
        ("after it (the next track's)", after),
    ] {
        let times: Vec<f64> = beats
            .iter()
            .map(|&t| window_out + (t - plan.start))
            .collect();
        report(what, &offsets(&rendered.beats, &times), times.len());
    }
    Ok(())
}

/// How far (ms) the nearest of `found` is from each of `times`, for those
/// within 100 ms, in order.
fn offsets(found: &[f64], times: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = times
        .iter()
        .filter_map(|&t| {
            let i = found.partition_point(|&b| b < t);
            let near = [i.checked_sub(1), Some(i)]
                .into_iter()
                .flatten()
                .filter_map(|j| found.get(j))
                .map(|&b| (b - t).abs())
                .min_by(f64::total_cmp)?;
            (near < 0.1).then_some(near * 1000.0)
        })
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

fn report(what: &str, offsets: &[f64], of: usize) {
    match offsets {
        [] => println!("rendered beats {what}: none near where they belong"),
        _ => println!(
            "rendered beats {what}: median {:.1} ms, worst {:.1} ms from where they belong ({} of {of})",
            offsets[offsets.len() / 2],
            offsets[offsets.len() - 1],
            offsets.len(),
        ),
    }
}

/// A 24-bit stereo WAV of `samples` (24-bit, interleaved).
fn write_wav(path: &Path, rate: u32, samples: &[i32]) -> Result<(), String> {
    durable::write_atomic(path, &jodel_dsp::wav::pcm24(rate, 2, samples)).map_err(|e| e.to_string())
}
