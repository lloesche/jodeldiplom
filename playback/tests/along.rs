//! A head read along a time map ([`Command::Along`]) on an output of
//! lanes: its stems are read along the map as its parts and its main is
//! their sum. A click track's clicks land on the stream frames the map puts
//! them on (varispeed and stretched, at the stems' rate and resampled to
//! another); main is the parts' sum while it is read along the map; once
//! the map runs straight it is back on its file, bit for bit; a mono file
//! reads along too; with no separator the whole mix is the other
//! instruments' part; and a file of more than two channels is refused.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use common::lanes::{caps_of, quarter, wide_target, Wide};
use common::transition::{clicks, FULL};
use common::*;
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Command, Event, MakeStems, Player, PlayerOptions, SeparatorSource, Settings, TempoMode,
};

/// The files' rate (the stems').
const RATE: u32 = 44_100;

/// The output second the head is read along the map from.
const FROM: f64 = 1.0;

/// The map: straight to 1 s, then gliding up to 5 % faster by 2 s and on
/// at that to 4 s, then back to its own tempo over a second (straight from
/// 5 s).
fn map() -> Arc<TimeMap> {
    Arc::new(TimeMap::from_knots(
        vec![(1.0, 1.0), (2.0, 2.05), (4.0, 4.15)],
        vec![1.0, 1.05, 1.05],
        1.0,
    ))
}

/// A file of `channels` channels (each the same) of `seconds` from a
/// sample function of time.
fn file(dir: &Path, name: &str, channels: u16, seconds: f64, f: impl Fn(f64) -> f64) -> PathBuf {
    let path = dir.join(name);
    let frames = (seconds * f64::from(RATE)) as usize;
    let samples: Vec<i32> = (0..frames)
        .flat_map(|n| {
            let v = (f(n as f64 / f64::from(RATE)) * FULL).round() as i32;
            std::iter::repeat_n(v, usize::from(channels))
        })
        .collect();
    write_wav(&path, RATE, channels, &samples);
    path
}

/// Clicks every quarter second from a tenth in.
fn click(t: f64) -> f64 {
    clicks(0.25, 0.1, 3_000.0)(t, 0)
}

/// How a file is played along [`map`].
#[derive(Clone, Copy)]
struct Run {
    /// The device's rate (another than the file's: resampled), and its
    /// channels (the file's).
    out: u32,
    channels: u16,
    tempo: TempoMode,
    separator: Option<SeparatorSource>,
    make: MakeStems,
}

impl Run {
    /// Stereo at the file's rate, varispeed, the quarter's stems made.
    fn new() -> Run {
        Run {
            out: RATE,
            channels: 2,
            tempo: TempoMode::Varispeed,
            separator: Some(quarter),
            make: MakeStems::Separate,
        }
    }

    /// The output frame the head is read along the map from.
    fn from(&self) -> u64 {
        (FROM * f64::from(self.out)) as u64
    }

    /// The frames of the move's crossfade (the player's switch fade).
    fn fade(&self) -> u64 {
        (0.02 * f64::from(self.out)) as u64
    }

    /// Plays `path` on a lane device, read along [`map`] from [`FROM`],
    /// to the end (offline, the parts waited for); its capture and the
    /// events.
    fn play(&self, path: PathBuf) -> (Wide, Vec<Event>) {
        let (target, captures, gate) = wide_target(caps_of(&[self.out], self.channels));
        let options = PlayerOptions {
            settings: Settings {
                max_rate: None,
                resample: self.out != RATE,
                ..Settings::default()
            },
            position_interval: Duration::from_millis(1),
            separator: self.separator,
            offline: true,
            ..PlayerOptions::default()
        };
        let (player, events) = Player::new(Box::new(target), options);
        player.make_stems(self.make).unwrap();
        gate.allow(4_096);
        player.load(vec![path]).unwrap();
        player.play(0).unwrap();
        let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        let along = Command::Along {
            map: map(),
            from: self.from(),
            tempo: self.tempo,
        };
        player.send(along).unwrap();
        gate.allow_all();
        seen.extend(until_queue_end(&events));
        drop(player);
        (Wide::of(&captures), seen)
    }
}

/// The output frame the head went back onto its file at, and the file's
/// frame there (the second track start: the switch back's).
fn back_at(seen: &[Event]) -> (u64, u64) {
    let starts: Vec<_> = seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some((s.output_frame, s.from_frame)),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2, "started, and back on its file: {starts:?}");
    starts[1]
}

/// The output seconds where clicks start in `wide`'s main: its first
/// channel's first frame past 15 % of full scale at least 0.1 s after the
/// last.
fn onsets(wide: &Wide) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for (f, frame) in wide.main.chunks_exact(wide.channels).enumerate() {
        let t = f as f64 / f64::from(wide.rate);
        if f64::from(frame[0]).abs() / FULL > 0.15 && out.last().is_none_or(|&l| t - l > 0.1) {
            out.push(t);
        }
    }
    out
}

/// Every click on the output second the map puts it on (within `within`
/// seconds, and a frame or two for the onset's threshold), from before the
/// map to after the head is back on its file.
fn clicks_on_the_map(wide: &Wide, within: f64) {
    let map = map();
    let got = onsets(wide);
    assert!(got.len() >= 20, "{got:?}");
    for (i, &t) in got.iter().enumerate() {
        let track = 0.1 + 0.25 * i as f64;
        let want = map.inverse(track);
        assert!(
            (t - want).abs() < within + 2.0 / f64::from(wide.rate),
            "click {i} (the track's {track} s) at {t} s, the map has it at {want} s: {got:?}"
        );
    }
}

/// While read along the map (past the move's crossfade, up to the move
/// back): main is the parts' sum, and the parts are there.
fn main_is_the_sum(wide: &Wide, run: &Run, back: u64) {
    let warped = (run.from() + run.fade()) as usize..back as usize;
    let (worst, there) = wide.worst(warped.clone());
    assert_eq!(there, warped.len(), "the parts are there all along");
    assert!(worst <= 2, "{worst} steps off the parts' sum");
}

/// The clicks of a click track read along the map by `run` on the frames
/// the map puts them on, main the parts' sum, the parts there all through,
/// and the positions the map's.
fn clicks_land(run: Run, within: f64) {
    let dir = temp_dir(&format!("along-clicks-{:?}-{}", run.tempo, run.out));
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let (wide, seen) = run.play(path);
    clicks_on_the_map(&wide, within);
    let (back, _) = back_at(&seen);
    let rate = f64::from(run.out);
    assert!(back as f64 / rate > 5.3, "back at {back}");
    main_is_the_sum(&wide, &run, back);
    // From the file's parts to the map's and back, said there all through.
    assert!(wide.control.iter().all(|&c| c == 1));
    let map = map();
    let mut checked = 0;
    for e in &seen {
        if let Event::Position(p) = e {
            let o = p.output_frame;
            if o > run.from() && o < back {
                let want = map.at(o as f64 / rate);
                assert!(
                    (p.seconds - want).abs() < 1e-6,
                    "{} s at {o}, the map's {want}",
                    p.seconds
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "positions while read along the map");
}

#[test]
fn varispeed_clicks_land_on_the_frames_the_map_puts_them_on() {
    clicks_land(Run::new(), 0.001);
}

#[test]
fn stretched_clicks_land_on_the_frames_the_map_puts_them_on() {
    let run = Run {
        tempo: TempoMode::Stretch,
        ..Run::new()
    };
    clicks_land(run, 0.004);
}

/// At 48 kHz the stems are read along the map at their own rate and then
/// resampled to the output's, main and parts alike.
#[test]
fn resampled_to_another_rate_the_clicks_land_too() {
    let run = Run {
        out: 48_000,
        ..Run::new()
    };
    clicks_land(run, 0.001);
}

/// Once the map runs straight the head goes back onto its file: past the
/// switch's crossfade, main is the file bit for bit, from the frame the
/// map put it on.
#[test]
fn back_on_its_file_bit_for_bit_once_the_map_runs_straight() {
    let dir = temp_dir("along-back");
    let path = file(&dir, "tone.wav", 2, 7.0, |t| {
        0.4 * (2.0 * std::f64::consts::PI * 441.0 * t).sin()
    });
    let samples = read_all(&path);
    let run = Run::new();
    let (wide, seen) = run.play(path);
    let (back, frame) = back_at(&seen);
    // Where the map has it then (on the file's grid: a frame here).
    let heard = map().at(back as f64 / f64::from(RATE)) * f64::from(RATE);
    assert!(
        (heard - frame as f64).abs() <= 1.0,
        "{frame} for the map's {heard}"
    );
    let from = (back + run.fade()) as usize;
    let tail = &wide.main[from * 2..];
    let file = &samples[(frame as usize + from - back as usize) * 2..];
    let n = tail.len().min(file.len());
    assert!(n > RATE as usize, "{n} frames after the move back");
    assert!(
        tail[..n] == file[..n],
        "main is the file after the move back"
    );
    main_is_the_sum(&wide, &run, back);
}

/// A mono file reads along the map too: its one channel doubled into its
/// stems' pair, and their pairs averaged back into its parts.
#[test]
fn a_mono_file_reads_along_the_map() {
    let dir = temp_dir("along-mono");
    let path = file(&dir, "mono.wav", 1, 7.0, click);
    let run = Run {
        channels: 1,
        ..Run::new()
    };
    let (wide, seen) = run.play(path);
    assert_eq!(wide.channels, 1);
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    main_is_the_sum(&wide, &run, back);
}

/// With no separator, the whole mix is the other instruments' part while
/// the head is read along the map (no transform: the other part is main
/// but for the silent parts' dither), the rest silent (their dither of a
/// step); the parts are there only then.
#[test]
fn with_no_separator_the_whole_mix_is_the_other_part() {
    let dir = temp_dir("along-whole");
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let run = Run {
        separator: None,
        make: MakeStems::Never,
        ..Run::new()
    };
    let (wide, seen) = run.play(path);
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    main_is_the_sum(&wide, &run, back);
    for f in (run.from() + run.fade()) as usize..back as usize {
        for c in 0..2 {
            let i = 2 * f + c;
            let silent = wide.parts[..3].iter().all(|p| p[i].abs() <= 1);
            assert!(silent, "only other, at {f}");
            let off = (wide.parts[3][i] - wide.main[i]).abs();
            assert!(off <= 3, "other is main, at {f}: {off} steps off");
        }
    }
    // None before the map (none asked for), none once back on the file.
    let before = run.from() as usize - 1;
    assert!(wide.control[..before].iter().all(|&c| c == 0));
    assert_eq!(wide.control.last(), Some(&0));
    wide.keeps_the_promise(&[]).unwrap();
}

/// A file of more than two channels is not read along a map: a notice
/// says why, and it plays on as it was.
#[test]
fn more_than_two_channels_are_refused() {
    let dir = temp_dir("along-six");
    let path = file(&dir, "six.wav", 6, 1.0, |t| {
        0.3 * (2.0 * std::f64::consts::PI * 300.0 * t).sin()
    });
    let samples = read_all(&path);
    let run = Run {
        channels: 6,
        make: MakeStems::Never,
        ..Run::new()
    };
    let (wide, seen) = run.play(path);
    let notice = seen.iter().find_map(|e| match e {
        Event::Notice { message, .. } => Some(message.clone()),
        _ => None,
    });
    let notice = notice.expect("a notice");
    assert!(notice.contains("6 channels"), "{notice}");
    assert!(wide.main == samples, "it plays on as it was");
}
