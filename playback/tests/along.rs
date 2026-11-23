//! A head read along a time map ([`Command::Along`]) on an output of
//! lanes: its stems are read along the map as its parts and its main is
//! their sum. A click track's clicks land on the stream frames the map puts
//! them on (varispeed and stretched, at the stems' rate and resampled to
//! another); main is the parts' sum while it is read along the map, and is
//! crossfaded in once; once the map runs straight it is back on its file,
//! bit for bit (also stretched and resampled); new settings and a
//! transition wait or are refused meanwhile; a switch back that does not
//! come about is asked again; a mono file reads along too; with no
//! separator the whole mix is the other instruments' part; on a device,
//! stems that fall behind give way to the whole mix; and a file of more
//! than two channels is refused.

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::lanes::{caps_of, quarter, wide_target, Gated, Wide, CLOSE};
use common::transition::{beats, clicks, lanes, FULL};
use common::*;
use jodel_playback::stems::Separator;
use jodel_playback::warp::TimeMap;
use jodel_playback::{
    Command, Envelope, Event, MakeStems, Player, PlayerOptions, SeparatorSource, Settings,
    TempoMode, TransitionPlan, TransitionState,
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

/// Three tones that never line up again: any two stretches differ.
fn chord(t: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    0.2 * ((tau * 311.0 * t).sin() + (tau * 467.3 * t).sin() + (tau * 1_013.7 * t).sin())
}

/// How a file is played along a map.
#[derive(Clone)]
struct Run {
    /// The device's rate (another than the file's: resampled), and its
    /// channels (the file's).
    out: u32,
    channels: u16,
    tempo: TempoMode,
    separator: Option<SeparatorSource>,
    make: MakeStems,
    /// A device: the stems are not waited for.
    device: bool,
    map: Arc<TimeMap>,
}

impl Run {
    /// Stereo at the file's rate, varispeed along [`map`], offline, the
    /// quarter's stems made.
    fn new() -> Run {
        Run {
            out: RATE,
            channels: 2,
            tempo: TempoMode::Varispeed,
            separator: Some(quarter),
            make: MakeStems::Separate,
            device: false,
            map: map(),
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

    fn options(&self) -> PlayerOptions {
        PlayerOptions {
            settings: Settings {
                max_rate: None,
                resample: self.out != RATE,
                ..Settings::default()
            },
            position_interval: Duration::from_millis(1),
            separator: self.separator,
            offline: !self.device,
            ..PlayerOptions::default()
        }
    }

    /// Plays `paths` on a lane device, the first read along the map from
    /// [`FROM`], to the end; its capture and the events.
    fn play(&self, paths: Vec<PathBuf>) -> (Wide, Vec<Event>) {
        self.play_with(paths, &mut |_, _| {})
    }

    /// The same, the output let through a piece at a time, `on` told of
    /// each event as it comes (to send commands mid-way).
    fn play_with(
        &self,
        paths: Vec<PathBuf>,
        on: &mut dyn FnMut(&Player, &Event),
    ) -> (Wide, Vec<Event>) {
        let (target, captures, gate) = wide_target(caps_of(&[self.out], self.channels));
        let (player, events) = Player::new(Box::new(target), self.options());
        player.make_stems(self.make).unwrap();
        gate.allow(4_096);
        player.load(paths).unwrap();
        player.play(0).unwrap();
        let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        let along = Command::Along {
            map: Arc::clone(&self.map),
            from: self.from(),
            tempo: self.tempo,
        };
        player.send(along).unwrap();
        seen.extend(step_to_end(&gate, &events, &player, on));
        drop(player);
        (Wide::of(&captures), seen)
    }
}

/// Lets the output through 4096 frames at a time (the next piece once the
/// last is written), telling `on` of each event, until the queue ends.
fn step_to_end(
    gate: &Gate,
    events: &Receiver<Event>,
    player: &Player,
    on: &mut dyn FnMut(&Player, &Event),
) -> Vec<Event> {
    let deadline = Instant::now() + TIMEOUT * 2;
    let mut seen = Vec::new();
    loop {
        let target = gate.written() + 4_096;
        gate.allow(target);
        loop {
            assert!(Instant::now() < deadline, "the queue did not end");
            match events.recv_timeout(Duration::from_millis(2)) {
                Ok(e) => {
                    on(player, &e);
                    let end = matches!(e, Event::QueueEnded);
                    seen.push(e);
                    if end {
                        return seen;
                    }
                }
                Err(RecvTimeoutError::Timeout) if gate.written() >= target => break,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => panic!("{e}"),
            }
        }
    }
}

/// The track starts heard: (output frame, file frame).
fn starts(seen: &[Event]) -> Vec<(u64, u64)> {
    seen.iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some((s.output_frame, s.from_frame)),
            _ => None,
        })
        .collect()
}

/// The output frame the head went back onto its file at, and the file's
/// frame there (the second track start: the switch back's).
fn back_at(seen: &[Event]) -> (u64, u64) {
    let starts = starts(seen);
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
    let (wide, seen) = run.play(vec![path]);
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
/// switch's crossfade, main is `reference` (a plain play from the frame
/// the map put it on) to within `dither` steps (none at the file's rate).
fn back_bit_for_bit(run: Run, dither: i32, reference: impl Fn(&Path, u64) -> Vec<i32>) {
    let dir = temp_dir(&format!("along-back-{:?}-{}", run.tempo, run.out));
    let path = file(&dir, "chord.wav", 2, 7.0, chord);
    let (wide, seen) = run.play(vec![path.clone()]);
    let (back, frame) = back_at(&seen);
    // Where the map has it then, to a grid step of the file's route.
    let heard = map().at(back as f64 / f64::from(run.out)) * f64::from(RATE);
    assert!(
        (heard - frame as f64).abs() <= 1.0,
        "{frame} for the map's {heard}"
    );
    let plain = reference(&path, frame);
    // Past the crossfade and the reference resampler's run-up.
    let skip = run.fade() as usize + 1_024;
    let from = back as usize + skip;
    let tail = &wide.main[from * 2..];
    let plain = &plain[skip * 2..];
    let n = tail.len().min(plain.len());
    assert!(n > RATE as usize, "{n} frames after the move back");
    if let Some(i) = (0..n).find(|&i| (tail[i] - plain[i]).abs() > dither) {
        panic!(
            "main is not the file at {} after the move back",
            from + i / 2
        );
    }
    main_is_the_sum(&wide, &run, back);
}

/// The file from `frame` on, as it is.
fn the_file(path: &Path, frame: u64) -> Vec<i32> {
    read_all(path).split_off(frame as usize * 2)
}

#[test]
fn back_on_its_file_bit_for_bit_once_the_map_runs_straight() {
    back_bit_for_bit(Run::new(), 0, the_file);
}

/// Stretched, it goes back once the stretch is steered onto the map.
#[test]
fn stretched_back_on_its_file_bit_for_bit_too() {
    let run = Run {
        tempo: TempoMode::Stretch,
        ..Run::new()
    };
    back_bit_for_bit(run, 0, the_file);
}

/// Resampled to 48 kHz (a grid of 147 file frames to 160 of the output's),
/// it goes back on its route's grid, as the file plays on that route from
/// there: the same samples to within the dither (a resampler's dither runs
/// on from where it started, the switch's run-up, as a route switch's
/// does: `tests/switch.rs`).
#[test]
fn resampled_back_on_its_route_to_within_dither() {
    let run = Run {
        out: 48_000,
        ..Run::new()
    };
    let plain = |path: &Path, frame: u64| {
        let (target, captures, gate) = wide_target(caps_of(&[48_000], 2));
        let (player, events) = Player::new(Box::new(target), run.options());
        player.load(vec![path.to_path_buf()]).unwrap();
        player.play(0).unwrap();
        collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        player.seek(frame as f64 / f64::from(RATE)).unwrap();
        collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(&events);
        drop(player);
        Wide::of(&captures).main
    };
    back_bit_for_bit(run.clone(), 2, plain);
}

/// The move onto the map crossfades main from the file's once: over the
/// crossfade, main is the parts' sum as closely as the separator's stems
/// add up to the song (crossfaded twice, the new side would come in late).
#[test]
fn the_move_onto_the_map_crossfades_once() {
    let dir = temp_dir("along-fade");
    let path = file(&dir, "chord.wav", 2, 3.0, chord);
    // Half a second further into the track from the move on.
    let run = Run {
        map: Arc::new(TimeMap::from_knots(vec![(0.0, 0.5)], vec![1.0], 0.0)),
        ..Run::new()
    };
    let map = Arc::clone(&run.map);
    let (wide, _) = run.play(vec![path]);
    let fade = run.from() as usize..(run.from() + run.fade()) as usize;
    let (worst, there) = wide.worst(fade.clone());
    assert_eq!(there, fade.len());
    assert!(worst <= CLOSE, "{worst} steps off over the crossfade");
    // Past it, the track half a second on.
    let o = run.from() as usize + 2 * run.fade() as usize;
    let at = (map.at(o as f64 / f64::from(RATE)) * f64::from(RATE)).round() as usize;
    let off =
        (i64::from(wide.main[2 * o]) - (chord(at as f64 / f64::from(RATE)) * FULL) as i64).abs();
    assert!(off <= CLOSE, "{off} steps off the track half a second on");
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
    let (wide, seen) = run.play(vec![path]);
    assert_eq!(wide.channels, 1);
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    main_is_the_sum(&wide, &run, back);
}

/// With no separator, the whole mix is the other instruments' part while
/// the head is read along the map (no transform: the other part is main),
/// the rest exactly silent; the parts are there only then.
#[test]
fn with_no_separator_the_whole_mix_is_the_other_part() {
    let dir = temp_dir("along-whole");
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let run = Run {
        separator: None,
        make: MakeStems::Never,
        ..Run::new()
    };
    let (wide, seen) = run.play(vec![path]);
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    main_is_the_sum(&wide, &run, back);
    for f in (run.from() + run.fade()) as usize..back as usize {
        for c in 0..2 {
            let i = 2 * f + c;
            assert!(
                wide.parts[..3].iter().all(|p| p[i] == 0),
                "only other, at {f}"
            );
            assert_eq!(wide.parts[3][i], wide.main[i], "other is main, at {f}");
        }
    }
    // None before the map (none asked for), none once back on the file.
    let before = run.from() as usize - 1;
    assert!(wide.control[..before].iter().all(|&c| c == 0));
    assert_eq!(wide.control.last(), Some(&0));
    wide.keeps_the_promise(&[]).unwrap();
}

/// The first time a position past `seconds` is heard.
fn past(e: &Event, seconds: f64) -> bool {
    matches!(e, Event::Position(p) if p.output_frame as f64 / f64::from(RATE) > seconds)
}

/// New settings while the head is read along the map do not move it off
/// it: they apply once it is back on its file.
#[test]
fn new_settings_wait_for_the_map() {
    let dir = temp_dir("along-settings");
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let run = Run::new();
    let mut sent = false;
    let (wide, seen) = run.play_with(vec![path], &mut |player, e| {
        if !sent && past(e, 2.0) {
            let settings = Settings {
                max_rate: None,
                ..Settings::default()
            };
            player.send(Command::SetSettings(settings)).unwrap();
            sent = true;
        }
    });
    assert!(sent);
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    main_is_the_sum(&wide, &run, back);
}

/// A transition set up while the head is read along the map is refused.
#[test]
fn a_transition_while_read_along_a_map_is_refused() {
    let dir = temp_dir("along-transition");
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let next = file(&dir, "next.wav", 2, 1.0, click);
    let run = Run::new();
    let mut sent = false;
    let (wide, seen) = run.play_with(vec![path, next], &mut |player, e| {
        if !sent && past(e, 2.0) {
            let plan = TransitionPlan {
                from: 0,
                into: 1,
                start: 5.0,
                length: 1.0,
                entry: 0.1,
                beats: beats(5.0, 0.25, 0.1, 0.25, 4),
                lanes: lanes(Envelope::cut(1.0, 0.0, 0.0), Envelope::cut(0.0, 1.0, 0.0)),
                tempo: TempoMode::Varispeed,
                settle: 1.0,
                glide: false,
            };
            player.transition(plan).unwrap();
            sent = true;
        }
    });
    let refused = seen.iter().any(|e| {
        matches!(e, Event::Transition(TransitionState::Cancelled { reason })
            if reason.contains("read along a time map"))
    });
    assert!(refused, "the transition is refused");
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::Transition(TransitionState::Started { .. }))));
    clicks_on_the_map(&wide, 0.001);
}

/// A switch back onto the file that does not come about (no route with
/// the settings then) is asked again a second later, and comes about once
/// it can.
#[test]
fn a_switch_back_that_fails_is_asked_again() {
    let dir = temp_dir("along-retry");
    let path = file(&dir, "chord.wav", 2, 8.0, chord);
    let samples = read_all(&path);
    let run = Run::new();
    let (good, bad) = (
        run.options().settings,
        Settings {
            max_rate: Some(22_050),
            resample: false,
            ..Settings::default()
        },
    );
    let (mut sent, mut noticed, mut heard) = (false, None, 0);
    let (wide, seen) = run.play_with(vec![path], &mut |player, e| {
        if let Event::Position(p) = e {
            heard = p.output_frame;
        }
        if !sent && past(e, 2.0) {
            player.send(Command::SetSettings(bad)).unwrap();
            sent = true;
        }
        if noticed.is_none() && matches!(e, Event::Notice { .. }) {
            noticed = Some(heard);
            player.send(Command::SetSettings(good)).unwrap();
        }
    });
    let noticed = noticed.expect("a notice that the switch back has no route");
    let (back, frame) = back_at(&seen);
    assert!(
        back > noticed + u64::from(RATE) / 2,
        "back at {back}, noticed at {noticed}"
    );
    let from = (back + run.fade()) as usize;
    let tail = &wide.main[from * 2..];
    let file = &samples[(frame as usize + from - back as usize) * 2..];
    let n = tail.len().min(file.len());
    assert!(
        n > RATE as usize && tail[..n] == file[..n],
        "back on the file"
    );
}

/// A separator that separates nothing: the stems never come.
static NEVER: Gated = Gated::new();

fn never() -> Option<&'static dyn Separator> {
    Some(&NEVER)
}

/// On a device, stems that are not separated in time give way to the whole
/// mix: the head plays on along the map with no gap, its main the parts'
/// sum wherever they are said to be there (from a moment after the move,
/// a device's half second of them made ahead).
#[test]
fn on_a_device_stems_that_fall_behind_give_way_to_the_whole_mix() {
    let dir = temp_dir("along-device");
    let path = file(&dir, "clicks.wav", 2, 7.0, click);
    let run = Run {
        separator: Some(never),
        device: true,
        ..Run::new()
    };
    let (wide, seen) = run.play(vec![path]);
    NEVER.allow(usize::MAX);
    let fell_back = seen
        .iter()
        .any(|e| matches!(e, Event::PartsStopped { reason, .. } if reason.contains("whole mix")));
    assert!(fell_back, "the stems gave way to the whole mix");
    clicks_on_the_map(&wide, 0.001);
    let (back, _) = back_at(&seen);
    let warped = (run.from() + run.fade()) as usize..back as usize;
    let (worst, there) = wide.worst(warped.clone());
    assert!(
        there > warped.len() * 9 / 10,
        "{there} of {} frames",
        warped.len()
    );
    assert!(worst <= 2, "{worst} steps off the parts' sum");
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
    let (wide, seen) = run.play(vec![path]);
    let notice = seen.iter().find_map(|e| match e {
        Event::Notice { message, .. } => Some(message.clone()),
        _ => None,
    });
    let notice = notice.expect("a notice");
    assert!(notice.contains("6 channels"), "{notice}");
    assert!(wide.main == samples, "it plays on as it was");
}
