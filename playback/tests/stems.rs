//! Stem playback: a track separated as it plays and the stems that are on
//! summed, a mix changed while it plays, back to the file as it is, seeks,
//! track changes and a separator that fails. The separator here gives each
//! stem a quarter of the mix, so the sum of all four is the mix.

mod common;

use std::path::{Path, PathBuf};
use std::time::Duration;

use common::*;
use jodel_output::{SampleFormat, SimClock};
use jodel_playback::stems::{Separator, Stem, StemMix};
use jodel_playback::{
    Event, FilterDesign, JumpAt, MemoryTarget, PlayState, Player, PlayerOptions, Settings, Stage,
    TrackStart,
};
use SampleFormat::*;

const RATE: u32 = 44_100;
/// Eight seconds: two of the separator's chunks and a bit.
const FRAMES: usize = 8 * RATE as usize;

/// Each stem a quarter of the mix.
struct Quarter;

impl Separator for Quarter {
    fn name(&self) -> String {
        "quarter".into()
    }

    fn masks(&self, frames: usize, stft: &[f32]) -> Result<Vec<f32>, String> {
        let per_stem = stft.len();
        assert_eq!(per_stem % frames, 0);
        // Real parts a quarter, imaginary parts nothing.
        Ok((0..4 * per_stem)
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.0 })
            .collect())
    }
}

/// A separator that always fails.
struct Broken;

impl Separator for Broken {
    fn name(&self) -> String {
        "broken".into()
    }

    fn masks(&self, _: usize, _: &[f32]) -> Result<Vec<f32>, String> {
        Err("out of memory".into())
    }
}

static QUARTER: Quarter = Quarter;
static BROKEN: Broken = Broken;

fn quarter() -> Option<&'static dyn Separator> {
    Some(&QUARTER)
}

fn broken() -> Option<&'static dyn Separator> {
    Some(&BROKEN)
}

fn mix(stems: &[Stem]) -> StemMix {
    stems.iter().fold(StemMix::NONE, |m, &s| m.toggle(s))
}

fn settings(stems: StemMix) -> Settings {
    Settings {
        resample: false,
        stems,
        ..Settings::default()
    }
}

fn options(stems: StemMix, separator: jodel_playback::SeparatorSource) -> PlayerOptions {
    PlayerOptions {
        settings: settings(stems),
        position_interval: Duration::from_millis(1),
        separator: Some(separator),
        ..PlayerOptions::default()
    }
}

/// A 24-bit stereo WAV of two tones, `FRAMES` long.
fn tone_file(dir: &Path, name: &str) -> (PathBuf, Vec<i32>) {
    let path = dir.join(name);
    let samples = tones(RATE, FRAMES);
    write_wav(&path, RATE, 2, &samples);
    (path, samples)
}

/// The 24-bit samples of a stream the device took as `S24In32Low`.
fn samples(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| (i32::from_le_bytes([b[0], b[1], b[2], b[3]]) << 8) >> 8)
        .collect()
}

fn starts(events: &[Event]) -> Vec<&TrackStart> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(&**s),
            _ => None,
        })
        .collect()
}

/// The largest difference between `got` and `scale` times `want`, frame
/// `from` (inclusive) to `to`. The first frame of a separation is left out:
/// the chunk fades in from it.
fn worst(got: &[i32], want: &[i32], scale: f64, from: usize, to: usize) -> f64 {
    (2 * from..2 * to)
        .map(|i| (f64::from(got[i]) - scale * f64::from(want[i])).abs())
        .fold(0.0, f64::max)
}

/// Within this many 24-bit steps: dither and the transforms' rounding.
const CLOSE: f64 = 64.0;

#[test]
fn the_stems_that_are_on_are_summed_and_none_is_silence() {
    let dir = temp_dir("stems-sum");
    let (path, file) = tone_file(&dir, "tones.wav");
    for (stems, scale) in [
        (mix(&[Stem::Drums]), 0.25),
        (mix(&[Stem::Drums, Stem::Bass, Stem::Other]), 0.75),
        // A stem at a level: the voice at 40 % under the rest.
        (StemMix::ALL.with_level(Stem::Vocals, 40), 0.85),
        (StemMix::NONE, 0.0),
    ] {
        let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
        let captures = target.captures();
        let (player, events) = Player::new(Box::new(target), options(stems, quarter));
        player.load(vec![path.clone()]).unwrap();
        player.play(0).unwrap();
        let seen = until_queue_end(&events);
        let start = starts(&seen)[0];
        assert_eq!(start.route.stage, Stage::Stems, "{stems:?}");
        assert!(!start.bit_perfect);
        assert!(!seen
            .iter()
            .any(|e| matches!(e, Event::Error { .. } | Event::Notice { .. })));
        drop(player);
        let streams = captured(&captures);
        assert_eq!(streams.len(), 1);
        let got = samples(&streams[0].1);
        assert_eq!(got.len(), file.len(), "{stems:?}: the whole track");
        let off = worst(&got, &file, scale, 1, FRAMES);
        assert!(off <= CLOSE, "{stems:?}: {off} steps off");
    }
}

/// A new mix of the same route is heard after what was already made, with
/// no restart; all stems on again plays the file as it is from the frame
/// heard.
#[test]
fn a_mix_changes_in_place_and_all_stems_play_the_file_again() {
    let dir = temp_dir("stems-change");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let drums = mix(&[Stem::Drums]);
    let (player, events) = Player::new(Box::new(target), options(drums, quarter));
    player.load(vec![path]).unwrap();
    gate.allow(RATE.into());
    player.play(0).unwrap();
    gate.wait_written(RATE.into());
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(starts(&first)[0].route.stage, Stage::Stems);

    // Drums and bass: half the mix, once what was made is written.
    let changed = gate.written() as usize;
    player
        .set_settings(settings(mix(&[Stem::Drums, Stem::Bass])))
        .unwrap();
    // A pause and resume after it: once the engine answers, it has the mix.
    player.pause().unwrap();
    player.resume().unwrap();
    collect_until(&events, |e| matches!(e, Event::State(PlayState::Playing)));
    gate.allow(3 * u64::from(RATE));
    gate.wait_written(3 * u64::from(RATE));

    // Every stem: the file as it is, from the frame heard.
    let whole = gate.written() as usize;
    player.set_settings(settings(StemMix::ALL)).unwrap();
    let seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let start = starts(&seen)[0];
    assert_eq!(start.route.stage, Stage::Passthrough);
    assert!(start.route.lossless);
    // The file crossfades in at the frame written up to.
    assert_eq!(start.output_frame as usize, whole);
    assert_eq!(start.from_frame as usize, whole);
    gate.allow_all();
    let rest = until_queue_end(&events);
    let all = [first, seen, rest].concat();
    assert_eq!(starts(&all).len(), 2, "the mix change restarts nothing");
    assert!(!all.iter().any(|e| matches!(e, Event::Error { .. })));
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "the same output throughout");
    let got = samples(&streams[0].1);
    assert!(worst(&got, &file, 0.25, 1, changed) <= CLOSE);
    let settled = changed + 2 * 4096;
    assert!(worst(&got, &file, 0.5, settled, whole) <= CLOSE);
    assert_eq!(got.len() / 2, FRAMES);
    let faded = whole + FADE;
    assert!(
        got[2 * faded..] == file[2 * faded..],
        "bit-perfect from {faded}"
    );
}

/// The default crossfade of a switch, in frames.
const FADE: usize = RATE as usize / 50;

/// Turning a stem off moves the track from the file to the stems without
/// a break, in the same output: the file exactly up to where they meet,
/// then a crossfade into the stems.
#[test]
fn the_stems_crossfade_in_without_a_break() {
    let dir = temp_dir("stems-enter");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(StemMix::ALL, quarter));
    player.load(vec![path]).unwrap();
    gate.allow(2 * u64::from(RATE));
    player.play(0).unwrap();
    gate.wait_written(2 * u64::from(RATE));
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(starts(&first)[0].route.stage, Stage::Passthrough);
    player.set_settings(settings(mix(&[Stem::Drums]))).unwrap();
    let seen = trickle_until_started(&gate, &events);
    let start = starts(&seen)[0];
    assert_eq!(start.route.stage, Stage::Stems);
    let at = start.output_frame as usize;
    assert_eq!(start.from_frame as usize, at);
    assert!(at >= 2 * RATE as usize);
    gate.allow_all();
    let rest = until_queue_end(&events);
    assert!(!rest.iter().any(|e| matches!(e, Event::Error { .. })));
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "the same output throughout");
    let got = samples(&streams[0].1);
    assert_eq!(got.len(), file.len());
    assert!(got[..2 * at] == file[..2 * at], "the file up to {at}");
    assert!(worst(&got, &file, 0.25, at + FADE, FRAMES) <= CLOSE);
}

#[test]
fn a_seek_separates_from_the_new_position() {
    let dir = temp_dir("stems-seek");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(mix(&[Stem::Vocals]), quarter));
    player.load(vec![path]).unwrap();
    gate.allow(RATE.into());
    player.play(0).unwrap();
    gate.wait_written(RATE.into());
    collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    let at = gate.written() as usize;
    player.seek(5.0).unwrap();
    collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    let got = samples(&captured(&captures)[0].1);
    let from = 5 * RATE as usize;
    assert_eq!(got.len() / 2, at + FRAMES - from);
    let after: Vec<i32> = got[2 * at..].to_vec();
    assert!(worst(&after, &file[2 * from..], 0.25, 1, FRAMES - from) <= CLOSE);
}

#[test]
fn the_next_track_plays_its_stems_gaplessly() {
    let dir = temp_dir("stems-next");
    let (one, first) = tone_file(&dir, "one.wav");
    let (two, second) = tone_file(&dir, "two.wav");
    let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(mix(&[Stem::Other]), quarter));
    player.load(vec![one, two]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    let routes: Vec<Stage> = starts(&seen).iter().map(|s| s.route.stage).collect();
    assert_eq!(routes, [Stage::Stems, Stage::Stems]);
    drop(player);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one stream: gapless");
    let got = samples(&streams[0].1);
    assert_eq!(got.len(), first.len() + second.len());
    let both = [first, second].concat();
    assert!(worst(&got, &both, 0.25, 1, FRAMES) <= CLOSE);
    assert!(worst(&got, &both, 0.25, FRAMES + 1, 2 * FRAMES) <= CLOSE);
}

#[test]
fn failing_stems_fall_back_to_the_whole_mix() {
    let dir = temp_dir("stems-broken");
    let (path, file) = tone_file(&dir, "tones.wav");
    let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
    let captures = target.captures();
    let (player, events) = Player::new(Box::new(target), options(mix(&[Stem::Bass]), broken));
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    let stages: Vec<Stage> = starts(&seen).iter().map(|s| s.route.stage).collect();
    assert_eq!(stages.last(), Some(&Stage::Passthrough), "{seen:#?}");
    let stopped = seen.iter().find_map(|e| match e {
        Event::StemsStopped { index: 0, reason } => Some(reason.clone()),
        _ => None,
    });
    assert_eq!(stopped.as_deref(), Some("out of memory"), "{seen:#?}");
    assert!(!seen.iter().any(|e| matches!(
        e,
        Event::TrackEnded {
            finished: false,
            ..
        }
    )));
    drop(player);
    let streams = captured(&captures);
    let got = samples(&streams.last().unwrap().1);
    assert!(got == file, "the file as it is");
}

/// A high output rate: the stems at the separator's rate resampled to it.
const HIGH: u32 = 192_000;

/// Lengths of a track at [`HIGH`] (4 s and a little less) whose stems from
/// its start, resampled back, once came out two frames short and two over;
/// from a whole second on as well.
const AWKWARD: [usize; 2] = [4 * HIGH as usize - 512, 4 * HIGH as usize - 320];

/// Options for stems at [`HIGH`] with the quarter separator: `stems` on.
fn high_options(stems: StemMix) -> PlayerOptions {
    PlayerOptions {
        settings: Settings {
            stems,
            ..Settings::default()
        },
        position_interval: Duration::from_millis(1),
        separator: Some(quarter),
        ..PlayerOptions::default()
    }
}

/// Turning a stem off at a high output rate: the stems run up from a
/// second before the frame written, more output frames than the switch
/// keeps of a route that has not caught up, and nothing of them is there
/// until the first chunk is separated. The switch waits for them (it once
/// dropped output it did not have yet, and the engine stopped), then
/// crossfades in, in the same output, and the track ends on its last
/// frame (at awkward lengths too: the run-up counts in the stems' end).
#[test]
fn stems_at_a_high_rate_are_waited_for() {
    let dir = temp_dir("stems-high");
    for frames in [4 * HIGH as usize, AWKWARD[0], AWKWARD[1]] {
        let path = dir.join(format!("tones-{frames}.wav"));
        let file = tones(HIGH, frames);
        write_wav(&path, HIGH, 2, &file);
        let (target, captures, gate) = GatedTarget::new(caps(&[HIGH], &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), high_options(StemMix::ALL));
        player.load(vec![path]).unwrap();
        gate.allow(2 * u64::from(HIGH));
        player.play(0).unwrap();
        gate.wait_written(2 * u64::from(HIGH));
        collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
        player
            .set_settings(Settings {
                stems: mix(&[Stem::Drums]),
                ..Settings::default()
            })
            .unwrap();
        let seen = trickle_until_started(&gate, &events);
        let start = starts(&seen)[0];
        assert_eq!(start.route.stage, Stage::Stems);
        assert_eq!(start.output.format.rate, HIGH, "the same output");
        let at = start.output_frame as usize;
        gate.allow_all();
        let rest = until_queue_end(&events);
        assert!(!rest.iter().any(|e| matches!(e, Event::Error { .. })));
        drop(player);
        let streams = captured(&captures);
        assert_eq!(streams.len(), 1);
        let got = samples(&streams[0].1);
        assert_eq!(got.len(), file.len(), "{frames} frames");
        assert!(got[..2 * at] == file[..2 * at], "the file up to {at}");
        // A quarter of the tones after the crossfade (through 44.1 kHz and
        // back: the tones are far below its band edge).
        let fade = HIGH as usize / 50;
        let near = worst(&got, &file, 0.25, at + fade, frames - HIGH as usize / 10);
        assert!(near <= 4_096.0, "off by {near}");
    }
}

/// A track at a high rate on its stems ends on its own last frame: its
/// stems at the separator's rate end within a frame of that rate of it,
/// and resampled back they once ended a frame or two short of the file or
/// past it, by its length (or the frame the stems started at).
#[test]
fn stems_at_a_high_rate_end_with_the_track() {
    let dir = temp_dir("stems-high-end");
    for frames in AWKWARD {
        let path = dir.join(format!("tones-{frames}.wav"));
        let file = tones(HIGH, frames);
        write_wav(&path, HIGH, 2, &file);
        let target = MemoryTarget::new(caps(&[HIGH], &[S24In32Low]), SimClock::Instant);
        let captures = target.captures();
        let (player, events) = Player::new(Box::new(target), high_options(mix(&[Stem::Drums])));
        player.load(vec![path]).unwrap();
        player.play(0).unwrap();
        let seen = until_queue_end(&events);
        let start = starts(&seen)[0];
        assert_eq!(start.route.stage, Stage::Stems);
        assert_eq!(start.output.format.rate, HIGH);
        drop(player);
        let got = samples(&captured(&captures)[0].1);
        assert_eq!(got.len(), file.len(), "{frames} frames");
        let fade = HIGH as usize / 50;
        let near = worst(&got, &file, 0.25, fade, frames - HIGH as usize / 10);
        assert!(near <= 4_096.0, "off by {near}");
    }
}

/// A jump on a stems route at a high rate, at a time: the frames up to it,
/// then the track from the target to its last frame, nothing more or
/// less. The stream starts again a little before the target and drops
/// what it makes for that; a target nearer the end than the resampler's
/// filter reaches ends the stems before all of that is made.
#[test]
fn a_jump_at_a_high_rate_ends_with_the_track() {
    let dir = temp_dir("stems-high-jump");
    let reach = FilterDesign::new(RATE, HIGH).half_width() * HIGH as usize / RATE as usize;
    for frames in AWKWARD {
        let path = dir.join(format!("tones-{frames}.wav"));
        write_wav(&path, HIGH, 2, &tones(HIGH, frames));
        for to in [HIGH as usize, frames - reach / 2] {
            let (target, captures, gate) = GatedTarget::new(caps(&[HIGH], &[S24In32Low]));
            let (player, events) = Player::new(Box::new(target), high_options(mix(&[Stem::Drums])));
            player.jump_window(4.0, 4.0).unwrap();
            player.load(vec![path.clone()]).unwrap();
            gate.allow(u64::from(HIGH));
            player.play(0).unwrap();
            gate.wait_written(u64::from(HIGH));
            let mut seen = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
            let seconds = to as f64 / f64::from(HIGH);
            player.jump(0, seconds, JumpAt::At(2.0)).unwrap();
            gate.allow_all();
            seen.extend(until_queue_end(&events));
            drop(player);
            assert!(
                seen.iter().any(|e| matches!(e, Event::Jumped { .. })),
                "{seen:#?}"
            );
            assert!(!seen.iter().any(|e| matches!(e, Event::Error { .. })));
            let got = samples(&captured(&captures)[0].1);
            let want = 2 * HIGH as usize + frames - to;
            assert_eq!(got.len() / 2, want, "{frames} frames, to {to}");
        }
    }
}

/// Frames a stem takes to fade in or out when the mix changes (10 ms).
const STEM_RAMP: usize = RATE as usize / 100;

/// A change of the stems at a time of the track lands on that frame: the
/// stems before it up to there, the new ones from there (after their
/// 10 ms fade).
#[test]
fn a_mix_change_at_a_time_lands_on_its_frame() {
    let dir = temp_dir("stems-at");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(mix(&[Stem::Drums]), quarter));
    player.load(vec![path]).unwrap();
    gate.allow(RATE.into());
    player.play(0).unwrap();
    gate.wait_written(RATE.into());
    player
        .set_stems_at(mix(&[Stem::Drums, Stem::Bass]), 4.0)
        .unwrap();
    gate.allow_all();
    let seen = until_queue_end(&events);
    assert_eq!(starts(&seen).len(), 1, "nothing restarts");
    drop(player);
    let got = samples(&captured(&captures)[0].1);
    let at = 4 * RATE as usize;
    assert!(
        worst(&got, &file, 0.25, 1, at) <= CLOSE,
        "drums up to the time"
    );
    assert!(
        worst(&got, &file, 0.5, at + STEM_RAMP, FRAMES) <= CLOSE,
        "drums and bass from it"
    );
}

/// All stems on again at a time: the stems play on to it, all of them
/// from it, and then the track goes back to the file as it is.
#[test]
fn all_stems_at_a_time_come_back_there_then_the_file_plays() {
    let dir = temp_dir("stems-at-all");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(mix(&[Stem::Drums]), quarter));
    player.load(vec![path]).unwrap();
    gate.allow(RATE.into());
    player.play(0).unwrap();
    gate.wait_written(RATE.into());
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(starts(&first)[0].route.stage, Stage::Stems);
    player.set_stems_at(StemMix::ALL, 3.0).unwrap();
    // Up to a little before the time, then a little at a time: the file's
    // route meets the stems a little after what was written.
    let near = 3 * u64::from(RATE) - u64::from(RATE) / 10;
    gate.allow(near);
    gate.wait_written(near);
    let seen = trickle_until_started(&gate, &events);
    gate.allow_all();
    until_queue_end(&events);
    let legs = starts(&seen);
    let back = legs[0];
    assert_eq!(back.route.stage, Stage::Passthrough);
    let at = 3 * RATE as usize;
    let met = back.output_frame as usize;
    assert!(met >= at, "the file once the time is made ({met})");
    drop(player);
    let got = samples(&captured(&captures)[0].1);
    assert!(worst(&got, &file, 0.25, 1, at) <= CLOSE);
    assert!(worst(&got, &file, 1.0, at + STEM_RAMP, met) <= CLOSE);
    assert!(got[2 * (met + FADE)..] == file[2 * (met + FADE)..]);
}

/// The first stem off at a time: the track moves onto its stems before it
/// (all of them, so it sounds as the file), and the stem goes at the time.
#[test]
fn the_first_stem_off_at_a_time_goes_there() {
    let dir = temp_dir("stems-at-first");
    let (path, file) = tone_file(&dir, "tones.wav");
    let (target, captures, gate) = GatedTarget::new(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(StemMix::ALL, quarter));
    player.load(vec![path]).unwrap();
    gate.allow(RATE.into());
    player.play(0).unwrap();
    gate.wait_written(RATE.into());
    let first = collect_until(&events, |e| matches!(e, Event::TrackStarted(_)));
    assert_eq!(starts(&first)[0].route.stage, Stage::Passthrough);
    let drums_off = StemMix::ALL.toggle(Stem::Drums);
    player.set_stems_at(drums_off, 5.0).unwrap();
    let seen = trickle_until_started(&gate, &events);
    let onto = starts(&seen)[0];
    assert_eq!(onto.route.stage, Stage::Stems);
    let met = onto.output_frame as usize;
    let at = 5 * RATE as usize;
    assert!(met < at, "on the stems before the time ({met})");
    gate.allow_all();
    until_queue_end(&events);
    drop(player);
    let got = samples(&captured(&captures)[0].1);
    assert!(got[..2 * met] == file[..2 * met]);
    assert!(worst(&got, &file, 1.0, met + FADE, at) <= CLOSE);
    assert!(worst(&got, &file, 0.75, at + STEM_RAMP, FRAMES) <= CLOSE);
}

/// Stems kept for one file: its whole mix as the voice, nothing else.
struct Kept {
    track: PathBuf,
    stems: PathBuf,
}

impl jodel_playback::StemStore for Kept {
    fn open(&self, path: &Path) -> Option<jodel_playback::StemFile> {
        (path == self.track)
            .then(|| jodel_playback::StemFile::open(&self.stems).ok())
            .flatten()
    }
}

/// Stems that sum over full scale (three of them each the whole track, at
/// half scale): what the device gets stays under the stems' ceiling, nothing
/// clipped, and held near it (the limiter takes off what is over, not
/// more).
#[test]
fn stems_summed_over_full_scale_are_kept_under_it() {
    let dir = temp_dir("stems-over");
    let (path, file) = tone_file(&dir, "tones.wav");
    let kept = dir.join("tones.stems");
    let mut w = jodel_playback::StemWriter::create(&kept, RATE, "test").unwrap();
    let full = f32::powi(2.0, 23);
    let left: Vec<f32> = file.iter().step_by(2).map(|&s| s as f32 / full).collect();
    let right: Vec<f32> = file
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&s| s as f32 / full)
        .collect();
    let drums = Stem::ALL.iter().position(|&s| s == Stem::Drums).unwrap();
    let silent = vec![0.0f32; left.len()];
    let parts: [[&[f32]; 2]; 4] = std::array::from_fn(|s| {
        if s == drums {
            [&silent[..], &silent[..]]
        } else {
            [&left[..], &right[..]]
        }
    });
    w.push(parts).unwrap();
    w.finish().unwrap();
    let store: std::sync::Arc<dyn jodel_playback::StemStore> = std::sync::Arc::new(Kept {
        track: path.clone(),
        stems: kept,
    });
    let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
    let captures = target.captures();
    let options = PlayerOptions {
        stem_store: Some(store),
        ..options(StemMix::ALL.toggle(Stem::Drums), broken)
    };
    let (player, events) = Player::new(Box::new(target), options);
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    assert_eq!(starts(&seen)[0].route.stage, Stage::Stems);
    drop(player);
    let got = samples(&captured(&captures)[0].1);
    assert_eq!(got.len(), file.len());
    let ceiling = jodel_playback::stems::CEILING * f64::from(1 << 23);
    let peak = got.iter().map(|&s| f64::from(s.abs())).fold(0.0, f64::max);
    assert!(peak <= ceiling + 2.0, "{peak} over {ceiling}");
    assert!(peak > 0.95 * ceiling, "held at {peak}, under {ceiling}");
    assert!(!got.iter().any(|&s| s.abs() >= (1 << 23) - 1), "clipped");
}

/// A track whose stems are kept plays them: nothing is separated (the
/// separator here only fails), the stems that are on are the ones heard,
/// also from a seek.
#[test]
fn kept_stems_play_without_a_separation() {
    let dir = temp_dir("stems-kept");
    let (path, file) = tone_file(&dir, "tones.wav");
    let kept = dir.join("tones.stems");
    let mut w = jodel_playback::StemWriter::create(&kept, RATE, "test").unwrap();
    let full = f32::powi(2.0, 23);
    let left: Vec<f32> = file.iter().step_by(2).map(|&s| s as f32 / full).collect();
    let right: Vec<f32> = file
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&s| s as f32 / full)
        .collect();
    let silent = vec![0.0f32; left.len()];
    let voice = Stem::ALL.iter().position(|&s| s == Stem::Vocals).unwrap();
    let parts: [[&[f32]; 2]; 4] = std::array::from_fn(|s| {
        if s == voice {
            [&left[..], &right[..]]
        } else {
            [&silent[..], &silent[..]]
        }
    });
    w.push(parts).unwrap();
    w.finish().unwrap();
    let store: std::sync::Arc<dyn jodel_playback::StemStore> = std::sync::Arc::new(Kept {
        track: path.clone(),
        stems: kept,
    });
    for (stems, scale) in [
        (mix(&[Stem::Vocals]), 1.0),
        (mix(&[Stem::Drums, Stem::Bass]), 0.0),
    ] {
        let target = MemoryTarget::new(caps(&[RATE], &[S24In32Low]), SimClock::Instant);
        let captures = target.captures();
        let options = PlayerOptions {
            stem_store: Some(store.clone()),
            ..options(stems, broken)
        };
        let (player, events) = Player::new(Box::new(target), options);
        player.load(vec![path.clone()]).unwrap();
        player.play(0).unwrap();
        let seen = until_queue_end(&events);
        assert_eq!(starts(&seen)[0].route.stage, Stage::Stems, "{stems:?}");
        assert!(
            !seen.iter().any(|e| matches!(
                e,
                Event::Error { .. } | Event::Notice { .. } | Event::StemsStopped { .. }
            )),
            "{stems:?}: nothing separated, nothing failed"
        );
        drop(player);
        let streams = captured(&captures);
        let got = samples(&streams[0].1);
        assert_eq!(got.len(), file.len(), "{stems:?}: the whole track");
        // Within the 16 bits the stems are kept in.
        let off = worst(&got, &file, scale, 0, FRAMES);
        assert!(off <= 600.0, "{stems:?}: {off} steps off");
    }
}
