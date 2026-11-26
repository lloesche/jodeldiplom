//! Stem levels on the desk: the engine hands each stem on beside the file
//! as a part of a mixer's lane input ([`MakeStems`]), each on its strip
//! inside the player's, and the mixer plays the stems at their strips'
//! levels in place of the file while one is touched. A level is heard
//! within the device's buffer; untouched, the file is heard bit for bit
//! again; seeks, gapless changes, a failing separator and stems kept in a
//! store. The separator here gives each stem a quarter of the mix, so all
//! four add up to it.

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use common::lanes::{broken, device_samples, options, quarter, tone_file, CLOSE, RATE};
use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role, Strip};
use jodel_output::{Capabilities, SampleFormat, SimClock};
use jodel_playback::stems::{lane, PerStem, Stem};
use jodel_playback::{
    Captures, Event, MakeStems, MemoryTarget, Player, PlayerOptions, StemDesk, StemFile, StemStore,
    StemWriter,
};
use jodel_testkit::TempDir;
use SampleFormat::*;

/// The part strip of `stem` among the desk's (in
/// [`jodel_playback::stems::LANES`]' order).
fn part(lanes: &MixerLanes, stem: Stem) -> &Strip {
    &lanes.parts[lane(stem)]
}

/// The largest step between `n` frames of `got` from its frame `at` and
/// `scale` times as many of `want` from its frame `from` (stereo).
fn off(got: &[i32], at: usize, want: &[i32], from: usize, n: usize, scale: f64) -> f64 {
    (0..2 * n)
        .map(|i| (f64::from(got[2 * at + i]) - scale * f64::from(want[2 * from + i])).abs())
        .fold(0.0, f64::max)
}

/// The tones of `frames` at [`RATE`] in `dir`, and their samples.
fn tones_of(dir: &Path, name: &str, frames: usize) -> (PathBuf, Vec<i32>) {
    let path = tone_file(dir, name, RATE, frames);
    (path, tones(RATE, frames))
}

/// The one stream the device played, as 24-bit samples.
fn played(captures: &Captures) -> Vec<i32> {
    let streams = captured(captures);
    assert_eq!(streams.len(), 1, "one stream");
    device_samples(&streams[0].1)
}

/// A desk whose device plays as far as a test lets it ([`held_desk`]).
struct Held {
    player: Player,
    events: Receiver<Event>,
    captures: Captures,
    /// What the mixer's device takes, and what the engine has written.
    device: Arc<Gate>,
    engine: Arc<Gate>,
    lanes: MixerLanes,
}

/// The engine on a mixer's lane input as the app has it (the player's
/// strip a group, a strip for each stem inside it) with the quarter's
/// stems made and waited for; the mixer's device gated, so the engine runs
/// ahead of what is played by the mixer's input ring.
fn held_desk(caps: Capabilities) -> Held {
    let device = MemoryTarget::new(caps, SimClock::Instant);
    let captures = device.captures();
    let (device, gate) = GatedTarget::around(Box::new(device));
    let mixer = Mixer::open(Box::new(device), MixerOptions::default());
    let (desk, head) = StemDesk::new(&mixer, "Player", Role::Lead, 1.0);
    let (input, engine) = GatedTarget::around(head.target);
    engine.allow_all();
    let (player, events) = Player::new(Box::new(input), options(true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    let lanes = MixerLanes {
        parts: desk.stems,
        faders: head.faders.lanes,
        mixer,
    };
    Held {
        player,
        events,
        captures,
        device: gate,
        engine,
        lanes,
    }
}

impl Held {
    /// Lets the mixer's device play `frames` in all, and waits for the
    /// engine to be a quarter of a second ahead of it, on the mixer's
    /// input ring. Returns the frames played.
    fn play_to(&self, frames: u64) -> usize {
        self.device.allow(frames);
        self.device.wait_written(frames);
        let ahead = frames + u64::from(RATE) / 4;
        assert!(
            self.engine.written_within(ahead, TIMEOUT),
            "the engine wrote {} frames, the device played {frames}",
            self.engine.written()
        );
        self.device.written() as usize
    }
}

/// A stem's level changed on its strip is heard from the next frames the
/// mixer mixes, a few milliseconds after what the device has played (the
/// engine a quarter of a second ahead of it at least, on the mixer's input
/// ring); every strip back as it was, the file again bit for bit.
#[test]
fn a_stem_level_on_the_desk_is_heard_at_once_and_untouched_is_the_file_again() {
    let dir = TempDir::new("stems-desk");
    let frames = 3 * RATE as usize;
    let (path, file) = tones_of(&dir, "tones.wav", frames);
    let desk = held_desk(caps(&[RATE], &[S24In32Low]));
    part(&desk.lanes, Stem::Drums).set_mute(true);
    desk.player.load(vec![path]).unwrap();
    desk.player.play(0).unwrap();
    let rate = u64::from(RATE);
    // The voice down to 40 % a second in: drums off, bass and other, 40 %
    // of the voice (0.6 of the file).
    let lowered = desk.play_to(rate);
    part(&desk.lanes, Stem::Vocals).set_level(0.4);
    // Every strip as it was at two seconds.
    let reset = desk.play_to(2 * rate);
    part(&desk.lanes, Stem::Vocals).set_level(1.0);
    part(&desk.lanes, Stem::Drums).set_mute(false);
    desk.device.allow_all();
    let seen = until_queue_end(&desk.events);
    let Held {
        player,
        captures,
        lanes,
        ..
    } = desk;
    drop(player);
    drop(lanes);
    assert!(!seen
        .iter()
        .any(|e| matches!(e, Event::Error { .. } | Event::PartsStopped { .. })));
    let got = played(&captures);
    assert_eq!(got.len(), file.len());
    // The mixer's period and the strip's ramp, and a little more: 50 ms.
    let soon = RATE as usize / 20;
    let drums_off = off(&got, 1_000, &file, 1_000, lowered - 1_000, 0.75);
    assert!(drums_off <= CLOSE as f64, "{drums_off} steps off 0.75");
    let at = lowered + soon;
    let voice_down = off(&got, at, &file, at, reset - at, 0.6);
    assert!(voice_down <= CLOSE as f64, "{voice_down} steps off 0.6");
    // Main again once the ramps and the mixer's switch are over.
    let back = 2 * (reset + soon + RATE as usize / 50);
    assert!(got[back..] == file[back..], "the file again from {back}");
}

/// A seek with a stem muted: the stems start again from the seek's frame,
/// heard there (at a high output rate too, the stems resampled to it as
/// main is).
#[test]
fn a_seek_with_a_stem_muted_plays_its_stems_from_there() {
    let dir = TempDir::new("stems-seek");
    let frames = 4 * RATE as usize;
    let (path, file) = tones_of(&dir, "tones.wav", frames);
    for out in [RATE, 192_000] {
        let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[out], &[S24In32Low]));
        let (player, events) = Player::new(Box::new(target), options(true, quarter));
        player.make_stems(MakeStems::Separate).unwrap();
        part(&lanes, Stem::Drums).set_mute(true);
        player.load(vec![path.clone()]).unwrap();
        gate.allow(u64::from(out));
        player.play(0).unwrap();
        gate.wait_written(u64::from(out));
        player.seek(2.5).unwrap();
        collect_until(&events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(&events);
        drop(player);
        drop(lanes);
        let got = played(&captures);
        let from = 5 * RATE as usize / 2;
        let (before, after) = if out == RATE {
            (file[..2 * from].to_vec(), file[2 * from..].to_vec())
        } else {
            (
                reference_resample(RATE, out, 2, &file),
                reference_resample(RATE, out, 2, &file[2 * from..]),
            )
        };
        // What was mixed before the seek, then the track from it.
        let seam = got.len() / 2 - after.len() / 2;
        let edge = out as usize / 40;
        let early = off(&got, edge, &before, edge, seam - 2 * edge, 0.75);
        let late = off(
            &got,
            seam + edge,
            &after,
            edge,
            after.len() / 2 - 2 * edge,
            0.75,
        );
        println!("{out} Hz: {early} steps off before the seek, {late} after");
        assert!(early.max(late) <= CLOSE as f64, "{out}: {early}, {late}");
    }
}

/// The next track follows gaplessly with its stems: one stem soloed, its
/// quarter of each track throughout.
#[test]
fn the_next_track_plays_its_stems_gaplessly() {
    let dir = TempDir::new("stems-next");
    let frames = 2 * RATE as usize;
    let (one, first) = tones_of(&dir, "one.wav", frames);
    let (two, second) = tones_of(&dir, "two.wav", frames);
    let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(true, quarter));
    player.make_stems(MakeStems::Separate).unwrap();
    part(&lanes, Stem::Other).set_solo(true);
    gate.allow_all();
    player.load(vec![one, two]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(lanes);
    let starts = seen
        .iter()
        .filter(|e| matches!(e, Event::TrackStarted(_)))
        .count();
    assert_eq!(starts, 2);
    let got = played(&captures);
    assert_eq!(got.len(), first.len() + second.len(), "gapless");
    let both = [first, second].concat();
    let worst = off(&got, 1_000, &both, 1_000, 2 * frames - 1_000, 0.25);
    assert!(worst <= CLOSE as f64, "{worst} steps off a quarter");
}

/// A separator that fails: the parts stop (and say why), and the mixer
/// plays main, the file as it is, though a stem is muted.
#[test]
fn failing_stems_stop_the_parts_and_the_file_plays_as_it_is() {
    let dir = TempDir::new("stems-broken");
    let (path, file) = tones_of(&dir, "tones.wav", 2 * RATE as usize);
    let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[RATE], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(target), options(true, broken));
    player.make_stems(MakeStems::Separate).unwrap();
    part(&lanes, Stem::Bass).set_mute(true);
    gate.allow_all();
    player.load(vec![path]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(lanes);
    let stopped = seen.iter().find_map(|e| match e {
        Event::PartsStopped { index: 0, reason } => Some(reason.clone()),
        _ => None,
    });
    assert_eq!(stopped.as_deref(), Some("out of memory"), "{seen:#?}");
    assert!(!seen.iter().any(|e| matches!(
        e,
        Event::Error { .. }
            | Event::TrackEnded {
                finished: false,
                ..
            }
    )));
    assert!(played(&captures) == file, "the file as it is");
}

/// Stems kept for one file: its whole mix as the voice, nothing else.
struct Kept {
    track: PathBuf,
    stems: PathBuf,
}

impl StemStore for Kept {
    fn open(&self, path: &Path) -> Option<StemFile> {
        (path == self.track)
            .then(|| StemFile::open(&self.stems).ok())
            .flatten()
    }
}

/// Stems of `file` kept for `track` in `dir`: `gain` times the file as each
/// stem `stems` names, the others silent.
fn kept(dir: &Path, track: &Path, file: &[i32], stems: &[Stem], gain: f32) -> Arc<dyn StemStore> {
    let path = dir.join("kept.stems");
    let mut w = StemWriter::create(&path, RATE, "test").unwrap();
    let full = f32::powi(2.0, 23);
    let side = |c: usize| -> Vec<f32> {
        file.iter()
            .skip(c)
            .step_by(2)
            .map(|&s| gain * s as f32 / full)
            .collect()
    };
    let (left, right) = (side(0), side(1));
    let silent = vec![0.0f32; left.len()];
    let parts = PerStem::from_fn(|s| {
        if stems.contains(&s) {
            [&left[..], &right[..]]
        } else {
            [&silent[..], &silent[..]]
        }
    });
    w.push(parts).unwrap();
    w.finish().unwrap();
    Arc::new(Kept {
        track: track.to_path_buf(),
        stems: path,
    })
}

/// A track whose stems are kept plays them: nothing is separated (the
/// separator here only fails), and the strips play them (the voice soloed:
/// the file; muted: nothing).
#[test]
fn kept_stems_play_without_a_separation() {
    let dir = TempDir::new("stems-kept");
    let frames = 2 * RATE as usize;
    let (path, file) = tones_of(&dir, "tones.wav", frames);
    let store = kept(&dir, &path, &file, &[Stem::Vocals], 1.0);
    for (solo, scale) in [(true, 1.0), (false, 0.0)] {
        let (target, captures, gate, lanes) = gated_mixer_lanes(caps(&[RATE], &[S24In32Low]));
        let options = PlayerOptions {
            stem_store: Some(store.clone()),
            ..options(true, broken)
        };
        let (player, events) = Player::new(Box::new(target), options);
        player.make_stems(MakeStems::Kept).unwrap();
        let vocals = part(&lanes, Stem::Vocals);
        if solo {
            vocals.set_solo(true);
        } else {
            vocals.set_mute(true);
        }
        gate.allow_all();
        player.load(vec![path.clone()]).unwrap();
        player.play(0).unwrap();
        let seen = until_queue_end(&events);
        drop(player);
        drop(lanes);
        assert!(
            !seen.iter().any(|e| matches!(
                e,
                Event::Error { .. } | Event::Notice { .. } | Event::PartsStopped { .. }
            )),
            "solo {solo}: nothing separated, nothing failed"
        );
        let got = played(&captures);
        assert_eq!(got.len(), file.len());
        // Within the 16 bits the stems are kept in, past the mixer's switch.
        let worst = off(&got, 1_000, &file, 1_000, frames - 1_000, scale);
        assert!(worst <= 600.0, "solo {solo}: {worst} steps off");
    }
}
