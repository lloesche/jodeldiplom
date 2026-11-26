//! The engine through a mixer ([`jodel_output::mixer`]): on its Lead input,
//! alone at gain 1 (and on a desk with two heads, the second idle), the
//! device gets exactly the bytes it gets without one at the desk's format,
//! on every route and through seeks, jumps and pauses; and
//! [`Command::PlayAt`] starts a track a share of its length in. The device
//! here plays what it is given at once, so the mixer mixes a period as
//! soon as the engine has written one: its one path, as on a device.

mod common;

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use common::*;
use jodel_output::mixer::desk_format;
use jodel_output::{Capabilities, SampleFormat, StreamFormat};
use jodel_playback::{Captures, Command, Event, JumpAt, PlayState, Player, PlayerOptions};
use jodel_testkit::{fixture, TempDir};
use SampleFormat::*;

#[derive(Clone, Copy, Debug)]
enum Via {
    Direct,
    Mixer,
    /// A player's two heads on the desk (`StemDesk::two_heads`).
    Desk,
}

type Script<'a> = &'a dyn Fn(&Player, &Receiver<Event>, &Gate, &Captures);

/// Runs `script` on a player on a gated device that takes `caps`, directly
/// or on a mixer's Lead input, and returns what the device got.
fn run(
    via: Via,
    caps: Capabilities,
    options: PlayerOptions,
    script: Script,
) -> Vec<(StreamFormat, Vec<u8>)> {
    let (player, events, captures, gate, mixer) = match via {
        Via::Direct => {
            let (target, captures, gate) = GatedTarget::new(caps);
            let (player, events) = Player::new(Box::new(target), options);
            (player, events, captures, gate, None)
        }
        Via::Mixer => {
            let (target, captures, gate, mixer) = gated_mixer(caps);
            let (player, events) = Player::new(Box::new(target), options);
            (player, events, captures, gate, Some(mixer))
        }
        Via::Desk => {
            let (lead, guest, captures, gate, _desk, mixer) =
                gated_desk(caps, jodel_output::SimClock::Instant);
            let (player, events) = Player::on_desk(lead, guest, options);
            (player, events, captures, gate, Some(mixer))
        }
    };
    script(&player, &events, &gate, &captures);
    drop(player);
    drop(mixer);
    captured(&captures)
}

/// The same bytes, stream for stream, without the mixer (on a device at
/// the format the desk runs one that takes `caps` at), with it, and on a
/// desk with two heads.
fn same_through_the_mixer(caps: Capabilities, options: PlayerOptions, script: Script) {
    let desk = desk_format(&caps).expect("a desk's format");
    let one = [desk].into_iter().collect();
    let direct = run(Via::Direct, one, options.clone(), script);
    assert!(!direct.is_empty() && direct.iter().all(|(_, b)| !b.is_empty()));
    for via in [Via::Mixer, Via::Desk] {
        let mixed = run(via, caps.clone(), options.clone(), script);
        assert_eq!(direct.len(), mixed.len(), "streams {via:?}");
        for ((format, a), (format_mixed, b)) in direct.iter().zip(&mixed) {
            assert_eq!(format, format_mixed);
            assert!(
                a == b,
                "{format}: {} bytes directly, {} through {via:?}",
                a.len(),
                b.len()
            );
        }
    }
}

/// Plays `files` to the end of the queue.
fn to_the_end(files: Vec<PathBuf>) -> impl Fn(&Player, &Receiver<Event>, &Gate, &Captures) {
    move |player, events, gate, _| {
        gate.allow_all();
        player.load(files.clone()).unwrap();
        player.play(0).unwrap();
        until_queue_end(events);
    }
}

#[test]
fn flac_gapless_and_resampled_are_bit_exact() {
    let cases = [
        (caps(&[44_100], &[S24In32Low]), vec![F44]),
        (caps(&[48_000], &[S24Packed]), vec![F48]),
        (caps(&[44_100], &[S24In32Low]), vec![F44, F44]),
        (caps(&[48_000], &[S24In32Low]), vec![F44, F44]),
        // As it is, then resampled, gapless.
        (caps(&[44_100], &[S24In32Low]), vec![F44, F48]),
    ];
    for (caps, files) in cases {
        let files = files.into_iter().map(fixture).collect();
        same_through_the_mixer(caps, player_options(), &to_the_end(files));
    }
}

#[test]
fn plain_and_resampled_pcm_are_bit_exact() {
    let dir = TempDir::new("mixer-pcm");
    let (w48, w44) = (dir.join("48.wav"), dir.join("44.wav"));
    ramp_wav(&w48, 48_000, 30_000);
    ramp_wav(&w44, 44_100, 30_000);
    same_through_the_mixer(
        caps(&[48_000], &[S24In32Low]),
        player_options(),
        &to_the_end(vec![w48]),
    );
    same_through_the_mixer(
        caps(&[48_000], &[S24In32]),
        player_options(),
        &to_the_end(vec![w44]),
    );
}

#[test]
fn seek_pause_and_jump_are_bit_exact() {
    let dir = TempDir::new("mixer-moves");
    let wav = dir.join("ramp.wav");
    ramp_wav(&wav, 44_100, 3 * 44_100);
    // Each move is made once the device has every frame written (whole
    // periods of the mixer: 441 frames at 44.1 kHz, and past the output's
    // prefill, from which it plays): a seek drops what the device has not
    // taken, and it is to take the same with or without the mixer.
    let up_to = |gate: &Gate, captures: &Captures, frames: u64| {
        gate.allow(frames);
        gate.wait_written(frames);
        wait_captured(captures, frames);
    };
    let script = |player: &Player, events: &Receiver<Event>, gate: &Gate, captures: &Captures| {
        player.load(vec![wav.clone()]).unwrap();
        gate.allow(45 * 441);
        player.play(0).unwrap();
        up_to(gate, captures, 45 * 441);
        player.pause().unwrap();
        collect_until(events, |e| matches!(e, Event::State(PlayState::Paused)));
        player.resume().unwrap();
        collect_until(events, |e| matches!(e, Event::State(PlayState::Playing)));
        up_to(gate, captures, 90 * 441);
        player.seek(2.0).unwrap();
        collect_until(events, |e| matches!(e, Event::Seeked { .. }));
        up_to(gate, captures, 136 * 441);
        player.jump(0, 0.5, JumpAt::Now).unwrap();
        // Taken before anything more is let through: the rest of the
        // packet under way is not to go out on one run and not the other.
        collect_until(events, |e| matches!(e, Event::Jumped { .. }));
        gate.allow_all();
        until_queue_end(events);
    };
    same_through_the_mixer(caps(&[44_100], &[S24In32Low]), player_options(), &script);
    // A FLAC file's seek, too.
    let script = |player: &Player, events: &Receiver<Event>, gate: &Gate, captures: &Captures| {
        player.load(vec![fixture(F44)]).unwrap();
        gate.allow(11 * 441);
        player.play(0).unwrap();
        up_to(gate, captures, 11 * 441);
        player.seek(3_000.0 / 44_100.0).unwrap();
        collect_until(events, |e| matches!(e, Event::Seeked { .. }));
        gate.allow_all();
        until_queue_end(events);
    };
    same_through_the_mixer(caps(&[44_100], &[S24In32Low]), player_options(), &script);
}

#[test]
fn play_at_starts_a_share_of_the_length_in() {
    let dir = TempDir::new("mixer-play-at");
    let wav = dir.join("ramp.wav");
    let frames = 44_101u64;
    let samples = ramp_wav(&wav, 44_100, frames as u32);
    let (lead, captures, mixer) = mixer_lead(caps(&[44_100], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(lead), player_options());
    player.load(vec![wav]).unwrap();
    player
        .send(Command::PlayAt {
            index: 0,
            share: 1.0 / 3.0,
        })
        .unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(mixer);
    let from = (frames as f64 / 3.0).round() as usize;
    let start = seen.iter().find_map(|e| match e {
        Event::TrackStarted(s) => Some(s.from_frame),
        _ => None,
    });
    assert_eq!(start, Some(from as u64));
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1);
    assert!(streams[0].1 == device_bytes(&samples[2 * from..], S24In32Low));
}

/// On a desk (stereo only), a mono file plays doubled onto both sides,
/// exactly, and a 5.1 file folded down to stereo ([`Remix`]), each said so
/// in its route; neither opens the device again.
#[test]
fn mono_and_5_1_files_play_as_stereo() {
    use jodel_playback::remix::Remix;
    let dir = TempDir::new("mixer-channels");
    let frames = 4_800usize;
    let mono: Vec<i32> = (0..frames).map(|k| k as i32 * 37 - 90_000).collect();
    let surround: Vec<i32> = (0..frames * 6)
        .map(|i| ((i % 6) as i32 + 1) * 100_000 + (i / 6) as i32)
        .collect();
    let (m, s) = (dir.join("1-mono.wav"), dir.join("2-surround.wav"));
    write_wav(&m, 48_000, 1, &mono);
    write_wav(&s, 48_000, 6, &surround);
    let (lead, captures, mixer) = mixer_lead(caps(&[48_000], &[S24In32Low]));
    let (player, events) = Player::new(Box::new(lead), player_options());
    player.load(vec![m, s]).unwrap();
    player.play(0).unwrap();
    let seen = until_queue_end(&events);
    drop(player);
    drop(mixer);
    let routes: Vec<_> = seen
        .iter()
        .filter_map(|e| match e {
            Event::TrackStarted(s) => Some(s.route.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(routes.len(), 2);
    for (route, from) in routes.iter().zip([1, 6]) {
        assert_eq!((route.input_channels, route.channels), (from, 2));
        assert!(!route.lossless);
    }
    assert_eq!(routes[0].describe(), "48 kHz mono → stereo");
    assert_eq!(routes[1].describe(), "48 kHz 5.1 → stereo");
    let mut want: Vec<i32> = mono.iter().flat_map(|&s| [s, s]).collect();
    Remix::to_stereo(6).unwrap().apply(&surround, &mut want);
    let streams = captured(&captures);
    assert_eq!(streams.len(), 1, "one opening, gapless");
    assert!(streams[0].1 == device_bytes(&want, S24In32Low));
}
