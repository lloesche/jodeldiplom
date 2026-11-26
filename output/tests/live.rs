//! The effects input's live synth, on a simulated device: a note sent on
//! a stream frame is heard from that very frame (and the synth's
//! look-ahead after it); one late by a little starts at once, one late by
//! more never does, nor one waiting when its output is flushed (a seek);
//! one on the desk's own frames sounds with no output open; and with the
//! synth resting, the device gets the input bit for bit.

mod common;

use std::time::Duration;

use common::*;
use jodel_output::mixer::{Effects, Fader, InputTarget, Mixer, Role};
use jodel_output::pack::unpack;
use jodel_output::{Output, OutputTarget};
use jodel_synth::{Commands, PatchId};

/// Every frame of stream `i`, (left, right).
fn both(device: &Device, i: usize) -> Vec<(i32, i32)> {
    let (sink, _) = device.streams.lock().unwrap()[i].clone();
    let v = unpack(&sink.bytes(), device.format());
    v.chunks_exact(2).map(|f| (f[0], f[1])).collect()
}

/// Where the right side starts sounding after silence (what is mixed is
/// dithered: under a hundred-thousandth of full scale is silence): each
/// first frame.
///
/// A note rises from silence over the samples of its first block (the
/// synth ramps a voice's level), so its first few frames are under that:
/// it is heard from at most [`RISE`] frames after it starts.
fn onsets(frames: &[(i32, i32)]) -> Vec<usize> {
    let mut quiet = 1_000;
    let mut out = Vec::new();
    for (k, &(_, r)) in frames.iter().enumerate() {
        if r.unsigned_abs() < 20_000 {
            quiet += 1;
            continue;
        }
        if quiet >= 1_000 {
            out.push(k);
        }
        quiet = 0;
    }
    out
}

/// How many frames after it starts a note is heard by [`onsets`].
const RISE: u64 = 8;

/// Whether a note heard from `heard` started on `frame`, the synth's
/// look-ahead `latency` later.
fn on(heard: u64, frame: u64, latency: u64) -> bool {
    (frame + latency..=frame + latency + RISE).contains(&heard)
}

/// A desk: the player's strip with a Lead input on it, the effects input
/// on a strip of its own playing a tone synth at `rate` (its look-ahead
/// `latency` frames).
struct Desk {
    mixer: Mixer,
    device: Device,
    lead: InputTarget,
    fader: Fader,
    effects: Effects,
    patch: PatchId,
    latency: u64,
    rate: u32,
}

fn desk_at(rate: u32) -> Desk {
    desk_with(rate, rate)
}

/// The same, the synth at `synth_rate`.
fn desk_with(rate: u32, synth_rate: u32) -> Desk {
    let (mixer, device) = instant_mixer_at(rate);
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(synth_rate, 0.02);
    let latency = synth.latency() as u64;
    effects.live(synth);
    let (lead, fader) = mixer.input(&player, Role::Lead, 1.0);
    Desk {
        mixer,
        device,
        lead,
        fader,
        effects,
        patch,
        latency,
        rate,
    }
}

impl Desk {
    fn open(&mut self) -> Box<dyn Output> {
        let out = self
            .lead
            .open(&request(self.rate).prefill(Duration::ZERO))
            .unwrap();
        taken(&mut self.lead);
        out
    }

    /// A note on stream frame `frame` of the lead's output.
    fn note(&self, frame: u64) -> bool {
        let c = tone(self.patch, frame, 1.0);
        self.effects.commands(Some(&self.fader)).send(c)
    }
}

#[test]
fn a_note_is_heard_from_the_stream_frame_it_was_sent_on() {
    for rate in [RATE, 96_000] {
        let mut d = desk_at(rate);
        let mut out = d.open();
        for frame in [12_345, 30_000] {
            assert!(d.note(frame));
        }
        out.write(&frames(0, 48_000, |k| k as i32), None).unwrap();
        out.drain(None).unwrap();
        let played = both(&d.device, 0);
        // The left side is the stream frame playing (the notes are on the
        // right alone): each heard from its frame, the look-ahead later.
        let starts = onsets(&played);
        assert_eq!(starts.len(), 2, "{rate}: {starts:?}");
        for (&at, want) in starts.iter().zip([12_345, 30_000]) {
            let heard = played[at].0 as u64;
            assert!(
                on(heard, want, d.latency),
                "{rate}: {heard} for {want} (+{})",
                d.latency
            );
        }
        // The left side as written all along (dithered where mixed).
        let off = played
            .iter()
            .enumerate()
            .find(|(k, f)| f.0.abs_diff(*k as i32) > 256);
        assert_eq!(off, None);
        let view = d.effects.view();
        assert_eq!((view.commands, view.missed, view.live), (2, 0, rate));
    }
}

#[test]
fn a_late_note_starts_at_once_or_is_dropped_and_a_seek_calls_off_what_waits() {
    let mut d = desk_at(RATE);
    let mut out = d.open();
    out.write(&frames(0, 24_000, |k| k as i32), None).unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 24_000);
    // Far behind: never; a little: at once; ahead: waits.
    assert!(d.note(0));
    assert!(d.note(23_000));
    assert!(d.note(40_000));
    out.write(&frames(24_000, 4_800, |k| k as i32), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 28_800);
    // A seek: the frames go on, but they are of another place in the song;
    // what waits on them is called off.
    out.flush().unwrap();
    out.write(&frames(0, 24_000, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    let played = both(&d.device, 0);
    let starts = onsets(&played);
    assert_eq!(starts.len(), 1, "{starts:?}");
    let heard = played[starts[0]].0 as u64;
    assert!(on(heard, 24_000, d.latency), "{heard}");
    let view = d.effects.view();
    assert_eq!((view.commands, view.missed), (2, 1), "{view:?}");
}

/// The song over and its output closed, a note on the desk's own frames
/// (a little after the frame mixed now) is heard on the device kept open.
#[test]
fn a_note_on_the_desks_frames_sounds_after_the_song() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, 0.05);
    let latency = synth.latency() as u64;
    effects.live(synth);
    wait_until("open", || mixer.desk().master.stream.is_some());
    let at = mixer.timebase().frame + 9_600;
    assert!(effects.commands(None).send(tone(patch, at, 1.0)));
    wait_until("heard", || {
        let right: Vec<i32> = both(&device, 0).iter().map(|f| f.1).collect();
        right
            .iter()
            .filter(|v| v.unsigned_abs() > 1_000_000)
            .count()
            > 1_000
    });
    let played = both(&device, 0);
    let starts = onsets(&played);
    assert_eq!(starts.len(), 1, "{starts:?}");
    // On its frame (the device's frames are the desk's, from its opening),
    // its look-ahead after.
    let heard = starts[0] as u64;
    assert!(on(heard, at, latency), "{heard} for {at}");
    assert_eq!(effects.view().missed, 0);
}

/// With a synth playing nothing (and after a note, once its tail is
/// over), what goes to the device is the input as written.
#[test]
fn with_the_synth_resting_the_input_is_bit_for_bit() {
    let input = |k: u64| ((k * 2_654_435_761) % 16_000_000) as i32 - 8_000_000;
    let mut d = desk_at(RATE);
    let mut out = d.open();
    let data = frames(0, 24_000, input);
    out.write(&data, None).unwrap();
    out.drain(None).unwrap();
    let played: Vec<i32> = both(&d.device, 0)
        .into_iter()
        .flat_map(|(l, r)| [l, r])
        .collect();
    assert_eq!(played, data);
    assert!(d.mixer.passthrough());
    // A note, then its tail: then as written again.
    assert!(d.note(24_100));
    let tail = (jodel_output::mixer::LIVE_TAIL * f64::from(RATE)) as u64;
    let more = 24_000 + tail + 24_000;
    out.write(&frames(24_000, more, input), None).unwrap();
    out.drain(None).unwrap();
    let played = both(&d.device, 0);
    let after = 24_000 + 4_800 + tail as usize + 9_600;
    let off = (after..played.len()).find(|&k| played[k] != (input(k as u64), 0));
    assert_eq!(off, None);
    assert!(played[24_100 + 2_000].1 != 0, "the note sounded");
    assert!(d.mixer.passthrough(), "bit for bit again after");
}

/// A synth made at another rate than the device's renders nothing: its
/// notes are dropped (counted), the input goes to the device bit for bit.
#[test]
fn a_synth_at_another_rate_renders_nothing() {
    let input = |k: u64| ((k * 2_654_435_761) % 16_000_000) as i32 - 8_000_000;
    let mut d = desk_with(RATE, 96_000);
    let mut out = d.open();
    assert!(d.note(1_000));
    let data = frames(0, 24_000, input);
    out.write(&data, None).unwrap();
    out.drain(None).unwrap();
    let played: Vec<i32> = both(&d.device, 0)
        .into_iter()
        .flat_map(|(l, r)| [l, r])
        .collect();
    assert_eq!(played, data);
    assert!(d.mixer.passthrough());
    let view = d.effects.view();
    assert_eq!((view.commands, view.missed, view.live), (0, 1, 96_000));
}

/// A note further ahead than a game plans (of a clock gone wrong) is
/// dropped (counted), and the synth stays at rest: the input bit for bit.
#[test]
fn a_note_too_far_ahead_is_dropped_and_the_synth_rests() {
    let input = |k: u64| ((k * 2_654_435_761) % 16_000_000) as i32 - 8_000_000;
    let mut d = desk_at(RATE);
    let mut out = d.open();
    let ahead = (jodel_output::mixer::LIVE_AHEAD * f64::from(RATE)) as u64;
    assert!(d.note(ahead + 48_000));
    let desk = d.mixer.timebase().frame + ahead + 48_000;
    assert!(d.effects.commands(None).send(tone(d.patch, desk, 1.0)));
    let data = frames(0, 24_000, input);
    out.write(&data, None).unwrap();
    out.drain(None).unwrap();
    let played: Vec<i32> = both(&d.device, 0)
        .into_iter()
        .flat_map(|(l, r)| [l, r])
        .collect();
    assert_eq!(played, data);
    assert!(d.mixer.passthrough());
    let view = d.effects.view();
    assert_eq!((view.commands, view.missed), (0, 2), "{view:?}");
}
