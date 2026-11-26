//! The desk's effects input, on a simulated device: with nothing sounding,
//! or its strip down, the device gets the input bit for bit; what sounds
//! is metered on its strip; the microphone's tap hears the effects with
//! the rest of what plays, so its echo canceller takes them out of the
//! voice; and the device kept open plays them after the song. (Its live
//! synth's notes on a stream's frames: `live.rs`.)

mod common;

use std::time::Duration;

use common::*;
use jodel_output::mixer::{Effects, Fader, InputTarget, Mixer, Role};
use jodel_output::pack::unpack;
use jodel_output::{Output, OutputTarget};
use jodel_synth::{Commands, PatchId};
use jodel_testkit::real_time;

/// Every frame of stream `i`, (left, right).
fn both(device: &Device, i: usize) -> Vec<(i32, i32)> {
    let (sink, _) = device.streams.lock().unwrap()[i].clone();
    let v = unpack(&sink.bytes(), device.format());
    v.chunks_exact(2).map(|f| (f[0], f[1])).collect()
}

/// The notes heard on the right side, (first frame, length): where it is
/// loud, a gap shorter than 20 ms (a tone's own crossings of zero) inside
/// one.
fn loud_runs(frames: &[(i32, i32)]) -> Vec<(usize, usize)> {
    const GAP: usize = 960;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (k, &(_, r)) in frames.iter().enumerate() {
        if r.abs() <= 1_000_000 {
            continue;
        }
        match runs.last_mut() {
            Some((from, length)) if k < *from + *length + GAP => *length = k + 1 - *from,
            _ => runs.push((k, 1)),
        }
    }
    runs
}

/// A desk: the player's strip with a Lead input on it, an effects strip
/// with the effects input playing a tone synth (a note held `gate`
/// seconds).
struct Desk {
    mixer: Mixer,
    device: Device,
    lead: InputTarget,
    fader: Fader,
    effects: Effects,
    strip: jodel_output::mixer::Strip,
    patch: PatchId,
}

fn desk(gate: f32) -> Desk {
    let (mixer, device) = instant_mixer_at(RATE);
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, gate);
    effects.live(synth);
    let (lead, fader) = mixer.input(&player, Role::Lead, 1.0);
    Desk {
        mixer,
        device,
        lead,
        fader,
        effects,
        strip,
        patch,
    }
}

impl Desk {
    fn open(&mut self) -> Box<dyn Output> {
        let out = self
            .lead
            .open(&request(RATE).prefill(Duration::ZERO))
            .unwrap();
        taken(&mut self.lead);
        out
    }

    /// A note on the right side alone on stream frame `frame` of the
    /// lead's output.
    fn note(&self, frame: u64) -> bool {
        let c = tone(self.patch, frame, 1.0);
        self.effects.commands(Some(&self.fader)).send(c)
    }
}

#[test]
fn with_nothing_sounding_or_its_strip_down_the_input_is_bit_for_bit() {
    let input = |k: u64| ((k * 2_654_435_761) % 16_000_000) as i32 - 8_000_000;
    // Nothing sent; then a note on the strip muted, and at -inf.
    for case in 0..3 {
        let mut d = desk(0.1);
        match case {
            1 => d.strip.set_mute(true),
            2 => d.strip.set_level(0.0),
            _ => {}
        }
        let mut out = d.open();
        if case > 0 {
            assert!(d.note(1_000));
        }
        let data = frames(0, 24_000, input);
        out.write(&data, None).unwrap();
        out.drain(None).unwrap();
        let played: Vec<i32> = both(&d.device, 0)
            .into_iter()
            .flat_map(|(l, r)| [l, r])
            .collect();
        assert_eq!(played, data, "case {case}");
        assert!(d.mixer.passthrough());
        let view = d.effects.view();
        assert_eq!(view.commands, u64::from(case > 0), "case {case}");
    }
}

#[test]
fn a_sound_is_metered_on_its_strip() {
    let mut d = desk(0.1);
    let mut out = d.open();
    assert!(d.note(100));
    out.write(&frames(0, 9_600, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    let meters = d.mixer.meters();
    let (_, peaks) = meters
        .strips
        .iter()
        .find(|(id, _)| *id == d.strip.id())
        .unwrap();
    assert!(peaks[0] < 1e-6 && peaks[1] > 0.1, "{peaks:?}");
    assert_eq!(loud_runs(&both(&d.device, 0)).len(), 1);
}

/// The microphone's tap is on what goes to the device: its echo
/// canceller's reference (what plays) has the effects in it, where they
/// play, as it has the song.
#[test]
fn the_microphone_hears_the_effects_as_what_plays() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, 0.15);
    effects.live(synth);
    let singer = Singer::new(|_| 0.0);
    mixer.set_mic(Some(singer.input.clone()));
    let (mut lead, _) = mixer.input(&player, Role::Lead, 1.0);
    let mut out = lead.open(&request(RATE)).unwrap();
    let total = 3 * u64::from(RATE);
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, total, |_| 0), None).unwrap();
        out.drain(None).unwrap();
        out
    });
    wait_until("hearing", || {
        mixer.mic().port == jodel_output::mixer::MicPort::Hearing
    });
    let dir = std::env::temp_dir().join(format!("jodel-fx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("echo.wav");
    let mut recording = singer.input.record_echo(&path, 2.0).unwrap();
    // In the middle, a little after the frame mixed now.
    let at = mixer.timebase().frame + 2_400;
    assert!(effects.commands(None).send(tone(patch, at, 0.0)));
    let _out = writer.join().unwrap();
    let file = loop {
        if let Some(done) = recording.done() {
            break done.unwrap();
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let bytes = std::fs::read(&file).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let floats: Vec<f32> = bytes[80..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // Channel 1 of 3: what plays.
    let reference: Vec<f32> = floats.chunks_exact(3).map(|f| f[1]).collect();
    let loud = reference.iter().filter(|v| v.abs() > 0.1).count();
    let quiet = reference.iter().filter(|v| v.abs() < 1e-4).count();
    assert!(loud > 2_000, "the effect is in what plays: {loud}");
    assert!(quiet > 20_000, "and only where it plays: {quiet}");
    assert_eq!(effects.view().commands, 1);
    // The microphone itself never reaches the device: what played is the
    // effect alone.
    let played = device.left(0);
    let sounded = played
        .iter()
        .filter(|v| v.unsigned_abs() > 1_000_000)
        .count();
    assert!((2_000..24_000).contains(&sounded), "{sounded}");
}

/// The loud frames of the right side of `device`'s first stream.
fn loud(device: &Device) -> usize {
    (both(device, 0).iter())
        .filter(|f| f.1.unsigned_abs() > 1_000_000)
        .count()
}

/// A sound sent on the desk's frames as the song ends (its last phrase
/// rated, the next track held back, nothing running) is heard to its end,
/// and one sounding as the song pauses is not frozen: the effects keep the
/// device playing until they are done. (On a device in real time on the
/// test's time: a sound sent on the frame the mixer mixes next is never
/// late, however busy the machine.)
#[test]
fn the_last_sound_is_heard_after_the_song_stops() {
    let (mixer, device) = manual_mixer();
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, 0.1);
    effects.live(synth);
    let (mut lead, _fader) = mixer.input(&player, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 9_600, |_| 0), None).unwrap();
    device.drain(&mixer, &mut *out);
    // Paused, the song over: a sound now, heard whole.
    out.pause().unwrap();
    let now = || mixer.timebase().frame + 480;
    assert!(effects.commands(None).send(tone(patch, now(), 1.0)));
    device.tick_until(&mixer, "heard", || loud(&device) > 4_000);
    device.tick_until(&mixer, "at rest", || mixer.passthrough());
    let one = loud(&device);
    assert!(one > 4_000, "{one}");
    // A sound, then the song paused at once.
    out.resume().unwrap();
    out.write(&frames(0, 4_800, |_| 0), None).unwrap();
    assert!(effects.commands(None).send(tone(patch, now(), 1.0)));
    out.pause().unwrap();
    // The device plays on (it stays open, in real time): both heard whole.
    device.tick_until(&mixer, "both heard", || loud(&device) > one + 4_000);
    device.tick_until(&mixer, "at rest", || mixer.passthrough());
    let two = loud(&device);
    assert!(two.abs_diff(2 * one) < one / 20, "{one} then {two}");
    assert_eq!(effects.view().missed, 0);
}

/// The device is open before anything plays and stays open: the song over
/// and its output closed, sounds sent then (a moment apart, on the desk's
/// frames) are heard whole on the same opening of it, and it is still open
/// after them. (On the test's time, as above.)
#[test]
fn sounds_after_the_song_play_on_the_device_kept_open() {
    let (mixer, device) = manual_mixer();
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, 0.1);
    effects.live(synth);
    let open_now = || mixer.desk().master.stream.is_some();
    wait_until("open from the start", open_now);
    let (mut lead, _fader) = mixer.input(&player, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 9_600, |_| 0), None).unwrap();
    device.drain(&mixer, &mut *out);
    out.close().unwrap();
    let mut heard = Vec::new();
    for n in 1..=2 {
        // A moment apart: 300 ms of the device's.
        device.ticks(&mixer, 30);
        let at = mixer.timebase().frame + 480;
        assert!(effects.commands(None).send(tone(patch, at, 1.0)));
        let before = heard.last().copied().unwrap_or(0);
        device.tick_until(&mixer, "heard", || loud(&device) > before + 4_000);
        device.tick_until(&mixer, "at rest", || mixer.passthrough());
        assert_eq!(effects.view().commands, n);
        heard.push(loud(&device));
    }
    assert!(heard[0] > 4_000, "{heard:?}");
    assert!(heard[1].abs_diff(2 * heard[0]) < heard[0] / 20, "{heard:?}");
    device.ticks(&mixer, 30);
    assert!(open_now(), "still open");
    assert_eq!(device.streams.lock().unwrap().len(), 1, "opened once");
    assert_eq!(effects.view().missed, 0);
}

/// An effects input dropped takes only itself off the desk: one made after
/// it plays on.
#[test]
fn dropping_an_older_effects_input_leaves_the_newer() {
    let mut d = desk(0.1);
    let newer = d.mixer.effects(&d.strip);
    let (synth, patch) = tone_synth(RATE, 0.1);
    newer.live(synth);
    let older = std::mem::replace(&mut d.effects, newer);
    drop(older);
    d.patch = patch;
    let mut out = d.open();
    assert!(d.note(1_000));
    out.write(&frames(0, 24_000, |k| k as i32), None).unwrap();
    out.drain(None).unwrap();
    assert_eq!(loud_runs(&both(&d.device, 0)).len(), 1);
    assert_eq!(d.effects.view().commands, 1);
}
