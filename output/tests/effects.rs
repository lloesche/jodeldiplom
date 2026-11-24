//! The desk's effects input, on a simulated device: a sound cued on a
//! stream frame starts on that very frame; one late by a little starts at
//! once, one late by more, cancelled or of an output flushed never does;
//! with nothing sounding, or its strip down, the device gets the input
//! bit for bit; and the microphone's tap hears the effects with the rest
//! of what plays, so its echo canceller takes them out of the voice.

mod common;

use std::time::Duration;

use common::*;
use jodel_output::mixer::{EffectCue, Effects, Fader, InputTarget, Mixer, Role, SoundBank};
use jodel_output::pack::unpack;
use jodel_output::{Output, OutputTarget};

/// A sound of `n` frames on the right side only, at `level` (full scale
/// 1), its first frame as loud as the rest.
fn right_only(n: usize, level: f32) -> Vec<f32> {
    (0..n).flat_map(|_| [0.0, level]).collect()
}

/// Every frame of stream `i`, (left, right).
fn both(device: &Device, i: usize) -> Vec<(i32, i32)> {
    let (sink, _) = device.streams.lock().unwrap()[i].clone();
    let v = unpack(&sink.bytes(), device.format());
    v.chunks_exact(2).map(|f| (f[0], f[1])).collect()
}

/// The runs where the right side is loud: (first frame, length).
fn loud_runs(frames: &[(i32, i32)]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut at = None;
    for (k, &(_, r)) in frames.iter().enumerate() {
        match (at, r.abs() > 1_000_000) {
            (None, true) => at = Some(k),
            (Some(from), false) => {
                runs.push((from, k - from));
                at = None;
            }
            _ => {}
        }
    }
    if let Some(from) = at {
        runs.push((from, frames.len() - from));
    }
    runs
}

/// A desk: the player's strip with a Lead input on it, an effects strip
/// with the effects input, its bank one sound (`sound`) at `rate`.
struct Desk {
    mixer: Mixer,
    device: Device,
    lead: InputTarget,
    fader: Fader,
    effects: Effects,
    strip: jodel_output::mixer::Strip,
    rate: u32,
}

fn desk(sound: Vec<f32>) -> Desk {
    desk_at(sound, RATE)
}

fn desk_at(sound: Vec<f32>, rate: u32) -> Desk {
    let (mixer, device) = instant_mixer();
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let mut bank = SoundBank::new(rate);
    assert_eq!(bank.add(sound), 0);
    effects.load(bank);
    let (lead, fader) = mixer.input(&player, Role::Lead, 1.0);
    Desk {
        mixer,
        device,
        lead,
        fader,
        effects,
        strip,
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
}

#[test]
fn a_sound_starts_on_the_stream_frame_it_was_cued_on() {
    // At the device's rates: the bank at each.
    for rate in [RATE, 96_000] {
        let mut d = desk_at(right_only(480, 0.4), rate);
        let mut out = d.open();
        let cue = EffectCue::new(0);
        for frame in [12_345, 30_000, 30_100] {
            assert!(d.effects.play_at(cue, &d.fader, frame));
        }
        out.write(&frames(0, 48_000, |k| k as i32), None).unwrap();
        out.drain(None).unwrap();
        let played = both(&d.device, 0);
        // The left side is the stream frame playing: each sound starts on
        // its own (the last two overlap: one run, louder where both sound).
        let runs = loud_runs(&played);
        assert_eq!(runs.len(), 2, "{rate}: {runs:?}");
        for (&(from, len), (want, long)) in runs.iter().zip([(12_345, 480), (30_000, 580)]) {
            let at = played[from].0;
            assert!(at.abs_diff(want) <= 2, "{rate}: {at} for {want}");
            assert!(played[from - 1].1.abs() < 1_000, "nothing just before");
            assert_eq!(len, long);
        }
        let (r1, r2) = (played[30_050].1, played[30_150].1);
        assert!(r2 > r1 + r1 / 2, "{r1} {r2}");
        assert!(played[12_345 + 100].1.abs_diff(3_355_443) <= 2);
        // Where no sound is, the input as written.
        let off = |from: usize, to: usize| (from..to).find(|&k| played[k] != (k as i32, 0));
        // (The periods a sound sounds in are summed, dithered, whole.)
        let off = off(0, 11_000).or(off(40_000, 48_000));
        assert_eq!(off, None, "{rate}: {:?}", off.map(|k| played[k]));
        assert!(d.mixer.passthrough(), "bit for bit again after");
        let view = d.effects.view();
        assert_eq!((view.played, view.missed, view.sounding), (3, 0, 0));
        assert_eq!(view.rate, rate);
    }
}

#[test]
fn a_late_cue_starts_at_once_or_is_missed() {
    let mut d = desk(right_only(480, 0.5));
    let mut out = d.open();
    out.write(&frames(0, 24_000, |k| k as i32), None).unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 24_000);
    let cue = EffectCue::new(0);
    // Far behind: never; a little: at once; cancelled: never.
    d.effects.play_at(cue, &d.fader, 0);
    d.effects.play_at(cue, &d.fader, 23_000);
    d.effects.play_at(cue, &d.fader, 40_000);
    out.write(&frames(24_000, 4_800, |k| k as i32), None)
        .unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 28_800);
    d.effects.cancel();
    // A sound the bank has not, never.
    d.effects.play(EffectCue::new(7));
    out.write(&frames(28_800, 19_200, |k| k as i32), None)
        .unwrap();
    out.drain(None).unwrap();
    let played = both(&d.device, 0);
    let runs = loud_runs(&played);
    assert_eq!(runs.len(), 1, "{runs:?}");
    assert!(played[runs[0].0].0.abs_diff(24_000) <= 2);
    let view = d.effects.view();
    assert_eq!((view.played, view.missed), (1, 3));
}

#[test]
fn a_cue_of_an_output_flushed_never_comes() {
    let mut d = desk(right_only(480, 0.5));
    let mut out = d.open();
    out.write(&frames(0, 9_600, |k| k as i32), None).unwrap();
    wait_until("mixed", || d.mixer.timebase().frame == 9_600);
    d.effects.play_at(EffectCue::new(0), &d.fader, 20_000);
    out.write(&frames(9_600, 480, |k| k as i32), None).unwrap();
    wait_until("taken", || d.mixer.timebase().frame == 10_080);
    // A seek: the stream's frames go on, but they are of another place.
    out.flush().unwrap();
    out.write(&frames(0, 24_000, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    let played = both(&d.device, 0);
    assert!(loud_runs(&played).is_empty());
    assert_eq!(d.effects.view().missed, 1);
}

#[test]
fn with_nothing_sounding_or_its_strip_down_the_input_is_bit_for_bit() {
    let input = |k: u64| ((k * 2_654_435_761) % 16_000_000) as i32 - 8_000_000;
    // Nothing cued; then a cue on the strip muted, and at -inf.
    for case in 0..3 {
        let mut d = desk(right_only(4_800, 0.9));
        match case {
            1 => d.strip.set_mute(true),
            2 => d.strip.set_level(0.0),
            _ => {}
        }
        let mut out = d.open();
        if case > 0 {
            d.effects.play_at(EffectCue::new(0), &d.fader, 1_000);
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
        assert_eq!(view.played, u64::from(case > 0), "case {case}");
    }
}

#[test]
fn a_sound_is_metered_on_its_strip() {
    let mut d = desk(right_only(4_800, 0.5));
    let mut out = d.open();
    d.effects.play_at(EffectCue::new(0), &d.fader, 100);
    out.write(&frames(0, 9_600, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    let meters = d.mixer.meters();
    let (_, peaks) = meters
        .strips
        .iter()
        .find(|(id, _)| *id == d.strip.id())
        .unwrap();
    assert!(
        peaks[0] < 1e-6 && (peaks[1] - 0.5).abs() < 1e-3,
        "{peaks:?}"
    );
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
    let mut bank = SoundBank::new(RATE);
    let tone: Vec<f32> = (0..9_600)
        .flat_map(|k| {
            let v = 0.5 * (k as f32 * 0.13).sin();
            [v, v]
        })
        .collect();
    bank.add(tone);
    effects.load(bank);
    let singer = Singer::new(|_| 0.0);
    mixer.set_mic(Some(singer.input.clone()));
    let (mut lead, fader) = mixer.input(&player, Role::Lead, 1.0);
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
    // Cued half a second on from what is mixed now.
    let at = fader.mixed_to().unwrap() + 24_000;
    assert!(effects.play_at(EffectCue::new(0), &fader, at));
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
    let loud = reference.iter().filter(|v| v.abs() > 0.2).count();
    let quiet = reference.iter().filter(|v| v.abs() < 1e-4).count();
    assert!(loud > 2_000, "the effect is in what plays: {loud}");
    assert!(quiet > 20_000, "and only where it plays: {quiet}");
    assert_eq!(effects.view().played, 1);
    // The microphone itself never reaches the device: what played is the
    // effect alone.
    let played = device.left(0);
    let sounded = played
        .iter()
        .filter(|v| v.unsigned_abs() > 1_000_000)
        .count();
    assert!((2_000..9_600).contains(&sounded), "{sounded}");
}

/// A sound cued as the song ends (its last phrase rated, the next track
/// held back, nothing running) is heard to its end, and one sounding as
/// the song pauses is not frozen: the effects keep the device playing
/// until they are done.
#[test]
fn the_last_sound_is_heard_after_the_song_stops() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let player = mixer.strip("Player");
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let mut bank = SoundBank::new(RATE);
    bank.add(vec![0.5; 2 * 4_800]);
    effects.load(bank);
    let (mut lead, _fader) = mixer.input(&player, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 9_600, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    // Paused, the song over: a sound now.
    out.pause().unwrap();
    assert!(effects.play(EffectCue::new(0)));
    wait_until("heard whole", || {
        let v = effects.view();
        v.played == 1 && v.sounding == 0
    });
    // A sound, then the song paused at once.
    out.resume().unwrap();
    out.write(&frames(0, 4_800, |_| 0), None).unwrap();
    assert!(effects.play(EffectCue::new(0)));
    out.pause().unwrap();
    wait_until("heard whole", || {
        let v = effects.view();
        v.played == 2 && v.sounding == 0
    });
    // The device plays out what it holds (its buffer, 200 ms here) before
    // it pauses: both heard whole.
    let loud = || {
        device
            .left(0)
            .iter()
            .filter(|v| v.abs_diff(4_194_304) <= 2)
            .count()
    };
    wait_until("both heard whole", || loud() == 2 * 4_800);
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(loud(), 2 * 4_800);
    assert_eq!(effects.view().missed, 0);
}

/// An effects input dropped takes only itself off the desk: one made after
/// it plays on.
#[test]
fn dropping_an_older_effects_input_leaves_the_newer() {
    let mut d = desk(right_only(480, 0.4));
    let newer = d.mixer.effects(&d.strip);
    let mut bank = SoundBank::new(RATE);
    bank.add(right_only(480, 0.4));
    newer.load(bank);
    let older = std::mem::replace(&mut d.effects, newer);
    drop(older);
    let mut out = d.open();
    assert!(d.effects.play_at(EffectCue::new(0), &d.fader, 1_000));
    out.write(&frames(0, 9_600, |k| k as i32), None).unwrap();
    out.drain(None).unwrap();
    assert_eq!(loud_runs(&both(&d.device, 0)).len(), 1);
    assert_eq!(d.effects.view().played, 1);
}
