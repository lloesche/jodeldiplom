//! The mixer in real time, on a simulated device: an input's clock is what
//! the device has played, a ramp lands on its frame, a crossfade's two
//! sides start on the same frame, a starved input leaves the one heard bit
//! for bit, a flush drops what the device held, the microphone's port hears
//! song after song in each Lead's frames on the one opening of the device
//! and never reaches it, the monitor sees the mix, every input is offered
//! the desk's one format and the device stays open; a closed input cannot
//! open again, and one withdrawn before its first audio leaves the
//! crossfade's other side as it was, while one heard is left as it is.
//! Strips: 0 dB is bit for bit, a level change is a 20 ms linear ramp,
//! mute and solo silence without a click, and the meters read each strip
//! and the sum. Where frames must line up exactly, the device plays what
//! it is given at once ([`instant_mixer`]); in real time it plays silence
//! before anything is written ([`lead_in`] finds where an output starts).

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use jodel_output::mixer::{
    equal_power, Curve, MicPort, MicView, Monitor, Ramp, Role, Start, Timebase,
};
use jodel_output::{Clock, Error, Output, OutputTarget, StreamFormat, StreamInfo, VoiceBlock};
use jodel_testkit::real_time;

#[test]
fn an_inputs_position_is_what_the_device_has_played() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let clock = out.clock();
    // The writer goes on until told to stop: the samples below are taken
    // while it plays, however long a busy machine takes to give them.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = stop.clone();
    let writer = std::thread::spawn(move || {
        let chunk = u64::from(RATE) / 10;
        let mut written = 0;
        while !stopped.load(std::sync::atomic::Ordering::Relaxed) {
            out.write(&frames(written, chunk, |_| 1_000), None).unwrap();
            written += chunk;
        }
        out.drain(None).unwrap();
        (out, written)
    });
    wait_until("playing", || clock.position() > 4_800);
    let period = u64::from(RATE / 100);
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    // The output's position read between two of the mixer's timebases (a
    // thread put off between the reads cannot make them disagree). The
    // mixer tells each output what the device has played before it
    // publishes the timebase of that turn, so the second is the next one
    // published after the read: it is at least as new.
    let next_timebase = |was: Timebase| {
        let until = Instant::now() + Duration::from_millis(100);
        loop {
            let tb = mixer.timebase();
            if tb != was || Instant::now() > until {
                return tb;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    };
    let mut samples = Vec::new();
    while samples.len() < 100 || clock.position() < 3 * u64::from(RATE) / 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "{} samples in a minute",
            samples.len()
        );
        let before = mixer.timebase();
        let position = clock.position();
        let after = next_timebase(mixer.timebase());
        // What the mixer took but the device has not played is not heard.
        assert!(
            after.frame > position + period,
            "position {position}, mixed {}",
            after.frame
        );
        assert_eq!((after.epoch, after.rate), (1, RATE));
        samples.push((before.heard, position, after.heard));
        std::thread::sleep(Duration::from_millis(2));
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (out, written) = writer.join().unwrap();
    assert_eq!(out.position().unwrap(), written);
    assert_eq!(device.underruns(), 0);
    // The device frames heard are the output's and, where it had none when
    // the mixer took a period (a writer put off: the mixer fills the
    // device's buffer ahead, and does not wait), silence: each position is
    // the output's frames among those heard, to a period.
    let gaps = out.stats().underruns;
    let left = device.left(0);
    let ours = |heard: u64| {
        left[..(heard as usize).min(left.len())]
            .iter()
            .filter(|&&v| v == 1_000)
            .count() as u64
    };
    for (before, position, after) in samples {
        let (from, to) = (ours(before), ours(after));
        assert!(
            from <= position + period && position <= to + period,
            "position {position}: heard {before} then {after}, of them the output's {from} \
             then {to} ({gaps} gaps)"
        );
    }
}

#[test]
fn a_ramp_lands_on_its_frame_and_unity_solo_is_bit_exact() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    let level = 1_000_000;
    let clock = out.clock();
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, total, |_| level), None).unwrap();
        out.drain(None).unwrap();
    });
    wait_until("playing", || clock.position() > 4_800);
    let tb = mixer.timebase();
    let at = tb.frame + 9_600;
    let over = 480u64;
    fader.set(Ramp {
        start: tb.at(at),
        to: 0.0,
        over: frames_of(over),
        curve: Curve::Linear,
    });
    writer.join().unwrap();
    assert!(fader.settled() && fader.gain() == 0.0);
    drop(mixer);
    assert_eq!(device.underruns(), 0);
    let left = device.left(0);
    let at = at as usize;
    let start = lead_in(&left, |_| level);
    assert!(left[..start].iter().all(|&v| v == 0), "silence before it");
    // As written, bit for bit, up to the ramp.
    assert!(left[start..at].iter().all(|&v| v == level));
    for k in 0..over as usize {
        let want = f64::from(level) * (1.0 - k as f64 / over as f64);
        // Dithered: within the dither's reach.
        assert!((f64::from(left[at + k]) - want).abs() < 1.5, "{k}");
    }
    // Then silence, exactly.
    assert!(left[at + over as usize..start + total as usize]
        .iter()
        .all(|&v| v == 0));
}

#[test]
fn a_crossfade_starts_both_sides_on_the_first_frame_of_what_comes_in() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let (a, b) = (1_000_000, 2_000_000);
    let mut out = open(&mut lead, RATE);
    let clock = out.clock();
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, u64::from(RATE), |_| a), None).unwrap();
        out
    });
    wait_until("the song plays", || clock.position() > 4_800);
    let over = 960u64;
    mixer.crossfade(
        &guest_fader,
        Start::FirstAudio,
        &[&lead_fader],
        frames_of(over),
    );
    std::thread::sleep(Duration::from_millis(30));
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 24_000, |_| b), None).unwrap();
    g.drain(None).unwrap();
    assert!(lead_fader.settled() && lead_fader.gain() == 0.0);
    assert!(guest_fader.settled() && guest_fader.gain() == 1.0);
    let _out = writer.join().unwrap();
    drop(mixer);
    let left = device.left(0);
    // The song as written up to the guest's first frame; then, from that
    // very frame, the song falling and the guest rising, equal-power; then
    // the guest as written. (The first frame of the fade is the song's,
    // dithered: it may read as written.)
    let fits = |start: usize| {
        (0..over as usize).all(|k| {
            let t = k as f64 / over as f64;
            let (fall, rise) = (equal_power(1.0, 0.0, t), equal_power(0.0, 1.0, t));
            let want = f64::from(a) * fall + f64::from(b) * rise;
            (f64::from(left[start + k]) - want).abs() < 1.5
        })
    };
    let song = lead_in(&left, |_| a);
    assert!(left[..song].iter().all(|&v| v == 0), "silence before it");
    let changed = song + left[song..].iter().position(|&v| v != a).unwrap();
    let start = [changed - 1, changed].into_iter().find(|&s| fits(s));
    assert!(
        start.is_some(),
        "no fade from {changed}: {:?}",
        &left[changed - 2..changed + 4]
    );
    let start = start.unwrap();
    assert!(left[song..start].iter().all(|&v| v == a));
    // (The frames of the period the fade ends in, and of the next, go
    // through the mix, dithered.)
    let after = start + over as usize + 1_500;
    assert!(left[after..after + 10_000].iter().all(|&v| v == b));
}

#[test]
fn a_starved_input_leaves_the_one_heard_bit_for_bit() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, _guest_fader) = mixer.input(&strip, Role::Guest, 1.0);
    let mut out = open(&mut lead, RATE);
    let clock = out.clock();
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, 2 * u64::from(RATE), |k| k as i32), None)
            .unwrap();
        out.drain(None).unwrap();
    });
    wait_until("the song plays", || clock.position() > 4_800);
    // A guest that gives its prefill of silence, then nothing: running,
    // starved, at gain 1.
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    wait_until("the guest is starved", || g.position().unwrap() == 4_800);
    let from = mixer.timebase().frame + 4_800;
    wait_until("mixed past it", || mixer.timebase().frame > from + 24_000);
    assert!(mixer.desk().master.passthrough);
    writer.join().unwrap();
    drop(g);
    drop(mixer);
    let left = device.left(0);
    let from = from as usize;
    let song = lead_in(&left, |k| k as i32);
    assert!(left[..song].iter().all(|&v| v == 0), "silence before it");
    let wrong = (from..from + 24_000).find(|&d| left[d] != (d - song) as i32);
    assert_eq!(
        wrong.map(|d| (d, left[d])),
        None,
        "the song as written while the guest is starved ({} underruns)",
        device.underruns()
    );
}

#[test]
fn a_flush_drops_what_the_device_holds_of_it() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 24_000, |_| 111), None).unwrap();
    wait_until("it plays", || out.position().unwrap() > 9_600);
    out.flush().unwrap();
    assert_eq!(out.position().unwrap(), out.written());
    // Where the device was at the flush: as the mixer tells it next (what
    // it told last is from before it, however long before the flush came).
    let before = mixer.timebase().heard;
    wait_until("heard told again", || mixer.timebase().heard != before);
    let flushed = mixer.timebase().heard as usize;
    out.write(&frames(0, 9_600, |_| 222), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let left = device.left(0);
    // Gone at once (the device holds 200 ms here), and nothing of it after
    // what came after the flush.
    let last = left.iter().rposition(|&v| v == 111).unwrap();
    assert!(last < flushed + 2 * 480, "{last} {flushed}");
    let first = left.iter().position(|&v| v == 222).unwrap();
    assert!(first > last);
    assert_eq!(left[first..].iter().filter(|&&v| v == 222).count(), 9_600);
}

#[test]
fn a_seek_keeps_a_fade_under_way_and_a_pause_ends_it() {
    let _guard = real_time();
    let (mixer, _device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 24_000, |_| 1_000), None).unwrap();
    wait_until("it plays", || out.position().unwrap() > 4_800);
    fader.set(Ramp::fade(0.0, Duration::from_secs(2), Start::Now));
    wait_until("the fade is under way", || fader.gain() < 0.99);
    // A seek: the output is held back until its prefill, and the fade is
    // where it was, not at its end.
    out.flush().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let g = fader.gain();
    assert!(g > 0.5 && g < 0.99 && !fader.settled(), "{g}");
    out.write(&frames(0, 24_000, |_| 1_000), None).unwrap();
    wait_until("it fades on", || fader.gain() < g);
    // Paused, nothing of it is heard: the fade ends.
    out.pause().unwrap();
    wait_until("it settles", || fader.settled());
    assert_eq!(fader.gain(), 0.0);
}

#[test]
fn the_microphone_hears_the_device_in_the_leads_frames() {
    let _guard = real_time();
    let (mixer, _device) = mixer();
    let strip = mixer.strip("Player");
    let singer = Singer::new(loud);
    mixer.set_mic(Some(singer.input.clone()));
    // A guest plays (silent) first, so the device is ahead of the Lead.
    let (mut guest, _guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 9_600, |k| k as i32), None).unwrap();
    wait_until("the guest plays", || g.position().unwrap() == 9_600);
    assert_eq!(mixer.mic().port, MicPort::Idle);
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    out.write(&frames(0, total, |k| k as i32), None).unwrap();
    out.drain(None).unwrap();
    let hearing = MicView {
        port: MicPort::Hearing,
        refused: 0,
    };
    assert_eq!(mixer.desk().mic, hearing);
    // Cleaned to the Lead's last frames.
    let voice = singer.voice_until("the voice to the end", |v| {
        v.iter().any(|b| b.stream_frame > total - 4_800)
    });
    assert!(voice.len() > 100, "{} blocks", voice.len());
    // Told in the Lead's frames: the device's run 9 600 and more ahead.
    let last = voice.iter().map(|b| b.stream_frame).max().unwrap();
    assert!(last <= total && last > total - 4_800, "{last}");
    // (Before the Lead's first frame and after its last the frame told
    // stays where the Lead is.)
    let within = |w: &&[VoiceBlock]| w[0].stream_frame > 0 && w[1].stream_frame < total - 128;
    for w in voice.windows(2).filter(|w| w[1].continues).filter(within) {
        assert_eq!(w[1].stream_frame, w[0].stream_frame + 128);
    }
    // Outputs on inputs run no taps of their own.
    assert!(matches!(out.set_tap(None), Err(Error::Unsupported(_))));
    assert!(matches!(g.set_tap(None), Err(Error::Unsupported(_))));
    // Taken away: no tap.
    mixer.set_mic(None);
    wait_until("no microphone", || mixer.mic().port == MicPort::None);
}

/// The microphone is tapped for song after song (one output closed, the
/// next opened) on the one opening of the device, each told in its own
/// Lead output's frames, with no complaint.
#[test]
fn the_microphone_hears_song_after_song_on_one_opening() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let singer = Singer::new(loud);
    mixer.set_mic(Some(singer.input.clone()));
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut out = open(&mut lead, RATE);
        out.write(&frames(0, u64::from(RATE) / 2, |_| 0), None)
            .unwrap();
        out.drain(None).unwrap();
        assert_eq!(mixer.mic().port, MicPort::Hearing);
        let voice = singer.voice_until("the voice", |v| v.len() > 20);
        runs.push(voice.iter().map(|b| b.stream_frame).max().unwrap());
    }
    assert_eq!(device.streams.lock().unwrap().len(), 1);
    assert_eq!(mixer.timebase().epoch, 1);
    // Each song told its own Lead output's frames.
    assert!(runs.iter().all(|&r| r <= 24_000), "{runs:?}");
    assert_eq!(mixer.mic().refused, 0);
}

/// The microphone is listened to only: singing loudly into it, the device
/// gets the Lead's silence, bit for bit.
#[test]
fn the_microphone_never_reaches_the_device() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let singer = Singer::new(loud);
    mixer.set_mic(Some(singer.input.clone()));
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, u64::from(RATE), |_| 0), None).unwrap();
    out.drain(None).unwrap();
    singer.voice_until("the voice heard", |v| v.len() > 100);
    let desk = mixer.desk();
    assert!(desk.master.passthrough);
    drop(out);
    drop(mixer);
    let (sink, _) = device.streams.lock().unwrap()[0].clone();
    assert!(sink.bytes().iter().all(|&b| b == 0), "only silence played");
}

/// What a monitor saw.
#[derive(Default)]
struct Watched {
    rate: u32,
    /// The device frame the next frames should start at.
    next: u64,
    left: Vec<i32>,
    open: bool,
    out_of_order: bool,
}

#[derive(Clone, Default)]
struct Watch(Arc<Mutex<Watched>>);

impl Monitor for Watch {
    fn opened(&mut self, info: &StreamInfo, _: Clock) {
        let mut w = self.0.lock().unwrap();
        w.rate = info.format.rate;
        w.open = true;
    }

    fn mixed(&mut self, samples: &[i32], channels: usize, first_frame: u64) {
        let mut w = self.0.lock().unwrap();
        w.out_of_order |= first_frame != w.next;
        w.next += (samples.len() / channels) as u64;
        w.left.extend(samples.iter().step_by(channels));
    }

    fn closed(&mut self) {
        self.0.lock().unwrap().open = false;
    }
}

#[test]
fn the_monitor_sees_the_mix_of_two_inputs() {
    let (mixer, _device) = instant_mixer();
    let strip = mixer.strip("Player");
    let watch = Watch::default();
    mixer.set_monitor(Some(Box::new(watch.clone())));
    let (mut a, _fa) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&strip, Role::Guest, 0.5);
    // Both running short of a period: nothing is mixed until both hold
    // one.
    let (oa, ob) = (
        open_started(&mut a, RATE, 100_000),
        open_started(&mut b, RATE, 200_000),
    );
    let (oa, ob) = play_both(
        oa,
        ob,
        frames(1, 23_999, |_| 100_000),
        frames(1, 23_999, |_| 200_000),
    );
    {
        let w = watch.0.lock().unwrap();
        assert_eq!(w.rate, RATE);
        assert!(w.open && !w.out_of_order);
        // Both heard at once all through: 100k + 0.5 × 200k, dithered.
        let both = w.left.iter().filter(|&&v| (v - 200_000).abs() <= 1).count();
        assert_eq!(both, 24_000);
    }
    drop((oa, ob));
    // The device stays open with no output.
    std::thread::sleep(Duration::from_millis(50));
    assert!(watch.0.lock().unwrap().open);
}

/// The device is open before anything plays, at the desk's format, and
/// every input, Lead or Guest, is offered that one format: an output asked
/// at another is refused, and the device is never opened again.
#[test]
fn every_input_is_offered_the_desks_one_format() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let desk = StreamFormat::new(RATE, 2, FORMAT);
    assert_eq!(mixer.format(), Some(desk));
    wait_until("open from the start", || {
        mixer.desk().master.stream.is_some()
    });
    let strip = mixer.strip("Player");
    let (mut guest, _fg) = mixer.input(&strip, Role::Guest, 1.0);
    let (mut lead, _fl) = mixer.input(&strip, Role::Lead, 1.0);
    for target in [&mut guest, &mut lead] {
        let only: Vec<StreamFormat> = target.capabilities().unwrap().iter().copied().collect();
        assert_eq!(only, [desk]);
        assert!(matches!(
            target.open(&request(96_000)),
            Err(Error::FormatNotAccepted { .. })
        ));
    }
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 4_800, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    let mut out = open(&mut guest, RATE);
    out.write(&frames(0, 4_800, |_| 0), None).unwrap();
    out.drain(None).unwrap();
    assert_eq!(mixer.timebase().epoch, 1);
    assert_eq!(device.streams.lock().unwrap().len(), 1);
}

#[test]
fn a_closed_input_cannot_open_again_and_the_device_stays() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut g1, f1) = mixer.input(&strip, Role::Guest, 1.0);
    let mut o1 = open(&mut g1, RATE);
    o1.write(&frames(0, 100, |_| 0), None).unwrap();
    // Closed while its target is still held: its output closes, and it
    // cannot open again; the device plays on.
    f1.close();
    assert_eq!(g1.capabilities().err(), Some(Error::Closed));
    assert_eq!(o1.write(&frames(0, 100, |_| 0), None), Err(Error::Closed));
    assert!(matches!(g1.open(&request(RATE)), Err(Error::Closed)));
    // The next guest opens on it as it is.
    let (mut g2, _f2) = mixer.input(&strip, Role::Guest, 1.0);
    let o2 = open(&mut g2, RATE);
    assert_eq!(o2.format().rate, RATE);
    drop(f1);
    drop(g1);
    // Everything above taken by the mixer, and the device mixed into:
    // still the one opening, at its rate.
    taken(&mut g2);
    wait_until("the device mixed into", || mixer.timebase().epoch != 0);
    let tb = mixer.timebase();
    assert_eq!((tb.epoch, tb.rate), (1, RATE));
    assert_eq!(device.streams.lock().unwrap().len(), 1, "opened once");
}

#[test]
fn a_crossfade_withdrawn_before_its_first_audio_leaves_the_other_side_as_it_was() {
    let (mixer, device) = instant_mixer();
    let strip = mixer.strip("Player");
    let (mut lead, lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let level = 1_000_000;
    let mut out = open_started(&mut lead, RATE, level);
    out.write(&frames(1, 4_799, |_| level), None).unwrap();
    wait_until("the song plays", || mixer.timebase().frame == 4_800);
    mixer.crossfade(
        &guest_fader,
        Start::FirstAudio,
        &[&lead_fader],
        frames_of(960),
    );
    // Silent, its fade in waiting: shut, and the song never falls.
    assert!(guest_fader.withdraw());
    assert!(matches!(guest.open(&request(RATE)), Err(Error::Closed)));
    out.write(&frames(4_800, 4_800, |_| level), None).unwrap();
    out.drain(None).unwrap();
    assert!(lead_fader.settled() && lead_fader.gain() == 1.0);
    drop(out);
    drop(mixer);
    assert_eq!(device.underruns(), 0);
    let left = device.left(0);
    assert_eq!(left.len(), 9_600);
    assert_eq!(left.iter().position(|&v| v != level), None);
}

#[test]
fn a_crossfade_under_way_is_not_withdrawn() {
    let (mixer, device) = instant_mixer();
    let strip = mixer.strip("Player");
    let (mut lead, lead_fader) = mixer.input(&strip, Role::Lead, 1.0);
    let (mut guest, guest_fader) = mixer.input(&strip, Role::Guest, 0.0);
    let (a, b) = (1_000_000, 2_000_000);
    let mut out = open_started(&mut lead, RATE, a);
    out.write(&frames(1, 4_799, |_| a), None).unwrap();
    wait_until("the song plays", || mixer.timebase().frame == 4_800);
    let over = 960u64;
    mixer.crossfade(
        &guest_fader,
        Start::FirstAudio,
        &[&lead_fader],
        frames_of(over),
    );
    // A period of the guest mixed: its fade in is under way, half in.
    let mut g = open_started(&mut guest, RATE, b);
    g.write(&frames(1, 479, |_| b), None).unwrap();
    out.write(&frames(4_800, 4_800, |_| a), None).unwrap();
    wait_until("the fade begins", || mixer.timebase().frame == 5_280);
    assert!(!guest_fader.withdraw());
    // Left as it is: the fade runs on to its end.
    let (_out, _g) = play_both(out, g, Vec::new(), frames(480, 4_320, |_| b));
    assert!(lead_fader.settled() && lead_fader.gain() == 0.0);
    assert!(guest_fader.settled() && guest_fader.gain() == 1.0);
    drop(mixer);
    let left = device.left(0);
    for k in 0..over as usize {
        let t = k as f64 / over as f64;
        let (fall, rise) = (equal_power(1.0, 0.0, t), equal_power(0.0, 1.0, t));
        let want = f64::from(a) * fall + f64::from(b) * rise;
        let got = f64::from(left[4_800 + k]);
        assert!((got - want).abs() < 1.5, "{k}: {got} for {want}");
    }
    assert_eq!(
        left[4_800 + over as usize..].iter().position(|&v| v != b),
        None
    );
}

#[test]
fn a_fade_out_on_an_input_with_nothing_to_play_settles() {
    let _guard = real_time();
    let (mixer, _device) = mixer();
    let strip = mixer.strip("Player");
    let (mut target, fader) = mixer.input(&strip, Role::Guest, 1.0);
    // Nothing open at all.
    fader.set(Ramp::fade(0.0, Duration::from_millis(200), Start::Now));
    wait_until("it settles", || fader.settled());
    assert_eq!(fader.gain(), 0.0);
    // An output that ran dry and was closed.
    fader.set(Ramp::set(1.0));
    let mut out = open(&mut target, RATE);
    out.write(&frames(0, 4_800, |_| 1), None).unwrap();
    out.drain(None).unwrap();
    out.close().unwrap();
    fader.set(Ramp::fade(0.0, Duration::from_millis(200), Start::Now));
    wait_until("it settles", || fader.settled());
    assert_eq!(fader.gain(), 0.0);
    // One waiting for audio that never comes does not.
    fader.set(Ramp::fade(
        1.0,
        Duration::from_millis(200),
        Start::FirstAudio,
    ));
    std::thread::sleep(Duration::from_millis(50));
    assert!(!fader.settled());
}

#[test]
fn a_strip_at_0_db_keeps_its_input_bit_for_bit() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let player = mixer.strip("Player");
    let preview = mixer.strip("Preview");
    // Down and back up before anything plays: exactly 0 dB again.
    player.set_level(0.5);
    player.set_level(1.0);
    let (mut lead, _fader) = mixer.input(&player, Role::Lead, 1.0);
    let (_guest, _guest_fader) = mixer.input(&preview, Role::Guest, 0.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    out.write(&frames(0, total, |k| (k as i32) * 97 - 2_000_000), None)
        .unwrap();
    out.drain(None).unwrap();
    let desk = mixer.desk();
    assert!(desk.master.passthrough);
    let names: Vec<&str> = desk.strips.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["Player", "Preview"]);
    assert_eq!(desk.strips[0].level, 1.0);
    let input = &desk.strips[0].inputs[0];
    assert_eq!((input.role, input.gain), (Role::Lead, 1.0));
    assert_eq!(input.format, Some(StreamFormat::new(RATE, 2, FORMAT)));
    assert!(input.capacity > 0);
    assert_eq!(desk.strips[1].inputs[0].format, None);
    drop(out);
    drop(mixer);
    let left = device.left(0);
    let song = lead_in(&left, |k| (k as i32) * 97 - 2_000_000);
    assert!(left[..song].iter().all(|&v| v == 0), "silence before it");
    let wrong = (0..total as usize).find(|&k| left[song + k] != (k as i32) * 97 - 2_000_000);
    assert_eq!(wrong, None);
}

#[test]
fn a_strips_level_is_a_20_ms_linear_ramp() {
    let _guard = real_time();
    let (mixer, device) = mixer();
    let strip = mixer.strip("Player");
    let (mut lead, _fader) = mixer.input(&strip, Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    let level = 1_000_000;
    let clock = out.clock();
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, total, |_| level), None).unwrap();
        out.drain(None).unwrap();
    });
    wait_until("playing", || clock.position() > 9_600);
    let to = 0.501_187_2; // -6 dB
    strip.set_level(to);
    writer.join().unwrap();
    assert_eq!(mixer.desk().strips[0].level, to);
    drop(mixer);
    let left = device.left(0);
    let over = 960; // 20 ms
    let fits = |start: usize| {
        (0..over).all(|k| {
            let want = f64::from(level) * (1.0 + (to - 1.0) * k as f64 / over as f64);
            (f64::from(left[start + k]) - want).abs() < 1.5
        })
    };
    let song = lead_in(&left, |_| level);
    assert!(left[..song].iter().all(|&v| v == 0), "silence before it");
    let changed = song + left[song..].iter().position(|&v| v != level).unwrap();
    let start = [changed - 1, changed].into_iter().find(|&s| fits(s));
    assert!(
        start.is_some(),
        "no ramp from {changed}: {:?}",
        &left[changed - 2..changed + 4]
    );
    let start = start.unwrap();
    assert!(left[song..start].iter().all(|&v| v == level));
    // Its start, middle and end.
    let at = |k: usize| f64::from(left[start + k]) / f64::from(level);
    assert!((at(0) - 1.0).abs() < 2e-6);
    assert!((at(over / 2) - (1.0 + to) / 2.0).abs() < 2e-6);
    assert!((at(over) - to).abs() < 2e-6);
    let held = &left[start + over..song + total as usize];
    assert!(held
        .iter()
        .all(|&v| (f64::from(v) - f64::from(level) * to).abs() < 1.5));
}

#[test]
fn mute_and_solo_silence_strips_without_a_click() {
    let (mixer, device) = instant_mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let (mut a, _fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&sb, Role::Guest, 1.0);
    let (mut oa, mut ob) = (
        open_started(&mut a, RATE, 1_000_000),
        open_started(&mut b, RATE, 2_000_000),
    );
    let step = 4_800;
    let mut written = 1;
    // Up to the next step, both, mixed.
    let mut play = |oa: &mut Box<dyn Output>, ob: &mut Box<dyn Output>| {
        let n = step - written % step;
        oa.write(&frames(0, n, |_| 1_000_000), None).unwrap();
        ob.write(&frames(0, n, |_| 2_000_000), None).unwrap();
        written += n;
        let w = written;
        wait_until("mixed", || mixer.timebase().frame == w);
    };
    play(&mut oa, &mut ob);
    sa.set_mute(true);
    play(&mut oa, &mut ob);
    assert!(mixer.desk().strips[0].mute);
    sa.set_mute(false);
    play(&mut oa, &mut ob);
    sa.set_solo(true);
    play(&mut oa, &mut ob);
    let desk = mixer.desk();
    assert!(desk.strips[0].solo && !desk.strips[1].solo);
    sa.set_solo(false);
    play(&mut oa, &mut ob);
    let outs = play_both(oa, ob, Vec::new(), Vec::new());
    drop(outs);
    drop(mixer);
    let left = device.left(0);
    let levels = [1_000_000, 2_000_000, 3_000_000];
    let (seen, largest) = plateaus(&left, &levels, 3_000_000);
    assert_eq!(
        seen,
        [3_000_000, 2_000_000, 3_000_000, 1_000_000, 3_000_000],
        "both, A muted, both, A soloed, both"
    );
    // Each change from the frame it was set at, over 20 ms.
    let changes = [
        (3_000_000, 2_000_000),
        (2_000_000, 3_000_000),
        (3_000_000, 1_000_000),
        (1_000_000, 3_000_000),
    ];
    for (k, &(from, to)) in changes.iter().enumerate() {
        let at = (k + 1) * step as usize;
        let near = |i: usize, v: i32| left[i].abs_diff(v) <= 2;
        assert!(near(at - 1, from), "{k}: {}", left[at - 1]);
        assert!(near(at + 480, (from + to) / 2), "{k}: {}", left[at + 480]);
        assert!(near(at + 960, to), "{k}: {}", left[at + 960]);
    }
    // Ramped: 2 000 000 over 960 frames is about 2 083 a frame.
    assert!(largest < 2_200, "a step of {largest}");
}

#[test]
fn the_meters_read_each_strip_and_the_sum() {
    let (mixer, _device) = instant_mixer();
    let (sa, sb) = (mixer.strip("A"), mixer.strip("B"));
    let (mut a, _fa) = mixer.input(&sa, Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(&sb, Role::Guest, 1.0);
    let alternating = |k: u64| if k % 2 == 0 { 1_000_000 } else { -500_000 };
    let (oa, ob) = (
        open_started(&mut a, RATE, alternating(0)),
        open_started(&mut b, RATE, 2_000_000),
    );
    let _outs = play_both(
        oa,
        ob,
        frames(1, 23_999, alternating),
        frames(1, 23_999, |_| 2_000_000),
    );
    let desk = mixer.desk();
    let meters = mixer.meters();
    let full = 8_388_608.0f32;
    let ids: Vec<u64> = desk.strips.iter().map(|s| s.id).collect();
    assert_eq!(ids, [sa.id(), sb.id()]);
    assert_eq!(
        meters.strips,
        [
            (sa.id(), vec![1_000_000.0 / full, 0.0]),
            (sb.id(), vec![2_000_000.0 / full, 0.0])
        ]
    );
    let master = &meters.master;
    assert_eq!(master.len(), 2);
    assert!(
        (master[0] - 3_000_000.0 / full).abs() <= 2.0 / full,
        "{master:?}"
    );
    assert!(master[1] <= 1.0 / full, "{master:?}");
    assert!(!meters.limited);
    // Nothing mixed since the last look: nothing to read.
    let again = mixer.meters();
    assert!(again.strips.iter().all(|(_, p)| *p == [0.0, 0.0]));
    assert_eq!(again.master, [0.0, 0.0]);
}
