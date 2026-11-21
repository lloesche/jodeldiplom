//! The mixer in real time, on a simulated device: an input's clock is what
//! the device has played, a ramp lands on its frame, a crossfade's two
//! sides start on the same frame, a starved input leaves the one heard bit
//! for bit, a flush drops what the device held, a microphone's tap on the
//! device is told the Lead's frames, the monitor sees the mix, and only the
//! Lead opens the device again at another format; a closed input never
//! does.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_output::mixer::{Curve, InputTarget, Mixer, MixerOptions, Monitor, Ramp, Role, Start};
use jodel_output::pack::unpack;
use jodel_output::{
    Capabilities, Clock, Error, FileOutput, MemorySink, OpenRequest, Output, OutputTarget,
    SampleFormat, SimClock, StreamFormat, StreamInfo, Tap,
};

const RATE: u32 = 48_000;
const FORMAT: SampleFormat = SampleFormat::S32;

/// A device in real time that keeps what it played: every stream opened,
/// with its clock.
#[derive(Clone, Default)]
struct Device {
    streams: Arc<Mutex<Vec<(MemorySink, Clock)>>>,
}

impl OutputTarget for Device {
    fn name(&self) -> String {
        "test device".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([RATE, 96_000]
            .into_iter()
            .map(|r| StreamFormat::new(r, 2, FORMAT))
            .collect())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let sink = MemorySink::default();
        let caps = self.capabilities()?;
        let out = FileOutput::open(sink.clone(), &caps, request, SimClock::Realtime)?;
        self.streams.lock().unwrap().push((sink, out.clock()));
        Ok(Box::new(out))
    }
}

impl Device {
    /// The left channel of stream `i`.
    fn left(&self, i: usize) -> Vec<i32> {
        let (sink, _) = self.streams.lock().unwrap()[i].clone();
        unpack(&sink.bytes(), FORMAT)
            .into_iter()
            .step_by(2)
            .collect()
    }

    fn underruns(&self) -> u64 {
        let streams = self.streams.lock().unwrap();
        streams.iter().map(|(_, c)| c.stats().underruns).sum()
    }
}

fn mixer() -> (Mixer, Device) {
    let device = Device::default();
    let options = MixerOptions {
        // Room for a busy test machine; the device period is the default.
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    (Mixer::open(Box::new(device.clone()), options), device)
}

fn request(rate: u32) -> OpenRequest {
    OpenRequest::new(rate, 2).formats(&[FORMAT])
}

fn open(target: &mut InputTarget, rate: u32) -> Box<dyn Output> {
    target.open(&request(rate)).unwrap()
}

/// `frames` stereo frames, the left one `f(frame)`, the right zero.
fn frames(from: u64, n: u64, f: impl Fn(u64) -> i32) -> Vec<i32> {
    (from..from + n).flat_map(|k| [f(k), 0]).collect()
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn frames_of(n: u64) -> Duration {
    Duration::from_secs_f64(n as f64 / f64::from(RATE))
}

#[test]
fn an_inputs_position_is_what_the_device_has_played() {
    let (mixer, device) = mixer();
    let (mut lead, _fader) = mixer.input(Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let clock = out.clock();
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, 2 * u64::from(RATE), |_| 1_000), None)
            .unwrap();
        out.drain(None).unwrap();
        out
    });
    wait_until("playing", || mixer.timebase().heard > 4_800);
    let period = u64::from(RATE / 100);
    let mut checked = 0;
    while clock.position() < 3 * u64::from(RATE) / 2 {
        let (position, tb) = (clock.position(), mixer.timebase());
        assert!(
            position.abs_diff(tb.heard) <= period,
            "position {position}, heard {}",
            tb.heard
        );
        // What the mixer took but the device has not played is not heard.
        assert!(tb.frame > position + period);
        assert_eq!((tb.epoch, tb.rate), (1, RATE));
        checked += 1;
        std::thread::sleep(Duration::from_millis(7));
    }
    assert!(checked > 50);
    let out = writer.join().unwrap();
    assert_eq!(out.position().unwrap(), 2 * u64::from(RATE));
    assert_eq!(device.underruns(), 0);
}

#[test]
fn a_ramp_lands_on_its_frame_and_unity_solo_is_bit_exact() {
    let (mixer, device) = mixer();
    let (mut lead, fader) = mixer.input(Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let total = u64::from(RATE);
    let level = 1_000_000;
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, total, |_| level), None).unwrap();
        out.drain(None).unwrap();
    });
    wait_until("playing", || mixer.timebase().frame > 4_800);
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
    // As written, bit for bit, up to the ramp.
    assert!(left[..at].iter().all(|&v| v == level));
    for k in 0..over as usize {
        let want = f64::from(level) * (1.0 - k as f64 / over as f64);
        // Dithered: within the dither's reach.
        assert!((f64::from(left[at + k]) - want).abs() < 1.5, "{k}");
    }
    // Then silence, exactly.
    assert!(left[at + over as usize..total as usize]
        .iter()
        .all(|&v| v == 0));
}

#[test]
fn a_crossfade_starts_both_sides_on_the_first_frame_of_what_comes_in() {
    let (mixer, device) = mixer();
    let (mut lead, lead_fader) = mixer.input(Role::Lead, 1.0);
    let (mut guest, guest_fader) = mixer.input(Role::Guest, 0.0);
    let (a, b) = (1_000_000, 2_000_000);
    let mut out = open(&mut lead, RATE);
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, u64::from(RATE), |_| a), None).unwrap();
        out
    });
    wait_until("the song plays", || mixer.timebase().heard > 4_800);
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
    // very frame, the song's cosine and the guest's sine; then the guest
    // as written. (The first frame of the fade is the song's, dithered:
    // it may read as written.)
    let fits = |start: usize| {
        (0..over as usize).all(|k| {
            let x = std::f64::consts::FRAC_PI_2 * k as f64 / over as f64;
            let want = f64::from(a) * x.cos() + f64::from(b) * x.sin();
            (f64::from(left[start + k]) - want).abs() < 1.5
        })
    };
    let changed = left.iter().position(|&v| v != a).unwrap();
    let start = [changed - 1, changed].into_iter().find(|&s| fits(s));
    assert!(
        start.is_some(),
        "no fade from {changed}: {:?}",
        &left[changed - 2..changed + 4]
    );
    let start = start.unwrap();
    assert!(left[..start].iter().all(|&v| v == a));
    // (The frames of the period the fade ends in, and of the next, go
    // through the mix, dithered.)
    let after = start + over as usize + 1_500;
    assert!(left[after..after + 10_000].iter().all(|&v| v == b));
}

#[test]
fn a_starved_input_leaves_the_one_heard_bit_for_bit() {
    let (mixer, device) = mixer();
    let (mut lead, _lead_fader) = mixer.input(Role::Lead, 1.0);
    let (mut guest, _guest_fader) = mixer.input(Role::Guest, 1.0);
    let mut out = open(&mut lead, RATE);
    let writer = std::thread::spawn(move || {
        out.write(&frames(0, 2 * u64::from(RATE), |k| k as i32), None)
            .unwrap();
        out.drain(None).unwrap();
    });
    wait_until("the song plays", || mixer.timebase().heard > 4_800);
    // A guest that gives its prefill of silence, then nothing: running,
    // starved, at gain 1.
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 4_800, |_| 0), None).unwrap();
    wait_until("the guest is starved", || g.position().unwrap() == 4_800);
    let from = mixer.timebase().frame + 4_800;
    wait_until("mixed past it", || mixer.timebase().frame > from + 24_000);
    assert!(mixer.passthrough());
    writer.join().unwrap();
    drop(g);
    drop(mixer);
    let left = device.left(0);
    let from = from as usize;
    let wrong = (from..from + 24_000).find(|&d| left[d] != d as i32);
    assert_eq!(
        wrong.map(|d| (d, left[d])),
        None,
        "the song as written while the guest is starved ({} underruns)",
        device.underruns()
    );
}

#[test]
fn a_flush_drops_what_the_device_holds_of_it() {
    let (mixer, device) = mixer();
    let (mut lead, _fader) = mixer.input(Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 24_000, |_| 111), None).unwrap();
    wait_until("it plays", || mixer.timebase().heard > 9_600);
    out.flush().unwrap();
    let flushed = mixer.timebase().heard as usize;
    assert_eq!(out.position().unwrap(), out.written());
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
    let (mixer, _device) = mixer();
    let (mut lead, fader) = mixer.input(Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    out.write(&frames(0, 24_000, |_| 1_000), None).unwrap();
    wait_until("it plays", || mixer.timebase().heard > 4_800);
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

/// One buffer a tap saw: the stream frame it was told, how many stream
/// frames, and its first left sample.
type Buffer = (u64, usize, Option<i32>);

/// What a tap saw.
#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<Buffer>>>);

impl Tap for Seen {
    fn begin(&mut self, _: usize, stream_frame: u64, stream_frames: usize, _: StreamFormat) {
        self.0
            .lock()
            .unwrap()
            .push((stream_frame, stream_frames, None));
    }

    fn see(&mut self, samples: &[i32], _: StreamFormat) {
        if let Some(last) = self.0.lock().unwrap().last_mut() {
            last.2 = last.2.or(samples.first().copied());
        }
    }
}

#[test]
fn a_leads_tap_hears_the_device_in_the_leads_frames() {
    let (mixer, _device) = mixer();
    // A guest plays (silent) first, so the device is ahead of the Lead.
    let (mut guest, _guest_fader) = mixer.input(Role::Guest, 0.0);
    let mut g = open(&mut guest, RATE);
    g.write(&frames(0, 9_600, |k| k as i32), None).unwrap();
    wait_until("the guest plays", || mixer.timebase().heard > 9_600);
    let (mut lead, _fader) = mixer.input(Role::Lead, 1.0);
    let mut out = open(&mut lead, RATE);
    let seen = Seen::default();
    out.set_tap(Some(Box::new(seen.clone()))).unwrap();
    out.write(&frames(0, u64::from(RATE), |k| k as i32), None)
        .unwrap();
    out.drain(None).unwrap();
    let seen = seen.0.lock().unwrap();
    let with = seen.iter().filter(|s| s.1 > 0).count();
    assert!(with > 50, "{} buffers with Lead frames", with);
    for &(frame, n, first) in seen.iter() {
        if n > 0 {
            assert_eq!(first, Some(frame as i32), "{seen:?}");
        }
    }
    // A guest cannot tap the device.
    assert!(matches!(g.set_tap(None), Err(Error::Unsupported(_))));
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
    let (mixer, _device) = mixer();
    let watch = Watch::default();
    mixer.set_monitor(Some(Box::new(watch.clone())));
    let (mut a, _fa) = mixer.input(Role::Lead, 1.0);
    let (mut b, _fb) = mixer.input(Role::Guest, 0.5);
    let (mut oa, mut ob) = (open(&mut a, RATE), open(&mut b, RATE));
    oa.pause().unwrap();
    ob.pause().unwrap();
    oa.write(&frames(0, 24_000, |_| 100_000), None).unwrap();
    ob.write(&frames(0, 24_000, |_| 200_000), None).unwrap();
    oa.resume().unwrap();
    ob.resume().unwrap();
    oa.drain(None).unwrap();
    ob.drain(None).unwrap();
    {
        let w = watch.0.lock().unwrap();
        assert_eq!(w.rate, RATE);
        assert!(w.open && !w.out_of_order);
        // Both heard at once: 100k + 0.5 × 200k, dithered.
        let both = w.left.iter().filter(|&&v| (v - 200_000).abs() <= 1).count();
        assert!(both > 20_000, "{both}");
    }
    drop((oa, ob));
    wait_until("the device closes with the last output", || {
        !watch.0.lock().unwrap().open
    });
}

#[test]
fn only_the_lead_opens_the_device_again() {
    let (mixer, _device) = mixer();
    // Nothing open: a guest is offered all, and opens the device.
    let (mut g1, _f1) = mixer.input(Role::Guest, 1.0);
    assert_eq!(g1.capabilities().unwrap().iter().count(), 2);
    let mut o1 = open(&mut g1, RATE);
    o1.write(&frames(0, 100, |_| 0), None).unwrap();
    // A guest made then is offered the device's format, and nothing else
    // opens: the first guest plays on.
    let (mut g2, _f2) = mixer.input(Role::Guest, 1.0);
    let only: Vec<StreamFormat> = g2.capabilities().unwrap().iter().copied().collect();
    assert_eq!(only, [StreamFormat::new(RATE, 2, FORMAT)]);
    assert!(matches!(
        g2.open(&request(96_000)),
        Err(Error::FormatNotAccepted { .. })
    ));
    o1.write(&frames(0, 100, |_| 0), None).unwrap();
    // The Lead at another format: the guests' outputs are closed.
    let (mut lead, _fl) = mixer.input(Role::Lead, 1.0);
    let mut out = lead.open(&request(96_000)).unwrap();
    assert_eq!(out.format().rate, 96_000);
    assert_eq!(o1.write(&frames(0, 100, |_| 0), None), Err(Error::Closed));
    out.write(&frames(0, 100, |_| 0), None).unwrap();
    wait_until("a new epoch", || mixer.timebase().epoch == 2);
}

#[test]
fn a_closed_input_never_takes_the_device_again() {
    let (mixer, _device) = mixer();
    let (mut g1, f1) = mixer.input(Role::Guest, 1.0);
    let mut o1 = open(&mut g1, 96_000);
    o1.write(&frames(0, 100, |_| 0), None).unwrap();
    // Closed while its target is still held: its output and the device
    // close, and it cannot open again.
    f1.close();
    assert_eq!(g1.capabilities().err(), Some(Error::Closed));
    assert_eq!(o1.write(&frames(0, 100, |_| 0), None), Err(Error::Closed));
    assert!(matches!(g1.open(&request(96_000)), Err(Error::Closed)));
    wait_until("the device closes", || mixer.timebase().rate == 0);
    // The next guest opens the device at its own format.
    let (mut g2, _f2) = mixer.input(Role::Guest, 1.0);
    let o2 = open(&mut g2, RATE);
    assert_eq!(o2.format().rate, RATE);
    drop(f1);
    drop(g1);
    wait_until("a new epoch at its rate", || {
        let tb = mixer.timebase();
        (tb.epoch, tb.rate) == (2, RATE)
    });
}

#[test]
fn a_fade_out_on_an_input_with_nothing_to_play_settles() {
    let (mixer, _device) = mixer();
    let (mut target, fader) = mixer.input(Role::Guest, 1.0);
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
