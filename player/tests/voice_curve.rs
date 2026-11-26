//! The voice's curve over a song's words, as the mixer plays it: a word
//! marked as not to be sung turns the vocals up over it, smoothly, from
//! the Voice slider's level, and back after it; the curve is the vocals
//! part's automation on the player's lane input (as `desk_wiring` gives
//! it), on a simulated device that takes what it is given at once.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use jodel_lyrics::{Lyrics, Source};
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{Automation, GainCurve, Mixer, MixerOptions, Role};
use jodel_output::pack::unpack;
use jodel_output::{
    Capabilities, Error, FileOutput, MemorySink, OpenRequest, Output, OutputTarget, SampleFormat,
    SimClock, StreamFormat,
};
use jodeldiplom::core::desk::{taper, StripKind};
use jodeldiplom::core::voice_curve::{levels, on_frames, NOT_SUNG_RAISE_DB, SMOOTH};

const RATE: u32 = 48_000;
const FORMAT: SampleFormat = SampleFormat::S32;
/// The voice as written: well under full scale.
const X: i32 = 4_000_000;

/// A device that takes what it is given at once and keeps it.
#[derive(Clone, Default)]
struct Device {
    streams: Arc<Mutex<Vec<MemorySink>>>,
}

impl OutputTarget for Device {
    fn name(&self) -> String {
        "test device".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        Ok([StreamFormat::new(RATE, 2, FORMAT)].into_iter().collect())
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        let sink = MemorySink::default();
        let caps = self.capabilities()?;
        let out = FileOutput::open(sink.clone(), &caps, request, SimClock::Instant)?;
        self.streams.lock().unwrap().push(sink);
        Ok(Box::new(out.into_stream()))
    }
}

/// `n` lane frames: main and the vocals part the voice (`X`), the other
/// parts silent, the parts there.
fn voice_only(n: usize, vocals: usize) -> Vec<i32> {
    let mut v = Vec::with_capacity(n * lane_width(2, 5));
    for _ in 0..n {
        v.extend([X, 0]);
        for k in 0..4 {
            v.extend([if k == vocals { X } else { 0 }, 0]);
        }
        v.push(Parts::Present.word());
    }
    v
}

#[test]
fn the_voice_rises_over_a_word_not_to_be_sung() {
    // "la la la", the second not to be sung, from 0.75 to 1.1 s.
    let mut song = Lyrics::parse_text("la la la\n", Source::Unknown);
    for (k, w) in song.lines[0].words.iter_mut().enumerate() {
        let at = 0.25 + 0.5 * k as f64;
        (w.start, w.end) = (Some(at), Some(at + 0.35));
    }
    song.lines[0].words[1].not_sung = true;
    let voice_db = -12.0;
    let points = on_frames(&levels(&song, voice_db), 0.0, f64::from(RATE));
    let (base, up) = (
        taper::gain(voice_db),
        taper::gain(voice_db + NOT_SUNG_RAISE_DB),
    );

    let device = Device::default();
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(device.clone()), options);
    let player = mixer.strip("Player");
    let strips: Vec<_> = ["Drums", "Bass", "Vocals", "Other"]
        .iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<_> = strips.iter().collect();
    let (mut target, fader, _parts) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let vocals = StripKind::Vocals.lane().unwrap();
    let request = OpenRequest::new(RATE, 2)
        .formats(&[FORMAT])
        .lanes(5)
        .prefill(Duration::ZERO);
    let mut out = target.open(&request).unwrap();
    out.write(&voice_only(1, vocals), None).unwrap();
    let curve = GainCurve::new(&points).unwrap();
    fader
        .automate(Automation::new().part(vocals, curve))
        .unwrap();
    // Once the mixer has taken it, the rest of two seconds.
    target.capabilities().unwrap();
    let frames = 2 * RATE as usize;
    out.write(&voice_only(frames - 1, vocals), None).unwrap();
    out.drain(None).unwrap();
    drop(out);
    drop(mixer);
    let bytes = device.streams.lock().unwrap()[0].bytes();
    let left: Vec<i32> = unpack(&bytes, FORMAT).into_iter().step_by(2).collect();
    assert!(left.len() >= frames, "{}", left.len());

    let at = |t: f64| f64::from(left[(t * f64::from(RATE)) as usize]);
    let close = |got: f64, gain: f64| (got - f64::from(X) * gain).abs() <= 16.0;
    // At the slider's level before it and after it; 6 dB up over it.
    assert!(close(at(0.5), base), "{} for {base}", at(0.5));
    assert!(close(at(0.9), up), "{} for {up}", at(0.9));
    assert!(close(at(1.4), base), "{} for {base}", at(1.4));
    // Up over the smoothing before it, not at once: no step larger than a
    // small share of the rise.
    let from = ((0.75 - 2.0 * SMOOTH) * f64::from(RATE)) as usize;
    let to = ((1.1 + 2.0 * SMOOTH) * f64::from(RATE)) as usize;
    let largest = left[from..to]
        .windows(2)
        .map(|w| (f64::from(w[1]) - f64::from(w[0])).abs())
        .fold(0.0, f64::max);
    let rise = f64::from(X) * (up - base);
    assert!(
        largest < rise / 100.0,
        "a step of {largest} in a rise of {rise}"
    );
}
