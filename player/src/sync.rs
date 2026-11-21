//! The sync's clicks played and its answers heard ([`crate::core::sync`]
//! says what they mean).
//!
//! The clicks play on an output of their own, a Lead input on the
//! Player's strip of the mixer, opened at the format the device runs at
//! (else at a common one it takes), so nothing reopens; the song's player
//! is stopped first, so nothing else is heard. Being the newest Lead
//! output, the microphone's tap follows it: the voice is told in its
//! stream frames, the frames the clicks are written at. A thread of its
//! own opens it (a device can take a while to answer: the window never
//! waits for it) and writes them: [`sync::CLICKS`] tone bursts of
//! [`CLICK_LENGTH`] at [`CLICK_HZ`], [`CLICK_LEVEL`] below full scale,
//! with raised-cosine edges. The output closes when they are over, or when
//! the sync is called off ([`Clicks`] dropped).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use jodel_output::mixer::{Fader, InputTarget, Mixer, Role, Strip};
use jodel_output::{
    Clock, OpenRequest, Output, OutputTarget, SampleFormat, StreamFormat, VoiceBlock,
};

use crate::core::sync::{self, Onsets, SyncCommand, SyncEvent};
use crate::core::Action;

/// A click's length, in seconds.
pub const CLICK_LENGTH: f64 = 0.015;
/// Its tone, in Hz.
pub const CLICK_HZ: f64 = 1000.0;
/// Its level, in dB below full scale.
pub const CLICK_LEVEL: f64 = 12.0;
/// Its edges, each, in seconds.
const EDGE: f64 = 0.003;
/// Full scale of the engine's samples (24 bits).
const FULL: f64 = 8_388_607.0;
/// How long the writer waits on the output at a time, to see a stop.
const WAIT: Duration = Duration::from_millis(50);

/// The sync's output: what it plays, in the engine's samples (`channels`
/// a frame at `rate`): silence, the clicks at [`sync::click_frames`], and
/// silence after the last to answer it.
pub fn click_track(rate: u32, channels: u16) -> Vec<i32> {
    let r = f64::from(rate);
    let frames = (sync::length() * r).round() as usize;
    let ch = usize::from(channels);
    let mut out = vec![0i32; frames * ch];
    let amplitude = FULL * 10f64.powf(-CLICK_LEVEL / 20.0);
    let length = (CLICK_LENGTH * r).round() as usize;
    let edge = EDGE * r;
    for start in sync::click_frames(rate) {
        for i in 0..length {
            let at = start as usize + i;
            if at >= frames {
                break;
            }
            let x = i as f64;
            let ramp = |n: f64| 0.5 - 0.5 * (std::f64::consts::PI * (n / edge).min(1.0)).cos();
            let envelope = ramp(x).min(ramp(length as f64 - x));
            let v = amplitude * envelope * (std::f64::consts::TAU * CLICK_HZ * x / r).sin();
            out[at * ch..(at + 1) * ch].fill(v.round() as i32);
        }
    }
    out
}

/// What the clicks' output asks of the device: `format` (the device's,
/// while it is open), else a common one it takes (48 or 44.1 kHz, stereo
/// where it can).
fn request(
    format: Option<StreamFormat>,
    target: &mut dyn OutputTarget,
) -> Result<OpenRequest, String> {
    let at = |f: StreamFormat| OpenRequest::new(f.rate, f.channels).formats(&[f.sample]);
    if let Some(format) = format {
        return Ok(at(format));
    }
    let caps = target.capabilities().map_err(|e| e.to_string())?;
    let mut channels = vec![2];
    channels.extend(caps.channel_counts());
    [48_000, 44_100, 96_000, 88_200, 192_000, 176_400]
        .into_iter()
        .flat_map(|rate| channels.iter().map(move |&c| (rate, c)))
        .find_map(|(rate, c)| caps.choose(rate, c, &SampleFormat::PREFERENCE))
        .or_else(|| caps.iter().next().copied())
        .map(at)
        .ok_or_else(|| "the device takes no format".to_owned())
}

/// The clicks' output once open: its clock, its rate, and where each click
/// starts in its frames.
struct Opened {
    clock: Clock,
    rate: u32,
    clicks: Vec<u64>,
}

/// The sync's clicks playing, and the answers heard.
pub struct Clicks {
    /// Set once the output is open.
    opened: Arc<OnceLock<Opened>>,
    fader: Fader,
    stop: Arc<AtomicBool>,
    /// Set once every frame has played (or the output failed: why).
    done: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
    /// Whether the voice is the clicks' yet: the microphone's tap moves to
    /// their output, and its first block there does not follow the one
    /// before; what comes before is still the song's.
    hearing: bool,
    /// The sounds heard in the voice.
    onsets: Onsets,
    /// What was last told, and whether the end was.
    told: Option<(usize, Vec<Option<f64>>)>,
    ended: bool,
}

impl Clicks {
    /// Starts the clicks on a Lead input of `strip` on `mixer`: the output
    /// is opened, and the clicks written, by a thread of their own.
    pub fn start(mixer: &Mixer, strip: &Strip) -> Result<Clicks, String> {
        let (target, fader) = mixer.input(strip, Role::Lead, 1.0);
        let format = mixer.desk().master.stream.map(|s| s.format);
        let opened: Arc<OnceLock<Opened>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let failure: Arc<Mutex<Option<String>>> = Arc::default();
        let (open, stopped, finished, failed) =
            (opened.clone(), stop.clone(), done.clone(), failure.clone());
        let thread = std::thread::Builder::new()
            .name("jodel-sync".into())
            .spawn(move || {
                if let Err(e) = run(target, format, &open, &stopped) {
                    *failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                }
                finished.store(true, Ordering::Release);
            })
            .map_err(|e| e.to_string())?;
        Ok(Clicks {
            opened,
            fader,
            stop,
            done,
            failure,
            thread: Some(thread),
            hearing: false,
            onsets: Onsets::default(),
            told: None,
            ended: false,
        })
    }

    /// The clicks' output's clock, once it is open.
    pub fn clock(&self) -> Option<Clock> {
        self.opened.get().map(|o| o.clock.clone())
    }

    /// The voice heard since the last frame. Until the microphone's tap is
    /// on the clicks' output (its first block there starts a run), and
    /// past what was written to it, a block is the song's: left out.
    pub fn hear(&mut self, voice: &[VoiceBlock]) {
        let Some(o) = self.opened.get() else {
            return;
        };
        let written = o.clock.written();
        for block in voice {
            if !block.continues {
                self.hearing = true;
            }
            if !self.hearing || block.stream_frame > written {
                continue;
            }
            let per = f64::from(o.rate) / f64::from(block.rate.max(1));
            self.onsets.push(
                block.stream_frame,
                per,
                block.rate,
                block.continues,
                &block.samples,
            );
        }
    }

    /// Each click's answer so far.
    pub fn answers(&self) -> Vec<Option<f64>> {
        match self.opened.get() {
            Some(o) => sync::answers(&o.clicks, o.rate, &self.onsets.found),
            None => vec![None; sync::CLICKS],
        }
    }

    /// What the core is to be told now: how far it is, when that changed;
    /// then, once every frame has played, the end (or why it failed).
    pub fn news(&mut self) -> Vec<SyncEvent> {
        let mut out = Vec::new();
        if self.ended {
            return out;
        }
        let played = self.opened.get().map_or(0, |o| {
            let heard = o.clock.position();
            o.clicks.iter().filter(|&&c| c <= heard).count()
        });
        let now = (played, self.answers());
        if self.told.as_ref() != Some(&now) {
            out.push(SyncEvent::Progress {
                played: now.0,
                answers: now.1.clone(),
            });
            self.told = Some(now);
        }
        if self.done.load(Ordering::Acquire) {
            self.ended = true;
            let failure = self
                .failure
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            out.push(failure.map_or(SyncEvent::Ended, SyncEvent::Failed));
        }
        out
    }
}

/// The clicks' thread: opens the output on `target` (at `format`, else
/// at one the device takes), and plays them.
fn run(
    mut target: InputTarget,
    format: Option<StreamFormat>,
    opened: &OnceLock<Opened>,
    stop: &AtomicBool,
) -> Result<(), String> {
    let request = request(format, &mut target)?;
    let mut out = target.open(&request).map_err(|e| e.to_string())?;
    let format = out.format();
    let _ = opened.set(Opened {
        clock: out.clock(),
        rate: format.rate,
        clicks: sync::click_frames(format.rate),
    });
    let track = click_track(format.rate, format.channels);
    let played = play(&mut *out, &track, usize::from(format.channels), stop);
    let _ = out.close();
    played
}

/// Writes `track` (`channels` a frame) to `out` and waits until it has
/// played, a little at a time so a stop is seen.
fn play(
    out: &mut dyn Output,
    track: &[i32],
    channels: usize,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut at = 0;
    while at < track.len() {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let n = out
            .write(&track[at..], Some(WAIT))
            .map_err(|e| e.to_string())?;
        at += n * channels;
    }
    while !stop.load(Ordering::Acquire) {
        if out.drain(Some(WAIT)).map_err(|e| e.to_string())? {
            break;
        }
    }
    Ok(())
}

impl Drop for Clicks {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.fader.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Carries out `command` for the sync: the clicks started on the mixer's
/// `strip` (none without an engine), or stopped. What the core is to be
/// told at once (the clicks could not start).
pub fn command(
    command: SyncCommand,
    engine: Option<(&Mixer, &Strip)>,
    clicks: &mut Option<Clicks>,
) -> Option<Action> {
    *clicks = None;
    if command == SyncCommand::Stop {
        return None;
    }
    let started = match engine {
        Some((mixer, strip)) => Clicks::start(mixer, strip),
        None => Err("no output device".to_owned()),
    };
    match started {
        Ok(c) => *clicks = Some(c),
        Err(why) => return Some(Action::Sync(SyncEvent::Failed(why))),
    }
    None
}

/// A frame of the sync under way: the voice heard since the last, and what
/// the core is to be told; the clicks let go of once over.
pub fn frame(clicks: &mut Option<Clicks>, voice: &[VoiceBlock]) -> Vec<Action> {
    let Some(c) = clicks else {
        return Vec::new();
    };
    c.hear(voice);
    let news = c.news();
    if c.ended {
        *clicks = None;
    }
    news.into_iter().map(Action::Sync).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clicks_are_short_bursts_at_minus_12_db_where_the_core_has_them() {
        let rate = 48_000;
        let track = click_track(rate, 2);
        assert_eq!(
            track.len(),
            (sync::length() * 48_000.0).round() as usize * 2
        );
        let left: Vec<i32> = track.iter().step_by(2).copied().collect();
        let peak = left.iter().map(|v| v.unsigned_abs()).max().unwrap();
        let want = FULL * 10f64.powf(-0.6);
        assert!((f64::from(peak) - want).abs() < want * 0.01, "{peak}");
        for (k, &c) in sync::click_frames(rate).iter().enumerate() {
            let c = c as usize;
            // Silent before, sounding within it, silent 15 ms on.
            assert!(left[c - 100..c].iter().all(|&v| v == 0), "click {k}");
            assert!(left[c..c + 720]
                .iter()
                .any(|&v| v.unsigned_abs() > peak / 2));
            assert!(left[c + 721..c + 2000].iter().all(|&v| v == 0));
        }
        assert_eq!(sync::CLICKS, 8);
    }

    /// A block of voice as the cleaner gives it.
    fn block(frame: u64, continues: bool, loud: bool) -> VoiceBlock {
        VoiceBlock {
            stream_frame: frame,
            rate: 48_000,
            continues,
            samples: (0..128)
                .map(|i| {
                    if loud {
                        0.5 * ((i % 7) as f32 - 3.0) / 3.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        }
    }

    /// The song's last blocks, still queued when the sync starts, are left
    /// out: the voice counts from the first block on the clicks' output,
    /// and none stamped past what was written to it.
    #[test]
    fn the_songs_blocks_left_at_the_start_are_left_out() {
        let (mixer, strip) = {
            let target = jodel_playback::NullTarget::new(
                jodel_playback::any_capabilities(),
                jodel_output::SimClock::Realtime,
            );
            let mixer = Mixer::open(Box::new(target), Default::default());
            let strip = mixer.strip("Player");
            (mixer, strip)
        };
        let mut clicks = Clicks::start(&mixer, &strip).unwrap();
        while clicks.clock().is_none_or(|c| c.written() < 60_000) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let first = sync::click_frames(48_000)[0];
        // A loud song block that goes on from before (its frame happens to
        // be near the first click), then one stamped far past what was
        // written, then silence starting the run on the clicks' output.
        clicks.hear(&[block(first + 4_800, true, true)]);
        clicks.hear(&[block(10_000_000, false, true)]);
        assert!(clicks.onsets.found.is_empty() && clicks.hearing);
        // On the clicks' output now: heard.
        let mut voice = vec![block(first, false, false)];
        for k in 1..60 {
            voice.push(block(first + k * 128, true, (20..40).contains(&k)));
        }
        clicks.hear(&voice);
        assert_eq!(clicks.answers().iter().flatten().count(), 1);
    }
}
