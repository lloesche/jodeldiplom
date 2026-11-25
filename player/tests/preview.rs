//! The preview's players on a mixer (a null device): each file fades in
//! from its first frame on the very frame what was heard fades out, the
//! song runs on under it, only the latest file asked for is loaded, a file
//! that plays to its end makes room, files at another rate than the
//! device's are resampled to it, and the song fades back in at the end.
//!
//! Where frames must line up exactly, the device takes what the mixer
//! gives it no faster than real time but plays only what it is given
//! ([`paced_mixer`]): the mixer waits for every input, so a player put off
//! on a busy machine is late, never a gap. The others run on a device in
//! real time ([`realtime_mixer`]).

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jodel_output::mixer::{equal_power, Mixer, MixerOptions, Monitor, Role};
use jodel_output::{
    Capabilities, Clock, Error, OpenRequest, Output, OutputTarget, SimClock, Stats, StreamInfo, Tap,
};
use jodel_playback::{
    any_capabilities, Command, Event, NullTarget, Player, PlayerOptions, Settings,
};
use jodeldiplom::preview::Previews;

const RATE: u32 = 48_000;
/// The fades' length in frames at [`RATE`].
const FADE: usize = 9_600;

/// A 24-bit stereo WAV at `rate` of `frames`, the left sample of frame `n`
/// `f(n)`.
fn wav(path: &Path, rate: u32, frames: u32, f: impl Fn(u32) -> i32) {
    let data_len = frames * 6;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 6).to_le_bytes());
    out.extend_from_slice(&6u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for n in 0..frames {
        out.extend_from_slice(&f(n).to_le_bytes()[..3]);
        out.extend_from_slice(&[0, 0, 0]);
    }
    std::fs::write(path, out).unwrap();
}

/// The left channel of every frame the device gets, in order.
#[derive(Clone, Default)]
struct Heard(Arc<Mutex<Vec<i32>>>);

impl Monitor for Heard {
    fn opened(&mut self, _: &StreamInfo, _: Clock) {}
    fn mixed(&mut self, samples: &[i32], channels: usize, _: u64) {
        self.0
            .lock()
            .unwrap()
            .extend(samples.iter().step_by(channels));
    }
    fn closed(&mut self) {}
}

impl Heard {
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// The last `n` samples are all within `within` of `v`.
    fn ends_near(&self, n: usize, v: i32, within: i32) -> bool {
        let h = self.0.lock().unwrap();
        h.len() >= n && h[h.len() - n..].iter().all(|&x| (x - v).abs() <= within)
    }

    /// How many samples from `from` on are `v`.
    fn count(&self, from: usize, v: i32) -> usize {
        self.0.lock().unwrap()[from..]
            .iter()
            .filter(|&&x| x == v)
            .count()
    }
}

/// A null device that plays only what it is given, taking it no faster
/// than real time (after a first 100 ms): to the mixer it is a device that
/// waits for its inputs, to the players one that runs about as a device
/// does.
struct Paced(NullTarget);

impl OutputTarget for Paced {
    fn name(&self) -> String {
        "paced null output".into()
    }

    fn capabilities(&mut self) -> Result<Capabilities, Error> {
        self.0.capabilities()
    }

    fn open(&mut self, request: &OpenRequest) -> Result<Box<dyn Output>, Error> {
        Ok(Box::new(PacedOutput {
            inner: self.0.open(request)?,
            start: None,
        }))
    }
}

struct PacedOutput {
    inner: Box<dyn Output>,
    /// When the first frame was taken.
    start: Option<Instant>,
}

impl Output for PacedOutput {
    fn info(&self) -> &StreamInfo {
        self.inner.info()
    }
    fn write(&mut self, samples: &[i32], timeout: Option<Duration>) -> Result<usize, Error> {
        let format = self.inner.format();
        let ch = usize::from(format.channels);
        let frames = (samples.len() / ch) as u64;
        let deadline = timeout.map(|t| Instant::now() + t);
        let start = *self.start.get_or_insert_with(Instant::now);
        loop {
            let ahead = u64::from(format.rate) / 10;
            let due = ahead + (start.elapsed().as_secs_f64() * f64::from(format.rate)) as u64;
            let room = due.saturating_sub(self.inner.written()).min(frames);
            if room > 0 {
                return self.inner.write(&samples[..room as usize * ch], timeout);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(0);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    fn drain(&mut self, timeout: Option<Duration>) -> Result<bool, Error> {
        self.inner.drain(timeout)
    }
    fn position(&self) -> Result<u64, Error> {
        self.inner.position()
    }
    fn written(&self) -> u64 {
        self.inner.written()
    }
    fn latency(&self) -> Duration {
        self.inner.latency()
    }
    fn pause(&mut self) -> Result<(), Error> {
        self.inner.pause()
    }
    fn resume(&mut self) -> Result<(), Error> {
        self.inner.resume()
    }
    fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.inner.flush()
    }
    fn set_tap(&mut self, tap: Option<Box<dyn Tap>>) -> Result<(), Error> {
        self.inner.set_tap(tap)
    }
    fn stats(&self) -> Stats {
        self.inner.stats()
    }
    fn clock(&self) -> Clock {
        self.inner.clock()
    }
    fn close(&mut self) -> Result<(), Error> {
        self.inner.close()
    }
}

/// A mixer on a null device in real time that runs at `rate` (as a shared
/// device runs at the system's): a period an input is short of is mixed
/// without it, as on a device.
fn realtime_mixer(rate: u32) -> (Mixer, Heard) {
    let device = NullTarget::new(any_capabilities(rate), SimClock::Realtime);
    mixer_on(Box::new(device))
}

/// A mixer on a [`Paced`] device at 48 kHz: every period is mixed once
/// each running input holds it.
fn paced_mixer() -> (Mixer, Heard) {
    let device = Paced(NullTarget::new(any_capabilities(48_000), SimClock::Instant));
    mixer_on(Box::new(device))
}

fn mixer_on(device: Box<dyn OutputTarget>) -> (Mixer, Heard) {
    let mixer = Mixer::open(
        device,
        MixerOptions {
            buffer: Duration::from_millis(200),
            ..MixerOptions::default()
        },
    );
    let heard = Heard::default();
    mixer.set_monitor(Some(Box::new(heard.clone())));
    (mixer, heard)
}

/// The first frame of `h[range]` that is not `v`, and what is around it.
fn first_not(h: &[i32], range: std::ops::Range<usize>, v: i32) -> Option<String> {
    let at = range.clone().find(|&i| h[i] != v)?;
    let around = &h[at.saturating_sub(3)..(at + 4).min(h.len())];
    let zeros = h[range].iter().filter(|&&x| x == 0).count();
    Some(format!(
        "frame {at} is {} (around it {around:?}; {zeros} frames of silence)",
        h[at]
    ))
}

fn wait(
    what: &str,
    mixer: &Mixer,
    previews: &mut Previews,
    mut done: impl FnMut(&Previews) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done(previews) {
        assert!(Instant::now() < deadline, "timed out: {what}");
        let said = previews.pump(mixer);
        assert!(said.is_empty(), "{said:?}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_preview_crossfades_on_one_frame_over_the_song_running_on() {
    let dir = common::temp_dir("preview");
    let (song, ramp, two, three) = (
        dir.join("song.wav"),
        dir.join("ramp.wav"),
        dir.join("two.wav"),
        dir.join("three.wav"),
    );
    let level = 1_000_000;
    wav(&song, RATE, 20 * RATE, |_| level);
    let ramp_frames = 3 * RATE;
    wav(&ramp, RATE, ramp_frames, |n| n as i32);
    wav(&two, RATE, 3 * RATE, |_| 2_000_000);
    wav(&three, RATE, 3 * RATE, |_| 3_000_000);

    let (mixer, heard) = paced_mixer();
    let (target, lead) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let (player, events) = Player::new(Box::new(target), PlayerOptions::default());
    player.load(vec![song.clone()]).unwrap();
    player.play(0).unwrap();
    let mut previews = Previews::new(
        vec![lead.clone()],
        mixer.strip("Preview"),
        Settings::default(),
    );
    wait("the song plays", &mixer, &mut previews, |_| {
        heard.len() > 4_800
    });

    // The first file: from a third in, its fade in and the song's fade out
    // on the same frames.
    let asked = heard.len();
    previews.play(&mixer, ramp.clone());
    wait("the song is down", &mixer, &mut previews, |_| {
        lead.settled() && lead.gain() == 0.0
    });
    wait("the ramp is heard", &mixer, &mut previews, |_| {
        heard.len() > asked + 3 * FADE
    });
    {
        let h = heard.0.lock().unwrap();
        let from = (f64::from(ramp_frames) / 3.0).round() as i32;
        // Heard as it is now (the song all down): the file's frame says
        // where it started; the two fades must start there, to the frame.
        let last = h.len() - 1;
        let start = last - (h[last] - from) as usize;
        // No sooner than asked for (on a device that waits for its inputs,
        // as soon as the next period).
        assert!(
            start >= asked,
            "the file from {start}, asked for at {asked}"
        );
        // The song as it is up to there (the fade's first frame is the
        // song's, dithered: it may read as it is).
        assert_eq!(
            first_not(&h, asked..(start - 1).max(asked), level),
            None,
            "the song before the file's first frame {start} (asked for at {asked})"
        );
        for k in 0..FADE {
            let t = k as f64 / FADE as f64;
            let (fall, rise) = (equal_power(1.0, 0.0, t), equal_power(0.0, 1.0, t));
            let want = f64::from(level) * fall + f64::from(from + k as i32) * rise;
            assert!(
                (f64::from(h[start + k]) - want).abs() < 1.5,
                "{k}: {} for {want}",
                h[start + k]
            );
        }
    }

    // Two more asked for at once: only the latest is loaded and heard.
    let moved = heard.len();
    previews.play(&mixer, two);
    previews.play(&mixer, three);
    wait("the latest is heard", &mixer, &mut previews, |_| {
        heard.ends_near(2 * FADE, 3_000_000, 0)
    });
    assert_eq!(heard.count(moved, 2_000_000), 0);

    // Over: the song fades back in, and played on all the while.
    previews.end();
    wait("the players are gone", &mixer, &mut previews, |p| p.done());
    wait("the song is back", &mixer, &mut previews, |_| {
        lead.settled() && lead.gain() == 1.0
    });
    wait("the song as it is", &mixer, &mut previews, |_| {
        heard.ends_near(FADE, level, 0)
    });
    let mut positions = Vec::new();
    let mut queue_changes = 0;
    while let Ok(event) = events.try_recv() {
        match event {
            Event::Position(p) => positions.push(p.seconds),
            Event::QueueChanged { .. } => queue_changes += 1,
            _ => {}
        }
    }
    assert_eq!(queue_changes, 1, "the song's queue was loaded once only");
    assert!(positions.windows(2).all(|w| w[1] >= w[0]));
    let ran = positions.last().copied().unwrap_or(0.0);
    assert!(ran > 0.5, "the song ran on: {ran} s");
    player.send(Command::Stop).unwrap();
    drop(player);
    drop(mixer);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_file_played_to_its_end_makes_room_and_another_rate_is_resampled() {
    let _guard = common::real_time();
    let dir = common::temp_dir("preview-end");
    let (short, one, other) = (
        dir.join("short.wav"),
        dir.join("one.wav"),
        dir.join("other.wav"),
    );
    // The device runs at 44.1 kHz, as the files.
    wav(&short, 44_100, 44_100, |_| 500_000);
    wav(&one, 44_100, 3 * 44_100, |_| 1_000_000);
    wav(&other, 96_000, 3 * 96_000, |_| 3_000_000);
    let (mixer, heard) = realtime_mixer(44_100);
    let (_target, lead) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let mut previews = Previews::new(vec![lead], mixer.strip("Preview"), Settings::default());
    // A file heard to its end (from a third in: 0.67 s): its player goes.
    previews.play(&mixer, short);
    wait("it is heard", &mixer, &mut previews, |_| {
        heard.ends_near(1_000, 500_000, 0)
    });
    assert_eq!(mixer.timebase().rate, 44_100);
    wait(
        "it played out: its player went",
        &mixer,
        &mut previews,
        |p| p.players() == 0,
    );
    // And for 100 ms after, nothing more happens: nothing said, no player
    // back, the device playing on.
    for _ in 0..50 {
        assert!(previews.pump(&mixer).is_empty());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(previews.players(), 0, "its player was let go of");
    assert_eq!(mixer.timebase().rate, 44_100);
    // So two more files each have a place: the second, at 96 kHz, comes
    // in over the first, resampled to the device's 44.1 kHz.
    previews.play(&mixer, one);
    wait("the first is heard", &mixer, &mut previews, |_| {
        heard.ends_near(1_000, 1_000_000, 0)
    });
    let epoch = mixer.timebase().epoch;
    previews.play(&mixer, other);
    wait("the second is heard", &mixer, &mut previews, |_| {
        heard.ends_near(4_410, 3_000_000, 4)
    });
    let tb = mixer.timebase();
    assert_eq!(
        (tb.epoch, tb.rate),
        (epoch, 44_100),
        "the device was not opened again"
    );
    previews.end();
    wait("the players are gone", &mixer, &mut previews, |p| p.done());
    drop(mixer);
    let _ = std::fs::remove_dir_all(dir);
}

/// Where `h[start..]` is not `from` falling and `to` rising, equal-power,
/// over a fade, from frame `start` on: the first frame that is not, its
/// value and the one wanted (`None`: all are).
fn crossfade_off(h: &[i32], start: usize, from: i32, to: i32) -> Option<(usize, i32, f64)> {
    (0..FADE).find_map(|k| {
        let t = k as f64 / FADE as f64;
        let (fall, rise) = (equal_power(1.0, 0.0, t), equal_power(0.0, 1.0, t));
        let want = f64::from(from) * fall + f64::from(to) * rise;
        let got = h.get(start + k).copied().unwrap_or(i32::MIN);
        ((f64::from(got) - want).abs() >= 1.5).then_some((start + k, got, want))
    })
}

#[test]
fn every_move_is_a_crossfade_and_the_players_crossfaded_out_go() {
    let dir = common::temp_dir("preview-moves");
    let song = dir.join("song.wav");
    let level = 1_000_000;
    wav(&song, RATE, 30 * RATE, |_| level);
    let values = [1_500_000, 2_000_000, 2_500_000, 3_000_000, 3_500_000];
    let files: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let path = dir.join(format!("{i}.wav"));
            wav(&path, RATE, 6 * RATE, move |_| v);
            path
        })
        .collect();
    let (mixer, heard) = paced_mixer();
    let (target, lead) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let (player, _events) = Player::new(Box::new(target), PlayerOptions::default());
    player.load(vec![song]).unwrap();
    player.play(0).unwrap();
    let mut previews = Previews::new(vec![lead], mixer.strip("Preview"), Settings::default());
    wait("the song plays", &mixer, &mut previews, |_| {
        heard.ends_near(4_800, level, 0)
    });
    // Five moves in a row, each once the last is heard.
    let mut moves = Vec::new();
    for (path, &v) in files.iter().zip(&values) {
        moves.push(heard.len());
        previews.play(&mixer, path.clone());
        assert!(previews.players() <= 2);
        wait("the file is heard alone", &mixer, &mut previews, |_| {
            heard.ends_near(2 * FADE, v, 0)
        });
        wait("the one before is let go of", &mixer, &mut previews, |p| {
            p.players() == 1
        });
    }
    {
        let h = heard.0.lock().unwrap();
        let mut from = level;
        for (&asked, &to) in moves.iter().zip(&values) {
            // As it was up to the frame the next file's first audio is;
            // from that frame the one falls as the other rises: no gap.
            let changed = asked + h[asked..].iter().position(|&x| x != from).unwrap();
            let offs = [changed - 1, changed].map(|s| crossfade_off(&h, s, from, to));
            let start = [changed - 1, changed]
                .into_iter()
                .zip(offs)
                .find_map(|(s, off)| off.is_none().then_some(s));
            let around = &h[changed.saturating_sub(3)..(changed + 4).min(h.len())];
            let zeros = h[asked..].iter().filter(|&&x| x == 0).count();
            assert!(
                start.is_some(),
                "no crossfade {from} → {to} at {changed} (asked for at {asked}; around it \
                 {around:?}; off (frame, value, wanted) from either frame {offs:?}; {zeros} \
                 frames of silence since)"
            );
            assert_eq!(first_not(&h, asked..start.unwrap().max(asked), from), None);
            from = to;
        }
    }
    // A fast run of moves: only the last is loaded; what is heard comes
    // down to it in one crossfade, never through silence.
    let burst = heard.len();
    for i in [2, 3, 4, 1] {
        previews.play(&mixer, files[i].clone());
        assert!(previews.players() <= 2);
        for _ in 0..3 {
            assert!(previews.pump(&mixer).is_empty());
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    wait("the last is heard alone", &mixer, &mut previews, |_| {
        heard.ends_near(2 * FADE, values[1], 0)
    });
    wait("the others are let go of", &mixer, &mut previews, |p| {
        p.players() == 1
    });
    {
        let h = heard.0.lock().unwrap();
        let low = h[burst..].iter().min().copied().unwrap();
        let at = burst + h[burst..].iter().position(|&x| x == low).unwrap();
        let zeros = h[burst..].iter().filter(|&&x| x == 0).count();
        let levels: Vec<i32> = (burst..h.len()).step_by(480).map(|i| h[i]).collect();
        assert!(
            low >= values[1] - 2,
            "a dip to {low} at {at} (from {burst}; {zeros} frames of silence; every 480th \
             frame from there: {levels:?})"
        );
    }
    previews.end();
    wait("the players are gone", &mixer, &mut previews, |p| p.done());
    player.send(Command::Stop).unwrap();
    drop(player);
    drop(mixer);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn quick_moves_across_rates_with_nothing_playing_fail_nothing() {
    let _guard = common::real_time();
    let dir = common::temp_dir("preview-rates");
    let specs = [
        (44_100, 1_000_000),
        (96_000, 2_000_000),
        (48_000, 1_500_000),
        (96_000, 3_000_000),
    ];
    let files: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(i, &(rate, v))| {
            let path = dir.join(format!("{i}.wav"));
            wav(&path, rate, 3 * rate, move |_| v);
            path
        })
        .collect();
    let (mixer, heard) = realtime_mixer(48_000);
    let (_target, lead) = mixer.input(&mixer.strip("Player"), Role::Lead, 1.0);
    let mut previews = Previews::new(vec![lead], mixer.strip("Preview"), Settings::default());
    // Moved on before anything has opened the device (a player opens a
    // WAV file's output within a millisecond or two: no pause between).
    for path in &files[..3] {
        previews.play(&mixer, path.clone());
        assert!(previews.pump(&mixer).is_empty());
    }
    wait("the last is heard", &mixer, &mut previews, |_| {
        heard.ends_near(2 * FADE, 1_500_000, 0)
    });
    let tb = mixer.timebase();
    assert_eq!(tb.rate, 48_000);
    // Then another rate: resampled to the device, which stays as it is.
    previews.play(&mixer, files[3].clone());
    wait("the next is heard", &mixer, &mut previews, |_| {
        heard.ends_near(4_800, 3_000_000, 4)
    });
    let now = mixer.timebase();
    assert_eq!((now.epoch, now.rate), (tb.epoch, 48_000));
    previews.end();
    wait("the players are gone", &mixer, &mut previews, |p| p.done());
    drop(mixer);
    let _ = std::fs::remove_dir_all(dir);
}

/// The song is both heads of its player (a transition's next track plays
/// on the second): the preview ducks both, so neither is heard under it,
/// and brings both back at its end.
#[test]
fn a_preview_ducks_both_heads_of_the_song() {
    let dir = common::temp_dir("preview-heads");
    let (a, b, file) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("file.wav"));
    let (level_a, level_b, level) = (1_000_000, 500_000, 2_000_000);
    wav(&a, RATE, 30 * RATE, |_| level_a);
    wav(&b, RATE, 30 * RATE, |_| level_b);
    wav(&file, RATE, 6 * RATE, |_| level);
    let (mixer, heard) = paced_mixer();
    let (_desk, lead, guest) = jodel_playback::StemDesk::two_heads(&mixer, None, "Player", 1.0);
    let song = vec![lead.faders.input.clone(), guest.faders.input.clone()];
    // Each head played on by a player of its own here, as the engine plays
    // a transition's two tracks on them: the Lead's first, which opens the
    // device (the guest's is offered its format).
    let mut previews = Previews::new(song.clone(), mixer.strip("Preview"), Settings::default());
    let mut players = Vec::new();
    for (head, path) in [(lead, &a), (guest, &b)] {
        let (player, _events) = Player::new(head.target, PlayerOptions::default());
        player.load(vec![path.clone()]).unwrap();
        player.play(0).unwrap();
        players.push(player);
        let heard_before = heard.len();
        wait("the head plays", &mixer, &mut previews, |_| {
            heard.len() > heard_before + 4_800
        });
    }
    wait("both heads play", &mixer, &mut previews, |_| {
        heard.ends_near(4_800, level_a + level_b, 1)
    });
    previews.play(&mixer, file);
    wait("both heads are down", &mixer, &mut previews, |_| {
        song.iter().all(|f| f.settled() && f.gain() == 0.0)
    });
    // The file alone: neither head is heard under it.
    wait("the file is heard alone", &mixer, &mut previews, |_| {
        heard.ends_near(2 * FADE, level, 1)
    });
    previews.end();
    wait("the players are gone", &mixer, &mut previews, |p| p.done());
    wait("both heads are back", &mixer, &mut previews, |_| {
        song.iter().all(|f| f.settled() && f.gain() == 1.0)
    });
    wait("the song as it was", &mixer, &mut previews, |_| {
        heard.ends_near(FADE, level_a + level_b, 1)
    });
    for player in &players {
        player.send(Command::Stop).unwrap();
    }
    drop(players);
    drop(mixer);
    let _ = std::fs::remove_dir_all(dir);
}
