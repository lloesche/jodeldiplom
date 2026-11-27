//! What the mixer thread allocates and frees while the effects input plays:
//! nothing, from the very first sound after the device opens, at each
//! rate a desk runs at (here 48 and 96 kHz). A counting allocator (this
//! binary's alone) counts every call on the thread named `jodel-mixer`
//! while its live synth plays: notes sent on stream frames and on the
//! desk's, waiting, starting (more than its voices), let go, called off,
//! too late, its level glided, its patch edited as it sounds (to another
//! engine too), and the synth replaced as it plays (the one
//! before let go of elsewhere); the simulated device's own memory, which
//! grows as it plays on that thread, is left out. The synth and the
//! effects input are messages (freed and reserved as the mixer thread
//! takes them), the commands are not: they come through a queue of fixed
//! size. The song plays a little before the sounds, so its own stream's
//! buffers are at their sizes.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::mixer_count::{count, QuietDevice, COUNTING, MIXER_ALLOCATIONS};
use common::*;
use jodel_output::mixer::{Mixer, MixerOptions, Role};
use jodel_output::{OpenRequest, OutputTarget};
use jodel_synth::{Command, Commands, Tag};
use jodel_testkit::Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting(count);

/// `n` frames of `channels` channels from stream frame `from`: the first
/// channel the frame's number.
fn wide(from: u64, n: u64, channels: usize) -> Vec<i32> {
    (from..from + n)
        .flat_map(|k| std::iter::once(k as i32).chain(std::iter::repeat_n(0, channels - 1)))
        .collect()
}

/// One counting test at a time (the count is the binary's).
static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn playing_live_allocates_nothing_on_the_mixer_thread() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    for rate in [RATE, 96_000] {
        let device = Device {
            instant: true,
            rate: Some(rate),
            ..Device::default()
        };
        let options = MixerOptions {
            buffer: Duration::from_millis(200),
            ..MixerOptions::default()
        };
        let mixer = Mixer::open(Box::new(QuietDevice(device.clone())), options);
        let player = mixer.strip("Player");
        let strip = mixer.strip("Effects");
        let effects = mixer.effects(&strip);
        let (mut target, fader) = mixer.input(&player, Role::Lead, 1.0);
        let (synth, patch) = tone_synth(rate, 0.01);
        effects.live(synth);
        let request = OpenRequest::new(rate, 2)
            .formats(&[device.format()])
            .prefill(Duration::ZERO);
        let mut out = target.open(&request).unwrap();
        taken(&mut target);
        out.write(&wide(0, 9_600, 2), None).unwrap();
        wait_until("mixed", || mixer.timebase().frame == 9_600);
        MIXER_ALLOCATIONS.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        let release = Command::At {
            frame: 17_000,
            tag: Tag(3),
            cue: jodel_synth::Cue::Release,
        };
        // The patch edited as it sounds (a Synth tab's edit): quieter, then
        // another engine's.
        let quieter = jodel_synth::Patch {
            level: -6.0,
            gate: 0.01,
            ..jodel_synth::Patch::new(jodel_synth::engines::analog::SINE.into())
        };
        let bell = jodel_synth::Patch {
            gate: 0.01,
            ..jodel_synth::Patch::new(jodel_synth::engines::fm::GLASS_BELL.into())
        };
        {
            let mut song = effects.commands(Some(&fader));
            // The first note; then more at once than the synth has voices;
            // then one every 1 000 frames; one far too late; a release.
            song.send(tone(patch, 10_000, 0.0));
            for _ in 0..12 {
                song.send(tone(patch, 13_000, -0.5));
            }
            song.patch(patch, quieter);
            for k in 0..18 {
                if k == 9 {
                    song.patch(patch, bell);
                }
                song.send(tone(patch, 14_000 + 1_000 * k, 0.5));
            }
            song.send(tone(patch, 0, 0.0));
            song.send(release);
            song.send(Command::Tempo { bpm: 97.0 });
            song.send(Command::Level { db: -2.5 });
        }
        let desk = mixer.timebase().frame;
        effects.commands(None).send(tone(patch, desk + 3_000, 0.0));
        out.write(&wide(9_600, 10_560, 2), None).unwrap();
        wait_until("mixed", || mixer.timebase().frame == 20_160);
        // Everything waiting called off; a synth in its place.
        effects
            .commands(Some(&fader))
            .send(Command::Cancel { tag: None });
        let (synth, patch) = tone_synth(rate, 0.01);
        effects.live(synth);
        effects
            .commands(Some(&fader))
            .send(tone(patch, 22_000, 0.0));
        out.write(&wide(20_160, 20_160, 2), None).unwrap();
        wait_until("mixed", || mixer.timebase().frame == 40_320);
        COUNTING.store(false, Ordering::Relaxed);
        assert_eq!(MIXER_ALLOCATIONS.load(Ordering::Relaxed), 0, "{rate}");
        let view = effects.view();
        assert_eq!(view.live, rate);
        assert!(view.commands >= 30, "{rate}: {view:?}");
        assert_eq!(view.missed, 1, "{rate}: the one too late");
        out.drain(None).unwrap();
    }
}

/// In real time with no output open at all, the device playing on
/// (silence): nothing is allocated as the mixer goes on, nor as sounds on
/// the desk's frames are played on it.
#[test]
fn idle_in_real_time_allocates_nothing() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    // In real time on the test's time: a sound sent on the frame mixed
    // next is never late, however busy the machine.
    let device = Device {
        manual: true,
        ..Device::default()
    };
    let mixer = Mixer::open(Box::new(QuietDevice(device.clone())), options);
    let strip = mixer.strip("Effects");
    let effects = mixer.effects(&strip);
    let (synth, patch) = tone_synth(RATE, 0.02);
    effects.live(synth);
    wait_until("open", || mixer.timebase().rate == RATE);
    device.tick(&mixer);
    // A first sound: what it reserves is reserved.
    let soon = || mixer.timebase().frame + 480;
    assert!(effects.commands(None).send(tone(patch, soon(), 0.0)));
    device.tick_until(&mixer, "heard", || effects.view().commands == 1);
    MIXER_ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let from = mixer.timebase().frame;
    for _ in 0..5 {
        assert!(effects.commands(None).send(tone(patch, soon(), 0.0)));
        device.ticks(&mixer, 6);
    }
    device.tick_until(&mixer, "idle a while, every sound heard", || {
        mixer.timebase().frame > from + u64::from(RATE) / 2 && effects.view().commands == 6
    });
    COUNTING.store(false, Ordering::Relaxed);
    assert_eq!(MIXER_ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(effects.view().missed, 0);
}
