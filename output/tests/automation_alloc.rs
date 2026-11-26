//! What the mixer thread allocates and frees while it evaluates automation:
//! nothing. A counting allocator (this binary's alone) counts every call on
//! the thread named `jodel-mixer` while a lane input plays with every one
//! of its curves moving on every frame, so each gain is worked out frame by
//! frame; the simulated device's own memory, which grows as it plays on
//! that thread, is left out.
//!
//! A set replaced (`Fader::automate` again) is freed on the mixer thread,
//! as every message it takes is (the channel's blocks, a crossfade's list,
//! an input forgotten): that is not counted here, only the mixing.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::mixer_count::{count, QuietDevice, COUNTING, MIXER_ALLOCATIONS};
use common::*;
use jodel_output::lanes::{lane_width, Parts};
use jodel_output::mixer::{Automation, GainCurve, Mixer, MixerOptions, Role};
use jodel_output::{OpenRequest, OutputTarget};
use jodel_testkit::Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting(count);

const PARTS: u16 = 4;

/// `n` lane frames: main `x` on the left, each part a quarter, the parts
/// there.
fn lane_frames(n: u64) -> Vec<i32> {
    let x = 4_000_000;
    let mut v = Vec::with_capacity(n as usize * lane_width(2, 1 + PARTS));
    for _ in 0..n {
        v.extend([x, 0]);
        for _ in 0..PARTS {
            v.extend([x / 4, 0]);
        }
        v.push(Parts::Present.word());
    }
    v
}

#[test]
fn evaluating_curves_allocates_nothing_on_the_mixer_thread() {
    let device = Device {
        instant: true,
        ..Device::default()
    };
    let options = MixerOptions {
        buffer: Duration::from_millis(200),
        ..MixerOptions::default()
    };
    let mixer = Mixer::open(Box::new(QuietDevice(device.clone())), options);
    let player = mixer.strip("Player");
    let parts: Vec<_> = ["Drums", "Bass", "Vocals", "Other"]
        .iter()
        .map(|name| mixer.strip_in(&player, name))
        .collect();
    let refs: Vec<_> = parts.iter().collect();
    let (mut target, fader, _parts) = mixer.input_lanes(&player, &refs, Role::Lead, 1.0);
    let request = OpenRequest::new(RATE, 2)
        .formats(&[device.format()])
        .lanes(1 + PARTS)
        .prefill(Duration::ZERO);
    let mut out = target.open(&request).unwrap();
    out.write(&lane_frames(1), None).unwrap();
    // Every curve moving on every frame (one with a cut), so every gain is
    // one a frame.
    let moving = |to: f64| GainCurve::new(&[(0, 1.0), (48_000, to)]).unwrap();
    let cut = GainCurve::new(&[(0, 1.0), (10_000, 0.9), (10_000, 0.5), (48_000, 0.1)]).unwrap();
    let mut auto = Automation::new().input(moving(0.5)).main(cut);
    for k in 0..usize::from(PARTS) {
        auto = auto.part(k, moving(0.1 * k as f64));
    }
    fader.automate(auto).unwrap();
    taken(&mut target);
    // Each gain's buffer grown to a chunk first.
    out.write(&lane_frames(4_799), None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 4_800);
    let data = lane_frames(24_000 - 4_800);
    MIXER_ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    out.write(&data, None).unwrap();
    wait_until("mixed", || mixer.timebase().frame == 24_000);
    COUNTING.store(false, Ordering::Relaxed);
    assert_eq!(MIXER_ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(!mixer.passthrough(), "the curves were mixed");
    out.drain(None).unwrap();
}
