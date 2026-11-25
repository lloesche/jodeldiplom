//! The clock bridge across the microphone's and the device's clocks: a
//! varispeed resampler (cubic) from the capture rate to the output's, and
//! the loop that steers its ratio to hold the ring's fill.

use super::capture::MicShared;
use super::MAX_ADJUST;
use crate::spsc;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The loop's proportional gain, per second of fill error.
const KP: f64 = 0.35;
/// The loop's integral gain, per second² (with `KP`: 0.25 rad/s, damping
/// 0.7, so a clock 0.1 % off settles in about 20 s, within 2 ms of the
/// target on the way).
const KI: f64 = 0.0625;
/// The time constant of the fill's smoothing, in seconds: well inside the
/// loop, and long enough to quiet the jitter of when threads wake.
const SMOOTHING: f64 = 0.5;
/// The most input samples one output sample may need (a ratio of 16).
pub(super) const MAX_RATIO: usize = 16;
/// How long both sides run before the voice starts: devices deliver and
/// take in bursts as they start.
const WARM_UP: Duration = Duration::from_millis(100);

/// The varispeed resampler and the loop that steers it.
pub(super) struct Steer {
    /// The rates it was set up for (capture, output); 0 before the first
    /// buffer.
    rates: (u32, u32),
    nominal: f64,
    adjust: f64,
    integral: f64,
    /// The smoothed fill, in capture frames; `None` until measured.
    smoothed: Option<f64>,
    /// Running (else waiting for the ring to fill to its target).
    primed: bool,
    /// Output frames seen since the rates were set, up to the warm-up.
    seen: u64,
    /// The input position of the next output sample, in 32.32 fixed
    /// point, between `history[1]` and `history[2]`.
    position: u64,
    step: u64,
    /// The four input samples around the position, oldest first.
    history: [f32; 4],
}

impl Steer {
    pub(super) fn new() -> Steer {
        Steer {
            rates: (0, 0),
            nominal: 1.0,
            adjust: 0.0,
            integral: 0.0,
            smoothed: None,
            primed: false,
            seen: 0,
            position: 0,
            step: 1 << 32,
            history: [0.0; 4],
        }
    }

    fn set_step(&mut self) {
        let ratio = self.nominal * (1.0 + self.adjust);
        self.step = (ratio * 4_294_967_296.0).round() as u64;
    }

    /// Capture samples `frames` output frames consume from here.
    fn need(&self, frames: usize) -> usize {
        ((self.position + frames as u64 * self.step) >> 32) as usize
    }

    /// Before a buffer of `frames` at `out_rate`: steers the ratio on the
    /// fill this buffer will leave, and says whether the ring holds the
    /// whole buffer's voice (else the buffer gets none). Whether it reads
    /// the ring now (primed) is told to the capture, which counts what it
    /// cannot queue as lost only then.
    pub(super) fn begin(
        &mut self,
        shared: &MicShared,
        ring: &mut spsc::Consumer<f32>,
        out_rate: u32,
        frames: usize,
    ) -> bool {
        let live = self.steer_buffer(shared, ring, out_rate, frames);
        shared.bridge.reading.store(self.primed, Ordering::Release);
        live
    }

    /// [`Steer::begin`] but for telling whether it reads.
    fn steer_buffer(
        &mut self,
        shared: &MicShared,
        ring: &mut spsc::Consumer<f32>,
        out_rate: u32,
        frames: usize,
    ) -> bool {
        let in_rate = shared.capture.rate.load(Ordering::Acquire);
        if in_rate == 0 || out_rate == 0 {
            return false;
        }
        if self.rates != (in_rate, out_rate) {
            *self = Steer::new();
            self.rates = (in_rate, out_rate);
            self.nominal = f64::from(in_rate) / f64::from(out_rate);
            self.set_step();
        }
        if self.nominal * (1.0 + 2.0 * MAX_ADJUST) >= MAX_RATIO as f64 {
            return false;
        }
        let packet = shared.capture.packet.load(Ordering::Relaxed);
        let period = shared
            .capture
            .period
            .load(Ordering::Relaxed)
            .max(packet)
            .max(1);
        let target = 2 * period as usize;
        shared.bridge.target.store(target as u32, Ordering::Relaxed);
        let queued = ring.len();
        let arriving = self.arriving(shared, packet);
        // What to keep queued for the fill to be at its target after this
        // buffer.
        let keep = (target + self.need(frames)) as f64 - arriving;
        let keep = (keep.round().max(0.0) as usize).max(self.need(frames));
        if !self.primed {
            let warm = |rate: u32| rate as u64 * WARM_UP.as_millis() as u64 / 1000;
            let first_at = shared.capture.first_at.load(Ordering::Relaxed);
            let captured = if first_at == 0 {
                u64::MAX
            } else {
                shared.now_ns().saturating_sub(first_at - 1)
            };
            let output_warm = self.seen >= warm(out_rate);
            self.seen = (self.seen + frames as u64).min(warm(out_rate));
            if !output_warm || captured < WARM_UP.as_nanos() as u64 || queued < keep {
                return false;
            }
            ring.skip(queued - keep);
            self.primed = true;
            self.smoothed = None;
            return true;
        }
        let need = self.need(frames);
        if queued < need {
            // The capture stalled: no voice until the target is queued
            // again. The loop holds its estimate of the clocks meanwhile.
            shared.bridge.underruns.fetch_add(1, Ordering::Relaxed);
            self.primed = false;
            self.history = [0.0; 4];
            self.position = 0;
            return false;
        }
        if queued > 4 * target + 2 * need {
            // Far too much queued (the device thread stalled): skip to the
            // target.
            ring.skip(queued - keep);
            shared.bridge.resyncs.fetch_add(1, Ordering::Relaxed);
            self.smoothed = None;
        } else {
            let fill = queued as f64 - need as f64 + arriving;
            self.steer(shared, frames, target, fill);
            // Held well over the target (a burst of capture, as devices
            // deliver when they start): skip the excess rather than drain
            // it at the loop's limit for seconds.
            let over = self.smoothed.unwrap_or(fill) - target as f64;
            let excess = (fill - target as f64).round();
            if over > f64::from(period) && excess >= 1.0 {
                ring.skip(excess as usize);
                self.smoothed = Some(fill - excess);
                shared.bridge.resyncs.fetch_add(1, Ordering::Relaxed);
            }
        }
        // A faster ratio may need a sample more than is queued.
        ring.len() >= self.need(frames)
    }

    /// Fills `voice` with the capture resampled; the buffer's
    /// [`Steer::begin`] made sure the ring holds it.
    pub(super) fn render(
        &mut self,
        ring: &mut spsc::Consumer<f32>,
        voice: &mut [f32],
        taken: &mut [f32],
    ) {
        let need = self.need(voice.len()).min(taken.len());
        let got = ring.pop_slice(&mut taken[..need]);
        let mut next = taken[..got].iter();
        for out in voice.iter_mut() {
            let t = (self.position & 0xFFFF_FFFF) as f32 / 4_294_967_296.0;
            *out = catmull_rom(&self.history, t);
            self.position = (self.position & 0xFFFF_FFFF) + self.step;
            for _ in 0..(self.position >> 32) {
                let x = next.next().copied().unwrap_or(0.0);
                self.history = [self.history[1], self.history[2], self.history[3], x];
            }
        }
        self.position &= 0xFFFF_FFFF;
    }

    /// What to add to the queue's length for the fill as if the voice
    /// arrived continuously: what has come in since the last packet, minus
    /// half a packet (so the fill means what is queued on average).
    fn arriving(&self, shared: &MicShared, packet: u32) -> f64 {
        let pushed_at = shared.capture.pushed_at.load(Ordering::Acquire);
        let since = if pushed_at == 0 {
            0.0
        } else {
            let ns = shared.now_ns().saturating_sub(pushed_at - 1);
            (ns as f64 * 1e-9 * f64::from(self.rates.0)).min(f64::from(packet))
        };
        since - f64::from(packet) / 2.0
    }

    /// Moves the ratio on `fill`, what a buffer of `frames` will leave
    /// queued (as [`Steer::arriving`] counts it).
    fn steer(&mut self, shared: &MicShared, frames: usize, target: usize, fill: f64) {
        let (in_rate, out_rate) = self.rates;
        let fin = f64::from(in_rate);
        let dt = frames as f64 / f64::from(out_rate);
        let smoothed = match self.smoothed {
            None => fill,
            Some(s) => s + (fill - s) * (dt / SMOOTHING).min(1.0),
        };
        self.smoothed = Some(smoothed);
        let error = (smoothed - target as f64) / fin;
        // The integral is the estimate of the clocks' offset: it does not
        // wind up while the ratio is held at its limit.
        let integral = self.integral + error * dt;
        let wanted = KP * error + KI * integral;
        if wanted.abs() <= MAX_ADJUST || (wanted > 0.0) != (error > 0.0) {
            self.integral = integral;
        }
        self.adjust = (KP * error + KI * self.integral).clamp(-MAX_ADJUST, MAX_ADJUST);
        self.set_step();
        shared
            .bridge
            .fill
            .store(smoothed.to_bits(), Ordering::Relaxed);
        shared
            .bridge
            .adjust
            .store(self.adjust.to_bits(), Ordering::Relaxed);
    }
}

/// Catmull-Rom between `h[1]` and `h[2]` at `t` in [0, 1).
fn catmull_rom(h: &[f32; 4], t: f32) -> f32 {
    let [a, b, c, d] = *h;
    let c1 = 0.5 * (c - a);
    let c2 = a - 2.5 * b + 2.0 * c - 0.5 * d;
    let c3 = 0.5 * (d - a) + 1.5 * (b - c);
    ((c3 * t + c2) * t + c1) * t + b
}

#[cfg(test)]
mod tests {
    use super::super::capture::{MicClock, Opened, Wire};
    use super::super::Mono;
    use super::*;
    use crate::input::Producer;

    /// Queues `samples` captured at time 0, then an empty packet at 1 s:
    /// the capture is past its warm-up, and nothing is arriving.
    fn queued_a_second_ago(capture: &mut Producer, samples: &[f32]) {
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        capture.push(&bytes, Wire::F32, 1, samples.len());
        capture.shared().set_time(Duration::from_secs(1));
        capture.push(&[], Wire::F32, 1, 0);
    }

    #[test]
    fn catmull_rom_is_exact_on_its_points_and_on_quadratics() {
        let f = |x: f32| 0.1 * x * x * x - 0.3 * x * x + 0.2 * x + 0.05;
        let h = [f(-1.0), f(0.0), f(1.0), f(2.0)];
        assert_eq!(catmull_rom(&h, 0.0), h[1]);
        // Catmull-Rom reproduces quadratics exactly.
        let q = |x: f32| 0.4 * x * x - 0.2 * x + 0.1;
        let hq = [q(-1.0), q(0.0), q(1.0), q(2.0)];
        for t in [0.25f32, 0.5, 0.75] {
            assert!((catmull_rom(&hq, t) - q(t)).abs() < 1e-6);
        }
    }

    #[test]
    fn same_rates_pass_the_voice_through_unchanged() {
        let (mut capture, opened) = Opened::new(48_000, 240, Mono::Auto, MicClock::caller());
        let shared = &*opened.shared;
        let mut ring = shared.capture.ring.consumer().unwrap();
        let mut steer = Steer::new();
        let ramp: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        queued_a_second_ago(&mut capture, &ramp);
        shared.capture.packet.store(240, Ordering::Relaxed);
        let mut taken = vec![0.0; 4096 * MAX_RATIO];
        let mut voice = vec![0.0; 480];
        // The output's warm-up first, consuming nothing.
        for _ in 0..10 {
            assert!(!steer.begin(shared, &mut ring, 48_000, 480));
        }
        assert!(steer.begin(shared, &mut ring, 48_000, 480));
        steer.render(&mut ring, &mut voice, &mut taken);
        // The first buffer runs at the nominal ratio, three samples late
        // (the interpolator's history), from the target (and half a packet,
        // with none arriving) before the end.
        let start = 48_000 - (480 + 480 + 120);
        for k in 3..480 {
            assert_eq!(voice[k], ramp[start + k - 3], "{k}");
        }
    }
}
