//! The tempo-synced stereo delay every voice shares, fed by the voices'
//! sends ([`Delay`]). Its lines are made once, at the synth's rate;
//! nothing allocates as it runs.
//!
//! Sources: our own design: two lines, a one-pole low-pass in the
//! feedback, straight or ping-pong.

use crate::dsp::{flush, Rate};
use crate::BLOCK;

/// The delay's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaySettings {
    /// The time between echoes, beats (at the tempo the synth is told).
    pub beats: f32,
    /// How much of each echo comes back, 0 to below 1.
    pub feedback: f32,
    /// How much each echo darkens, 0 (not at all) to 1.
    pub damp: f32,
    /// How loud the echoes are, 0 (silent) to 1.
    pub wet: f32,
    /// The echoes going from side to side.
    pub ping_pong: bool,
}

impl Default for DelaySettings {
    fn default() -> DelaySettings {
        DelaySettings {
            beats: 0.75,
            feedback: 0.35,
            damp: 0.3,
            wet: 0.5,
            ping_pong: true,
        }
    }
}

/// The longest delay, seconds.
pub const DELAY_MAX: f32 = 2.0;

/// A stereo delay: a line a side, the echoes fed back through a one-pole
/// low-pass, straight or from side to side. Its time follows the tempo,
/// gliding there over a few blocks when it changes.
#[derive(Clone, Debug)]
pub struct Delay {
    left: Vec<f32>,
    right: Vec<f32>,
    at: usize,
    /// The time now and the time it glides to, samples.
    time: f32,
    target: f32,
    /// The feedback's low-passes.
    lp: [f32; 2],
    settings: DelaySettings,
    rate: f32,
}

impl Delay {
    /// A silent one at `rate`, at 120 beats a minute.
    pub fn new(rate: Rate) -> Delay {
        let len = (DELAY_MAX * rate.hz()) as usize + 2;
        let mut d = Delay {
            left: vec![0.0; len],
            right: vec![0.0; len],
            at: 0,
            time: 0.0,
            target: 0.0,
            lp: [0.0; 2],
            settings: DelaySettings::default(),
            rate: rate.hz(),
        };
        d.retime(120.0);
        d.time = d.target;
        d
    }

    /// Its settings now.
    pub fn settings(&self) -> DelaySettings {
        self.settings
    }

    /// Takes `s` at `bpm`, its time at once: for setting up (a jump in
    /// the time while echoes ring shifts them; a tempo change goes through
    /// [`Delay::retime`], which glides).
    pub fn set(&mut self, s: DelaySettings, bpm: f32) {
        self.settings = s;
        self.retime(bpm);
        self.time = self.target;
    }

    /// The time for `bpm`, gliding there.
    pub fn retime(&mut self, bpm: f32) {
        let seconds = self.settings.beats.max(0.0) * 60.0 / bpm.max(1.0);
        self.target = (seconds * self.rate).clamp(1.0, (self.left.len() - 2) as f32);
    }

    /// The block's sends `send` (left, right) through, the echoes added to
    /// `left` and `right`.
    pub fn process(
        &mut self,
        send: [&[f32; BLOCK]; 2],
        left: &mut [f32; BLOCK],
        right: &mut [f32; BLOCK],
    ) {
        let s = self.settings;
        self.time += (self.target - self.time) * 0.1;
        if (self.time - self.target).abs() < 0.01 {
            self.time = self.target;
        }
        let feedback = s.feedback.clamp(0.0, 0.99);
        let keep = s.damp.clamp(0.0, 1.0) * 0.95;
        let len = self.left.len();
        let back = self.time.floor() as usize;
        let frac = self.time - back as f32;
        for i in 0..BLOCK {
            // Read between the two samples the time falls between.
            let a = (self.at + len - back) % len;
            let b = (a + len - 1) % len;
            let yl = self.left[a] + (self.left[b] - self.left[a]) * frac;
            let yr = self.right[a] + (self.right[b] - self.right[a]) * frac;
            self.lp[0] = flush(yl + (self.lp[0] - yl) * keep);
            self.lp[1] = flush(yr + (self.lp[1] - yr) * keep);
            let [fl, fr] = self.lp;
            let (inl, inr) = if s.ping_pong {
                (
                    0.5 * (send[0][i] + send[1][i]) + fr * feedback,
                    fl * feedback,
                )
            } else {
                (send[0][i] + fl * feedback, send[1][i] + fr * feedback)
            };
            self.left[self.at] = inl;
            self.right[self.at] = inr;
            self.at = (self.at + 1) % len;
            left[i] += yl * s.wet;
            right[i] += yr * s.wet;
        }
    }
}
