//! An input's lanes as the mixer thread runs them ([`crate::lanes`]): main
//! (lane 0) and its parts, each on a strip of its own, and the switch from
//! main to the parts and back.
//!
//! The switch has a gain of its own, `s`, linear over [`SWITCH_FADE`]:
//!
//! - main is heard at the input's fader × (1 − s) × its strip's path;
//! - part k at the input's fader × s × part k's fader × its strip's path;
//! - each times the input's automation: its own curve, and main's or
//!   part k's (`automation.rs`).
//!
//! `s` goes to 1 when the parts are touched (for some part, its fader's
//! target times its strip's own gain is not 1, its strip's solo gate is
//! not main's, or its automation's curve is not exactly 1, see
//! `automation.rs`) while the control word says [`Parts::Present`]. It
//! goes back to 0 only once nothing is touched and every part's ramps have
//! settled (and their automation), so main returns bit for bit; from the
//! frame the control word turns
//! [`Parts::Ending`] it fades to 0 (the writer keeps the parts valid that
//! long), and on a frame that says [`Parts::None`] it is 0 at once (which a
//! writer does straight after `Present` only on a new track's first frame,
//! where main cuts anyway). A change of the control word, or the parts
//! touched, lands on its very frame within the chunk.
//!
//! A frame the stream does not supply (it is not running, a flush keeps
//! the mixer out, or it ran dry) has no word: nothing of the input is heard
//! there, and the switch keeps what it knew. On the first frame supplied
//! after such a gap (a seek, a resume, a new output, an underrun, the very
//! first) `s` is at its aim at once: with nothing heard before it, there is
//! nothing to crossfade from.
//!
//! "Before the chunk" is one convention here and in
//! [`super::strips::Look`]: a frame noted at or before the chunk's first
//! means from its start.

use std::time::Duration;

use super::envelope::Envelope;
use super::ramp::{start_of, Gain, GainId, Waiting, When};
use super::strips::Strips;
use super::Curve;
use crate::lanes::{Parts, SWITCH_FADE};

/// A lane of an input.
pub(super) struct LaneRun {
    /// Its fader's gain, by id: the input's own for main (lane 0), the
    /// part's for a part.
    pub(super) id: GainId,
    /// Its strip.
    pub(super) strip: u64,
    pub(super) gain: Gain,
    /// Ramps set on that fader and taken.
    pub(super) ramps: u64,
    /// Its samples this chunk, when its stream is a lane stream (a plain
    /// stream's are its own).
    pub(super) samples: Vec<i32>,
    /// This chunk's gains: what it is heard at.
    pub(super) gains: Envelope,
    /// It was heard in the last chunk: it supplied frames at a gain other
    /// than 0.
    pub(super) heard: bool,
    /// A part present in its stream but not heard (`s` = 0): its strip's
    /// meter still shows it, at `meter` (what it would be heard at with the
    /// parts switched in).
    pub(super) metering: bool,
    pub(super) meter: Envelope,
    /// Its gain at the last frame mixed.
    pub(super) last: f64,
    /// It may reach the device as written: main (a plain input's only
    /// lane, or a lane stream's main). A part never does.
    pub(super) exact: bool,
}

impl LaneRun {
    pub(super) fn new(id: GainId, strip: u64, gain: f64) -> LaneRun {
        LaneRun {
            id,
            strip,
            gain: Gain::new(gain),
            ramps: 0,
            samples: Vec::new(),
            gains: Envelope::new(gain),
            heard: false,
            metering: false,
            meter: Envelope::new(0.0),
            last: 0.0,
            exact: id.lane == 0,
        }
    }
}

/// Splits a lane stream's frames (`samples`, every lane of `channels`
/// samples and the control word, a frame after another) into each lane's
/// samples, and the control words of the `got` frames the stream supplied
/// from frame `at` (the frames around them have none).
pub(super) fn split(
    samples: &[i32],
    channels: usize,
    at: usize,
    got: usize,
    lanes: &mut [LaneRun],
    words: &mut Vec<i32>,
) {
    let width = channels * lanes.len() + 1;
    let frames = samples.len() / width;
    words.clear();
    let supplied = samples.chunks_exact(width).skip(at).take(got);
    words.extend(supplied.map(|f| f[width - 1]));
    for (k, lane) in lanes.iter_mut().enumerate() {
        lane.samples.clear();
        lane.samples.reserve(frames * channels);
        for frame in samples.chunks_exact(width) {
            lane.samples
                .extend_from_slice(&frame[k * channels..(k + 1) * channels]);
        }
    }
}

/// How an input's parts stand this chunk, for its switch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Touch {
    /// Some part is touched, as of the chunk's end.
    pub(super) touched: bool,
    /// The earliest frame anything that counts changed at (`None`: nothing
    /// noted; at or before the chunk's first: from its start).
    pub(super) at: Option<u64>,
    /// Every part's ramps (its fader's, its strip's, main's strip's) have
    /// settled by the chunk's end.
    pub(super) settled: bool,
}

impl Touch {
    /// Nothing touched, nothing changed, everything settled.
    pub(super) const UNTOUCHED: Touch = Touch {
        touched: false,
        at: None,
        settled: true,
    };

    /// This and `other` together: touched where either is, from the
    /// earlier change, settled where both are.
    pub(super) fn merge(self, other: Touch) -> Touch {
        let at = match (self.at, other.at) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        Touch {
            touched: self.touched || other.touched,
            at,
            settled: self.settled && other.settled,
        }
    }
}

/// How the parts of an input (`lanes`, main first) stand after this
/// chunk's fills: their faders' and strips' targets (`starts`: the faders'
/// ramps starting in it).
pub(super) fn touch(lanes: &[LaneRun], strips: &Strips, starts: &[(GainId, u64)]) -> Touch {
    let main = strips.look(lanes[0].strip);
    let mut t = Touch {
        touched: false,
        at: main.changed,
        settled: !main.moving,
    };
    for lane in &lanes[1..] {
        let part = strips.look(lane.strip);
        t.touched |= lane.gain.target() * part.own != 1.0 || part.gate != main.lead_gate;
        t.settled &= !lane.gain.moving() && !part.moving;
        let started = start_of(starts, lane.id);
        for at in [part.changed, started].into_iter().flatten() {
            t.at = Some(t.at.map_or(at, |a| a.min(at)));
        }
    }
    t
}

/// The switch of an input with parts between main and them.
pub(super) struct Switch {
    gain: Gain,
    target: f64,
    /// This chunk: the parts' share `s`, and main's, `1 − s`.
    pub(super) share: Envelope,
    pub(super) rest: Envelope,
    /// This chunk's control words: one a frame the stream supplied, from
    /// the chunk's frame `from`.
    pub(super) words: Vec<i32>,
    pub(super) from: usize,
    /// The control word of the last frame supplied, whether the parts were
    /// touched there, and whether their ramps had settled by the last
    /// chunk's end.
    pub(super) control: Parts,
    touched: bool,
    settled: bool,
    /// The last frame was not supplied (or none has been yet).
    absent: bool,
    /// `s` at the last frame mixed.
    pub(super) last: f64,
}

impl Switch {
    pub(super) fn new() -> Switch {
        Switch {
            gain: Gain::new(0.0),
            target: 0.0,
            share: Envelope::new(0.0),
            rest: Envelope::new(1.0),
            words: Vec::new(),
            from: 0,
            control: Parts::None,
            touched: false,
            settled: true,
            absent: true,
            last: 0.0,
        }
    }

    /// The switch's gains for frames `first..first + n` at `rate`: with
    /// `lanes`, the stream is a lane stream (or none) and supplied the
    /// frames it has words for; without, it is a plain stream, with no
    /// parts on any frame. `touch` is how the parts stand.
    pub(super) fn fill(&mut self, first: u64, n: usize, rate: u32, lanes: bool, touch: Touch) {
        let end = first + n as u64;
        let flip = (touch.touched != self.touched)
            .then(|| (touch.at.unwrap_or(first).clamp(first, end - 1) - first) as usize);
        let supplied = if lanes {
            self.from..self.from + self.words.len()
        } else {
            0..n
        };
        self.share.cut(n);
        // At the chunk's start, ramps settled by the last one's end count.
        if !self.absent {
            self.aim(self.settled);
        }
        let mut from = 0;
        for k in 0..n {
            let flips = flip == Some(k);
            if !supplied.contains(&k) {
                // No word: nothing changes but what is noted.
                self.absent = true;
                if flips {
                    self.touched = touch.touched;
                }
                continue;
            }
            let control = if lanes {
                Parts::from_word(self.words[k - self.from])
            } else {
                Parts::None
            };
            if !self.absent && control == self.control && !flips {
                continue;
            }
            self.share.fill_part(&mut self.gain, first, from, k, rate);
            from = k;
            self.control = control;
            if flips {
                self.touched = touch.touched;
            }
            if std::mem::take(&mut self.absent) {
                // The first frame after a gap: nothing was heard before.
                self.jump();
            } else {
                // Something just changed: nothing has settled.
                self.aim(false);
            }
        }
        self.share.fill_part(&mut self.gain, first, from, n, rate);
        self.share.join();
        self.rest.assign_complement(&self.share);
        self.settled = touch.settled;
        self.last = self.share.last(n);
    }

    /// The parts' share that the control word and the touch give.
    fn aimed(&self) -> f64 {
        if self.control == Parts::Present && self.touched {
            1.0
        } else {
            0.0
        }
    }

    /// `s` at its aim at once.
    fn jump(&mut self) {
        self.target = self.aimed();
        self.gain.set(Waiting {
            when: When::Now,
            to: self.target,
            over: Duration::ZERO,
            curve: Curve::Linear,
        });
    }

    /// `s` aims at what the control word and the parts give now
    /// (`settled`: their ramps have).
    fn aim(&mut self, settled: bool) {
        match self.control {
            Parts::None => self.jump(),
            Parts::Ending => self.ramp(0.0),
            Parts::Present if self.touched => self.ramp(1.0),
            Parts::Present if settled => self.ramp(0.0),
            Parts::Present => {}
        }
    }

    fn ramp(&mut self, to: f64) {
        if to != self.target {
            self.target = to;
            self.gain.set(Waiting {
                when: When::Now,
                to,
                over: SWITCH_FADE,
                curve: Curve::Linear,
            });
        }
    }

    /// Nothing of the input is mixed: the switch's ramp ends at once.
    pub(super) fn settle(&mut self) {
        self.gain.settle();
    }

    /// The device opened again: a ramp under way ends at once.
    pub(super) fn new_epoch(&mut self) {
        self.gain.new_epoch();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const FADE: usize = 480; // 10 ms

    /// A lane stream of 1 and of 3 channels (the desk mixes stereo, the
    /// lanes take any width): each lane's samples, and the words of the
    /// frames supplied.
    #[test]
    fn a_lane_stream_splits_at_any_width() {
        for channels in [1usize, 3] {
            let mut lanes: Vec<LaneRun> = (0..3)
                .map(|k| LaneRun::new(GainId::new(1, k), 0, 1.0))
                .collect();
            let width = channels * 3 + 1;
            assert_eq!(crate::lanes::lane_width(channels as u16, 3), width);
            // Frame f: lane k's sample c is f·100 + k·10 + c, the word f.
            let samples: Vec<i32> = (0..4)
                .flat_map(|f| {
                    let lanes = (0..3)
                        .flat_map(move |k| (0..channels).map(move |c| f * 100 + k * 10 + c as i32));
                    lanes.chain(std::iter::once(f))
                })
                .collect();
            let mut words = Vec::new();
            split(&samples, channels, 1, 2, &mut lanes, &mut words);
            assert_eq!(words, [1, 2]);
            for (k, lane) in lanes.iter().enumerate() {
                let want: Vec<i32> = (0..4)
                    .flat_map(|f| (0..channels).map(move |c| f * 100 + k as i32 * 10 + c as i32))
                    .collect();
                assert_eq!(lane.samples, want, "{channels} channels, lane {k}");
            }
        }
    }

    /// The words of `n` frames supplied, each from the last `(from, word)`
    /// at or before it.
    fn words(sw: &mut Switch, w: &[(usize, Parts)], n: usize) {
        sw.words.clear();
        for k in 0..n {
            let p = w
                .iter()
                .rev()
                .find(|(from, _)| *from <= k)
                .map_or(Parts::None, |p| p.1);
            sw.words.push(p.word());
        }
    }

    fn share(sw: &Switch, n: usize) -> Vec<f64> {
        (0..n).map(|k| sw.share.at(k)).collect()
    }

    fn touched(at: Option<u64>) -> Touch {
        Touch {
            touched: true,
            at,
            settled: false,
        }
    }

    #[test]
    fn the_switch_follows_the_control_word_on_its_frames() {
        let mut sw = Switch::new();
        // Touched before the parts are there: main.
        words(&mut sw, &[(0, Parts::None)], 100);
        sw.fill(0, 100, RATE, true, touched(None));
        assert_eq!(share(&sw, 100), [0.0; 100]);
        // Present from frame 130: the switch from that very frame.
        words(&mut sw, &[(0, Parts::None), (30, Parts::Present)], 1_000);
        sw.fill(100, 1_000, RATE, true, touched(None));
        let s = share(&sw, 1_000);
        assert_eq!(s[..31], [0.0; 31]);
        for k in 0..FADE {
            assert!((s[30 + k] - k as f64 / FADE as f64).abs() < 1e-12, "{k}");
        }
        assert_eq!(s[30 + FADE..], [1.0; 1_000 - 30 - FADE]);
        assert_eq!(sw.rest.flat(), None);
        // Ending at 1 150: back to main over the fade, from there.
        words(&mut sw, &[(0, Parts::Present), (50, Parts::Ending)], 1_000);
        sw.fill(1_100, 1_000, RATE, true, touched(None));
        let s = share(&sw, 1_000);
        assert_eq!(s[..51], [1.0; 51]);
        assert!((s[50 + FADE / 2] - 0.5).abs() < 1e-12);
        assert_eq!(s[50 + FADE..], [0.0; 1_000 - 50 - FADE]);
        // Present again (the next parts): the switch back in from there.
        words(&mut sw, &[(0, Parts::Ending), (20, Parts::Present)], 600);
        sw.fill(2_100, 600, RATE, true, touched(None));
        let s = share(&sw, 600);
        assert_eq!(s[..21], [0.0; 21]);
        assert_eq!(s[20 + FADE..], [1.0; 600 - 20 - FADE]);
        // None while switched: main at once.
        words(&mut sw, &[(0, Parts::Present), (7, Parts::None)], 100);
        sw.fill(2_700, 100, RATE, true, touched(None));
        let s = share(&sw, 100);
        assert!(s[6] > 0.0 && s[7..] == [0.0; 93]);
    }

    #[test]
    fn the_switch_goes_back_only_once_the_parts_have_settled() {
        let mut sw = Switch::new();
        words(&mut sw, &[(0, Parts::Present)], 600);
        sw.fill(0, 600, RATE, true, touched(None));
        // Touched from the first frame supplied: switched at once.
        assert_eq!(sw.share.flat(), Some(1.0));
        // Untouched, the parts' ramps still under way: it holds.
        let back = Touch {
            touched: false,
            at: Some(650),
            settled: false,
        };
        sw.fill(600, 600, RATE, true, back);
        assert_eq!(sw.share.flat(), Some(1.0));
        // Settled by that chunk's end: back from the next one's start.
        let rest = Touch {
            touched: false,
            at: None,
            settled: true,
        };
        sw.fill(1_200, 600, RATE, true, rest);
        assert_eq!(
            sw.share.flat(),
            Some(1.0),
            "settled only by this chunk's end"
        );
        sw.fill(1_800, 600, RATE, true, rest);
        let s = share(&sw, 600);
        assert!((s[0] - 1.0).abs() < 1e-12 && s[FADE..] == [0.0; 600 - FADE]);
        // A plain stream on it: no parts, main.
        sw.fill(2_400, 10, RATE, false, touched(None));
        assert_eq!(sw.share.flat(), Some(0.0));
    }

    #[test]
    fn words_from_inside_the_chunk_count_from_there() {
        let mut sw = Switch::new();
        // A follower's frames from frame 100 of the chunk: a gap before.
        words(&mut sw, &[(0, Parts::Present)], 380);
        sw.from = 100;
        sw.fill(0, 480, RATE, true, touched(None));
        let s = share(&sw, 480);
        assert_eq!(s[..100], [0.0; 100]);
        assert_eq!(s[100..], [1.0; 380]);
    }

    #[test]
    fn a_gap_in_the_stream_keeps_the_switch_and_resumes_at_its_aim() {
        let mut sw = Switch::new();
        // Touched at frame 100, the switch half way at the chunk's end.
        words(&mut sw, &[(0, Parts::Present)], 340);
        sw.fill(0, 340, RATE, true, touched(Some(100)));
        assert!((sw.last - 239.0 / FADE as f64).abs() < 1e-12, "{}", sw.last);
        // The stream runs dry after 40 frames, and gives nothing for a
        // whole chunk: no word, nothing changes.
        words(&mut sw, &[(0, Parts::Present)], 40);
        sw.fill(340, 480, RATE, true, touched(None));
        assert_eq!(sw.control, Parts::Present);
        sw.words.clear();
        sw.fill(820, 480, RATE, true, touched(None));
        // Back: at its aim at once, no ramp.
        words(&mut sw, &[(0, Parts::Present)], 470);
        sw.fill(1_300, 470, RATE, true, touched(None));
        assert_eq!(sw.share.flat(), Some(1.0));
        // A gap while untouched comes back on main at once, whatever ramp
        // the switch was on.
        let rest = Touch {
            touched: false,
            at: None,
            settled: false,
        };
        sw.words.clear();
        sw.fill(1_770, 480, RATE, true, rest);
        words(&mut sw, &[(0, Parts::Present)], 480);
        sw.fill(2_250, 480, RATE, true, rest);
        assert_eq!(sw.share.flat(), Some(0.0));
    }
}
