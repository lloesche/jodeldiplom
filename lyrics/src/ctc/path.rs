//! The Viterbi path itself: a model's emissions, the tokens' states, the
//! skips and costs a path may take, and the most likely path.

use std::ops::Range;

/// A model's output for a stretch of voice: a log-probability for every
/// frame and token.
#[derive(Clone, Debug, PartialEq)]
pub struct Emissions {
    /// The log-probabilities, frame by frame (`frames × tokens`).
    pub logp: Vec<f32>,
    /// Tokens a frame.
    pub tokens: usize,
    /// Seconds a frame.
    pub frame_seconds: f64,
}

impl Emissions {
    /// How many frames there are.
    pub fn frames(&self) -> usize {
        self.logp.len().checked_div(self.tokens).unwrap_or(0)
    }

    /// The log-probability of `token` at `frame`.
    pub fn at(&self, frame: usize, token: u32) -> f32 {
        self.logp[frame * self.tokens + token as usize]
    }

    /// Its frames' clock.
    pub fn clock(&self) -> FrameClock {
        FrameClock {
            frame_seconds: self.frame_seconds,
            frames: self.frames(),
        }
    }
}

/// Frames of a stretch of voice, from 0 s: seconds turned into frames, each
/// way of rounding named, and frames back into seconds. A time before 0 is
/// frame 0, and none is past the last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameClock {
    /// Seconds a frame.
    pub frame_seconds: f64,
    /// How many frames there are.
    pub frames: usize,
}

impl FrameClock {
    /// `frames` frames of `hop` samples of a voice at `rate` Hz.
    pub fn of_rate(rate: u32, hop: usize, frames: usize) -> FrameClock {
        FrameClock {
            frame_seconds: hop as f64 / f64::from(rate),
            frames,
        }
    }

    /// How long the frames last, in seconds.
    pub fn seconds(&self) -> f64 {
        self.time(self.frames)
    }

    /// When frame `frame` starts, in seconds (the end of the one before).
    pub fn time(&self, frame: usize) -> f64 {
        frame as f64 * self.frame_seconds
    }

    /// The frame starting nearest to `t` seconds.
    pub fn nearest(&self, t: f64) -> usize {
        ((t / self.frame_seconds).round() as usize).min(self.frames)
    }

    /// The frames a state may be entered on to start from `a` to `b`
    /// seconds: those rounded outward ([`FrameClock::outward`]), and the
    /// frame after, on which a start at `b` is entered.
    pub fn entry_window(&self, a: f64, b: f64) -> Range<usize> {
        let frames = self.outward(a, b);
        frames.start..frames.end + 1
    }

    /// The frames from `a` to `b` seconds, rounded outward: every frame
    /// with some of that time.
    pub fn outward(&self, a: f64, b: f64) -> Range<usize> {
        let from = ((a / self.frame_seconds).floor() as usize).min(self.frames);
        let to = ((b / self.frame_seconds).ceil() as usize).min(self.frames);
        from..to
    }

    /// The frames from `margin` after `a` to `margin` before `b` seconds,
    /// rounded inward: only frames wholly within that time.
    pub fn inward(&self, a: f64, b: f64, margin: f64) -> Range<usize> {
        let from = (((a + margin) / self.frame_seconds).ceil() as usize).min(self.frames);
        let to = (((b - margin) / self.frame_seconds).floor() as usize).min(self.frames);
        from..to
    }
}

/// Tokens the path may leave out, at a cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Skip {
    /// The tokens left out.
    pub tokens: Range<usize>,
    /// The cost, in log-probability (positive: subtracted from the path's
    /// score).
    pub penalty: f32,
}

/// The path the alignment found.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    /// The state at each frame: `2i` the blank before token `i`, `2i + 1`
    /// token `i`, `2N` the blank after the last.
    pub states: Vec<u32>,
    /// Its score: the sum of its log-probabilities, less the costs of its
    /// skips.
    pub score: f32,
}

impl Path {
    /// The frames on each of `count` tokens; `None` for a token the path
    /// left out.
    pub fn token_frames(&self, count: usize) -> Vec<Option<Range<usize>>> {
        let mut spans: Vec<Option<Range<usize>>> = vec![None; count];
        for (f, &s) in self.states.iter().enumerate() {
            if let Some(k) = token_at(s as usize) {
                let span = &mut spans[k];
                match span {
                    Some(r) => r.end = f + 1,
                    None => *span = Some(f..f + 1),
                }
            }
        }
        spans
    }
}

/// The states of a path through `tokens` tokens: a blank before each, each
/// token, and a blank after the last.
pub(crate) fn state_count(tokens: usize) -> usize {
    2 * tokens + 1
}

/// Token `k`'s state.
pub(crate) fn token_state(k: usize) -> usize {
    2 * k + 1
}

/// The blank before token `k` (after the last, for `k` their count).
pub(crate) fn blank_state(k: usize) -> usize {
    2 * k
}

/// The token state `s` is on; `None` for a blank.
pub(crate) fn token_at(s: usize) -> Option<usize> {
    (s % 2 == 1).then_some(s / 2)
}

/// The token blank state `s` is before; `None` for a token's state.
pub(crate) fn blank_before(s: usize) -> Option<usize> {
    (s % 2 == 0).then_some(s / 2)
}

/// The states of `tokens` and of the blanks between them: from the first's
/// to the last's.
pub(crate) fn states_of(tokens: Range<usize>) -> Range<usize> {
    token_state(tokens.start)..blank_state(tokens.end)
}

/// How a predecessor reaches a state, as a back-pointer byte: from itself,
/// the state before, two before, or the `k`-th skip landing on it
/// (`EXTRA + k`).
const STAY: u8 = 0;
const STEP: u8 = 1;
const JUMP: u8 = 2;
const EXTRA: u8 = 3;

/// One kind of cost a frame on a state pays, in log-probability (nothing
/// past a slice's end).
#[derive(Clone, Debug, PartialEq)]
pub enum StayCost {
    /// A frame on state `s` costs `by_state[s]` where `frames` marks it,
    /// or on every frame when there is no mark.
    Marked {
        /// The cost on each state.
        by_state: Vec<f32>,
        /// The frames that cost.
        frames: Option<Vec<bool>>,
    },
    /// A frame on state `s` outside the frames of the window `s` has
    /// costs that window's penalty.
    Outside(Vec<Option<(Range<usize>, f32)>>),
    /// A frame `f` on a state outside the states frame `f` holds the path
    /// to costs that hold's penalty.
    Held(Vec<Option<(Range<usize>, f32)>>),
}

impl StayCost {
    /// What it costs a frame `f` on state `s`.
    fn at(&self, f: usize, s: usize) -> f32 {
        match self {
            StayCost::Marked { by_state, frames } => {
                let marked = frames
                    .as_ref()
                    .is_none_or(|m| m.get(f).copied().unwrap_or(false));
                if marked {
                    by_state.get(s).copied().unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            StayCost::Outside(windows) => match windows.get(s) {
                Some(Some((frames, penalty))) if !frames.contains(&f) => *penalty,
                _ => 0.0,
            },
            StayCost::Held(holds) => match holds.get(f) {
                Some(Some((states, penalty))) if !states.contains(&s) => *penalty,
                _ => 0.0,
            },
        }
    }
}

/// What a path pays besides its log-probabilities, by state `s` and frame
/// `f` (in log-probability; nothing past a slice's end).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Costs {
    /// What a frame costs, each kind added in turn, in this order.
    pub stay: Vec<StayCost>,
    /// Entering state `s` outside the frames of each of `entry[s]` costs
    /// that one's penalty (each cost source adds its own).
    pub entry: Vec<Vec<(Range<usize>, f32)>>,
}

impl Costs {
    /// What a frame `f` on state `s` costs.
    fn stay(&self, f: usize, s: usize) -> f32 {
        self.stay.iter().fold(0.0, |sum, cost| sum + cost.at(f, s))
    }

    /// What entering state `s` at frame `f` costs.
    fn entering(&self, f: usize, s: usize) -> f32 {
        self.entry.get(s).map_or(0.0, |of| {
            of.iter()
                .filter(|(frames, _)| !frames.contains(&f))
                .map(|(_, penalty)| penalty)
                .sum()
        })
    }
}

/// The most likely path of `ids` through `emissions`, with `blank` the
/// blank's index, leaving out `skips` at their costs, and paying `costs`.
/// `None` when no path fits (more tokens than frames).
pub fn force_align(
    emissions: &Emissions,
    ids: &[u32],
    blank: u32,
    skips: &[Skip],
    costs: &Costs,
) -> Option<Path> {
    let frames = emissions.frames();
    let n = ids.len();
    let states = state_count(n);
    if frames == 0 {
        return None;
    }
    let token_of = |s: usize| token_at(s).map_or(blank, |k| ids[k]);
    let cost = |f: usize, s: usize| costs.stay(f, s);
    // The skips' extra edges, by the state they land on: (from, cost).
    let mut extra: Vec<Vec<(usize, f32)>> = vec![Vec::new(); states];
    for skip in skips {
        let (a, b) = (skip.tokens.start, skip.tokens.end);
        if a >= b || b > n {
            continue;
        }
        let from = [Some(blank_state(a)), (a > 0).then(|| token_state(a - 1))];
        let to = [Some(blank_state(b)), (b < n).then(|| token_state(b))];
        for f in from.into_iter().flatten() {
            for t in to.into_iter().flatten() {
                // Two tokens in a row that are the same need a blank
                // between them.
                if matches!((token_at(f), token_at(t)), (Some(x), Some(y)) if ids[x] == ids[y]) {
                    continue;
                }
                if extra[t].len() < usize::from(u8::MAX - EXTRA) {
                    extra[t].push((f, skip.penalty));
                }
            }
        }
    }
    let mut score = vec![f32::NEG_INFINITY; states];
    let mut next = vec![f32::NEG_INFINITY; states];
    let mut back = vec![STAY; frames * states];
    // The path starts on the first blank or the first token.
    for (s, first) in score.iter_mut().enumerate().take(2) {
        *first = emissions.at(0, token_of(s)) - cost(0, s) - costs.entering(0, s);
    }
    for f in 1..frames {
        let row = &mut back[f * states..(f + 1) * states];
        for s in 0..states {
            next[s] = f32::NEG_INFINITY;
            let mut best = score[s];
            let mut how = STAY;
            let entry = costs.entering(f, s);
            if s >= 1 && score[s - 1] - entry > best {
                best = score[s - 1] - entry;
                how = STEP;
            }
            // Over the blank between two different tokens.
            let past_blank = token_at(s).is_some_and(|k| k >= 1 && ids[k] != ids[k - 1]);
            if past_blank && score[s - 2] - entry > best {
                best = score[s - 2] - entry;
                how = JUMP;
            }
            for (k, &(from, cost)) in extra[s].iter().enumerate() {
                let through = score[from] - cost - entry;
                if through > best {
                    best = through;
                    how = EXTRA + k as u8;
                }
            }
            if best > f32::NEG_INFINITY {
                next[s] = best + emissions.at(f, token_of(s)) - cost(f, s);
                row[s] = how;
            }
        }
        std::mem::swap(&mut score, &mut next);
    }
    let mut end = states - 1;
    if n > 0 && score[states - 2] > score[end] {
        end = states - 2;
    }
    let best = score[end];
    if best == f32::NEG_INFINITY {
        return None;
    }
    let mut path = vec![0u32; frames];
    let mut s = end;
    for f in (0..frames).rev() {
        path[f] = s as u32;
        if f == 0 {
            break;
        }
        s = match back[f * states + s] {
            STAY => s,
            STEP => s - 1,
            JUMP => s - 2,
            k => extra[s][usize::from(k - EXTRA)].0,
        };
    }
    Some(Path {
        states: path,
        score: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seconds to frames and back, each rounding as named, nothing past
    /// the last frame.
    #[test]
    fn the_clock_turns_seconds_into_frames_and_back() {
        let clock = FrameClock::of_rate(16_000, 320, 100);
        assert_eq!(clock.frame_seconds, 0.02);
        assert_eq!(clock.time(50), 1.0);
        assert_eq!(clock.seconds(), 2.0);
        assert_eq!(clock.nearest(0.509), 25);
        assert_eq!(clock.nearest(0.511), 26);
        assert_eq!(clock.nearest(-1.0), 0);
        assert_eq!(clock.nearest(9.0), 100, "clamped to the frames");
        assert_eq!(clock.outward(0.51, 0.59), 25..30);
        assert_eq!(clock.entry_window(0.51, 0.59), 25..31);
        assert_eq!(clock.inward(0.51, 0.69, 0.0), 26..34);
    }

    /// A token's state and the blanks either side, there and back.
    #[test]
    fn the_states_are_a_blank_before_each_token_and_one_after() {
        assert_eq!(state_count(3), 7);
        for k in 0..3 {
            assert_eq!(token_at(token_state(k)), Some(k));
            assert_eq!(blank_before(blank_state(k)), Some(k));
            assert_eq!(token_at(blank_state(k)), None);
            assert_eq!(blank_before(token_state(k)), None);
        }
        assert_eq!(blank_before(blank_state(3)), Some(3));
        // Tokens 1 and 2 and the blank between them.
        assert_eq!(states_of(1..3), 3..6);
        assert_eq!(states_of(0..1), 1..2);
    }
}
