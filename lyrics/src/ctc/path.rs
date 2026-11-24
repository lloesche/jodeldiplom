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
            if s % 2 == 1 {
                let span = &mut spans[(s / 2) as usize];
                match span {
                    Some(r) => r.end = f + 1,
                    None => *span = Some(f..f + 1),
                }
            }
        }
        spans
    }
}

/// How a predecessor reaches a state, as a back-pointer byte: from itself,
/// the state before, two before, or the `k`-th skip landing on it
/// (`EXTRA + k`).
const STAY: u8 = 0;
const STEP: u8 = 1;
const JUMP: u8 = 2;
const EXTRA: u8 = 3;

/// What a path pays besides its log-probabilities, by state `s` and frame
/// `f` (in log-probability; nothing past a slice's end).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Costs {
    /// A frame on state `s` costs `charged[s]` where [`Costs::charging`]
    /// says so.
    pub charged: Vec<f32>,
    /// The frames that charge.
    pub charging: Vec<bool>,
    /// A frame on state `s` costs `always[s]`.
    pub always: Vec<f32>,
    /// A frame on state `s` costs `spanned[s]` where [`Costs::unheard`]
    /// says so.
    pub spanned: Vec<f32>,
    /// The frames where nothing with words was heard.
    pub unheard: Vec<bool>,
    /// Entering state `s` outside the frames of each of `entry[s]` costs
    /// that one's penalty (each cost source adds its own).
    pub entry: Vec<Vec<(Range<usize>, f32)>>,
    /// A frame on state `s` outside the frames of `outside[s]` costs its
    /// penalty.
    pub outside: Vec<Option<(Range<usize>, f32)>>,
}

impl Costs {
    /// What a frame `f` on state `s` costs.
    fn stay(&self, f: usize, s: usize) -> f32 {
        let charged = if self.charging.get(f).copied().unwrap_or(false) {
            self.charged.get(s).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let spanned = if self.unheard.get(f).copied().unwrap_or(false) {
            self.spanned.get(s).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let outside = match self.outside.get(s) {
            Some(Some((frames, penalty))) if !frames.contains(&f) => *penalty,
            _ => 0.0,
        };
        charged + spanned + outside + self.always.get(s).copied().unwrap_or(0.0)
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
    let states = 2 * n + 1;
    if frames == 0 {
        return None;
    }
    let token_of = |s: usize| if s % 2 == 1 { ids[s / 2] } else { blank };
    let cost = |f: usize, s: usize| costs.stay(f, s);
    // The skips' extra edges, by the state they land on: (from, cost).
    let mut extra: Vec<Vec<(usize, f32)>> = vec![Vec::new(); states];
    for skip in skips {
        let (a, b) = (skip.tokens.start, skip.tokens.end);
        if a >= b || b > n {
            continue;
        }
        let from = [Some(2 * a), (a > 0).then(|| 2 * a - 1)];
        let to = [Some(2 * b), (b < n).then(|| 2 * b + 1)];
        for f in from.into_iter().flatten() {
            for t in to.into_iter().flatten() {
                // Two tokens in a row that are the same need a blank
                // between them.
                if f % 2 == 1 && t % 2 == 1 && ids[f / 2] == ids[t / 2] {
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
            if s >= 3 && s % 2 == 1 && ids[s / 2] != ids[s / 2 - 1] && score[s - 2] - entry > best {
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
