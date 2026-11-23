//! Forced alignment of the lyrics' tokens to a CTC model's output.
//!
//! The model gives, for every frame of voice (20 ms for wav2vec2), the
//! log-probability of each token and of the blank ([`Emissions`]). The
//! lyrics' tokens are known ([`crate::tokens`]); what is not known is
//! where each falls. The forced alignment finds the most likely path
//! through the sequence `blank t1 blank t2 ... tN blank`, one state a
//! frame, which at each frame stays, steps to the next state, or skips a
//! blank between two different tokens (the Viterbi algorithm). A token's
//! frames are where the path is on it.
//!
//! Seven additions:
//!
//! - **Pauses inside a word** ([`CtcSettings::inside_word`]): a frame of
//!   blank between two letters of a word costs where the voice pauses
//!   ([`voice_pauses`]), so a word is held over singing but does not
//!   bridge a silence.
//! - **Backing words** ([`Skip`]): the path may jump over a word's tokens
//!   (with a word break beside them) at a cost in log-probability, since
//!   the voice stem often leaves out, or changes, what is in parentheses.
//! - **Anchors** ([`Anchor`]): with LRC line times, a line's first token
//!   must be reached within a band of frames around its time, so a stretch
//!   the model hears badly cannot pull the rest of the song with it.
//! - **Pins** ([`CtcSettings::window_penalty`]): where a recogniser heard
//!   a line ([`crate::anchors`], a window of 20 to 30 s), its word costs
//!   to start outside it, so a stretch (an intro's chant) cannot take the
//!   lines of the verse it sounds like.
//! - **The opening pulled together** ([`CtcSettings::opening_pull`]): the
//!   song is aligned once, the first words heard surely found (an island),
//!   and the words before them aligned again with a cost on the time
//!   between them, so a first line heard weakly stays near the lines after
//!   it rather than going to anything sung earlier.
//! - **Glimpsed line starts** ([`crate::glimpses::line_starts`]): in that
//!   second alignment, a line's first word costs to start outside the
//!   second or two where the recogniser's glimpses near it heard it
//!   start, so a "yeah, yeah, yo" before a line cannot take its "you".
//! - **Nothing heard** ([`CtcSettings::unheard`]): a line costs, from its
//!   first letter to its last, over a stretch where the glimpses heard no
//!   words, so its last words do not go on to a choir's "oh" or a synth
//!   left in the voice.
//!
//! The work is linear in frames × tokens; a 4-minute song (about 12,000
//! frames by 3,000 states) takes some tens of milliseconds and a byte of
//! back-pointer a frame and state.
//!
//! [`CtcAligner`] puts it together: the tokens, the model's emissions, the
//! path, each word's times and confidence (the mean posterior of its
//! frames), and the words under the confidence threshold timed between
//! their neighbours.

use std::ops::Range;

use crate::asr::Part;
use crate::spread;
use crate::tokens::{tokenize, TokenWord, Tokens, Vocab, WAV2VEC2_960H};
use crate::{Aligner, Detail, Lyrics, Maker, Timing, CONFIDENT, VERSION};

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

/// A token the path must reach within a stretch of frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Anchor {
    /// The token.
    pub token: usize,
    /// The frames in which the path may first reach it.
    pub frames: Range<usize>,
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
    /// Entering state `s` outside the frames of `enter[s]` costs its
    /// penalty.
    pub enter: Vec<Option<(Range<usize>, f32)>>,
    /// And outside those of `near[s]` its penalty too.
    pub near: Vec<Option<(Range<usize>, f32)>>,
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
        charged + spanned + self.always.get(s).copied().unwrap_or(0.0)
    }

    /// What entering state `s` at frame `f` costs.
    fn entry(&self, f: usize, s: usize) -> f32 {
        let outside = |of: &[Option<(Range<usize>, f32)>]| match of.get(s) {
            Some(Some((frames, penalty))) if !frames.contains(&f) => *penalty,
            _ => 0.0,
        };
        outside(&self.enter) + outside(&self.near)
    }
}

/// The most likely path of `ids` through `emissions`, with `blank` the
/// blank's index, leaving out `skips` at their costs, reaching each
/// anchor's token in its frames, and paying `costs`. `None` when no path
/// fits (more tokens than frames, or anchors that cannot all be met).
pub fn force_align(
    emissions: &Emissions,
    ids: &[u32],
    blank: u32,
    skips: &[Skip],
    anchors: &[Anchor],
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
    // The frames each state may be in: a state at or past an anchor's
    // token only from its band's start; a state before it only up to two
    // frames before the band's end, so the token is reached in the band.
    let mut earliest = vec![0i64; states];
    let mut latest = vec![i64::MAX; states];
    for anchor in anchors {
        if anchor.token >= n || anchor.frames.is_empty() {
            continue;
        }
        let at = 2 * anchor.token + 1;
        for e in &mut earliest[at..] {
            *e = (*e).max(anchor.frames.start as i64);
        }
        for l in &mut latest[..at] {
            *l = (*l).min(anchor.frames.end as i64 - 2);
        }
    }
    let allowed = |s: usize, f: usize| (earliest[s]..=latest[s]).contains(&(f as i64));
    let mut score = vec![f32::NEG_INFINITY; states];
    let mut next = vec![f32::NEG_INFINITY; states];
    let mut back = vec![STAY; frames * states];
    // The path starts on the first blank or the first token.
    for (s, first) in score.iter_mut().enumerate().take(2) {
        if allowed(s, 0) {
            *first = emissions.at(0, token_of(s)) - cost(0, s) - costs.entry(0, s);
        }
    }
    for f in 1..frames {
        let row = &mut back[f * states..(f + 1) * states];
        for s in 0..states {
            next[s] = f32::NEG_INFINITY;
            if !allowed(s, f) {
                continue;
            }
            let mut best = score[s];
            let mut how = STAY;
            let entry = costs.entry(f, s);
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

/// How the CTC alignment goes.
#[derive(Clone, Debug, PartialEq)]
pub struct CtcSettings {
    /// The cost of leaving out a backing word, in log-probability.
    pub skip_penalty: f32,
    /// How far a line's first word may start from its LRC time, in
    /// seconds.
    pub anchor_band: f64,
    /// Words under this confidence are timed between their neighbours.
    pub confident: f32,
    /// Below this ([`align_heard`]'s measure) the model has not heard the
    /// words well enough: lyrics with line times keep them,
    /// split over their words ([`spread::split_lines`]).
    pub trusted: f32,
    /// The cost of a second of a pause in the voice inside a word, between
    /// two of its letters, in log-probability. A word is sung in one go:
    /// held over singing it costs nothing (a long note, where the model
    /// hears little but its first letters), but its letters are not
    /// spread over a pause (letters heard in an intro's leftovers, a word
    /// bridging silence to where the next is heard). Between words, a
    /// pause is free.
    pub inside_word: f32,
    /// The cost of a word starting outside the stretch a recogniser heard
    /// it in ([`Hints::windows`]), in log-probability; 0 for none. It keeps
    /// a stretch the model hears badly (an intro's chant taken for the
    /// first verse) from shifting the lines with it.
    pub window_penalty: f32,
    /// The cost of a second off the letters (on a blank or a word break,
    /// within words too, or a word could stretch over the gap) from the
    /// first word to the first island (words heard surely,
    /// [`CtcSettings::island_words`] in a row), in log-probability; 0 for
    /// none. The song is aligned once, then again with this cost, so the
    /// opening's lines stay near the lines after them: a song's silence
    /// before its first words is free, and without it a first line heard
    /// weakly may go to anything sung earlier.
    pub opening_pull: f32,
    /// The cost of a second of a line (from its first letter to its last)
    /// over a stretch where the recogniser's glimpses heard no words
    /// ([`crate::glimpses::unheard`]), in log-probability; 0 for none. A
    /// line is sung in one go: its last words do not go on to a choir's
    /// "oh" or a synth left in the voice seconds after the singing.
    pub unheard: f32,
    /// The cost of a line's first word starting outside where the
    /// recogniser's glimpses near it heard it start
    /// ([`crate::glimpses::line_starts`]), in log-probability; 0 for none.
    pub glimpse_penalty: f32,
    /// How many words in a row make an island.
    pub island_words: usize,
    /// The least mean probability of a word's letters, on the frames the
    /// path has them, for it to be heard surely.
    pub island_heard: f32,
}

impl Default for CtcSettings {
    fn default() -> Self {
        CtcSettings {
            skip_penalty: 4.0,
            anchor_band: 1.5,
            confident: CONFIDENT,
            trusted: TRUSTED,
            inside_word: INSIDE_WORD,
            window_penalty: WINDOW_PENALTY,
            opening_pull: OPENING_PULL,
            unheard: UNHEARD,
            glimpse_penalty: GLIMPSE_PENALTY,
            island_words: ISLAND_WORDS,
            island_heard: ISLAND_HEARD,
        }
    }
}

/// [`CtcSettings::window_penalty`]'s default. On 138 songs (English and
/// German, 6,800 lines) the pins, with the opening pull, timed 42 lines
/// within 1 s of LRCLIB's that were not and 5 the other way (a stretch
/// shifted 6 to 58 s, first lines 8 to 25 s off); first lines over 3 s off
/// went from 22 to 16. At 100 it made no difference. Soft, since a
/// recogniser's windows are not always right.
pub const WINDOW_PENALTY: f32 = 30.0;

/// [`CtcSettings::opening_pull`]'s default. On its own, on the same songs,
/// it timed 6 lines right (first lines 7 to 25 s off, the rest of the
/// song already right) and none wrong; at 2 a second the same.
pub const OPENING_PULL: f32 = 1.0;

/// [`CtcSettings::unheard`]'s default. On 153 songs (100 English, 53
/// German; with the glimpse pins) the lines within 1 s of LRCLIB's: 2 a
/// second 82.5 % and 81.3 % (of the songs whose LRCLIB times fit), 5 82.6 %
/// and 81.3 %, 10 82.5 % and 81.2 %; none 82.0 % and 81.2 %. The lines
/// ending over 1 s after the next line's time: 447 and 251 with none, 420
/// and 256 at 5 (before the glimpses 466 and 253).
pub const UNHEARD: f32 = 5.0;

/// [`CtcSettings::glimpse_penalty`]'s default. On the same songs, beside
/// the transcript's pins: 0 82.2 % and 81.3 %, 5 82.8 % and 81.2 %, 10
/// 82.8 % and 81.2 %, 20 82.6 % and 81.3 %, 30 82.3 % and 81.0 % (before
/// the glimpses 81.4 % and 81.3 %); 30 where the glimpses heard a line's
/// first three words and less where fewer did, no better. Lower than the
/// transcript's: a glimpse is heard with less around it, and German heard
/// as English more often.
pub const GLIMPSE_PENALTY: f32 = 20.0;

/// [`CtcSettings::island_words`]'s default.
pub const ISLAND_WORDS: usize = 4;

/// [`CtcSettings::island_heard`]'s default.
pub const ISLAND_HEARD: f32 = 0.5;

/// The longest gap, in seconds, between two words of an island: its words
/// are sung together.
pub const ISLAND_GAP: f64 = 2.0;

/// What is known of the voice besides the model's emissions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hints {
    /// Frame by frame, where the voice pauses ([`voice_pauses`]); empty
    /// when that is not known, and then a word may be held over anything.
    pub pauses: Vec<bool>,
    /// For each word of the lyrics (line after line), the seconds within
    /// which a recogniser heard it start, if it did ([`crate::anchors`]).
    pub windows: Vec<Option<(f64, f64)>>,
    /// What a recogniser heard in short glimpses of the voice
    /// ([`crate::glimpses`]); empty when it was not heard so.
    pub glimpses: Vec<Part>,
}

/// [`CtcSettings::inside_word`]'s default. On 134 songs (English and
/// German) it keeps the lines' starts where they were (81 % within 1 s of
/// LRCLIB's), corrects first lines that bridged a pause (11 to 25 s early),
/// and keeps held notes; from 5 to 20 a second it makes no difference, and
/// a cost on every frame inside a word, sung or not, pushed held notes
/// away.
pub const INSIDE_WORD: f32 = 5.0;

/// [`CtcSettings::trusted`]'s default: songs wav2vec2 times well hear their
/// voices' letters at 0.44–0.74 (AURORA), one it cannot (two-part
/// harmonies, from 0.2 s) at 0.25.
pub const TRUSTED: f32 = 0.35;

/// `lyrics` timed by `emissions` in `vocab`: each spelt word's start and
/// end from the path, its confidence the mean posterior of its frames;
/// left-out and unspellable words, and words under the threshold, timed
/// between their neighbours, by `hints` of the voice. `None` when no path
/// fits.
pub fn align_emissions(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
) -> Option<Lyrics> {
    align_heard(emissions, lyrics, vocab, settings, hints).map(|(lyrics, _)| lyrics)
}

/// [`align_emissions`], and how well the model heard the words: the mean
/// probability of the letters on the frames the path has them (the blanks
/// between are likely anywhere, so they tell nothing).
pub fn align_heard(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
) -> Option<(Lyrics, f32)> {
    let tokens: Tokens = tokenize(lyrics, vocab);
    let fs = emissions.frame_seconds;
    let frames = emissions.frames();
    let skips: Vec<Skip> = tokens
        .words
        .iter()
        .filter_map(|w| w.skip.clone())
        .map(|tokens| Skip {
            tokens,
            penalty: settings.skip_penalty,
        })
        .collect();
    let band = (settings.anchor_band / fs).round() as i64;
    let mut anchors = Vec::new();
    let mut anchored_line = None;
    for w in &tokens.words {
        let (Some(range), Some(time)) = (&w.tokens, lyrics.lines[w.line].time) else {
            continue;
        };
        if anchored_line == Some(w.line) {
            continue;
        }
        anchored_line = Some(w.line);
        let centre = (time / fs).round() as i64;
        let start = (centre - band).clamp(0, frames as i64) as usize;
        let end = (centre + band + 1).clamp(0, frames as i64) as usize;
        // A time too near the end for the rest of the text to follow it
        // (one for another, longer version) cannot be met.
        let rest = tokens.ids.len() - range.start;
        if start < end && start + rest <= frames {
            anchors.push(Anchor {
                token: range.start,
                frames: start..end,
            });
        }
    }
    // The blanks inside a word (state 2i between its letters i - 1 and i)
    // cost on the frames where the voice pauses.
    let inside = settings.inside_word * fs as f32;
    let ids = &tokens.ids;
    let states = 2 * ids.len() + 1;
    let sep = vocab.separator;
    let mut costs = Costs {
        charged: (0..states)
            .map(|s| {
                let i = s / 2;
                let between =
                    s % 2 == 0 && i > 0 && i < ids.len() && ids[i - 1] != sep && ids[i] != sep;
                if between {
                    inside
                } else {
                    0.0
                }
            })
            .collect(),
        charging: hints.pauses.clone(),
        ..Costs::default()
    };
    // Each word's window, where a recogniser heard it start.
    let flat_start: Vec<usize> = lyrics
        .lines
        .iter()
        .scan(0, |at, l| {
            let here = *at;
            *at += l.words.len();
            Some(here)
        })
        .collect();
    let window_of = |w: &TokenWord| {
        hints
            .windows
            .get(flat_start[w.line] + w.word)
            .copied()
            .flatten()
    };
    // A word heard there costs to start elsewhere.
    if settings.window_penalty > 0.0 && !hints.windows.is_empty() {
        costs.enter = vec![None; states];
        for w in &tokens.words {
            let (Some(range), Some((a, b))) = (&w.tokens, window_of(w)) else {
                continue;
            };
            costs.enter[2 * range.start + 1] =
                Some((frames_within(a, b, fs, frames), settings.window_penalty));
        }
    }
    let lines = line_tokens(&tokens);
    if settings.unheard > 0.0 && !hints.glimpses.is_empty() {
        let cost = settings.unheard * fs as f32;
        costs.spanned = spanned(lyrics, &lines, states, cost);
        costs.unheard = crate::glimpses::unheard(&hints.glimpses, frames, fs);
    }
    let align = |anchors: &[Anchor], costs: &Costs| {
        force_align(emissions, ids, vocab.blank, &skips, anchors, costs)
    };
    let mut path = align(&anchors, &costs).or_else(|| align(&[], &costs))?;
    // Aligned again: with each line's start where the glimpses heard it,
    // near where this path put it, and the opening, up to the first island,
    // pulled together (a second between its words costs).
    let mut again = false;
    if settings.glimpse_penalty > 0.0 && !hints.glimpses.is_empty() {
        let spans = path.token_frames(ids.len());
        let mut first = vec![None; lyrics.lines.len()];
        for &(l, a, _) in &lines {
            first[l] = spans[a].as_ref().map(|r| r.start as f64 * fs);
        }
        let starts = crate::glimpses::line_starts(lyrics, &hints.glimpses, &first);
        costs.near = vec![None; states];
        for w in tokens.words.iter().filter(|w| w.word == 0) {
            let (Some(range), Some(start)) = (&w.tokens, starts[w.line]) else {
                continue;
            };
            // Within the part the transcript has it in, if it has it (whose
            // pin stays).
            let (c, d) = window_of(w).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            let (a, b) = (start.from.max(c), start.to.min(d));
            if a < b {
                costs.near[2 * range.start + 1] =
                    Some((frames_within(a, b, fs, frames), settings.glimpse_penalty));
                again = true;
            }
        }
    }
    if settings.opening_pull > 0.0 {
        if let Some((first, island)) =
            first_island(&path, emissions, &tokens, sep, &window_of, settings)
        {
            let pull = settings.opening_pull * fs as f32;
            costs.always = (0..states)
                .map(|s| {
                    let inside = s > 2 * first + 1 && s < 2 * island + 1;
                    if inside && !on_letter(ids, sep, s) {
                        pull
                    } else {
                        0.0
                    }
                })
                .collect();
            again = true;
        }
    }
    if again {
        if let Some(p) = align(&anchors, &costs).or_else(|| align(&[], &costs)) {
            path = p;
        }
    }
    let spans = path.token_frames(tokens.ids.len());
    let mut out = lyrics.clone();
    for w in &tokens.words {
        let word = &mut out.lines[w.line].words[w.word];
        word.start = None;
        word.end = None;
        word.confidence = 0.0;
        let Some(range) = &w.tokens else {
            continue;
        };
        let (Some(first), Some(last)) = (&spans[range.start], &spans[range.end - 1]) else {
            continue;
        };
        let frames = first.start..last.end;
        let sum: f32 = frames
            .clone()
            .map(|f| {
                let s = path.states[f] as usize;
                let token = if s % 2 == 1 {
                    tokens.ids[s / 2]
                } else {
                    vocab.blank
                };
                emissions.at(f, token).exp()
            })
            .sum();
        word.start = Some(frames.start as f64 * fs);
        word.end = Some(frames.end as f64 * fs);
        word.confidence = sum / frames.len() as f32;
        word.letters = letter_times(
            &word.text,
            range.clone(),
            &tokens.ids,
            &spans,
            vocab,
            fs,
            frames.start,
        );
    }
    spread::interpolate(&mut out.lines, settings.confident);
    out.timing = Timing {
        detail: Detail::Words,
        by: None,
        fit: None,
    };
    let (sum, count) = path
        .states
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s % 2 == 1)
        .map(|(f, &s)| emissions.at(f, tokens.ids[s as usize / 2]).exp())
        .fold((0.0f32, 0usize), |(sum, n), p| (sum + p, n + 1));
    Some((out, sum / count.max(1) as f32))
}

/// Each line with spelt words: the line, its first token and its last.
fn line_tokens(tokens: &Tokens) -> Vec<(usize, usize, usize)> {
    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for w in &tokens.words {
        let Some(range) = &w.tokens else {
            continue;
        };
        match out.last_mut() {
            Some(last) if last.0 == w.line => last.2 = range.end - 1,
            _ => out.push((w.line, range.start, range.end - 1)),
        }
    }
    out
}

/// The frames from `a` to `b` seconds (a frame either side), of `frames`.
fn frames_within(a: f64, b: f64, fs: f64, frames: usize) -> Range<usize> {
    let from = (a / fs).floor().max(0.0) as usize;
    let to = ((b / fs).ceil().max(0.0) as usize).min(frames) + 1;
    from..to
}

/// Where nothing with words was heard, what a frame on each of `states`
/// costs: `cost` on every state of a line (`lines`, [`line_tokens`]) from
/// its first letter to its last, none on a line of nothing but "ooh"s,
/// which is sung where no words are.
fn spanned(lyrics: &Lyrics, lines: &[(usize, usize, usize)], states: usize, cost: f32) -> Vec<f32> {
    let mut out = vec![0.0; states];
    for &(l, a, b) in lines {
        if lyrics.lines[l]
            .words
            .iter()
            .all(|w| crate::glimpses::vocable(&w.text))
        {
            continue;
        }
        for c in &mut out[2 * a + 1..2 * b + 2] {
            *c = cost;
        }
    }
    out
}

/// Whether state `s` of `ids` is on a letter (not a blank, nor a word
/// break).
fn on_letter(ids: &[u32], sep: u32, s: usize) -> bool {
    s % 2 == 1 && ids[s / 2] != sep
}

/// Where `path` first hears the words surely: the first token of the
/// lyrics, and that of the first of [`CtcSettings::island_words`] words in
/// a row whose letters it hears at [`CtcSettings::island_heard`] or more
/// (their mean probability on the frames it has them) and none of which
/// starts outside the window a recogniser heard it in (`window_of`).
/// `None` when there is no such run.
fn first_island(
    path: &Path,
    emissions: &Emissions,
    tokens: &Tokens,
    sep: u32,
    window_of: &dyn Fn(&TokenWord) -> Option<(f64, f64)>,
    settings: &CtcSettings,
) -> Option<(usize, usize)> {
    let ids = &tokens.ids;
    let spans = path.token_frames(ids.len());
    let fs = emissions.frame_seconds;
    let heard = |range: &Range<usize>| {
        let (mut sum, mut n) = (0.0f32, 0usize);
        for k in range.clone().filter(|&k| ids[k] != sep) {
            for f in spans[k].clone().unwrap_or_default() {
                sum += emissions.at(f, ids[k]).exp();
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    };
    let mut first = None;
    let (mut run, mut island, mut last_end) = (0, 0, f64::NEG_INFINITY);
    for w in &tokens.words {
        let Some(range) = &w.tokens else {
            continue;
        };
        first.get_or_insert(range.start);
        let start = spans[range.start].as_ref().map(|r| r.start as f64 * fs);
        let end = spans[range.end - 1].as_ref().map(|r| r.end as f64 * fs);
        let placed = match (start, window_of(w)) {
            (None, _) => false,
            (Some(t), Some((a, b))) => a <= t && t <= b,
            (Some(_), None) => true,
        };
        if placed && heard(range) >= settings.island_heard {
            // Sung together: a gap breaks the run.
            if run > 0 && start.unwrap_or(0.0) - last_end > ISLAND_GAP {
                run = 0;
            }
            last_end = end.unwrap_or(f64::NEG_INFINITY);
            if run == 0 {
                island = range.start;
            }
            run += 1;
            if run >= settings.island_words {
                return Some((first?, island));
            }
        } else {
            run = 0;
        }
    }
    None
}

/// When each character of `text` is sung, in seconds after the word's first
/// frame `first`: a letter from the first frame the path has its token on
/// (the word's tokens are `ids[range]`, in order, apostrophes and breaks
/// among them); anything else (punctuation, a letter the path has no frame
/// for) at the character's before. Never earlier than the one before.
fn letter_times(
    text: &str,
    range: Range<usize>,
    ids: &[u32],
    spans: &[Option<Range<usize>>],
    vocab: &Vocab,
    fs: f64,
    first: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(text.chars().count());
    let mut next = range.start;
    let mut last = 0.0f32;
    for c in text.chars() {
        let spelt = crate::tokens::latin(c);
        let id = spelt
            .and_then(|s| s.chars().next())
            .and_then(|l| vocab.id(l));
        if let (Some(spelt), Some(id)) = (spelt, id) {
            let at = (next..range.end).find(|&k| ids[k] == id);
            if let Some(k) = at {
                if let Some(span) = &spans[k] {
                    let t = (span.start.saturating_sub(first)) as f64 * fs;
                    last = last.max(t as f32);
                }
                next = (k + spelt.len()).min(range.end);
            }
        }
        out.push(last);
    }
    out
}

/// For each of `emissions`' frames, whether the voice (`vocals_16k`, which
/// they were made from) pauses there: outside every
/// [`spread::vocal_stretches`].
pub fn voice_pauses(vocals_16k: &[f32], emissions: &Emissions) -> Vec<bool> {
    spread::pauses(
        &spread::vocal_stretches(vocals_16k),
        emissions.frames(),
        emissions.frame_seconds,
    )
}

/// A speech model giving CTC emissions for the voice.
pub trait Acoustic {
    /// Its name and version.
    fn maker(&self) -> Maker;

    /// Its vocabulary.
    fn vocab(&self) -> Vocab {
        WAV2VEC2_960H
    }

    /// Its emissions for the voice at [`crate::VOCALS_RATE`]; `None` when
    /// it cannot run.
    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions>;
}

/// A model shared between threads is a model.
impl<A: Acoustic + ?Sized> Acoustic for std::sync::Arc<A> {
    fn maker(&self) -> Maker {
        (**self).maker()
    }

    fn vocab(&self) -> Vocab {
        (**self).vocab()
    }

    fn emissions(&self, vocals_16k: &[f32]) -> Option<Emissions> {
        (**self).emissions(vocals_16k)
    }
}

/// The name and version a [`CtcAligner`] over a model made by `model`
/// records in its results: the model's, and this crate's timing
/// ([`VERSION`]). A timing kept by another is made again.
pub fn aligner_maker(model: &Maker) -> Maker {
    Maker::new(format!("ctc:{}@{}", model.name, model.version), VERSION)
}

/// The CTC aligner: a model's emissions, then [`align_emissions`]. Without
/// emissions, or with no path, the times come from [`spread::fallback`].
pub struct CtcAligner<A> {
    /// The model.
    pub model: A,
    /// How the alignment goes.
    pub settings: CtcSettings,
}

impl<A: Acoustic> CtcAligner<A> {
    /// An aligner over `model` with the default settings.
    pub fn new(model: A) -> CtcAligner<A> {
        CtcAligner {
            model,
            settings: CtcSettings::default(),
        }
    }
}

impl<A: Acoustic> Aligner for CtcAligner<A> {
    fn maker(&self) -> Maker {
        aligner_maker(&self.model.maker())
    }

    fn align(&self, vocals_16k: &[f32], lyrics: &Lyrics) -> Lyrics {
        self.align_heard_in(vocals_16k, lyrics, &[], &[])
    }
}

impl<A: Acoustic> CtcAligner<A> {
    /// [`Aligner::align`], with what a recogniser heard in the voice (its
    /// transcript's `parts` and its `glimpses`, none when it was not heard)
    /// pinning the words to where it heard them ([`crate::anchors`],
    /// [`crate::glimpses`]).
    pub fn align_heard_in(
        &self,
        vocals_16k: &[f32],
        lyrics: &Lyrics,
        parts: &[Part],
        glimpses: &[Part],
    ) -> Lyrics {
        let aligned = self.model.emissions(vocals_16k).and_then(|e| {
            let hints = Hints {
                pauses: voice_pauses(vocals_16k, &e),
                windows: crate::anchors::windows(lyrics, parts),
                glimpses: glimpses.to_vec(),
            };
            align_heard(&e, lyrics, &self.model.vocab(), &self.settings, &hints)
        });
        let lined = lyrics.lines.iter().any(|l| l.time.is_some());
        match aligned {
            // Heard too poorly: the line times are surer.
            Some((_, heard)) if lined && heard < self.settings.trusted => {
                spread::split_lines(lyrics)
            }
            Some((mut out, _)) => {
                out.timing.by = Some(self.maker());
                out
            }
            None => spread::fallback(vocals_16k, lyrics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::normalize;
    use crate::{Line, Source};

    const FS: f64 = 0.02;

    /// Emissions of `frames` frames where the planted token of each frame
    /// (the blank where none is) has probability `p` and the others share
    /// the rest.
    fn planted(frames: usize, plants: &[(usize, u32)], p: f32) -> Emissions {
        let mut on = vec![WAV2VEC2_960H.blank; frames];
        for &(f, t) in plants {
            on[f] = t;
        }
        emissions_of(&on, p)
    }

    fn emissions_of(on: &[u32], p: f32) -> Emissions {
        let v = WAV2VEC2_960H.len();
        let other = ((1.0 - p) / (v - 1) as f32).ln();
        let mut logp = vec![other; on.len() * v];
        for (f, &t) in on.iter().enumerate() {
            logp[f * v + t as usize] = p.ln();
        }
        Emissions {
            logp,
            tokens: v,
            frame_seconds: FS,
        }
    }

    /// Plants `word`'s letters from `frame`, `each` frames a letter with a
    /// blank between, then a word break after; gives the frame after.
    fn plant(on: &mut [u32], word: &str, frame: usize, each: usize) -> usize {
        let mut f = frame;
        let letters = normalize(word).unwrap();
        for (i, c) in letters.chars().enumerate() {
            if i > 0 {
                f += 1; // a blank
            }
            for _ in 0..each {
                on[f] = WAV2VEC2_960H.id(c).unwrap();
                f += 1;
            }
        }
        f
    }

    fn lyrics(lines: &[&str], times: &[Option<f64>]) -> Lyrics {
        Lyrics {
            lines: lines
                .iter()
                .zip(times)
                .map(|(text, &time)| Line {
                    time,
                    ..Line::from_text(text)
                })
                .collect(),
            source: Source::Unknown,
            timing: Timing::default(),
        }
    }

    fn times(l: &Lyrics) -> Vec<(f64, f64)> {
        l.lines
            .iter()
            .flat_map(|l| &l.words)
            .map(|w| (w.start.unwrap(), w.end.unwrap()))
            .collect()
    }

    fn close(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x.0 - y.0).abs() < 1e-9 && (x.1 - y.1).abs() < 1e-9)
    }

    #[test]
    fn the_path_finds_planted_word_times_exactly() {
        let mut on = vec![WAV2VEC2_960H.blank; 400];
        let sep = WAV2VEC2_960H.separator;
        // "Hello world, all ok": words at planted frames, a break between.
        let mut planted_at = Vec::new();
        let mut f = 20;
        for (word, gap) in [("Hello", 15), ("world,", 40), ("all", 7), ("ok", 0)] {
            let end = plant(&mut on, word, f, 2);
            planted_at.push((f as f64 * FS, end as f64 * FS));
            if gap > 0 {
                on[end + gap / 2] = sep;
            }
            f = end + gap;
        }
        let e = emissions_of(&on, 0.9);
        let text = lyrics(&["Hello world,", "all ok"], &[None, None]);
        let out = align_emissions(
            &e,
            &text,
            &WAV2VEC2_960H,
            &CtcSettings::default(),
            &Hints::default(),
        )
        .unwrap();
        assert!(close(&times(&out), &planted_at), "{:?}", times(&out));
        assert!(out
            .lines
            .iter()
            .flat_map(|l| &l.words)
            .all(|w| w.confidence > 0.85));
        assert_eq!(out.timing.detail, Detail::Words);
        // Each letter where it was planted (three frames apart); the comma,
        // not sung, at the letter's before it.
        let words: Vec<&crate::Word> = out.lines.iter().flat_map(|l| &l.words).collect();
        let hundredths = |w: &crate::Word| -> Vec<u32> {
            w.letters
                .iter()
                .map(|t| (t * 100.0).round() as u32)
                .collect()
        };
        assert_eq!(hundredths(words[0]), [0, 6, 12, 18, 24]);
        assert_eq!(hundredths(words[1]), [0, 6, 12, 18, 24, 24]);
        let third = words[0].letter_at(2).unwrap();
        assert!((third - (20.0 + 6.0) * FS).abs() < 1e-6, "{third}");
    }

    #[test]
    fn repeated_letters_need_a_blank_between() {
        // "LL" can only be two tokens with a blank between.
        let l = WAV2VEC2_960H.id('L').unwrap();
        let e = planted(3, &[(0, l), (2, l)], 0.9);
        let path = force_align(&e, &[l, l], 0, &[], &[], &Costs::default()).unwrap();
        assert_eq!(path.states, [1, 2, 3]);
        let e = planted(1, &[(0, l)], 0.9);
        assert!(force_align(&e, &[l, l], 0, &[], &[], &Costs::default()).is_none());
    }

    #[test]
    fn a_word_is_held_over_singing_but_not_over_a_pause() {
        // "go": a sure G at frame 10 and an O at 200, and a weaker G at
        // 196. From the first G the word is surer; it is a held note
        // where the voice sings on, and letters heard in a pause where it
        // does not.
        let (g, o) = (
            WAV2VEC2_960H.id('G').unwrap(),
            WAV2VEC2_960H.id('O').unwrap(),
        );
        let mut e = planted(300, &[(10, g), (200, o)], 0.9);
        let v = WAV2VEC2_960H.len();
        e.logp[196 * v + g as usize] = 0.5f32.ln();
        e.logp[196 * v + WAV2VEC2_960H.blank as usize] = 0.4f32.ln();
        let text = lyrics(&["go"], &[None]);
        let settings = CtcSettings::default();
        let start = |pauses: &[bool]| {
            let hints = Hints {
                pauses: pauses.to_vec(),
                ..Hints::default()
            };
            align_emissions(&e, &text, &WAV2VEC2_960H, &settings, &hints)
                .unwrap()
                .lines[0]
                .words[0]
                .start
                .unwrap()
        };
        assert_eq!(start(&[]), 10.0 * FS);
        assert_eq!(start(&[false; 300]), 10.0 * FS);
        let mut pauses = vec![false; 300];
        pauses[11..196].fill(true);
        assert_eq!(start(&pauses), 196.0 * FS);
        // Between words, a pause is free.
        let mut on = vec![WAV2VEC2_960H.blank; 300];
        let a = plant(&mut on, "go", 10, 2);
        on[a + 1] = WAV2VEC2_960H.separator;
        let b = plant(&mut on, "on", 200, 2);
        let e = emissions_of(&on, 0.9);
        let out = align_emissions(
            &e,
            &lyrics(&["go on"], &[None]),
            &WAV2VEC2_960H,
            &settings,
            &Hints {
                pauses: vec![true; 300],
                ..Hints::default()
            },
        )
        .unwrap();
        assert!(close(
            &times(&out),
            &[(10.0 * FS, a as f64 * FS), (200.0 * FS, b as f64 * FS)]
        ));
    }

    /// Frame `f` of `e` with `token` at probability `p`, the blank at
    /// `blank`, the other tokens sharing the rest.
    fn set(e: &mut Emissions, f: usize, token: u32, p: f32, blank: f32) {
        let v = WAV2VEC2_960H.len();
        let other = ((1.0 - p - blank) / (v - 2) as f32).ln();
        for t in 0..v {
            e.logp[f * v + t] = other;
        }
        e.logp[f * v + token as usize] = p.ln();
        e.logp[f * v + WAV2VEC2_960H.blank as usize] = blank.ln();
    }

    /// Plants `word` in `e` from `frame`, two frames a letter and a blank
    /// between, each letter at `p` and the blank under it at `blank`; gives
    /// the frame after.
    fn plant_at(e: &mut Emissions, word: &str, frame: usize, p: f32, blank: f32) -> usize {
        let mut f = frame;
        for (i, c) in normalize(word).unwrap().chars().enumerate() {
            if i > 0 {
                f += 1;
            }
            for _ in 0..2 {
                set(e, f, WAV2VEC2_960H.id(c).unwrap(), p, blank);
                f += 1;
            }
        }
        f
    }

    #[test]
    fn a_word_heard_elsewhere_costs_its_pin() {
        // "sun" heard surely at 0.4 s (a sound like it) and less surely
        // at 6 s, where a recogniser heard it.
        let mut e = planted(400, &[], 0.9);
        plant_at(&mut e, "sun", 20, 0.9, 0.09);
        plant_at(&mut e, "sun", 300, 0.4, 0.5);
        let text = lyrics(&["sun"], &[None]);
        let start = |settings: &CtcSettings, windows: Vec<Option<(f64, f64)>>| {
            let hints = Hints {
                windows,
                ..Hints::default()
            };
            align_emissions(&e, &text, &WAV2VEC2_960H, settings, &hints)
                .unwrap()
                .lines[0]
                .words[0]
                .start
                .unwrap()
        };
        let settings = CtcSettings::default();
        assert_eq!(start(&settings, vec![]), 20.0 * FS);
        assert_eq!(start(&settings, vec![Some((5.0, 7.0))]), 300.0 * FS);
        // Unpinned, or a window that holds the sure one, changes nothing.
        let free = CtcSettings {
            window_penalty: 0.0,
            ..settings.clone()
        };
        assert_eq!(start(&free, vec![Some((5.0, 7.0))]), 20.0 * FS);
        assert_eq!(start(&settings, vec![Some((0.0, 2.0))]), 20.0 * FS);
    }

    #[test]
    fn the_opening_stays_near_the_lines_after_it() {
        // "go", then "one two three four" sung surely from 8 s. "go" is
        // heard a little more surely at 0.4 s (something sung earlier)
        // than at 7.4 s, just before the rest.
        let mut e = planted(600, &[], 0.9);
        plant_at(&mut e, "go", 20, 0.7, 0.25);
        plant_at(&mut e, "go", 370, 0.6, 0.35);
        let mut f = 400;
        for word in ["one", "two", "three", "four"] {
            f = plant_at(&mut e, word, f, 0.95, 0.04);
            set(&mut e, f + 1, WAV2VEC2_960H.separator, 0.95, 0.04);
            f += 4;
        }
        let text = lyrics(&["go", "one two three four"], &[None, None]);
        let start = |settings: &CtcSettings| {
            align_emissions(&e, &text, &WAV2VEC2_960H, settings, &Hints::default())
                .unwrap()
                .lines[0]
                .words[0]
                .start
                .unwrap()
        };
        let loose = CtcSettings {
            opening_pull: 0.0,
            ..CtcSettings::default()
        };
        assert_eq!(start(&loose), 20.0 * FS);
        assert_eq!(start(&CtcSettings::default()), 370.0 * FS);
    }

    fn glimpse(start: f64, text: &str) -> Part {
        Part {
            start,
            end: start + crate::glimpses::GLIMPSE,
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_line_starts_where_the_glimpses_heard_it_not_on_the_chant_before() {
        // "it hurts" at 1 s; a chant's "yo" at 3 s heard surely as "you";
        // the line "do you wanna" sung at 4.6 s, its first two words heard
        // less surely.
        let mut e = planted(500, &[], 0.9);
        let mut f = plant_at(&mut e, "it", 50, 0.95, 0.04);
        plant_at(&mut e, "hurts", f + 2, 0.95, 0.04);
        plant_at(&mut e, "you", 150, 0.95, 0.04);
        f = plant_at(&mut e, "do", 230, 0.5, 0.45);
        f = plant_at(&mut e, "you", f + 2, 0.5, 0.45);
        plant_at(&mut e, "wanna", f + 2, 0.95, 0.04);
        let text = lyrics(&["it hurts", "do you wanna"], &[None, None]);
        let start = |glimpses: Vec<Part>| {
            let hints = Hints {
                glimpses,
                ..Hints::default()
            };
            align_emissions(&e, &text, &WAV2VEC2_960H, &CtcSettings::default(), &hints)
                .unwrap()
                .lines[1]
                .words[0]
                .start
                .unwrap()
        };
        assert!(start(Vec::new()) < 3.0, "{}", start(Vec::new()));
        let heard = vec![
            glimpse(0.0, "It hurts."),
            glimpse(1.5, "It hurts. Yeah, yeah."),
            glimpse(3.0, "Yeah. Do you wanna?"),
            glimpse(4.5, "Do you wanna?"),
            glimpse(6.0, ""),
        ];
        assert_eq!(start(heard), 230.0 * FS);
    }

    #[test]
    fn a_line_does_not_go_on_over_a_stretch_with_no_words_heard() {
        // "come home" sung at 1 s; an "oh" at 6 s, where nothing with words
        // was sung, sounds more like "home" than the real one.
        let mut e = planted(500, &[], 0.9);
        let f = plant_at(&mut e, "come", 50, 0.95, 0.04);
        plant_at(&mut e, "home", f + 2, 0.5, 0.45);
        plant_at(&mut e, "home", 300, 0.6, 0.35);
        let text = lyrics(&["come home"], &[None]);
        let end = |glimpses: Vec<Part>| {
            let hints = Hints {
                glimpses,
                ..Hints::default()
            };
            align_emissions(&e, &text, &WAV2VEC2_960H, &CtcSettings::default(), &hints)
                .unwrap()
                .lines[0]
                .words[1]
                .end
                .unwrap()
        };
        assert!(end(Vec::new()) > 6.0);
        let heard = vec![
            glimpse(0.0, "Come home."),
            glimpse(1.5, ""),
            glimpse(3.0, ""),
            glimpse(4.5, ""),
            glimpse(6.0, ""),
            glimpse(7.5, "Go."),
        ];
        assert!(end(heard) < 2.0);
    }

    #[test]
    fn a_backing_word_not_sung_is_left_out_and_filled_in() {
        let mut on = vec![WAV2VEC2_960H.blank; 300];
        let sep = WAV2VEC2_960H.separator;
        let a = plant(&mut on, "come", 10, 2);
        on[a + 5] = sep;
        let b = plant(&mut on, "home", a + 40, 2);
        let e = emissions_of(&on, 0.9);
        let text = lyrics(&["come (yeah yeah) home"], &[None]);
        let out = align_emissions(
            &e,
            &text,
            &WAV2VEC2_960H,
            &CtcSettings::default(),
            &Hints::default(),
        )
        .unwrap();
        let w = &out.lines[0].words;
        assert_eq!(
            (w[0].start, w[0].end),
            (Some(10.0 * FS), Some(a as f64 * FS))
        );
        assert_eq!(
            (w[3].start, w[3].end),
            (Some((a + 40) as f64 * FS), Some(b as f64 * FS))
        );
        // Left out: timed between, with no confidence.
        for backing in &w[1..3] {
            assert_eq!(backing.confidence, 0.0);
            let (s, e) = (backing.start.unwrap(), backing.end.unwrap());
            assert!(s >= w[0].end.unwrap() && e <= w[3].start.unwrap() && s < e);
        }
        // Sung, the same words are found.
        let mut on2 = vec![WAV2VEC2_960H.blank; 300];
        let a = plant(&mut on2, "come", 10, 2);
        on2[a + 2] = sep;
        let y1 = plant(&mut on2, "yeah", a + 5, 2);
        on2[y1 + 2] = sep;
        let y2 = plant(&mut on2, "yeah", y1 + 5, 2);
        on2[y2 + 2] = sep;
        plant(&mut on2, "home", y2 + 5, 2);
        let e = emissions_of(&on2, 0.9);
        let out = align_emissions(
            &e,
            &text,
            &WAV2VEC2_960H,
            &CtcSettings::default(),
            &Hints::default(),
        )
        .unwrap();
        let w = &out.lines[0].words;
        assert_eq!(w[1].start, Some((a + 5) as f64 * FS));
        assert_eq!(w[2].start, Some((y1 + 5) as f64 * FS));
        assert!(w[1].confidence > 0.8);
    }

    #[test]
    fn line_anchors_hold_a_corrupted_stretch() {
        // Three one-word lines at 1 s, 5 s and 9 s. The second line's
        // stretch is garbled, and at 7 s the voice stem carries an echo of
        // the third line's word, surer than the word itself (no letter of
        // it in the second's, so only the third word can take it).
        let mut on = vec![WAV2VEC2_960H.blank; 600];
        let sep = WAV2VEC2_960H.separator;
        let first = plant(&mut on, "sun", 50, 2);
        on[first + 3] = sep;
        plant(&mut on, "rain", 250, 2);
        let third_end = plant(&mut on, "fog", 450, 2);
        let echo_end = plant(&mut on, "fog", 350, 2);
        let mut e = emissions_of(&on, 0.6);
        let v = WAV2VEC2_960H.len();
        for (f, &token) in on.iter().enumerate().take(echo_end).skip(350) {
            let t = token as usize;
            if t != WAV2VEC2_960H.blank as usize {
                for k in 0..v {
                    e.logp[f * v + k] = (0.01 / (v - 1) as f32).ln();
                }
                e.logp[f * v + t] = 0.99f32.ln();
            }
        }
        for f in 240..270 {
            for t in 0..v {
                e.logp[f * v + t] = (1.0 / v as f32).ln();
            }
        }
        let timed = lyrics(&["sun", "rain", "fog"], &[Some(1.0), Some(5.0), Some(9.0)]);
        let untimed = lyrics(&["sun", "rain", "fog"], &[None, None, None]);
        let settings = CtcSettings::default();
        let start = |l: &Lyrics, i: usize| l.lines[i].words[0].start.unwrap();
        // Without the line times, the echo takes the third word.
        let loose =
            align_emissions(&e, &untimed, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        assert!(
            (start(&loose, 2) - 7.0).abs() < 1e-9,
            "{}",
            start(&loose, 2)
        );
        // With them, each line stays by its own time; the garbled word is
        // filled in from its line's time.
        let held =
            align_emissions(&e, &timed, &WAV2VEC2_960H, &settings, &Hints::default()).unwrap();
        assert!((start(&held, 0) - 1.0).abs() < 1e-9);
        assert!((start(&held, 1) - 5.0).abs() < 1e-9);
        assert!(held.lines[1].words[0].confidence < CONFIDENT);
        assert!((start(&held, 2) - 9.0).abs() < 1e-9);
        assert!((held.lines[2].words[0].end.unwrap() - third_end as f64 * FS).abs() < 1e-9);
        assert!(held.lines[2].words[0].confidence > 0.5);
    }

    #[test]
    fn anchors_that_cannot_be_met_fall_back_to_a_free_path() {
        let mut on = vec![WAV2VEC2_960H.blank; 200];
        plant(&mut on, "hi", 20, 2);
        let e = emissions_of(&on, 0.9);
        // The second line's time before the first's: no path meets both.
        let text = lyrics(&["hi", "ho"], &[Some(3.5), Some(0.5)]);
        let out = align_emissions(
            &e,
            &text,
            &WAV2VEC2_960H,
            &CtcSettings::default(),
            &Hints::default(),
        )
        .unwrap();
        assert_eq!(out.lines[0].words[0].start, Some(20.0 * FS));
        // Too few frames for the tokens: no path at all.
        let short = emissions_of(&on[..2], 0.9);
        assert!(align_emissions(
            &short,
            &text,
            &WAV2VEC2_960H,
            &CtcSettings::default(),
            &Hints::default()
        )
        .is_none());
    }

    struct Planted(Emissions);

    impl Acoustic for Planted {
        fn maker(&self) -> Maker {
            Maker::new("planted", 7)
        }
        fn emissions(&self, _: &[f32]) -> Option<Emissions> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn the_aligner_records_itself() {
        let mut on = vec![WAV2VEC2_960H.blank; 200];
        plant(&mut on, "hi", 20, 2);
        let aligner = CtcAligner::new(Planted(emissions_of(&on, 0.9)));
        let out = aligner.align(&[], &lyrics(&["hi"], &[None]));
        assert_eq!(out.timing.by, Some(Maker::new("ctc:planted@7", VERSION)));
        assert_eq!(out.lines[0].words[0].start, Some(20.0 * FS));
    }
}
