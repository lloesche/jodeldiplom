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
//! - **The source's line times** ([`CtcSettings::scaffold`]): where a
//!   synced text's times fit the recording, each line's words cost to
//!   start, and to be, outside the window around its time (the `lines`
//!   [`align_heard`] is given, made by [`crate::scaffold`], which aligns
//!   the song without them first to check them).
//!
//! The work is linear in frames × tokens; a 4-minute song (about 12,000
//! frames by 3,000 states) takes some tens of milliseconds and a byte of
//! back-pointer a frame and state.
//!
//! [`align_heard`] puts it together: the tokens, the costs, the path, each
//! word's times and confidence (the mean posterior of its frames), and the
//! words under the confidence threshold timed between their neighbours;
//! [`crate::aligner::CtcAligner`] runs it on a model's emissions.

use std::ops::Range;

use crate::asr::Part;
use crate::spread;
use crate::tokens::{tokenize, TokenWord, Tokens, Vocab};
use crate::{Detail, Lyrics, Timing, CONFIDENT};

mod path;

pub use path::{force_align, Costs, Emissions, Path, Skip};

/// How the CTC alignment goes.
#[derive(Clone, Debug, PartialEq)]
pub struct CtcSettings {
    /// The cost of leaving out a backing word, in log-probability.
    pub skip_penalty: f32,
    /// Words under this confidence are timed between their neighbours.
    pub confident: f32,
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
    /// The cost of a word of a line starting outside the window its
    /// source's synced times hold it in ([`align_heard`]'s `lines`,
    /// [`crate::scaffold`]), and of each second of the line (from its first
    /// letter to its last) outside it, in log-probability; 0 for none. A
    /// short line put far away pays for its words, a word held on past its
    /// line's end for the time.
    pub scaffold: f32,
}

impl Default for CtcSettings {
    fn default() -> Self {
        CtcSettings {
            skip_penalty: 4.0,
            confident: CONFIDENT,
            inside_word: INSIDE_WORD,
            window_penalty: WINDOW_PENALTY,
            opening_pull: OPENING_PULL,
            unheard: UNHEARD,
            glimpse_penalty: GLIMPSE_PENALTY,
            island_words: ISLAND_WORDS,
            island_heard: ISLAND_HEARD,
            scaffold: SCAFFOLD,
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

/// [`CtcSettings::scaffold`]'s default.
pub const SCAFFOLD: f32 = 10.0;

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
    align_heard(emissions, lyrics, vocab, settings, hints, &[]).map(|(lyrics, _)| lyrics)
}

/// [`align_emissions`], with each line's words held within `lines` (for
/// each line of the lyrics, the seconds where its source's synced times
/// say it is sung, [`crate::scaffold::held`]; empty for none:
/// [`CtcSettings::scaffold`]), and how well the model heard the words: the
/// mean probability of the letters on the frames the path has them (the
/// blanks between are likely anywhere, so they tell nothing).
pub fn align_heard(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
    lines: &[Option<(f64, f64)>],
) -> Option<(Lyrics, f32)> {
    let job = Job::new(emissions, lyrics, vocab, settings, hints);
    let mut costs = Costs {
        entry: vec![Vec::new(); job.states()],
        ..Costs::default()
    };
    pause_costs(&job, &mut costs);
    pin_costs(&job, &mut costs);
    scaffold_costs(&job, &mut costs, lines);
    unheard_costs(&job, &mut costs);
    let skips = job.skips();
    let align = |costs: &Costs| force_align(emissions, &job.tokens.ids, vocab.blank, &skips, costs);
    let mut path = align(&costs)?;
    // Aligned again: with each line's start where the glimpses heard it,
    // near where this path put it, and the opening, up to the first island,
    // pulled together.
    let near = glimpse_costs(&job, &mut costs, &path);
    let pulled = opening_costs(&job, &mut costs, &path);
    if near || pulled {
        if let Some(p) = align(&costs) {
            path = p;
        }
    }
    Some(job.read(&path))
}

/// What one alignment works with: the lyrics as tokens, and what is known
/// of the voice.
struct Job<'a> {
    emissions: &'a Emissions,
    lyrics: &'a Lyrics,
    vocab: &'a Vocab,
    settings: &'a CtcSettings,
    hints: &'a Hints,
    tokens: Tokens,
    /// Each line with spelt words: the line, its first token and its last
    /// ([`line_tokens`]).
    lines: Vec<(usize, usize, usize)>,
    /// Each line's first word's index among all the lyrics' words.
    flat_start: Vec<usize>,
}

impl<'a> Job<'a> {
    fn new(
        emissions: &'a Emissions,
        lyrics: &'a Lyrics,
        vocab: &'a Vocab,
        settings: &'a CtcSettings,
        hints: &'a Hints,
    ) -> Job<'a> {
        let tokens = tokenize(lyrics, vocab);
        let lines = line_tokens(&tokens);
        let flat_start = lyrics
            .lines
            .iter()
            .scan(0, |at, l| {
                let here = *at;
                *at += l.words.len();
                Some(here)
            })
            .collect();
        Job {
            emissions,
            lyrics,
            vocab,
            settings,
            hints,
            tokens,
            lines,
            flat_start,
        }
    }

    fn fs(&self) -> f64 {
        self.emissions.frame_seconds
    }

    fn frames(&self) -> usize {
        self.emissions.frames()
    }

    fn states(&self) -> usize {
        2 * self.tokens.ids.len() + 1
    }

    /// The frames from `a` to `b` seconds ([`frames_within`]).
    fn within(&self, a: f64, b: f64) -> Range<usize> {
        frames_within(a, b, self.fs(), self.frames())
    }

    /// Where a recogniser heard word `w` start, if it did
    /// ([`Hints::windows`]).
    fn heard_window(&self, w: &TokenWord) -> Option<(f64, f64)> {
        self.hints
            .windows
            .get(self.flat_start[w.line] + w.word)
            .copied()
            .flatten()
    }

    /// The backing words the path may leave out.
    fn skips(&self) -> Vec<Skip> {
        self.tokens
            .words
            .iter()
            .filter_map(|w| w.skip.clone())
            .map(|tokens| Skip {
                tokens,
                penalty: self.settings.skip_penalty,
            })
            .collect()
    }

    /// The lyrics timed by `path`, and how well the model heard them.
    fn read(&self, path: &Path) -> (Lyrics, f32) {
        let (emissions, tokens, vocab, fs) = (self.emissions, &self.tokens, self.vocab, self.fs());
        let spans = path.token_frames(tokens.ids.len());
        let mut out = self.lyrics.clone();
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
        spread::interpolate(&mut out.lines, self.settings.confident);
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
        (out, sum / count.max(1) as f32)
    }
}

/// The blanks inside a word (state 2i between its letters i - 1 and i)
/// cost where the voice pauses ([`CtcSettings::inside_word`]).
fn pause_costs(job: &Job, costs: &mut Costs) {
    let ids = &job.tokens.ids;
    let sep = job.vocab.separator;
    let inside = job.settings.inside_word * job.fs() as f32;
    costs.charged = (0..job.states())
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
        .collect();
    costs.charging = job.hints.pauses.clone();
}

/// A word a recogniser heard costs to start outside where it heard it
/// ([`CtcSettings::window_penalty`]).
fn pin_costs(job: &Job, costs: &mut Costs) {
    let penalty = job.settings.window_penalty;
    if penalty <= 0.0 || job.hints.windows.is_empty() {
        return;
    }
    for w in &job.tokens.words {
        let (Some(range), Some((a, b))) = (&w.tokens, job.heard_window(w)) else {
            continue;
        };
        costs.entry[2 * range.start + 1].push((job.within(a, b), penalty));
    }
}

/// A line's words cost to start outside its window in `windows`, and its
/// letters (and the blanks between them) to be there
/// ([`CtcSettings::scaffold`]).
fn scaffold_costs(job: &Job, costs: &mut Costs, windows: &[Option<(f64, f64)>]) {
    let penalty = job.settings.scaffold;
    if penalty <= 0.0 || windows.is_empty() {
        return;
    }
    let line_window = |l: usize| {
        let &(from, to) = windows.get(l)?.as_ref()?;
        Some(job.within(from, to))
    };
    let per_frame = penalty * job.fs() as f32;
    costs.outside = vec![None; job.states()];
    for &(l, a, b) in &job.lines {
        if let Some(within) = line_window(l) {
            for c in &mut costs.outside[2 * a + 1..2 * b + 2] {
                *c = Some((within.clone(), per_frame));
            }
        }
    }
    for w in &job.tokens.words {
        if let (Some(range), Some(within)) = (&w.tokens, line_window(w.line)) {
            costs.entry[2 * range.start + 1].push((within, penalty));
        }
    }
}

/// A line costs over a stretch where the glimpses heard no words
/// ([`CtcSettings::unheard`]).
fn unheard_costs(job: &Job, costs: &mut Costs) {
    if job.settings.unheard <= 0.0 || job.hints.glimpses.is_empty() {
        return;
    }
    let cost = job.settings.unheard * job.fs() as f32;
    costs.spanned = spanned(job.lyrics, &job.lines, job.states(), cost);
    costs.unheard = crate::glimpses::unheard(&job.hints.glimpses, job.frames(), job.fs());
}

/// Each line's first word costs to start outside where the glimpses near
/// where `path` put it heard it start, within the part the transcript has
/// it in when it has it ([`CtcSettings::glimpse_penalty`]). Whether any
/// does.
fn glimpse_costs(job: &Job, costs: &mut Costs, path: &Path) -> bool {
    let penalty = job.settings.glimpse_penalty;
    if penalty <= 0.0 || job.hints.glimpses.is_empty() {
        return false;
    }
    let fs = job.fs();
    let spans = path.token_frames(job.tokens.ids.len());
    let mut first = vec![None; job.lyrics.lines.len()];
    for &(l, a, _) in &job.lines {
        first[l] = spans[a].as_ref().map(|r| r.start as f64 * fs);
    }
    let starts = crate::glimpses::line_starts(job.lyrics, &job.hints.glimpses, &first);
    let mut any = false;
    for w in job.tokens.words.iter().filter(|w| w.word == 0) {
        let (Some(range), Some(start)) = (&w.tokens, starts[w.line]) else {
            continue;
        };
        let (c, d) = job
            .heard_window(w)
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        let (a, b) = (start.from.max(c), start.to.min(d));
        if a < b {
            costs.entry[2 * range.start + 1].push((job.within(a, b), penalty));
            any = true;
        }
    }
    any
}

/// The opening, from the first word to the first island `path` hears,
/// pulled together: every frame off the letters between costs
/// ([`CtcSettings::opening_pull`]). Whether there is one.
fn opening_costs(job: &Job, costs: &mut Costs, path: &Path) -> bool {
    if job.settings.opening_pull <= 0.0 {
        return false;
    }
    let ids = &job.tokens.ids;
    let sep = job.vocab.separator;
    let heard_window = |w: &TokenWord| job.heard_window(w);
    let Some((first, island)) = first_island(
        path,
        job.emissions,
        &job.tokens,
        sep,
        &heard_window,
        job.settings,
    ) else {
        return false;
    };
    let pull = job.settings.opening_pull * job.fs() as f32;
    costs.always = (0..job.states())
        .map(|s| {
            let inside = s > 2 * first + 1 && s < 2 * island + 1;
            if inside && !on_letter(ids, sep, s) {
                pull
            } else {
                0.0
            }
        })
        .collect();
    true
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
/// starts outside the window a recogniser heard it in (`heard_window`).
/// `None` when there is no such run.
fn first_island(
    path: &Path,
    emissions: &Emissions,
    tokens: &Tokens,
    sep: u32,
    heard_window: &dyn Fn(&TokenWord) -> Option<(f64, f64)>,
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
        let placed = match (start, heard_window(w)) {
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

#[cfg(test)]
pub(crate) mod tests;
