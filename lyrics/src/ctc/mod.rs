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
//! Eight additions:
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
//!   [`align_held`] is given, made by [`crate::scaffold`], which aligns
//!   the song without them first to check them).
//! - **Words set by hand** ([`crate::WordEdit::pin`]): a word moved by hand
//!   is held to its times, all but hard ([`PINNED`] a frame off them), and
//!   nothing else may be sung there, so the words around it fit around
//!   it; it keeps its times exactly, and its letters are read from the
//!   path within them.
//!
//! The work is linear in frames × tokens; a 4-minute song (about 12,000
//! frames by 3,000 states) takes some tens of milliseconds and a byte of
//! back-pointer a frame and state.
//!
//! [`align_held`] puts it together: the tokens, the costs (one source a
//! function, in `costs`), the path, each word's times and confidence (the
//! mean posterior of its frames), and the words under the confidence
//! threshold timed between their neighbours; [`crate::aligner::CtcAligner`]
//! runs it on a model's emissions.

use std::ops::Range;

use crate::asr::Part;
use crate::spread;
use crate::tokens::{tokenize, TokenWord, Tokens, Vocab};
use crate::{Detail, Lyrics, Timing, CONFIDENT};

mod costs;
mod path;

pub use path::{force_align, Costs, Emissions, FrameClock, Path, Skip, StayCost};
use path::{state_count, states_of, token_at};

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
    /// source's synced times hold it in ([`align_held`]'s `lines`,
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

/// What a frame costs off a word's times set by hand
/// ([`crate::WordEdit::pin`]): its letters on a frame outside them, or
/// another word's on a frame inside them. Hard, in effect: a frame of the
/// least likely letter costs some tens.
pub const PINNED: f32 = 1000.0;

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
    align_held(emissions, lyrics, vocab, settings, hints, &[])
}

/// [`align_emissions`], with each line's words held within `lines` (for
/// each line of the lyrics, the seconds where its source's synced times
/// say it is sung, [`crate::scaffold::held`]; empty for none:
/// [`CtcSettings::scaffold`]).
pub fn align_held(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
    lines: &[Option<(f64, f64)>],
) -> Option<Lyrics> {
    align_path(emissions, lyrics, vocab, settings, hints, lines).map(|(lyrics, _)| lyrics)
}

/// [`align_held`], and the path the words were read from (how well the
/// model heard them on it: [`heard`]).
pub fn align_path(
    emissions: &Emissions,
    lyrics: &Lyrics,
    vocab: &Vocab,
    settings: &CtcSettings,
    hints: &Hints,
    lines: &[Option<(f64, f64)>],
) -> Option<(Lyrics, Path)> {
    let job = Job::new(emissions, lyrics, vocab, settings, hints);
    let mut costs = Costs {
        stay: Vec::new(),
        entry: vec![Vec::new(); job.states()],
    };
    // A frame's costs are added up in the order they are put in, and a sum
    // of floats depends on its order: keep this one.
    costs::pauses(&job, &mut costs);
    costs::heard_windows(&job, &mut costs);
    costs::unheard(&job, &mut costs);
    costs::scaffold(&job, &mut costs, lines);
    costs::pinned(&job, &mut costs);
    let skips = job.skips();
    let align = |costs: &Costs| force_align(emissions, &job.tokens.ids, vocab.blank, &skips, costs);
    let mut path = align(&costs)?;
    // Aligned again: with each line's start where the glimpses heard it,
    // near where this path put it, and the opening, up to the first island,
    // pulled together.
    let near = costs::glimpsed(&job, &mut costs, &path);
    let pulled = costs::opening(&job, &mut costs, &path);
    if near || pulled {
        if let Some(p) = align(&costs) {
            path = p;
        }
    }
    Some((job.read(&path), path))
}

/// How well the model heard the words of `lyrics` on `path` (through
/// `emissions`, as [`align_path`] found it): the mean probability of the
/// letters on the frames the path has them (the blanks between are likely
/// anywhere, so they tell nothing).
pub fn heard(emissions: &Emissions, lyrics: &Lyrics, vocab: &Vocab, path: &Path) -> f32 {
    let ids = tokenize(lyrics, vocab).ids;
    let (sum, count) = (path.states.iter().enumerate())
        .filter_map(|(f, &s)| Some((f, token_at(s as usize)?)))
        .map(|(f, k)| emissions.at(f, ids[k]).exp())
        .fold((0.0f32, 0usize), |(sum, n), p| (sum + p, n + 1));
    sum / count.max(1) as f32
}

/// What one alignment works with: the lyrics as tokens, and what is known
/// of the voice.
struct Job<'a> {
    emissions: &'a Emissions,
    lyrics: &'a Lyrics,
    vocab: &'a Vocab,
    settings: &'a CtcSettings,
    hints: &'a Hints,
    /// The emissions' frames.
    clock: FrameClock,
    tokens: Tokens,
    /// Each line with spelt words: the line, and its tokens from its first
    /// word's first to its last word's last ([`line_tokens`]).
    lines: Vec<(usize, Range<usize>)>,
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
        Job {
            emissions,
            lyrics,
            vocab,
            settings,
            hints,
            clock: emissions.clock(),
            tokens,
            lines,
        }
    }

    fn states(&self) -> usize {
        state_count(self.tokens.ids.len())
    }

    /// Where a recogniser heard word `k` (among all the lyrics' words)
    /// start, if it did ([`Hints::windows`]).
    fn heard_window(&self, k: usize) -> Option<(f64, f64)> {
        self.hints.windows.get(k).copied().flatten()
    }

    /// Where nothing with words was heard, what a frame on each state
    /// costs: `cost` on every state of a line ([`Job::lines`]) from its
    /// first letter to its last, none on a line of nothing but "ooh"s,
    /// which is sung where no words are.
    fn spanned(&self, cost: f32) -> Vec<f32> {
        let mut out = vec![0.0; self.states()];
        for (l, tokens) in &self.lines {
            if self.lyrics.lines[*l]
                .words
                .iter()
                .all(|w| crate::glimpses::vocable(&w.text))
            {
                continue;
            }
            out[states_of(tokens.clone())].fill(cost);
        }
        out
    }

    /// Where `path` first hears the words surely: the first token of the
    /// lyrics, and that of the first of [`CtcSettings::island_words`] words
    /// in a row whose letters it hears at [`CtcSettings::island_heard`] or
    /// more (their mean probability on the frames it has them) and none of
    /// which starts outside the window a recogniser heard it in
    /// ([`Job::heard_window`]). `None` when there is no such run.
    fn first_island(&self, path: &Path) -> Option<(usize, usize)> {
        let (emissions, settings, clock) = (self.emissions, self.settings, self.clock);
        let ids = &self.tokens.ids;
        let sep = self.vocab.separator;
        let spans = path.token_frames(ids.len());
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
        for (k, w) in self.tokens.words.iter().enumerate() {
            let Some(range) = &w.tokens else {
                continue;
            };
            first.get_or_insert(range.start);
            let start = spans[range.start].as_ref().map(|r| clock.time(r.start));
            let end = spans[range.end - 1].as_ref().map(|r| clock.time(r.end));
            let placed = match (start, self.heard_window(k)) {
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

    /// The times word `w` was set to by hand, if it was
    /// ([`crate::WordEdit::pin`]).
    fn pin(&self, w: &TokenWord) -> Option<(f64, f64)> {
        self.lyrics.word(w.at).and_then(|word| word.edit.pin)
    }

    /// The backing words the path may leave out (not one set by hand).
    fn skips(&self) -> Vec<Skip> {
        self.tokens
            .words
            .iter()
            .filter(|w| self.pin(w).is_none())
            .filter_map(|w| w.skip.clone())
            .map(|tokens| Skip {
                tokens,
                penalty: self.settings.skip_penalty,
            })
            .collect()
    }

    /// The lyrics timed by `path`.
    fn read(&self, path: &Path) -> Lyrics {
        let (emissions, tokens, vocab, clock) =
            (self.emissions, &self.tokens, self.vocab, self.clock);
        let spans = path.token_frames(tokens.ids.len());
        let mut out = self.lyrics.clone();
        for w in &tokens.words {
            let word = &mut out[w.at];
            word.clear_times();
            let on_path = w.tokens.as_ref().and_then(|range| {
                let (first, last) = (spans[range.start].as_ref()?, spans[range.end - 1].as_ref()?);
                Some((range, first.start..last.end))
            });
            if let Some((range, frames)) = on_path {
                let sum: f32 = frames
                    .clone()
                    .map(|f| {
                        let s = path.states[f] as usize;
                        let token = token_at(s).map_or(vocab.blank, |k| tokens.ids[k]);
                        emissions.at(f, token).exp()
                    })
                    .sum();
                word.start = Some(clock.time(frames.start));
                word.end = Some(clock.time(frames.end));
                word.confidence = sum / frames.len() as f32;
                // Set by hand: its letters read from its own start.
                let first = word
                    .edit
                    .pin
                    .map_or(frames.start, |(a, _)| clock.nearest(a));
                word.letters = letter_times(
                    &word.text,
                    range.clone(),
                    &tokens.ids,
                    &spans,
                    vocab,
                    clock,
                    first,
                );
            }
            // Set by hand: its own times, whatever the path has for it
            // (none where it is unspelt or left out), its letters in them.
            if word.keep_pin() {
                word.clip_letters();
            }
        }
        spread::interpolate(&mut out.lines, self.settings.confident);
        out.timing = Timing {
            detail: Detail::Words,
            by: None,
            fit: None,
        };
        out
    }
}

/// Each line with spelt words: the line, and its tokens from its first
/// word's first to its last word's last.
fn line_tokens(tokens: &Tokens) -> Vec<(usize, Range<usize>)> {
    let mut out: Vec<(usize, Range<usize>)> = Vec::new();
    for w in &tokens.words {
        let Some(range) = &w.tokens else {
            continue;
        };
        match out.last_mut() {
            Some((line, spelt)) if *line == w.at.line => spelt.end = range.end,
            _ => out.push((w.at.line, range.clone())),
        }
    }
    out
}

/// When each character of `text` is sung, in seconds after the word's first
/// frame `first` of `clock`: a letter from the first frame the path has its
/// token on (the word's tokens are `ids[range]`, in order, apostrophes and
/// breaks among them); anything else (punctuation, a letter the path has no
/// frame for) at the character's before. Never earlier than the one before.
fn letter_times(
    text: &str,
    range: Range<usize>,
    ids: &[u32],
    spans: &[Option<Range<usize>>],
    vocab: &Vocab,
    clock: FrameClock,
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
                    let t = clock.time(span.start.saturating_sub(first));
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
    spread::pauses(&spread::vocal_stretches(vocals_16k), emissions.clock())
}

#[cfg(test)]
pub(crate) mod tests;
