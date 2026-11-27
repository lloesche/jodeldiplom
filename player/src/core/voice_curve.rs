//! The voice stem's level in Karaoke mode, word by word: the vocals heard
//! at the Voice slider's level, but over a word marked as not to be sung
//! (a backing vocal, an ad-lib the singer leaves to the record) turned up
//! [`NOT_SUNG_RAISE_DB`] (and to at least [`NOT_SUNG_LEAST_DB`], so it is
//! heard with the voice all the way down), and over a word with a level of
//! its own ([`jodel_lyrics::WordEdit::voice_db`]) at that level; each change
//! smoothed over [`SMOOTH`] before the word starts and after it ends.
//!
//! On the mixer it is the vocals part's automation curve on the playing
//! head's input ([`VoiceCurve`], [`jodel_output::mixer::Automation`]),
//! timed on its stream frames by the last position the engine told and its
//! speed through the track ([`crate::core::Playing::frame_at`]); the
//! vocals lane's own gain is then 1, the curve carrying the slider's level
//! too, so a voice turned all the way down still lets a word not sung be
//! heard. The mixer thread runs a curve it is given without allocating or
//! locking. It is in force only in the mode, with no transition armed or
//! mixing (whose automation the head's would be: entering the mode calls
//! one armed off), and for a song with such words; it follows every
//! change of the slider, the song, the words, and a seek (until two
//! positions tell the speed again, one gain at the slider's level).

use jodel_lyrics::{Lyrics, Word};

use super::desk::taper;
use super::transition::Status;
use super::{karaoke, sliders, Effect, State};

/// How much a word not to be sung turns the voice up over the slider's
/// level, in dB.
pub const NOT_SUNG_RAISE_DB: f64 = 6.0;

/// The least level of the voice over a word not to be sung, in dB against
/// the voice as recorded.
pub const NOT_SUNG_LEAST_DB: f64 = -6.0;

/// How long the voice takes to change its level, in seconds: before a
/// word with a level of its own starts, and after it ends.
pub const SMOOTH: f64 = 0.05;

/// How far, in seconds, the frames a curve was timed on may drift from
/// where the positions told now put them before it is timed again.
const DRIFT: f64 = 0.005;

/// The voice's level over a word in the mode, and why, in dB against the
/// voice as recorded ([`voice_level`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VoiceLevel {
    /// A level set over it.
    Set(f64),
    /// The slider's raised, as it is not to be sung.
    Raised(f64),
    /// The Voice slider's.
    Slider(f64),
}

impl VoiceLevel {
    /// The level.
    pub fn db(self) -> f64 {
        match self {
            VoiceLevel::Set(db) | VoiceLevel::Raised(db) | VoiceLevel::Slider(db) => db,
        }
    }

    /// Whether it is the word's own (set over it, or raised as it is not
    /// sung), not the slider's.
    pub fn is_own(self) -> bool {
        !matches!(self, VoiceLevel::Slider(_))
    }
}

/// The voice's level over `word` in the mode, `voice_db` the slider's:
/// the level set over it, else raised over a word not to be sung, else the
/// slider's.
pub fn voice_level(word: &Word, voice_db: f64) -> VoiceLevel {
    match (word.edit.voice_db, word.edit.not_sung) {
        (Some(db), _) => VoiceLevel::Set(f64::from(db)),
        (None, true) => VoiceLevel::Raised((voice_db + NOT_SUNG_RAISE_DB).max(NOT_SUNG_LEAST_DB)),
        (None, false) => VoiceLevel::Slider(voice_db),
    }
}

/// The voice's level over each word with one of its own, in order and none
/// overlapping the one before: from, to (seconds) and the level in dB
/// against the voice as recorded; `voice_db` the slider's.
pub fn spans(lyrics: &Lyrics, voice_db: f64) -> Vec<(f64, f64, f64)> {
    let mut out: Vec<(f64, f64, f64)> = Vec::new();
    for w in lyrics.lines.iter().flat_map(|l| &l.words) {
        let level = voice_level(w, voice_db);
        if !level.is_own() {
            continue;
        }
        let db = level.db();
        let (Some(a), Some(b)) = (w.start, w.end) else {
            continue;
        };
        let a = out.last().map_or(a, |last| a.max(last.1));
        if b > a {
            out.push((a, b, db));
        }
    }
    out
}

/// The voice's gain over the song as breakpoints (seconds, linear gain), in
/// order: the slider's level (`voice_db`), the [`spans`]' levels over
/// their words, each change over [`SMOOTH`] (two words closer than that
/// go from one level to the next between them). Empty when no word has a
/// level of its own.
pub fn levels(lyrics: &Lyrics, voice_db: f64) -> Vec<(f64, f64)> {
    let base = taper::gain(voice_db);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (a, b, db) in spans(lyrics, voice_db) {
        let g = taper::gain(db);
        match out.last().copied() {
            Some((end, _)) if a - SMOOTH < end + SMOOTH => {
                out.push((a.max((end + SMOOTH).min(b)), g));
            }
            Some((end, _)) => {
                out.push((end + SMOOTH, base));
                out.push((a - SMOOTH, base));
                out.push((a, g));
            }
            None => {
                out.push((a - SMOOTH, base));
                out.push((a, g));
            }
        }
        out.push((b, g));
    }
    if let Some(&(end, _)) = out.last() {
        out.push((end + SMOOTH, base));
    }
    out
}

/// `levels` on the stream frames of an output that has 0 s of the track
/// on frame `zero` and goes through it at `speed` frames a second: what the
/// mixer is given ([`VoiceCurve::Set`]). The frames before the output's
/// first are its first, the gain there as the curve has it.
pub fn on_frames(levels: &[(f64, f64)], zero: f64, speed: f64) -> Vec<(u64, f64)> {
    let mut out: Vec<(u64, f64)> = Vec::with_capacity(levels.len());
    let mut before: Option<(f64, f64)> = None;
    for &(t, g) in levels {
        let f = zero + t * speed;
        if f < 0.0 {
            before = Some((f, g));
            continue;
        }
        if let Some((f0, g0)) = before.take() {
            // The gain at frame 0, between the last point before it and
            // this one.
            let share = -f0 / (f - f0).max(1e-9);
            out.push((0, g0 + (g - g0) * share));
        }
        out.push((f.round() as u64, g));
    }
    if let (true, Some((_, g))) = (out.is_empty(), before) {
        out.push((0, g));
    }
    out
}

/// What the mixer is given for the voice: the vocals part's curve on a
/// head's input, or none.
#[derive(Clone, Debug, PartialEq)]
pub enum VoiceCurve {
    /// The vocals heard through `points` (stream frame, gain) of `head`'s
    /// output, its vocals lane at 1.
    Set {
        /// The head ([`crate::core::HeadFrame::head`]).
        head: usize,
        /// The curve's points.
        points: Vec<(u64, f64)>,
    },
    /// The head's automation ended (every curve back to 1).
    Clear {
        /// The head.
        head: usize,
    },
}

/// What the mixer was last given for the voice, so it is given a curve
/// again only when that changes: the head and track, a hash of the levels,
/// and how they were put on frames (`None`: one gain, the speed not known).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceSent {
    head: usize,
    index: usize,
    levels: u64,
    mapped: Option<Mapped>,
}

/// How a curve's levels were put on an output's stream frames.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mapped {
    /// The frame of 0 s into the track.
    zero: f64,
    /// Frames a second of the track.
    speed: f64,
    /// The first and last times the levels have, in seconds.
    span: (f64, f64),
}

impl VoiceSent {
    /// Whether `now` times the same curve on the same frames, give or take
    /// [`DRIFT`].
    fn same(&self, now: &VoiceSent) -> bool {
        let close = match (self.mapped, now.mapped) {
            (None, None) => true,
            (Some(was), Some(is)) => [was.span.0, was.span.1].iter().all(|t| {
                ((is.zero + t * is.speed) - (was.zero + t * was.speed)).abs() <= DRIFT * is.speed
            }),
            _ => false,
        };
        (self.head, self.index, self.levels) == (now.head, now.index, now.levels) && close
    }
}

/// The voice's levels in force: the head and the track they are for, and
/// the levels ([`levels`]).
struct Due {
    head: usize,
    index: usize,
    levels: Vec<(f64, f64)>,
}

/// The voice's levels in force now: in the mode, no transition armed or
/// mixing, the song playing told on a head, its lyrics with words of their
/// own levels.
fn due(state: &State) -> Option<Due> {
    let draft = &state.transition.draft;
    if !karaoke::in_mode(state) || draft.status != Status::Draft || draft.in_force().is_some() {
        return None;
    }
    let index = state.playing.track.as_ref()?.index;
    let head = state.playing.anchor.filter(|a| a.index == index)?.at.head;
    let Some(karaoke::SongLyrics::Found { shown, .. }) = karaoke::lyrics_of(state, index) else {
        return None;
    };
    let levels = levels(shown, sliders::voice_db(state.config.karaoke_voice));
    (!levels.is_empty()).then_some(Due {
        head,
        index,
        levels,
    })
}

/// Whether the voice's levels are in force ([`due`]): the vocals lane is
/// then at 1, the curve carrying the slider's level.
pub fn in_force(state: &State) -> bool {
    due(state).is_some()
}

/// After any action: the mixer given the voice's curve when it changed
/// (another song, other levels, the slider moved, a seek), or told it is
/// gone.
pub(crate) fn follow(state: &mut State, fx: &mut Vec<Effect>) {
    let sent = state.stems_sent.voice;
    let Some(Due {
        head,
        index,
        levels,
    }) = due(state)
    else {
        if let Some(s) = sent {
            fx.push(Effect::VoiceCurve(VoiceCurve::Clear { head: s.head }));
            state.stems_sent.voice = None;
        }
        return;
    };
    let mut hash = jodel_dsp::Fnv1a::new();
    for (t, g) in &levels {
        hash.write(&t.to_bits().to_le_bytes());
        hash.write(&g.to_bits().to_le_bytes());
    }
    let hash = hash.finish();
    let anchor = state.playing.anchor.filter(|a| a.index == index);
    let span = (levels[0].0, levels[levels.len() - 1].0);
    let mapped = state.playing.speed.zip(anchor).map(|(speed, a)| Mapped {
        zero: a.at.frame as f64 - a.seconds * speed,
        speed,
        span,
    });
    let now = VoiceSent {
        head,
        index,
        levels: hash,
        mapped,
    };
    if sent.is_some_and(|s| s.same(&now)) {
        return;
    }
    if let Some(s) = sent.filter(|s| s.head != head) {
        fx.push(Effect::VoiceCurve(VoiceCurve::Clear { head: s.head }));
    }
    let points = match mapped {
        Some(m) => on_frames(&levels, m.zero, m.speed),
        None => vec![(
            0,
            taper::gain(sliders::voice_db(state.config.karaoke_voice)),
        )],
    };
    fx.push(Effect::VoiceCurve(VoiceCurve::Set { head, points }));
    state.stems_sent.voice = Some(now);
}

#[cfg(test)]
mod tests {
    use super::*;
    use jodel_lyrics::Source;

    fn song() -> Lyrics {
        let mut l = Lyrics::parse_text("one two three four\n", Source::Unknown);
        for (k, w) in l.lines[0].words.iter_mut().enumerate() {
            (w.start, w.end) = (Some(k as f64), Some(k as f64 + 0.5));
        }
        l
    }

    #[test]
    fn a_word_not_sung_turns_the_voice_up_smoothly() {
        let mut l = song();
        assert!(levels(&l, -12.0).is_empty());
        l.lines[0].words[1].edit.not_sung = true;
        let base = taper::gain(-12.0);
        let up = taper::gain(-6.0);
        let got = levels(&l, -12.0);
        assert_eq!(
            got,
            [
                (1.0 - SMOOTH, base),
                (1.0, up),
                (1.5, up),
                (1.5 + SMOOTH, base)
            ]
        );
        // With the voice all the way down, it is heard all the same; with
        // the voice up, 6 dB over it.
        assert_eq!(
            spans(&l, f64::NEG_INFINITY),
            [(1.0, 1.5, NOT_SUNG_LEAST_DB)]
        );
        assert_eq!(spans(&l, -3.0), [(1.0, 1.5, 3.0)]);
        // A level of its own wins.
        l.lines[0].words[1].edit.voice_db = Some(-20.0);
        assert_eq!(spans(&l, -3.0), [(1.0, 1.5, -20.0)]);
    }

    #[test]
    fn words_close_together_go_from_one_level_to_the_next() {
        let mut l = song();
        let w = &mut l.lines[0].words;
        (w[1].edit.not_sung, w[2].edit.voice_db) = (true, Some(0.0));
        w[2].start = Some(1.52);
        let got = levels(&l, -12.0);
        let up = taper::gain(-6.0);
        // From the first word's level straight to the next's, over SMOOTH.
        assert_eq!(
            got[..3],
            [(1.0 - SMOOTH, taper::gain(-12.0)), (1.0, up), (1.5, up)]
        );
        assert_eq!(got[3], (1.5 + SMOOTH, 1.0));
        assert!(got.windows(2).all(|p| p[0].0 <= p[1].0));
    }

    #[test]
    fn levels_land_on_their_frames_and_none_before_the_first() {
        let levels = [(-1.0, 0.5), (1.0, 1.5), (2.0, 1.5)];
        assert_eq!(
            on_frames(&levels, 0.0, 100.0),
            [(0, 1.0), (100, 1.5), (200, 1.5)]
        );
        assert_eq!(
            on_frames(&levels, 1_000.0, 100.0),
            [(900, 0.5), (1_100, 1.5), (1_200, 1.5)]
        );
        assert_eq!(on_frames(&[(-2.0, 0.5)], 0.0, 100.0), [(0, 0.5)]);
    }
}
