//! The singing scored: the notes of a song's lines (from the voice's
//! melody, under the words' letters: [`Notes`]) and the singer's pitch
//! against them, frame by frame, word by word, phrase by phrase.
//!
//! Each frame of the singer's pitch (from the microphone, at the time in
//! the song it was sung) is held against the note sung there, in any key:
//! the pitch is moved by how far, all along, the singer has sung from the
//! song (the median over the last [`KEY_FRAMES`]), octaves aside. A frame
//! within [`TOLERANCE`] (the melody's shape, 3 semitones) hits. A word
//! whose note time is hit for at least [`SHARE`] of it scores
//! [`WORD_POINTS`]. Words hit in a row are a streak, and the points a word
//! scores are times the streak's [`multiplier`] (a miss ends it).
//!
//! Each phrase (a sung line), once its frames are in, is measured and held
//! against the original singer's same phrase ([`crate::core::shine`], the
//! [`Reference`] set with [`Singing::set_reference`]): its shine and the
//! tier it reached. Without a reference the phrases have no shine, the
//! words are scored all the same. What happens (a word scored, a bubble
//! bursting, a shot rescuing a single word missed, the streak up or lost,
//! a phrase's tier) is one stream of moments ([`Singing::moments_since`]).
//!
//! The song played in another key ([`Singing::set_key`]) has its notes in
//! that key: they move by its semitones, so the singing is held against
//! what is heard.
//!
//! The singer's pitch as drawn ([`Heard::cents`]): a frame that hits sits
//! on its note; off the notes it is the pitch sung, its jitter smoothed (a
//! median over [`SMOOTH_FRAMES`], then a glide).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::Lyrics;
use jodel_rhythm::waveform::{Melody, Waveform};

use crate::core::shine::{self, Reference, Shone};

mod bursts;
pub mod judge;
mod moments;
mod notes;
mod result;
mod tally;
mod undo;

pub use bursts::Rescue;
pub use judge::{KeyStep, SungKey, KEY_FRAMES, SHARE, TOLERANCE};
pub use moments::{Burst, Cue, Cursor, Moment, Since};
pub use notes::{notes_of, LineNotes, Note, Notes};
pub use result::{Medal, SongResult};
pub use tally::{Run, Tally};

/// What a word hit scores.
pub const WORD_POINTS: u32 = 100;
/// How long after a word's end it is scored (the last frames of it come
/// in a little late).
const SCORED_AFTER: f64 = 0.3;
/// The multiplier of a streak of `streak` words hit in a row: 1 to 4.
pub fn multiplier(streak: u32) -> u32 {
    match streak {
        0..5 => 1,
        5..10 => 2,
        10..20 => 3,
        _ => 4,
    }
}

/// A word as it was scored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scored {
    /// When in the song it ended, in seconds.
    pub end: f64,
    /// Whether it was hit (else missed).
    pub hit: bool,
    /// The points it scored.
    pub points: u32,
    /// The streak after it.
    pub streak: u32,
}

/// The frames the pitch drawn as sung is the median of (the last 50 ms).
pub const SMOOTH_FRAMES: usize = 5;
/// How far each frame the pitch drawn as sung goes on to the median: half
/// way (a slide followed within about 30 ms).
const GLIDE: f64 = 0.5;

/// A frame of the singer's pitch, as held against the song.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heard {
    /// When in the song it was sung, in seconds.
    pub t: f64,
    /// The pitch as drawn, in hundredths of a MIDI note: moved into the
    /// song's key (and octave, by the note there), on the note where it
    /// hits.
    pub cents: f64,
    /// How far off the note it was (none off a note), in hundredths of a
    /// semitone: above the note when positive.
    pub off: Option<f64>,
    /// Whether it hit the note.
    pub hit: bool,
    /// Which line's note it was on.
    pub line: Option<usize>,
}

/// A waveform of only `melody` (the notes need no more of it).
pub fn melodic(melody: &Melody) -> Arc<Waveform> {
    Arc::new(Waveform {
        melody: Some(melody.clone()),
        ..Waveform::default()
    })
}

/// What the notes were made from: the song, its lyrics and its waveform
/// (told apart by identity, never by address alone).
#[derive(Clone, Debug)]
struct Source {
    path: PathBuf,
    lyrics: Arc<Lyrics>,
    waveform: Arc<Waveform>,
}

impl Source {
    fn is(&self, path: &std::path::Path, lyrics: &Arc<Lyrics>, waveform: &Arc<Waveform>) -> bool {
        self.path == path
            && Arc::ptr_eq(&self.lyrics, lyrics)
            && Arc::ptr_eq(&self.waveform, waveform)
    }
}

/// One song's singing, scored as it goes.
#[derive(Clone, Debug, Default)]
pub struct Singing {
    /// What the notes are of.
    source: Option<Source>,
    /// The notes.
    pub notes: Notes,
    /// The original singer's phrases the shine holds these to.
    reference: Option<Arc<Reference>>,
    /// The singer's pitch frames so far, in time.
    pub heard: Vec<Heard>,
    /// Every frame since the singing last started over, as the shine
    /// measures it.
    frames: Vec<shine::Frame>,
    /// What happened so far ([`Singing::moments_since`]), and what it
    /// tells of the bubbles (those burst, the words rescued).
    log: moments::Log,
    /// How far the singer sings from the song (octaves aside).
    sung_key: SungKey,
    /// Each word's frames on a note, and those that hit.
    counts: HashMap<(usize, usize), (u32, u32)>,
    /// How far off the last frames sung were, and the last drawn as sung,
    /// for the pitch drawn as sung.
    recent: VecDeque<f64>,
    glide: Option<f64>,
    /// When the last frame (sung or not) was, in the song.
    last: Option<f64>,
    /// The scoring so far.
    pub tally: Tally,
    /// What each of the last frames did, to be taken back when sought back
    /// a little ([`Singing::hear`]).
    undo: undo::Log,
}

impl Singing {
    /// Ready for `path`'s singing with its `lyrics` (their sung lines in
    /// order) and the melody of its `waveform`: its notes made anew when
    /// any of them is another (and its scoring started over, its reference
    /// let go), in the key they were in.
    pub fn prepare(
        &mut self,
        path: &std::path::Path,
        lyrics: &Arc<Lyrics>,
        waveform: &Arc<Waveform>,
    ) {
        if self
            .source
            .as_ref()
            .is_some_and(|s| s.is(path, lyrics, waveform))
        {
            return;
        }
        let shift = self.notes.shift();
        self.notes = waveform
            .melody
            .as_ref()
            .map_or_else(Notes::default, |m| Notes::new(lyrics, m));
        self.notes.set_key(shift);
        self.source = Some(Source {
            path: path.to_owned(),
            lyrics: Arc::clone(lyrics),
            waveform: Arc::clone(waveform),
        });
        self.reference = None;
        self.restart();
    }

    /// The song is heard `semitones` from its own key from now: its notes
    /// (and the singing drawn on them) move by the change, their heights in
    /// their lines as they were; the scoring goes on.
    pub fn set_key(&mut self, semitones: i8) {
        let by = self.notes.set_key(semitones);
        for heard in &mut self.heard {
            heard.cents += by;
        }
    }

    /// The key its notes are in, in semitones from the song's own.
    pub fn shift(&self) -> i8 {
        self.notes.shift()
    }

    /// Whether its notes are those of `path`.
    pub fn is_of(&self, path: &std::path::Path) -> bool {
        self.source.as_ref().is_some_and(|s| s.path == path)
    }

    /// Whether its notes were made from these.
    pub fn is_made_of(
        &self,
        path: &std::path::Path,
        lyrics: &Arc<Lyrics>,
        waveform: &Arc<Waveform>,
    ) -> bool {
        self.source
            .as_ref()
            .is_some_and(|s| s.is(path, lyrics, waveform))
    }

    /// The song its notes are of.
    pub fn song(&self) -> Option<&std::path::Path> {
        self.source.as_ref().map(|s| s.path.as_path())
    }

    /// The original singer's phrases the shine holds these to (none: no
    /// shine); taken where it has a phrase a line.
    pub fn set_reference(&mut self, reference: Option<Arc<Reference>>) {
        self.reference = reference.filter(|r| r.phrases.len() == self.notes.lines.len());
    }

    /// The original singer's phrases, where known.
    pub fn reference(&self) -> Option<&Arc<Reference>> {
        self.reference.as_ref()
    }

    /// The scoring started over (the song from its start): no frames, no
    /// score, no moments.
    pub fn restart(&mut self) {
        self.heard.clear();
        self.frames.clear();
        self.log.clear();
        self.sung_key.clear();
        self.counts.clear();
        self.recent.clear();
        self.glide = None;
        self.last = None;
        self.tally = Tally::default();
        self.undo.clear();
    }

    /// The frames heard on line `l`'s word `w` so far (on its notes, with a
    /// pitch or not): what the scoring holds its hits against.
    pub fn noted(&self, l: usize, w: usize) -> u32 {
        self.counts.get(&(l, w)).map_or(0, |f| f.0)
    }

    /// How line `l`'s word `w` was rescued, if it was.
    pub fn rescue_of(&self, l: usize, w: usize) -> Option<Rescue> {
        self.log.index().rescues.get(&(l, w)).copied()
    }

    /// The words rescued so far (their line and word) and how.
    pub fn rescues(&self) -> impl Iterator<Item = ((usize, usize), &Rescue)> {
        self.log.index().rescues.iter().map(|(&w, r)| (w, r))
    }

    /// When the last frame was, in seconds into the song.
    pub fn last_heard(&self) -> Option<f64> {
        self.last
    }

    /// A frame at `t` coming: from before the last by more than a second
    /// (the song started again, sought back far), the scoring starts over;
    /// by less (sought back a little, the delay nudged), what the frames
    /// from `t` on did is taken back (they are heard again), the score
    /// before them kept.
    fn sought(&mut self, t: f64) {
        if self.last.is_some_and(|last| t < last - 1.0) {
            self.restart();
            return;
        }
        let from = t - jodel_rhythm::yin::HOP / 2.0;
        let back = self.undo.take_from(from);
        let Some(first) = back.last() else {
            return;
        };
        self.tally = first.tally;
        for u in back {
            if let Some((word, hit)) = u.word {
                if let Some(c) = self.counts.get_mut(&word) {
                    c.0 -= 1;
                    c.1 -= u32::from(hit);
                }
            }
            if let Some(step) = u.key {
                self.sung_key.undo(step);
            }
        }
        let keep = self.heard.partition_point(|h| h.t < from);
        self.heard.truncate(keep);
        let keep = self
            .frames
            .partition_point(|f| f.t < from - shine::FRAME_LATE);
        self.frames.truncate(keep);
        self.log.take_back(from);
        self.recent.clear();
        self.glide = None;
    }

    /// The multiplier now.
    pub fn multiplier(&self) -> u32 {
        multiplier(self.tally.streak)
    }

    /// The note sung at `t`: its line and index ([`Notes::note_at`]).
    pub fn note_at(&self, t: f64) -> Option<(usize, usize)> {
        self.notes.note_at(t)
    }

    /// The note sung at `t` ([`Notes::word_at`]).
    pub fn word_at(&self, t: f64) -> Option<((usize, usize), f64)> {
        self.notes.word_at(t)
    }

    /// A frame of the singer: `cents` (none where nothing is sung) at `t`
    /// seconds into the song. A frame from before the last by more than a
    /// second (the song started again, sought back) starts the scoring
    /// over; from before it by less, what the frames from it on did is
    /// taken back first.
    pub fn hear(&mut self, t: f64, cents: Option<f32>) {
        self.sought(t);
        self.last = Some(t);
        let mut undo = undo::Undo {
            t,
            word: None,
            key: None,
            tally: self.tally,
        };
        if let Some((l, k)) = self.notes.note_at(t) {
            let n = self.notes.lines[l].notes[k];
            let counts = self.counts.entry((l, n.word)).or_insert((0, 0));
            counts.0 += 1;
            undo.word = Some(((l, n.word), false));
            if let Some(c) = cents {
                let c = f64::from(c);
                undo.key = Some(self.sung_key.hear(c, n.cents));
                let off = self.sung_key.off(c, n.cents);
                let hit = judge::hits(off);
                undo.word = Some(((l, n.word), hit));
                counts.1 += u32::from(hit);
                self.recent.push_back(off);
                if self.recent.len() > SMOOTH_FRAMES {
                    self.recent.pop_front();
                }
                let drawn = if hit {
                    self.glide = None;
                    0.0
                } else {
                    let mut sorted: Vec<f64> = self.recent.iter().copied().collect();
                    sorted.sort_by(f64::total_cmp);
                    // The median takes out a frame heard wrong; half way on
                    // from the last drawn softens a wobble, and still
                    // follows a slide within a few frames.
                    let median = sorted[sorted.len() / 2];
                    let drawn = self.glide.map_or(median, |g| g + GLIDE * (median - g));
                    self.glide = Some(drawn);
                    drawn
                };
                self.heard.push(Heard {
                    t,
                    cents: n.cents + drawn,
                    off: Some(off),
                    hit,
                    line: Some(l),
                });
            } else {
                self.recent.clear();
                self.glide = None;
            }
        } else {
            self.recent.clear();
            self.glide = None;
        }
        self.frames.push(shine::Frame {
            t: t - shine::FRAME_LATE,
            cents,
            key: self.sung_key.median(),
        });
        self.score_words(t);
        self.pops(t);
        self.shine_phrases(t);
        self.undo.push(undo);
    }

    /// The words over by `t`, scored: each a moment, the streak's level up
    /// or lost with it. A word missed alone waits, rescued at once if the
    /// next word burst already ([`bursts`]).
    fn score_words(&mut self, t: f64) {
        while let Some(&(l, w, end)) = self.notes.words().get(self.tally.next_word) {
            if end + SCORED_AFTER > t {
                break;
            }
            let order = self.tally.next_word;
            self.tally.next_word += 1;
            let counts = self.counts.get(&(l, w)).copied().unwrap_or((0, 0));
            let before = multiplier(self.tally.streak);
            if !self.tally.score_word(counts, end, order) {
                continue;
            }
            let streak = self.tally.streak;
            let hit = self.tally.last_scored.is_some_and(|s| s.hit);
            self.log.push(t, t, Cue::Word { end, hit, streak });
            let now = multiplier(streak);
            if now > before {
                self.log.push(t, t, Cue::StreakUp(now - 1));
            } else if now < before && before > 1 {
                self.log.push(t, t, Cue::StreakLost);
            }
            if self.tally.run == Run::Waiting(order) {
                self.rescue_waiting(t);
            }
        }
    }

    /// The phrases whose frames are all in by `t` (seconds into the song),
    /// measured and held against the original's: each a moment with its
    /// shine and tier.
    fn shine_phrases(&mut self, t: f64) {
        let ready = t - shine::FRAME_LATE - shine::measure::MASK_AROUND;
        while let Some(&(l, end)) = self.notes.phrases().get(self.tally.next_phrase) {
            if end > ready {
                break;
            }
            self.tally.next_phrase += 1;
            let Some(original) = self.reference.as_ref().and_then(|r| r.phrase(l)) else {
                continue;
            };
            let measured =
                shine::measure::phrase(&self.notes, l, &self.frames, &original.yardstick);
            let Some(s) = measured.map(|m| shine::shine(&m, &original.yardstick)) else {
                continue;
            };
            let shone = Shone::new(l, end, &s, original.par);
            self.log.push(t, t, Cue::Phrase(shone));
        }
    }

    /// The song sung to its end: silence heard on to past its last note,
    /// so its last words and phrases are scored (nothing when nothing was
    /// heard).
    pub fn finish(&mut self) {
        let Some(mut t) = self.last else {
            return;
        };
        let ends = self.notes.words().last().map_or(0.0, |w| w.2);
        let ends = ends.max(self.notes.phrases().last().map_or(0.0, |p| p.1));
        let until = ends + SCORED_AFTER + shine::measure::MASK_AROUND + shine::FRAME_LATE + 0.02;
        while t < until {
            t += jodel_rhythm::yin::HOP;
            self.hear(t, None);
        }
    }

    /// The moments new to `cursor` (which moves past them), and whether
    /// what it read before was taken back.
    pub fn moments_since(&self, cursor: &mut Cursor) -> Since<'_> {
        self.log.since(cursor)
    }

    /// All the moments so far, in the order they were decided.
    pub fn moments(&self) -> &[Moment] {
        self.log.all()
    }

    /// The phrases that shone so far, in order.
    pub fn shone(&self) -> impl Iterator<Item = &Shone> {
        self.log.all().iter().filter_map(|m| match &m.cue {
            Cue::Phrase(p) => Some(p),
            _ => None,
        })
    }

    /// A phrase's shine decided elsewhere (a take replayed, a picture
    /// drawn of a moment): into the moments at `at` (the singing's time),
    /// as if decided here.
    pub fn decided(&mut self, at: f64, shone: Shone) {
        self.log.push(at, at, Cue::Phrase(shone));
    }

    /// The last phrase that shone, if any.
    pub fn last_phrase(&self) -> Option<&Shone> {
        self.log.all().iter().rev().find_map(|m| match &m.cue {
            Cue::Phrase(p) => Some(p),
            _ => None,
        })
    }

    /// The words with notes in the song.
    pub fn words(&self) -> usize {
        self.notes.words().len()
    }

    /// The frames heard on line `l` from `from` to `to` seconds.
    pub fn heard_on(&self, l: usize, from: f64, to: f64) -> impl Iterator<Item = &Heard> {
        let a = self.heard.partition_point(|h| h.t < from);
        self.heard[a..]
            .iter()
            .take_while(move |h| h.t <= to)
            .filter(move |h| h.line == Some(l))
    }
}

#[cfg(test)]
mod tests;
