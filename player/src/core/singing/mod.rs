//! The singing scored: the notes of a song's lines (from the voice's
//! melody, under the words' letters) and the singer's pitch against them,
//! frame by frame, word by word.
//!
//! Each frame of the singer's pitch (from the microphone, at the time in
//! the song it was sung) is held against the note sung there. On
//! [`Level::Easy`] and [`Level::Medium`] the singer may sing in any key:
//! their pitch is moved by how far, all along, they have sung from the
//! song (the median over the last [`KEY_FRAMES`]), octaves aside. On
//! [`Level::Hard`] only octaves are let off. A frame within the level's
//! [`Level::tolerance`] hits; on [`Level::Medium`] one a little further
//! off, within [`Level::near`], half hits. A word whose note time is hit
//! for at least the level's share of it ([`Level::share`]: a tenth on
//! easy, a quarter on medium, two fifths on hard; a pitch is heard on only
//! about half of a word sung well, its consonants and breaths having none)
//! scores [`WORD_POINTS`]; one that only gets there with its half hits,
//! half of that. Words hit in a row are a
//! streak, and the points a word scores are times the streak's
//! [`multiplier`] (a half hit keeps the streak, a miss ends it).
//!
//! The song played in another key ([`Singing::set_key`]) has its notes in
//! that key: they move by its semitones, so the singing is held against
//! what is heard.
//!
//! The singer's pitch as drawn ([`Heard::cents`]) follows the level: on
//! easy and medium a frame that hits sits on its note, one that half hits
//! on the semitone it is nearest; on hard, and off the notes, it is the
//! pitch sung, its jitter smoothed (a median over [`SMOOTH_FRAMES`], then
//! a glide).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use jodel_rhythm::waveform::Melody;

mod judge;
mod tally;
mod undo;

pub use judge::{Credit, KeyStep, SungKey, KEY_FRAMES};
pub use tally::Tally;

/// How strictly the singing is held to the song's notes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Level {
    /// The shape of the melody, in any key: within 3 semitones.
    Easy,
    /// Its steps, in any key: within 1.5 semitones.
    #[default]
    Medium,
    /// The song's own notes, octaves aside: within a semitone.
    Hard,
}

impl Level {
    /// All, easiest first.
    pub const ALL: [Level; 3] = [Level::Easy, Level::Medium, Level::Hard];

    /// How far from a note a frame may be and still hit, in hundredths of
    /// a semitone.
    pub fn tolerance(self) -> f64 {
        match self {
            Level::Easy => 300.0,
            Level::Medium => 150.0,
            Level::Hard => 100.0,
        }
    }

    /// How far from a note a frame may be and still half hit (none: no
    /// half hits at the level).
    pub fn near(self) -> Option<f64> {
        (self == Level::Medium).then_some(300.0)
    }

    /// Whether the singer may sing in their own key.
    pub fn any_key(self) -> bool {
        self != Level::Hard
    }

    /// Whether the singer's pitch is drawn on the note they hit (else
    /// always as sung).
    pub fn snaps(self) -> bool {
        self != Level::Hard
    }

    /// Its name, as shown and in the settings.
    pub fn name(self) -> &'static str {
        match self {
            Level::Easy => "easy",
            Level::Medium => "medium",
            Level::Hard => "hard",
        }
    }

    /// The level named `name`.
    pub fn parse(name: &str) -> Option<Level> {
        Level::ALL.into_iter().find(|l| l.name() == name)
    }

    /// What it asks of the singer.
    pub fn what(self) -> &'static str {
        match self {
            Level::Easy => "any key, the melody's shape",
            Level::Medium => "any key, its steps",
            Level::Hard => "the song's own key",
        }
    }

    /// The next, after the hardest the easiest.
    pub fn next(self) -> Level {
        match self {
            Level::Easy => Level::Medium,
            Level::Medium => Level::Hard,
            Level::Hard => Level::Easy,
        }
    }
}

/// A note sung in a line: under which of its characters (from `from` to
/// `to`, fractions of a character where a letter is sung through it), when
/// (`start` to `end`, seconds into the song), its pitch (hundredths of a
/// MIDI note, an octave heard wrong folded back), and how high that is
/// against the line's middle, from -1 (its lowest) to 1 (its highest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// Where it starts, in characters of the line's text.
    pub from: f32,
    /// Where it ends.
    pub to: f32,
    /// How high: -1 to 1, 0 the line's middle.
    pub height: f32,
    /// When it starts, in seconds into the song.
    pub start: f64,
    /// When it ends.
    pub end: f64,
    /// Its pitch, in hundredths of a MIDI note.
    pub cents: f64,
    /// Which of the line's words it is of.
    pub word: usize,
}

/// A line's notes, and the pitch its heights are measured from and over.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LineNotes {
    /// Its notes, in order.
    pub notes: Vec<Note>,
    /// The middle (height 0), in hundredths of a MIDI note.
    pub mid: f64,
    /// How far from the middle height 1 is.
    pub range: f64,
}

impl LineNotes {
    /// How high `cents` is against the line's middle (-1 to 1).
    pub fn height(&self, cents: f64) -> f32 {
        ((cents - self.mid) / self.range.max(1.0)).clamp(-1.0, 1.0) as f32
    }
}

/// A medal for a song sung: by the words hit, a half hit as half of one
/// (not by the points, which the streaks multiply).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medal {
    /// Half of them.
    Bronze,
    /// 70 %.
    Silver,
    /// 85 %.
    Gold,
}

impl Medal {
    /// The medal for `hit` and `half` hit of `words` words, if any.
    pub fn of(hit: u32, half: u32, words: u32) -> Option<Medal> {
        if words == 0 {
            return None;
        }
        let share = (f64::from(hit) + f64::from(half) / 2.0) / f64::from(words);
        if share >= 0.85 {
            Some(Medal::Gold)
        } else if share >= 0.7 {
            Some(Medal::Silver)
        } else if share >= 0.5 {
            Some(Medal::Bronze)
        } else {
            None
        }
    }

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Medal::Bronze => "Bronze",
            Medal::Silver => "Silver",
            Medal::Gold => "Gold",
        }
    }
}

/// How a song was sung, shown when it ends.
#[derive(Clone, Debug, PartialEq)]
pub struct SongResult {
    /// The song's title.
    pub title: String,
    /// The points.
    pub points: u32,
    /// The words hit.
    pub hit: u32,
    /// The words half hit.
    pub half: u32,
    /// The words scored.
    pub words: u32,
    /// The most words hit in a row.
    pub streak: u32,
    /// The level sung at.
    pub level: Level,
    /// The medal won, if any.
    pub medal: Option<Medal>,
    /// The best score of the song at that level before (none: the first).
    pub best: Option<u32>,
    /// Whether the song was sung whole (from its start): only then a medal
    /// and a best.
    pub whole: bool,
}

impl SongResult {
    /// Whether it beat the best before (or is the first).
    pub fn new_best(&self) -> bool {
        self.best.is_none_or(|b| self.points > b)
    }
}

/// The least range, in hundredths of a semitone, a line's notes are spread
/// over each way from its middle: a line that stays closer keeps its notes
/// nearer the middle (4 semitones each way to the top).
const NOTE_RANGE: f64 = 400.0;
/// The most: notes further out (an octave each way) stay at the edge.
const NOTE_RANGE_MOST: f64 = 1200.0;
/// The least of a character a note is drawn under.
const NOTE_LEAST_CHARS: f32 = 0.25;

/// Where in `w` (in characters, a fraction through one) the voice is at
/// `t`: by its letters' times where it has them (each letter from its
/// time to the next's), else spread evenly over the word.
fn char_at(w: &jodel_lyrics::Word, t: f64) -> f32 {
    let n = w.text.chars().count();
    let (Some(start), Some(end)) = (w.start, w.end) else {
        return 0.0;
    };
    if n == 0 {
        return 0.0;
    }
    let times: Option<Vec<f64>> = (w.letters.len() == n)
        .then(|| (0..n).map(|i| w.letter_at(i)).collect())
        .flatten();
    let Some(times) = times else {
        return (n as f64 * ((t - start) / (end - start).max(1e-3)).clamp(0.0, 1.0)) as f32;
    };
    for i in 0..n {
        let from = times[i];
        let to = if i + 1 < n { times[i + 1] } else { end };
        if t < to || i + 1 == n {
            let share = ((t - from) / (to - from).max(1e-3)).clamp(0.0, 1.0);
            return (i as f64 + share) as f32;
        }
    }
    n as f32
}

/// The notes `melody` sings `line` on ([`Melody::notes_between`] of each
/// lead word, not a backing one, what is not a note taken out), each under
/// the letters sung then; its height its pitch against the line's middle
/// (the median over its notes' length), an octave off folded back (the
/// pitch heard an octave wrong), spread over the line's own range
/// ([`NOTE_RANGE`] to [`NOTE_RANGE_MOST`]). A word's notes go under all of
/// it, end to end, and over all its time. Every lead word gets a note: one
/// without a pitch of its own one between its neighbours', under all of
/// it. None with fewer than two notes heard.
///
/// [`Melody::notes_between`]: jodel_rhythm::waveform::Melody::notes_between
pub fn notes_of(line: &jodel_lyrics::Line, melody: &Melody) -> LineNotes {
    let mut at = 0.0f32;
    /// A word: where it is in the text (from, to), when (start, end),
    /// whether it gets notes, and its own (each with how long it is heard).
    type Word = (f32, f32, (f64, f64), bool, Vec<(Note, f64)>);
    let mut words: Vec<Word> = Vec::new();
    for (k, w) in line.words.iter().enumerate() {
        if k > 0 {
            at += 1.0;
        }
        let n = w.text.chars().count() as f32;
        // Only the lead's words: a backing part in parentheses (an echo) is
        // quieter under it, its pitch unsure and its time squeezed between
        // the lead's. A lead word heard weakly is timed between its
        // neighbours, close enough for the melody sung there.
        let lead = w.start.is_some() && w.end.is_some() && !w.backing;
        let (ws, we) = (w.start.unwrap_or(0.0), w.end.unwrap_or(0.0));
        let mut own: Vec<(Note, f64)> = Vec::new();
        if lead {
            for (a, b, c) in melody.notes_between(ws, we) {
                let (from, to) = (at + char_at(w, a), at + char_at(w, b));
                // Too short under the letters to read (a note on a quick
                // consonant).
                if to - from >= NOTE_LEAST_CHARS {
                    let note = Note {
                        from,
                        to,
                        height: 0.0,
                        start: a,
                        end: b,
                        cents: c,
                        word: k,
                    };
                    own.push((note, b - a));
                }
            }
        }
        // Under the whole word, end to end and over all its time: its
        // first note from its first letter, its last to its last, each on
        // to the next.
        if let Some(first) = own.first_mut() {
            first.0.from = at;
            first.0.start = ws;
        }
        if let Some(last) = own.last_mut() {
            last.0.to = at + n;
            last.0.end = we;
        }
        for i in 1..own.len() {
            own[i - 1].0.to = own[i].0.from;
            own[i - 1].0.end = own[i].0.start;
        }
        words.push((at, at + n, (ws, we), lead, own));
        at += n;
    }
    let heard: Vec<(f64, f64)> = words
        .iter()
        .flat_map(|w| w.4.iter().map(|(n, d)| (n.cents, *d)))
        .collect();
    if heard.len() < 2 {
        return LineNotes::default();
    }
    // The middle: half the singing below it, half above.
    let middle = |notes: &[(f64, f64)]| {
        let mut by = notes.to_vec();
        by.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = by.iter().map(|n| n.1).sum::<f64>() / 2.0;
        let mut sum = 0.0;
        for (c, d) in &by {
            sum += d;
            if sum >= half {
                return *c;
            }
        }
        by.last().map_or(0.0, |n| n.0)
    };
    let mid = middle(&heard);
    for w in &mut words {
        for (n, _) in &mut w.4 {
            while n.cents - mid > 900.0 {
                n.cents -= 1200.0;
            }
            while mid - n.cents > 900.0 {
                n.cents += 1200.0;
            }
        }
    }
    let heard: Vec<(f64, f64)> = words
        .iter()
        .flat_map(|w| w.4.iter().map(|(n, d)| (n.cents, *d)))
        .collect();
    let mid = middle(&heard);
    let range = heard
        .iter()
        .map(|n| (n.0 - mid).abs())
        .fold(NOTE_RANGE, f64::max)
        .min(NOTE_RANGE_MOST);
    // Every lead word gets a note: one without a pitch of its own, one
    // under all of it between the word before's last note and the next's
    // first (or the one there is).
    let own: Vec<bool> = words.iter().map(|w| !w.4.is_empty()).collect();
    for k in 0..words.len() {
        if !words[k].3 || own[k] {
            continue;
        }
        let before = (0..k).rev().find_map(|j| {
            own[j]
                .then(|| words[j].4.last().map(|n| n.0.cents))
                .flatten()
        });
        let after = (k + 1..words.len()).find_map(|j| {
            own[j]
                .then(|| words[j].4.first().map(|n| n.0.cents))
                .flatten()
        });
        let guess = match (before, after) {
            (Some(a), Some(b)) => (a + b) / 2.0,
            (Some(c), None) | (None, Some(c)) => c,
            (None, None) => continue,
        };
        let (from, to, (ws, we)) = (words[k].0, words[k].1, words[k].2);
        words[k].4.push((
            Note {
                from,
                to,
                height: 0.0,
                start: ws,
                end: we,
                cents: guess,
                word: k,
            },
            0.0,
        ));
    }
    let mut out = LineNotes {
        notes: Vec::new(),
        mid,
        range,
    };
    for w in words {
        for (mut n, _) in w.4 {
            n.height = out.height(n.cents);
            out.notes.push(n);
        }
    }
    out
}

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
    /// How it was hit (none: missed).
    pub hit: Option<Hit>,
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

/// How a frame hit its note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// Within the level's tolerance.
    Full,
    /// Close, within [`Level::near`].
    Half,
}

/// A frame of the singer's pitch, as held against the song.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heard {
    /// When in the song it was sung, in seconds.
    pub t: f64,
    /// The pitch as drawn, in hundredths of a MIDI note: moved into the
    /// song's key (and octave, by the note there) as the level allows, on
    /// the note or a semitone where it snaps.
    pub cents: f64,
    /// How far off the note it was (none off a note), in hundredths of a
    /// semitone: above the note when positive.
    pub off: Option<f64>,
    /// Whether it hit the note.
    pub hit: Option<Hit>,
    /// Which line's note it was on.
    pub line: Option<usize>,
}

/// One song's singing, scored as it goes.
#[derive(Clone, Debug, Default)]
pub struct Singing {
    /// The song the notes are of: its path, and the lyrics and melody they
    /// were made from (by address).
    key: Option<(PathBuf, usize, usize)>,
    /// Each sung line's notes, by its index among the song's sung lines.
    pub lines: Vec<LineNotes>,
    /// Every note in time: start, end, line, note.
    order: Vec<(f64, f64, usize, usize)>,
    /// Every word with notes, by when it ends: its line, word and end.
    words: Vec<(usize, usize, f64)>,
    /// The level it is scored at.
    pub level: Level,
    /// The key the song is heard in, in semitones from its own: the notes
    /// are moved by it.
    shift: i8,
    /// The singer's pitch frames so far, in time.
    pub heard: Vec<Heard>,
    /// How far the singer sings from the song (octaves aside).
    sung_key: SungKey,
    /// Each word's frames on a note, those that hit and those that half
    /// hit.
    frames: HashMap<(usize, usize), (u32, u32, u32)>,
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
    /// order) and `melody` (`melody_id` telling one from another): its
    /// notes made anew when any of them changed (and its scoring started
    /// over). `level`: the level to score at (a change starts the scoring
    /// over too).
    pub fn prepare(
        &mut self,
        path: &std::path::Path,
        lyrics: &Arc<jodel_lyrics::Lyrics>,
        melody: &Melody,
        melody_id: usize,
        level: Level,
    ) {
        let key = (path.to_owned(), Arc::as_ptr(lyrics) as usize, melody_id);
        if self.key.as_ref() != Some(&key) {
            let sung: Vec<&jodel_lyrics::Line> = lyrics
                .lines
                .iter()
                .filter(|l| !l.words.is_empty())
                .collect();
            self.lines = sung.iter().map(|l| notes_of(l, melody)).collect();
            let shift = std::mem::take(&mut self.shift);
            self.set_key(shift);
            self.order = self
                .lines
                .iter()
                .enumerate()
                .flat_map(|(l, ln)| {
                    ln.notes
                        .iter()
                        .enumerate()
                        .map(move |(k, n)| (n.start, n.end, l, k))
                })
                .collect();
            self.order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut ends: HashMap<(usize, usize), f64> = HashMap::new();
            for (l, ln) in self.lines.iter().enumerate() {
                for n in &ln.notes {
                    let e = ends.entry((l, n.word)).or_insert(n.end);
                    *e = e.max(n.end);
                }
            }
            self.words = ends.into_iter().map(|((l, w), e)| (l, w, e)).collect();
            self.words.sort_by(|a, b| a.2.total_cmp(&b.2));
            self.key = Some(key);
            self.level = level;
            self.restart();
        } else if self.level != level {
            self.level = level;
            self.restart();
        }
    }

    /// The song is heard `semitones` from its own key from now: its notes
    /// (and the singing drawn on them) move by the change, their heights in
    /// their lines as they were; the scoring goes on.
    pub fn set_key(&mut self, semitones: i8) {
        let by = 100.0 * f64::from(i32::from(semitones) - i32::from(self.shift));
        self.shift = semitones;
        if by == 0.0 {
            return;
        }
        for line in &mut self.lines {
            line.mid += by;
            for note in &mut line.notes {
                note.cents += by;
            }
        }
        for heard in &mut self.heard {
            heard.cents += by;
        }
    }

    /// The key its notes are in, in semitones from the song's own.
    pub fn shift(&self) -> i8 {
        self.shift
    }

    /// Whether its notes are those of `path`.
    pub fn is_of(&self, path: &std::path::Path) -> bool {
        self.key.as_ref().is_some_and(|k| k.0 == path)
    }

    /// The song its notes are of.
    pub fn song(&self) -> Option<&std::path::Path> {
        self.key.as_ref().map(|k| k.0.as_path())
    }

    /// The scoring started over (the song from its start, or another
    /// level): no frames, no score.
    pub fn restart(&mut self) {
        self.heard.clear();
        self.sung_key.clear();
        self.frames.clear();
        self.recent.clear();
        self.glide = None;
        self.last = None;
        self.tally = Tally::default();
        self.undo.clear();
    }

    /// The frames heard on line `l`'s word `w` so far (on its notes, with a
    /// pitch or not): what the scoring holds its hits against.
    pub fn noted(&self, l: usize, w: usize) -> u32 {
        self.frames.get(&(l, w)).map_or(0, |f| f.0)
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
                if let Some(c) = self.frames.get_mut(&word) {
                    c.0 -= 1;
                    c.1 -= u32::from(hit == Some(Hit::Full));
                    c.2 -= u32::from(hit == Some(Hit::Half));
                }
            }
            if let Some(step) = u.key {
                self.sung_key.undo(step);
            }
        }

        let keep = self.heard.partition_point(|h| h.t < from);
        self.heard.truncate(keep);
        self.recent.clear();
        self.glide = None;
    }

    /// The multiplier now.
    pub fn multiplier(&self) -> u32 {
        multiplier(self.tally.streak)
    }

    /// The note sung at `t`: its line and index.
    fn note_at(&self, t: f64) -> Option<(usize, usize)> {
        let i = self.order.partition_point(|n| n.0 <= t);
        let n = self.order.get(i.checked_sub(1)?)?;
        (t < n.1).then_some((n.2, n.3))
    }

    /// The note sung at `t` seconds into the song (none between the
    /// notes): the word it is of (its line and its index there) and its
    /// pitch, in hundredths of a MIDI note.
    pub fn word_at(&self, t: f64) -> Option<((usize, usize), f64)> {
        let (l, k) = self.note_at(t)?;
        let n = &self.lines[l].notes[k];
        Some(((l, n.word), n.cents))
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
        let note = self.note_at(t);
        if let Some((l, k)) = note {
            let n = self.lines[l].notes[k];
            let counts = self.frames.entry((l, n.word)).or_insert((0, 0, 0));
            counts.0 += 1;
            undo.word = Some(((l, n.word), None));
            if let Some(c) = cents {
                let c = f64::from(c);
                undo.key = Some(self.sung_key.hear(c, n.cents));
                let off = self.sung_key.off(self.level, c, n.cents);
                let hit = self.level.judge(off);
                undo.word = Some(((l, n.word), hit));
                match hit {
                    Some(Hit::Full) => counts.1 += 1,
                    Some(Hit::Half) => counts.2 += 1,
                    None => {}
                }
                self.recent.push_back(off);
                if self.recent.len() > SMOOTH_FRAMES {
                    self.recent.pop_front();
                }
                let drawn = match hit {
                    Some(Hit::Full) if self.level.snaps() => 0.0,
                    Some(Hit::Half) if self.level.snaps() => (off / 100.0).round() * 100.0,
                    _ => {
                        let mut sorted: Vec<f64> = self.recent.iter().copied().collect();
                        sorted.sort_by(f64::total_cmp);
                        // The median takes out a frame heard wrong; half
                        // way on from the last drawn softens a wobble, and
                        // still follows a slide within a few frames.
                        let median = sorted[sorted.len() / 2];
                        let drawn = self.glide.map_or(median, |g| g + GLIDE * (median - g));
                        self.glide = Some(drawn);
                        drawn
                    }
                };
                if hit.is_some() && self.level.snaps() {
                    self.glide = None;
                }
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
        // The words over by now, scored.
        while let Some(&(l, w, end)) = self.words.get(self.tally.next_word) {
            if end + SCORED_AFTER > t {
                break;
            }
            self.tally.next_word += 1;
            let counts = self.frames.get(&(l, w)).copied().unwrap_or((0, 0, 0));
            self.tally.score_word(self.level, counts, end);
        }
        self.undo.push(undo);
    }

    /// The words with notes in the song.
    pub fn words(&self) -> usize {
        self.words.len()
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
