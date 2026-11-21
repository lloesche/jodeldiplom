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
//! for at least half ([`WORD_HIT`]) scores [`WORD_POINTS`]; one that only
//! gets there with its half hits, half of that. Words hit in a row are a
//! streak, and the points a word scores are times the streak's
//! [`multiplier`] (a half hit keeps the streak, a miss ends it).
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

/// `d` (hundredths of a semitone) brought within half an octave of 0, a
/// whole number of octaves taken off.
fn fold(d: f64) -> f64 {
    (d + 600.0).rem_euclid(1200.0) - 600.0
}

/// The frames of the singer's pitch the key they sing in is told from (the
/// last 3 s of them on a note).
pub const KEY_FRAMES: usize = 300;
/// The share of a word's note time to hit for it to count.
pub const WORD_HIT: f64 = 0.5;
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
    /// The singer's pitch frames so far, in time.
    pub heard: Vec<Heard>,
    /// How far the singer sings from the song (octaves aside), frame by
    /// frame of late, and its median.
    keys: VecDeque<f64>,
    key_off: Option<f64>,
    /// Each word's frames on a note, those that hit and those that half
    /// hit.
    frames: HashMap<(usize, usize), (u32, u32, u32)>,
    /// How far off the last frames sung were, and the last drawn as sung,
    /// for the pitch drawn as sung.
    recent: VecDeque<f64>,
    glide: Option<f64>,
    /// The words scored so far (of [`Singing::words`]).
    next_word: usize,
    /// When the last frame (sung or not) was, in the song.
    last: Option<f64>,
    /// The score.
    pub score: u32,
    /// The words hit.
    pub hit_words: u32,
    /// The words half hit.
    pub half_words: u32,
    /// The words scored (hit or not).
    pub scored_words: u32,
    /// The words hit in a row up to now (a half hit keeps it).
    pub streak: u32,
    /// The longest streak.
    pub best_streak: u32,
    /// The last word scored.
    pub last_scored: Option<Scored>,
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
        self.keys.clear();
        self.key_off = None;
        self.frames.clear();
        self.recent.clear();
        self.glide = None;
        self.next_word = 0;
        self.last = None;
        self.score = 0;
        self.hit_words = 0;
        self.half_words = 0;
        self.scored_words = 0;
        self.streak = 0;
        self.best_streak = 0;
        self.last_scored = None;
    }

    /// The multiplier now.
    pub fn multiplier(&self) -> u32 {
        multiplier(self.streak)
    }

    /// The note sung at `t`: its line and index.
    fn note_at(&self, t: f64) -> Option<(usize, usize)> {
        let i = self.order.partition_point(|n| n.0 <= t);
        let n = self.order.get(i.checked_sub(1)?)?;
        (t < n.1).then_some((n.2, n.3))
    }

    /// A frame of the singer: `cents` (none where nothing is sung) at `t`
    /// seconds into the song. A frame from before the last by more than a
    /// second (the song started again, sought back) starts the scoring
    /// over.
    pub fn hear(&mut self, t: f64, cents: Option<f32>) {
        if self.last.is_some_and(|last| t < last - 1.0) {
            self.restart();
        }
        self.last = Some(t);
        let note = self.note_at(t);
        if let Some((l, k)) = note {
            let n = self.lines[l].notes[k];
            let counts = self.frames.entry((l, n.word)).or_insert((0, 0, 0));
            counts.0 += 1;
            if let Some(c) = cents {
                let c = f64::from(c);
                // How far the singer sings from the song, octaves aside,
                // near what it has been.
                let from = fold(c - n.cents);
                let near = self.key_off.map_or(from, |k| k + fold(from - k));
                self.keys.push_back(near);
                if self.keys.len() > KEY_FRAMES {
                    self.keys.pop_front();
                }
                let mut sorted: Vec<f64> = self.keys.iter().copied().collect();
                sorted.sort_by(f64::total_cmp);
                self.key_off = Some(sorted[sorted.len() / 2]);
                let shift = if self.level.any_key() {
                    self.key_off.unwrap_or(0.0)
                } else {
                    0.0
                };
                let off = fold(c - shift - n.cents);
                let hit = if off.abs() <= self.level.tolerance() {
                    counts.1 += 1;
                    Some(Hit::Full)
                } else if self.level.near().is_some_and(|near| off.abs() <= near) {
                    counts.2 += 1;
                    Some(Hit::Half)
                } else {
                    None
                };
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
        while let Some(&(l, w, end)) = self.words.get(self.next_word) {
            if end + SCORED_AFTER > t {
                break;
            }
            self.next_word += 1;
            let (noted, hit, half) = self.frames.get(&(l, w)).copied().unwrap_or((0, 0, 0));
            if noted == 0 {
                // Not heard at all (sought past): not scored.
                continue;
            }
            self.scored_words += 1;
            let enough = WORD_HIT * f64::from(noted);
            let (got, base) = if f64::from(hit) >= enough {
                self.hit_words += 1;
                self.streak += 1;
                (Some(Hit::Full), WORD_POINTS)
            } else if f64::from(hit + half) >= enough {
                self.half_words += 1;
                (Some(Hit::Half), WORD_POINTS / 2)
            } else {
                self.streak = 0;
                (None, 0)
            };
            self.best_streak = self.best_streak.max(self.streak);
            let points = base * self.multiplier();
            self.score += points;
            self.last_scored = Some(Scored {
                end,
                hit: got,
                points,
                streak: self.streak,
            });
        }
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
mod tests {
    use super::*;

    /// A melody sure at `cents` over the frames `runs` give.
    fn melody(frames: usize, runs: &[(usize, usize, u16)]) -> Melody {
        let mut m = Melody {
            cents: vec![0; frames],
            sure: vec![0; frames],
        };
        for &(a, b, c) in runs {
            for i in a..b {
                m.cents[i] = c;
                m.sure[i] = 255;
            }
        }
        m
    }

    fn word(text: &str, s: f64, e: f64) -> jodel_lyrics::Word {
        jodel_lyrics::Word {
            start: Some(s),
            end: Some(e),
            confidence: 0.9,
            ..jodel_lyrics::Word::new(text)
        }
    }

    fn line(words: Vec<jodel_lyrics::Word>) -> jodel_lyrics::Line {
        jodel_lyrics::Line {
            words,
            ..jodel_lyrics::Line::default()
        }
    }

    #[test]
    fn a_line_s_notes_go_under_their_letters_as_high_as_they_are_sung() {
        // "go" on C4; "up" (its "p" from 0.4 to 0.75 s) on C4, then a
        // fifth up, then an octave and a third up (heard an octave wrong:
        // a third up, between).
        let m = melody(
            200,
            &[
                (0, 20, 6000),
                (30, 45, 6000),
                (45, 60, 6700),
                (60, 75, 7600),
            ],
        );
        let mut up = word("up", 0.3, 0.75);
        up.letters = vec![0.0, 0.1];
        let mut go = word("go", 0.0, 0.2);
        go.letters = vec![0.0, 0.1];
        let l = line(vec![go, up]);
        let notes = notes_of(&l, &m);
        let r = |x: f32| (x * 100.0).round() / 100.0;
        let got: Vec<(f32, f32, f32)> = notes
            .notes
            .iter()
            .map(|n| (r(n.from), r(n.to), r(n.height)))
            .collect();
        // The middle is C4 (half the singing); spread over 4 semitones
        // each way at least (here 7: the fifth at the top).
        assert_eq!(
            got,
            [
                (0.0, 2.0, 0.0),
                (3.0, 4.14, 0.0),
                (4.14, 4.57, 1.0),
                (4.57, 5.0, 0.57)
            ]
        );
        // Over the words' whole time, the octave folded back.
        assert_eq!((notes.notes[1].start, notes.notes[3].end), (0.3, 0.75));
        assert_eq!(notes.notes[3].cents, 6400.0);
        // A backing word gets none: here that leaves one note, too few.
        let mut backing = l.clone();
        backing.words[1].backing = true;
        assert!(notes_of(&backing, &m).notes.is_empty());
    }

    #[test]
    fn a_word_without_a_pitch_gets_one_between_its_neighbours() {
        // "low" on F3, "mid" with no pitch at all, "high" a fifth up on C4.
        let m = melody(300, &[(0, 50, 5300), (200, 250, 6000)]);
        let l = line(vec![
            word("low", 0.0, 0.5),
            word("mid", 1.0, 1.5),
            word("high", 2.0, 2.5),
        ]);
        let got: Vec<(f32, f32, f32, f64, f64)> = notes_of(&l, &m)
            .notes
            .iter()
            .map(|n| {
                let h = (n.height * 100.0).round() / 100.0;
                (n.from, n.to, h, n.start, n.end)
            })
            .collect();
        // The middle the lower (half the singing), spread over 7
        // semitones: "mid" halfway, under the whole word and its time.
        assert_eq!(
            got,
            [
                (0.0, 3.0, 0.0, 0.0, 0.5),
                (4.0, 7.0, 0.5, 1.0, 1.5),
                (8.0, 12.0, 1.0, 2.0, 2.5)
            ]
        );
    }

    /// A song of one line, "la la la la", each word a second on its note:
    /// C4, E4, G4, C4.
    fn song() -> (Arc<jodel_lyrics::Lyrics>, Arc<Melody>) {
        let m = melody(
            500,
            &[
                (0, 100, 6000),
                (100, 200, 6400),
                (200, 300, 6700),
                (300, 400, 6000),
            ],
        );
        let l = line(
            (0..4)
                .map(|k| word("la", f64::from(k), f64::from(k) + 1.0))
                .collect(),
        );
        let lyrics = jodel_lyrics::Lyrics {
            lines: vec![l],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        };
        (Arc::new(lyrics), Arc::new(m))
    }

    /// `singing` hears `pitch(t)` every 10 ms from 0 to 4.5 s.
    fn sing(singing: &mut Singing, pitch: impl Fn(f64) -> Option<f32>) {
        for i in 0..450 {
            let t = f64::from(i) / 100.0;
            singing.hear(t, pitch(t));
        }
    }

    fn melody_at(t: f64) -> f32 {
        [6000.0, 6400.0, 6700.0, 6000.0][(t as usize).min(3)]
    }

    #[test]
    fn the_song_sung_scores_every_word_at_every_level() {
        let (lyrics, m) = song();
        for level in Level::ALL {
            let mut s = Singing::default();
            s.prepare(Path::new("a"), &lyrics, &m, 1, level);
            sing(&mut s, |t| Some(melody_at(t)));
            assert_eq!(
                (s.hit_words, s.scored_words, s.score),
                (4, 4, 400),
                "{level:?}"
            );
        }
    }

    #[test]
    fn a_fourth_down_counts_on_easy_and_medium_but_not_on_hard() {
        let (lyrics, m) = song();
        let score = |level| {
            let mut s = Singing::default();
            s.prepare(Path::new("a"), &lyrics, &m, 1, level);
            sing(&mut s, |t| Some(melody_at(t) - 500.0));
            s.hit_words
        };
        // The key followed from the first frames on: every word.
        assert_eq!(score(Level::Easy), 4);
        assert_eq!(score(Level::Medium), 4);
        assert_eq!(score(Level::Hard), 0);
        // An octave down counts even on hard.
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Hard);
        sing(&mut s, |t| Some(melody_at(t) - 1200.0));
        assert_eq!(s.hit_words, 4);
    }

    #[test]
    fn the_shape_is_enough_on_easy_the_steps_on_medium() {
        let (lyrics, m) = song();
        // Every note 2 semitones off its own way (the key held), the second
        // and third flat by two semitones: within easy's 3, not medium's 1.5.
        let off = |t: f64| {
            let k = t as usize;
            Some(melody_at(t) + if k == 1 || k == 2 { -200.0 } else { 0.0 })
        };
        let score = |level| {
            let mut s = Singing::default();
            s.prepare(Path::new("a"), &lyrics, &m, 1, level);
            sing(&mut s, off);
            s.hit_words
        };
        assert_eq!(score(Level::Easy), 4);
        assert_eq!(score(Level::Medium), 2);
    }

    #[test]
    fn close_on_medium_scores_half_and_draws_on_the_nearest_semitone() {
        let (lyrics, m) = song();
        // The second word 2 semitones flat: beyond medium's 1.5, within its
        // half hits' 3.
        let off = |t: f64| Some(melody_at(t) + if t as usize == 1 { -210.0 } else { 0.0 });
        let sung = |level| {
            let mut s = Singing::default();
            s.prepare(Path::new("a"), &lyrics, &m, 1, level);
            sing(&mut s, off);
            s
        };
        let s = sung(Level::Medium);
        assert_eq!((s.hit_words, s.half_words, s.score), (3, 1, 350));
        // The half hit kept the streak: the first, third and last hit.
        assert_eq!((s.streak, s.best_streak), (3, 3));
        // Drawn on the note where it hit, a semitone step off where it half
        // hit.
        let at = |s: &Singing, t: f64| {
            s.heard
                .iter()
                .find(|h| (h.t - t).abs() < 0.001)
                .map(|h| (h.cents, h.hit))
        };
        assert_eq!(at(&s, 0.5), Some((6000.0, Some(Hit::Full))));
        assert_eq!(at(&s, 1.5), Some((6200.0, Some(Hit::Half))));
        // On easy it all hits, on its notes.
        let s = sung(Level::Easy);
        assert_eq!((s.hit_words, s.half_words, s.score), (4, 0, 400));
        assert_eq!(at(&s, 1.5), Some((6400.0, Some(Hit::Full))));
        // On hard: as sung, no half hits.
        let s = sung(Level::Hard);
        assert_eq!((s.hit_words, s.half_words), (3, 0));
        let (cents, hit) = at(&s, 1.5).unwrap();
        assert!((cents - 6190.0).abs() < 0.01 && hit.is_none(), "{cents}");
    }

    #[test]
    fn the_pitch_drawn_on_hard_is_smoothed() {
        let (lyrics, m) = song();
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Hard);
        // A frame now and then a semitone and a half sharp: drawn as the
        // rest.
        sing(&mut s, |t| {
            let i = (t * 100.0).round() as u32;
            Some(melody_at(t) + if i.is_multiple_of(4) { 150.0 } else { 20.0 })
        });
        let h = s.heard.iter().find(|h| (h.t - 0.4).abs() < 0.001).unwrap();
        assert!((h.cents - 6020.0).abs() < 0.01, "{}", h.cents);
    }

    #[test]
    fn medals_go_by_the_words() {
        assert_eq!(Medal::of(4, 0, 4), Some(Medal::Gold));
        // Half the words half hit, the rest hit: 75 %.
        assert_eq!(Medal::of(2, 2, 4), Some(Medal::Silver));
        assert_eq!(Medal::of(2, 0, 4), Some(Medal::Bronze));
        assert_eq!(Medal::of(1, 1, 4), None);
        assert_eq!(Medal::of(0, 0, 0), None);
    }

    #[test]
    fn a_streak_multiplies_the_points_and_a_miss_ends_it() {
        assert_eq!(
            [0, 4, 5, 9, 10, 19, 20, 99].map(multiplier),
            [1, 1, 2, 2, 3, 3, 4, 4]
        );
        // A song of 14 words, a second each on C4: the first 12 sung, the
        // 13th missed, the last sung.
        let m = melody(1500, &[(0, 1400, 6000)]);
        let l = line(
            (0..14)
                .map(|k| word("la", f64::from(k), f64::from(k) + 1.0))
                .collect(),
        );
        let lyrics = Arc::new(jodel_lyrics::Lyrics {
            lines: vec![l],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        });
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &Arc::new(m), 1, Level::Medium);
        for i in 0..1500 {
            let t = f64::from(i) / 100.0;
            let missed = (12.0..13.0).contains(&t);
            s.hear(t, (!missed).then_some(6000.0));
        }
        // Words 1-4 at ×1, 5-9 at ×2, 10-12 at ×3, the 13th nothing, the
        // 14th at ×1 again.
        assert_eq!(s.score, 4 * 100 + 5 * 200 + 3 * 300 + 100);
        assert_eq!((s.streak, s.best_streak), (1, 12));
        let last = s.last_scored.unwrap();
        assert_eq!(
            (last.hit, last.points, last.streak),
            (Some(Hit::Full), 100, 1)
        );
    }

    #[test]
    fn silence_scores_nothing_and_singing_again_from_the_start_starts_over() {
        let (lyrics, m) = song();
        let mut s = Singing::default();
        s.prepare(Path::new("a"), &lyrics, &m, 1, Level::Medium);
        sing(&mut s, |_| None);
        assert_eq!((s.hit_words, s.scored_words, s.score), (0, 4, 0));
        // From the start again: over.
        sing(&mut s, |t| Some(melody_at(t)));
        assert_eq!((s.hit_words, s.scored_words), (4, 4));
    }

    use std::path::Path;
}
