//! A song's notes, line by line: from the voice's melody, under the
//! words' letters ([`notes_of`]).

use jodel_rhythm::waveform::Melody;

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

    /// How high its notes `notes` are together: their heights' mean over
    /// their time (0 for none).
    pub fn word_height(&self, notes: std::ops::Range<usize>) -> f32 {
        let notes = self.notes.get(notes).unwrap_or_default();
        let long = |n: &Note| (n.end - n.start).max(1e-3);
        let time: f64 = notes.iter().map(long).sum();
        let sum: f64 = notes.iter().map(|n| f64::from(n.height) * long(n)).sum();
        (sum / time.max(1e-9)) as f32
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
/// lead word, not a backing one nor one marked as not to be sung
/// ([`jodel_lyrics::Word::not_sung`]), what is not a note taken out), each under
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
        // neighbours, close enough for the melody sung there. Not a word
        // marked as not to be sung: the record sings it.
        let lead = w.start.is_some() && w.end.is_some() && !w.backing && !w.edit.not_sung;
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
        // So a word's notes lie end to end, in order, none overlapping
        // another (what a word's bubble drawn through them counts on).
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

/// A song's notes as the game holds the singing to them: each sung line's
/// ([`notes_of`]), and in time every note, every word (by when it ends) and
/// every line with notes (by when it ends); moved into the key the song is
/// heard in ([`Notes::set_key`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Notes {
    /// Each sung line's notes, by its index among the song's sung lines.
    pub lines: Vec<LineNotes>,
    /// Every note in time: start, end, line, note.
    order: Vec<(f64, f64, usize, usize)>,
    /// Every word with notes, by when it ends: its line, word and end.
    words: Vec<(usize, usize, f64)>,
    /// Each of those words' place among them, by its line and word.
    word_order: std::collections::HashMap<(usize, usize), usize>,
    /// Every line with notes, by when it ends: its line and end.
    phrases: Vec<(usize, f64)>,
    /// The key they are in, in semitones from the song's own.
    shift: i8,
}

impl Notes {
    /// The notes `melody` sings `lyrics`' sung lines (those with words) on,
    /// in the song's own key.
    pub fn new(lyrics: &jodel_lyrics::Lyrics, melody: &Melody) -> Notes {
        let lines: Vec<LineNotes> = lyrics
            .lines
            .iter()
            .filter(|l| !l.words.is_empty())
            .map(|l| notes_of(l, melody))
            .collect();
        let mut order: Vec<(f64, f64, usize, usize)> = lines
            .iter()
            .enumerate()
            .flat_map(|(l, ln)| {
                ln.notes
                    .iter()
                    .enumerate()
                    .map(move |(k, n)| (n.start, n.end, l, k))
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut ends: std::collections::HashMap<(usize, usize), f64> = Default::default();
        for (l, ln) in lines.iter().enumerate() {
            for n in &ln.notes {
                let e = ends.entry((l, n.word)).or_insert(n.end);
                *e = e.max(n.end);
            }
        }
        let mut words: Vec<(usize, usize, f64)> =
            ends.into_iter().map(|((l, w), e)| (l, w, e)).collect();
        words.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        let mut phrases: Vec<(usize, f64)> = (lines.iter().enumerate())
            .filter_map(|(l, ln)| Some((l, ln.notes.last()?.end)))
            .collect();
        phrases.sort_by(|a, b| a.1.total_cmp(&b.1));
        let word_order = (words.iter().enumerate())
            .map(|(k, &(l, w, _))| ((l, w), k))
            .collect();
        Notes {
            lines,
            order,
            words,
            word_order,
            phrases,
            shift: 0,
        }
    }

    /// Into the key `semitones` from the song's own: every note moved by
    /// the change, their heights in their lines as they were. How far they
    /// moved, in hundredths of a semitone.
    pub fn set_key(&mut self, semitones: i8) -> f64 {
        let by = 100.0 * f64::from(i32::from(semitones) - i32::from(self.shift));
        self.shift = semitones;
        for line in &mut self.lines {
            line.mid += by;
            for note in &mut line.notes {
                note.cents += by;
            }
        }
        by
    }

    /// The key they are in, in semitones from the song's own.
    pub fn shift(&self) -> i8 {
        self.shift
    }

    /// The note sung at `t` (none between the notes): its line and index.
    pub fn note_at(&self, t: f64) -> Option<(usize, usize)> {
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

    /// Every word with notes, by when it ends: its line, word and end.
    pub fn words(&self) -> &[(usize, usize, f64)] {
        &self.words
    }

    /// Line `l`'s word `w`'s place among [`Notes::words`].
    pub fn word_order(&self, l: usize, w: usize) -> Option<usize> {
        self.word_order.get(&(l, w)).copied()
    }

    /// Every line with notes, by when it ends: its line and end.
    pub fn phrases(&self) -> &[(usize, f64)] {
        &self.phrases
    }
}
