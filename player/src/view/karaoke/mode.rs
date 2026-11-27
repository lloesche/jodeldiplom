//! Karaoke mode on the screen: one picture over the whole screen
//! ([`KaraokePicture`], made by [`picture`]: the lyrics as they are sung
//! ([`lines_now`]), the count-in, the score, the song's result, Next)
//! that the window draws over the cover's ground (its stage on the GPU,
//! else the lines in its own widgets), and its controls' bar over it
//! while it shows ([`KaraokePicture::controls`]): the voice slider, the
//! seek bar, the transport and the keys ([`hint`]). Next is no control
//! but the stage's button ([`KaraokePicture::next`]).

use super::super::palette;
use super::{plain_lines, result_now, ResultShown};
use crate::core::karaoke;
use crate::core::keymap::{self, Context, Op};
use crate::core::singing::{notes_of, LineNotes, Note, Singing};
use crate::core::text;
use crate::core::State;

/// Next's button on the stage, while the mode waits for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Next {
    /// Whether the song it plays can be sung yet (else greyed).
    pub ready: bool,
}

/// The keys of the mode, as its controls' bar says them, from the keymap
/// (↑ ↓ do nothing there).
pub fn hint() -> String {
    let key = |op| keymap::hint(Context::KaraokeMode, op);
    format!(
        "{} leaves  ·  {} {} level  ·  {} / {} Next  ·  {} skips the song  ·  hold {} demo",
        key(Op::Back),
        key(Op::SliderDown),
        key(Op::SliderUp),
        key(Op::KaraokeNext),
        key(Op::PlayPause),
        key(Op::Skip),
        key(Op::KaraokeDemo),
    )
}

/// The mode's picture now (`heard`: seconds into the track as heard), its
/// controls' bar taking `bar` of the screen's height from the bottom (the
/// window lays it out).
pub fn picture(state: &State, bar: f32, heard: f64) -> KaraokePicture {
    let count = match state.karaoke.resume_at {
        // A paused song's count-in, on the clock.
        Some(_) => karaoke::resume_count(state, state.drawn_ms.unwrap_or(state.now_ms)),
        None => state.karaoke.countdown.and_then(|c| {
            let playing = state.playing.track.as_ref().map(|t| t.index);
            (playing == Some(c.index)).then(|| c.at(heard)).flatten()
        }),
    };
    // Before the singing starts (however the song started) or comes back
    // after a break: the count-in again, the first one at its full size,
    // later ones half as large in the same place. Never two at once.
    let (count, cue) = match count {
        Some(c) => (Some(c), None),
        None => match song_cue(state, heard) {
            Some(c) if c.line == 0 => (Some((c.number, c.through)), None),
            other => (None, other),
        },
    };
    // The words a little ahead of the voice, to be read before they are
    // sung (`karaoke_lead`).
    let t = heard + f64::from(state.config.karaoke_lead) / 1000.0;
    let lines = lines_now(state, t);
    let (before, moved) = scrolling(&lines, t, |t| lines_now(state, t));
    let result = result_now(state);
    // Next lit once the song it plays can be sung (N goes on).
    let next = karaoke::waiting(state).then(|| Next {
        ready: karaoke::song_index(state).is_none_or(|i| karaoke::not_singable(state, i).is_none()),
    });
    KaraokePicture {
        bar,
        reach: karaoke::controls::reach(state),
        lines,
        count,
        cue,
        controls: karaoke::controls::shown(state).then(|| karaoke::controls::fade(state)),
        next,
        before,
        moved,
        // The result's own points instead.
        score: score_now(state).filter(|_| result.is_none()),
        result,
    }
}

/// What a lyric line is to the one being sung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    /// Sung now (large, filling as it is sung).
    Current,
    /// Coming next (smaller, dim).
    Next,
    /// Just sung (smaller, fading).
    Previous,
    /// Not a lyric: the title, the artist, a note.
    Info,
    /// A warning (lyrics no trusted source has), in the warning colour.
    Warning,
}

/// A line of the mode's picture.
#[derive(Clone, Debug, PartialEq)]
pub struct LyricLine {
    /// The text.
    pub text: String,
    /// How much of it is sung, 0 to 1 of its characters.
    pub fill: f64,
    /// What it is.
    pub kind: LineKind,
    /// Which of the song's lines it is (from 0), for a lyric; what moves
    /// where as the lines scroll.
    pub id: Option<usize>,
    /// How much of it shows, 0 to 1 (less while the song's title fades).
    pub alpha: f64,
    /// The notes its words are sung on, the line being sung only
    /// ([`notes_of`]); none where they are not known or not shown.
    pub notes: Vec<Note>,
    /// Where the singer hit them (the game on): from and to, in
    /// characters; a bubble full swept full ([`crate::core::bubbles`]).
    pub hits: Vec<(f32, f32)>,
    /// The bubbles with liquid in them, bursting or burst (the game on).
    pub liquid: Vec<super::super::bubbles::Liquid>,
    /// The singer's pitch along them: where (characters) and how high
    /// (as the notes' heights).
    pub trail: Vec<(f32, f32)>,
    /// The singer now: where, how high, and which way to go (1 higher, -1
    /// lower, 0 on the note).
    pub voice: Option<(f32, f32, i8)>,
}

/// What the singing scored so far, shown top right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    /// The points.
    pub points: u32,
    /// The words hit in a row now.
    pub streak: u32,
}

impl Score {
    /// The multiplier's rung now ([`crate::core::singing::rung`]).
    pub fn rung(&self) -> Option<crate::core::shine::Tier> {
        crate::core::singing::rung(self.streak)
    }

    /// The rung under the points, as the badge names it: its tier and its
    /// multiplier ("Amazing ×16"); none at ×1.
    pub fn rung_label(&self) -> Option<String> {
        self.rung()
            .map(|t| format!("{} ×{}", t.name(), t.multiplier()))
    }
}

/// The notes of the line being sung among `lines` (the song's own scoring
/// notes, the same [`notes_of`], else made here), by the melody of song
/// `index` where its analysis has one; with the singer's hits, trail and
/// pitch now on them where they sing (`t`: seconds into the song as the
/// words show it, ahead of what is heard by `karaoke_lead`). None with
/// the notes hidden (`karaoke_notes`).
fn with_notes(
    state: &State,
    index: usize,
    sung: &[&jodel_lyrics::Line],
    lines: &mut [LyricLine],
    t: f64,
) {
    if !state.config.karaoke_notes {
        return;
    }
    let path = &state.queue.paths[index];
    let singing = &state.karaoke.singing;
    let ours = singing.is_of(path) && !singing.notes.lines.is_empty();
    let melody = match state.rhythm.get(path) {
        Some(crate::core::Rhythm::Ready(r)) => r.waveform.as_ref().and_then(|w| w.melody.clone()),
        _ => None,
    };
    let heard = t - f64::from(state.config.karaoke_lead) / 1000.0;
    for l in lines.iter_mut().filter(|l| l.kind == LineKind::Current) {
        let Some(i) = l.id else { continue };
        let notes = if ours {
            singing.notes.lines.get(i).cloned().unwrap_or_default()
        } else {
            match (sung.get(i), &melody) {
                (Some(line), Some(m)) => notes_of(line, m),
                _ => continue,
            }
        };
        if ours {
            let game = (state.config.karaoke_game, state.karaoke.lag.late);
            sung_on(singing, game, i, &notes, heard, l);
        }
        l.notes = notes.notes;
    }
}

/// The singer's hits and the bubbles bursting (`game` on,
/// [`super::super::bubbles::fill`], by the singing's time: `heard` less `late`,
/// how late the voice comes), trail and pitch now, as `singing` heard
/// them, on line `i`'s `notes`, up to `heard` seconds into the song, into
/// `l`.
pub fn sung_on(
    singing: &Singing,
    (game, late): (bool, f64),
    i: usize,
    notes: &LineNotes,
    heard: f64,
    l: &mut LyricLine,
) {
    let (Some(first), Some(last)) = (notes.notes.first(), notes.notes.last()) else {
        return;
    };
    // Where on the line (characters) the song is at `t`, by its notes.
    let char_at = |t: f64| -> Option<f32> {
        let n = notes.notes.iter().find(|n| t >= n.start && t < n.end)?;
        let share = ((t - n.start) / (n.end - n.start).max(1e-3)) as f32;
        Some(n.from + share * (n.to - n.from))
    };
    let tolerance = crate::core::singing::TOLERANCE;
    let mut last_at = None;
    for h in singing.heard_on(i, first.start, heard.min(last.end)) {
        let Some(at) = char_at(h.t) else { continue };
        let height = notes.height(h.cents);
        l.trail.push((at, height));
        last_at = Some((h.t, at, height, h.off));
    }
    if game {
        let clock = crate::core::lag::sung_clock(heard, late);
        super::super::bubbles::fill(singing, i, clock, l);
    }
    // The singer now: their last pitch, if just sung.
    if let Some((t, at, height, off)) = last_at {
        // By the singing's time: its frames come that much after the song.
        if crate::core::lag::sung_clock(heard, late) - t < 0.15 {
            let way = match off {
                Some(o) if o > tolerance => -1,
                Some(o) if o < -tolerance => 1,
                _ => 0,
            };
            l.voice = Some((at, height, way));
        }
    }
}

/// Karaoke mode's picture: the whole screen, wherever the window shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct KaraokePicture {
    /// How much of the screen's height, from its bottom, the controls' bar
    /// takes, 0 to 1, shown or not: what the line under a result stays clear
    /// above. The window says, as it lays the bar out.
    pub bar: f32,
    /// How much of the screen's height, from its bottom, the pointer
    /// brings the controls up in ([`karaoke::controls::reach`]): what
    /// Next's button stays clear above.
    pub reach: f32,
    /// The lines, top to bottom, centred.
    pub lines: Vec<LyricLine>,
    /// The count-in's number and how far through its beat, 0 to 1.
    pub count: Option<(usize, f64)>,
    /// A smaller count-in before the singing comes back after a break
    /// ([`break_cue`]), shown at the top as the count-in is, apart from
    /// the lines.
    pub cue: Option<Cue>,
    /// How much the controls' bar shows while it is laid out (the
    /// window's widgets over the picture), 0 to 1
    /// ([`karaoke::controls::fade`]).
    pub controls: Option<f64>,
    /// Next's button (the stage's), while the mode waits for it.
    pub next: Option<Next>,
    /// The lines as they were just before they last changed, while they
    /// move to where they are now (empty once they are there).
    pub before: Vec<LyricLine>,
    /// How far they have moved, 0 to 1, eased.
    pub moved: f64,
    /// The singing's score so far (the game on).
    pub score: Option<Score>,
    /// How the song before was sung, while the next waits for Next (the
    /// game on): drawn on the stage, revealed beat by beat (its words are
    /// the lines then, for a window without one).
    pub result: Option<ResultShown>,
}

pub(super) fn line(text: impl Into<String>, fill: f64, kind: LineKind) -> LyricLine {
    LyricLine {
        text: text.into(),
        fill,
        kind,
        id: None,
        alpha: 1.0,
        notes: Vec::new(),
        hits: Vec::new(),
        liquid: Vec::new(),
        trail: Vec::new(),
        voice: None,
    }
}

/// A lyric line: [`line`] with which of the song's lines it is.
fn lyric(lines: &[&jodel_lyrics::Line], i: usize, fill: f64, kind: LineKind) -> LyricLine {
    LyricLine {
        id: Some(i),
        ..line(lines[i].text(), fill, kind)
    }
}

/// How long the lines take to move up when they change, in seconds.
pub const SCROLL: f64 = 0.3;

/// What the mode says with no song queued.
pub(crate) fn nothing_queued() -> String {
    format!("Queue some songs, then {PRESS_NEXT}")
}

/// What starts the next song, as the mode says it.
pub(crate) const PRESS_NEXT: &str = "press Next";

/// How early a line comes up before it is sung, in seconds.
const EARLY: f64 = 2.0;
/// A pause between lines longer than this shows a note instead of the
/// line just sung, from a second after it ends.
const PAUSE: f64 = 4.0;

/// The lines the mode shows at `t` seconds into the song. With timed
/// lyrics: the line before, the line sung (filling word by word), the two
/// after; as the song starts, its title and artist alone, fading out
/// ([`karaoke::title_shown`]), then the first line alone until it is
/// sung; a `♪` through long pauses. Untimed lyrics: their first lines.
/// None: the title and artist and what is known.
pub fn lines_now(state: &State, t: f64) -> Vec<LyricLine> {
    let Some(index) = karaoke::song_index(state) else {
        return vec![line(nothing_queued(), 0.0, LineKind::Info)];
    };
    // How the song before was sung, until the next is started: its words
    // (the stage draws it instead where it draws: [`KaraokePicture::result`]).
    if let Some(r) = result_now(state) {
        return plain_lines(&r.shown.result, &r.next_words());
    }
    let path = &state.queue.paths[index];
    let info = state
        .playing
        .track
        .as_ref()
        .filter(|tr| tr.index == index)
        .map(|tr| tr.info.clone());
    // From the file's details (read for the queue) first, so they are the
    // same while the song waits as once it plays; the opened track's tags
    // where they are missing.
    let details = state.details.get(path);
    let title = details
        .and_then(|d| d.title.clone())
        .or_else(|| info.as_ref().map(|i| text::track_title(i)))
        .unwrap_or_else(|| crate::core::shown_title(state, path));
    let artist = details
        .and_then(|d| d.artist.clone())
        .or_else(|| info.as_ref().and_then(|i| i.tags.artist.clone()))
        .unwrap_or_default();
    let heading = |note: &str| {
        let mut lines = vec![line(title.clone(), 0.0, LineKind::Current)];
        if !artist.is_empty() {
            lines.push(line(artist.clone(), 0.0, LineKind::Next));
        }
        if !note.is_empty() {
            lines.push(line(note, 0.0, LineKind::Info));
        }
        lines
    };
    // Waiting for Next while still being prepared: what is still to come
    // for it (and, Next pressed, that it starts once that is done).
    if let (true, Some(karaoke::NotSingable::Preparing(what))) =
        (karaoke::waiting(state), karaoke::not_singable(state, index))
    {
        let said = if state.karaoke.start_when_ready == Some(index) {
            "Starts as soon as it is prepared"
        } else {
            "Being prepared"
        };
        return heading(&format!("{} {said}: {what}", text::spinner(state)));
    }
    match karaoke::lyrics_of(state, index) {
        Some(karaoke::SongLyrics::Found { shown, .. }) => {
            let sung: Vec<&jodel_lyrics::Line> =
                shown.lines.iter().filter(|l| !l.words.is_empty()).collect();
            if sung.iter().all(|l| l.start().is_none()) {
                let mut lines = heading("Lyrics not timed yet");
                lines.extend(
                    sung.iter()
                        .take(3)
                        .map(|l| line(l.text(), 0.0, LineKind::Next)),
                );
                return lines;
            }
            // The title and artist alone as the song starts, fading out;
            // then the lines.
            let title = karaoke::title_shown(state, index, t);
            if title > 0.0 {
                let mut lines = heading("");
                if karaoke::lyrics_of(state, index).is_some_and(karaoke::SongLyrics::heard) {
                    lines.push(line(karaoke::HEARD_WARNING, 0.0, LineKind::Warning));
                }
                for l in &mut lines {
                    l.alpha = title;
                }
                return lines;
            }
            match timed_lines(&sung, t) {
                Some(mut lines) => {
                    with_notes(state, index, &sung, &mut lines, t);
                    lines
                }
                None => heading(""),
            }
        }
        Some(karaoke::SongLyrics::Instrumental) => heading("Instrumental"),
        Some(karaoke::SongLyrics::NotFound) => heading("No lyrics found"),
        Some(karaoke::SongLyrics::Failed(_)) => heading("The lyrics could not be looked up"),
        Some(karaoke::SongLyrics::Pending) => {
            heading(&format!("{} Looking for the lyrics…", text::spinner(state)))
        }
        None => heading(""),
    }
}

/// The lines around `t` of timed `lines` (the first alone before it is
/// sung); `None` without any. A line is sung from its first timed word
/// (the line's own time only
/// without one): the word timing follows the voice, and a line time that
/// is early would end the line before its last words are sung.
pub fn timed_lines(lines: &[&jodel_lyrics::Line], t: f64) -> Option<Vec<LyricLine>> {
    let starts: Vec<f64> = {
        let mut last = f64::NEG_INFINITY;
        lines
            .iter()
            .map(|l| {
                let sung = l.words.iter().find_map(|w| w.start).or(l.time);
                // In order, even where two lines' times cross.
                last = sung.map_or(last, |s| s.max(last));
                last.max(0.0)
            })
            .collect()
    };
    let end_of = |i: usize| -> f64 {
        let own = lines[i].words.iter().rev().find_map(|w| w.end);
        let next = starts.get(i + 1).copied();
        match (own, next) {
            (Some(e), Some(n)) => e.min(n),
            (Some(e), None) => e,
            (None, Some(n)) => n,
            (None, None) => starts[i] + 4.0,
        }
    };
    let current = starts.iter().rposition(|&s| s <= t);
    let Some(i) = current else {
        // Before the first line: it alone, waiting for its count-in; the
        // lines after it come up as it is sung.
        starts.first()?;
        return Some(vec![lyric(lines, 0, 0.0, LineKind::Current)]);
    };
    let mut out = Vec::new();
    let next = starts.get(i + 1).copied();
    // A pause: from a second after the line sung ends, when the next is
    // more than PAUSE away.
    let pausing = t > end_of(i) + 1.0 && next.is_none_or(|n| n - end_of(i) > PAUSE);
    if pausing && next.is_none_or(|n| n - t > EARLY) {
        out.push(lyric(lines, i, 1.0, LineKind::Previous));
        out.push(line("♪", 0.0, LineKind::Info));
        out.extend((i + 1..lines.len().min(i + 3)).map(|k| lyric(lines, k, 0.0, LineKind::Next)));
        return Some(out);
    }
    if pausing {
        // The pause ending: the next line moves up into its place, to be
        // sung; the line before stays where it was (never sung again).
        out.push(lyric(lines, i, 1.0, LineKind::Previous));
        out.push(lyric(lines, i + 1, 0.0, LineKind::Current));
        out.extend((i + 2..lines.len().min(i + 4)).map(|k| lyric(lines, k, 0.0, LineKind::Next)));
        return Some(out);
    }
    if i > 0 {
        out.push(lyric(lines, i - 1, 1.0, LineKind::Previous));
    }
    out.push(lyric(lines, i, fill_at(lines[i], t), LineKind::Current));
    out.extend((i + 1..lines.len().min(i + 3)).map(|k| lyric(lines, k, 0.0, LineKind::Next)));
    Some(out)
}

/// A break in the singing at least this long (from a line's last word to
/// the next line's first, or from the song's start) gets a count-in before
/// the singing comes back, in seconds.
pub const BREAK: f64 = 8.0;

/// A count-in before the singing comes back: its number, how far through
/// its second (0 to 1), and the line it leads into (which of the song's
/// lines, as [`LyricLine::id`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cue {
    /// 3, 2 or 1.
    pub number: usize,
    /// How far through its second, 0 to 1.
    pub through: f64,
    /// The line it leads into.
    pub line: usize,
}

/// The count-in at `t` (seconds into the song, as heard) before the singing
/// of timed `lines` starts (however the song was started) or comes back
/// after a break of [`BREAK`] or more: 3, 2, 1, a second each (as the
/// count-in before a song), "1" ending as the line's first word is sung.
/// Whole or not at all: only when the song has been heard without a jump
/// since `from` (seconds) from before the count would begin. `None`
/// outside such a count.
pub fn break_cue(lines: &[&jodel_lyrics::Line], t: f64, from: f64) -> Option<Cue> {
    let mut last = f64::NEG_INFINITY;
    let starts: Vec<Option<f64>> = lines
        .iter()
        .map(|l| {
            let sung = l.words.iter().find_map(|w| w.start).or(l.time)?;
            last = sung.max(last);
            Some(last.max(0.0))
        })
        .collect();
    // The next line to be sung after `t`.
    let j = starts.iter().position(|s| s.is_some_and(|s| s > t))?;
    let start = starts[j]?;
    // Where the singing before it ended: the line before's last word (its
    // start, without one), or the song's start.
    let ended = (0..j)
        .rev()
        .find_map(|k| {
            lines[k]
                .words
                .iter()
                .rev()
                .find_map(|w| w.end)
                .or(starts[k])
        })
        .unwrap_or(0.0);
    // The first line always: the singing starts there.
    if j > 0 && start - ended < BREAK {
        return None;
    }
    // Whole or not at all: less than a count's room since the song was
    // heard from there, none.
    if start - karaoke::COUNT as f64 * karaoke::COUNT_STEP < from - 0.05 {
        return None;
    }
    let (number, through) = karaoke::Countdown::before(0, start).at(t)?;
    Some(Cue {
        number,
        through,
        line: j,
    })
}

/// How much of `l` (a share of its characters, spaces between words
/// included) is sung at `t`: its words to the one being sung, and of that
/// one as much as its time has passed.
pub fn fill_at(l: &jodel_lyrics::Line, t: f64) -> f64 {
    let total = l.text().chars().count();
    if total == 0 {
        return 0.0;
    }
    let mut sung = 0.0;
    for (k, w) in l.words.iter().enumerate() {
        let chars = w.text.chars().count() as f64;
        let space = if k > 0 { 1.0 } else { 0.0 };
        match (w.start, w.end) {
            (Some(_), Some(e)) if t >= e => sung += space + chars,
            (Some(s), Some(e)) if t >= s => {
                sung += space + letters_sung(w, t).unwrap_or(chars * ((t - s) / (e - s).max(1e-3)));
                break;
            }
            (Some(_), _) => break,
            // An untimed word: sung once the next timed one is.
            (None, _) => sung += space + chars,
        }
    }
    (sung / total as f64).clamp(0.0, 1.0)
}

/// How many of `w`'s characters are sung at `t`, by its letters' times
/// (each filling from its own time to the next's, the last to the word's
/// end): a held vowel holds the fill. `None` without them.
fn letters_sung(w: &jodel_lyrics::Word, t: f64) -> Option<f64> {
    let n = w.text.chars().count();
    if w.letters.len() != n || n == 0 {
        return None;
    }
    let end = w.end?;
    let mut sung = 0.0;
    for i in 0..n {
        let from = w.letter_at(i)?;
        let to = if i + 1 < n { w.letter_at(i + 1)? } else { end };
        if t >= to {
            sung += 1.0;
        } else {
            if t > from {
                sung += (t - from) / (to - from).max(1e-3);
            }
            break;
        }
    }
    Some(sung)
}

/// The score of the song in the mode, the game on and its notes known.
fn score_now(state: &State) -> Option<Score> {
    let singing = &state.karaoke.singing;
    let index = karaoke::song_index(state)?;
    let path = state.queue.paths.get(index)?;
    (state.config.karaoke_game && singing.is_of(path) && singing.words() > 0).then_some(Score {
        points: singing.tally.score,
        streak: singing.tally.streak,
    })
}

/// [`break_cue`] for the song playing, its words timed, at `heard`.
fn song_cue(state: &State, heard: f64) -> Option<Cue> {
    let track = state.playing.track.as_ref()?;
    if state.playing.state != jodel_playback::PlayState::Playing {
        return None;
    }
    // Heard from: as the last tick noted it, unless it jumped since (a
    // seek between two ticks: from here then).
    let now = state.drawn_ms.unwrap_or(state.now_ms);
    let from = match state.karaoke.last_heard {
        Some((i, was, then)) if i == track.index => {
            let went = now.saturating_sub(then) as f64 / 1000.0;
            if (heard - was - went).abs() > 1.0 {
                heard
            } else {
                karaoke::counts_from(state, track.index)
            }
        }
        _ => heard,
    };
    let Some(karaoke::SongLyrics::Found { shown, .. }) = karaoke::lyrics_of(state, track.index)
    else {
        return None;
    };
    let sung: Vec<&jodel_lyrics::Line> =
        shown.lines.iter().filter(|l| !l.words.is_empty()).collect();
    break_cue(&sung, heard, from)
}

/// Whether the lines shown at `t` (`now`) changed within the last
/// [`SCROLL`]: the lines just before the change and how far (eased) they
/// have moved since; else none, and 1. `at` gives the lines at a time.
pub fn scrolling(
    now: &[LyricLine],
    t: f64,
    at: impl Fn(f64) -> Vec<LyricLine>,
) -> (Vec<LyricLine>, f64) {
    // What is where: a fill alone moves nothing.
    let shape = |lines: &[LyricLine]| -> Vec<(Option<usize>, LineKind, String)> {
        lines
            .iter()
            .map(|l| (l.id, l.kind, l.text.clone()))
            .collect()
    };
    let here = shape(now);
    let (mut lo, mut hi) = (t - SCROLL, t);
    let mut before = at(lo);
    if shape(&before) == here {
        return (Vec::new(), 1.0);
    }
    // The moment of the change, to a millisecond or so.
    for _ in 0..9 {
        let mid = (lo + hi) / 2.0;
        let lines = at(mid);
        if shape(&lines) == here {
            hi = mid;
        } else {
            lo = mid;
            before = lines;
        }
    }
    let p = ((t - hi) / SCROLL).clamp(0.0, 1.0);
    // Ease out: quick at first, settling.
    let eased = 1.0 - (1.0 - p).powi(3);
    (before, eased)
}

/// Points as shown: thousands apart by a thin space ("12 400").
pub fn points_text(points: u32) -> String {
    let digits = points.to_string();
    let mut out = String::new();
    for (k, c) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k).is_multiple_of(3) {
            out.push('\u{2009}');
        }
        out.push(c);
    }
    out
}

/// The colour of a line's unsung text in the window.
pub fn line_color(kind: LineKind) -> [u8; 3] {
    match kind {
        LineKind::Current => palette::TEXT,
        LineKind::Next => [190, 190, 198],
        LineKind::Previous | LineKind::Info => palette::DIM,
        LineKind::Warning => palette::WARN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under the points the rung, named as the badge names it (none at
    /// ×1), whatever the last phrase reached.
    #[test]
    fn the_score_names_the_rung() {
        let score = |streak| Score { points: 0, streak };
        assert_eq!(score(0).rung_label(), None);
        for t in crate::core::shine::Tier::ALL {
            let label = score(t.spec().streak).rung_label();
            assert_eq!(label, Some(format!("{} ×{}", t.name(), t.multiplier())));
        }
    }

    fn lines(ids: &[usize]) -> Vec<LyricLine> {
        ids.iter()
            .map(|&i| LyricLine {
                id: Some(i),
                ..line(format!("line {i}"), 0.0, LineKind::Next)
            })
            .collect()
    }

    fn timed(time: f64, words: &[(&str, f64, f64)]) -> jodel_lyrics::Line {
        jodel_lyrics::Line {
            words: words
                .iter()
                .map(|&(text, s, e)| jodel_lyrics::Word {
                    start: Some(s),
                    end: Some(e),
                    ..jodel_lyrics::Word::new(text)
                })
                .collect(),
            time: Some(time),
            ..jodel_lyrics::Line::default()
        }
    }

    /// With its letters' times a word fills letter by letter as each is
    /// sung: a held letter holds the fill; without them, evenly.
    #[test]
    fn a_word_fills_as_its_letters_are_sung() {
        let mut line = timed(1.0, &[("go", 1.0, 2.0)]);
        // "g" from 1.0 to 1.8 (held), "o" from 1.8 to the end.
        line.words[0].letters = vec![0.0, 0.8];
        assert!((fill_at(&line, 1.4) - 0.25).abs() < 1e-6);
        assert!((fill_at(&line, 1.9) - 0.75).abs() < 1e-6);
        line.words[0].letters.clear();
        assert!((fill_at(&line, 1.4) - 0.4).abs() < 1e-9);
    }

    /// The line sung is the one whose words are being sung: a line time
    /// earlier than its words neither ends the line before nor starts the
    /// next one early.
    #[test]
    fn a_line_lasts_until_its_words_are_sung() {
        // Each line's own time is a second and a half before its words.
        let a = timed(10.0, &[("one", 11.5, 12.5), ("two", 12.5, 14.0)]);
        let b = timed(13.0, &[("three", 14.5, 15.5), ("four", 15.5, 17.0)]);
        let lines = [&a, &b];
        let current = |t: f64| {
            timed_lines(&lines, t)
                .unwrap()
                .into_iter()
                .find(|l| l.kind == LineKind::Current)
                .unwrap()
        };
        // Past the next line's own time, the first is still sung, two
        // thirds of the way.
        let now = current(13.5);
        assert_eq!(now.id, Some(0));
        assert!(now.fill > 0.5 && now.fill < 1.0, "{}", now.fill);
        // Its last word done, the next comes in on its first word.
        assert_eq!(current(14.2).id, Some(0));
        let next = current(14.6);
        assert_eq!(next.id, Some(1));
        assert!(next.fill > 0.0, "{}", next.fill);
    }

    /// Before the singing comes back after a long break, and before the
    /// first line, a count-in: 3, 2, 1, a second each, "1" ending on the
    /// first word; none after a short break; whole or not at all.
    #[test]
    fn a_long_break_counts_in_again_whole_or_not_at_all() {
        let a = timed(6.0, &[("one", 6.0, 7.0), ("two", 7.0, 8.0)]);
        let b = timed(30.0, &[("three", 30.0, 31.0)]);
        let c = timed(34.0, &[("four", 34.0, 35.0)]);
        let lines = [&a, &b, &c];
        let cue = |t: f64, from: f64| break_cue(&lines, t, from).map(|c| (c.number, c.line));
        // Before the first line (heard from the start): 3, 2, 1.
        assert_eq!(cue(2.5, 0.0), None, "before the count");
        assert_eq!(cue(3.5, 0.0), Some((3, 0)));
        assert_eq!(cue(4.5, 0.0), Some((2, 0)));
        assert_eq!(cue(5.5, 0.0), Some((1, 0)));
        assert_eq!(cue(6.0, 0.0), None, "over as the word is sung");
        // After 22 s without singing: again, before "three".
        assert_eq!(cue(26.5, 0.0), None, "not yet");
        assert_eq!(cue(27.5, 0.0), Some((3, 1)));
        assert_eq!(cue(29.9, 0.0), Some((1, 1)));
        // After 3 s: none.
        assert_eq!(cue(32.5, 0.0), None);
        // Heard from under three seconds before a first word (the mode
        // entered there, a seek): no count at all, not a part of one.
        assert_eq!(cue(28.5, 28.0), None);
        assert_eq!(cue(28.5, 26.9), Some((2, 1)), "heard from before it began");
        assert_eq!(cue(4.0, 3.5), None);
        // A first line under three seconds into the song: none.
        let early = timed(1.5, &[("hi", 1.5, 2.0)]);
        assert_eq!(break_cue(&[&early], 0.5, 0.0), None);
    }

    /// After a pause the next line comes up from below, and the line sung
    /// before it is never the one sung again: the lines only ever move up.
    #[test]
    fn after_a_pause_the_lines_go_on_up() {
        let a = timed(10.0, &[("one", 10.0, 11.0), ("two", 11.0, 12.0)]);
        let b = timed(30.0, &[("three", 30.0, 31.0), ("four", 31.0, 32.0)]);
        let c = timed(33.0, &[("five", 33.0, 34.0)]);
        let lines = [&a, &b, &c];
        let shown = |t: f64| timed_lines(&lines, t).unwrap();
        let current = |t: f64| {
            shown(t)
                .into_iter()
                .find(|l| l.kind == LineKind::Current)
                .and_then(|l| l.id)
        };
        // In the pause: the note.
        assert!(shown(20.0).iter().any(|l| l.text == "♪"));
        assert_eq!(current(20.0), None);
        // From two seconds before the next line to its end: that one, and
        // the first only ever as the line before.
        let mut t = 27.0;
        while t < 32.0 {
            let now = current(t);
            assert_ne!(now, Some(0), "at {t}");
            if t >= 28.05 {
                assert_eq!(now, Some(1), "at {t}");
            }
            t += 0.05;
        }
        // Coming up, unfilled; then filling.
        let up = shown(29.0);
        let line_b = up.iter().find(|l| l.id == Some(1)).unwrap();
        assert_eq!((line_b.kind, line_b.fill), (LineKind::Current, 0.0));
        assert!(up
            .iter()
            .any(|l| l.id == Some(0) && l.kind == LineKind::Previous));
        // The same lines in the same places as it starts: nothing scrolls.
        let ids = |v: Vec<LyricLine>| v.into_iter().map(|l| (l.id, l.kind)).collect::<Vec<_>>();
        assert_eq!(ids(shown(29.9)), ids(shown(30.2)));
    }

    /// The lines move for [`SCROLL`] after they change, from what they were
    /// just before; a fill changing alone moves nothing.
    #[test]
    fn the_lines_scroll_from_where_they_were() {
        let at = |t: f64| {
            if t < 10.0 {
                lines(&[0, 1, 2])
            } else {
                lines(&[1, 2, 3])
            }
        };
        let (before, moved) = scrolling(&at(10.15), 10.15, at);
        assert_eq!(before, lines(&[0, 1, 2]));
        // Half the time: eased, most of the way.
        assert!((moved - 0.875).abs() < 0.02, "{moved}");
        let (before, moved) = scrolling(&at(10.5), 10.5, at);
        assert!(before.is_empty() && moved == 1.0);
        let filling = |t: f64| {
            let mut l = lines(&[1, 2]);
            l[0].fill = t / 20.0;
            l
        };
        let (before, moved) = scrolling(&filling(11.0), 11.0, filling);
        assert!(before.is_empty() && moved == 1.0);
    }
}
