//! The Karaoke tab and Karaoke mode ([`crate::core::karaoke`]).
//!
//! The tab is cells: the cover, the button into the mode, the sliders and
//! a few notes. The mode is one picture over the whole screen
//! ([`KaraokePicture`]: the lyrics, the count-in) that the window paints
//! in pixels over the blurred cover and a terminal draws in cells
//! ([`karaoke_cells`]); its controls are cells over it.

use ratatui::buffer::Buffer;
use ratatui::layout::{HorizontalAlignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Widget};
use ratatui::Frame;

use super::colors::*;
use super::{keep_start, Pictures, Scene};
use crate::core::karaoke;
use crate::core::lanes::{self, WordLines};
use crate::core::layout::SWITCH;
use crate::core::layout::{
    KaraokeLayout, KaraokeTabLayout, SliderLayout, KARAOKE_ENTER, KARAOKE_START,
};
use crate::core::singing::{notes_of, Hit, Level, LineNotes, Medal, Note, Singing, SongResult};
use crate::core::sliders::Slider;
use crate::core::text::{self, width};
use crate::core::{Gpu, State};

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
    /// The medal a song sung won (none: none this time), in its colour.
    Medal(Option<Medal>),
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
    /// characters, and whether fully (else half).
    pub hits: Vec<(f32, f32, bool)>,
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
    /// The level they are scored at.
    pub level: Level,
    /// The words hit in a row now.
    pub streak: u32,
}

impl Score {
    /// The multiplier now (1 to 4).
    pub fn multiplier(&self) -> u32 {
        crate::core::singing::multiplier(self.streak)
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
    let ours = singing.is_of(path) && !singing.lines.is_empty();
    let melody = match state.rhythm.get(path) {
        Some(crate::core::Rhythm::Ready(r)) => r.waveform.as_ref().and_then(|w| w.melody.clone()),
        _ => None,
    };
    let heard = t - f64::from(state.config.karaoke_lead) / 1000.0;
    for l in lines.iter_mut().filter(|l| l.kind == LineKind::Current) {
        let Some(i) = l.id else { continue };
        let notes = if ours {
            singing.lines.get(i).cloned().unwrap_or_default()
        } else {
            match (sung.get(i), &melody) {
                (Some(line), Some(m)) => notes_of(line, m),
                _ => continue,
            }
        };
        if ours {
            sung_on(singing, state.config.karaoke_game, i, &notes, heard, l);
        }
        l.notes = notes.notes;
    }
}

/// The singer's hits (`game` on), trail and pitch now, as `singing` heard
/// them, on line `i`'s `notes`, up to `heard` seconds into the song, into
/// `l`.
pub fn sung_on(
    singing: &Singing,
    game: bool,
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
    let tolerance = singing.level.tolerance();
    let mut last_at = None;
    for h in singing.heard_on(i, first.start, heard.min(last.end)) {
        let Some(at) = char_at(h.t) else { continue };
        let height = notes.height(h.cents);
        l.trail.push((at, height));
        if let (true, Some(hit)) = (game, h.hit) {
            let full = hit == Hit::Full;
            let to = char_at(h.t + 0.0099).unwrap_or(at);
            match l.hits.last_mut() {
                Some(span) if span.2 == full && (at - span.1).abs() < 0.05 => {
                    span.1 = to.max(span.1)
                }
                _ => l.hits.push((at, to.max(at), full)),
            }
        }
        last_at = Some((h.t, at, height, h.off));
    }
    // The singer now: their last pitch, if just sung.
    if let Some((t, at, height, off)) = last_at {
        if heard - t < 0.15 {
            let way = match off {
                Some(o) if o > tolerance => -1,
                Some(o) if o < -tolerance => 1,
                _ => 0,
            };
            l.voice = Some((at, height, way));
        }
    }
}

/// Karaoke mode's picture: the whole screen.
#[derive(Clone, Debug, PartialEq)]
pub struct KaraokePicture {
    /// The screen.
    pub area: Rect,
    /// The lines, top to bottom, centred.
    pub lines: Vec<LyricLine>,
    /// The count-in's number and how far through its beat, 0 to 1.
    pub count: Option<(usize, f64)>,
    /// A smaller count-in before the singing comes back after a break
    /// ([`break_cue`]), shown at the top as the count-in is, apart from
    /// the lines.
    pub cue: Option<Cue>,
    /// The controls' bar, while it shows (its cells are text over the
    /// picture).
    pub controls: Option<Rect>,
    /// The START button in it.
    pub start: Option<Rect>,
    /// The lines as they were just before they last changed, while they
    /// move to where they are now (empty once they are there).
    pub before: Vec<LyricLine>,
    /// How far they have moved, 0 to 1, eased.
    pub moved: f64,
    /// The singing's score so far (the game on).
    pub score: Option<Score>,
}

fn line(text: impl Into<String>, fill: f64, kind: LineKind) -> LyricLine {
    LyricLine {
        text: text.into(),
        fill,
        kind,
        id: None,
        alpha: 1.0,
        notes: Vec::new(),
        hits: Vec::new(),
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
        return vec![line("Queue some songs, then START", 0.0, LineKind::Info)];
    };
    // How the song before was sung, until the next is started.
    if let Some(result) = &state.karaoke.result {
        let waiting = state.karaoke.held.is_some()
            && state.karaoke.countdown.is_none()
            && state.karaoke.resume_at.is_none();
        if waiting {
            let next = state
                .details
                .get(&state.queue.paths[index])
                .and_then(|d| d.title.clone())
                .unwrap_or_else(|| crate::core::shown_title(state, &state.queue.paths[index]));
            return result_lines(result, &next);
        }
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

/// How a song was sung, as lines: its medal, its points and words hit, its
/// level and whether it beat the best, and the song coming next.
pub fn result_lines(r: &SongResult, next: &str) -> Vec<LyricLine> {
    let medal = match r.medal {
        Some(m) => format!("{} medal", m.name()),
        None if !r.whole => "Not sung from the start: no medal, no best".to_owned(),
        None => "No medal this time".to_owned(),
    };
    let best = if !r.whole {
        format!(
            "{} · your best {}",
            r.level.name(),
            points_text(r.best.unwrap_or(0))
        )
    } else if r.new_best() {
        match r.best {
            Some(_) => format!("{} · a new best!", r.level.name()),
            None => format!("{} · your first score", r.level.name()),
        }
    } else {
        format!(
            "{} · your best {}",
            r.level.name(),
            points_text(r.best.unwrap_or(0))
        )
    };
    vec![
        line(r.title.clone(), 0.0, LineKind::Info),
        line(medal, 0.0, LineKind::Medal(r.medal)),
        line(
            if r.half > 0 {
                format!(
                    "{} points · {} of {} words, {} half · {} in a row",
                    points_text(r.points),
                    r.hit,
                    r.words,
                    r.half,
                    r.streak
                )
            } else {
                format!(
                    "{} points · {} of {} words · {} in a row",
                    points_text(r.points),
                    r.hit,
                    r.words,
                    r.streak
                )
            },
            0.0,
            LineKind::Next,
        ),
        line(best, 0.0, LineKind::Info),
        line(format!("Next: {next} · START"), 0.0, LineKind::Info),
    ]
}

/// The colour of a medal (none: dim).
pub fn medal_color(medal: Option<Medal>) -> Color {
    match medal {
        Some(Medal::Gold) => HIT,
        Some(Medal::Silver) => Color::Rgb(205, 210, 222),
        Some(Medal::Bronze) => Color::Rgb(210, 135, 70),
        None => DIM,
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

/// The mode's picture now (`heard`: seconds into the track as heard).
pub fn picture(state: &State, layout: &KaraokeLayout, heard: f64) -> KaraokePicture {
    let count = match state.karaoke.resume_at {
        // A paused song's count-in, on the clock.
        Some(_) => karaoke::resume_count(state, state.drawn_ms.unwrap_or(state.now_ms)),
        None => state.karaoke.countdown.and_then(|c| {
            let playing = state.playing.track.as_ref().map(|t| t.index);
            (playing == Some(c.index)).then(|| c.at(heard)).flatten()
        }),
    };
    // Before the singing starts (however the song started) or comes back
    // after a break: the count-in again, the first one as large as START's,
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
    KaraokePicture {
        area: layout.screen,
        lines,
        count,
        cue,
        controls: layout.controls.as_ref().map(|c| c.bar),
        start: layout.controls.as_ref().and_then(|c| c.start),
        before,
        moved,
        score: score_now(state),
    }
}

/// The score of the song in the mode, the game on and its notes known.
fn score_now(state: &State) -> Option<Score> {
    let singing = &state.karaoke.singing;
    let index = karaoke::song_index(state)?;
    let path = state.queue.paths.get(index)?;
    (state.config.karaoke_game && singing.is_of(path) && singing.words() > 0).then_some(Score {
        points: singing.score,
        level: singing.level,
        streak: singing.streak,
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

/// What to do about a microphone giving only silence, for this system.
const SILENT_HINT: &str = if cfg!(target_os = "macos") {
    "Allow it in System Settings › Privacy & Security › Microphone (Jodeldiplom, or the terminal it runs in)."
} else {
    "Check the input is not muted in the system's sound settings."
};

/// Draws the tab.
pub fn draw_tab(
    frame: &mut Frame,
    scene: &Scene,
    layout: &KaraokeTabLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    if layout.cover.width > 0 && layout.cover.height > 0 {
        pictures.cover(frame, layout.cover, scene.background, false);
    }
    // The song playing, or the one the mode starts with.
    let (title, mut artist) = match &state.playing.track {
        Some(t) => (
            text::track_title(&t.info),
            t.info.tags.artist.clone().unwrap_or_default(),
        ),
        None => ("Nothing playing".into(), String::new()),
    };
    // The key it is in, once found.
    let key = crate::core::lanes::rhythmed(state, crate::core::lanes::Block::Now)
        .and_then(|r| r.analysis.key);
    if let Some(key) = key {
        if !artist.is_empty() {
            artist.push_str(" · ");
        }
        artist.push_str(&text::key_text(&key));
    }
    let max = usize::from(layout.title.width);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                keep_start(&title, max),
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(keep_start(&artist, max), Style::new().fg(DIM))),
        ]),
        layout.title,
    );
    // Greyed out until the song's lyrics are ready, why beside it.
    let why = karaoke::song_index(state).and_then(|i| karaoke::not_singable(state, i));
    button(
        frame.buffer_mut(),
        layout.enter,
        KARAOKE_ENTER,
        why.is_none(),
    );
    if let Some(why) = why {
        let x = layout.enter.right() + 2;
        let room = usize::from((layout.title.right()).saturating_sub(x));
        if room > 8 && layout.enter.height > 0 {
            frame.buffer_mut().set_string(
                x,
                layout.enter.y + layout.enter.height / 2,
                keep_start(why, room),
                Style::new().fg(DIM),
            );
        }
    }
    for s in &layout.sliders {
        slider(frame.buffer_mut(), state, s, true);
    }
    game_settings(frame.buffer_mut(), state, layout);
    super::sync::row(frame.buffer_mut(), state, layout);
    let gpu = match state.gpu {
        Gpu::Present(_) => "Here and in the mode the voice plays at its level, the rest as it is.",
        Gpu::Looking => "Still looking for a GPU for the stems.",
        Gpu::Absent => "No GPU: the voice is turned down only where its stems are kept.",
    };
    // What is known of the lyrics of the song and the next two.
    let mut lines: Vec<Line> = Vec::new();
    if let Some(first) = karaoke::song_index(state) {
        for (k, path) in state.queue.paths.iter().enumerate().skip(first).take(3) {
            let what = match karaoke::lyrics_of(state, k) {
                Some(karaoke::SongLyrics::Pending) => format!("{} looking…", text::spinner(state)),
                Some(l) => l.describe(),
                None => String::new(),
            };
            let heard = karaoke::lyrics_of(state, k).is_some_and(karaoke::SongLyrics::heard);
            let what_color = if heard { WARN } else { DIM };
            let who = if k == first { "Lyrics" } else { "  next" };
            let name = crate::core::shown_title(state, path);
            let max = usize::from(layout.notes.width).saturating_sub(width(who) + width(&what) + 5);
            // The song stands out (the one playing most); what was found of
            // its lyrics, less.
            let name_style = if k == first {
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(TEXT)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{who}  "), Style::new().fg(TEXT)),
                Span::styled(keep_start(&name, max.max(8)), name_style),
                Span::styled(format!(": {what}"), Style::new().fg(what_color)),
            ]));
            // The song's own lyrics heard by machine learning: said loudly.
            if heard && k == first {
                lines.push(Line::from(Span::styled(
                    format!("        ⚠ {}", karaoke::HEARD_WARNING),
                    Style::new().fg(WARN).add_modifier(Modifier::BOLD),
                )));
            }
        }
        lines.push(Line::default());
    }
    // The microphone: which, and how loud the singer is now (the music's
    // echo taken out).
    let mic = &state.karaoke.mic;
    let mic_line = match (&mic.name, &mic.error) {
        (Some(name), _) if mic.silent => Line::from(vec![
            Span::styled("Mic     ", Style::new().fg(TEXT)),
            Span::styled(
                format!("{name}: only silence comes in"),
                Style::new().fg(FAIL),
            ),
        ]),
        (Some(name), _) => {
            // −60 dB to 0 dB in twelve steps.
            let peak = state.mic.map_or(0.0, |m| m.singer());
            let db = 20.0 * f64::from(peak.max(1e-6)).log10();
            let [lit, unlit] = super::blocks((db + 60.0) / 60.0, 12, ACCENT);
            Line::from(vec![
                Span::styled("Mic     ", Style::new().fg(TEXT)),
                lit,
                unlit,
                Span::styled(format!("  {name}  (i: the next one)"), Style::new().fg(DIM)),
            ])
        }
        (None, Some(why)) => Line::from(vec![
            Span::styled("Mic     ", Style::new().fg(TEXT)),
            Span::styled(
                format!("not open: {why}  (i: another one)"),
                Style::new().fg(FAIL),
            ),
        ]),
        (None, None) => Line::from(Span::styled(
            format!("Mic     {} opening…", text::spinner(state)),
            Style::new().fg(DIM),
        )),
    };
    lines.push(mic_line);
    if mic.name.is_some() && mic.silent {
        lines.push(Line::from(Span::styled(
            format!("        {SILENT_HINT}"),
            Style::new().fg(DIM),
        )));
    }
    // The echo canceller: how much of the music's echo it takes out of
    // what the microphone hears.
    if mic.name.is_some() && !mic.silent {
        let echo = echo_text(state);
        lines.push(Line::from(vec![
            Span::styled("Echo    ", Style::new().fg(TEXT)),
            Span::styled(echo, Style::new().fg(DIM)),
        ]));
    }
    let words = match &state.karaoke.model {
        karaoke::ModelState::Missing => {
            "not timed: the speech model is not in this build (make puts it there)".to_owned()
        }
        karaoke::ModelState::Coming => {
            "timed once the speech model is fetched (378 MB, when first needed)".to_owned()
        }
        karaoke::ModelState::Ready => "timed against the voice by the speech model".to_owned(),
    };
    lines.push(Line::from(vec![
        Span::styled("Words   ", Style::new().fg(TEXT)),
        Span::styled(words, Style::new().fg(DIM)),
    ]));
    lines.push(Line::default());
    let notes = [
        gpu,
        "Each song waits for START, then counts in 3, 2, 1, a second each; again after a break of 8 s.",
        "The microphone only listens: it is never played; the music stays as it is.",
        "The music's echo is taken out of what it hears; your voice never is.",
        "esc in the mode comes back here.",
    ];
    lines.extend(
        notes
            .iter()
            .map(|n| Line::from(Span::styled(*n, Style::new().fg(DIM)))),
    );
    frame.render_widget(Paragraph::new(lines), layout.notes);
    if let Some(voice) = layout.voice {
        draw_voice(frame, scene, voice, pictures);
    }
    if let Some(panel) = layout.panel {
        super::sync::panel(frame.buffer_mut(), state, panel, &layout.panel_buttons);
    }
}

/// Cells a second the voice's lane shows without lyrics to fit.
const VOICE_PER: f64 = 6.0;

/// How tall the words on the tab's voice lane are where the frontend draws
/// text at any size (the window), in cells: large enough to read as they
/// go by, small enough that a song sung fast does not make them race.
pub const VOICE_TEXT: u16 = 2;

/// The span the voice's lane on the tab shows and where it starts: at the
/// zoom that leaves out no word of the song's lyrics, each where it is sung
/// ([`crate::core::lanes::lyrics_per_second`]), with each character `scale`
/// cells wide, the playhead a third of the way in.
pub fn voice_window(state: &State, width: u16, heard: f64, scale: u16) -> (f64, f64) {
    use crate::core::lanes::{fitted_words, lyrics_per_second, Block};
    let per = lanes::lyrics(state, Block::Now)
        .and_then(|l| lyrics_per_second(&fitted_words(&l), WordLines::KARAOKE.rows))
        .unwrap_or(VOICE_PER);
    let span = f64::from(width.max(1)) / (per * f64::from(scale.max(1)));
    let start = (heard - VOICE_LEAD * span).max(0.0);
    (start, start + span)
}
/// The share of the voice lane before the playhead.
pub const VOICE_LEAD: f64 = 1.0 / 3.0;

/// The voice's lane of the song playing, its words on it as they are sung.
fn draw_voice(frame: &mut Frame, scene: &Scene, area: Rect, pictures: &mut dyn Pictures) {
    use crate::core::lanes::{rhythmed, Block};
    use crate::core::layout::Lane;
    use crate::view::lanes::{lane_color, WavePlot};
    let state = scene.state;
    let r = rhythmed(state, Block::Now);
    let waveform = r
        .and_then(|r| r.waveform.clone())
        .filter(|w| w.stems.is_some());
    let (Some(r), Some(waveform)) = (r, waveform) else {
        let note = match (&state.playing.track, &state.gpu) {
            (None, _) => "The voice and its words show here while a song plays.",
            (_, Gpu::Absent) => "No GPU for the stems: the voice cannot be shown.",
            _ => "The voice and its words show here once the song's stems are in.",
        };
        let y = area.y + area.height / 2;
        frame.render_widget(
            Paragraph::new(Span::styled(note, Style::new().fg(DIM)))
                .alignment(HorizontalAlignment::Center),
            Rect::new(area.x, y, area.width, 1),
        );
        return;
    };
    let heard = scene.heard;
    // Larger words where they can be drawn larger (and the lane tall
    // enough for both their lines).
    let lines = WordLines::KARAOKE;
    let scale = pictures
        .lane_text_scale()
        .min(area.height / lines.rows as u16)
        .max(1);
    let (start, end) = voice_window(state, area.width, heard, scale);
    // The whole track's bars: the lane may be scrolled between frames.
    let a = &r.analysis;
    let downbeats = a
        .downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect();
    pictures.wave(
        frame,
        &WavePlot {
            area,
            waveform,
            lane: Lane::Vocals,
            start,
            end,
            color: lane_color(Lane::Vocals),
            downbeats,
            unclear: Vec::new(),
            selection: None,
            shown: None,
            playhead: state.playing.track.as_ref().map(|_| heard),
            cursor: None,
            cue: None,
            transition: None,
            gain: Vec::new(),
            edit: None,
            rows: (0, area.height),
            lyrics: lanes::lyrics(state, Block::Now),
            text_scale: scale,
            lines,
            follow: Some(VOICE_LEAD),
            deep: true,
            key: a.key,
        },
    );
}

/// Draws the mode: its picture, then its controls over it.
pub fn draw_mode(
    frame: &mut Frame,
    scene: &Scene,
    layout: &KaraokeLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    let picture = picture(state, layout, scene.heard);
    pictures.karaoke(frame, &picture);
    let Some(c) = &layout.controls else {
        return;
    };
    super::draw_progress(frame, state, c.elapsed, c.seek, c.total);
    let buf = frame.buffer_mut();
    for s in &c.sliders {
        slider(buf, state, s, true);
    }
    if let Some(start) = c.start {
        // Greyed out until the song's lyrics are ready (N goes on).
        let singable =
            karaoke::song_index(state).is_none_or(|i| karaoke::not_singable(state, i).is_none());
        button(buf, start, KARAOKE_START, singable);
    }
    let hint = "esc leaves  ·  ↑ ↓ slider  ·  ← → level  ·  enter / space START  ·  N next song";
    let max = usize::from(c.hint.width);
    buf.set_string(
        c.hint.x,
        c.hint.y,
        keep_start(hint, max),
        Style::new().fg(DIM),
    );
}

/// A setting's row on the tab: its name, then its choices side by side,
/// the one chosen lit (a click on one picks it), then what it means.
#[allow(clippy::too_many_arguments)]
fn choices(
    buf: &mut Buffer,
    x: u16,
    y: u16,
    name: &str,
    items: &[(Rect, &str, bool)],
    lit: bool,
    what: &str,
    right: u16,
) {
    buf.set_string(x, y, name, Style::new().fg(TEXT));
    let mut end = x;
    for &(rect, label, chosen) in items {
        let style = if chosen && lit {
            Style::new().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)
        } else if chosen {
            Style::new().fg(BG).bg(DIM)
        } else {
            Style::new().fg(DIM).bg(TRACK)
        };
        let text = format!(
            " {label:<w$}",
            w = usize::from(rect.width.saturating_sub(1))
        );
        buf.set_string(
            rect.x,
            rect.y,
            keep_start(&text, usize::from(rect.width)),
            style,
        );
        end = end.max(rect.right());
    }
    let room = usize::from(right.saturating_sub(end + 2));
    if room > 4 {
        buf.set_string(end + 2, y, keep_start(what, room), Style::new().fg(DIM));
    }
}

/// The game's settings on the tab: the game on or off, the level, the
/// notes shown or hidden; each a click (or its key: g, d, p).
fn game_settings(buf: &mut Buffer, state: &State, layout: &KaraokeTabLayout) {
    let config = &state.config;
    let right = layout.title.right();
    let switch = |rect: Rect, on: bool| -> Vec<(Rect, &'static str, bool)> {
        let at = rect.x + 12;
        vec![(
            Rect::new(at, rect.y, SWITCH, 1),
            if on { "on" } else { "off" },
            true,
        )]
    };
    let game = config.karaoke_game;
    choices(
        buf,
        layout.game.x,
        layout.game.y,
        "Game  (g)",
        &switch(layout.game, game),
        game,
        if game {
            "your singing scored, a medal and a best for each song"
        } else {
            "just singing, nothing scored"
        },
        right,
    );
    if let Some(&(first, _)) = layout.levels.first() {
        let items: Vec<(Rect, &str, bool)> = layout
            .levels
            .iter()
            .map(|&(r, level)| (r, level.name(), level == config.karaoke_level))
            .collect();
        choices(
            buf,
            layout.game.x,
            first.y,
            "Level (d)",
            &items,
            game,
            config.karaoke_level.what(),
            right,
        );
    }
    let notes = config.karaoke_notes;
    choices(
        buf,
        layout.show_notes.x,
        layout.show_notes.y,
        "Notes (p)",
        &switch(layout.show_notes, notes),
        notes,
        if notes {
            "the song's notes run by in the mode, for you to sing on"
        } else {
            "only the words"
        },
        right,
    );
    let keep = config.karaoke_keep;
    choices(
        buf,
        layout.keep.x,
        layout.keep.y,
        "Keep",
        &switch(layout.keep, keep),
        keep,
        if keep {
            "your singing of each song saved (recordings, beside the settings), to tune the game"
        } else {
            "your singing is not saved"
        },
        right,
    );
}

/// A button: a rounded border around its label, in the accent colour.
fn button(buf: &mut Buffer, area: Rect, label: &str, lit: bool) {
    if area.width < 3 || area.height == 0 {
        return;
    }
    let color = if lit { ACCENT } else { DIM };
    if area.height >= 3 {
        Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(color))
            .render(area, buf);
    }
    let y = area.y + area.height / 2;
    let w = width(label) as u16;
    let x = area.x + area.width.saturating_sub(w) / 2;
    buf.set_string(
        x,
        y,
        label,
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    );
}

/// A slider: its label (in the accent colour when the arrow keys move
/// it), the track with its knob, and the value.
fn slider(buf: &mut Buffer, state: &State, s: &SliderLayout, show_focus: bool) {
    let focused = show_focus && state.karaoke.focus == s.slider;
    let label_style = if focused {
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(TEXT)
    };
    buf.set_string(s.row.x, s.row.y, s.slider.label(), label_style);
    let n = usize::from(s.track.width);
    if n >= 2 {
        let knob = usize::from(
            crate::core::layout::Track::columns(s.track).cell(s.slider.share(state)) - s.track.x,
        );
        let on = Style::new().fg(ACCENT);
        let off = Style::new().fg(TRACK);
        for i in 0..n {
            let (symbol, style) = match i.cmp(&knob) {
                std::cmp::Ordering::Less => ("━", on),
                std::cmp::Ordering::Equal => ("●", on.add_modifier(Modifier::BOLD)),
                std::cmp::Ordering::Greater => ("─", off),
            };
            buf.set_string(s.track.x + i as u16, s.track.y, symbol, style);
        }
    }
    buf.set_string(
        s.value.x,
        s.value.y,
        s.slider.text(state),
        Style::new().fg(TEXT),
    );
    // The voice's level means nothing without stems.
    if s.slider == Slider::Voice && !matches!(state.gpu, Gpu::Present(_)) {
        buf.set_style(s.row, Style::new().fg(DIM));
    }
}

/// The digits of a count-in, five rows of blocks each.
const DIGITS: [[&str; 5]; 3] = [
    ["  █  ", " ██  ", "  █  ", "  █  ", " ███ "],
    [" ███ ", "    █", " ███ ", "█    ", "█████"],
    ["████ ", "    █", " ███ ", "    █", "████ "],
];

/// The mode's picture in cells (a terminal): the lines centred, the line
/// sung in bold with what is sung in the accent colour, and the count-in
/// in block digits.
pub fn karaoke_cells(buf: &mut Buffer, picture: &KaraokePicture) {
    let area = picture.area;
    // The lines are centred on the whole screen (the controls never move
    // them), and drawn only above the controls' bar while it shows.
    let bottom = picture.controls.map_or(area.bottom(), |c| c.y);
    // The count-in at the top, in block digits, over the lines.
    if let Some((n, _)) = picture.count {
        if let Some(rows) = n.checked_sub(1).and_then(|i| DIGITS.get(i)) {
            let y0 = area.y + 1;
            for (k, row) in rows.iter().enumerate() {
                let x = area.x + area.width.saturating_sub(5) / 2;
                buf.set_string(
                    x,
                    y0 + k as u16,
                    row,
                    Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
                );
            }
        }
    }
    // The count-in before the singing comes back after a break: its number
    // in the top row, apart from the lines (it never moves with them).
    if let Some(cue) = picture.cue {
        buf.set_string(
            area.x + area.width.saturating_sub(1) / 2,
            area.y + 1,
            cue.number.to_string(),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        );
    }
    // The score, top right: the points and the level.
    if let Some(score) = picture.score {
        let times = match score.multiplier() {
            1 => String::new(),
            m => format!("×{m}  "),
        };
        let text = format!(
            "{}  {times}{}",
            score.level.name(),
            points_text(score.points)
        );
        let w = width(&text) as u16;
        if w + 2 <= area.width {
            buf.set_string(
                area.x + area.width - w - 1,
                area.y,
                text,
                Style::new().fg(HIT).add_modifier(Modifier::BOLD),
            );
        }
    }
    let n = picture.lines.len() as u16;
    // Below the digits while they show.
    let top = if picture.count.is_some() {
        area.y + 7
    } else if picture.cue.is_some() {
        area.y + 2
    } else {
        area.y
    };
    let mut y = (area.y + area.bottom().saturating_sub(area.y + 2 * n) / 2).max(top);
    for line in &picture.lines {
        if y + 1 >= bottom {
            break;
        }
        let max = usize::from(area.width.saturating_sub(4));
        let text = keep_start(&line.text, max);
        let x = area.x + area.width.saturating_sub(width(&text) as u16) / 2;
        let (sung, rest) = match line.kind {
            // Fading out (a song's title): dim for its second half.
            _ if line.alpha < 0.5 => (Style::new().fg(DIM), Style::new().fg(DIM)),
            LineKind::Current => (
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
                Style::new().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            LineKind::Next => (Style::new().fg(TEXT), Style::new().fg(TEXT)),
            LineKind::Previous | LineKind::Info => (Style::new().fg(DIM), Style::new().fg(DIM)),
            LineKind::Warning => (
                Style::new().fg(WARN).add_modifier(Modifier::BOLD),
                Style::new().fg(WARN).add_modifier(Modifier::BOLD),
            ),
            LineKind::Medal(m) => (
                Style::new().fg(medal_color(m)).add_modifier(Modifier::BOLD),
                Style::new().fg(medal_color(m)).add_modifier(Modifier::BOLD),
            ),
        };
        let chars: Vec<char> = text.chars().collect();
        let split =
            ((line.fill.clamp(0.0, 1.0) * chars.len() as f64).round() as usize).min(chars.len());
        let head: String = chars[..split].iter().collect();
        let tail: String = chars[split..].iter().collect();
        let line_widget = Line::from(vec![Span::styled(head, sung), Span::styled(tail, rest)]);
        Paragraph::new(line_widget)
            .alignment(HorizontalAlignment::Left)
            .render(Rect::new(x, y, width(&text) as u16, 1), buf);
        // The notes its words are sung on, in the row under it, under the
        // letters sung on them (each character's cells), at five heights;
        // those sung already brighter.
        if y + 1 < bottom && line.alpha >= 0.5 && !line.notes.is_empty() {
            let mut cell_of = Vec::with_capacity(chars.len() + 1);
            let mut at = 0u16;
            for c in &chars {
                cell_of.push(at);
                at += width(&c.to_string()) as u16;
            }
            cell_of.push(at);
            let last = chars.len();
            for note in &line.notes {
                let level = (note.height * 2.0).round().clamp(-2.0, 2.0) as i32;
                let glyph = NOTE_GLYPHS[(level + 2) as usize];
                let a = (note.from.floor().max(0.0) as usize).min(last);
                let b = (note.to.ceil() as usize).clamp(a + 1, last.max(a + 1));
                let (c0, c1) = (
                    cell_of[a.min(last)],
                    cell_of[b.min(last)].max(cell_of[a.min(last)] + 1),
                );
                for c in c0..c1 {
                    let ch = cell_of.iter().rposition(|&k| k <= c).unwrap_or(0);
                    let mid = ch as f32 + 0.5;
                    let hit = line.hits.iter().find(|h| mid >= h.0 && mid < h.1);
                    let color = if let Some(&(_, _, full)) = hit {
                        if full {
                            HIT
                        } else {
                            HALF_HIT
                        }
                    } else if ch < split {
                        crate::view::lanes::NOTE
                    } else {
                        NOTE_AHEAD
                    };
                    buf[(x + c, y + 1)].set_symbol(glyph).set_fg(color);
                }
            }
            // The singer now: which way to go, or on the note.
            if let Some((at, _, way)) = line.voice {
                let ch = (at.floor().max(0.0) as usize).min(last.saturating_sub(1));
                let glyph = match way {
                    1 => "↑",
                    -1 => "↓",
                    _ => "●",
                };
                buf[(x + cell_of[ch], y + 1)]
                    .set_symbol(glyph)
                    .set_fg(SINGER);
            }
        }
        y += 2;
    }
}

/// A note's mark in the terminal, lowest to highest: scan lines at five
/// heights of a cell.
const NOTE_GLYPHS: [&str; 5] = ["⎽", "⎼", "─", "⎻", "⎺"];

/// The notes' colour: those sung ([`NOTE`]) and those to come.
///
/// [`NOTE`]: crate::view::lanes::NOTE
pub const NOTE_AHEAD: Color = Color::Rgb(78, 120, 146);
/// A note where the singer hit it: gold.
pub const HIT: Color = Color::Rgb(255, 190, 60);
/// Where they half hit it: a paler gold.
pub const HALF_HIT: Color = Color::Rgb(176, 150, 104);
/// The singer's own pitch: coral.
pub const SINGER: Color = Color::Rgb(240, 110, 80);

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
pub fn line_color(kind: LineKind) -> Color {
    match kind {
        LineKind::Current => TEXT,
        LineKind::Next => Color::Rgb(190, 190, 198),
        LineKind::Previous | LineKind::Info => DIM,
        LineKind::Warning => WARN,
        LineKind::Medal(m) => medal_color(m),
    }
}

/// What the tab says of the echo canceller.
fn echo_text(state: &State) -> String {
    match (state.config.karaoke_echo, state.mic.and_then(|m| m.echo())) {
        (false, _) => "off (karaoke_echo in the settings file)".to_owned(),
        (true, None) => "on, once music plays".to_owned(),
        (true, Some((_, false, _))) => format!("{} learning the room…", text::spinner(state)),
        (true, Some((db, true, delay))) => match delay {
            Some(d) => format!(
                "the music's echo taken out, {db:.0} dB (it comes {:.0} ms late)",
                d * 1000.0
            ),
            None => format!("the music's echo taken out, {db:.0} dB"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The tab says what the canceller does: off, waiting for music,
    /// learning, or how much of the echo it takes out.
    #[test]
    fn the_tab_says_what_the_canceller_does() {
        let mut state = State::new("/m".into(), crate::config::Config::default(), (120, 40));
        state.config.karaoke_echo = true;
        assert_eq!(echo_text(&state), "on, once music plays");
        let mut mic = crate::core::mic::MicReading {
            echo_on: true,
            voice: Some(0.1),
            ..Default::default()
        };
        state.mic = Some(mic);
        assert!(echo_text(&state).ends_with("learning the room…"));
        mic.removed_db = 42.0;
        mic.learnt = true;
        mic.delay = Some(0.04);
        state.mic = Some(mic);
        assert_eq!(
            echo_text(&state),
            "the music's echo taken out, 42 dB (it comes 40 ms late)"
        );
        state.config.karaoke_echo = false;
        assert!(echo_text(&state).starts_with("off"));
    }

    #[test]
    fn the_terminal_draws_the_notes_under_their_letters() {
        let mut current = line("high low", 0.5, LineKind::Current);
        current.notes = vec![
            Note {
                from: 0.0,
                to: 4.0,
                height: 1.0,
                start: 0.0,
                end: 1.0,
                cents: 6000.0,
                word: 0,
            },
            Note {
                from: 5.0,
                to: 8.0,
                height: -1.0,
                start: 0.0,
                end: 1.0,
                cents: 6000.0,
                word: 0,
            },
        ];
        let picture = KaraokePicture {
            area: Rect::new(0, 0, 20, 5),
            lines: vec![current],
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
        };
        let mut buf = Buffer::empty(picture.area);
        karaoke_cells(&mut buf, &picture);
        let row = |y: u16| -> String { (0..20).map(|x| buf[(x, y)].symbol().to_owned()).collect() };
        let text_row = (0..5).find(|&y| row(y).contains("high low")).unwrap();
        assert_eq!(row(text_row + 1).trim(), "⎺⎺⎺⎺ ⎽⎽⎽");
        // Sung (the first half) brighter than to come.
        let x = (0..20)
            .find(|&x| buf[(x, text_row + 1)].symbol() == "⎺")
            .unwrap();
        assert_eq!(buf[(x, text_row + 1)].fg, crate::view::lanes::NOTE);
        assert_eq!(buf[(x + 6, text_row + 1)].fg, NOTE_AHEAD);
    }
}
