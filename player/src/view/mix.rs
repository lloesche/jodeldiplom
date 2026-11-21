//! The Mix tab: per block a header, the whole track in one row, the bar
//! numbers and the lanes (the mix and each stem) over the span shown.
//!
//! The waveforms are [`WavePlot`]s, drawn by the [`Pictures`]: in cells as
//! braille (two dot columns and four dot rows a cell, mirrored about the
//! middle, here), in pixels by the window. Bars start where a lane's cell
//! is shaded (a line in pixels), the selection is tinted, the playhead is
//! the accent colour, and stretches where the beat is unclear are dimmer.
//! The voice's lane carries the lyrics, once timed, along its bottom row:
//! each word where it is sung, each letter in the cell where the aligner
//! heard it, a line where a sound is held (`Mi───ster`, `go────`; a word
//! without its letters' times spread by syllables), the word heard now in
//! the accent colour ([`lane_labels`]), on the lane's middle row. `L` (the
//! `mix_lyrics` setting) turns them off.

use std::sync::Arc;

use jodel_rhythm::waveform::{self, Waveform};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::{ACCENT, DIM, TEXT};
use super::plot::{braille, braille_bit};
use super::{Pictures, Scene};
use crate::core::layout::{Lane, MixBlockLayout, MixLayout};
use crate::core::mix::{self, Block};
use crate::core::transition::Pairing;
use crate::core::{text, Rhythm, State};

/// Text on a coloured label.
const BG_TEXT: Color = Color::Rgb(20, 20, 24);
/// A bar's first beat, behind the waveform.
pub const BAR_BG: Color = Color::Rgb(40, 40, 52);
/// The selection, behind the waveform.
pub const SELECT_BG: Color = Color::Rgb(58, 58, 86);
/// The span the lanes show, in the overview.
pub const SHOWN_BG: Color = Color::Rgb(46, 46, 58);
/// The rest of the track, in the overview: shaded, so the span shown
/// stands out.
pub const OUTSIDE_BG: Color = Color::Rgb(8, 8, 10);
/// How bright the waveform is outside the span shown, in the overview.
pub const OUTSIDE_GAIN: f32 = 0.35;
/// The cursor's beat.
pub const CURSOR_BG: Color = Color::Rgb(96, 90, 28);
/// The next track's cue bar.
pub const CUE: Color = Color::Rgb(60, 200, 140);
/// A transition's window, behind the waveform.
pub const TRANSITION_BG: Color = Color::Rgb(34, 46, 58);
/// A transition's fade curve.
pub const CURVE: Color = Color::Rgb(250, 250, 255);
/// The voice's notes on the Karaoke tab's lane: Okabe–Ito's sky blue,
/// lighter, apart from the voice's own colour and the words'.
pub const NOTE: Color = Color::Rgb(128, 200, 240);

/// `color` at `gain` of its brightness over the background (a stem faded
/// down by a transition looks dimmer).
pub fn faded(color: Color, gain: f32) -> Color {
    let g = gain.clamp(0.0, 1.0);
    match color {
        Color::Rgb(r, gr, b) => {
            let mix = |c: u8, bg: u8| {
                (f32::from(bg) + (f32::from(c) - f32::from(bg)) * (0.25 + 0.75 * g)) as u8
            };
            Color::Rgb(mix(r, 17), mix(gr, 17), mix(b, 21))
        }
        other => other,
    }
}

/// A stem turned off (not heard), whatever its colour.
pub const OFF: Color = Color::Rgb(78, 78, 86);

/// Whether a lane is a stem turned off.
fn is_off(state: &State, lane: Lane) -> bool {
    lane.stem()
        .is_some_and(|i| !state.stems.contains(jodel_playback::stems::Stem::ALL[i]))
}

/// A lane's colour.
pub fn lane_color(lane: Lane) -> Color {
    match lane {
        // The stems in the Okabe–Ito hues (orange, sky blue, bluish green,
        // reddish purple): told apart with every common colour-vision
        // deficiency; the mix a neutral grey.
        Lane::Mix => Color::Rgb(169, 171, 181),
        Lane::Drums => Color::Rgb(230, 159, 0),
        Lane::Bass => Color::Rgb(86, 180, 233),
        Lane::Other => Color::Rgb(0, 158, 115),
        Lane::Vocals => Color::Rgb(204, 121, 167),
    }
}

/// A colour a share of the way to the background (for unclear beats).
pub fn dimmed(color: Color) -> Color {
    match color {
        Color::Rgb(r, g, b) => Color::Rgb(r / 2 + 8, g / 2 + 8, b / 2 + 10),
        other => other,
    }
}

/// One lane's waveform over `start..end` seconds, with the marks over it.
#[derive(Clone, Debug)]
pub struct WavePlot {
    /// Where it goes.
    pub area: Rect,
    /// The track's waveform.
    pub waveform: Arc<Waveform>,
    /// Which lane of it.
    pub lane: Lane,
    /// The seconds shown.
    pub start: f64,
    /// Up to.
    pub end: f64,
    /// The waveform's colour.
    pub color: Color,
    /// The bars' first beats, in seconds.
    pub downbeats: Vec<f64>,
    /// Stretches where the beat is unclear.
    pub unclear: Vec<(f64, f64)>,
    /// The selection.
    pub selection: Option<(f64, f64)>,
    /// In the overview: the span the lanes show.
    pub shown: Option<(f64, f64)>,
    /// What is heard now (the playing track).
    pub playhead: Option<f64>,
    /// The cursor's beat.
    pub cursor: Option<f64>,
    /// The next track's cue bar.
    pub cue: Option<f64>,
    /// A transition's window on this track.
    pub transition: Option<(f64, f64)>,
    /// The lane's gain over the window: times and gains, in order (empty
    /// without a transition).
    pub gain: Vec<(f64, f32)>,
    /// The bar being edited.
    pub edit: Option<f64>,
    /// The rows of the lane above `area` (out of view) and all its rows:
    /// `area` shows part of it while its block slides.
    pub rows: (u16, u16),
    /// The song's lyrics, whose timed words go along the lane (the voice's
    /// lane only): shared, never copied, frame to frame.
    pub lyrics: Option<Arc<jodel_lyrics::Lyrics>>,
    /// How many cells tall, and each character how many wide, the words
    /// are drawn (1: the cells' own; [`super::Pictures::lane_text_scale`]).
    pub text_scale: u16,
    /// The lines the words go on ([`mix::WordLines`]).
    pub lines: mix::WordLines,
    /// The lane follows the playhead, which stays this share of the way in
    /// (its start `playhead − share × span`, not before the track's): a
    /// frontend may scroll it by itself between frames. Its marks (the
    /// bars) are then the whole track's.
    pub follow: Option<f64>,
    /// Words are read over it (the Karaoke tab's voice lane): where it
    /// is drawn in pixels, its body deep and its peaks bright, and the
    /// voice's notes over it ([`jodel_rhythm::waveform::Melody`]).
    pub deep: bool,
    /// The song's key, whose scale the notes stand on (lines across the
    /// lane where it is `deep`).
    pub key: Option<jodel_rhythm::key::Key>,
}

/// What a label holds a sound with: a line, joined from cell to cell.
pub const HOLD: char = '─';

/// A word's label on the voice's lane.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Where it starts, in the lane's units (cells, or pixels).
    pub at: f64,
    /// Its characters, one a character's width, [`HOLD`] where a sound is
    /// held.
    pub text: String,
    /// It is heard now.
    pub heard: bool,
    /// Its line (0: the first; [`mix::WordLines`]).
    pub row: usize,
}

pub use crate::core::mix::timed_words;

/// `word` over `wide` cells (at least its characters), each character in
/// the cell its time (`letters`, seconds after the word's start) falls in
/// at `per` cells a second, after the one before and leaving room for
/// those after; [`HOLD`] in the cells between, and after the last up to
/// `wide` (held).
pub fn timed_label(word: &str, letters: &[f32], per: f64, wide: usize) -> String {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    if n == 0 || letters.len() != n || wide < n {
        return word_label(word, wide);
    }
    let mut out = vec![HOLD; wide];
    let mut prev: Option<usize> = None;
    for (i, (&c, &t)) in chars.iter().zip(letters).enumerate() {
        let mut p = (f64::from(t) * per).floor().max(0.0) as usize;
        p = p.min(wide - (n - i));
        if let Some(q) = prev {
            p = p.max(q + 1);
        }
        out[p] = c;
        prev = Some(p);
    }
    out.into_iter().collect()
}

/// `word` cut into its syllables, roughly: a part a group of vowels, the
/// consonants between two groups split before the last one (`Mis`, `ter`;
/// `ba`, `by`), or before the last two where they go together (`ta`,
/// `ble`; `se`, `cret`), a silent final `e` kept with the part before. A
/// word with no Latin vowels (CJK) is a part a character.
pub fn syllable_parts(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let vowel = |c: char| {
        let c = c.to_lowercase().next().unwrap_or(c);
        "aeiouyàáâäãåèéêëìíîïòóôöõùúûüýÿæœø".contains(c)
    };
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if vowel(c) {
            match groups.last_mut() {
                Some(g) if g.1 == i => g.1 = i + 1,
                _ => groups.push((i, i + 1)),
            }
        }
    }
    if groups.is_empty() {
        let letters = chars.iter().filter(|c| c.is_alphabetic()).count();
        if letters > 1 && chars.iter().all(|c| !c.is_ascii() || !c.is_alphanumeric()) {
            return chars.iter().map(|c| c.to_string()).collect();
        }
        return vec![word.to_owned()];
    }
    // A silent final `e` (after a consonant, not `-le`) is no syllable.
    let last_letter = chars.iter().rposition(|c| c.is_alphabetic());
    if let (Some(&(s, e)), Some(end)) = (groups.last(), last_letter) {
        let lower = |i: usize| chars[i].to_lowercase().next().unwrap_or(chars[i]);
        let silent = groups.len() > 1
            && e == s + 1
            && s == end
            && lower(s) == 'e'
            && s >= 2
            && !vowel(chars[s - 1])
            && !(lower(s - 1) == 'l' && !vowel(chars[s - 2]));
        if silent {
            groups.pop();
        }
    }
    let lower = |i: usize| chars[i].to_lowercase().next().unwrap_or(chars[i]);
    let mut cuts = vec![0];
    for pair in groups.windows(2) {
        let (end, next) = (pair[0].1, pair[1].0);
        // A consonant and `l` or `r` begin a syllable together.
        let blend = next >= end + 2
            && matches!(lower(next - 1), 'l' | 'r')
            && "bcdfgkpt".contains(lower(next - 2));
        let cut = if blend {
            next - 2
        } else {
            next.saturating_sub(1)
        };
        cuts.push(end.max(cut));
    }
    cuts.push(chars.len());
    cuts.windows(2)
        .map(|c| chars[c[0]..c[1]].iter().collect())
        .filter(|p: &String| !p.is_empty())
        .collect()
}

/// `word` over `slots` character cells: as it is when that is its length
/// or less (cut to fit); else its syllables with [`HOLD`] between them,
/// spread evenly over the slots (one syllable: [`HOLD`] after it, held).
pub fn word_label(word: &str, slots: usize) -> String {
    let n = word.chars().count();
    if slots <= n {
        return word.chars().take(slots).collect();
    }
    let extra = slots - n;
    let parts = syllable_parts(word);
    let hold = |n: usize| std::iter::repeat_n(HOLD, n).collect::<String>();
    if parts.len() < 2 {
        return format!("{word}{}", hold(extra));
    }
    let gaps = parts.len() - 1;
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        out.push_str(part);
        if i < gaps {
            out.push_str(&hold(extra * (i + 1) / gaps - extra * i / gaps));
        }
    }
    out
}

/// The labels of `words` (timed, in order: the whole song's) along a lane,
/// those reaching into `view` (`lo..hi`, in the lane's units), as
/// [`mix::lay_out_words`] places them: each a character `cell` units wide
/// at `per` units a second, as wide as it is sung ([`timed_label`], else
/// [`word_label`]), and whether it is heard at `playhead`.
#[allow(clippy::too_many_arguments)]
pub fn lane_labels(
    words: &[&jodel_lyrics::Word],
    x_of: &dyn Fn(f64) -> f64,
    per: f64,
    cell: f64,
    lines: mix::WordLines,
    playhead: Option<f64>,
    (lo, hi): (f64, f64),
) -> Vec<Label> {
    let chars_per_second = per / cell.max(1e-9);
    let mut out = Vec::new();
    // Words come in order of their start, but one placed late (on a line
    // freeing after it) can start past the view while a later one, on
    // another line, is still in it: stop only once a whole line's width
    // past it.
    let stop = hi + (hi - lo).max(0.0);
    mix::lay_out_words(words, x_of, per, cell, lines, &mut |k, at, wide, row| {
        if at >= stop {
            return false;
        }
        if at >= hi {
            return true;
        }
        // Only what reaches into the view is made.
        if at + wide as f64 * cell <= lo {
            return true;
        }
        let w = words[k];
        let text = if w.letters.is_empty() {
            word_label(&w.text, wide)
        } else {
            timed_label(&w.text, &w.letters, chars_per_second, wide)
        };
        let heard = playhead
            .is_some_and(|p| w.start.is_some_and(|s| p >= s) && w.end.is_some_and(|e| p < e));
        out.push(Label {
            at,
            text,
            heard,
            row,
        });
        true
    });
    out
}

impl WavePlot {
    /// The lane's gain at `time` (1 outside the transition's curve).
    pub fn gain_at(&self, time: f64) -> f32 {
        let g = &self.gain;
        if g.is_empty() {
            return 1.0;
        }
        let i = g.partition_point(|p| p.0 <= time);
        match (i.checked_sub(1).map(|j| g[j]), g.get(i)) {
            (Some(a), Some(b)) if b.0 > a.0 => {
                let u = ((time - a.0) / (b.0 - a.0)) as f32;
                a.1 + (b.1 - a.1) * u
            }
            (Some(a), _) => a.1,
            (None, Some(b)) => b.1,
            (None, None) => 1.0,
        }
    }

    /// The seconds at the left edge of column `x` of `columns`.
    pub fn time_at(&self, x: f64, columns: f64) -> f64 {
        self.start + (self.end - self.start) * x / columns.max(1.0)
    }

    /// The lane's levels.
    pub fn levels(&self) -> Option<&waveform::Lane> {
        match self.lane.stem() {
            None => Some(&self.waveform.mix),
            Some(i) => self.waveform.stems.as_ref().map(|s| &s[i]),
        }
    }

    /// Whether `time` is in a stretch where the beat is unclear.
    pub fn is_unclear(&self, time: f64) -> bool {
        self.unclear.iter().any(|&(a, b)| time >= a && time < b)
    }

    /// Whether `time` falls in `start..end` of the column it starts.
    fn within(time: Option<f64>, start: f64, end: f64) -> bool {
        time.is_some_and(|t| t >= start && t < end)
    }
}
/// A lane in braille: for each half cell its peak, mirrored about the
/// middle row, and the marks as cell backgrounds.
pub fn wave_cells(buf: &mut Buffer, plot: &WavePlot) {
    let area = plot.area.intersection(buf.area);
    if area.width == 0 || area.height == 0 {
        return;
    }
    let Some(levels) = plot.levels() else {
        return;
    };
    // The whole lane's rows, of which `area` shows those from `skip`.
    let (skip, full) = (plot.rows.0, plot.rows.1.max(area.height));
    let dots_x = 2 * usize::from(area.width);
    let dots_y = 4 * usize::from(full);
    let half = dots_y as f64 / 2.0;
    let mut bits = vec![0u8; usize::from(area.width) * usize::from(full)];
    for dx in 0..dots_x {
        let (t0, t1) = (
            plot.time_at(dx as f64, dots_x as f64),
            plot.time_at(dx as f64 + 1.0, dots_x as f64),
        );
        let v = f64::from(levels.peak_between(t0, t1));
        if v <= 0.02 {
            continue;
        }
        let reach = (v * half).round().max(1.0) as usize;
        let mid = dots_y / 2;
        for dy in mid.saturating_sub(reach)..(mid + reach).min(dots_y) {
            let (cx, cy) = (dx / 2, dy / 4);
            bits[cy * usize::from(area.width) + cx] |= braille_bit(dx % 2, dy % 4);
        }
    }
    // A transition's fade as a line of dots, from the bottom (off) to the
    // top (full), each dot column joined to the last; its cells take the
    // curve's colour. In a lane of one row the brightness says it alone.
    let mut curve = vec![0u8; bits.len()];
    if !plot.gain.is_empty() && full >= 2 {
        let mut last: Option<usize> = None;
        for dx in 0..dots_x {
            let t = plot.time_at(dx as f64 + 0.5, dots_x as f64);
            if !plot.transition.is_some_and(|(a, b)| t >= a && t <= b) {
                last = None;
                continue;
            }
            let g = f64::from(plot.gain_at(t).clamp(0.0, 1.0));
            let y = ((1.0 - g) * (dots_y - 1) as f64).round() as usize;
            let (lo, hi) = last.map_or((y, y), |p| (p.min(y), p.max(y)));
            for dy in lo..=hi {
                curve[(dy / 4) * usize::from(area.width) + dx / 2] |= braille_bit(dx % 2, dy % 4);
            }
            last = Some(y);
        }
    }
    for cx in 0..area.width {
        let (t0, t1) = (
            plot.time_at(f64::from(cx), f64::from(area.width)),
            plot.time_at(f64::from(cx) + 1.0, f64::from(area.width)),
        );
        let bar = plot.downbeats.iter().any(|&d| d >= t0 && d < t1);
        let selected = plot.selection.is_some_and(|(a, b)| t1 > a && t0 < b);
        let shown = plot.shown.is_some_and(|(a, b)| t1 > a && t0 < b);
        let outside = plot.shown.is_some() && !shown;
        let mid = 0.5 * (t0 + t1);
        let mut fg = if plot.is_unclear(mid) {
            dimmed(plot.color)
        } else {
            plot.color
        };
        fg = faded(fg, plot.gain_at(mid));
        if outside {
            fg = faded(fg, OUTSIDE_GAIN);
        }
        let in_transition = plot.transition.is_some_and(|(a, b)| t1 > a && t0 < b);
        let bg = if WavePlot::within(plot.edit, t0, t1) {
            Some(CURVE)
        } else if WavePlot::within(plot.cursor, t0, t1) {
            Some(CURSOR_BG)
        } else if WavePlot::within(plot.cue, t0, t1) {
            Some(dimmed(CUE))
        } else if selected {
            Some(SELECT_BG)
        } else if shown {
            Some(SHOWN_BG)
        } else if bar {
            Some(BAR_BG)
        } else if in_transition {
            Some(TRANSITION_BG)
        } else if outside {
            Some(OUTSIDE_BG)
        } else {
            None
        };
        let head = WavePlot::within(plot.playhead, t0, t1);
        if head {
            fg = ACCENT;
        }
        for cy in 0..area.height {
            let row = usize::from((cy + skip).min(full - 1));
            let at = row * usize::from(area.width) + usize::from(cx);
            let (b, line) = (bits[at] | curve[at], curve[at] != 0);
            let cell = &mut buf[(area.x + cx, area.y + cy)];
            if head && b == 0 {
                cell.set_symbol("│");
            } else {
                let mut utf8 = [0u8; 4];
                cell.set_symbol(braille(b).encode_utf8(&mut utf8));
            }
            cell.set_fg(if line && !head { CURVE } else { fg });
            if let Some(bg) = bg {
                cell.set_bg(bg);
            }
        }
    }
    // The words along the lane's middle row (the upper of two), and the
    // lines under it, where they show.
    let rows = plot.lines.rows.max(1) as u16;
    let middle = ((full - 1) / 2).min(full.saturating_sub(rows));
    let lyrics = plot.lyrics.as_deref();
    if let Some(lyrics) = lyrics.filter(|_| middle >= skip && middle - skip < area.height) {
        let y0 = area.y + middle - skip;
        let per = f64::from(area.width) / (plot.end - plot.start).max(1e-9);
        let words = timed_words(lyrics);
        // Whole cells from the track's start, so the words keep their
        // cells as the view scrolls.
        let x_of = |t: f64| (t * per).floor();
        let lo = (plot.start * per).floor();
        let view = (lo, lo + f64::from(area.width));
        let labels = lane_labels(&words, &x_of, per, 1.0, plot.lines, plot.playhead, view);
        for label in labels {
            let y = y0 + label.row as u16;
            if y >= area.bottom() {
                continue;
            }
            let fg = if label.heard { ACCENT } else { TEXT };
            for (i, c) in label.text.chars().enumerate() {
                let x = label.at - lo + i as f64;
                if x < 0.0 || x >= f64::from(area.width) {
                    continue;
                }
                let mut utf8 = [0u8; 4];
                buf[(area.x + x as u16, y)]
                    .set_symbol(c.encode_utf8(&mut utf8))
                    .set_fg(fg);
            }
        }
    }
}

/// The seconds of the bars' first beats of `a` within `start..end`.
fn downbeats_within(a: &jodel_rhythm::Analysis, start: f64, end: f64) -> Vec<f64> {
    a.downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .filter(|&t| t >= start && t < end)
        .collect()
}

/// Where the beat is unclear in `a`, as stretches of seconds.
fn unclear(a: &jodel_rhythm::Analysis) -> Vec<(f64, f64)> {
    a.regions
        .iter()
        .filter(|r| r.confidence < jodel_rhythm::CLEAR)
        .map(|r| (r.start, r.end))
        .collect()
}

/// The header line of a block.
fn header(state: &State, which: Block, focused: bool, heard: f64) -> Line<'static> {
    let mark = if focused { "▸ " } else { "  " };
    let name = match which {
        Block::Now => "Now",
        Block::Next => "Next",
        Block::Prev => "Last",
    };
    let style = if focused {
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(DIM)
    };
    let mut spans = vec![Span::styled(format!("{mark}{name:<5}"), style)];
    let Some(path) = mix::path(state, which) else {
        let what = match which {
            Block::Now => "nothing playing",
            Block::Next => "nothing after it in the queue",
            Block::Prev => "",
        };
        spans.push(Span::styled(what, Style::new().fg(DIM)));
        return Line::from(spans);
    };
    spans.push(Span::styled(
        crate::core::shown_title(state, path),
        Style::new().fg(TEXT),
    ));
    let status = if !state.config.bpm {
        " · tempo is off (M)".to_owned()
    } else {
        match state.rhythm.get(path) {
            None | Some(Rhythm::Pending) => " · analysing…".to_owned(),
            Some(Rhythm::Failed) => " · could not be analysed".to_owned(),
            Some(Rhythm::Ready(r)) => {
                let a = &r.analysis;
                let mut s = String::new();
                if let Some(sum) = a.summary {
                    s.push_str(&format!(" · {} BPM", text::bpm_text(sum.bpm)));
                }
                if let Some(key) = a.key {
                    s.push_str(&format!(" · {}", text::key_text(&key)));
                }
                let bars = a.downbeats.len();
                if let Some(c) = mix::cursor(state, which, heard) {
                    s.push_str(&format!(" · bar {} / {bars}", mix::bar_of(a, c)));
                }
                if r.waveform.as_ref().is_some_and(|w| w.stems.is_none()) {
                    s.push_str(" · no stems");
                }
                if r.fixed() {
                    s.push_str(" · grid fixed by hand");
                }
                if which == Block::Now {
                    s.push_str(&transition_text(state));
                }
                s
            }
        }
    };
    spans.push(Span::styled(status, Style::new().fg(DIM)));
    Line::from(spans)
}

/// The transition set up, in a few words.
fn transition_text(state: &State) -> String {
    use crate::core::transition::Status;
    let d = &state.mix.draft;
    // The preset of the one sent (a skip sends its own), else the one set.
    let preset = match (&d.status, &d.sent) {
        (Status::Armed | Status::Mixing(_), Some(sent)) => sent.preset,
        _ => d.preset,
    };
    let mut s = format!(
        " · transition: {}, {}",
        preset.name(),
        mix::tempo_name(d.tempo)
    );
    let bars = mix::draft_bars(state);
    if d.lanes.as_ref().is_some_and(|l| *l != d.preset.lanes(bars)) {
        s.push_str(" (edited)");
    }
    let cued = |b: Block| mix::block(state, b).cue.is_some();
    match &d.status {
        Status::Draft if cued(Block::Now) && cued(Block::Next) => s.push_str(", g arms"),
        Status::Draft => s.push_str(", enter sets the cues"),
        Status::Armed => s.push_str(", armed"),
        Status::Mixing(_) => s.push_str(", mixing"),
    }
    // The tempi of the beats that are paired, where they meet.
    if let (Some(a), Some(b)) = mix::meeting_tempi(state) {
        // The next track's tempo in the playing track's beats, by the
        // pairing (2:1: its beat is two of these), and the glide to it.
        let per = mix::pairing(state);
        let to = b * per.ratio();
        // Set by hand: what the transition set up (or armed) takes; one
        // already mixing plays what it was sent with.
        let set = d.pairing.is_some() && !matches!(d.status, Status::Mixing(_));
        s.push_str(" · next ");
        if per != Pairing::One || set {
            s.push_str(per.label());
            s.push_str(if set { " (set) " } else { " " });
        }
        s.push_str(&format!("{:+.1} %", 100.0 * (to / a - 1.0)));
        if (to / a - 1.0).abs() >= 0.002 {
            // The glide goes as far as the engine takes it; the next track
            // makes up the rest after the window.
            let most = jodel_playback::warp::GLIDE_MOST;
            let reach = a * (to / a).clamp(1.0 / most, most);
            s.push_str(&format!(" · tempo {a:.0} → {reach:.0} across the window"));
            if (reach - to).abs() >= 0.5 {
                s.push_str(&format!(", {to:.0} after it"));
            }
        }
    }
    s
}

/// What a block shows of the transition: its window on that track, each
/// lane's gain curve over it, and the edited bar.
struct Shown {
    window: (f64, f64),
    curves: Vec<(Lane, Vec<(f64, f32)>)>,
    edit: Option<(Lane, f64)>,
}

/// The transition as block `which` shows it: the one armed or mixing, else
/// the one set up at the two cues, else the fades of this track's side
/// over its own cue (one cue alone).
fn shown(state: &State, which: Block) -> Option<Shown> {
    use crate::core::transition::Status;
    use jodel_playback::stems::STEMS;
    use jodel_playback::transition::Side;
    let d = &state.mix.draft;
    // The playing track's seconds of the window, its bars, the side, this
    // track's time at the playing track's, and each stem's gains at a
    // share of the window.
    type Gains = Box<dyn Fn(f64) -> [f64; STEMS]>;
    type Here = Box<dyn Fn(f64) -> f64>;
    let (start, end, bars, side, here, gains): (f64, f64, usize, Side, Here, Gains) =
        match mix::shown_plan(state) {
            Some((plan, bars)) => {
                // The side by the track (after the hand-over the track
                // mixed in is the playing one, the one left slides away
                // above it).
                let index = mix::index(state, which)?;
                let side = if index == plan.from {
                    Side::From
                } else if index == plan.into {
                    Side::Into
                } else {
                    return None;
                };
                let map = plan.time_map();
                let here: Here = match side {
                    Side::From => Box::new(|t| t),
                    Side::Into => Box::new(move |t| map.at(t)),
                };
                let (start, end) = (plan.start, plan.start + plan.length);
                let gains: Gains = Box::new(move |share| plan.gains(side, share));
                (start, end, bars, side, here, gains)
            }
            None if d.status == Status::Draft => {
                let (start, end, _) = mix::cue_window(state, which)?;
                let side = match which {
                    Block::Now => Side::From,
                    Block::Next => Side::Into,
                    Block::Prev => return None,
                };
                let bars = mix::draft_bars(state);
                let lanes = d.lanes(bars);
                let base = if side == Side::From { 0 } else { STEMS };
                let gains: Gains =
                    Box::new(move |share| std::array::from_fn(|s| lanes[base + s].gain(share)));
                (start, end, bars, side, Box::new(|t| t), gains)
            }
            None => return None,
        };
    let window = (here(start), here(end));
    let steps = 96;
    let curves = Lane::ALL
        .iter()
        .map(|&lane| {
            let points = (0..=steps)
                .map(|k| {
                    let share = f64::from(k) / f64::from(steps);
                    let t = start + (end - start) * share;
                    let g = gains(share);
                    let gain = match lane.stem() {
                        Some(i) => g[i],
                        None => g.iter().sum::<f64>() / g.len() as f64,
                    };
                    (here(t), gain as f32)
                })
                .collect();
            (lane, points)
        })
        .collect();
    let edit = d.edit.and_then(|e| {
        let mine = match side {
            Side::From => e.lane < STEMS,
            Side::Into => e.lane >= STEMS,
        };
        let bars = bars.max(1);
        let lane = Lane::ALL
            .iter()
            .copied()
            .find(|l| l.stem() == Some(e.lane % STEMS))?;
        mine.then(|| {
            (
                lane,
                here(start + (end - start) * e.bar as f64 / bars as f64),
            )
        })
    });
    Some(Shown {
        window,
        curves,
        edit,
    })
}

/// The bar numbers over a block's lanes, over `start..end` seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct RulerPlot {
    /// Where it goes (one row).
    pub area: Rect,
    /// The seconds shown.
    pub start: f64,
    /// Up to.
    pub end: f64,
    /// Each bar's first beat in the span, and its number.
    pub bars: Vec<(f64, usize)>,
    /// The cursor's beat.
    pub cursor: Option<f64>,
    /// The next track's cue bar.
    pub cue: Option<f64>,
    /// It follows the playhead with the lanes under it
    /// ([`WavePlot::follow`]); its bars are then the whole track's.
    pub follow: Option<f64>,
}

impl RulerPlot {
    /// Where `time` falls, in units of `width` across the span.
    pub fn x_of(&self, time: f64, width: f64) -> f64 {
        (time - self.start) / (self.end - self.start).max(1e-9) * width
    }
}

/// The bar numbers in cells: at each bar's first beat where there is room
/// (else a tick), the cursor as `▼`, the cue as `◆`.
pub fn ruler_cells(buf: &mut Buffer, plot: &RulerPlot) {
    let area = plot.area.intersection(buf.area);
    if area.width == 0 || area.height == 0 {
        return;
    }
    let w = f64::from(area.width);
    let column = |t: f64| plot.x_of(t, w).floor();
    let marks: Vec<f64> = [plot.cue, plot.cursor]
        .into_iter()
        .flatten()
        .map(column)
        .collect();
    let mut free = 0.0;
    for &(t, bar) in &plot.bars {
        let mut x = column(t);
        if x < 0.0 || x >= w {
            continue;
        }
        if x < free {
            buf[(area.x + x as u16, area.y)].set_symbol("╵").set_fg(DIM);
            continue;
        }
        // A mark on the bar goes before its number.
        if marks.contains(&x) {
            x += 1.0;
            if x >= w {
                continue;
            }
        }
        let label = bar.to_string();
        buf.set_string(area.x + x as u16, area.y, &label, Style::new().fg(DIM));
        free = x + label.len() as f64 + 1.0;
    }
    for (time, symbol, color) in [(plot.cue, "◆", CUE), (plot.cursor, "▼", ACCENT)] {
        if let Some(x) = time.map(column).filter(|&x| x >= 0.0 && x < w) {
            buf[(area.x + x as u16, area.y)]
                .set_symbol(symbol)
                .set_fg(color);
        }
    }
}

/// One block: its header, overview, ruler and lanes.
fn draw_block(frame: &mut Frame, scene: &Scene, b: &MixBlockLayout, pictures: &mut dyn Pictures) {
    let state = scene.state;
    let heard = scene.heard;
    let focused = state.mix.focus == b.block;
    frame.render_widget(
        Paragraph::new(header(state, b.block, focused, heard)),
        b.header,
    );
    for (lane, label, _) in &b.lanes {
        frame.render_widget(
            Paragraph::new(Span::styled(
                lane.label(),
                Style::new().fg(if is_off(state, *lane) {
                    DIM
                } else {
                    lane_color(*lane)
                }),
            )),
            *label,
        );
    }
    let Some(r) = mix::rhythmed(state, b.block) else {
        return;
    };
    let Some(waveform) = r.waveform.clone() else {
        return;
    };
    let a = &r.analysis;
    let (start, end) = mix::window(state, b.block, heard);
    let beat_time = |i: Option<usize>| i.and_then(|i| a.beats.get(i).copied());
    let block = mix::block(state, b.block);
    let cursor = focused
        .then(|| beat_time(mix::cursor(state, b.block, heard)))
        .flatten();
    let cue = beat_time(block.cue.map(|c| c.beat));
    let selection = mix::selection(state, b.block, heard);
    let playhead = mix::playhead(state, b.block, heard);
    let duration = waveform.duration().max(a.duration).max(1e-3);
    let shown = shown(state, b.block);
    let transition = shown.as_ref().map(|s| s.window);
    pictures.wave(
        frame,
        &WavePlot {
            area: b.overview,
            waveform: Arc::clone(&waveform),
            lane: Lane::Mix,
            start: 0.0,
            end: duration,
            color: DIM,
            downbeats: Vec::new(),
            unclear: Vec::new(),
            selection,
            shown: (!mix::whole(state, b.block)).then_some((start, end)),
            playhead,
            cursor: None,
            cue,
            transition,
            gain: Vec::new(),
            edit: None,
            rows: (0, 1),
            lyrics: None,
            text_scale: 1,
            lines: mix::WordLines::ONE,
            follow: None,
            deep: false,
            key: None,
        },
    );
    let follows = b.block == Block::Now
        && mix::block(state, b.block).start.is_none()
        && mix::slide(state).is_none()
        && b.top_cut == 0;
    let bars = a
        .downbeats
        .iter()
        .enumerate()
        .filter_map(|(k, &d)| a.beats.get(d).map(|&t| (t, k + 1)))
        .filter(|&(t, _)| follows || (t >= start && t < end))
        .collect();
    pictures.ruler(
        frame,
        &RulerPlot {
            area: b.ruler,
            start,
            end,
            bars,
            cursor,
            cue,
            follow: follows.then_some(mix::LEAD),
        },
    );
    let downbeats = downbeats_within(a, start, end);
    let unclear = unclear(a);
    // The playing track's block following the playhead (not while blocks
    // slide): a frontend may scroll its ruler and lanes by itself, so they
    // carry the whole track's bars.
    let follow = (b.block == Block::Now
        && mix::block(state, b.block).start.is_none()
        && mix::slide(state).is_none()
        && b.top_cut == 0)
        .then_some(mix::LEAD);
    let downbeats = if follow.is_some() {
        downbeats_within(a, f64::NEG_INFINITY, f64::INFINITY)
    } else {
        downbeats
    };
    let lyrics = state
        .config
        .mix_lyrics
        .then(|| mix::lyrics_of_block(state, b.block))
        .flatten();
    for (i, (lane, label, area)) in b.lanes.iter().enumerate() {
        let skip = if i == 0 { b.top_cut } else { 0 };
        let color = if is_off(state, *lane) {
            OFF
        } else {
            lane_color(*lane)
        };
        let gain = shown
            .as_ref()
            .and_then(|s| s.curves.iter().find(|c| c.0 == *lane))
            .map(|c| c.1.clone())
            .unwrap_or_default();
        let edit = shown
            .as_ref()
            .and_then(|s| s.edit)
            .filter(|e| e.0 == *lane)
            .map(|e| e.1);
        if edit.is_some() {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    lane.label(),
                    Style::new().fg(BG_TEXT).bg(lane_color(*lane)),
                )),
                *label,
            );
        }
        pictures.wave(
            frame,
            &WavePlot {
                area: *area,
                waveform: Arc::clone(&waveform),
                lane: *lane,
                start,
                end,
                color,
                downbeats: downbeats.clone(),
                unclear: unclear.clone(),
                selection,
                shown: None,
                playhead,
                cursor,
                cue,
                transition,
                gain,
                edit,
                rows: (skip, b.lane_rows.max(area.height)),
                lyrics: if *lane == Lane::Vocals {
                    lyrics.clone()
                } else {
                    None
                },
                text_scale: 1,
                lines: mix::WordLines::ONE,
                follow,
                deep: false,
                key: None,
            },
        );
    }
}

/// The Mix tab.
pub fn draw(frame: &mut Frame, scene: &Scene, m: &MixLayout, pictures: &mut dyn Pictures) {
    for b in &m.blocks {
        draw_block(frame, scene, b, pictures);
    }
}

/// The Mix tab's footer: the focused track's zoom and the main keys.
pub fn footer(state: &State) -> String {
    // The focused track's zoom (its own, set by the wheel or `z`, else the
    // tab's).
    let zoom = match mix::ZOOMS[mix::zoom_of(state, state.mix.focus)] {
        0 => "whole tracks".to_owned(),
        1 => "1 bar".to_owned(),
        bars => format!("{bars} bars"),
    };
    if let Some(e) = state.mix.draft.edit {
        return format!(
            "editing {} at bar {} · ↑↓ lane · ←→ bar · +/- 0 1 gain · c cut · esc done",
            crate::core::transition::lane_name(e.lane),
            e.bar
        );
    }
    format!(
        "{zoom} · ←→ bars · 1-9 jump · enter cue · p preset · r pairing · g go · e edit · ? keys"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_split_into_their_syllables() {
        let parts = |w: &str| syllable_parts(w);
        assert_eq!(parts("Mister"), ["Mis", "ter"]);
        assert_eq!(parts("baby"), ["ba", "by"]);
        assert_eq!(parts("go"), ["go"]);
        assert_eq!(parts("time"), ["time"], "a silent e");
        assert_eq!(parts("table"), ["ta", "ble"]);
        assert_eq!(parts("beautiful,"), ["beau", "ti", "ful,"]);
        assert_eq!(parts("愛してる"), ["愛", "し", "て", "る"]);
        assert_eq!(parts("..."), ["..."]);
    }

    #[test]
    fn the_overview_shades_the_track_outside_the_span_shown() {
        // 100 s over 100 cells, the lanes showing 40 to 60 s.
        let frames = (100.0 * waveform::FPS) as usize;
        let lane = waveform::Lane {
            peak: vec![200; frames],
            rms: vec![150; frames],
        };
        let plot = WavePlot {
            area: Rect::new(0, 0, 100, 1),
            waveform: Arc::new(Waveform {
                mix: lane,
                stems: None,
                melody: None,
            }),
            lane: Lane::Mix,
            start: 0.0,
            end: 100.0,
            color: DIM,
            downbeats: Vec::new(),
            unclear: Vec::new(),
            selection: None,
            shown: Some((40.0, 60.0)),
            playhead: None,
            cursor: None,
            cue: None,
            transition: None,
            gain: Vec::new(),
            edit: None,
            rows: (0, 1),
            lyrics: None,
            text_scale: 1,
            lines: mix::WordLines::ONE,
            follow: None,
            deep: false,
            key: None,
        };
        let mut buf = Buffer::empty(plot.area);
        wave_cells(&mut buf, &plot);
        let at = |x: u16| &buf[(x, 0)];
        assert_eq!(at(50).bg, SHOWN_BG);
        assert_eq!(at(50).fg, DIM);
        for x in [5, 30, 70, 95] {
            assert_eq!(at(x).bg, OUTSIDE_BG, "cell {x}");
            assert_eq!(at(x).fg, faded(DIM, OUTSIDE_GAIN), "cell {x}");
        }
        // The whole track shown: nothing shaded.
        let whole = WavePlot {
            shown: None,
            ..plot
        };
        let mut buf = Buffer::empty(whole.area);
        wave_cells(&mut buf, &whole);
        assert_ne!(buf[(5, 0)].bg, OUTSIDE_BG);
    }

    #[test]
    fn words_not_heard_do_not_set_the_zoom() {
        // A line sung at an easy pace, then an echo the aligner did not
        // hear, squeezed between its neighbours (Suddenly I See, bar 58).
        let heard = |s: f64, e: f64, t: &str| jodel_lyrics::Word {
            confidence: 0.9,
            ..word(s, e, t, &[])
        };
        let words = vec![
            heard(0.0, 0.5, "She"),
            heard(0.6, 1.1, "got"),
            heard(1.2, 1.7, "the"),
            heard(1.8, 2.6, "power"),
            word(2.6, 2.7, "(Suddenly", &[]),
            word(2.7, 2.75, "I", &[]),
            word(2.75, 2.8, "see)", &[]),
            heard(2.8, 3.3, "She"),
        ];
        let lyrics = jodel_lyrics::Lyrics {
            lines: vec![jodel_lyrics::Line {
                words,
                ..jodel_lyrics::Line::default()
            }],
            source: jodel_lyrics::Source::Unknown,
            timing: jodel_lyrics::Timing::default(),
        };
        let all = mix::lyrics_per_second(&mix::timed_words(&lyrics), 1).unwrap();
        let fitted = mix::lyrics_per_second(&mix::fitted_words(&lyrics), 1).unwrap();
        assert!(
            all > 2.0 * fitted,
            "the echo zooms in: {all} against {fitted}"
        );
        assert_eq!(mix::fitted_words(&lyrics).len(), 5);
    }

    #[test]
    fn a_word_is_spread_over_the_cells_it_is_sung() {
        assert_eq!(word_label("Mister", 6), "Mister");
        assert_eq!(word_label("Mister", 9), "Mis───ter");
        assert_eq!(word_label("Mister", 4), "Mist", "cut where it has no room");
        assert_eq!(word_label("go", 5), "go───", "held");
        assert_eq!(word_label("beautiful", 13), "beau──ti──ful");
    }

    fn word(start: f64, end: f64, text: &str, letters: &[f32]) -> jodel_lyrics::Word {
        jodel_lyrics::Word {
            start: Some(start),
            end: Some(end),
            letters: letters.to_vec(),
            ..jodel_lyrics::Word::new(text)
        }
    }

    /// Labels over `start..end` across `cells` cells.
    fn labels(
        words: &[jodel_lyrics::Word],
        start: f64,
        end: f64,
        cells: f64,
        heard: Option<f64>,
    ) -> Vec<Label> {
        let all: Vec<&jodel_lyrics::Word> = words.iter().collect();
        let per = cells / (end - start);
        lane_labels(
            &all,
            &|t| (t - start) * per,
            per,
            1.0,
            mix::WordLines::ONE,
            heard,
            (0.0, cells),
        )
    }

    /// On two lines a word crowding the one before goes under it, where it
    /// is sung, and the next goes back on the first line; a word with no
    /// room on either goes after the one that frees first, kept (on the
    /// Karaoke tab) or left out (one line, the Mix lanes).
    #[test]
    fn crowded_words_go_on_the_line_under() {
        // A tenth of a second a cell.
        let words = [
            word(1.0, 1.2, "Mister", &[]),
            word(1.3, 1.4, "rock", &[]),
            word(1.8, 1.9, "and", &[]),
            word(1.85, 1.9, "roll", &[]),
            word(1.9, 1.95, "baby", &[]),
        ];
        let all: Vec<&jodel_lyrics::Word> = words.iter().collect();
        let x_of = |t: f64| (t * 10.0).floor();
        let two = lane_labels(
            &all,
            &x_of,
            10.0,
            1.0,
            mix::WordLines {
                rows: 2,
                keep_all: true,
            },
            None,
            (0.0, 100.0),
        );
        let placed: Vec<(f64, usize)> = two.iter().map(|l| (l.at, l.row)).collect();
        // "Mister" 10..16, "rock" at 13 under it, "and" at 18 back on top,
        // "roll" at 18 under it (free from 18), "baby" on neither: after
        // "and" (18..21, a cell between), late.
        assert_eq!(
            placed,
            vec![(10.0, 0), (13.0, 1), (18.0, 0), (18.0, 1), (22.0, 0)]
        );
        let one = lane_labels(
            &all,
            &x_of,
            10.0,
            1.0,
            mix::WordLines::ONE,
            None,
            (0.0, 100.0),
        );
        assert!(one.len() < words.len(), "one line leaves some out");
        assert_eq!(mix::words_left_out(&all, 10.0, 1), words.len() - one.len());
    }
    #[test]
    fn the_lane_labels_stand_where_their_words_are_sung() {
        // Ten seconds over 100 cells: a tenth of a second a cell.
        let words = [
            word(1.0, 2.0, "Mister", &[]),
            word(2.2, 2.4, "Rock", &[]),
            word(2.5, 2.6, "and", &[]),
        ];
        let l = labels(&words, 0.0, 10.0, 100.0, Some(1.5));
        assert_eq!(
            l[0],
            Label {
                at: 10.0,
                text: "Mis────ter".into(),
                heard: true,
                row: 0,
            }
        );
        // "Rock" from cell 22; "and" would run into it, so it goes after
        // it, a cell between.
        assert_eq!(
            (l[1].at, l[1].text.as_str(), l[1].heard),
            (22.0, "Rock", false)
        );
        assert_eq!(l[2].at, 27.0);
        // Out of the span: none.
        assert!(labels(&words, 5.0, 10.0, 50.0, None).is_empty());
        // In pixels, 8 to a character: the same places, eight times over.
        let shown: Vec<&jodel_lyrics::Word> = words.iter().collect();
        let px = lane_labels(
            &shown,
            &|t| t * 80.0,
            80.0,
            8.0,
            mix::WordLines::ONE,
            None,
            (0.0, 1e9),
        );
        assert_eq!((px[1].at, px[2].at), (176.0, 216.0));
    }

    /// Words crowded so that each pushes the next along stand where they
    /// stand whatever part of the song is in view: scrolling never moves
    /// them against one another.
    #[test]
    fn crowded_words_keep_their_places_as_the_view_scrolls() {
        // A word every 0.2 s, each six letters long, at a cell per 0.1 s:
        // each pushed further by the one before.
        let words: Vec<jodel_lyrics::Word> = (0..40)
            .map(|k| word(f64::from(k) * 0.2, f64::from(k) * 0.2 + 0.15, "wordss", &[]))
            .collect();
        let all: Vec<&jodel_lyrics::Word> = words.iter().collect();
        let x_of = |t: f64| (t * 10.0).floor();
        let place = |lo: f64| -> Vec<(f64, String)> {
            lane_labels(
                &all,
                &x_of,
                10.0,
                1.0,
                mix::WordLines::ONE,
                None,
                (lo, lo + 30.0),
            )
            .into_iter()
            .map(|l| (l.at, l.text))
            .collect()
        };
        for lo in [10.0, 11.0, 17.0, 23.0] {
            let later = place(lo + 3.0);
            for label in place(lo) {
                if label.0 >= lo + 3.0 {
                    assert!(later.contains(&label), "{label:?} moved at {lo}");
                }
            }
        }
    }

    /// With the letters' times, each letter stands where it is sung: a
    /// held vowel stretches with `-` after it, the last held to the end.
    #[test]
    fn heard_letters_stand_where_they_are_sung() {
        // "Mister" over a second at 10 cells a second: M, i held to 0.4,
        // s t e quickly, e held, r at 0.8.
        let letters = [0.0, 0.05, 0.42, 0.45, 0.5, 0.8];
        assert_eq!(timed_label("Mister", &letters, 10.0, 10), "Mi──ste─r─");
        // Too close for their cells: one after another all the same.
        assert_eq!(timed_label("Mister", &[0.0; 6], 10.0, 8), "Mister──");
        // No room for the late ones: kept to the cells there are.
        assert_eq!(timed_label("go", &[0.0, 5.0], 10.0, 4), "g──o");
        // Without times: by syllables.
        assert_eq!(timed_label("Mister", &[], 10.0, 9), "Mis───ter");
        let l = labels(
            &[word(1.0, 2.0, "Mister", &letters)],
            0.0,
            10.0,
            100.0,
            None,
        );
        assert_eq!(l[0].text, "Mi──ste─r─");
    }
}
