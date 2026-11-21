//! The lanes a track is drawn in, shared by the Transition and Karaoke
//! tabs and the window: the colours, a lane's waveform ([`WavePlot`]), and
//! the lyrics' words along the voice's lane ([`lane_labels`]).
//!
//! The waveforms are drawn by the [`super::Pictures`]: in cells as braille
//! (two dot columns and four dot rows a cell, mirrored about the middle,
//! here), in pixels by the window. Bars start where a lane's cell is shaded
//! (a line in pixels), the selection is tinted, the playhead is the accent
//! colour, and stretches where the beat is unclear are dimmer. The voice's
//! lane carries the lyrics, once timed, along its bottom row: each word
//! where it is sung, each letter in the cell where the aligner heard it, a
//! line where a sound is held (`Mi───ster`, `go────`; a word without its
//! letters' times spread by syllables), the word heard now in the accent
//! colour ([`lane_labels`]), on the lane's middle row.

use std::sync::Arc;

use jodel_rhythm::waveform::{self, Waveform};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

use super::colors::{ACCENT, TEXT};
use super::plot::{braille, braille_bit};
use crate::core::lanes::{self, WordLines};
use crate::core::layout::Lane;

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

/// A lane's colour.
pub fn lane_color(lane: Lane) -> Color {
    match lane {
        // The stems in the Okabe–Ito hues (orange, sky blue, bluish green,
        // reddish purple): told apart with every common colour-vision
        // deficiency; the whole song a neutral grey.
        Lane::Song => Color::Rgb(169, 171, 181),
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
    /// The lines the words go on ([`WordLines`]).
    pub lines: WordLines,
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
    /// Its line (0: the first; [`WordLines`]).
    pub row: usize,
}

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
/// [`lanes::lay_out_words`] places them: each a character `cell` units wide
/// at `per` units a second, as wide as it is sung ([`timed_label`], else
/// [`word_label`]), and whether it is heard at `playhead`.
#[allow(clippy::too_many_arguments)]
pub fn lane_labels(
    words: &[&jodel_lyrics::Word],
    x_of: &dyn Fn(f64) -> f64,
    per: f64,
    cell: f64,
    lines: WordLines,
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
    lanes::lay_out_words(words, x_of, per, cell, lines, &mut |k, at, wide, row| {
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
        let words = lanes::timed_words(lyrics);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::colors::DIM;

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
            lane: Lane::Song,
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
            lines: WordLines::ONE,
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
        let all = lanes::lyrics_per_second(&lanes::timed_words(&lyrics), 1).unwrap();
        let fitted = lanes::lyrics_per_second(&lanes::fitted_words(&lyrics), 1).unwrap();
        assert!(
            all > 2.0 * fitted,
            "the echo zooms in: {all} against {fitted}"
        );
        assert_eq!(lanes::fitted_words(&lyrics).len(), 5);
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
            WordLines::ONE,
            heard,
            (0.0, cells),
        )
    }

    /// On two lines a word crowding the one before goes under it, where it
    /// is sung, and the next goes back on the first line; a word with no
    /// room on either goes after the one that frees first, kept (on the
    /// Karaoke tab) or left out (one line, the Transition lanes).
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
            WordLines {
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
        let one = lane_labels(&all, &x_of, 10.0, 1.0, WordLines::ONE, None, (0.0, 100.0));
        assert!(one.len() < words.len(), "one line leaves some out");
        assert_eq!(
            lanes::words_left_out(&all, 10.0, 1),
            words.len() - one.len()
        );
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
            WordLines::ONE,
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
                WordLines::ONE,
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
