//! Text both frontends draw and the mouse needs to locate: rates, times,
//! the signal path, the status light and the list toggle.

use jodel_playback::{PlayState, Stage, Stem};
use unicode_width::UnicodeWidthStr;

use crate::core::keymap::Op;
use crate::core::{Rhythm, Side, State};

/// A rate in kHz without the unit: `44.1`, `48`, `352.8`.
pub fn khz_short(rate: u32) -> String {
    let khz = f64::from(rate) / 1000.0;
    if (khz - khz.round()).abs() < 0.05 {
        format!("{khz:.0}")
    } else {
        format!("{khz:.1}")
    }
}

/// Milliseconds with their sign: `+30 ms`, `−20 ms`, `0 ms`.
pub fn signed_ms(ms: i32) -> String {
    match ms.signum() {
        1 => format!("+{ms} ms"),
        -1 => format!("−{} ms", ms.unsigned_abs()),
        _ => "0 ms".to_owned(),
    }
}

/// A time as `m:ss`, or `h:mm:ss` from an hour on.
pub fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).floor() as u64;
    let (h, m, s) = (total / 3600, total / 60 % 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// How a piece of text is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// Ordinary text.
    Plain,
    /// A title (bold).
    Title,
    /// The artist.
    Accent,
    /// Secondary text.
    Dim,
    /// The file's rate.
    Source,
    /// The output device (clickable).
    Device,
    /// The playing track's title (a link to it in the browser).
    TitleLink,
    /// Its artist (a link to the artist folder).
    ArtistLink,
    /// Its album (a link to the album folder).
    AlbumLink,
    /// Good news (bit-perfect).
    Good,
    /// A warning (not bit-perfect, not exclusive).
    Warn,
    /// The beat pulse: drawn brighter for a moment on each beat.
    Pulse,
}

/// A piece of coloured text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    /// The text.
    pub text: String,
    /// Its colour.
    pub tone: Tone,
}

fn seg(text: impl Into<String>, tone: Tone) -> Seg {
    Seg {
        text: text.into(),
        tone,
    }
}

/// The signal path of the track being heard, for example
/// `file 96 → DAC (exclusive), bit-perfect`, or with a resampler
/// `file 44.1 → resample 48 (160/147) → DAC (exclusive), not bit-perfect`.
/// A lossy file is never called bit-perfect; its path ends with what it
/// was decoded from: `file 44.1 → DAC (exclusive), decoded from MP3`.
pub fn signal_path(state: &State) -> Option<Vec<Seg>> {
    let track = state.playing.track.as_ref()?;
    let route = &track.route;
    let arrow = || seg(" → ", Tone::Dim);
    let input = khz_short(route.input_rate);
    let output = khz_short(route.decoded_rate());
    let mut out = vec![seg(format!("file {input}"), Tone::Source)];
    match route.stage {
        Stage::Passthrough => {}
        Stage::Stems => {
            out.push(arrow());
            out.push(seg(
                format!("stems {output} ({})", state.stems_in_force().describe()),
                Tone::Warn,
            ));
        }
    }
    if let Some(step) = route.resample {
        out.push(arrow());
        out.push(seg(
            format!("resample {} ({})", khz_short(step.to), step.detail()),
            Tone::Warn,
        ));
    }
    out.push(arrow());
    out.push(seg(track.device.clone(), Tone::Device));
    if track.output.exclusive {
        match &state.device_volume {
            Some(volume) => out.push(seg(
                format!(" (exclusive, volume {})", volume.label()),
                Tone::Dim,
            )),
            None => out.push(seg(" (exclusive)", Tone::Dim)),
        }
    } else {
        out.push(seg(" (not exclusive)", Tone::Warn));
    }
    out.push(seg(", ", Tone::Dim));
    if let Some(codec) = route.decoded_from {
        out.push(seg(format!("decoded from {codec}"), Tone::Dim));
    } else if track.bit_perfect && !state.playing.mixed {
        out.push(seg("bit-perfect", Tone::Good));
    } else {
        out.push(seg("not bit-perfect", Tone::Warn));
    }
    if state.playing.clipped > 0 {
        out.push(seg(
            format!(", {} samples clipped", state.playing.clipped),
            Tone::Warn,
        ));
    }
    Some(out)
}

/// The width of a text in cells.
pub fn width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Wraps coloured text into lines of at most `max` cells, breaking at
/// spaces where it can.
pub fn wrap(segs: &[Seg], max: usize) -> Vec<Vec<Seg>> {
    let max = max.max(1);
    // Words keep their trailing spaces and their tone.
    let mut words: Vec<Seg> = Vec::new();
    for s in segs {
        let mut start = 0;
        let bytes: Vec<(usize, char)> = s.text.char_indices().collect();
        for (i, &(at, c)) in bytes.iter().enumerate() {
            let next_is_word = bytes.get(i + 1).is_some_and(|&(_, n)| n != ' ');
            if c == ' ' && next_is_word {
                let end = at + 1;
                words.push(seg(&s.text[start..end], s.tone));
                start = end;
            }
        }
        if start < s.text.len() {
            words.push(seg(&s.text[start..], s.tone));
        }
    }
    let mut lines: Vec<Vec<Seg>> = vec![Vec::new()];
    let mut used = 0;
    for word in words {
        let trimmed = width(word.text.trim_end());
        if used > 0 && used + trimmed > max {
            lines.push(Vec::new());
            used = 0;
        }
        if used == 0 && trimmed > max {
            // Hard-split a word longer than a line.
            let mut piece = String::new();
            for c in word.text.chars() {
                if width(&piece) + width(c.encode_utf8(&mut [0; 4])) > max {
                    lines
                        .last_mut()
                        .unwrap()
                        .push(seg(piece.clone(), word.tone));
                    lines.push(Vec::new());
                    piece.clear();
                }
                piece.push(c);
            }
            used = width(&piece);
            lines.last_mut().unwrap().push(seg(piece, word.tone));
            continue;
        }
        let text = if used == 0 {
            word.text.trim_start().to_owned()
        } else {
            word.text.clone()
        };
        used += width(&text);
        lines.last_mut().unwrap().push(seg(text, word.tone));
    }
    for line in &mut lines {
        if let Some(last) = line.last_mut() {
            last.text = last.text.trim_end().to_owned();
        }
    }
    lines
}

/// Where the cells of one tone lie in wrapped lines: (line, first column,
/// width) for each piece.
pub fn locate(lines: &[Vec<Seg>], tone: Tone) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (row, line) in lines.iter().enumerate() {
        let mut x = 0;
        for s in line {
            let w = width(&s.text);
            if s.tone == tone {
                let trimmed = width(s.text.trim_end());
                out.push((row, x, trimmed));
            }
            x += w;
        }
    }
    out
}

/// The status light: symbol, label, tone.
pub fn status_light(state: &State) -> (&'static str, String, Tone) {
    let playing = &state.playing;
    let Some(track) = &playing.track else {
        return ("○", "nothing playing".into(), Tone::Dim);
    };
    if track.info.is_lossy() {
        return ("○", lossy_label(&track.info), Tone::Dim);
    }
    if track.route.stage == Stage::Stems {
        return ("○", "stems".into(), Tone::Dim);
    }
    ("○", "PCM".into(), Tone::Dim)
}

/// What the status light says for a lossy file: the codec, the bit rate
/// when known, and that it is lossy (`MP3 320 kbps · lossy`).
pub fn lossy_label(info: &jodel_playback::TrackInfo) -> String {
    match info.bitrate {
        Some(bitrate) => format!(
            "{} {} kbps · lossy",
            info.format_name(),
            (bitrate + 500) / 1000
        ),
        None => format!("{} · lossy", info.format_name()),
    }
}

/// A stem button by the transport buttons: ` drums `, or ` D ` on a
/// narrow screen.
pub fn stem_button(stem: Stem, wide: bool) -> String {
    let name = stem.id();
    if wide {
        format!(" {name} ")
    } else {
        format!(" {} ", name[..1].to_uppercase())
    }
}

/// The play state symbol: `▶`, `▮▮` or `■`.
pub fn play_symbol(state: PlayState) -> &'static str {
    match state {
        PlayState::Playing => "▶",
        PlayState::Paused => "▮▮",
        PlayState::Stopped => "■",
    }
}

/// The list toggle in the list pane's top border: the pieces, which side
/// each selects, and whether it is the shown side.
pub fn toggle(state: &State) -> Vec<(String, Option<Side>, bool)> {
    let queue = format!(" Queue ({}) ", state.queue.paths.len());
    vec![
        (
            " Browser ".into(),
            Some(Side::Browser),
            state.side == Side::Browser,
        ),
        ("│".into(), None, false),
        (queue, Some(Side::Queue), state.side == Side::Queue),
        ("│".into(), None, false),
        (
            " Transition ".into(),
            Some(Side::Transition),
            state.side == Side::Transition,
        ),
        ("│".into(), None, false),
        (
            " Karaoke ".into(),
            Some(Side::Karaoke),
            state.side == Side::Karaoke,
        ),
        ("│".into(), None, false),
        (" Mix ".into(), Some(Side::Mix), state.side == Side::Mix),
    ]
}

/// The lines of the track panel (the right column), at most `max` cells
/// wide: the title, artist and album each on one line (shortened with
/// `…` when too long), the status light and the signal path (wrapped
/// between its steps, [`wrap_path`]). The view draws exactly these lines,
/// top-aligned, and the mouse finds the device name and the title, artist
/// and album links ([`LINKS`]) in them.
pub fn info_lines(state: &State, max: usize) -> Vec<Vec<Seg>> {
    let mut lines: Vec<Vec<Seg>> = Vec::new();
    let one = |text: &str, tone: Tone| vec![seg(shorten(text, max), tone)];
    let Some(track) = &state.playing.track else {
        lines.push(one("Nothing playing", Tone::Title));
        lines.extend(wrap(
            &[seg("enter plays the selected file; A queues it", Tone::Dim)],
            max,
        ));
        return lines;
    };
    let tags = &track.info.tags;
    lines.push(one(&track_title(&track.info), Tone::TitleLink));
    if let Some(artist) = &tags.artist {
        lines.push(one(artist, Tone::ArtistLink));
    }
    if let Some(album) = &tags.album {
        lines.push(one(album, Tone::AlbumLink));
    }
    lines.push(Vec::new());
    let (light, label, tone) = status_light(state);
    lines.extend(wrap(&[seg(format!("{light} {label}"), tone)], max));
    if let Some(bpm) = bpm_line(state) {
        lines.extend(wrap(&bpm, max));
    }
    if let Some(path) = signal_path(state) {
        lines.extend(wrap_path(&path, max));
    }
    lines
}

/// The link tones and where each goes.
pub const LINKS: [(Tone, Op); 3] = [
    (Tone::TitleLink, Op::GoTrack),
    (Tone::ArtistLink, Op::GoArtist),
    (Tone::AlbumLink, Op::GoAlbum),
];

/// The cells the track panel's lines take unwrapped: the width the right
/// column would need to show each on one line.
pub fn info_width(state: &State) -> usize {
    if state.playing.track.is_none() {
        return 0;
    }
    info_lines(state, usize::MAX / 2)
        .iter()
        .map(|line| line.iter().map(|s| width(&s.text)).sum::<usize>())
        .max()
        .unwrap_or(0)
}

/// Shortens `text` to `max` cells with a `…` at the end.
pub fn shorten(text: &str, max: usize) -> String {
    if width(text) <= max {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut used = 1;
    for c in text.chars() {
        let w = width(c.encode_utf8(&mut [0; 4]));
        if used + w > max {
            break;
        }
        used += w;
        out.push(c);
    }
    out.trim_end().to_owned() + "…"
}

/// Wraps a signal path ([`signal_path`]) to `max` cells between its
/// steps: a new line starts with a step's `→`, and the device's
/// `(exclusive), bit-perfect` stays one piece (with the device's name when
/// both fit on a line), so no phrase is split and no word is left alone.
/// A piece longer than a line is wrapped at its spaces.
pub fn wrap_path(path: &[Seg], max: usize) -> Vec<Vec<Seg>> {
    // The pieces: each starts at an arrow, or is the device's tail.
    let mut pieces: Vec<Vec<Seg>> = Vec::new();
    let mut tail = false;
    for s in path {
        let starts = s.text == " → " || (s.text.starts_with(" (") && !tail);
        if starts || pieces.is_empty() {
            tail = s.text.starts_with(" (");
            pieces.push(Vec::new());
        }
        pieces.last_mut().unwrap().push(s.clone());
    }
    let piece_width = |p: &[Seg]| p.iter().map(|s| width(&s.text)).sum::<usize>();
    // The device and its tail stay together when they fit on a line.
    let tail = pieces
        .iter()
        .position(|p| p.first().is_some_and(|s| s.text.starts_with(" (")));
    if let Some(t) = tail.filter(|&t| t > 0) {
        if piece_width(&pieces[t - 1]) + piece_width(&pieces[t]) <= max + 1 {
            let tail = pieces.remove(t);
            pieces[t - 1].extend(tail);
        }
    }
    let mut lines: Vec<Vec<Seg>> = vec![Vec::new()];
    let mut used = 0;
    for mut piece in pieces {
        if used > 0 && used + piece_width(&piece) > max {
            lines.push(Vec::new());
            used = 0;
        }
        if used == 0 {
            if let Some(first) = piece.first_mut() {
                first.text = first.text.trim_start().to_owned();
            }
            piece.retain(|s| !s.text.is_empty());
        }
        let w = piece_width(&piece);
        if used == 0 && w > max {
            let mut wrapped = wrap(&piece, max);
            if lines.last().is_some_and(Vec::is_empty) {
                lines.pop();
            }
            used = wrapped
                .last()
                .map_or(0, |l| l.iter().map(|s| width(&s.text)).sum());
            lines.append(&mut wrapped);
            continue;
        }
        used += w;
        lines.last_mut().unwrap().extend(piece);
    }
    lines.retain(|l| !l.is_empty());
    lines
}

/// The Now playing caption, wrapped to `max` cells: the title, artist ·
/// album and the status light. The title, artist and album are links, as
/// in [`info_lines`].
pub fn caption_lines(state: &State, max: usize) -> Vec<Vec<Seg>> {
    let Some(track) = &state.playing.track else {
        return wrap(&[seg("Nothing playing", Tone::Title)], max);
    };
    let tags = &track.info.tags;
    let mut lines = wrap(&[seg(track_title(&track.info), Tone::TitleLink)], max);
    let mut by: Vec<Seg> = Vec::new();
    if let Some(artist) = &tags.artist {
        by.push(seg(artist.clone(), Tone::ArtistLink));
    }
    if let Some(album) = &tags.album {
        if !by.is_empty() {
            by.push(seg(" · ", Tone::Dim));
        }
        by.push(seg(album.clone(), Tone::AlbumLink));
    }
    if !by.is_empty() {
        lines.extend(wrap(&by, max));
    }
    let (light, label, tone) = status_light(state);
    lines.extend(wrap(&[seg(format!("{light} {label}"), tone)], max));
    if let Some(bpm) = bpm_line(state) {
        lines.extend(wrap(&bpm, max));
    }
    lines
}

/// A tempo for people: whole when it is within 0.05 of a whole number,
/// else with one decimal.
pub fn bpm_text(bpm: f64) -> String {
    if (bpm - bpm.round()).abs() < 0.05 {
        format!("{bpm:.0}")
    } else {
        format!("{bpm:.1}")
    }
}

/// The playing track's tempo, and its key once found (`· E major (12B)`):
/// `♩ 128 BPM` when steady; `♩ 98 BPM · 92–104`
/// when it varies (the tempo here, then the range), or `♩ 92–104 BPM`
/// where the beat is unclear; `♩ … BPM` while it is analysed; `♩ – BPM`
/// when the beat is clear nowhere or the file could not be analysed.
/// `None` with the tempo off or nothing playing. The `♩` is the beat
/// pulse ([`Tone::Pulse`]).
pub fn bpm_line(state: &State) -> Option<Vec<Seg>> {
    if !state.config.bpm {
        return None;
    }
    let track = state.playing.track.as_ref()?;
    let path = state.queue.paths.get(track.index)?;
    let text = match state.rhythm.get(path) {
        None | Some(Rhythm::Pending) => "… BPM".to_owned(),
        Some(Rhythm::Failed) => "– BPM".to_owned(),
        Some(Rhythm::Ready(r)) => match r.analysis.summary {
            None => "– BPM".to_owned(),
            Some(s) if s.steady => format!("{} BPM", bpm_text(s.bpm)),
            Some(s) => {
                let range = format!("{:.0}–{:.0}", s.low, s.high);
                let at = state.playing.position;
                match r.analysis.bpm_at(at).filter(|_| r.analysis.is_clear_at(at)) {
                    Some(now) => format!("{:.0} BPM · {range}", now),
                    None => format!("{range} BPM"),
                }
            }
        },
    };
    let key = match state.rhythm.get(path) {
        Some(Rhythm::Ready(r)) => r.analysis.key.map(|k| format!(" · {}", key_text(&k))),
        _ => None,
    };
    Some(vec![
        seg("♩", Tone::Pulse),
        seg(format!(" {text}{}", key.unwrap_or_default()), Tone::Dim),
    ])
}

/// A key as shown: `E major (12B)`, its place on the Camelot wheel after.
pub fn key_text(key: &jodel_rhythm::key::Key) -> String {
    format!("{} ({})", key.name(), key.camelot())
}

/// The name a track is shown by: its title tag, else its file name.
pub fn track_title(info: &jodel_playback::TrackInfo) -> String {
    info.tags
        .title
        .clone()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| {
            info.path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
}

/// A spinner's glyphs, in turn.
pub const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A spinner's glyph for something still loading, turning ten times a
/// second by the clock last drawn at.
pub fn spinner(state: &State) -> char {
    let ms = state.drawn_ms.unwrap_or(state.now_ms);
    SPINNER[(ms / 100 % 10) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_clocks() {
        assert_eq!(khz_short(44_100), "44.1");
        assert_eq!(khz_short(48_000), "48");
        assert_eq!(khz_short(352_800), "352.8");
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(61.9), "1:01");
        assert_eq!(clock(3_725.0), "1:02:05");
    }

    #[test]
    fn wrapping_keeps_tones_and_widths() {
        let segs = vec![
            seg("file 44.1", Tone::Source),
            seg(" → ", Tone::Dim),
            seg("Speakers (USB DAC)", Tone::Device),
            seg(", bit-perfect", Tone::Good),
        ];
        let lines = wrap(&segs, 16);
        for line in &lines {
            let w: usize = line.iter().map(|s| width(&s.text)).sum();
            assert!(w <= 16, "{line:?}");
        }
        let joined: String = lines
            .iter()
            .flatten()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(joined.contains("Speakers"));
        let device = locate(&lines, Tone::Device);
        assert!(!device.is_empty());
        // One wide line holds everything.
        assert_eq!(wrap(&segs, 200).len(), 1);
        // A word longer than a line is split.
        let long = wrap(&[seg("abcdefghij", Tone::Plain)], 4);
        assert_eq!(long.len(), 3);
    }
}
