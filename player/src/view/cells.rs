//! The text helpers of the cell renderer: widths in cells, wrapping and
//! shortening to a number of cells, and where a tone lies in wrapped
//! lines (for the mouse), and the right column's lines wrapped to its
//! cells.

use unicode_width::UnicodeWidthStr;

use crate::core::text::{bpm_line, seg, signal_path, status_light, track_title, Seg, Tone};
use crate::core::State;

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

/// The lines of the track panel (the right column), at most `max` cells
/// wide: the title, artist and album each on one line (shortened with
/// `…` when too long), the status light and the signal path (wrapped
/// between its steps, [`wrap_path`]). The view draws exactly these lines,
/// top-aligned, and the mouse finds the device name and the title, artist
/// and album links ([`crate::core::text::LINKS`]) in them.
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
/// steps: a new line starts with a step's `→`, and what follows the
/// device (`, decoded from MP3`, the samples clipped) stays one piece
/// (with the device's name when both fit on a line), so no phrase is split
/// and no word is left alone. A piece longer than a line is wrapped at its
/// spaces.
pub fn wrap_path(path: &[Seg], max: usize) -> Vec<Vec<Seg>> {
    // The pieces: each starts at an arrow, or is the device's tail.
    let mut pieces: Vec<Vec<Seg>> = Vec::new();
    let mut tail = false;
    for s in path {
        let starts = s.text == " → " || (s.text.starts_with(", ") && !tail);
        if starts || pieces.is_empty() {
            tail = s.text.starts_with(", ");
            pieces.push(Vec::new());
        }
        pieces.last_mut().unwrap().push(s.clone());
    }
    let piece_width = |p: &[Seg]| p.iter().map(|s| width(&s.text)).sum::<usize>();
    // The device and its tail stay together when they fit on a line.
    let tail = pieces
        .iter()
        .position(|p| p.first().is_some_and(|s| s.text.starts_with(", ")));
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
            // A line starts with the step's arrow, or the tail's words.
            if let Some(first) = piece.first_mut() {
                first.text = first.text.trim_start().trim_start_matches(", ").to_owned();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_keeps_tones_and_widths() {
        let segs = vec![
            seg("file 44.1", Tone::Source),
            seg(" → ", Tone::Dim),
            seg("Speakers (USB DAC)", Tone::Device),
            seg(", decoded from MP3", Tone::Dim),
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
