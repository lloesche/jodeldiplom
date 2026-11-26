//! Our own record in an LRC file, which other players pass over: header
//! tags naming the source, the aligner and the fit, and a comment after a
//! line with its words' confidences, their letters' times and when its
//! source synced it.

use std::fmt::Write as _;

use crate::{Line, Lyrics, Synced};

/// Our header: where the text came from.
pub(super) const SOURCE_TAG: &str = "source";
/// Our header: the aligner that timed it (`name version`).
pub(super) const ALIGNER_TAG: &str = "aligner";
/// Our header: how well the text fit what was heard (`78`, `24 swapped`).
pub(super) const FIT_TAG: &str = "fit";
/// Our per-line comment with the words' confidences.
const CONFIDENCE: &str = "confidence";
/// Our per-line comment with the words' letters' times.
const LETTERS: &str = "letters";
/// Our per-line comment with when the source synced the line
/// ([`crate::Line::synced`]): its start and end in seconds, `-` for none.
const SYNCED: &str = "synced";

/// Our header tags.
pub(super) const HEADER_TAGS: [&str; 3] = [SOURCE_TAG, ALIGNER_TAG, FIT_TAG];

/// Our header for `lyrics`: its source, and its aligner and fit when known.
pub(super) fn write_header(s: &mut String, lyrics: &Lyrics) {
    let _ = writeln!(s, "[{SOURCE_TAG}:{}]", lyrics.source.name());
    if let Some(by) = &lyrics.timing.by {
        let _ = writeln!(s, "[{ALIGNER_TAG}:{}]", by.to_text());
    }
    if let Some(fit) = &lyrics.timing.fit {
        let _ = writeln!(s, "[{FIT_TAG}:{}]", fit.to_text());
    }
}

/// Our comments after `line`: when its source synced it, and when its
/// words are timed and the text was `aligned` by us, their confidences and
/// their letters' times.
pub(super) fn write_comments(s: &mut String, line: &Line, aligned: bool) {
    write_synced(s, line);
    if !(aligned && line.words.iter().any(|w| w.start.is_some())) {
        return;
    }
    let _ = write!(s, "[#:{CONFIDENCE}");
    for w in &line.words {
        let _ = write!(s, " {}", (w.confidence * 100.0).round() as u32);
    }
    s.push_str("]\n");
    if line.words.iter().any(|w| !w.letters.is_empty()) {
        let _ = write!(s, "[#:{LETTERS}");
        for w in &line.words {
            if w.letters.is_empty() {
                s.push_str(" -");
                continue;
            }
            for (k, t) in w.letters.iter().enumerate() {
                let sep = if k == 0 { ' ' } else { ',' };
                let _ = write!(s, "{sep}{}", (t * 100.0).round() as u32);
            }
        }
        s.push_str("]\n");
    }
}

/// Applies our comment `value` (after `#:`) to `lines`, the lines the
/// lyrics line before it made; passes over one it does not know.
pub(super) fn apply_comment(lines: &mut [(f64, Line)], value: &str) {
    let value = value.trim();
    if let Some(values) = value.strip_prefix(CONFIDENCE) {
        confide(lines, values);
    } else if let Some(values) = value.strip_prefix(LETTERS) {
        letter(lines, values);
    } else if let Some(values) = value.strip_prefix(SYNCED) {
        let synced = synced_of(values);
        for (_, line) in lines {
            line.synced = synced;
        }
    }
}

/// Sets the words' confidences in `lines` from `values` (percentages).
fn confide(lines: &mut [(f64, Line)], values: &str) {
    let values: Option<Vec<f32>> = values
        .split_whitespace()
        .map(|v| v.parse::<f32>().ok().map(|p| p / 100.0))
        .collect();
    let Some(values) = values else {
        return;
    };
    for (_, line) in lines {
        if line.words.len() == values.len() {
            for (w, &c) in line.words.iter_mut().zip(&values) {
                w.confidence = c;
            }
        }
    }
}

/// The [`SYNCED`] comment for `line`, when its source synced it.
fn write_synced(s: &mut String, line: &Line) {
    if let Some(synced) = line.synced {
        let end = synced.end.map_or("-".into(), |e| format!("{e:.2}"));
        let _ = writeln!(s, "[#:{SYNCED} {:.2} {end}]", synced.start);
    }
}

/// When a line was synced, from a [`SYNCED`] comment's `values` (`start
/// end`, the end `-` when there is none).
fn synced_of(values: &str) -> Option<Synced> {
    let mut v = values.split_whitespace();
    let start = v.next()?.parse::<f64>().ok().filter(|t| t.is_finite())?;
    let end = match v.next()? {
        "-" => None,
        e => Some(e.parse::<f64>().ok().filter(|t| t.is_finite())?),
    };
    Some(Synced { start, end })
}

/// Sets the words' letters' times in `lines` from `values` (a word's
/// hundredths joined by commas, `-` for none); a word whose count is not
/// its characters' keeps none.
fn letter(lines: &mut [(f64, Line)], values: &str) {
    let words: Vec<&str> = values.split_whitespace().collect();
    for (_, line) in lines {
        if line.words.len() != words.len() {
            continue;
        }
        for (w, &v) in line.words.iter_mut().zip(&words) {
            let times: Option<Vec<f32>> = (v != "-")
                .then(|| {
                    v.split(',')
                        .map(|t| t.parse::<u32>().ok().map(|h| h as f32 / 100.0))
                        .collect()
                })
                .flatten();
            w.letters = times
                .filter(|t| t.len() == w.text.chars().count())
                .unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lrc::{parse, write};
    use crate::{Lyrics, Source};

    #[test]
    fn when_a_line_was_synced_is_written_and_read_back() {
        let found = Lyrics::parse_text(
            "[00:04.00]Out of time\n[00:07.25]Still here\n[00:09.00]\n",
            Source::Lrclib,
        );
        let written = write(&found);
        assert!(
            written.contains("Out of time\n[#:synced 4.00 7.25]\n"),
            "{written}"
        );
        assert!(
            written.contains("Still here\n[#:synced 7.25 9.00]\n"),
            "{written}"
        );
        let back = parse(&written, Source::Unknown);
        assert_eq!(back.lines, found.lines);
        assert!(back.lines.iter().all(|l| l.time.is_none()));
        // A comment that does not read leaves the line unsynced.
        let broken = parse("Line\n[#:synced soon -]\n", Source::Unknown);
        assert_eq!(broken.lines[0].synced, None);
    }
}
