//! Lyrics as text: plain lines, LRC and Enhanced LRC.
//!
//! - **Plain text**: one line of lyrics a line; blank lines are dropped.
//! - **LRC**: `[mm:ss.xx]` before a line gives its time; several such tags
//!   repeat the line at each time. `[ar:...]`-style header tags are read
//!   (`offset` shifts every time) or passed over.
//! - **Enhanced LRC**: `<mm:ss.xx>` before a word gives its start; it also
//!   ends the word before. A tag followed by nothing but white space ends
//!   the word before it without starting one, so a pause between words is
//!   kept. A tag inside a word (`Hel<00:01.20>lo`, timing syllables) is
//!   read as one word.
//!
//! Section tags are dropped: a line wholly in brackets (`[Chorus]`,
//! `[Verse 2: Artist]`), and a line naming a section, such as `Chorus:`,
//! `Verse 2`, `Chorus x2` or `(Bridge)`. A line marking a break in the
//! singing (a note, `...`, an empty timed line, `[Instrumental]`,
//! `(Guitar solo)`, `*Interlude*`) is a break: no words, its time kept,
//! several in a row as one, written back as `♪`. Notes around sung words
//! are dropped.
//!
//! Written back, our own timings carry a header naming the source and the
//! aligner (and how well the text fit what was heard, when checked), a `[#:confidence ...]` line after each timed line with its
//! words' confidences in percent, and where the aligner heard the letters,
//! a `[#:letters ...]` line: for each word its characters' times after its
//! start in hundredths, joined by commas (`-` for a word without them);
//! other players pass these over.

use std::fmt::Write as _;

use crate::{Detail, Fit, Line, Lyrics, Maker, Source, Timing, Word, GIVEN};

/// The names a section of a song goes by, for the section tags dropped.
const SECTIONS: [&str; 17] = [
    "intro",
    "outro",
    "verse",
    "chorus",
    "pre-chorus",
    "prechorus",
    "post-chorus",
    "postchorus",
    "hook",
    "bridge",
    "refrain",
    "interlude",
    "instrumental",
    "breakdown",
    "solo",
    "coda",
    "drop",
];

/// The header tags of LRC files, passed over when read.
const HEADER_TAGS: [&str; 12] = [
    "ti", "ar", "al", "au", "by", "re", "ve", "length", "tool", "lang", "la", "id",
];

/// Our header: where the text came from.
const SOURCE_TAG: &str = "source";
/// Our header: the aligner that timed it (`name version`).
const ALIGNER_TAG: &str = "aligner";
/// Our header: how well the text fit what was heard (`78`, `24 swapped`).
const FIT_TAG: &str = "fit";
/// Our per-line comment with the words' confidences.
const CONFIDENCE: &str = "confidence";
/// Our per-line comment with the words' letters' times.
const LETTERS: &str = "letters";

/// Whether `text` has LRC times (`[mm:ss.xx]` at the start of a line).
pub fn is_synced(text: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim();
        line.strip_prefix('[')
            .and_then(|rest| rest.split_once(']'))
            .is_some_and(|(inner, _)| parse_time(inner).is_some())
    })
}

/// The lyrics in `text`: plain, LRC or Enhanced LRC, from `source` (or
/// the source its header names).
pub fn parse(text: &str, source: Source) -> Lyrics {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut source = source;
    let mut by = None;
    let mut fit = None;
    let mut offset = 0.0;
    // Each line with the time it sorts by: its own, else the last one
    // seen (an untimed line stays where it was).
    let mut lines: Vec<(f64, Line)> = Vec::new();
    let mut last_time = 0.0;
    // The lines the last lyrics line made (more than one when repeated).
    let mut last_made = 0..0;
    for raw in text.lines() {
        let mut rest = raw.trim();
        if rest.is_empty() {
            continue;
        }
        let mut times = Vec::new();
        while let Some((inner, after)) = rest.strip_prefix('[').and_then(|r| r.split_once(']')) {
            match parse_time(inner) {
                Some(t) => {
                    times.push(t);
                    rest = after.trim_start();
                }
                None => break,
            }
        }
        if times.is_empty() {
            if let Some((key, value)) = header_tag(rest) {
                match key.as_str() {
                    "offset" => {
                        offset = value.trim().parse::<f64>().unwrap_or(0.0) / 1000.0;
                    }
                    SOURCE_TAG => source = Source::parse(&value).unwrap_or(source),
                    ALIGNER_TAG => by = Maker::parse(&value),
                    FIT_TAG => fit = Fit::parse(&value),
                    "#" => {
                        if let Some(values) = value.trim().strip_prefix(CONFIDENCE) {
                            confide(&mut lines[last_made.clone()], values);
                        } else if let Some(values) = value.trim().strip_prefix(LETTERS) {
                            letter(&mut lines[last_made.clone()], values);
                        }
                    }
                    _ => {}
                }
                continue;
            }
        }
        let mut line = words_of(rest);
        if is_break(&line.text()) {
            line = Line {
                is_break: true,
                ..Line::default()
            };
        } else if is_section(rest) {
            continue;
        } else {
            // A note around sung words (`♪ la la ♪`) is not sung.
            line.words.retain(|w| !w.text.chars().all(is_note));
        }
        let first = lines.len();
        if times.is_empty() {
            if line.words.is_empty() && !line.is_break {
                continue;
            }
            lines.push((last_time, line));
        } else {
            let repeated = times.len() > 1;
            for &t in &times {
                let mut copy = line.clone();
                copy.time = Some(t);
                if repeated {
                    // Word times belong to one place in the song.
                    for w in &mut copy.words {
                        w.start = None;
                        w.end = None;
                        w.confidence = 0.0;
                    }
                }
                lines.push((t, copy));
            }
            last_time = times.iter().copied().fold(f64::MIN, f64::max);
        }
        last_made = first..lines.len();
    }
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut lines: Vec<Line> = lines.into_iter().map(|(_, l)| l).collect();
    // Breaks in a row are one, from the first.
    lines.dedup_by(|later, kept| later.is_break && kept.is_break);
    if offset != 0.0 {
        let shift = |t: &mut Option<f64>| {
            if let Some(t) = t {
                *t = (*t - offset).max(0.0);
            }
        };
        for line in &mut lines {
            shift(&mut line.time);
            for w in &mut line.words {
                shift(&mut w.start);
                shift(&mut w.end);
            }
        }
    }
    let detail = Lyrics::detail_of_lines(&lines);
    Lyrics {
        lines,
        source,
        timing: Timing {
            detail,
            by: by.filter(|_| detail != Detail::None),
            fit,
        },
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

/// A line of text with its word times, if any.
fn words_of(text: &str) -> Line {
    // Segments of text, each after the word time before it.
    let mut segments: Vec<(Option<f64>, &str)> = Vec::new();
    let mut rest = text;
    let mut tag = None;
    loop {
        let found = rest.find('<').and_then(|open| {
            let close = open + rest[open..].find('>')?;
            Some((open, close, parse_time(&rest[open + 1..close])?))
        });
        match found {
            Some((open, close, t)) => {
                segments.push((tag, &rest[..open]));
                tag = Some(t);
                rest = &rest[close + 1..];
            }
            None => {
                segments.push((tag, rest));
                break;
            }
        }
    }
    let mut line = Line::default();
    let mut previous_text = "";
    for (tag, segment) in segments {
        let joins = !segment.is_empty()
            && !segment.starts_with(char::is_whitespace)
            && !previous_text.is_empty()
            && !previous_text.ends_with(char::is_whitespace)
            && !line.words.is_empty();
        let mut words = segment.split_whitespace();
        if joins {
            if let (Some(last), Some(first)) = (line.words.last_mut(), words.next()) {
                last.text.push_str(first);
            }
        } else if let (Some(t), Some(last)) = (tag, line.words.last_mut()) {
            if last.end.is_none() {
                last.end = Some(t);
            }
        }
        for (i, text) in words.enumerate() {
            let mut word = Word::new(text);
            if i == 0 && !joins {
                word.start = tag;
            }
            line.words.push(word);
        }
        if !segment.is_empty() {
            previous_text = segment;
        }
    }
    for w in &mut line.words {
        if w.is_timed() {
            w.confidence = GIVEN;
        }
    }
    line.mark_backing();
    line
}

/// `[key:value]` as (lower-case key, value), when `text` is just that and
/// the key is a header's (or a comment, `#`).
fn header_tag(text: &str) -> Option<(String, String)> {
    let inner = text.strip_prefix('[')?.strip_suffix(']')?;
    let (key, value) = inner.split_once(':')?;
    let key = key.trim().to_ascii_lowercase();
    let known = HEADER_TAGS.contains(&key.as_str())
        || key == "offset"
        || key == "#"
        || key == SOURCE_TAG
        || key == ALIGNER_TAG
        || key == FIT_TAG;
    known.then(|| (key, value.trim().to_owned()))
}

/// A music symbol: a note, notes, a clef.
fn is_note(c: char) -> bool {
    matches!(
        c,
        '♩' | '♪'
            | '♫'
            | '♬'
            | '𝄞'
            | '🎵'
            | '🎶'
            | '🎼'
            | '🎸'
            | '🎹'
            | '🎷'
            | '🎺'
            | '🥁'
    )
}

/// Words that say a line is a break in the singing: one is needed.
const BREAK_WORDS: [&str; 17] = [
    "instrumental",
    "instrumentals",
    "interlude",
    "break",
    "solo",
    "solos",
    "music",
    "musik",
    "música",
    "musica",
    "musique",
    "zwischenspiel",
    "instrumentalteil",
    "pause",
    "inst",
    "instr",
    "no-lyrics",
];

/// Words that only say which break: `guitar solo`, `instrumental bridge`.
const BREAK_HELPERS: [&str; 46] = [
    "guitar",
    "guitars",
    "piano",
    "sax",
    "saxophone",
    "drum",
    "drums",
    "bass",
    "violin",
    "violins",
    "strings",
    "synth",
    "synths",
    "keyboard",
    "keyboards",
    "keys",
    "organ",
    "trumpet",
    "horn",
    "horns",
    "harmonica",
    "flute",
    "cello",
    "gitarre",
    "schlagzeug",
    "guitarra",
    "guitare",
    "de",
    "la",
    "the",
    "a",
    "an",
    "and",
    "no",
    "lyrics",
    "vocals",
    "musical",
    "long",
    "short",
    "extended",
    "intro",
    "outro",
    "bridge",
    "section",
    "only",
    "end",
];

/// Whether a line's `text` marks a break in the singing rather than being
/// sung: nothing in it but symbols (`♪`, `...`, empty), or a few words
/// that say so (`[Instrumental]`, `(Guitar solo)`, `♪ Break ♪`,
/// `*Interlude*`). Not wrapped, only words no one sings (`Instrumental`,
/// `Guitar solo`; not `Music` or `The music`).
fn is_break(text: &str) -> bool {
    if !text.chars().any(char::is_alphanumeric) {
        return true;
    }
    let edge = |c: char| !c.is_alphanumeric();
    let inner = text.trim_matches(edge);
    // Wrapped in brackets, stars, dashes, notes (not in quotes, and not a
    // sentence's own stops).
    let wrapper =
        |c: Option<char>| c.is_some_and(|c| !c.is_alphanumeric() && !".,!?;:'\"‘’“”…".contains(c));
    let trimmed = text.trim();
    let wrapped = wrapper(trimmed.chars().next()) || wrapper(trimmed.chars().last());
    let words: Vec<String> = inner
        .split(|c: char| c.is_whitespace() || c == '/' || c == ',')
        .map(|w| w.trim_matches(edge).to_lowercase())
        .filter(|w| !w.is_empty() && !is_count(w))
        .collect();
    let breaking = words
        .iter()
        .filter(|w| BREAK_WORDS.contains(&w.as_str()))
        .count();
    if breaking == 0 || words.len() > 5 {
        return false;
    }
    if !words
        .iter()
        .all(|w| BREAK_WORDS.contains(&w.as_str()) || BREAK_HELPERS.contains(&w.as_str()))
    {
        return false;
    }
    let has = |list: &[&str]| words.iter().any(|w| list.contains(&w.as_str()));
    wrapped
        || has(&[
            "instrumental",
            "instrumentals",
            "interlude",
            "zwischenspiel",
            "instrumentalteil",
        ])
        // `Guitar solo`, not `Solo`.
        || (has(&["solo", "solos"]) && words.len() > 1)
}

/// Whether `text` names a section of the song rather than being sung.
fn is_section(text: &str) -> bool {
    let text = text.trim();
    if text.len() > 2 && text.starts_with('[') && text.ends_with(']') {
        return true;
    }
    let (inner, wrapped) = match text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        Some(inner) => (inner.trim(), true),
        None => (text, false),
    };
    let colon = inner.ends_with(':');
    let inner = inner.trim_end_matches(':').trim();
    let words: Vec<String> = inner
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                .to_lowercase()
        })
        .collect();
    let Some(first) = words.first() else {
        return false;
    };
    if !SECTIONS.contains(&first.as_str()) {
        return false;
    }
    if colon && words.len() <= 6 {
        return true;
    }
    if wrapped && words.len() <= 4 {
        return true;
    }
    // `Verse 2`, `Chorus x2`, `Verse two`: a name and a count.
    words[1..].iter().all(|w| is_count(w)) && words.len() <= 3
}

/// A number or a repeat count: `2`, `ii`, `two`, `x2`, `2x`, `×2`.
fn is_count(word: &str) -> bool {
    const NUMBERS: [&str; 14] = [
        "one", "two", "three", "four", "five", "six", "i", "ii", "iii", "iv", "v", "vi", "a", "b",
    ];
    let bare = word
        .trim_start_matches(['x', '×'])
        .trim_end_matches(['x', '×']);
    (!bare.is_empty() && bare.chars().all(|c| c.is_ascii_digit())) || NUMBERS.contains(&word)
}

/// A time as `mm:ss.xx`, `mm:ss.xxx`, `mm:ss` or `mm:ss:xx`, in seconds.
pub fn parse_time(text: &str) -> Option<f64> {
    let text = text.trim();
    let (minutes, rest) = text.split_once(':')?;
    let (seconds, fraction) = match rest.split_once(['.', ':']) {
        Some((s, f)) => (s, f),
        None => (rest, ""),
    };
    let digits = |s: &str| !s.is_empty() && s.len() <= 3 && s.bytes().all(|b| b.is_ascii_digit());
    let long_minutes = minutes.len() > 3 && minutes.bytes().all(|b| b.is_ascii_digit());
    if !(digits(minutes) || long_minutes) {
        return None;
    }
    if !(digits(seconds) && (fraction.is_empty() || digits(fraction))) {
        return None;
    }
    let minutes: u64 = minutes.parse().ok()?;
    let seconds: u64 = seconds.parse().ok()?;
    if seconds >= 60 {
        return None;
    }
    // In thousandths, so a time read back is the same number written.
    let thousandths: u64 = match fraction.len() {
        0 => 0,
        1 => fraction.parse::<u64>().ok()? * 100,
        2 => fraction.parse::<u64>().ok()? * 10,
        _ => fraction.parse().ok()?,
    };
    let total = (minutes * 60 + seconds) * 1000 + thousandths;
    if total % 10 == 0 {
        Some((total / 10) as f64 / 100.0)
    } else {
        Some(total as f64 / 1000.0)
    }
}

/// `t` seconds as `mm:ss.xx` (minutes may pass 99).
pub fn format_time(t: f64) -> String {
    let hundredths = (t.max(0.0) * 100.0).round() as u64;
    format!(
        "{:02}:{:02}.{:02}",
        hundredths / 6000,
        hundredths / 100 % 60,
        hundredths % 100
    )
}

/// The lyrics as Enhanced LRC: a header with the source (and the aligner,
/// for our own times), then each line with its time and its words' times,
/// as far as they are known.
pub fn write(lyrics: &Lyrics) -> String {
    let mut s = String::from("[re:jodel-lyrics]\n");
    let _ = writeln!(s, "[{SOURCE_TAG}:{}]", lyrics.source.name());
    if let Some(by) = &lyrics.timing.by {
        let _ = writeln!(s, "[{ALIGNER_TAG}:{}]", by.to_text());
    }
    if let Some(fit) = &lyrics.timing.fit {
        let _ = writeln!(s, "[{FIT_TAG}:{}]", fit.to_text());
    }
    for line in &lyrics.lines {
        if let Some(t) = line.start() {
            let _ = write!(s, "[{}]", format_time(t));
        } else if line.words.is_empty() && !line.is_break {
            continue;
        }
        if line.is_break {
            let _ = writeln!(s, "{}", crate::BREAK);
            continue;
        }
        let timed = line.words.iter().any(|w| w.start.is_some());
        for (i, w) in line.words.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            if let Some(t) = w.start {
                let _ = write!(s, "<{}>", format_time(t));
            }
            s.push_str(&w.text);
            // An end of its own: before a pause, after the last word, or
            // before a word without a start.
            if let Some(end) = w.end {
                let next = line.words.get(i + 1).and_then(|n| n.start);
                if next.is_none_or(|n| format_time(n) != format_time(end)) {
                    let _ = write!(s, "<{}>", format_time(end));
                }
            }
        }
        s.push('\n');
        if timed && lyrics.timing.by.is_some() {
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
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(lyrics: &Lyrics) -> Vec<String> {
        lyrics.lines.iter().map(Line::text).collect()
    }

    #[test]
    fn times_read_and_write() {
        assert_eq!(parse_time("01:02.34"), Some(62.34));
        assert_eq!(parse_time("01:02.345"), Some(62.345));
        assert_eq!(parse_time("1:02"), Some(62.0));
        assert_eq!(parse_time("00:02:50"), Some(2.5));
        assert_eq!(parse_time("00:02.5"), Some(2.5));
        assert_eq!(parse_time("ar:Someone"), None);
        assert_eq!(parse_time("00:61.00"), None);
        assert_eq!(parse_time("Chorus"), None);
        assert_eq!(format_time(62.34), "01:02.34");
        assert_eq!(format_time(6005.0), "100:05.00");
        assert_eq!(parse_time(&format_time(123.45)), Some(123.45));
    }

    #[test]
    fn plain_text_drops_section_tags_and_keeps_backing_words() {
        let text = "[Verse 1: Someone]\nI walked along the road\n\nVerse 2:\n\
                    Chorus x2\n(Chorus)\nCHORUS\nBridge over the water (over the water)\n\
                    (Ooh, ooh)\n[Instrumental]\n";
        let lyrics = parse(text, Source::Unknown);
        assert_eq!(
            texts(&lyrics),
            [
                "I walked along the road",
                "Bridge over the water (over the water)",
                "(Ooh, ooh)",
                ""
            ]
        );
        assert!(lyrics.lines[3].is_break, "[Instrumental]");
        assert_eq!(lyrics.timing.detail, Detail::None);
        let bridge = &lyrics.lines[1];
        assert!(!bridge.backing);
        assert_eq!(
            bridge.words.iter().filter(|w| w.backing).count(),
            3,
            "(over the water)"
        );
        assert!(lyrics.lines[2].backing);
    }

    #[test]
    fn lrc_lines_with_times_repeats_and_offsets() {
        let text = "[ar:Someone]\n[ti:A Song]\n[offset:+500]\n\
                    [00:10.50]First line\n[00:20.00][00:40.00]Again and again\n\
                    [00:30.00]\n[00:31.00][Chorus]\n[00:35.00]Third line\n";
        assert!(is_synced(text));
        let lyrics = parse(text, Source::Lrclib);
        assert_eq!(lyrics.timing.detail, Detail::Lines);
        let times: Vec<Option<f64>> = lyrics.lines.iter().map(|l| l.time).collect();
        assert_eq!(
            times,
            [Some(10.0), Some(19.5), Some(29.5), Some(34.5), Some(39.5)]
        );
        assert_eq!(
            texts(&lyrics),
            [
                "First line",
                "Again and again",
                "",
                "Third line",
                "Again and again"
            ]
        );
        assert!(!is_synced("Just words\n[Chorus]\n"));
    }

    #[test]
    fn breaks_are_marked_one_way_and_runs_of_them_are_one() {
        let text = "[00:10.00]Sung here ♪\n[00:20.00]♪\n[00:21.00]<00:21.00>♫<00:21.00>\n\
                    [00:22.00][Instrumental Break]\n[00:23.00]\n\
                    [00:30.00]♪ And here ... ♪\n[00:40.00](Guitar solo)\n\
                    [00:50.00]The music\n[00:55.00]*Interlude*\n[01:00.00]Music\n";
        let lyrics = parse(text, Source::Lrclib);
        let shown: Vec<(String, bool)> = lyrics
            .lines
            .iter()
            .map(|l| (l.text(), l.is_break))
            .collect();
        let line = |t: &str| (t.to_owned(), false);
        let rest = (String::new(), true);
        assert_eq!(
            shown,
            [
                line("Sung here"),
                rest.clone(),
                line("And here ..."),
                rest.clone(),
                line("The music"),
                rest,
                line("Music")
            ]
        );
        assert_eq!(lyrics.lines[1].time, Some(20.0), "the first of the run");
        // Written as one note each, and read back the same.
        let written = write(&lyrics);
        assert!(written.contains("[00:20.00]♪\n[00:30.00]"), "{written}");
        assert_eq!(parse(&written, Source::Unknown).lines, lyrics.lines);
        // Plain text keeps its breaks, one in a row.
        let plain = Lyrics::parse_text("One\n♪\n(Instrumental)\nTwo\n", Source::Lrclib);
        let plain: Vec<(String, bool)> =
            plain.lines.iter().map(|l| (l.text(), l.is_break)).collect();
        assert_eq!(plain, [line("One"), (String::new(), true), line("Two")]);
    }

    #[test]
    fn what_is_a_break_and_what_is_sung() {
        for text in [
            "♪",
            "♪♪",
            "...",
            "---",
            "",
            "[Break]",
            "(break)",
            "[Instrumental]",
            "Instrumental",
            "(Instrumental Break)",
            "[Guitar Solo]",
            "Guitar solo",
            "♪ Instrumental ♪",
            "*Interlude*",
            "~ music ~",
            "[Musik]",
            "(Zwischenspiel)",
            "[Solo x2]",
            "- Pause -",
        ] {
            assert!(is_break(text), "{text:?} is a break");
        }
        for text in [
            "Music",
            "Solo",
            "Break",
            "The music",
            "Pause",
            "(Break it down)",
            "[Chorus]",
            "Break my heart",
            "\"Music\"",
            "Music!",
            "Instrumental in my life",
            "Oh, music",
        ] {
            assert!(!is_break(text), "{text:?} is sung (or a section)");
        }
    }

    #[test]
    fn enhanced_lrc_word_times_pauses_and_split_words() {
        let text =
            "[00:01.00]<00:01.00>Hello<00:01.40> <00:02.00>world <00:02.50>again<00:03.10>\n\
                    [00:04.00]<00:04.00>Hel<00:04.20>lo <00:04.60>there<00:05.00>\n";
        let lyrics = parse(text, Source::Sidecar);
        assert_eq!(lyrics.timing.detail, Detail::Words);
        let w = &lyrics.lines[0].words;
        let times: Vec<(Option<f64>, Option<f64>)> = w.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(
            times,
            [
                (Some(1.0), Some(1.4)),
                (Some(2.0), Some(2.5)),
                (Some(2.5), Some(3.1))
            ]
        );
        assert!(w.iter().all(|w| w.confidence == GIVEN));
        let w = &lyrics.lines[1].words;
        assert_eq!(w[0].text, "Hello");
        assert_eq!((w[0].start, w[0].end), (Some(4.0), Some(4.6)));
        assert_eq!((w[1].start, w[1].end), (Some(4.6), Some(5.0)));
    }

    fn timed(text: &str, start: f64, end: f64, confidence: f32) -> Word {
        Word {
            start: Some(start),
            end: Some(end),
            confidence,
            ..Word::new(text)
        }
    }

    #[test]
    fn enhanced_lrc_round_trips_with_its_provenance() {
        let mut backing = Line {
            words: vec![timed("(yeah", 7.5, 7.8, 0.3), timed("yeah)", 7.8, 8.2, 0.9)],
            time: Some(7.4),
            ..Line::default()
        };
        backing.mark_backing();
        let lyrics = Lyrics {
            lines: vec![
                Line {
                    words: vec![
                        Word {
                            letters: [0, 4, 10, 12, 30, 30]
                                .iter()
                                .map(|&h| h as f32 / 100.0)
                                .collect(),
                            ..timed("Hello,", 1.0, 1.4, 0.95)
                        },
                        timed("world", 2.0, 2.5, 0.88),
                        timed("again", 2.5, 3.1, 0.2),
                    ],
                    time: Some(0.9),
                    ..Line::default()
                },
                Line {
                    time: Some(5.0),
                    is_break: true,
                    ..Line::default()
                },
                backing,
                Line {
                    words: vec![timed("Over", 62.34, 63.0, 1.0)],
                    time: Some(62.34),
                    ..Line::default()
                },
            ],
            source: Source::Lrclib,
            timing: Timing {
                detail: Detail::Words,
                by: Some(Maker::new("ctc:wav2vec2-base-960h", 1)),
                fit: Some(Fit {
                    percent: 64,
                    swapped: true,
                }),
            },
        };
        let text = write(&lyrics);
        assert!(text
            .contains("[source:lrclib]\n[aligner:ctc:wav2vec2-base-960h 1]\n[fit:64 swapped]\n"));
        assert!(text.contains("<00:01.40> <00:02.00>world <00:02.50>again<00:03.10>"));
        assert!(text.contains("[#:confidence 95 88 20]"));
        assert!(text.contains("[#:letters 0,4,10,12,30,30 - -]"));
        let back = parse(&text, Source::Unknown);
        assert_eq!(back, lyrics);
        assert_eq!(write(&back), text);
    }

    #[test]
    fn plain_and_line_timed_text_round_trips() {
        let plain = parse("One line\nTwo (lines)\n", Source::Tag);
        assert_eq!(parse(&write(&plain), Source::Unknown), plain);
        let lines = parse(
            "[00:01.00]One line\n[00:02.50]Two (lines)\n",
            Source::Sidecar,
        );
        assert_eq!(lines.timing.detail, Detail::Lines);
        assert_eq!(parse(&write(&lines), Source::Unknown), lines);
    }
}
