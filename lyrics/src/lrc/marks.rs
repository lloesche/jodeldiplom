//! What a line of lyrics is besides its words: a break in the singing, a
//! section's name, or sung, and the notes around sung words.

use crate::Line;

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

/// What a line read as `line` (its words) from `text` is: a break (no
/// words, written back as `♪`), nothing (a section's name, dropped), or
/// its sung words (the notes around them dropped).
pub(super) fn classify(mut line: Line, text: &str) -> Option<Line> {
    if is_break(&line.text()) {
        return Some(Line {
            is_break: true,
            ..Line::default()
        });
    }
    if is_section(text) {
        return None;
    }
    // A note around sung words (`♪ la la ♪`) is not sung.
    line.words.retain(|w| !w.text.chars().all(is_note));
    Some(line)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
