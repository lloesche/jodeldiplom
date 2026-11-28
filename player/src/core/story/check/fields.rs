//! An answer's fields read forgivingly: a field missing its default, one
//! unknown left out (said), a count past its most its first kept, a name
//! made a name, the words asking for writing found.

use serde_json::{Map, Value};

use super::super::cast;
use super::super::follow::figures;
use super::super::repair::{list, number, text};
use super::super::rules::{self, CHARACTERS, KEYWORDS, MOTIFS, PALETTE};
use super::super::{Character, Confidence, Rgb, Shot, Style};
use super::{ArtRead, Drafted};

/// Words that ask for text in a picture (whole words, case ignored): the
/// image model would write it. Things that may carry writing (a sign, a
/// billboard, a label) are only asked against in the prompt: the model
/// writes them often, and the pictures are to be checked for text anyway
/// (`docs/todo.md`).
const TEXT_WORDS: &[&str] = &[
    "text",
    "texts",
    "letter",
    "letters",
    "lettering",
    "writing",
    "handwriting",
    "written",
    "poster",
    "posters",
    "logo",
    "logos",
    "caption",
    "captions",
    "headline",
    "typography",
    "font",
    "graffiti",
];

/// The answer's fields.
pub(super) const FIELDS: [&str; 5] = ["story", "characters", "scenes", "art", "shader"];

/// A scene's fields (with those its first line is looked for in, and
/// `last_line`, which older answers wrote and the scenes' first lines
/// make needless: passed over).
const SCENE_FIELDS: [&str; 11] = [
    "first_line",
    "start_line",
    "line",
    "lines",
    "same_as",
    "section",
    "shot",
    "characters",
    "shows",
    "prompt",
    "last_line",
];

/// The fields a scene's first line is looked for in, in order: its own,
/// then what a model writes instead.
const FIRST_LINE: [&str; 4] = ["first_line", "start_line", "line", "lines"];

/// The art's fields.
const ART_FIELDS: [&str; 9] = [
    "knowledge",
    "confidence",
    "genre",
    "era",
    "style",
    "style_keywords",
    "palette",
    "motifs",
    "why",
];

/// The fields of `fields` (of `what`) none of `known`, said left out in
/// `notes` (each once).
pub(super) fn unknown(
    fields: &Map<String, Value>,
    known: &[&str],
    what: &str,
    notes: &mut Vec<String>,
) {
    for key in fields.keys().filter(|k| !known.contains(&k.as_str())) {
        let note = format!("{what}'s field {key:?} is not asked for: left out");
        if !notes.contains(&note) {
            notes.push(note);
        }
    }
}

/// The characters in `value`: each named as [`rules::name`] has it, its
/// looks its appearance alone ([`cast::appearance`]); one without a name
/// or looks, one named twice, a group, and those past [`CHARACTERS`]' most
/// left out (each said in `notes`).
pub(super) fn characters(value: Option<&Value>, notes: &mut Vec<String>) -> Vec<Character> {
    let mut kept: Vec<Character> = Vec::new();
    for v in list(value) {
        if let Value::Object(fields) = &v {
            unknown(fields, &["name", "looks"], "a character", notes);
        }
        let written = text(v.get("name")).unwrap_or_default();
        let name = rules::name(&written);
        let (looks, doings) = cast::appearance(&text(v.get("looks")).unwrap_or_default());
        if !doings.is_empty() {
            notes.push(format!(
                "the character {written:?}'s looks are appearance only: {doings:?} left out"
            ));
        }
        if name.is_empty() || looks.is_empty() {
            notes.push(format!(
                "the character {written:?} without a name or looks: left out"
            ));
            continue;
        }
        if rules::is_group(&name) {
            notes.push(format!(
                "the character {name:?} is a group, no portrait: left to the scenes' prompts"
            ));
            continue;
        }
        if kept.iter().any(|c| c.name == name) {
            notes.push(format!("two characters named {name:?}: the first kept"));
            continue;
        }
        if name != written {
            notes.push(format!("the character {written:?} named {name:?}"));
        }
        kept.push(Character { name, looks });
    }
    notes.extend(CHARACTERS.hold(&mut kept));
    kept
}

/// A scene's `fields` read (the `at`th the model wrote): its first line (as a number, the first
/// figures of a string, or the first of a list), its characters among
/// `characters` (a name no one described dropped, a name twice once),
/// each mending said in `notes`.
pub(super) fn scene(
    fields: &Map<String, Value>,
    at: usize,
    characters: &[Character],
    notes: &mut Vec<String>,
) -> Drafted {
    unknown(fields, &SCENE_FIELDS, "a scene", notes);
    let written = FIRST_LINE.iter().find_map(|key| match fields.get(*key) {
        Some(Value::Array(items)) => number(items.first()),
        Some(Value::String(s)) => figures(s).map(|n| n as f64),
        other => number(other),
    });
    let mut seen: Vec<String> = Vec::new();
    for v in list(fields.get("characters")) {
        let Some(written) = text(Some(&v)) else {
            continue;
        };
        let name = rules::name(&written);
        if !characters.iter().any(|c| c.name == name) {
            notes.push(format!(
                "a scene names {written:?}, whom no one described: the name dropped"
            ));
        } else if !seen.contains(&name) {
            seen.push(name);
        }
    }
    let same_as = number(fields.get("same_as")).map(|l| super::Again::Written(l.round() as i64));
    Drafted {
        at,
        written,
        first: 0,
        standing: false,
        bounds: None,
        same_as,
        section: text(fields.get("section")).unwrap_or_default(),
        shot: shot(text(fields.get("shot")).as_deref(), notes),
        characters: seen,
        shows: text(fields.get("shows")).unwrap_or_default(),
        prompt: text(fields.get("prompt")).unwrap_or_default(),
    }
}

/// The shot `written`: by its name, a close one or a far one by what it
/// says, else medium (said in `notes`, but where it is its name or none
/// is written).
fn shot(written: Option<&str>, notes: &mut Vec<String>) -> Shot {
    let Some(written) = written.filter(|w| !w.is_empty()) else {
        return Shot::Medium;
    };
    let w = written.to_lowercase();
    let far = [
        "wide",
        "aerial",
        "establishing",
        "long",
        "landscape",
        "panoram",
    ];
    let shot = match w.as_str() {
        "wide" => return Shot::Wide,
        "medium" => return Shot::Medium,
        "close-up" => return Shot::CloseUp,
        w if w.contains("close") => Shot::CloseUp,
        w if far.iter().any(|f| w.contains(f)) => Shot::Wide,
        _ => Shot::Medium,
    };
    notes.push(format!("the shot {w:?} taken as {:?}", shot.name()));
    shot
}

/// The art in `value`, a field missing its default, the style as written
/// where it is none, the keywords asking for writing left out, colours
/// that are none left out, the counts past their most (each said in
/// `notes`).
pub(super) fn art(value: Option<&Value>, notes: &mut Vec<String>) -> ArtRead {
    let field = |key: &str| value.and_then(|v| v.get(key));
    let words = |key: &str| text(field(key)).unwrap_or_default();
    if let Some(Value::Object(fields)) = value {
        unknown(fields, &ART_FIELDS, "the art", notes);
    }
    let confidence = match words("confidence").to_lowercase().as_str() {
        "high" => Confidence::High,
        "medium" => Confidence::Medium,
        "low" => Confidence::Low,
        other => {
            notes.push(format!("the confidence {other:?} taken as \"low\""));
            Confidence::Low
        }
    };
    let style_written = words("style");
    let style = Style::named(&style_written).ok_or(style_written);
    // A list of keywords is taken as written one after another.
    let keywords: Vec<String> = match field("style_keywords") {
        Some(Value::Array(items)) => items.iter().filter_map(|v| text(Some(v))).collect(),
        other => (text(other).unwrap_or_default().split(','))
            .map(|k| k.trim().to_owned())
            .collect(),
    };
    let mut keywords: Vec<String> = (keywords.into_iter())
        .filter(|k| !k.is_empty())
        .filter(|k| match text_word(k) {
            Some(word) => {
                notes.push(format!(
                    "the keyword {k:?} asks for writing ({word:?}): left out"
                ));
                false
            }
            None => true,
        })
        .collect();
    notes.extend(KEYWORDS.hold(&mut keywords));
    let mut palette = Vec::new();
    for v in list(field("palette")) {
        let written = text(Some(&v)).unwrap_or_default();
        match colour(&written) {
            Some(rgb) => palette.push(rgb),
            None => notes.push(format!(
                "the colour {written:?} is no \"#rrggbb\": left out"
            )),
        }
    }
    notes.extend(PALETTE.hold(&mut palette));
    let mut motifs: Vec<String> = (list(field("motifs")).iter())
        .filter_map(|v| text(Some(v)))
        .filter(|m| !m.is_empty())
        .collect();
    notes.extend(MOTIFS.hold(&mut motifs));
    ArtRead {
        knowledge: words("knowledge"),
        confidence,
        genre: words("genre"),
        era: words("era"),
        style,
        style_keywords: keywords.join(", "),
        palette,
        motifs,
        why: words("why"),
    }
}

/// The colour `written`: `#rrggbb`, its `#` missing, or `#rgb`.
fn colour(written: &str) -> Option<Rgb> {
    let hex = written.trim().trim_start_matches('#');
    let long: String = match hex.len() {
        3 => hex.chars().flat_map(|c| [c, c]).collect(),
        _ => hex.to_owned(),
    };
    format!("#{long}").parse().ok()
}

/// The first word of `prompt` that asks for text in a picture.
pub fn text_word(prompt: &str) -> Option<&str> {
    prompt
        .split(|c: char| !c.is_alphanumeric())
        .find(|w| TEXT_WORDS.iter().any(|t| t.eq_ignore_ascii_case(w)))
}

/// `prompt` without the words that ask for text in a picture.
pub(super) fn without_text(prompt: &str) -> String {
    let kept: Vec<&str> = (prompt.split(' '))
        .filter(|w| text_word(w).is_none())
        .collect();
    kept.join(" ")
}

/// The words of the lines of a scene, one after another: what a scene
/// with nothing else is drawn of.
pub(super) fn words_of(lines: &[String]) -> String {
    let words: Vec<&str> = (lines.iter())
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    words.join(", ")
}

/// A line's words, as two lines are compared: its letters and figures in
/// lower case, a space between its words.
pub(super) fn plain(line: &str) -> String {
    let words: Vec<String> = (line.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    words.join(" ")
}
