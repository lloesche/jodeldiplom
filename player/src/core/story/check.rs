//! A model's answer read as a plan and held to every rule ([`read`]).
//!
//! The answer is read as the model writes it ([`Answer`]: no field
//! unknown, none missing, none twice), then checked: the scenes taking
//! every line exactly once in order, the characters named and described,
//! the pictures' prompts asking for no text, the counts of [`rules`],
//! and the shaders making a look of the catalogue
//! ([`super::shader::Shader::to_look`]). The rules broken are the error
//! (JSON that does not read: that alone; else the first broken of the
//! characters, of the lines, of each scene and of the art, and every
//! fault of the shaders), each said precisely, so that the model can be
//! asked again with all of them. Nothing is repaired, and one thing only
//! is taken wider than asked: a pass in the shaders' `add` that the look
//! has already stays where it is (the model lists there the passes it
//! moves as often as those it adds; a test holds it).

use super::rules::{self, CHARACTERS, KEYWORDS, MOTIFS, PALETTE, PROMPT_WORDS};
use super::shader::Shader;
use super::{Art, Character, Plan, Scene, Shot, Song};

/// Words that ask for text in a picture (whole words, case ignored): a
/// prompt with one is refused, the image model would write it. Things
/// that may carry writing (a sign, a billboard, a label) are only asked
/// against in the prompt: the model writes them often, and the pictures
/// are to be checked for text anyway (`docs/todo.md`).
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

/// The answer as the model writes it.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Answer {
    story: String,
    characters: Vec<Character>,
    scenes: Vec<Asked>,
    art: Art,
    shader: Shader,
}

/// A scene as the model writes it: its lines, no times.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Asked {
    first_line: usize,
    last_line: usize,
    section: String,
    shot: Shot,
    characters: Vec<String>,
    shows: String,
    prompt: String,
}

/// The plan in a model's `answer` for `song`, checked, its scenes timed
/// by their lines. The JSON is taken from the answer's first `{` to its
/// last `}` (a model may fence it); else the rules it breaks (see the
/// module's doc), said precisely.
pub fn read(answer: &str, song: &Song) -> Result<Plan, String> {
    let json = match (answer.find('{'), answer.rfind('}')) {
        (Some(a), Some(b)) if a < b => &answer[a..=b],
        _ => return Err("the answer holds no JSON object".into()),
    };
    let a: Answer = serde_json::from_str(json).map_err(|e| {
        let e = e.to_string();
        // serde's own words (the table test of the errors holds them).
        let hint = if e.contains("duplicate field") {
            " (write each field of an object once)"
        } else {
            ""
        };
        format!("the JSON does not read: {e}{hint}")
    })?;
    let mut broken: Vec<String> = Vec::new();
    if a.story.trim().is_empty() {
        broken.push("\"story\" is empty".into());
    }
    broken.extend(characters(&a.characters).err());
    broken.extend(lines(&a.scenes, song).err());
    for (k, scene) in a.scenes.iter().enumerate() {
        broken.extend(self::scene(&a.characters, k, scene).err());
    }
    broken.extend(art(&a.art).err());
    let look = a.shader.to_look(&a.art.why);
    if let Err(faults) = &look {
        broken.extend(faults.iter().cloned());
    }
    match look {
        Ok(look) if broken.is_empty() => Ok(Plan {
            story: a.story,
            scenes: timed(a.scenes, song),
            characters: a.characters,
            art: a.art,
            look,
        }),
        _ => Err(broken.join("; ")),
    }
}

/// The characters: as many as [`CHARACTERS`], each named as
/// [`rules::NAME`] has it (no two the same) and described.
fn characters(characters: &[Character]) -> Result<(), String> {
    CHARACTERS.check(characters.len())?;
    for (k, c) in characters.iter().enumerate() {
        if !rules::is_name(&c.name) {
            return Err(format!(
                "character {k}'s name {:?} is not {}",
                c.name,
                rules::NAME
            ));
        }
        if characters[..k].iter().any(|o| o.name == c.name) {
            return Err(format!("two characters are named {:?}", c.name));
        }
        if c.looks.trim().is_empty() {
            return Err(format!("character {:?} has no looks", c.name));
        }
    }
    Ok(())
}

/// The scenes' lines: as many scenes as [`rules::scenes`] takes, together
/// taking every line once, in order.
fn lines(scenes: &[Asked], song: &Song) -> Result<(), String> {
    let n = song.lines.len();
    rules::scenes(song).check(scenes.len())?;
    let mut next = 0;
    for (k, s) in scenes.iter().enumerate() {
        if s.first_line > s.last_line {
            return Err(format!(
                "scene {k} starts at line {} after its last line {}",
                s.first_line, s.last_line
            ));
        }
        if s.last_line >= n {
            return Err(format!(
                "scene {k} ends at line {}, but the last line is {}",
                s.last_line,
                n - 1
            ));
        }
        if s.first_line > next {
            return Err(format!(
                "{} in no scene (scene {k} starts at line {})",
                some_lines(next, s.first_line - 1),
                s.first_line
            ));
        }
        if s.first_line < next {
            return Err(format!(
                "scenes {} and {k} both take line {}: each line belongs to one scene",
                k - 1,
                s.first_line
            ));
        }
        next = s.last_line + 1;
    }
    if next < n {
        return Err(format!(
            "{} in no scene: the last scene must end at line {}",
            some_lines(next, n - 1),
            n - 1
        ));
    }
    Ok(())
}

/// Lines `a` to `b`, as an error says them.
fn some_lines(a: usize, b: usize) -> String {
    if a == b {
        format!("line {a} is")
    } else {
        format!("lines {a} to {b} are")
    }
}

/// Scene `k`: its characters described, each once; its section and what
/// it shows said; its prompt of [`PROMPT_WORDS`], asking for no text.
fn scene(characters: &[Character], k: usize, s: &Asked) -> Result<(), String> {
    for (j, name) in s.characters.iter().enumerate() {
        if !characters.iter().any(|c| c.name == *name) {
            let known: Vec<&str> = characters.iter().map(|c| c.name.as_str()).collect();
            return Err(format!(
                "scene {k} shows {name:?}, who is not among the characters ({})",
                known.join(", ")
            ));
        }
        if s.characters[..j].contains(name) {
            return Err(format!("scene {k} names {name:?} twice"));
        }
    }
    if s.section.trim().is_empty() {
        return Err(format!("scene {k} has no section"));
    }
    if s.shows.trim().is_empty() {
        return Err(format!("scene {k} does not say what it shows"));
    }
    PROMPT_WORDS
        .check(s.prompt.split_whitespace().count())
        .map_err(|e| format!("scene {k}'s prompt: {e}"))?;
    if let Some(word) = text_word(&s.prompt) {
        return Err(format!(
            "scene {k}'s prompt asks for writing in the picture ({word:?}): nothing may be written in it"
        ));
    }
    Ok(())
}

/// The first word of `prompt` that asks for text in a picture.
pub fn text_word(prompt: &str) -> Option<&str> {
    prompt
        .split(|c: char| !c.is_alphanumeric())
        .find(|w| TEXT_WORDS.iter().any(|t| t.eq_ignore_ascii_case(w)))
}

/// The scenes with their time spans: from the first line's start (the
/// first scene's from the song's start) to the next scene's start (the
/// last one's to the song's end).
fn timed(asked: Vec<Asked>, song: &Song) -> Vec<Scene> {
    let starts: Vec<f64> = (asked.iter().enumerate())
        .map(|(k, s)| {
            if k == 0 {
                0.0
            } else {
                song.lines[s.first_line].start
            }
        })
        .collect();
    let end = (song.lines.iter().map(|l| l.end)).fold(song.duration, f64::max);
    (asked.into_iter().enumerate())
        .map(|(k, s)| Scene {
            first_line: s.first_line,
            last_line: s.last_line,
            from: starts[k],
            to: starts.get(k + 1).copied().unwrap_or(end),
            section: s.section,
            shot: s.shot,
            characters: s.characters,
            shows: s.shows,
            prompt: s.prompt,
        })
        .collect()
}

/// The art: its keywords as many as [`KEYWORDS`], asking for no text; a
/// palette of [`PALETTE`]; [`MOTIFS`], none empty.
fn art(art: &Art) -> Result<(), String> {
    let keywords: Vec<&str> = (art.style_keywords.split(','))
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .collect();
    KEYWORDS
        .check(keywords.len())
        .map_err(|e| format!("\"style_keywords\": {e}"))?;
    if let Some(word) = text_word(&art.style_keywords) {
        return Err(format!(
            "\"style_keywords\" ask for writing ({word:?}): nothing may be written in the pictures"
        ));
    }
    PALETTE
        .check(art.palette.len())
        .map_err(|e| format!("the palette: {e}"))?;
    MOTIFS.check(art.motifs.len())?;
    if art.motifs.iter().any(|m| m.trim().is_empty()) {
        return Err("a motif is empty".into());
    }
    Ok(())
}
