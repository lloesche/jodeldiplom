//! What of a plan needs the model's judgement, asked once more in the same
//! conversation ([`Question`], [`ask`]), and its answer, a small patch,
//! read ([`answers`]): its prompt and its first answer are still in the
//! model's cache, so the follow-up costs only its own words. Only what is
//! wrong is asked, each question with its choices listed (the nearest few,
//! where there are many); the patch answers each by its number, and an
//! answer that is no valid choice is taken as the nearest one
//! (`super::check::Read::settle`). A scene of the very same words as an
//! earlier one drawn anew is asked about too, by itself if nothing else
//! is (whether it shows that one's picture again: a follow-up costs a few
//! seconds, a picture far more).

use serde_json::Value;

use super::repair;
use super::rules::{nearest, PROMPT_WORDS};
use super::Style;

/// The most choices a question lists where there are many (the nearest
/// to what was written); the catalogue in the prompt has the rest.
pub const CHOICES: usize = 8;

/// A question of the follow-up: what it settles, and its words.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    /// What it settles.
    pub what: What,
    /// Its words, as the model is asked them.
    pub asks: String,
}

/// What a question settles.
#[derive(Clone, Debug, PartialEq)]
pub enum What {
    /// The first line of the scene `scene` (from 0), one of `lo` to `hi`.
    Start {
        /// The scene.
        scene: usize,
        /// The first line it may take.
        lo: usize,
        /// The last.
        hi: usize,
    },
    /// The earlier scene the scene `scene` shows again (by a line of it),
    /// or none.
    Again {
        /// The scene.
        scene: usize,
    },
    /// The picture's prompt of the scene `scene`.
    Prompt {
        /// The scene.
        scene: usize,
    },
    /// The art style.
    Style,
    /// The shaders' preset.
    Preset,
    /// The `index`th pass to add.
    Pass {
        /// Its place among those to add.
        index: usize,
    },
    /// Whether the scene `scene` shows the picture of the earlier scene
    /// `earlier` again (they have the same words).
    Reuse {
        /// The scene.
        scene: usize,
        /// The earlier scene.
        earlier: usize,
    },
}

/// What the follow-up's words begin with (kept in the plan's hash, as it
/// changes what is answered).
pub const PREAMBLE: &str = "Thank you. A few things in your plan need settling. Answer each \
    question below by its number, all in one JSON object (such as {\"1\": 12, \"2\": \"a \
    name\"}), and nothing else: the rest of your plan stays as it is.\n";

/// The follow-up's words for `questions` (one at least): each by its
/// number, and how to answer them.
pub fn ask(questions: &[Question]) -> String {
    let mut words = String::from(PREAMBLE);
    for (k, q) in questions.iter().enumerate() {
        words.push_str(&format!("{}. {}\n", k + 1, q.asks));
    }
    words
}

/// The answers the follow-up's `answer` gives to `n` questions, in their
/// order, read forgivingly: the JSON object it holds, each answer by its
/// number (a key whose first figures are it: `"1"`, `"1."`, `"Q1"`,
/// `"question 1"`), or a list of them in order; one object or list
/// wrapping it all taken as it; a value that is an object of one field
/// taken as that field's value; none where it gives none (or holds no
/// JSON). What was mended is said in `notes`.
pub fn answers(answer: &str, n: usize, notes: &mut Vec<String>) -> Vec<Option<Value>> {
    let mut patch = repair::any(answer, notes);
    if let Some(Value::Object(one)) = &patch {
        // One field not named by a number, holding the answers.
        let wrapped = (one.iter().next())
            .filter(|(key, v)| {
                one.len() == 1 && figures(key).is_none() && (v.is_object() || v.is_array())
            })
            .map(|(_, v)| v);
        if let Some(inner) = wrapped.cloned() {
            patch = Some(inner);
        }
    }
    let by_number = |k: usize| -> Option<Value> {
        match patch.as_ref()? {
            Value::Array(items) => items.get(k - 1).cloned(),
            Value::Object(fields) => (fields.iter())
                .find(|(key, _)| figures(key) == Some(k))
                .map(|(_, v)| v.clone()),
            _ => None,
        }
    };
    (1..=n)
        .map(|k| match by_number(k)? {
            Value::Object(one) if one.len() == 1 => one.into_iter().next().map(|(_, v)| v),
            other => Some(other),
        })
        .collect()
}

/// The number the first figures in `key` make.
pub fn figures(key: &str) -> Option<usize> {
    let digits: String = (key.chars())
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// A scene as a question names it: its place in the answer (from 1) and
/// what it shows.
pub fn scene_named(scene: usize, shows: &str) -> String {
    let shows = shows.trim();
    if shows.is_empty() {
        format!("Scene {} of your plan", scene + 1)
    } else {
        format!("Scene {} of your plan (\"{shows}\")", scene + 1)
    }
}

/// `names` quoted and joined with commas.
pub fn quoted<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let all: Vec<String> = names.into_iter().map(|n| format!("\"{n}\"")).collect();
    all.join(", ")
}

/// The [`CHOICES`] of `among` nearest to `written`, the nearest first.
pub fn nearest_few<'a>(written: &str, among: &[&'a str]) -> Vec<&'a str> {
    let mut left: Vec<&str> = among.to_vec();
    let mut few = Vec::new();
    while few.len() < CHOICES {
        let Some(next) = nearest(written, left.iter().copied()) else {
            break;
        };
        left.retain(|n| *n != next);
        few.push(next);
    }
    few
}

/// The question of the scene `scene` (showing `shows`) whose first line
/// `written` (none: none written) fits nowhere: one of `lo` to `hi`.
pub fn start(
    scene: usize,
    shows: &str,
    written: Option<f64>,
    (lo, hi): (usize, usize),
) -> Question {
    let has = match written {
        Some(line) => {
            format!("starts at line {line}, which does not fit between the scenes around it")
        }
        None => "has no first line".to_owned(),
    };
    Question {
        what: What::Start { scene, lo, hi },
        asks: format!(
            "{} {has}: which line does it start at? A line's number from {lo} to {hi}.",
            scene_named(scene, shows)
        ),
    }
}

/// The question of the scene `scene` (showing `shows`) that shows again
/// the scene at line `line`, which is none before it: which of those
/// starting at `earlier` (their first lines), or none.
pub fn again(scene: usize, shows: &str, line: i64, earlier: &[usize]) -> Question {
    let lines: Vec<String> = earlier.iter().map(usize::to_string).collect();
    Question {
        what: What::Again { scene },
        asks: format!(
            "{} shows again the scene at line {line}, but no scene before it starts there: which \
             earlier scene does it show again? The first line of one of the scenes starting at \
             lines {}, or null for a picture of its own.",
            scene_named(scene, shows),
            lines.join(", ")
        ),
    }
}

/// The question of the scene `scene` (showing `shows`, its lines `lines`)
/// whose prompt asks for writing in the picture (`word`), or that has none.
pub fn prompt(scene: usize, shows: &str, lines: &str, word: Option<&str>) -> Question {
    let why = match word {
        Some(word) => format!(
            "has a picture prompt that asks for writing in the picture ({word:?}): write it again, \
             nothing written in the picture"
        ),
        None => format!("(its lines: {lines}) has no picture prompt: write it"),
    };
    Question {
        what: What::Prompt { scene },
        asks: format!(
            "{} {why}, {}, a concrete visual description of what the song is about there. Answer \
             the prompt as a string.",
            scene_named(scene, shows),
            PROMPT_WORDS.said()
        ),
    }
}

/// The question of the art style written `written` (empty: none), which
/// is none of them.
pub fn style(written: &str) -> Question {
    let has = if written.is_empty() {
        "Your plan has no art style".to_owned()
    } else {
        format!("The art style {written:?} is not one of the styles")
    };
    Question {
        what: What::Style,
        asks: format!(
            "{has}: which is it? Exactly one of {}.",
            quoted(Style::ALL.iter().map(|s| s.name()))
        ),
    }
}

/// The question of the shaders' preset written `written` (empty: none),
/// which is none of `ids`; its choices the nearest to it (or, where none
/// was written, to `genre`).
pub fn preset(written: &str, genre: &str, ids: &[&str]) -> Question {
    let has = if written.is_empty() {
        "Your plan has no shader preset".to_owned()
    } else {
        format!("The shader's preset {written:?} is not in the catalogue")
    };
    let to = if written.is_empty() { genre } else { written };
    Question {
        what: What::Preset,
        asks: format!(
            "{has}: which is it? One of {}, or another preset id of the catalogue, exactly as it \
             quotes it.",
            quoted(nearest_few(to, ids))
        ),
    }
}

/// The question of the `index`th pass to add, written `written`, which is
/// none of `names`.
pub fn pass(index: usize, written: &str, names: &[&str]) -> Question {
    Question {
        what: What::Pass { index },
        asks: format!(
            "The pass {written:?} to add is not in the catalogue: which is it? One of {}, or \
             another pass of the catalogue, exactly as it names it.",
            quoted(nearest_few(written, names))
        ),
    }
}

/// The question of the scene `scene` (its lines `a` to `b`) that has the
/// very words of the earlier scene `earlier`, starting at line `line`.
pub fn reuse(
    scene: usize,
    shows: &str,
    (a, b): (usize, usize),
    (earlier, line): (usize, usize),
) -> Question {
    Question {
        what: What::Reuse { scene, earlier },
        asks: format!(
            "{} (lines {a} to {b}) has the same words as the scene starting at line {line}: does it \
             show that scene's picture again? true or false.",
            scene_named(scene, shows)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The follow-up asks each question by its number; its answer is read
    /// forgivingly, each question's value by its number (one wrapped in an
    /// object of one field unwrapped), none where it gave none.
    #[test]
    fn a_follow_up_is_asked_by_number_and_its_patch_read() {
        let questions = [style("fresco"), pass(0, "lazer", &["laser", "grain"])];
        let words = ask(&questions);
        assert!(words.contains("\n1. The art style \"fresco\" is not one of the styles"));
        assert!(words.contains("\n2. The pass \"lazer\" to add is not in the catalogue: which is it? One of \"laser\", \"grain\""));
        let mut notes = Vec::new();
        let got = answers(
            "```json\n{\"1\": \"watercolor\", \"2\": {\"pass\": \"laser\"},}\n```",
            3,
            &mut notes,
        );
        assert_eq!(got, [Some("watercolor".into()), Some("laser".into()), None]);
        assert!(notes.iter().any(|n| n.contains("a comma too many")));
        assert_eq!(answers("I cannot say.", 2, &mut notes), [None, None]);
        // Keys with their numbers in them, a wrapping object, a list.
        let keyed = "{\"answers\": {\"1.\": 7, \"Q2\": \"ink wash\", \"question 3\": true}}";
        let want = [Some(7.into()), Some("ink wash".into()), Some(true.into())];
        assert_eq!(answers(keyed, 3, &mut notes), want);
        assert_eq!(answers("[7, \"ink wash\", true]", 3, &mut notes), want);
        assert_eq!(figures("scene 12"), Some(12));
    }

    /// The few choices listed are the nearest to what was written.
    #[test]
    fn the_choices_listed_are_the_nearest() {
        let names = [
            "film", "folk", "noir", "folklore", "vhs", "a", "b", "c", "d", "e",
        ];
        let few = nearest_few("folk-forever", &names);
        assert_eq!(few.len(), CHOICES);
        assert_eq!(&few[..2], ["folk", "folklore"]);
    }
}
