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
//! seconds, a picture far more), and so is a storyboard that reads as a
//! portrait session: some of its scenes rewritten whole, in one question
//! of all that was found ([`redraw`] of [`posed`], [`close`],
//! [`camera`]), answered by every rewritten scene in the patch.

use serde_json::Value;

use super::repair;
use super::rules::{nearest, PROMPT_WORDS};
use super::{Angle, Shot, Style};

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
    /// Some of the scenes `scenes` (their places) rewritten as a
    /// storyboard's shots, the model choosing which.
    Redraw {
        /// The scenes it may rewrite.
        scenes: Vec<usize>,
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

/// The answers the follow-up's `answer` gives to `questions`, in their
/// order, read forgivingly: the JSON object it holds, each answer by its
/// number (a key whose first figures are it: `"1"`, `"1."`, `"Q1"`,
/// `"question 1"`), or a list of them in order; one object or list
/// wrapping it all taken as it; a value that is an object of one field
/// taken as that field's value; none where it gives none (or holds no
/// JSON). The storyboard's question ([`What::Redraw`]) is answered by
/// every rewritten scene in the patch wherever it sits ([`scene_shaped`]:
/// a model numbers them as it likes, one key a scene, or a bare list),
/// as a list; the other questions never by a rewritten scene. What was
/// mended is said in `notes`.
pub fn answers(
    answer: &str,
    questions: &[Question],
    notes: &mut Vec<String>,
) -> Vec<Option<Value>> {
    let patch = repair::any(answer, notes);
    let mut scenes = Vec::new();
    if let Some(patch) = &patch {
        rewritten(patch, 3, &mut scenes);
    }
    let mut held = Vec::new();
    let numbered = numbered(patch, questions.len());
    (questions.iter().zip(numbered))
        .map(|(q, by_number)| match q.what {
            What::Redraw { .. } => (!scenes.is_empty()).then(|| Value::Array(scenes.clone())),
            _ => by_number.filter(|v| {
                held.clear();
                rewritten(v, 1, &mut held);
                held.is_empty()
            }),
        })
        .collect()
}

/// The rewritten scenes in `value` ([`scene_shaped`]), `depth` levels of
/// lists and objects down at most, in order, into `scenes`.
fn rewritten(value: &Value, depth: usize, scenes: &mut Vec<Value>) {
    if scene_shaped(value) {
        scenes.push(value.clone());
        return;
    }
    let inner: Vec<&Value> = match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(fields) => fields.values().collect(),
        _ => Vec::new(),
    };
    if depth > 0 {
        for v in inner {
            rewritten(v, depth - 1, scenes);
        }
    }
}

/// The answers of a `patch` to `n` questions by their numbers (see
/// [`answers`]).
fn numbered(mut patch: Option<Value>, n: usize) -> Vec<Option<Value>> {
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

/// What a storyboard's question asks for after what it found: the scenes
/// rewritten whole, and how they are answered.
const REDRAW: &str = "Rewrite them whole as a storyboard's shots: something happening in a \
    place, often with other people around; a character doing something, small in the frame or \
    seen from behind, never posed or looking at the viewer, or no character at all (the place, \
    an object, the weather); each from another angle and distance than the scenes before and \
    after it. Answer a list of the scenes rewritten, each {\"scene\": its number, \"shot\": \
    \"...\", \"angle\": \"...\", \"characters\": [], \"shows\": \"...\", \"prompt\": \"...\"}.";

/// The fields a rewritten scene is told by: an answer's object holding one
/// of them is a scene ([`scene_shaped`]); no other question's answer is.
const SCENE_SHAPED: [&str; 4] = ["prompt", "shows", "shot", "angle"];

/// The fields a rewritten scene names itself by.
const SCENE_NUMBER: [&str; 4] = ["scene", "number", "place", "index"];

/// Whether `value` is a rewritten scene: an object with one of
/// [`SCENE_SHAPED`]'s fields and its number ([`SCENE_NUMBER`]), or two of
/// them at least (one alone, `{"prompt": "…"}`, is another question's
/// answer: a prompt asked for).
fn scene_shaped(value: &Value) -> bool {
    value.as_object().is_some_and(|o| {
        let shaped = SCENE_SHAPED.iter().filter(|f| o.contains_key(**f)).count();
        let numbered = SCENE_NUMBER.iter().any(|f| o.contains_key(*f));
        shaped >= 2 || (shaped == 1 && numbered)
    })
}

/// The scenes as a question names them, by the model's own numbers (`at`,
/// their places in its answer from 0, said from 1): `Scenes 1, 2 and 4`.
fn scenes_named(at: &[usize]) -> String {
    let all: Vec<String> = at.iter().map(|k| (k + 1).to_string()).collect();
    match all.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("Scenes {} and {last}", rest.join(", ")),
        _ => format!("Scene {}", all.join("")),
    }
}

/// What a storyboard's look at its scenes found: the scenes that share
/// something (their places among the scenes read, and the model's own
/// numbers for them), in words what they share and what some of them
/// could show instead, and how many of them to rewrite.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    /// The scenes (their places among the scenes read).
    pub scenes: Vec<usize>,
    /// What they share and what to show instead, in words.
    pub says: String,
    /// How many of them to rewrite.
    pub k: usize,
}

impl Finding {
    /// The `scenes` (each its place and the model's number for it) that
    /// share what `share` says, `k` of them to show what `instead` says.
    fn new(scenes: &[(usize, usize)], share: &str, k: usize, instead: &str) -> Finding {
        let at: Vec<usize> = scenes.iter().map(|&(_, at)| at).collect();
        Finding {
            scenes: scenes.iter().map(|&(k, _)| k).collect(),
            says: format!(
                "{} {share} ({k} of them could {instead})",
                scenes_named(&at)
            ),
            k,
        }
    }
}

/// The `scenes` (each its place and the model's number) where the
/// character `name` could read as a portrait: `k` of them to show
/// something else.
pub fn posed(name: &str, scenes: &[(usize, usize)], k: usize) -> Finding {
    let share = format!("show \"{name}\" near and facing us");
    let instead = format!(
        "show the place, an object or an action, \"{name}\" small in the frame, seen from \
         behind, or not there"
    );
    Finding::new(scenes, &share, k, &instead)
}

/// The `scenes` that are all close-ups: `k` of them to show from further
/// away.
pub fn close(scenes: &[(usize, usize)], k: usize) -> Finding {
    let instead = "be a wide shot of the place or a medium shot of an action";
    Finding::new(scenes, "are all close-ups", k, instead)
}

/// The `scenes` that all share the shot `shot` and the angle `angle`: `k`
/// of them to show from another.
pub fn camera(shot: Shot, angle: Angle, scenes: &[(usize, usize)], k: usize) -> Finding {
    let share = format!(
        "are all {} shots at the angle \"{}\"",
        shot.name(),
        angle.name()
    );
    let angles = quoted(Angle::ALL.iter().map(|a| a.name()));
    let instead = format!("be seen from another distance and angle (one of {angles})");
    Finding::new(scenes, &share, k, &instead)
}

/// The storyboard's one question of what was `found` (one finding at
/// least) among the `drawn` scenes drawn: as many of the scenes found as
/// the most any finding asks, rewritten (asked once, so that the
/// rewrites never pull against each other).
pub fn redraw(found: &[Finding], drawn: usize) -> Question {
    let mut scenes: Vec<usize> = found.iter().flat_map(|f| f.scenes.clone()).collect();
    scenes.sort_unstable();
    scenes.dedup();
    let k = (found.iter().map(|f| f.k).max().unwrap_or(1)).min(scenes.len());
    let says: Vec<&str> = found.iter().map(|f| f.says.as_str()).collect();
    Question {
        asks: format!(
            "Of the {drawn} scenes drawn, the storyboard reads as a portrait session: {}. \
             Choose {k} of these scenes and rewrite them. {REDRAW}",
            says.join("; ")
        ),
        what: What::Redraw { scenes },
    }
}

/// Every word the storyboard's question can be asked with, as one text
/// (kept in the plan's hash, as it changes what is answered): the
/// question of one of each finding.
pub fn board_words() -> String {
    let one = [(0, 0)];
    let found = [
        posed("name", &one, 1),
        close(&one, 1),
        camera(Shot::Medium, Angle::EyeLevel, &one, 1),
    ];
    redraw(&found, 1).asks
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
        let three = [
            style("fresco"),
            pass(0, "lazer", &["laser"]),
            preset("x", "", &[]),
        ];
        let got = answers(
            "```json\n{\"1\": \"watercolor\", \"2\": {\"pass\": \"laser\"},}\n```",
            &three,
            &mut notes,
        );
        assert_eq!(got, [Some("watercolor".into()), Some("laser".into()), None]);
        assert!(notes.iter().any(|n| n.contains("a comma too many")));
        assert_eq!(
            answers("I cannot say.", &questions, &mut notes),
            [None, None]
        );
        // Keys with their numbers in them, a wrapping object, a list.
        let keyed = "{\"answers\": {\"1.\": 7, \"Q2\": \"ink wash\", \"question 3\": true}}";
        let want = [Some(7.into()), Some("ink wash".into()), Some(true.into())];
        assert_eq!(answers(keyed, &three, &mut notes), want);
        assert_eq!(answers("[7, \"ink wash\", true]", &three, &mut notes), want);
        assert_eq!(figures("scene 12"), Some(12));
    }

    /// The storyboard's question is answered by every rewritten scene in
    /// the patch, wherever it sits: one key a scene (the numbers the
    /// model's own), a bare list, a list under the question's number; the
    /// other questions never by a scene, and still by their numbers.
    #[test]
    fn every_rewritten_scene_answers_the_storyboard_s_question() {
        let questions = [
            style("fresco"),
            redraw(&[close(&[(0, 0), (1, 1), (2, 2)], 2)], 3),
        ];
        let scene = |n: usize| serde_json::json!({"scene": n, "prompt": format!("p{n}")});
        let all = Some(Value::Array(vec![scene(1), scene(2), scene(5)]));
        let mut notes = Vec::new();
        for patch in [
            r#"{"1": {"scene": 1, "prompt": "p1"}, "2": {"scene": 2, "prompt": "p2"}, "3": {"scene": 5, "prompt": "p5"}}"#,
            r#"[{"scene": 1, "prompt": "p1"}, {"scene": 2, "prompt": "p2"}, {"scene": 5, "prompt": "p5"}]"#,
            r#"{"1": "ink wash", "2": [{"scene": 1, "prompt": "p1"}, {"scene": 2, "prompt": "p2"}, {"scene": 5, "prompt": "p5"}]}"#,
        ] {
            let got = answers(patch, &questions, &mut notes);
            assert_eq!(got[1], all, "{patch}");
            let style = got[0].clone();
            assert!(
                style.is_none() || style == Some("ink wash".into()),
                "{patch}"
            );
        }
        assert_eq!(
            answers("{\"1\": \"gouache painting\"}", &questions, &mut notes)[1],
            None
        );
        // A prompt asked for, in an object of its one field: that
        // question's answer, never a rewritten scene.
        let both = [prompt(0, "", "", None), questions[1].clone()];
        let got = answers("{\"1\": {\"prompt\": \"A quay\"}}", &both, &mut notes);
        assert_eq!(got, [Some("A quay".into()), None]);
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
