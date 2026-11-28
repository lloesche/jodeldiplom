//! The plan's forgiving reading on good and broken answers (every slip
//! mended and said, what needs judgement asked, the nearest valid choice
//! where it is not answered), the prompt (a snapshot, every count it asks
//! for, the most scenes never said), the streamed prose, the look the
//! shaders make and the song as the plan is made from it; no model
//! needed.

use serde_json::Value;

use super::check::{read, text_word, Read, NOTHING};
use super::follow::What;
use super::prompt::prompt;
use super::prose::Prose;
use super::rules::{self, COUNTS, MOST_SCENES};
use super::*;

/// A song of twelve lines over two minutes (words of our own).
pub(crate) fn song() -> Song {
    let words = [
        "I walked down to the harbour",
        "The boats were sleeping in the rain",
        "You came running with your coat undone",
        "And we laughed at the grey sky",
        "Carry me over the water",
        "Carry me home before the night",
        "My father mended nets all winter",
        "His hands were rough as rope",
        "Now the lighthouse turns for no one",
        "And the gulls have gone inland",
        "Carry me over the water",
        "Carry me home before the night",
    ];
    let lines = (words.iter().enumerate())
        .map(|(k, w)| Line {
            start: 8.0 + 9.0 * k as f64,
            end: 8.0 + 9.0 * k as f64 + 7.5,
            text: (*w).to_owned(),
        })
        .collect();
    Song {
        title: "Harbour Song".into(),
        artist: "The Example Band".into(),
        album: "Grey Water".into(),
        tags: vec![("date".into(), "1994".into())],
        duration: 120.0,
        lines,
    }
}

/// The story of [`GOOD`].
const STORY: &str = "A woman remembers the harbour of her youth and a friend who ran to her in the rain. Her father's work is over and the town is empty, and she longs to be carried home.";

/// A good answer for [`song`], fenced as the model writes it: its second
/// chorus shows the first again.
pub(crate) const GOOD: &str = r##"```json
{"story": "A woman remembers the harbour of her youth and a friend who ran to her in the rain. Her father's work is over and the town is empty, and she longs to be carried home.",
 "characters": [{"name": "narrator", "looks": "Woman, late 20s, slender, long dark hair, yellow raincoat"},
                {"name": "you", "looks": "Man, late 20s, tall, curly red hair, open navy coat"},
                {"name": "father", "looks": "Man, 60s, broad, grey beard, wool sweater"}],
 "scenes": [
  {"first_line": 0, "section": "verse 1", "shot": "wide", "characters": ["narrator"], "shows": "The narrator walks to a rainy harbour.", "prompt": "A small harbour in steady rain at dusk, narrator walking along the quay, moored fishing boats in the foreground, grey hills behind"},
  {"first_line": 2, "section": "verse 1", "shot": "medium", "characters": ["narrator", "you"], "shows": "The two laugh under a grey sky.", "prompt": "Rain-soaked quay in the afternoon, you running toward narrator with his coat open, both laughing, puddles in front, low clouds over the sea"},
  {"first_line": 4, "section": "chorus", "shot": "wide", "characters": [], "shows": "The sea at nightfall.", "prompt": "Wide dark sea at nightfall, a small rowing boat crossing toward distant lights, waves in the foreground, a fading sky above"},
  {"first_line": 6, "section": "verse 2", "shot": "close-up", "characters": ["father"], "shows": "The father mends nets.", "prompt": "Close view of father's rough hands mending a fishing net by lamplight in winter, coils of rope in front, a frosted window behind"},
  {"first_line": 8, "section": "verse 2", "shot": "wide", "characters": [], "shows": "An empty lighthouse turns.", "prompt": "Lonely lighthouse on a rocky point at night, its beam sweeping over empty water, rocks in the foreground, clouds drifting behind"},
  {"first_line": 10, "section": "chorus", "same_as": 4}
 ],
 "art": {"knowledge": "I do not know this band.", "confidence": "low", "genre": "folk rock", "era": "90s",
  "style": "watercolor", "style_keywords": "watercolor, soft washes, rain, muted greys, paper texture, 1990s",
  "palette": ["#3a4a5a", "#8a9aa8", "#d9c27a", "#2b2b2b", "#f0ece0"], "motifs": ["boats", "rain", "lighthouse"],
  "why": "Soft washes fit a remembered, rainy town."},
 "shader": {"preset": "folk", "add": ["grain"], "knobs": {"canvas": {"amount": 0.5}, "grain": {"amount": 0.45, "size": 2}},
            "drive": [{"pass": "liquid", "knob": "amount", "by": "chorus", "amount": 0.6}], "mood": ["longing", "rain"]}}
```"##;

/// [`GOOD`] with `from` replaced by `to` (which must be there).
fn bad(from: &str, to: &str) -> String {
    assert!(GOOD.contains(from), "{from}");
    GOOD.replacen(from, to, 1)
}

/// `answer` read for [`song`], which must hold something to draw.
fn reading(answer: &str) -> Read {
    read(answer, &song()).expect("something to draw")
}

/// The plan `read` makes with `answers` to its questions (none: none
/// answered), and what was mended and asked.
fn settled(mut read: Read, answers: &[Value]) -> (Plan, Vec<String>) {
    let questions = read.questions();
    let answers: Vec<Option<Value>> = answers.iter().cloned().map(Some).collect();
    read.settle(&questions, &answers);
    read.plan()
}

/// The plan of `answer` for [`song`], its questions answered by none:
/// the nearest valid choices.
pub(crate) fn planned(answer: &str) -> Plan {
    settled(reading(answer), &[]).0
}

/// The plan of `answer` and what was mended, its questions not answered.
fn mended(answer: &str) -> (Plan, Vec<String>) {
    settled(reading(answer), &[])
}

/// Whether one of `notes` says `said`.
fn says(notes: &[String], said: &str) -> bool {
    notes.iter().any(|n| n.contains(said))
}

/// What `answer`'s questions settle.
fn asked(answer: &str) -> Vec<What> {
    reading(answer)
        .questions()
        .into_iter()
        .map(|q| q.what)
        .collect()
}

/// A good answer is read whole, nothing mended or asked: its characters,
/// its scenes from their first lines alone (each to the next one's start;
/// the first from the song's start, the last to its end), the chorus that
/// comes back showing the first one again, its art, its look; kept and
/// read back the same.
#[test]
fn a_good_answer_is_read_and_its_scenes_timed() {
    let read = reading(GOOD);
    assert!(read.notes.is_empty(), "{:?}", read.notes);
    // Only how the chorus that comes back was read is said.
    assert!(read.questions().is_empty());
    let (plan, notes) = mended(GOOD);
    assert_eq!(
        notes,
        ["read: scene 5 shows scene 2 again (same_as 4 read as a line of it)"]
    );
    assert_eq!(plan.characters.len(), 3);
    assert_eq!(plan.scenes.len(), 6);
    let lines: Vec<(usize, usize)> = plan
        .scenes
        .iter()
        .map(|s| (s.first_line, s.last_line))
        .collect();
    assert_eq!(lines, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]);
    let spans: Vec<(f64, f64)> = plan.scenes.iter().map(|s| (s.from, s.to)).collect();
    assert_eq!(spans[0], (0.0, 26.0));
    assert_eq!(spans[1], (26.0, 44.0));
    assert_eq!(spans[5], (98.0, 120.0));
    let again: Vec<Option<usize>> = plan.scenes.iter().map(|s| s.same_as).collect();
    assert_eq!(again, [None, None, None, None, None, Some(2)]);
    assert_eq!(plan.scenes[3].shot, Shot::CloseUp);
    assert_eq!(plan.art.confidence, Confidence::Low);
    assert_eq!(plan.art.style, Style::Watercolor);
    assert_eq!(plan.art.palette[0], Rgb([0x3a, 0x4a, 0x5a]));
    assert_eq!(plan.art.palette[0].to_string(), "#3a4a5a");
    assert_eq!(Plan::from_json(&plan.to_json()), Ok(plan));
}

/// Only an answer with nothing to draw is refused: no JSON object, or no
/// scene in it.
#[test]
fn only_an_answer_with_nothing_to_draw_is_refused() {
    for answer in [
        "I would rather not.",
        "{\"story\": \"a story\"}",
        "{\"scenes\": [\"one\", 2]}",
    ] {
        let refused = read(answer, &song()).expect_err(answer);
        assert!(refused.why.starts_with(NOTHING), "{}", refused.why);
    }
}

/// Each slip a small model makes is mended and said, nothing usable
/// thrown away, the plan taken.
#[test]
fn every_slip_is_mended_and_said() {
    let seven: Vec<String> = (0..7)
        .map(|k| format!("{{\"name\": \"person {k}\", \"looks\": \"Someone\"}}"))
        .collect();
    let characters =
        &GOOD[GOOD.find("[{\"name\"").unwrap()..GOOD.find("],\n \"scenes\"").unwrap() + 1];
    let cases: Vec<(String, &str)> = vec![
        // The JSON.
        (bad("\"era\": \"90s\",", "\"era\": \"90s\",,"), "a comma too many"),
        (bad("\"mood\": [\"longing\", \"rain\"]", "\"mood\": [\"longing\", \"rain\",]"), "a comma too many"),
        (bad("The sea at nightfall.", "The sea \"at\" nightfall."), "a quote inside a string"),
        (bad("\"story\": \"A", "\"stroy\": \"x\", \"story\": \"A"), "field \"stroy\" is not asked for"),
        (bad("\"shot\": \"wide\"", "\"shot\": \"wide\", \"shot\": \"medium\""), "the field \"shot\" twice"),
        (bad("\"first_line\": 0,", "\"first_line\": 0, \"from\": 3,"), "field \"from\" is not asked for"),
        // The characters.
        (bad(characters, &format!("[{}]", seven.join(", "))), "7 characters: the first 6 kept"),
        (bad("{\"name\": \"father\"", "{\"name\": \"Father\""), "the character \"Father\" named \"father\""),
        (bad("{\"name\": \"father\"", "{\"name\": \"you\""), "two characters named \"you\": the first kept"),
        (bad("[\"father\"]", "[\"mother\"]"), "names \"mother\", whom no one described: the name dropped"),
        // The scenes.
        (bad("\"shot\": \"wide\"", "\"shot\": \"aerial\""), "the shot \"aerial\" taken as \"wide\""),
        (bad("\"shot\": \"close-up\"", "\"shot\": \"extreme close up\""), "taken as \"close-up\""),
        (bad("\"first_line\": 0,", "\"first_line\": 1,"), "the first scene starts at line 0"),
        // The art.
        (bad("\"#3a4a5a\"", "\"slate\""), "the colour \"slate\" is no \"#rrggbb\": left out"),
        (bad("paper texture, 1990s", "paper texture, 1990s, hand lettering"), "the keyword \"hand lettering\" asks for writing"),
        (bad("\"confidence\": \"low\"", "\"confidence\": \"unsure\""), "the confidence \"unsure\" taken as \"low\""),
        // The shaders.
        (bad("\"size\": 2", "\"size\": 4"), "grain.size set to 4: taken as 3"),
        (bad("\"size\": 2", "\"sise\": 2"), "grain.sise taken as grain.size"),
        (bad("\"add\": [\"grain\"]", "\"add\": []"), "grain is moved, so it is added to the look"),
        // A real slip ("Suddenly I See", refused twice for it before): a
        // pass the look has not, moved by the music.
        (bad("\"pass\": \"liquid\"", "\"pass\": \"glow\""), "glow is moved, so it is added to the look"),
        (bad("\"canvas\": {\"amount\": 0.5}", "\"canvas\": 0.5"), "canvas set to 0.5 alone: taken as its amount"),
        (bad("\"by\": \"chorus\"", "\"by\": \"choruses\""), "moved by \"choruses\": taken as \"chorus\""),
        (bad("\"amount\": 0.6", "\"amount\": 1.6"), "liquid.amount's amount set to 1.6: taken as 1"),
        (
            bad("\"amount\": 0.6}]", "\"amount\": 0.6}, {\"pass\": \"liquid\", \"knob\": \"amount\", \"by\": \"bass\", \"amount\": 0.2}]"),
            "liquid.amount moved twice: the first kept",
        ),
        (
            bad("\"add\": [\"grain\"], \"knobs\": {", "\"add\": [\"grain\", \"cutout\"], \"knobs\": {\"cutout\": {\"layers\": 2.5}, "),
            "cutout.layers set to 2.5: taken as 3",
        ),
    ];
    for (answer, said) in cases {
        let (plan, notes) = mended(&answer);
        assert!(says(&notes, said), "{said:?} not in {notes:?}");
        assert_eq!(plan.scenes.len(), 6, "{said}: every scene kept");
        assert!(
            plan.look.faults().is_empty(),
            "{said}: {:?}",
            plan.look.faults()
        );
    }
    // A name twice in a scene is named once.
    let (plan, _) = mended(&bad(
        "[\"narrator\", \"you\"]",
        "[\"narrator\", \"Narrator\", \"you\"]",
    ));
    assert_eq!(plan.scenes[1].characters, ["narrator", "you"]);
    // A real slip: a name written with an underscore (the model's answer
    // was refused twice for it before).
    let (plan, notes) = mended(
        &bad("\"father\", \"looks\"", "\"white_man\", \"looks\"")
            .replacen("[\"father\"]", "[\"white_man\"]", 1)
            .replacen("view of father's", "view of white_man's", 1),
    );
    assert!(
        says(&notes, "the character \"white_man\" named \"white man\""),
        "{notes:?}"
    );
    assert_eq!(plan.scenes[3].characters, ["white man"]);
}

/// A crowd is never a character (it would be one portrait): it stays in
/// the scenes' prompts; a character seen in no scene drawn has no
/// portrait either.
#[test]
fn a_group_or_a_character_never_seen_is_no_portrait() {
    let crowd = bad(
        "{\"name\": \"father\", \"looks\": \"Man, 60s, broad, grey beard, wool sweater\"}",
        "{\"name\": \"father\", \"looks\": \"Man, 60s, broad, grey beard, wool sweater\"}, {\"name\": \"the crowd\", \"looks\": \"Dozens of dancers\"}, {\"name\": \"cat\", \"looks\": \"A black cat\"}",
    )
    .replacen("\"characters\": []", "\"characters\": [\"the crowd\"]", 1);
    let (plan, notes) = mended(&crowd);
    assert!(
        says(
            &notes,
            "the character \"the crowd\" is a group, no portrait"
        ),
        "{notes:?}"
    );
    assert!(
        says(
            &notes,
            "the character \"cat\" is seen in no scene drawn: left out"
        ),
        "{notes:?}"
    );
    let names: Vec<&str> = plan.characters.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["narrator", "you", "father"]);
    assert!(plan.scenes[2].characters.is_empty());
}

/// The scenes' lines come from their first lines alone: the first scene
/// from line 0; those going up in order (the most of them) stand; one out
/// of order, out of the song or with none is asked where it starts,
/// between the scenes around it (else, not answered, put where it fits
/// nearest); one with no line left between them is left out, its lines
/// shown by the scene before; every line in one scene.
#[test]
fn the_scenes_lines_come_from_their_first_lines_alone() {
    // Scene 3 starts at line 40 (out of the song), scene 4 at 3 (out of
    // order): asked, from 5 to 7 and 7 to 9.
    let misplaced = bad("\"first_line\": 6,", "\"first_line\": 40,").replacen(
        "\"first_line\": 8,",
        "\"first_line\": 3,",
        1,
    );
    let starts: Vec<What> = asked(&misplaced);
    assert_eq!(
        starts,
        [
            What::Start {
                scene: 3,
                lo: 5,
                hi: 8
            },
            What::Start {
                scene: 4,
                lo: 6,
                hi: 9
            },
        ]
    );
    let (plan, notes) = settled(reading(&misplaced), &[Value::from(6), Value::from(8)]);
    let lines: Vec<(usize, usize)> = plan
        .scenes
        .iter()
        .map(|s| (s.first_line, s.last_line))
        .collect();
    assert_eq!(lines, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]);
    assert!(
        says(&notes, "scene 3 starts at line 6, as answered"),
        "{notes:?}"
    );
    // Not answered (or answered wrong): the nearest, each after the one
    // before.
    let (plan, notes) = settled(reading(&misplaced), &[Value::from(99), Value::from("soon")]);
    let firsts: Vec<usize> = plan.scenes.iter().map(|s| s.first_line).collect();
    assert_eq!(firsts, [0, 2, 4, 8, 9, 10]);
    assert!(says(&notes, "(the nearest; answered 99)"), "{notes:?}");
    // A scene with no line left between two that stand: left out, the
    // one before showing its lines.
    let crowded = bad("\"first_line\": 4,", "\"first_line\": 2,").replacen(
        "\"first_line\": 6,",
        "\"first_line\": 3,",
        1,
    );
    let (plan, notes) = mended(&crowded);
    assert!(
        says(
            &notes,
            "fits nowhere between the scenes around it: left out"
        ),
        "{notes:?}"
    );
    let lines: Vec<(usize, usize)> = plan
        .scenes
        .iter()
        .map(|s| (s.first_line, s.last_line))
        .collect();
    assert_eq!(lines, [(0, 1), (2, 2), (3, 7), (8, 9), (10, 11)]);
    // The first line read from a string, or a range of lines.
    let written = bad("\"first_line\": 2,", "\"lines\": \"2-3\",").replacen(
        "\"first_line\": 4,",
        "\"first_line\": \"4\",",
        1,
    );
    let (plan, _) = mended(&written);
    assert_eq!(plan.scenes[1].first_line, 2);
    assert_eq!(plan.scenes[2].first_line, 4);
}

/// The model decides how many scenes: one is enough (it shows the whole
/// song), and those past the most are merged into the last kept, which
/// shows on to the song's end.
#[test]
fn a_plan_has_one_scene_or_more_and_at_most_its_ceiling() {
    let mut long = song();
    long.duration = 600.0;
    long.lines = (0..60)
        .map(|k| Line {
            start: 5.0 + 9.0 * k as f64,
            end: 5.0 + 9.0 * k as f64 + 8.0,
            text: format!("line {k}"),
        })
        .collect();
    let scene = |k: usize| {
        format!("{{\"first_line\": {}, \"shot\": \"wide\", \"shows\": \"s\", \"prompt\": \"a field of wheat at noon {k}\"}}", 2 * k)
    };
    let all: Vec<String> = (0..25).map(scene).collect();
    let answer = format!("{{\"story\": \"s\", \"scenes\": [{}], \"art\": {{\"style\": \"watercolor\"}}, \"shader\": {{\"preset\": \"folk\"}}}}", all.join(", "));
    let mut read = read(&answer, &long).unwrap();
    assert!(says(
        &read.notes,
        "25 scenes: those past the 20th merged into it"
    ));
    let questions = read.questions();
    read.settle(&questions, &[]);
    let plan = read.plan().0;
    assert_eq!(plan.scenes.len(), MOST_SCENES);
    let last = plan.scenes.last().unwrap();
    assert_eq!((last.first_line, last.last_line, last.to), (38, 59, 600.0));
    // One scene.
    let one = "{\"scenes\": [{\"first_line\": 0, \"prompt\": \"a harbour in the rain at dusk\"}]}";
    let (plan, notes) = mended(one);
    assert_eq!(plan.scenes.len(), 1);
    assert_eq!(
        (
            plan.scenes[0].from,
            plan.scenes[0].to,
            plan.scenes[0].last_line
        ),
        (0.0, 120.0, 11)
    );
    // Its art and shaders missing: their defaults, the nearest style and
    // preset.
    assert!(says(&notes, "question"), "{notes:?}");
    assert!(plan.look.faults().is_empty());
}

/// A scene shown again draws nothing new: it names a line of an earlier
/// scene (its first or any other); one naming no scene before it is
/// asked which (or none), and not answered is drawn anew where it has a
/// prompt, else shows the nearest earlier scene of its section.
#[test]
fn a_scene_shown_again_draws_nothing_new() {
    let (plan, _) = mended(&bad("\"same_as\": 4", "\"same_as\": 5"));
    assert_eq!(plan.scenes[5].same_as, Some(2), "a line inside the chorus");
    let later = bad("\"same_as\": 4", "\"same_as\": 11");
    assert_eq!(asked(&later), [What::Again { scene: 5 }]);
    let (plan, notes) = settled(reading(&later), &[Value::from(4)]);
    assert_eq!(plan.scenes[5].same_as, Some(2));
    assert!(
        says(&notes, "scene 5 shows scene 2 again, as answered"),
        "{notes:?}"
    );
    // Not answered: no prompt of its own, so it shows the earlier chorus.
    let (plan, notes) = mended(&later);
    assert_eq!(plan.scenes[5].same_as, Some(2), "{notes:?}");
    // Answered null, with a prompt of its own: drawn anew.
    let own = bad(
        "\"same_as\": 4}",
        "\"same_as\": 11, \"prompt\": \"the open sea at dawn, a gull over the waves\"}",
    );
    let (plan, _) = settled(reading(&own), &[Value::Null]);
    assert_eq!(plan.scenes[5].same_as, None);
    assert_eq!(pictures::pictures(&plan).scenes.len(), 6);
}

/// The real slips of the review and the runs: `same_as` written as the
/// earlier scene's place among the scenes ("Suddenly I See": its chorus
/// named scene 4, read as line 4, a pre-chorus) is read as a place where
/// only that is of its section; a first line written "line 4" is read; a
/// pass or a knob no one knows, far from any, is left out rather than
/// taken as something else.
#[test]
fn a_slip_is_read_as_it_was_meant_or_left_out() {
    let by_place = bad("\"same_as\": 4", "\"same_as\": 2");
    let (plan, notes) = mended(&by_place);
    assert_eq!(plan.scenes[5].same_as, Some(2), "{notes:?}");
    assert!(
        says(
            &notes,
            "same_as 2 read as its place among the scenes, of the same section"
        ),
        "{notes:?}"
    );
    let (plan, _) = mended(&bad("\"first_line\": 4,", "\"first_line\": \"line 4\","));
    assert_eq!(plan.scenes[2].first_line, 4);
    let far = bad(
        "\"knobs\": {",
        "\"knobs\": {\"zebra\": {\"intensity\": 0.9}, \"grain\": {\"wobble\": 1}, ",
    );
    let (plan, notes) = mended(&far);
    assert!(
        says(&notes, "the pass \"zebra\" is none: left out"),
        "{notes:?}"
    );
    assert!(
        says(&notes, "grain has no knob \"wobble\": left out"),
        "{notes:?}"
    );
    assert!(plan.look.faults().is_empty());
    // Yes in any case and form.
    let twice = bad(
        "{\"first_line\": 10, \"section\": \"chorus\", \"same_as\": 4}",
        "{\"first_line\": 10, \"section\": \"chorus\", \"prompt\": \"A dark sea at night under a thin moon\"}",
    );
    for yes in [
        Value::from("Yes"),
        Value::from("TRUE"),
        Value::from(1),
        Value::from(4),
    ] {
        let (plan, _) = settled(reading(&twice), std::slice::from_ref(&yes));
        assert_eq!(plan.scenes[5].same_as, Some(2), "{yes}");
    }
    let (plan, _) = settled(reading(&twice), &[Value::from("No")]);
    assert_eq!(plan.scenes[5].same_as, None);
}

/// Where the follow-up asks anyway, a scene of the very words of an
/// earlier one drawn anew is asked whether it shows that one again; not
/// asked for itself alone.
#[test]
fn a_chorus_drawn_twice_is_asked_whether_it_shows_the_first_again() {
    let twice = bad(
        "{\"first_line\": 10, \"section\": \"chorus\", \"same_as\": 4}",
        "{\"first_line\": 10, \"section\": \"chorus\", \"shot\": \"wide\", \"shows\": \"The sea again.\", \"prompt\": \"A dark sea at night under a thin moon, a boat far out, waves in front\"}",
    );
    assert_eq!(
        asked(&twice),
        [What::Reuse {
            scene: 5,
            earlier: 2
        }],
        "asked by itself"
    );
    let both = twice.replacen("\"style\": \"watercolor\"", "\"style\": \"fresco\"", 1);
    assert_eq!(
        asked(&both),
        [
            What::Style,
            What::Reuse {
                scene: 5,
                earlier: 2
            }
        ]
    );
    let (plan, notes) = settled(
        reading(&both),
        &[Value::from("watercolour"), Value::from(true)],
    );
    assert_eq!(plan.scenes[5].same_as, Some(2));
    assert_eq!(plan.art.style, Style::Watercolor);
    assert!(
        says(&notes, "the style \"watercolor\", the nearest"),
        "{notes:?}"
    );
    let (plan, _) = settled(
        reading(&both),
        &[Value::from("ink wash"), Value::from(false)],
    );
    assert_eq!(
        (plan.scenes[5].same_as, plan.art.style),
        (None, Style::InkWash)
    );
}

/// A name no one knows (the style, the preset, a pass to add) and a
/// prompt asking for writing are asked again, their choices listed; an
/// answer that is a valid choice is taken, any other the nearest.
#[test]
fn what_needs_judgement_is_asked_and_the_patch_merged() {
    let wrong = bad("\"style\": \"watercolor\"", "\"style\": \"fresco\"")
        .replacen("\"preset\": \"folk\"", "\"preset\": \"folk-forever\"", 1)
        .replacen(
            "\"add\": [\"grain\"]",
            "\"add\": [\"grain\", \"glitchy\"]",
            1,
        )
        .replacen("Lonely lighthouse", "Lonely lighthouse with a poster", 1);
    let read = reading(&wrong);
    let questions = read.questions();
    let whats: Vec<&What> = questions.iter().map(|q| &q.what).collect();
    assert_eq!(
        whats,
        [
            &What::Prompt { scene: 4 },
            &What::Style,
            &What::Preset,
            &What::Pass { index: 1 }
        ]
    );
    let words = super::follow::ask(&questions);
    assert!(words.contains("1. Scene 5 of your plan (\"An empty lighthouse turns.\") has a picture prompt that asks for writing in the picture (\"poster\")"), "{words}");
    assert!(
        words.contains("\"watercolor\", \"gouache painting\""),
        "{words}"
    );
    assert!(words.contains("One of \"folk\""), "{words}");
    let patch = "{\"1\": \"Lonely lighthouse on a rocky point at night, its beam over the water\", \"2\": \"ink wash\", \"3\": \"rock\", \"4\": \"glow\"}";
    let mut notes = Vec::new();
    let answers = super::follow::answers(patch, questions.len(), &mut notes);
    let values: Vec<Value> = answers.into_iter().map(Option::unwrap).collect();
    let (plan, notes) = settled(read, &values);
    assert_eq!(
        plan.scenes[4].prompt,
        "Lonely lighthouse on a rocky point at night, its beam over the water"
    );
    assert_eq!(plan.art.style, Style::InkWash);
    assert_eq!(plan.look.id, "rock+song");
    assert!(
        says(&notes, "question 4: the pass to add \"glow\""),
        "{notes:?}"
    );
    // The patch wrong too: the nearest of each.
    let (plan, notes) = settled(
        reading(&wrong),
        &[
            Value::from("a sign with a poster"),
            Value::from(3),
            Value::Null,
        ],
    );
    assert_eq!(plan.scenes[4].prompt, "Lonely lighthouse with a on a rocky point at night, its beam sweeping over empty water, rocks in the foreground, clouds drifting behind");
    assert!(text_word(&plan.scenes[4].prompt).is_none());
    assert_eq!(plan.look.id, "folk+song", "{notes:?}");
    assert!(
        says(&notes, "the pass \"glitchy\" taken as \"glitch\""),
        "{notes:?}"
    );
    assert!(plan.look.faults().is_empty());
}

/// An answer cut off keeps every whole scene (the one cut off left out);
/// the art and shaders it never wrote are asked (else their nearest), and
/// the last scene kept shows on to the song's end.
#[test]
fn an_answer_cut_off_keeps_every_whole_scene() {
    let cut = &GOOD[..GOOD.find("Close view of father").unwrap()];
    let read = reading(cut);
    assert!(says(&read.notes, "its end cut off"), "{:?}", read.notes);
    assert_eq!(
        read.questions()
            .iter()
            .map(|q| q.what.clone())
            .collect::<Vec<_>>(),
        [What::Style, What::Preset]
    );
    let (plan, notes) = mended(cut);
    assert_eq!(plan.scenes.len(), 3);
    assert_eq!((plan.scenes[2].last_line, plan.scenes[2].to), (11, 120.0));
    assert_eq!(plan.story, STORY);
    assert!(says(&notes, "question 1: the style"), "{notes:?}");
    assert!(plan.look.faults().is_empty());
}

/// The words that ask for writing are found as whole words only, and no
/// art style asks for any.
#[test]
fn no_style_asks_for_writing() {
    assert_eq!(
        text_word("a fountain by the postern gate, its texture worn"),
        None
    );
    assert_eq!(text_word("a torn Poster on the door"), Some("Poster"));
    for s in Style::ALL {
        assert_eq!(text_word(s.words()), None, "{}", s.name());
        assert!(
            !s.words().contains("poster") && !s.name().contains("poster"),
            "{}",
            s.name()
        );
        let read: Style = serde_json::from_str(&format!("\"{}\"", s.name())).unwrap();
        assert_eq!(read, *s);
    }
}

/// The prompt, its catalogue left out, is as the snapshot has it
/// (`tests/snapshots/story_prompt.txt`; `UPDATE_SNAPSHOTS=1` writes it
/// anew); the same song always gives the same prompt; it says every count
/// an answer is held to, but never how many scenes (nor the most), and its
/// example has no narrator.
#[test]
fn the_prompt_is_as_its_snapshot_and_says_every_count() {
    let song = song();
    let asked = prompt(&song);
    assert_eq!(asked, prompt(&song), "the same prompt every time");
    let brief = jodel_looks::catalogue::brief();
    assert!(asked.contains(brief));
    for count in COUNTS {
        assert!(asked.contains(&count.said()), "{count:?}");
    }
    for most in [
        format!("{MOST_SCENES} scenes"),
        format!("most {MOST_SCENES}"),
    ] {
        assert!(!asked.contains(&most), "the most scenes is never said");
    }
    assert!(!asked.contains("narrator"));
    let shown = asked.replace(brief, "<the catalogue>\n");
    let file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/story_prompt.txt");
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        durable::write_atomic(&file, shown.as_bytes()).unwrap();
    }
    let kept = std::fs::read_to_string(&file).unwrap_or_default();
    assert!(
        kept == shown,
        "the story's prompt changed (UPDATE_SNAPSHOTS=1 to keep it):\n{shown}"
    );
}

/// A name is read as one: lower case, an underscore a space, its first
/// two words; a group is told apart; the nearest of names is the same
/// but for case and marks first, then one holding the other, then the
/// fewest changes.
#[test]
fn names_are_read_and_the_nearest_found() {
    assert_eq!(rules::name(" White_Man "), "white man");
    assert_eq!(rules::name("the old fisherman"), "the old");
    assert_eq!(rules::name("Zoë's dog!"), "zoë's dog");
    assert!(rules::is_group("the crowd") && rules::is_group("street-dancers"));
    assert!(rules::is_group("dancers") && !rules::is_group("father"));
    let styles = ["watercolor", "ink wash", "oil painting"];
    assert_eq!(rules::nearest("Ink-Wash", styles), Some("ink wash"));
    assert_eq!(rules::nearest("oil", styles), Some("oil painting"));
    assert_eq!(rules::nearest("watercolour", styles), Some("watercolor"));
    assert_eq!(rules::nearest("x", [] as [&str; 0]), None);
}

/// The prose of the good answer streamed in pieces of any size is the
/// same: the story, every character with its looks, what every scene
/// shows, the art's style and why, and nothing else.
#[test]
fn the_prose_of_an_answer_streams_out_of_its_json() {
    let plan = planned(GOOD);
    let mut want = format!("{STORY}\n\n");
    for c in &plan.characters {
        want.push_str(&format!("{}: {}\n", c.name, c.looks));
    }
    for s in plan.scenes.iter().filter(|s| !s.shows.is_empty()) {
        want.push_str(&format!("· {}\n", s.shows));
    }
    want.push_str(&format!(
        "\nThe look: {}\n{}\n",
        plan.art.style.name(),
        plan.art.why
    ));
    let chars: Vec<char> = GOOD.chars().collect();
    for size in [1, 2, 3, 7, 64, chars.len()] {
        let mut prose = Prose::new();
        let got: String = (chars.chunks(size))
            .map(|c| prose.feed(&c.iter().collect::<String>()))
            .collect();
        assert_eq!(got, want, "pieces of {size}");
    }
}

/// Escapes are read whole however the pieces split them, a line break as
/// a space; a key of the same name elsewhere is not shown.
#[test]
fn the_prose_reads_escapes_and_only_its_fields() {
    let answer = r##"{"story": "She said \"go\"\nand left été 🚗.", "shader": {"name": "not shown", "knobs": {"why": 1}},
 "art": {"style": "watercolor", "palette": ["#000000"], "why": "Soft."}}"##;
    let want = "She said \"go\" and left été 🚗.\n\n\nThe look: watercolor\nSoft.\n";
    // Half a pair alone is dropped, and the string still ends at its quote.
    let lone = r#"{"story": "a\ud83d", "art": {"why": "b\ud83dx\ud83d\n"}}"#;
    let mut prose = Prose::new();
    assert_eq!(prose.feed(lone), "a\n\nbx \n");
    for size in [1, 2, 5, answer.len()] {
        let mut prose = Prose::new();
        let chars: Vec<char> = answer.chars().collect();
        let got: String = (chars.chunks(size))
            .map(|c| prose.feed(&c.iter().collect::<String>()))
            .collect();
        assert_eq!(got, want, "pieces of {size}");
    }
}

/// The shaders make a look of their preset: the pass added after its
/// passes, the knobs moved, the music moving what was asked on top of the
/// pass's own drives (a mix too), its mood among the tags; checked,
/// nothing of it is held or dropped.
#[test]
fn the_shaders_make_a_valid_look() {
    // "canvas" is the preset's already: it stays where it is.
    let answer = bad("\"add\": [\"grain\"]", "\"add\": [\"canvas\", \"grain\"]").replacen(
        "\"by\": \"chorus\", \"amount\": 0.6}]",
        "\"by\": \"chorus\", \"amount\": 0.6}, {\"pass\": \"grain\", \"knob\": \"mix\", \"by\": \"quiet\", \"amount\": -0.5}]",
        1,
    );
    let (plan, notes) = mended(&answer);
    assert_eq!(
        notes,
        ["read: scene 5 shows scene 2 again (same_as 4 read as a line of it)"]
    );
    let look = plan.look;
    assert_eq!(look.id, "folk+song");
    assert!(look
        .tags
        .ends_with(&["longing".to_owned(), "rain".to_owned()]));
    let passes: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
    assert_eq!(passes, ["watercolor", "canvas", "liquid", "grain"]);
    let layer = |pass: &str| look.layers.iter().find(|l| l.pass == pass).unwrap();
    assert_eq!(layer("canvas").knobs.get("amount"), Some(&0.5));
    assert_eq!(layer("grain").knobs.get("size"), Some(&2.0));
    let liquid = layer("liquid").drive.as_ref().expect("asked");
    let amount: Vec<_> = liquid.iter().filter(|d| d.knob == "amount").collect();
    assert_eq!(
        amount.len(),
        1,
        "its own amount driver replaced: {liquid:?}"
    );
    assert_eq!(
        (amount[0].by, amount[0].amount),
        (jodel_looks::Driver::Chorus, 0.6)
    );
    let grain = layer("grain").drive.as_ref().expect("asked");
    assert!(grain
        .iter()
        .any(|d| d.knob == "mix" && d.by == jodel_looks::Driver::Quiet));
    let folk = jodel_looks::presets::named("folk").unwrap();
    assert_eq!(layer("canvas").drive, folk.layers[1].drive, "not asked");
    assert!(look.faults().is_empty());
    let checked = look.check();
    assert!(checked.notes.is_empty(), "{:?}", checked.notes);
    assert_eq!(checked.layers.len(), look.layers.len());
}

/// A shot is told to the image model by the name the model writes it
/// with.
#[test]
fn a_shot_is_named_as_the_model_writes_it() {
    for shot in [Shot::Wide, Shot::Medium, Shot::CloseUp] {
        let written = serde_json::to_string(&shot).unwrap();
        assert_eq!(written, format!("\"{}\"", shot.name()));
    }
}

/// Shaders over the budget are taken without their dearest passes (as the
/// catalogue tells the model), none of them kept: what went cost more
/// than anything that stayed; the look within the budget, nothing more
/// dropped.
#[test]
fn shaders_over_the_budget_lose_their_dearest_passes() {
    let asked = [
        "watercolor",
        "canvas",
        "liquid",
        "grain",
        "glow",
        "halation",
    ];
    let answer = bad(
        "\"add\": [\"grain\"]",
        "\"add\": [\"grain\", \"glow\", \"halation\"]",
    );
    let (plan, notes) = mended(&answer);
    assert!(
        says(&notes, "over the budget: 4 of 6 passes kept"),
        "{notes:?}"
    );
    let look = plan.look;
    let kept: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
    assert_eq!(kept, ["watercolor", "canvas", "liquid", "grain"]);
    let cost = |p: &str| jodel_looks::passes::named(p).unwrap().cost;
    let gone = asked.iter().filter(|p| !kept.contains(p));
    for g in gone {
        assert!(kept.iter().all(|&k| cost(g) >= cost(k)), "{g}");
    }
    assert!(look.faults().is_empty(), "{:?}", look.faults());
    assert!(look.check().dropped.is_empty());
}

/// A song from a file's tags and its lyrics: the title, artist and album;
/// each telling tag once (no technical ones, none empty); its length;
/// its sung lines only (no break, no backing line, none untimed), each
/// from its start to its last word's end, else its source's end, else
/// the next line's start, else a while after it.
#[test]
fn a_song_is_made_of_its_tags_and_sung_lines() {
    use jodel_lyrics::{Line as Sung, Lyrics, Source, Synced, Word};
    let timed = |text: &str, start: f64, end: f64| Word {
        start: Some(start),
        end: Some(end),
        ..Word::new(text)
    };
    let words = |text: &str| text.split(' ').map(Word::new).collect::<Vec<_>>();
    let mut backing = Sung::from_text("(ooh ooh)");
    backing.time = Some(14.0);
    let lyrics = Lyrics {
        lines: vec![
            Sung {
                words: vec![timed("I", 10.0, 10.4), timed("walked", 10.5, 12.0)],
                ..Sung::default()
            },
            Sung {
                is_break: true,
                time: Some(13.0),
                ..Sung::default()
            },
            backing,
            Sung {
                words: words("Carry me over"),
                synced: Some(Synced {
                    start: 20.0,
                    end: Some(24.0),
                }),
                ..Sung::default()
            },
            Sung {
                words: words("never timed"),
                ..Sung::default()
            },
            Sung {
                words: words("Carry me home"),
                time: Some(26.0),
                ..Sung::default()
            },
            Sung {
                words: words("the end"),
                time: Some(30.0),
                ..Sung::default()
            },
        ],
        source: Source::Lrclib,
        timing: Default::default(),
    };
    let info = jodel_playback::TrackInfo {
        path: "/music/harbour.flac".into(),
        container: jodel_playback::Container::Flac,
        codec: jodel_playback::Codec::Flac,
        rate: 48_000,
        channels: 2,
        bits: 16,
        float: false,
        bitrate: None,
        frames: Some(48_000 * 120),
        tags: jodel_playback::Tags {
            title: Some("Harbour Song".into()),
            artist: Some("The Example Band".into()),
            album: None,
            track_number: None,
            all: [
                ("ENCODER", "x 1.0"),
                ("DATE", "1994"),
                ("date", "2000"),
                ("Genre", "folk"),
                ("comment", " "),
            ]
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .into(),
        },
        cover: None,
    };
    let song = Song::of(&info, &lyrics);
    assert_eq!(
        (
            song.title.as_str(),
            song.artist.as_str(),
            song.album.as_str()
        ),
        ("Harbour Song", "The Example Band", "")
    );
    assert_eq!(
        song.tags,
        [
            ("date".to_owned(), "1994".to_owned()),
            ("genre".to_owned(), "folk".to_owned())
        ]
    );
    assert_eq!(song.duration, 120.0);
    let lines: Vec<(f64, f64, &str)> = (song.lines.iter())
        .map(|l| (l.start, l.end, l.text.as_str()))
        .collect();
    assert_eq!(
        lines,
        [
            (10.0, 12.0, "I walked"),
            (20.0, 24.0, "Carry me over"),
            (26.0, 30.0, "Carry me home"),
            (30.0, 30.0 + rules::UNTIMED_LINE_SECONDS, "the end"),
        ]
    );
}
