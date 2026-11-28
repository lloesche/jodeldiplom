//! The plan's conversations with writers that answer as told (taken at
//! once, a follow-up asked in the same conversation and its patch merged,
//! a patch wrong too, refused, cut off, failed), what is kept of them and
//! how, and the whole job on the model, end to end, where it is here.

use std::cell::RefCell;

use super::*;
use crate::core::story::tests::{song, GOOD};

/// The context the tests' plans are written in.
const CONTEXT: usize = 32_752;

/// A writer that answers each turn with the next of its answers (said to
/// think first where the answer has a thought), its text streamed in
/// pieces of seven characters, and keeps each turn's words with its
/// conversation's number and its rooms.
struct Fake {
    answers: RefCell<Vec<Result<Written, Failure>>>,
    asked: RefCell<Vec<Turn>>,
    talks: RefCell<usize>,
}

/// A turn a fake writer was told: its conversation, its words, its rooms.
#[derive(Clone, Debug, PartialEq)]
struct Turn {
    talk: usize,
    words: String,
    rooms: (usize, usize),
}

impl Fake {
    fn new(answers: Vec<Result<Written, Failure>>) -> Fake {
        Fake {
            answers: RefCell::new(answers),
            asked: RefCell::new(Vec::new()),
            talks: RefCell::new(0),
        }
    }

    /// The words of every turn, in order.
    fn words(&self) -> Vec<String> {
        self.asked
            .borrow()
            .iter()
            .map(|t| t.words.clone())
            .collect()
    }
}

/// A conversation of a [`Fake`].
struct FakeTalk<'a> {
    fake: &'a Fake,
    talk: usize,
}

impl Writer for Fake {
    fn converse(&self, _context: usize) -> Box<dyn Conversation + '_> {
        let mut talks = self.talks.borrow_mut();
        *talks += 1;
        Box::new(FakeTalk {
            fake: self,
            talk: *talks,
        })
    }
}

impl Conversation for FakeTalk<'_> {
    fn say(
        &mut self,
        words: &str,
        rooms: (usize, usize),
        seen: &mut dyn FnMut(Writing),
    ) -> Result<Written, Failure> {
        self.fake.asked.borrow_mut().push(Turn {
            talk: self.talk,
            words: words.to_owned(),
            rooms,
        });
        let answer = self.fake.answers.borrow_mut().remove(0)?;
        if answer.thought_tokens > 0 {
            seen(Writing::Thinking);
        }
        let chars: Vec<char> = answer.text.chars().collect();
        for piece in chars.chunks(7) {
            seen(Writing::Answer(&piece.iter().collect::<String>()));
        }
        Ok(answer)
    }
}

/// An answer that ended by itself.
fn says(text: &str) -> Result<Written, Failure> {
    Ok(Written {
        text: text.to_owned(),
        stopped: true,
        ..Written::default()
    })
}

/// [`GOOD`] with its style one no one knows and the pass it adds
/// misspelt: two questions for the follow-up.
fn unsure() -> String {
    GOOD.replacen("\"style\": \"watercolor\"", "\"style\": \"fresco\"", 1)
        .replacen("\"add\": [\"grain\"]", "\"add\": [\"grains\"]", 1)
}

/// The plan `fake` writes for [`song`], and what it told (its thinking
/// as `[thinking]`).
fn planned_by(fake: &Fake) -> (Planned, String) {
    let song = song();
    let mut told = String::new();
    let planned = plan(
        fake,
        &song,
        (&prompt::prompt(&song), CONTEXT),
        &mut |story| match story {
            Story::Words(words) => told.push_str(&words),
            Story::Thinking => told.push_str("[thinking]"),
            _ => {}
        },
    );
    (planned, told)
}

/// The model thinks before it writes: "thinking…" told first, the
/// thought not in the answer read.
#[test]
fn a_thought_is_told_before_the_plan() {
    let thought = Written {
        thought_tokens: 900,
        ..says(GOOD).unwrap()
    };
    let (planned, told) = planned_by(&Fake::new(vec![Ok(thought)]));
    assert!(planned.plan.is_ok());
    assert!(told.starts_with("[thinking]A woman remembers"), "{told}");
}

/// A thought that never ends is cut off: where the context filled its
/// room before its most, no story in this context (short of room); not
/// asked again (the model writes greedily: it would think as long again).
/// At its most, or ended within itself on a stop token, cut off for good.
#[test]
fn a_thought_that_never_ends_is_cut_off() {
    let endless = Written {
        thinking: true,
        thought_tokens: 3_000,
        thought_room: 3_000,
        ..Written::default()
    };
    let (planned, told) = planned_by(&Fake::new(vec![Ok(endless)]));
    let Err(Failure::Short(why)) = planned.plan else {
        panic!("cut off: {:?}", planned.plan);
    };
    assert_eq!(
        why,
        "the model's thought did not end (cut off after 3000 tokens)"
    );
    assert_eq!(planned.tries.len(), 1, "not asked again");
    assert!(told.starts_with("[thinking]") && told.contains("No story for this song"));
    // At its most, cut off for good (a larger context gives it no more).
    let runaway = Written {
        thinking: true,
        thought_tokens: MOST_THOUGHT,
        thought_room: MOST_THOUGHT,
        ..Written::default()
    };
    let (planned, _) = planned_by(&Fake::new(vec![Ok(runaway)]));
    let Err(Failure::CutOff(why)) = planned.plan else {
        panic!("cut off: {:?}", planned.plan);
    };
    assert_eq!(
        why,
        format!("the model's thought did not end (cut off after {MOST_THOUGHT} tokens)")
    );
    assert_eq!(planned.tries.len(), 1, "not asked again");
    // Ended on a stop token within its room: for good too.
    let stopped = Written {
        thinking: true,
        thought_tokens: 100,
        thought_room: 3_000,
        ..Written::default()
    };
    let (planned, _) = planned_by(&Fake::new(vec![Ok(stopped)]));
    assert!(matches!(planned.plan, Err(Failure::CutOff(_))));
}

/// A good answer is taken at once, its prose told as it came, the model
/// asked nothing more.
#[test]
fn a_good_answer_is_taken_at_once() {
    let fake = Fake::new(vec![says(GOOD)]);
    let (planned, told) = planned_by(&fake);
    assert!(planned.plan.is_ok());
    assert_eq!(planned.tries.len(), 1);
    assert_eq!(
        planned.notes,
        ["read: scene 5 shows scene 2 again (same_as 4 read as a line of it)"]
    );
    assert!(told.starts_with("A woman remembers") && told.contains("father: Man, 60s"));
    assert_eq!(fake.asked.borrow()[0].rooms, (MOST_THOUGHT, MOST_ANSWER));
}

/// What needs judgement goes back to the model once, in the same
/// conversation: only what is wrong, its choices listed, a patch the
/// answer; the patch merged into the plan, the rest of it as it was; the
/// follow-up and every mending in the notes (for the trace).
#[test]
fn what_needs_judgement_is_asked_in_the_same_conversation() {
    let patch = "```json\n{\"1\": \"ink wash\", \"2\": \"grain\"}\n```";
    let fake = Fake::new(vec![says(&unsure()), says(patch)]);
    let (planned, told) = planned_by(&fake);
    let plan = planned.plan.expect("a plan");
    assert_eq!(plan.art.style, crate::core::story::Style::InkWash);
    assert!(plan.look.layers.iter().any(|l| l.pass == "grain"));
    assert_eq!(plan.scenes.len(), 6, "the rest as it was");
    let asked = fake.asked.borrow();
    assert_eq!(asked.len(), 2);
    assert_eq!(asked[1].talk, asked[0].talk, "the same conversation");
    assert_eq!(asked[1].rooms, (FOLLOW_UP_THOUGHT, MOST_PATCH));
    let follow_up = &asked[1].words;
    assert!(
        follow_up.contains("1. The art style \"fresco\" is not one of the styles"),
        "{follow_up}"
    );
    assert!(
        follow_up.contains(
            "2. The pass \"grains\" to add is not in the catalogue: which is it? One of \"grain\""
        ),
        "{follow_up}"
    );
    assert!(!follow_up.contains("You direct"), "only what is wrong");
    assert_eq!(
        planned.tries[1].follow_up.as_deref(),
        Some(follow_up.as_str())
    );
    assert!(
        planned
            .notes
            .iter()
            .any(|n| n.contains("question 1: the style \"ink wash\", as answered")),
        "{:?}",
        planned.notes
    );
    assert!(
        told.contains("(Settling 2 things with the model.)"),
        "{told}"
    );
    assert!(!told.contains("ink wash"), "the patch is not prose");
}

/// A patch that is wrong too, a follow-up that fails to end its thought,
/// one with no room in the context, or a model that fails in it: the
/// nearest valid choices taken, the plan kept (the first answer is not
/// thrown away). Where the context left the follow-up short of room, the
/// plan says so (kept with that context: planned again in a larger
/// one).
#[test]
fn a_patch_wrong_too_takes_the_nearest_valid_choice() {
    let wrong = "{\"1\": \"pastel crayon\", \"2\": 7}";
    let unended = Written {
        thinking: true,
        ..Written::default()
    };
    for (second, cramped) in [
        (says(wrong), false),
        (Ok(unended), true),
        (Err(Failure::Short("no room".into())), true),
        (Err(Failure::Model("device lost".into())), false),
    ] {
        let fake = Fake::new(vec![says(&unsure()), second]);
        let (planned, _) = planned_by(&fake);
        assert_eq!(planned.cramped, cramped);
        let plan = planned.plan.expect("a plan");
        assert!(
            plan.look.layers.iter().any(|l| l.pass == "grain"),
            "grains: the nearest"
        );
        assert!(plan.look.faults().is_empty());
        assert!(
            planned.notes.iter().any(|n| n.contains("question 2")),
            "{:?}",
            planned.notes
        );
    }
}

/// An answer cut off at its room keeps every whole scene, the art and
/// shaders it never wrote asked in the follow-up; only one with nothing to
/// draw is refused (for good: kept as no story), and a song is never
/// asked twice from the start.
#[test]
fn only_an_answer_with_nothing_to_draw_is_refused() {
    let cut = Written {
        text: GOOD[..GOOD.find("Close view of father").unwrap()].to_owned(),
        stopped: false,
        ..Written::default()
    };
    let patch = "{\"1\": \"watercolor\", \"2\": \"folk\"}";
    let fake = Fake::new(vec![Ok(cut), says(patch)]);
    let (planned, _) = planned_by(&fake);
    let plan = planned.plan.expect("a plan");
    assert_eq!(plan.scenes.len(), 3);
    assert_eq!(plan.look.id, "folk+song");
    assert!(planned.notes[0].contains(&format!("cut off at {MOST_ANSWER} tokens")));
    let fake = Fake::new(vec![says("I would rather not say.")]);
    let (planned, told) = planned_by(&fake);
    let Err(Failure::Refused(why)) = &planned.plan else {
        panic!("refused: {:?}", planned.plan);
    };
    assert!(why.starts_with(check::NOTHING), "{why}");
    assert_eq!(fake.asked.borrow().len(), 1, "not asked again");
    assert!(told.ends_with(&format!("\nNo story for this song: {why}\n")));
    for failure in [
        Failure::Short("too long".into()),
        Failure::Model("device lost".into()),
    ] {
        let fake = Fake::new(vec![Err(failure.clone())]);
        let (planned, _) = planned_by(&fake);
        assert_eq!(planned.plan, Err(failure));
        assert!(planned.tries.is_empty());
    }
}

/// A song with no timed sung lines is not asked at all: it has no story,
/// and the writer is never called (it has no answer to give).
#[test]
fn a_song_with_no_lines_is_not_asked() {
    let mut silent = song();
    silent.lines.clear();
    let fake = Fake::new(vec![]);
    let mut told = String::new();
    let asked = prompt::prompt(&silent);
    let planned = plan(&fake, &silent, (&asked, CONTEXT), &mut |story| {
        if let Story::Words(words) = story {
            told.push_str(&words);
        }
    });
    assert_eq!(planned.plan, Err(Failure::Refused(NO_LINES.into())));
    assert!(planned.tries.is_empty() && fake.words().is_empty());
    assert!(told.contains(NO_LINES));
}

/// What is kept is read back the same, a plan or that there is none; one
/// made by another version or for another ask is not read; one short of
/// room with the context it had.
#[test]
fn a_plan_is_kept_with_what_made_it() {
    let plan = crate::core::story::tests::planned(GOOD);
    let asked = "0123456789abcdef";
    let text = to_text(
        asked,
        &Kept::Plan {
            plan: Box::new(plan.clone()),
            settled_short_at: None,
        },
    );
    assert!(text.starts_with("jodeldiplom-cache story-plan 3 by "));
    assert_eq!(
        from_text(&text, asked),
        Some(Kept::Plan {
            plan: Box::new(plan.clone()),
            settled_short_at: None
        })
    );
    // Settled short of room: with the context it was settled in.
    let settled = Kept::Plan {
        plan: Box::new(plan),
        settled_short_at: Some(CONTEXT),
    };
    assert_eq!(from_text(&to_text(asked, &settled), asked), Some(settled));
    assert_eq!(from_text(&text, "fedcba9876543210"), None, "another ask");
    let older = text.replace(&format!("{MADE} {VERSION}"), &format!("{MADE} 0"));
    assert_eq!(from_text(&older, asked), None, "another maker");
    let none = to_text(asked, &Kept::NoStory("too few\nscenes".into()));
    assert_eq!(
        from_text(&none, asked),
        Some(Kept::NoStory("too few scenes".into()))
    );
    let short = Kept::Short {
        context: CONTEXT,
        why: "the thought did not end".into(),
    };
    assert_eq!(from_text(&to_text(asked, &short), asked), Some(short));
}

/// A job runs its plan, tells it and keeps it, found again for the song
/// as it is asked (other words: planned again), whatever context wrote
/// it; a song that has no story is kept as such (nothing to draw, its
/// thought at its most too), one short of room with the context it had; a
/// model that failed is not kept.
#[test]
fn a_job_keeps_what_it_planned_for_the_song_as_asked() {
    let dir = jodel_testkit::TempDir::new("story");
    let file = dir.join("song.flac");
    durable::write_atomic(&file, b"not really a song").unwrap();
    let mut told = Vec::new();
    let tell = &mut |story: Story| told.push(story);
    let job = StoryJob::new(file.clone(), song(), dir.join("stories"));
    assert_eq!(job.kept(), None, "none kept yet");
    let failed = Fake::new(vec![Err(Failure::Model("device lost".into()))]);
    assert!(matches!(
        job.run(&failed, CONTEXT, tell),
        Err(Failure::Model(_))
    ));
    assert_eq!(job.kept(), None, "a model's failure is not kept");
    let plan = job
        .run(&Fake::new(vec![says(GOOD)]), CONTEXT, tell)
        .unwrap();
    assert_eq!(
        job.kept(),
        Some(Kept::Plan {
            plan: Box::new(plan),
            settled_short_at: None
        })
    );
    let mut other = song();
    other.lines[0].text = "I ran down to the harbour".into();
    let other = StoryJob::new(file.clone(), other, dir.join("stories"));
    assert_eq!(other.kept(), None, "other words, another ask");
    let refused = Fake::new(vec![says("no plan")]);
    assert!(matches!(
        other.run(&refused, CONTEXT, tell),
        Err(Failure::Refused(_))
    ));
    assert!(told.iter().any(|s| matches!(s, Story::Words(_))));
    assert!(matches!(other.kept(), Some(Kept::NoStory(_))));
    // A thought cut off by the context: kept with that context.
    let job = |line: &str| {
        let mut song = song();
        song.lines[0].text = line.into();
        StoryJob::new(file.clone(), song, dir.join("stories"))
    };
    let endless = |tokens| Written {
        thinking: true,
        thought_tokens: tokens,
        thought_room: tokens,
        ..Written::default()
    };
    let fourth = job("I flew over the harbour");
    let short = fourth.run(&Fake::new(vec![Ok(endless(3_000))]), CONTEXT, &mut |_| {});
    assert!(matches!(short, Err(Failure::Short(_))));
    assert!(matches!(
        fourth.kept(),
        Some(Kept::Short {
            context: CONTEXT,
            ..
        })
    ));
    // A thought that runs to its most: none, whatever the context.
    let fifth = job("I sailed out of the harbour");
    let cut = fifth.run(
        &Fake::new(vec![Ok(endless(MOST_THOUGHT))]),
        CONTEXT,
        &mut |_| {},
    );
    assert!(matches!(cut, Err(Failure::CutOff(_))));
    assert!(matches!(fifth.kept(), Some(Kept::NoStory(_))));
}

/// The whole job on the model, end to end: `JODEL_GEMMA` names the
/// model's file and `JODEL_STORY_SONG` a song as the plan is made from it
/// (a [`Song`] as JSON); skipped where either is not there. The song is
/// planned twice, the same answers both times (greedy), the plan read and
/// its look checked whole; the plan, the answers (the follow-up's too),
/// what was mended and the times go to the folder `JODEL_STORY_OUT`
/// names, where it names one.
#[test]
#[ignore = "runs the language model: JODEL_GEMMA, JODEL_STORY_SONG"]
fn a_song_is_planned_on_the_model() {
    let (Some(model), Some(input)) = (
        std::env::var_os("JODEL_GEMMA").map(PathBuf::from),
        std::env::var_os("JODEL_STORY_SONG").map(PathBuf::from),
    ) else {
        eprintln!("skipped: JODEL_GEMMA and JODEL_STORY_SONG name the model and a song");
        return;
    };
    if !model.exists() {
        eprintln!("skipped: no model at {}", model.display());
        return;
    }
    let song: Song =
        serde_json::from_str(&std::fs::read_to_string(&input).expect("the song")).expect("a song");
    let asked = prompt::prompt(&song);
    let started = std::time::Instant::now();
    let gemma = Gemma::open(&model, jodel_gpu::Precision::Fast).expect("the model");
    let loaded = started.elapsed();
    let mut runs = Vec::new();
    for _ in 0..2 {
        let started = std::time::Instant::now();
        let mut prose = String::new();
        let context = context(&model).expect("its context");
        let planned = plan(&gemma, &song, (&asked, context), &mut |story| {
            if let Story::Words(words) = story {
                prose.push_str(&words);
            }
        });
        runs.push((planned, prose, started.elapsed()));
    }
    let (first, prose, took) = &runs[0];
    let mut report = format!(
        "model loaded in {:.1} s on {}; GPU memory held {:.0} MB\n",
        loaded.as_secs_f64(),
        gemma.gpu_name(),
        gemma.held().total() as f64 / f64::from(1 << 20)
    );
    for (k, (planned, _, took)) in runs.iter().enumerate() {
        report.push_str(&format!("run {k}: {:.1} s\n", took.as_secs_f64()));
        for t in &planned.tries {
            let w = &t.written;
            report.push_str(&format!(
                "  {}prompt {} tokens in {:.2} s; thought {}; answer {} tokens in {:.1} s ({:.0} a second){}\n",
                if t.follow_up.is_some() { "follow-up: " } else { "" },
                w.prompt_tokens,
                w.prompt_time.as_secs_f64(),
                w.thought_tokens,
                w.answer_tokens,
                w.answer_time.as_secs_f64(),
                w.answer_tokens as f64 / w.answer_time.as_secs_f64().max(1e-9),
                if w.stopped { "" } else { "; cut off" },
            ));
        }
    }
    eprintln!("{report}");
    if let Some(out) = std::env::var_os("JODEL_STORY_OUT").map(PathBuf::from) {
        let put = |name: &str, text: &str| {
            durable::write_atomic(&out.join(name), text.as_bytes()).unwrap();
        };
        put("timings.txt", &report);
        put("prose.txt", prose);
        for (k, t) in first.tries.iter().enumerate() {
            put(&format!("answer-{k}.txt"), &t.written.text);
            if let Some(asked) = &t.follow_up {
                put(&format!("follow-up-{k}.txt"), asked);
            }
        }
        put("mended.txt", &first.notes.join("\n"));
        if let Ok(plan) = &first.plan {
            put("plan.json", &plan.to_json());
        }
    }
    let answers =
        |p: &Planned| -> Vec<String> { p.tries.iter().map(|t| t.written.text.clone()).collect() };
    assert_eq!(
        answers(first),
        answers(&runs[1].0),
        "the same song, the same plan"
    );
    let plan = first.plan.as_ref().expect("a plan");
    assert!(!prose.is_empty() && took.as_secs() < 600);
    assert!(plan.look.faults().is_empty(), "{:?}", plan.look.faults());
    assert!(plan.look.check().notes.is_empty());
}
