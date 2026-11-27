//! The plan's tries with writers that answer as told (taken, asked again,
//! refused, cut off, failed), what is kept of them and how, and the whole
//! job on the model, end to end, where it is here.

use std::cell::RefCell;

use super::*;
use crate::core::story::tests::{song, GOOD};

/// The context the tests' plans are written in.
const CONTEXT: usize = 32_752;

/// A writer that answers each prompt with the next of its answers (said
/// to think first where the answer has a thought), its text streamed in
/// pieces of seven characters, and keeps the prompts.
struct Fake {
    answers: RefCell<Vec<Result<Written, Failure>>>,
    asked: RefCell<Vec<String>>,
}

impl Fake {
    fn new(answers: Vec<Result<Written, Failure>>) -> Fake {
        Fake {
            answers: RefCell::new(answers),
            asked: RefCell::new(Vec::new()),
        }
    }
}

impl Writer for Fake {
    fn write(
        &self,
        prompt: &str,
        _room: (usize, usize),
        seen: &mut dyn FnMut(Writing),
    ) -> Result<Written, Failure> {
        self.asked.borrow_mut().push(prompt.to_owned());
        let answer = self.answers.borrow_mut().remove(0)?;
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

/// [`GOOD`] with a line of the lyrics in no scene.
fn broken() -> String {
    GOOD.replacen(
        "\"first_line\": 10, \"last_line\": 11",
        "\"first_line\": 10, \"last_line\": 10",
        1,
    )
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

/// A thought that never ends is cut off: no story in this context (short
/// of room), not asked again (the model writes greedily: it would think
/// as long again).
#[test]
fn a_thought_that_never_ends_is_cut_off() {
    let endless = Written {
        thinking: true,
        thought_tokens: 3_000,
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
}

/// A good answer is taken at once, its prose told as it came.
#[test]
fn a_good_answer_is_taken_at_once() {
    let fake = Fake::new(vec![says(GOOD)]);
    let (planned, told) = planned_by(&fake);
    assert!(planned.plan.is_ok());
    assert_eq!(planned.tries.len(), 1);
    assert!(told.starts_with("A woman remembers") && told.contains("father: Man, 60s"));
}

/// An answer refused is asked again once, the error said with the prompt
/// (and told), and the second answer taken.
#[test]
fn a_refused_answer_is_asked_again_with_its_error() {
    let fake = Fake::new(vec![says(&broken()), says(GOOD)]);
    let (planned, told) = planned_by(&fake);
    assert!(planned.plan.is_ok());
    let why = "line 11 is in no scene: the last scene must end at line 11";
    assert_eq!(planned.tries[0].refused.as_deref(), Some(why));
    assert_eq!(planned.tries[1].refused, None);
    let asked = fake.asked.borrow();
    assert_eq!(asked[1], prompt::again(&asked[0], why));
    assert!(told.contains(&format!("(That plan did not hold: {why}. Once more.)")));
}

/// Two answers refused: the song has no story (kept as such); the last
/// cut off at its length: none either (kept as such too: written greedily,
/// it would be cut off again); a prompt too long or a model that fails: no
/// second try.
#[test]
fn a_plan_that_never_holds_is_refused_cut_off_or_failed() {
    let fake = Fake::new(vec![says(&broken()), says(&broken())]);
    let (planned, told) = planned_by(&fake);
    assert!(matches!(planned.plan, Err(Failure::Refused(_))));
    assert!(told.ends_with(
        "\nNo story for this song: line 11 is in no scene: the last scene must end at line 11\n"
    ));
    let cut = Written {
        text: GOOD[..GOOD.len() / 2].to_owned(),
        stopped: false,
        ..Written::default()
    };
    let fake = Fake::new(vec![says(&broken()), Ok(cut)]);
    let (planned, _) = planned_by(&fake);
    let Err(Failure::CutOff(why)) = planned.plan else {
        panic!("cut off: {:?}", planned.plan);
    };
    assert!(why.ends_with(&format!("(the answer was cut off at {MOST_ANSWER} tokens)")));
    // A first answer cut off is not asked again (a second call would
    // find no answer and panic).
    let cut = Written {
        text: GOOD[..GOOD.len() / 2].to_owned(),
        stopped: false,
        ..Written::default()
    };
    let (planned, _) = planned_by(&Fake::new(vec![Ok(cut)]));
    assert!(matches!(planned.plan, Err(Failure::CutOff(_))));
    assert_eq!(planned.tries.len(), 1);
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
    assert!(planned.tries.is_empty() && fake.asked.borrow().is_empty());
    assert!(told.contains(NO_LINES));
}

/// What is kept is read back the same, a plan or that there is none (in
/// any context); one made by another version or for another ask is not
/// read; one short of room in a context is read in one as small, not in a
/// larger one.
#[test]
fn a_plan_is_kept_with_what_made_it() {
    let plan = check::read(GOOD, &song()).unwrap();
    let asked = "0123456789abcdef";
    let text = to_text(asked, &Kept::Plan(Box::new(plan.clone())));
    assert!(text.starts_with("jodeldiplom-cache story-plan 2 by "));
    for context in [LEAST, CONTEXT, 4 * CONTEXT] {
        assert_eq!(
            from_text(&text, asked, context),
            Some(Kept::Plan(Box::new(plan.clone())))
        );
    }
    assert_eq!(
        from_text(&text, "fedcba9876543210", CONTEXT),
        None,
        "another ask"
    );
    let older = text.replace(&format!("{MADE} {VERSION}"), &format!("{MADE} 0"));
    assert_eq!(from_text(&older, asked, CONTEXT), None, "another maker");
    let none = to_text(asked, &Kept::NoStory("too few\nscenes".into()));
    assert_eq!(
        from_text(&none, asked, 4 * CONTEXT),
        Some(Kept::NoStory("too few scenes".into()))
    );
    let short = Kept::Short {
        context: CONTEXT,
        why: "the thought did not end".into(),
    };
    let text = to_text(asked, &short);
    assert_eq!(from_text(&text, asked, LEAST), Some(short.clone()));
    assert_eq!(from_text(&text, asked, CONTEXT), Some(short));
    assert_eq!(from_text(&text, asked, CONTEXT + 1), None, "more room");
}

/// The least context a card is given.
const LEAST: usize = jodel_gpu::llm::LEAST_CONTEXT;

/// A job runs its plan, tells it and keeps it, found again for the song
/// as it is asked (other words: planned again) in any context; a song
/// that has no story is kept as such (its answers refused, or cut off),
/// one short of room as such in its context (planned again in a larger
/// one), a model that failed is not kept.
#[test]
fn a_job_keeps_what_it_planned_for_the_song_as_asked() {
    let dir = jodel_testkit::TempDir::new("story");
    let file = dir.join("song.flac");
    durable::write_atomic(&file, b"not really a song").unwrap();
    let mut told = Vec::new();
    let tell = &mut |story: Story| told.push(story);
    let job = StoryJob::new(file.clone(), song(), dir.join("stories"), CONTEXT);
    assert_eq!(job.kept(), None, "none kept yet");
    let failed = Fake::new(vec![Err(Failure::Model("device lost".into()))]);
    assert!(matches!(job.run(&failed, tell), Err(Failure::Model(_))));
    assert_eq!(job.kept(), None, "a model's failure is not kept");
    let plan = job.run(&Fake::new(vec![says(GOOD)]), tell).unwrap();
    assert_eq!(job.kept(), Some(Kept::Plan(Box::new(plan.clone()))));
    let larger = StoryJob::new(file.clone(), song(), dir.join("stories"), 2 * CONTEXT);
    assert_eq!(
        larger.kept(),
        Some(Kept::Plan(Box::new(plan))),
        "any context"
    );
    let mut other = song();
    other.lines[0].text = "I ran down to the harbour".into();
    let other = StoryJob::new(file.clone(), other, dir.join("stories"), CONTEXT);
    assert_eq!(other.kept(), None, "other words, another ask");
    let refused = Fake::new(vec![says(&broken()), says(&broken())]);
    assert!(matches!(
        other.run(&refused, tell),
        Err(Failure::Refused(_))
    ));
    assert!(told.iter().any(|s| matches!(s, Story::Words(_))));
    assert!(matches!(other.kept(), Some(Kept::NoStory(_))));
    // An answer cut off at its length: kept as none too, for this ask.
    let mut third = song();
    third.lines[0].text = "I swam out past the harbour".into();
    let third = StoryJob::new(dir.join("song.flac"), third, dir.join("stories"), CONTEXT);
    let cut = Written {
        text: GOOD[..GOOD.len() / 2].to_owned(),
        stopped: false,
        ..Written::default()
    };
    let cut = Fake::new(vec![Ok(cut)]);
    assert!(matches!(
        third.run(&cut, &mut |_| {}),
        Err(Failure::CutOff(_))
    ));
    assert!(matches!(third.kept(), Some(Kept::NoStory(_))));
    // A thought cut off by the context: none in it, planned again in more.
    let mut fourth = song();
    fourth.lines[0].text = "I flew over the harbour".into();
    let at = |context| StoryJob::new(file.clone(), fourth.clone(), dir.join("stories"), context);
    let endless = Written {
        thinking: true,
        thought_tokens: 3_000,
        ..Written::default()
    };
    let short = at(CONTEXT).run(&Fake::new(vec![Ok(endless)]), &mut |_| {});
    assert!(matches!(short, Err(Failure::Short(_))));
    assert!(matches!(
        at(CONTEXT).kept(),
        Some(Kept::Short {
            context: CONTEXT,
            ..
        })
    ));
    assert!(matches!(at(LEAST).kept(), Some(Kept::Short { .. })));
    assert_eq!(at(2 * CONTEXT).kept(), None, "more room: planned again");
}

/// The whole job on the model, end to end: `JODEL_GEMMA` names the
/// model's file and `JODEL_STORY_SONG` a song as the plan is made from it
/// (a [`Song`] as JSON); skipped where either is not there. The song is
/// planned twice, the same answer both times (greedy), the plan read and
/// its look checked whole; the plan, the answers and the times go to the
/// folder `JODEL_STORY_OUT` names, where it names one.
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
                "  prompt {} tokens in {:.2} s; answer {} tokens in {:.1} s ({:.0} a second){}{}\n",
                w.prompt_tokens,
                w.prompt_time.as_secs_f64(),
                w.answer_tokens,
                w.answer_time.as_secs_f64(),
                w.answer_tokens as f64 / w.answer_time.as_secs_f64().max(1e-9),
                if w.stopped { "" } else { "; cut off" },
                t.refused
                    .as_ref()
                    .map_or_else(String::new, |why| format!("; refused: {why}"))
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
        }
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
