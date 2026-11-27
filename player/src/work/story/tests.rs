//! The plan's tries with writers that answer as told (taken, asked again,
//! refused, cut off, failed), what is kept of them and how, and the whole
//! job on the model, end to end, where it is here.

use std::cell::RefCell;
use std::sync::mpsc;
use std::sync::Arc;

use super::*;
use crate::core::story::tests::{song, GOOD};

/// A writer that answers each prompt with the next of its answers, its
/// text streamed in pieces of seven characters, and keeps the prompts.
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
        _most: usize,
        seen: &mut dyn FnMut(&str),
    ) -> Result<Written, Failure> {
        self.asked.borrow_mut().push(prompt.to_owned());
        let answer = self.answers.borrow_mut().remove(0)?;
        let chars: Vec<char> = answer.text.chars().collect();
        for piece in chars.chunks(7) {
            seen(&piece.iter().collect::<String>());
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

/// The plan `fake` writes for [`song`], and what it told.
fn planned_by(fake: &Fake) -> (Planned, String) {
    let song = song();
    let mut told = String::new();
    let planned = plan(fake, &song, &prompt::prompt(&song), &mut |story| {
        if let Story::Words(words) = story {
            told.push_str(&words);
        }
    });
    (planned, told)
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
/// cut off at its length: none either, but the length is to blame (not
/// kept); a prompt too long or a model that fails: no second try.
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
        Failure::Refused("too long".into()),
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
    let planned = plan(&fake, &silent, &prompt::prompt(&silent), &mut |story| {
        if let Story::Words(words) = story {
            told.push_str(&words);
        }
    });
    assert_eq!(planned.plan, Err(Failure::Refused(NO_LINES.into())));
    assert!(planned.tries.is_empty() && fake.asked.borrow().is_empty());
    assert!(told.contains(NO_LINES));
}

/// What is kept is read back the same, a plan or that there is none; one
/// made by another version or for another ask is not read.
#[test]
fn a_plan_is_kept_with_what_made_it() {
    let plan = check::read(GOOD, &song()).unwrap();
    let text = to_text("0123456789abcdef", &Kept::Plan(Box::new(plan.clone())));
    assert!(text.starts_with("jodeldiplom-cache story-plan 1 by "));
    assert_eq!(
        from_text(&text, "0123456789abcdef"),
        Some(Kept::Plan(Box::new(plan)))
    );
    assert_eq!(from_text(&text, "fedcba9876543210"), None, "another ask");
    let older = text.replace(&format!("{MADE} {VERSION}"), &format!("{MADE} 0"));
    assert_eq!(from_text(&older, "0123456789abcdef"), None, "another maker");
    let none = to_text("0123456789abcdef", &Kept::NoStory("too few\nscenes".into()));
    assert_eq!(
        from_text(&none, "0123456789abcdef"),
        Some(Kept::NoStory("too few scenes".into()))
    );
}

/// A job runs its plan, tells it on the outbox and keeps it, found again
/// for the song as it is asked (other words: planned again); a song that
/// has no story is kept as such, a model that failed is not kept.
#[test]
fn a_job_keeps_what_it_planned_for_the_song_as_asked() {
    let dir = jodel_testkit::TempDir::new("story");
    let file = dir.join("song.flac");
    durable::write_atomic(&file, b"not really a song").unwrap();
    let (tx, rx) = mpsc::channel();
    let outbox = Outbox::new(tx, Arc::new(|| {}));
    let job = StoryJob::new(file.clone(), song(), dir.join("stories"));
    assert_eq!(job.kept(), None, "none kept yet");
    let failed = Fake::new(vec![Err(Failure::Model("device lost".into()))]);
    assert!(matches!(job.run(&failed, &outbox), Err(Failure::Model(_))));
    assert_eq!(job.kept(), None, "a model's failure is not kept");
    let plan = job.run(&Fake::new(vec![says(GOOD)]), &outbox).unwrap();
    assert_eq!(job.kept(), Some(Kept::Plan(Box::new(plan))));
    assert!(rx.try_iter().any(|a| matches!(
        a,
        Action::Heard {
            part: Heard::Story(Story::Words(_)),
            ..
        }
    )));
    let mut other = song();
    other.lines[0].text = "I ran down to the harbour".into();
    let other = StoryJob::new(file, other, dir.join("stories"));
    assert_eq!(other.kept(), None, "other words, another ask");
    let refused = Fake::new(vec![says(&broken()), says(&broken())]);
    assert!(matches!(
        other.run(&refused, &outbox),
        Err(Failure::Refused(_))
    ));
    assert!(matches!(other.kept(), Some(Kept::NoStory(_))));
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
        let planned = plan(&gemma, &song, &asked, &mut |story| {
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
