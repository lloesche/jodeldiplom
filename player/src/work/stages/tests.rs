//! The stages on fake models of round sizes (not the real models'), on
//! fake cards of 8 GB and 16 GB, a stepped clock; the planning done by
//! hand.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use super::plan::{Plan, Step, IDLE};
use super::*;
use crate::work::story_models::StoryModels;

const GB: u64 = 1 << 30;

/// What the window and the rest of the process hold on the card.
const BASE: u64 = GB / 2;
/// The budgets of an 8 GB card and a 16 GB one.
const EIGHT: u64 = 7 * GB + GB / 2;
const SIXTEEN: u64 = 15 * GB;

/// A model of `gb` gigabytes of weights and a quarter of that working.
fn model(gb: u64) -> Footprint {
    Footprint {
        resident: gb * GB,
        working: gb * GB / 4,
    }
}

/// A plan of models of 4 GB each (two do not fit together on 8 GB, all
/// of them do not on 16 GB, three do).
fn plan() -> Plan {
    let mut plan = Plan::default();
    for stage in Stage::ALL {
        plan.declare(stage, model(4));
    }
    plan
}

/// `plan` made again at `now` on a card of `budget` holding [`BASE`] and
/// what it has loaded.
fn step(plan: &mut Plan, budget: u64, now: Instant) -> Step {
    let usage = BASE.saturating_add(plan.held());
    plan.step(Some(Budget { budget, usage }), now)
}

/// What a step grants and lets go.
fn did(granted: &[u64], released: &[Stage]) -> Step {
    Step {
        released: released.to_vec(),
        granted: granted.to_vec(),
    }
}

/// A stage's queue is first in first out, one job at a time.
#[test]
fn a_stage_runs_its_jobs_first_in_first_out_one_at_a_time() {
    let (mut plan, t) = (plan(), Instant::now());
    let tickets: Vec<u64> = (0..3).map(|_| plan.queue(&[Stage::Stems], false)).collect();
    for &ticket in &tickets {
        assert_eq!(step(&mut plan, SIXTEEN, t), did(&[ticket], &[]));
        assert_eq!(step(&mut plan, SIXTEEN, t), did(&[], &[]), "one at a time");
        plan.done(ticket, t);
    }
}

/// A job needing two models runs once it is first in both queues.
#[test]
fn a_job_needing_two_models_runs_once_both_are_its_turn() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(1));
    }
    let voice = plan.queue(&[Stage::Stems], false);
    let analysis = plan.queue(&[Stage::Stems, Stage::Beats], false);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[voice], &[]));
    plan.done(voice, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[analysis], &[]));
    assert!(plan.loaded(Stage::Beats));
}

/// Where two models do not fit together, one drains what was queued for it
/// before the other takes its room.
#[test]
fn on_a_small_card_a_model_drains_its_queue_before_the_next_takes_its_room() {
    let (mut plan, t) = (plan(), Instant::now());
    let (a, b) = (
        plan.queue(&[Stage::Hearing], false),
        plan.queue(&[Stage::Hearing], false),
    );
    assert_eq!(step(&mut plan, EIGHT, t), did(&[a], &[]));
    let timing = plan.queue(&[Stage::Timing], false);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[], &[]), "no room yet");
    plan.done(a, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[b], &[]), "queued before");
    plan.done(b, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[timing], &[Stage::Hearing]));
    assert!(plan.held() + BASE <= EIGHT, "never over the budget");
}

/// A job queued for a model let go while another drains its queue waits
/// until that queue is empty.
#[test]
fn a_song_queued_meanwhile_waits_until_the_models_queue_is_empty() {
    let (mut plan, t) = (plan(), Instant::now());
    let first = plan.queue(&[Stage::Scenes], false);
    let second = plan.queue(&[Stage::Scenes], false);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[first], &[]));
    let late = plan.queue(&[Stage::Hearing], false);
    plan.done(first, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[second], &[]));
    plan.done(second, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[late], &[Stage::Scenes]));
}

/// A stage asked for again and again holds no other off: once a job waits
/// for room, jobs queued after it wait behind it, the loaded stage's too.
#[test]
fn a_stage_asked_again_and_again_holds_no_other_off() {
    let (mut plan, t) = (plan(), Instant::now());
    let running = plan.queue(&[Stage::Stems], false);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[running], &[]));
    let other = plan.queue(&[Stage::Scenes], false);
    let again = plan.queue(&[Stage::Stems], false);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[], &[]));
    plan.done(running, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[other], &[Stage::Stems]));
    plan.done(other, t);
    assert_eq!(step(&mut plan, EIGHT, t), did(&[again], &[Stage::Scenes]));
}

/// While a song is sung, a job held back (its model not loaded) holds no
/// queue's front: the song's own job behind it goes on.
#[test]
fn a_job_held_back_by_singing_holds_no_queue() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(1));
    }
    let first = plan.queue(&[Stage::Stems], false);
    step(&mut plan, SIXTEEN, t);
    plan.done(first, t);
    plan.sing(true);
    let analysis = plan.queue(&[Stage::Stems, Stage::Beats], false);
    let shine = plan.queue(&[Stage::Stems], true);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[shine], &[]));
    plan.done(shine, t);
    plan.sing(false);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[analysis], &[]));
}

/// A model whose footprint is not known yet (its files not there yet)
/// loads beside nothing: the others not in use go first, one at work is
/// waited for, and nothing loads beside it.
#[test]
fn a_model_whose_footprint_is_not_known_runs_alone() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(1));
    }
    plan.declare(Stage::Scenes, jodel_gpu::UNKNOWN);
    let (stems, beats) = (
        plan.queue(&[Stage::Stems], false),
        plan.queue(&[Stage::Beats], false),
    );
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[stems, beats], &[]));
    plan.done(stems, t);
    let scenes = plan.queue(&[Stage::Scenes], false);
    assert_eq!(
        step(&mut plan, SIXTEEN, t),
        did(&[], &[]),
        "the beats at work"
    );
    plan.done(beats, t);
    assert_eq!(
        step(&mut plan, SIXTEEN, t),
        did(&[scenes], &[Stage::Stems, Stage::Beats])
    );
    let hearing = plan.queue(&[Stage::Hearing], false);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[], &[]), "none beside it");
    plan.done(scenes, t);
    assert_eq!(
        step(&mut plan, SIXTEEN, t),
        did(&[hearing], &[Stage::Scenes])
    );
}

/// Models that fit together stay loaded together, a song after another,
/// nothing let go.
#[test]
fn models_that_fit_stay_loaded_together() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(1));
    }
    for _song in 0..2 {
        for stage in Stage::ALL {
            let ticket = plan.queue(&[stage], false);
            assert_eq!(step(&mut plan, SIXTEEN, t), did(&[ticket], &[]));
            plan.done(ticket, t);
        }
    }
    assert!(Stage::ALL.iter().all(|&s| plan.loaded(s)));
    assert!(plan.held() + BASE <= SIXTEEN - plan::MARGIN);
}

/// Where the next model does not fit, the models not in use make room,
/// the one used longest ago first, never one at work; a model that fits
/// nowhere runs alone.
#[test]
fn the_models_not_in_use_make_room_the_oldest_first() {
    let (mut plan, mut t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(3));
    }
    for stage in [Stage::Stems, Stage::Beats, Stage::Hearing] {
        let ticket = plan.queue(&[stage], false);
        step(&mut plan, SIXTEEN, t);
        plan.done(ticket, t);
        t += Duration::from_secs(1);
    }
    // 9 GB loaded of the 13.5 GB room: the timing's 3.75 GB fits, the
    // scenes' then needs the stems' room (used longest ago).
    let timing = plan.queue(&[Stage::Timing], false);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[timing], &[]));
    let scenes = plan.queue(&[Stage::Scenes], false);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[scenes], &[Stage::Stems]));
    // On a card too small for any two, a model runs alone.
    plan.done(timing, t);
    plan.done(scenes, t);
    let small = 4 * GB;
    let stems = plan.queue(&[Stage::Stems], false);
    let made = step(&mut plan, small, t);
    assert_eq!(made.granted, [stems]);
    assert_eq!(made.released.len(), 4, "{made:?}");
}

/// Once every queue has been empty for a minute, every model goes, on any
/// card; a job queued meanwhile starts the minute again.
#[test]
fn every_model_goes_after_a_minute_with_nothing_to_do() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in [Stage::Stems, Stage::Beats] {
        let ticket = plan.queue(&[stage], false);
        step(&mut plan, SIXTEEN, t);
        plan.done(ticket, t);
    }
    assert_eq!(plan.deadline(), Some(t + IDLE));
    assert_eq!(step(&mut plan, SIXTEEN, t + IDLE / 2), did(&[], &[]));
    let ticket = plan.queue(&[Stage::Beats], false);
    step(&mut plan, SIXTEEN, t + IDLE / 2);
    assert_eq!(plan.deadline(), None, "something to do");
    let later = t + IDLE / 2 + Duration::from_secs(1);
    plan.done(ticket, later);
    assert_eq!(step(&mut plan, SIXTEEN, t + IDLE), did(&[], &[]));
    assert_eq!(
        step(&mut plan, SIXTEEN, later + IDLE),
        did(&[], &[Stage::Stems, Stage::Beats])
    );
    assert_eq!(plan.held(), 0);
    assert_eq!(plan.deadline(), None, "nothing left to let go");
}

/// While a song is sung nothing new goes on the card: a model already
/// loaded goes on with its jobs, a job for the song loads its model, any
/// other job for a model not loaded waits for the song's end.
#[test]
fn singing_takes_the_gpu_first() {
    let (mut plan, t) = (plan(), Instant::now());
    for stage in Stage::ALL {
        plan.declare(stage, model(1));
    }
    let first = plan.queue(&[Stage::Stems], false);
    step(&mut plan, SIXTEEN, t);
    plan.done(first, t);
    plan.sing(true);
    let stems = plan.queue(&[Stage::Stems], false);
    let scenes = plan.queue(&[Stage::Scenes], false);
    let shine = plan.queue(&[Stage::Hearing], true);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[stems, shine], &[]));
    plan.sing(false);
    assert_eq!(step(&mut plan, SIXTEEN, t), did(&[scenes], &[]));
}

/// A fake model: its releases told on a channel with the thread that
/// carried them out.
struct Fake {
    footprint: Footprint,
    released: Mutex<mpsc::Sender<thread::ThreadId>>,
}

impl Resident for Fake {
    fn footprint(&self) -> Footprint {
        self.footprint
    }

    fn release(&self) {
        let told = self.released.lock().unwrap();
        let _ = told.send(thread::current().id());
    }
}

/// A fake model with bugs: its footprint and its release panic.
struct Buggy;

impl Resident for Buggy {
    fn footprint(&self) -> Footprint {
        panic!("a bug in a footprint")
    }

    fn release(&self) {
        panic!("a bug in a release")
    }
}

/// Stages on a stepped clock and an 8 GB card holding [`BASE`] and what
/// the plan has loaded, a fake model of 4 GB registered for each of
/// `stages`; their releases.
fn stages(of: &[Stage]) -> (Arc<Stages>, mpsc::Receiver<thread::ThreadId>, AppClock) {
    let (tx, released) = mpsc::channel();
    let clock = AppClock::stepped();
    let held = Arc::new(Mutex::new(0u64));
    let stages = Arc::new(Stages::new(clock.clone(), move || {
        Some(Budget {
            budget: EIGHT,
            usage: BASE + *held.lock().unwrap(),
        })
    }));
    for &stage in of {
        let fake = Fake {
            footprint: model(4),
            released: Mutex::new(tx.clone()),
        };
        stages.register(stage, Arc::new(fake));
    }
    (stages, released, clock)
}

/// How long a test waits for another thread at most, however busy the
/// machine.
const GUARD: Duration = Duration::from_secs(20);

/// Waits (within [`GUARD`]) until `stages`' plan has `n` jobs waiting for
/// `stage`.
fn until_waiting(stages: &Stages, stage: Stage, n: usize) {
    let inner = stages.lock();
    let (inner, timeout) = stages
        .changed
        .wait_timeout_while(inner, GUARD, |i| i.plan.waiting(stage) != n)
        .unwrap();
    assert!(!timeout.timed_out(), "{stage:?} never had {n} waiting");
    drop(inner);
}

/// A job in a thread of its own, telling when it starts and waiting to be
/// let end.
fn job(
    stages: &Arc<Stages>,
    stage: Stage,
) -> (mpsc::Receiver<()>, mpsc::Sender<()>, thread::JoinHandle<()>) {
    let (started, start) = mpsc::channel();
    let (end, ended) = mpsc::channel::<()>();
    let stages = Arc::clone(stages);
    let handle = thread::spawn(move || {
        stages.run(&[stage], "a job", || {
            started.send(()).unwrap();
            ended.recv().unwrap();
        });
    });
    (start, end, handle)
}

/// A job starts only once the planner has planned (queuing, ending and
/// singing only tell it), on its own thread; a model is let go by the
/// planner, not by the jobs, before the job that needs its room starts;
/// every model goes after a minute with nothing to do.
#[test]
fn the_planner_lets_models_go_and_then_starts_the_jobs() {
    let (stages, released, clock) = stages(&[Stage::Hearing, Stage::Timing]);
    let (started, end, hearing) = job(&stages, Stage::Hearing);
    until_waiting(&stages, Stage::Hearing, 1);
    assert!(
        started.recv_timeout(Duration::from_millis(50)).is_err(),
        "not planned yet"
    );
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    let (timing_started, timing_end, timing) = job(&stages, Stage::Timing);
    until_waiting(&stages, Stage::Timing, 1);
    stages.plan_once();
    assert!(
        timing_started
            .recv_timeout(Duration::from_millis(50))
            .is_err(),
        "no room"
    );
    end.send(()).unwrap();
    hearing.join().unwrap();
    assert!(released.try_recv().is_err(), "a job's end lets nothing go");
    stages.plan_once();
    assert_eq!(
        released.try_recv(),
        Ok(thread::current().id()),
        "the planner's"
    );
    timing_started.recv_timeout(GUARD).unwrap();
    timing_end.send(()).unwrap();
    timing.join().unwrap();
    clock.step(IDLE - Duration::from_secs(1));
    stages.plan_once();
    assert!(stages.loaded(Stage::Timing));
    clock.step(Duration::from_secs(1));
    stages.plan_once();
    assert!(!stages.loaded(Stage::Timing));
    assert_eq!(released.try_recv(), Ok(thread::current().id()));
}

/// While a song is sung, a job for a model not loaded waits for its end;
/// one for the song does not.
#[test]
fn while_a_song_is_sung_a_model_not_loaded_waits_for_its_end() {
    let (stages, _, _) = stages(&[Stage::Scenes, Stage::Hearing]);
    let singing = stages.sing();
    let (started, end, scenes) = job(&stages, Stage::Scenes);
    let shine = {
        let stages = Arc::clone(&stages);
        thread::spawn(move || {
            for_the_song_sung();
            stages.run(&[Stage::Hearing], "the song's", || 7)
        })
    };
    until_waiting(&stages, Stage::Scenes, 1);
    until_waiting(&stages, Stage::Hearing, 1);
    stages.plan_once();
    assert_eq!(shine.join().unwrap(), 7, "the song's own job");
    stages.plan_once();
    assert!(started.recv_timeout(Duration::from_millis(50)).is_err());
    drop(singing);
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    end.send(()).unwrap();
    scenes.join().unwrap();
}

/// A turn inside a turn panics rather than wait for ever; the outer turn
/// ends with it, and the next job goes.
#[test]
fn a_turn_inside_a_turn_panics() {
    let (stages, _, _) = stages(&[Stage::Stems, Stage::Beats]);
    let nested = {
        let stages = Arc::clone(&stages);
        thread::spawn(move || {
            stages.run(&[Stage::Stems], "outer", || {
                stages.run(&[Stage::Beats], "inner", || ());
            })
        })
    };
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    assert!(nested.join().is_err(), "it panicked");
    let (started, end, next) = job(&stages, Stage::Stems);
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    end.send(()).unwrap();
    next.join().unwrap();
}

/// A model whose footprint and release panic stops no work: it is taken
/// as not known (it runs alone), and as let go; the jobs after go on.
#[test]
fn a_model_that_panics_stops_no_work() {
    let (stages, _, clock) = stages(&[]);
    stages.register(Stage::Stems, Arc::new(Buggy));
    let (started, end, first) = job(&stages, Stage::Stems);
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    end.send(()).unwrap();
    first.join().unwrap();
    clock.step(IDLE);
    stages.plan_once();
    assert!(
        !stages.loaded(Stage::Stems),
        "let go, though its release panicked"
    );
    let (started, end, next) = job(&stages, Stage::Stems);
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    end.send(()).unwrap();
    next.join().unwrap();
}

/// A job that panics ends its turn: the next goes.
#[test]
fn a_job_that_panicked_lets_the_next_go() {
    let (stages, _, _) = stages(&[Stage::Stems]);
    let panicked = {
        let stages = Arc::clone(&stages);
        thread::spawn(move || stages.run(&[Stage::Stems], "a bug", || panic!("a bug")))
    };
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    assert!(panicked.join().is_err());
    let (started, end, next) = job(&stages, Stage::Stems);
    until_waiting(&stages, Stage::Stems, 1);
    stages.plan_once();
    started.recv_timeout(GUARD).unwrap();
    end.send(()).unwrap();
    next.join().unwrap();
}

/// The models that run today register themselves where they are made: the
/// background separator and the beat network with the analyses' worker,
/// the speech recogniser and the speech model with the word timer, the
/// story's four (its three, and the scenes' depth and cut-out) with its
/// models; their footprints said before loading.
#[test]
fn the_models_register_themselves_where_they_are_made() {
    crate::work::rhythm::register_networks();
    let models = crate::work::lyrics::models::Models::new("speech".into(), "asr".into());
    let _timer = crate::work::lyrics::timer::SpeechTimer::new(
        models,
        None,
        crate::setup::Unreadable::none(),
    );
    let story = StoryModels {
        language: "gemma.gguf".into(),
        transformer: "transformer".into(),
        pipeline: "pipeline".into(),
        encoder: "encoder".into(),
    };
    let scenes = crate::work::scenes::SceneModels {
        depth: "depth".into(),
        matte: "matte".into(),
    };
    let _story = crate::work::story::worker::GpuModels::new(
        &story,
        scenes,
        crate::setup::Unreadable::none(),
    );
    let inner = Stages::global().lock();
    for stage in Stage::ALL {
        assert!(inner.models[stage as usize].is_some(), "{stage:?}");
    }
    let stems = inner.models[Stage::Stems as usize].as_ref().unwrap();
    assert!(stems.footprint().resident > 0, "worked out from its shapes");
}

/// The process's planner plans on a thread of its own: a job queued there
/// starts with no one planning by hand.
#[test]
fn the_process_planner_plans_on_its_own() {
    let counted = Arc::new(AtomicUsize::new(0));
    let ran = Arc::clone(&counted);
    thread::spawn(move || {
        run(&[Stage::Scenes], "a song's scenes", || {
            ran.fetch_add(1, Ordering::SeqCst)
        });
    })
    .join()
    .unwrap();
    assert_eq!(counted.load(Ordering::SeqCst), 1);
}
