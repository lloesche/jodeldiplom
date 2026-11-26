//! The sound conductor and its synth allocate nothing once they are made:
//! every kind of event, the announcer's calls, the clock (from an
//! analysis) and the key told, held sounds started and let go, a seek, and
//! rendering through the synth's bus, counted by a global allocator that
//! counts on this thread.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use jodel_rhythm::Analysis;
use jodel_testkit::{Call, Counting};
use jodeldiplom::sfx::conductor::{Called, Clock, Conductor, Diploma, Event};
use jodeldiplom::tiers::Tier;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether this thread's allocations count (the harness has its own
    /// threads).
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn count(call: Call) {
    if call == Call::Alloc && COUNTING.with(Cell::get) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting(count);

#[test]
fn conducting_allocates_nothing() {
    const RATE: u32 = 48_000;
    let beats: Vec<f64> = (0..200).map(|k| 0.5 * f64::from(k)).collect();
    let a = Analysis {
        downbeats: (0..200).step_by(4).collect(),
        local_bpm: vec![120.0; beats.len()],
        beats,
        ..Analysis::default()
    };
    let (mut c, mut synth) = Conductor::new(RATE, &jodeldiplom::theme::CLASSIC);
    let mut out = vec![0.0f32; 2 * 480];
    COUNTING.with(|on| on.set(true));
    // The count counts.
    drop(std::hint::black_box(Vec::<u8>::with_capacity(8)));
    assert_eq!(ALLOCATIONS.swap(0, Ordering::Relaxed), 1);
    let mut streak = 0;
    for block in 0..2_000u64 {
        let now = synth.frame();
        if block % 10 == 0 {
            let mut lines = [None; Clock::LINES];
            lines[0] = Some((now + 10_000, now + 100_000));
            let t = now as f64 / f64::from(RATE);
            let clock = Clock::from_analysis(&a, t, now, f64::from(RATE), lines);
            c.set_clock(&mut synth, clock);
        }
        if block % 20 == 0 {
            streak += 1;
            c.event(
                &mut synth,
                now,
                Event::Word {
                    streak,
                    pan: 0.3,
                    size: 0.0,
                },
            );
            if streak == 60 {
                c.event(&mut synth, now, Event::Lost(Tier::TOP));
                streak = 0;
            }
        }
        match block % 400 {
            100 => {
                let tier = Tier::ALL[(block / 400) as usize % Tier::COUNT];
                c.event(&mut synth, now, Event::Phrase(tier));
            }
            150 => {
                c.event(&mut synth, now, Event::Jackpot);
            }
            200 => {
                c.event(&mut synth, now, Event::Zap { pan: -0.4 });
            }
            220 => {
                // The top rung reached: its stinger, payout and call; and
                // the announcer calling the rest out.
                let top = Tier::TOP.spec().streak;
                let word = Event::Word {
                    streak: top,
                    pan: 0.0,
                    size: 0.0,
                };
                c.event(&mut synth, now, word);
                for t in Tier::ALL {
                    c.announce(&mut synth, now, now + 4_800, Called::Name(t));
                    c.announce(&mut synth, now, now + 9_600, Called::Times(t));
                }
            }
            250..=273 => {
                let k = (block % 400 - 250) as u32;
                let tick = Diploma::Tick { k, of: 24 };
                c.event(&mut synth, now, Event::Diploma(tick));
            }
            275 => {
                let gem = Diploma::Gem {
                    tier: Some(Tier::ALL[3]),
                    pan: -0.2,
                };
                c.event(&mut synth, now, Event::Diploma(gem));
            }
            280 => {
                let points = Diploma::Points(Some(Tier::TOP));
                c.event(&mut synth, now, Event::Diploma(points));
            }
            285 => {
                let rung = Diploma::Rung(Tier::TOP);
                c.event(&mut synth, now, Event::Diploma(rung));
            }
            300 => {
                let key = jodel_rhythm::key::Key::on((block % 12) as u8, true);
                c.set_key(Some(&key), -2);
            }
            390 => c.cancel(&mut synth, now),
            _ => {}
        }
        synth.render_to(&mut out);
    }
    COUNTING.with(|on| on.set(false));
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(out.iter().all(|s| s.is_finite()));
    assert_eq!(synth.dropped(), 0);
}

/// The theme switched as its sounds ring (the settings' row mid-song):
/// its patches told the synth that plays, as commands, nothing allocated;
/// a chord held on the choir plays on in the new theme's voice with no
/// step in the sound bigger than its own before, and goes with its keys
/// (no note stuck); the next word's pop is the new theme's, and so is
/// every atom's note.
#[test]
fn a_theme_switched_as_it_sounds_plays_its_patches_and_allocates_nothing() {
    use jodeldiplom::sfx::conductor::Atom;
    use jodeldiplom::theme::{CLASSIC, SPOOKY};
    const RATE: u32 = 48_000;
    const BLOCK: usize = 480;
    let (mut c, mut synth) = Conductor::new(RATE, &CLASSIC);
    let word = Event::Word {
        streak: 1,
        pan: 0.0,
        size: 0.3,
    };
    let mut out = vec![0.0f32; 2 * BLOCK];
    // The biggest step from sample to sample over `blocks` blocks.
    let mut steps = |synth: &mut jodel_synth::Synth, blocks: usize| {
        let mut most = 0.0f32;
        for _ in 0..blocks {
            let last = out[out.len() - 2];
            synth.render_to(&mut out);
            most = (out.iter().step_by(2))
                .fold((most, last), |(m, before), &s| {
                    (m.max((s - before).abs()), s)
                })
                .0;
        }
        most
    };
    COUNTING.with(|on| on.set(true));
    for note in [60.0, 64.0] {
        c.key_down(&mut synth, 0, Atom::Choir, note);
    }
    c.event(&mut synth, 0, word);
    // The chord settled, the pop gone.
    steps(&mut synth, 40);
    let before = steps(&mut synth, 20);
    for atom in Atom::ALL {
        c.set_patch(&mut synth, atom, SPOOKY.patch(atom));
    }
    let across = steps(&mut synth, 20);
    let choir = |synth: &jodel_synth::Synth, held: bool| {
        (synth.voices())
            .filter(|v| v.held == held && c.atom_of(v.patch) == Some(Atom::Choir))
            .all(|v| synth.patch(v.patch) == Some(&SPOOKY.patch(Atom::Choir)))
    };
    let held = synth.voices().filter(|v| v.held).count();
    let rings_on = choir(&synth, true);
    let at = synth.frame();
    c.event(&mut synth, at, word);
    steps(&mut synth, 2);
    let pops = (synth.voices())
        .filter(|v| c.atom_of(v.patch) == Some(Atom::Pop))
        .map(|v| synth.patch(v.patch) == Some(&SPOOKY.patch(Atom::Pop)))
        .fold((0, true), |(n, all), new| (n + 1, all && new));
    let at = synth.frame();
    for note in [60.0, 64.0] {
        c.key_up(&mut synth, at, Atom::Choir, note);
    }
    steps(&mut synth, 300);
    // Every atom's note on its patch in the new theme.
    let mut amiss = None;
    let mut block = [0.0f32; 2 * jodel_synth::BLOCK];
    for atom in Atom::ALL {
        let at = synth.frame();
        c.key_down(&mut synth, at, atom, 67.0);
        // Block by block until it sounds (past a spread's start: a choir's
        // voices come in one by one).
        let mut theirs = false;
        for _ in 0..100 {
            synth.render_to(&mut block);
            let mut on = (synth.voices()).filter(|v| c.atom_of(v.patch) == Some(atom));
            if let Some(v) = on.next() {
                theirs = synth.patch(v.patch) == Some(&SPOOKY.patch(atom));
                break;
            }
        }
        if !theirs {
            amiss = amiss.or(Some(atom));
        }
        let at = synth.frame();
        c.key_up(&mut synth, at, atom, 67.0);
        steps(&mut synth, 300);
    }
    COUNTING.with(|on| on.set(false));
    assert_eq!(amiss, None, "every atom on the new theme's patch");
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(held, 2, "the chord plays on");
    assert!(rings_on, "in the new theme's choir");
    assert!(
        across <= 1.5 * before + 1e-4,
        "no click: {across} across the switch, {before} before"
    );
    assert!(pops.0 > 0 && pops.1, "the pop is the new theme's: {pops:?}");
    assert_ne!(SPOOKY.patch(Atom::Pop), CLASSIC.patch(Atom::Pop));
    assert_eq!(
        synth.voices().filter(|v| v.held).count(),
        0,
        "no note stuck"
    );
    assert_eq!((c.dropped(), synth.dropped()), (0, 0));
}
