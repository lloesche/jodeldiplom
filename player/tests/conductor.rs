//! The sound conductor and its synth allocate nothing once they are made:
//! every kind of event, the clock (from an analysis) and the key told,
//! held sounds started and let go, a seek, and rendering through the
//! synth's bus, counted by a global allocator that counts on this thread.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use jodel_rhythm::Analysis;
use jodeldiplom::sfx::conductor::{Clock, Conductor, Diploma, Event};
use jodeldiplom::tiers::Tier;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether this thread's allocations count (the harness has its own
    /// threads).
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

fn count() {
    if COUNTING.with(Cell::get) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call goes straight to the system allocator, with the same
// layout and pointer; it only counts on the way.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        count();
        // SAFETY: the caller's contract is passed on as it is.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

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
    let (mut c, mut synth) = Conductor::new(RATE);
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
            c.event(&mut synth, now, Event::Word { streak, pan: 0.3 });
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
