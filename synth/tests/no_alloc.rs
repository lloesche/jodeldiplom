//! Nothing allocates after the synth is made: cueing notes on every
//! engine, retriggering and stealing voices, a note let go of alone,
//! patches edited as they sound (to another engine too), and rendering
//! through the bus (the chorus, the delay, the hall, the compressor, the
//! level gliding and the limiter), counted by a global allocator that
//! counts.

use jodel_synth::chorus::{ChorusSettings, Wave};
use jodel_synth::engines::{
    analog, drum, fm, formant, lofi, modal, noise, phase, string, supersaw, AnyParams,
};
use jodel_synth::params::Knobs;
use jodel_synth::patch::{Lfo, LfoRate};
use jodel_synth::{Bank, Command, NoteOn, Patch, Synth};
use jodel_testkit::{Call, Counting};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether this thread's allocations count (the test harness has its
    /// own threads).
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
fn rendering_allocates_nothing() {
    let mut bank = Bank::new();
    let lfo = Lfo {
        rate: LfoRate::Beats(0.5),
        pitch: 0.2,
        bright: 1.0,
        amp: 0.3,
        shape: 0.5,
        delay: 0.1,
        fade: 0.2,
    };
    let ids = [
        bank.add(Patch {
            reverb: 0.5,
            ..Patch::new(noise::WHOOSH.into())
        }),
        bank.add(Patch {
            delay: 0.5,
            lfo,
            ..Patch::new(analog::ZAP.into())
        }),
        bank.add(Patch {
            mono: true,
            ..Patch::new(lofi::BLIP.into())
        }),
        bank.add(Patch {
            reverb: 0.3,
            delay: 0.3,
            ..Patch::new(phase::BELL.into())
        }),
        bank.add(Patch {
            mono: true,
            reverb: 0.2,
            ..Patch::new(modal::SLOT_BELL.into())
        }),
        bank.add(Patch {
            gate: 2.0,
            ..Patch::new(string::HARP.into())
        }),
        bank.add(Patch {
            delay: 0.2,
            ..Patch::new(formant::AWW.into())
        }),
        bank.add(Patch {
            mono: true,
            lfo,
            chorus: 0.5,
            ..Patch::new(fm::BRASS.into())
        }),
        bank.add(Patch {
            reverb: 0.3,
            ..Patch::new(fm::GLASS_BELL.into())
        }),
        bank.add(Patch {
            lfo,
            chorus: 1.0,
            ..Patch::new(supersaw::BIG_CHORD.into())
        }),
        bank.add(Patch {
            gate: 10.0,
            ..Patch::new(drum::CLAP.into())
        }),
        bank.add(Patch {
            gate: 10.0,
            mono: true,
            ..Patch::new(drum::OPEN_HAT.into())
        }),
    ];
    let mut synth = Synth::new(48_000, bank, 24);
    let mut out = vec![0.0f32; 2 * 1_000];
    COUNTING.with(|c| c.set(true));
    // The count counts.
    drop(std::hint::black_box(Vec::<u8>::with_capacity(8)));
    assert_eq!(ALLOCATIONS.swap(0, Ordering::Relaxed), 1);
    for k in 0..400u32 {
        // More notes than voices, so some are stolen; some with lengths of
        // their own.
        let id = ids[k as usize % ids.len()];
        let note = NoteOn {
            note: 48.0 + (k % 24) as f32,
            velocity: 0.9,
            seed: k,
            pan: 0.0,
            shape: 0.0,
        };
        let length = (k % 3 == 0).then_some(u64::from(k % 7) * 500);
        synth.note(u64::from(k) * 300, id, note, length);
        synth.set_tempo(100.0 + (k % 3) as f32);
        synth.command(Command::Level {
            db: -3.0 * (k % 4) as f32,
        });
        synth.set_chorus(ChorusSettings {
            wave: if k % 2 == 0 {
                Wave::Sine
            } else {
                Wave::Triangle
            },
            taps: 1 + k as usize % 3,
            ..ChorusSettings::default()
        });
        // A patch edited as it sounds: a parameter of the same engine, or
        // another engine (its voices fading out); a held note let go.
        if k % 5 == 0 {
            let mut to = *synth.patch(id).expect("in the bank");
            to.set(k as usize % to.count(), (k % 11) as f32 / 10.0);
            if k % 10 == 0 {
                let name = AnyParams::ENGINES[k as usize / 10 % AnyParams::ENGINES.len()];
                to.engine = AnyParams::of_engine(name).expect("an engine");
            }
            synth.command(Command::Patch { patch: id, to });
            synth.command(Command::Release {
                frame: synth.frame(),
                patch: id,
                note: Some(note.note),
                tag: jodel_synth::Tag::NONE,
            });
        }
        synth.render_to(&mut out[..2 * 300]);
    }
    let _ = synth.voices().count();
    COUNTING.with(|c| c.set(false));
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert!(synth.stolen() > 0, "no voice was stolen");
}
