//! The host: notes start on their frame to the sample; voices are taken
//! oldest-released first, fading out without a click, even two in one
//! block; a mono retrigger does not click either, and slides to its note
//! where its patch slides; a note's own shape moves its sound; a release
//! ends a voice;
//! the queue drops what it cannot hold; the spreads come from the seed; the
//! glides, the filter envelope, the tremolo and an LFO in beats move what
//! they should; a note's own length overrides the gate; the pool takes 128
//! voices; the bus's level comes before the limiter, gliding while a
//! sound plays.

mod common;

use common::measure::{biggest_jump, centroid, cents, magnitude, pitch, rms};
use common::{left, on, render, RATE};
use jodel_synth::engines::analog;
use jodel_synth::patch::{Adsr, Glide, Lfo, LfoRate, Spread};
use jodel_synth::{Bank, Command, NoteOn, Patch, PatchId, Synth, Tag, CUES};

/// A plain sine held `gate` seconds, released over `release`.
fn sine(gate: f32, release: f32) -> Patch {
    Patch {
        adsr: Adsr {
            release,
            ..Adsr::default()
        },
        gate,
        ..Patch::new(analog::SINE.into())
    }
}

/// The same with a soft attack, held long.
fn soft() -> Patch {
    Patch {
        adsr: Adsr {
            attack: 0.02,
            ..Adsr::default()
        },
        ..sine(10.0, 0.05)
    }
}

fn synth(patches: &[Patch], voices: usize) -> (Synth, Vec<PatchId>) {
    let mut bank = Bank::new();
    let ids = patches.iter().map(|p| bank.add(*p)).collect();
    (Synth::new(RATE, bank, voices), ids)
}

/// The largest step of a plain `soft()` sine on `note`, once it holds.
fn steady_jump(note: f32) -> f32 {
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(note, 1), None);
    biggest_jump(&left(&render(&mut s, 24_000))[4_800..])
}

#[test]
fn a_note_starts_on_its_frame_to_the_sample() {
    for at in [0u64, 1, 31, 32, 1_000, 4_097] {
        let (mut s, ids) = synth(&[sine(1.0, 0.1)], 8);
        assert!(s.note(at, ids[0], on(69.0, 1), None));
        let out = left(&render(&mut s, 6_000));
        let first = out.iter().position(|x| *x != 0.0).expect("it sounds");
        // The ramp starts from 0 on the frame itself; the limiter holds the
        // output back by its latency.
        assert_eq!(first as u64, at + 1 + s.latency() as u64, "{at}");
    }
}

#[test]
fn rendering_in_pieces_is_rendering_at_once() {
    let make = || {
        let (mut s, ids) = synth(&[sine(0.05, 0.05)], 8);
        for k in 0..20u64 {
            let n = NoteOn {
                velocity: 0.8,
                pan: 0.3,
                ..on(60.0 + k as f32, k as u32)
            };
            s.note(k * 1_234, ids[0], n, None);
        }
        s
    };
    let whole = render(&mut make(), 30_000);
    let mut s = make();
    let mut pieces = Vec::new();
    for n in [1usize, 17, 31, 32, 33, 500, 7, 29_379] {
        pieces.extend(render(&mut s, n));
    }
    assert_eq!(s.frame(), 30_000);
    assert_eq!(whole, pieces);
}

#[test]
fn a_voice_is_taken_from_the_oldest_let_go() {
    let (mut s, ids) = synth(&[sine(10.0, 5.0), sine(0.01, 5.0)], 32);
    // An old note let go (ringing in its long release), then the pool
    // filled with held ones.
    s.note(0, ids[1], on(40.0, 0), None);
    for k in 1..32u64 {
        s.note(k, ids[0], on(50.0 + k as f32, 0), None);
    }
    render(&mut s, 2_000);
    assert_eq!(s.voices().count(), 32);
    assert!(s.voices().any(|v| v.note == 40.0 && !v.held));
    s.note(2_000, ids[0], on(100.0, 0), None);
    render(&mut s, 64);
    assert_eq!(s.voices().count(), 32);
    assert!(!s.voices().any(|v| v.note == 40.0), "the released one went");
    assert!(s.voices().any(|v| v.note == 100.0));
    // With every voice held, the oldest goes.
    s.note(2_100, ids[0], on(101.0, 0), None);
    render(&mut s, 64);
    assert!(!s.voices().any(|v| v.note == 51.0), "the oldest went");
    assert_eq!(s.stolen(), 2);
}

#[test]
fn a_voice_taken_fades_out_without_a_click() {
    // One voice: a held A4, taken mid-block by an E5.
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(69.0, 1), None);
    s.note(10_013, ids[0], on(76.0, 2), None);
    let out = left(&render(&mut s, 20_000));
    let jump = biggest_jump(&out[4_800..]);
    assert!(jump < 1.5 * steady_jump(76.0), "{jump}");
    // The A4 fades out after the frame, not before it.
    assert!(out[10_013].abs() > 0.0 && out[10_000..10_013].iter().any(|x| x.abs() > 0.1));
    assert_eq!(s.stolen(), 1);
}

#[test]
fn two_notes_on_one_voice_in_one_block_fade_the_first() {
    // One voice, a held A4; two notes land on it in one block: the A4 fades
    // out (never cut), the first of the two never sounds, the second plays.
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(69.0, 1), None);
    s.note(9_605, ids[0], on(72.0, 2), None);
    s.note(9_620, ids[0], on(76.0, 3), None);
    let out = left(&render(&mut s, 20_000));
    let jump = biggest_jump(&out[4_800..]);
    assert!(jump < 1.5 * steady_jump(76.0), "{jump}");
    let notes: Vec<f32> = s.voices().map(|v| v.note).collect();
    assert_eq!(notes, [76.0]);
    let got = pitch(&out[14_400..19_200], RATE as f32);
    assert!(cents(got, 659.255).abs() < 5.0, "{got}");
}

#[test]
fn a_mono_retrigger_does_not_click() {
    let mono = Patch {
        mono: true,
        ..soft()
    };
    for to in [69.0, 74.0] {
        let (mut s, ids) = synth(&[mono], 4);
        s.note(0, ids[0], on(69.0, 1), None);
        for k in 1..6u64 {
            s.note(k * 3_001, ids[0], on(to, k as u32), None);
        }
        let out = left(&render(&mut s, 24_000));
        let jump = biggest_jump(&out[2_400..]);
        assert!(jump < 1.5 * steady_jump(to), "{to}: {jump}");
        assert_eq!(s.voices().count(), 1);
    }
}

/// A held sine retriggered on its own note twice in one block sounds as if
/// it was not: no sample between the two is left out.
#[test]
fn mono_retriggers_in_one_block_leave_no_gap() {
    let mono = Patch {
        mono: true,
        ..sine(10.0, 0.05)
    };
    let play = |retrigger: bool| {
        let (mut s, ids) = synth(&[mono], 4);
        s.note(0, ids[0], on(69.0, 1), None);
        if retrigger {
            // Both in the block from frame 3_200.
            s.note(3_205, ids[0], on(69.0, 2), None);
            s.note(3_220, ids[0], on(69.0, 3), None);
        }
        left(&render(&mut s, 6_000))
    };
    let (plain, twice) = (play(false), play(true));
    let worst = plain
        .iter()
        .zip(&twice)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "{worst}");
}

/// A note held until it is let go: it sounds on, and from its release's
/// frame it falls to silence and ends; a release before it changes
/// nothing.
#[test]
fn a_held_note_is_let_go_on_its_frame() {
    let held = Patch {
        mono: true,
        adsr: Adsr::held(0.0, 0.05),
        gate: 3_600.0,
        ..Patch::new(analog::SINE.into())
    };
    let (mut s, ids) = synth(&[held], 4);
    let release = |frame| Command::Release {
        frame,
        patch: ids[0],
        tag: Tag::NONE,
    };
    assert!(s.command(release(10)), "let go before it starts: nothing");
    s.note(100, ids[0], on(69.0, 1), None);
    let out = left(&render(&mut s, 24_000));
    assert!(rms(&out[20_000..]) > 0.1, "held on");
    assert!(s.voices().all(|v| v.held));
    assert!(s.command(release(30_000)));
    let out = left(&render(&mut s, 24_000));
    assert!(rms(&out[..5_000]) > 0.1, "until its frame");
    assert!(rms(&out[12_000..]) < 1e-4, "silent after");
    assert_eq!(s.voices().count(), 0, "ended");
}

/// What waits can be called off by its tag, or all of it; a synth's room
/// for cues is what it was made with.
#[test]
fn waiting_cues_are_called_off_by_tag() {
    // Notes of a hundredth of a second, so each is heard alone.
    let (_, ids) = synth(&[sine(0.01, 0.005)], 1);
    let mut bank = Bank::new();
    bank.add(sine(0.01, 0.005));
    let mut s = Synth::with_cues(RATE, bank, 8, 3);
    let note = |frame, tag| Command::Note {
        frame,
        patch: ids[0],
        note: on(69.0, 1),
        length: None,
        tag: Tag(tag),
    };
    assert!(s.command(note(1_000, 1)));
    assert!(s.command(note(2_000, 2)));
    assert!(s.command(note(3_000, 1)));
    assert!(!s.command(note(4_000, 2)), "room for three");
    assert_eq!(s.dropped(), 1);
    assert!(s.command(Command::Cancel { tag: Some(Tag(1)) }));
    // Untagged: not called off by its tag.
    assert!(s.command(note(5_000, 0)));
    assert!(s.command(Command::Cancel {
        tag: Some(Tag::NONE)
    }));
    let out = left(&render(&mut s, 6_000));
    let heard = |from: usize| rms(&out[from..from + 900]) > 0.01;
    assert!(!heard(1_100) && heard(2_100) && !heard(3_100) && heard(5_000));
    s.note(7_000, ids[0], on(69.0, 2), None);
    s.command(Command::Cancel { tag: None });
    assert!(rms(&left(&render(&mut s, 3_000))) == 0.0);
}

#[test]
fn a_release_ends_the_voice_in_silence() {
    let (mut s, ids) = synth(&[sine(0.1, 0.2)], 8);
    s.note(100, ids[0], on(69.0, 9), None);
    let out = render(&mut s, 48_000);
    assert_eq!(s.voices().count(), 0);
    // Sounding for its gate and release (0.3 s), then nothing at all.
    let last = out.iter().rposition(|x| *x != 0.0).unwrap() / 2;
    assert!((last as f32 / 48_000.0 - 0.3).abs() < 0.01, "{last}");
}

#[test]
fn a_note_of_its_own_length_overrides_the_gate() {
    let (mut s, ids) = synth(&[sine(1.0, 0.1)], 8);
    s.note(0, ids[0], on(69.0, 9), Some(4_800));
    let out = render(&mut s, 48_000);
    let last = out.iter().rposition(|x| *x != 0.0).unwrap() / 2;
    assert!((last as f32 / 48_000.0 - 0.2).abs() < 0.01, "{last}");
}

#[test]
fn a_full_queue_drops_and_counts() {
    let (mut s, ids) = synth(&[sine(0.1, 0.1)], 8);
    for k in 0..CUES as u64 {
        assert!(s.note(1_000_000 + k, ids[0], on(60.0, 0), None));
    }
    assert!(!s.note(5, ids[0], on(60.0, 0), None));
    assert_eq!(s.dropped(), 1);
}

#[test]
fn the_spreads_come_from_the_seed() {
    let spread = Patch {
        spread: Spread {
            detune: 50.0,
            level: 6.0,
            start: 0.01,
            lfo_phase: 1.0,
        },
        ..sine(0.2, 0.1)
    };
    let once = |seed: u32| {
        let (mut s, ids) = synth(&[spread], 8);
        s.note(0, ids[0], on(69.0, seed), None);
        render(&mut s, 9_600)
    };
    assert_eq!(once(5), once(5));
    let (a, b) = (left(&once(5)), left(&once(6)));
    assert_ne!(a, b);
    let pa = pitch(&a[2_400..], 48_000.0);
    let pb = pitch(&b[2_400..], 48_000.0);
    assert!(cents(pa, 440.0).abs() <= 50.5 && cents(pb, 440.0).abs() <= 50.5);
    assert!((pa - pb).abs() > 0.01, "{pa} {pb}");
}

#[test]
fn a_mono_patch_keeps_one_voice() {
    let mono = Patch {
        mono: true,
        ..sine(1.0, 0.1)
    };
    let (mut s, ids) = synth(&[mono], 8);
    for k in 0..5u64 {
        s.note(k * 1_000, ids[0], on(60.0 + k as f32, 0), None);
    }
    render(&mut s, 6_000);
    let notes: Vec<f32> = s.voices().map(|v| v.note).collect();
    assert_eq!(notes, [64.0]);
}

/// A mono patch with a slide: a retrigger glides from the note before to
/// its own over the slide (halfway there half the slide in), then holds
/// it; without one it jumps.
#[test]
fn a_mono_retrigger_slides_to_its_note() {
    let legato = |slide: f32| Patch {
        mono: true,
        slide,
        ..sine(10.0, 0.05)
    };
    let play = |slide: f32| {
        let (mut s, ids) = synth(&[legato(slide)], 4);
        s.note(0, ids[0], on(57.0, 1), None);
        s.note(9_600, ids[0], on(69.0, 2), None);
        let out = left(&render(&mut s, 24_000));
        let at = |from: usize| pitch(&out[from..from + 1_440], 48_000.0);
        (at(4_800), at(9_600 + 2_400 - 720), at(19_200))
    };
    let (before, half, after) = play(0.1);
    assert!(cents(before, 220.0).abs() < 5.0, "{before}");
    // 0.05 s into a 0.1 s slide over an octave: about half an octave up.
    assert!(cents(half, 220.0 * 2f32.sqrt()).abs() < 300.0, "{half}");
    assert!(cents(after, 440.0).abs() < 5.0, "{after}");
    let (_, jumped, _) = play(0.0);
    assert!(cents(jumped, 440.0).abs() < 5.0, "{jumped}");
}

/// A note's own shape moves what its engine's shape moves, as the patch's
/// glide of it does: a vowel of its own on a formant voice ("i" with its
/// second formant up at 2.3 kHz where "u" has none, "a" with its first at
/// 0.8 kHz where the others have none).
#[test]
fn a_note_s_own_shape_moves_its_vowel() {
    use jodel_synth::engines::formant;
    let voice = Patch::new(
        formant::Params {
            vowel: formant::A,
            ..formant::Params::default()
        }
        .into(),
    );
    let sung = |shape: f32| {
        let (mut s, ids) = synth(&[Patch { gate: 1.0, ..voice }], 4);
        s.note(
            0,
            ids[0],
            NoteOn {
                shape,
                ..on(57.0, 1)
            },
            None,
        );
        let out = left(&render(&mut s, 24_000));
        let x = &out[12_000..14_048];
        (
            magnitude(x, 2_200.0, 48_000.0),
            magnitude(x, 880.0, 48_000.0),
        )
    };
    let (i, a, u) = (sung(-1.0), sung(0.0), sung(1.0));
    assert!(i.0 > 4.0 * u.0, "i {i:?} u {u:?}");
    assert!(
        a.1 > 4.0 * u.1 && a.1 > 4.0 * i.1,
        "i {i:?} a {a:?} u {u:?}"
    );
}

#[test]
fn the_pitch_glide_bends_down_to_the_note() {
    let pew = Patch {
        pitch: Glide {
            from: 12.0,
            time: 0.2,
            curve: 1.0,
        },
        ..sine(1.0, 0.1)
    };
    let (mut s, ids) = synth(&[pew], 8);
    s.note(0, ids[0], on(57.0, 0), None);
    let out = left(&render(&mut s, 24_000));
    let early = pitch(&out[480..2_880], 48_000.0);
    let late = pitch(&out[14_400..19_200], 48_000.0);
    assert!(early > 360.0 && early < 440.0, "{early}");
    assert!(cents(late, 220.0).abs() < 5.0, "{late}");
}

#[test]
fn the_filter_envelope_brightens_then_settles() {
    let pluck = Patch {
        adsr: Adsr {
            decay: 0.3,
            sustain: 0.2,
            ..Adsr::default()
        },
        filter_env: 4.0,
        ..sine(1.0, 0.1)
    };
    let pluck = Patch {
        engine: analog::Params {
            cutoff: 300.0,
            ..analog::Params::default()
        }
        .into(),
        ..pluck
    };
    let (mut s, ids) = synth(&[pluck], 8);
    s.note(0, ids[0], on(48.0, 0), None);
    let out = left(&render(&mut s, 48_000));
    let (early, late) = (
        centroid(&out[0..2_048], 48_000.0),
        centroid(&out[38_000..40_048], 48_000.0),
    );
    assert!(early > 2.0 * late, "{early} {late}");
}

/// The level every 10 ms of `x`.
fn envelope(x: &[f32]) -> Vec<f32> {
    x.chunks(480).map(rms).collect()
}

/// The period of `e` (one value every 10 ms), seconds: its strongest
/// autocorrelation between 0.2 and 1.5 s.
fn period(e: &[f32]) -> f32 {
    let mean = e.iter().sum::<f32>() / e.len() as f32;
    let d: Vec<f32> = e.iter().map(|v| v - mean).collect();
    let corr = |lag: usize| d.iter().zip(&d[lag..]).map(|(a, b)| a * b).sum::<f32>();
    let lag = (20..150)
        .max_by(|&a, &b| corr(a).total_cmp(&corr(b)))
        .unwrap();
    lag as f32 * 0.01
}

#[test]
fn a_tremolo_in_beats_follows_the_tempo() {
    let tremolo = Patch {
        lfo: Lfo {
            rate: LfoRate::Beats(1.0),
            amp: 1.0,
            ..Lfo::default()
        },
        ..sine(10.0, 0.1)
    };
    for (bpm, beat) in [(120.0, 0.5), (60.0, 1.0)] {
        let (mut s, ids) = synth(&[tremolo], 8);
        s.set_tempo(bpm);
        s.note(0, ids[0], on(69.0, 0), None);
        let e = envelope(&left(&render(&mut s, 4 * 48_000)));
        assert!((period(&e) - beat).abs() < 0.03, "{bpm}: {}", period(&e));
        // Full depth: it dips to near silence once a beat.
        let (lo, hi) = e[20..]
            .iter()
            .fold((f32::MAX, 0.0f32), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(lo < 0.1 * hi, "{lo} {hi}");
    }
}

#[test]
fn the_lfo_waits_and_fades_in() {
    let vibrato = Patch {
        lfo: Lfo {
            rate: LfoRate::Hz(6.0),
            amp: 1.0,
            delay: 0.5,
            fade: 0.5,
            ..Lfo::default()
        },
        ..sine(10.0, 0.1)
    };
    let (mut s, ids) = synth(&[vibrato], 8);
    s.note(0, ids[0], on(69.0, 0), None);
    let e = envelope(&left(&render(&mut s, 2 * 48_000)));
    let swing = |w: &[f32]| {
        w.iter().cloned().fold(0.0f32, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min)
    };
    let (before, during, after) = (swing(&e[5..45]), swing(&e[55..95]), swing(&e[110..190]));
    // (A 10 ms window holds no whole number of A4's cycles: a little
    // ripple even with no tremolo.)
    assert!(
        before < 0.02 && during > 0.05 && after > during,
        "{before} {during} {after}"
    );
}

#[test]
fn the_pool_takes_128_voices() {
    let (mut s, ids) = synth(&[sine(10.0, 0.1)], 128);
    for k in 0..128u64 {
        s.note(k * 7, ids[0], on(40.0 + (k % 60) as f32, k as u32), None);
    }
    render(&mut s, 4_800);
    assert_eq!(s.voices().count(), 128);
    assert_eq!(s.stolen(), 0);
    s.note(5_000, ids[0], on(100.0, 0), None);
    render(&mut s, 480);
    assert_eq!((s.voices().count(), s.stolen()), (128, 1));
}

#[test]
fn a_pile_of_sounds_never_passes_the_ceiling() {
    // Sixty-four loud notes at once, a jackpot's worth: the bus's limiter
    // holds them under -1 dBFS.
    let loud = Patch {
        level: 12.0,
        reverb: 1.0,
        ..Patch::new(analog::Params::default().into())
    };
    let (mut s, ids) = synth(&[loud], 64);
    for k in 0..64u64 {
        s.note(
            k * 3,
            ids[0],
            on(36.0 + (k % 36) as f32, k as u32),
            Some(24_000),
        );
    }
    let out = render(&mut s, 48_000);
    let ceiling = 10f32.powf(-1.0 / 20.0);
    let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak <= ceiling + 1e-4 && peak > 0.5 * ceiling, "{peak}");
}

/// A held sine at the bus's level `db` (set before it sounds), its left
/// side over its second second.
fn at_level(db: f32) -> Vec<f32> {
    let (mut s, ids) = synth(&[soft()], 1);
    assert!(s.command(Command::Level { db }));
    s.note(0, ids[0], on(69.0, 1), None);
    left(&render(&mut s, 2 * RATE as usize))[RATE as usize..].to_vec()
}

#[test]
fn the_bus_level_comes_before_the_limiter() {
    let plain = at_level(0.0);
    // After the compressor: the same sound, only lower.
    let down = at_level(-6.0);
    let ratio = rms(&down) / rms(&plain);
    assert!((ratio - 10f32.powf(-6.0 / 20.0)).abs() < 1e-3, "{ratio}");
    // Before the limiter: lifted far, it is held under the ceiling.
    let ceiling = 10f32.powf(-1.0 / 20.0);
    let up = at_level(18.0);
    let peak = up.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak <= ceiling + 1e-4 && peak > 0.8, "{peak}");
    let (mut s, _) = synth(&[soft()], 1);
    assert!(!s.command(Command::Level { db: f32::NAN }));
    assert_eq!(s.level(), 0.0);
}

#[test]
fn the_bus_level_glides_while_a_sound_plays() {
    let (mut s, ids) = synth(&[soft()], 1);
    s.note(0, ids[0], on(69.0, 1), None);
    let before = left(&render(&mut s, RATE as usize / 2));
    let step = biggest_jump(&before[RATE as usize / 4..]);
    assert!(s.command(Command::Level { db: -12.0 }));
    assert!((s.level() + 12.0).abs() < 1e-4);
    let gliding = left(&render(&mut s, 3 * RATE as usize));
    // No jump: the first 50 ms are still about where they were, and no
    // step is bigger than the sine's own.
    let tenth = RATE as usize / 20;
    let start = rms(&gliding[..tenth]) / rms(&before[before.len() - tenth..]);
    assert!(start > 0.97, "{start}");
    assert!(biggest_jump(&gliding) <= step * 1.01);
    // There after its glide.
    let end = rms(&gliding[gliding.len() - tenth..]) / rms(&before[before.len() - tenth..]);
    assert!((end - 10f32.powf(-12.0 / 20.0)).abs() < 0.01, "{end}");
}
