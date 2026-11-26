//! The sound conductor playing a scripted game, to listen to: a minute of
//! a song's beat analysis (112 beats a minute in 4/4, E minor, lines of
//! two bars and a bar's gap, a stretch where the grid is unsure) with
//! words hit in a row up the rungs, a rescue shot, the streak lost at ×16,
//! built up again to ×32 (the payout), phrases rated up to the top (the
//! jackpot) and the diploma's count-up. Rendered in 10 ms periods as the
//! effects path would, the clock made from the analysis every period, the
//! conductor's notes going straight to its synth (its frames the song's).
//!
//! Into OUT_DIR: `game.wav` (the conductor alone), `game_with_backing.wav`
//! (with a backing of drums, bass, chords and a guide voice on the sung
//! lines, made on a synth of its own, so the beat and the key can be
//! heard), and in `sounds/` each sound alone: the ladder up to the top,
//! the five stingers, the five tier chimes, the jackpot, the streak lost
//! from each rung, clinks, the zap, a pop, the count-up and the shimmer;
//! in `diploma/` each of the sheet's own sounds alone, and whole diplomas
//! revealed as the screen plays them (frame by frame at 60 a second) for
//! each kind of verdict: a Gold one with a new best, a Silver one short of
//! its best, one without a medal, one not sung from its start, one with
//! nothing scored, and one the demo key sang. Prints each file's length
//! and levels, and what the game cost.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example conductor -- OUT_DIR
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use jodel_rhythm::{Analysis, Region};
use jodel_synth::engines::{analog, drum, formant, supersaw};
use jodel_synth::patch::{Adsr, Lfo, LfoRate};
use jodel_synth::{Bank, NoteOn, Patch, PatchId, Synth};
use jodeldiplom::core::shine::Shone;
use jodeldiplom::core::singing::{rung, Medal, Shown, SongResult};
use jodeldiplom::sfx::conductor::{Clock, Conductor, Diploma, Event, Sheet};
use jodeldiplom::sfx::reveal::RevealSounds;
use jodeldiplom::tiers::Tier;

const RATE: u32 = 48_000;
const TEMPO: f64 = 112.0;
const SECONDS: f64 = 66.0;
/// A device period.
const PERIOD: usize = 480;
/// The tier table's rungs.
const RUNGS: [u32; 5] = [8, 20, 40, 75, 120];
/// The song: two bars before the first line, lines of two bars sung and
/// a bar's gap, nine of them; the words' places in a line, in beats.
const INTRO: f64 = 8.0;
const LINE: f64 = 12.0;
const SUNG: f64 = 8.0;
const LINE_COUNT: usize = 9;
const WORDS: [f64; 11] = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 6.5, 7.0];
/// The guide melody, a line's words: up and down E minor's pentatonic.
const MELODY: [f32; 11] = [
    64.0, 67.0, 69.0, 71.0, 69.0, 67.0, 71.0, 74.0, 71.0, 67.0, 64.0,
];
/// Where the grid is unsure, beats.
const UNSURE: (f64, f64) = (50.0, 62.0);
/// The phrases' tiers, line by line (the top one's line also hits the
/// jackpot).
const PHRASES: [usize; LINE_COUNT] = [0, 1, 1, 2, 3, 2, 3, 4, 4];
/// The word missed (streak lost at ×16) and the one a shot rescues.
const MISSED: usize = 37;
const RESCUED: usize = 24;

fn seconds_a_beat() -> f64 {
    60.0 / TEMPO
}

fn frame(beats: f64) -> u64 {
    (beats * seconds_a_beat() * f64::from(RATE)).round() as u64
}

/// The song's beat analysis: steady beats, a bar every four, sure but in
/// [`UNSURE`].
fn analysis(seconds: f64) -> Analysis {
    let n = (seconds / seconds_a_beat()) as usize + 8;
    let beats: Vec<f64> = (0..n).map(|k| k as f64 * seconds_a_beat()).collect();
    let region = |from: f64, to: f64, confidence| Region {
        start: from * seconds_a_beat(),
        end: to * seconds_a_beat(),
        confidence,
    };
    Analysis {
        duration: seconds,
        downbeats: (0..n).step_by(4).collect(),
        local_bpm: vec![TEMPO; n],
        regions: vec![
            region(0.0, UNSURE.0, 0.9),
            region(UNSURE.0, UNSURE.1, 0.15),
            region(UNSURE.1, n as f64, 0.9),
        ],
        beats,
        ..Analysis::default()
    }
}

/// The sung lines, first and end frames.
fn lines() -> Vec<(u64, u64)> {
    (0..LINE_COUNT)
        .map(|l| {
            let start = INTRO + LINE * l as f64;
            (frame(start), frame(start + SUNG))
        })
        .collect()
}

/// The clock at frame `now` from `a`, with the lines not over yet (none:
/// nothing sung).
fn clock_at(a: &Analysis, now: u64, sung: bool) -> Clock {
    let mut ahead = [None; Clock::LINES];
    if sung {
        let lines = lines().into_iter().filter(|&(_, end)| end > now);
        for (slot, line) in ahead.iter_mut().zip(lines) {
            *slot = Some(line);
        }
    }
    let t = now as f64 / f64::from(RATE);
    Clock::from_analysis(a, t, now, f64::from(RATE), ahead)
}

/// The game's events, in order: their frames and what.
fn script() -> Vec<(u64, Event)> {
    let mut events = Vec::new();
    let mut streak = 0u32;
    let mut word = 0usize;
    for (line, &tier) in PHRASES.iter().enumerate() {
        let start = INTRO + LINE * line as f64;
        for (w, &at) in WORDS.iter().enumerate() {
            let f = frame(start + at);
            let pan = -0.6 + 1.2 * w as f32 / (WORDS.len() - 1) as f32;
            if word == RESCUED {
                events.push((f.saturating_sub(frame(0.25)), Event::Zap { pan }));
            }
            if word == MISSED {
                if let Some(from) = rung(streak) {
                    events.push((f, Event::Lost(from)));
                }
                streak = 0;
            } else {
                streak += 1;
                events.push((
                    f,
                    Event::Word {
                        streak,
                        pan,
                        size: 0.0,
                    },
                ));
            }
            word += 1;
        }
        let end = frame(start + SUNG + 0.25);
        let tier = Tier::ALL[tier];
        events.push((end, Event::Phrase(tier)));
        if tier == Tier::TOP {
            events.push((end, Event::Jackpot));
        }
    }
    // The diploma, after the last line.
    let after = INTRO + LINE * LINE_COUNT as f64;
    events.extend(count_up(frame(after + 0.5), 24));
    for (k, tier) in [1, 2, 3, 4].into_iter().enumerate() {
        let pan = -0.3 + 0.2 * k as f32;
        let gem = Event::Diploma(Diploma::Gem {
            tier: Some(Tier::ALL[tier]),
            pan,
        });
        events.push((frame(after + 4.0 + 0.5 * k as f64), gem));
    }
    let points = Diploma::Points(Some(Tier::TOP));
    events.push((frame(after + 6.5), Event::Diploma(points)));
    events.sort_by_key(|&(f, _)| f);
    events
}

/// The diploma's count-up of `of` ticks from `at`, faster and faster as
/// the reveal draws it (two seconds in all).
fn count_up(at: u64, of: u32) -> Vec<(u64, Event)> {
    (0..of)
        .map(|k| {
            let share = (f64::from(k) / f64::from(of)).powf(0.6);
            let t = at + (2.0 * share * f64::from(RATE)) as u64;
            (t, Event::Diploma(Diploma::Tick { k, of }))
        })
        .collect()
}

/// What playing cost: the wall seconds in all, each period's (sorted:
/// its events planned and its frames rendered), the most voices sounding.
struct Cost {
    took: f64,
    periods: Vec<f64>,
    voices: usize,
}

/// `events` played on a conductor in `key` for `seconds`, in periods, the
/// clock from `clock_of`; the output and what it cost.
fn play(
    events: &[(u64, Event)],
    key: &jodel_rhythm::key::Key,
    seconds: f64,
    clock_of: &dyn Fn(u64) -> Clock,
) -> (Vec<f32>, Cost) {
    let (mut c, mut synth) = Conductor::new(RATE);
    c.set_key(Some(key), 0);
    let frames = (seconds * f64::from(RATE)) as usize / PERIOD * PERIOD;
    let mut out = vec![0.0f32; 2 * frames];
    let mut next = 0;
    let (mut voices, mut periods) = (0, Vec::new());
    let started = Instant::now();
    for period in out.as_chunks_mut::<{ 2 * PERIOD }>().0 {
        let one = Instant::now();
        let now = synth.frame();
        c.set_clock(&mut synth, clock_of(now));
        while next < events.len() && events[next].0 < now + PERIOD as u64 {
            let (at, e) = events[next];
            c.event(&mut synth, at, e);
            next += 1;
        }
        synth.render_to(period);
        periods.push(one.elapsed().as_secs_f64());
        voices = voices.max(synth.voices().count());
    }
    let took = started.elapsed().as_secs_f64();
    assert_eq!(synth.dropped() + c.dropped(), 0, "commands dropped");
    periods.sort_by(f64::total_cmp);
    (
        out,
        Cost {
            took,
            periods,
            voices,
        },
    )
}

/// The backing: drums on the grid, a bass and chords a bar each (Em, C,
/// G, D), and a guide voice singing the lines' melody.
fn backing(seconds: f64) -> Vec<f32> {
    let mut bank = Bank::new();
    let kick = bank.add(Patch {
        gate: 2.0,
        level: -6.0,
        ..Patch::new(drum::KICK.into())
    });
    let snare = bank.add(Patch {
        gate: 2.0,
        level: -10.0,
        reverb: 0.15,
        ..Patch::new(drum::CLAP.into())
    });
    let hat = bank.add(Patch {
        gate: 1.0,
        level: -18.0,
        ..Patch::new(drum::HAT.into())
    });
    let bass = bank.add(Patch {
        adsr: Adsr::hit(1.2),
        level: -8.0,
        ..Patch::new(analog::BASS.into())
    });
    let pad = bank.add(Patch {
        adsr: Adsr::held(0.3, 0.6),
        level: -22.0,
        reverb: 0.3,
        chorus: 0.5,
        ..Patch::new(supersaw::CHOIR_PAD.into())
    });
    let voice = bank.add(Patch {
        adsr: Adsr::held(0.04, 0.12),
        lfo: Lfo {
            rate: LfoRate::Hz(5.5),
            pitch: 0.15,
            delay: 0.15,
            fade: 0.2,
            ..Lfo::NONE
        },
        level: -10.0,
        reverb: 0.25,
        ..Patch::new(
            formant::Params {
                vowel: formant::A,
                ..formant::AWW
            }
            .into(),
        )
    });
    let mut notes: Vec<(u64, PatchId, NoteOn, Option<u64>)> = Vec::new();
    let mut cue = |at: u64, p, note: f32, velocity: f32, length: Option<u64>| {
        let seed = notes.len() as u32 + 1;
        let n = NoteOn {
            note,
            velocity,
            seed,
            pan: 0.0,
        };
        notes.push((at, p, n, length));
    };
    let beats = (seconds * TEMPO / 60.0) as u32;
    // E, C, G, D a bar each.
    const ROOTS: [i32; 4] = [40, 36, 43, 38];
    const THIRDS: [i32; 4] = [3, 4, 4, 4];
    for b in 0..beats {
        let at = frame(f64::from(b));
        if b % 2 == 0 {
            cue(at, kick, 36.0, 1.0, None);
        } else {
            cue(at, snare, 60.0, 0.8, None);
        }
        cue(at, hat, 60.0, 0.6, None);
        cue(at + frame(0.5), hat, 60.0, 0.4, None);
        if b % 4 == 0 {
            let bar = (b / 4) as usize % 4;
            let root = ROOTS[bar];
            cue(at, bass, root as f32, 0.9, Some(frame(1.8)));
            cue(at + frame(2.0), bass, root as f32, 0.7, Some(frame(1.8)));
            for interval in [12, 12 + THIRDS[bar], 19] {
                let note = (root + 12 + interval) as f32;
                cue(at, pad, note, 0.7, Some(frame(3.9)));
            }
        }
    }
    for line in 0..LINE_COUNT {
        let start = INTRO + LINE * line as f64;
        for (w, &at) in WORDS.iter().enumerate() {
            let next = WORDS.get(w + 1).copied().unwrap_or(SUNG);
            let length = frame((next - at) * 0.9);
            cue(frame(start + at), voice, MELODY[w], 0.8, Some(length));
        }
    }
    notes.sort_by_key(|n| n.0);
    let mut s = Synth::new(RATE, bank, 64);
    s.set_tempo(TEMPO as f32);
    let frames = (seconds * f64::from(RATE)) as usize / PERIOD * PERIOD;
    let mut out = vec![0.0; 2 * frames];
    let mut next = 0;
    for period in out.as_chunks_mut::<{ 2 * PERIOD }>().0 {
        let end = s.frame() + PERIOD as u64;
        while next < notes.len() && notes[next].0 < end {
            let (at, p, n, length) = notes[next];
            assert!(s.note(at, p, n, length));
            next += 1;
        }
        s.render_to(period);
    }
    out
}

/// Writes `out` (interleaved stereo) into `dir` as `name`, a 16-bit WAV,
/// printing its length and levels.
fn save(dir: &Path, name: &str, out: &[f32]) {
    std::fs::create_dir_all(dir).expect("the output folder");
    let path = dir.join(format!("{name}.wav"));
    let words: Vec<i16> = out
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
        .collect();
    let bytes = jodeldiplom::keep::wav(&words, RATE, 2);
    durable::write_atomic(&path, &bytes).expect("the WAV written");
    let db = |x: f32| 20.0 * x.max(1e-9).log10();
    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len().max(1) as f32).sqrt();
    println!(
        "{:<44} {:6.2} s  peak {:6.1} dBFS  rms {:6.1} dBFS",
        path.display(),
        out.len() as f64 / 2.0 / f64::from(RATE),
        db(peak),
        db(rms)
    );
}

/// Each sound alone into `dir`, on the song's grid with nothing sung.
fn sounds(dir: &Path, key: &jodel_rhythm::key::Key, a: &Analysis) {
    let one = |name: &str, events: Vec<(u64, Event)>, seconds: f64| {
        let (out, _) = play(&events, key, seconds, &|now| clock_at(a, now, false));
        save(dir, name, &out);
    };
    let word = |streak, pan| Event::Word {
        streak,
        pan,
        size: 0.0,
    };
    // Two thirds into a beat: the pickup at once, the hit on the next.
    let at = frame(2.0 / 3.0);
    let mut ladder: Vec<(u64, Event)> = (1..=60u32)
        .map(|streak| (frame(0.5 * f64::from(streak)), word(streak, 0.0)))
        .collect();
    ladder.push((frame(31.0), Event::Lost(Tier::TOP)));
    one("ladder_to_the_top", ladder, 36.0);
    for (r, &streak) in (1u32..).zip(&RUNGS) {
        one(
            &format!("stinger_rung{r}"),
            vec![(at, word(streak, 0.0))],
            8.0,
        );
        // The rung reached (its stinger), then lost two bars on.
        let from = Event::Lost(Tier::ALL[r as usize - 1]);
        let lost = vec![(0, word(streak, 0.0)), (frame(8.0), from)];
        one(&format!("lost_from_rung{r}"), lost, 10.0);
    }
    for t in Tier::ALL {
        let name = format!("chime_{}", t.name().to_lowercase());
        one(&name, vec![(at, Event::Phrase(t))], 4.0);
    }
    one("jackpot", vec![(at, Event::Jackpot)], 6.0);
    let clinks = (0..16)
        .map(|k| {
            (
                frame(0.5 * f64::from(k)),
                word(25 + k, -0.5 + k as f32 / 15.0),
            )
        })
        .collect();
    one("words_with_clinks", clinks, 9.0);
    one("zap", vec![(at, Event::Zap { pan: 0.3 })], 1.5);
    one("pop", vec![(at, word(1, 0.0))], 1.0);
    let mut count = count_up(at, 24);
    let points = Diploma::Points(Some(Tier::TOP));
    count.push((frame(7.0), Event::Diploma(points)));
    one("count_up", count, 11.0);
    // Held on by a word every half beat, then lost.
    let mut shimmer: Vec<(u64, Event)> = (0..18u32)
        .map(|k| (at + frame(0.5 * f64::from(k)), word(35 + k, 0.0)))
        .collect();
    shimmer.push((frame(9.0), Event::Lost(Tier::named("Amazing"))));
    one("shimmer_held_then_lost", shimmer, 12.0);
}

/// A result of `points`, with `hit` words of 45 and a best streak of
/// `streak`, its phrases in `tiers`, its `medal`, the `best` before, sung
/// whole or not.
fn result(
    (points, hit, streak): (u32, u32, u32),
    tiers: &[Option<Tier>],
    (medal, best, whole): (Option<Medal>, Option<u32>, bool),
) -> SongResult {
    let phrase = |&tier: &Option<Tier>| Shone {
        line: 0,
        end: 1.0,
        value: 0.9,
        top: 0.9,
        tier,
    };
    SongResult {
        title: "Song".into(),
        points,
        hit,
        words: 45,
        streak,
        phrases: tiers.iter().map(phrase).collect(),
        medal,
        best,
        whole,
        demo: false,
    }
}

/// Each sound of the sheet alone into `dir`, and whole diplomas revealed.
fn diploma(dir: &Path, key: &jodel_rhythm::key::Key) {
    let free = |_| Clock::free(RATE);
    for sound in Sheet::ALL {
        let sheet = Diploma::Sheet {
            sound,
            gain: 1.0,
            pan: 0.0,
        };
        let (out, _) = play(&[(0, Event::Diploma(sheet))], key, 4.0, &free);
        save(dir, &sound.name(), &out);
    }
    let named = |t: &str| Some(Tier::named(t));
    let six = [
        Some(Tier::TOP),
        None,
        named("Good"),
        named("Great"),
        named("Amazing"),
        Some(Tier::TOP),
    ];
    let gold = result(
        (12_400, 41, 23),
        &six,
        (Some(Medal::Gold), Some(9_000), true),
    );
    let silver = result(
        (7_300, 34, 12),
        &six[..4],
        (Some(Medal::Silver), Some(9_000), true),
    );
    let cheer = result(
        (2_100, 14, 5),
        &[None, named("Good"), None],
        (None, None, true),
    );
    let whole = result((5_600, 30, 9), &six[..3], (None, Some(4_000), false));
    let silent = result((0, 0, 0), &[], (None, None, false));
    let demo = SongResult {
        demo: true,
        ..result((9_100, 40, 30), &six, (Some(Medal::Gold), None, true))
    };
    let runs = [
        ("run-diploma-gold-new-best", gold),
        ("run-diploma-silver", silver),
        ("run-diploma-cheer", cheer),
        ("run-diploma-whole", whole),
        ("run-diploma-silent", silent),
        ("run-diploma-demo", demo),
    ];
    for (name, result) in runs {
        let shown = Shown::new(result, 1, 0);
        let mut reveal = RevealSounds::default();
        let mut events = Vec::new();
        let settled = shown.reveal.settled();
        for f in 0..(60.0 * (settled + 1.0)) as u64 {
            let at = f * u64::from(RATE) / 60;
            let heard = reveal.frame(Some((&shown, shown.time(f * 1000 / 60))));
            events.extend(heard.into_iter().map(|d| (at, Event::Diploma(d))));
        }
        let (out, _) = play(&events, key, settled + 3.0, &free);
        save(dir, name, &out);
    }
}

fn main() {
    let Some(root) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: conductor OUT_DIR");
        std::process::exit(2);
    };
    // E minor, as the analysis would find it.
    let key = jodel_rhythm::key::Key::on(4, true);
    let a = analysis(SECONDS + 10.0);
    let events = script();
    let (game, cost) = play(&events, &key, SECONDS, &|now| clock_at(&a, now, true));
    let period = PERIOD as f64 / f64::from(RATE);
    let share = |q: f64| {
        let k = ((cost.periods.len() - 1) as f64 * q) as usize;
        100.0 * cost.periods[k] / period
    };
    println!(
        "the game: {SECONDS} s planned and rendered in {:.3} s ({:.0}x real time, {:.2} % \
         of a core); a {} ms period's share of its time: median {:.2} %, 99th percentile \
         {:.2} %, 99.9th {:.2} %, the longest {:.1} %; {} voices at most",
        cost.took,
        SECONDS / cost.took,
        100.0 * cost.took / SECONDS,
        1_000.0 * period,
        share(0.5),
        share(0.99),
        share(0.999),
        share(1.0),
        cost.voices
    );
    save(&root, "game", &game);
    let band = backing(SECONDS);
    let mut mixed: Vec<f32> = game.iter().zip(&band).map(|(g, b)| g + 0.6 * b).collect();
    let peak = mixed.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak > 0.95 {
        for s in &mut mixed {
            *s *= 0.95 / peak;
        }
    }
    save(&root, "game_with_backing", &mixed);
    sounds(&root.join("sounds"), &key, &a);
    diploma(&root.join("diploma"), &key);
}
