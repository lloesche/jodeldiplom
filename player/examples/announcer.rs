//! The announcer's calls, to listen to and to measure: every word it
//! sings (the tiers' names, the rungs' multipliers) alone in four keys and
//! tempos (and once in Spooky's ghostly voice), the top tier's phrase (its
//! chime, jackpot and call) and the top rung (its stinger, payout and
//! call) as the game plays them, the same rung with a line being sung (the
//! call ducked), the top rung's stinger as the diploma plays it (no call,
//! to set the call against); and, given a song, the
//! game's calls over that song's instrumental (its stems but the vocals,
//! separated on the GPU), on its own beat grid and in its own key.
//!
//! Into OUT_DIR: `alone/` and `moments/`, and with a song `song/`: the
//! instrumental, the game alone over it and the two mixed (16-bit WAVs).
//! Prints for each its length, peak, loudness and the pitch track of what
//! it sings (the voiced stretches' notes, and how far each is from the
//! nearest tone of the key's pentatonic).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example announcer -- OUT_DIR [SONG [FROM_SECONDS]]
//! ```

use std::path::{Path, PathBuf};

use jodel_gpu::GpuSeparator;
use jodel_playback::source::Source;
use jodel_rhythm::grid::When;
use jodel_rhythm::key::Key;
use jodel_rhythm::loudness::Meter;
use jodel_rhythm::stems::{Splitter, Stem};
use jodel_rhythm::{Analysis, Method, Region};
use jodel_synth::Synth;
use jodeldiplom::core::RhythmRecipe;
use jodeldiplom::rhythm::analyze_file;
use jodeldiplom::sfx::conductor::{Called, Clock, Conductor, Diploma, Event};
use jodeldiplom::theme::{Theme, CLASSIC, SPOOKY};
use jodeldiplom::tiers::Tier;

/// A device period, frames.
const PERIOD: usize = 480;
/// How long a stretch of the song is played, seconds.
const STRETCH: f64 = 24.0;

/// A steady song at `bpm` in 4/4 for `seconds`, sure all along.
fn steady(bpm: f64, seconds: f64) -> Analysis {
    let beat = 60.0 / bpm;
    let n = (seconds / beat) as usize + 8;
    let beats: Vec<f64> = (0..n).map(|k| k as f64 * beat).collect();
    Analysis {
        duration: seconds,
        downbeats: (0..n).step_by(4).collect(),
        local_bpm: vec![bpm; n],
        regions: vec![Region {
            start: 0.0,
            end: n as f64 * beat,
            confidence: 0.9,
        }],
        beats,
        ..Analysis::default()
    }
}

/// What a render is played on: the analysis (its seconds from `offset`
/// on the render's frame 0), the key, the sung lines (frames).
struct Song<'a> {
    analysis: &'a Analysis,
    offset: f64,
    key: Key,
    lines: &'a [(u64, u64)],
    lufs: Option<f64>,
    theme: &'a Theme,
}

/// What the conductor does on a frame: an event, or a call alone (landing
/// by the grid's rule as it says).
#[derive(Clone, Copy)]
enum Cue {
    Event(Event),
    Call(Called, When),
}

/// `seconds` of the conductor at `rate` playing `cues` on `song`, in
/// device periods, the clock told each one.
fn play(song: &Song, rate: u32, cues: &[(u64, Cue)], seconds: f64) -> Vec<f32> {
    let (mut c, mut synth): (Conductor, Synth) = Conductor::new(rate, song.theme);
    c.set_key(Some(&song.key), 0);
    c.set_loudness(&mut synth, song.lufs);
    let frames = (seconds * f64::from(rate)) as usize / PERIOD * PERIOD;
    let mut out = vec![0.0f32; 2 * frames];
    let mut next = 0;
    for period in out.as_chunks_mut::<{ 2 * PERIOD }>().0 {
        let now = synth.frame();
        let mut ahead = [None; Clock::LINES];
        let lines = song.lines.iter().filter(|&&(_, end)| end > now);
        for (slot, &line) in ahead.iter_mut().zip(lines) {
            *slot = Some(line);
        }
        let t = song.offset + now as f64 / f64::from(rate);
        let clock = Clock::from_analysis(song.analysis, t, now, f64::from(rate), ahead);
        c.set_clock(&mut synth, clock);
        while next < cues.len() && cues[next].0 < now + PERIOD as u64 {
            let (at, cue) = cues[next];
            match cue {
                Cue::Event(e) => {
                    c.event(&mut synth, at, e);
                }
                Cue::Call(called, when) => {
                    let lands = clock.land(at, when);
                    c.announce(&mut synth, at, lands, called);
                }
            }
            next += 1;
        }
        synth.render_to(period);
    }
    assert_eq!(synth.dropped() + c.dropped(), 0, "commands dropped");
    out
}

/// The integrated loudness of `stereo` (interleaved) at `rate`, LUFS.
fn loudness(stereo: &[f32], rate: u32) -> Option<f64> {
    let (l, r): (Vec<f32>, Vec<f32>) = stereo
        .as_chunks::<2>()
        .0
        .iter()
        .map(|f| (f[0], f[1]))
        .unzip();
    let mut meter = Meter::new(rate);
    meter.push(&l, &r);
    meter.integrated()
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-9).log10()
}

/// The notes' names.
const NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

fn name(midi: f64) -> String {
    let n = midi.round() as i32;
    format!(
        "{}{}",
        NAMES[n.rem_euclid(12) as usize],
        n.div_euclid(12) - 1
    )
}

/// The pitch classes of `key`'s pentatonic.
fn pentatonic(key: &Key) -> [i32; 5] {
    let steps = if key.minor() {
        [0, 3, 5, 7, 10]
    } else {
        [0, 2, 4, 7, 9]
    };
    steps.map(|s| (i32::from(key.tonic()) + s) % 12)
}

/// The pitch (MIDI) of `x` at `rate` by its normalised autocorrelation
/// (80 Hz to 1 kHz), if it is clearly pitched.
fn pitch(x: &[f32], rate: f64) -> Option<f64> {
    let lo = (rate / 1_000.0) as usize;
    let hi = (rate / 80.0) as usize;
    let n = x.len().checked_sub(hi + 1)?;
    let corr = |lag: usize| {
        let (mut s, mut a, mut b) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let (u, v) = (f64::from(x[i]), f64::from(x[i + lag]));
            s += u * v;
            a += u * u;
            b += v * v;
        }
        s / (a * b).sqrt().max(1e-20)
    };
    let c: Vec<f64> = (lo..=hi).map(corr).collect();
    let peak = |k: usize| k > 0 && k + 1 < c.len() && c[k] >= c[k - 1] && c[k] >= c[k + 1];
    let mut k = (1..c.len() - 1)
        .filter(|&k| peak(k))
        .max_by(|&a, &b| c[a].total_cmp(&c[b]))?;
    if c[k] < 0.8 {
        return None;
    }
    // Its period, not a multiple of it: the shortest lag a whole fraction
    // of it as high.
    for d in (2..=4).rev() {
        let near = ((lo + k) as f64 / d as f64).round() as usize;
        if near < lo + 2 {
            continue;
        }
        let found = (near - lo - 2..=near - lo + 2).filter(|&j| peak(j));
        if let Some(j) = found.max_by(|&a, &b| c[a].total_cmp(&c[b])) {
            if c[j] >= 0.95 * c[k] {
                k = j;
                break;
            }
        }
    }
    let (a, b, g) = (c[k - 1], c[k], c[k + 1]);
    let lag = (lo + k) as f64 + 0.5 * (a - g) / (a - 2.0 * b + g);
    Some(69.0 + 12.0 * (rate / lag / 440.0).log2())
}

/// What `out` (stereo at `rate`) measures as a call in `key`: printed.
fn measure(what: &str, out: &[f32], rate: u32, key: &Key) {
    let sr = f64::from(rate);
    let mono: Vec<f32> = out
        .as_chunks::<2>()
        .0
        .iter()
        .map(|f| 0.5 * (f[0] + f[1]))
        .collect();
    let peak = mono.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let step = (0.01 * sr) as usize;
    let rms: Vec<f64> = mono
        .chunks(step)
        .map(|c| (c.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let top = rms.iter().copied().fold(0.0, f64::max);
    let start = rms.iter().position(|&r| r > top * 0.003).unwrap_or(0);
    let end = rms.iter().rposition(|&r| r > top * 0.03).unwrap_or(0);
    let lufs = loudness(out, rate);
    // The pitch track: 40 ms windows every 10 ms where it is loud, its
    // voiced stretches each a note.
    let window = (0.04 * sr) as usize;
    let mut track: Vec<(f64, f64, Vec<f64>)> = Vec::new();
    for (k, &r) in rms.iter().enumerate() {
        let from = k * step;
        if r < top * 0.1 || from + window > mono.len() {
            continue;
        }
        if let Some(p) = pitch(&mono[from..from + window], sr) {
            let t = k as f64 * 0.01;
            match track.last_mut() {
                Some((_, until, ps))
                    if (p - ps[ps.len() - 1]).abs() < 0.7 && t - *until < 0.025 =>
                {
                    *until = t;
                    ps.push(p);
                }
                _ => track.push((t, t, vec![p])),
            }
        }
    }
    let scale = pentatonic(key);
    let notes: Vec<String> = track
        .into_iter()
        .filter(|(from, until, _)| until - from >= 0.03)
        .map(|(from, until, mut ps)| {
            ps.sort_by(f64::total_cmp);
            let p = ps[ps.len() / 2];
            let class = (p.round() as i32).rem_euclid(12);
            let mark = if scale.contains(&class) { "" } else { "?" };
            let cents = (100.0 * (p - p.round())).round();
            format!(
                "{}{mark}{cents:+}c {:.2}-{:.2}s",
                name(p),
                from - start as f64 * 0.01,
                until - start as f64 * 0.01
            )
        })
        .collect();
    println!(
        "{what:<34} {:5.2} s  peak {:6.1} dBFS  {:6.1} LUFS  [{}]",
        (end.saturating_sub(start) + 1) as f64 * 0.01,
        db(f64::from(peak)),
        lufs.unwrap_or(f64::NEG_INFINITY),
        notes.join(", ")
    );
}

fn save(dir: &Path, name: &str, out: &[f32], rate: u32) {
    std::fs::create_dir_all(dir).expect("the output folder");
    let words: Vec<i16> = out
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
        .collect();
    let bytes = jodel_dsp::wav::pcm16(rate, 2, &words);
    durable::write_atomic(&dir.join(format!("{name}.wav")), &bytes).expect("the WAV written");
}

/// A moment of the game to render: its name, its cues, the lines sung.
type Moment<'a> = (&'a str, Vec<(u64, Cue)>, &'a [(u64, u64)]);

/// Each word alone in four keys and tempos (and in Spooky's voice), and
/// the top's moments.
fn alone(dir: &Path) {
    const RATE: u32 = 48_000;
    let keys = [
        (Key::on(4, true), 112.0, &CLASSIC),
        (Key::on(0, false), 90.0, &CLASSIC),
        (Key::on(9, true), 128.0, &CLASSIC),
        (Key::on(6, false), 150.0, &CLASSIC),
        (Key::on(4, true), 112.0, &SPOOKY),
    ];
    for (key, bpm, theme) in keys {
        let a = steady(bpm, 30.0);
        let song = Song {
            analysis: &a,
            offset: 0.0,
            key,
            lines: &[],
            lufs: None,
            theme,
        };
        let beat = 60.0 / bpm * f64::from(RATE);
        let at = (4.0 * beat) as u64;
        let spooky = if *theme == CLASSIC { "" } else { "spooky_" };
        let name = key.name().replace(' ', "_").replace('♯', "s");
        let tag = format!("{spooky}{name}_{bpm}");
        let calls = Tier::ALL
            .into_iter()
            .flat_map(|t| [Called::Name(t), Called::Times(t)]);
        for called in calls {
            let word = match called {
                Called::Name(t) => t.name().to_lowercase(),
                Called::Times(t) => format!("x{}", t.multiplier()),
            };
            let out = play(&song, RATE, &[(at, Cue::Call(called, When::Beat))], 6.0);
            let skip = 2 * (at as usize - PERIOD);
            measure(&format!("{tag} {word}"), &out[skip..], RATE, &key);
            save(
                &dir.join("alone"),
                &format!("{tag}_{word}"),
                &out[skip..],
                RATE,
            );
        }
    }
    // The top's moments as the game plays them, in E minor at 112.
    let key = Key::on(4, true);
    let a = steady(112.0, 30.0);
    let beat = 60.0 / 112.0 * f64::from(RATE);
    let at = (4.0 * beat + beat * 2.0 / 3.0) as u64;
    let top = Tier::TOP.spec().streak;
    let word = |streak| Event::Word {
        streak,
        pan: 0.0,
        size: 0.0,
    };
    let sung = [(at - 2 * beat as u64, at + 8 * beat as u64)];
    let moments: [Moment; 4] = [
        (
            "phrase_diva_with_jackpot",
            vec![
                (at, Cue::Event(Event::Phrase(Tier::TOP))),
                (at, Cue::Event(Event::Jackpot)),
            ],
            &[],
        ),
        (
            "rung_x32",
            vec![
                (at - 4_000, Cue::Event(word(top - 1))),
                (at, Cue::Event(word(top))),
            ],
            &[],
        ),
        (
            "rung_x32_over_a_sung_line",
            vec![
                (at - 4_000, Cue::Event(word(top - 1))),
                (at, Cue::Event(word(top))),
            ],
            &sung,
        ),
        (
            "stinger_x32_as_the_diploma_plays_it_no_call",
            vec![(at, Cue::Event(Event::Diploma(Diploma::Rung(Tier::TOP))))],
            &[],
        ),
    ];
    for (name, cues, lines) in moments {
        let song = Song {
            analysis: &a,
            offset: 0.0,
            key,
            lines,
            lufs: None,
            theme: &CLASSIC,
        };
        let out = play(&song, RATE, &cues, 10.0);
        measure(name, &out, RATE, &key);
        save(&dir.join("moments"), name, &out, RATE);
    }
}

/// `STRETCH` seconds of `path` from `from` seconds: its instrumental (the
/// stems but the vocals) with the game's calls over it.
fn over_song(dir: &Path, path: &Path, from: f64) -> Result<(), String> {
    let recipe = RhythmRecipe {
        methods: Method::ALL.to_vec(),
        stems: false,
    };
    let a = analyze_file(path, &recipe)?;
    let key = a.key.ok_or("the song has no key")?;
    let mut source = Source::open(path).map_err(|e| e.to_string())?;
    source.make_stereo();
    let rate = source.info().rate;
    let first = (from * f64::from(rate)) as u64;
    source.seek(first).map_err(|e| e.to_string())?;
    let frames = (STRETCH * f64::from(rate)) as usize / PERIOD * PERIOD;
    let mut raw = Vec::new();
    while raw.len() < 2 * frames {
        if source.read(&mut raw).map_err(|e| e.to_string())? == 0 {
            break;
        }
    }
    raw.resize(2 * frames, 0);
    let scale = 1.0 / f64::from(1 << 23);
    let (left, right): (Vec<f32>, Vec<f32>) = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|f| {
            (
                (f64::from(f[0]) * scale) as f32,
                (f64::from(f[1]) * scale) as f32,
            )
        })
        .unzip();
    // The instrumental: every stem but the vocals.
    let gpu = GpuSeparator::shared().ok_or("stems need a GPU")?;
    if rate != jodel_rhythm::stems::RATE {
        return Err(format!(
            "the separator takes {} Hz",
            jodel_rhythm::stems::RATE
        ));
    }
    let mut music = vec![0.0f32; 2 * frames];
    let mut take = |b: &jodel_rhythm::stems::StemBlock| {
        for (stem, [left, right]) in b.stems.iter() {
            if stem == Stem::Vocals {
                continue;
            }
            for (i, (l, r)) in left.iter().zip(*right).enumerate() {
                if let Some(f) = music.get_mut(2 * (b.start + i)..2 * (b.start + i) + 2) {
                    f[0] += l;
                    f[1] += r;
                }
            }
        }
    };
    let mut splitter = Splitter::new(gpu);
    splitter.push(&left, &right, &mut take)?;
    splitter.finish(&mut take)?;
    let lufs = loudness(&music, rate);
    println!(
        "{}: {} at {:.1} BPM, the instrumental {:.1} LUFS",
        path.display(),
        key.name(),
        a.summary.map_or(0.0, |s| s.bpm),
        lufs.unwrap_or(f64::NEG_INFINITY)
    );
    // The top phrase twice (its call on a downbeat with room), and the top
    // rung reached while a line is sung (its call ducked).
    let second = |s: f64| (s * f64::from(rate)) as u64;
    let top = Tier::TOP.spec().streak;
    let word = |streak| {
        Cue::Event(Event::Word {
            streak,
            pan: 0.0,
            size: 0.0,
        })
    };
    let cues = [
        (second(2.3), Cue::Event(Event::Phrase(Tier::TOP))),
        (second(2.3), Cue::Event(Event::Jackpot)),
        (second(10.6), word(top - 1)),
        (second(10.9), word(top)),
        (second(17.3), Cue::Event(Event::Phrase(Tier::TOP))),
        (second(17.3), Cue::Event(Event::Jackpot)),
    ];
    let lines = [(second(9.5), second(14.0))];
    let song = Song {
        analysis: &a,
        offset: from,
        key,
        lines: &lines,
        lufs,
        theme: &CLASSIC,
    };
    let game = play(&song, rate, &cues, STRETCH);
    let mixed: Vec<f32> = music.iter().zip(&game).map(|(m, g)| m + g).collect();
    let out = dir.join("song");
    save(&out, "instrumental", &music, rate);
    save(&out, "game_alone", &game, rate);
    save(&out, "mixed", &mixed, rate);
    let peak = mixed.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!("mixed: peak {:.1} dBFS", db(f64::from(peak)));
    // The calls alone, where the game sang them (a phrase's on the
    // jackpot's downbeat, its three beats of room; the rung's on the next
    // beat), against the instrumental there.
    let beats = |at: f64| 60.0 / a.bpm_at(from + at).unwrap_or(120.0) * f64::from(rate);
    let downbeat = |at: f64| When::Downbeat {
        length: 3.0 * beats(at),
    };
    let calls = [
        (
            "diva",
            2.3,
            Cue::Call(Called::Name(Tier::TOP), downbeat(2.3)),
        ),
        (
            "x32 ducked",
            10.9,
            Cue::Call(Called::Times(Tier::TOP), When::Beat),
        ),
        (
            "diva again",
            17.3,
            Cue::Call(Called::Name(Tier::TOP), downbeat(17.3)),
        ),
    ];
    for (what, at, cue) in calls {
        let alone = play(&song, rate, &[(second(at), cue)], STRETCH);
        let span = 2 * second(at - 0.3) as usize..2 * second(at + 3.0) as usize;
        let music_lufs = loudness(&music[span.clone()], rate).unwrap_or(f64::NEG_INFINITY);
        measure(
            &format!("over the song ({music_lufs:.1} LUFS): {what}"),
            &alone[span],
            rate,
            &key,
        );
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("an output folder"));
    alone(&dir);
    if let Some(song) = args.next() {
        let from = args.next().map_or(60.0, |s| s.parse().expect("seconds"));
        if let Err(e) = over_song(&dir, Path::new(&song), from) {
            eprintln!("{song}: {e}");
            std::process::exit(1);
        }
    }
}
