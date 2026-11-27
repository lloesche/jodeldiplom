//! A short scripted song sung through the whole player, to listen to the
//! game's sounds as the desk plays them: the player on a memory device in
//! real time (as the window runs it: its frames drawn every 1/60 s, the
//! game's sounds played with each), a song of its own (112 beats a minute
//! in 4/4, E minor: drums, a bass and chords made on our synth, as loud as
//! a modern track's instrumental, about -11 LUFS), its words and their
//! notes given, the beat grid and the key as the analysis would find
//! them, Karaoke mode started and the demo key held, so the game
//! hears a perfect singer: every word pops up the ladder, the rungs come
//! on the beat, a word missed alone is rescued by a shot, two missed lose
//! the streak (from ×16), it builds up again to ×32 (the payout); each
//! line's phrase is given a tier, rising to the top's jackpot; then the
//! result's diploma is revealed on the device kept open, its paper, pen,
//! stamps and tunes, its ticks, gems, points and streak's rung all by the
//! synth.
//!
//! Writes `game_through_player.wav` into OUT_DIR (16-bit, the device's
//! rate: what the device played, song and sounds; the song plays from
//! autoplay first, then again from Next), the song alone as `song.wav`
//! and its lyrics into `OUT_DIR/work`, the game's sounds alone (the song
//! taken out where it was sung) as `game_sounds_alone.wav`, and prints the
//! result and how loud the song and the sounds were.
//!
//! ```text
//! cargo run -p jodeldiplom --example game_through_player -- OUT_DIR
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodel_output::{Capabilities, SampleFormat, SimClock, StreamFormat};
use jodel_playback::PlayState;
use jodel_rhythm::waveform::{Melody, Waveform};
use jodel_rhythm::{Analysis, Region};
use jodel_synth::dynamics::CompressorSettings;
use jodel_synth::engines::{analog, drum, supersaw};
use jodel_synth::patch::Adsr;
use jodel_synth::{Bank, NoteOn, Patch, Synth};
use jodeldiplom::app::{App, AppOptions, MemoryOutput, Output};
use jodeldiplom::core::karaoke::{self, SongLyrics};
use jodeldiplom::core::keymap::Op;
use jodeldiplom::core::shine::Shone;
use jodeldiplom::core::{Action, Rhythm, Rhythmed, Side};
use jodeldiplom::tiers::Tier;

const RATE: u32 = 48_000;
const TEMPO: f64 = 112.0;
/// Two bars before the first line; lines of two bars sung and a bar's
/// gap; the words' places in a line, in beats, and their notes (MIDI, up
/// and down E minor's pentatonic).
const INTRO: f64 = 8.0;
const LINE: f64 = 12.0;
const SUNG: f64 = 8.0;
const LINES: usize = 10;
const WORDS: [f64; 11] = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 6.5, 7.0];
const MELODY: [f64; 11] = [
    64.0, 67.0, 69.0, 71.0, 69.0, 67.0, 71.0, 74.0, 71.0, 67.0, 64.0,
];
/// The words missed (first and last, by their places in the song):
/// one alone, a line's last (the next line's first word's shot rescues
/// it), then a line's last two (the streak lost from ×16).
const MISSED: [(usize, usize); 2] = [(21, 21), (42, 43)];
/// Each line's phrase's tier (a place in the tier table).
const PHRASES: [usize; LINES] = [0, 1, 1, 2, 3, 1, 2, 3, 4, 4];
/// How long the song goes on after its last line, beats.
const OUTRO: f64 = 8.0;

fn beat_seconds() -> f64 {
    60.0 / TEMPO
}

fn seconds(beats: f64) -> f64 {
    beats * beat_seconds()
}

fn length() -> f64 {
    seconds(INTRO + LINE * LINES as f64 + OUTRO)
}

/// Each word: its line, its place there, its start and end (seconds), its
/// note.
fn words() -> Vec<(usize, usize, f64, f64, f64)> {
    let mut out = Vec::new();
    for l in 0..LINES {
        let start = INTRO + LINE * l as f64;
        for (w, &at) in WORDS.iter().enumerate() {
            let next = WORDS.get(w + 1).copied().unwrap_or(SUNG);
            let (from, to) = (seconds(start + at), seconds(start + at + 0.9 * (next - at)));
            out.push((l, w, from, to, MELODY[w]));
        }
    }
    out
}

/// The song's beat analysis: steady beats, a bar every four, sure, in E
/// minor.
fn analysis() -> Analysis {
    let n = (length() / beat_seconds()) as usize + 8;
    let beats: Vec<f64> = (0..n).map(|k| k as f64 * beat_seconds()).collect();
    Analysis {
        duration: length(),
        downbeats: (0..n).step_by(4).collect(),
        local_bpm: vec![TEMPO; n],
        regions: vec![Region {
            start: 0.0,
            end: length() + 10.0,
            confidence: 0.9,
        }],
        beats,
        key: Some(jodel_rhythm::key::Key::on(4, true)),
        ..Analysis::default()
    }
}

/// The song: kick and clap on the beats, hats on the eighths, a bass and
/// chords a bar each (Em, C, G, D), made on a synth of its own and
/// mastered loud on its bus (squeezed 3:1 from -22 dBFS, lifted 12 dB into
/// the limiter at -1 dBFS): about -11 LUFS, as a modern track's
/// instrumental is.
fn song() -> Vec<f32> {
    let mut bank = Bank::new();
    let kick = bank.add(Patch {
        gate: 2.0,
        level: -4.0,
        ..Patch::new(drum::KICK.into())
    });
    let clap = bank.add(Patch {
        gate: 2.0,
        level: -2.0,
        reverb: 0.15,
        ..Patch::new(drum::CLAP.into())
    });
    let hat = bank.add(Patch {
        gate: 1.0,
        level: -8.0,
        ..Patch::new(drum::HAT.into())
    });
    let bass = bank.add(Patch {
        adsr: Adsr::hit(1.2),
        level: 0.0,
        ..Patch::new(analog::BASS.into())
    });
    let pad = bank.add(Patch {
        adsr: Adsr::held(0.3, 0.6),
        level: 0.0,
        reverb: 0.3,
        chorus: 0.5,
        ..Patch::new(supersaw::CHOIR_PAD.into())
    });
    let frame = |beats: f64| (seconds(beats) * f64::from(RATE)).round() as u64;
    let mut s = Synth::new(RATE, bank, 64);
    s.set_tempo(TEMPO as f32);
    s.set_compressor(CompressorSettings {
        threshold: -22.0,
        ratio: 3.0,
        knee: 6.0,
        attack: 0.01,
        release: 0.15,
        makeup: 12.0,
    });
    let mut seed = 0;
    let mut cue = |s: &mut Synth, at: u64, p, note: f32, velocity: f32, length: Option<u64>| {
        seed += 1;
        let n = NoteOn {
            note,
            velocity,
            seed,
            pan: 0.0,
            shape: 0.0,
        };
        assert!(s.note(at, p, n, length));
    };
    const ROOTS: [i32; 4] = [40, 36, 43, 38];
    const THIRDS: [i32; 4] = [3, 4, 4, 4];
    let beats = (length() / beat_seconds()) as u32;
    let total = frame(f64::from(beats)) as usize;
    let mut out = vec![0.0f32; 2 * total];
    let period = 2 * 4_800;
    for (k, chunk) in out.chunks_mut(period).enumerate() {
        let from = (k * period / 2) as u64;
        let to = from + (chunk.len() / 2) as u64;
        for b in 0..beats {
            let at = frame(f64::from(b));
            if at < from || at >= to {
                continue;
            }
            let drum = if b % 2 == 0 { kick } else { clap };
            cue(
                &mut s,
                at,
                drum,
                if b % 2 == 0 { 36.0 } else { 60.0 },
                0.9,
                None,
            );
            cue(&mut s, at, hat, 60.0, 0.6, None);
            cue(&mut s, at + frame(0.5), hat, 60.0, 0.4, None);
            if b % 4 == 0 {
                let bar = (b / 4) as usize % 4;
                let root = ROOTS[bar];
                cue(&mut s, at, bass, root as f32, 0.9, Some(frame(1.8)));
                let second = at + frame(2.0);
                cue(&mut s, second, bass, root as f32, 0.7, Some(frame(1.8)));
                for interval in [12, 12 + THIRDS[bar], 19] {
                    let note = (root + 12 + interval) as f32;
                    cue(&mut s, at, pad, note, 0.7, Some(frame(3.9)));
                }
            }
        }
        s.render_to(chunk);
    }
    out
}

/// `samples` (stereo, interleaved) as a 16-bit WAV at `path`.
fn save(path: &Path, samples: &[f32]) {
    let words: Vec<i16> = samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32_767.0).round() as i16)
        .collect();
    durable::write_atomic(path, &jodel_dsp::wav::pcm16(RATE, 2, &words)).expect("written");
    let db = |x: f32| 20.0 * x.max(1e-9).log10();
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    println!(
        "{}: {:.1} s, peak {:.1} dBFS",
        path.display(),
        samples.len() as f64 / 2.0 / f64::from(RATE),
        db(peak)
    );
}

/// Where in `played` (stereo, interleaved) `song` plays from its start,
/// near frame `near` (within a quarter of a second), in frames, and at
/// what gain: where its second second lines up best.
fn find(played: &[f32], song: &[f32], near: usize) -> (usize, f32) {
    let second = 2 * RATE as usize;
    let excerpt = &song[second..2 * second];
    let fit = |at: usize| -> f32 {
        let got = &played[2 * at + second..][..excerpt.len()];
        got.iter().zip(excerpt).map(|(a, b)| a * b).sum()
    };
    let reach = RATE as usize / 4;
    let at = (near.saturating_sub(reach)..near + reach)
        .max_by(|&a, &b| fit(a).total_cmp(&fit(b)))
        .unwrap_or(near);
    let got = &played[2 * at..][..song.len().min(played.len() - 2 * at)];
    let dot: f32 = got.iter().zip(song).map(|(a, b)| a * b).sum();
    let energy: f32 = song[..got.len()].iter().map(|v| v * v).sum();
    (at, dot / energy.max(1e-12))
}

/// Draws `app`'s frames as the window does (every 1/60 s) until `done`,
/// for `wait` at most; whether it came.
fn frames(app: &mut App, wait: Duration, mut done: impl FnMut(&mut App) -> bool) -> bool {
    let end = Instant::now() + wait;
    while Instant::now() < end {
        if done(app) {
            return true;
        }
        app.pump_wait(Duration::from_millis(16));
        app.heard_for_frame();
    }
    done(app)
}

fn main() {
    let Some(root) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: game_through_player OUT_DIR");
        std::process::exit(2);
    };
    let work = root.join("work");
    std::fs::create_dir_all(&work).expect("the output folder");
    let path = work.join("song.wav");
    save(&path, &song());
    let text: Vec<String> = (0..LINES)
        .map(|_| vec!["la"; WORDS.len()].join(" "))
        .collect();
    let text = text.join("\n");
    durable::write_atomic(&work.join("song.lrc"), text.as_bytes()).expect("the lyrics");

    let mut caps = Capabilities::new();
    caps.insert(StreamFormat::new(RATE, 2, SampleFormat::S24In32Low));
    let memory = MemoryOutput::new(caps, SimClock::Realtime);
    let mut options = AppOptions::new(work.clone());
    options.output = Output::Memory(memory.clone());
    options.config.beat_voting = false;
    options.config.beat_method = jodel_rhythm::Method::Flux;
    options.devices = Arc::new(|| Ok(Vec::new()));
    options.mic = Arc::new(|_| Err("no microphone".into()));
    options.mics = Arc::new(|| Ok(Vec::new()));
    options.separator = || None;
    options.autoplay = Some(path.clone());
    let mut app = App::new(options);
    // As a window of 960 × 684 points (the old 120 × 36 cells) reports.
    let size = iced::Size::new(960.0, 684.0);
    app.state.fit = jodeldiplom::gui::metrics::fit(size, &app.state, 8.0);
    let wait = Duration::from_secs(60);
    let playing = |a: &App| a.state.playing.state == PlayState::Playing;
    assert!(frames(&mut app, wait, |a| playing(a)
        && matches!(a.state.rhythm.get(&path), Some(Rhythm::Ready(_)))));
    // The grid, the key and the notes as given.
    let frames_of = (length() * 100.0) as usize + 100;
    let mut melody = Melody {
        cents: vec![0; frames_of],
        sure: vec![0; frames_of],
    };
    for &(_, _, from, to, note) in &words() {
        for i in (from * 100.0) as usize..(to * 100.0) as usize {
            (melody.cents[i], melody.sure[i]) = ((note * 100.0) as u16, 255);
        }
    }
    let waveform = Waveform {
        melody: Some(melody),
        ..Default::default()
    };
    let rhythm = Rhythmed::new(analysis(), Some(waveform));
    app.state
        .rhythm
        .insert(path.clone(), Rhythm::Ready(Arc::new(rhythm)));
    let mut lyrics = jodel_lyrics::Lyrics::parse_text(&text, jodel_lyrics::Source::Lrclib);
    for &(l, w, from, to, _) in &words() {
        let word = &mut lyrics.lines[l].words[w];
        (word.start, word.end) = (Some(from), Some(to));
    }
    let found = jodeldiplom::work::lyrics::Answer::Found {
        lyrics,
        timed: None,
    };
    app.state
        .karaoke
        .lyrics
        .insert(path.clone(), SongLyrics::from_answer(found));
    app.dispatch(Action::ShowSide(Side::Karaoke));
    assert!(frames(&mut app, wait, |a| karaoke::not_singable(
        &a.state, 0
    )
    .is_none()));
    app.dispatch(Action::Op(Op::KaraokeEnter));
    app.dispatch(Action::Op(Op::KaraokeNext));
    // Next plays the song from its start under the countdown: about where
    // on the device that is (from where it is heard, its last second).
    let mut zero = None;
    let started = |a: &mut App| {
        let into = a.state.heard();
        if playing(a) && (0.5..3.0).contains(&into) {
            let heard = a.desk().map_or(0, |d| d.master.timebase.heard);
            zero = Some(heard.saturating_sub((into * f64::from(RATE)) as u64) as usize);
        }
        a.state.karaoke.countdown.is_none() && playing(a)
    };
    assert!(frames(&mut app, wait, started), "Next");
    app.dispatch(Action::Op(Op::KaraokeDemo));
    // Sung to its end: the demo let go over the words missed, each line's
    // phrase given its tier as its last word is over.
    let words = words();
    let mut rated = 0;
    // Where the singer humming got to.
    let mut hummed: Option<f64> = None;
    let deadline = Instant::now() + Duration::from_secs_f64(length() + 30.0);
    let shown = |a: &App| karaoke::result_now(&a.state).is_some();
    while !shown(&app) {
        assert!(Instant::now() < deadline, "the song ends");
        app.pump_wait(Duration::from_millis(16));
        app.heard_for_frame();
        let state = &mut app.state;
        let t = state.karaoke.lag.sung_clock(state.heard());
        // Over a miss the demo is let go (and stays so a while after, so it
        // starts again where the song is), and a singer heard in its place
        // humming no note at all.
        let miss = MISSED.iter().find_map(|&(first, last)| {
            let (from, to) = (words[first].2, words[last].3);
            (t > from - 0.1 && t < to + 0.7).then_some((from, to))
        });
        match (miss, state.karaoke.demo.held()) {
            (Some(_), true) => app.dispatch(Action::DemoLetGo),
            (None, false) if state.playing.state == PlayState::Playing => {
                app.dispatch(Action::Op(Op::KaraokeDemo))
            }
            _ => {}
        }
        let state = &mut app.state;
        if let Some((from, to)) = miss {
            let mut at = hummed.filter(|&a| a >= from).unwrap_or(from);
            while at <= t.min(to) {
                state.karaoke.singing.hear(at, None);
                at += jodel_rhythm::yin::HOP;
            }
            hummed = Some(at);
        }
        let line_end = |l: usize| seconds(INTRO + LINE * l as f64 + SUNG);
        if rated < LINES && t > line_end(rated) + 0.4 {
            let shone = Shone {
                line: rated,
                end: line_end(rated),
                value: 0.9,
                top: 0.9,
                tier: Some(Tier::ALL[PHRASES[rated]]),
            };
            state.karaoke.singing.decided(t, shone);
            rated += 1;
        }
    }
    // The diploma revealed, and a while after it.
    let (settled, r) = karaoke::result_now(&app.state)
        .map(|(s, _)| (s.reveal.settled(), s.result.clone()))
        .expect("shown");
    let tiers: Vec<&str> = r
        .phrases
        .iter()
        .filter_map(|p| Some(p.tier?.name()))
        .collect();
    println!(
        "the result: {} points, {} of {} words, the best streak {}, {:?}, the phrases {tiers:?}",
        r.points, r.hit, r.words, r.streak, r.medal
    );
    let until = Instant::now() + Duration::from_secs_f64(settled + 3.0);
    frames(&mut app, wait, |_| Instant::now() >= until);
    app.shutdown();
    let streams = memory.captures().last().expect("a device").list();
    let samples: Vec<f32> = (streams[0].sink.bytes().as_chunks::<4>().0.iter())
        .map(|b| ((i32::from_le_bytes(*b) << 8) >> 8) as f32 / 8_388_608.0)
        .collect();
    save(&root.join("game_through_player.wav"), &samples);
    // The song taken out where it was sung (its last play): the game's
    // sounds as the device played them.
    let song = song();
    let (at, gain) = find(&samples, &song, zero.expect("the song heard"));
    let mut alone = samples[2 * at..].to_vec();
    for (x, s) in alone.iter_mut().zip(&song) {
        *x -= gain * s;
    }
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let db = |x: f32| 20.0 * x.max(1e-9).log10();
    let sung = song.len().min(alone.len());
    let intro = 2 * (seconds(INTRO) * f64::from(RATE)) as usize;
    println!(
        "sung from {:.2} s at {gain:.3}: the song {:.1} dBFS rms; what is left of it before the \
         first word {:.1}; the game's sounds {:.1} dBFS rms while sung, {:.1} after (the \
         diploma)",
        at as f64 / f64::from(RATE),
        db(rms(&song)),
        db(rms(&alone[..intro])),
        db(rms(&alone[intro..sung])),
        db(rms(&alone[sung..]))
    );
    save(&root.join("game_sounds_alone.wav"), &alone);
}
