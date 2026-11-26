//! The echo canceller on real music, without hardware: a song plays
//! through a simulated laptop (small speakers with a limiter and soft
//! clipping, a room's tail), the microphone hears that and a singer (a
//! separated voice from another song), and the canceller takes the song's
//! echo out of what it hears, the residual suppressor after it (as the game
//! hears the voice). The microphone is never played, so there is no loop.
//! Prints, a second at a time, how it is doing, then what each kept of the
//! singer and what it took out of the song. The song plays as one channel
//! on both speakers (the laptop is made up of one).
//!
//! ```text
//! cargo run --release -p jodel-playback --example aec_room -- SONG VOICE.wav [ECHO_DB] [DELAY_MS] [SECONDS] [linear]
//! ```
//!
//! `VOICE.wav` is mono 16-bit (any rate); `ECHO_DB` how loud the song's
//! echo is at the microphone against the singer (10 by default: laptop
//! speakers beside it, the singer at arm's length); `DELAY_MS` the way
//! from what is written to the device to the microphone (30); `SECONDS`
//! how much of the song (60); `linear`: speakers with no limiter and no
//! clipping. The voice starts 10 s in, after the canceller has heard the
//! song alone.

use std::f32::consts::PI;
use std::path::Path;

use jodel_output::aec::{Whole, BLOCK};

const RATE: u32 = 48_000;

fn resample(mono: &[f64], from: u32) -> Vec<f32> {
    let mut r = jodel_playback::Resampler::new(from, RATE, 1);
    let mut out = Vec::new();
    r.process_f64(mono, &mut out);
    r.finish_f64(&mut out);
    out.into_iter().map(|v| v as f32).collect()
}

fn song(path: &Path, seconds: f64) -> Vec<f32> {
    let mut source = jodel_playback::Source::open(path).unwrap();
    let (rate, channels) = (
        source.info().rate,
        usize::from(source.info().channels).max(1),
    );
    let limit = (seconds * f64::from(rate)) as usize;
    let mut mono = Vec::new();
    let mut block = Vec::new();
    while mono.len() < limit {
        block.clear();
        if source.read_unquantized(&mut block).unwrap() == 0 {
            break;
        }
        for f in block.chunks(channels) {
            // The source's samples are in 24-bit units.
            mono.push(f.iter().sum::<f64>() / channels as f64 / 8_388_608.0);
        }
    }
    mono.truncate(limit);
    resample(&mono, rate)
}

/// A mono 16-bit WAV.
fn wav(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let wav = jodel_dsp::wav::parse(&bytes).expect("a WAV file");
    let rate = wav.format.rate;
    let mono: Vec<f64> = wav
        .pcm16()
        .expect("16-bit samples")
        .iter()
        .map(|&s| f64::from(s) / 32_768.0)
        .collect();
    resample(&mono, rate)
}

struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

/// What a laptop's microphone hears of `reference`: small speakers (no
/// bass, a limiter, soft clipping), then the room: the direct sound
/// `delay_ms` late, a tail of 0.3 s.
fn laptop(reference: &[f32], delay_ms: f32, linear: bool) -> Vec<f32> {
    let (mut hp_x, mut hp_y, mut env) = (0.0f32, 0.0f32, 0.0f32);
    let a = (-2.0 * PI * 150.0 / RATE as f32).exp();
    let spoken: Vec<f32> = reference
        .iter()
        .map(|&x| {
            hp_y = a * (hp_y + x - hp_x);
            hp_x = x;
            let level = hp_y.abs();
            env += (level - env) * if level > env { 0.3 } else { 0.0005 };
            if linear {
                return hp_y;
            }
            let gain = (0.2 / env.max(1e-6)).min(1.0);
            (1.8 * hp_y * gain).tanh() / 1.8
        })
        .collect();
    let delay = (delay_ms / 1000.0 * RATE as f32) as usize;
    let tail = (0.3 * RATE as f32) as usize;
    let mut noise = Noise(5);
    // Sparse taps: the direct sound, then the tail every fourth sample.
    let mut taps = vec![(delay, 0.6f32)];
    let decay = 6.91 / (0.3 * RATE as f32);
    for i in (1..tail).step_by(4) {
        taps.push((delay + i, 0.6 * noise.next() * (-(i as f32) * decay).exp()));
    }
    (0..spoken.len())
        .map(|t| {
            taps.iter()
                .take_while(|(k, _)| *k <= t)
                .map(|&(k, v)| v * spoken[t - k])
                .sum()
        })
        .collect()
}

fn power(s: &[f32]) -> f64 {
    s.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / s.len().max(1) as f64
}

fn db(p: f64) -> f64 {
    10.0 * p.max(1e-20).log10()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let echo_db: f64 = args.get(2).map_or(10.0, |s| s.parse().unwrap());
    let delay_ms: f32 = args.get(3).map_or(30.0, |s| s.parse().unwrap());
    let seconds: f64 = args.get(4).map_or(60.0, |s| s.parse().unwrap());
    let music = song(Path::new(&args[0]), seconds);
    let n = music.len() / BLOCK * BLOCK;
    let linear = args.get(5).is_some_and(|a| a == "linear");
    let echo = laptop(&music[..n], delay_ms, linear);
    // The singer from 10 s in, scaled so the echo is `echo_db` over them
    // (over their voiced stretches).
    let start = 10 * RATE as usize;
    let raw = wav(Path::new(&args[1]));
    let mut voice = vec![0.0f32; n];
    for (i, v) in raw.iter().enumerate() {
        if start + i < n {
            voice[start + i] = *v;
        }
    }
    let frame = RATE as usize / 50;
    let voiced: Vec<bool> = (0..n / frame)
        .map(|f| power(&voice[f * frame..(f + 1) * frame]) > 1e-5)
        .collect();
    let pick = |s: &[f32], on: bool| -> Vec<f32> {
        (start / frame..n / frame)
            .filter(|&f| voiced[f] == on)
            .flat_map(|f| s[f * frame..(f + 1) * frame].to_vec())
            .collect()
    };
    let gain = (db(power(&pick(&echo, true))) - db(power(&pick(&voice, true))) - echo_db) / 20.0;
    let gain = 10f64.powf(gain) as f32;
    for v in &mut voice {
        *v *= gain;
    }
    let mut noise = Noise(9);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let mut learnt_at = None;
    let cleaned = Whole::new(RATE).run(&mic, &music[..n], &music[..n], |end, s| {
        if s.learnt && learnt_at.is_none() {
            learnt_at = Some(end as f32 / RATE as f32);
        }
        if end % (5 * RATE as usize) < BLOCK {
            println!(
                "{:4.0} s  learnt {:5}  removed {:5.1} dB  filter {:5.1} dB  delay {:?}",
                end as f32 / RATE as f32,
                s.learnt,
                s.removed_db,
                s.filter_db,
                s.delay
            );
        }
    });
    let (out, suppressed) = (cleaned.cancelled, cleaned.suppressed);
    println!("learnt after {learnt_at:?} s");
    for (name, out) in [("cancelled", &out), ("suppressed", &suppressed)] {
        // What came out where the singer sang, against the singer: the
        // share kept (its projection), and how far over the rest of what
        // came out.
        let (o, v) = (pick(out, true), pick(&voice, true));
        let a = o.iter().zip(&v).map(|(o, v)| f64::from(o * v)).sum::<f64>()
            / v.iter().map(|v| f64::from(v * v)).sum::<f64>();
        let rest: Vec<f32> = o.iter().zip(&v).map(|(o, v)| o - a as f32 * v).collect();
        let over = db(a * a * power(&v)) - db(power(&rest));
        // Where the singer did not sing: how much of the song's echo went.
        let removed = db(power(&pick(&echo, false))) - db(power(&pick(out, false)));
        println!(
            "{name}: the singer kept at {:+.2} dB, {over:.1} dB over the rest; the echo removed {removed:.1} dB where they did not sing",
            20.0 * a.abs().max(1e-9).log10()
        );
    }
}
