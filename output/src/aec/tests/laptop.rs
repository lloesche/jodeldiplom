//! Laptops: speakers with a limiter and soft clipping, a reverberant room,
//! singers over the echo.

use super::*;

/// A laptop's speakers in a living room: a limiter and soft clipping on
/// the way out, no bass, a long reverberant tail (`rt60` seconds), the
/// echo arriving `delay_ms` late.
pub(super) fn laptop(reference: &[f32], delay_ms: f32, rt60: f32) -> Vec<f32> {
    // Speaker: high-pass, limiter (fast attack, slow release), tanh.
    let mut hp_x = 0.0f32;
    let mut hp_y = 0.0f32;
    let mut env = 0.0f32;
    let a = (-2.0 * PI * 150.0 / RATE as f32).exp();
    let spoken: Vec<f32> = reference
        .iter()
        .map(|&x| {
            hp_y = a * (hp_y + x - hp_x);
            hp_x = x;
            let level = hp_y.abs();
            let k = if level > env { 0.3 } else { 0.0005 };
            env += (level - env) * k;
            let gain = (0.2 / env.max(1e-6)).min(1.0);
            (1.8 * hp_y * gain).tanh() / 1.8
        })
        .collect();
    // Room: the direct sound, then a diffuse tail.
    let delay = (delay_ms / 1000.0 * RATE as f32) as usize;
    let tail = (rt60 * RATE as f32) as usize;
    let mut noise = Noise(5);
    let mut h = vec![0.0f32; delay + tail];
    h[delay] = 0.6;
    let decay = 6.91 / (rt60 * RATE as f32);
    for i in 1..tail {
        h[delay + i] = 0.25 * noise.next() * (-(i as f32) * decay).exp();
    }
    convolve(&spoken, &h)
}

/// The meter's reading: a peak falling over 0.3 s, as the insert's.
fn meter(s: &[f32]) -> Vec<f32> {
    let decay = (-1.0 / (RATE as f32 * 0.3)).exp();
    let mut p = 0.0f32;
    s.iter()
        .map(|v| {
            p = v.abs().max(p * decay);
            p
        })
        .collect()
}

/// What the canceller does in a laptop's room: how far the meter falls
/// with the music only (8 to 12 s in, once learnt), how much of a
/// voice singing over it after is kept (dB) and how far over the rest
/// of what comes out (dB), and the delay it found.
struct Laptop {
    meter_db: f32,
    kept_db: f32,
    clean_db: f32,
    delay: Option<f32>,
}

fn laptop_room(delay_ms: f32, rt60: f32, singer: f32) -> Laptop {
    let n = RATE as usize * 16;
    let s = RATE as usize;
    let reference = music(n, 21);
    let echo = laptop(&reference, delay_ms, rt60);
    let sung: Vec<f32> = voice(n, 12 * s).iter().map(|v| singer * v).collect();
    let mut noise = Noise(8);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + sung[i] + 3e-4 * noise.next())
        .collect();
    let (out, aec) = run(&mic, &reference, &reference);
    let w = 8 * s..12 * s;
    let peak = |m: Vec<f32>| m[w.clone()].iter().cloned().fold(0.0f32, f32::max);
    let meter_db = 20.0 * (peak(meter(&mic)) / peak(meter(&out)).max(1e-9)).log10();
    let v = 13 * s..16 * s;
    let (kept_db, clean_db) = kept(&out[v.clone()], &sung[v]);
    let found = Laptop {
        meter_db,
        kept_db,
        clean_db,
        delay: aec.status().delay,
    };
    eprintln!(
        "{delay_ms} ms late, rt60 {rt60} s, the singer at {singer}: the meter falls {meter_db:.1} dB; the voice kept at {kept_db:.1} dB, {clean_db:.1} dB over the rest; the delay found {:?}",
        found.delay
    );
    found
}

/// How much of `voice` is in `out` (dB), and how far over the rest
/// of `out` it is (dB).
pub(super) fn kept(out: &[f32], voice: &[f32]) -> (f32, f32) {
    let a = out.iter().zip(voice).map(|(o, v)| o * v).sum::<f32>()
        / voice.iter().map(|v| v * v).sum::<f32>().max(1e-20);
    let rest: Vec<f32> = out.iter().zip(voice).map(|(o, v)| o - a * v).collect();
    let part: Vec<f32> = voice.iter().map(|v| a * v).collect();
    (
        20.0 * a.abs().max(1e-9).log10(),
        db(power(&part)) - db(power(&rest)),
    )
}

/// The music alone moves the meter by less than a 20 dB step; a voice
/// as loud as the echo comes through whole (within 0.5 dB), 10 dB over
/// the rest; the delay is found to within 15 ms.
fn check(delay_ms: f32, rt60: f32) {
    let found = laptop_room(delay_ms, rt60, 1.0);
    assert!(
        found.meter_db > 20.0,
        "the meter falls {:.1} dB",
        found.meter_db
    );
    assert!(
        found.kept_db.abs() < 0.5,
        "the voice kept at {:.1} dB",
        found.kept_db
    );
    assert!(
        found.clean_db > 10.0,
        "the voice {:.1} dB over the rest",
        found.clean_db
    );
    let delay = found.delay.expect("a delay found") * 1000.0;
    assert!(
        (delay - delay_ms).abs() < 15.0,
        "the delay found {delay:.0} ms"
    );
}

#[test]
fn a_laptop_in_a_small_room() {
    check(40.0, 0.15);
}

#[test]
fn a_laptop_in_a_living_room() {
    check(40.0, 0.4);
}

/// The window moves after the echo's loudest part, for more of the
/// room's tail.
#[test]
fn speakers_further_away() {
    check(120.0, 0.4);
}

/// A wireless speaker: the echo starts past the window as it is, and
/// the delay finder moves it there.
#[test]
fn a_wireless_speaker() {
    check(300.0, 0.3);
}

/// A singer as a singer sounds, from `from`: syllables of 0.15 to 0.5 s
/// with 50 to 150 ms between them, each a voiced tone (its fundamental
/// between 150 and 350 Hz with a vibrato, harmonics to 4 kHz falling
/// 6 dB an octave) rising in 20 ms; every fourth soft (−15 dB); every
/// third starting with a consonant (noise above 3 kHz, 50 ms). Gives
/// the samples, and each syllable's (start, end, soft).
pub(super) fn singer(n: usize, from: usize) -> (Vec<f32>, Vec<(usize, usize, bool)>) {
    let mut out = vec![0.0f32; n];
    let mut syllables = Vec::new();
    let mut noise = Noise(77);
    let mut at = from;
    let mut k = 0usize;
    let rate = RATE as f32;
    while at < n {
        let long = (0.15 + 0.35 * ((k * 7 % 11) as f32 / 10.0)) * rate;
        let end = (at + long as usize).min(n);
        let soft = k % 4 == 3;
        let level = if soft { 0.18 * 0.178 } else { 0.18 };
        let f0 = 150.0 + 200.0 * ((k * 5 % 9) as f32 / 8.0);
        let mut phase = 0.0f32;
        let mut last = 0.0f32;
        for (j, i) in (at..end).enumerate() {
            let t = j as f32 / rate;
            let f = f0 * (1.0 + 0.02 * (2.0 * PI * 5.5 * t).sin());
            phase += 2.0 * PI * f / rate;
            let mut v = 0.0;
            let mut h = 1;
            while h as f32 * f < 4000.0 {
                v += (h as f32 * phase).sin() / h as f32;
                h += 1;
            }
            let attack = (t / 0.02).min(1.0);
            let release = (((end - i) as f32 / rate) / 0.03).min(1.0);
            let mut s = level * 0.5 * v * attack * release;
            if k % 3 == 0 && t < 0.05 {
                let x = noise.next();
                s += level * 1.2 * (x - last);
                last = x;
            }
            out[i] = s;
        }
        syllables.push((at, end, soft));
        at = end + ((0.05 + 0.1 * ((k * 3 % 7) as f32 / 6.0)) * rate) as usize;
        k += 1;
    }
    (out, syllables)
}

/// How a singer comes through a laptop's room (40 ms, 0.4 s of tail),
/// once learnt: each syllable's energy out against its own (dB), the
/// first 60 ms of each (the onsets), the soft ones; and how far the
/// meter falls with the music alone.
struct Sung {
    meter_db: f32,
    median: f32,
    worst: f32,
    onsets: f32,
    soft: f32,
}

fn sung_through(singer_level: f32) -> Sung {
    let n = RATE as usize * 20;
    let s = RATE as usize;
    let reference = music(n, 21);
    let echo = laptop(&reference, 40.0, 0.4);
    let (voice, syllables) = singer(n, 10 * s);
    let voice: Vec<f32> = voice.iter().map(|v| singer_level * v).collect();
    let mut noise = Noise(8);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let (out, _) = run(&mic, &reference, &reference);
    let w = 6 * s..10 * s;
    let peak = |m: Vec<f32>| m[w.clone()].iter().cloned().fold(0.0f32, f32::max);
    let meter_db = 20.0 * (peak(meter(&mic)) / peak(meter(&out)).max(1e-9)).log10();
    let kept_in = |r: std::ops::Range<usize>| db(power(&out[r.clone()])) - db(power(&voice[r]));
    let mut all: Vec<f32> = syllables.iter().map(|&(a, b, _)| kept_in(a..b)).collect();

    let mut onsets: Vec<f32> = syllables
        .iter()
        .map(|&(a, b, _)| kept_in(a..(a + s * 6 / 100).min(b)))
        .collect();
    let mut soft: Vec<f32> = syllables
        .iter()
        .filter(|x| x.2)
        .map(|&(a, b, _)| kept_in(a..b))
        .collect();
    let median = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let sung = Sung {
        meter_db,
        median: median(&mut all),
        worst: all[all.len() / 10],
        onsets: median(&mut onsets),
        soft: median(&mut soft),
    };
    let sung_power = syllables
        .iter()
        .map(|&(a, b, _)| power(&voice[a..b]))
        .sum::<f32>()
        / syllables.len() as f32;
    let over = db(power(&echo[w.clone()])) - db(sung_power);
    eprintln!(
        "a singer at {singer_level} (the echo {over:.1} dB over them): the meter falls {:.1} dB; syllables kept {:.1} dB (median), {:.1} dB (worst tenth), onsets {:.1} dB, soft ones {:.1} dB",
        sung.meter_db, sung.median, sung.worst, sung.onsets, sung.soft
    );
    sung
}

/// A singer through a laptop's room, from twice as loud as the music's
/// echo at the microphone to 24 dB under it: no syllable loses anything
/// (the soft ones and the onsets neither: the canceller never takes the
/// singer out), and the music alone moves the meter by less than a
/// 20 dB step. Under the echo, what the filter leaves of it adds to the
/// syllables (a few dB to the soft ones of a singer 24 dB under it).
#[test]
fn a_singer_through_a_laptop() {
    for level in [2.0, 1.0, 0.3, 0.1] {
        let sung = sung_through(level);
        assert!(
            sung.meter_db > 20.0,
            "the meter falls {:.1} dB",
            sung.meter_db
        );
        assert!(sung.median > -0.5, "syllables kept {:.1} dB", sung.median);
        assert!(
            sung.worst > -1.0,
            "the worst tenth kept {:.1} dB",
            sung.worst
        );
        assert!(sung.onsets > -1.0, "onsets kept {:.1} dB", sung.onsets);
        assert!(sung.soft > -1.0, "soft syllables kept {:.1} dB", sung.soft);
    }
}

/// A singer quieter than the music's echo: heard whole.
#[test]
fn a_quiet_singer() {
    let found = laptop_room(40.0, 0.4, 0.5);
    assert!(
        found.meter_db > 20.0,
        "the meter falls {:.1} dB",
        found.meter_db
    );
    assert!(
        found.kept_db.abs() < 0.5,
        "the voice kept at {:.1} dB",
        found.kept_db
    );
}

/// A band of held chords over a held bass (as a song's band plays with
/// its singer turned down), a singer in its key over its echo through
/// a laptop, the microphone with an offset and a rumble of its own:
/// the filters stay well within the microphone's level, the echo is
/// still taken out once the singer stops, and the singer is kept. (On a
/// real recording the filter that learns grew beyond numbers without
/// [`WHITE_GAIN_MOST`]; this signal does not drive it that far, so it
/// holds what must always hold.)
#[test]
fn a_steady_band_and_a_singer_never_blow_the_filters_up() {
    let n = RATE as usize * 20;
    let s = RATE as usize;
    let mut noise = Noise(5);
    let reference: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            // A chord changing every two seconds, held, under it its
            // root two octaves down, loud (most of a song's power is
            // its bass), and a little noise. The bass, held, whitened
            // is all but gone, and the laptop does not play it: there
            // its steps were the ones too large.
            let root = [220.0, 196.0, 247.0, 165.0][(t / 2.0) as usize % 4];
            let chord: f32 = [1.0, 1.26, 1.5, 2.0]
                .iter()
                .map(|r| (2.0 * PI * root * r * t).sin())
                .sum();
            let bass = (2.0 * PI * root / 4.0 * t).sin();
            0.08 * chord + 0.3 * bass + 0.002 * noise.next()
        })
        .collect();
    let echo = laptop(&reference, 80.0, 0.4);
    // The singer on the chord's notes (in its key, as a singer is):
    // their harmonics land on the band's held tones.
    let mut syllables = Vec::new();
    let voice: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            if !(4.0..16.0).contains(&t) {
                return 0.0;
            }
            let root = [220.0, 196.0, 247.0, 165.0][(t / 2.0) as usize % 4];
            let note = root * [1.0, 1.26, 1.5][((t * 2.0) as usize) % 3];
            let f = note * (1.0 + 0.005 * (2.0 * PI * 5.5 * t).sin());
            let on = ((t * 2.0).fract() < 0.85) as u8 as f32;
            let v: f32 = (1..8)
                .map(|h| (2.0 * PI * f * h as f32 * t).sin() / h as f32)
                .sum();
            0.12 * on * v
        })
        .collect();
    for k in 8..32 {
        let a = k * s / 2;
        syllables.push((a + s / 20, a + s * 8 / 20, false));
    }
    // A laptop's microphone: an offset of its own and a slow rumble
    // under the room (none of it the music's).
    let mic: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            echo[i] + voice[i] + 0.01 + 0.004 * (2.0 * PI * 0.7 * t).sin()
        })
        .collect();
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; n];
    let mut block = [0.0f32; BLOCK];
    let mut worst = f32::MIN;
    for b in 0..n / BLOCK {
        let r = b * BLOCK..(b + 1) * BLOCK;
        aec.process(
            &mic[r.clone()],
            &reference[r.clone()],
            &reference[r.clone()],
            &mut block,
        );
        out[r].copy_from_slice(&block);
        worst = worst.max(aec.shadow_level / aec.mic_level.max(1e-12));
    }
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(
        worst < 10.0,
        "the shadow left {:.1} dB over the microphone",
        db(worst)
    );
    let w = 17 * s..20 * s;
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
    assert!(removed > 3.0, "the echo removed {removed:.1} dB after");
    let kept: Vec<f32> = syllables
        .iter()
        .filter(|&&(_, b, _)| b < 16 * s)
        .map(|&(a, b, _)| db(power(&out[a..b])) - db(power(&voice[a..b])))
        .collect();
    let median = {
        let mut k = kept.clone();
        k.sort_by(|a, b| a.total_cmp(b));
        k[k.len() / 2]
    };
    eprintln!("a steady band: shadow at most {:.1} dB over the microphone, removed {removed:.1} dB, syllables kept {median:.1} dB", db(worst));
    assert!(median > -0.5, "syllables kept {median:.1} dB");
}
