//! The canceller's tests: the echo of made-up music through made-up rooms.

mod laptop;
mod stereo;

use laptop::kept;

use super::*;
use std::f32::consts::PI;

const RATE: u32 = 48_000;

/// A repeatable noise, -0.5 to 0.5.
struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

/// Music-like: noise coloured by a one-pole low-pass, beating
/// tones on it.
fn music(n: usize, seed: u32) -> Vec<f32> {
    let mut noise = Noise(seed);
    let mut lp = 0.0f32;
    (0..n)
        .map(|i| {
            lp += (noise.next() - lp) * 0.2;
            let t = i as f32 / RATE as f32;
            0.4 * lp
                + 0.08 * (2.0 * PI * 220.0 * t).sin()
                + 0.05 * (2.0 * PI * 554.0 * t).sin() * (2.0 * PI * 0.5 * t).sin()
        })
        .collect()
}

/// A room: a delay, then a decaying tail, at a gain.
fn room(delay_ms: f32, gain: f32) -> Vec<f32> {
    let delay = (delay_ms / 1000.0 * RATE as f32) as usize;
    let mut noise = Noise(99);
    let tail = (0.04 * RATE as f32) as usize;
    let mut h = vec![0.0f32; delay + tail];
    h[delay] = gain;
    for i in 1..tail {
        h[delay + i] = gain * 0.6 * noise.next() * (-(i as f32) / (0.008 * RATE as f32)).exp();
    }
    h
}

fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for (i, &hv) in h.iter().enumerate() {
        if hv == 0.0 {
            continue;
        }
        for n in i..x.len() {
            y[n] += hv * x[n - i];
        }
    }
    y
}

/// A voice: a gliding tone in bursts (syllables).
fn voice(n: usize, from: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i < from {
                return 0.0;
            }
            let t = (i - from) as f32 / RATE as f32;
            let on = ((t * 3.0).fract() < 0.7) as u8 as f32;
            let f = 300.0 + 150.0 * (2.0 * PI * 0.7 * t).sin();
            0.25 * on * (2.0 * PI * f * t).sin()
        })
        .collect()
}

fn power(s: &[f32]) -> f32 {
    s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32
}

fn db(p: f32) -> f32 {
    10.0 * p.max(1e-20).log10()
}

/// Runs the canceller over whole signals; the output aligned back by
/// its block.
fn run(mic: &[f32], left: &[f32], right: &[f32]) -> (Vec<f32>, EchoCanceller) {
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; mic.len()];
    let mut block = [0.0f32; BLOCK];
    for b in 0..mic.len() / BLOCK {
        let r = b * BLOCK..(b + 1) * BLOCK;
        aec.process(
            &mic[r.clone()],
            &left[r.clone()],
            &right[r.clone()],
            &mut block,
        );
        out[r].copy_from_slice(&block);
    }
    (out, aec)
}

/// The music's echo through a room 45 ms away is taken out by more
/// than 25 dB once learnt, and a voice singing over it afterwards
/// comes through, the filter holding still meanwhile.
#[test]
fn the_echo_goes_and_a_voice_over_it_stays() {
    let n = RATE as usize * 12;
    let reference = music(n, 7);
    let echo = convolve(&reference, &room(45.0, 0.5));
    let sung = voice(n, RATE as usize * 7);
    let mut noise = Noise(3);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + sung[i] + 1e-4 * noise.next())
        .collect();
    let (out, aec) = run(&mic, &reference, &reference);
    let s = RATE as usize;
    // Music alone, after learning for a few seconds.
    let before = power(&echo[4 * s..7 * s]);
    let after = power(&out[4 * s..7 * s]);
    let removed = db(before) - db(after);
    assert!(removed > 25.0, "echo removed {removed:.1} dB");
    assert!(aec.status().learnt);
    // The meter says about as much, singing or not.
    assert!(aec.status().removed_db > 25.0, "{:?}", aec.status());
    // Singing over it: the voice is what comes out, the echo far under
    // it.
    let w = 8 * s..12 * s;
    let residual: Vec<f32> = w.clone().map(|i| out[i] - sung[i]).collect();
    let kept = db(power(&out[w.clone()])) - db(power(&sung[w.clone()]));
    let clean = db(power(&sung[w.clone()])) - db(power(&residual));
    eprintln!(
        "echo removed {removed:.1} dB (the meter says {:.1} dB); voice kept at {kept:.1} dB, {clean:.1} dB over the rest",
        aec.status().removed_db
    );
    assert!(kept > -3.0, "the voice kept at {kept:.1} dB");
    assert!(clean > 10.0, "the voice {clean:.1} dB over what is not it");
}

/// Nothing played: the microphone passes as it is.
#[test]
fn without_a_reference_the_voice_passes() {
    let n = RATE as usize * 3;
    let sung = voice(n, 0);
    let silence = vec![0.0; n];
    let (out, _) = run(&sung, &silence, &silence);
    let w = RATE as usize..2 * RATE as usize;
    let residual: Vec<f32> = w.clone().map(|i| out[i] - sung[i]).collect();
    assert!(db(power(&sung[w])) - db(power(&residual)) > 30.0);
}

/// The echo's way changing (the speaker moved): it learns the new one. It
/// trusts the change only once the filter has been worse than nothing for
/// a while, and moves its window only where the delay finder hears the new
/// echo: within 9 s it has taken it out by 25 dB.
#[test]
fn a_changed_room_is_learnt_again() {
    let n = RATE as usize * 16;
    let reference = music(n, 11);
    let first = convolve(&reference, &room(30.0, 0.6));
    let second = convolve(&reference, &room(80.0, 0.4));
    let half = 7 * RATE as usize;
    let mic: Vec<f32> = (0..n)
        .map(|i| if i < half { first[i] } else { second[i] })
        .collect();
    let (out, _) = run(&mic, &reference, &reference);
    let s = RATE as usize;
    let w = n - 3 * s..n;
    let removed = db(power(&second[w.clone()])) - db(power(&out[w]));
    assert!(removed > 25.0, "removed {removed:.1} dB after the change");
}

/// The volume turned up 12 dB once learnt: the leftover jumps by as much
/// as a singer would, yet the new loudness is learnt within seconds
/// (held still while that looked like singing, it never was).
#[test]
fn the_volume_turned_up_is_learnt_again() {
    let n = RATE as usize * 16;
    let s = RATE as usize;
    let reference = music(n, 13);
    let quiet = convolve(&reference, &room(40.0, 0.25));
    let mic: Vec<f32> = (0..n)
        .map(|i| if i < 8 * s { quiet[i] } else { 4.0 * quiet[i] })
        .collect();
    let (out, aec) = run(&mic, &reference, &reference);
    let w = 12 * s..16 * s;
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
    eprintln!(
        "the volume turned up: {removed:.1} dB removed after, {:?}",
        aec.status()
    );
    assert!(
        removed > 25.0,
        "removed {removed:.1} dB after the volume went up"
    );
    assert!(aec.status().learnt);
}

/// A filter in use far too loud: one learnt from a singer over music
/// near silence (a song's quiet start, whose band stem keeps a trace of
/// the singer) met the music at full level with an estimate 45 dB over
/// all the microphone heard, on a real song through a made-up laptop.
/// This signal does not lead it there, so the filters are spoilt by
/// hand. The estimate takes nothing out, no block comes out louder than
/// the microphone, and the echo is learnt again.
#[test]
fn a_filter_far_too_loud_takes_nothing_out_and_starts_over() {
    let n = RATE as usize * 12;
    let s = RATE as usize;
    let reference = music(n, 13);
    let echo = convolve(&reference, &room(40.0, 0.4));
    let mut noise = Noise(17);
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + 1e-4 * noise.next()).collect();
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; n];
    let mut block = [0.0f32; BLOCK];
    let mut worst = f32::MIN;
    for b in 0..n / BLOCK {
        if b * BLOCK == 4 * s {
            assert!(aec.status().learnt);
            for w in aec.in_use.filter.iter_mut().chain(aec.shadow.iter_mut()) {
                for c in w.iter_mut().flatten() {
                    c.re *= 200.0;
                    c.im *= 200.0;
                }
            }
        }
        let r = b * BLOCK..(b + 1) * BLOCK;
        aec.process(
            &mic[r.clone()],
            &reference[r.clone()],
            &reference[r.clone()],
            &mut block,
        );
        out[r.clone()].copy_from_slice(&block);
        if b * BLOCK >= 4 * s {
            worst = worst.max(db(power(&block)) - db(power(&mic[r])));
        }
    }
    assert!(
        worst < 0.5,
        "a block came out {worst:.1} dB over the microphone"
    );
    assert!(aec.status().learnt, "{:?}", aec.status());
    let w = 10 * s..12 * s;
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
    assert!(removed > 20.0, "the echo removed {removed:.1} dB after");
}

/// Filters gone beyond numbers (whatever did it) are started over, and
/// the canceller learns again: nothing stays spoilt until the player
/// starts again.
#[test]
fn filters_beyond_numbers_start_over() {
    let n = RATE as usize * 12;
    let s = RATE as usize;
    let reference = music(n, 11);
    let echo = convolve(&reference, &room(30.0, 0.5));
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; n];
    let mut block = [0.0f32; BLOCK];
    for b in 0..n / BLOCK {
        if b * BLOCK == 4 * s {
            assert!(aec.status().learnt);
            // The shadow, then the filter in use too, spoilt.
            aec.shadow[0][0][3].re = f32::NAN;
            aec.shadow_level = f32::NAN;
        }
        if b * BLOCK == 6 * s {
            aec.in_use.filter[0][0][3].re = f32::INFINITY;
        }
        let r = b * BLOCK..(b + 1) * BLOCK;
        aec.process(
            &echo[r.clone()],
            &reference[r.clone()],
            &reference[r.clone()],
            &mut block,
        );
        out[r].copy_from_slice(&block);
    }
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(aec.shadow_level.is_finite() && aec.in_use.error_level.is_finite());
    assert!(aec.status().learnt, "{:?}", aec.status());
    let w = 10 * s..12 * s;
    let removed = db(power(&echo[w.clone()])) - db(power(&out[w]));
    assert!(removed > 20.0, "the echo removed {removed:.1} dB after");
}

/// A note held for four seconds over the music, once learnt: the
/// filter learns nothing from it (it would learn to take the note out),
/// stays learnt, and the note comes through whole.
#[test]
fn a_held_note_never_spoils_the_filter() {
    let n = RATE as usize * 10;
    let s = RATE as usize;
    let reference = music(n, 7);
    let echo = convolve(&reference, &room(30.0, 0.5));
    let note: Vec<f32> = (0..n)
        .map(|i| {
            if i < 6 * s {
                0.0
            } else {
                0.2 * (2.0 * PI * 330.0 * i as f32 / RATE as f32).sin()
            }
        })
        .collect();
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + note[i]).collect();
    let (out, aec) = run(&mic, &reference, &reference);
    let w = 7 * s..10 * s;
    let (kept_db, clean_db) = kept(&out[w.clone()], &note[w]);
    eprintln!(
        "a held note: kept {kept_db:.1} dB, {clean_db:.1} dB over the rest, {:?}",
        aec.status()
    );
    assert!(aec.status().learnt, "the filter was spoilt");
    assert!(kept_db.abs() < 0.5, "the note kept at {kept_db:.1} dB");
    assert!(clean_db > 20.0, "the note {clean_db:.1} dB over the rest");
}

#[test]
fn the_transform_inverts() {
    let fft = fft::Fft::new();
    let a: Vec<f32> = (0..BLOCK).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..BLOCK).map(|i| (i as f32 * 0.11).cos()).collect();
    let x = fft.real(&a, &b);
    let back = fft.inverse(&x);
    for i in 0..BLOCK {
        assert!((back[i] - a[i]).abs() < 1e-4 && (back[BLOCK + i] - b[i]).abs() < 1e-4);
    }
}
