//! Telling singing from the echo: a quiet singer over a laptop's echo, one
//! too quiet for their level to tell, a room that changes under them, a hum
//! in the room, the echo alone; and how soon two speakers heard apart are
//! learnt.

use super::laptop::{kept, laptop, singer};
use super::stereo::stereo;
use super::*;

/// A stereo song through a laptop's two speakers (a limiter and soft
/// clipping each, a living room's tail), the microphone nearer the left
/// one: the echo at the microphone.
fn two_laptop_speakers(left: &[f32], right: &[f32], delay_ms: f32) -> Vec<f32> {
    let near = laptop(left, delay_ms, 0.3);
    let far = laptop(right, delay_ms + 2.0, 0.3);
    near.iter().zip(&far).map(|(a, b)| a + 0.5 * b).collect()
}

/// `voice` scaled so that over its syllables it is `under` dB under the
/// echo `echo` there.
fn under(voice: &[f32], syllables: &[(usize, usize, bool)], echo: &[f32], under: f32) -> Vec<f32> {
    let sung = |s: &[f32]| {
        syllables
            .iter()
            .map(|&(a, b, _)| power(&s[a..b]) * (b - a) as f32)
            .sum::<f32>()
    };
    let g = (sung(echo) / sung(voice) / 10f32.powf(under / 10.0)).sqrt();
    voice.iter().map(|v| g * v).collect()
}

/// Runs the canceller as [`run`] does; and, block by block, whether it took
/// the block for sung over.
fn run_heard(mic: &[f32], left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<bool>) {
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; mic.len()];
    let mut heard = Vec::new();
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
        heard.push(aec.detector.singing());
    }
    (out, heard)
}

/// The share of the blocks in `w` (in samples) taken for sung over.
fn share(heard: &[bool], w: std::ops::Range<usize>) -> f32 {
    let blocks = &heard[w.start / BLOCK..w.end / BLOCK];
    blocks.iter().filter(|&&h| h).count() as f32 / blocks.len() as f32
}

/// The echo removed in `w`: what came out less the voice (the echo the
/// canceller left) against the echo.
fn removed_under(out: &[f32], voice: &[f32], echo: &[f32], w: std::ops::Range<usize>) -> f32 {
    let left: Vec<f32> = w.clone().map(|i| out[i] - voice[i]).collect();
    db(power(&echo[w])) - db(power(&left))
}

/// A singer 8 dB under the echo of a stereo song through a laptop (a
/// MacBook's singer was), for 14 s once the canceller has learnt: the
/// filter, learning from the singing at a tenth of its pace, still takes
/// the echo out under them within 2 dB of what it took out before they
/// sang, and the singer is kept whole. (When the leftover's level alone
/// told the singing apart, the filter learnt from it at the full pace and
/// took 10 dB less out by the end.)
#[test]
fn a_quiet_singer_never_draws_the_filter_away() {
    let n = RATE as usize * 22;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.5);
    let echo = two_laptop_speakers(&left, &right, 40.0);
    let (voice, syllables) = singer(n, 8 * s);
    let voice = under(&voice, &syllables, &echo, 8.0);
    let mut noise = Noise(9);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let (out, aec) = run(&mic, &left, &right);
    let before = removed_under(&out, &voice, &echo, 5 * s..8 * s);
    let early = removed_under(&out, &voice, &echo, 9 * s..12 * s);
    let late = removed_under(&out, &voice, &echo, 18 * s..22 * s);
    let (kept_db, _) = kept(&out[9 * s..22 * s], &voice[9 * s..22 * s]);
    eprintln!(
        "a quiet singer: {before:.1} dB of the echo out before, {early:.1} dB under the singer at first, {late:.1} dB at the end; the singer kept at {kept_db:.1} dB; {:?}",
        aec.status()
    );
    assert!(before > 15.0, "{before:.1} dB out before");
    assert!(early > before - 2.0, "{early:.1} dB out at first");
    assert!(late > before - 2.0, "{late:.1} dB out at the end");
    assert!(kept_db.abs() < 0.5, "the singer kept at {kept_db:.1} dB");
    assert!(aec.status().learnt);
}

/// A singer 14 dB under the echo: once the canceller has learnt, their
/// leftover is mostly under [`singing::LOUDER`] times the usual one, so
/// only their pitch tells them apart. Most of their syllables are heard as
/// sung (96 %; by their level alone, 26 %), and the echo under them still
/// goes as before they sang.
#[test]
fn a_singer_too_quiet_for_their_level_is_heard_by_their_pitch() {
    let n = RATE as usize * 20;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.5);
    let echo = two_laptop_speakers(&left, &right, 40.0);
    let (voice, syllables) = singer(n, 8 * s);
    let voice = under(&voice, &syllables, &echo, 14.0);
    let mut noise = Noise(12);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let (out, heard) = run_heard(&mic, &left, &right);
    let sung: Vec<bool> = (0..n / BLOCK)
        .map(|b| {
            let at = b * BLOCK;
            syllables
                .iter()
                .any(|&(a, e, soft)| !soft && at >= a && at + BLOCK <= e)
        })
        .collect();
    let late = |b: &usize| *b * BLOCK >= 10 * s;
    let syllable_blocks: Vec<usize> = (0..n / BLOCK).filter(|b| sung[*b] && late(b)).collect();
    let heard_share =
        syllable_blocks.iter().filter(|&&b| heard[b]).count() as f32 / syllable_blocks.len() as f32;
    let before = removed_under(&out, &voice, &echo, 5 * s..8 * s);
    let after = removed_under(&out, &voice, &echo, 16 * s..20 * s);
    eprintln!(
        "a singer 14 dB under: {:.0} % of their syllables heard; {before:.1} dB of the echo out before, {after:.1} dB at the end",
        100.0 * heard_share
    );
    assert!(heard_share > 0.5, "{:.0} % heard", 100.0 * heard_share);
    assert!(after > before - 2.0, "{after:.1} dB out at the end");
}

/// The laptop moved while a quiet singer sings (8 dB under the echo): the
/// echo comes 25 ms later, another way. The singing never keeps the
/// canceller from learning the new way: 5 s on it takes the new echo out
/// by more than 10 dB under the singer (before, 7), who is kept whole.
#[test]
fn a_room_changed_under_a_singer_is_learnt_again() {
    let n = RATE as usize * 20;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.5);
    let first = two_laptop_speakers(&left, &right, 40.0);
    let second = two_laptop_speakers(&right, &left, 65.0);
    let moved = 9 * s;
    let echo: Vec<f32> = (0..n)
        .map(|i| if i < moved { first[i] } else { 0.8 * second[i] })
        .collect();
    let (voice, syllables) = singer(n, 6 * s);
    let voice = under(&voice, &syllables, &echo, 8.0);
    let mut noise = Noise(10);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let (out, aec) = run(&mic, &left, &right);
    let after = removed_under(&out, &voice, &echo, 14 * s..20 * s);
    let (kept_db, _) = kept(&out[14 * s..20 * s], &voice[14 * s..20 * s]);
    eprintln!(
        "moved under a singer: {after:.1} dB of the new echo out 5 s on, the singer kept at {kept_db:.1} dB; {:?}",
        aec.status()
    );
    assert!(after > 10.0, "{after:.1} dB of the new echo out");
    assert!(kept_db.abs() < 0.5, "the singer kept at {kept_db:.1} dB");
}

/// A steady hum in the room (100 Hz and its harmonics, 20 dB under the
/// echo: a fridge, a fan), clearly pitched and nothing of the music's, is
/// no singer: once learnt, hardly a block is taken for sung over (with the
/// leftover's floor starting at nothing, every one was), and the canceller
/// learns as it would without it.
#[test]
fn a_hum_in_the_room_is_not_singing() {
    let n = RATE as usize * 12;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.5);
    let echo = two_laptop_speakers(&left, &right, 40.0);
    let hum: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            (1..6)
                .map(|h| (2.0 * PI * 100.0 * h as f32 * t).sin() / h as f32)
                .sum()
        })
        .collect();
    let g = (power(&echo) / power(&hum) / 100.0).sqrt();
    let hum: Vec<f32> = hum.iter().map(|v| g * v).collect();
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + hum[i]).collect();
    let (out, heard) = run_heard(&mic, &left, &right);
    let (plain, _) = run(&echo, &left, &right);
    let w = 8 * s..12 * s;
    let with = removed_under(&out, &hum, &echo, w.clone());
    let without = db(power(&echo[w.clone()])) - db(power(&plain[w]));
    let sung = share(&heard, 4 * s..n);
    eprintln!(
        "a hum: {:.1} % of the blocks taken for sung over; {with:.1} dB of the echo out under it, {without:.1} dB without it",
        100.0 * sung
    );
    assert!(sung < 0.02, "{:.1} % sung over", 100.0 * sung);
    assert!(
        with > without - 2.0,
        "{with:.1} dB out, {without:.1} without the hum"
    );
}

/// The echo alone, of a stereo song through a laptop's two speakers: once
/// learnt, few blocks are taken for sung over (6 %, before 14: where this
/// music's pure tones leave a pitch that the estimate, its other tone and
/// its noise louder, does not seem to hold; on a MacBook's real echo alone
/// none were).
#[test]
fn the_echo_alone_is_not_singing() {
    let n = RATE as usize * 14;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.5);
    let echo = two_laptop_speakers(&left, &right, 40.0);
    let mut noise = Noise(13);
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + 3e-4 * noise.next()).collect();
    let (_, heard) = run_heard(&mic, &left, &right);
    let sung = share(&heard, 4 * s..n);
    eprintln!(
        "the echo alone: {:.1} % of the blocks taken for sung over",
        100.0 * sung
    );
    assert!(sung < 0.08, "{:.1} % sung over", 100.0 * sung);
}

/// Two speakers playing a stereo song, each its own way to the
/// microphone (one 40 ms away, the other 55 ms, louder and with a longer
/// tail): learnt soon, the echo down by more than 12 dB in its third
/// second and by more than 22 dB once learnt. (Its step along both
/// channels at once was twice what it is now; on this echo, without noise
/// or a voice, that learnt faster, 17.7 and 27 dB, and it let a voice the
/// detector missed in twice as much.)
#[test]
fn two_speakers_apart_are_learnt_within_seconds() {
    let n = RATE as usize * 10;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.6);
    let near = convolve(&left, &room(40.0, 0.4));
    let far = convolve(&right, &room(55.0, 0.6));
    let mut noise = Noise(11);
    let mic: Vec<f32> = (0..n)
        .map(|i| near[i] + far[i] + 1e-4 * noise.next())
        .collect();
    let (out, _) = run(&mic, &left, &right);
    let removed = |w: std::ops::Range<usize>| db(power(&mic[w.clone()])) - db(power(&out[w]));
    let soon = removed(2 * s..3 * s);
    let learnt = removed(7 * s..10 * s);
    eprintln!(
        "two speakers apart: {soon:.1} dB out in the third second, {learnt:.1} dB once learnt"
    );
    assert!(soon > 12.0, "{soon:.1} dB out in the third second");
    assert!(learnt > 22.0, "{learnt:.1} dB out once learnt");
}
