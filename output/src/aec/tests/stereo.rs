//! Two speakers heard apart, a song's beat, a speaker's limiter, and the
//! residual suppressor.

use super::laptop::{kept, laptop, singer};
use super::*;

/// A stereo song: what both sides play (music), and what each plays of
/// its own (wide guitars, reverb: music of its own, `wide` as loud).
fn stereo(n: usize, wide: f32) -> (Vec<f32>, Vec<f32>) {
    let common = music(n, 31);
    let a = music(n, 32);
    let b = music(n, 33);
    let left = (0..n).map(|i| common[i] + wide * a[i]).collect();
    let right = (0..n).map(|i| common[i] + wide * b[i]).collect();
    (left, right)
}

/// A laptop's microphone beside its left speaker, the right one further
/// and quieter: a song whose sides play a good part of their sound apart
/// is taken out by over 25 dB from left and right, over 12 dB more than
/// from their mean (all the canceller had before).
#[test]
fn two_speakers_heard_apart_are_taken_out() {
    let n = RATE as usize * 12;
    let s = RATE as usize;
    let (left, right) = stereo(n, 0.6);
    let near = convolve(&left, &room(40.0, 0.6));
    let far = convolve(&right, &room(41.0, 0.25));
    let mut noise = Noise(4);
    let mic: Vec<f32> = (0..n)
        .map(|i| near[i] + far[i] + 1e-4 * noise.next())
        .collect();
    let w = 8 * s..12 * s;
    let (out, aec) = run(&mic, &left, &right);
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w.clone()]));
    let mean: Vec<f32> = (0..n).map(|i| 0.5 * (left[i] + right[i])).collect();
    let (from_mean, _) = run(&mic, &mean, &mean);
    let mean_removed = db(power(&mic[w.clone()])) - db(power(&from_mean[w]));
    eprintln!("two speakers: {removed:.1} dB removed from left and right, {mean_removed:.1} dB from their mean; {:?}", aec.status());
    assert!(removed > 25.0, "removed {removed:.1} dB");
    assert!(
        removed > mean_removed + 12.0,
        "from the mean {mean_removed:.1} dB"
    );
    assert!(aec.status().learnt);
}

/// Music whose loudness repeats every 250 ms (a beat), the echo 300 ms late
/// (past the window as it starts): the delay is found to within 15 ms, not
/// a beat off.
#[test]
fn the_delay_is_found_not_a_beat_off() {
    let n = RATE as usize * 16;
    let s = RATE as usize;
    let beat = RATE as usize / 4;
    let base = music(n, 41);
    let reference: Vec<f32> = (0..n)
        .map(|i| {
            let into = (i % beat) as f32 / beat as f32;
            base[i] * (0.15 + 1.7 * (-into * 8.0).exp())
        })
        .collect();
    let echo = convolve(&reference, &room(300.0, 0.5));
    let mut noise = Noise(6);
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + 1e-4 * noise.next()).collect();
    let (out, aec) = run(&mic, &reference, &reference);
    let delay = aec.status().delay.expect("a delay found") * 1000.0;
    let w = 12 * s..16 * s;
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
    eprintln!("a beat: the delay found {delay:.0} ms, {removed:.1} dB removed");
    assert!(
        (delay - 300.0).abs() < 15.0,
        "the delay found {delay:.0} ms"
    );
    assert!(removed > 20.0, "removed {removed:.1} dB");
}

/// A loud passage through a speaker's limiter: the music 10 dB up, the
/// speaker's gain pulled down 8 dB at once and let up again over a third of
/// a second, so the filter learnt on the quiet part overshoots for moments.
/// It never starts over (learnt throughout), and the louder part is learnt.
#[test]
fn a_limiter_riding_a_loud_passage_never_starts_it_over() {
    let n = RATE as usize * 16;
    let s = RATE as usize;
    let loud_from = 8 * s;
    let base = music(n, 17);
    let reference: Vec<f32> = (0..n)
        .map(|i| {
            if i < loud_from {
                0.3 * base[i]
            } else {
                base[i]
            }
        })
        .collect();
    let gain: Vec<f32> = (0..n)
        .map(|i| {
            if i < loud_from {
                1.0
            } else {
                let t = (i - loud_from) as f32 / RATE as f32;
                0.4 + 0.2 * (1.0 - (-t / 0.3).exp())
            }
        })
        .collect();
    let spoken: Vec<f32> = (0..n).map(|i| gain[i] * reference[i]).collect();
    let echo = convolve(&spoken, &room(40.0, 0.5));
    let mut noise = Noise(12);
    let mic: Vec<f32> = (0..n).map(|i| echo[i] + 1e-4 * noise.next()).collect();
    let mut aec = EchoCanceller::new(RATE);
    let mut out = vec![0.0f32; n];
    let mut block = [0.0f32; BLOCK];
    let mut lost = 0;
    for b in 0..n / BLOCK {
        let r = b * BLOCK..(b + 1) * BLOCK;
        aec.process(
            &mic[r.clone()],
            &reference[r.clone()],
            &reference[r.clone()],
            &mut block,
        );
        out[r].copy_from_slice(&block);
        if b * BLOCK > 4 * s && !aec.status().learnt {
            lost += 1;
        }
    }
    let w = 13 * s..16 * s;
    let removed = db(power(&mic[w.clone()])) - db(power(&out[w]));
    eprintln!("a limiter: {lost} blocks not learnt after 4 s, {removed:.1} dB removed at the end");
    assert_eq!(lost, 0, "started over");
    assert!(removed > 20.0, "removed {removed:.1} dB");
}

/// The canceller and the suppressor after it, over whole signals: what
/// each gives, lined up.
fn suppressed(mic: &[f32], reference: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let cleaned = Whole::new(RATE).run(mic, reference, reference, |_, _| {});
    (cleaned.cancelled, cleaned.suppressed)
}

/// Nothing played (a headset): the suppressor gives the voice exactly as it
/// came, a block late.
#[test]
fn without_an_echo_the_suppressor_passes_the_voice() {
    let n = RATE as usize * 3;
    let (sung, _) = singer(n, 0);
    let (cancelled, out) = suppressed(&sung, &vec![0.0; n]);
    for i in 0..n / BLOCK * BLOCK - BLOCK {
        assert!(
            (out[i] - cancelled[i]).abs() < 1e-5,
            "{i}: {} {}",
            out[i],
            cancelled[i]
        );
    }
}

/// A laptop's distorting speakers: what the filter leaves of the music
/// alone is taken down by the suppressor by more than 6 dB besides, while
/// a singer as loud as the echo keeps all but a dB, and stands no less over
/// the rest than after the canceller alone (the rest counts what the gains
/// change of the voice too).
#[test]
fn the_suppressor_takes_the_leftover_down_and_keeps_the_singer() {
    let n = RATE as usize * 20;
    let s = RATE as usize;
    let reference = music(n, 21);
    let echo = laptop(&reference, 40.0, 0.4);
    let (voice, _) = singer(n, 12 * s);
    let mut noise = Noise(8);
    let mic: Vec<f32> = (0..n)
        .map(|i| echo[i] + voice[i] + 3e-4 * noise.next())
        .collect();
    let (cancelled, out) = suppressed(&mic, &reference);
    let quiet = 8 * s..12 * s;
    let more = db(power(&cancelled[quiet.clone()])) - db(power(&out[quiet]));
    let sung = 13 * s..19 * s;
    let (kept_before, over_before) = kept(&cancelled[sung.clone()], &voice[sung.clone()]);
    let (kept_after, over_after) = kept(&out[sung.clone()], &voice[sung]);
    eprintln!(
        "the suppressor: the leftover {more:.1} dB further down; the singer kept {kept_before:.1} → {kept_after:.1} dB, {over_before:.1} → {over_after:.1} dB over the rest"
    );
    assert!(more > 6.0, "the leftover down by {more:.1} dB");
    assert!(kept_after > -1.0, "the singer kept {kept_after:.1} dB");
    assert!(
        over_after >= over_before,
        "{over_before:.1} → {over_after:.1} dB"
    );
}

/// What the suppressor learnt goes with the filter it was learnt of: once
/// it takes the voice down (the canceller's estimate all of it, learnt as
/// leftover), a block from a new filter ([`Echo::generation`]) has it learn
/// again, and until it has, it passes the voice exactly as it came.
#[test]
fn the_suppressor_learns_again_for_a_new_filter() {
    let mut res = Suppressor::new(RATE);
    let mut noise = Noise(23);
    let mut echo = Echo {
        estimate: [0.0; BLOCK],
        echo_only: true,
        learnt: true,
        generation: 0,
    };
    let mut out = [0.0f32; BLOCK];
    let mut last = [0.0f32; BLOCK];
    let mut ratio = |res: &mut Suppressor, echo: &mut Echo, last: &mut [f32; BLOCK]| {
        let voice: [f32; BLOCK] = std::array::from_fn(|_| 0.2 * noise.next());
        echo.estimate = voice;
        res.process(&voice, echo, &mut out);
        let r = db(power(&out)) - db(power(last));
        *last = voice;
        r
    };
    let mut before = 0.0;
    // The share starts at −30 dB and follows over about 1.5 s.
    for _ in 0..4000 {
        before = ratio(&mut res, &mut echo, &mut last);
    }
    assert!(before < -10.0, "learnt: {before:.1} dB");
    echo.generation = 1;
    ratio(&mut res, &mut echo, &mut last);
    for _ in 0..100 {
        let after = ratio(&mut res, &mut echo, &mut last);
        assert!(after.abs() < 0.01, "a new filter: {after:.2} dB");
    }
}
