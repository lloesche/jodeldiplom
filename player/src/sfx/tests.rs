use super::*;
use crate::core::bubbles::SWELL;
use crate::core::shine::Tier;

#[test]
fn the_bank_has_every_sound_ending_in_silence() {
    let b = bank(44_100);
    assert_eq!((b.len(), b.rate()), (Sound::count(), 44_100));
    for s in Sound::all() {
        let x = b.sound(s.index()).unwrap();
        assert!(x.len() > 2_000, "{s:?}");
        assert!(x[x.len() - 2..].iter().all(|v| v.abs() < 1e-6), "{s:?}");
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.3, "{s:?} under the song: {peak}");
    }
}

/// A WAV file of 32-bit float stereo at `rate`.
fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 4) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 4);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 8).to_le_bytes());
    out.extend_from_slice(&8u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for v in samples {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// `sounds` (each at its time, seconds, with its gain and pan) mixed into
/// one stereo take, as the desk mixes them (the near side whole).
fn sequence(rate: u32, sounds: &[(f64, Sound, f32, f32)]) -> Vec<f32> {
    let end = sounds
        .iter()
        .map(|&(t, s, ..)| t + s.render(rate).seconds())
        .fold(0.0, f64::max);
    let mut out = vec![0.0f32; 2 * (end * f64::from(rate)).ceil() as usize + 2];
    for &(t, s, gain, pan) in sounds {
        let at = 2 * (t * f64::from(rate)).round() as usize;
        let p = pan.clamp(-1.0, 1.0);
        let sides = [gain * (1.0 - p).min(1.0), gain * (1.0 + p).min(1.0)];
        for (k, v) in s.render(rate).interleaved().into_iter().enumerate() {
            out[at + k] += sides[k % 2] * v;
        }
    }
    out
}

/// A result of `points`, with `hit` words and a best streak of `streak`,
/// `tiers` its phrases', its `medal`, the `best` before, sung whole or not.
fn result(
    (points, hit, streak): (u32, u32, u32),
    tiers: &[Option<Tier>],
    (medal, best, whole): (Option<crate::core::singing::Medal>, Option<u32>, bool),
) -> crate::core::singing::SongResult {
    crate::core::singing::SongResult {
        title: "Song".into(),
        points,
        hit,
        words: 45,
        streak,
        phrases: tiers
            .iter()
            .map(|&tier| crate::core::shine::Shone {
                line: 0,
                end: 1.0,
                value: 0.9,
                top: 0.9,
                tier,
            })
            .collect(),
        medal,
        best,
        whole,
        demo: false,
    }
}

/// The sounds of `result`'s reveal as the screen plays them, frame by
/// frame at 60 a second: each at the frame it is cued on.
fn revealed(result: crate::core::singing::SongResult) -> Vec<(f64, Sound, f32, f32)> {
    let shown = crate::core::singing::Shown::new(result, 1, 0);
    let mut sounds = reveal::RevealSounds::default();
    let mut heard = Vec::new();
    for f in 0..(60.0 * (shown.reveal.settled() + 1.0)) as u64 {
        let t = shown.time(f * 1000 / 60);
        for c in sounds.frame(Some((&shown, t))) {
            heard.push((t, c.sound, c.gain, c.pan));
        }
    }
    heard
}

/// Writes every sound, and a few runs of them as the game plays them
/// (a whole diploma revealed among them, for each kind of verdict), as
/// WAV files into the folder `JODEL_SFX_WAVS` names, to listen to and to
/// look at: `cargo test -p jodeldiplom sfx::tests::write_the_sounds --
/// --ignored`.
#[test]
#[ignore]
fn write_the_sounds() {
    use crate::core::singing::Medal;
    let Some(dir) = std::env::var_os("JODEL_SFX_WAVS") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let rate = 48_000;
    for s in Sound::all() {
        let path = dir.join(format!("{}.wav", s.name()));
        durable::write_atomic(&path, &wav(&s.render(rate).interleaved(), rate)).unwrap();
    }
    // A run of pops up and down a line, each splashing as it tears.
    let mut ripple = Vec::new();
    for (k, step) in [0, 2, 4, 6, 8, 7, 5, 3, 1, 4].into_iter().enumerate() {
        let t = 0.16 * k as f64;
        ripple.push((t, Sound::Pop(step), 1.0, 0.0));
        ripple.push((t + SWELL, Sound::Sparkle(step), 1.0, 0.0));
    }
    // A word's bubbles bursting together: a ripple.
    for j in 0..4 {
        let t = 2.0 + moments::RIPPLE * j as f64;
        ripple.push((t, Sound::Pop(3 + j), 1.0, 0.0));
    }
    ripple.push((
        2.0 + 3.0 * moments::RIPPLE + SWELL,
        Sound::Sparkle(6),
        1.0,
        0.0,
    ));
    // Up every rung, then lost.
    let mut streak: Vec<_> = (Tier::ALL.into_iter().enumerate())
        .map(|(k, t)| (k as f64, Sound::StepUp(t), 1.0, 0.0))
        .collect();
    streak.push((Tier::COUNT as f64 + 0.2, Sound::Break, 1.0, 0.0));
    // Every tier's chime, each as the one before rings out.
    let mut tiers = Vec::new();
    let mut at = 0.0;
    for t in Tier::ALL {
        let chime = Sound::Tier(t);
        tiers.push((at, chime, 1.0, 0.0));
        at += 0.8 * chime.render(rate).l.len() as f64 / f64::from(rate);
    }
    // Diplomas revealed, as the screen plays them: a Gold one with a new
    // best, a Silver one short of its best, one without a medal, one not
    // sung from its start, one with nothing scored, and one the demo key
    // sang (its best line no best, a tap).
    let six = [
        Some(Tier::TOP),
        None,
        Some(Tier::named("Good")),
        Some(Tier::named("Great")),
    ]
    .into_iter()
    .chain([Some(Tier::named("Amazing")), Some(Tier::TOP)])
    .collect::<Vec<_>>();
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
        &[None, Some(Tier::named("Good")), None],
        (None, None, true),
    );
    let whole = result((5_600, 30, 9), &six[..3], (None, Some(4_000), false));
    let silent = result((0, 0, 0), &[], (None, None, false));
    let demo = crate::core::singing::SongResult {
        demo: true,
        ..result((9_100, 40, 30), &six, (Some(Medal::Gold), None, true))
    };
    let runs = [
        ("run-ripple", ripple),
        ("run-streak", streak),
        ("run-tiers", tiers),
        ("run-diploma-gold-new-best", revealed(gold)),
        ("run-diploma-silver", revealed(silver)),
        ("run-diploma-cheer", revealed(cheer)),
        ("run-diploma-whole", revealed(whole)),
        ("run-diploma-silent", revealed(silent)),
        ("run-diploma-demo", revealed(demo)),
    ];
    for (name, run) in runs {
        let take = sequence(rate, &run);
        let peak = take.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 0.5, "{name}: {peak}");
        let path = dir.join(format!("{name}.wav"));
        durable::write_atomic(&path, &wav(&take, rate)).unwrap();
    }
}

/// How long a bank takes to make, and how big it is, at the device's
/// rates: `cargo test --release -p jodeldiplom sfx::tests::bank_times --
/// --ignored --nocapture`.
#[test]
#[ignore]
fn bank_times() {
    let dir = std::env::temp_dir().join(format!("jodel-sfx-times-{}", std::process::id()));
    let store = store::BankStore::new(&dir);
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    for rate in [44_100, 48_000, 96_000, 192_000, 384_000] {
        let start = std::time::Instant::now();
        let b = bank(rate);
        let made = ms(start);
        let start = std::time::Instant::now();
        store.save(&b).unwrap();
        let saved = ms(start);
        let start = std::time::Instant::now();
        let read = store.load(rate).unwrap();
        let loaded = ms(start);
        assert_eq!(read, b);
        let size = std::fs::metadata(store.file(rate)).unwrap().len();
        println!(
            "{rate:>6} Hz: made {made:>7.1} ms, written {saved:>6.1} ms, read {loaded:>6.1} ms, file {:>5.2} MB",
            size as f64 / 1e6
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
