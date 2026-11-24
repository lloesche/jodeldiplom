use super::*;
use crate::core::bubbles::SWELL;
use crate::core::shine::Tier;

#[test]
fn the_bank_has_every_sound_ending_in_silence() {
    let b = bank(44_100);
    assert_eq!((b.len(), b.rate()), (Sound::COUNT, 44_100));
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

/// `sounds` (each at its time, seconds) mixed into one stereo take.
fn sequence(rate: u32, sounds: &[(f64, Sound, f32)]) -> Vec<f32> {
    let end = sounds
        .iter()
        .map(|&(t, s, _)| t + s.render(rate).seconds())
        .fold(0.0, f64::max);
    let mut out = vec![0.0f32; 2 * (end * f64::from(rate)).ceil() as usize + 2];
    for &(t, s, gain) in sounds {
        let at = 2 * (t * f64::from(rate)).round() as usize;
        for (k, v) in s.render(rate).interleaved().into_iter().enumerate() {
            out[at + k] += gain * v;
        }
    }
    out
}

/// Writes every sound, and a few runs of them as the game plays them, as
/// WAV files into the folder `JODEL_SFX_WAVS` names, to listen to and to
/// look at: `cargo test -p jodeldiplom sfx::tests::write_the_sounds --
/// --ignored`.
#[test]
#[ignore]
fn write_the_sounds() {
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
        ripple.push((t, Sound::Pop(step), 1.0));
        ripple.push((t + SWELL, Sound::Sparkle(step), 1.0));
    }
    // A word's bubbles bursting together: a ripple.
    for j in 0..4 {
        let t = 2.0 + moments::RIPPLE * j as f64;
        ripple.push((t, Sound::Pop(3 + j), 1.0));
    }
    ripple.push((2.0 + 3.0 * moments::RIPPLE + SWELL, Sound::Sparkle(6), 1.0));
    let streak = [
        (0.0, Sound::StepUp(1), 1.0),
        (1.0, Sound::StepUp(2), 1.0),
        (2.0, Sound::StepUp(3), 1.0),
        (3.2, Sound::Break, 1.0),
    ];
    let tiers = [
        (0.0, Sound::Tier(Tier::Good), 1.0),
        (1.2, Sound::Tier(Tier::Great), 1.0),
        (2.6, Sound::Tier(Tier::Amazing), 1.0),
        (4.4, Sound::Tier(Tier::Diva), 1.0),
    ];
    // A Gold result with a new best revealed, as the screen plays it.
    let result = crate::core::singing::SongResult {
        title: "Song".into(),
        points: 12_400,
        hit: 41,
        words: 45,
        streak: 23,
        phrases: [Some(Tier::Diva), None, Some(Tier::Good), Some(Tier::Great)]
            .into_iter()
            .chain([Some(Tier::Amazing), Some(Tier::Diva)])
            .map(|tier| crate::core::shine::Shone {
                line: 0,
                end: 1.0,
                value: 0.9,
                diva: 0.9,
                tier,
            })
            .collect(),
        medal: Some(crate::core::singing::Medal::Gold),
        best: Some(9_000),
        whole: true,
    };
    let shown = crate::core::singing::Shown::new(result, 1, 0);
    let revealed: Vec<(f64, Sound, f32)> = shown
        .reveal
        .beats()
        .iter()
        .filter_map(|&(at, b)| reveal::sound_of(b, &shown).map(|(s, gain, _)| (at, s, gain)))
        .collect();
    for (name, run) in [
        ("run-ripple", &ripple[..]),
        ("run-streak", &streak[..]),
        ("run-tiers", &tiers[..]),
        ("run-result", &revealed[..]),
    ] {
        let path = dir.join(format!("{name}.wav"));
        durable::write_atomic(&path, &wav(&sequence(rate, run), rate)).unwrap();
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
