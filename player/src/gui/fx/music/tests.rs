use super::*;
use crate::core::Rhythmed;
use jodel_rhythm::waveform::{Lane, Melody, Waveform};
use jodel_rhythm::Region;

/// A grid at 120 a minute from 1 s for 20 beats, bars of four from the
/// first, sure of its first twelve beats only (its tempo worked out as
/// the analysis does).
fn grid() -> Analysis {
    let beats: Vec<f64> = (0..20).map(|i| 1.0 + f64::from(i) * 0.5).collect();
    let mut a = Analysis {
        downbeats: vec![0, 4, 8, 12, 16],
        regions: vec![
            Region {
                start: 1.0,
                end: 6.5,
                confidence: 0.9,
            },
            Region {
                start: 6.5,
                end: 10.5,
                confidence: 0.1,
            },
        ],
        beats,
        ..Analysis::default()
    };
    a.derive();
    a
}

/// The music `song` is heard as at `now`, sung by `singing`.
fn at(song: &Song, singing: &Singing, now: f64) -> Music {
    read(&Heard {
        song: Some(song),
        singing,
        now,
        clock: now,
        length: Some(20.0),
    })
}

/// Between two beats, as far as the moment is; the bar from its first
/// beat; the tempo the beats'.
#[test]
fn the_beat_and_the_bar_run_between_the_grid_s_beats() {
    let a = grid();
    let at = beat_at(&a, 1.25);
    assert!((at.phase - 0.5).abs() < 1e-9 && (at.tempo - 120.0).abs() < 1e-6);
    assert!((at.bar - 0.125).abs() < 1e-9, "{at:?}");
    assert_eq!(at.since_bar, Some(0.25));
    let later = beat_at(&a, 2.75);
    assert!((later.bar - 0.875).abs() < 1e-9, "{later:?}");
    assert!((later.since_beat - 0.25).abs() < 1e-9);
    let next_bar = beat_at(&a, 3.0);
    assert!(next_bar.bar < 1e-9 && next_bar.since_bar == Some(0.0));
    assert_eq!(beat_at(&a, 2.0).sure, 1.0, "a sure region");
    assert_eq!(beat_at(&a, 8.0).sure, 0.0, "an unsure one");
}

/// Before the grid and after it: running freely at the tempo, not sure
/// at all.
#[test]
fn off_the_grid_the_beat_runs_freely() {
    let a = grid();
    let before = beat_at(&a, 0.25);
    assert_eq!((before.sure, before.since_bar), (0.0, None));
    assert!((before.tempo - FREE_TEMPO).abs() < 1e-6);
    assert!((0.0..1.0).contains(&before.phase));
    let after = beat_at(&a, 30.1);
    assert!((after.phase - 0.2).abs() < 1e-6, "{after:?}");
    let empty = beat_at(&Analysis::default(), 3.3);
    assert!((empty.phase - 0.6).abs() < 1e-9);
}

/// The pulses fall away after each beat and each bar; unsure, they fade
/// to none for what moves with them, while the shaders still have them as
/// the grid says; no song, no music.
#[test]
fn the_pulses_fall_away_and_fade_where_the_grid_is_unsure() {
    let singing = Singing::default();
    let song = Song::of(&Rhythmed::new(grid(), None));
    let on = at(&song, &singing, 3.0);
    let off = at(&song, &singing, 3.2);
    assert!((on.beat - 1.0).abs() < 1e-6 && (on.downbeat - 1.0).abs() < 1e-6);
    assert!(off.beat < 0.2 && off.downbeat > off.beat, "{off:?}");
    assert_eq!(on.get(Signal::Beat), on.beat, "sure");
    let unsure = at(&song, &singing, 9.0);
    let faded = unsure.get(Signal::Beat);
    assert!(unsure.beat > 0.99 && faded == 0.0, "{unsure:?}");
    assert_eq!(unsure.get(Signal::BeatPhase), 0.0);
    assert!((on.progress - 0.15).abs() < 1e-6);
    let beat = [on.beat_phase, on.bar_phase, on.beat, on.downbeat];
    assert_eq!(on.uniform()[0], beat);
    let none = read(&Heard {
        song: None,
        singing: &singing,
        now: 3.0,
        clock: 3.0,
        length: None,
    });
    assert_eq!(none, Music::NEUTRAL);
}

/// The levels as the backdrop hears them (each stem its own, the mix for
/// each without stems), the melody's pitch where it is sure, and a
/// section's swell from its start.
#[test]
fn the_levels_the_melody_and_the_sections_come_from_the_waveform() {
    let lane = |level: fn(f64) -> u8| {
        let v: Vec<u8> = (0..3000).map(|i| level(f64::from(i) / 100.0)).collect();
        Lane {
            peak: v.clone(),
            rms: v,
        }
    };
    let loud_after = |t: f64| if t >= 12.0 { 220 } else { 90 };
    let mut melody = Melody {
        cents: vec![0; 3000],
        sure: vec![0; 3000],
    };
    melody.cents[500] = 6900;
    melody.sure[500] = 255;
    melody.cents[501] = 6000;
    let mut stems = [lane(|_| 100), lane(|_| 100), lane(|_| 100), lane(|_| 100)];
    stems[Stem::Drums as usize] = lane(loud_after);
    let waveform = Waveform {
        mix: lane(loud_after),
        stems: Some(stems),
        melody: Some(melody),
    };
    let mut a = grid();
    a.beats = (0..60).map(|i| f64::from(i) * 0.5).collect();
    a.downbeats = (0..15).map(|i| i * 4).collect();
    a.derive();
    let song = Song::of(&Rhythmed::new(a, Some(waveform)));
    let singing = Singing::default();
    let at = |now: f64| at(&song, &singing, now);
    let (quiet, loud, later) = (at(5.0), at(12.5), at(16.0));
    let drums = Signal::Stem(Stem::Drums);
    assert!(loud.get(drums) > 0.9 && quiet.get(drums) < 0.1, "{loud:?}");
    let bass = Signal::Stem(Stem::Bass);
    assert_eq!(loud.get(bass), quiet.get(bass), "the bass stays");
    assert!(loud.level > 0.9 && quiet.level < 0.1);
    assert!((at(5.005).melody - 69.0).abs() < 1e-6, "A4 where sure");
    assert_eq!(at(5.015).melody, 0.0, "not where unsure");
    assert!((at(5.005).get(Signal::Melody) - (69.0 - 36.0) / 48.0).abs() < 1e-6);
    let swelling = loud.section > 0.6 && later.section < loud.section;
    assert!(swelling, "{loud:?} {later:?}");
    assert_eq!(quiet.section, 0.0);
}

/// The song heard is made once for an analysis and shared: following the
/// same analysis keeps it, a new one (a grid fixed by hand) makes it
/// again, none drops it.
#[test]
fn the_song_heard_is_made_once_an_analysis() {
    let rhythm = Rhythmed::new(grid(), None);
    let mut kept = None;
    Song::follow(&mut kept, Some(&rhythm));
    let first = kept.clone().expect("made");
    Song::follow(&mut kept, Some(&rhythm));
    assert!(std::sync::Arc::ptr_eq(&first, kept.as_ref().unwrap()));
    let regridded = Rhythmed::new(grid(), None);
    Song::follow(&mut kept, Some(&regridded));
    assert!(!std::sync::Arc::ptr_eq(&first, kept.as_ref().unwrap()));
    Song::follow(&mut kept, None);
    assert!(kept.is_none());
}
