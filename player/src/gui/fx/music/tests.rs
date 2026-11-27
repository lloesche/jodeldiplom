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
        duration: 20.0,
        ..Analysis::default()
    };
    a.derive();
    a
}

/// The music `song` is heard as at `now`, sung by `singing`: its drivers
/// the song's curves read then.
fn at(song: &Song, singing: &Singing, now: f64) -> Music {
    let curves = jodel_looks::Curves::new(song.analysis(), song.waveform(), &song.sections, &[]);
    read(&Heard {
        song: Some(song),
        singing,
        now,
        clock: now,
        signals: curves.at(now),
    })
}

/// Between two beats, as far as the moment is; the bar from its first
/// beat; the tempo the beats'; as sure as the region (faded): the song's
/// grid at the moment ([`Analysis::grid_at`]).
#[test]
fn the_beat_and_the_bar_run_between_the_grid_s_beats() {
    let singing = Singing::default();
    let song = Song::of(&Rhythmed::new(grid(), None));
    let at = |t: f64| super::tests::at(&song, &singing, t);
    let m = at(1.25);
    assert!((m.beat_phase - 0.5).abs() < 1e-6 && (m.tempo - 120.0).abs() < 1e-4);
    assert!((m.bar_phase - 0.125).abs() < 1e-6, "{m:?}");
    assert!(
        (m.downbeat - (-1.0f32).exp()).abs() < 1e-5,
        "a quarter second on"
    );
    let later = at(2.75);
    assert!((later.bar_phase - 0.875).abs() < 1e-6, "{later:?}");
    assert!((later.beat - (-2.5f32).exp()).abs() < 1e-5);
    let next_bar = at(3.0);
    assert!(next_bar.bar_phase < 1e-6 && (next_bar.downbeat - 1.0).abs() < 1e-6);
    assert_eq!(at(2.0).sure, 1.0, "a sure region");
    assert_eq!(at(8.0).sure, 0.0, "an unsure one");
}

/// Before the grid's first beat and after its last: the beats go on at
/// the tempo there, not sure at all before; no beats at all, no beat (the
/// grid's free one, still on its beat), unsure.
#[test]
fn off_the_grid_the_beat_runs_on() {
    let singing = Singing::default();
    let song = Song::of(&Rhythmed::new(grid(), None));
    let before = at(&song, &singing, 0.25);
    assert_eq!(before.sure, 0.0);
    assert!((before.beat_phase - 0.5).abs() < 1e-6, "{before:?}");
    let after = at(&song, &singing, 30.1);
    assert!((after.beat_phase - 0.2).abs() < 1e-5, "{after:?}");
    let empty = Song::of(&Rhythmed::new(Analysis::default(), None));
    let none = at(&empty, &singing, 3.3);
    assert_eq!((none.sure, none.beat_phase), (0.0, 0.0));
    assert!(!none.is_sure());
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
    assert_eq!(on.get(Driver::Beat), on.beat, "sure");
    let unsure = at(&song, &singing, 9.0);
    let faded = unsure.get(Driver::Beat);
    assert!(unsure.beat > 0.99 && faded == 0.0, "{unsure:?}");
    let unsure_off = at(&song, &singing, 9.2);
    assert_eq!(
        unsure_off.get(Driver::BeatPhase),
        unsure_off.beat_phase,
        "the phase runs on"
    );
    assert!((on.progress - 0.15).abs() < 1e-6);
    let beat = [on.beat_phase, on.bar_phase, on.beat, on.downbeat];
    assert_eq!(on.uniform()[0], beat);
    let none = read(&Heard {
        song: None,
        singing: &singing,
        now: 3.0,
        clock: 3.0,
        signals: Signals::QUIET,
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
    let drums = Driver::Drums;
    assert!(loud.get(drums) > 0.9 && quiet.get(drums) < 0.1, "{loud:?}");
    let bass = Driver::Bass;
    assert_eq!(loud.get(bass), quiet.get(bass), "the bass stays");
    assert!(loud.level > 0.9 && quiet.level < 0.1);
    assert!((at(5.005).melody - 69.0).abs() < 1e-6, "A4 where sure");
    assert_eq!(at(5.015).melody, 0.0, "not where unsure");
    assert!((at(5.005).get(Driver::Melody) - (69.0 - 36.0) / 48.0).abs() < 1e-6);
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

/// The beats counted from the grid's first and the bar's length come with
/// the beat, on the grid and before it; the shaders get them too.
#[test]
fn the_beats_are_counted_and_the_bar_measured() {
    let singing = Singing::default();
    let song = Song::of(&Rhythmed::new(grid(), None));
    let m = at(&song, &singing, 3.25);
    assert!((m.beats - 4.5).abs() < 1e-6 && m.bar_beats == 4.0, "{m:?}");
    let before = at(&song, &singing, 0.25);
    assert!(
        (before.beats + 1.5).abs() < 1e-6,
        "before the first: {before:?}"
    );
    assert_eq!(m.uniform()[4], [m.beats, 4.0, 0.0, 0.0]);
}

/// A steady grid beats as a grid read does: its phases and pulses.
#[test]
fn a_steady_grid_beats_as_a_read_one() {
    let singing = Singing::default();
    let song = Song::of(&Rhythmed::new(grid(), None));
    // 3.3 s: 4.6 beats from the grid's first, at 120 a minute.
    let read = at(&song, &singing, 3.3);
    let steady = Music::steady(4.6, 120.0, 4, 0.9, 2.3);
    for (a, b) in [
        (read.beat_phase, steady.beat_phase),
        (read.bar_phase, steady.bar_phase),
        (read.beat, steady.beat),
        (read.downbeat, steady.downbeat),
        (read.beats, steady.beats),
        (read.sure, steady.sure),
    ] {
        assert!((a - b).abs() < 1e-4, "{read:?} {steady:?}");
    }
    assert!(steady.is_sure() && !Music::steady(4.6, 120.0, 4, 0.2, 2.3).is_sure());
    assert!((steady.beat_length() - 0.5).abs() < 1e-6);
}

/// A hit lands by the grid's one rule ([`Grid::land`]) on the singing's
/// clock (the song's less how late the voice comes), with the lines sung
/// from then on: on the next beat near enough, on a bar's first with room
/// before a sung line; at once where the grid is unsure or there is none.
#[test]
fn a_hit_lands_by_the_grid_s_one_rule() {
    let a = grid();
    let song = Song::of(&Rhythmed::new(a.clone(), None));
    let singing = Singing::default();
    let late = 0.2;
    let heard = |now: f64| {
        read(&Heard {
            song: Some(&song),
            singing: &singing,
            now,
            clock: now - late,
            signals: Signals::QUIET,
        })
    };
    // A tenth of a second before the beat at 3.5 s: on it.
    let m = heard(3.4);
    let clock = 3.4 - late;
    assert!((m.land(clock, When::Beat) - (3.5 - late)).abs() < 1e-9);
    // With a line sung over the next bar's first (5.0 s), the one after.
    let mut sung = m;
    sung.lines[0] = Some((4.9 - late, 5.6 - late));
    let length = 1.5;
    let on = sung.land(clock, When::Downbeat { length });
    assert!((on - (7.0 - late)).abs() < 1e-9, "{on}");
    let grid = a.grid_at(3.4).on(3.4, clock, 1.0);
    let lines = [(4.9 - late, 5.6 - late)];
    assert_eq!(on, grid.land(clock, When::Downbeat { length }, &lines));
    // Unsure there, or no song: at once.
    assert_eq!(heard(8.1).land(8.1 - late, When::Beat), 8.1 - late);
    assert_eq!(Music::NEUTRAL.land(1.0, When::Downbeat { length }), 1.0);
}
