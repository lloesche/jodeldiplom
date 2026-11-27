use super::*;
use crate::demo::Song;

/// A moment's pulse peaks on the moment itself (1 there, less just before
/// and just after), rising only [`LEAD`] before it.
#[test]
fn a_pulse_peaks_on_its_moment() {
    let m = [1.0, 2.0];
    assert_eq!(pulse_of(&m, 1.0, 0.1), 1.0);
    assert!(pulse_of(&m, 0.99, 0.1) < 1.0 && pulse_of(&m, 0.99, 0.1) > 0.0);
    assert!(pulse_of(&m, 1.01, 0.1) < 1.0);
    assert_eq!(pulse_of(&m, 1.0 - LEAD - 0.001, 0.1), 0.0);
    assert!(pulse_of(&m, 1.5, 0.1) < 0.01);
}

/// The made-up song's curves: the beat where it put it (a bar of four
/// every two seconds), the kicks on one and three and the snares on two
/// and four, choruses found where it is loudest, singing in its lines,
/// and the same curves every time.
#[test]
fn the_curves_find_what_the_song_has() {
    let song = Song::new(90.0, 120.0);
    let c = song.curves();
    assert_eq!(c, song.curves(), "the same every time");
    let b = c.beat(4.0);
    assert!(b.bar_phase.abs() < 1e-6 && b.phase.abs() < 1e-6, "{b:?}");
    assert!((c.beat(2.5).bar_phase - 0.25).abs() < 1e-6 && b.bar_beats == 4.0);
    assert!(
        c.kicks().len() > 20 && c.snares().len() > 20,
        "{} {}",
        c.kicks().len(),
        c.snares().len()
    );
    for &k in c.kicks() {
        let beat = (k / 0.5).round() as usize;
        assert_eq!(beat % 2, 0, "a kick on {k}");
    }
    for &s in c.snares() {
        let beat = (s / 0.5).round() as usize;
        assert_eq!(beat % 2, 1, "a snare on {s}");
    }
    // The first chorus is the third section (32 s to 48 s).
    assert!(
        c.at(40.0).get(Driver::Chorus) > 0.8,
        "{}",
        c.at(40.0).get(Driver::Chorus)
    );
    assert!(c.at(8.0).get(Driver::Chorus) < 0.2);
    assert!(c.at(40.0).get(Driver::Energy) > c.at(8.0).get(Driver::Energy));
    let (a, b) = song.lines[0];
    assert!(c.at((a + b) / 2.0).get(Driver::Singing) > 0.9);
    assert!(c.at(4.0).get(Driver::Instrumental) > 0.5);
    // Each section a new part.
    assert_ne!(c.at(20.0).get(Driver::Part), c.at(40.0).get(Driver::Part));
}

/// What is read at a time is the same however the frames came to it: a
/// curve between its frames, a pulse from its moments, nothing kept: the
/// times read in another order, from curves made again, read the same.
#[test]
fn the_same_time_reads_the_same() {
    let song = Song::new(40.0, 100.0);
    let c = song.curves();
    let times = [0.0, 3.21, 17.005, 39.9, 8.0];
    let forward: Vec<Signals> = times.iter().map(|&t| c.at(t)).collect();
    let again = song.curves();
    let back: Vec<Signals> = times.iter().rev().map(|&t| again.at(t)).collect();
    assert!(forward.iter().eq(back.iter().rev()));
    let downbeat = song.analysis.beats[song.analysis.downbeats[3]];
    assert_eq!(c.at(downbeat).get(Driver::Downbeat), 1.0);
    let lvl = |t: f64| c.at(t).get(Driver::Level);
    let mid = lvl(10.005);
    let (a, b) = (lvl(10.0), lvl(10.01));
    assert!(mid >= a.min(b) - 1e-6 && mid <= a.max(b) + 1e-6);
}

/// Every driver from the song is set by its curves somewhere in a song
/// that has all a song can have (a minor key, a melody), and none of the
/// live ones ever is: those are the singing's to set.
#[test]
fn every_song_driver_is_set_and_no_live_one() {
    let mut song = Song::new(90.0, 120.0);
    song.analysis.key = Some(jodel_rhythm::key::Key {
        class: 3,
        confidence: 1.0,
        tagged: true,
    });
    let frames = song.waveform.mix.rms.len();
    song.waveform.melody = Some(jodel_rhythm::waveform::Melody {
        cents: (0..frames).map(|i| 5700 + (i % 1200) as u16).collect(),
        sure: vec![255; frames],
    });
    let c = song.curves();
    assert_eq!(c.at(10.0).get(Driver::Minor), 1.0, "a minor key");
    let times: Vec<f64> = (0..1800).map(|i| f64::from(i) * 0.05).collect();
    for d in Driver::ALL {
        let moved = times.iter().any(|&t| c.at(t).get(d) != 0.0);
        match d.origin() {
            Origin::Song => assert!(moved, "{} never set", d.name()),
            Origin::Live => assert!(!moved, "{} set from the song", d.name()),
        }
    }
}

/// Where the grid is unsure the beat's and the bar's pulses are none, and
/// its phases run on, unbroken across the regions' edge: a ramp moves on as
/// the grid does.
#[test]
fn an_unsure_grid_pulses_not_and_its_phases_run_on() {
    let mut song = Song::new(20.0, 120.0);
    song.analysis.regions = vec![
        jodel_rhythm::Region {
            start: 0.0,
            end: 10.25,
            confidence: 0.9,
        },
        jodel_rhythm::Region {
            start: 10.25,
            end: 20.0,
            confidence: 0.05,
        },
    ];
    let c = song.curves();
    assert_eq!(c.at(6.0).get(Driver::Beat), 1.0, "on a sure beat");
    assert_eq!(c.at(12.0).get(Driver::Beat), 0.0, "on an unsure one");
    assert_eq!(c.at(12.0).get(Driver::Downbeat), 0.0);
    // Across the edge at 10.25 s, half a beat in, the phase moves on as it
    // did before it.
    let phase = |t: f64| c.at(t).get(Driver::BeatPhase);
    assert!((phase(10.2) - 0.4).abs() < 1e-4 && (phase(10.3) - 0.6).abs() < 1e-4);
    assert!((phase(12.25) - 0.5).abs() < 1e-4, "it runs on");
}
