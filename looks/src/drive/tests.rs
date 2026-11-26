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

/// The made-up song's curves: the beats and bars where it put them, the
/// kicks on one and three and the snares on two and four, choruses found
/// where it is loudest, singing in its lines, and the same curves every
/// time.
#[test]
fn the_curves_find_what_the_song_has() {
    let song = Song::new(90.0, 120.0);
    let c = song.curves();
    assert_eq!(c, song.curves(), "the same every time");
    assert_eq!(c.beats().len(), song.analysis.beats.len());
    assert!((c.bars()[1] - 2.0).abs() < 1e-9);
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
/// curve between its frames, a pulse from its moments, nothing kept.
#[test]
fn the_same_time_reads_the_same() {
    let c = Song::new(40.0, 100.0).curves();
    for t in [0.0, 3.21, 17.005, 39.9] {
        assert_eq!(c.at(t), c.at(t));
    }
    let downbeat = c.bars()[3];
    assert_eq!(c.at(downbeat).get(Driver::Downbeat), 1.0);
    let lvl = |t: f64| c.at(t).get(Driver::Level);
    let mid = lvl(10.005);
    let (a, b) = (lvl(10.0), lvl(10.01));
    assert!(mid >= a.min(b) - 1e-6 && mid <= a.max(b) + 1e-6);
}
