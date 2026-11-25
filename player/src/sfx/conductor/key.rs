//! The song's key as the conductor plays in it: a pentatonic view
//! ([`Pentatonic`]) of the key the analysis found
//! ([`jodel_rhythm::key::Key`]), shifted as the song is heard, on which
//! every pitched sound of the game sits. A pentatonic has no half step to
//! clash with the song's chords, so no chord needs tracking.

/// Major or minor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    /// Its pentatonic 1 2 3 5 6.
    Major,
    /// Its pentatonic 1 ♭3 4 5 ♭7.
    Minor,
}

/// A key's pentatonic: its tonic's pitch class (0 C to 11 B) and its mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Pentatonic {
    tonic: u8,
    mode: Mode,
}

/// The degrees of a pentatonic in an octave.
pub(crate) const DEGREES: i32 = 5;

/// The steps of the major and the minor pentatonic, semitones over the
/// tonic.
const MAJOR: [i32; 5] = [0, 2, 4, 7, 9];
const MINOR: [i32; 5] = [0, 3, 5, 7, 10];

impl Pentatonic {
    /// The pentatonic of `key` heard `shift` semitones up (down when
    /// negative).
    pub(crate) fn of(key: &jodel_rhythm::key::Key, shift: i8) -> Pentatonic {
        let tonic = (i32::from(key.tonic()) + i32::from(shift)).rem_euclid(12);
        Pentatonic {
            tonic: tonic as u8,
            mode: if key.minor() {
                Mode::Minor
            } else {
                Mode::Major
            },
        }
    }

    /// The note `degree` steps up the pentatonic from its tonic at `home`
    /// (a MIDI note on the tonic; degree 5 the octave over it, -1 the step
    /// under it).
    pub(crate) fn note(&self, degree: i32, home: i32) -> f32 {
        let steps = match self.mode {
            Mode::Major => MAJOR,
            Mode::Minor => MINOR,
        };
        let octave = degree.div_euclid(DEGREES);
        let step = steps[degree.rem_euclid(DEGREES) as usize];
        (home + 12 * octave + step) as f32
    }

    /// The tonic as a MIDI note at or above `floor`, within an octave of
    /// it.
    pub(crate) fn home(&self, floor: i32) -> i32 {
        floor + (i32::from(self.tonic) - floor).rem_euclid(12)
    }

    /// The degree of the `k`-th tone of the tonic triad up the pentatonic
    /// (0 the tonic, 1 the third, 2 the fifth, 3 the octave, and on): the
    /// chord tones the arpeggios run through.
    pub(crate) fn chord(&self, k: i32) -> i32 {
        let third = match self.mode {
            Mode::Major => 2,
            Mode::Minor => 1,
        };
        let tone = [0, third, 3][k.rem_euclid(3) as usize];
        DEGREES * k.div_euclid(3) + tone
    }
}
