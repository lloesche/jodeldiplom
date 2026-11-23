//! The Karaoke tab's sliders ([`Slider`]): the voice stem's level, the
//! singer's delay and the key the song is played in.

use jodel_playback::KEY_RANGE;

use crate::core::desk::taper;
use crate::core::State;

/// The tab's sliders: the voice stem's level, the singer's delay in force
/// ([`crate::core::lag`]), which the arrows and the pointer nudge, and the
/// key ([`crate::core::key`]). The microphone is never played, so it has
/// none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Slider {
    /// The voice stem's level on the tab and in the mode: a position along
    /// its travel, 0 to 100 %, read on the fader's taper ([`voice_db`]).
    #[default]
    Voice,
    /// The singer's delay in force, in ms, from 0 to [`DELAY_MOST`]: moved,
    /// it moves the nudge (the tab only).
    Delay,
    /// The key the song plays in, in semitones from its own,
    /// −[`KEY_RANGE`] to +[`KEY_RANGE`] (the tab only).
    Key,
}

impl Slider {
    /// The tab's, top to bottom.
    pub const ALL: [Slider; 3] = [Slider::Voice, Slider::Delay, Slider::Key];
    /// The mode's.
    pub const MODE: [Slider; 1] = [Slider::Voice];

    /// Its label.
    pub fn label(self) -> &'static str {
        match self {
            Slider::Voice => "Voice",
            Slider::Delay => "Delay",
            Slider::Key => "Key",
        }
    }

    /// Its lowest value, in its unit.
    pub fn least(self) -> i32 {
        match self {
            Slider::Voice => 0,
            Slider::Delay => 0,
            Slider::Key => -i32::from(KEY_RANGE),
        }
    }

    /// Its highest value.
    pub fn most(self) -> i32 {
        match self {
            Slider::Voice => 100,
            Slider::Delay => DELAY_MOST,
            Slider::Key => i32::from(KEY_RANGE),
        }
    }

    /// The finest step of its value, in its unit: where a click or drag
    /// on it lands.
    pub fn unit(self) -> i32 {
        match self {
            Slider::Voice | Slider::Key => 1,
            Slider::Delay => super::lag::NUDGE_STEP,
        }
    }

    /// `value` a step down or `up`, as `←`/`→` and the wheel move it
    /// (`fine`: with shift), within its range. The voice goes by 1 dB on
    /// the taper, from −∞ up to [`taper::LAST_MARK_DB`] and from there (or
    /// below it) down to −∞, as a fader's keys do; with shift by 1 % of
    /// its travel. The delay goes by its nudge's step, the key by a
    /// semitone.
    pub fn stepped(self, value: i32, up: bool, fine: bool) -> i32 {
        let sign = if up { 1 } else { -1 };
        let moved = match self {
            Slider::Voice if !fine => voice_stepped(value, up),
            Slider::Voice | Slider::Key => value + sign,
            Slider::Delay => value + sign * super::lag::NUDGE_STEP,
        };
        moved.clamp(self.least(), self.most())
    }

    /// Its value in `state`.
    pub fn value(self, state: &State) -> i32 {
        match self {
            Slider::Voice => i32::from(state.config.karaoke_voice),
            Slider::Delay => delay_ms(state),
            Slider::Key => i32::from(state.playing.key),
        }
    }

    /// Its value as a share of its range, 0 to 1.
    pub fn share(self, state: &State) -> f64 {
        f64::from(self.value(state) - self.least()) / f64::from(self.most() - self.least())
    }

    /// Its value as shown, right-aligned: ` −12 dB`, `  −∞ dB`, ` 150 ms`,
    /// `     +2`.
    pub fn text(self, state: &State) -> String {
        match self {
            Slider::Voice => {
                let db = voice_db(state.config.karaoke_voice);
                format!("{:>7}", format!("{} dB", taper::number(db, 0, false)))
            }
            Slider::Delay => format!("{:>4} ms", self.value(state)),
            Slider::Key => format!("{:>7}", super::key::text(state.playing.key)),
        }
    }
}

/// The voice's level in dB at `percent` of the Voice slider's travel: the
/// fader's taper from its bottom (−∞, 0 %) up to its 0 dB (100 %, the
/// voice at its own level), so the slider spends its travel where a change
/// is heard, as a fader does.
pub fn voice_db(percent: u8) -> f64 {
    taper::db_at(f64::from(percent.min(100)) / 100.0 * taper::UNITY)
}

/// The Voice slider's position (0 to 100 %) nearest `db` (0 dB and above:
/// the top).
fn voice_percent(db: f64) -> i32 {
    (taper::position_of(db.min(0.0)) / taper::UNITY * 100.0).round() as i32
}

/// The Voice slider at `value` moved by 1 dB down or `up`, to the next
/// whole dB (the value shown); from below the last mark up to it, from it
/// (or below it) down to −∞. A dB is a whole percent of the travel or
/// more down to the last mark, so each step moves the slider.
fn voice_stepped(value: i32, up: bool) -> i32 {
    let last = voice_percent(taper::LAST_MARK_DB);
    if up && value < last {
        return last;
    }
    if !up && value <= last {
        return 0;
    }
    let db = voice_db(value.clamp(0, 100) as u8).round();
    let to = if up { db + 1.0 } else { db - 1.0 };
    let moved = voice_percent(to.max(taper::LAST_MARK_DB));
    if moved != value {
        moved
    } else if up {
        value + 1
    } else {
        value - 1
    }
}

/// The most the Delay slider shows, in ms.
pub const DELAY_MOST: i32 = 600;

/// The singer's delay in force as the Delay slider has it, in ms: the
/// base the scoring takes and the nudge on it (on the tab, where no song
/// is sung, it is what the scoring takes at once).
fn delay_ms(state: &State) -> i32 {
    let base = (state.karaoke.lag.base * 1000.0).round() as i32;
    base + super::lag::nudge_ms(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_voice_slider_is_the_taper_up_to_0_db_and_steps_a_db() {
        assert_eq!(voice_db(100), 0.0);
        assert_eq!(voice_db(0), f64::NEG_INFINITY);
        assert!((voice_db(25) - -30.0).abs() < 1e-9);
        assert!((voice_db(10) - taper::LAST_MARK_DB).abs() < 1e-9);
        assert_eq!(voice_db(44).round(), -20.0);
        for percent in 0..=100u8 {
            assert_eq!(voice_percent(voice_db(percent)), i32::from(percent));
        }
        let v = Slider::Voice;
        // From the top a dB at a time, each shown as a whole dB, down to
        // the last mark, then off; and back up the same way.
        let mut value = 100;
        let mut shown = 0.0;
        while value > 10 {
            value = v.stepped(value, false, false);
            shown -= 1.0;
            assert_eq!(voice_db(value as u8).round(), shown, "{value}");
        }
        assert_eq!((value, shown), (10, taper::LAST_MARK_DB));
        assert_eq!(v.stepped(10, false, false), 0);
        assert_eq!(v.stepped(5, false, false), 0);
        assert_eq!(v.stepped(0, false, false), 0);
        assert_eq!(v.stepped(0, true, false), 10);
        assert_eq!(v.stepped(5, true, false), 10);
        assert_eq!(v.stepped(10, true, false), 11);
        assert_eq!(v.stepped(99, true, false), 100);
        assert_eq!(v.stepped(100, true, false), 100);
        // With shift, a percent of the travel.
        assert_eq!(v.stepped(5, true, true), 6);
        assert_eq!(v.stepped(0, false, true), 0);
        assert_eq!(Slider::Delay.stepped(150, true, false), 160);
        // The key a semitone a step, within its range.
        assert_eq!(Slider::Key.stepped(2, true, false), 3);
        assert_eq!(Slider::Key.stepped(6, true, false), 6);
        assert_eq!(Slider::Key.stepped(-6, false, true), -6);
    }
}
