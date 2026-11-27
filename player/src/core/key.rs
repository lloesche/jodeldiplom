//! The key the song plays in: `n` semitones from its own, its tempo kept
//! (the engine's [`Command::Key`], every stem but the drums shifted; the
//! whole mix without stems). Mainly for karaoke, to sing in a comfortable
//! key.
//!
//! `-` and `+` (or `=`) move it a semitone, `0` puts the song's own back,
//! on the Karaoke tab (where it is a slider of its own, `Key`, beside the
//! voice and the delay) and on the Mixer tab (shown on the PLAYER strip);
//! the palette has them too. A change is heard within about a tenth of a
//! second. It is shown at once as asked ([`Playing::key`]); the engine
//! says what it plays in ([`told`]: a change it refused, a track picked
//! putting it back at 0). The song's notes on the Karaoke tab and the
//! scoring follow the key heard ([`heard`]), so singing in the new key
//! scores. The key stays for the songs the queue runs into, and goes back
//! to 0 when another song is picked (the song it is on picked again, as
//! Karaoke's Next does, keeps it).
//!
//! [`Playing::key`]: super::Playing::key

use jodel_playback::{Command, KEY_RANGE};

use super::keymap::Op;
use super::{Effect, State};

/// The key as shown: `+2`, `0`, `−3`.
pub fn text(semitones: i8) -> String {
    match semitones {
        0 => "0".to_owned(),
        s if s > 0 => format!("+{s}"),
        s => format!("−{}", s.unsigned_abs()),
    }
}

/// Carries out a key's op ([`Op::KeyDown`], [`Op::KeyUp`],
/// [`Op::KeyReset`]: the tabs' keys, the palette). With the Delay slider
/// at hand on the Karaoke tab or in the mode, `0` puts the delay back as
/// not nudged instead ([`super::lag::unset_nudge`]).
pub(crate) fn step(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let karaoke = state.side == super::Side::Karaoke || super::karaoke::in_mode(state);
    if op == Op::KeyReset && karaoke && state.karaoke.focus == super::sliders::Slider::Delay {
        return super::lag::unset_nudge(state);
    }
    let key = i32::from(state.playing.key);
    let to = match op {
        Op::KeyDown => key - 1,
        Op::KeyUp => key + 1,
        _ => 0,
    };
    set(state, to, fx);
}

/// The key asked for at `semitones` (within its range): shown at once and
/// sent to the engine, and said.
pub(crate) fn set(state: &mut State, semitones: i32, fx: &mut Vec<Effect>) {
    let range = i32::from(KEY_RANGE);
    let key = semitones.clamp(-range, range) as i8;
    if key == state.playing.key {
        let edge = if key == 0 {
            "its own"
        } else {
            "as far as it goes"
        };
        return state.info(format!("Key {}: {edge}", text(key)));
    }
    state.playing.key = key;
    fx.push(Effect::Engine(Command::Key { semitones: key }));
    let said = if key == 0 {
        "Key 0: the song's own".to_owned()
    } else {
        format!("Key {}: the tempo kept, the drums as they are", text(key))
    };
    state.info(said);
}

/// A new engine is started (another output, the engine started again):
/// nothing plays yet, the key asked for stays. What the new engine is told
/// first: `queue` loaded (unless it is empty), then the key (unless it is
/// the song's own), for the song played next.
pub fn for_new_engine(state: &mut State, queue: Vec<std::path::PathBuf>) -> Vec<Command> {
    let key = state.playing.key;
    state.playing = super::Playing {
        key,
        ..super::Playing::default()
    };
    let load = (!queue.is_empty()).then_some(Command::Load(queue));
    let key = (key != 0).then_some(Command::Key { semitones: key });
    load.into_iter().chain(key).collect()
}

/// The engine plays in `semitones` from now ([`jodel_playback::Event::Key`]).
pub(crate) fn told(state: &mut State, semitones: i8) {
    state.playing.key = semitones;
}

/// A position heard in `semitones`: the song's notes and the scoring are
/// in that key from there.
pub(crate) fn heard(state: &mut State, semitones: i8) {
    state.karaoke.singing.set_key(semitones);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_shows_with_its_sign() {
        assert_eq!(text(2), "+2");
        assert_eq!(text(0), "0");
        assert_eq!(text(-3), "−3");
    }
}
