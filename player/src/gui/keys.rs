//! The keyboard as the core sees it: an iced key press as a [`Key`]
//! ([`core::keymap`](crate::core::keymap) says what it does), and the setup
//! screen's names for its keys.

use iced::keyboard::{key::Named, Key as IcedKey, Modifiers};

use crate::core::keymap::{self, Key, KeyCode, Op};
use crate::core::State;

/// An iced key as the core's [`Key`], with `modifiers` held.
pub fn key(key: &IcedKey, modifiers: Modifiers) -> Option<Key> {
    let code = match key {
        IcedKey::Named(named) => match named {
            Named::Enter => KeyCode::Enter,
            Named::Escape => KeyCode::Esc,
            Named::Backspace => KeyCode::Backspace,
            Named::Tab => KeyCode::Tab,
            Named::ArrowUp => KeyCode::Up,
            Named::ArrowDown => KeyCode::Down,
            Named::ArrowLeft => KeyCode::Left,
            Named::ArrowRight => KeyCode::Right,
            Named::PageUp => KeyCode::PageUp,
            Named::PageDown => KeyCode::PageDown,
            Named::Home => KeyCode::Home,
            Named::End => KeyCode::End,
            Named::Delete => KeyCode::Delete,
            Named::Space => KeyCode::Char(' '),
            _ => return None,
        },
        IcedKey::Character(text) => KeyCode::Char(text.chars().next()?),
        IcedKey::Unidentified => return None,
    };
    // Cmd+V on a Mac pastes, as ctrl+v does elsewhere.
    let paste = cfg!(target_os = "macos") && modifiers.logo() && code == KeyCode::Char('v');
    Some(Key {
        code,
        ctrl: modifiers.control() || paste,
        alt: modifiers.alt(),
        shift: !matches!(code, KeyCode::Char(_)) && modifiers.shift(),
    })
}

/// A key press as the core's [`Key`]: `plain` is the key without the
/// modifiers, `logical` with them (as the layout makes it). With alt the
/// key as if alone (Option+1 on a Mac is 1); a digit with shift is that
/// digit, shifted (whatever symbol the layout puts there).
pub fn pressed(plain: &IcedKey, logical: &IcedKey, modifiers: Modifiers) -> Option<Key> {
    let digit = matches!(plain, IcedKey::Character(t) if t.chars().next().is_some_and(|c| c.is_ascii_digit()));
    let shifted_digit = modifiers.shift() && digit;
    let which = if modifiers.alt() || shifted_digit {
        plain
    } else {
        logical
    };
    key(which, modifiers).map(|mut k| {
        k.shift |= shifted_digit;
        k
    })
}

/// Whether `key` let go of is the demo key, held
/// ([`crate::core::singing::Demo`]): the key alone, whatever is held with
/// it now or shows over the mode.
pub fn lets_go_of_demo(state: &State, key: &IcedKey) -> bool {
    let demo = |k: Key| {
        keymap::KEYMAP
            .iter()
            .any(|b| b.op == Op::KaraokeDemo && b.keys.contains(&k))
    };
    state.karaoke.demo.held() && self::key(key, Modifiers::empty()).is_some_and(demo)
}

/// The key's name as the setup screen takes it: one character, `"enter"`
/// or `"esc"` ([`crate::view::setup::choice`]).
pub fn setup_name(key: &IcedKey) -> String {
    match key {
        IcedKey::Named(Named::Enter) => "enter".to_owned(),
        IcedKey::Named(Named::Escape) => "esc".to_owned(),
        IcedKey::Character(text) => text.to_string(),
        _ => String::new(),
    }
}

/// Ctrl alone held (not AltGr, which is ctrl+alt): held in the browser it
/// previews the file selected.
pub fn ctrl_alone(modifiers: Modifiers) -> bool {
    modifiers.control() && !modifiers.alt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(c: &str) -> IcedKey {
        IcedKey::Character(c.into())
    }

    #[test]
    fn shifted_digits_and_alt_keys_are_the_plain_key_with_its_modifier() {
        let shift = Modifiers::SHIFT;
        // Shift+1 on a German layout is "!": the digit, shifted.
        let k = pressed(&ch("1"), &ch("!"), shift).unwrap();
        assert_eq!((k.code, k.shift), (KeyCode::Char('1'), true));
        // A capital is its letter, no shift.
        let k = pressed(&ch("v"), &ch("V"), shift).unwrap();
        assert_eq!((k.code, k.shift), (KeyCode::Char('V'), false));
        // Option+1 on a Mac makes "¡": the 1, with alt.
        let k = pressed(&ch("1"), &ch("¡"), Modifiers::ALT).unwrap();
        assert_eq!((k.code, k.alt), (KeyCode::Char('1'), true));
        let k = key(&IcedKey::Named(Named::ArrowUp), shift).unwrap();
        assert_eq!((k.code, k.shift), (KeyCode::Up, true));
        assert!(key(&IcedKey::Named(Named::F1), Modifiers::empty()).is_none());
    }
}
