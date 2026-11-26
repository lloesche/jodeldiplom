//! The help screen and the command palette as they are used: the help
//! opened, searched and scrolled, the palette typed, moved through and
//! run. What they list comes from the keymap ([`keymap::help_matching`],
//! [`keymap::palette_matches`]).

use crate::core::keymap;
use crate::core::update::{apply_op, offset};
use crate::core::{Effect, Filter, Palette, Screen, State};

/// The palette's matches and the first one shown in `height` rows (the
/// input line takes the first row).
pub fn palette_window(state: &State, height: usize) -> (Vec<keymap::PaletteEntry>, usize) {
    let Some(palette) = &state.palette else {
        return (Vec::new(), 0);
    };
    let matches = keymap::palette_matches(&palette.text);
    let shown = height.saturating_sub(1).max(1);
    let start = palette.cursor.saturating_sub(shown - 1);
    (matches, start)
}

/// The help screen's sections, as its search leaves them.
pub fn help_shown(state: &State) -> Vec<keymap::HelpSection> {
    keymap::help_matching(state.help_search.as_ref().map_or("", |s| s.text.as_str()))
}

/// The number of lines the help screen has, as its search leaves them.
pub fn help_line_count(state: &State) -> usize {
    help_shown(state).iter().map(|s| s.lines.len() + 2).sum()
}

/// How many lines of the help's text show: the window's
/// ([`crate::core::Fit::help_rows`]), but for the search line's while the
/// help is searched.
fn help_height(state: &State) -> usize {
    let search = state.help_search.is_some();
    usize::from(state.fit.help_rows).saturating_sub(usize::from(search))
}

/// Whether the search of the help is being typed.
pub(crate) fn typing(state: &State) -> bool {
    state.screen == Screen::Help && state.help_search.as_ref().is_some_and(|s| s.editing)
}

/// Scrolls the help `delta` lines, its last line no higher than the foot.
pub(crate) fn scroll(state: &mut State, delta: i32) {
    let visible = help_height(state).max(1);
    let max = help_line_count(state).saturating_sub(visible);
    state.help_scroll = offset(state.help_scroll, delta).min(max);
}

/// [`keymap::Op::Help`]: the help opens whole (a search from before is gone), or
/// closes.
pub(crate) fn toggle(state: &mut State) {
    state.screen = if state.screen == Screen::Help {
        Screen::Main
    } else {
        state.help_search = None;
        Screen::Help
    };
}

/// [`keymap::Op::Filter`] on the help: its search is typed (again with what was
/// typed).
pub(crate) fn search(state: &mut State) {
    let text = state.help_search.take().map(|s| s.text).unwrap_or_default();
    state.help_search = Some(Filter {
        text,
        editing: true,
    });
}

/// [`keymap::Op::Palette`]: the palette opens over the main screen.
pub(crate) fn open_palette(state: &mut State) {
    state.screen = Screen::Main;
    state.palette = Some(Palette::default());
}

/// Runs the palette's selected match, and closes it.
pub(crate) fn run_palette(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(palette) = state.palette.take() else {
        return;
    };
    let matches = keymap::palette_matches(&palette.text);
    if let Some(entry) = matches.get(palette.cursor) {
        state.screen = Screen::Main;
        apply_op(state, entry.op, fx);
    }
}

/// [`crate::core::Action::PaletteRun`]: the palette's match `index` runs.
pub(crate) fn run_palette_at(state: &mut State, index: usize, fx: &mut Vec<Effect>) {
    if let Some(palette) = &mut state.palette {
        palette.cursor = index;
    }
    run_palette(state, fx);
}

/// `↑` `↓` in the palette, `delta` matches; whether it is open to take them.
pub(crate) fn palette_moved(state: &mut State, delta: i32) -> bool {
    let Some(palette) = &mut state.palette else {
        return false;
    };
    let n = keymap::palette_matches(&palette.text).len();
    palette.cursor = offset(palette.cursor, delta).min(n.saturating_sub(1));
    true
}

/// `text` typed (or pasted) into the palette or the help's search, where
/// one is being typed; whether one took it.
pub(crate) fn typed(state: &mut State, text: &str) -> bool {
    if let Some(palette) = &mut state.palette {
        palette.text.push_str(text);
        palette.cursor = 0;
    } else if typing(state) {
        if let Some(search) = state.help_search.as_mut() {
            search.text.push_str(text);
        }
        state.help_scroll = 0;
    } else {
        return false;
    }
    true
}

/// `backspace` in the palette or the help's search: its last character
/// goes, and with none left the palette or the search; whether one took
/// it.
pub(crate) fn erased(state: &mut State) -> bool {
    if let Some(palette) = &mut state.palette {
        if palette.text.pop().is_none() {
            state.palette = None;
        }
    } else if typing(state) {
        if let Some(search) = &mut state.help_search {
            if search.text.pop().is_none() {
                state.help_search = None;
            }
        }
        state.help_scroll = 0;
    } else {
        return false;
    }
    true
}

/// `enter` in the palette (its match runs) or the help's search (it stays,
/// the keys are the help's again; an empty one goes); whether one took it.
pub(crate) fn accepted(state: &mut State, fx: &mut Vec<Effect>) -> bool {
    if state.palette.is_some() {
        run_palette(state, fx);
    } else if typing(state) {
        if let Some(search) = state.help_search.as_mut() {
            search.editing = false;
        }
        if state
            .help_search
            .as_ref()
            .is_some_and(|s| s.text.is_empty())
        {
            state.help_search = None;
        }
    } else {
        return false;
    }
    true
}

/// `esc` on the help: its search is cleared first, then the help is left.
pub(crate) fn back(state: &mut State) {
    if state.help_search.is_some() {
        state.help_search = None;
        state.help_scroll = 0;
    } else {
        state.screen = Screen::Main;
    }
}

/// Whether [`keymap::Op::Paste`] has somewhere to paste to: the palette, the help's
/// search being typed, or the main screen (the browser's filter).
pub(crate) fn can_paste(state: &State) -> bool {
    state.palette.is_some() || state.screen == Screen::Main || typing(state)
}
