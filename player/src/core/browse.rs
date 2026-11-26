//! The browser side of the list: its rows, the filter and the search below
//! a folder, typing to jump to a name, folders listed, entered and left;
//! and the cursor and scroll of either side of the list.

use std::path::{Path, PathBuf};

use crate::browser::{self, EntryKind};
use crate::core::keymap::Op;
use crate::core::{
    preview, queue, Effect, Filter, Placing, Screen, Search, Side, State, TYPE_AHEAD_MS,
};

/// One row of the browser side of the list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// [`crate::core::BrowserState::entries`]`[i]`.
    Entry(usize),
    /// [`Search::hits`]`[i]`.
    Hit(usize),
    /// "Search subfolders for …", under a filtered folder.
    SearchBelow,
}

/// The rows the browser side shows: the search results, or the folder
/// (filtered, with a "search subfolders" row under a non-empty filter).
pub fn browser_rows(state: &State) -> Vec<Row> {
    let b = &state.browser;
    if let Some(search) = &b.search {
        return (0..search.hits.len()).map(Row::Hit).collect();
    }
    match b.filter.as_ref().filter(|f| !f.text.is_empty()) {
        Some(filter) => {
            let mut rows: Vec<Row> = b
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| browser::matches(e, &filter.text, state.details.get(&e.path)))
                .map(|(i, _)| Row::Entry(i))
                .collect();
            rows.push(Row::SearchBelow);
            rows
        }
        None => (0..b.entries.len()).map(Row::Entry).collect(),
    }
}

/// The entry a browser row stands for.
pub fn row_entry(state: &State, row: Row) -> Option<&browser::Entry> {
    match row {
        Row::Entry(i) => state.browser.entries.get(i),
        Row::Hit(i) => state.browser.search.as_ref()?.hits.get(i),
        Row::SearchBelow => None,
    }
}

/// The entry of the browser row under the cursor, if it is one.
pub(crate) fn selected_entry(state: &State) -> Option<&browser::Entry> {
    let row = *browser_rows(state).get(state.browser.cursor)?;
    row_entry(state, row)
}

/// The browser row showing `path`, if one does.
fn row_of(state: &State, path: &Path) -> Option<usize> {
    browser_rows(state)
        .iter()
        .position(|&row| row_entry(state, row).is_some_and(|e| e.path == path))
}

/// The path of the browser row under the cursor, if it is an entry.
fn selected_path(state: &State) -> Option<PathBuf> {
    selected_entry(state).map(|e| e.path.clone())
}

/// After the browser's rows changed (a filter typed or cleared): the cursor
/// stays on `path` where it is still shown, else goes to the first row.
fn reselect(state: &mut State, path: Option<PathBuf>) {
    let row = path.and_then(|path| row_of(state, &path)).unwrap_or(0);
    set_cursor(state, Side::Browser, row);
}

/// The cursor and the first row shown of `side`, where it is a list (the
/// browser or the queue), to be changed.
fn view_mut(state: &mut State, side: Side) -> Option<(&mut usize, &mut usize)> {
    match side {
        Side::Browser => Some((&mut state.browser.cursor, &mut state.browser.scroll)),
        Side::Queue => Some((&mut state.queue.cursor, &mut state.queue.scroll)),
        Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth | Side::Shader => None,
    }
}

/// How many rows `side` has (none, for a tab that is no list).
pub fn list_len(state: &State, side: Side) -> usize {
    match side {
        Side::Browser => browser_rows(state).len(),
        Side::Queue => state.queue.paths.len(),
        Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth | Side::Shader => 0,
    }
}

/// The selected row and the first row shown of `side` (0 and 0, for a tab
/// that is no list).
pub fn list_cursor(state: &State, side: Side) -> (usize, usize) {
    match side {
        Side::Browser => (state.browser.cursor, state.browser.scroll),
        Side::Queue => (state.queue.cursor, state.queue.scroll),
        Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth | Side::Shader => (0, 0),
    }
}

/// Puts `side`'s cursor on `cursor` (on its last row, past the end), with
/// that row in view, whether `side` shows or not.
pub(crate) fn set_cursor(state: &mut State, side: Side, cursor: usize) {
    let last = list_len(state, side).saturating_sub(1);
    if let Some((at, _)) = view_mut(state, side) {
        *at = cursor.min(last);
    }
    fix_scroll(state, side);
}

/// How many rows `side` shows on the main screen: the window's
/// ([`crate::core::Fit::list_rows`]), but for the filter line's in the
/// browser while it is filtered.
pub fn list_height(state: &State, side: Side) -> usize {
    let filter = side == Side::Browser && state.browser.filter.is_some();
    usize::from(state.fit.list_rows).saturating_sub(usize::from(filter))
}

/// Scrolls `side` so that its cursor's row is in view.
pub(crate) fn fix_scroll(state: &mut State, side: Side) {
    let height = list_height(state, side).max(1);
    let len = list_len(state, side);
    let Some((&mut cursor, scroll)) = view_mut(state, side) else {
        return;
    };
    *scroll = (*scroll).min(len.saturating_sub(height));
    if cursor < *scroll {
        *scroll = cursor;
    } else if cursor >= *scroll + height {
        *scroll = cursor + 1 - height;
    }
}

/// Just the `..` row of `dir` (none at the very top).
fn parent_row(dir: &Path) -> Vec<browser::Entry> {
    browser::parent_of(dir)
        .map(browser::Entry::parent)
        .into_iter()
        .collect()
}

/// [`crate::core::Action::Listed`] for the folder shown: its entries (or
/// why not), the row to select in it, and the file to play once it is
/// listed.
pub(crate) fn listed(
    state: &mut State,
    result: Result<Vec<browser::Entry>, String>,
    fx: &mut Vec<Effect>,
) {
    // The remembered folder is gone (or its drive): the start folder.
    if let (Err(_), Some(fallback)) = (&result, state.browser.fallback.take()) {
        return enter_dir(state, fallback, fx);
    }
    state.browser.fallback = None;
    let b = &mut state.browser;
    b.loading = false;
    match result {
        Ok(entries) => {
            b.entries = entries;
            b.error = None;
        }
        Err(error) => {
            b.entries = parent_row(&b.dir);
            b.error = Some(error);
        }
    }
    let select = state
        .autoplay
        .clone()
        .or_else(|| state.browser.last_child.get(&state.browser.dir).cloned());
    let row = select.and_then(|path| row_of(state, &path)).unwrap_or(0);
    set_cursor(state, Side::Browser, row);
    if let Some(path) = state.autoplay.take() {
        match row_of(state, &path) {
            Some(row) => queue::play_browser_from(state, row, fx),
            None => state.error(format!("Cannot play {}", path.display())),
        }
    }
}

/// Shows `dir` in the browser on the main screen, with `select` selected
/// once it is listed.
pub(crate) fn jump_to(
    state: &mut State,
    dir: PathBuf,
    select: Option<PathBuf>,
    fx: &mut Vec<Effect>,
) {
    state.screen = Screen::Main;
    state.side = Side::Browser;
    state.browser.filter = None;
    enter_dir(state, dir.clone(), fx);
    match select {
        Some(file) => {
            state.browser.last_child.insert(dir, file);
        }
        None => {
            state.browser.last_child.remove(&dir);
        }
    }
}

/// Where [`Op::GoTrack`], [`Op::GoAlbum`] and [`Op::GoArtist`] go for
/// the playing track: the folder to show and what to select in it. The
/// track's folder is its album; the folder above that its artist, unless
/// it is the root of a drive or there is none (then the album).
pub(crate) fn playing_jump(state: &State, op: Op) -> Option<(PathBuf, Option<PathBuf>)> {
    let track = state.playing.track.as_ref()?;
    let path = track.info.path.clone();
    let album = path.parent()?.to_path_buf();
    let artist = album
        .parent()
        .filter(|a| a.parent().is_some())
        .map(PathBuf::from);
    match (op, artist) {
        (Op::GoArtist, Some(artist)) => Some((artist, Some(album))),
        _ => Some((album, Some(path))),
    }
}

/// Shows `dir` in the browser, listed in the background; until it is, just
/// its `..` row.
pub(crate) fn enter_dir(state: &mut State, dir: PathBuf, fx: &mut Vec<Effect>) {
    // Gone somewhere else: no falling back from there.
    state.browser.fallback = None;
    if let Some(selected) = selected_path(state) {
        let current = state.browser.dir.clone();
        state.browser.last_child.insert(current, selected);
    }
    if state.browser.search.take().is_some() {
        fx.push(Effect::CancelSearch);
    }
    // Going up reselects the folder we came from.
    let came_from = state.browser.dir.clone();
    if browser::parent_of(&came_from).as_deref() == Some(dir.as_path()) {
        state.browser.last_child.insert(dir.clone(), came_from);
    }
    let b = &mut state.browser;
    b.dir = dir.clone();
    // What was typed to jump to a name belonged to the folder left.
    b.typed.clear();
    // `..` is there at once, so a slow or failing listing never traps us.
    b.entries = parent_row(&dir);
    b.loading = true;
    b.error = None;
    b.cursor = 0;
    b.scroll = 0;
    b.filter = None;
    fx.push(Effect::ListDir(dir));
}

/// Up one folder, with the folder we came from selected; out of a search
/// first.
pub(crate) fn go_parent(state: &mut State, fx: &mut Vec<Effect>) {
    if state.browser.search.is_some() {
        state.browser.search = None;
        fx.push(Effect::CancelSearch);
        set_cursor(state, Side::Browser, 0);
        return;
    }
    if let Some(parent) = browser::parent_of(&state.browser.dir) {
        enter_dir(state, parent, fx);
    }
}

/// A character typed on the list: into the browser's filter while it is
/// typed, else a jump to the name starting with it.
pub(crate) fn typed(state: &mut State, c: char) {
    if state.browser.filter.as_ref().is_some_and(|f| f.editing) {
        let selected = selected_path(state);
        if let Some(filter) = state.browser.filter.as_mut() {
            filter.text.push(c);
        }
        reselect(state, selected);
    } else {
        type_ahead(state, c);
    }
}

/// Jumps to the browser row whose name starts with what was typed, as a
/// file explorer does: keys within [`TYPE_AHEAD_MS`] build one prefix; one
/// letter typed again moves on to the next name starting with it.
fn type_ahead(state: &mut State, c: char) {
    let now = state.now_ms;
    let b = &mut state.browser;
    if now.saturating_sub(b.typed_at) > TYPE_AHEAD_MS {
        b.typed.clear();
    }
    b.typed_at = now;
    b.typed.extend(c.to_lowercase());
    let typed = b.typed.clone();
    let first = typed.chars().next().unwrap_or(c);
    // "a", "aa", "aaa": cycle through the names starting with "a".
    let cycling = typed.chars().all(|t| t == first);
    let prefix = if cycling { first.to_string() } else { typed };
    // The names as the list shows them (the title tag once it is read).
    let side = state.side;
    let names: Vec<Option<String>> = match side {
        Side::Browser => browser_rows(state)
            .iter()
            .map(|&row| {
                row_entry(state, row)
                    .filter(|e| e.kind != EntryKind::Parent)
                    .map(|e| shown_name(state, e).to_lowercase())
            })
            .collect(),
        Side::Queue => state
            .queue
            .paths
            .iter()
            .map(|path| Some(shown_title(state, path).to_lowercase()))
            .collect(),
        Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth | Side::Shader => Vec::new(),
    };
    if names.is_empty() {
        return;
    }
    let cursor = list_cursor(state, side).0.min(names.len() - 1);
    // Cycling starts after the current row; a longer prefix may stay on it.
    let start = if cycling { cursor + 1 } else { cursor };
    let hit = (0..names.len())
        .map(|i| (start + i) % names.len())
        .find(|&i| names[i].as_deref().is_some_and(|n| n.starts_with(&prefix)));
    if let Some(row) = hit {
        set_cursor(state, side, row);
    }
}

/// The name a browser row shows: a search hit's relative path, a folder's
/// name, a file's title tag once read, else its file name.
fn shown_name(state: &State, entry: &browser::Entry) -> String {
    if entry.relative.is_some() || entry.kind != EntryKind::File {
        return entry.label().to_owned();
    }
    state
        .details
        .get(&entry.path)
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| entry.name.clone())
}

/// The name a queue row shows: the title tag once read, else the file
/// name.
pub fn shown_title(state: &State, path: &Path) -> String {
    state
        .details
        .get(path)
        .and_then(|d| d.title.clone())
        .unwrap_or_else(|| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
}

/// Searches the folder's subfolders for the filter's text, its results in
/// place of the folder's rows (the "search subfolders" row under a filter,
/// and a filter that names no path).
pub(crate) fn search_below(state: &mut State, fx: &mut Vec<Effect>) {
    let query = state
        .browser
        .filter
        .as_ref()
        .map(|f| f.text.clone())
        .unwrap_or_default();
    if query.is_empty() {
        return;
    }
    let b = &mut state.browser;
    b.next_search += 1;
    let id = b.next_search;
    b.search = Some(Search {
        id,
        query: query.clone(),
        hits: Vec::new(),
        done: false,
    });
    if let Some(filter) = &mut b.filter {
        filter.editing = false;
    }
    let root = b.dir.clone();
    state.side = Side::Browser;
    set_cursor(state, Side::Browser, 0);
    fx.push(Effect::Search { id, root, query });
}

/// [`crate::core::Action::SearchHits`]: more results of search `id`, while
/// it is the one shown.
pub(crate) fn search_hits(state: &mut State, id: u64, hits: Vec<browser::Entry>) {
    if let Some(search) = state.browser.search.as_mut().filter(|s| s.id == id) {
        search.hits.extend(hits);
    }
}

/// [`crate::core::Action::SearchDone`]: search `id` found all it will.
pub(crate) fn search_done(state: &mut State, id: u64) {
    if let Some(search) = state.browser.search.as_mut().filter(|s| s.id == id) {
        search.done = true;
    }
}

/// [`crate::core::Action::JumpMissed`]: the filter's `text` is no path, so
/// it is searched for below the folder (while it is still what is typed).
pub(crate) fn jump_missed(state: &mut State, text: &str, fx: &mut Vec<Effect>) {
    if state
        .browser
        .filter
        .as_ref()
        .is_some_and(|f| f.text == text)
    {
        search_below(state, fx);
    }
}

/// `/` on the main screen: the browser's filter typed (again with what was
/// typed), out of a search.
pub(crate) fn open_filter(state: &mut State, fx: &mut Vec<Effect>) {
    state.screen = Screen::Main;
    state.side = Side::Browser;
    if state.browser.search.take().is_some() {
        fx.push(Effect::CancelSearch);
    }
    let text = state
        .browser
        .filter
        .take()
        .map(|f| f.text)
        .unwrap_or_default();
    let selected = selected_path(state);
    state.browser.filter = Some(Filter {
        text,
        editing: true,
    });
    reselect(state, selected);
}

/// `enter` on the filter: the keys are the list's again; a path that
/// exists is jumped to, anything else is searched for below the folder
/// ([`crate::core::Action::JumpMissed`]).
pub(crate) fn accept_filter(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(filter) = &mut state.browser.filter else {
        return;
    };
    filter.editing = false;
    if !filter.text.trim().is_empty() {
        fx.push(Effect::Jump {
            text: filter.text.clone(),
            from: state.browser.dir.clone(),
        });
    }
    if filter.text.is_empty() {
        state.browser.filter = None;
    }
    fix_scroll(state, state.side);
}

/// `backspace` on the filter: its last character goes, and with none left
/// the filter itself.
pub(crate) fn erase_filter(state: &mut State) {
    let selected = selected_path(state);
    if let Some(filter) = &mut state.browser.filter {
        if filter.text.pop().is_none() {
            state.browser.filter = None;
        }
        reselect(state, selected);
    }
}

/// Puts pasted `text` (one line) into the filter being typed; else opens
/// the filter with it, on the main screen only.
pub(crate) fn paste(state: &mut State, text: &str) {
    if state.screen != Screen::Main {
        return;
    }
    // The filter is the browser's (the Transition tab has none to show).
    state.side = Side::Browser;
    let selected = selected_path(state);
    let filter = state.browser.filter.get_or_insert_with(Filter::default);
    if !filter.editing {
        filter.text.clear();
        filter.editing = true;
    }
    filter.text.push_str(text);
    reselect(state, selected);
}

/// `esc` on the main screen: the filter being typed goes, else the search,
/// else the filter; with nothing to clear, the browser goes up a folder.
pub(crate) fn back(state: &mut State, fx: &mut Vec<Effect>) {
    let selected = selected_path(state);
    let b = &mut state.browser;
    if b.filter.as_ref().is_some_and(|f| f.editing) {
        b.filter = None;
        reselect(state, selected);
    } else if b.search.take().is_some() {
        fx.push(Effect::CancelSearch);
    } else if b.filter.is_some() {
        b.filter = None;
        reselect(state, selected);
    } else if state.side == Side::Browser {
        // Nothing to clear: up one folder, with the folder we came
        // from selected (as `backspace`).
        go_parent(state, fx);
        return;
    }
    fix_scroll(state, state.side);
}

/// [`crate::core::Action::ClickRow`]: the row selected; a second click on it soon after
/// does what `enter` does on it.
pub(crate) fn click_row(state: &mut State, row: usize, fx: &mut Vec<Effect>) {
    state.screen = Screen::Main;
    let side = state.side;
    let double = state.last_click.press((side, row), state.now_ms);
    set_cursor(state, side, row);
    // A queue row held can be dragged to another place.
    state.queue.dragging = side == Side::Queue;
    if double {
        // As `enter`, which ends a preview too.
        preview::end(state, fx);
        queue::activate(state, Placing::Append, fx);
    }
}
