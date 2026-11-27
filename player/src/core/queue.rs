//! The play queue: what goes into it from the browser (a file, a folder
//! walked, a playlist read) and how; its rows removed and moved, here at
//! once and in the engine; the queues saved as playlists.

use std::path::PathBuf;

use jodel_playback::Command;

use crate::browser::{self, EntryKind};
use crate::core::browse::{self, browser_rows, row_entry, Row};
use crate::core::keymap::{self, Context, Op};
use crate::core::{Effect, ListKind, Placing, Screen, Side, State, WalkPurpose};

fn tracks(n: usize) -> &'static str {
    if n == 1 {
        "track"
    } else {
        "tracks"
    }
}

/// Plays the browser's file at row `row` with its whole folder (or all
/// search results) queued around it.
pub(crate) fn play_browser_from(state: &mut State, row: usize, fx: &mut Vec<Effect>) {
    let rows = browser_rows(state);
    let Some(first) = rows.get(row).and_then(|&r| row_entry(state, r)) else {
        return;
    };
    let first = first.path.clone();
    // The whole folder (or all search results), unfiltered, so a filter
    // only picks where to start; playing starts at the file, and previous
    // goes back to the songs before it.
    let pool: Vec<PathBuf> = if state.browser.search.is_some() {
        rows.iter()
            .filter_map(|&r| row_entry(state, r))
            .filter(|e| e.kind == EntryKind::File)
            .map(|e| e.path.clone())
            .collect()
    } else {
        state
            .browser
            .entries
            .iter()
            .filter(|e| e.kind == EntryKind::File)
            .map(|e| e.path.clone())
            .collect()
    };
    let (paths, at) = match pool.iter().position(|p| *p == first) {
        Some(at) => (pool, at),
        None => (vec![first], 0),
    };
    fx.push(Effect::Engine(Command::Load(paths)));
    fx.push(Effect::Engine(Command::Play(at)));
}

/// Play with nothing playing or paused: the queue from its current track,
/// else the browser's file selected with its folder.
pub(crate) fn play_stopped(state: &mut State, fx: &mut Vec<Effect>) {
    if !state.queue.paths.is_empty() {
        let index = state.queue.current.unwrap_or(0);
        fx.push(Effect::Engine(Command::Play(index)));
    } else if browse::selected_entry(state).is_some_and(|e| e.kind == EntryKind::File) {
        play_browser_from(state, state.browser.cursor, fx);
    } else {
        let enter = keymap::hint(Context::List, Op::Open);
        state.info(format!("Nothing to play: pick a file and press {enter}"));
    }
}

/// Puts `paths` where `how` says: after the queue; after it, playing the
/// first; or in its place, playing from `start`. `name` is what they are,
/// for the message.
fn place(
    state: &mut State,
    paths: Vec<PathBuf>,
    start: usize,
    how: Placing,
    name: &str,
    fx: &mut Vec<Effect>,
) {
    let n = paths.len();
    match how {
        Placing::Append => {
            fx.push(Effect::Engine(Command::Enqueue(paths)));
            state.info(format!("Queued {name} ({n} {})", tracks(n)));
        }
        Placing::AppendPlay => {
            let at = state.queue.paths.len();
            fx.push(Effect::Engine(Command::Enqueue(paths)));
            fx.push(Effect::Engine(Command::Play(at)));
            state.info(format!("Queued and playing {name}"));
        }
        Placing::Replace => {
            fx.push(Effect::Engine(Command::Load(paths)));
            fx.push(Effect::Engine(Command::Play(start)));
            state.queue.cursor = start;
            state.info(format!(
                "Playing {name} ({n} {}, in place of the queue)",
                tracks(n)
            ));
        }
    }
}

/// Takes the selected queue row out, here at once and in the engine.
pub(crate) fn remove(state: &mut State, fx: &mut Vec<Effect>) {
    let q = &mut state.queue;
    let index = q.cursor;
    if state.side != Side::Queue || index >= q.paths.len() {
        return;
    }
    q.paths.remove(index);
    let shift = |i: usize| (i > index).then(|| i - 1).or((i < index).then_some(i));
    q.current = q.current.and_then(shift);
    q.pending += 1;
    if let Some(track) = state.playing.track.as_mut() {
        if let Some(i) = shift(track.index) {
            track.index = i;
        }
    }
    fx.push(Effect::Engine(Command::Remove(index)));
    browse::set_cursor(state, Side::Queue, index);
}

/// Moves queue row `from` to `to`, here at once and in the engine; the
/// cursor goes with it.
fn move_row(state: &mut State, from: usize, to: usize, fx: &mut Vec<Effect>) {
    let q = &mut state.queue;
    let len = q.paths.len();
    if from >= len || to >= len || from == to {
        return;
    }
    let path = q.paths.remove(from);
    q.paths.insert(to, path);
    let moved = |i: usize| {
        if i == from {
            to
        } else if from < to && i > from && i <= to {
            i - 1
        } else if to < from && i >= to && i < from {
            i + 1
        } else {
            i
        }
    };
    q.current = q.current.map(moved);
    q.pending += 1;
    if let Some(track) = state.playing.track.as_mut() {
        track.index = moved(track.index);
    }
    fx.push(Effect::Engine(Command::Move { from, to }));
    browse::set_cursor(state, Side::Queue, to);
}

/// The selected queue row `by` rows up (`-1`) or down (`1`), while the
/// queue shows; a row at the top or the bottom stays.
pub(crate) fn move_selected(state: &mut State, by: isize, fx: &mut Vec<Effect>) {
    if state.side != Side::Queue {
        return;
    }
    let from = state.queue.cursor;
    if let Some(to) = from.checked_add_signed(by) {
        move_row(state, from, to, fx);
    }
}

/// A queue row dragged with the mouse onto row `to`.
pub(crate) fn drag(state: &mut State, to: usize, fx: &mut Vec<Effect>) {
    let from = state.queue.cursor;
    if state.side == Side::Queue && to != from {
        move_row(state, from, to, fx);
    }
}

/// [`jodel_playback::Event::QueueChanged`]: the engine's queue, unless one
/// from before changes made here; the cursor stays where it is in it.
pub(crate) fn engine_changed(state: &mut State, queue: Vec<PathBuf>, current: Option<usize>) {
    // A queue from before changes this side made is not taken.
    if state.queue.pending > 0 {
        state.queue.pending -= 1;
        if state.queue.pending > 0 {
            return;
        }
    }
    state.queue.paths = queue;
    state.queue.current = current;
    let cursor = state.queue.cursor;
    browse::set_cursor(state, Side::Queue, cursor);
}

/// `enter` (`how`: [`Placing::Append`]), `shift+enter`
/// ([`Placing::AppendPlay`]) and `alt+enter` ([`Placing::Replace`]) on the
/// list, and a double click on a row: in the queue, the track plays; in
/// the browser, a folder opens with `enter` (and goes into the queue, all
/// of it, with the others), and a file or playlist goes into the queue as
/// `how` says: after it, after it and playing, or in its place with its
/// whole folder (playing from the file).
pub(crate) fn activate(state: &mut State, how: Placing, fx: &mut Vec<Effect>) {
    match state.side.list() {
        None => {}
        Some(ListKind::Queue) => {
            if state.queue.cursor < state.queue.paths.len() {
                fx.push(Effect::Engine(Command::Play(state.queue.cursor)));
            }
        }
        Some(ListKind::Browser) => {
            let rows = browser_rows(state);
            let Some(&row) = rows.get(state.browser.cursor) else {
                return;
            };
            if row == Row::SearchBelow {
                browse::search_below(state, fx);
                return;
            }
            let Some(entry) = row_entry(state, row).cloned() else {
                return;
            };
            match (entry.kind, how) {
                // Up one folder, with the folder we came from selected.
                (EntryKind::Parent, _) => browse::go_parent(state, fx),
                (EntryKind::Dir, Placing::Append) => browse::enter_dir(state, entry.path, fx),
                (EntryKind::File, Placing::Replace) => {
                    play_browser_from(state, state.browser.cursor, fx);
                    state.info(format!(
                        "Playing {} with its folder, in place of the queue",
                        entry.name
                    ));
                }
                _ => queue_entry(state, entry, how, fx),
            }
        }
    }
}

/// A right click on a browser row ([`crate::core::Action::EnqueueRow`]):
/// the file, or the folder with everything below it, or the playlist's
/// tracks, after the queue.
pub(crate) fn enqueue(state: &mut State, fx: &mut Vec<Effect>) {
    if state.side != Side::Browser {
        return;
    }
    if let Some(entry) = browse::selected_entry(state).cloned() {
        queue_entry(state, entry, Placing::Append, fx);
    }
}

/// Puts `entry` into the queue as `how` says: a file at once, a folder's
/// files once it is walked, a playlist's tracks once it is read.
fn queue_entry(state: &mut State, entry: browser::Entry, how: Placing, fx: &mut Vec<Effect>) {
    match entry.kind {
        EntryKind::File => place(state, vec![entry.path], 0, how, &entry.name, fx),
        EntryKind::Dir => {
            state.info(format!("Collecting {}…", entry.name));
            fx.push(Effect::Walk {
                root: entry.path,
                purpose: WalkPurpose {
                    name: entry.name,
                    how,
                },
            });
        }
        EntryKind::Playlist => fx.push(Effect::ReadPlaylist {
            path: entry.path,
            how,
        }),
        EntryKind::Parent => {}
    }
}

/// [`crate::core::Action::Walked`]: the files a folder walk found, into the
/// queue as it was asked.
pub(crate) fn walked(
    state: &mut State,
    purpose: WalkPurpose,
    paths: Vec<PathBuf>,
    fx: &mut Vec<Effect>,
) {
    let WalkPurpose { name, how } = purpose;
    if paths.is_empty() {
        state.info(format!("Nothing to play in {name}"));
    } else {
        place(state, paths, 0, how, &name, fx);
    }
}

/// [`crate::core::Action::PlaylistRead`]: a playlist's tracks, into the
/// queue as `how` says.
pub(crate) fn playlist_read(
    state: &mut State,
    path: &std::path::Path,
    tracks: Result<Vec<PathBuf>, String>,
    how: Placing,
    fx: &mut Vec<Effect>,
) {
    let name = file_stem(path);
    match tracks {
        Err(why) => state.error(format!("{name}: {why}")),
        Ok(tracks) if tracks.is_empty() => {
            state.info(format!("Nothing to play in {name}"));
        }
        Ok(tracks) => place(state, tracks, 0, how, &name, fx),
    }
}

fn file_stem(path: &std::path::Path) -> String {
    path.file_stem()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `ctrl+s`: the queue saved as a playlist among the saved queues.
pub(crate) fn save(state: &mut State, fx: &mut Vec<Effect>) {
    if state.queue.paths.is_empty() {
        return state.info("The queue is empty: nothing to save");
    }
    if state.queues_dir.is_none() {
        return state.info("No folder to save queues in");
    }
    fx.push(Effect::SaveQueue(state.queue.paths.clone()));
}

/// [`crate::core::Action::QueueSaved`]: where the queue was saved, or why
/// it was not.
pub(crate) fn saved(state: &mut State, result: Result<PathBuf, String>) {
    match result {
        Ok(path) => {
            let name = file_stem(&path);
            let open = keymap::hint(Context::Global, Op::QueueOpenSaved);
            state.info(format!(
                "Queue saved as {name} ({open} shows the saved queues)"
            ));
        }
        Err(why) => state.error(format!("The queue was not saved: {why}")),
    }
}

/// `ctrl+l`: the saved queues, in the browser.
pub(crate) fn open_saved(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(dir) = state.queues_dir.clone() else {
        return state.info("No folder of saved queues");
    };
    state.screen = Screen::Main;
    state.side = Side::Browser;
    browse::enter_dir(state, dir, fx);
}
