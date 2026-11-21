//! `update(state, action) -> effects`: the UI's logic, without IO.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_playback::{Command, Event, Headroom, PlayState, StemMix};

use crate::browser::{self, EntryKind};
use crate::core::karaoke;
use crate::core::keymap::{self, Op};
use crate::core::layout;
use crate::core::mix;
use crate::core::preview;
use crate::core::text;
use crate::core::transition::Preset;
use crate::core::{
    Action, Click, DeviceChoice, DeviceVolume, Effect, Facts, Filter, FolderPick, FolderRow, Gpu,
    Heard, Job, Level, Message, NowTrack, Palette, Placing, Rhythm, RhythmRecipe, Screen, Search,
    SettingsRow, Side, State, VolumeChange, WalkPurpose, DOUBLE_CLICK_MS, MESSAGE_MS, RHYTHM_AHEAD,
    SEEK_STEP, SETTINGS_DELAY_MS, TYPE_AHEAD_MS,
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

/// How many rows the shown side has.
pub fn list_len(state: &State) -> usize {
    match state.side {
        Side::Browser => browser_rows(state).len(),
        Side::Queue => state.queue.paths.len(),
        Side::Mix | Side::Karaoke => 0,
    }
}

/// The selected row and the first row shown, of the shown side.
pub fn list_cursor(state: &State) -> (usize, usize) {
    match state.side {
        Side::Browser => (state.browser.cursor, state.browser.scroll),
        Side::Queue => (state.queue.cursor, state.queue.scroll),
        Side::Mix | Side::Karaoke => (0, 0),
    }
}

/// The path of the browser row under the cursor, if it is an entry.
fn selected_path(state: &State) -> Option<PathBuf> {
    let rows = browser_rows(state);
    rows.get(state.browser.cursor)
        .and_then(|&row| row_entry(state, row))
        .map(|e| e.path.clone())
}

/// After the browser's rows changed (a filter typed or cleared): the cursor
/// stays on `path` where it is still shown, else goes to the first row.
fn reselect(state: &mut State, path: Option<PathBuf>) {
    let rows = browser_rows(state);
    let row = path
        .and_then(|path| {
            rows.iter()
                .position(|&row| row_entry(state, row).is_some_and(|e| e.path == path))
        })
        .unwrap_or(0);
    let side = state.side;
    state.side = Side::Browser;
    set_cursor(state, row);
    state.side = side;
}

fn set_cursor(state: &mut State, cursor: usize) {
    let len = list_len(state);
    let cursor = cursor.min(len.saturating_sub(1));
    match state.side {
        Side::Browser => state.browser.cursor = cursor,
        Side::Queue => state.queue.cursor = cursor,
        Side::Mix | Side::Karaoke => {}
    }
    fix_scroll(state);
}

/// How many list rows the main screen shows.
pub fn list_height(state: &State) -> usize {
    layout::main_for(state).map_or(0, |m| usize::from(m.rows.height))
}

fn fix_scroll(state: &mut State) {
    let height = list_height(state).max(1);
    let len = list_len(state);
    let (cursor, scroll) = list_cursor(state);
    let mut scroll = scroll.min(len.saturating_sub(height));
    if cursor < scroll {
        scroll = cursor;
    } else if cursor >= scroll + height {
        scroll = cursor + 1 - height;
    }
    match state.side {
        Side::Browser => state.browser.scroll = scroll,
        Side::Queue => state.queue.scroll = scroll,
        Side::Mix | Side::Karaoke => {}
    }
}

/// The files on screen whose details are not known yet.
pub fn visible_missing(state: &State) -> Vec<PathBuf> {
    // Karaoke (the tab or the mode): the song it shows and the next two,
    // their titles and artists shown before they play.
    if state.screen == Screen::Karaoke
        || (state.screen == Screen::Main && state.side == Side::Karaoke)
    {
        let Some(index) = karaoke::song_index(state) else {
            return Vec::new();
        };
        return state
            .queue
            .paths
            .iter()
            .skip(index)
            .take(3)
            .filter(|p| !state.details.contains_key(*p))
            .cloned()
            .collect();
    }
    if state.screen != Screen::Main {
        return Vec::new();
    }
    let height = list_height(state);
    let (_, scroll) = list_cursor(state);
    let paths: Vec<PathBuf> = match state.side {
        Side::Browser => browser_rows(state)
            .into_iter()
            .skip(scroll)
            .take(height)
            .filter_map(|row| row_entry(state, row))
            .filter(|e| e.kind == EntryKind::File)
            .map(|e| e.path.clone())
            .collect(),
        Side::Queue => state
            .queue
            .paths
            .iter()
            .skip(scroll)
            .take(height)
            .cloned()
            .collect(),
        Side::Mix | Side::Karaoke => Vec::new(),
    };
    paths
        .into_iter()
        .filter(|p| !state.details.contains_key(p))
        .collect()
}

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

/// The first device shown in a device list of `height` rows.
pub fn device_window(state: &State, height: usize) -> usize {
    state
        .settings
        .device_cursor
        .saturating_sub(height.max(1) - 1)
}

/// The first row of the data folder's list shown in `height` rows (the
/// selected one in view).
pub fn folder_window(state: &State, height: usize) -> usize {
    state
        .settings
        .folders
        .as_ref()
        .map_or(0, |p| p.cursor.saturating_sub(height.max(1) - 1))
}

/// The data folder's list opens on the folder the data folder is in. While
/// a move is under way it does not: one that stopped goes on instead.
fn open_folders(state: &mut State, fx: &mut Vec<Effect>) {
    if let Some(m) = state.data_move.as_ref().filter(|m| !m.finished) {
        if m.failure.is_some() {
            fx.push(Effect::RetryMove);
            say(state, Level::Info, "Moving the data folder again");
        } else {
            say(
                state,
                Level::Info,
                "The data folder is moving: choose another once it is there",
            );
        }
        return;
    }
    let dir = state
        .data_dir
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    folder_go(state, dir, fx);
}

/// The data folder's list shows `dir` (`None`: the drives).
fn folder_go(state: &mut State, dir: Option<PathBuf>, fx: &mut Vec<Effect>) {
    state.settings.folders = Some(FolderPick {
        dir: dir.clone(),
        folders: None,
        cursor: 0,
    });
    fx.push(Effect::ListFolders(dir));
}

/// Up a folder in the data folder's list (from a drive's top, to the drives).
fn folder_parent(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(pick) = &state.settings.folders else {
        return;
    };
    let Some(dir) = &pick.dir else {
        return;
    };
    let up = dir.parent().map(Path::to_path_buf);
    folder_go(state, up, fx);
}

/// `enter` on the data folder's list: the row selected.
fn folder_open(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(pick) = &state.settings.folders else {
        return;
    };
    match pick.rows().get(pick.cursor).cloned() {
        Some(FolderRow::Choose(dir)) => {
            state.settings.folders = None;
            choose_data_folder(state, dir, fx);
        }
        Some(FolderRow::Up) => folder_parent(state, fx),
        Some(FolderRow::Folder(dir)) => folder_go(state, Some(dir), fx),
        None => {}
    }
}

/// The data folder is to be made in `dir` (or is `dir`, when that is a data
/// folder already): the app moves it there.
fn choose_data_folder(state: &mut State, dir: PathBuf, fx: &mut Vec<Effect>) {
    let parent = if dir.file_name().is_some_and(|n| n == crate::dirs::DATA) {
        dir.parent().map(Path::to_path_buf).unwrap_or(dir)
    } else {
        dir
    };
    let to = parent.join(crate::dirs::DATA);
    if let Some(from) = &state.data_dir {
        if &to == from {
            return say(state, Level::Info, "The data folder is there already");
        }
        if to.starts_with(from) {
            return say(
                state,
                Level::Info,
                "Not inside the data folder itself: choose a folder outside it",
            );
        }
    }
    fx.push(Effect::MoveData(parent));
}

/// The help screen's sections, as its search leaves them.
pub fn help_shown(state: &State) -> Vec<keymap::HelpSection> {
    keymap::help_matching(state.help_search.as_ref().map_or("", |s| s.text.as_str()))
}

/// The number of lines the help screen has, as its search leaves them.
pub fn help_line_count(state: &State) -> usize {
    help_shown(state).iter().map(|s| s.lines.len() + 2).sum()
}

/// Whether the search of the help is being typed.
fn help_typing(state: &State) -> bool {
    state.screen == Screen::Help && state.help_search.as_ref().is_some_and(|s| s.editing)
}

/// A message for the status line, as information.
pub(crate) fn say_info(state: &mut State, text: impl Into<String>) {
    say(state, Level::Info, text);
}

fn say(state: &mut State, level: Level, text: impl Into<String>) {
    state.message = Some(Message {
        text: text.into(),
        level,
        until_ms: state.now_ms + MESSAGE_MS,
    });
}

/// Applies `action` to `state` and returns the work to carry out.
pub fn update(state: &mut State, action: Action) -> Vec<Effect> {
    let mut fx = Vec::new();
    let mut check_details = true;
    // A jump to a name is typed in one go: any other key or click (Enter,
    // a folder opened or left, a move) starts the next one afresh.
    if is_input(&action) && !matches!(action, Action::Type(_) | Action::Paste(_)) {
        state.browser.typed.clear();
    }
    // A key or click other than a move ends a preview first (and Ctrl
    // pressed with a key starts none); a move is followed below.
    let input = is_input(&action);
    if input {
        preview::before_input(state, &action, &mut fx);
    }
    let selected = preview::watched(state);
    match action {
        Action::Ctrl(how) => preview::ctrl(state, how, &mut fx),
        Action::Op(op) => apply_op(state, op, &mut fx),
        Action::Type(c) => {
            if let Some(palette) = &mut state.palette {
                palette.text.push(c);
                palette.cursor = 0;
            } else if help_typing(state) {
                if let Some(search) = state.help_search.as_mut() {
                    search.text.push(c);
                }
                state.help_scroll = 0;
            } else if state.browser.filter.as_ref().is_some_and(|f| f.editing) {
                let selected = selected_path(state);
                if let Some(filter) = state.browser.filter.as_mut() {
                    filter.text.push(c);
                }
                reselect(state, selected);
            } else {
                type_ahead(state, c);
            }
        }
        Action::Paste(text) => paste(state, &text),
        Action::MixPoint {
            block,
            time,
            extend,
        } => {
            if mix::point(state, block, time, extend) {
                apply_op(state, Op::Accept, &mut fx);
            }
        }
        Action::MixJump { block, time } => match block {
            mix::Block::Now => mix::seek(state, &mut fx, time),
            _ => mix::jump(state, block, time),
        },
        Action::MixWheel {
            block,
            steps,
            scroll,
            at,
        } => {
            if scroll {
                mix::scroll(state, block, f64::from(steps) / 10.0);
            } else {
                mix::wheel_zoom(state, block, steps.signum() as isize, at);
            }
        }
        Action::MixPanStart {
            block,
            column,
            width,
            overview,
        } => mix::pan_start(state, block, column, width, overview),
        Action::MixPan(column) => mix::pan_to(state, column),
        Action::MixPanEnd => mix::pan_end(state),
        Action::KaraokeSlide { slider, share } => {
            state.karaoke.dragging = Some(slider);
            karaoke::slide_to(state, slider, share, &mut fx);
        }
        Action::KaraokeSlideEnd => state.karaoke.dragging = None,
        Action::KaraokeLevel(level) => karaoke::set_level(state, level, &mut fx),
        Action::KaraokeStep { slider, by } => karaoke::step(state, slider, by, &mut fx),
        Action::PointerMoved { row } => karaoke::pointer_moved(state, row),
        Action::SelectRow(row) => {
            state.screen = Screen::Main;
            set_cursor(state, row);
        }
        Action::OpenRow(row) => {
            set_cursor(state, row);
            open(state, &mut fx);
        }
        Action::ClickRow(row) => {
            state.screen = Screen::Main;
            let now = state.now_ms;
            let double = state.last_click.is_some_and(|c| {
                c.side == state.side
                    && c.row == row
                    && now.saturating_sub(c.at_ms) <= DOUBLE_CLICK_MS
            });
            set_cursor(state, row);
            // A queue row held can be dragged to another place.
            state.queue.dragging = state.side == Side::Queue;
            if double {
                // As `enter`, which ends a preview too.
                state.last_click = None;
                preview::end(state, &mut fx);
                open(state, &mut fx);
            } else {
                state.last_click = Some(Click {
                    side: state.side,
                    row,
                    at_ms: now,
                });
            }
        }
        Action::EnqueueRow(row) => {
            set_cursor(state, row);
            enqueue(state, &mut fx);
        }
        Action::QueueDrag(row) => {
            let from = state.queue.cursor;
            if state.side == Side::Queue && row != from {
                move_in_queue(state, from, row, &mut fx);
            }
        }
        Action::QueueDragEnd => state.queue.dragging = false,
        Action::Scroll(delta) => match state.screen {
            Screen::Help => scroll_help(state, delta),
            _ => {
                let (cursor, _) = list_cursor(state);
                set_cursor(state, offset(cursor, delta));
            }
        },
        Action::ShowSide(side) => {
            state.side = side;
            state.screen = Screen::Main;
            fix_scroll(state);
        }
        Action::SeekTo(fraction) => {
            state.seek_drag = None;
            if karaoke::seek_refused(state) {
                // Held for START or counting in: nothing to seek yet.
            } else if let Some(duration) = state.playing.duration() {
                let seconds = fraction.clamp(0.0, 1.0) * duration;
                state.playing.position = seconds;
                fx.push(Effect::Engine(Command::Seek(seconds)));
            }
        }
        Action::SeekDrag(fraction) => {
            if state.playing.duration().is_some() && !karaoke::seek_refused(state) {
                state.seek_drag = Some(fraction.clamp(0.0, 1.0));
            }
        }
        Action::OpenSettings(row) => {
            open_settings(state, &mut fx);
            state.settings.row = SettingsRow::ALL.iter().position(|r| *r == row).unwrap_or(0);
            if row == SettingsRow::Device {
                state.settings.picking = true;
            }
        }
        Action::SettingsClick(row, step) => {
            state.settings.row = row.min(SettingsRow::ALL.len() - 1);
            state.settings.picking = false;
            state.settings.folders = None;
            if step == 0 {
                settings_open(state, &mut fx);
            } else {
                settings_step(state, step, &mut fx);
            }
        }
        Action::ChooseDevice(index) => {
            state.settings.device_cursor = index;
            choose_device(state, &mut fx);
        }
        Action::FolderRow(index) => {
            if let Some(pick) = &mut state.settings.folders {
                pick.cursor = index;
            }
            folder_open(state, &mut fx);
        }
        Action::FoldersListed(dir, folders) => {
            if let Some(pick) = &mut state.settings.folders {
                if pick.dir == dir {
                    pick.folders = Some(folders);
                    pick.cursor = pick.cursor.min(pick.rows().len().saturating_sub(1));
                }
            }
        }
        Action::PaletteRun(index) => {
            if let Some(palette) = &mut state.palette {
                palette.cursor = index;
            }
            run_palette(state, &mut fx);
        }
        Action::HelpScroll(delta) => scroll_help(state, delta),
        Action::Resize(cols, rows) => {
            if state.size != (cols, rows) {
                state.info_width_held = 0;
            }
            state.size = (cols, rows);
            fix_scroll(state);
        }
        Action::Tick(now) => {
            state.now_ms = now;
            mix::tick(state);
            karaoke::tick(state, &mut fx);
            preview::tick(state, &mut fx);
            if state.message.as_ref().is_some_and(|m| m.until_ms <= now) {
                state.message = None;
            }
            if state.settings_due.is_some_and(|due| due <= now) {
                send_settings(state, &mut fx);
            }
            check_details = false;
        }
        Action::Listed { dir, result } => {
            if dir == state.browser.dir {
                listed(state, result, &mut fx);
            }
        }
        Action::Details(path, details) => {
            state.pending.remove(&path);
            state.details.insert(path, details);
            check_details = false;
        }
        Action::DetailsDropped(paths) => {
            for path in paths {
                state.pending.remove(&path);
            }
            check_details = false;
        }
        Action::SearchHits { id, hits } => {
            if let Some(search) = state.browser.search.as_mut().filter(|s| s.id == id) {
                search.hits.extend(hits);
            }
        }
        Action::SearchDone { id } => {
            if let Some(search) = state.browser.search.as_mut().filter(|s| s.id == id) {
                search.done = true;
            }
        }
        Action::Walked { purpose, paths } => match purpose {
            WalkPurpose::Enqueue { name, how } => {
                if paths.is_empty() {
                    say(state, Level::Info, format!("Nothing to play in {name}"));
                } else {
                    place(state, paths, 0, how, &name, &mut fx);
                }
            }
        },
        Action::JumpMissed { text } => {
            if state
                .browser
                .filter
                .as_ref()
                .is_some_and(|f| f.text == text)
            {
                search_below(state, &mut fx);
            }
        }
        Action::JumpTo { dir, select } => jump_to(state, dir, select, &mut fx),
        Action::Devices(result) => match result {
            Ok(list) => {
                let mut choices = vec![DeviceChoice {
                    id: None,
                    name: "System default".into(),
                    summary: "whatever the system's default output is".into(),
                }];
                choices.extend(list);
                state.settings.device_cursor = choices
                    .iter()
                    .position(|c| c.id == state.config.device)
                    .unwrap_or(0);
                state.settings.devices = Some(choices);
            }
            Err(error) => {
                state.settings.devices = Some(Vec::new());
                say(
                    state,
                    Level::Error,
                    format!("Listing devices failed: {error}"),
                );
            }
        },
        Action::DeviceRates(rates) => device_rates(state, rates),
        Action::Engine(event) => {
            // A track starting (the device opened, or kept) shows the
            // device's volume as it is.
            if matches!(event, Event::TrackStarted(_)) && !state.null_output {
                fx.push(Effect::DeviceVolume(VolumeChange::Read));
            }
            engine_event(state, event)
        }
        Action::DeviceVolume { result, changed } => device_volume(state, result, changed),
        Action::Rhythm(path, recipe, result) => {
            // Only for a file still wanted, by the recipe in force.
            if state.rhythm_wanted.contains(&path) && recipe == state.rhythm_recipe {
                let rhythm = match result {
                    Ok(analysis) => Rhythm::Ready(analysis),
                    Err(_) => Rhythm::Failed,
                };
                state.rhythm.insert(path, rhythm);
            }
        }
        Action::Gpu(name) => {
            state.gpu = match name {
                Some(name) => Gpu::Present(name),
                None => Gpu::Absent,
            };
        }
        Action::Lyrics(path, answer) => karaoke::lyrics_arrived(state, path, answer),
        Action::Working { path, job, on } => {
            let jobs = state.working.entry(path.clone()).or_default();
            jobs.retain(|j| *j != job);
            if on {
                jobs.push(job);
            }
            if jobs.is_empty() {
                state.working.remove(&path);
            }
        }
        Action::Heard { path, part } => {
            let facts = state.facts.entry(path).or_default();
            match part {
                Heard::Key(found) => facts.key = Some(found),
                Heard::Stems => facts.stems = Some(true),
            }
        }
        Action::Ready { path, job, facts } => {
            match facts {
                Some(Facts::Rhythm(f)) => state.facts.entry(path.clone()).or_default().analysed(&f),
                Some(Facts::Lyrics(f)) => {
                    state.facts.entry(path.clone()).or_default().lyrics = Some(f)
                }
                None => {}
            }
            let jobs = state.ready.entry(path).or_default();
            if !jobs.contains(&job) {
                jobs.push(job);
            }
        }
        Action::Failed(text) => say(state, Level::Error, text),
        Action::Info(text) => say(state, Level::Info, text),
        Action::QueueSaved(result) => match result {
            Ok(path) => {
                let name = path
                    .file_stem()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                say(
                    state,
                    Level::Info,
                    format!("Queue saved as {name} (ctrl+l shows the saved queues)"),
                );
            }
            Err(why) => say(
                state,
                Level::Error,
                format!("The queue was not saved: {why}"),
            ),
        },
        Action::PlaylistRead { path, tracks, how } => {
            let name = path
                .file_stem()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            match tracks {
                Err(why) => say(state, Level::Error, format!("{name}: {why}")),
                Ok(tracks) if tracks.is_empty() => {
                    say(state, Level::Info, format!("Nothing to play in {name}"));
                }
                Ok(tracks) => place(state, tracks, 0, how, &name, &mut fx),
            }
        }
        Action::EngineGone(_) => {
            // The new engine plays the file as it is: if the stems were
            // what failed, they do not fail again at once.
            state.stems = StemMix::ALL;
            say(
                state,
                Level::Error,
                "The playback engine failed and was started again",
            );
        }
    }
    // While Ctrl is held, the preview follows the selection as it is
    // moved (not as a folder's listing lands).
    if input && state.preview.ctrl {
        preview::moved(state, selected, &mut fx);
    }
    want_rhythm(state, &mut fx);
    karaoke::want_lyrics(state, &mut fx);
    karaoke::follow_stems(state, &mut fx);
    prepare(state, &mut fx);
    state.info_width_held = state.info_width_held.max(text::info_width(state));
    if check_details {
        // Everything on screen still unknown, including what is already
        // asked for: the worker replaces its queue with this.
        let wanted = visible_missing(state);
        if wanted.iter().any(|p| !state.pending.contains(p)) {
            state.pending.extend(wanted.iter().cloned());
            fx.push(Effect::WantDetails(wanted));
        }
    }
    fx
}

fn tracks(n: usize) -> &'static str {
    if n == 1 {
        "track"
    } else {
        "tracks"
    }
}

fn offset(value: usize, delta: i32) -> usize {
    if delta < 0 {
        value.saturating_sub(delta.unsigned_abs() as usize)
    } else {
        value.saturating_add(delta as usize)
    }
}

fn scroll_help(state: &mut State, delta: i32) {
    let visible = match layout::compute(state) {
        layout::Layout::Help(h) => usize::from(h.text.height),
        _ => 1,
    };
    let max = help_line_count(state).saturating_sub(visible);
    state.help_scroll = offset(state.help_scroll, delta).min(max);
}

/// Just the `..` row of `dir` (none at the very top).
fn parent_row(dir: &std::path::Path) -> Vec<browser::Entry> {
    browser::parent_of(dir)
        .map(browser::Entry::parent)
        .into_iter()
        .collect()
}

fn listed(state: &mut State, result: Result<Vec<browser::Entry>, String>, fx: &mut Vec<Effect>) {
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
    let row = select
        .and_then(|path| {
            browser_rows(state)
                .iter()
                .position(|&row| row_entry(state, row).is_some_and(|e| e.path == path))
        })
        .unwrap_or(0);
    let was = state.side;
    state.side = Side::Browser;
    set_cursor(state, row);
    state.side = was;
    if let Some(path) = state.autoplay.take() {
        let rows = browser_rows(state);
        if let Some(row) = rows
            .iter()
            .position(|&r| row_entry(state, r).is_some_and(|e| e.path == path))
        {
            play_browser_from(state, &rows, row, fx);
        } else {
            say(
                state,
                Level::Error,
                format!("Cannot play {}", path.display()),
            );
        }
    }
}

/// Shows `dir` in the browser on the main screen, with `select` selected
/// once it is listed.
fn jump_to(state: &mut State, dir: PathBuf, select: Option<PathBuf>, fx: &mut Vec<Effect>) {
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
pub fn playing_jump(state: &State, op: Op) -> Option<(PathBuf, Option<PathBuf>)> {
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

fn enter_dir(state: &mut State, dir: PathBuf, fx: &mut Vec<Effect>) {
    // Gone somewhere else: no falling back from there.
    state.browser.fallback = None;
    let rows = browser_rows(state);
    if let Some(entry) = rows
        .get(state.browser.cursor)
        .and_then(|&r| row_entry(state, r))
    {
        let selected = entry.path.clone();
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

/// Whether `action` is the user's own key, click or scroll (not the
/// clock or a background result).
fn is_input(action: &Action) -> bool {
    matches!(
        action,
        Action::Op(_)
            | Action::Type(_)
            | Action::Paste(_)
            | Action::SelectRow(_)
            | Action::OpenRow(_)
            | Action::ClickRow(_)
            | Action::EnqueueRow(_)
            | Action::QueueDrag(_)
            | Action::QueueDragEnd
            | Action::Scroll(_)
            | Action::ShowSide(_)
            | Action::MixPoint { .. }
            | Action::MixJump { .. }
            | Action::MixWheel { .. }
            | Action::MixPanStart { .. }
            | Action::MixPan(_)
            | Action::MixPanEnd
            | Action::KaraokeSlide { .. }
            | Action::KaraokeSlideEnd
            | Action::KaraokeLevel(_)
            | Action::KaraokeStep { .. }
            | Action::SeekTo(_)
            | Action::SeekDrag(_)
            | Action::OpenSettings(_)
            | Action::SettingsClick(..)
            | Action::ChooseDevice(_)
            | Action::FolderRow(_)
            | Action::PaletteRun(_)
            | Action::HelpScroll(_)
    )
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
    let names: Vec<Option<String>> = match state.side {
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
        Side::Mix | Side::Karaoke => Vec::new(),
    };
    if names.is_empty() {
        return;
    }
    let cursor = match state.side {
        Side::Browser => state.browser.cursor,
        Side::Queue | Side::Mix | Side::Karaoke => state.queue.cursor,
    }
    .min(names.len() - 1);
    // Cycling starts after the current row; a longer prefix may stay on it.
    let start = if cycling { cursor + 1 } else { cursor };
    let hit = (0..names.len())
        .map(|i| (start + i) % names.len())
        .find(|&i| names[i].as_deref().is_some_and(|n| n.starts_with(&prefix)));
    if let Some(row) = hit {
        set_cursor(state, row);
    }
}

/// The name a browser row shows: a search hit's relative path, a folder's
/// name, a file's title tag once read, else its file name.
pub fn shown_name(state: &State, entry: &browser::Entry) -> String {
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
pub fn shown_title(state: &State, path: &std::path::Path) -> String {
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

fn go_parent(state: &mut State, fx: &mut Vec<Effect>) {
    if state.browser.search.is_some() {
        state.browser.search = None;
        fx.push(Effect::CancelSearch);
        set_cursor(state, 0);
        return;
    }
    if let Some(parent) = browser::parent_of(&state.browser.dir) {
        enter_dir(state, parent, fx);
    }
}

/// Plays the browser's file at `rows[row]` with its whole folder (or all
/// search results) queued around it.
fn play_browser_from(state: &mut State, rows: &[Row], row: usize, fx: &mut Vec<Effect>) {
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
            say(
                state,
                Level::Info,
                format!("Queued {name} ({n} {})", tracks(n)),
            );
        }
        Placing::AppendPlay => {
            let at = state.queue.paths.len();
            fx.push(Effect::Engine(Command::Enqueue(paths)));
            fx.push(Effect::Engine(Command::Play(at)));
            say(state, Level::Info, format!("Queued and playing {name}"));
        }
        Placing::Replace => {
            fx.push(Effect::Engine(Command::Load(paths)));
            fx.push(Effect::Engine(Command::Play(start)));
            state.queue.cursor = start;
            say(
                state,
                Level::Info,
                format!("Playing {name} ({n} {}, in place of the queue)", tracks(n)),
            );
        }
    }
}
/// Takes the selected queue row out, here at once and in the engine.
fn remove_from_queue(state: &mut State, fx: &mut Vec<Effect>) {
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
    set_cursor(state, index);
    queue_changed(state, fx);
}

/// Moves queue row `from` to `to`, here at once and in the engine; the
/// cursor goes with it.
fn move_in_queue(state: &mut State, from: usize, to: usize, fx: &mut Vec<Effect>) {
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
    set_cursor(state, to);
    queue_changed(state, fx);
}

/// The queue changed here: it is saved with the session, soon.
fn queue_changed(state: &mut State, fx: &mut Vec<Effect>) {
    let _ = (state, fx);
}

/// `enter`, and a double click on a row: a folder opens, a file or
/// playlist goes into the queue, a queue row plays.
fn open(state: &mut State, fx: &mut Vec<Effect>) {
    activate(state, Placing::Append, fx);
}

/// `enter` (`how`: [`Placing::Append`]), `shift+enter`
/// ([`Placing::AppendPlay`]) and `alt+enter` ([`Placing::Replace`]) on the
/// list: in the queue, the track plays; in the browser, a folder opens
/// with `enter` (and goes into the queue, all of it, with the others), and
/// a file or playlist goes into the queue as `how` says: after it, after
/// it and playing, or in its place with its whole folder (playing from
/// the file).
fn activate(state: &mut State, how: Placing, fx: &mut Vec<Effect>) {
    match state.side {
        Side::Mix | Side::Karaoke => {}
        Side::Queue => {
            if state.queue.cursor < state.queue.paths.len() {
                fx.push(Effect::Engine(Command::Play(state.queue.cursor)));
            }
        }
        Side::Browser => {
            let rows = browser_rows(state);
            let Some(&row) = rows.get(state.browser.cursor) else {
                return;
            };
            if row == Row::SearchBelow {
                search_below(state, fx);
                return;
            }
            let Some(entry) = row_entry(state, row).cloned() else {
                return;
            };
            match (entry.kind, how) {
                // Up one folder, with the folder we came from selected.
                (EntryKind::Parent, _) => go_parent(state, fx),
                (EntryKind::Dir, Placing::Append) => enter_dir(state, entry.path, fx),
                (EntryKind::Dir, how) => {
                    say(state, Level::Info, format!("Collecting {}…", entry.name));
                    fx.push(Effect::Walk {
                        root: entry.path.clone(),
                        purpose: WalkPurpose::Enqueue {
                            name: entry.name,
                            how,
                        },
                    });
                }
                (EntryKind::File, Placing::Replace) => {
                    play_browser_from(state, &rows, state.browser.cursor, fx);
                    say(
                        state,
                        Level::Info,
                        format!(
                            "Playing {} with its folder, in place of the queue",
                            entry.name
                        ),
                    );
                }
                (EntryKind::File, how) => place(state, vec![entry.path], 0, how, &entry.name, fx),
                (EntryKind::Playlist, how) => fx.push(Effect::ReadPlaylist {
                    path: entry.path,
                    how,
                }),
            }
        }
    }
}

/// Right click (and the palette): the file, or the folder with everything
/// below it, or the playlist's tracks, after the queue.
fn enqueue(state: &mut State, fx: &mut Vec<Effect>) {
    if state.side != Side::Browser {
        return;
    }
    let rows = browser_rows(state);
    let Some(entry) = rows
        .get(state.browser.cursor)
        .and_then(|&r| row_entry(state, r))
        .cloned()
    else {
        return;
    };
    match entry.kind {
        EntryKind::File => place(state, vec![entry.path], 0, Placing::Append, &entry.name, fx),
        EntryKind::Dir => {
            say(state, Level::Info, format!("Collecting {}…", entry.name));
            fx.push(Effect::Walk {
                root: entry.path.clone(),
                purpose: WalkPurpose::Enqueue {
                    name: entry.name,
                    how: Placing::Append,
                },
            });
        }
        EntryKind::Playlist => fx.push(Effect::ReadPlaylist {
            path: entry.path,
            how: Placing::Append,
        }),
        EntryKind::Parent => {}
    }
}
/// `ctrl+s`: the queue saved as a playlist among the saved queues.
fn save_queue(state: &mut State, fx: &mut Vec<Effect>) {
    if state.queue.paths.is_empty() {
        return say(state, Level::Info, "The queue is empty: nothing to save");
    }
    if state.queues_dir.is_none() {
        return say(state, Level::Info, "No folder to save queues in");
    }
    fx.push(Effect::SaveQueue(state.queue.paths.clone()));
}

/// `ctrl+l`: the saved queues, in the browser.
fn open_saved_queues(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(dir) = state.queues_dir.clone() else {
        return say(state, Level::Info, "No folder of saved queues");
    };
    state.screen = Screen::Main;
    state.side = Side::Browser;
    enter_dir(state, dir, fx);
}

fn search_below(state: &mut State, fx: &mut Vec<Effect>) {
    let query = state
        .browser
        .filter
        .as_ref()
        .map(|f| f.text.clone())
        .unwrap_or_default();
    if query.is_empty() {
        say(
            state,
            Level::Info,
            "Type a filter first (/), then search below it",
        );
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
    set_cursor(state, 0);
    fx.push(Effect::Search { id, root, query });
}

fn open_settings(state: &mut State, fx: &mut Vec<Effect>) {
    state.screen = Screen::Settings;
    state.palette = None;
    if state.settings.devices.is_none() {
        fx.push(Effect::ListDevices);
    }
}

fn settings_open(state: &mut State, fx: &mut Vec<Effect>) {
    match state.settings_row() {
        SettingsRow::DataFolder => open_folders(state, fx),
        SettingsRow::Device => {
            state.settings.picking = true;
            if state.settings.devices.is_none() {
                fx.push(Effect::ListDevices);
            }
        }
        _ => settings_step(state, 1, fx),
    }
}

fn cycle<T: PartialEq + Copy>(all: &[T], current: T, step: i32) -> T {
    let at = all.iter().position(|x| *x == current).unwrap_or(0) as i32;
    let n = all.len() as i32;
    all[((at + step) % n + n) as usize % n as usize]
}

fn settings_step(state: &mut State, step: i32, fx: &mut Vec<Effect>) {
    let config = &mut state.config;
    match SettingsRow::ALL[state.settings.row.min(SettingsRow::ALL.len() - 1)] {
        SettingsRow::Device => {
            state.settings.picking = true;
            if state.settings.devices.is_none() {
                fx.push(Effect::ListDevices);
            }
            return;
        }
        SettingsRow::RateCap => {
            // From the cap in force, through the caps the device can
            // satisfy.
            let from = state.effective_max_rate();
            state.config.max_rate = cycle(&state.rate_caps(), from, step);
        }
        SettingsRow::Resample => config.resample = !config.resample,
        SettingsRow::Headroom => config.headroom = cycle(&Headroom::CHOICES, config.headroom, step),
        // The tempo rows are not the engine's: saved, not sent.
        SettingsRow::Tempo => {
            config.bpm = !config.bpm;
            fx.push(Effect::SaveConfig);
            return;
        }
        SettingsRow::Voting => {
            config.beat_voting = !config.beat_voting;
            fx.push(Effect::SaveConfig);
            return;
        }
        SettingsRow::Method(method) => {
            if !config.beat_voting {
                config.beat_method = method;
            } else if !config.beat_methods.contains(&method) {
                config.beat_methods.push(method);
                config.beat_methods.sort_unstable();
            } else if config.beat_methods.len() > 1 {
                config.beat_methods.retain(|m| *m != method);
            } else {
                say(state, Level::Info, "One beat method stays on".to_owned());
                return;
            }
            fx.push(Effect::SaveConfig);
            return;
        }
        // Not the engine's either: how the player sends stem changes.
        SettingsRow::SkipMix => {
            config.skip_mix = !config.skip_mix;
            fx.push(Effect::SaveConfig);
            return;
        }
        SettingsRow::DataFolder => {
            open_folders(state, fx);
            return;
        }
        SettingsRow::SkipPreset => {
            // The Mix tab's, then each preset, either way round.
            let choices: Vec<Option<Preset>> = std::iter::once(None)
                .chain(Preset::ALL.into_iter().map(Some))
                .collect();
            config.skip_preset = cycle(&choices, config.skip_preset, step);
            fx.push(Effect::SaveConfig);
            return;
        }
        SettingsRow::StemSwitch => {
            config.stems_on_bar = !config.stems_on_bar;
            fx.push(Effect::SaveConfig);
            return;
        }
        SettingsRow::Stems => {
            match state.gpu {
                Gpu::Present(_) => {
                    state.config.stems = !state.config.stems;
                    fx.push(Effect::SaveConfig);
                }
                Gpu::Absent => say(state, Level::Info, "Stems need a GPU".to_owned()),
                Gpu::Looking => say(state, Level::Info, "Still looking for a GPU".to_owned()),
            }
            return;
        }
    }
    settings_changed(state, fx);
}

/// The settings changed: the UI shows them at once and the file gets them;
/// the engine gets them [`SETTINGS_DELAY_MS`] after the last change (on
/// [`Action::Tick`]), so stepping through values reroutes the track once.
fn settings_changed(state: &mut State, fx: &mut Vec<Effect>) {
    state.settings_due = Some(state.now_ms + SETTINGS_DELAY_MS);
    fx.push(Effect::SaveConfig);
}

/// [`send_settings`], for the other core modules.
pub(crate) fn send_settings_now(state: &mut State, fx: &mut Vec<Effect>) {
    send_settings(state, fx);
}

/// Sends the settings in force to the engine now.
fn send_settings(state: &mut State, fx: &mut Vec<Effect>) {
    state.settings_due = None;
    state.karaoke.stems_sent = karaoke::stems_due(state);
    fx.push(Effect::Engine(Command::SetSettings(
        state.engine_settings(),
    )));
}

/// The files whose beats should be known: the playing track and the next
/// [`RHYTHM_AHEAD`] in the queue; none while the tempo is off.
fn rhythm_wanted(state: &State) -> Vec<PathBuf> {
    match state.queue.current {
        Some(current) if state.config.bpm => {
            // The track a transition just left, while its block slides away.
            let from = match state.mix.slide {
                Some(_) => current.saturating_sub(1),
                None => current,
            };
            state
                .queue
                .paths
                .iter()
                .skip(from)
                .take(1 + RHYTHM_AHEAD + current - from)
                .cloned()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The recipe in force: the methods, and stems when they are on and there
/// is a GPU; `None` while stems are on and the GPU is still being looked
/// for, so nothing is analysed twice.
fn rhythm_recipe(state: &State) -> Option<RhythmRecipe> {
    let stems = match state.gpu {
        _ if !state.config.stems => false,
        Gpu::Looking => return None,
        Gpu::Absent => false,
        Gpu::Present(_) => true,
    };
    Some(RhythmRecipe {
        methods: state.config.methods_in_force(),
        stems,
    })
}

/// When the files to analyse or the recipe change, forgets the analyses
/// no longer wanted (all of them, for another recipe) and asks for those
/// not done yet, in order.
fn want_rhythm(state: &mut State, fx: &mut Vec<Effect>) {
    let wanted = rhythm_wanted(state);
    let Some(recipe) = rhythm_recipe(state) else {
        return;
    };
    // Nothing wanted before or now: another recipe changes nothing.
    if wanted == state.rhythm_wanted && (recipe == state.rhythm_recipe || wanted.is_empty()) {
        state.rhythm_recipe = recipe;
        return;
    }
    if recipe != state.rhythm_recipe {
        state.rhythm.clear();
    }
    state.rhythm.retain(|path, _| wanted.contains(path));
    let mut ask = Vec::new();
    for path in &wanted {
        let rhythm = state.rhythm.entry(path.clone()).or_insert(Rhythm::Pending);
        if matches!(rhythm, Rhythm::Pending) {
            ask.push(path.clone());
        }
    }
    state.rhythm_wanted = wanted;
    state.rhythm_recipe = recipe.clone();
    fx.push(Effect::WantRhythm(ask, recipe));
}

/// Gets the whole queue ready ahead when it or the track playing changed:
/// from the track playing (or the current one) to the end, then from the
/// start; analysed by the recipe in force (none while the tempo is off)
/// and their lyrics looked up and timed, into the caches.
fn prepare(state: &mut State, fx: &mut Vec<Effect>) {
    let recipe = if state.config.bpm {
        // Stems on and the GPU still being looked for: not yet.
        let Some(recipe) = rhythm_recipe(state) else {
            return;
        };
        Some(recipe)
    } else {
        None
    };
    let paths = &state.queue.paths;
    let from = state
        .playing
        .track
        .as_ref()
        .map(|t| t.index)
        .or(state.queue.current)
        .unwrap_or(0)
        .min(paths.len());
    let order = || paths[from..].iter().chain(&paths[..from]);
    let (old, old_recipe) = &state.prepared;
    if *old_recipe == recipe && old.len() == paths.len() && old.iter().eq(order()) {
        return;
    }
    let order: Vec<PathBuf> = order().cloned().collect();
    // Another recipe: the analyses made by the last are not this one's.
    if *old_recipe != recipe {
        for jobs in state.ready.values_mut() {
            jobs.retain(|j| *j != Job::Rhythm);
        }
        for facts in state.facts.values_mut() {
            (facts.key, facts.stems, facts.tempo) = (None, None, None);
        }
    }
    state.prepared = (order.clone(), recipe.clone());
    fx.push(Effect::Prepare(order, recipe));
}

/// `X` (or the setting's row): the next-track key cuts or mixes in, the
/// other way from now; kept in the settings. Which preset it mixes by is a
/// row of its own (the Mix tab's, or one set there).
fn cycle_skip_mix(state: &mut State, fx: &mut Vec<Effect>) {
    state.config.skip_mix = !state.config.skip_mix;
    let what = if state.config.skip_mix {
        format!("mixes into it: {}", skip_preset_name(state))
    } else {
        "cuts to it".to_owned()
    };
    say(state, Level::Info, format!("Next track {what}"));
    fx.push(Effect::SaveConfig);
}

/// The preset the next-track key mixes by, named: its own, or the Mix
/// tab's as set now.
pub(crate) fn skip_preset_name(state: &State) -> String {
    match state.config.skip_preset {
        Some(p) => p.name().to_owned(),
        None => format!("{} (the Mix tab's)", state.mix.draft.preset.name()),
    }
}

/// How far a file in the queue is got ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// A worker is at it.
    Working,
    /// Not yet: some work for it is still to come.
    Waiting,
    /// Everything is in the caches: its analysis and waveforms (while the
    /// tempo is on) and its lyrics, looked up and timed (where they are got
    /// ready).
    Done,
}

/// How far `path` is got ready ([`Readiness`]).
pub fn readiness(state: &State, path: &Path) -> Readiness {
    if state.working.contains_key(path) {
        return Readiness::Working;
    }
    let done = state.ready.get(path);
    let has = |job: Job| done.is_some_and(|jobs| jobs.contains(&job));
    // Told by the workers, or known here already.
    let analysed = matches!(
        state.rhythm.get(path),
        Some(Rhythm::Ready(_) | Rhythm::Failed)
    );
    let settled = match state.karaoke.lyrics.get(path) {
        Some(karaoke::SongLyrics::Found { shown, .. }) => {
            shown.timing.detail == jodel_lyrics::Detail::Words
                || state.karaoke.model == karaoke::ModelState::Missing
        }
        Some(karaoke::SongLyrics::Pending) | None => false,
        Some(_) => true,
    };
    let rhythm = state.prepared.1.is_none() || has(Job::Rhythm) || analysed;
    let lyrics = !state.prepares_lyrics || has(Job::Lyrics) || settled;
    if rhythm && lyrics {
        Readiness::Done
    } else {
        Readiness::Waiting
    }
}

/// Seconds ahead of what is heard that a stem change can be made at the
/// soonest on the stems: the output holds about half a second.
const STEMS_SOON: f64 = 1.0;
/// The same for the first stem off away from the Mix tab: the track moves
/// onto its stems first, separated from where they meet. On the Mix tab
/// the playing track's stems are kept as it plays, so [`STEMS_SOON`].
const STEMS_ROUTE_SOON: f64 = 2.5;

/// Where a change from the stems heard to `stems` is to be heard, with
/// the stems switched on bars: the first bar of the playing track the
/// engine can still make it at ([`STEMS_SOON`], [`STEMS_ROUTE_SOON`]);
/// `None` without bars there (then at once).
pub fn stems_bar(state: &State, stems: StemMix) -> Option<(f64, usize)> {
    if !state.config.stems_on_bar {
        return None;
    }
    let a = &mix::rhythmed(state, mix::Block::Now)?.analysis;
    let soon = if state.stems.is_full() && !stems.is_full() && state.side != Side::Mix {
        STEMS_ROUTE_SOON
    } else {
        STEMS_SOON
    };
    let after = state.heard() + soon;
    a.downbeats
        .iter()
        .enumerate()
        .map(|(k, &d)| (a.beats[d], k + 1))
        .find(|&(t, _)| t >= after)
}

/// The stems heard become `stems`: from the next bar the engine can make
/// ([`stems_bar`]), else at once, as soon as the output buffer has played.
/// The first stem off and all on again move the track to its other route
/// where it is.
fn set_stems(state: &mut State, stems: StemMix, fx: &mut Vec<Effect>) {
    match state.gpu {
        Gpu::Present(_) => {}
        Gpu::Looking => return say(state, Level::Info, "Still looking for a GPU for the stems"),
        Gpu::Absent => return say(state, Level::Info, "Stems need a GPU"),
    }
    if stems == state.stems {
        return;
    }
    let bar = stems_bar(state, stems);
    state.stems = stems;
    let mut what = if stems.is_full() {
        "Stems: all on, the file as it is".to_owned()
    } else {
        format!("Stems: {} (not bit-perfect)", stems.describe())
    };
    match bar {
        Some((at, bar)) => {
            what.push_str(&format!(", from bar {bar}"));
            fx.push(Effect::Engine(Command::SetStemsAt { stems, at }));
        }
        None => send_settings(state, fx),
    }
    say(state, Level::Info, what);
}

/// The device's rates arrived: a rate cap it cannot satisfy is shown as
/// the cap in force ([`State::effective_max_rate`]), with a message.
fn device_rates(state: &mut State, rates: Option<Vec<u32>>) {
    state.device_rates = rates.map(|mut rates| {
        rates.sort_unstable();
        rates.dedup();
        rates
    });
    let (Some(saved), Some(lowest)) = (
        state.config.max_rate,
        state.device_rates.as_ref().and_then(|r| r.first().copied()),
    ) else {
        return;
    };
    if saved >= lowest {
        return;
    }
    let using = match state.effective_max_rate() {
        Some(cap) => format!("{} kHz", text::khz_short(cap)),
        None => "no cap".into(),
    };
    let device = state.device_label();
    say(
        state,
        Level::Info,
        format!(
            "Rate cap {} kHz is below every rate {device} takes ({} kHz up); using {using}",
            text::khz_short(saved),
            text::khz_short(lowest)
        ),
    );
}

fn choose_device(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(choice) = state
        .settings
        .devices
        .as_ref()
        .and_then(|list| list.get(state.settings.device_cursor))
        .cloned()
    else {
        return;
    };
    state.settings.picking = false;
    state.config.device = choice.id.clone();
    fx.push(Effect::SaveConfig);
    if state.null_output {
        say(
            state,
            Level::Info,
            format!(
                "Saved {}; this run keeps playing into the null output",
                choice.name
            ),
        );
    } else {
        say(state, Level::Info, format!("Playing on {}", choice.name));
        state.device_volume = None;
        fx.push(Effect::SwitchDevice(choice.id));
    }
}

/// The device's volume control as read or set ([`Action::DeviceVolume`]):
/// kept for the signal path; a change by a key is said.
fn device_volume(state: &mut State, result: Result<Option<DeviceVolume>, String>, changed: bool) {
    match result {
        Ok(Some(volume)) => {
            state.device_volume = Some(volume);
            if changed {
                say(
                    state,
                    Level::Info,
                    format!("Device volume {}", volume.label()),
                );
            }
        }
        Ok(None) => {
            state.device_volume = None;
            if changed {
                let text = if state.null_output {
                    "The null output has no volume"
                } else {
                    "The system's volume keys set this device's volume"
                };
                say(state, Level::Info, text);
            }
        }
        Err(error) => {
            if changed {
                say(state, Level::Error, format!("Device volume: {error}"));
            }
        }
    }
}

fn run_palette(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(palette) = state.palette.take() else {
        return;
    };
    let matches = keymap::palette_matches(&palette.text);
    if let Some(entry) = matches.get(palette.cursor) {
        state.screen = Screen::Main;
        apply_op(state, entry.op, fx);
    }
}

fn move_by(state: &mut State, delta: i32, fx: &mut Vec<Effect>) {
    let _ = fx;
    if let Some(palette) = &mut state.palette {
        let n = keymap::palette_matches(&palette.text).len();
        palette.cursor = offset(palette.cursor, delta).min(n.saturating_sub(1));
        return;
    }
    match state.screen {
        Screen::Main => {
            let (cursor, _) = list_cursor(state);
            set_cursor(state, offset(cursor, delta));
        }
        Screen::Settings if state.settings.folders.is_some() => {
            if let Some(pick) = &mut state.settings.folders {
                let n = pick.rows().len();
                pick.cursor = offset(pick.cursor, delta).min(n.saturating_sub(1));
            }
        }
        Screen::Settings if state.settings.picking => {
            let n = state.settings.devices.as_ref().map_or(0, Vec::len);
            state.settings.device_cursor =
                offset(state.settings.device_cursor, delta).min(n.saturating_sub(1));
        }
        Screen::Settings => {
            state.settings.row = offset(state.settings.row, delta).min(SettingsRow::ALL.len() - 1);
        }
        Screen::Help => scroll_help(state, delta),
        Screen::NowPlaying | Screen::Karaoke => {}
    }
}

fn page(state: &State) -> i32 {
    match state.screen {
        Screen::Help => i32::from(state.size.1.saturating_sub(6)).max(1),
        _ => (list_height(state) as i32 - 1).max(1),
    }
}

fn apply_op(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    if karaoke::apply(state, op, fx) || mix::apply(state, op, fx) {
        return;
    }
    match op {
        Op::PlayPause => match state.playing.state {
            PlayState::Playing => fx.push(Effect::Engine(Command::Pause)),
            PlayState::Paused => fx.push(Effect::Engine(Command::Resume)),
            PlayState::Stopped => {
                if !state.queue.paths.is_empty() {
                    let index = state.queue.current.unwrap_or(0);
                    fx.push(Effect::Engine(Command::Play(index)));
                } else {
                    let rows = browser_rows(state);
                    let cursor = state.browser.cursor;
                    let is_file = rows
                        .get(cursor)
                        .and_then(|&r| row_entry(state, r))
                        .is_some_and(|e| e.kind == EntryKind::File);
                    if is_file {
                        play_browser_from(state, &rows, cursor, fx);
                    } else {
                        say(
                            state,
                            Level::Info,
                            "Nothing to play: pick a file and press enter",
                        );
                    }
                }
            }
        },
        Op::SeekBack | Op::SeekForward => {
            if state.playing.track.is_some() {
                let step = if op == Op::SeekBack {
                    -SEEK_STEP
                } else {
                    SEEK_STEP
                };
                let mut target = (state.playing.position + step).max(0.0);
                if let Some(duration) = state.playing.duration() {
                    target = target.min(duration);
                }
                state.playing.position = target;
                fx.push(Effect::Engine(Command::Seek(target)));
            }
        }
        Op::Next => mix::skip(state, fx),
        Op::SkipMix => cycle_skip_mix(state, fx),
        Op::Previous => fx.push(Effect::Engine(Command::Previous)),
        Op::Stop => fx.push(Effect::Engine(Command::Stop)),
        Op::ToggleSide => {
            // Typing a filter ends; the filter itself stays.
            if let Some(filter) = state.browser.filter.as_mut() {
                filter.editing = false;
            }
            let at = Side::ALL.iter().position(|&s| s == state.side).unwrap_or(0);
            state.side = Side::ALL[(at + 1) % Side::ALL.len()];
            state.screen = Screen::Main;
            fix_scroll(state);
        }
        Op::GoTrack | Op::GoAlbum | Op::GoArtist => match playing_jump(state, op) {
            Some((dir, select)) => jump_to(state, dir, select, fx),
            None => say(state, Level::Info, "Nothing playing"),
        },
        Op::NowPlaying => {
            state.screen = if state.screen == Screen::NowPlaying {
                Screen::Main
            } else {
                Screen::NowPlaying
            };
        }
        Op::Settings => {
            if state.screen == Screen::Settings {
                state.screen = Screen::Main;
            } else {
                open_settings(state, fx);
            }
        }
        Op::Help => {
            state.screen = if state.screen == Screen::Help {
                Screen::Main
            } else {
                // The help opens whole: a search from before is gone.
                state.help_search = None;
                Screen::Help
            };
        }
        Op::Palette => {
            state.screen = Screen::Main;
            state.palette = Some(Palette::default());
        }
        Op::Filter if state.screen == Screen::Help => {
            // On the help, `/` searches it (again with what was typed).
            let text = state.help_search.take().map(|s| s.text).unwrap_or_default();
            state.help_search = Some(Filter {
                text,
                editing: true,
            });
        }
        Op::Filter => {
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
        Op::Quit => {
            state.quit = true;
            if state.settings_due.is_some() {
                send_settings(state, fx);
            }
            fx.push(Effect::Quit);
        }
        Op::ToggleDrums | Op::ToggleBass | Op::ToggleVocals | Op::ToggleOther => {
            if let Some(stem) = op.stem() {
                let stems = state.stems.toggle(stem);
                set_stems(state, stems, fx);
            }
        }
        Op::AllStems => set_stems(state, StemMix::ALL, fx),
        Op::JumpBar(offset) => mix::jump_bar(state, isize::from(offset), false, fx),
        Op::JumpBarAtEnd(offset) => mix::jump_bar(state, isize::from(offset), true, fx),
        Op::VolumeUp => fx.push(Effect::DeviceVolume(VolumeChange::Step(1))),
        Op::VolumeDown => fx.push(Effect::DeviceVolume(VolumeChange::Step(-1))),
        Op::ToggleBpm => {
            state.config.bpm = !state.config.bpm;
            let what = if state.config.bpm { "on" } else { "off" };
            say(state, Level::Info, format!("Tempo and beat pulse {what}"));
            fx.push(Effect::SaveConfig);
        }
        Op::Back => back(state, fx),
        Op::Up => move_by(state, -1, fx),
        Op::Down => move_by(state, 1, fx),
        Op::PageUp => {
            let p = page(state);
            move_by(state, -p, fx)
        }
        Op::PageDown => {
            let p = page(state);
            move_by(state, p, fx)
        }
        Op::Top => set_cursor(state, 0),
        Op::Bottom => set_cursor(state, usize::MAX),
        Op::Open => match state.screen {
            Screen::Settings if state.settings.folders.is_some() => folder_open(state, fx),
            Screen::Settings if state.settings.picking => choose_device(state, fx),
            Screen::Settings => settings_open(state, fx),
            _ => open(state, fx),
        },
        Op::Parent if state.screen == Screen::Settings && state.settings.folders.is_some() => {
            folder_parent(state, fx)
        }
        Op::Parent => {
            if state.side == Side::Browser {
                go_parent(state, fx);
            }
        }
        Op::Enqueue => enqueue(state, fx),
        Op::QueuePlay => activate(state, Placing::AppendPlay, fx),
        Op::OpenReplace => activate(state, Placing::Replace, fx),
        Op::QueueRemove => remove_from_queue(state, fx),
        Op::QueueSave => save_queue(state, fx),
        Op::QueueOpenSaved => open_saved_queues(state, fx),
        Op::QueueMoveUp => {
            let from = state.queue.cursor;
            if from > 0 {
                move_in_queue(state, from, from - 1, fx);
            }
        }
        Op::QueueMoveDown => {
            let from = state.queue.cursor;
            if from + 1 < state.queue.paths.len() {
                move_in_queue(state, from, from + 1, fx);
            }
        }
        Op::SearchBelow => search_below(state, fx),
        Op::Accept => {
            if state.palette.is_some() {
                run_palette(state, fx);
            } else if help_typing(state) {
                // The search stays, the keys are the help's again.
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
            } else if let Some(filter) = &mut state.browser.filter {
                filter.editing = false;
                // A path that exists is jumped to; anything else is
                // searched for below the folder (`Action::JumpMissed`).
                if !filter.text.trim().is_empty() {
                    fx.push(Effect::Jump {
                        text: filter.text.clone(),
                        from: state.browser.dir.clone(),
                    });
                }
                if filter.text.is_empty() {
                    state.browser.filter = None;
                }
                fix_scroll(state);
            }
        }
        Op::Erase => {
            if let Some(palette) = &mut state.palette {
                if palette.text.pop().is_none() {
                    state.palette = None;
                }
                return;
            }
            if help_typing(state) {
                if let Some(search) = &mut state.help_search {
                    if search.text.pop().is_none() {
                        state.help_search = None;
                    }
                }
                state.help_scroll = 0;
                return;
            }
            let selected = selected_path(state);
            if let Some(filter) = &mut state.browser.filter {
                if filter.text.pop().is_none() {
                    state.browser.filter = None;
                }
                reselect(state, selected);
            }
        }
        Op::Paste => {
            if state.palette.is_some() || state.screen == Screen::Main || help_typing(state) {
                fx.push(Effect::ReadClipboard);
            }
        }
        Op::MixBarBack
        | Op::MixBarForward
        | Op::MixBeatBack
        | Op::MixBeatForward
        | Op::MixSelectBack
        | Op::MixSelectForward
        | Op::MixZoomIn
        | Op::MixZoomOut
        | Op::MixFitLyrics
        | Op::MixFollow
        | Op::MixLyrics
        | Op::GridEarlier
        | Op::GridLater
        | Op::GridNudgeBack
        | Op::GridNudgeForward
        | Op::GridHalf
        | Op::GridDouble
        | Op::GridOne
        | Op::GridReset
        | Op::TransitionGo
        | Op::TransitionPreset
        | Op::TransitionTempo
        | Op::TransitionPairing
        | Op::TransitionEdit
        | Op::EditBack
        | Op::EditForward
        | Op::EditRaise
        | Op::EditLower
        | Op::EditOff
        | Op::EditFull
        | Op::EditCut => say(state, Level::Info, "That is for the Mix tab (tab)"),
        Op::KaraokeEnter | Op::KaraokeEnterHere => {
            // From anywhere (the palette): the tab first, then the mode.
            state.palette = None;
            state.side = Side::Karaoke;
            state.screen = Screen::Main;
            karaoke::enter(state, fx, op == Op::KaraokeEnterHere);
        }
        Op::KaraokeStart
        | Op::KaraokeMicNext
        | Op::KaraokeRecordEcho
        | Op::KaraokeGame
        | Op::KaraokeLevel
        | Op::KaraokeNotes
        | Op::KaraokeKeep
        | Op::SliderDown
        | Op::SliderUp
        | Op::SliderFineDown
        | Op::SliderFineUp => say(state, Level::Info, "That is for the Karaoke tab (tab)"),
        Op::ValueDown => settings_step(state, -1, fx),
        Op::ValueUp => settings_step(state, 1, fx),
    }
}

/// Pasted text as one line: line breaks and tabs become spaces, other
/// control characters go, and the ends are trimmed.
fn one_line(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            '\n' | '\r' | '\t' => Some(' '),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

/// Puts pasted `text` into the palette or the filter being typed; on the
/// main screen otherwise, opens the filter with it.
fn paste(state: &mut State, text: &str) {
    let text = one_line(text);
    if text.is_empty() {
        return;
    }
    if let Some(palette) = &mut state.palette {
        palette.text.push_str(&text);
        palette.cursor = 0;
        return;
    }
    if help_typing(state) {
        if let Some(search) = state.help_search.as_mut() {
            search.text.push_str(&text);
        }
        state.help_scroll = 0;
        return;
    }
    if state.screen != Screen::Main {
        return;
    }
    // The filter is the browser's (the Mix tab has none to show).
    state.side = Side::Browser;
    let selected = selected_path(state);
    let filter = state.browser.filter.get_or_insert_with(Filter::default);
    if !filter.editing {
        filter.text.clear();
        filter.editing = true;
    }
    filter.text.push_str(&text);
    reselect(state, selected);
}

fn back(state: &mut State, fx: &mut Vec<Effect>) {
    if state.palette.take().is_some() {
        return;
    }
    match state.screen {
        Screen::Main => {
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
            fix_scroll(state);
        }
        Screen::Settings if state.settings.folders.is_some() => state.settings.folders = None,
        Screen::Settings if state.settings.picking => state.settings.picking = false,
        // A search of the help is cleared first, then the help is left.
        Screen::Help if state.help_search.is_some() => {
            state.help_search = None;
            state.help_scroll = 0;
        }
        _ => state.screen = Screen::Main,
    }
}

fn engine_event(state: &mut State, event: Event) {
    let playing = &mut state.playing;
    match event {
        Event::QueueChanged { queue, current } => {
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
            let side = state.side;
            state.side = Side::Queue;
            set_cursor(state, cursor);
            state.side = side;
        }
        Event::State(new) => {
            playing.state = new;
            if new == PlayState::Stopped {
                playing.track = None;
                playing.position = 0.0;
                playing.clipped = 0;
            }
        }
        Event::TrackStarted(start) => {
            let start = *start;
            mix::track_started(state, start.index);
            let playing = &mut state.playing;
            // Where it starts: 0, or the position a switch of route (or a
            // seek) took it up at.
            let at = start.from_frame as f64 / f64::from(start.info.rate.max(1));
            state.queue.current = Some(start.index);
            playing.track = Some(NowTrack {
                index: start.index,
                info: Arc::clone(&start.info),
                route: start.route,
                output: start.output,
                device: start.device,
                bit_perfect: start.bit_perfect,
            });
            playing.position = at;
            playing.clipped = 0;
        }
        Event::Position(position) => {
            if playing
                .track
                .as_ref()
                .is_some_and(|t| t.index == position.index)
            {
                playing.position = position.seconds;
                playing.anchor = Some((position.index, position.output_frame, position.seconds));
            }
        }
        Event::Seeked { index, seconds } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.position = seconds;
                playing.clipped = 0;
            }
        }
        Event::Jumped { index, seconds } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.position = seconds;
            }
        }
        Event::Clipped { index, samples, .. } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.clipped = samples;
            }
        }
        Event::TrackEnded { .. } => {}
        Event::Error {
            index,
            error,
            skipped,
        } => {
            let name = index
                .and_then(|i| state.queue.paths.get(i))
                .and_then(|p| p.file_name())
                .map(|n| format!("{}: ", n.to_string_lossy()))
                .unwrap_or_default();
            let tail = if skipped { " (skipped)" } else { "" };
            say(state, Level::Error, format!("{name}{error}{tail}"));
        }
        Event::Notice { message, .. } => say(state, Level::Info, message),
        Event::Transition(t) => mix::transition_event(state, t),
        Event::StemsStopped { reason, .. } => {
            // The engine plays the whole mix now: so do the buttons.
            state.stems = StemMix::ALL;
            say(
                state,
                Level::Error,
                format!("The stems stopped ({reason}); playing the whole mix"),
            );
        }
        Event::QueueEnded => say(state, Level::Info, "End of the queue"),
        Event::Held { index } => karaoke::held(state, index),
    }
}
