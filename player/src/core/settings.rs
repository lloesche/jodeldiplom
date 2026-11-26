//! The settings screen: its rows stepped and opened, the device list and
//! the data folder's list; the settings saved, and sent to the engine a
//! moment after the last change.

use std::path::{Path, PathBuf};

use jodel_playback::{Command, Headroom};

use crate::core::transition::Preset;
use crate::core::update::offset;
use crate::core::{
    DeviceChoice, Effect, FolderPick, FolderRow, Gpu, Screen, SettingsRow, State, SETTINGS_DELAY_MS,
};

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

/// The settings screen, its devices listed the first time.
pub(crate) fn open(state: &mut State, fx: &mut Vec<Effect>) {
    state.screen = Screen::Settings;
    state.palette = None;
    if state.settings.devices.is_none() {
        fx.push(Effect::ListDevices);
    }
}

/// [`crate::core::Action::OpenSettings`]: the settings screen on `row`
/// (the device list open, for the device's).
pub(crate) fn open_at(state: &mut State, row: SettingsRow, fx: &mut Vec<Effect>) {
    open(state, fx);
    state.settings.row = SettingsRow::ALL.iter().position(|r| *r == row).unwrap_or(0);
    if row == SettingsRow::Device {
        state.settings.picking = true;
    }
}

/// [`crate::core::Action::SettingsClick`]: row `row` clicked, opened
/// (`step` 0) or stepped `step` values on.
pub(crate) fn click(state: &mut State, row: usize, step: i32, fx: &mut Vec<Effect>) {
    state.settings.row = row.min(SettingsRow::ALL.len() - 1);
    state.settings.picking = false;
    state.settings.folders = None;
    if step == 0 {
        open_row(state, fx);
    } else {
        step_row(state, step, fx);
    }
}

/// `enter` on the settings screen: the folder or the device selected in
/// the list open, else the row opened.
pub(crate) fn enter(state: &mut State, fx: &mut Vec<Effect>) {
    if state.settings.folders.is_some() {
        folder_open(state, fx);
    } else if state.settings.picking {
        choose_device(state, fx);
    } else {
        open_row(state, fx);
    }
}

/// `esc` on the settings screen: the list open closes, else the screen.
pub(crate) fn back(state: &mut State) {
    if state.settings.folders.is_some() {
        state.settings.folders = None;
    } else if state.settings.picking {
        state.settings.picking = false;
    } else {
        state.screen = Screen::Main;
    }
}

/// `↑` `↓` on the settings screen, `delta` rows: in the list open, else
/// through the rows.
pub(crate) fn move_by(state: &mut State, delta: i32) {
    let s = &mut state.settings;
    if let Some(pick) = &mut s.folders {
        let n = pick.rows().len();
        pick.cursor = offset(pick.cursor, delta).min(n.saturating_sub(1));
    } else if s.picking {
        let n = s.devices.as_ref().map_or(0, Vec::len);
        s.device_cursor = offset(s.device_cursor, delta).min(n.saturating_sub(1));
    } else {
        s.row = offset(s.row, delta).min(SettingsRow::ALL.len() - 1);
    }
}

/// `enter` (or a click) on the selected row: the data folder's list or
/// the device list opens; any other row steps on.
fn open_row(state: &mut State, fx: &mut Vec<Effect>) {
    match state.settings_row() {
        SettingsRow::DataFolder => open_folders(state, fx),
        SettingsRow::Device => pick_device(state, fx),
        _ => step_row(state, 1, fx),
    }
}

/// The device list opens (the devices listed the first time).
fn pick_device(state: &mut State, fx: &mut Vec<Effect>) {
    state.settings.picking = true;
    if state.settings.devices.is_none() {
        fx.push(Effect::ListDevices);
    }
}

/// The one of `all` `step` away from `current` (from the first, where it
/// is not among them), going round; `current` itself where `all` is empty.
fn cycle<T: PartialEq + Copy>(all: &[T], current: T, step: i32) -> T {
    let n = all.len() as i32;
    if n == 0 {
        return current;
    }
    let at = all.iter().position(|x| *x == current).unwrap_or(0) as i32;
    all[((at + step) % n + n) as usize % n as usize]
}

/// `←` `→` (and a click) on the selected row: its value `step` on. The
/// headroom reaches the engine too; the other rows are the player's,
/// saved only.
pub(crate) fn step_row(state: &mut State, step: i32, fx: &mut Vec<Effect>) {
    let row = state.settings_row();
    let config = &mut state.config;
    match row {
        SettingsRow::Device => return pick_device(state, fx),
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
                state.info("One beat method stays on");
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
        SettingsRow::DataFolder => return open_folders(state, fx),
        SettingsRow::SkipPreset => {
            // The Transition tab's, then each preset, either way round.
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
                Gpu::Absent => state.info("Stems need a GPU"),
                Gpu::Looking => state.info("Still looking for a GPU"),
            }
            return;
        }
    }
    changed(state, fx);
}

/// The settings changed: the UI shows them at once and the file gets them;
/// the engine gets them [`SETTINGS_DELAY_MS`] after the last change (on
/// [`crate::core::Action::Tick`]), so stepping through values reroutes the
/// track once.
fn changed(state: &mut State, fx: &mut Vec<Effect>) {
    state.settings_due = Some(state.now_ms + SETTINGS_DELAY_MS);
    fx.push(Effect::SaveConfig);
}

/// Sends the settings in force to the engine now.
pub(crate) fn send(state: &mut State, fx: &mut Vec<Effect>) {
    state.settings_due = None;
    fx.push(Effect::Engine(Command::SetSettings(
        state.config.settings(),
    )));
}

/// `T`: tempo and the beat pulse on or off, kept in the settings.
pub(crate) fn toggle_bpm(state: &mut State, fx: &mut Vec<Effect>) {
    state.config.bpm = !state.config.bpm;
    let what = if state.config.bpm { "on" } else { "off" };
    state.info(format!("Tempo and beat pulse {what}"));
    fx.push(Effect::SaveConfig);
}

/// `X` (or the setting's row): the next-track key cuts or mixes in, the
/// other way from now; kept in the settings. Which preset it mixes by is a
/// row of its own (the Transition tab's, or one set there).
pub(crate) fn cycle_skip_mix(state: &mut State, fx: &mut Vec<Effect>) {
    state.config.skip_mix = !state.config.skip_mix;
    let what = if state.config.skip_mix {
        format!("mixes into it: {}", skip_preset_name(state))
    } else {
        "cuts to it".to_owned()
    };
    state.info(format!("Next track {what}"));
    fx.push(Effect::SaveConfig);
}

/// The preset the next-track key mixes by, named: its own, or the Transition
/// tab's as set now.
pub(crate) fn skip_preset_name(state: &State) -> String {
    match state.config.skip_preset {
        Some(p) => p.name().to_owned(),
        None => format!(
            "{} (the Transition tab's)",
            state.transition.draft.preset.name()
        ),
    }
}

/// [`crate::core::Action::Devices`]: the devices to choose from, "System
/// default" first, the one in the settings selected; or why there are
/// none.
pub(crate) fn devices(state: &mut State, result: Result<Vec<DeviceChoice>, String>) {
    match result {
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
            state.error(format!("Listing devices failed: {error}"));
        }
    }
}

/// [`crate::core::Action::ChooseDevice`]: device `index` of the list.
pub(crate) fn choose_device_at(state: &mut State, index: usize, fx: &mut Vec<Effect>) {
    state.settings.device_cursor = index;
    choose_device(state, fx);
}

/// The device selected in the list is chosen: saved, and played on from
/// now (unless this run plays into the null output).
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
        state.info(format!(
            "Saved {}; this run keeps playing into the null output",
            choice.name
        ));
    } else {
        state.info(format!("Playing on {}", choice.name));
        fx.push(Effect::SwitchDevice(choice.id));
    }
}

/// The data folder's list opens on the folder the data folder is in. While
/// a move is under way it does not: one that stopped goes on instead.
fn open_folders(state: &mut State, fx: &mut Vec<Effect>) {
    if let Some(m) = state.data_move.as_ref().filter(|m| !m.finished) {
        if m.failure.is_some() {
            fx.push(Effect::RetryMove);
            state.info("Moving the data folder again");
        } else {
            state.info("The data folder is moving: choose another once it is there");
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
pub(crate) fn folder_parent(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(pick) = &state.settings.folders else {
        return;
    };
    let Some(dir) = &pick.dir else {
        return;
    };
    let up = dir.parent().map(Path::to_path_buf);
    folder_go(state, up, fx);
}

/// [`crate::core::Action::FolderRow`]: row `index` of the data folder's
/// list, as `enter` on it.
pub(crate) fn folder_row(state: &mut State, index: usize, fx: &mut Vec<Effect>) {
    if let Some(pick) = &mut state.settings.folders {
        pick.cursor = index;
    }
    folder_open(state, fx);
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

/// [`crate::core::Action::FoldersListed`]: the subfolders of `dir` (or the
/// drives), while the list still shows it.
pub(crate) fn folders_listed(state: &mut State, dir: Option<PathBuf>, folders: Vec<PathBuf>) {
    if let Some(pick) = &mut state.settings.folders {
        if pick.dir == dir {
            pick.folders = Some(folders);
            pick.cursor = pick.cursor.min(pick.rows().len().saturating_sub(1));
        }
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
            return state.info("The data folder is there already");
        }
        if to.starts_with(from) {
            return state.info("Not inside the data folder itself: choose a folder outside it");
        }
    }
    fx.push(Effect::MoveData(parent));
}
