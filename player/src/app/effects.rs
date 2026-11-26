//! The core's effects carried out by the [`App`]: the engine told, the
//! workers asked, the background jobs started (each sends its answer
//! through the [`crate::work::outbox::Outbox`]), the files written.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use super::App;
use crate::browser;
use crate::core::{self, Action, Effect};
use crate::desk_wiring::Output;

impl App {
    /// Carries out one effect.
    pub fn run(&mut self, effect: Effect) {
        match effect {
            Effect::Engine(command) => self.engine_command(command),
            Effect::Preview(what) => self.preview(what),
            Effect::ListDir(dir) => self.outbox.spawn(move || {
                let result = browser::list_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()));
                Action::Listed { dir, result }
            }),
            Effect::Jump { text, from } => {
                self.outbox
                    .spawn(move || match browser::resolve_jump(&text, &from) {
                        Some((dir, select)) => Action::JumpTo { dir, select },
                        None => Action::JumpMissed { text },
                    });
            }
            Effect::WantDetails(paths) => self.details.want(paths),
            Effect::WantRhythm(paths, recipe) => self.rhythm.want(paths, recipe),
            Effect::WantLyrics(paths) => self.lyrics.want(paths),
            Effect::SaveLyricEdits(path, edits) => self.lyrics.edited(path, edits),
            Effect::VoiceCurve(curve) => {
                if let Some(engine) = &self.engine {
                    engine.voice_curve(curve);
                }
            }
            Effect::Prepare(paths, recipe) => {
                if let Some(covers) = &self.covers {
                    covers.prepare(paths.clone());
                }
                self.lyrics.prepare(paths.clone());
                self.rhythm.prepare(paths, recipe);
            }
            Effect::NextMic => self.next_mic(),
            Effect::RecordEcho => {
                let said = self.mic.record_echo(&self.state);
                self.dispatch(said);
            }
            Effect::Sync(command) => {
                let engine = self
                    .engine
                    .as_ref()
                    .map(|e| (&e.mixer, e.strip(core::desk::StripKind::Player)));
                if let Some(action) = crate::sync::command(command, engine, &mut self.clicks) {
                    self.dispatch(action);
                }
            }
            Effect::Desk(change) => self.desk_change(change),
            Effect::Bench(ask) => {
                if let Some(engine) = &mut self.engine {
                    engine.bench(&self.state, ask);
                }
            }
            Effect::KeepPatch { theme, atom, patch } => {
                if let Some(store) = &self.patches {
                    store.keep(theme, atom, patch);
                }
            }
            Effect::Search { id, root, query } => {
                self.cancel_search();
                let cancel = Arc::new(AtomicBool::new(false));
                self.search = Some(cancel.clone());
                let outbox = self.outbox.clone();
                thread::spawn(move || {
                    browser::search(&root, &query, &cancel, |hits| {
                        outbox.send(Action::SearchHits { id, hits });
                    });
                    outbox.send(Action::SearchDone { id });
                });
            }
            Effect::CancelSearch => self.cancel_search(),
            Effect::Walk { root, purpose } => self.outbox.spawn(move || {
                let paths = browser::walk_playable(&root);
                Action::Walked { purpose, paths }
            }),
            Effect::ListDevices => {
                let devices = self.devices.clone();
                self.outbox.spawn(move || Action::Devices(devices()));
            }
            Effect::ListFolders(dir) => self.outbox.spawn(move || {
                let folders = browser::list_folders(dir.as_deref());
                Action::FoldersListed(dir, folders)
            }),
            Effect::MoveData(parent) => self.move_data(parent),
            Effect::RetryMove => self.moving.retry(),
            Effect::SwitchDevice(id) => {
                if !matches!(self.output, Output::Null(_)) {
                    self.restart_on(Output::Device(id));
                }
            }
            Effect::SaveGrid(path, grid) => {
                if let Some(store) = &self.grids {
                    store.save(path, grid);
                }
            }
            Effect::ResetGrid(path) => {
                if let Some(store) = &self.grids {
                    store.remove(path);
                }
            }
            Effect::ReadClipboard => {
                let text = arboard::Clipboard::new().and_then(|mut c| c.get_text());
                match text {
                    Ok(text) => self.dispatch(Action::Paste(text)),
                    Err(e) => self.dispatch(Action::Failed(format!("Nothing to paste: {e}"))),
                }
            }
            Effect::SaveQueue(tracks) => {
                let result = match &self.state.queues_dir {
                    Some(dir) => {
                        let path = crate::store::session::queue_name(dir, &tracks);
                        crate::store::session::write_playlist(&path, &tracks, &[])
                            .map(|()| path)
                            .map_err(|e| e.to_string())
                    }
                    None => Err("no folder to save queues in".into()),
                };
                self.dispatch(Action::QueueSaved(result));
            }
            Effect::ReadPlaylist { path, how } => {
                let tracks = crate::store::session::read_playlist(&path).map_err(|e| e.to_string());
                self.dispatch(Action::PlaylistRead { path, tracks, how });
            }
            Effect::SaveConfig => {
                // Only what was changed in the player is saved; the command
                // line's overrides stay out of the file.
                let now = self.state.config.clone();
                let file = self.saved.merge_changes(&self.baseline, &now);
                self.saved = file.clone();
                self.baseline = now;
                if let Some(path) = self.state.config_file.clone() {
                    if let Err(error) = file.save(&path) {
                        self.dispatch(Action::Failed(format!(
                            "Could not save {}: {error}",
                            path.display()
                        )));
                    }
                }
            }
            Effect::Quit => {
                if let Some(engine) = &self.engine {
                    let _ = engine.player.stop();
                }
            }
        }
    }

    /// The search under way, if any, cancelled.
    pub(super) fn cancel_search(&mut self) {
        if let Some(cancel) = self.search.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The data folder goes into `parent`: the move written down (from now
    /// on what is written goes there), the settings saved, the files moved
    /// in the background ([`crate::store::mover`]).
    fn move_data(&mut self, parent: PathBuf) {
        if self.moving.busy() {
            return self.dispatch(Action::Failed(
                "The data folder is moving already: choose another once it is there".into(),
            ));
        }
        let Some(from) = self.state.data_dir.clone() else {
            return;
        };
        let to = crate::store::dirs::Data::in_parent(&parent).0;
        // The journal beside the settings file in use ([`crate::store::dirs::Folders`]).
        let settings = self
            .state
            .config_file
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let Some(settings) = settings else {
            return self.dispatch(Action::Failed(
                "No settings folder to note the move in".into(),
            ));
        };
        let notify = self.outbox.notify();
        let shown = &mut self.state.data_move;
        if let Err(why) = self
            .moving
            .begin(&settings, (from, to.clone()), notify, shown)
        {
            return self.dispatch(Action::Failed(why));
        }
        // The system's place is kept as none, so it follows the system.
        let default = crate::store::dirs::default_data_parent();
        self.state.config.data_folder = (default.as_ref() != Some(&parent)).then_some(parent);
        self.run(Effect::SaveConfig);
        self.state.data_dir = Some(to.clone());
        self.dispatch(Action::Info(format!(
            "The data folder is now {}",
            to.display()
        )));
    }

    /// `i`: the next microphone, kept in the settings.
    fn next_mic(&mut self) {
        let mixer = self.engine.as_ref().map(|e| &e.mixer);
        match self.mic.next(&mut self.state, mixer) {
            Ok(name) => {
                self.run(Effect::SaveConfig);
                self.mic_turn();
                self.state.message = Some(core::Message {
                    text: format!("Microphone: {name}"),
                    level: core::Level::Info,
                    until_ms: self.state.now_ms + core::MESSAGE_MS,
                });
            }
            Err(why) => self.dispatch(Action::Failed(why)),
        }
    }

    /// Tells the preview's players, made for the first file of a preview.
    fn preview(&mut self, what: core::preview::Preview) {
        use core::preview::Preview;
        let Some(engine) = &mut self.engine else {
            return;
        };
        let (song, strip) = (
            engine.song_faders(),
            engine.strip(core::desk::StripKind::Preview).clone(),
        );
        match (what, &mut engine.previews) {
            (Preview::Play { path }, previews) => previews
                .get_or_insert_with(|| {
                    crate::preview::Previews::new(song, strip, self.state.config.settings())
                })
                .play(&engine.mixer, path),
            (Preview::Silent, Some(p)) => p.silent(),
            (Preview::End, Some(p)) => p.end(),
            (_, None) => {}
        }
    }

    /// Sets a strip of the mixer as the Mixer tab has it.
    fn desk_change(&self, change: core::desk::DeskChange) {
        if let Some(engine) = &self.engine {
            engine.apply(change);
        }
    }
}
