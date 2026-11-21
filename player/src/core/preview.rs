//! Previews while Ctrl is held in the browser, as tuning a radio: the
//! song playing fades out and runs on unheard, and the file selected plays
//! from a third in ([`SHARE`]); while Ctrl stays down, every other file
//! selected (the arrows, the page keys, a click, the wheel, the pointer
//! over a row) crossfades in from its third in turn, a folder selected is
//! silence. Releasing Ctrl ends it: the song fades back in where it has
//! got to. Nothing of it reaches the queue or the song's play state: the
//! files play on players of their own ([`Effect::Preview`], the app's
//! [`crate::preview`]).
//!
//! The window tells when the key goes down and up ([`Ctrl`]; a terminal
//! does not tell the key alone, so the terminal player has no preview).
//! Ctrl held alone starts a preview after [`HOLD_MS`], so `ctrl+s`, `ctrl+l`,
//! `ctrl+v` and `ctrl+c` do what they do without one; any key or click
//! other than a move of the selection ends a preview first (`enter` then
//! queues the file, as ever).

use std::path::PathBuf;

use crate::browser::EntryKind;
use crate::core::keymap::Op;
use crate::core::update::{browser_rows, row_entry, shown_title};
use crate::core::{Action, Effect, Screen, Side, State};

/// Ctrl held alone this long starts a preview, in milliseconds: a key
/// pressed with it sooner does what it does, and no preview starts.
pub const HOLD_MS: u64 = 300;
/// Where a preview starts, as a share of the file's length (its start when
/// the length is not known).
pub const SHARE: f64 = 1.0 / 3.0;

/// What the preview's players are to do.
#[derive(Clone, Debug, PartialEq)]
pub enum Preview {
    /// Play `path` from [`SHARE`] of its length, crossfading from what is
    /// heard (the song, or the file before).
    Play {
        /// The file.
        path: PathBuf,
    },
    /// Nothing for now: the file heard fades out.
    Silent,
    /// Over: the file heard fades out and the song fades back in.
    End,
}

/// How a frontend saw the Ctrl key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctrl {
    /// The key went down: a preview starts once it has been held for
    /// [`HOLD_MS`] (or at once with a move of the selection).
    Pressed,
    /// The key went up.
    Released,
}

/// The preview's state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreviewState {
    /// Ctrl is held, as the frontend last said.
    pub ctrl: bool,
    /// When Ctrl held alone starts a preview ([`State::now_ms`] clock).
    pub due: Option<u64>,
    /// The preview, while it is on.
    pub on: Option<On>,
}

/// A preview that is on.
#[derive(Clone, Debug, PartialEq)]
pub struct On {
    /// The file heard; `None` while a row that is not a file is selected.
    pub path: Option<PathBuf>,
}

/// Whether a preview can be on now: the browser on the main screen,
/// nothing being typed.
pub fn allowed(state: &State) -> bool {
    state.screen == Screen::Main
        && state.side == Side::Browser
        && state.palette.is_none()
        && !state.browser.filter.as_ref().is_some_and(|f| f.editing)
}

/// Whether Ctrl is held for a preview: then `ctrl+↑` is `↑`, and the
/// pointer over a row selects it.
pub fn held(state: &State) -> bool {
    state.preview.ctrl && allowed(state)
}

/// Whether a preview is on.
pub fn active(state: &State) -> bool {
    state.preview.on.is_some()
}

/// The file selected in the browser, if a file is.
fn selected_file(state: &State) -> Option<PathBuf> {
    let rows = browser_rows(state);
    rows.get(state.browser.cursor)
        .and_then(|&row| row_entry(state, row))
        .filter(|e| e.kind == EntryKind::File)
        .map(|e| e.path.clone())
}

/// What the selection is, for [`moved`] to tell a move: the file selected
/// while Ctrl is held (`None` otherwise, or on a row that is not a file).
pub(crate) fn watched(state: &State) -> Option<Option<PathBuf>> {
    held(state).then(|| selected_file(state))
}

/// [`Action::Ctrl`].
pub(crate) fn ctrl(state: &mut State, how: Ctrl, fx: &mut Vec<Effect>) {
    match how {
        Ctrl::Pressed => {
            if !state.preview.ctrl && state.preview.on.is_none() && allowed(state) {
                state.preview.due = Some(state.now_ms + HOLD_MS);
            }
            state.preview.ctrl = true;
        }
        Ctrl::Released => {
            state.preview.ctrl = false;
            end(state, fx);
        }
    }
}

/// The clock: Ctrl held alone long enough starts a preview.
pub(crate) fn tick(state: &mut State, fx: &mut Vec<Effect>) {
    if state.preview.due.is_some_and(|due| due <= state.now_ms) {
        state.preview.due = None;
        if held(state) && state.preview.on.is_none() {
            start(state, fx);
        }
    }
}

/// Starts a preview of the file selected, if a file is.
fn start(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(path) = selected_file(state) else {
        return;
    };
    // The status line shows the preview.
    state.message = None;
    state.preview.on = Some(On {
        path: Some(path.clone()),
    });
    fx.push(Effect::Preview(Preview::Play { path }));
}

/// The selection moved while Ctrl is held (`before`: what [`watched`]
/// said before): the preview follows it, or starts with it.
pub(crate) fn moved(state: &mut State, before: Option<Option<PathBuf>>, fx: &mut Vec<Effect>) {
    let Some(now) = watched(state) else {
        return;
    };
    if before.as_ref() == Some(&now) {
        return;
    }
    match &mut state.preview.on {
        Some(on) => {
            if on.path != now {
                on.path = now.clone();
                state.message = None;
                fx.push(Effect::Preview(match now {
                    Some(path) => Preview::Play { path },
                    None => Preview::Silent,
                }));
            }
        }
        None => {
            state.preview.due = None;
            start(state, fx);
        }
    }
}

/// Ends the preview, if one is on: the song is heard again.
pub(crate) fn end(state: &mut State, fx: &mut Vec<Effect>) {
    state.preview.due = None;
    if state.preview.on.take().is_some() {
        fx.push(Effect::Preview(Preview::End));
    }
}

/// Before an input: any but a move of the selection ends a preview first,
/// and Ctrl held alone starts none (a key pressed with it, as `ctrl+s`).
pub(crate) fn before_input(state: &mut State, action: &Action, fx: &mut Vec<Effect>) {
    let moves = matches!(
        action,
        Action::Op(Op::Up | Op::Down | Op::PageUp | Op::PageDown | Op::Top | Op::Bottom)
            | Action::SelectRow(_)
            | Action::ClickRow(_)
            | Action::Scroll(_)
    );
    if !moves {
        end(state, fx);
    }
}

/// The status line while a preview is on.
pub fn line(state: &State) -> Option<String> {
    let on = state.preview.on.as_ref()?;
    let after = if state.playing.track.is_some() {
        "release Ctrl to go back"
    } else {
        "release Ctrl to stop"
    };
    Some(match &on.path {
        Some(path) => format!(
            "Preview: {} from a third in · {after}",
            shown_title(state, path)
        ),
        None => format!("Preview: select a file · {after}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::Entry;
    use crate::config::Config;
    use crate::core::keymap::{self, Key, KeyCode, Meaning};
    use crate::core::layout::{self, Layout};
    use crate::core::mouse::{self, MouseInput, MouseKind};
    use crate::core::update;
    use jodel_playback::{Command, PlayState};

    /// The browser on a folder (row 2) among files, a song playing from
    /// the queue.
    fn state() -> State {
        let mut s = State::new("/m".into(), Config::default(), (120, 40));
        s.browser.loading = false;
        s.browser.entries = (0..10)
            .map(|i| Entry {
                name: format!("{i:02}.flac"),
                path: format!("/m/{i:02}.flac").into(),
                kind: if i == 2 {
                    EntryKind::Dir
                } else {
                    EntryKind::File
                },
                relative: None,
            })
            .collect();
        s.queue.paths = vec!["/q/song.flac".into(), "/q/next.flac".into()];
        s.queue.current = Some(0);
        s.playing.state = PlayState::Playing;
        s
    }

    fn file(i: usize) -> PathBuf {
        format!("/m/{i:02}.flac").into()
    }

    /// The previews among `fx`.
    fn previews(fx: &[Effect]) -> Vec<Preview> {
        fx.iter()
            .filter_map(|e| match e {
                Effect::Preview(p) => Some(p.clone()),
                _ => None,
            })
            .collect()
    }

    fn play(i: usize) -> Preview {
        Preview::Play { path: file(i) }
    }

    /// Whether `fx` change the queue or what it plays.
    fn touches_the_queue(fx: &[Effect]) -> bool {
        fx.iter().any(|e| {
            matches!(
                e,
                Effect::Engine(
                    Command::Load(_)
                        | Command::Enqueue(_)
                        | Command::Remove(_)
                        | Command::Move { .. }
                        | Command::Play(_)
                        | Command::PlayIn { .. }
                        | Command::Next
                        | Command::Previous
                        | Command::Seek(_)
                        | Command::Stop
                )
            )
        })
    }

    /// Ctrl pressed at 1000 and held until the preview starts.
    fn previewing(s: &mut State) -> Vec<Effect> {
        update(s, Action::Tick(1_000));
        let mut fx = update(s, Action::Ctrl(Ctrl::Pressed));
        fx.extend(update(s, Action::Tick(1_000 + HOLD_MS)));
        fx
    }

    #[test]
    fn ctrl_held_previews_the_file_and_leaves_the_song_playing() {
        let mut s = state();
        s.browser.cursor = 1;
        update(&mut s, Action::Tick(1_000));
        let fx = update(&mut s, Action::Ctrl(Ctrl::Pressed));
        // Not at once: a key pressed with it is a chord.
        assert!(previews(&fx).is_empty());
        let fx = update(&mut s, Action::Tick(1_000 + HOLD_MS - 50));
        assert!(previews(&fx).is_empty());
        let fx = update(&mut s, Action::Tick(1_000 + HOLD_MS));
        assert_eq!(previews(&fx), [play(1)]);
        assert_eq!(SHARE, 1.0 / 3.0);
        assert_eq!(s.playing.state, PlayState::Playing);
        assert!(!touches_the_queue(&fx));
        let line = line(&s).unwrap();
        assert!(line.starts_with("Preview: 01.flac"), "{line}");
        assert!(line.contains("release Ctrl"), "{line}");
    }

    #[test]
    fn a_move_while_ctrl_is_held_previews_the_file_selected() {
        let mut s = state();
        s.browser.cursor = 0;
        previewing(&mut s);
        // ctrl+↓ is ↓ while Ctrl is held for a preview.
        let down = Key {
            code: KeyCode::Down,
            ctrl: true,
            alt: false,
            shift: false,
        };
        assert_eq!(keymap::lookup(&s, down), Some(Meaning::Op(Op::Down)));
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(previews(&fx), [play(1)]);
        // Fast moves each ask for the file they land on (the engine plays
        // only the latest one).
        let fx: Vec<Effect> = [Op::Down, Op::Down, Op::Down]
            .into_iter()
            .flat_map(|op| update(&mut s, Action::Op(op)))
            .collect();
        assert_eq!(previews(&fx), [Preview::Silent, play(3), play(4)]);
        assert_eq!(previews(&fx).last(), Some(&play(4)));
        // A click and the wheel move it too.
        let fx = update(&mut s, Action::ClickRow(6));
        assert_eq!(previews(&fx), [play(6)]);
        let fx = update(&mut s, Action::Scroll(3));
        assert_eq!(previews(&fx), [play(9)]);
        // So does the pointer over a row.
        let Layout::Main(m) = layout::compute(&s) else {
            panic!()
        };
        let over = MouseInput {
            kind: MouseKind::Moved,
            col: m.rows.x + 1,
            row: m.rows.y + 5,
        };
        let a = mouse::action(&s, over).unwrap();
        assert!(matches!(a, Action::SelectRow(5)));
        let fx = update(&mut s, a);
        assert_eq!(previews(&fx), [play(5)]);
        assert!(!touches_the_queue(&fx));
        assert_eq!(s.playing.state, PlayState::Playing);
    }

    #[test]
    fn a_folder_selected_is_silence() {
        let mut s = state();
        s.browser.cursor = 1;
        previewing(&mut s);
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(previews(&fx), [Preview::Silent]);
        assert_eq!(s.preview.on.as_ref().unwrap().path, None);
        assert!(line(&s).unwrap().contains("select a file"));
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(previews(&fx), [play(3)]);
    }

    #[test]
    fn ctrl_on_a_folder_previews_once_a_file_is_selected() {
        let mut s = state();
        s.browser.cursor = 2;
        let fx = previewing(&mut s);
        assert!(previews(&fx).is_empty());
        assert_eq!(s.playing.state, PlayState::Playing);
        let fx = update(&mut s, Action::Op(Op::Down));
        assert_eq!(previews(&fx), [play(3)]);
    }

    #[test]
    fn releasing_ctrl_goes_back_and_the_queue_is_as_it_was() {
        let mut s = state();
        s.browser.cursor = 0;
        let queue = s.queue.paths.clone();
        let mut fx = previewing(&mut s);
        fx.extend(update(&mut s, Action::Op(Op::Down)));
        fx.extend(update(&mut s, Action::Op(Op::PageDown)));
        let back = update(&mut s, Action::Ctrl(Ctrl::Released));
        assert_eq!(previews(&back), [Preview::End]);
        fx.extend(back);
        assert!(!touches_the_queue(&fx));
        assert_eq!(s.queue.paths, queue);
        assert_eq!(s.queue.current, Some(0));
        assert_eq!(s.playing.state, PlayState::Playing);
        assert!(line(&s).is_none());
        // Moves after it preview nothing.
        let fx = update(&mut s, Action::Op(Op::Down));
        assert!(previews(&fx).is_empty());
    }

    #[test]
    fn a_paused_song_stays_paused() {
        let mut s = state();
        s.playing.state = PlayState::Paused;
        s.browser.cursor = 0;
        let mut fx = previewing(&mut s);
        assert_eq!(s.playing.state, PlayState::Paused);
        fx.extend(update(&mut s, Action::Ctrl(Ctrl::Released)));
        assert_eq!(s.playing.state, PlayState::Paused);
        assert!(!fx
            .iter()
            .any(|e| matches!(e, Effect::Engine(Command::Pause | Command::Resume))));
    }

    #[test]
    fn ctrl_with_another_key_starts_no_preview() {
        let mut s = state();
        s.browser.cursor = 0;
        update(&mut s, Action::Tick(1_000));
        update(&mut s, Action::Ctrl(Ctrl::Pressed));
        // ctrl+s: the queue saved, as ever.
        let ctrl_s = Key::ctrl('s');
        assert_eq!(keymap::lookup(&s, ctrl_s), Some(Meaning::Op(Op::QueueSave)));
        update(&mut s, Action::Op(Op::QueueSave));
        let fx = update(&mut s, Action::Tick(1_000 + 2 * HOLD_MS));
        assert!(previews(&fx).is_empty());
        assert!(!active(&s));
    }

    #[test]
    fn enter_while_previewing_queues_the_file_and_ends_the_preview() {
        let mut s = state();
        s.browser.cursor = 0;
        previewing(&mut s);
        update(&mut s, Action::Op(Op::Down));
        let enter = Key {
            code: KeyCode::Enter,
            ctrl: true,
            alt: false,
            shift: false,
        };
        assert_eq!(keymap::lookup(&s, enter), Some(Meaning::Op(Op::Open)));
        let fx = update(&mut s, Action::Op(Op::Open));
        assert_eq!(previews(&fx), [Preview::End]);
        assert!(fx
            .iter()
            .any(|e| *e == Effect::Engine(Command::Enqueue(vec![file(1)]))));
        assert!(!active(&s));
        assert_eq!(s.playing.state, PlayState::Playing);
    }

    #[test]
    fn no_preview_outside_the_browser() {
        let mut s = state();
        s.side = Side::Queue;
        let fx = previewing(&mut s);
        assert!(previews(&fx).is_empty());
        let fx = update(&mut s, Action::Op(Op::Down));
        assert!(previews(&fx).is_empty());
    }

    #[test]
    fn another_command_ends_the_preview_first() {
        let mut s = state();
        s.browser.cursor = 0;
        previewing(&mut s);
        let fx = update(&mut s, Action::Op(Op::Next));
        let first = fx
            .iter()
            .position(|e| matches!(e, Effect::Preview(Preview::End)));
        assert_eq!(first, Some(0), "{fx:?}");
        assert!(!active(&s));
    }
}
