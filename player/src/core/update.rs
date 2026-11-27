//! `update(state, action) -> effects`: the UI's logic, without IO.
//!
//! Here: [`update`] itself, the ops of the keys and the palette
//! ([`apply_op`]) and what the engine reports. The rest by concern:
//! [`super::browse`] (the browser side and the list's cursor),
//! [`super::queue`], [`super::settings`], [`super::work`] (the background
//! work on the files) and [`super::help`] (the help screen and the
//! palette); the tabs and Karaoke in their own modules.

use std::sync::Arc;

use jodel_playback::{Command, Event, PlayState};

use crate::core::browse::{self, fix_scroll, list_cursor, list_height, set_cursor};
use crate::core::desk;
use crate::core::karaoke;
use crate::core::keymap::Op;
use crate::core::lanes;
use crate::core::preview;
use crate::core::timeview::PanOf;
use crate::core::transition::{draft, jump, tab, viewport};
use crate::core::{
    help, queue, settings, work, Action, Effect, Gpu, NowTrack, Placing, Screen, Side, State,
    SEEK_STEP,
};

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
        Action::Released(op) => released(state, op, &mut fx),
        Action::Op(op) => apply_op(state, op, &mut fx),
        Action::Type(c) => {
            if !help::typed(state, c.encode_utf8(&mut [0; 4]))
                && !super::shader_tab::typed(state, c)
            {
                browse::typed(state, c);
            }
        }
        Action::Paste(text) => paste(state, &text),
        Action::TransitionPoint {
            block,
            time,
            extend,
        } => {
            if viewport::point(state, block, time, extend) {
                apply_op(state, Op::Accept, &mut fx);
            }
        }
        Action::TransitionJump { block, time } => match block {
            lanes::Block::Now => viewport::seek(state, &mut fx, time),
            _ => viewport::jump(state, block, time),
        },
        Action::TransitionWheel {
            block,
            steps,
            scroll,
            at,
        } => {
            if scroll {
                viewport::scroll(state, block, f64::from(steps) / 10.0);
            } else {
                viewport::wheel_zoom(state, block, steps.signum() as isize, at);
            }
        }
        Action::Pan { of, input } => match of {
            PanOf::Block { block, overview } => viewport::pan(state, block, overview, input),
            PanOf::Lyrics => super::lyric_edit::pan(state, input),
        },
        Action::KaraokeSlide { slider, share } => {
            state.karaoke.dragging = Some(slider);
            karaoke::slide_to(state, slider, share, &mut fx);
        }
        Action::KaraokeSlideEnd => state.karaoke.dragging = None,
        Action::KaraokeStep { slider, by } => karaoke::step(state, slider, by, &mut fx),
        Action::Sync(event) => super::sync::event(state, event),
        Action::Desk(what) => desk::act(state, what, &mut fx),
        Action::Synth(what) => super::synth_tab::act(state, what, &mut fx),
        Action::Shader(what) => super::shader_tab::act(state, what, &mut fx),
        Action::PointerMoved { from_bottom } => {
            karaoke::controls::pointer_moved(state, from_bottom);
        }
        Action::PointerLeft => karaoke::controls::pointer_left(state),
        Action::SelectRow(row) => {
            state.screen = Screen::Main;
            select(state, row);
        }
        Action::ClickRow(row) => browse::click_row(state, row, &mut fx),
        Action::EnqueueRow(row) => {
            select(state, row);
            queue::enqueue(state, &mut fx);
        }
        Action::QueueDrag(row) => queue::drag(state, row, &mut fx),
        Action::QueueDragEnd => state.queue.dragging = false,
        Action::Scroll(delta) => match state.screen {
            Screen::Help => help::scroll(state, delta),
            _ => select_by(state, delta),
        },
        Action::ShowSide(side) => {
            state.side = side;
            state.screen = Screen::Main;
            fix_scroll(state, side);
        }
        Action::SeekTo(fraction) => {
            state.seek_drag = None;
            if karaoke::seek_refused(state) {
                // Held for Next or counting in: nothing to seek yet.
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
        Action::OpenSettings(row) => settings::open_at(state, row, &mut fx),
        Action::SettingsClick(row, step) => settings::click(state, row, step, &mut fx),
        Action::ChooseDevice(index) => settings::choose_device_at(state, index, &mut fx),
        Action::FolderRow(index) => settings::folder_row(state, index, &mut fx),
        Action::FoldersListed(dir, folders) => settings::folders_listed(state, dir, folders),
        Action::PaletteRun(index) => help::run_palette_at(state, index, &mut fx),
        Action::HelpScroll(delta) => help::scroll(state, delta),
        Action::Fit(fit) => {
            state.fit = fit;
            fix_scroll(state, state.side);
        }
        Action::Tick(now) => {
            state.now_ms = now;
            viewport::tick(state);
            karaoke::tick(state, &mut fx);
            preview::tick(state, &mut fx);
            if state.message.as_ref().is_some_and(|m| m.until_ms <= now) {
                state.message = None;
            }
            if state.settings_due.is_some_and(|due| due <= now) {
                settings::send(state, &mut fx);
            }
            check_details = false;
        }
        Action::Listed { dir, result } => {
            if dir == state.browser.dir {
                browse::listed(state, result, &mut fx);
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
        Action::SearchHits { id, hits } => browse::search_hits(state, id, hits),
        Action::SearchDone { id } => browse::search_done(state, id),
        Action::Walked { purpose, paths } => queue::walked(state, purpose, paths, &mut fx),
        Action::JumpMissed { text } => browse::jump_missed(state, &text, &mut fx),
        Action::JumpTo { dir, select } => browse::jump_to(state, dir, select, &mut fx),
        Action::Devices(result) => settings::devices(state, result),
        Action::Engine(event) => engine_event(state, event),
        Action::Rhythm(path, recipe, result) => work::rhythm(state, path, recipe, result),
        Action::Gpu(name) => {
            state.gpu = match name {
                Ok(name) => Gpu::Present(name),
                Err(why) => Gpu::Absent(why),
            };
        }
        Action::Lyrics(path, answer) => karaoke::lyrics_arrived(state, path, answer),
        Action::LyricsEdit(what) => super::lyric_edit::act(state, what, &mut fx),
        Action::Working { path, job, on } => {
            super::prep::working(state, &path, on);
            work::working(state, path.clone(), job, on);
            super::prep::follow(state, &path);
        }
        Action::Heard { path, part } => {
            super::prep::heard(state, &path, &part);
            work::heard(state, path.clone(), &part);
            super::prep::follow(state, &path);
        }
        Action::Ready { path, job, facts } => {
            work::ready(state, path.clone(), job, facts);
            super::prep::follow(state, &path);
        }
        Action::Failed(text) => state.error(text),
        Action::Info(text) => state.info(text),
        Action::QueueSaved(result) => queue::saved(state, result),
        Action::PlaylistRead { path, tracks, how } => {
            queue::playlist_read(state, &path, tracks, how, &mut fx);
        }
        Action::EngineGone(_) => state.error("The playback engine failed and was started again"),
    }
    // While Ctrl is held, the preview follows the selection as it is
    // moved (not as a folder's listing lands).
    if input && state.preview.ctrl {
        preview::moved(state, selected, &mut fx);
    }
    work::want_rhythm(state, &mut fx);
    karaoke::want_lyrics(state, &mut fx);
    super::stems::follow(state, &mut fx);
    super::sync::follow(state, &mut fx);
    desk::settle(state, &mut fx);
    super::synth_tab::settle(state, &mut fx);
    super::shader_tab::settle(state);
    work::prepare(state, &mut fx);
    work::want_stories(state, &mut fx);
    karaoke::start_when_ready(state, &mut fx);
    if check_details {
        work::want_details(state, &mut fx);
    }
    fx
}

/// Whether `action` is the user's own key, click or scroll (not the
/// clock or a background result). Every action is named, so a new one is
/// put on one side or the other.
fn is_input(action: &Action) -> bool {
    match action {
        Action::Op(_)
        | Action::Type(_)
        | Action::Paste(_)
        | Action::SelectRow(_)
        | Action::ClickRow(_)
        | Action::EnqueueRow(_)
        | Action::QueueDrag(_)
        | Action::QueueDragEnd
        | Action::Scroll(_)
        | Action::ShowSide(_)
        | Action::TransitionPoint { .. }
        | Action::TransitionJump { .. }
        | Action::TransitionWheel { .. }
        | Action::Pan { .. }
        | Action::KaraokeSlide { .. }
        | Action::KaraokeSlideEnd
        | Action::KaraokeStep { .. }
        | Action::Desk(_)
        | Action::Synth(_)
        | Action::Shader(_)
        | Action::LyricsEdit(_)
        | Action::SeekTo(_)
        | Action::SeekDrag(_)
        | Action::OpenSettings(_)
        | Action::SettingsClick(..)
        | Action::ChooseDevice(_)
        | Action::FolderRow(_)
        | Action::PaletteRun(_)
        | Action::HelpScroll(_) => true,
        // Ctrl alone is the preview's own; a key let go of, the pointer
        // moving and the sync's clicks heard press nothing.
        Action::Ctrl(_)
        | Action::Released(_)
        | Action::PointerMoved { .. }
        | Action::PointerLeft
        | Action::Sync(_)
        | Action::Fit(_)
        | Action::Tick(_)
        | Action::Listed { .. }
        | Action::Details(..)
        | Action::DetailsDropped(_)
        | Action::SearchHits { .. }
        | Action::SearchDone { .. }
        | Action::Walked { .. }
        | Action::JumpMissed { .. }
        | Action::JumpTo { .. }
        | Action::FoldersListed(..)
        | Action::Devices(_)
        | Action::Engine(_)
        | Action::EngineGone(_)
        | Action::Rhythm(..)
        | Action::Gpu(_)
        | Action::Lyrics(..)
        | Action::Working { .. }
        | Action::Ready { .. }
        | Action::Heard { .. }
        | Action::QueueSaved(_)
        | Action::PlaylistRead { .. }
        | Action::Failed(_)
        | Action::Info(_) => false,
    }
}

/// `value` moved `delta` on, stopping at 0.
pub(crate) fn offset(value: usize, delta: i32) -> usize {
    if delta < 0 {
        value.saturating_sub(delta.unsigned_abs() as usize)
    } else {
        value.saturating_add(delta as usize)
    }
}

/// The shown side's cursor on `row` ([`set_cursor`]).
fn select(state: &mut State, row: usize) {
    let side = state.side;
    set_cursor(state, side, row);
}

/// The shown side's cursor `delta` rows on.
fn select_by(state: &mut State, delta: i32) {
    let side = state.side;
    let (cursor, _) = list_cursor(state, side);
    set_cursor(state, side, offset(cursor, delta));
}

/// `↑` `↓` and the pages, `delta` rows: in the palette while it is open,
/// else on the screen shown.
fn move_by(state: &mut State, delta: i32) {
    if help::palette_moved(state, delta) {
        return;
    }
    match state.screen {
        Screen::Main => select_by(state, delta),
        Screen::Settings => settings::move_by(state, delta),
        Screen::Help => help::scroll(state, delta),
        Screen::NowPlaying | Screen::Karaoke | Screen::Lyrics(_) => {}
    }
}

fn page(state: &State) -> i32 {
    match state.screen {
        // Two lines of the last page stay in view.
        Screen::Help => i32::from(state.fit.help_rows.saturating_sub(2)).max(1),
        _ => (list_height(state, state.side) as i32 - 1).max(1),
    }
}

/// Gives `op` to the screen or tab the keys are on (none under a palette);
/// whether it took it.
fn shown_takes(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if state.palette.is_some() {
        return false;
    }
    match state.screen {
        Screen::Lyrics(_) => super::lyric_edit::screen::apply(state, op, fx),
        Screen::Karaoke => karaoke::apply_mode(state, op, fx),
        Screen::Main => match state.side {
            Side::Transition => tab::apply(state, op, fx),
            Side::Karaoke => karaoke::apply(state, op, fx),
            Side::Mixer => desk::apply(state, op, fx),
            Side::Synth => super::synth_tab::apply(state, op, fx),
            Side::Shader => super::shader_tab::apply(state, op, fx),
            Side::Browser | Side::Queue => false,
        },
        Screen::NowPlaying | Screen::Settings | Screen::Help => false,
    }
}

/// Does what `op` means where it is pressed (or run from the palette): the
/// screen or tab shown first; an operation of another's ([`Op::home`])
/// says where it is done; the rest as anywhere.
pub(crate) fn apply_op(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    if shown_takes(state, op, fx) {
        return;
    }
    match op {
        Op::PlayPause => match state.playing.state {
            PlayState::Playing => fx.push(Effect::Engine(Command::Pause)),
            PlayState::Paused => fx.push(Effect::Engine(Command::Resume)),
            PlayState::Stopped => queue::play_stopped(state, fx),
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
        Op::Skip => draft::skip(state, fx),
        Op::SkipMix => settings::cycle_skip_mix(state, fx),
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
            fix_scroll(state, state.side);
        }
        Op::GoTrack | Op::GoAlbum | Op::GoArtist => match browse::playing_jump(state, op) {
            Some((dir, select)) => browse::jump_to(state, dir, select, fx),
            None => state.info("Nothing playing"),
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
                settings::open(state, fx);
            }
        }
        Op::Help => help::toggle(state),
        Op::Palette => help::open_palette(state),
        Op::Filter if state.screen == Screen::Help => help::search(state),
        Op::Filter => browse::open_filter(state, fx),
        Op::Quit => {
            state.quit = true;
            if state.settings_due.is_some() {
                settings::send(state, fx);
            }
            fx.push(Effect::Quit);
        }
        Op::ToggleDrums | Op::ToggleBass | Op::ToggleVocals | Op::ToggleOther => {
            if let Some(stem) = op.stem() {
                super::stems::toggle(state, stem, fx);
            }
        }
        Op::AllStems => super::stems::reset(state, fx),
        Op::JumpBar(offset) => jump::jump_bar(state, isize::from(offset), false, fx),
        Op::JumpBarAtEnd(offset) => jump::jump_bar(state, isize::from(offset), true, fx),
        Op::VolumeUp | Op::VolumeDown => {
            let by = if op == Op::VolumeUp { 1.0 } else { -1.0 };
            let db = super::desk::taper::step(state.desk.volume, by);
            super::desk::set_volume(state, db, fx);
            let volume = super::desk::taper::label(state.desk.volume);
            state.info(format!("Volume {volume}"));
        }
        Op::ToggleBpm => settings::toggle_bpm(state, fx),
        Op::Back => back(state, fx),
        Op::Up => move_by(state, -1),
        Op::Down => move_by(state, 1),
        Op::PageUp => {
            let p = page(state);
            move_by(state, -p);
        }
        Op::PageDown => {
            let p = page(state);
            move_by(state, p);
        }
        Op::Top => select(state, 0),
        Op::Bottom => select(state, usize::MAX),
        Op::Open if state.screen == Screen::Settings => settings::enter(state, fx),
        Op::Open => queue::activate(state, Placing::Append, fx),
        Op::Parent if state.screen == Screen::Settings && state.settings.folders.is_some() => {
            settings::folder_parent(state, fx)
        }
        Op::Parent => {
            if state.side == Side::Browser {
                browse::go_parent(state, fx);
            }
        }
        Op::QueuePlay => queue::activate(state, Placing::AppendPlay, fx),
        Op::OpenReplace => queue::activate(state, Placing::Replace, fx),
        Op::QueueRemove => queue::remove(state, fx),
        Op::QueueSave => queue::save(state, fx),
        Op::QueueOpenSaved => queue::open_saved(state, fx),
        Op::QueueMoveUp => queue::move_selected(state, -1, fx),
        Op::QueueMoveDown => queue::move_selected(state, 1, fx),
        Op::Accept => {
            if !help::accepted(state, fx) {
                browse::accept_filter(state, fx);
            }
        }
        Op::Erase => {
            if !help::erased(state) {
                browse::erase_filter(state);
            }
        }
        Op::Paste => {
            if help::can_paste(state) {
                fx.push(Effect::ReadClipboard);
            }
        }
        Op::KaraokeEnter | Op::KaraokeEnterHere => {
            // From anywhere (the palette): the tab first, then the mode.
            state.palette = None;
            state.side = Side::Karaoke;
            state.screen = Screen::Main;
            karaoke::enter(state, fx, op == Op::KaraokeEnterHere);
        }
        Op::KaraokeSync => {
            // From anywhere (the palette): the tab first, then the sync.
            state.palette = None;
            state.side = Side::Karaoke;
            state.screen = Screen::Main;
            super::sync::begin(state, fx);
        }
        Op::KeyDown | Op::KeyUp | Op::KeyReset => super::key::step(state, op, fx),
        Op::MicNext => fx.push(Effect::NextMic),
        Op::DeskEcho => desk::toggle_echo(state, fx),
        Op::ValueDown => settings::step_row(state, -1, fx),
        Op::ValueUp => settings::step_row(state, 1, fx),
        Op::LyricsEditor => {
            // From anywhere (the palette): the song karaoke is about.
            state.palette = None;
            super::lyric_edit::open(state);
        }
        // Another tab's or screen's own, from elsewhere: where it is done.
        _ => state.info(format!("That is for: {}", op.home().title())),
    }
}

/// The key of a held row's `op` let go of: what it holds ends, where it
/// is going on (the demo singing, a key of the Synth tab's keyboard
/// holding its note); nothing where it is not.
fn released(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    match op {
        Op::KaraokeDemo => state.karaoke.demo.let_go(),
        Op::SynthNote(k) => super::synth_tab::key_up(state, k, fx),
        _ => {}
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

/// Puts pasted `text` into the palette, the help's search being typed or
/// the browser's filter ([`browse::paste`]).
fn paste(state: &mut State, text: &str) {
    let text = one_line(text);
    if !text.is_empty() && !help::typed(state, &text) {
        browse::paste(state, &text);
    }
}

/// `esc`: the palette closes, else the screen shown steps back.
fn back(state: &mut State, fx: &mut Vec<Effect>) {
    if state.palette.take().is_some() {
        return;
    }
    match state.screen {
        Screen::Main => browse::back(state, fx),
        Screen::Settings => settings::back(state),
        Screen::Help => help::back(state),
        Screen::NowPlaying | Screen::Karaoke | Screen::Lyrics(_) => state.screen = Screen::Main,
    }
}

fn engine_event(state: &mut State, event: Event) {
    let playing = &mut state.playing;
    match event {
        Event::QueueChanged { queue, current } => queue::engine_changed(state, queue, current),
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
            draft::track_started(state, start.index);
            karaoke::track_started(state);
            // Another track has stems of its own (or none) to find out.
            state.desk.parts_failed.retain(|&i| i == start.index);
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
            });
            playing.position = at;
            playing.clipped = 0;
            playing.forget_anchor();
        }
        Event::Position(position) => {
            playing.told(&position);
            super::key::heard(state, position.key);
        }
        Event::Seeked { index, seconds } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.position = seconds;
                playing.clipped = 0;
                playing.forget_anchor();
            }
        }
        Event::Jumped { index, seconds } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.position = seconds;
                playing.forget_anchor();
            }
        }
        Event::Clipped { index, samples, .. } => {
            if playing.track.as_ref().is_some_and(|t| t.index == index) {
                playing.clipped = samples;
            }
        }
        Event::TrackEnded {
            index,
            finished: true,
        } => karaoke::ended(state, index),
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
            state.error(format!("{name}{error}{tail}"));
        }
        Event::Notice { message, .. } => state.info(message),
        Event::Transition(t) => draft::transition_event(state, t),
        // The stems as parts of the output stopped: the file plays on.
        Event::PartsStopped { index, reason } => super::stems::stopped(state, index, &reason),
        Event::QueueEnded => state.info("End of the queue"),
        Event::Held { index } => karaoke::held(state, index),
        Event::Key { semitones } => super::key::told(state, semitones),
    }
}
