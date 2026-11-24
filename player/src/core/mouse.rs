//! Mouse input: the named rects of the layout and what a click, drag or
//! wheel on each does. The window reports the pointer in cells (and
//! within its cell, [`action_at`]).

use ratatui::layout::{Position, Rect};

use crate::core::karaoke;
use crate::core::keymap::Op;
use crate::core::layout::{self, Bottom, Layout};
use crate::core::sliders::Slider;
use crate::core::text::{self, Tone};
use crate::core::update::{device_window, folder_window, list_cursor, list_len, palette_window};
use crate::core::{Action, Screen, SettingsRow, Side, State};

/// A mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    /// The main button.
    Left,
    /// The secondary button.
    Right,
    /// The wheel button.
    Middle,
}

/// What the mouse did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseKind {
    /// A button went down.
    Down(MouseButton),
    /// A button went up.
    Up(MouseButton),
    /// The pointer moved with a button held.
    Drag(MouseButton),
    /// The pointer moved.
    Moved,
    /// The wheel turned up.
    ScrollUp,
    /// The wheel turned down.
    ScrollDown,
    /// The pointer went out of the window.
    Out,
}

/// A mouse event at a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MouseInput {
    /// What happened.
    pub kind: MouseKind,
    /// The cell column.
    pub col: u16,
    /// The cell row.
    pub row: u16,
}

/// A named place on the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// The Browser or Queue half of the toggle.
    Toggle(Side),
    /// A row of the list (its index in the list, not on screen).
    Row(usize),
    /// The cover.
    Cover,
    /// The track text (on Now playing, a click returns).
    Info,
    /// The device name in the signal path.
    Device,
    /// The progress bar.
    Bar,
    /// A button.
    Button(Op),
    /// A settings row's `‹`.
    SettingsLess(usize),
    /// A settings row's value.
    SettingsValue(usize),
    /// A settings row's `›`.
    SettingsMore(usize),
    /// A device in the open list.
    DeviceRow(usize),
    /// A row of the data folder's list.
    FolderRow(usize),
    /// A palette match.
    PaletteRow(usize),
    /// The help text.
    HelpText,
    /// The list's rows area (for the wheel).
    ListArea,
    /// The title, artist or album of the playing track: where it goes.
    Link(Op),
    /// A Transition block's lanes (or its ruler).
    TransitionLane(crate::core::lanes::Block),
    /// A Transition block's whole-track row.
    TransitionOverview(crate::core::lanes::Block),
    /// A Karaoke slider's row.
    KaraokeSlider(Slider),
}

fn tone_rects(lines: &[Vec<text::Seg>], tone: Tone, area: Rect) -> Vec<Rect> {
    text::locate(lines, tone)
        .into_iter()
        .filter(|(row, _, _)| *row < usize::from(area.height))
        .map(|(row, x, w)| Rect::new(area.x + x as u16, area.y + row as u16, w as u16, 1))
        .map(|r| r.intersection(area))
        .collect()
}

fn info_device_rects(state: &State, info: Rect) -> Vec<Rect> {
    let lines = text::info_lines(state, usize::from(info.width));
    tone_rects(&lines, Tone::Device, info)
}

/// Where the title, artist and album links are in `lines` drawn in
/// `area`, and where each goes.
pub fn link_rects(lines: &[Vec<text::Seg>], area: Rect) -> Vec<(Rect, Op)> {
    text::LINKS
        .iter()
        .flat_map(|&(tone, op)| {
            tone_rects(lines, tone, area)
                .into_iter()
                .map(move |r| (r, op))
        })
        .collect()
}

fn bottom_regions(bottom: &Bottom, out: &mut Vec<(Rect, Target)>) {
    for button in &bottom.buttons {
        out.push((button.rect, Target::Button(button.op)));
    }
    out.push((bottom.bar, Target::Bar));
}

/// Every hit region of the screen shown, most specific first.
pub fn regions(state: &State) -> Vec<(Rect, Target)> {
    let mut out = Vec::new();
    match layout::compute(state) {
        Layout::TooSmall(_) => {}
        Layout::Main(m) => {
            for (rect, side) in &m.toggle {
                out.push((*rect, Target::Toggle(*side)));
            }
            if let Some(info) = m.info {
                for rect in info_device_rects(state, info) {
                    out.push((rect, Target::Device));
                }
                let lines = text::info_lines(state, usize::from(info.width));
                for (rect, op) in link_rects(&lines, info) {
                    out.push((rect, Target::Link(op)));
                }
                out.push((info, Target::Info));
            }
            if let Some(cover) = m.cover {
                out.push((cover, Target::Cover));
            }
            if let Some(mini) = m.mini {
                out.push((mini, Target::Cover));
            }
            if let (Some(k), None) = (&m.karaoke, &state.palette) {
                for &(rect, op) in &k.panel_buttons {
                    out.push((rect, Target::Button(op)));
                }
                out.push((k.enter, Target::Button(Op::KaraokeEnter)));
                for s in &k.sliders {
                    out.push((s.row, Target::KaraokeSlider(s.slider)));
                }
                out.push((k.game, Target::Button(Op::KaraokeGame)));
                out.push((k.show_notes, Target::Button(Op::KaraokeNotes)));
                out.push((k.keep, Target::Button(Op::KaraokeKeep)));
                out.push((k.sync, Target::Button(Op::KaraokeSync)));
            }
            if let (Some(tab), None) = (&m.transition, &state.palette) {
                if let Some(fit) = tab.fit {
                    out.push((fit, Target::Button(Op::TransitionFitLyrics)));
                }
                // The block sliding away after a hand-over takes no clicks.
                for b in tab
                    .blocks
                    .iter()
                    .filter(|b| b.block != crate::core::lanes::Block::Prev)
                {
                    out.push((b.overview, Target::TransitionOverview(b.block)));
                    out.push((b.ruler, Target::TransitionLane(b.block)));
                    for (_, _, lane) in &b.lanes {
                        out.push((*lane, Target::TransitionLane(b.block)));
                    }
                }
            }
            if state.palette.is_some() {
                let height = usize::from(m.rows.height);
                let (matches, start) = palette_window(state, height);
                for (i, index) in (start..matches.len()).enumerate() {
                    let y = m.rows.y + 1 + i as u16;
                    if y >= m.rows.bottom() {
                        break;
                    }
                    out.push((
                        Rect::new(m.rows.x, y, m.rows.width, 1),
                        Target::PaletteRow(index),
                    ));
                }
            } else {
                let (_, scroll) = list_cursor(state);
                let len = list_len(state);
                for i in 0..usize::from(m.rows.height) {
                    let index = scroll + i;
                    if index >= len {
                        break;
                    }
                    let rect = Rect::new(m.rows.x, m.rows.y + i as u16, m.rows.width, 1);
                    out.push((rect, Target::Row(index)));
                }
            }
            out.push((m.rows, Target::ListArea));
            bottom_regions(&m.bottom, &mut out);
        }
        Layout::NowPlaying(n) => {
            let lines = text::caption_lines(state, usize::from(n.caption.width));
            for (rect, op) in link_rects(&lines, n.caption) {
                out.push((rect, Target::Link(op)));
            }
            out.push((n.screen, Target::Cover));
        }
        Layout::Settings(s) => {
            if let (Some(list), Some(pick)) = (s.devices, &state.settings.folders) {
                let start = folder_window(state, usize::from(list.height));
                let n = pick.rows().len();
                for i in 0..usize::from(list.height) {
                    if start + i >= n {
                        break;
                    }
                    let rect = Rect::new(list.x, list.y + i as u16, list.width, 1);
                    out.push((rect, Target::FolderRow(start + i)));
                }
            } else if let Some(devices) = s.devices {
                let start = device_window(state, usize::from(devices.height));
                let n = state.settings.devices.as_ref().map_or(0, Vec::len);
                for i in 0..usize::from(devices.height) {
                    if start + i >= n {
                        break;
                    }
                    let rect = Rect::new(devices.x, devices.y + i as u16, devices.width, 1);
                    out.push((rect, Target::DeviceRow(start + i)));
                }
            }
            for (i, line) in s.lines.iter().enumerate() {
                out.push((line.less, Target::SettingsLess(i)));
                out.push((line.more, Target::SettingsMore(i)));
                out.push((line.rect, Target::SettingsValue(i)));
            }
            bottom_regions(&s.bottom, &mut out);
        }
        Layout::Help(h) => {
            out.push((h.text, Target::HelpText));
            bottom_regions(&h.bottom, &mut out);
        }
        Layout::Karaoke(k) => {
            if let Some(c) = &k.controls {
                for s in &c.sliders {
                    out.push((s.row, Target::KaraokeSlider(s.slider)));
                }
                out.push((c.seek, Target::Bar));
            }
        }
    }
    out
}

/// The target under a cell.
pub fn hit(state: &State, col: u16, row: u16) -> Option<Target> {
    let at = Position::new(col, row);
    regions(state)
        .into_iter()
        .find(|(rect, _)| rect.contains(at))
        .map(|(_, target)| target)
}

fn bar_fraction(state: &State, col: u16) -> Option<f64> {
    let bar = match layout::compute(state) {
        Layout::Karaoke(k) => k.controls?.seek,
        other => other.bottom()?.bar,
    };
    if bar.width < 2 {
        return None;
    }
    let x = col.clamp(bar.x, bar.right() - 1) - bar.x;
    Some(f64::from(x) / f64::from(bar.width - 1))
}

/// The seconds into a Transition block's track at column `col`: of the span the
/// lanes show, or of the whole track for its overview.
fn transition_time(
    state: &State,
    block: crate::core::lanes::Block,
    col: u16,
    overview: bool,
) -> Option<f64> {
    use crate::core::lanes;
    use crate::core::transition::viewport;
    let Layout::Main(m) = layout::compute(state) else {
        return None;
    };
    let b = m
        .transition?
        .blocks
        .into_iter()
        .find(|b| b.block == block)?;
    // The lanes and the overview share their columns (the overview may be
    // out of view while a block slides).
    let area = b.lanes.first().map_or(b.overview, |l| l.2);
    if area.width == 0 {
        return None;
    }
    let share =
        (f64::from(col.clamp(area.x, area.right() - 1) - area.x) + 0.5) / f64::from(area.width);
    let (start, end) = if overview {
        let r = lanes::rhythmed(state, block)?;
        let duration = r
            .waveform
            .as_ref()
            .map_or(0.0, |w| w.duration())
            .max(r.analysis.duration);
        (0.0, duration)
    } else {
        viewport::window(state, block, state.heard())
    };
    Some(start + share * (end - start))
}

/// The width of a Transition block's waveforms, in columns.
fn transition_width(state: &State, block: crate::core::lanes::Block) -> Option<u16> {
    let Layout::Main(m) = layout::compute(state) else {
        return None;
    };
    let b = m
        .transition?
        .blocks
        .into_iter()
        .find(|b| b.block == block)?;
    let width = b.lanes.first().map_or(b.overview.width, |l| l.2.width);
    (width > 0).then_some(width)
}

/// The share of `slider`'s range with the pointer at `x` cells from the
/// left edge. The middle of the track's first cell is 0 and of its last
/// cell 1, where the knob is drawn for those shares.
fn slider_share(state: &State, slider: Slider, x: f64) -> Option<f64> {
    let sliders = match layout::compute(state) {
        Layout::Main(m) => m.karaoke?.sliders,
        Layout::Karaoke(k) => k.controls?.sliders,
        _ => return None,
    };
    let track = sliders.into_iter().find(|s| s.slider == slider)?.track;
    layout::Track::columns(track).share_at(x)
}

/// What a mouse event does, if anything. The pointer is taken to be in
/// the middle of its cell, over the cells alone.
pub fn action(state: &State, input: MouseInput) -> Option<Action> {
    action_at(state, input, f64::from(input.col) + 0.5, None)
}

/// [`action`] with the pointer known within its cell and over what the
/// window draws in pixels: `x` is in cells from the left edge (the GUI
/// knows it to the pixel), and `over` the target drawn in pixels under the
/// pointer, if any (Karaoke mode's START pill, on the stage), which is hit
/// before the cells under it. Only a Karaoke slider uses `x`: cell by cell,
/// a 40-cell track moves 2.6 % a step (5.1 % for the microphone's 200 %);
/// pixel by pixel, every whole percent is in reach.
pub fn action_at(state: &State, input: MouseInput, x: f64, over: Option<Target>) -> Option<Action> {
    use MouseButton::{Left, Right};
    use MouseKind::*;
    // A drag on the progress bar follows the pointer anywhere.
    if state.seek_drag.is_some() {
        return match input.kind {
            Drag(Left) | Moved => bar_fraction(state, input.col).map(Action::SeekDrag),
            Up(Left) => bar_fraction(state, input.col).map(Action::SeekTo),
            _ => None,
        };
    }
    // So does a drag of a Transition block's view.
    if state.transition.pan.is_some() {
        return match input.kind {
            Drag(Right) | Moved => Some(Action::TransitionPan(input.col)),
            Up(Right) => Some(Action::TransitionPanEnd),
            _ => None,
        };
    }
    // And a Karaoke slider held.
    if state.karaoke.dragging.is_some() {
        return match input.kind {
            Drag(Left) | Moved => state
                .karaoke
                .dragging
                .and_then(|s| slider_share(state, s, x).map(|share| (s, share)))
                .map(|(slider, share)| Action::KaraokeSlide { slider, share }),
            Up(Left) => Some(Action::KaraokeSlideEnd),
            _ => None,
        };
    }
    // While a sync shows, only its panel's buttons take a click.
    if state.karaoke.sync.is_some() {
        return match (input.kind, hit(state, input.col, input.row)) {
            (Down(Left), Some(Target::Button(op @ (Op::SyncKeep | Op::SyncAgain | Op::Back)))) => {
                Some(Action::Op(op))
            }
            _ => None,
        };
    }
    // Moving over Karaoke mode's bottom shows its controls; off the
    // window they go.
    if karaoke::in_mode(state) {
        match input.kind {
            Moved => return Some(Action::PointerMoved { row: input.row }),
            Out => return Some(Action::PointerLeft),
            _ => {}
        }
    }
    // And a queue row held: it goes where the pointer is.
    if state.queue.dragging {
        match input.kind {
            Up(Left) => return Some(Action::QueueDragEnd),
            Drag(Left) => {
                return match hit(state, input.col, input.row) {
                    Some(Target::Row(row)) => Some(Action::QueueDrag(row)),
                    _ => None,
                };
            }
            _ => {}
        }
    }
    // Ctrl held for a preview: the pointer over a browser row selects it
    // (and the preview follows).
    if input.kind == Moved && crate::core::preview::held(state) {
        return match hit(state, input.col, input.row) {
            Some(Target::Row(row)) if row != state.browser.cursor => Some(Action::SelectRow(row)),
            _ => None,
        };
    }
    let target = over.or_else(|| hit(state, input.col, input.row))?;
    match (input.kind, target) {
        (Down(Right), Target::TransitionLane(block) | Target::TransitionOverview(block)) => {
            Some(Action::TransitionPanStart {
                block,
                column: input.col,
                width: transition_width(state, block)?,
                overview: matches!(target, Target::TransitionOverview(_)),
            })
        }
        (
            ScrollUp | ScrollDown,
            Target::TransitionLane(block) | Target::TransitionOverview(block),
        ) => {
            let overview = matches!(target, Target::TransitionOverview(_));
            Some(Action::TransitionWheel {
                block,
                steps: if input.kind == ScrollUp { -1 } else { 1 },
                // Over the whole track the wheel moves the view; over the
                // lanes it zooms that block, around the pointer.
                scroll: overview,
                at: (!overview)
                    .then(|| transition_time(state, block, input.col, false))
                    .flatten(),
            })
        }
        (Down(Left), Target::TransitionLane(block)) => {
            transition_time(state, block, input.col, false).map(|time| Action::TransitionPoint {
                block,
                time,
                extend: false,
            })
        }
        (Drag(Left), Target::TransitionLane(block)) => {
            transition_time(state, block, input.col, false).map(|time| Action::TransitionPoint {
                block,
                time,
                extend: true,
            })
        }
        (Down(Left), Target::KaraokeSlider(slider)) => {
            slider_share(state, slider, x).map(|share| Action::KaraokeSlide { slider, share })
        }
        (ScrollUp | ScrollDown, Target::KaraokeSlider(slider)) => Some(Action::KaraokeStep {
            slider,
            by: if input.kind == ScrollUp { 1 } else { -1 },
        }),
        (Down(Left), Target::TransitionOverview(block)) => {
            transition_time(state, block, input.col, true)
                .map(|time| Action::TransitionJump { block, time })
        }
        (ScrollUp | ScrollDown, _) => {
            let step = if input.kind == ScrollUp { -3 } else { 3 };
            match state.screen {
                Screen::Help => Some(Action::HelpScroll(step)),
                Screen::Main if state.palette.is_none() => Some(Action::Scroll(step)),
                _ => None,
            }
        }
        (Down(Left), Target::Bar) => bar_fraction(state, input.col).map(Action::SeekDrag),
        (Down(Left), Target::Toggle(side)) => Some(Action::ShowSide(side)),
        (Down(Left), Target::Row(row)) => Some(Action::ClickRow(row)),
        (Down(Right), Target::Row(row)) => Some(Action::EnqueueRow(row)),
        (Down(Left), Target::Device) => Some(Action::OpenSettings(SettingsRow::Device)),
        (Down(Left), Target::Cover) => Some(Action::Op(Op::NowPlaying)),
        (Down(Left), Target::Button(op) | Target::Link(op)) => Some(Action::Op(op)),
        (Down(Left), Target::SettingsLess(i)) => Some(Action::SettingsClick(i, -1)),
        (Down(Left), Target::SettingsMore(i)) => Some(Action::SettingsClick(i, 1)),
        (Down(Left), Target::SettingsValue(i)) => Some(Action::SettingsClick(i, 0)),
        (Down(Left), Target::DeviceRow(i)) => Some(Action::ChooseDevice(i)),
        (Down(Left), Target::FolderRow(i)) => Some(Action::FolderRow(i)),
        (Down(Left), Target::PaletteRow(i)) => Some(Action::PaletteRun(i)),
        _ => None,
    }
}

/// What a click does with shift held: a few buttons have a second meaning
/// (Enter Karaoke Mode: enter where the song is); the rest are the same.
pub fn shifted(action: Action) -> Action {
    match action {
        Action::Op(Op::KaraokeEnter) => Action::Op(Op::KaraokeEnterHere),
        action => action,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::{Entry, EntryKind};
    use crate::config::Config;
    use crate::core::update;

    fn state() -> State {
        let mut s = State::new("/m".into(), Config::default(), (120, 40));
        s.browser.loading = false;
        s.browser.entries = (0..50)
            .map(|i| Entry {
                name: format!("{i:02}.flac"),
                path: format!("/m/{i:02}.flac").into(),
                kind: EntryKind::File,
                relative: None,
            })
            .collect();
        s
    }

    fn down(col: u16, row: u16) -> MouseInput {
        MouseInput {
            kind: MouseKind::Down(MouseButton::Left),
            col,
            row,
        }
    }

    #[test]
    fn rows_toggle_and_buttons_are_hit() {
        let s = state();
        let Layout::Main(m) = layout::compute(&s) else {
            panic!()
        };
        assert_eq!(hit(&s, m.rows.x + 2, m.rows.y), Some(Target::Row(0)));
        assert_eq!(hit(&s, m.rows.x + 2, m.rows.y + 3), Some(Target::Row(3)));
        let (queue_rect, _) = m.toggle[1];
        assert_eq!(
            action(&s, down(queue_rect.x + 1, queue_rect.y)).map(|a| format!("{a:?}")),
            Some("ShowSide(Queue)".into())
        );
        let play = m
            .bottom
            .buttons
            .iter()
            .find(|b| b.op == Op::PlayPause)
            .unwrap();
        assert!(matches!(
            action(&s, down(play.rect.x, play.rect.y)),
            Some(Action::Op(Op::PlayPause))
        ));
        assert!(matches!(
            action(&s, down(m.cover.unwrap().x + 3, m.cover.unwrap().y + 1)),
            Some(Action::Op(Op::NowPlaying))
        ));
    }

    #[test]
    fn a_double_click_is_enter_and_the_wheel_scrolls() {
        let mut s = state();
        let Layout::Main(m) = layout::compute(&s) else {
            panic!()
        };
        let at = down(m.rows.x + 1, m.rows.y + 2);
        let a = action(&s, at).unwrap();
        assert!(update(&mut s, a).is_empty() || s.browser.cursor == 2);
        assert_eq!(s.browser.cursor, 2);
        update(&mut s, Action::Tick(100));
        let a = action(&s, at).unwrap();
        let fx = update(&mut s, a);
        // What enter does on it: queued, nothing played.
        let mut enter = state();
        enter.browser.cursor = 2;
        let by_key = update(&mut enter, Action::Op(Op::Open));
        let engine = |fx: &[crate::core::Effect]| -> Vec<jodel_playback::Command> {
            fx.iter()
                .filter_map(|e| match e {
                    crate::core::Effect::Engine(c) => Some(c.clone()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(engine(&fx), engine(&by_key));
        assert!(matches!(
            engine(&fx)[..],
            [jodel_playback::Command::Enqueue(ref p)] if p.len() == 1
        ));
        // Clicks far apart only select.
        update(&mut s, Action::Tick(5_000));
        let a = action(&s, down(m.rows.x + 1, m.rows.y + 4)).unwrap();
        update(&mut s, a);
        update(&mut s, Action::Tick(6_000));
        let a = action(&s, down(m.rows.x + 1, m.rows.y + 4)).unwrap();
        assert!(update(&mut s, a)
            .iter()
            .all(|e| !matches!(e, crate::core::Effect::Engine(_))));
        let wheel = MouseInput {
            kind: MouseKind::ScrollDown,
            col: m.rows.x + 1,
            row: m.rows.y + 1,
        };
        let a = action(&s, wheel).unwrap();
        update(&mut s, a);
        assert_eq!(s.browser.cursor, 7);
    }

    #[test]
    fn dragging_the_bar_seeks_on_release() {
        let mut s = state();
        let Layout::Main(m) = layout::compute(&s) else {
            panic!()
        };
        let bar = m.bottom.bar;
        // Nothing playing: no drag starts.
        let a = action(&s, down(bar.x + 5, bar.y)).unwrap();
        update(&mut s, a);
        assert_eq!(s.seek_drag, None);
        // While dragging, the pointer may leave the bar.
        s.seek_drag = Some(0.1);
        let drag = MouseInput {
            kind: MouseKind::Drag(MouseButton::Left),
            col: bar.right() + 5,
            row: 0,
        };
        assert!(matches!(action(&s, drag), Some(Action::SeekDrag(f)) if f == 1.0));
        let up = MouseInput {
            kind: MouseKind::Up(MouseButton::Left),
            col: bar.x,
            row: bar.y,
        };
        assert!(matches!(action(&s, up), Some(Action::SeekTo(f)) if f == 0.0));
    }

    #[test]
    fn settings_rows_step_values() {
        let mut s = state();
        s.screen = Screen::Settings;
        let Layout::Settings(l) = layout::compute(&s) else {
            panic!()
        };
        // The third row: resampling.
        let more = l.lines[2].more;
        let a = action(&s, down(more.x, more.y)).unwrap();
        let fx = update(&mut s, a);
        assert!(!s.config.resample);
        assert!(fx.contains(&crate::core::Effect::SaveConfig));
    }
}
