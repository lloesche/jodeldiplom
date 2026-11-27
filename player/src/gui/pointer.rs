//! The pointer: what a widget's press holds ([`Grab`]), followed wherever
//! the pointer goes; and the wheel, for every widget alike, by whole steps
//! ([`Wheel`]).
//!
//! The widgets only say what the pointer does over them; this is the one
//! owner of what is held. A button is let go of wherever the pointer is
//! (the window hears every release, whichever widget it was over), and a
//! release with nothing held is dropped.

use iced::mouse::ScrollDelta;
use iced::{Point, Rectangle};

use crate::app::App;
use crate::core::desk::Channel;
use crate::core::lanes::{self, Block};
use crate::core::lyric_edit::screen;
use crate::core::lyric_edit::{self, LyricsEdit, Part};
use crate::core::shader_tab::ShaderAction;
use crate::core::sliders::Slider;
use crate::core::synth_tab::{self, SynthAction};
use crate::core::timeview::{self, PanInput, PanOf};
use crate::core::transition::viewport;
use crate::core::{Action, State};
use jodel_lyrics::WordAt;

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
use crate::gui::widgets::piano;

/// What a widget's press holds until its button is let go of: the
/// pointer's moves anywhere in the window go to it, and its release lands
/// it, each turned into the core's actions by what it holds. Its bounds
/// are in the window's points, as the widget was laid out when pressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grab {
    /// The seek bar's track: the share across it under the pointer is
    /// where a drag would go, sought on release.
    Seek(Rectangle),
    /// A queue row (its click told apart, [`Action::ClickRow`]): the
    /// track goes to the row under the pointer as it moves. `rows` is
    /// where the rows in view are, the first at its top, each `row`
    /// points high.
    QueueRow {
        /// The rows in view.
        rows: Rectangle,
        /// A row's height.
        row: f32,
    },
    /// A Transition block's lanes (`bounds`, their width the span shown):
    /// the cursor goes to the beat under the pointer, and, dragged, bars
    /// are selected to the beat it is over across them.
    Lanes {
        /// The block.
        block: Block,
        /// Where its lanes are.
        bounds: Rectangle,
    },
    /// A view held with the secondary button (a Transition block's, over
    /// its lanes or its whole track; the lyrics editor's, over its lane),
    /// `bounds` what it was pressed on: it moves with the pointer.
    Pan {
        /// The view.
        of: PanOf,
        /// Where its lanes, its whole track or its lane are.
        bounds: Rectangle,
    },
    /// A slider of the Karaoke tab or of Karaoke mode's controls, its
    /// track: set to the share across it under the pointer, as pressed and
    /// as it moves (landing on its finest step, [`crate::core::sliders::Slider::unit`]).
    Slider(Slider, Rectangle),
    /// A word of the lyrics editor, by its start, middle or end, over its
    /// lane (`lane`, its width the span shown): selected as pressed, then
    /// that part goes to the time under the pointer as it moves, set by
    /// hand.
    LyricWord {
        /// The word.
        word: WordAt,
        /// What of it.
        part: Part,
        /// Where the lane is.
        lane: Rectangle,
    },
    /// A parameter's slider on the Synth tab (its place in the patch's
    /// parameters), its track: set to the share across it under the
    /// pointer, as pressed and as it moves.
    SynthKnob(usize, Rectangle),
    /// A knob's slider on the Shader tab (its place in the selected
    /// pass's knobs, or the mix's; its driver's amount, or the knob
    /// itself), its track: set to the share across it under the pointer.
    ShaderKnob(usize, bool, Rectangle),
    /// The Synth tab's keyboard (its bounds): the key under the pointer
    /// sounds while it is held, the pointer sliding from key to key.
    Piano(Rectangle),
}

impl Grab {
    /// The button that holds it.
    fn button(self) -> MouseButton {
        match self {
            Grab::Seek(_)
            | Grab::QueueRow { .. }
            | Grab::Lanes { .. }
            | Grab::Slider(..)
            | Grab::LyricWord { .. }
            | Grab::SynthKnob(..)
            | Grab::ShaderKnob(..)
            | Grab::Piano(_) => MouseButton::Left,
            Grab::Pan { .. } => MouseButton::Right,
        }
    }

    /// What the press at `at` does (a queue row's click is told by the
    /// row itself).
    fn pressed(self, at: Point, state: &State) -> Option<Action> {
        match self {
            Grab::Seek(_)
            | Grab::Slider(..)
            | Grab::SynthKnob(..)
            | Grab::ShaderKnob(..)
            | Grab::Piano(_) => self.moved(at, state),
            Grab::QueueRow { .. } => None,
            Grab::Lanes { block, bounds } => Some(Action::TransitionPoint {
                block,
                time: lanes_time(state, block, bounds, at),
                extend: false,
            }),
            Grab::Pan { of, bounds } => Some(Action::Pan {
                of,
                input: PanInput::Start {
                    at: at.x,
                    width: bounds.width,
                },
            }),
            Grab::LyricWord { word, part, lane } => {
                let (start, end) = screen::window(state)?;
                // A tremble of a few points is no drag.
                let slop = f64::from(SLOP / lane.width.max(1.0)) * (end - start);
                Some(lyrics(LyricsEdit::Hold {
                    word,
                    part,
                    time: lyric_time(state, lane, at)?,
                    slop,
                }))
            }
        }
    }

    /// What the pointer at `at` does while it holds.
    pub fn moved(self, at: Point, state: &State) -> Option<Action> {
        match self {
            Grab::Seek(track) => Some(Action::SeekDrag(share_across(track, at))),
            Grab::QueueRow { rows, row } => {
                if !rows.contains(at) || row <= 0.0 {
                    return None;
                }
                let index = state.queue.scroll + ((at.y - rows.y) / row) as usize;
                (index < state.queue.paths.len()).then_some(Action::QueueDrag(index))
            }
            Grab::Lanes { block, bounds } => {
                // Only to another beat: a hand's tremble between the two
                // presses of a double click selects nothing.
                let time = lanes_time(state, block, bounds, at);
                let a = &lanes::rhythmed(state, block)?.analysis;
                let cursor = viewport::cursor(state, block, state.heard());
                (a.beat_at(time).unwrap_or(0) != cursor?).then_some(Action::TransitionPoint {
                    block,
                    time,
                    extend: true,
                })
            }
            Grab::Pan { of, .. } => Some(Action::Pan {
                of,
                input: PanInput::To(at.x),
            }),
            Grab::Slider(slider, track) => Some(Action::KaraokeSlide {
                slider,
                share: share_across(track, at),
            }),
            Grab::LyricWord { lane, .. } => {
                Some(lyrics(LyricsEdit::Drag(lyric_time(state, lane, at)?)))
            }
            Grab::SynthKnob(knob, track) => Some(Action::Synth(SynthAction::Slide {
                knob,
                share: share_across(track, at) as f32,
            })),
            Grab::ShaderKnob(knob, amount, track) => {
                let share = share_across(track, at) as f32;
                Some(Action::Shader(if amount {
                    ShaderAction::Amount { knob, share }
                } else {
                    ShaderAction::Slide { knob, share }
                }))
            }
            Grab::Piano(keys) => {
                let key = piano::key_at(keys, synth_tab::KEYS / 12, at);
                let note = key.map(|k| synth_tab::key_note(0, k));
                (note != state.synth.mouse).then_some(Action::Synth(SynthAction::Press(note)))
            }
        }
    }

    /// What letting go at `at` does.
    fn released(self, at: Point) -> Option<Action> {
        match self {
            Grab::Seek(track) => Some(Action::SeekTo(share_across(track, at))),
            Grab::QueueRow { .. } => Some(Action::QueueDragEnd),
            Grab::Lanes { .. } => None,
            Grab::Pan { of, .. } => Some(Action::Pan {
                of,
                input: PanInput::End,
            }),
            Grab::Slider(..) => Some(Action::KaraokeSlideEnd),
            Grab::LyricWord { .. } => Some(lyrics(LyricsEdit::Drop)),
            Grab::SynthKnob(..) => Some(Action::Synth(SynthAction::SlideEnd)),
            Grab::ShaderKnob(..) => Some(Action::Shader(ShaderAction::SlideEnd)),
            Grab::Piano(_) => Some(Action::Synth(SynthAction::Press(None))),
        }
    }

    /// Whether the core took the press (with nothing to seek, no drag
    /// starts; a row is held on the queue's side): it holds on only then.
    fn taken(self, state: &State) -> bool {
        match self {
            Grab::Seek(_) => state.seek_drag.is_some(),
            Grab::QueueRow { .. } => state.queue.dragging,
            Grab::Lanes { block, .. } => lanes::rhythmed(state, block).is_some(),
            Grab::Pan { of, .. } => timeview::panning(state, of),
            Grab::Slider(..) => state.karaoke.dragging.is_some(),
            Grab::LyricWord { .. } => editing(state).is_some_and(|s| s.held.is_some()),
            Grab::SynthKnob(..) => state.synth.dragging.is_some(),
            Grab::ShaderKnob(..) => state.shader.dragging.is_some(),
            Grab::Piano(_) => state.synth.mouse.is_some(),
        }
    }
}

/// How far, in points, the pointer may tremble on a word it presses and
/// change nothing.
const SLOP: f32 = 3.0;

/// `what` the lyrics editor does, as the core's action.
fn lyrics(what: LyricsEdit) -> Action {
    Action::LyricsEdit(what)
}

/// The lyrics editor's screen, while it is open.
fn editing(state: &State) -> Option<&screen::EditorScreen> {
    lyric_edit::editor(state).map(|e| &e.screen)
}

/// The seconds into the lyrics editor's song under `at` across its lane
/// (`lane`, the span shown), wherever the pointer is: before it the view's
/// start, past it its end.
fn lyric_time(state: &State, lane: Rectangle, at: Point) -> Option<f64> {
    let (start, end) = screen::window(state)?;
    Some(start + share_across(lane, at) * (end - start))
}

/// The seconds into `block`'s track under `at` across its lanes
/// (`bounds`, the span shown), wherever the pointer is: before them their
/// first, past them their last.
fn lanes_time(state: &State, block: Block, bounds: Rectangle, at: Point) -> f64 {
    let (start, end) = viewport::window(state, block, state.heard());
    // Short of the right edge: the time there is the next view's.
    let share = share_across(bounds, at).min(1.0 - 1e-6);
    start + share * (end - start)
}

/// The share of `bounds`'s width left of `at` (0 to 1).
fn share_across(bounds: Rectangle, at: Point) -> f64 {
    if bounds.width <= 0.0 {
        return 0.0;
    }
    f64::from(((at.x - bounds.x) / bounds.width).clamp(0.0, 1.0))
}

/// Where the pointer was, and what it holds.
#[derive(Debug, Default)]
pub struct PointerState {
    /// What a widget's press holds, and where the pointer was last seen
    /// while it does (in the window's points).
    grab: Option<(Grab, Point)>,
}

impl PointerState {
    /// Whether a widget's press holds the pointer.
    pub fn grabbed(&self) -> bool {
        self.grab.is_some()
    }

    /// A widget's press at `at` holds `grab` (unless another holds the
    /// pointer already): what it does there carried out, and held on to if
    /// the core took it. Whether it did anything.
    pub fn grab(&mut self, app: &mut App, grab: Grab, at: Point) -> bool {
        if self.grab.is_some() {
            return false;
        }
        if let Some(action) = grab.pressed(at, &app.state) {
            dispatch(app, action);
        }
        if grab.taken(&app.state) {
            self.grab = Some((grab, at));
        }
        true
    }

    /// The pointer moved to `at` in the window: what the grab held does
    /// there. Whether it did anything.
    pub fn moved(&mut self, app: &mut App, at: Point) -> bool {
        let Some((grab, last)) = &mut self.grab else {
            return false;
        };
        if *last == at {
            return false;
        }
        *last = at;
        let Some(action) = grab.moved(at, &app.state) else {
            return false;
        };
        dispatch(app, action);
        true
    }

    /// A button let go of, wherever the pointer is: the grab's lands
    /// where the pointer was last seen; any other is dropped.
    pub fn up(&mut self, app: &mut App, button: MouseButton) -> bool {
        match self.grab {
            Some((grab, at)) if grab.button() == button => {
                self.grab = None;
                if let Some(action) = grab.released(at) {
                    dispatch(app, action);
                }
                true
            }
            _ => false,
        }
    }
}

/// `action` carried out, the clock first.
fn dispatch(app: &mut App, action: Action) {
    app.dispatch(Action::Tick(app.now_ms()));
    app.dispatch(action);
}

/// What the wheel turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelOn {
    /// What the screen scrolls (the list, the help), from the widgets that
    /// scroll it (the list, the bottom bar), three rows a step.
    Screen,
    /// A strip of the Mixer tab: its fader, a step a line.
    Strip(Channel),
    /// A Transition block's lanes, or its whole track (`true`): zoomed or
    /// scrolled, a step a line.
    Lanes(Block, bool),
    /// A slider of the Karaoke tab or of Karaoke mode's controls: a step
    /// of its own a line.
    Slider(Slider),
    /// The lyrics editor's lane: zoomed, a step a line.
    Lyrics,
}

/// The wheel's turn not yet a whole step, for what it turns (a touchpad's
/// small pixel deltas add up; another target starts afresh).
#[derive(Debug, Default)]
pub struct Wheel {
    lines: f64,
    on: Option<WheelOn>,
}

impl Wheel {
    /// `delta` over `on`, a line being `line_px` pixels: the whole steps it
    /// makes now (up positive).
    pub fn turn(&mut self, on: WheelOn, delta: ScrollDelta, line_px: f32) -> i32 {
        if self.on != Some(on) {
            self.on = Some(on);
            self.lines = 0.0;
        }
        self.lines += match delta {
            ScrollDelta::Lines { y, .. } => f64::from(y),
            ScrollDelta::Pixels { y, .. } => f64::from(y / line_px.max(1.0)),
        };
        let whole = self.lines.trunc();
        self.lines -= whole;
        whole as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seek grab reads the share across its track wherever the pointer
    /// is: before the track 0, past it 1; the drag as it moves, the seek
    /// as it is let go of.
    #[test]
    fn a_seek_grab_follows_the_pointer_anywhere() {
        let grab = Grab::Seek(Rectangle::new(
            Point::new(100.0, 500.0),
            iced::Size::new(400.0, 16.0),
        ));
        let at = |x, y| Point::new(x, y);
        let state = State::new("/m".into(), crate::config::Config::default());
        let moved = |p| grab.moved(p, &state);
        assert!(matches!(moved(at(300.0, 508.0)), Some(Action::SeekDrag(f)) if f == 0.5));
        assert!(matches!(moved(at(20.0, 0.0)), Some(Action::SeekDrag(f)) if f == 0.0));
        assert!(matches!(moved(at(900.0, 900.0)), Some(Action::SeekDrag(f)) if f == 1.0));
        assert!(matches!(grab.released(at(200.0, 10.0)), Some(Action::SeekTo(f)) if f == 0.25));
        assert_eq!(grab.button(), MouseButton::Left);
    }

    /// A Karaoke slider's grab sets the slider to the share across its
    /// track wherever the pointer is, from its press on; the release lets
    /// go of it; it holds on only while the core holds the slider.
    #[test]
    fn a_slider_grab_follows_the_pointer_anywhere() {
        let track = Rectangle::new(Point::new(600.0, 700.0), iced::Size::new(160.0, 16.0));
        let grab = Grab::Slider(Slider::Voice, track);
        let mut state = State::new("/m".into(), crate::config::Config::default());
        let at = |x, y| Point::new(x, y);
        let slide = |a: Option<Action>| match a {
            Some(Action::KaraokeSlide { slider, share }) => Some((slider, share)),
            _ => None,
        };
        assert_eq!(
            slide(grab.pressed(at(640.0, 708.0), &state)),
            Some((Slider::Voice, 0.25))
        );
        assert_eq!(
            slide(grab.moved(at(10.0, 0.0), &state)),
            Some((Slider::Voice, 0.0))
        );
        assert_eq!(
            slide(grab.moved(at(2000.0, 900.0), &state)),
            Some((Slider::Voice, 1.0))
        );
        assert!(matches!(
            grab.released(at(0.0, 0.0)),
            Some(Action::KaraokeSlideEnd)
        ));
        assert!(!grab.taken(&state));
        state.karaoke.dragging = Some(Slider::Voice);
        assert!(grab.taken(&state));
        assert_eq!(grab.button(), MouseButton::Left);
    }

    /// A queue row held goes to the row in view under the pointer, scrolled
    /// or not; off the rows, or past the queue's end, it stays; the
    /// release ends the drag.
    #[test]
    fn a_queue_row_grab_moves_the_track_to_the_row_under_the_pointer() {
        let grab = Grab::QueueRow {
            rows: Rectangle::new(Point::new(10.0, 100.0), iced::Size::new(300.0, 340.0)),
            row: 34.0,
        };
        let mut state = State::new("/m".into(), crate::config::Config::default());
        state.queue.paths = (0..30).map(|i| format!("/q/{i}.flac").into()).collect();
        let at = |x, y| Point::new(x, y);
        assert!(matches!(
            grab.moved(at(50.0, 100.0 + 34.0 * 3.5), &state),
            Some(Action::QueueDrag(3))
        ));
        state.queue.scroll = 12;
        assert!(matches!(
            grab.moved(at(50.0, 101.0), &state),
            Some(Action::QueueDrag(12))
        ));
        assert!(grab.moved(at(400.0, 120.0), &state).is_none());
        assert!(grab.moved(at(50.0, 90.0), &state).is_none());
        state.queue.paths.truncate(14);
        assert!(grab.moved(at(50.0, 100.0 + 34.0 * 5.0), &state).is_none());
        assert!(matches!(
            grab.released(at(0.0, 0.0)),
            Some(Action::QueueDragEnd)
        ));
        assert!(grab.pressed(at(50.0, 120.0), &state).is_none());
    }

    #[test]
    fn small_pixel_turns_add_up_to_a_step_and_another_target_starts_afresh() {
        let mut wheel = Wheel::default();
        let px = |y| ScrollDelta::Pixels { x: 0.0, y };
        let strip = WheelOn::Strip(Channel::Main);
        // Five touchpad deltas of 4 pixels, a line being 18: one step.
        let steps: Vec<i32> = (0..5).map(|_| wheel.turn(strip, px(4.0), 18.0)).collect();
        assert_eq!(steps, [0, 0, 0, 0, 1]);
        // What was left does not carry over to the screen.
        assert_eq!(wheel.turn(WheelOn::Screen, px(15.0), 18.0), 0);
        assert_eq!(wheel.turn(WheelOn::Screen, px(4.0), 18.0), 1);
        // A mouse's notches are lines; down is negative.
        let lines = ScrollDelta::Lines { x: 0.0, y: -2.0 };
        assert_eq!(wheel.turn(WheelOn::Strip(Channel::Mic), lines, 18.0), -2);
    }
}
