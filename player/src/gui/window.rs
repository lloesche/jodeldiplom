//! The window as an iced application around the player ([`App`]): its
//! messages, what they do, what it shows and what it listens to.
//!
//! - Keys go through [`crate::core::keymap`] ([`super::keys`]); the
//!   widgets iced draws send the core's actions ([`Message::Act`]), and a
//!   press that holds the pointer ([`Message::Grab`], [`super::pointer`]):
//!   its moves come here from anywhere while it is held. Every button's
//!   release comes here, whichever widget it was over, so nothing stays
//!   held.
//! - Ticks: one decision each ([`super::paint`]): every refresh of the
//!   display while something moves smoothly (from inside each redraw,
//!   [`super::pacer`]), 30 a second while only the spectrum and the clock
//!   do; the rest's tick ([`REST`]) always; the player's threads wake it
//!   ([`super::wake`]).
//! - What shows: [`super::screens`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use iced::mouse::ScrollDelta;
use iced::window;
use iced::{event, keyboard, mouse, Element, Point, Size, Subscription, Task};

use super::lanes::Lanes;
use super::mixer;
use super::pacer::Pacer;
use super::paint::{Pace, Paint, Place, Seen, REST};
use super::pointer::{Grab, MouseButton, PointerState, Wheel, WheelOn};
use super::screens::Shown;
use super::stage_view::StageView;
use super::wake::{Wake, Wakes};
use super::widgets::cover::Covers;
use crate::app::{App, AppOptions};
use crate::core::desk::{Channel, DeskAction};
use crate::core::keymap;
use crate::core::preview::Ctrl;
use crate::core::{Action, Screen, State};
use crate::setup::{Setup, Status};
use crate::view::setup::Choice;

/// What the window takes.
#[derive(Clone, Debug)]
pub enum Message {
    /// A tick at the pace of the moment (every refresh, or at an interval).
    Frame(Instant),
    /// The rest's tick, whatever the pace.
    Rest(Instant),
    /// A player's thread has news.
    Wake,
    /// A key, a button let go of, or the window opened, resized, rescaled,
    /// unfocused, closing.
    Event(iced::Event, window::Id),
    /// A widget's action for the core.
    Act(Action),
    /// A widget's command for the core, and the one it is with shift held
    /// (Enter Karaoke Mode: in where the song is).
    OpShifted(keymap::Op, keymap::Op),
    /// A widget's press at a point holds the pointer until its button is
    /// let go of (the seek bar's, a queue row's).
    Grab(Grab, Point),
    /// The wheel over a strip of the Mixer tab.
    DeskWheel(Channel, ScrollDelta),
    /// The wheel over a widget that scrolls what the screen scrolls (the
    /// list, the bottom bar).
    Wheel(ScrollDelta),
    /// The wheel over `on`: each whole step up does the first action, each
    /// down the second (the Transition tab's lanes zoom in or out, a Karaoke
    /// slider steps).
    Steps(WheelOn, ScrollDelta, Box<(Action, Action)>),
    /// The window's last frame, for `--save-frame`.
    Shot(window::Screenshot),
    /// A choice on the setup screen.
    Setup(Choice),
}

/// How the window was asked to run.
pub(super) struct Start {
    pub(super) options: AppOptions,
    pub(super) wake: Wake,
    pub(super) wakes: Wakes,
    pub(super) exit_after: Option<f64>,
    pub(super) save_frame: Option<PathBuf>,
    /// Where an error that ends the window is told back.
    pub(super) error: Arc<Mutex<Option<String>>>,
}

/// Where the window stands.
enum Phase {
    /// Setup gets the player's files ([`crate::setup`]); the app starts
    /// with these options.
    Setup { setup: Setup, options: AppOptions },
    /// The player runs.
    Running(App),
    /// The window closes; what showed shows as it was until the last frame
    /// is taken.
    Leaving(Last),
}

/// What shows while the window closes.
#[expect(
    clippy::large_enum_variant,
    reason = "one, in the phase that holds the app anyway"
)]
enum Last {
    /// The app.
    App(App),
    /// The setup screen, where setup stood.
    Setup(Status),
    Nothing,
}

/// The window's state: the player, and what iced needs beside it.
pub struct Gui {
    phase: Phase,
    /// Karaoke mode's stage.
    stage: StageView,
    paint: Paint,
    pacer: Pacer,
    pointer: PointerState,
    wheel: Wheel,
    caps: mixer::Caps,
    /// The covers the screen shows, their art fitted to their places.
    covers: Covers,
    /// The lanes the screen shows.
    lanes: Lanes,
    wake: Wake,
    window: Option<window::Id>,
    /// The window's size in logical pixels, and its scale factor.
    size: Size,
    scale: f32,
    modifiers: keyboard::Modifiers,
    /// The window is fullscreen for Karaoke mode.
    fullscreen: bool,
    /// When the last tick at the refresh came, and whether they stopped
    /// coming (the window covered): nothing is drawn then.
    last_refresh: Option<Instant>,
    occluded: bool,
    started: Instant,
    exit_after: Option<f64>,
    save_frame: Option<PathBuf>,
    error: Arc<Mutex<Option<String>>>,
}

impl Gui {
    pub(super) fn new(start: Start) -> (Gui, Task<Message>) {
        let Start {
            mut options,
            wake,
            wakes,
            exit_after,
            save_frame,
            error,
        } = start;
        let started = Instant::now();
        // The moves written down (a data folder chosen anew) go on in the
        // background from now.
        crate::cli::start_mover(&mut options);
        let phase = match crate::cli::models_ready(&options) {
            Ok(paths) => {
                crate::cli::set_models(&mut options, &paths);
                Phase::Running(App::new(options))
            }
            Err((manifest, places)) => {
                // The files are not all here: the setup screen first.
                let setup = Setup::start(
                    manifest,
                    places,
                    Arc::new(crate::setup::Real),
                    options.notify.clone(),
                );
                Phase::Setup { setup, options }
            }
        };
        crate::trace::took("window: the app started", started, true);
        let mut gui = Gui::around(phase, wake, started);
        gui.exit_after = exit_after;
        gui.save_frame = save_frame;
        gui.error = error;
        (gui, wakes.task())
    }

    /// The window's state around `app` with no window, for drawing it all
    /// the same (the shots, the view's cost): its size and scale come as
    /// a window's events would tell them.
    pub fn headless(app: App) -> Gui {
        let (wake, _) = Wake::new();
        Gui::around(Phase::Running(app), wake, Instant::now())
    }

    /// The window's state on the setup screen with no window, `setup`
    /// under way (the app to start with `options` once it is done): for
    /// the shots and the screen's tests.
    pub fn headless_setup(setup: Setup, options: AppOptions) -> Gui {
        let (wake, _) = Wake::new();
        Gui::around(Phase::Setup { setup, options }, wake, Instant::now())
    }

    /// The window's state in `phase`, started at `started`, before it
    /// hears from its window.
    fn around(phase: Phase, wake: Wake, started: Instant) -> Gui {
        Gui {
            phase,
            stage: StageView::default(),
            paint: Paint::new(),
            pacer: Pacer::default(),
            pointer: PointerState::default(),
            wheel: Wheel::default(),
            caps: mixer::Caps::default(),
            covers: Covers::default(),
            lanes: Lanes::default(),
            wake,
            window: None,
            size: Size::ZERO,
            scale: 1.0,
            modifiers: keyboard::Modifiers::empty(),
            fullscreen: false,
            last_refresh: None,
            occluded: false,
            started,
            exit_after: None,
            save_frame: None,
            error: Arc::default(),
        }
    }

    /// The player's state replaced by `state` (a screen as a shot sets it
    /// up): the next tick draws it whole.
    pub fn show(&mut self, state: State) {
        if let Phase::Running(app) = &mut self.phase {
            app.state = state;
            self.paint.dirty = true;
        }
    }

    /// The cover shown replaced, or none (a shot without art): the next
    /// tick draws it whole.
    pub fn set_art(&mut self, image: Option<image::RgbImage>) {
        if let Phase::Running(app) = &mut self.phase {
            app.set_art(image);
            self.paint.dirty = true;
        }
    }

    /// The player's state, while there is a player (for the tests).
    pub fn state(&self) -> Option<&State> {
        self.app().map(|app| &app.state)
    }

    /// What keeps the row or the entry the keys are on in view, on a
    /// screen that scrolls them (the settings): run after each key.
    pub fn in_view(&self) -> Option<super::widgets::Reveal> {
        self.state()
            .filter(|state| state.screen == crate::core::Screen::Settings)
            .map(super::screens::settings::reveal)
    }

    pub(super) fn title(&self) -> String {
        "jodeldiplom".to_owned()
    }

    /// The app, while there is one to show.
    fn app(&self) -> Option<&App> {
        match &self.phase {
            Phase::Running(app) | Phase::Leaving(Last::App(app)) => Some(app),
            _ => None,
        }
    }

    /// What `message` does to the player and the window.
    pub fn update(&mut self, message: Message) -> Task<Message> {
        if let Phase::Leaving(_) = self.phase {
            return match message {
                Message::Shot(shot) => self.save(&shot),
                Message::Event(iced::Event::Window(window::Event::CloseRequested), _) => {
                    self.quit()
                }
                _ => Task::none(),
            };
        }
        let writing = self.writing();
        let task = match message {
            Message::Frame(at) => self.tick(at, true),
            Message::Rest(at) => self.tick(at, false),
            Message::Wake => {
                self.wake.taken();
                if let Phase::Running(app) = &mut self.phase {
                    self.paint.news |= app.pump();
                }
                Task::none()
            }
            Message::Event(event, id) => self.event(event, id),
            Message::Act(action) => {
                self.act(action);
                Task::none()
            }
            Message::OpShifted(plain, shifted) => {
                let shift = self.modifiers.shift();
                self.act(Action::Op(if shift { shifted } else { plain }));
                Task::none()
            }
            Message::Grab(grab, at) => {
                if let Phase::Running(app) = &mut self.phase {
                    self.paint.dirty |= self.pointer.grab(app, grab, at);
                }
                Task::none()
            }
            Message::Wheel(delta) => {
                // A step scrolls three rows: a touchpad's pixels move the
                // list as far as they go.
                let line = 3.0 * super::look::height::ROW;
                let steps = self.wheel.turn(WheelOn::Screen, delta, line);
                for _ in 0..steps.unsigned_abs() {
                    let action = self.app().and_then(|app| scrolled(&app.state, steps > 0));
                    if let Some(action) = action {
                        self.act(action);
                    }
                }
                Task::none()
            }
            Message::Steps(on, delta, actions) => {
                let steps = self.wheel.turn(on, delta, self.line_px());
                let (up, down) = *actions;
                for _ in 0..steps.unsigned_abs() {
                    self.act(if steps > 0 { up.clone() } else { down.clone() });
                }
                Task::none()
            }
            Message::DeskWheel(channel, delta) => {
                let steps = self
                    .wheel
                    .turn(WheelOn::Strip(channel), delta, self.line_px());
                if steps != 0 {
                    self.act(Action::Desk(DeskAction::Wheel(channel, steps)));
                }
                Task::none()
            }
            Message::Shot(_) => Task::none(),
            Message::Setup(choice) => self.setup_choice(choice),
        };
        if matches!(&self.phase, Phase::Running(app) if app.state.quit) {
            return self.leave();
        }
        // A text to write in the lyrics editor: its field takes the keys,
        // the text in it selected to be typed over.
        if !writing && self.writing() {
            let field = super::screens::lyrics::FIELD;
            return Task::batch([
                task,
                iced::widget::operation::focus(field),
                iced::widget::operation::text_input::select_all(field),
            ]);
        }
        task
    }

    /// Whether a text is being written in the lyrics editor.
    fn writing(&self) -> bool {
        self.app()
            .is_some_and(|app| crate::core::lyric_edit::screen::writing(&app.state))
    }

    /// The core's `action`, with the clock first.
    fn act(&mut self, action: Action) {
        if let Phase::Running(app) = &mut self.phase {
            app.dispatch(Action::Tick(app.now_ms()));
            app.dispatch(action);
            self.paint.dirty = true;
        }
    }

    /// A wheel line in points.
    fn line_px(&self) -> f32 {
        super::metrics::LINE
    }

    fn event(&mut self, event: iced::Event, id: window::Id) -> Task<Message> {
        match event {
            iced::Event::Window(event) => {
                self.window = Some(id);
                match event {
                    window::Event::Opened {
                        size, scale_factor, ..
                    } => {
                        #[cfg(target_os = "macos")]
                        super::set_dock_icon();
                        self.size = size;
                        self.scale = scale_factor;
                    }
                    window::Event::Resized(size) => self.size = size,
                    window::Event::Rescaled(scale) => self.scale = scale,
                    window::Event::CloseRequested => return self.leave(),
                    // Ctrl (or the demo key) let go of in another window is
                    // never seen: let go of here.
                    window::Event::Unfocused => {
                        if self.app().is_some_and(|app| app.state.preview.ctrl) {
                            self.act(Action::Ctrl(Ctrl::Released));
                        }
                        if self.app().is_some_and(|app| app.state.karaoke.demo.held()) {
                            self.act(Action::DemoLetGo);
                        }
                    }
                    _ => {}
                }
                self.paint.dirty = true;
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                if let Phase::Running(app) = &mut self.phase {
                    self.paint.dirty |= self.pointer.moved(app, position);
                }
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(button)) => {
                let button = match button {
                    mouse::Button::Left => MouseButton::Left,
                    mouse::Button::Right => MouseButton::Right,
                    mouse::Button::Middle => MouseButton::Middle,
                    _ => return Task::none(),
                };
                // A fader held is let go of wherever the pointer is.
                let desk = (button == MouseButton::Left)
                    .then(|| {
                        self.app()
                            .and_then(|app| crate::core::desk::let_go(&app.state))
                    })
                    .flatten();
                if let Some(action) = desk {
                    self.act(Action::Desk(action));
                }
                if let Phase::Running(app) = &mut self.phase {
                    self.paint.dirty |= self.pointer.up(app, button);
                }
            }
            iced::Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                let was = super::keys::ctrl_alone(self.modifiers);
                self.modifiers = modifiers;
                let now = super::keys::ctrl_alone(modifiers);
                if now != was {
                    self.act(Action::Ctrl(if now {
                        Ctrl::Pressed
                    } else {
                        Ctrl::Released
                    }));
                }
            }
            iced::Event::Keyboard(keyboard::Event::KeyReleased { key, .. }) => {
                if self
                    .app()
                    .is_some_and(|app| super::keys::lets_go_of_demo(&app.state, &key))
                {
                    self.act(Action::DemoLetGo);
                }
            }
            iced::Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modified_key,
                modifiers,
                ..
            }) => match &mut self.phase {
                Phase::Setup { setup, .. } => {
                    let failed = setup.status().failure.is_some();
                    let name = super::keys::setup_name(&modified_key);
                    if let Some(choice) = crate::view::setup::choice(&name, failed) {
                        return self.setup_choice(choice);
                    }
                }
                Phase::Running(app) => {
                    // The clock first: what a key means can depend on the
                    // time (space while a name is being typed).
                    app.dispatch(Action::Tick(app.now_ms()));
                    let action = super::keys::pressed(&key, &modified_key, modifiers)
                        .and_then(|k| keymap::action(&app.state, k));
                    if let Some(action) = action {
                        app.dispatch(action);
                        self.paint.dirty = true;
                    }
                    // What the keys are on stays in view.
                    if let Some(reveal) = self.in_view() {
                        return iced::advanced::widget::operate(reveal);
                    }
                }
                Phase::Leaving(_) => {}
            },
            _ => {}
        }
        Task::none()
    }

    /// The person's choice on the setup screen (while it shows).
    fn setup_choice(&mut self, choice: Choice) -> Task<Message> {
        let Phase::Setup { setup, .. } = &self.phase else {
            return Task::none();
        };
        match choice {
            Choice::Quit => self.leave(),
            Choice::Retry => {
                setup.retry();
                Task::none()
            }
        }
    }

    /// Setup finished: the app starts with the files it found.
    fn setup_done(&mut self, paths: &[PathBuf]) {
        if let Phase::Setup { mut options, .. } =
            std::mem::replace(&mut self.phase, Phase::Leaving(Last::Nothing))
        {
            crate::cli::set_models(&mut options, paths);
            self.phase = Phase::Running(App::new(options));
        }
        self.paint.dirty = true;
    }

    /// Whether the window can be seen: not minimised, its frames coming.
    fn visible(&self) -> bool {
        self.size.width > 0.0 && self.size.height > 0.0 && !self.occluded
    }

    /// Where a frame is drawn now.
    fn place(&self) -> Place {
        Place {
            size: (
                (self.size.width * self.scale).round() as u32,
                (self.size.height * self.scale).round() as u32,
            ),
            scale: self.scale,
            points: self.size,
        }
    }

    /// What the plan needs to know of the window.
    fn seen(&self) -> Seen {
        Seen {
            visible: self.visible(),
            following: self.lanes.following(),
            gliding: self.caps.moving(),
            stage: self.stage.gpu(),
            pictures_moving: self.stage.moving(),
        }
    }

    /// A tick at `at` (`refresh`: one at the display's refresh): the
    /// player's news taken in, what is due done, the window made
    /// fullscreen or not, what is heard now looked at, and what the plan
    /// says drawn.
    fn tick(&mut self, at: Instant, refresh: bool) -> Task<Message> {
        if self
            .exit_after
            .is_some_and(|s| self.started.elapsed().as_secs_f64() >= s)
        {
            return self.leave();
        }
        if refresh {
            self.last_refresh = Some(at);
            self.occluded = false;
        }
        let (place, seen) = (self.place(), self.seen());
        match &mut self.phase {
            Phase::Setup { setup, .. } => {
                // Setup tells the window as it goes (the view is built
                // again each time); that it is done is seen at a tick.
                if let Some(paths) = setup.status().finished {
                    self.setup_done(&paths);
                }
                Task::none()
            }
            Phase::Running(app) => {
                self.paint.news |= app.pump();
                if app.state.quit {
                    return self.leave();
                }
                let heard = app.heard_for_frame();
                self.caps.step(&app.state);
                let mut plan = self.paint.plan(app, seen, at);
                // Ticks at the refresh stopped coming: the window is
                // covered; nothing is drawn until they come again.
                let stale = self
                    .last_refresh
                    .is_none_or(|last| at.saturating_duration_since(last) > 2 * REST);
                if !refresh && plan.pace == Pace::Refresh && stale {
                    self.occluded = true;
                    plan.draw = super::paint::Draw::Nothing;
                }
                let parts = (&self.stage, &mut self.lanes);
                self.paint.draw(plan, parts, app, place, (heard, at));
                let places = super::screens::covers(&app.state, self.size, self.scale);
                self.covers.show(app.art(), &places, self.scale);
                // Karaoke mode is fullscreen; leaving it, the window comes
                // back.
                let fullscreen = crate::core::karaoke::in_mode(&app.state);
                match self.window {
                    Some(id) if fullscreen != self.fullscreen => {
                        self.fullscreen = fullscreen;
                        self.paint.dirty = true;
                        // macOS: the screen covered by the window itself, not
                        // a fullscreen Space ([`super::cover`]).
                        #[cfg(target_os = "macos")]
                        return window::run(id, move |w| super::cover::cover(w, fullscreen))
                            .discard();
                        #[cfg(not(target_os = "macos"))]
                        window::set_mode(
                            id,
                            if fullscreen {
                                window::Mode::Fullscreen
                            } else {
                                window::Mode::Windowed
                            },
                        )
                    }
                    _ => Task::none(),
                }
            }
            Phase::Leaving(_) => Task::none(),
        }
    }

    /// The player ends: its last frame taken first where it is to be saved
    /// (what shows stays as it is until then: the screenshot is of the
    /// frame drawn last, with the widgets that drew it).
    fn leave(&mut self) -> Task<Message> {
        let last = match std::mem::replace(&mut self.phase, Phase::Leaving(Last::Nothing)) {
            Phase::Running(app) => Last::App(app),
            Phase::Setup { setup, .. } => Last::Setup(setup.status()),
            Phase::Leaving(_) => Last::Nothing,
        };
        self.phase = Phase::Leaving(last);
        match (&self.save_frame, self.window) {
            (Some(_), Some(id)) => window::screenshot(id).map(Message::Shot),
            (Some(path), None) => {
                self.fail(format!("{}: no window to save a frame of", path.display()));
                self.quit()
            }
            (None, _) => self.quit(),
        }
    }

    /// The app stops (the session and the singing kept saved, the engine
    /// stopped), the window closes.
    fn quit(&mut self) -> Task<Message> {
        self.phase = Phase::Leaving(Last::Nothing);
        iced::exit()
    }

    /// `--save-frame`: the window's last frame into the file.
    fn save(&mut self, shot: &window::Screenshot) -> Task<Message> {
        if let Some(path) = self.save_frame.clone() {
            if let Err(e) = save_png(&path, shot) {
                self.fail(format!("{}: {e}", path.display()));
            }
        }
        self.quit()
    }

    /// An error the window ends with, for [`super::run`] to return.
    fn fail(&self, error: String) {
        *self.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
    }

    pub(super) fn subscription(&self) -> Subscription<Message> {
        let events = event::listen_with(|event, _status, id| match &event {
            iced::Event::Keyboard(_) | iced::Event::Mouse(mouse::Event::ButtonReleased(_)) => {
                Some(Message::Event(event, id))
            }
            iced::Event::Window(
                window::Event::Opened { .. }
                | window::Event::Resized(_)
                | window::Event::Rescaled(_)
                | window::Event::CloseRequested
                | window::Event::Unfocused,
            ) => Some(Message::Event(event, id)),
            _ => None,
        });
        // While a press holds the pointer, its moves anywhere.
        let moves = if self.pointer.grabbed() {
            event::listen_with(|event, _status, id| match event {
                iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                    Some(Message::Event(event, id))
                }
                _ => None,
            })
        } else {
            Subscription::none()
        };
        let rest = iced::time::every(REST).map(Message::Rest);
        let pace = match &self.phase {
            Phase::Running(app) if self.size.width > 0.0 && self.size.height > 0.0 => {
                self.paint.plan(app, self.seen(), Instant::now()).pace
            }
            _ => Pace::Every(REST),
        };
        // At the refresh the pacer ticks, from inside each redraw.
        self.pacer.tick_every_redraw(pace == Pace::Refresh);
        let frames = match pace {
            Pace::Refresh => Subscription::none(),
            Pace::Every(every) if every != REST => iced::time::every(every).map(Message::Frame),
            Pace::Every(_) => Subscription::none(),
        };
        Subscription::batch([events, moves, rest, frames])
    }

    /// What the window shows ([`super::screens`]), ticked at the refresh
    /// ([`super::pacer`]).
    pub fn view(&self) -> Element<'_, Message> {
        let shown = match &self.phase {
            Phase::Setup { setup, .. } => Shown::Setup(setup.status()),
            Phase::Leaving(Last::Setup(status)) => Shown::Setup(status.clone()),
            _ => Shown::Player(self.app().map(|app| &app.state)),
        };
        let parts = super::screens::Parts {
            stage: &self.stage,
            caps: &self.caps,
            covers: &self.covers,
            lanes: &self.lanes,
            heard: &self.paint.spectrum,
            size: self.size,
            scale: self.scale,
        };
        self.pacer.view(super::screens::view(shown, &parts))
    }
}

/// What a step of the wheel `up` (or down) over a widget that scrolls
/// what the screen scrolls does: the help's text or the list scroll by
/// three lines.
fn scrolled(state: &State, up: bool) -> Option<Action> {
    let step = if up { -3 } else { 3 };
    match state.screen {
        Screen::Help => Some(Action::HelpScroll(step)),
        Screen::Main if state.palette.is_none() => Some(Action::Scroll(step)),
        _ => None,
    }
}

/// A screenshot of the window as a PNG file.
fn save_png(path: &Path, shot: &window::Screenshot) -> Result<(), String> {
    let image = image::RgbaImage::from_raw(shot.size.width, shot.size.height, shot.rgba.to_vec())
        .ok_or("the screenshot's size")?;
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    durable::write_atomic(path, &png).map_err(|e| e.to_string())
}
