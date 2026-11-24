//! The window as an iced application around the player ([`App`]): its
//! messages, what they do, what it shows and what it listens to.
//!
//! - Keys go through [`crate::core::keymap`] ([`super::keys`]); the
//!   pointer over the old canvas through [`crate::core::mouse`]
//!   ([`super::pointer`]); the Mixer tab's own widgets send the desk's
//!   actions. Every button's release comes here, whichever widget it was
//!   over, so nothing stays held.
//! - Ticks: one decision each ([`super::paint`]): every refresh of the
//!   display while something moves smoothly, 30 a second while only the
//!   spectrum and the clock do; the rest's tick ([`REST`]) always; the
//!   player's threads wake it ([`super::wake`]).
//! - What shows: always the old canvas filling the window
//!   ([`super::embed`]) with a slot over it, the Mixer tab
//!   ([`super::mixer`]) or nothing: the same shape, so the canvas's widget
//!   keeps its place in iced's tree.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use iced::mouse::ScrollDelta;
use iced::widget::{opaque, pin, space, stack};
use iced::window;
use iced::{event, keyboard, mouse, Element, Size, Subscription, Task, Widget as _};

use super::embed::Embed;
use super::mixer;
use super::paint::{Pace, Paint, Place, Seen, REST};
use super::pointer::{Pointer, PointerKind, PointerState, Wheel, WheelOn};
use super::wake::{Wake, Wakes};
use crate::app::{App, AppOptions};
use crate::core::desk::{Channel, DeskAction};
use crate::core::keymap;
use crate::core::layout::{self, Layout};
use crate::core::mouse::MouseButton;
use crate::core::preview::Ctrl;
use crate::core::Action;

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
    /// The pointer over the old canvas.
    Pointer(Pointer),
    /// The Mixer tab's widgets.
    Desk(DeskAction),
    /// The wheel over a strip of the Mixer tab.
    DeskWheel(Channel, ScrollDelta),
    /// The window's last frame, for `--save-frame`.
    Shot(window::Screenshot),
}

/// The setup screen under way ([`crate::setup`]), before the app starts.
pub(super) struct SetupRun {
    pub(super) setup: crate::setup::Setup,
    pub(super) buttons: crate::view::setup::Buttons,
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
    /// Setup gets the player's files; the app starts with these options.
    Setup { run: SetupRun, options: AppOptions },
    /// The player runs.
    Running(App),
    /// The window closes; the app (while there is one) shows as it was
    /// until the last frame is taken.
    Leaving(Option<App>),
}

/// The window's state: the player, and what iced needs beside it.
pub(super) struct Gui {
    phase: Phase,
    embed: Embed,
    paint: Paint,
    pointer: PointerState,
    wheel: Wheel,
    caps: mixer::Caps,
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
                let setup = crate::setup::Setup::start(
                    manifest,
                    places,
                    Arc::new(crate::setup::Real),
                    options.notify.clone(),
                );
                let run = SetupRun {
                    setup,
                    buttons: Default::default(),
                };
                Phase::Setup { run, options }
            }
        };
        crate::trace::took("window: the app started", started, true);
        let gui = Gui {
            phase,
            embed: Embed::default(),
            paint: Paint::new(),
            pointer: PointerState::default(),
            wheel: Wheel::default(),
            caps: mixer::Caps::default(),
            wake,
            window: None,
            size: Size::ZERO,
            scale: 1.0,
            modifiers: keyboard::Modifiers::empty(),
            fullscreen: false,
            last_refresh: None,
            occluded: false,
            started,
            exit_after,
            save_frame,
            error,
        };
        (gui, wakes.task())
    }

    pub(super) fn title(&self) -> String {
        "jodeldiplom".to_owned()
    }

    /// The app, while there is one to show.
    fn app(&self) -> Option<&App> {
        match &self.phase {
            Phase::Running(app) | Phase::Leaving(Some(app)) => Some(app),
            _ => None,
        }
    }

    pub(super) fn update(&mut self, message: Message) -> Task<Message> {
        if let Phase::Leaving(_) = self.phase {
            return match message {
                Message::Shot(shot) => self.save(&shot),
                Message::Event(iced::Event::Window(window::Event::CloseRequested), _) => {
                    self.quit()
                }
                _ => Task::none(),
            };
        }
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
            Message::Pointer(pointer) => self.pointer(pointer),
            Message::Desk(action) => {
                self.act(Action::Desk(action));
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
        };
        if matches!(&self.phase, Phase::Running(app) if app.state.quit) {
            return self.leave();
        }
        // macOS: the window's view, for its Metal layer, asked for once the
        // window is known (its opening may come before the window listens).
        #[cfg(target_os = "macos")]
        if let Some(id) = self.window {
            return Task::batch([task, self.embed.find_view(id)]);
        }
        task
    }

    /// The core's `action`, with the clock first.
    fn act(&mut self, action: Action) {
        if let Phase::Running(app) = &mut self.phase {
            app.dispatch(Action::Tick(app.now_ms()));
            app.dispatch(action);
            self.paint.dirty = true;
        }
    }

    /// A wheel line in pixels: a cell's height.
    fn line_px(&self) -> f32 {
        self.embed.cell().1
    }

    fn pointer(&mut self, pointer: Pointer) -> Task<Message> {
        let shift = self.modifiers.shift();
        match &mut self.phase {
            Phase::Setup { run, .. } => {
                if let PointerKind::Down(MouseButton::Left) = pointer.kind {
                    if let Some(choice) = run.buttons.at(pointer.x as u16, pointer.y as u16) {
                        return self.setup_choice(choice);
                    }
                }
            }
            Phase::Running(app) => {
                let dirty = match pointer.kind {
                    PointerKind::Wheel(delta) => {
                        let line = self.embed.cell().1;
                        let steps = self.wheel.turn(WheelOn::Canvas, delta, line);
                        steps != 0 && self.pointer.scroll(app, pointer, steps, shift)
                    }
                    _ => self.pointer.take(app, pointer, shift),
                };
                self.paint.dirty |= dirty;
            }
            Phase::Leaving(_) => {}
        }
        Task::none()
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
                    // Ctrl let go of in another window is never seen: let
                    // go of here.
                    window::Event::Unfocused
                        if self.app().is_some_and(|app| app.state.preview.ctrl) =>
                    {
                        self.act(Action::Ctrl(Ctrl::Released));
                    }
                    _ => {}
                }
                self.paint.dirty = true;
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
                let shift = self.modifiers.shift();
                if let Phase::Running(app) = &mut self.phase {
                    self.paint.dirty |= self.pointer.up(app, button, shift);
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
            iced::Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                modified_key,
                modifiers,
                ..
            }) => match &mut self.phase {
                Phase::Setup { run, .. } => {
                    let failed = run.setup.status().failure.is_some();
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
                }
                Phase::Leaving(_) => {}
            },
            _ => {}
        }
        Task::none()
    }

    /// The person's choice on the setup screen.
    fn setup_choice(&mut self, choice: crate::view::setup::Choice) -> Task<Message> {
        match choice {
            crate::view::setup::Choice::Quit => self.leave(),
            crate::view::setup::Choice::Retry => {
                if let Phase::Setup { run, .. } = &self.phase {
                    run.setup.retry();
                }
                Task::none()
            }
        }
    }

    /// Setup finished: the app starts with the files it found.
    fn setup_done(&mut self, paths: &[PathBuf]) {
        if let Phase::Setup { mut options, .. } =
            std::mem::replace(&mut self.phase, Phase::Leaving(None))
        {
            crate::cli::set_models(&mut options, paths);
            options.size = self.embed.lock().canvas.grid();
            self.phase = Phase::Running(App::new(options));
        }
        self.paint.dirty = true;
    }

    /// Whether the window can be seen: not minimised, its frames coming.
    fn visible(&self) -> bool {
        self.size.width > 0.0 && self.size.height > 0.0 && !self.occluded
    }

    /// Where the old canvas draws now.
    fn place(&self) -> Place {
        Place {
            size: (
                (self.size.width * self.scale).round() as u32,
                (self.size.height * self.scale).round() as u32,
            ),
            scale: self.scale,
            pointer: self.pointer.inside(),
        }
    }

    /// What the plan needs to know of the window.
    fn seen(&self) -> Seen {
        Seen {
            visible: self.visible(),
            layer: self.embed.gpu() && self.embed.has_layer(),
            gliding: self.caps.moving(),
            stage: self.embed.gpu(),
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
            Phase::Setup { run, .. } => {
                // Setup tells the loop as it goes: a frame for each tick.
                if let Some(paths) = run.setup.status().finished {
                    self.setup_done(&paths);
                } else if seen.visible {
                    self.paint.setup(&self.embed, run, place);
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
                self.paint
                    .draw(plan.draw, &self.embed, app, place, (heard, at));
                // Karaoke mode is fullscreen; leaving it, the window comes
                // back.
                let fullscreen = crate::core::karaoke::in_mode(&app.state);
                match self.window {
                    Some(id) if fullscreen != self.fullscreen => {
                        self.fullscreen = fullscreen;
                        self.paint.dirty = true;
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
        let app = match std::mem::replace(&mut self.phase, Phase::Leaving(None)) {
            Phase::Running(app) => Some(app),
            _ => None,
        };
        self.phase = Phase::Leaving(app);
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
        self.phase = Phase::Leaving(None);
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
        let rest = iced::time::every(REST).map(Message::Rest);
        let pace = match &self.phase {
            Phase::Running(app) if self.size.width > 0.0 && self.size.height > 0.0 => {
                self.paint.plan(app, self.seen(), Instant::now()).pace
            }
            Phase::Setup { .. } if self.paint.dirty => Pace::Refresh,
            _ => Pace::Every(REST),
        };
        // At the refresh the canvas's widget ticks, from inside each redraw.
        self.embed.tick_every_redraw(pace == Pace::Refresh);
        let frames = match pace {
            Pace::Refresh => Subscription::none(),
            Pace::Every(every) if every != REST => iced::time::every(every).map(Message::Frame),
            Pace::Every(_) => Subscription::none(),
        };
        Subscription::batch([events, rest, frames])
    }

    pub(super) fn view(&self) -> Element<'_, Message> {
        let desk = self.app().and_then(|app| self.desk(app));
        stack![self.embed.view(), desk.unwrap_or_else(|| space().boxed())].boxed()
    }

    /// The Mixer tab, over its place on the old canvas, while it shows.
    fn desk<'a>(&'a self, app: &'a App) -> Option<Element<'a, Message>> {
        let state = &app.state;
        let Layout::Main(m) = layout::compute(state) else {
            return None;
        };
        let (Some(desk), None) = (&m.desk, &state.palette) else {
            return None;
        };
        let (cw, ch) = self.embed.cell();
        let r = m.rows;
        let view = mixer::view(state, &self.caps, desk, r, (cw, ch));
        Some(
            pin(opaque(view))
                .x(f32::from(r.x) * cw)
                .y(f32::from(r.y) * ch)
                .boxed(),
        )
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
