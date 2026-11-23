//! The window frontend: winit for the window and input, the GPU (or,
//! without one, softbuffer) to put the [`Canvas`] pixels on screen. The
//! same widgets, layout, keymap and mouse regions as the terminal; the
//! pointer is mapped to cells.

pub mod canvas;
pub mod fonts;
pub mod karaoke;
pub mod liquid;
pub mod notes;
pub mod particles;
pub mod pop;
pub mod present;
pub mod rainbow;
pub mod raster;
#[cfg(test)]
mod shader_check;
pub mod shards;
pub mod sparks;
pub mod stage;
pub mod stage_gpu;
pub mod streak;

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton as WinButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key as WinKey, ModifiersState, NamedKey};
use winit::window::{Fullscreen, Icon, Window, WindowId};

use crate::app::{App, AppOptions};
use crate::cli::Args;
use crate::core::keymap::{Key, KeyCode};
use crate::core::mouse::{self, MouseButton, MouseInput, MouseKind};
use crate::core::preview::Ctrl;
use crate::core::Action;
use crate::tui::key_action;
use crate::view::{colors, Links, Scene};

pub use canvas::{font_px, Canvas, FONT};

/// How often the window redraws while something moves smoothly (the
/// Karaoke words going by, a drag, a slide) where the display's refresh
/// rate is not known: 60 a second. Where it is, every refresh
/// ([`display_frame`]): on a 120 Hz screen a frame at 60 shows twice, and
/// what moves stands still for two refreshes and then jumps (to the eye, a
/// doubled, smeared image).
const FRAME: Duration = Duration::from_millis(16);

/// The fastest the window redraws, whatever the display (240 a second).
const FRAME_LEAST: Duration = Duration::from_micros(4_166);

/// How often the display's refresh rate is looked at again (the window may
/// move to another screen).
const DISPLAY_EVERY: Duration = Duration::from_secs(2);

/// A frame for every refresh of the display `window` is on, if its rate is
/// known ([`FRAME`] otherwise), no faster than [`FRAME_LEAST`].
fn display_frame(window: &Window) -> Duration {
    window
        .current_monitor()
        .and_then(|m| m.refresh_rate_millihertz())
        .filter(|&mhz| mhz >= 1_000)
        .map_or(FRAME, |mhz| {
            Duration::from_secs_f64(1_000.0 / f64::from(mhz)).clamp(FRAME_LEAST, FRAME)
        })
}

/// How often it redraws while only the spectrum and the clock move: 30 a
/// second, half the work.
const FRAME_CALM: Duration = Duration::from_millis(33);

/// The Dock icon: macOS has no window icons, and a bare executable (not
/// in the app bundle, whose `Info.plist` names the icon) would show a
/// generic one. Set on the main thread, once the application exists.
#[cfg(target_os = "macos")]
fn set_dock_icon() {
    use objc2::{AllocAnyThread, MainThreadMarker};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let png = crate::icon::png(crate::icon::Style::MacOs, 512);
    let data = NSData::with_bytes(&png);
    let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    // SAFETY: called on the main thread (the marker proves it) with an
    // image AppKit made from our PNG; AppKit keeps its own reference.
    unsafe { app.setApplicationIconImage(Some(&image)) };
}

/// The player's icon at `size` pixels, for the window.
fn window_icon(size: u32) -> Option<Icon> {
    Icon::from_rgba(
        crate::icon::rgba(crate::icon::Style::Windows, size),
        size,
        size,
    )
    .ok()
}

/// A winit key as the core's [`Key`].
pub fn key(logical: &WinKey, modifiers: ModifiersState) -> Option<Key> {
    let code = match logical {
        WinKey::Named(named) => match named {
            NamedKey::Enter => KeyCode::Enter,
            NamedKey::Escape => KeyCode::Esc,
            NamedKey::Backspace => KeyCode::Backspace,
            NamedKey::Tab => KeyCode::Tab,
            NamedKey::ArrowUp => KeyCode::Up,
            NamedKey::ArrowDown => KeyCode::Down,
            NamedKey::ArrowLeft => KeyCode::Left,
            NamedKey::ArrowRight => KeyCode::Right,
            NamedKey::PageUp => KeyCode::PageUp,
            NamedKey::PageDown => KeyCode::PageDown,
            NamedKey::Home => KeyCode::Home,
            NamedKey::End => KeyCode::End,
            NamedKey::Delete => KeyCode::Delete,
            NamedKey::Space => KeyCode::Char(' '),
            _ => return None,
        },
        WinKey::Character(text) => KeyCode::Char(text.chars().next()?),
        _ => return None,
    };
    // Cmd+V on a Mac pastes, as ctrl+v does elsewhere.
    let paste = cfg!(target_os = "macos") && modifiers.super_key() && code == KeyCode::Char('v');
    Some(Key {
        code,
        ctrl: modifiers.control_key() || paste,
        alt: modifiers.alt_key(),
        shift: !matches!(code, KeyCode::Char(_)) && modifiers.shift_key(),
    })
}

/// The key pressed without its modifiers, where the platform says.
#[cfg(any(windows, target_os = "macos", target_os = "linux"))]
fn plain_key(event: &winit::event::KeyEvent) -> WinKey {
    use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
    event.key_without_modifiers()
}

/// The key pressed (no platform support for it without modifiers).
#[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
fn plain_key(event: &winit::event::KeyEvent) -> WinKey {
    event.logical_key.clone()
}

/// A wake-up from a background thread.
#[derive(Debug)]
struct Wake;

/// The setup screen under way ([`crate::setup`]), before the app starts.
struct SetupRun {
    setup: crate::setup::Setup,
    buttons: crate::view::setup::Buttons,
}

struct Gui {
    options: Option<AppOptions>,
    app: Option<App>,
    /// Setup, while the player waits for its files.
    setup: Option<SetupRun>,
    canvas: Option<Canvas>,
    window: Option<Arc<Window>>,
    screen: Option<Screen>,
    modifiers: ModifiersState,
    cell: (u16, u16),
    /// The pointer's column in cells, with the share of its cell (a
    /// Karaoke slider follows it to the pixel).
    x: f64,
    /// The pointer's row in cells, with the share of its cell (a Mixer tab
    /// fader follows it to the pixel).
    y: f64,
    /// The pointer's cell while it is in the window (links underline
    /// under it).
    pointer: Option<(u16, u16)>,
    held: Option<MouseButton>,
    wheel: f64,
    dirty: bool,
    /// The window cannot be seen (minimised, on another desktop, covered):
    /// nothing is drawn until it can.
    hidden: bool,
    /// The redraw asked for is the next frame of what moves (not input,
    /// nor the system's): it is shown only if a pixel changed.
    paced: bool,
    /// The size of the frame the window shows; `None`: it needs one.
    presented: Option<(u32, u32)>,
    /// When the last frame with Karaoke mode's stage was drawn.
    last_stage: Option<Instant>,
    /// The singing's streak as the picture shines with it.
    shine: streak::PictureShine,
    /// The window is fullscreen for Karaoke mode.
    fullscreen: bool,
    last_draw: Instant,
    /// When the layer alone was last moved.
    last_layer: Instant,
    /// A frame of what moves smoothly: one a refresh of the display
    /// ([`display_frame`]), and when that was looked at.
    frame: Duration,
    frame_checked: Instant,
    /// When the last frame of what moves began (for the trace).
    last_paced: Option<Instant>,
    /// The frames, for the trace.
    frames: crate::trace::Frames,
    started: Instant,
    exit_after: Option<f64>,
    error: Option<String>,
}

impl Gui {
    fn act(&mut self, action: Option<Action>) {
        let (Some(app), Some(action)) = (self.app.as_mut(), action) else {
            return;
        };
        app.dispatch(Action::Tick(app.now_ms()));
        app.dispatch(action);
        self.dirty = true;
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn mouse(&mut self, kind: MouseKind) {
        let Some(app) = &self.app else { return };
        let input = MouseInput {
            kind,
            col: self.cell.0,
            row: self.cell.1,
        };
        let mut action = mouse::action_at(&app.state, input, self.x, self.y);
        if self.modifiers.shift_key() {
            action = action.map(mouse::shifted);
        }
        self.act(action);
    }

    /// Moves the pointer; redraws when it enters or leaves a link.
    fn hover(&mut self, pointer: Option<(u16, u16)>) {
        let link = |gui: &Gui| {
            let (Some(app), Some((col, row))) = (&gui.app, gui.pointer) else {
                return None;
            };
            match mouse::hit(&app.state, col, row) {
                Some(mouse::Target::Link(op)) => Some(op),
                _ => None,
            }
        };
        let before = link(self);
        self.pointer = pointer;
        if link(self) != before {
            self.dirty = true;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    /// The setup screen's frame.
    fn redraw_setup(&mut self) {
        let (Some(window), Some(screen), Some(canvas), Some(run)) = (
            &self.window,
            &mut self.screen,
            &mut self.canvas,
            &mut self.setup,
        ) else {
            return;
        };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return;
        };
        canvas.fit(size.width, size.height);
        let status = run.setup.status();
        run.buttons = canvas.draw_widgets(|frame| {
            crate::view::setup::draw(frame, &status, crate::view::colors::BG)
        });
        let changes = canvas.take_changes();
        self.presented = match screen {
            Screen::Gpu(gpu) => match gpu.show(canvas, &changes, size.width, size.height, None) {
                Ok(present::Shown::Yes) => Some((size.width, size.height)),
                _ => None,
            },
            Screen::Soft(surface) => {
                if surface.resize(w, h).is_err() {
                    return;
                }
                let Ok(mut buffer) = surface.buffer_mut() else {
                    return;
                };
                canvas.write_xrgb(&mut buffer, size.width as usize, size.height as usize);
                buffer
                    .present()
                    .is_ok()
                    .then_some((size.width, size.height))
            }
        };
        self.dirty = false;
    }

    /// The person's choice on the setup screen.
    fn setup_choice(&mut self, event_loop: &ActiveEventLoop, choice: crate::view::setup::Choice) {
        match choice {
            crate::view::setup::Choice::Quit => event_loop.exit(),
            crate::view::setup::Choice::Retry => {
                if let Some(run) = &self.setup {
                    run.setup.retry();
                }
            }
        }
    }

    /// Setup finished: the app starts with the files it found.
    fn setup_done(&mut self, paths: &[std::path::PathBuf]) {
        self.setup = None;
        if let Some(mut options) = self.options.take() {
            crate::cli::set_models(&mut options, paths);
            if let Some(canvas) = &self.canvas {
                options.size = canvas.grid();
            }
            self.app = Some(App::new(options));
        }
        self.dirty = true;
        self.presented = None;
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn redraw(&mut self) {
        if self.setup.is_some() {
            return self.redraw_setup();
        }
        let self_pointer = self.pointer;
        let (Some(window), Some(screen), Some(canvas), Some(app)) = (
            &self.window,
            &mut self.screen,
            &mut self.canvas,
            &mut self.app,
        ) else {
            return;
        };
        let size = window.inner_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return;
        };
        let grid = canvas.fit(size.width, size.height);
        if grid != app.state.size {
            app.dispatch(Action::Resize(grid.0, grid.1));
        }
        app.pump();
        let shape = (size.width, size.height);
        // Only a layer moves (the Karaoke tab's voice lane going by): the
        // GPU moves it, and nothing else is drawn, until the next frame of
        // the rest is due ([`FRAME_CALM`]).
        let layer_only = matches!(screen, Screen::Gpu(_))
            && canvas.has_layer()
            && !self.dirty
            && !app.smooth()
            && self.presented == Some(shape)
            && self.last_draw.elapsed() < FRAME_CALM;
        if layer_only {
            let started = Instant::now();
            let heard = app.heard_for_frame();
            let layer = canvas.layer_frame(heard);
            let drawn = started.elapsed();
            let paced = std::mem::take(&mut self.paced);
            let gap = paced
                .then(|| self.last_paced.map(|at| started - at))
                .flatten()
                .filter(|&gap| gap < Duration::from_millis(250));
            if paced {
                self.last_paced = Some(started);
            }
            let showing = Instant::now();
            if let Screen::Gpu(gpu) = screen {
                gpu.set_stage(None, size.width, size.height);
                let none = canvas::Changes::default();
                if gpu
                    .show(canvas, &none, size.width, size.height, layer)
                    .is_err()
                {
                    self.presented = None;
                }
            }
            let late = matches!(screen, Screen::Gpu(gpu) if gpu.behind);
            if paced {
                jodel_gpu::pace::frame(late);
            }
            self.last_layer = Instant::now();
            self.frames
                .frame(gap, late, drawn, showing.elapsed(), "the layer alone");
            return;
        }
        let spectrum = app.spectrum();
        let heard = app.heard_for_frame();
        let scene = Scene {
            state: &app.state,
            spectrum: &spectrum,
            background: colors::BG,
            links: Links::OnHover(self_pointer),
            heard,
        };
        let drawing = std::time::Instant::now();
        // Frames of what moves, and how far apart they come (a long pause
        // is the window resting), for the trace.
        let paced = std::mem::take(&mut self.paced);
        let gap = paced
            .then(|| self.last_paced.map(|at| drawing - at))
            .flatten()
            .filter(|&gap| gap < Duration::from_millis(250));
        if paced {
            self.last_paced = Some(drawing);
        }
        canvas.draw(&scene, app.art());
        let drawn = drawing.elapsed();
        crate::trace::took("frame: drawn", drawing, false);
        self.dirty = false;
        self.last_draw = Instant::now();
        let what = if crate::core::karaoke::in_mode(&app.state) {
            "Karaoke mode".to_owned()
        } else {
            format!("{:?}, {:?}", app.state.screen, app.state.side)
        };
        // The layer where it is now.
        let layer = canvas.layer_frame(heard);
        // Karaoke mode's stage, stepped by the time since the last.
        let now = Instant::now();
        let dt = self
            .last_stage
            .map_or(0.0, |at| now.duration_since(at).as_secs_f32());
        let stage = match screen {
            Screen::Gpu(_) => {
                let stage = canvas.stage_frame(&app.state, heard, dt);
                // The picture's bubbles' liquid and bursts, and its streak's
                // shine, over it on the GPU.
                let clock = app.state.karaoke.lag.sung_clock(heard);
                let over = canvas.picture_over();
                let song = streak::Sung {
                    singing: &app.state.karaoke.singing,
                    game: app.state.config.karaoke_game,
                };
                let at = streak::Moment { now: heard, clock };
                let (edge, ups) = self.shine.frame(over, song, at);
                let ours = over.into_iter();
                let bubbles = ours.clone().flat_map(|o| o.bubbles.iter().copied());
                let bubbles: Vec<_> = bubbles.chain(edge).collect();
                let sprays = ours.flat_map(|o| o.sprays.iter()).chain(ups);
                let sprays: Vec<_> = sprays.copied().collect();
                stage_gpu::with_picture(stage, &bubbles, &sprays, clock as f32)
            }
            Screen::Soft(_) => None,
        };
        self.last_stage = stage.is_some().then_some(now);
        if let Screen::Gpu(gpu) = screen {
            gpu.set_stage(stage.as_ref(), size.width, size.height);
        }
        // A frame of what moves that changed no pixel is not shown: the
        // window still shows the last (unless a layer moved).
        if paced
            && !canvas.changed
            && layer.is_none()
            && stage.is_none()
            && self.presented == Some(shape)
        {
            jodel_gpu::pace::frame(false);
            self.frames.frame(gap, false, drawn, Duration::ZERO, &what);
            return;
        }
        let changes = canvas.take_changes();
        let showing = std::time::Instant::now();
        let first = self.presented.is_none();
        self.presented = match screen {
            Screen::Gpu(gpu) => match gpu.show(canvas, &changes, size.width, size.height, layer) {
                Ok(present::Shown::Yes) => Some(shape),
                Ok(present::Shown::NotNow) => None,
                Err(_) => {
                    // The GPU gave up on the window: softbuffer from now,
                    // every lane in the pixmap.
                    if let Some(soft) = Screen::soft(window) {
                        *screen = soft;
                        canvas.set_layers(false);
                    }
                    None
                }
            },
            Screen::Soft(surface) => {
                if surface.resize(w, h).is_err() {
                    return;
                }
                let Ok(mut buffer) = surface.buffer_mut() else {
                    return;
                };
                canvas.write_xrgb(&mut buffer, size.width as usize, size.height as usize);
                buffer.present().is_ok().then_some(shape)
            }
        };
        crate::trace::took("frame: shown", showing, first);
        // Late: the frame before had not got through the GPU by now, kept
        // from it by other work. Background GPU work gives way then.
        let late = matches!(screen, Screen::Gpu(gpu) if gpu.behind);
        if paced {
            jodel_gpu::pace::frame(late);
        }
        self.frames
            .frame(gap, late, drawn, showing.elapsed(), &what);
    }
}

/// What puts the frames on screen.
enum Screen {
    /// The GPU: only the changed pixels go over.
    Gpu(Box<present::GpuWindow>),
    /// softbuffer: the whole window each frame.
    Soft(softbuffer::Surface<Arc<Window>, Arc<Window>>),
}

impl Screen {
    /// The GPU's, unless `JODELDIPLOM_SOFTBUFFER` is set or it cannot be
    /// set up; softbuffer's otherwise; an error if neither.
    fn new(window: &Arc<Window>) -> Result<Screen, String> {
        if std::env::var_os("JODELDIPLOM_SOFTBUFFER").is_none() {
            if let Ok(gpu) = present::GpuWindow::new(window.clone()) {
                return Ok(Screen::Gpu(Box::new(gpu)));
            }
        }
        softbuffer::Context::new(window.clone())
            .and_then(|context| softbuffer::Surface::new(&context, window.clone()))
            .map(Screen::Soft)
            .map_err(|e| format!("cannot draw into the window: {e}"))
    }

    /// softbuffer's, if it can be set up.
    fn soft(window: &Arc<Window>) -> Option<Screen> {
        softbuffer::Context::new(window.clone())
            .and_then(|context| softbuffer::Surface::new(&context, window.clone()))
            .map(Screen::Soft)
            .ok()
    }
}

impl ApplicationHandler<Wake> for Gui {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        crate::trace::note("window: resumed");
        let started = std::time::Instant::now();
        let attributes = Window::default_attributes()
            .with_title("jodeldiplom")
            .with_window_icon(window_icon(32))
            .with_inner_size(LogicalSize::new(1280.0, 800.0))
            .with_min_inner_size(LogicalSize::new(480.0, 300.0));
        // The taskbar's (and Alt+Tab's) larger icon.
        #[cfg(windows)]
        let attributes = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attributes.with_taskbar_icon(window_icon(64))
        };
        #[cfg(target_os = "macos")]
        set_dock_icon();
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.error = Some(format!("cannot open a window: {error}"));
                event_loop.exit();
                return;
            }
        };
        crate::trace::took("window: created", started, true);
        let started = std::time::Instant::now();
        let screen = match Screen::new(&window) {
            Ok(screen) => screen,
            Err(error) => {
                self.error = Some(error);
                event_loop.exit();
                return;
            }
        };
        crate::trace::took("window: frames' screen set up", started, true);
        let started = std::time::Instant::now();
        let mut canvas = Canvas::new(80, 24, font_px(window.scale_factor()));
        // Lanes the window scrolls by itself, where the GPU shows it.
        canvas.set_layers(matches!(screen, Screen::Gpu(_)));
        let size = window.inner_size();
        let grid = canvas.fit(size.width, size.height);
        crate::trace::took("window: canvas", started, true);
        let started = std::time::Instant::now();
        if let Some(mut options) = self.options.take() {
            options.size = grid;
            // The moves written down (a data folder chosen anew) go on in
            // the background from now.
            crate::cli::start_mover(&mut options);
            match crate::cli::models_ready(&options) {
                Ok(paths) => {
                    crate::cli::set_models(&mut options, &paths);
                    self.app = Some(App::new(options));
                }
                Err((manifest, places)) => {
                    // The files are not all here: the setup screen first.
                    let setup = crate::setup::Setup::start(
                        manifest,
                        places,
                        Arc::new(crate::setup::Real),
                        options.notify.clone(),
                    );
                    self.setup = Some(SetupRun {
                        setup,
                        buttons: Default::default(),
                    });
                    self.options = Some(options);
                }
            }
        }
        crate::trace::took("window: the app started", started, true);
        self.canvas = Some(canvas);
        self.screen = Some(screen);
        self.frame = display_frame(&window);
        self.frame_checked = Instant::now();
        self.window = Some(window);
        self.dirty = true;
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: Wake) {
        // What came is taken in by `about_to_wait`, next.
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // The setup screen takes its keys and clicks; the rest (drawing,
        // the size, the pointer, closing) goes on as ever.
        if let Some(run) = &self.setup {
            let choice = match &event {
                WindowEvent::KeyboardInput { event: key, .. }
                    if key.state == ElementState::Pressed =>
                {
                    let name = match &key.logical_key {
                        WinKey::Named(NamedKey::Enter) => "enter".to_owned(),
                        WinKey::Named(NamedKey::Escape) => "esc".to_owned(),
                        WinKey::Character(text) => text.to_string(),
                        _ => String::new(),
                    };
                    let failed = run.setup.status().failure.is_some();
                    Some(crate::view::setup::choice(&name, failed))
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: WinButton::Left,
                    ..
                } => Some(run.buttons.at(self.cell.0, self.cell.1)),
                WindowEvent::KeyboardInput { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. } => Some(None),
                _ => None,
            };
            if let Some(choice) = choice {
                if let Some(choice) = choice {
                    self.setup_choice(event_loop, choice);
                }
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::Resized(_) => self.dirty = true,
            WindowEvent::Occluded(hidden) => {
                self.hidden = hidden;
                if !hidden {
                    self.dirty = true;
                    self.presented = None;
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(canvas) = &mut self.canvas {
                    canvas.set_font_px(font_px(scale_factor));
                }
                self.dirty = true;
                self.presented = None;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                // Ctrl alone (not AltGr, which is ctrl+alt): held in the
                // browser it previews the file selected.
                let ctrl = |m: ModifiersState| m.control_key() && !m.alt_key();
                let was = ctrl(self.modifiers);
                self.modifiers = modifiers.state();
                let now = ctrl(self.modifiers);
                if now != was {
                    self.act(Some(Action::Ctrl(if now {
                        Ctrl::Pressed
                    } else {
                        Ctrl::Released
                    })));
                }
            }
            // Ctrl let go of in another window is never seen: let go of here.
            WindowEvent::Focused(false) => {
                if self.app.as_ref().is_some_and(|app| app.state.preview.ctrl) {
                    self.act(Some(Action::Ctrl(Ctrl::Released)));
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                // The clock first: what a key means can depend on the time
                // (space while a name is being typed).
                if let Some(app) = self.app.as_mut() {
                    app.dispatch(Action::Tick(app.now_ms()));
                }
                // With alt the key as if alone: Option+1 on a Mac is 1. A
                // digit with shift is that digit, shifted (whatever symbol
                // the layout puts there).
                let plain = plain_key(&event);
                let digit = matches!(&plain, WinKey::Character(t) if t.chars().next().is_some_and(|c| c.is_ascii_digit()));
                let shifted_digit = self.modifiers.shift_key() && digit;
                let logical = if self.modifiers.alt_key() || shifted_digit {
                    plain
                } else {
                    event.logical_key.clone()
                };
                let action = key(&logical, self.modifiers)
                    .map(|mut k| {
                        k.shift |= shifted_digit;
                        k
                    })
                    .and_then(|k| self.app.as_ref().and_then(|app| key_action(&app.state, k)));
                self.act(action);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let Some(canvas) = &self.canvas else { return };
                let (cw, ch) = canvas.cell();
                let (cols, rows) = canvas.grid();
                let cell = (
                    ((position.x.max(0.0) as usize) / cw).min(usize::from(cols.saturating_sub(1)))
                        as u16,
                    ((position.y.max(0.0) as usize) / ch).min(usize::from(rows.saturating_sub(1)))
                        as u16,
                );
                let x = (position.x.max(0.0) / cw as f64).min(f64::from(cols));
                let y = (position.y.max(0.0) / ch as f64).min(f64::from(rows));
                let moved = x != self.x || y != self.y;
                self.x = x;
                self.y = y;
                // A Karaoke slider or a Mixer tab fader held follows the
                // pointer within a cell.
                let sliding = self.held == Some(MouseButton::Left)
                    && self.app.as_ref().is_some_and(|app| {
                        app.state.karaoke.dragging.is_some() || app.state.desk.dragging.is_some()
                    });
                if cell == self.cell && self.pointer.is_some() && moved && sliding {
                    self.mouse(MouseKind::Drag(MouseButton::Left));
                } else if cell != self.cell || self.pointer.is_none() {
                    self.hover(Some(cell));
                    self.cell = cell;
                    if let Some(button) = self.held {
                        self.mouse(MouseKind::Drag(button));
                    } else if self.app.as_ref().is_some_and(|app| {
                        crate::core::karaoke::in_mode(&app.state)
                            || crate::core::preview::held(&app.state)
                    }) {
                        // Karaoke mode shows its controls while it moves; a
                        // preview selects the row under it.
                        self.mouse(MouseKind::Moved);
                    }
                }
            }
            WindowEvent::CursorLeft { .. } => self.hover(None),
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    WinButton::Left => MouseButton::Left,
                    WinButton::Right => MouseButton::Right,
                    WinButton::Middle => MouseButton::Middle,
                    _ => return,
                };
                if state == ElementState::Pressed {
                    self.held = Some(button);
                    self.mouse(MouseKind::Down(button));
                } else {
                    self.held = None;
                    self.mouse(MouseKind::Up(button));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => {
                        let ch = self.canvas.as_ref().map_or(16, |c| c.cell().1) as f64;
                        p.y / ch
                    }
                };
                self.wheel += lines;
                while self.wheel >= 1.0 {
                    self.wheel -= 1.0;
                    self.mouse(MouseKind::ScrollUp);
                }
                while self.wheel <= -1.0 {
                    self.wheel += 1.0;
                    self.mouse(MouseKind::ScrollDown);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(run) = &self.setup {
            if self
                .exit_after
                .is_some_and(|s| self.started.elapsed().as_secs_f64() >= s)
            {
                event_loop.exit();
                return;
            }
            // Setup tells the loop as it goes: a frame for each telling.
            if let Some(paths) = run.setup.status().finished {
                self.setup_done(&paths);
            } else if let Some(window) = &self.window {
                window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(250),
            ));
            return;
        }
        let Some(app) = &mut self.app else {
            return;
        };
        // What the engine and the workers sent: drawn at once only while
        // nothing is paced; otherwise the next paced frame shows it, so
        // frames stay evenly apart.
        let arrived = app.pump();
        if arrived && !app.animating() {
            self.dirty = true;
        }
        let expired = self
            .exit_after
            .is_some_and(|s| self.started.elapsed().as_secs_f64() >= s);
        if app.state.quit || expired {
            event_loop.exit();
            return;
        }
        // Karaoke mode is fullscreen; leaving it, the window comes back.
        let fullscreen = crate::core::karaoke::in_mode(&app.state);
        if fullscreen != self.fullscreen {
            self.fullscreen = fullscreen;
            if let Some(window) = &self.window {
                window.set_fullscreen(fullscreen.then_some(Fullscreen::Borderless(None)));
            }
            self.dirty = true;
            self.presented = None;
        }
        let animating = app.animating();
        if self.frame_checked.elapsed() >= DISPLAY_EVERY {
            if let Some(window) = &self.window {
                self.frame = display_frame(window);
            }
            self.frame_checked = Instant::now();
        }
        if self.hidden {
            // Nothing to see: the player plays on, the window waits.
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(250),
            ));
        } else if self.dirty || animating {
            // Input and smooth motion at every refresh of the display, the
            // rest at 30 a second.
            let frame = if self.dirty || app.smooth() || app.layer_moves() {
                self.frame
            } else {
                FRAME_CALM
            };

            // The last frame of either kind.
            let last = self.last_draw.max(self.last_layer);
            if last.elapsed() >= frame {
                if let Some(window) = &self.window {
                    // Input asks for its frame itself; this one is paced.
                    self.paced = !self.dirty;
                    window.request_redraw();
                }
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(last + frame));
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(250),
            ));
        }
        // What is due next (changed settings, a count-in, a preview) on
        // time.
        if let Some(due) = app.state.due() {
            let at = Instant::now() + Duration::from_millis(due.saturating_sub(app.now_ms()));
            match event_loop.control_flow() {
                ControlFlow::WaitUntil(sooner) if sooner <= at => {}
                _ => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            }
        }
    }
}

/// Runs the player in a window until it is closed.
pub fn run(args: &Args) -> Result<(), String> {
    let mut options = crate::cli::app_options(args)?;
    let event_loop = EventLoop::<Wake>::with_user_event()
        .build()
        .map_err(|e| format!("cannot start the window system: {e}"))?;
    let proxy = Mutex::new(event_loop.create_proxy());
    options.notify = Arc::new(move || {
        if let Ok(proxy) = proxy.lock() {
            let _ = proxy.send_event(Wake);
        }
    });
    let mut gui = Gui {
        options: Some(options),
        app: None,
        setup: None,
        canvas: None,
        window: None,
        screen: None,
        modifiers: ModifiersState::empty(),
        cell: (0, 0),
        x: 0.0,
        y: 0.0,
        pointer: None,
        held: None,
        wheel: 0.0,
        dirty: true,
        hidden: false,
        paced: false,
        presented: None,
        last_stage: None,
        shine: streak::PictureShine::default(),
        fullscreen: false,
        last_draw: Instant::now() - FRAME,
        last_layer: Instant::now(),
        frame: FRAME,
        frame_checked: Instant::now(),
        last_paced: None,
        frames: crate::trace::Frames::default(),
        started: Instant::now(),
        exit_after: args.exit_after,
        error: None,
    };
    event_loop
        .run_app(&mut gui)
        .map_err(|e| format!("the window system failed: {e}"))?;
    if let Some(mut app) = gui.app.take() {
        app.shutdown();
    }
    if let (Some(path), Some(canvas)) = (&args.save_frame, &gui.canvas) {
        canvas
            .image()
            .save(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    match gui.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
